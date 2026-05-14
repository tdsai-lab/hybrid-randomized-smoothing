#!/usr/bin/env python3
# ============================================================
# Shared-D Hybrid RS (text L0 + image L2)
#
# Approach B:
#   - sample once under the full ambient kernel K_D
#   - estimate a single pA under that shared kernel
#   - certify all d <= D by changing only the NP threshold terms
#     induced by the discrete threat model at budget d
# ============================================================

import argparse
import logging
import os
import random

import torch
from torchvision.transforms.functional import to_pil_image, to_tensor
from tqdm import tqdm

from hybrid_rs.certify.certify_multimodal_text_append_suffix_image import (
    batch_noise_uniform_append_only,
    build_allowed_token_ids,
    build_append_only_base_ids,
    build_cached_mm_prompt_template,
    forbid_initialization,
    noise_uniform_suffix_inplace,
    parse_guard,
    save_testing_result,
)
from hybrid_rs.data import add_multimodal_dataset_args, get_multimodal_dataset_loader_from_args
from hybrid_rs.models import LLaVAGuardSafetyDetector
from hybrid_rs.utils_hybrid import (
    add_gaussian_noise,
    certify_r_hybrid,
    clopper_pearson_lcb,
    grouped_absorb,
    grouped_uniform,
)


logging.getLogger("transformers.image_utils").setLevel(logging.ERROR)


def main():
    forbid_initialization()
    random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["uniform", "absorb"], default="uniform")
    parser.add_argument(
        "--text_threat",
        choices=["inplace_suffix", "append_only"],
        default="append_only",
    )

    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--tau", type=float, default=4.6e-5)

    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=20)

    parser.add_argument("--d_max", type=int, default=8, help="max d for inplace_suffix")
    parser.add_argument(
        "--m", type=int, default=8, help="max appended tokens for append_only"
    )
    parser.add_argument(
        "--pad_token", type=str, default=None, help="optional explicit pad token string"
    )

    parser.add_argument("--r_max", type=float, default=50.0)
    parser.add_argument("--max_examples", type=int, default=400)
    parser.add_argument(
        "--out", type=str, default="./results/shared_D_hybrid_rs.json"
    )
    parser.add_argument(
        "--use_diffusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable diffusion denoising (default: on); pass --no-use_diffusion to disable",
    )
    parser.add_argument(
        "--flash_attn",
        action="store_true",
        help="enable Flash Attention 2",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="load model in 4-bit quantization",
    )

    add_multimodal_dataset_args(parser, default_dataset="hateful_memes_mm")

    parser.add_argument(
        "--max_chars",
        type=int,
        default=4000,
        help="truncate text0 to avoid tokenizer pathologies",
    )
    parser.add_argument(
        "--allowed_token_scan",
        type=int,
        default=50000,
        help="how many token ids to scan to build allowed set",
    )
    parser.add_argument(
        "--min_allowed_tokens",
        type=int,
        default=2000,
        help="sanity threshold for allowed token ids",
    )
    parser.add_argument(
        "--skip_unparseable",
        action="store_true",
        help="skip MC draws where output is not exactly Safe/Unsafe",
    )
    parser.add_argument(
        "--guard_decision_rule",
        choices=["generate", "logit_mass"],
        default="logit_mass",
        help="Use full generation+parse or single-forward aggregated Safe/Unsafe token mass.",
    )

    args = parser.parse_args()

    args.out = os.path.join(
        "results_shared_D",
        f"shared_D_hybrid_rs_{args.text_threat}_{args.kernel}_b{args.beta}_s{args.sigma}_n{args.n}_max{args.max_examples}_diff_{args.use_diffusion}_dataset_{args.dataset}.json",
    )

    detector = LLaVAGuardSafetyDetector(
        use_flash_attn=args.flash_attn,
        load_in_4bit=args.load_in_4bit,
    )
    tok = detector.tokenizer
    prompt_prefix, prompt_suffix = build_cached_mm_prompt_template(detector.processor)

    print(args)
    print(f"Using guard decision rule: {args.guard_decision_rule}")

    denoiser = None
    if args.use_diffusion:
        from hybrid_rs.utils_diffusion import Denoiser
        denoiser = Denoiser()
        denoiser.eval().to("cuda")
        print("Using diffusion denoiser model for image noising.")

    print(f"Using dataset: {args.dataset}")
    loader = get_multimodal_dataset_loader_from_args(args)

    results = []

    allowed_ids = build_allowed_token_ids(tok, args.allowed_token_scan)
    v_eff = len(allowed_ids)
    print(f"Allowed token IDs: {v_eff} tokens (scan up to {args.allowed_token_scan})")

    bang_id = tok.convert_tokens_to_ids(",")
    if bang_id is None or bang_id in getattr(tok, "all_special_ids", []):
        raise RuntimeError("Tokenizer does not have a usable ',' token.")

    for i, ((text0, image0), _) in enumerate(loader):
        if i >= int(args.max_examples):
            break

        print(f"example {i}: text0:", text0)

        if args.text_threat == "append_only":
            base_ids, pad_span = build_append_only_base_ids(
                tok, text0, int(args.m), bang_id
            )
            d_max_eff = int(args.m)
        else:
            ids0 = tok(text0, add_special_tokens=False).input_ids
            d_max_eff = min(int(args.d_max), len(ids0))

        d_ambient = int(d_max_eff)
        nA = 0
        done = 0
        pbar = tqdm(total=int(args.n), leave=False)
        pbar.set_description(f"MC (shared D={d_ambient})")

        while done < int(args.n):
            b = min(int(args.batch_size), int(args.n) - done)

            if args.text_threat == "append_only":
                texts = batch_noise_uniform_append_only(
                    tok,
                    base_ids,
                    pad_span,
                    float(args.beta),
                    d_ambient,
                    allowed_ids,
                    b,
                )
            else:
                texts = [
                    noise_uniform_suffix_inplace(
                        tokenizer=tok,
                        text=text0,
                        beta=float(args.beta),
                        d=d_ambient,
                        allowed_ids=allowed_ids,
                    )
                    for _ in range(b)
                ]

            if denoiser is not None:
                x0 = to_tensor(image0)
                x0_bchw = x0.unsqueeze(0).repeat(b, 1, 1, 1)
                xhat_bchw = denoiser(x0_bchw, float(args.sigma))
                xhat_bchw_cpu = xhat_bchw.detach().cpu()
                images_for_proc = [to_pil_image(xhat_bchw_cpu[k]) for k in range(b)]
            else:
                images_for_proc = [
                    add_gaussian_noise(image0, float(args.sigma)) for _ in range(b)
                ]

            prompts = [f"{prompt_prefix}{t}{prompt_suffix}" for t in texts]

            inputs = detector.processor(
                text=prompts,
                images=images_for_proc,
                return_tensors="pt",
                padding=True,
            )
            if args.guard_decision_rule == "logit_mass":
                unsafe_mask = detector.classify_safe_unsafe_from_inputs(inputs)
                nA += int(unsafe_mask.sum().item())
            else:
                inputs = {k: v.to(detector.device) for k, v in inputs.items()}
                out = detector.model.generate(
                    **inputs,
                    max_new_tokens=3,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                    pad_token_id=detector.processor.tokenizer.eos_token_id,
                )

                prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
                for j in range(b):
                    gen_ids = out[j, int(prompt_lens[j]) :]
                    output = detector.processor.decode(gen_ids, skip_special_tokens=True)
                    mm = parse_guard(output)
                    if mm:
                        nA += 1

            done += b
            pbar.update(b)
            pbar.set_postfix_str(f"n {done}/{args.n} | nA {nA}")

        pbar.close()

        pA_shared = clopper_pearson_lcb(int(args.n), int(nA), float(args.alpha))
        print(f"  Shared D={d_ambient}: nA={nA}/{args.n}, pA_lcb={pA_shared:.6f}")

        certified_curve = []
        d_cert = 0
        rs = []

        for d in range(1, d_ambient + 1):
            if args.kernel == "uniform":
                pc, pa, g = grouped_uniform(d, float(args.beta), int(v_eff))
            else:
                pc, pa, g = grouped_absorb(d, float(args.beta))

            r_d = certify_r_hybrid(
                pA=float(pA_shared),
                tau=float(args.tau),
                sigma=float(args.sigma),
                p_clean=pc,
                p_adv=pa,
                gamma=g,
                r_max=float(args.r_max),
            )

            rec = dict(
                d=int(d),
                nA_unsafe=int(nA),
                pA_lcb=float(pA_shared),
                r=float(r_d),
            )
            certified_curve.append(rec)

            ok_d = r_d > 0.0
            if ok_d:
                d_cert = d
                rs.append(r_d)

            print(f"  d={d} r={r_d:.6f} ok={ok_d}")
            if not ok_d:
                break

        results.append(
            dict(
                example_id=int(i),
                approach="shared_D",
                text_threat=str(args.text_threat),
                guard_decision_rule=str(args.guard_decision_rule),
                kernel=str(args.kernel),
                beta=float(args.beta),
                sigma=float(args.sigma),
                tau=float(args.tau),
                n=int(args.n),
                alpha=float(args.alpha),
                D_ambient=int(d_ambient),
                d_max_effective=int(d_ambient),
                d_cert=int(d_cert),
                pA_shared=float(pA_shared),
                nA_shared=int(nA),
                certified_curve=certified_curve,
            )
        )

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    save_testing_result(results, args.out, verbose_result=False)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
