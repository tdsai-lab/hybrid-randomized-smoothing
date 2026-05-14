#!/usr/bin/env python3
# ============================================================
# TEXT-ONLY certification (text L0) with MULTIMODAL INPUT kept
#
# You still feed (image, text) to the detector exactly as before,
# but you DO NOT add any noise to the image.
#
# Text threat: L0_inplace (attacker may change <= d tokens anywhere)
# Text smoothing kernel K (tokenwise i.i.d.):
#   each token is replaced with prob beta by Unif(allowed \ {t})
#
# Non-wasteful:
#   Kernel does not depend on d, so we estimate pA once per example,
#   then reuse it for all d to compute the TEXT certificate.
#
# Note:
#   grouped_uniform(d, beta, V) returns vectors (pc, pa, g); the correct
#   NP lower bound is computed by the "fill" procedure (sort by g ascending
#   and fill clean mass up to pA). Using pc*pA + pa*(1-pA) is only valid
#   in the 2-atom case.
# ============================================================

import argparse
import random
import logging
import os
import json
import re
import torch
from tqdm import tqdm
from typing import Dict, List, Union

from hybrid_rs.models import LLaVAGuardSafetyDetector, POLICY_MULTI_MODAL
from hybrid_rs.data import (
    add_multimodal_dataset_args,
    get_multimodal_dataset_loader_from_args,
)
from hybrid_rs.utils_hybrid import (
    clopper_pearson_lcb,
    grouped_uniform,
    grouped_absorb,
)

logging.getLogger("transformers.image_utils").setLevel(logging.ERROR)


# ------------------------------------------------------------
# init + IO
# ------------------------------------------------------------
def forbid_initialization():
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def save_testing_result(
    results: Union[Dict, List[Dict]], saving_path: str = None, verbose_result: bool = True
) -> None:
    if saving_path is not None:
        with open(saving_path, "w") as json_file:
            json.dump(results, json_file, ensure_ascii=False, indent=4)


# ------------------------------------------------------------
# Allowed token ids (stable alphabet)
# ------------------------------------------------------------
def build_allowed_token_ids(tokenizer, max_check: int = 50000, min_allowed: int = 2000):
    forbidden = set(getattr(tokenizer, "all_special_ids", []))
    allowed = []
    V = int(tokenizer.vocab_size)

    for tid in range(min(V, int(max_check))):
        if tid in forbidden:
            continue
        try:
            s = tokenizer.decode(
                [tid], skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
        except Exception:
            continue
        s = (s or "").strip()
        if not s:
            continue
        if any(ord(c) < 32 for c in s):
            continue
        if len(s) > 8:
            continue
        allowed.append(int(tid))

    if len(allowed) < int(min_allowed):
        raise RuntimeError(f"allowed token set too small: {len(allowed)} (increase max_check)")
    return allowed


def _sample_uniform_except_from_list(allowed_ids: List[int], t: int) -> int:
    while True:
        r = random.choice(allowed_ids)
        if int(r) != int(t):
            return int(r)


def noise_uniform_l0_full_inplace_ids(tokenizer, text: str, beta: float, allowed_ids: List[int]):
    """
    Text smoothing kernel K:
      For each token position i:
        w.p. 1-beta keep token,
        w.p. beta replace uniformly in allowed_ids \\ {t_i}.
    """
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) == 0:
        return text, ids

    out = list(map(int, ids))
    for i, t in enumerate(out):
        if random.random() < float(beta):
            out[i] = _sample_uniform_except_from_list(allowed_ids, int(t))

    noised_text = tokenizer.decode(
        out, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return noised_text, out


# ------------------------------------------------------------
# Guard parsing + counting policy
# ------------------------------------------------------------
_UNSAFE_RE = re.compile(r"^\s*unsafe\s*[\.\!\:\,]*\s*$", re.IGNORECASE)
_SAFE_RE = re.compile(r"^\s*safe\s*[\.\!\:\,]*\s*$", re.IGNORECASE)


def parse_guard(output: str | None):
    """
    Returns:
      True  => Unsafe
      False => Safe
      None  => unparseable
    """
    if output is None:
        return None
    s = output.strip()
    if not s:
        return None
    first = s.splitlines()[0].strip()
    if _UNSAFE_RE.match(first):
        return True
    if _SAFE_RE.match(first):
        return False
    return None


def unsafe_from_mm(mm, skip_unparseable: bool):
    if mm is None:
        return None if skip_unparseable else True
    return bool(mm)


# ------------------------------------------------------------
# NP "fill" for grouped_uniform/grouped_absorb outputs
# ------------------------------------------------------------
@torch.no_grad()
def np_fill_lower_bound(pA: float, pc: torch.Tensor, pa: torch.Tensor, g: torch.Tensor) -> float:
    """
    Compute:
      min sum_i pa[i] f_i
      s.t. sum_i pc[i] f_i = pA, 0<=f_i<=1
    given groups (pc, pa) and likelihood ratios g=pa/pc.

    Optimal: sort by g ascending; take full groups until reaching pA, then fraction.
    """
    pc = pc.detach().cpu().double()
    pa = pa.detach().cpu().double()
    g = g.detach().cpu().double()

    idx = torch.argsort(g)
    pc = pc[idx]
    pa = pa[idx]

    target = float(pA)
    cum_pc = 0.0
    cum_pa = 0.0

    for i in range(pc.numel()):
        vi = float(pc[i].item())
        if vi <= 0.0:
            continue

        if cum_pc + vi <= target:
            cum_pc += vi
            cum_pa += float(pa[i].item())
        else:
            delta = target - cum_pc
            lam = max(0.0, min(1.0, delta / vi))
            cum_pa += lam * float(pa[i].item())
            return float(cum_pa)

    return float(cum_pa)


def certify_text_only(
    pA: float,
    tau: float,
    d: int,
    beta: float,
    V_eff: int,
    kernel: str,
):
    """
    Returns:
      ok (bool), p_adv_lb (float)

    Semantics: certify that for any adversary changing <= d tokens (L0),
    the UNSAFE probability remains >= tau.
    """
    if kernel == "uniform":
        pc, pa, g = grouped_uniform(int(d), float(beta), int(V_eff))
    elif kernel == "absorb":
        pc, pa, g = grouped_absorb(int(d), float(beta))
    else:
        raise ValueError(f"unknown kernel={kernel}")

    p_adv_lb = np_fill_lower_bound(float(pA), pc, pa, g)
    ok = (p_adv_lb >= float(tau))
    return ok, float(p_adv_lb)


# ------------------------------------------------------------
# MC estimate pA ONCE (no image noise; still pass image to model)
# ------------------------------------------------------------
@torch.no_grad()
def estimate_pA_lcb_once_text_only(
    detector: LLaVAGuardSafetyDetector,
    text0: str,
    image0,
    allowed_ids: List[int],
    beta: float,
    alpha: float,
    n: int,
    batch_size: int,
    skip_unparseable: bool,
    guard_decision_rule: str,
):
    tok = detector.tokenizer

    nA = 0
    accepted = 0
    pbar = tqdm(total=int(n), leave=False)
    pbar.set_description("MC pA (text-only)")

    while accepted < int(n):
        b = min(int(batch_size), int(n) - accepted)

        # 1) texts (noised)
        texts = []
        for _ in range(b):
            t, _ = noise_uniform_l0_full_inplace_ids(
                tokenizer=tok,
                text=text0,
                beta=float(beta),
                allowed_ids=allowed_ids,
            )
            texts.append(t)

        # 2) images: NO noise (repeat same clean image)
        images_for_proc = [image0 for _ in range(b)]

        # 3) prompts (unchanged)
        prompts = []
        for t in texts:
            text_block = {
                "type": "text",
                "text": f"{POLICY_MULTI_MODAL}\n\nUSER TEXT:\n{t}\n",
            }
            conv = [{"role": "user", "content": [{"type": "image"}, text_block]}]
            prompts.append(detector.processor.apply_chat_template(conv, add_generation_prompt=True))

        # 4) batched guard decision
        inputs = detector.processor(
            text=prompts, images=images_for_proc, return_tensors="pt", padding=True
        )
        if str(guard_decision_rule) == "logit_mass":
            unsafe_mask = detector.classify_safe_unsafe_from_inputs(inputs)
            nA += int(unsafe_mask.sum().item())
            accepted += int(unsafe_mask.numel())
        else:
            # Legacy branch: generate label text, then decode+parse Safe/Unsafe.
            inputs = {k: v.to(detector.device) for k, v in inputs.items()}

            out = detector.model.generate(
                **inputs,
                max_new_tokens=3,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=detector.processor.tokenizer.eos_token_id,
            )

            # 5) decode
            prompt_lens = inputs["attention_mask"].sum(dim=1).tolist()
            for j in range(b):
                gen_ids = out[j, int(prompt_lens[j]) :]
                o = detector.processor.decode(gen_ids, skip_special_tokens=True)

                mm = parse_guard(o)
                unsafe = unsafe_from_mm(mm, skip_unparseable=bool(skip_unparseable))
                if unsafe is None:
                    continue
                if unsafe:
                    nA += 1
                accepted += 1

        pbar.n = accepted
        pbar.refresh()
        pbar.set_postfix_str(f"accepted {accepted}/{n} | nA {nA}")

    pbar.close()
    pA_lcb = clopper_pearson_lcb(int(n), int(nA), float(alpha))
    return float(pA_lcb), int(nA)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    forbid_initialization()
    random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel", choices=["uniform", "absorb"], default="uniform")

    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--tau", type=float, default=4.6e-5)

    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=1)

    parser.add_argument("--d_max", type=int, default=8, help="max L0 budget d (token changes)")
    parser.add_argument("--max_examples", type=int, default=400)
    parser.add_argument("--out", type=str, default="./results/text_only_rs_l0_loopd.json")

    add_multimodal_dataset_args(parser, default_dataset="hateful_memes_mm")

    parser.add_argument("--max_chars", type=int, default=4000)
    parser.add_argument("--allowed_token_scan", type=int, default=50000)
    parser.add_argument("--min_allowed_tokens", type=int, default=2000)
    parser.add_argument("--skip_unparseable", action="store_true")
    parser.add_argument(
        "--guard_decision_rule",
        choices=["generate", "logit_mass"],
        default="logit_mass",
        help="Use full generation+parse or single-forward aggregated Safe/Unsafe token mass.",
    )

    args = parser.parse_args()

    args.out = os.path.join(
        "results_text_l0",
        f"text_only_rs_l0_loopd_{args.kernel}_b{args.beta}_n{args.n}_max_exemples{args.max_examples}_dataset_{args.dataset}.json",
    )

    detector = LLaVAGuardSafetyDetector()
    tok = detector.tokenizer
    print(f"Using guard decision rule: {args.guard_decision_rule}")

    print(f"Using dataset: {args.dataset}")
    loader = get_multimodal_dataset_loader_from_args(args)

    allowed_ids = build_allowed_token_ids(tok, args.allowed_token_scan, args.min_allowed_tokens)
    V_eff = len(allowed_ids)
    print(f"Allowed token IDs: {V_eff} tokens")

    results = []

    for i, ((text0, image0), _) in enumerate(loader):
        if i >= int(args.max_examples):
            break

        if isinstance(text0, str) and len(text0) > int(args.max_chars):
            text0 = text0[: int(args.max_chars)]

        ids0 = tok(text0, add_special_tokens=False).input_ids
        d_max_eff = min(int(args.d_max), len(ids0))

        # ---- estimate pA once for this example (text noise only; image clean) ----
        pA_lcb, nA_unsafe = estimate_pA_lcb_once_text_only(
            detector=detector,
            text0=text0,
            image0=image0,
            allowed_ids=allowed_ids,
            beta=float(args.beta),
            alpha=float(args.alpha),
            n=int(args.n),
            batch_size=int(args.batch_size),
            skip_unparseable=bool(args.skip_unparseable),
            guard_decision_rule=str(args.guard_decision_rule),
        )

        certified_curve = []
        ok_all_prev = True
        d_cert = 0

        for d in range(1, d_max_eff + 1):
            ok_d, p_adv_lb = certify_text_only(
                pA=float(pA_lcb),
                tau=float(args.tau),
                d=int(d),
                beta=float(args.beta),
                V_eff=int(V_eff),
                kernel=str(args.kernel),
            )

            certified_curve.append(
                dict(
                    d=int(d),
                    nA_unsafe=int(nA_unsafe),
                    pA_lcb=float(pA_lcb),
                    p_adv_lb=float(p_adv_lb),
                    ok=bool(ok_d),
                )
            )
            
            print(
                "Example %d | d=%d | pA_lcb=%.6g | p_adv_lb=%.6g | ok=%s | nA_unsafe=%d"
                % (i, d, pA_lcb, p_adv_lb, str(ok_d), nA_unsafe)
            )

            if ok_all_prev and ok_d:
                d_cert = d
            else:
                ok_all_prev = False
                break

            

        results.append(
            dict(
                example_id=int(i),
                text_threat="l0_inplace",
                guard_decision_rule=str(args.guard_decision_rule),
                kernel=str(args.kernel),
                beta=float(args.beta),
                tau=float(args.tau),
                n=int(args.n),
                alpha=float(args.alpha),
                d_max_effective=int(d_max_eff),
                d_cert=int(d_cert),
                certified_curve=certified_curve,
            )
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_testing_result(results, args.out, verbose_result=False)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
