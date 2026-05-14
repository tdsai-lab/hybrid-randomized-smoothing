#!/usr/bin/env python3
import argparse
import random
import logging
import json
import os

import torch
from tqdm import tqdm
from typing import List, Optional

from hybrid_rs.models import LLaVAGuardSafetyDetector, POLICY_MULTI_MODAL
from hybrid_rs.data import add_multimodal_dataset_args, get_multimodal_dataset_loader_from_args
from hybrid_rs.attack.predict_adv_text_image import JointTextImageAttack, clopper_pearson_lcb
from hybrid_rs.utils_hybrid import add_gaussian_noise

# python predict_adv_text_image_multimodal.py --attack_steps 6 --attack_verbose

logging.getLogger("transformers.image_utils").setLevel(logging.ERROR)


# attack is defined like this
# [policy]∥[USER TEXT]∥[adv_text].
# Length: it is entirely determined by adv_text_inits

# ============================================================================
# Utility functions
# ============================================================================
import numpy as np
from PIL import Image

def pixel_values_to_pil(processor, pv4: torch.Tensor) -> Image.Image:
    """
    pv4: [1,3,H,W] or [3,H,W] in processor pixel_values space
         i.e. after image_processor normalization: (x - mean)/std, typically on 0..1 origin.
    Returns: PIL RGB image.
    """
    if pv4.dim() == 4:
        pv = pv4[0]
    elif pv4.dim() == 3:
        pv = pv4
    else:
        raise ValueError(f"Expected pv4 with 3 or 4 dims, got {tuple(pv4.shape)}")

    pv = pv.detach().cpu().float()  # [3,H,W]

    ip = processor.image_processor
    mean = torch.tensor(ip.image_mean).view(3, 1, 1)
    std = torch.tensor(ip.image_std).view(3, 1, 1)

    x01 = (pv * std + mean).clamp(0.0, 1.0)              # back to [0,1]
    x_uint8 = (x01.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(x_uint8, mode="RGB")

def _single_token_id_any(tok, candidates):
    for s in candidates:
        ids = tok(s, add_special_tokens=False).input_ids
        if len(ids) == 1:
            return int(ids[0]), s
    return None, None


@torch.no_grad()
def forced_choice_safe_unsafe_from_last_logits(logits_last, safe_id: int, unsafe_id: int):
    ls = logits_last[:, safe_id]
    lu = logits_last[:, unsafe_id]
    return ["Unsafe" if (lu[i] > ls[i]) else "Safe" for i in range(logits_last.size(0))]


def sigma_pixel_to_pixel_values(processor, sigma_pixel: float, device: torch.device, dtype: torch.dtype) -> float:
    ip = processor.image_processor
    std = torch.tensor(ip.image_std, device=device, dtype=dtype)
    std_min = float(std.min())
    return float(sigma_pixel / max(std_min, 1e-12))


def get_nonascii_toks(tokenizer, device="cpu"):
    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        decoded = tokenizer.decode([i])
        if not is_ascii(decoded) or len(decoded) == 0:
            ascii_toks.append(i)
    
    special_tokens = [
        tokenizer.bos_token_id,
        tokenizer.eos_token_id,
        tokenizer.pad_token_id,
        tokenizer.unk_token_id
    ]
    for tok_id in special_tokens:
        if tok_id is not None:
            ascii_toks.append(tok_id)
    
    return torch.tensor(ascii_toks, device=device)


def get_filtered_cands(tokenizer, control_cand, filter_cand=True, curr_control=None) -> List[str]:
    cands, count = [], 0
    for i in range(control_cand.shape[0]):
        decoded_str = tokenizer.decode(control_cand[i], skip_special_tokens=True)
        if filter_cand:
            if decoded_str != curr_control and len(
                tokenizer(decoded_str, add_special_tokens=False).input_ids
            ) == len(control_cand[i]):
                cands.append(decoded_str)
            else:
                count += 1
        else:
            cands.append(decoded_str)

    if filter_cand:
        cands = cands + [cands[-1]] * (len(control_cand) - len(cands))
    return cands


# ============================================================================
# Build inputs with adversarial text
# ============================================================================

def build_chat_inputs_with_adv_text(detector, user_text: str, adv_text: str, image_pil, 
                                     adv_pixel_values: Optional[torch.Tensor], device: torch.device):
    """Build chat inputs with adversarial text appended to user text"""
    processor = detector.processor
    
    # Combine prompt with adversarial text
    full_text = user_text + " " + adv_text if adv_text else user_text
    
    text_block = {"type": "text", "text": f"{POLICY_MULTI_MODAL}\n\nUSER TEXT:\n{full_text}\n"}
    conversation = [{"role": "user", "content": [{"type": "image"}, text_block]}]
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)

    base = processor(images=image_pil, text=text_prompt, return_tensors="pt")
    base = {k: v.to(device) if torch.is_tensor(v) else v for k, v in base.items()}
    
    
    pv = base.get("pixel_values", None)
    if pv is None:
        raise RuntimeError("processor did not return pixel_values")
        
    # Replace with adversarial pixel values if provided
    if adv_pixel_values is not None:
        base["pixel_values"] = adv_pixel_values
    
    if "pixel_values" not in base:
        raise RuntimeError("processor did not return pixel_values")
    return base

# ============================================================================
# Randomized Smoothing under Joint Attack
# ============================================================================

@torch.no_grad()
def _batchify_processor_inputs(base_inputs: dict, b: int) -> dict:
    """Replicate processor inputs to batch size b"""
    cur = {}
    for k, v in base_inputs.items():
        if torch.is_tensor(v):
            if v.size(0) == 1:
                cur[k] = v.repeat((b,) + (1,) * (v.dim() - 1))
            elif v.size(0) == b:
                cur[k] = v
            else:
                cur[k] = v
        elif isinstance(v, (list, tuple)):
            if len(v) == 1:
                cur[k] = list(v) * b
            elif len(v) == b:
                cur[k] = v
            else:
                cur[k] = v
        else:
            cur[k] = v
    return cur


# =========================
# Certify-style text kernels
# =========================

def _sample_uniform_except(V: int, forbidden: set[int], t: int) -> int:
    while True:
        r = random.randrange(V)
        if r != t and r not in forbidden:
            return r

def build_padded_ids(tokenizer, x: str, m_pad_tokens: int, pad_token_id: int):
    x_ids = tokenizer(x, add_special_tokens=False).input_ids
    base_ids = x_ids + [int(pad_token_id)] * int(m_pad_tokens)
    pad_start = len(x_ids)
    pad_end = len(base_ids)
    return base_ids, slice(pad_start, pad_end)

def noise_uniform_suffix_inplace(tokenizer, text: str, beta: float, d: int) -> str:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if d <= 0 or len(ids) == 0:
        return text
    d = min(int(d), len(ids))
    prefix, suffix = ids[:-d], ids[-d:]
    V = tokenizer.vocab_size
    forbidden = set(getattr(tokenizer, "all_special_ids", []))

    out = []
    for t in suffix:
        if random.random() < beta:
            out.append(_sample_uniform_except(V, forbidden, t))
        else:
            out.append(t)

    return tokenizer.decode(prefix + out, skip_special_tokens=True, clean_up_tokenization_spaces=False)

def noise_uniform_on_first_d_padding_tokens(tokenizer, base_ids, pad_span, beta: float, d: int) -> str:
    ids = list(base_ids)
    pad_len = pad_span.stop - pad_span.start
    if d <= 0 or pad_len <= 0:
        return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    d = min(int(d), int(pad_len))
    V = tokenizer.vocab_size
    forbidden = set(getattr(tokenizer, "all_special_ids", []))

    for pos in range(pad_span.start, pad_span.start + d):
        t = ids[pos]
        if random.random() < beta:
            ids[pos] = _sample_uniform_except(V, forbidden, t)

    return tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


# =========================
# 2) Add a BATCH builder (needed for RS)
# =========================

from typing import Dict

def build_chat_inputs_with_full_text_batch(
    detector,
    full_texts: List[str],      # already includes adv suffix if you want
    image_pil_list: List,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    processor = detector.processor
    text_prompts = []
    for full_text in full_texts:
        text_block = {"type": "text", "text": f"{POLICY_MULTI_MODAL}\n\nUSER TEXT:\n{full_text}\n"}
        conversation = [{"role": "user", "content": [{"type": "image"}, text_block]}]
        text_prompts.append(processor.apply_chat_template(conversation, add_generation_prompt=True))

    base = processor(images=image_pil_list, text=text_prompts, return_tensors="pt", padding=True)
    base = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in base.items()}
    if "pixel_values" not in base:
        raise RuntimeError("processor did not return pixel_values")
    return base



@torch.no_grad()
def rs_predict_hybrid_under_joint_attack(
    detector,
    *,
    prompt: str,
    image0_pil,                    # PIL, used for size/metadata
    adv_text: str,                 # attacked suffix (string)
    adv_image_pil,                 # PIL center of RS (the attacked image in pixel space)
    text_threat: str,              # "append_only" or "inplace_suffix"
    kernel: str,                   # "uniform" only (like certify)
    beta: float,
    m: int,                        # padding length for append_only
    d_list: List[int],
    sigma_pixel: float,            # sigma in pixel space (same meaning as certify)
    n: int,
    batch_size: int,
    alpha: float,
    device: torch.device,
    use_diffusion: bool = True,
    denoiser=None,                 # if use_diffusion: callable pil -> pil, i.e., denoiser(pil, sigma_pixel)
):
    """
    Option A (coherent with certify):
      - Text kernel K_d exactly like certify (append-only pad corruption or inplace suffix).
      - Image kernel in pixel space, either:
           (i) Gaussian: add_gaussian_noise(PIL, sigma_pixel)  [same as certify]
          (ii) Diffusion: denoiser(PIL, sigma_pixel)           [same as certify]
      - Forward uses the processor each time (like generate_full does, but logits-only path).
      - Decision rule is forced-choice Safe/Unsafe from last-token logits (your RS code path).
    """

    if kernel != "uniform":
        raise RuntimeError("Only kernel='uniform' is implemented for suffix/append-only here (same as certify).")

    model = detector.model
    tok = detector.tokenizer
    processor = detector.processor

    safe_id, _ = _single_token_id_any(tok, ["Safe", " safe", " Safe", "safe"])
    unsafe_id, _ = _single_token_id_any(tok, ["Unsafe", " unsafe", " Unsafe", "unsafe"])
    if safe_id is None or unsafe_id is None:
        raise RuntimeError("Could not find Safe/Unsafe token ids")

    # ------------------------------------------------------------------
    # Precompute center objects for text kernel (exactly certify logic)
    # ------------------------------------------------------------------
    if text_threat == "append_only":
        pad_token_id = getattr(tok, "pad_token_id", None) or getattr(tok, "eos_token_id", None)
        if pad_token_id is None:
            raise RuntimeError("Tokenizer has no pad_token_id/eos_token_id; cannot use append_only.")
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        pad_token_id = tok.pad_token_id

        base_ids, pad_span = build_padded_ids(tok, prompt, int(m), int(pad_token_id))
        adv_ids = tok(adv_text, add_special_tokens=False).input_ids

        base_ids_adv = list(base_ids)
        pad_len = pad_span.stop - pad_span.start
        L = min(len(adv_ids), pad_len)
        for j in range(L):
            base_ids_adv[pad_span.start + j] = int(adv_ids[j])

        def sample_text(d: int) -> str:
            return noise_uniform_on_first_d_padding_tokens(
                tok, base_ids_adv, pad_span, float(beta), int(d)
            )

    elif text_threat == "inplace_suffix":
        attacked_full_text = (prompt + " " + adv_text).strip()

        def sample_text(d: int) -> str:
            return noise_uniform_suffix_inplace(tok, attacked_full_text, float(beta), int(d))

    else:
        raise ValueError("text_threat must be 'append_only' or 'inplace_suffix'.")

    # ------------------------------------------------------------------
    # Image kernel in pixel space (exactly certify meaning)
    # ------------------------------------------------------------------
    if use_diffusion:
        if denoiser is None:
            raise RuntimeError("use_diffusion=True but denoiser is None.")
        # Must match certify: denoiser(image, sigma) returns a PIL-like image or tensor convertible by processor
        def sample_image():
            return denoiser(adv_image_pil, float(sigma_pixel))
    else:
        # Must match certify: add_gaussian_noise(image, sigma) returns PIL-like image
        def sample_image():
            return add_gaussian_noise(adv_image_pil, float(sigma_pixel))

    # ------------------------------------------------------------------
    # Helper: build processor batch matching the chat-template pipeline
    # used by build_chat_inputs_with_adv_text_batch.
    # ------------------------------------------------------------------
    def build_chat_inputs_with_full_text_batch(full_texts: List[str], images: List):
        text_prompts = []
        for full_text in full_texts:
            text_block = {
                "type": "text",
                "text": f"{POLICY_MULTI_MODAL}\n\nUSER TEXT:\n{full_text}\n",
            }
            conversation = [{"role": "user", "content": [{"type": "image"}, text_block]}]
            text_prompts.append(processor.apply_chat_template(conversation, add_generation_prompt=True))

        base = processor(images=images, text=text_prompts, return_tensors="pt", padding=True)
        base = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in base.items()}
        if "pixel_values" not in base:
            raise RuntimeError("processor did not return pixel_values")
        return base

    # ------------------------------------------------------------------
    # RS loop over d
    # ------------------------------------------------------------------
    results_d = []

    for d in d_list:
        counts = {"Safe": 0, "Unsafe": 0}
        done = 0

        while done < int(n):
            b = min(int(batch_size), int(n) - done)

            # sample K_d texts
            full_texts = [sample_text(int(d)) for _ in range(b)]

            # sample image kernel around attacked image (pixel space)
            ims = [sample_image() for _ in range(b)]

            # processor forward
            inputs = build_chat_inputs_with_full_text_batch(full_texts, ims)

            out = model(**inputs, use_cache=False)
            logits_last = out.logits[:, -1, :]
            labels = forced_choice_safe_unsafe_from_last_logits(logits_last, int(safe_id), int(unsafe_id))
            for lab in labels:
                counts[lab] += 1

            done += b

        # CP-LCB exactly like certify (direct counts)
        pred_label = "Unsafe" if counts["Unsafe"] >= counts["Safe"] else "Safe"
        nA = max(counts["Safe"], counts["Unsafe"])
        pA_lcb = clopper_pearson_lcb(int(n), int(nA), float(alpha))

        results_d.append(
            dict(
                d=int(d),
                pred_label=str(pred_label),
                counts=dict(counts),
                nA_majority=int(nA),
                pA_lcb=float(pA_lcb),
                beta=float(beta),
                sigma_pixel=float(sigma_pixel),
                text_threat=str(text_threat),
                kernel=str(kernel),
                use_diffusion=bool(use_diffusion),
            )
        )

    return results_d

# ============================================================================
# Main
# ============================================================================

def save_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def make_adv_text_init(tokenizer, adv_len: int, base_token: str = "!") -> str:
    ids = tokenizer(base_token, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError("base_token must map to exactly one token")
    return tokenizer.decode(ids * adv_len)

def forbid_initialization():
    def skip(*args, **kwargs):
        pass

    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def main():
    forbid_initialization()

    random.seed(0)
    torch.manual_seed(0)

    parser = argparse.ArgumentParser()

    # Dataset plumbing (same as certify)
    add_multimodal_dataset_args(
        parser,
        default_dataset="unsafebench_interaction_unsafe_mm",
        include_hateful_memes=False,
    )
    parser.add_argument("--max_examples", type=int, default=20)

    # Hybrid RS knobs (match certify)
    parser.add_argument("--kernel", choices=["uniform", "absorb"], default="uniform")
    parser.add_argument(
        "--text_threat",
        choices=["append_only", "inplace_suffix"],
        default="append_only",
    )
    parser.add_argument("--beta", type=float, default=0.25)
    parser.add_argument("--sigma", type=float, default=1.0)   # sigma in pixel space (as certify)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)

    # Threat-region sizes (match certify)
    parser.add_argument("--m", type=int, default=8, help="append_only padding length")
    parser.add_argument("--d_max", type=int, default=8, help="inplace_suffix max d")

    # Image kernel (match certify)
    parser.add_argument(
        "--use_diffusion",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable diffusion denoising (default: on); pass --no-use_diffusion to disable",
    )

    # Attack knobs
    parser.add_argument("--attack_steps", type=int, default=20)
    parser.add_argument("--attack_text_steps", type=int, default=1)
    parser.add_argument("--attack_image_steps", type=int, default=5)
    parser.add_argument("--attack_eps", type=float, default=0.3)
    parser.add_argument("--attack_step", type=float, default=0.05)
    parser.add_argument("--attack_adv_len", type=int, default=2)
    parser.add_argument("--attack_target_label", type=str, default="Safe", choices=["Safe", "Unsafe"])
    parser.add_argument("--attack_verbose", action="store_true")
    parser.add_argument("--attack_log_every", type=int, default=5)

    # Misc
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=str, default=None)

    args = parser.parse_args()

    if args.out is None:
        args.out = os.path.join(
            "results",
            f"rs_attack_then_rs_{args.dataset}_{args.text_threat}_{args.kernel}"
            f"_b{args.beta}_s{args.sigma}_n{args.n}_m{args.m}_dmax{args.d_max}"
            f"_diff_{args.use_diffusion}.json",
        )

    dev = torch.device(args.device)

    print("Building model...")
    detector = LLaVAGuardSafetyDetector()
    detector.model.to(dev)
    detector.model.eval()

    # Optional diffusion denoiser; passed through to rs_predict_hybrid_under_joint_attack.
    denoiser = None
    if args.use_diffusion:
        from hybrid_rs.utils_diffusion import Denoiser
        denoiser = Denoiser().eval().to(dev)
        print("Using diffusion denoiser for RS image kernel.")

    # Dataset loader exactly like certify
    print(f"Using dataset: {args.dataset}")
    loader = get_multimodal_dataset_loader_from_args(
        args,
        include_hateful_memes=False,
    )

    # Build attack init exactly from tokenizer (single-token base)
    adv_text_init = make_adv_text_init(detector.tokenizer, adv_len=int(args.attack_adv_len))
    print("adv_text_init:", repr(adv_text_init))

    results = []

    for i, ((text0, image0), _) in enumerate(loader):
        if i >= int(args.max_examples):
            break

        print(f"\n{'='*80}")
        print(f"Example {i}: {text0[:120]}...")
        print(f"{'='*80}")

        # 1) Attack: craft (adv_text, adv_image_pv4)
        print("[1/2] Running joint text-image attack...")
        attacker = JointTextImageAttack(
            detector=detector,
            original_image_pil=image0,
            prompt=text0,
            target_label=str(args.attack_target_label),
            adv_text_init=adv_text_init,
            num_steps=int(args.attack_steps),
            text_steps_per_iter=int(args.attack_text_steps),
            image_steps_per_iter=int(args.attack_image_steps),
            image_epsilon=float(args.attack_eps),
            image_step_size=float(args.attack_step),
            device=str(dev),
            verbose=bool(args.attack_verbose),
            log_every=int(args.attack_log_every),
        )

        adv_text, adv_image_pv4 = attacker.attack()          # adv_image_pv4: [1,3,H,W] (pixel_values space)
        
        adv_image_pil = pixel_values_to_pil(detector.processor, adv_image_pv4)

        # adv_image_pv4 = adv_image_pv4.to(dev)

        print("Attack complete.")
        print("adv_text:", repr(adv_text))

        # 2) RS: certify-compatible kernels, loop over d
        if args.text_threat == "append_only":
            d_max_eff = int(args.m)
        else:
            ids0 = detector.tokenizer(text0, add_special_tokens=False).input_ids
            d_max_eff = min(int(args.d_max), len(ids0))
        d_list = list(range(1, d_max_eff + 1))

        print(f"[2/2] RS prediction at attacked point over d=1..{d_max_eff} ...")

        rs_curve = rs_predict_hybrid_under_joint_attack(
            detector,
            prompt=text0,
            image0_pil=image0,
            adv_text=adv_text,
            adv_image_pil=adv_image_pil,          # <-- FIX
            text_threat=str(args.text_threat),
            kernel=str(args.kernel),
            beta=float(args.beta),
            m=int(args.m),
            d_list=d_list,
            sigma_pixel=float(args.sigma),
            n=int(args.n),
            batch_size=int(args.batch_size),
            alpha=float(args.alpha),
            device=dev,
            use_diffusion=bool(args.use_diffusion),
            denoiser=denoiser,
        )

        # Basic printout
        for rec_d in rs_curve:
            c = rec_d["counts"]
            print(
                f" d={rec_d['d']:2d} pred={rec_d['pred_label']:<6s} "
                f"counts(Safe={c['Safe']}, Unsafe={c['Unsafe']}) pA_lcb={rec_d['pA_lcb']:.6f}"
            )

        results.append(
            dict(
                example_id=int(i),
                dataset=str(args.dataset),
                policy="POLICY_MULTI_MODAL",
                kernel=str(args.kernel),
                text_threat=str(args.text_threat),
                beta=float(args.beta),
                sigma=float(args.sigma),
                n=int(args.n),
                alpha=float(args.alpha),
                m=int(args.m),
                d_max=int(args.d_max),
                d_max_effective=int(d_max_eff),
                use_diffusion=bool(args.use_diffusion),
                attack=dict(
                    target_label=str(args.attack_target_label),
                    steps=int(args.attack_steps),
                    text_steps_per_iter=int(args.attack_text_steps),
                    image_steps_per_iter=int(args.attack_image_steps),
                    image_epsilon=float(args.attack_eps),
                    image_step_size=float(args.attack_step),
                    adv_text_init=str(adv_text_init),
                    adv_text=str(adv_text),
                ),
                rs_curve=rs_curve,
            )
        )

    payload = dict(
        meta=dict(
            model="llava_guard",
            attack="joint_text_image_gcg_pgd",
            rs="hybrid_like_certify",
        ),
        results=results,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
