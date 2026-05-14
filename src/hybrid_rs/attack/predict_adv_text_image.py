#!/usr/bin/env python3
import json
import logging
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from statsmodels.stats.proportion import proportion_confint
from scipy.stats import norm
from typing import List, Tuple, Optional

from hybrid_rs.models import POLICY_MULTI_MODAL

logging.getLogger("transformers.image_utils").setLevel(logging.ERROR)


# attack is defined like this
# [policy]∥[USER TEXT]∥[adv_text].
# Length: it is entirely determined by adv_text_inits

# ============================================================================
# Utility functions
# ============================================================================

def clopper_pearson_lcb(n: int, nA: int, alpha: float) -> float:
    n = int(n * 10E5)
    nA = int(nA * 10E5)
    return float(proportion_confint(nA, n, alpha=2 * alpha, method="beta")[0])


def gaussian_rs_radius(pA: float, sigma: float) -> float:
    if pA <= 0.0:
        return float("-inf")
    if pA >= 1.0:
        return float("inf")
    return float(sigma * norm.ppf(pA))


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
# Joint Text-Image Attack (GCG + L2 PGD)
# ============================================================================

class JointTextImageAttack:
    def __init__(
        self,
        detector,
        original_image_pil,
        prompt: str,
        target_label: str = "Safe",  # Attack towards "Safe" to evade detection
        adv_text_init: str = "! " * 20,
        num_steps: int = 200,
        text_steps_per_iter: int = 1,
        image_steps_per_iter: int = 5,
        text_batch_size: int = 256,
        text_topk: int = 256,
        image_epsilon: float = 0.3,
        image_step_size: float = 0.05,
        device: str = "cuda",
        verbose: bool = False,
        log_every: int = 20,
    ):
        self.detector = detector
        self.original_image_pil = original_image_pil
        self.prompt = prompt
        self.target_label = target_label
        self.adv_text_init = adv_text_init
        self.num_steps = num_steps
        self.text_steps_per_iter = text_steps_per_iter
        self.image_steps_per_iter = image_steps_per_iter
        
        self.text_batch_size = text_batch_size
        self.text_topk = text_topk
        
        self.image_epsilon = image_epsilon
        self.image_step_size = image_step_size
        
        self.device = torch.device(device)
        self.verbose = verbose
        self.log_every = log_every
        
        self.model = detector.model
        self.tokenizer = detector.tokenizer
        self.processor = detector.processor
        
        # Get target token IDs
        safe_id, _ = _single_token_id_any(self.tokenizer, ["Safe", " safe", " Safe", "safe"])
        unsafe_id, _ = _single_token_id_any(self.tokenizer, ["Unsafe", " unsafe", " Unsafe", "unsafe"])
        if safe_id is None or unsafe_id is None:
            raise RuntimeError("Could not find Safe/Unsafe token ids")
        
        if target_label == "Safe":
            self.target_id = int(safe_id)
            self.other_id = int(unsafe_id)
        else:
            self.target_id = int(unsafe_id)
            self.other_id = int(safe_id)
        
        # Initialize
        self.adv_text = adv_text_init
        self.adv_image_pv = None
        self.original_image_pv = None
        
        self.not_allowed_tokens = get_nonascii_toks(self.tokenizer, device=self.device)
    
    def _expand_to_two_slots(self, pv4: torch.Tensor) -> torch.Tensor:
        # pv4: [B,3,H,W] -> [B,2,3,H,W]
        return pv4.unsqueeze(1).repeat(1, 2, 1, 1, 1)
    
    def _pv5(self) -> torch.Tensor:
        # returns a VIEW: [B,2,3,H,W] without allocating
        return self.adv_image_pv4.unsqueeze(1).expand(-1, 2, -1, -1, -1)

    def initialize_adversarial_image(self):
        """Initialize adversarial image from original image"""
        base = build_chat_inputs_with_adv_text(
            self.detector, 
            user_text=self.prompt,
            adv_text=self.adv_text,
            image_pil=self.original_image_pil,
            adv_pixel_values=None,
            device=self.device
        )
        # self.adv_image_pv = base["pixel_values"].detach().clone()
        # self.original_image_pv = self.adv_image_pv.clone()
        
        pv = base["pixel_values"].detach()          # [1,2,3,H,W]
        self.original_image_pv4 = pv[:, 0].clone()  # [1,3,H,W]
        self.adv_image_pv4 = self.original_image_pv4.clone()
        
        # pv = base["pixel_values"].detach()          # [1,2,3,H,W]
        # self.original_image_pv4 = pv[:, 0].clone()  # [1,3,H,W]
        # self.adv_image_pv4 = self.original_image_pv4.clone()
    
    def optimize_text_step(self) -> Tuple[float, float]:
        """One GCG text optimization step. Returns (loss, margin)"""
        # Get current text tokens
        full_text_without_adv = self.prompt
        adv_text_ids = self.tokenizer(self.adv_text, add_special_tokens=False).input_ids
        
        if len(adv_text_ids) == 0:
            # Fallback to empty optimization
            return 0.0, 0.0
        
        # Build inputs with current adversarial image and text
        inputs = build_chat_inputs_with_adv_text(
            self.detector,
            user_text=self.prompt,
            adv_text=self.adv_text,
            image_pil=self.original_image_pil,
            # adv_pixel_values=self.adv_image_pv,
            adv_pixel_values=self._pv5(),
            device=self.device
        )
        
        # Find where adversarial text tokens are in the full sequence
        full_ids = inputs["input_ids"].squeeze(0)
        prompt_ids = self.tokenizer(
            f"{POLICY_MULTI_MODAL}\n\nUSER TEXT:\n{self.prompt} ",
            add_special_tokens=False
        ).input_ids
        
        # Locate adv_text in full_ids (it comes after prompt)
        # This is approximate - we use the length
        adv_start = len(prompt_ids)
        adv_end = adv_start + len(adv_text_ids)
        adv_slice = slice(adv_start, adv_end)
        
        # Compute gradients
        coordinate_grad = self.compute_token_gradients(inputs, adv_slice, full_ids)
        
        # Sample and evaluate candidates
        adv_text_tokens = torch.tensor(adv_text_ids, device=self.device)
        new_adv_text, loss, margin = self.sample_and_evaluate_text(
            adv_text_tokens, coordinate_grad, inputs
        )
        
        self.adv_text = new_adv_text
        return loss, margin
    
    def compute_token_gradients(self, inputs: dict, adv_slice: slice, full_ids: torch.Tensor) -> torch.Tensor:
        """Compute gradients w.r.t. adversarial token positions"""
        # Get embeddings
        embed_layer = self.model.get_input_embeddings()
        embed_weights = embed_layer.weight
        
        adv_tokens = full_ids[adv_slice]
        
        # One-hot encoding
        one_hot = torch.zeros(
            adv_tokens.shape[0],
            embed_weights.shape[0],
            device=self.device,
            dtype=embed_weights.dtype
        )
        one_hot.scatter_(1, adv_tokens.unsqueeze(1), 1.0)
        one_hot.requires_grad_(True)
        
        # Compute embeddings with one-hot
        adv_embeds = one_hot @ embed_weights
        
        # Get full embeddings
        with torch.no_grad():
            full_embeds = embed_layer(full_ids.unsqueeze(0))
        
        # Stitch together
        full_embeds_adv = torch.cat([
            full_embeds[:, :adv_slice.start, :],
            adv_embeds.unsqueeze(0),
            full_embeds[:, adv_slice.stop:, :]
        ], dim=1)
        
        
        # Forward pass
        mm_kwargs = {}
        if "image_sizes" in inputs:
            mm_kwargs["image_sizes"] = inputs["image_sizes"]
        if "image_grid_thw" in inputs:          # depends on transformers version
            mm_kwargs["image_grid_thw"] = inputs["image_grid_thw"]
        if "pixel_attention_mask" in inputs:    # sometimes present
            mm_kwargs["pixel_attention_mask"] = inputs["pixel_attention_mask"]

        outputs = self.model(
            inputs_embeds=full_embeds_adv,
            attention_mask=inputs.get("attention_mask"),
            pixel_values=inputs.get("pixel_values"),
            **mm_kwargs,
        )

       
        # Loss: cross-entropy towards target token
        logits_last = outputs.logits[:, -1, :]
        target_tensor = torch.tensor([self.target_id], device=self.device)
        loss = F.cross_entropy(logits_last, target_tensor)
        
        loss.backward()
        
        grad = one_hot.grad.clone()
        grad = grad / (grad.norm(dim=-1, keepdim=True) + 1e-12)
        
        return grad
    
    def sample_and_evaluate_text(
        self, 
        adv_tokens: torch.Tensor,
        grad: torch.Tensor,
        current_inputs: dict
    ) -> Tuple[str, float, float]:
        """Sample text candidates and return best one with its loss and margin"""
        # Sample candidates
        new_token_candidates = self.sample_control(
            adv_tokens, grad, self.text_batch_size, topk=self.text_topk
        )
        
        # Decode candidates
        candidates = get_filtered_cands(
            self.tokenizer, new_token_candidates, filter_cand=False
        )
        
        # Evaluate a subset
        best_candidate = self.adv_text
        best_loss = float('inf')
        best_margin = float('-inf')
        
        eval_batch = candidates[:min(len(candidates), 32)]
        
        for candidate in eval_batch:
            inputs = build_chat_inputs_with_adv_text(
                self.detector,
                user_text=self.prompt,
                adv_text=candidate,
                image_pil=self.original_image_pil,
                # adv_pixel_values=self.adv_image_pv,
                adv_pixel_values=self._pv5(),   # FIX: was self.adv_image_pv (wrong)
                device=self.device
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits_last = outputs.logits[:, -1, :]
                
                # Loss
                target_tensor = torch.tensor([self.target_id], device=self.device)
                loss = F.cross_entropy(logits_last, target_tensor)
                
                # Margin
                margin = float((logits_last[0, self.target_id] - logits_last[0, self.other_id]).item())
                
                if loss < best_loss:
                    best_loss = loss.item()
                    best_margin = margin
                    best_candidate = candidate
        
        return best_candidate, best_loss, best_margin
    
    def sample_control(
        self, 
        control_toks: torch.Tensor, 
        grad: torch.Tensor, 
        batch_size: int, 
        topk: int = 256
    ) -> torch.Tensor:
        """
        Sample new token candidates based on gradient

        control_toks: current adversarial suffix, expressed as token IDs.

        batch size is unrelated to the model batch size or to RS. 
        It is an algorithmic hyper-parameter of GCG-style attacks: search breadth per iteration.
        
        sample multiple discrete candidates from this gradient signal. Each candidate is a full adversarial suffix of the same length, but with one (or a few) token positions changed according to the gradient
        
        fais penser à CLVQ ...
        """
        grad = grad.clone()

        # Ban tokens
        grad[:, self.not_allowed_tokens] = float("inf")

        # top_indices[pos, j] = j-th best token id for position pos
        top_indices = (-grad).topk(topk, dim=1).indices  # [L, topk], L = len(control_toks)

        L = control_toks.numel()
        if L == 0:
            return control_toks.unsqueeze(0).repeat(batch_size, 1)

        # Choose a position to change for each candidate
        pos = torch.randint(0, L, (batch_size,), device=self.device)  # [B]

        # Choose which of the top-k tokens to use for each candidate
        ksel = torch.randint(0, topk, (batch_size,), device=self.device)  # [B]

        # Gather the replacement token ids
        new_vals = top_indices[pos, ksel]  # [B]

        # Apply the single-coordinate replacement
        new_control = control_toks.unsqueeze(0).repeat(batch_size, 1)  # [B, L]
        new_control[torch.arange(batch_size, device=self.device), pos] = new_vals

        return new_control
    
    def optimize_image_step(self) -> Tuple[float, float]:
        """One L2 PGD image optimization step. Returns (loss, margin)"""
        # Build inputs
        self.adv_image_pv4 = self.adv_image_pv4.detach().requires_grad_(True)

        inputs = build_chat_inputs_with_adv_text(
            self.detector,
            user_text=self.prompt,
            adv_text=self.adv_text,
            image_pil=self.original_image_pil,
            # adv_pixel_values=self.adv_image_pv,
            adv_pixel_values=self.adv_image_pv4.unsqueeze(1).expand(-1, 2, -1, -1, -1),
            device=self.device
        )
        
        # self.adv_image_pv.requires_grad_(True)
        
        # Forward
        outputs = self.model(**inputs)
        logits_last = outputs.logits[:, -1, :]
        # Loss
        target_tensor = torch.tensor([self.target_id], device=self.device)
        loss = F.cross_entropy(logits_last, target_tensor)
        
        # Backward
        # grad = torch.autograd.grad(loss, self.adv_image_pv)[0]
        
        (grad4,) = torch.autograd.grad(loss, self.adv_image_pv4, retain_graph=False, create_graph=False)

        
        # Margin for logging
        with torch.no_grad():
            margin = float((logits_last[0, self.target_id] - logits_last[0, self.other_id]).item())
        
        # Convert epsilon to pixel_values space
        std_min = min(self.processor.image_processor.image_std)
        eps_pv = self.image_epsilon / std_min
        step_pv = self.image_step_size / std_min
        
        # PGD step
        with torch.no_grad():
            g_norm = grad4.view(grad4.size(0), -1).norm(p=2, dim=1).clamp_min(1e-12)
            g_norm = g_norm.view(-1, 1, 1, 1)

            # gradient step in 4D
            adv4 = self.adv_image_pv4 - step_pv * grad4 / g_norm

            # L2 projection in 4D
            delta = (adv4 - self.original_image_pv4).view(adv4.size(0), -1)
            delta_norm = delta.norm(p=2, dim=1).clamp_min(1e-12)
            factor = (eps_pv / delta_norm).clamp(max=1.0).view(-1, 1, 1, 1)

            self.adv_image_pv4 = (self.original_image_pv4 + (adv4 - self.original_image_pv4) * factor).detach()
        # with torch.no_grad():
        #     g_norm = grad4.view(grad4.size(0), -1).norm(p=2, dim=1).clamp_min(1e-12)
        #     g_norm = g_norm.view(-1, *([1] * (grad4.dim() - 1)))
            
        #     self.adv_image_pv = (self.adv_image_pv - step_pv * grad4 / g_norm).detach()
            
        #     # L2 projection
        #     delta = (self.adv_image_pv - self.original_image_pv).view(self.adv_image_pv.size(0), -1)
        #     delta_norm = delta.norm(p=2, dim=1).clamp_min(1e-12)
        #     factor = (eps_pv / delta_norm).clamp(max=1.0)
        #     factor = factor.view(-1, *([1] * (self.adv_image_pv.dim() - 1)))
            
        #     self.adv_image_pv = self.original_image_pv + (self.adv_image_pv - self.original_image_pv) * factor
        #     self.adv_image_pv = self.adv_image_pv.detach()
        
        return loss.item(), margin
    
    
    def attack(self) -> Tuple[str, torch.Tensor]:
        self.initialize_adversarial_image()

        if self.verbose:
            print(f"Starting joint attack for {self.num_steps} iterations...")

        for step in range(self.num_steps):

            # --- text block update ---
            if self.text_steps_per_iter > 0:
                for _ in range(self.text_steps_per_iter):
                    text_loss, text_margin = self.optimize_text_step()
            else:
                text_loss, text_margin = 0.0, 0.0

            # --- image block update ---
            if self.image_steps_per_iter > 0:
                for _ in range(self.image_steps_per_iter):
                    image_loss, image_margin = self.optimize_image_step()
            else:
                image_loss, image_margin = 0.0, 0.0

            # --- DEBUG / TRACKING (HERE) ---
            if self.verbose and (step % self.log_every == 0 or step == self.num_steps - 1):

                inputs = build_chat_inputs_with_adv_text(
                    self.detector,
                    user_text=self.prompt,
                    adv_text=self.adv_text,
                    image_pil=self.original_image_pil,
                    # adv_pixel_values=self.adv_image_pv,
                    adv_pixel_values=self._pv5(),
                    device=self.device,
                )

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits_last = outputs.logits[:, -1, :]

                    pred_id = logits_last.argmax(dim=-1).item()
                    tok = self.tokenizer.decode([pred_id])

                    margin = (
                        logits_last[0, self.target_id]
                        - logits_last[0, self.other_id]
                    ).item()

                print(
                    f"Step {step:3d}/{self.num_steps}: "
                    f"text_loss={text_loss:.4f} text_margin={text_margin:+.3f} | "
                    f"image_loss={image_loss:.4f} image_margin={image_margin:+.3f} | "
                    f"pred={repr(tok)} margin={margin:+.3f}"
                )

        return self.adv_text, self.adv_image_pv4.detach()


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


@torch.no_grad()
def rs_predict_under_joint_attack(
    detector,
    base_inputs: dict,
    adv_image_pv4: torch.Tensor,   # [1,3,H,W]
    sigma_pixel: float,
    n: int,
    batch_size: int,
    alpha: float,
):
    """
    Randomized smoothing prediction around the adversarial (text+image) example.
    
    Returns: (pred_label, nA_pred, pA_lcb, radius_L2, n_safe, n_unsafe)
    """
    model = detector.model
    processor = detector.processor
    tokenizer = detector.tokenizer
    dev = adv_image_pv4.device

    safe_id, _ = _single_token_id_any(tokenizer, ["Safe", " safe", " Safe", "safe"])
    unsafe_id, _ = _single_token_id_any(tokenizer, ["Unsafe", " unsafe", " Unsafe", "unsafe"])
    if safe_id is None or unsafe_id is None:
        raise RuntimeError("Could not find Safe/Unsafe token ids")

    sigma_pv = sigma_pixel_to_pixel_values(processor, sigma_pixel, device=dev, dtype=adv_image_pv4.dtype)

    count_safe = 0
    count_unsafe = 0

    for start in tqdm(range(0, n, batch_size), desc="RS sampling"):
        b = min(batch_size, n - start)
        cur = _batchify_processor_inputs(base_inputs, b)

        # Add Gaussian noise to the ADVERSARIAL image
        # noise = torch.randn((b,) + tuple(adv_image_pv.shape[1:]), device=dev, dtype=adv_image_pv4.dtype) * sigma_pv
        # x_rep = adv_image_pv.repeat((b,) + (1,) * (adv_image_pv.dim() - 1))
        # cur["pixel_values"] = x_rep + noise
        
        # Repeat center in 4D, add noise in 4D, then expand to 5D WITHOUT copy
        x4 = adv_image_pv4
        if x4.size(0) == 1 and b > 1:
            x4 = x4.repeat(b, 1, 1, 1)

        noise4 = torch.randn_like(x4) * sigma_pv
        x4_noised = x4 + noise4

        cur["pixel_values"] = x4_noised.unsqueeze(1).expand(-1, 2, -1, -1, -1)


        out = model(**cur)
        logits_last = out.logits[:, -1, :]
        labels = forced_choice_safe_unsafe_from_last_logits(logits_last, safe_id, unsafe_id)

        count_unsafe += sum(1 for t in labels if t == "Unsafe")
        count_safe += b - sum(1 for t in labels if t == "Unsafe")

    # Determine majority label
    if count_unsafe >= count_safe:
        pred_label = "Unsafe"
        nA = count_unsafe
    else:
        pred_label = "Safe"
        nA = count_safe

    pA_lcb = clopper_pearson_lcb(n, nA, alpha)
    R = gaussian_rs_radius(pA_lcb, float(sigma_pixel))
    
    return pred_label, int(nA), float(pA_lcb), float(R), int(count_safe), int(count_unsafe)


def save_json(path: str, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def make_adv_text_init(tokenizer, adv_len: int, base_token: str = "!") -> str:
    ids = tokenizer(base_token, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError("base_token must map to exactly one token")
    return tokenizer.decode(ids * adv_len)


