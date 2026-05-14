#!/usr/bin/env python3
import argparse
import logging
import os
from pathlib import Path

# from networkx import sigma
import torch
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm
import torch.nn as nn

from hybrid_rs.data import get_unsafebench_interaction_unsafe_mm_hf

try:
    from diffusion_denoised_smoothing.imagenet.guided_diffusion.script_util import (
        model_and_diffusion_defaults,
        create_model_and_diffusion,
        args_to_dict,
    )
except ImportError as _diffusion_import_error:
    raise ImportError(
        "Could not import `diffusion_denoised_smoothing`. This codebase depends on "
        "the upstream repo, which is not vendored here. Clone it at the root of "
        "this repository (so that `diffusion_denoised_smoothing/imagenet/...` is on "
        "your PYTHONPATH) before running anything that touches the diffusion "
        "denoiser:\n\n"
        "    git clone https://github.com/ethz-spylab/diffusion_denoised_smoothing\n"
    ) from _diffusion_import_error

from torchvision.utils import save_image, make_grid

logging.getLogger("transformers.image_utils").setLevel(logging.ERROR)


def _resolve_diffusion_ckpt() -> str:
    candidates = [
        os.environ.get("DIFFUSION_CKPT"),
        (
            str(Path(os.environ["HYBRID_RS_DATA_ROOT"]) / "256x256_diffusion_uncond.pt")
            if "HYBRID_RS_DATA_ROOT" in os.environ
            else None
        ),
        "imagenet/256x256_diffusion_uncond.pt",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    return str(Path("imagenet/256x256_diffusion_uncond.pt"))


PATH_TO_CKPT = _resolve_diffusion_ckpt()

class Args:
    image_size = 256
    num_channels = 256
    num_res_blocks = 2
    num_heads = 4
    num_heads_upsample = -1
    num_head_channels = 64
    attention_resolutions = "32,16,8"
    channel_mult = ""
    dropout = 0.0
    class_cond = False
    use_checkpoint = False
    use_scale_shift_norm = True
    resblock_updown = True
    use_fp16 = False
    use_new_attention_order = False
    clip_denoised = True
    learn_sigma = True
    diffusion_steps = 1000
    noise_schedule = "linear"
    timestep_respacing = None
    use_kl = False
    predict_xstart = False
    rescale_timesteps = False
    rescale_learned_sigmas = False

    # unused extras
    num_samples = 1
    batch_size = 1
    use_ddim = False
    model_path = ""
    classifier_path = ""
    classifier_scale = 1.0


def timestep_from_sigma(diffusion, sigma: float) -> int:
    if sigma <= 0:
        return 0
    a_all = diffusion.sqrt_alphas_cumprod
    b_all = diffusion.sqrt_one_minus_alphas_cumprod
    T = int(a_all.shape[0])

    t = 0
    real_sigma = 0.0
    while t + 1 < T and real_sigma < sigma:
        t += 1
        a = float(a_all[t])
        b = float(b_all[t])
        real_sigma = b / max(a, 1e-12)
    return int(t)


@torch.no_grad()
def denoise_dds(model, diffusion, x01: torch.Tensor, sigma: float):
    """
    x01: BCHW in [0,1], 256x256
    returns:
        x_t01  : noised image in [0,1]
        xhat01 : denoised image in [0,1]
    """
    device = next(model.parameters()).device
    x01 = x01.to(device)
    x0 = x01 * 2.0 - 1.0                      # [-1,1]

    t = timestep_from_sigma(diffusion, float(sigma))
    t_batch = torch.full((x0.shape[0],), t, device=device, dtype=torch.long)

    eps = torch.randn_like(x0)
    x_t = diffusion.q_sample(x_start=x0, t=t_batch, noise=eps)

    xhat0 = diffusion.p_sample(
        model, x_t, t_batch, clip_denoised=True
    )["pred_xstart"]

    # map back to [0,1]
    x_t01 = (x_t + 1.0) * 0.5
    xhat01 = (xhat0 + 1.0) * 0.5

    return x_t01.clamp(0.0, 1.0), xhat01.clamp(0.0, 1.0)


def init_diffusion_model(device="cuda"):
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(Args(), model_and_diffusion_defaults().keys())
    )
    sd = torch.load(PATH_TO_CKPT, map_location="cpu")
    model.load_state_dict(sd)
    model.eval().to(device)
    return model, diffusion


class Denoiser(nn.Module):
    def __init__(self, model_path=PATH_TO_CKPT):
        super().__init__()
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(Args(), model_and_diffusion_defaults().keys())
        )
        sd = torch.load(model_path, map_location="cpu")
        model.load_state_dict(sd)
        self.model = model
        self.diffusion = diffusion
        self.resize_crop = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
        ])
        self.transform = transforms.Compose([
            self.resize_crop,
            transforms.ToTensor(),
        ])
        
        self.to_tensor = transforms.ToTensor()

    def _prepare_bchw(self, x01):
        """
        Normalize supported inputs to BCHW tensors in [0, 1], resized/cropped to 256.
        Supports:
          - CHW torch.Tensor
          - BCHW torch.Tensor
          - single PIL image / ndarray accepted by ToTensor
          - list/tuple of CHW tensors or PIL images
        """
        if isinstance(x01, torch.Tensor):
            if x01.dim() == 3:
                x01 = x01.unsqueeze(0)
            elif x01.dim() != 4:
                raise ValueError(f"Expected CHW or BCHW tensor, got shape {tuple(x01.shape)}")
            return self.resize_crop(x01)

        if isinstance(x01, (list, tuple)):
            processed = []
            for item in x01:
                if isinstance(item, torch.Tensor):
                    if item.dim() != 3:
                        raise ValueError(
                            f"Expected CHW tensors inside sequence input, got shape {tuple(item.shape)}"
                        )
                    processed.append(self.resize_crop(item))
                else:
                    processed.append(self.transform(item))
            return torch.stack(processed, dim=0)

        return self.transform(x01).unsqueeze(0)


    def forward(self, x01: torch.Tensor, sigma: float):
        """
        x01: CHW or BCHW tensor in [0, 1], or a single PIL image.
        Returns: BCHW denoised images in [0, 1].
        """
        x01 = self._prepare_bchw(x01)
        _, xhat01 = denoise_dds(self.model, self.diffusion, x01, sigma)
        return xhat01
    
    def forward_to_tensor(self, x01: torch.Tensor, sigma: float):
        """
        assume there is batch dim 
        """
        x01 = self._prepare_bchw(x01)
        _, xhat01 = denoise_dds(self.model, self.diffusion, x01, sigma)
        return xhat01
    
    def get_xt_and_xhat(self, x01: torch.Tensor, sigma: float):
        x_t01, xhat01 = denoise_dds(self.model, self.diffusion, x01, sigma)
        return x_t01, xhat01

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str,
                        default="./data/unsafebench_interaction_unsafe_hf")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--max_examples", type=int, default=20)
    parser.add_argument("--sigma", type=float, default=0.5)
    parser.add_argument("--ckpt", type=str,
                        default=PATH_TO_CKPT)
    parser.add_argument("--out_dir", type=str, default="./denoiser_outputs")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # output folders
    orig_dir = os.path.join(args.out_dir, "original")
    den_dir = os.path.join(args.out_dir, f"denoised_sigma{args.sigma}")
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(den_dir, exist_ok=True)


    noised_dir = os.path.join(args.out_dir, f"noised_sigma{args.sigma}")
    os.makedirs(noised_dir, exist_ok=True)
    
    grid_dir = os.path.join(args.out_dir, f"grid_sigma{args.sigma}")
    os.makedirs(grid_dir, exist_ok=True)
        
    denoiser = Denoiser(model_path=args.ckpt)
    denoiser.eval().to(args.device)

    # image transform (must be 256x256)
    to_256 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])

    loader = get_unsafebench_interaction_unsafe_mm_hf(
        dataset_dir=args.dataset_dir,
        split=args.split,
        max_examples=args.max_examples,
        require_mm_unsafe=True,
    )

    for i, ((text0, pil_img), _) in enumerate(tqdm(loader, total=args.max_examples)):
        x = to_256(pil_img).unsqueeze(0)  # CPU

        x_t, xhat = denoiser.get_xt_and_xhat(x, sigma=args.sigma)  # returns on GPU

        # move to CPU for save_image / make_grid
        x_cpu = x.cpu()
        x_t_cpu = x_t.detach().cpu()
        xhat_cpu = xhat.detach().cpu()

        save_image(x_cpu,     os.path.join(orig_dir,   f"{i:06d}.png"))
        save_image(x_t_cpu,   os.path.join(noised_dir, f"{i:06d}.png"))
        save_image(xhat_cpu,  os.path.join(den_dir,    f"{i:06d}.png"))

        grid = make_grid(
            torch.cat([x_cpu, x_t_cpu, xhat_cpu], dim=0),
            nrow=3,
            padding=4,
            pad_value=1.0,
        )
        save_image(grid, os.path.join(grid_dir, f"{i:06d}.png"))

        if i + 1 >= args.max_examples:
            break

    print("Saved originals to:", orig_dir)
    print("Saved denoised to :", den_dir)
    print("DONE")


if __name__ == "__main__":
    main()
