import torch
from statsmodels.stats.proportion import proportion_confint
from PIL import Image
from scipy.special import comb
from torch.distributions.normal import Normal

_std_normal = Normal(0.0, 1.0)


# ------------------------------------------------------------
# Grouped discrete kernels (correct likelihood ratios)
# ------------------------------------------------------------

def grouped_absorb(d: int, beta: float, device="cpu"):
    v = beta ** d
    p_clean = torch.tensor([1 - v, v], dtype=torch.float64, device=device)
    p_adv   = torch.tensor([0.0, v], dtype=torch.float64, device=device)
    gamma   = torch.tensor([0.0, 1.0], dtype=torch.float64, device=device)
    return p_clean, p_adv, gamma

def grouped_uniform(d: int, beta: float, V: int, device="cpu"):
    alpha = beta / (V - 1)
    beta_bar = 1 - beta

    pc, pa, g = [], [], []
    for i in range(d + 1):
        for j in range(d + 1):
            if i + j < d:
                continue
            can = i + j - d
            cant = i - can
            num = binomial(d, i) * binomial(i, cant) * (V - 2) ** can
            pz_x = (alpha ** i) * (beta_bar ** (d - i))
            pz_xa = (alpha ** j) * (beta_bar ** (d - j))
            pc.append(pz_x * num)
            pa.append(pz_xa * num)
            g.append(0.0 if pz_x == 0 else pz_xa / pz_x)

    return (
        torch.tensor(pc, dtype=torch.float64, device=device),
        torch.tensor(pa, dtype=torch.float64, device=device),
        torch.tensor(g, dtype=torch.float64, device=device),
    )
    
    
# ------------------------------------------------------------
# Image noise (pixel space [0,1])
# ------------------------------------------------------------

try:
    from torchvision.transforms.functional import to_tensor, to_pil_image
except Exception:
    to_tensor = None
    to_pil_image = None
    
def add_gaussian_noise(image, sigma: float):
    if image is None or sigma <= 0:
        return image
    if to_tensor is None or to_pil_image is None:
        raise RuntimeError("torchvision is required for image noise (to_tensor/to_pil_image).")
    x = to_tensor(image)
    x = (x + sigma * torch.randn_like(x)).clamp(0, 1)
    return to_pil_image(x)

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

def random_noise_image(size=(224, 224)):
    x = torch.rand(3, *size)
    return Image.fromarray((255 * x.permute(1, 2, 0)).byte().numpy())




def clopper_pearson_lcb(n: int, nA: int, alpha: float) -> float:
    """
    One-sided lower confidence bound on Binomial proportion p with confidence 1-alpha.
    Uses two-sided beta interval with alpha' = 2*alpha.
    """
    return float(proportion_confint(nA, n, alpha=2 * alpha, method="beta")[0])

def binomial(n: int, m: int) -> int:
    return comb(n, m, exact=True)


# ------------------------------------------------------------
# Hybrid NP primitives
# ------------------------------------------------------------

def F_sigma(t, r, sigma, p_clean, gamma):
    eps = 1e-12
    t = torch.tensor(t, dtype=gamma.dtype, device=gamma.device)
    logt = torch.log(t + eps)
    base = 0.5 * r * r + sigma * sigma * (logt - torch.log(gamma + eps))
    arg = base / (sigma * r + eps)
    return (p_clean * _std_normal.cdf(arg)).sum().item()

def V_sigma(t, r, sigma, p_adv, gamma):
    eps = 1e-12
    t = torch.tensor(t, dtype=gamma.dtype, device=gamma.device)
    logt = torch.log(t + eps)
    base = 0.5 * r * r + sigma * sigma * (logt - torch.log(gamma + eps))
    arg = base / (sigma * r + eps) - r / (sigma + eps)
    return (p_adv * _std_normal.cdf(arg)).sum().item()

def solve_t_star(pA, r, sigma, p_clean, gamma):
    lo, hi = 1e-12, 1e12
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if F_sigma(mid, r, sigma, p_clean, gamma) >= pA:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)

def certify_r_hybrid(pA, tau, sigma, p_clean, p_adv, gamma, r_max):
    lo, hi = 0.0, float(r_max)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        t = solve_t_star(pA, mid, sigma, p_clean, gamma)
        v = V_sigma(t, mid, sigma, p_adv, gamma)
        if v >= tau:
            lo = mid
        else:
            hi = mid
    return float(lo)