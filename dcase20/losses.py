# losses.py
# ===========================================
# Custom Losses and Diffusion Utilities
# ===========================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ===========================================
# 1. Multi-scale Maximum Mean Discrepancy (MMD)
# ===========================================
def mmd_rbf(x, y, sigma_list=[2.0, 5.0, 10.0]):
    """
    Compute RBF-kernel Maximum Mean Discrepancy (MMD) between x and y.
    Args:
        x, y: Tensor [B, C, ...] (will be flattened to [B, -1])
        sigma_list: list of kernel bandwidths
    """
    def _kernel(a, b, s):
        a = a.view(a.size(0), -1)
        b = b.view(b.size(0), -1)
        dist2 = torch.cdist(a, b) ** 2
        return torch.exp(-dist2 / (2.0 * s * s))

    mmd = 0.0
    for s in sigma_list:
        k_xx = _kernel(x, x, s).mean()
        k_yy = _kernel(y, y, s).mean()
        k_xy = _kernel(x, y, s).mean()
        mmd = mmd + (k_xx + k_yy - 2.0 * k_xy)
    return mmd


def pyramid_mmd(feats_fake_list, feats_real_list, w=[1.0, 0.7, 0.5]):
    """
    Multi-layer pyramid MMD loss.
    Args:
        feats_fake_list / feats_real_list: list of Tensors (multi-level features from Discriminator)
        w: layer weights
    """
    loss = 0.0
    for i, (f_fake, f_real) in enumerate(zip(feats_fake_list, feats_real_list)):
        wi = w[i] if i < len(w) else 1.0
        loss = loss + wi * mmd_rbf(f_fake, f_real)
    return loss


# ===========================================
# 2. Diffusion Utilities
# ===========================================
def make_beta_schedule(T=1000, schedule='linear', s=0.008, beta_start=1e-4, beta_end=2e-2, device='cpu'):
    """
    Generate beta schedule for diffusion process.
    - linear: beta from beta_start to beta_end
    - cosine: Nichol & Dhariwal cosine schedule
    """
    if schedule == 'linear':
        betas = torch.linspace(beta_start, beta_end, T, device=device)
    elif schedule == 'cosine':
        steps = T + 1
        t = torch.linspace(0, T, steps, device=device) / T
        f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = f / f[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(1e-8, 0.999)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")
    return betas


class DiffusionHelper:
    """
    Precompute useful diffusion quantities for fast lookup.
    """
    def __init__(self, betas: torch.Tensor):
        device = betas.device
        self.T = betas.numel()
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.ones(1, device=device), self.alphas_cumprod[:-1]], dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

    @staticmethod
    def extract(a: torch.Tensor, t: torch.Tensor, x_shape):
        """
        Extract values from a vector a at indices t, reshape to [B,1,1,...] for broadcasting.
        """
        out = a.gather(0, t)
        return out.view(-1, *([1] * (len(x_shape) - 1)))


def sample_timesteps(batch_size: int, T: int, device):
    """
    Randomly sample timesteps for training.
    """
    return torch.randint(low=0, high=T, size=(batch_size,), device=device, dtype=torch.long)


def q_sample(x0: torch.Tensor, t: torch.Tensor, helper: DiffusionHelper, noise: torch.Tensor = None):
    """
    Forward diffusion process: sample x_t from x0 at timestep t.
    Returns: (x_t, noise)
    """
    if noise is None:
        noise = torch.randn_like(x0)
    sqrt_ac = DiffusionHelper.extract(helper.sqrt_alphas_cumprod, t, x0.shape)
    sqrt_om = DiffusionHelper.extract(helper.sqrt_one_minus_alphas_cumprod, t, x0.shape)
    x_t = sqrt_ac * x0 + sqrt_om * noise
    return x_t, noise


@torch.no_grad()
def ddim_step(x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor,
              eps_pred: torch.Tensor, helper: DiffusionHelper, eta: float = 0.0, noise: torch.Tensor = None):
    """
    One step of DDIM sampling: predict x_{t_prev} from x_t.
    """
    if noise is None:
        noise = torch.randn_like(x_t)

    a_t = DiffusionHelper.extract(helper.alphas_cumprod, t, x_t.shape)
    a_prev = DiffusionHelper.extract(helper.alphas_cumprod, t_prev, x_t.shape)

    sqrt_a_t = torch.sqrt(a_t)
    sqrt_one_minus_a_t = torch.sqrt(1.0 - a_t)

    # Predict x0
    x0_pred = (x_t - sqrt_one_minus_a_t * eps_pred) / (sqrt_a_t + 1e-8)

    # Compute sigma_t (stochasticity term)
    sigma_t = eta * torch.sqrt(torch.clamp((1 - a_prev) / (1 - a_t) * (1 - a_t / a_prev), min=0.0))

    # Direction term
    dir_xt = torch.sqrt(torch.clamp(1 - a_prev - sigma_t ** 2, min=0.0)) * eps_pred

    x_prev = torch.sqrt(a_prev) * x0_pred + dir_xt + sigma_t * noise
    return x_prev, x0_pred


@torch.no_grad()
def ddim_refine(x0: torch.Tensor,
                netU, cond_feats, helper: DiffusionHelper,
                steps: int = 15, eta: float = 0.0,
                guidance_fn=None, guidance_scale: float = 0.0):
    """
    Refine G's output x0 using DDIM for K steps.
    Args:
        netU: DiffusionRefiner network, interface eps = netU(x_t, t, cond_feats)
        cond_feats: list of features from Discriminator
        helper: DiffusionHelper instance
        steps: number of DDIM steps
        eta: stochasticity
        guidance_fn: optional guidance function, g = guidance_fn(x_t)
        guidance_scale: weight for guidance
    Returns:
        x_refined: refined sample
        diff_proxy: residual proxy score
    """
    B = x0.size(0)
    device = x0.device
    T = helper.T

    # Evenly spaced time steps
    t_seq = torch.linspace(T - 1, 0, steps + 1, device=device).long()
    t_pairs = list(zip(t_seq[:-1], t_seq[1:]))

    # Start from noisy x_t
    t_start = t_pairs[0][0]
    t_start_batch = t_start.repeat(B)
    x_t, _ = q_sample(x0, t_start_batch, helper)

    diff_proxy = 0.0

    for t_cur, t_prev in t_pairs:
        t_b = t_cur.repeat(B)
        eps_pred = netU(x_t, t_b, cond_feats)

        # Optional guidance
        if guidance_fn is not None and guidance_scale > 0.0:
            x_t.requires_grad_(True)
            score = guidance_fn(x_t)
            if score.ndim > 0:
                score = score.mean()
            g = torch.autograd.grad(score, x_t, retain_graph=False, create_graph=False)[0]
            eps_pred = eps_pred - guidance_scale * g
            x_t = x_t.detach()

        diff_proxy = diff_proxy + eps_pred.abs().mean().item()

        t_prev_b = t_prev.repeat(B)
        x_t, _ = ddim_step(x_t, t_b, t_prev_b, eps_pred, helper, eta=eta, noise=None)

    x_refined = x_t
    return x_refined, torch.tensor(diff_proxy, device=device)


# ===========================================
# Exported symbols
# ===========================================
__all__ = [
    "mmd_rbf", "pyramid_mmd",
    "make_beta_schedule", "DiffusionHelper",
    "sample_timesteps", "q_sample",
    "ddim_step", "ddim_refine",
]
