"""ArcCosine kernel for spherical Gaussian Processes.

Implements the ArcCosine Order 0 kernel: k(x,y) = 1 - arccos(x.y)/pi.
This is the only kernel used in the paper method (zero learnable parameters,
outperforms all Matern variants on S^d due to concentration of measure).

Reference: Cho & Saul (2009), Kernel Methods for Deep Learning.
"""

import math

import gpytorch
import torch
import torch.nn.functional as F


def _cosine_similarity(x1, x2, diag: bool) -> torch.Tensor:
    """Compute clamped cosine similarity between normalized inputs."""
    x1_norm = F.normalize(x1, p=2, dim=-1)
    x2_norm = F.normalize(x2, p=2, dim=-1)

    if diag:
        cos_sim = (x1_norm * x2_norm).sum(dim=-1)
    else:
        cos_sim = x1_norm @ x2_norm.transpose(-2, -1)

    return cos_sim.clamp(-1 + 1e-6, 1 - 1e-6)


class ArcCosineKernel(gpytorch.kernels.Kernel):
    """ArcCosine kernel order 0 for unit sphere data.

    k(x, y) = 1 - arccos(x.y) / pi

    Analogous to Matern-1/2 on R^d (rough, like Brownian motion paths).
    Zero learnable parameters -- parameterless > learnable kernels on S^d.
    """

    has_lengthscale = False

    def forward(self, x1, x2, diag=False, **params):
        cos_sim = _cosine_similarity(x1, x2, diag)
        return 1.0 - torch.arccos(cos_sim) / math.pi


def create_kernel(
    kernel_type: str = "arccosine",
    *,
    kernel_order: int = 0,
    use_scale: bool = True,
) -> gpytorch.kernels.Kernel:
    """Create an ArcCosine kernel, optionally wrapped in ScaleKernel.

    Args:
        kernel_type: Kernel type (only "arccosine" supported; accepted for API
            compatibility with the solver module).
        kernel_order: ArcCosine order (only 0 supported; accepted for API
            compatibility).
        use_scale: Wrap in ScaleKernel (adds one learnable outputscale).

    Returns:
        GPyTorch kernel instance.
    """
    if kernel_type != "arccosine":
        raise ValueError(f"Unsupported kernel type: {kernel_type}. Only 'arccosine' is available.")
    base = ArcCosineKernel()
    if use_scale:
        return gpytorch.kernels.ScaleKernel(base)
    return base
