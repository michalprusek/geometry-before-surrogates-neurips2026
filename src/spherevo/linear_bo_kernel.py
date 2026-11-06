"""Vendored SphericalLinearKernel from Doumont et al. 2025 (Linear BO).

Source: https://github.com/colmont/linear-bo, commit 6ee3bb6, file
``src/kernels/spherical_linear.py``. MIT License, Copyright (c) 2025 Colin Doumont;
the licence text is in LICENSES/linear-bo-MIT.txt (see THIRD_PARTY_NOTICES.md).
Changes from upstream: ``jaxtyping`` annotations removed, ``match`` rewritten as
``if``/``elif``, docstrings reflowed; no kernel math is altered.

Ref: Doumont, Fan, Maus, Gardner, Moss, Pleiss, "We Still Don't Understand
High-Dimensional Bayesian Optimization", arXiv:2512.00170 (2025).

Linear BO = a GP whose kernel maps box-normalized inputs through an inverse
stereographic projection onto a unit sphere and then applies a *linear*
(dot-product) kernel with learnable softmax weights on the constant and linear
terms. The spherical mapping mitigates the boundary-seeking behaviour of
box-supported BO. It is therefore itself a *geometry-aware, box-supported*
recipe — the natural fourth recipe for our controlled box-vs-sphere ablation.
"""

import math
from typing import Sequence

import gpytorch
import torch
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.models.exact_prediction_strategies import LinearPredictionStrategy
from gpytorch.priors.torch_priors import GammaPrior, LogNormalPrior
from linear_operator.operators import (
    LowRankRootLinearOperator,
    MatmulLinearOperator,
    RootLinearOperator,
)


def project_onto_unit_sphere(x):
    """Inverse stereographic projection of ``x`` onto the unit sphere.

    :param x: Input tensor of shape (..., N, D) to project.
    """
    x_sq_norm = x.square().sum(dim=-1, keepdim=True)
    x_ = torch.cat([2 * x, (x_sq_norm - 1.0)], dim=-1).mul(  # inverse stereographic projection
        1.0 / (1.0 + x_sq_norm)
    )
    return x_


def maybe_low_rank_root_lo(root):
    n, r = root.shape[-2:]
    if r >= n:
        return RootLinearOperator(root)
    else:
        return LowRankRootLinearOperator(root)


class SphericalLinearKernel(gpytorch.kernels.RBFKernel):
    r"""Apply a linear kernel after spherical projection (Doumont et al. 2025).

    :param ard_num_dims: The number of dimensions in the input space.
    :param prior: The hyperprior used for the lengthscales.
    :param bounds: The bounds of the input space. If a single (min, max) bound
        is given, it is used for all dimensions.
    """

    has_lengthscale = True

    def __init__(
        self,
        *,
        ard_num_dims: int,
        prior: str = "dsp_unscaled",
        bounds=(0.0, 1.0),
        batch_shape: torch.Size = torch.Size([]),
    ):
        if ard_num_dims == 1:
            raise ValueError(
                f"ard_num_dims must be equal to the dimensionality of the input data. Got {ard_num_dims}."
            )
        if isinstance(bounds[0], float):
            bounds = [(bounds[0], bounds[1])] * ard_num_dims

        if prior == "dsp_unscaled":
            lengthscale_prior = LogNormalPrior(
                loc=math.sqrt(2.0) + math.log(1) * 0.5, scale=math.sqrt(3.0)
            )  # DSP-like but no scaling by D
            lengthscale_constraint = GreaterThan(
                2.5e-2, transform=None, initial_value=lengthscale_prior.mode
            )
        elif prior == "gamma_3_6":
            lengthscale_prior = GammaPrior(3.0, 6.0)
            lengthscale_constraint = GreaterThan(
                1e-2, transform=None, initial_value=lengthscale_prior.mode
            )
        else:
            raise ValueError(f"Unknown prior {prior!r}")

        super().__init__(
            ard_num_dims=ard_num_dims,
            batch_shape=batch_shape,
            lengthscale_prior=lengthscale_prior,
            lengthscale_constraint=lengthscale_constraint,
        )

        # Create buffer for the center and length of each dimension in the space
        _dtype = self.raw_lengthscale.dtype
        _bounds = torch.tensor(bounds, dtype=_dtype)
        self.register_buffer("_mins", _bounds[..., 0])
        self.register_buffer("_maxs", _bounds[..., 1])
        self.register_buffer("_centers", (self._mins + self._maxs).div(2.0))
        assert torch.all(self._maxs > self._mins), f"Invalid bounds {bounds}."

        # Learnable coefficients "b_i" for constant and linear terms
        coeffs = torch.zeros(2, dtype=_dtype)
        self.register_parameter("raw_coeffs", torch.nn.Parameter(coeffs))

        # Global lengthscale "a"
        glob_ls = torch.zeros(1, dtype=_dtype)
        self.register_parameter("raw_glob_ls", torch.nn.Parameter(glob_ls))

    @property
    def coeffs(self) -> torch.Tensor:
        """The coefficients for the constant and linear terms"""
        return torch.nn.functional.softmax(self.raw_coeffs, dim=-1)

    @property
    def glob_ls(self) -> torch.Tensor:
        """The global lengthscale"""
        return torch.sigmoid(self.raw_glob_ls)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False, **params):  # noqa: D102
        x1_equal_x2 = torch.equal(x1, x2)

        # Make sure that we're within bounds
        assert torch.all(x1 <= self._maxs)
        assert torch.all(x1 >= self._mins)
        assert torch.all(x2 <= self._maxs)
        assert torch.all(x2 >= self._mins)

        # Get constants
        lengthscale = self.lengthscale
        max_sq_norm = (
            (self._maxs - self._mins)[..., None, :]  # Shape: (..., 1, D)
            .div(2.0 * lengthscale)
            .square()
            .sum(dim=-1, keepdim=True)  # Shape: (..., 1, 1)
        )
        glob_ls = torch.sqrt(self.glob_ls * max_sq_norm)  # O(\sqrt{D}) init

        # Center and scale inputs
        x1 = x1.sub(self._centers).div(lengthscale)
        x2 = x1 if x1_equal_x2 else x2.sub(self._centers).div(lengthscale)

        # Apply global lengthscale
        x1 = x1.div(glob_ls)
        x2 = x2.div(glob_ls)

        # Project the inputs onto the sphere
        x1_ = project_onto_unit_sphere(x1)
        x2_ = project_onto_unit_sphere(x2)

        # Sum up the (weighted) components for constant and linear terms
        terms = self.coeffs
        term0_sqrt = terms[0].sqrt()
        term1_sqrt = terms[1].sqrt()
        x1_ = torch.cat([x1_ * term1_sqrt, term0_sqrt.expand_as(x1_[..., :1])], dim=-1)
        if x1_equal_x2:
            kernel = maybe_low_rank_root_lo(x1_)
        else:
            x2_ = torch.cat([x2_ * term1_sqrt, term0_sqrt.expand_as(x2_[..., :1])], dim=-1)
            kernel = MatmulLinearOperator(x1_, x2_.mT)

        return kernel

    def prediction_strategy(self, train_inputs, train_prior_dist, train_labels, likelihood):
        # Allow for fast sampling
        return LinearPredictionStrategy(train_inputs, train_prior_dist, train_labels, likelihood)
