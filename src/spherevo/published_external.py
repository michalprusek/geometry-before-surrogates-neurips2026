# Copyright (c) 2019 Uber Technologies, Inc. (TuRBO, https://github.com/uber-research/TuRBO)
#
# Licensed under the Uber Non-Commercial License (the "License"); you may not use this file except in
# compliance with the License. A verbatim copy of the License is in LICENSES/turbo-LICENSE.txt. See the
# License for the specific language governing permissions and limitations under the License, including
# the disclaimer of warranty and the limitation of liability (its points 6 and 7), which apply to this file.
#
# This file contains a modified version of turbo/turbo_1.py and turbo/gp.py of TuRBO (commit de0db39). It is
# therefore distributed under the License above, for non-commercial purposes only; the MIT licence of the
# rest of this repository does not apply to it. Changed by the authors of this repository on 2026-10-13:
# the TuRBO-1 code was rewritten as an ask/tell optimiser; an initial design evaluated outside the optimiser
# can replace the Latin hypercube design; the search runs on the unit cube of a fixed problem box; a failed
# step proposes a uniform point instead of ending the run. The BAxUS part follows the BoTorch tutorial
# (MIT, LICENSES/botorch-MIT.txt); the CMA-ES part is the authors' own code.

"""Published external baselines as ask/tell optimisers (camera-ready, published-recipe block).

Three methods, each a port of the authors' released code, used by the LassoBench methods
``turbo_published`` / ``baxus_official`` / ``cmaes_full_published`` (``scripts/run_lassobench.py``),
the GuacaMol arm ``turbo_published`` (``src/spherevo/published_arms.py``) and the verification
harness (``scripts/cr/verify_baselines.py``).

``Turbo1Published``  TuRBO-1, Eriksson et al., NeurIPS 2019: a line-by-line port of the official
                     ``uber-research/TuRBO`` ``turbo/turbo_1.py`` and ``turbo/gp.py`` (local copy
                     ``cr/baselines/src/TuRBO``), on the unit cube of a FIXED problem box,
                     MINIMISING. numpy's global RNG drives the Sobol seed, the perturbation mask
                     and the LHS restarts exactly as there; torch's drives the Thompson sample.
``BaxusPublished``   BAxUS, Papenmeier, Nardi & Poloczek, NeurIPS 2022: a port of the BAxUS
                     tutorial the authors contributed to BoTorch (``tutorials/baxus``, local copy
                     ``cr/baselines/src/botorch``), MAXIMISING on [-1, 1]^D. Why this and not the
                     ``BAxUS`` package: it is the authors' own BoTorch implementation and runs
                     unchanged on our stack (botorch 0.17.2), whereas the package pins
                     ``gpytorch<=1.8.1`` and carries a non-commercial licence; the differences
                     from the package (GP and its fit) are listed in PUB_build.md and the
                     verification against BAxUS Fig. 3 decides whether the row may carry the name.
``run_cmaes_published``  pycma with the LassoBench protocol (``LassoBench/example/cma_example.py``):
                     popsize 20, sigma0 0.1, x0 = 0, bounds [-1, 1] (pycma BoundTransform).

Failure handling: none of the references catches a failure per step (an exception ends the
authors' run). Here a numerical failure of a TuRBO/BAxUS step gives a uniform point in the box for
that step and is recorded; the optimiser state is updated with the evaluated point as usual.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import gpytorch
import numpy as np
import torch
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from torch.quasirandom import SobolEngine

from src.spherevo.published_recipes import STEP_ERRORS, ModelFittingError

# ===========================================================================
# TuRBO-1 (official code: turbo/turbo_1.py, turbo/gp.py, turbo/utils.py)
# ===========================================================================
#   (line numbers of the local copy cr/baselines/src/TuRBO, re-checked in round 1, E-m14)
#   turbo_1.py:105-107   n_cand = min(100 * dim, 5000); failtol = ceil(max(4 / batch, dim / batch));
#                        succtol 3
#   turbo_1.py:111-113   length_min 0.5**7, length_max 1.6, length_init 0.8
#   turbo_1.py:137-150   _adjust_length: success iff min(fX_next) < min(_fX) - 1e-3 |min(_fX)| (:138)
#   turbo_1.py:159-160   mu = median(fX), sigma = fX.std() (< 1e-6 -> 1); fX standardised
#   turbo_1.py:164-176   GP on CPU float64 while len(X) < min_cuda (1024), max_cholesky_size 2000,
#                        train_gp(num_steps = n_training_steps = 50)
#   turbo_1.py:181-186   centre = argmin; lengthscale weights normalised to product 1;
#                        lb/ub = clip(centre -/+ weights * length / 2, 0, 1)
#   turbo_1.py:189-198   seed = np.random.randint(1e6); SobolEngine(dim, scramble, seed);
#                        mask = np.random.rand(n_cand, dim) <= min(20 / dim, 1); empty rows get
#                        one coordinate from np.random.randint(0, dim - 1)
#   turbo_1.py:214-216   Thompson sample of the posterior PREDICTIVE (likelihood(gp(X_cand)))
#   turbo_1.py:226-234   pick argmin per batch member, then remove it (y_cand[indbest, :] = inf)
#   turbo_1.py:244-259   restart: fresh latin_hypercube(n_init, dim) design, TR data reset
#   gp.py:48-53          Interval(5e-4, 0.2) noise, Interval(0.005, 2.0) ARD lengthscale,
#                        Interval(0.05, 20.0) outputscale; gp.py:26-39 Matern-5/2, ConstantMean
#   gp.py:74-92          init outputscale 1.0, lengthscale 0.5, noise 0.005 (:79-81); Adam lr 0.1
#                        (:85), num_steps iterations (:87-92)
#   utils.py:29-39       latin_hypercube with centre perturbation

TURBO_LENGTH_INIT = 0.8
TURBO_LENGTH_MIN = 0.5 ** 7
TURBO_LENGTH_MAX = 1.6
TURBO_SUCCESS_TOLERANCE = 3
TURBO_N_TRAINING_STEPS = 50
TURBO_MAX_CHOLESKY_SIZE = 2000
TURBO_MIN_CUDA = 1024
TURBO_NOISE_INTERVAL = (5e-4, 0.2)
TURBO_LENGTHSCALE_INTERVAL = (0.005, 2.0)
TURBO_OUTPUTSCALE_INTERVAL = (0.05, 20.0)
TURBO_INIT_HYPERS = {"covar_module.outputscale": 1.0,
                     "covar_module.base_kernel.lengthscale": 0.5,
                     "likelihood.noise": 0.005}
TURBO_ADAM_LR = 0.1


def turbo_latin_hypercube(n_pts: int, dim: int) -> np.ndarray:
    """``utils.py:29-39``: LHS with centre perturbation, numpy global RNG."""
    X = np.zeros((n_pts, dim))
    centers = (1.0 + 2.0 * np.arange(0.0, n_pts)) / float(2 * n_pts)
    for i in range(dim):
        X[:, i] = centers[np.random.permutation(n_pts)]
    pert = np.random.uniform(-1.0, 1.0, (n_pts, dim)) / float(2 * n_pts)
    X += pert
    return X


class _TurboGP(ExactGP):
    """``gp.py:26-39``."""

    def __init__(self, train_x, train_y, likelihood, lengthscale_constraint,
                 outputscale_constraint, ard_dims):
        super().__init__(train_x, train_y, likelihood)
        self.ard_dims = ard_dims
        self.mean_module = ConstantMean()
        base_kernel = MaternKernel(lengthscale_constraint=lengthscale_constraint,
                                   ard_num_dims=ard_dims, nu=2.5)
        self.covar_module = ScaleKernel(base_kernel, outputscale_constraint=outputscale_constraint)

    def forward(self, x):
        return MultivariateNormal(self.mean_module(x), self.covar_module(x))


def turbo_train_gp(train_x: torch.Tensor, train_y: torch.Tensor, use_ard: bool = True,
                   num_steps: int = TURBO_N_TRAINING_STEPS) -> _TurboGP:
    """``gp.py:41-95`` (``hypers={}`` as ``turbo_1.py`` always passes)."""
    assert train_x.ndim == 2 and train_y.ndim == 1 and train_x.shape[0] == train_y.shape[0]
    noise_constraint = Interval(*TURBO_NOISE_INTERVAL)
    if use_ard:
        lengthscale_constraint = Interval(*TURBO_LENGTHSCALE_INTERVAL)
    else:
        lengthscale_constraint = Interval(0.005, math.sqrt(train_x.shape[1]))
    outputscale_constraint = Interval(*TURBO_OUTPUTSCALE_INTERVAL)
    likelihood = GaussianLikelihood(noise_constraint=noise_constraint).to(
        device=train_x.device, dtype=train_y.dtype)
    model = _TurboGP(train_x=train_x, train_y=train_y, likelihood=likelihood,
                     lengthscale_constraint=lengthscale_constraint,
                     outputscale_constraint=outputscale_constraint,
                     ard_dims=train_x.shape[1] if use_ard else None,
                     ).to(device=train_x.device, dtype=train_x.dtype)
    model.train()
    likelihood.train()
    mll = ExactMarginalLogLikelihood(likelihood, model)
    model.initialize(**TURBO_INIT_HYPERS)
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=TURBO_ADAM_LR)
    for _ in range(num_steps):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
    model.eval()
    likelihood.eval()
    return model


@dataclass
class TurboStepInfo:
    fallback: Optional[str] = None
    kind: str = "bo"                 # "init" (restart design) or "bo"
    length: float = float("nan")
    t_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {"fallback": self.fallback, "events": [], "kind": self.kind,
                "tr_length": self.length, "t_acq_s": round(self.t_s, 4)}


class Turbo1Published:
    """TuRBO-1 as in ``turbo_1.py``, as an ask/tell optimiser on [0,1]^dim, MINIMISING.

    ``ask`` returns one point while an initial design is pending and a batch of ``batch_size``
    points otherwise; ``tell`` takes the evaluated rows (a batch adjusts the trust region once,
    with the minimum of the batch, as ``turbo_1.py:284-292``).

    ``seed_data`` installs an initial design that was evaluated outside (our cold starts);
    without it the first ``ask`` calls return the official LHS design of ``n_init`` points.
    """

    def __init__(self, dim: int, n_init: int, batch_size: int = 1, use_ard: bool = True,
                 max_cholesky_size: int = TURBO_MAX_CHOLESKY_SIZE,
                 n_training_steps: int = TURBO_N_TRAINING_STEPS):
        # q = batch_size as in turbo_1.py: failtol ceil(max(4/q, d/q)), q Thompson draws,
        # per-draw argmin with removal; the production arms use q = 1 (the verification of the
        # LassoBench pickles uses q = 10, LassoBench example/turbo_example.py:78)
        assert int(batch_size) >= 1
        self.dim, self.n_init, self.batch_size = dim, n_init, int(batch_size)
        self.use_ard, self.max_cholesky_size = use_ard, max_cholesky_size
        self.n_training_steps = n_training_steps
        self.n_cand = min(100 * dim, 5000)
        self.failtol = np.ceil(np.max([4.0 / batch_size, dim / batch_size]))
        self.succtol = TURBO_SUCCESS_TOLERANCE
        self.length_min, self.length_max = TURBO_LENGTH_MIN, TURBO_LENGTH_MAX
        self.length_init = TURBO_LENGTH_INIT
        self.n_restarts = 0
        self._pending_init: list[np.ndarray] = []
        self._last_kind = "bo"
        self._restart()
        self._need_initial_design = True     # LHS drawn at the first ask unless seed_data()

    def _restart(self) -> None:
        self._X = np.zeros((0, self.dim))
        self._fX = np.zeros((0, 1))
        self.failcount = 0
        self.succcount = 0
        self.length = self.length_init

    def seed_data(self, X_unit: np.ndarray, fX: np.ndarray) -> None:
        """Use an already evaluated design instead of the LHS one (our cold starts)."""
        self._restart()
        self._pending_init = []
        self._X = np.asarray(X_unit, dtype=np.float64).reshape(-1, self.dim).copy()
        self._fX = np.asarray(fX, dtype=np.float64).reshape(-1, 1).copy()
        self._need_initial_design = False

    def _adjust_length(self, fX_next: np.ndarray) -> None:
        if np.min(fX_next) < np.min(self._fX) - 1e-3 * math.fabs(np.min(self._fX)):
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1
        if self.succcount == self.succtol:
            self.length = min([2.0 * self.length, self.length_max])
            self.succcount = 0
        elif self.failcount == self.failtol:
            self.length /= 2.0
            self.failcount = 0

    def _create_candidates(self, X: np.ndarray, fX: np.ndarray, length: float):
        assert X.min() >= 0.0 and X.max() <= 1.0
        mu, sigma = np.median(fX), fX.std()
        sigma = 1.0 if sigma < 1e-6 else sigma
        fX = (fX.copy() - mu) / sigma
        device, dtype = torch.device("cpu"), torch.float64       # len(X) < min_cuda
        with gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_torch = torch.tensor(X).to(device=device, dtype=dtype)
            y_torch = torch.tensor(fX).to(device=device, dtype=dtype)
            gp = turbo_train_gp(X_torch, y_torch, use_ard=self.use_ard,
                                num_steps=self.n_training_steps)
        x_center = X[fX.argmin().item(), :][None, :]
        weights = gp.covar_module.base_kernel.lengthscale.cpu().detach().numpy().ravel()
        weights = weights / weights.mean()
        weights = weights / np.prod(np.power(weights, 1.0 / len(weights)))
        lb = np.clip(x_center - weights * length / 2.0, 0.0, 1.0)
        ub = np.clip(x_center + weights * length / 2.0, 0.0, 1.0)
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(self.dim, scramble=True, seed=seed)
        pert = sobol.draw(self.n_cand).to(dtype=dtype, device=device).cpu().detach().numpy()
        pert = lb + (ub - lb) * pert
        prob_perturb = min(20.0 / self.dim, 1.0)
        mask = np.random.rand(self.n_cand, self.dim) <= prob_perturb
        ind = np.where(np.sum(mask, axis=1) == 0)[0]
        mask[ind, np.random.randint(0, self.dim - 1, size=len(ind))] = 1
        X_cand = x_center.copy() * np.ones((self.n_cand, self.dim))
        X_cand[mask] = pert[mask]
        with torch.no_grad(), gpytorch.settings.max_cholesky_size(self.max_cholesky_size):
            X_cand_torch = torch.tensor(X_cand).to(device=device, dtype=dtype)
            y_cand = gp.likelihood(gp(X_cand_torch)).sample(
                torch.Size([self.batch_size])).t().cpu().detach().numpy()
        y_cand = mu + sigma * y_cand
        return X_cand, y_cand

    def _select_candidates(self, X_cand: np.ndarray, y_cand: np.ndarray) -> np.ndarray:
        X_next = np.ones((self.batch_size, self.dim))
        for i in range(self.batch_size):
            indbest = np.argmin(y_cand[:, i])
            X_next[i, :] = X_cand[indbest, :].copy()
            y_cand[indbest, :] = np.inf
        return X_next

    def ask(self) -> tuple[np.ndarray, TurboStepInfo]:
        """Next point (1 x dim, unit cube)."""
        info = TurboStepInfo()
        if self._need_initial_design:
            self._need_initial_design = False
            self._pending_init = [x for x in turbo_latin_hypercube(self.n_init, self.dim)]
        if not self._pending_init and self.length < self.length_min:
            self.n_restarts += 1
            self._restart()
            self._pending_init = [x for x in turbo_latin_hypercube(self.n_init, self.dim)]
        if self._pending_init:
            self._last_kind = info.kind = "init"
            info.length = self.length
            return self._pending_init.pop(0).reshape(1, -1), info
        self._last_kind = "bo"
        info.length = self.length
        t0 = time.perf_counter()
        try:
            X_cand, y_cand = self._create_candidates(self._X, self._fX.ravel(), self.length)
            x = self._select_candidates(X_cand, y_cand)
            if not np.all(np.isfinite(x)):
                raise ModelFittingError("non-finite TuRBO candidate")
        except STEP_ERRORS as e:
            info.fallback = (f"{type(e).__name__}: {e} (reference: no per-step handling)")
            x = np.random.rand(self.batch_size, self.dim)
        info.t_s = time.perf_counter() - t0
        return x, info

    def tell(self, x_unit: np.ndarray, fx) -> None:
        x = np.asarray(x_unit, dtype=np.float64).reshape(-1, self.dim)
        f = np.asarray(fx, dtype=np.float64).reshape(-1, 1)
        if self._last_kind == "bo":
            self._adjust_length(f)
        self._X = np.vstack((self._X, x))
        self._fX = np.vstack((self._fX, f))


# ===========================================================================
# BAxUS (BoTorch tutorial tutorials/baxus by the BAxUS authors)
# ===========================================================================
#   cells 8:   BaxusState: new_bins_on_split 3, length 0.8 / min 0.5**7 / max 1.6, success
#              tolerance 3, d_init and n_splits from the dimension, split_budget and
#              failure_tolerance from eval_budget; update_state (relative 1e-3 success)
#   cell 10:   embedding_matrix (sparse +-1 count-sketch, equal-size bins)
#   cell 12:   increase_embedding_and_observations (each bin split into n_new_bins + 1)
#   cell 17:   initial points: SobolEngine(d_init, scramble, seed), mapped to [-1, 1]
#   cell 19:   create_candidate, TS: TR centre +- weights * length clamped to [-1, 1],
#              perturbation prob min(20/d, 1); its default n_candidates (from the TARGET dim)
#              is overridden by the loop of cell 21
#   cell 21:   N_CANDIDATES = min(5000, max(2000, 200 * dim)) with dim = the INPUT dimension
#              (cell 6), passed to create_candidate (round-1 fix C5: 5000 for every D >= 25)
#   cell 21:   train_Y = (Y - mean) / std; SingleTaskGP(X, train_Y, likelihood=
#              GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))) (default kernel and
#              outcome transform of the installed BoTorch); fit_gpytorch_mll under
#              max_cholesky_size(inf), ModelFittingError -> Adam(lr 0.1) for 100 steps;
#              validate_input_scaling(False); eval_budget = total budget - n_init
#              (Fig. 3 of the paper: m_D = 1000 with a budget of 2000, so the verification passes
#              eval_budget = m_D - n_init, not the run budget; round-1 fix B5)

BAXUS_NEW_BINS_ON_SPLIT = 3
BAXUS_NOISE_INTERVAL = (1e-8, 1e-3)
BAXUS_ADAM_STEPS = 100
BAXUS_ADAM_LR = 0.1


@dataclass
class BaxusState:
    dim: int
    eval_budget: int
    new_bins_on_split: int = BAXUS_NEW_BINS_ON_SPLIT
    d_init: int = float("nan")
    target_dim: int = float("nan")
    n_splits: int = float("nan")
    length: float = 0.8
    length_init: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    success_counter: int = 0
    success_tolerance: int = 3
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        n_splits = round(math.log(self.dim, self.new_bins_on_split + 1))
        self.d_init = int(1 + np.argmin(np.abs(
            (1 + np.arange(self.new_bins_on_split))
            * (1 + self.new_bins_on_split) ** n_splits - self.dim)))
        self.target_dim = self.d_init
        self.n_splits = n_splits

    @property
    def split_budget(self) -> int:
        return round(-1 * (self.new_bins_on_split * self.eval_budget * self.target_dim)
                     / (self.d_init * (1 - (self.new_bins_on_split + 1) ** (self.n_splits + 1))))

    @property
    def failure_tolerance(self) -> int:
        if self.target_dim == self.dim:
            return self.target_dim
        k = math.floor(math.log(self.length_min / self.length_init, 0.5))
        split_budget = self.split_budget
        return min(self.target_dim, max(1, math.floor(split_budget / k)))


def baxus_update_state(state: BaxusState, Y_next: torch.Tensor) -> BaxusState:
    if max(Y_next) > state.best_value + 1e-3 * math.fabs(state.best_value):
        state.success_counter += 1
        state.failure_counter = 0
    else:
        state.success_counter = 0
        state.failure_counter += 1
    if state.success_counter == state.success_tolerance:
        state.length = min(2.0 * state.length, state.length_max)
        state.success_counter = 0
    elif state.failure_counter == state.failure_tolerance:
        state.length /= 2.0
        state.failure_counter = 0
    state.best_value = max(state.best_value, max(Y_next).item())
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def baxus_embedding_matrix(input_dim: int, target_dim: int, device, dtype) -> torch.Tensor:
    if target_dim >= input_dim:
        return torch.eye(input_dim, device=device, dtype=dtype)
    input_dims_perm = torch.randperm(input_dim, device=device) + 1
    bins = torch.tensor_split(input_dims_perm, target_dim)
    bins = torch.nn.utils.rnn.pad_sequence(bins, batch_first=True)
    mtrx = torch.zeros((target_dim, input_dim + 1), dtype=dtype, device=device)
    mtrx = mtrx.scatter_(
        1, bins, 2 * torch.randint(2, (target_dim, input_dim), dtype=dtype, device=device) - 1)
    return mtrx[:, 1:]


def baxus_increase_embedding_and_observations(S: torch.Tensor, X: torch.Tensor, n_new_bins: int):
    assert X.size(1) == S.size(0), "Observations don't lie in row space of S"
    device, dtype = S.device, S.dtype
    S_update = S.clone()
    X_update = X.clone()
    for row_idx in range(len(S)):
        row = S[row_idx]
        idxs_non_zero = torch.nonzero(row)
        idxs_non_zero = idxs_non_zero[torch.randperm(len(idxs_non_zero))].reshape(-1)
        if len(idxs_non_zero) <= 1:
            continue
        non_zero_elements = row[idxs_non_zero].reshape(-1)
        n_row_bins = min(n_new_bins + 1, len(idxs_non_zero))
        new_bins = torch.tensor_split(idxs_non_zero + 1, n_row_bins)[1:]
        elements_to_move = torch.tensor_split(non_zero_elements, n_row_bins)[1:]
        new_bins_padded = torch.nn.utils.rnn.pad_sequence(new_bins, batch_first=True)
        els_to_move_padded = torch.nn.utils.rnn.pad_sequence(elements_to_move, batch_first=True)
        S_stack = torch.zeros((n_row_bins - 1, len(row) + 1), device=device, dtype=dtype)
        S_stack = S_stack.scatter_(1, new_bins_padded, els_to_move_padded)
        S_update[row_idx, torch.hstack(new_bins) - 1] = 0
        X_update = torch.hstack(
            (X_update, X[:, row_idx].reshape(-1, 1).repeat(1, len(new_bins))))
        S_update = torch.vstack((S_update, S_stack[:, 1:]))
    return S_update, X_update


def baxus_project_down(X_input: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Least-squares projection input space -> target space: ``X S^T / diag(S S^T)``.

    ``S`` is the embedding's count-sketch matrix: entries in {-1, 0, +1} with one
    nonzero per INPUT coordinate, so its rows have disjoint supports,
    ``S S^T = diag(nnz)``, and the least-squares solution of ``min_t ||t S - x||`` is
    ``t = x S^T / nnz``.

    This is the identity on everything the tutorial itself produces, so it is not a
    change of the method but a definition where the tutorial has none:

    * for a point the optimiser proposed, ``x = (t S).clamp(-1, 1)`` with ``|t| <= 1``
      and entries of ``S`` equal to +-1, so no coordinate is clamped and
      ``x S^T / nnz = t (S S^T) / nnz = t`` exactly;
    * after a split each child row's support is a subset of its parent's with the same
      signs, so the projection returns the parent coordinate -- exactly the value
      ``baxus_increase_embedding_and_observations`` copies into the child bin.

    It therefore differs from the tutorial only on a design handed in from the input
    space (``BaxusPublished.seed_external``), for which the tutorial has no rule.
    Dropping the ``/ nnz`` -- the defect of the submitted wrapper -- inflates such a
    point by the nonzeros per row, about ``D / target_dim``, which at ``target_dim = 1``
    puts the cold starts near ``|t| ~ 10`` and collapses every later proposal onto a
    single corner of the box.
    """
    nnz = (S != 0).sum(dim=1).to(S.dtype).clamp(min=1.0)
    return (X_input @ S.T) / nnz


class BaxusPublished:
    """BAxUS (BoTorch tutorial) as an ask/tell optimiser on [-1, 1]^dim, MAXIMISING."""

    def __init__(self, dim: int, n_init: int, eval_budget: int, seed: int,
                 device: torch.device = torch.device("cpu")):
        self.dim, self.device, self.dtype = dim, device, torch.double
        self.state = BaxusState(dim=dim, eval_budget=eval_budget)
        self.S = baxus_embedding_matrix(dim, self.state.d_init, device, self.dtype)
        sobol = SobolEngine(dimension=self.state.d_init, scramble=True, seed=seed)
        X_init = 2 * sobol.draw(n=n_init).to(dtype=self.dtype, device=device) - 1
        self._pending = [x.reshape(1, -1) for x in X_init]
        self.X_target = torch.zeros((0, self.state.d_init), dtype=self.dtype, device=device)
        self.Y = torch.zeros((0, 1), dtype=self.dtype, device=device)
        self._last: Optional[torch.Tensor] = None
        self._last_kind = "init"
        self.n_splits_done = 0
        #: Input-space ([-1, 1]^dim) row of every observation, in order.  Always
        #: tracked (no RNG, no numerical effect); read only in the seeded mode.
        self.X_input = torch.zeros((0, dim), dtype=self.dtype, device=device)
        #: Set by :meth:`seed_external` only, and the only switch of the post-split
        #: re-projection: left False this class behaves exactly as before, which is
        #: what the verified LassoBench ``baxus_official`` row runs.
        self._external_seeding = False

    def seed_external(self, X_input: np.ndarray, Y: np.ndarray) -> dict[str, Any]:
        """Use an already evaluated design given in the INPUT space, instead of the
        tutorial's own Sobol design in the target space.

        The tutorial draws its initial design itself and has no rule for one that
        arrives from the input space.  This one enters the target space through
        :func:`baxus_project_down` and is clamped to [-1, 1]; after every split the
        whole history is re-projected the same way, which is the identity on the
        optimiser's own points and so changes nothing but these seeded rows.

        Returns the counts worth recording in the result file.
        """
        X = torch.as_tensor(np.asarray(X_input, dtype=np.float64), dtype=self.dtype,
                            device=self.device).reshape(-1, self.dim)
        y = torch.as_tensor(np.asarray(Y, dtype=np.float64), dtype=self.dtype,
                            device=self.device).reshape(-1, 1)
        n_input_clamped = int(((X < -1.0) | (X > 1.0)).sum())
        X = X.clamp(-1.0, 1.0)
        T = baxus_project_down(X, self.S)
        n_target_clamped = int(((T < -1.0) | (T > 1.0)).sum())
        self.X_input = X
        self.X_target = T.clamp(-1.0, 1.0)
        self.Y = y
        self._pending = []              # the tutorial's own Sobol design is discarded
        self._external_seeding = True
        return {"n_seeded": int(X.shape[0]),
                "cold_start_input_coordinates_clamped": n_input_clamped,
                "cold_start_target_coordinates_clamped": n_target_clamped,
                "target_dim_at_seeding": int(self.S.shape[0])}

    def _fit_and_propose(self, events: Optional[list[str]] = None) -> torch.Tensor:
        import botorch
        from botorch.fit import fit_gpytorch_mll
        from botorch.generation import MaxPosteriorSampling
        from botorch.models import SingleTaskGP

        train_Y = (self.Y - self.Y.mean()) / self.Y.std()
        with botorch.settings.validate_input_scaling(False):
            likelihood = GaussianLikelihood(noise_constraint=Interval(*BAXUS_NOISE_INTERVAL))
            model = SingleTaskGP(self.X_target, train_Y, likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            with gpytorch.settings.max_cholesky_size(float("inf")):
                try:
                    fit_gpytorch_mll(mll)
                except ModelFittingError as e:
                    if events is not None:     # tutorial behaviour, recorded (round-1 fix C7)
                        events.append(f"fit: ModelFittingError -> Adam fallback "
                                      f"({BAXUS_ADAM_STEPS} steps, lr {BAXUS_ADAM_LR}): {e}")
                    optimizer = torch.optim.Adam([{"params": model.parameters()}],
                                                 lr=BAXUS_ADAM_LR)
                    for _ in range(BAXUS_ADAM_STEPS):
                        optimizer.zero_grad()
                        output = model(self.X_target)
                        loss = -mll(output, train_Y.flatten())
                        loss.backward()
                        optimizer.step()
                X = self.X_target
                # cell 21: from the INPUT dimension (round-1 fix C5), not the target one
                n_candidates = min(5000, max(2000, 200 * self.dim))
                x_center = X[train_Y.argmax(), :].clone()
                weights = model.covar_module.lengthscale.detach().view(-1)
                weights = weights / weights.mean()
                weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
                tr_lb = torch.clamp(x_center - weights * self.state.length, -1.0, 1.0)
                tr_ub = torch.clamp(x_center + weights * self.state.length, -1.0, 1.0)
                d = X.shape[-1]
                sobol = SobolEngine(d, scramble=True)
                pert = sobol.draw(n_candidates).to(dtype=self.dtype, device=self.device)
                pert = tr_lb + (tr_ub - tr_lb) * pert
                prob_perturb = min(20.0 / d, 1.0)
                mask = torch.rand(n_candidates, d, dtype=self.dtype, device=self.device) \
                    <= prob_perturb
                ind = torch.where(mask.sum(dim=1) == 0)[0]
                mask[ind, torch.randint(0, d, size=(len(ind),), device=self.device)] = 1
                X_cand = x_center.expand(n_candidates, d).clone()
                X_cand[mask] = pert[mask]
                thompson_sampling = MaxPosteriorSampling(model=model, replacement=False)
                with torch.no_grad():
                    X_next = thompson_sampling(X_cand, num_samples=1)
        return X_next.reshape(1, -1)

    def ask(self) -> tuple[np.ndarray, dict[str, Any]]:
        """Next point in [-1, 1]^dim (lifted from the target space)."""
        info: dict[str, Any] = {"fallback": None, "events": [],
                                "target_dim": int(self.S.shape[0]),
                                "tr_length": float(self.state.length)}
        t0 = time.perf_counter()
        if self._pending:
            self._last, self._last_kind = self._pending.pop(0), "init"
            info["kind"] = "init"
        else:
            self._last_kind = info["kind"] = "bo"
            try:
                self._last = self._fit_and_propose(info["events"])
                if not torch.isfinite(self._last).all():
                    raise ModelFittingError("non-finite BAxUS candidate")
            except STEP_ERRORS as e:
                info["fallback"] = f"{type(e).__name__}: {e} (reference: no per-step handling)"
                self._last = 2 * torch.rand(1, self.X_target.shape[-1], dtype=self.dtype,
                                            device=self.device) - 1
        info["t_acq_s"] = round(time.perf_counter() - t0, 4)
        x_input = (self._last @ self.S).clamp(-1.0, 1.0)
        return x_input.detach().cpu().numpy(), info

    def tell(self, y: float) -> None:
        Y_next = torch.tensor([[float(y)]], dtype=self.dtype, device=self.device)
        if self._last_kind == "bo":      # the tutorial updates the state only after BO steps
            self.state = baxus_update_state(self.state, Y_next)
        self.X_target = torch.cat((self.X_target, self._last), dim=0)
        self.X_input = torch.cat(
            (self.X_input, (self._last @ self.S).clamp(-1.0, 1.0)), dim=0)
        self.Y = torch.cat((self.Y, Y_next), dim=0)
        if self.state.restart_triggered:
            self.state.restart_triggered = False
            # called unconditionally and with the same arguments in both modes, so a
            # split consumes the same draws of the global torch RNG either way
            self.S, X_target_copied = baxus_increase_embedding_and_observations(
                self.S, self.X_target, self.state.new_bins_on_split)
            self.X_target = (baxus_project_down(self.X_input, self.S).clamp(-1.0, 1.0)
                             if self._external_seeding else X_target_copied)
            self.state.target_dim = len(self.S)
            self.state.length = self.state.length_init
            self.state.failure_counter = 0
            self.state.success_counter = 0
            self.n_splits_done += 1


# ===========================================================================
# CMA-ES with the LassoBench protocol (LassoBench/example/cma_example.py:45-55)
# ===========================================================================
#   opts popsize 20, seed, bounds [-1, 1]; CMAEvolutionStrategy(np.zeros(d), sigma=0.1, opts);
#   es.optimize(f) = repeated ask / evaluate / tell of whole generations.

CMAES_POPSIZE = 20
CMAES_SIGMA0 = 0.1
CMAES_BOUNDS = (-1.0, 1.0)


def run_cmaes_published(f: Callable[[np.ndarray], float], dim: int, n_evals: int, seed: int,
                        *, x0: Optional[np.ndarray] = None, sigma0: float = CMAES_SIGMA0,
                        popsize: int = CMAES_POPSIZE,
                        on_eval: Optional[Callable[[np.ndarray, float, dict], None]] = None,
                        ) -> dict[str, Any]:
    """Minimise ``f`` with pycma under the LassoBench settings for exactly ``n_evals`` evaluations.

    A last, incomplete generation is evaluated up to the budget and not told (pycma tells whole
    generations). pycma's ``seed`` option treats 0 as "time", so the run uses ``seed + 1``.
    ``es.stop()`` is evaluated after every told generation, as ``es.optimize`` does, and recorded
    (``stop_conditions``; the first evaluation of the next generation carries it as an event); the
    run does NOT stop on it (round-1 fix C7: the official run would stop early, which can only
    lose evaluations).
    """
    import cma

    opts = cma.CMAOptions()
    opts.set("popsize", popsize)
    opts.set("seed", int(seed) + 1)
    opts.set("bounds", list(CMAES_BOUNDS))
    opts.set("verbose", -9)
    opts.set("verb_log", 0)
    opts.set("verb_disp", 0)
    es = cma.CMAEvolutionStrategy(np.zeros(dim) if x0 is None else np.asarray(x0, float),
                                  sigma0, opts)
    n, generations, told = 0, 0, 0
    stop_conditions: list[dict[str, Any]] = []
    pending_event: Optional[str] = None
    while n < n_evals:
        X = es.ask()
        fit = []
        for x in X:
            if n >= n_evals:
                break
            v = float(f(np.asarray(x)))
            fit.append(v)
            n += 1
            if on_eval is not None:
                events = [pending_event] if (pending_event and len(fit) == 1) else []
                on_eval(np.asarray(x), v, {"fallback": None, "events": events,
                                           "generation": generations})
        if len(fit) == len(X):
            es.tell(X, fit)
            told += 1
            stop = {str(k): str(v) for k, v in dict(es.stop()).items()}
            pending_event = None
            if stop:
                stop_conditions.append({"generation": generations, "evaluations": n,
                                        "conditions": stop})
                pending_event = (f"es.stop() after generation {generations}: "
                                 f"{sorted(stop)} (recorded; the run continues to the budget)")
        generations += 1
    return {"generations_started": generations, "generations_told": told,
            "popsize": popsize, "sigma0": sigma0, "cma_seed": int(seed) + 1,
            "cma_version": cma.__version__, "stop_conditions": stop_conditions}


def external_settings(method: str, dim: Optional[int] = None, batch_size: int = 1,
                      ) -> dict[str, Any]:
    """The literal settings of an external method, written once into every result record.
    ``batch_size`` (TuRBO only) is the run's q: it sets the failure tolerance (round-1 fix B6)."""
    if method == "turbo1":
        s = {"code": "port of uber-research/TuRBO turbo/turbo_1.py + turbo/gp.py",
             "gp": "ExactGP, ConstantMean, ScaleKernel(Matern-5/2 ARD); lengthscale "
                   "Interval(0.005, 2.0), outputscale Interval(0.05, 20), noise Interval(5e-4, 0.2); "
                   "init (1.0, 0.5, 0.005); Adam lr 0.1, 50 steps; max_cholesky_size 2000; CPU float64",
             "y_standardisation": "(fX - median) / std (std < 1e-6 -> 1)",
             "trust_region": {"length_init": 0.8, "length_min": 0.5 ** 7, "length_max": 1.6,
                              "success_tolerance": 3,
                              "failure_tolerance": "ceil(max(4 / q, d / q))"},
             "candidates": "min(100 d, 5000) Sobol (seed np.random.randint(1e6)), perturbation "
                           "prob min(20/d, 1), empty rows: np.random.randint(0, d - 1)",
             "acquisition": "Thompson sample of the posterior predictive, argmin per batch "
                            "member with removal", "batch_size": int(batch_size),
             "restart": "fresh latin_hypercube(n_init) design when length < length_min",
             "duplicates": "every proposal is evaluated and counted (as the reference)"}
        if dim is not None:
            s["n_candidates"] = min(100 * dim, 5000)
            s["failure_tolerance_value"] = int(np.ceil(max(4.0 / batch_size,
                                                           float(dim) / batch_size)))
    elif method == "baxus":
        s = {"code": "port of the BoTorch tutorial tutorials/baxus (by the BAxUS authors)",
             "embedding": "count-sketch, d_init from the dimension, 3 new bins per split",
             "gp": "SingleTaskGP default kernel and outcome transform of the installed BoTorch, "
                   "GaussianLikelihood(noise Interval(1e-8, 1e-3)); fit_gpytorch_mll, "
                   "ModelFittingError -> Adam lr 0.1, 100 steps; max_cholesky_size(inf)",
             "acquisition": "Thompson sampling, min(5000, max(2000, 200 D)) candidates, D = "
                            "the input dimension (tutorial cell 21)",
             "trust_region": "length 0.8 / 0.5**7 / 1.6, success tolerance 3, failure tolerance "
                             "and split budget from eval_budget",
             "initial_design": "own: scrambled Sobol in the d_init target space"}
        if dim is not None:
            st = BaxusState(dim=dim, eval_budget=1)
            s["d_init"], s["n_splits"] = st.d_init, st.n_splits
    elif method == "cmaes":
        s = {"code": "pycma CMAEvolutionStrategy, LassoBench example/cma_example.py settings",
             "popsize": CMAES_POPSIZE, "sigma0": CMAES_SIGMA0, "x0": "zeros (box centre)",
             "bounds": list(CMAES_BOUNDS), "boundary_handler": "pycma default (BoundTransform)",
             "seed": "pycma seed = run seed + 1 (0 means time in pycma)",
             "last_generation": "evaluated up to the budget, not told",
             "es_stop": "evaluated after every told generation and recorded; the run does not "
                        "stop on it"}
    else:
        raise ValueError(f"unknown external method {method!r}")
    s["method"] = method
    return s
