#!/usr/bin/env python3
"""BAxUS budget-scaling runner with GPU support.

BAxUS (Papenmeier et al., NeurIPS 2022) adapted from the NLP repo implementation.

Usage:
    PYTHONPATH=/path/to/NeurIPS CUDA_VISIBLE_DEVICES=1 \
        python scripts/run_baxus_budget.py --task adip --seed 42 --budget 1000
"""

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints import Interval
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from src.analysis.config import GUACAMOL_TO_POLI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_GP_FIT_ERRORS = (torch.linalg.LinAlgError, ValueError, RuntimeError)


def _is_cuda_error(e: Exception) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or (
        isinstance(e, RuntimeError) and "CUDA" in str(e)
    )


# ---------------------------------------------------------------------------
# BAxUS algorithm (from BoTorch tutorial / NLP repo)
# ---------------------------------------------------------------------------

@dataclass
class BaxusState:
    dim: int
    eval_budget: int
    new_bins_on_split: int = 3
    max_target_dim: int = 64
    d_init: int = 0
    target_dim: int = 0
    n_splits: int = 0
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
        self.d_init = int(1 + np.argmin(
            np.abs(
                (1 + np.arange(self.new_bins_on_split))
                * (1 + self.new_bins_on_split) ** n_splits
                - self.dim
            )
        ))
        self.target_dim = self.d_init
        self.n_splits = n_splits

    @property
    def split_budget(self) -> int:
        return round(
            -1 * (self.new_bins_on_split * self.eval_budget * self.target_dim)
            / (self.d_init * (1 - (self.new_bins_on_split + 1) ** (self.n_splits + 1)))
        )

    @property
    def failure_tolerance(self) -> int:
        if self.target_dim == self.dim:
            return self.target_dim
        k = math.floor(math.log(self.length_min / self.length_init, 0.5))
        sb = self.split_budget
        return min(self.target_dim, max(1, math.floor(sb / k)))


def _embedding_matrix(input_dim, target_dim, device, dtype):
    if target_dim >= input_dim:
        return torch.eye(input_dim, device=device, dtype=dtype)
    input_dims_perm = torch.randperm(input_dim, device=device) + 1
    bins = torch.tensor_split(input_dims_perm, target_dim)
    bins = torch.nn.utils.rnn.pad_sequence(bins, batch_first=True)
    mtrx = torch.zeros((target_dim, input_dim + 1), dtype=dtype, device=device)
    mtrx = mtrx.scatter_(
        1, bins,
        2 * torch.randint(2, (target_dim, input_dim), dtype=dtype, device=device) - 1,
    )
    return mtrx[:, 1:]


def _increase_embedding(S, X_target, n_new_bins, device, dtype):
    S_upd = S.clone()
    X_upd = X_target.clone()
    for row_idx in range(len(S)):
        row = S[row_idx]
        nz = torch.nonzero(row).reshape(-1)
        nz = nz[torch.randperm(len(nz), device=device)]
        if len(nz) <= 1:
            continue
        elems = row[nz]
        n_row_bins = min(n_new_bins + 1, len(nz))
        split_nz = torch.tensor_split(nz, n_row_bins)[1:]
        split_el = torch.tensor_split(elems, n_row_bins)[1:]
        for new_nz, new_el in zip(split_nz, split_el):
            new_row = torch.zeros(S_upd.shape[1], dtype=dtype, device=device)
            new_row[new_nz] = new_el
            S_upd[row_idx, new_nz] = 0
            S_upd = torch.vstack((S_upd, new_row.unsqueeze(0)))
            X_upd = torch.hstack((X_upd, X_target[:, row_idx:row_idx + 1]))
    return S_upd, X_upd


def _update_state(state, y_val):
    if y_val > state.best_value + 1e-3 * abs(state.best_value):
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
    state.best_value = max(state.best_value, y_val)
    if state.length < state.length_min:
        state.restart_triggered = True
    return state


def _baxus_candidate(state, model, X, Y, n_candidates=5000, device=None, dtype=torch.double):
    dim = X.shape[-1]
    x_center = X[Y.argmax(), :].clone()
    covar = model.covar_module
    if hasattr(covar, 'base_kernel') and hasattr(covar.base_kernel, 'lengthscale'):
        ls = covar.base_kernel.lengthscale.detach().view(-1)
    elif hasattr(covar, 'lengthscale'):
        ls = covar.lengthscale.detach().view(-1)
    else:
        ls = torch.ones(dim, dtype=dtype, device=device)
    weights = ls / ls.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))
    tr_lb = torch.clamp(x_center - weights * state.length, -1.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length, -1.0, 1.0)
    sobol = SobolEngine(dim, scramble=True)
    pert = sobol.draw(n_candidates).to(dtype=dtype, device=device)
    pert = tr_lb + (tr_ub - tr_lb) * pert
    prob_perturb = min(20.0 / dim, 1.0)
    mask = torch.rand(n_candidates, dim, dtype=dtype, device=device) <= prob_perturb
    ind = torch.where(mask.sum(dim=1) == 0)[0]
    if len(ind) > 0:
        mask[ind, torch.randint(0, dim, size=(len(ind),), device=device)] = 1
    X_cand = x_center.expand(n_candidates, dim).clone()
    X_cand[mask] = pert[mask]
    ts = MaxPosteriorSampling(model=model, replacement=False)
    with torch.no_grad():
        X_next = ts(X_cand, num_samples=1)
    return X_next


def _expand_subspace(state, S, X_input, device, dtype, max_target_dim):
    state.restart_triggered = False
    cap = min(max_target_dim, state.dim)
    if state.target_dim >= cap:
        state.length = state.length_init
        state.failure_counter = 0
        state.success_counter = 0
        X_target = X_input @ S.T
        return state, S, X_target
    X_target_raw = X_input @ S.T
    S, X_target_expanded = _increase_embedding(
        S, X_target_raw, state.new_bins_on_split, device, dtype,
    )
    new_dim = len(S)
    if new_dim > cap:
        S = S[:cap]
        X_target_expanded = X_target_expanded[:, :cap]
        new_dim = cap
    state.target_dim = new_dim
    state.length = state.length_init
    state.failure_counter = 0
    state.success_counter = 0
    return state, S, X_target_expanded


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_baxus(task_name, seed, budget, n_cold_start=5, device="cuda"):
    from src.codec.selfies_vae import SELFIESVAECodec
    from src.oracle.guacamol import GuacaMolOracle, load_zinc_smiles

    torch.manual_seed(seed)
    np.random.seed(seed)
    dtype = torch.double

    poli_task = GUACAMOL_TO_POLI.get(task_name, task_name)

    codec = SELFIESVAECodec.from_pretrained(device=device)
    oracle = GuacaMolOracle(task_name)
    dim = codec.embedding_dim

    # Cold start (same as run_benchmark.py)
    zinc_smiles = load_zinc_smiles()
    rng = np.random.RandomState(seed)
    cs_indices = rng.choice(len(zinc_smiles), n_cold_start, replace=False)
    cs_smiles = [zinc_smiles[i] for i in cs_indices]
    cs_scores = [oracle.score(s) for s in cs_smiles]
    Z_init = codec.encode(cs_smiles).to(device)

    # Normalize to [-1, 1]
    z_min = Z_init.min(dim=0).values
    z_max = Z_init.max(dim=0).values

    def normalize(z):
        return 2 * (z - z_min) / (z_max - z_min + 1e-8) - 1

    def denormalize(z_norm):
        return (z_norm + 1) / 2 * (z_max - z_min + 1e-8) + z_min

    Z_norm = normalize(Z_init).to(dtype)
    Y = torch.tensor(cs_scores, dtype=dtype, device=device).unsqueeze(-1)

    max_iter = budget - n_cold_start
    state = BaxusState(dim=dim, eval_budget=budget, max_target_dim=64)
    S = _embedding_matrix(dim, state.target_dim, device, dtype)

    X_input = Z_norm
    X_target = Z_norm @ S.T

    state.best_value = Y.max().item()
    best_score = state.best_value
    best_trace = cs_scores.copy()
    all_scores = cs_scores.copy()
    smiles_set = set(cs_smiles)
    gp_failures = 0
    decode_failures = 0

    for i in range(1, len(best_trace)):
        best_trace[i] = max(best_trace[i], best_trace[i - 1])

    logger.info(f"BAxUS-GPU: task={poli_task} seed={seed} budget={budget} "
                f"dim={dim} d_init={state.d_init} cold_start_best={best_score:.4f}")

    t0 = time.time()
    for it in range(max_iter):
        # Standardize Y for GP fitting
        Y_mean = Y.mean()
        Y_std = Y.std().clamp(min=1e-6)
        Y_std_train = (Y - Y_mean) / Y_std

        use_random = False
        try:
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            model = SingleTaskGP(
                X_target, Y_std_train, likelihood=likelihood,
            ).to(device)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
        except _GP_FIT_ERRORS as e:
            if _is_cuda_error(e):
                raise
            logger.warning(f"Iter {it}: GP fit failed ({type(e).__name__}), trying fallback")
            gp_failures += 1
            try:
                likelihood = GaussianLikelihood(noise_constraint=Interval(1e-4, 1.0))
                model = SingleTaskGP(
                    X_target, Y_std_train, likelihood=likelihood,
                ).to(device)
                mll = ExactMarginalLogLikelihood(model.likelihood, model)
                fit_gpytorch_mll(mll)
            except _GP_FIT_ERRORS as e2:
                if _is_cuda_error(e2):
                    raise
                logger.warning(f"Iter {it}: fallback GP also failed, using random candidate")
                gp_failures += 1
                use_random = True

        if use_random:
            X_next_target_norm = torch.rand(1, X_target.shape[-1], dtype=dtype, device=device) * 2 - 1
        else:
            model.eval()
            X_next_target_norm = _baxus_candidate(
                state=state, model=model, X=X_target, Y=Y,
                n_candidates=5000, device=device, dtype=dtype,
            )

        # Back-project target -> input space
        X_next_input = (X_next_target_norm @ S).clamp(-1.0, 1.0)
        z_full = denormalize(X_next_input.float())

        try:
            smiles = codec.decode(z_full)[0]
        except torch.cuda.OutOfMemoryError:
            raise
        except (RuntimeError, ValueError) as e:
            if _is_cuda_error(e):
                raise
            smiles = None
            decode_failures += 1

        if not smiles or smiles in smiles_set:
            score = float("nan")
            if smiles is None:
                decode_failures += 1
        else:
            score = oracle.score(smiles)
            smiles_set.add(smiles)

        # Always add point to GP data (use 0.0 for failures — nan breaks GP)
        gp_score = 0.0 if math.isnan(score) else score
        Y_next = torch.tensor([[gp_score]], dtype=dtype, device=device)
        state = _update_state(state, gp_score)

        X_input = torch.cat([X_input, X_next_input.to(dtype)], dim=0)
        X_target = X_input @ S.T
        Y = torch.cat([Y, Y_next], dim=0)

        all_scores.append(score)
        if not math.isnan(score):
            best_score = max(best_score, score)
        best_trace.append(best_score)

        if state.restart_triggered:
            state, S, X_target = _expand_subspace(
                state, S, X_input, device, dtype, 64
            )
            logger.info(f"Iter {it}: BAxUS expanded to {state.target_dim}D")

        if (it + 1) % 100 == 0:
            elapsed = time.time() - t0
            logger.info(f"Iteration {it + 1}/{max_iter}: best={best_score:.4f} "
                        f"target_dim={state.target_dim} gp_fail={gp_failures} "
                        f"decode_fail={decode_failures} elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0
    logger.info(f"BAxUS-GPU done: best={best_score:.4f} gp_failures={gp_failures} "
                f"decode_failures={decode_failures} elapsed={elapsed:.0f}s")

    return {
        "method": "baxus",
        "task_id": poli_task,
        "seed": seed,
        "best_score": best_score,
        "best_trace": best_trace,
        "all_scores": all_scores,
        "n_evaluated": len(all_scores),
        "wall_time_seconds": elapsed,
        "gp_failures": gp_failures,
        "decode_failures": decode_failures,
        "config": {
            "budget": budget,
            "n_cold_start": n_cold_start,
            "d_init": state.d_init,
            "max_target_dim": 64,
            "input_dim": dim,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="BAxUS budget-scaling runner (GPU)")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    result = run_baxus(args.task, args.seed, args.budget, device=args.device)

    out_dir = Path("results/guacamol/spherevo")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_b{args.budget}" if args.budget != 30 else ""
    out_path = out_dir / f"ours_baxus_{result['task_id']}_s{args.seed}_256d{suffix}.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
