#!/usr/bin/env python3
"""TuRBO budget-scaling runner with GPU support.

Direct BoTorch implementation (not poli-baselines) for 30x GPU speedup.
Based on the official BoTorch TuRBO tutorial.

Usage:
    PYTHONPATH=/path/to/NeurIPS CUDA_VISIBLE_DEVICES=1 \
        python scripts/run_turbo_budget.py --task adip --seed 42 --budget 1000
"""

import argparse
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from torch.quasirandom import SobolEngine

from src.analysis.config import GUACAMOL_TO_POLI

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Expected GP fitting exceptions (catch these, re-raise everything else)
_GP_FIT_ERRORS = (
    torch.linalg.LinAlgError,
    ValueError,
    RuntimeError,
)


def _is_cuda_error(e: Exception) -> bool:
    return isinstance(e, torch.cuda.OutOfMemoryError) or (
        isinstance(e, RuntimeError) and "CUDA" in str(e)
    )


# ---------------------------------------------------------------------------
# TuRBO state (from BoTorch tutorial)
# ---------------------------------------------------------------------------

@dataclass
class TurboState:
    dim: int
    batch_size: int = 1
    length: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    failure_counter: int = 0
    failure_tolerance: int = field(init=False)
    success_counter: int = 0
    success_tolerance: int = 10
    best_value: float = -float("inf")
    restart_triggered: bool = False

    def __post_init__(self):
        self.failure_tolerance = math.ceil(
            max(4.0 / self.batch_size, float(self.dim) / self.batch_size)
        )


def update_state(state, Y_next):
    if max(Y_next) > state.best_value + 1e-3 * abs(state.best_value):
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


def generate_batch(state, model, X, Y, n_candidates=5000, device="cpu", dtype=torch.double):
    x_center = X[Y.argmax(), :].clone()
    dim = X.shape[-1]

    covar = model.covar_module
    if hasattr(covar, 'base_kernel') and hasattr(covar.base_kernel, 'lengthscale'):
        ls = covar.base_kernel.lengthscale.detach().view(-1)
    else:
        ls = torch.ones(dim, dtype=dtype, device=device)
    weights = ls / ls.mean()
    weights = weights / torch.prod(weights.pow(1.0 / len(weights)))

    tr_lb = torch.clamp(x_center - weights * state.length / 2.0, 0.0, 1.0)
    tr_ub = torch.clamp(x_center + weights * state.length / 2.0, 0.0, 1.0)

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
        X_next = ts(X_cand, num_samples=state.batch_size)
    return X_next


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_turbo(task_name, seed, budget, n_cold_start=5, device="cuda"):
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

    # Normalize to [0, 1]^D
    z_min = Z_init.min(dim=0).values
    z_max = Z_init.max(dim=0).values
    span = z_max - z_min + 1e-8

    def normalize(z):
        return (z - z_min) / span

    def denormalize(z_norm):
        return z_norm * span + z_min

    X = normalize(Z_init).to(dtype=dtype, device=device)
    Y = torch.tensor(cs_scores, dtype=dtype, device=device).unsqueeze(-1)

    max_iter = budget - n_cold_start
    state = TurboState(dim=dim, batch_size=1)
    state.best_value = Y.max().item()

    best_score = state.best_value
    best_trace = cs_scores.copy()
    all_scores = cs_scores.copy()
    smiles_set = set(cs_smiles)
    gp_failures = 0
    decode_failures = 0

    for i in range(1, len(best_trace)):
        best_trace[i] = max(best_trace[i], best_trace[i - 1])

    logger.info(f"TuRBO-GPU: task={poli_task} seed={seed} budget={budget} "
                f"dim={dim} cold_start_best={best_score:.4f}")

    t0 = time.time()
    for it in range(max_iter):
        # Fit GP — catch specific errors, always re-raise CUDA errors
        use_random = False
        try:
            likelihood = GaussianLikelihood(
                noise_constraint=Interval(5e-4, 0.2)
            ).to(device)
            covar_module = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=dim)
            ).to(device)
            model = SingleTaskGP(
                X, Y,
                covar_module=covar_module,
                likelihood=likelihood,
                outcome_transform=Standardize(m=1),
            ).to(device)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
        except _GP_FIT_ERRORS as e:
            if _is_cuda_error(e):
                raise
            logger.warning(f"Iter {it}: GP fit failed ({type(e).__name__}), trying fallback")
            gp_failures += 1
            try:
                model = SingleTaskGP(
                    X, Y, outcome_transform=Standardize(m=1)
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
            X_next = torch.rand(1, dim, dtype=dtype, device=device)
        else:
            model.eval()
            X_next = generate_batch(
                state, model, X, Y,
                n_candidates=min(5000, max(2000, 10 * dim)),
                device=device, dtype=dtype,
            )

        # Denormalize and decode
        z_full = denormalize(X_next.float())
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

        # Use 0.0 for GP data when decode failed (nan breaks GP)
        gp_score = 0.0 if math.isnan(score) else score
        Y_next = torch.tensor([[gp_score]], dtype=dtype, device=device)
        X = torch.cat([X, X_next.to(dtype)], dim=0)
        Y = torch.cat([Y, Y_next], dim=0)

        state = update_state(state, Y_next)

        if state.restart_triggered:
            state = TurboState(dim=dim, batch_size=1)
            state.best_value = Y.max().item()

        all_scores.append(score)
        if not math.isnan(score):
            best_score = max(best_score, score)
        best_trace.append(best_score)

        if (it + 1) % 100 == 0:
            elapsed = time.time() - t0
            logger.info(f"Iteration {it + 1}/{max_iter}: best={best_score:.4f} "
                        f"TR_len={state.length:.4f} gp_fail={gp_failures} "
                        f"decode_fail={decode_failures} elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0
    logger.info(f"TuRBO-GPU done: best={best_score:.4f} gp_failures={gp_failures} "
                f"decode_failures={decode_failures} elapsed={elapsed:.0f}s")

    return {
        "method": "turbo",
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
            "input_dim": dim,
            "failure_tolerance": state.failure_tolerance,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="TuRBO budget-scaling runner (GPU)")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    result = run_turbo(args.task, args.seed, args.budget, device=args.device)

    out_dir = Path("results/guacamol/spherevo")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_b{args.budget}" if args.budget != 30 else ""
    out_path = out_dir / f"ours_turbo_{result['task_id']}_s{args.seed}_256d{suffix}.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
