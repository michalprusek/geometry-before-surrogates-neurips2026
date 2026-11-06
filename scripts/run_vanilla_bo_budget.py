#!/usr/bin/env python3
"""Vanilla BO (Hvarfner 2025) budget-scaling runner with GPU support.

Direct BoTorch implementation with Thompson Sampling acquisition.
Uses Matern-5/2 ARD kernel with Standardize outcome transform.

Usage:
    PYTHONPATH=/path/to/NeurIPS CUDA_VISIBLE_DEVICES=1 \
        python scripts/run_vanilla_bo_budget.py --task adip --seed 42 --budget 1000
"""

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll
from botorch.generation import MaxPosteriorSampling
from botorch.models import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.kernels import MaternKernel, ScaleKernel
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


def run_vanilla_bo(task_name, seed, budget, n_cold_start=5, device="cuda"):
    from src.codec.selfies_vae import SELFIESVAECodec
    from src.oracle.guacamol import GuacaMolOracle, load_zinc_smiles

    torch.manual_seed(seed)
    np.random.seed(seed)
    dtype = torch.double

    poli_task = GUACAMOL_TO_POLI.get(task_name, task_name)

    codec = SELFIESVAECodec.from_pretrained(device=device)
    oracle = GuacaMolOracle(task_name)
    dim = codec.embedding_dim

    # Cold start
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
    best_score = Y.max().item()
    best_trace = cs_scores.copy()
    all_scores = cs_scores.copy()
    smiles_set = set(cs_smiles)
    gp_failures = 0
    decode_failures = 0

    for i in range(1, len(best_trace)):
        best_trace[i] = max(best_trace[i], best_trace[i - 1])

    logger.info(f"VanillaBO-GPU: task={poli_task} seed={seed} budget={budget} "
                f"dim={dim} cold_start_best={best_score:.4f}")

    t0 = time.time()
    for it in range(max_iter):
        # Fit GP
        use_random = False
        try:
            covar_module = ScaleKernel(
                MaternKernel(nu=2.5, ard_num_dims=dim)
            ).to(device)
            model = SingleTaskGP(
                X, Y,
                covar_module=covar_module,
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
            # Thompson Sampling (much faster than EI optimization in 256D)
            try:
                n_candidates = min(5000, max(2000, 10 * dim))
                sobol = SobolEngine(dim, scramble=True)
                X_cand = sobol.draw(n_candidates).to(dtype=dtype, device=device)
                ts = MaxPosteriorSampling(model=model, replacement=False)
                with torch.no_grad():
                    X_next = ts(X_cand, num_samples=1)
            except _GP_FIT_ERRORS as e:
                if _is_cuda_error(e):
                    raise
                logger.warning(f"Iter {it}: Thompson Sampling failed, using random candidate")
                gp_failures += 1
                X_next = torch.rand(1, dim, dtype=dtype, device=device)

        # Decode and evaluate
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

        gp_score = 0.0 if math.isnan(score) else score
        Y_next = torch.tensor([[gp_score]], dtype=dtype, device=device)
        X = torch.cat([X, X_next.to(dtype)], dim=0)
        Y = torch.cat([Y, Y_next], dim=0)

        all_scores.append(score)
        if not math.isnan(score):
            best_score = max(best_score, score)
        best_trace.append(best_score)

        if (it + 1) % 100 == 0:
            elapsed = time.time() - t0
            logger.info(f"Iteration {it + 1}/{max_iter}: best={best_score:.4f} "
                        f"gp_fail={gp_failures} decode_fail={decode_failures} "
                        f"elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0
    logger.info(f"VanillaBO-GPU done: best={best_score:.4f} gp_failures={gp_failures} "
                f"decode_failures={decode_failures} elapsed={elapsed:.0f}s")

    return {
        "method": "vanilla_bo_hvarfner",
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
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Vanilla BO budget-scaling (GPU)")
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", type=int, default=1000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    result = run_vanilla_bo(args.task, args.seed, args.budget, device=args.device)

    out_dir = Path("results/guacamol/spherevo")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_b{args.budget}" if args.budget != 30 else ""
    out_path = out_dir / f"ours_vanilla_bo_hvarfner_{result['task_id']}_s{args.seed}_256d{suffix}.json"

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    logger.info(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
