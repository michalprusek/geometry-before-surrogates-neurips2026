#!/usr/bin/env python3
"""NAS-Bench-101 benchmark for SphereVO.

Tests CMA-ES subspace and baselines on the arch2vec 16D latent space
over NAS-Bench-101 (423K CIFAR-10 architectures).

Key differences from molecular benchmarks:
  - "Decode" is nearest-neighbor lookup in latent space (discrete arch set)
  - Oracle is table lookup returning validation accuracy (instant)
  - We MAXIMIZE validation accuracy (higher = better)
  - CMA-ES minimizes NEGATED accuracy internally

Since latent_dim=16 and default subspace_dim=16, CMA-ES subspace degenerates
to CMA-ES on the full sphere. The key ingredient tested is spherical
normalization, not dimensionality reduction.

Usage:
    PYTHONPATH=/path/to/NeurIPS python scripts/run_nasbench.py \
        --seeds 42-51 --methods cmaes_sub,turbo

    PYTHONPATH=/path/to/NeurIPS python scripts/run_nasbench.py \
        --seeds 42 --methods cmaes_sub,cmaes_full,random,turbo,gp_bo
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cma
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.analysis.config import NASBENCH_METHODS, RESULT_DIRS
from src.provenance import collect_provenance

RESULTS_DIR = PROJECT_ROOT / RESULT_DIRS["nasbench"]

METHODS = NASBENCH_METHODS


# ---------------------------------------------------------------------------
# LatentBlackBox for NAS-Bench-101
# ---------------------------------------------------------------------------


class NASBenchLatentBlackBox:
    """Wraps arch2vec codec + NASBench oracle as a callable.

    Input:  z in R^16 (latent vector)
    Output: validation accuracy (higher = better)

    "Decoding" maps the continuous latent vector to the nearest known
    architecture via L2 distance in the precomputed embedding table.
    """

    def __init__(self, codec, oracle):
        self.codec = codec
        self.oracle = oracle
        self.n_calls = 0

    def __call__(self, z: np.ndarray) -> float:
        """Evaluate a single latent vector. Returns validation accuracy."""
        nearest = self.codec.find_nearest(z, k=1)
        arch_hash, dist = nearest[0]
        score = self.oracle(arch_hash)
        self.n_calls += 1
        return float(score)

    def evaluate_batch(self, Z: np.ndarray) -> np.ndarray:
        """Evaluate multiple latent vectors. Returns [N] accuracy array."""
        Z = np.atleast_2d(Z)
        return np.array([self(z) for z in Z])


# ---------------------------------------------------------------------------
# Cold start: sample random architectures from the dataset
# ---------------------------------------------------------------------------


def generate_cold_start(
    codec, oracle, n_cold: int, seed: int
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Sample random architectures, encode to latent, score via oracle.

    Returns (z0 [n_cold, 16], y0 [n_cold], arch_hashes [n_cold]).
    """
    all_emb, all_hashes = codec.get_all_embeddings()
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(all_hashes), size=n_cold, replace=False)

    z0 = all_emb[idx]  # [n_cold, 16]
    hashes = [str(all_hashes[i]) for i in idx]  # str keys for oracle lookup
    y0 = np.array([oracle(h) for h in hashes])  # [n_cold]
    return z0, y0, hashes


# ---------------------------------------------------------------------------
# Spherical normalization helpers
# ---------------------------------------------------------------------------


def _normalize(x: np.ndarray) -> np.ndarray:
    """Normalize to unit sphere."""
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


# ---------------------------------------------------------------------------
# Method implementations
# ---------------------------------------------------------------------------


def run_cmaes_subspace(
    bb: NASBenchLatentBlackBox,
    all_emb: np.ndarray,
    D: int,
    d: int,
    n_cold: int,
    n_iter: int,
    pop_size: int,
    seed: int,
) -> dict:
    """CMA-ES in QR subspace with spherical normalization.

    Since D=16 and d=16 by default, this is CMA-ES on the full sphere.
    The key ingredient is spherical normalization constraining search to
    the decoder's support and eliminating the uninformative radial DOF.
    """
    rng = np.random.RandomState(seed)

    # Compute mean norm of all embeddings (for scaling after normalization)
    mean_norm = float(np.mean(np.linalg.norm(all_emb, axis=1)))

    # Cold start
    z0, y0, _ = generate_cold_start(bb.codec, bb.oracle, n_cold, seed)

    best_idx = int(np.argmax(y0))  # maximize accuracy
    best_score = y0[best_idx]
    best_z = z0[best_idx].copy()

    # QR projection
    A, _ = np.linalg.qr(rng.randn(D, d))

    # Project cold-start to subspace
    cs_proj = z0 @ A  # [n_cold, d]

    initial_mean = cs_proj[best_idx]
    _std = float(np.std(cs_proj))
    sigma0 = _std if _std > 1e-8 else 0.5

    es = cma.CMAEvolutionStrategy(
        initial_mean.flatten().tolist(),
        sigma0,
        {"popsize": pop_size, "verbose": -9, "seed": seed},
    )

    # Initial tell: CMA-ES minimizes, so negate accuracy
    _ = es.ask()
    if len(cs_proj) >= pop_size:
        top_idx = np.argsort(-y0)[:pop_size]  # best accuracy first
        es.tell(cs_proj[top_idx].tolist(), (-y0[top_idx]).tolist())
    else:
        n_extra = pop_size - len(cs_proj)
        extra_x = initial_mean + rng.randn(n_extra, d) * 0.1
        extra_f = np.full(n_extra, float(-np.min(y0)))  # worst cold-start score
        init_x = np.vstack([cs_proj, extra_x])
        init_f = np.concatenate([-y0, extra_f])
        es.tell(init_x.tolist(), init_f.tolist())

    trace = [float(best_score)]
    all_scores = list(y0)

    current_pop = None
    pop_idx = 0
    gen_raw, gen_scores_neg = [], []

    for i in range(n_iter):
        if current_pop is None or pop_idx >= len(current_pop):
            if gen_raw:
                es.tell(gen_raw, gen_scores_neg)
                gen_raw, gen_scores_neg = [], []
            current_pop = es.ask()
            pop_idx = 0

        raw = current_pop[pop_idx]
        gen_raw.append(raw)
        pop_idx += 1

        # Lift to ambient: project back, normalize to sphere, scale by mean_norm
        v = np.array(raw).reshape(1, -1)
        u = (v @ A.T).flatten()
        u_normed = _normalize(u.reshape(1, -1)).flatten() * mean_norm

        score = bb(u_normed)
        gen_scores_neg.append(-score)  # CMA-ES minimizes
        all_scores.append(score)

        if score > best_score:
            best_score = score
            best_z = u_normed.copy()
        trace.append(float(best_score))

    if gen_raw:
        es.tell(gen_raw, gen_scores_neg)

    return {
        "best_score": float(best_score),
        "best_trace": trace,
        "all_scores": [float(s) for s in all_scores],
    }


def run_cmaes_full(
    bb: NASBenchLatentBlackBox,
    all_emb: np.ndarray,
    D: int,
    n_cold: int,
    n_iter: int,
    pop_size: int,
    seed: int,
) -> dict:
    """CMA-ES in full 16D ambient space (no spherical normalization)."""
    rng = np.random.RandomState(seed)

    z0, y0, _ = generate_cold_start(bb.codec, bb.oracle, n_cold, seed)

    best_idx = int(np.argmax(y0))
    best_score = y0[best_idx]
    best_z = z0[best_idx].copy()

    # Bounds based on embedding distribution
    emb_min = float(np.min(all_emb))
    emb_max = float(np.max(all_emb))

    _std = float(np.std(all_emb))
    sigma0 = _std if _std > 1e-8 else 0.5

    es = cma.CMAEvolutionStrategy(
        best_z.flatten().tolist(),
        sigma0,
        {
            "popsize": pop_size,
            "verbose": -9,
            "seed": seed,
            "bounds": [emb_min, emb_max],
        },
    )

    _ = es.ask()
    if len(z0) >= pop_size:
        top_idx = np.argsort(-y0)[:pop_size]
        es.tell(z0[top_idx].tolist(), (-y0[top_idx]).tolist())
    else:
        es.tell(z0.tolist(), (-y0).tolist())

    trace = [float(best_score)]
    all_scores = list(y0)

    current_pop = None
    pop_idx = 0
    gen_raw, gen_scores_neg = [], []

    for i in range(n_iter):
        if current_pop is None or pop_idx >= len(current_pop):
            if gen_raw:
                es.tell(gen_raw, gen_scores_neg)
                gen_raw, gen_scores_neg = [], []
            current_pop = es.ask()
            pop_idx = 0

        raw = current_pop[pop_idx]
        gen_raw.append(raw)
        pop_idx += 1

        z = np.clip(np.array(raw), emb_min, emb_max)
        score = bb(z)
        gen_scores_neg.append(-score)
        all_scores.append(score)

        if score > best_score:
            best_score = score
            best_z = z.copy()
        trace.append(float(best_score))

    if gen_raw:
        es.tell(gen_raw, gen_scores_neg)

    return {
        "best_score": float(best_score),
        "best_trace": trace,
        "all_scores": [float(s) for s in all_scores],
    }


def run_geowalk(
    bb: NASBenchLatentBlackBox,
    all_emb: np.ndarray,
    D: int,
    n_cold: int,
    n_iter: int,
    seed: int,
) -> dict:
    """GeoWalk: geodesic random walk on the unit sphere with decaying angular step.

    Identical algorithm to ``GeoWalkSolver`` in src/spherevo/solver.py.
    16 random sphere samples followed by 9 geodesic refinement steps from the
    current best (argmax for NASBench since accuracy is maximized).
    The mean_norm of all_emb is used as the spherical radius (same convention
    as ``run_cmaes_subspace`` in this script).
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)

    mean_norm = float(np.mean(np.linalg.norm(all_emb, axis=1)))

    z0, y0, _ = generate_cold_start(bb.codec, bb.oracle, n_cold, seed)

    best_idx = int(np.argmax(y0))  # maximize accuracy
    best_score = float(y0[best_idx])
    best_z = z0[best_idx].copy()

    trace = [best_score]
    all_scores = list(y0)
    pts_history = list(z0)
    ys_history = list(y0)

    EXPLORE_STEPS = 16
    ANGLE_INIT = 0.60
    ANGLE_SHRINK = 0.45

    for step in range(n_iter):
        if step < EXPLORE_STEPS:
            # Random sphere sample
            x = rng.randn(D)
            x = x / (np.linalg.norm(x) + 1e-10) * mean_norm
        else:
            # Greedy geodesic walk from current best (argmax for accuracy)
            ys_arr = np.array(ys_history)
            pts_arr = np.array(pts_history)
            best_local = int(np.argmax(ys_arr))
            anc = pts_arr[best_local]
            anc_n = anc / (np.linalg.norm(anc) + 1e-10)

            v = rng.randn(D)
            v -= np.dot(v, anc_n) * anc_n
            v /= (np.linalg.norm(v) + 1e-10)

            t = (step - EXPLORE_STEPS) / 8.0
            angle = ANGLE_INIT - ANGLE_SHRINK * t

            x = np.cos(angle) * anc_n + np.sin(angle) * v
            x = x / (np.linalg.norm(x) + 1e-10) * mean_norm

        score = float(bb(x))
        all_scores.append(score)
        pts_history.append(x.copy())
        ys_history.append(score)

        if score > best_score:
            best_score = score
            best_z = x.copy()
        trace.append(best_score)

    return {
        "best_score": float(best_score),
        "best_trace": trace,
        "all_scores": [float(s) for s in all_scores],
    }


def run_random_search(
    bb: NASBenchLatentBlackBox,
    all_emb: np.ndarray,
    D: int,
    n_total: int,
    seed: int,
) -> dict:
    """Random search: sample random real architectures from the dataset.

    Samples uniformly from the precomputed architecture embeddings,
    ensuring the same evaluation pipeline as other methods (NN-decode
    in latent space). This is fairer than sampling Gaussian noise.
    """
    rng = np.random.RandomState(seed)

    _, all_hashes = bb.codec.get_all_embeddings()

    best_score = -float("inf")
    trace = []
    all_scores = []

    for _ in range(n_total):
        idx = rng.randint(0, len(all_hashes))
        arch_hash = str(all_hashes[idx])
        score = float(bb.oracle(arch_hash))
        all_scores.append(score)
        if score > best_score:
            best_score = score
        trace.append(float(best_score))

    return {
        "best_score": float(best_score),
        "best_trace": trace,
        "all_scores": [float(s) for s in all_scores],
    }


def run_turbo(
    bb: NASBenchLatentBlackBox,
    all_emb: np.ndarray,
    D: int,
    n_cold: int,
    n_iter: int,
    seed: int,
) -> dict:
    """TuRBO in full 16D latent space.

    GP should work well at 16D -- this tests whether surrogates help
    on a structured, low-dimensional latent space.
    """
    import math

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

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # Precompute embedding stats for GP fallback
    emb_mean = np.mean(all_emb, axis=0)
    emb_std = np.std(all_emb, axis=0)

    z0, y0, _ = generate_cold_start(bb.codec, bb.oracle, n_cold, seed)

    best_idx = int(np.argmax(y0))
    best_score = y0[best_idx]
    best_z = z0[best_idx].copy()

    all_points = list(z0)
    all_scores_list = list(y0)

    # TuRBO state
    tr_length = 0.8
    length_min = 0.5**7
    length_max = 1.6
    failure_tol = math.ceil(D / 1)  # batch_size=1
    success_tol = 3
    success_counter = 0
    failure_counter = 0

    trace = [float(best_score)]
    gp_fallbacks = 0

    for i in range(n_iter):
        # Rescale data to [0,1]^D for GP
        X_np = np.array(all_points)
        lb = X_np.min(axis=0)
        ub = X_np.max(axis=0)
        spread = ub - lb
        spread[spread < 1e-8] = 1.0
        X_unit = (X_np - lb) / spread

        # GP maximizes, so use raw scores (accuracy, higher=better)
        Y_vals = np.array(all_scores_list)

        train_X = torch.tensor(X_unit, dtype=torch.float64)
        train_Y = torch.tensor(Y_vals, dtype=torch.float64).unsqueeze(-1)

        candidate_np = None
        try:
            likelihood = GaussianLikelihood(
                noise_constraint=Interval(1e-8, 1e-3)
            )
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5,
                    ard_num_dims=D,
                    lengthscale_constraint=Interval(0.005, 4.0),
                ),
                outputscale_constraint=Interval(0.05, 20.0),
            )
            model = SingleTaskGP(
                train_X,
                train_Y,
                likelihood=likelihood,
                covar_module=covar_module,
                outcome_transform=Standardize(m=1),
            )
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            model.eval()

            # TR center = best point in [0,1]^D
            best_unit = torch.tensor(
                (best_z - lb) / spread, dtype=torch.float64
            )

            # Lengthscale-weighted TR bounds
            ls = (
                model.covar_module.base_kernel.lengthscale.detach()
                .view(-1)
                .double()
            )
            weights = ls / ls.mean()
            weights = weights / weights.prod().pow(1.0 / D)

            tr_lb = torch.clamp(
                best_unit - weights * tr_length / 2.0, 0.0, 1.0
            )
            tr_ub = torch.clamp(
                best_unit + weights * tr_length / 2.0, 0.0, 1.0
            )

            # Sobol candidates within TR
            n_cand = min(5000, max(2000, 10 * D))
            sobol = SobolEngine(D, scramble=True, seed=seed + i)
            pert = sobol.draw(n_cand).double()
            pert = tr_lb + (tr_ub - tr_lb) * pert

            # Probabilistic perturbation mask
            prob_perturb = min(20.0 / D, 1.0)
            mask = torch.rand(n_cand, D) <= prob_perturb
            no_pert = mask.sum(dim=1) == 0
            if no_pert.any():
                mask[
                    no_pert,
                    torch.randint(0, D, size=(int(no_pert.sum()),)),
                ] = True

            X_cand = best_unit.expand(n_cand, -1).clone()
            X_cand[mask] = pert[mask]

            # Thompson Sampling
            ts = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():
                X_next = ts(X_cand, num_samples=1)
            candidate_unit = X_next.squeeze(0).numpy()

            # Rescale back to latent space
            candidate_np = candidate_unit * spread + lb
        except Exception as e:
            logger.warning(f"GP failed at iter {i}: {type(e).__name__}: {e}")
            gp_fallbacks += 1
            candidate_np = emb_mean + rng.randn(D) * emb_std

        score = bb(candidate_np)
        all_points.append(candidate_np)
        all_scores_list.append(score)

        # Update TuRBO state (maximizing)
        if score > best_score + 1e-3 * abs(best_score):
            success_counter += 1
            failure_counter = 0
        else:
            success_counter = 0
            failure_counter += 1

        if success_counter == success_tol:
            tr_length = min(2.0 * tr_length, length_max)
            success_counter = 0
        elif failure_counter == failure_tol:
            tr_length /= 2.0
            failure_counter = 0

        if score > best_score:
            best_score = score
            best_z = candidate_np.copy()

        # Restart on TR collapse
        if tr_length < length_min:
            tr_length = 0.8
            success_counter = 0
            failure_counter = 0

        trace.append(float(best_score))

    return {
        "best_score": float(best_score),
        "best_trace": trace,
        "all_scores": [float(s) for s in all_scores_list],
        "gp_fallback_count": gp_fallbacks,
    }


def run_gp_bo(
    bb: NASBenchLatentBlackBox,
    all_emb: np.ndarray,
    D: int,
    n_cold: int,
    n_iter: int,
    seed: int,
) -> dict:
    """Vanilla GP-BO with Expected Improvement in 16D latent space."""
    import torch
    from botorch.acquisition import qExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # Precompute embedding stats for GP fallback
    emb_mean = np.mean(all_emb, axis=0)
    emb_std = np.std(all_emb, axis=0)

    z0, y0, _ = generate_cold_start(bb.codec, bb.oracle, n_cold, seed)

    best_idx = int(np.argmax(y0))
    best_score = y0[best_idx]

    # GP-BO maximizes, so use raw accuracy
    train_X = torch.tensor(z0, dtype=torch.float64)
    train_Y = torch.tensor(y0, dtype=torch.float64).unsqueeze(-1)

    trace = [float(best_score)]
    all_scores = list(y0)

    # Bounds from embedding distribution (with symmetric margin)
    _margin = 0.5 * (float(np.max(all_emb)) - float(np.min(all_emb)))
    emb_min = float(np.min(all_emb)) - _margin
    emb_max = float(np.max(all_emb)) + _margin
    gp_fallbacks = 0

    for i in range(n_iter):
        try:
            model = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            acqf = qExpectedImprovement(model, best_f=train_Y.max())
            bounds = torch.tensor(
                [[emb_min] * D, [emb_max] * D], dtype=torch.float64
            )
            candidate, _ = optimize_acqf(
                acqf,
                bounds=bounds,
                q=1,
                num_restarts=5,
                raw_samples=256,
            )
            z_new = candidate.detach().numpy().flatten()
        except Exception as e:
            logger.warning(f"GP-BO failed at iter {i}: {type(e).__name__}: {e}")
            gp_fallbacks += 1
            z_new = emb_mean + rng.randn(D) * emb_std

        score = bb(z_new)
        all_scores.append(score)

        if score > best_score:
            best_score = score
        trace.append(float(best_score))

        train_X = torch.cat(
            [train_X, torch.tensor(z_new, dtype=torch.float64).unsqueeze(0)]
        )
        train_Y = torch.cat(
            [train_Y, torch.tensor([[score]], dtype=torch.float64)]
        )

    return {
        "best_score": float(best_score),
        "best_trace": trace,
        "all_scores": [float(s) for s in all_scores],
        "gp_fallback_count": gp_fallbacks,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="NAS-Bench-101 benchmark for SphereVO"
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="42-51",
        help="Seed range (e.g. 42-51) or comma-separated (e.g. 42,43,44)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default="cmaes_sub,cmaes_full,random,turbo,gp_bo",
        help="Comma-separated methods",
    )
    parser.add_argument("--n-cold-start", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--subspace-dim", type=int, default=16)
    parser.add_argument("--pop-size", type=int, default=5)
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(RESULTS_DIR),
    )
    parser.add_argument(
        "--weights-dir",
        type=str,
        default=None,
        help="Path to arch2vec weights directory",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to NAS-Bench-101 data directory",
    )
    args = parser.parse_args()

    methods = args.methods.split(",")

    # Parse seeds
    if "-" in args.seeds:
        start, end = map(int, args.seeds.split("-"))
        seeds = list(range(start, end + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    os.makedirs(args.results_dir, exist_ok=True)

    # Load codec and oracle
    from src.codec.arch2vec import Arch2VecCodec
    from src.oracle.nasbench import NASBench101Oracle

    logger.info("Loading arch2vec codec...")
    codec = Arch2VecCodec.from_pretrained(
        weights_dir=args.weights_dir, device="cpu"
    )

    logger.info("Loading NAS-Bench-101 oracle...")
    oracle = NASBench101Oracle(data_dir=args.data_dir)
    oracle.load()

    # Precompute all embeddings once
    all_emb, all_hashes = codec.get_all_embeddings()
    D = codec.dim  # 16
    d = min(args.subspace_dim, D)

    logger.info(
        f"Loaded {len(all_hashes)} architectures, "
        f"latent_dim={D}, subspace_dim={d}"
    )

    bb = NASBenchLatentBlackBox(codec, oracle)

    provenance = collect_provenance()

    total = len(seeds) * len(methods)
    done = 0

    print(f"\n{'=' * 60}")
    print(f"NAS-Bench-101 Benchmark (D={D}, d={d})")
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds}")
    print(f"Budget: {args.n_cold_start} cold-start + {args.n_iter} iter = "
          f"{args.n_cold_start + args.n_iter} total")
    print(f"{'=' * 60}")

    for seed in seeds:
        for method in methods:
            result_file = os.path.join(
                args.results_dir, f"{method}_cifar10_s{seed}.json"
            )
            if os.path.exists(result_file):
                done += 1
                print(f"  [{done}/{total}] {method} s{seed}: SKIP (exists)")
                continue

            t0 = time.time()

            try:
                if method == "cmaes_sub":
                    result = run_cmaes_subspace(
                        bb,
                        all_emb,
                        D,
                        d,
                        args.n_cold_start,
                        args.n_iter,
                        args.pop_size,
                        seed,
                    )
                elif method == "cmaes_full":
                    result = run_cmaes_full(
                        bb,
                        all_emb,
                        D,
                        args.n_cold_start,
                        args.n_iter,
                        args.pop_size,
                        seed,
                    )
                elif method == "random":
                    result = run_random_search(
                        bb,
                        all_emb,
                        D,
                        args.n_cold_start + args.n_iter,
                        seed,
                    )
                elif method == "turbo":
                    result = run_turbo(
                        bb,
                        all_emb,
                        D,
                        args.n_cold_start,
                        args.n_iter,
                        seed,
                    )
                elif method == "gp_bo":
                    result = run_gp_bo(
                        bb,
                        all_emb,
                        D,
                        args.n_cold_start,
                        args.n_iter,
                        seed,
                    )
                elif method == "geowalk":
                    result = run_geowalk(
                        bb,
                        all_emb,
                        D,
                        args.n_cold_start,
                        args.n_iter,
                        seed,
                    )
                else:
                    raise ValueError(f"Unknown method: {method}")
            except Exception as e:
                logger.error(
                    f"{method} s{seed} failed: {type(e).__name__}: {e}"
                )
                result = {
                    "best_score": float("-inf"),
                    "best_trace": [],
                    "all_scores": [],
                    "error": f"{type(e).__name__}: {e}",
                }

            dt = time.time() - t0
            done += 1

            result.update(
                {
                    "method": method,
                    "task": "cifar10",
                    "D": D,
                    "d": d,
                    "seed": seed,
                    "n_cold_start": args.n_cold_start,
                    "n_iter": args.n_iter,
                    "n_total": args.n_cold_start + args.n_iter,
                    "higher_is_better": True,
                    "pop_size": args.pop_size if "cmaes" in method else None,
                    "elapsed_s": round(dt, 3),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            result.update(provenance)

            with open(result_file, "w") as f:
                json.dump(result, f, indent=2)

            print(
                f"  [{done}/{total}] {method} s{seed}: "
                f"{result['best_score']:.4f} {dt:.1f}s"
            )

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY — NAS-Bench-101 (cifar10)")
    print(f"{'=' * 60}")

    for method in methods:
        scores = []
        for seed in seeds:
            f = os.path.join(args.results_dir, f"{method}_cifar10_s{seed}.json")
            if os.path.exists(f):
                with open(f) as fh:
                    d_loaded = json.load(fh)
                scores.append(d_loaded["best_score"])
        if scores:
            print(
                f"  {method:15s}: {np.mean(scores):.4f} +/- {np.std(scores):.4f} "
                f"(n={len(scores)})"
            )


if __name__ == "__main__":
    main()
