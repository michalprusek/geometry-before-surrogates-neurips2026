"""LassoBench CMA-ES Subspace Benchmark.

Tests SphereVO's CMA-ES subspace approach on LassoBench tasks
(60D-7129D weighted Lasso hyperparameter optimization).

Matches GuacaMol/GSM8K setup: 5 cold-start + 25 iterations = 30 total evals.
Compares: CMA-ES subspace (ours), CMA-ES full-D, random search, GP-BO.

Tasks:
    Synthetic: synt_simple (60D), synt_medium (100D), synt_high (300D), synt_hard (1000D)
    Real-world: dna (180D), leukemia (7129D)

Usage:
    PYTHONPATH=/path/to/NeurIPS python scripts/run_lassobench.py
    PYTHONPATH=/path/to/NeurIPS python scripts/run_lassobench.py --tasks dna,leukemia --seeds 42-51
    PYTHONPATH=/path/to/NeurIPS python scripts/run_lassobench.py --real-only --seeds 42-51
"""

import argparse
import json
import logging
import os
import time
from datetime import datetime

import cma
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TASKS = ["synt_simple", "synt_medium", "synt_high", "synt_hard"]
TASK_DIMS = {"synt_simple": 60, "synt_medium": 100, "synt_high": 300, "synt_hard": 1000}

REAL_TASKS = {"dna": 180, "leukemia": 7129}


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(norms, 1e-8)


def _build_projection(method: str, D: int, d: int,
                      cs_points: np.ndarray, cs_scores: np.ndarray,
                      rng: np.random.RandomState) -> np.ndarray:
    """Build D x d projection matrix A using the specified method.

    Returns A such that z_low = x_high @ A  and  x_high ~ z_low @ A.T
    """
    if method == "random_qr":
        A, _ = np.linalg.qr(rng.randn(D, d))

    elif method == "pca":
        # PCA from cold-start: top-d principal components
        X_centered = cs_points - cs_points.mean(axis=0, keepdims=True)
        if X_centered.shape[0] >= d:
            _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
            A = Vt[:d].T  # [D, d]
        else:
            # Not enough points for d components; pad with random
            _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
            n_have = Vt.shape[0]
            A_part = Vt[:n_have].T  # [D, n_have]
            # Fill remaining with random orthogonal to existing
            A_rand, _ = np.linalg.qr(rng.randn(D, d))
            A = np.hstack([A_part, A_rand[:, n_have:d]])
            A, _ = np.linalg.qr(A)  # Re-orthogonalize
            A = A[:, :d]

    elif method == "pls":
        # PLS: supervised projection using cold-start X and y
        from sklearn.cross_decomposition import PLSRegression
        n = cs_points.shape[0]
        n_comp = min(d, n - 1, D)
        if n_comp < 1:
            A, _ = np.linalg.qr(rng.randn(D, d))
        else:
            pls = PLSRegression(n_components=n_comp)
            pls.fit(cs_points, cs_scores.reshape(-1, 1))
            # x_rotations_ is [D, n_comp] -- the projection directions
            A_part = pls.x_rotations_  # [D, n_comp]
            if n_comp < d:
                A_rand, _ = np.linalg.qr(rng.randn(D, d))
                A = np.hstack([A_part, A_rand[:, n_comp:d]])
                A, _ = np.linalg.qr(A)
                A = A[:, :d]
            else:
                A, _ = np.linalg.qr(A_part)
                A = A[:, :d]

    elif method == "random_coords":
        # Axis-aligned: pick d random coordinates
        chosen = rng.choice(D, size=d, replace=False)
        A = np.zeros((D, d))
        for j, c in enumerate(chosen):
            A[c, j] = 1.0

    elif method == "random_gauss":
        # Random Gaussian without orthogonalization
        A = rng.randn(D, d)
        # Normalize columns to unit length
        norms = np.linalg.norm(A, axis=0, keepdims=True)
        A = A / np.maximum(norms, 1e-8)

    else:
        raise ValueError(f"Unknown projection method: {method}")

    return A


PROJECTION_METHODS = ["random_qr", "pca", "pls", "random_coords", "random_gauss"]


def run_cmaes_subspace(bench, D: int, d: int, n_cold: int, n_iter: int,
                       pop_size: int, seed: int,
                       projection: str = "random_qr",
                       lq_K: int = 0, lq_gate: float = 0.3) -> dict:
    """CMA-ES in subspace with configurable projection method.

    If lq_K > 0, enables linear surrogate pre-screening (lq-CMA-ES):
    oversamples lq_K * pop_size candidates, ranks with a linear model,
    evaluates only the top pop_size. Spearman gating auto-disables on
    unstructured problems.
    """
    rng = np.random.RandomState(seed)

    # Cold start: random in [-1, 1]^D
    cs_points = rng.uniform(-1, 1, size=(n_cold, D))
    cs_scores = np.array([bench.evaluate(x) for x in cs_points])

    # Build projection
    A = _build_projection(projection, D, d, cs_points, cs_scores, rng)

    # Project cold-start, find best
    cs_proj = cs_points @ A  # [N, d]
    best_idx = int(np.argmin(cs_scores))  # minimize
    best_score = cs_scores[best_idx]
    best_x = cs_points[best_idx]

    # CMA-ES in subspace
    initial_mean = cs_proj[best_idx]
    es = cma.CMAEvolutionStrategy(
        initial_mean.flatten().tolist(), 0.5,
        {"popsize": pop_size, "verbose": -9, "seed": seed}
    )

    # Initial tell with cold-start
    if len(cs_proj) >= pop_size:
        top_idx = np.argsort(cs_scores)[:pop_size]
        init_x = cs_proj[top_idx]
        init_f = cs_scores[top_idx]
    else:
        init_x = cs_proj
        init_f = cs_scores
        n_extra = pop_size - len(cs_proj)
        extra_x = initial_mean + rng.randn(n_extra, d) * 0.1
        extra_f = np.full(n_extra, float(np.max(cs_scores)))
        init_x = np.vstack([init_x, extra_x])
        init_f = np.concatenate([init_f, extra_f])

    _ = es.ask()
    es.tell(init_x.tolist(), init_f.tolist())

    trace = [float(best_score)]
    all_scores = list(cs_scores)

    # lq-CMA-ES: linear surrogate state
    if lq_K > 0:
        lq_model_X = list(cs_proj)
        lq_model_y = list(cs_scores)
        lq_active = False

    current_pop = None
    pop_idx = 0
    gen_raw, gen_scores = [], []

    for i in range(n_iter):
        if current_pop is None or pop_idx >= len(current_pop):
            if gen_raw:
                # lq-CMA-ES: gating + accumulation
                if lq_K > 0:
                    if len(lq_model_X) >= d + 1 and len(gen_scores) >= 3:
                        from scipy.stats import spearmanr
                        Xm = np.array(lq_model_X)
                        Phi = np.column_stack([np.ones(len(Xm)), Xm])
                        w, _, _, _ = np.linalg.lstsq(
                            Phi, np.array(lq_model_y), rcond=None,
                        )
                        Xg = np.array(gen_raw)
                        Phi_g = np.column_stack([np.ones(len(Xg)), Xg])
                        preds = Phi_g @ w
                        rho, _ = spearmanr(preds, gen_scores)
                        lq_active = bool(
                            not np.isnan(rho) and rho >= lq_gate
                        )
                    for r_m, s_m in zip(gen_raw, gen_scores):
                        lq_model_X.append(np.array(r_m))
                        lq_model_y.append(s_m)
                es.tell(gen_raw, gen_scores)
                gen_raw, gen_scores = [], []
            # lq-CMA-ES: pre-screen if active
            if lq_K > 0 and lq_active and len(lq_model_X) >= d + 1:
                all_cands = es.ask(number=lq_K * pop_size)
                Xm = np.array(lq_model_X)
                Phi = np.column_stack([np.ones(len(Xm)), Xm])
                w, _, _, _ = np.linalg.lstsq(
                    Phi, np.array(lq_model_y), rcond=None,
                )
                Xc = np.array(all_cands)
                Phi_c = np.column_stack([np.ones(len(Xc)), Xc])
                preds = Phi_c @ w
                top_idx = np.argsort(preds)[:pop_size]  # minimize
                current_pop = [all_cands[j] for j in top_idx]
            else:
                current_pop = es.ask()
            pop_idx = 0

        raw = current_pop[pop_idx]
        gen_raw.append(raw)
        pop_idx += 1

        # Lift to ambient space: project back and clip to [-1, 1]
        v = np.array(raw).reshape(1, -1)
        u = (v @ A.T).flatten()
        x = np.clip(u, -1, 1)

        score = bench.evaluate(x)
        gen_scores.append(score)
        all_scores.append(score)

        if score < best_score:
            best_score = score
            best_x = x.copy()
        trace.append(float(best_score))

    if gen_raw:
        es.tell(gen_raw, gen_scores)

    return {
        "best_score": float(best_score),
        "best_trace": trace,
        "all_scores": all_scores,
        "best_x": best_x.tolist(),
    }


def run_cmaes_full(bench, D: int, n_cold: int, n_iter: int,
                   pop_size: int, seed: int) -> dict:
    """CMA-ES in full ambient space (baseline)."""
    rng = np.random.RandomState(seed)

    cs_points = rng.uniform(-1, 1, size=(n_cold, D))
    cs_scores = np.array([bench.evaluate(x) for x in cs_points])

    best_idx = int(np.argmin(cs_scores))
    best_score = cs_scores[best_idx]
    best_x = cs_points[best_idx]

    es = cma.CMAEvolutionStrategy(
        best_x.tolist(), 0.5,
        {"popsize": pop_size, "verbose": -9, "seed": seed,
         "bounds": [-1, 1]}
    )

    _ = es.ask()
    if len(cs_points) >= pop_size:
        top_idx = np.argsort(cs_scores)[:pop_size]
        es.tell(cs_points[top_idx].tolist(), cs_scores[top_idx].tolist())
    else:
        es.tell(cs_points.tolist(), cs_scores.tolist())

    trace = [float(best_score)]
    all_scores = list(cs_scores)

    current_pop = None
    pop_idx = 0
    gen_raw, gen_scores = [], []

    for i in range(n_iter):
        if current_pop is None or pop_idx >= len(current_pop):
            if gen_raw:
                es.tell(gen_raw, gen_scores)
                gen_raw, gen_scores = [], []
            current_pop = es.ask()
            pop_idx = 0

        raw = current_pop[pop_idx]
        gen_raw.append(raw)
        pop_idx += 1

        x = np.clip(np.array(raw), -1, 1)
        score = bench.evaluate(x)
        gen_scores.append(score)
        all_scores.append(score)

        if score < best_score:
            best_score = score
            best_x = x.copy()
        trace.append(float(best_score))

    if gen_raw:
        es.tell(gen_raw, gen_scores)

    return {
        "best_score": float(best_score),
        "best_trace": trace,
        "all_scores": all_scores,
    }


def run_geowalk(bench, D: int, n_cold: int, n_iter: int, seed: int) -> dict:
    """GeoWalk: geodesic random walk on the unit sphere with decaying angular step.

    Adapted to LassoBench: cold-start in ``[-1, 1]^D`` provides the spherical
    radius via ``mean_norm = mean(||x_cs||)``. After cold-start the algorithm
    runs 16 random sphere samples and 9 geodesic refinement steps from the
    current best, exactly mirroring ``GeoWalkSolver`` in src/spherevo/solver.py.
    Candidates are clipped to ``[-1, 1]`` after the geodesic step to stay
    within the LassoBench box (same convention as ``run_cmaes_subspace``).

    Note: LassoBench MINIMIZES, so anchoring uses argmin.
    """
    rng = np.random.RandomState(seed)

    # Cold start: random in [-1, 1]^D
    cs_points = rng.uniform(-1, 1, size=(n_cold, D))
    cs_scores = np.array([bench.evaluate(x) for x in cs_points])

    mean_norm = float(np.mean(np.linalg.norm(cs_points, axis=1)))
    if mean_norm < 1e-6:
        mean_norm = 1.0

    best_idx = int(np.argmin(cs_scores))
    best_score = float(cs_scores[best_idx])
    best_x = cs_points[best_idx].copy()

    trace = [best_score]
    all_scores = list(cs_scores)

    EXPLORE_STEPS = 16
    ANGLE_INIT = 0.60
    ANGLE_SHRINK = 0.45

    pts_history = list(cs_points)
    ys_history = list(cs_scores)

    for step in range(n_iter):
        if step < EXPLORE_STEPS:
            # Random sphere sample (clipped to box)
            x = rng.randn(D)
            x = x / (np.linalg.norm(x) + 1e-10) * mean_norm
            x = np.clip(x, -1, 1)
        else:
            # Greedy geodesic walk from current best (argmin)
            ys_arr = np.array(ys_history)
            pts_arr = np.array(pts_history)
            best_local = int(np.argmin(ys_arr))
            anc = pts_arr[best_local]
            anc_n = anc / (np.linalg.norm(anc) + 1e-10)

            v = rng.randn(D)
            v -= np.dot(v, anc_n) * anc_n
            v /= (np.linalg.norm(v) + 1e-10)

            t = (step - EXPLORE_STEPS) / 8.0
            angle = ANGLE_INIT - ANGLE_SHRINK * t

            x = np.cos(angle) * anc_n + np.sin(angle) * v
            x = x / (np.linalg.norm(x) + 1e-10) * mean_norm
            x = np.clip(x, -1, 1)

        score = bench.evaluate(x)
        all_scores.append(score)
        pts_history.append(x.copy())
        ys_history.append(score)

        if score < best_score:
            best_score = float(score)
            best_x = x.copy()
        trace.append(best_score)

    return {
        "best_score": float(best_score),
        "best_trace": trace,
        "all_scores": all_scores,
        "best_x": best_x.tolist(),
    }


def run_random_search(bench, D: int, n_total: int, seed: int) -> dict:
    """Pure random search baseline."""
    rng = np.random.RandomState(seed)
    best_score = float("inf")
    trace = []
    all_scores = []

    for _ in range(n_total):
        x = rng.uniform(-1, 1, size=D)
        score = bench.evaluate(x)
        all_scores.append(score)
        if score < best_score:
            best_score = score
        trace.append(float(best_score))

    return {
        "best_score": float(best_score),
        "best_trace": trace,
        "all_scores": all_scores,
    }


def run_turbo(bench, D: int, n_cold: int, n_iter: int, seed: int) -> dict:
    """TuRBO in full ambient D-dim (Eriksson et al. 2019).

    Operates in [0,1]^D normalized space with Matern-5/2 ARD GP,
    Thompson Sampling acquisition, and lengthscale-weighted trust region.
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

    # Cold start: uniform [-1,1]^D
    cs_points = rng.uniform(-1, 1, size=(n_cold, D))
    cs_scores = np.array([bench.evaluate(x) for x in cs_points])

    best_idx = int(np.argmin(cs_scores))
    best_score = cs_scores[best_idx]
    best_x = cs_points[best_idx]

    all_points = list(cs_points)
    all_scores_list = list(cs_scores)

    # TuRBO state
    tr_length = 0.8
    length_min = 0.5 ** 7
    length_max = 1.6
    failure_tol = math.ceil(D / 1)  # batch_size=1
    success_tol = 3
    success_counter = 0
    failure_counter = 0

    trace = [float(best_score)]

    for i in range(n_iter):
        # Rescale all data to [0,1]^D for GP
        X_np = np.array(all_points)
        lb = X_np.min(axis=0)
        ub = X_np.max(axis=0)
        spread = ub - lb
        spread[spread < 1e-8] = 1.0
        X_unit = (X_np - lb) / spread

        # Negate scores: LassoBench minimizes, GP+TS maximizes
        Y_neg = -np.array(all_scores_list)

        train_X = torch.tensor(X_unit, dtype=torch.float64)
        train_Y = torch.tensor(Y_neg, dtype=torch.float64).unsqueeze(-1)

        # Fit GP
        candidate_np = None
        try:
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5, ard_num_dims=D,
                    lengthscale_constraint=Interval(0.005, 4.0),
                ),
                outputscale_constraint=Interval(0.05, 20.0),
            )
            model = SingleTaskGP(train_X, train_Y, likelihood=likelihood,
                                 covar_module=covar_module,
                                 outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            model.eval()

            # TR center = best point in [0,1]^D
            best_unit = torch.tensor(
                (best_x - lb) / spread, dtype=torch.float64
            )

            # Lengthscale-weighted TR bounds
            ls = model.covar_module.base_kernel.lengthscale.detach().view(-1).double()
            weights = ls / ls.mean()
            weights = weights / weights.prod().pow(1.0 / D)

            tr_lb = torch.clamp(best_unit - weights * tr_length / 2.0, 0.0, 1.0)
            tr_ub = torch.clamp(best_unit + weights * tr_length / 2.0, 0.0, 1.0)

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
                mask[no_pert, torch.randint(0, D, size=(int(no_pert.sum()),))] = True

            X_cand = best_unit.expand(n_cand, -1).clone()
            X_cand[mask] = pert[mask]

            # Thompson Sampling
            ts = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():
                X_next = ts(X_cand, num_samples=1)
            candidate_unit = X_next.squeeze(0).numpy()

            # Rescale back to [-1,1]^D
            candidate_np = candidate_unit * spread + lb
            candidate_np = np.clip(candidate_np, -1, 1)
        except (np.linalg.LinAlgError, RuntimeError, ValueError) as _gp_err:
            logger.warning(f"TuRBO GP step failed: {type(_gp_err).__name__}: {_gp_err}")
            candidate_np = rng.uniform(-1, 1, size=D)

        score = bench.evaluate(candidate_np)
        all_points.append(candidate_np)
        all_scores_list.append(score)

        # Update TuRBO state
        y_neg = -score
        prev_best_neg = -best_score
        if y_neg > prev_best_neg + 1e-3 * abs(prev_best_neg):
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

        if score < best_score:
            best_score = score
            best_x = candidate_np.copy()

        # Restart on TR collapse
        if tr_length < length_min:
            tr_length = 0.8
            success_counter = 0
            failure_counter = 0

        trace.append(float(best_score))

    return {
        "best_score": float(best_score),
        "best_trace": trace,
        "all_scores": all_scores_list,
    }


def _baxus_embedding(D: int, d: int, rng: np.random.RandomState) -> np.ndarray:
    """Create BAxUS count-sketch embedding S ∈ {-1,0,+1}^{D×d}.

    Each row of S has exactly one non-zero entry (±1), assigning each
    ambient dimension to exactly one subspace dimension.
    """
    S = np.zeros((D, d))
    bins = rng.randint(0, d, size=D)
    signs = rng.choice([-1.0, 1.0], size=D)
    for i in range(D):
        S[i, bins[i]] = signs[i]
    return S


def _baxus_increase_dim(S: np.ndarray, d_new: int,
                        rng: np.random.RandomState) -> np.ndarray:
    """Increase BAxUS embedding dimension by splitting the largest bins.

    Splits bins with the most ambient dimensions until we reach d_new
    subspace dimensions. Existing bin assignments are preserved.
    """
    D = S.shape[0]
    d_old = S.shape[1]
    if d_new <= d_old or d_new > D:
        return S

    # Find bin assignments and signs
    bins = np.argmax(np.abs(S), axis=1)
    signs = np.array([S[i, bins[i]] for i in range(D)])

    d_cur = d_old
    while d_cur < d_new:
        # Count bin sizes
        bin_counts = np.bincount(bins, minlength=d_cur)
        # Split the largest bin
        largest = int(np.argmax(bin_counts))
        members = np.where(bins == largest)[0]
        if len(members) <= 1:
            break
        # Randomly split members into two groups
        rng.shuffle(members)
        half = len(members) // 2
        for idx in members[half:]:
            bins[idx] = d_cur
        d_cur += 1

    # Rebuild S
    S_new = np.zeros((D, d_cur))
    for i in range(D):
        S_new[i, bins[i]] = signs[i]
    return S_new


def run_baxus(bench, D: int, d_init: int, n_cold: int, n_iter: int,
              seed: int) -> dict:
    """BAxUS: Bayesian optimization in adaptively expanding subspaces.

    Uses count-sketch sparse embedding with adaptive dimension doubling
    on trust region collapse. GP + Thompson Sampling in low-d subspace.
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

    d = min(d_init, D)

    # Cold start: uniform [-1,1]^D
    cs_points = rng.uniform(-1, 1, size=(n_cold, D))
    cs_scores = np.array([bench.evaluate(x) for x in cs_points])

    best_idx = int(np.argmin(cs_scores))
    best_score = cs_scores[best_idx]
    best_x = cs_points[best_idx]

    all_points_D = list(cs_points)  # ambient space
    all_scores_list = list(cs_scores)

    # Initial count-sketch embedding
    S = _baxus_embedding(D, d, rng)

    # Project cold-start to subspace
    X_low = np.array([x @ S for x in all_points_D])  # [N, d]

    # TuRBO state for subspace
    tr_length = 0.8
    length_min = 0.5 ** 7
    length_max = 1.6
    failure_tol = math.ceil(d / 1)
    success_tol = 3
    success_counter = 0
    failure_counter = 0

    trace = [float(best_score)]

    for i in range(n_iter):
        # Rescale X_low to [0,1]^d for GP
        X_np = np.array(X_low)
        lb = X_np.min(axis=0)
        ub = X_np.max(axis=0)
        spread = ub - lb
        spread[spread < 1e-8] = 1.0
        X_unit = (X_np - lb) / spread

        Y_neg = -np.array(all_scores_list)

        train_X = torch.tensor(X_unit, dtype=torch.float64)
        train_Y = torch.tensor(Y_neg, dtype=torch.float64).unsqueeze(-1)

        candidate_D = None
        try:
            likelihood = GaussianLikelihood(noise_constraint=Interval(1e-8, 1e-3))
            covar_module = ScaleKernel(
                MaternKernel(
                    nu=2.5, ard_num_dims=d,
                    lengthscale_constraint=Interval(0.005, 4.0),
                ),
                outputscale_constraint=Interval(0.05, 20.0),
            )
            model = SingleTaskGP(train_X, train_Y, likelihood=likelihood,
                                 covar_module=covar_module,
                                 outcome_transform=Standardize(m=1))
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)
            model.eval()

            # TR center = best point in [0,1]^d
            best_low = best_x @ S
            best_unit = torch.tensor(
                (best_low - lb) / spread, dtype=torch.float64
            )

            # Lengthscale-weighted TR
            ls = model.covar_module.base_kernel.lengthscale.detach().view(-1).double()
            weights = ls / ls.mean()
            weights = weights / weights.prod().pow(1.0 / d)

            tr_lb = torch.clamp(best_unit - weights * tr_length / 2.0, 0.0, 1.0)
            tr_ub = torch.clamp(best_unit + weights * tr_length / 2.0, 0.0, 1.0)

            n_cand = min(5000, max(2000, 10 * d))
            sobol = SobolEngine(d, scramble=True, seed=seed + i)
            pert = sobol.draw(n_cand).double()
            pert = tr_lb + (tr_ub - tr_lb) * pert

            prob_perturb = min(20.0 / d, 1.0)
            mask = torch.rand(n_cand, d) <= prob_perturb
            no_pert = mask.sum(dim=1) == 0
            if no_pert.any():
                mask[no_pert, torch.randint(0, d, size=(int(no_pert.sum()),))] = True

            X_cand = best_unit.expand(n_cand, -1).clone()
            X_cand[mask] = pert[mask]

            ts = MaxPosteriorSampling(model=model, replacement=False)
            with torch.no_grad():
                X_next = ts(X_cand, num_samples=1)
            candidate_unit = X_next.squeeze(0).numpy()

            # Rescale back to subspace coordinates
            candidate_low = candidate_unit * spread + lb

            # Lift to ambient via S^T and clip
            candidate_D = (candidate_low.reshape(1, -1) @ S.T).flatten()
            candidate_D = np.clip(candidate_D, -1, 1)
        except (np.linalg.LinAlgError, RuntimeError, ValueError) as _gp_err:
            logger.warning(f"BAxUS GP step failed: {type(_gp_err).__name__}: {_gp_err}")
            candidate_D = rng.uniform(-1, 1, size=D)

        score = bench.evaluate(candidate_D)
        all_points_D.append(candidate_D)
        all_scores_list.append(score)
        X_low = np.vstack([X_low, (candidate_D @ S).reshape(1, -1)])

        # Update TuRBO state
        y_neg = -score
        prev_best_neg = -best_score
        if y_neg > prev_best_neg + 1e-3 * abs(prev_best_neg):
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

        if score < best_score:
            best_score = score
            best_x = candidate_D.copy()

        # On TR collapse: increase subspace dimension
        if tr_length < length_min:
            d_new = min(2 * d, D)
            if d_new > d:
                S = _baxus_increase_dim(S, d_new, rng)
                d = S.shape[1]
                # Reproject all data to new subspace
                X_low = np.array([x @ S for x in all_points_D])
                failure_tol = math.ceil(d / 1)
            # Reset TR
            tr_length = 0.8
            success_counter = 0
            failure_counter = 0

        trace.append(float(best_score))

    return {
        "best_score": float(best_score),
        "best_trace": trace,
        "all_scores": all_scores_list,
    }


def run_hvarfner_gp(bench, D: int, n_cold: int, n_iter: int, seed: int) -> dict:
    """Hvarfner et al. 2024 "Vanilla BO with sqrt(D) prior" — official recipe.

    Source: https://github.com/hvarfner/vanilla_bo_in_highdim
    - configs/model/default.yaml + benchmarking/gp_priors.py: 'default' model
    - configs/acq/qlognei.yaml + configs/acq_opt/highdim.yaml

    Spec:
    - RBFKernel ARD (NOT Matern), outputscale fixed at 1 (no ScaleKernel)
    - LogNormal lengthscale prior: loc = 1.4 + 0.5*log(D), scale = sqrt(3)
    - LogNormal noise prior: loc = -4, scale = 1
    - GreaterThan(1e-4) constraints on lengthscale and noise
    - qLogNoisyExpectedImprovement acquisition (prune_baseline=True)
    - optimize_acqf: num_restarts=4, raw_samples=512, sample_around_best=True (sigma=0.1)
    - Bounds [-1,1]^D (LassoBench native; matches paper's lasso_dna.yaml)
    """
    import math
    import torch
    from botorch.acquisition.logei import qLogNoisyExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.optim import optimize_acqf
    from gpytorch.constraints import GreaterThan
    from gpytorch.kernels import RBFKernel
    from gpytorch.likelihoods import GaussianLikelihood
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from gpytorch.priors import LogNormalPrior

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    cs_points = rng.uniform(-1, 1, size=(n_cold, D))
    cs_scores = np.array([bench.evaluate(x) for x in cs_points])

    train_X = torch.tensor(cs_points, dtype=torch.float64)
    train_Y = torch.tensor(-cs_scores, dtype=torch.float64).unsqueeze(-1)
    y_mean = train_Y.mean(); y_std = train_Y.std().clamp_min(1e-8)
    train_Y_std = (train_Y - y_mean) / y_std

    best_score = float(cs_scores.min())
    trace = [best_score]
    all_scores = list(cs_scores)

    # Hvarfner default.yaml: ls.loc=1.4, ls.scale=sqrt(3); dim_scaling=(0.5, 0)
    # Applied: loc + 0.5*log(D), scale unchanged
    ls_loc = 1.4 + 0.5 * math.log(D)
    ls_prior = LogNormalPrior(loc=ls_loc, scale=math.sqrt(3.0))
    noise_prior = LogNormalPrior(loc=-4.0, scale=1.0)

    bounds = torch.tensor([[-1.0] * D, [1.0] * D], dtype=torch.float64)

    for i in range(n_iter):
        try:
            # 'default' model: raw RBFKernel (no ScaleKernel; outputscale fixed at 1)
            covar_module = RBFKernel(
                ard_num_dims=D,
                lengthscale_prior=ls_prior,
                lengthscale_constraint=GreaterThan(1e-4),
            )
            likelihood = GaussianLikelihood(
                noise_prior=noise_prior,
                noise_constraint=GreaterThan(1e-4),
            )
            gp = SingleTaskGP(train_X, train_Y_std,
                              covar_module=covar_module,
                              likelihood=likelihood)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)

            acqf = qLogNoisyExpectedImprovement(
                model=gp, X_baseline=train_X, prune_baseline=True,
            )
            candidate, _ = optimize_acqf(
                acqf, bounds=bounds, q=1,
                num_restarts=4, raw_samples=512,
                options={"sample_around_best": True,
                         "sample_around_best_sigma": 0.1,
                         "maxiter": 300, "batch_limit": 64},
            )
            x = candidate.detach().numpy().flatten()
        except (np.linalg.LinAlgError, RuntimeError, ValueError) as _gp_err:
            logger.warning(f"Hvarfner step failed: {type(_gp_err).__name__}: {_gp_err}")
            x = rng.uniform(-1, 1, size=D)

        score = bench.evaluate(x)
        all_scores.append(score)
        if score < best_score:
            best_score = score
        trace.append(float(best_score))

        train_X = torch.cat([train_X, torch.tensor(x, dtype=torch.float64).unsqueeze(0)])
        train_Y = torch.cat([train_Y, torch.tensor([[-score]], dtype=torch.float64)])
        y_mean = train_Y.mean(); y_std = train_Y.std().clamp_min(1e-8)
        train_Y_std = (train_Y - y_mean) / y_std

    return {"best_score": float(best_score), "best_trace": trace, "all_scores": all_scores}


def run_papenmeier_msr(bench, D: int, n_cold: int, n_iter: int, seed: int) -> dict:
    """Papenmeier et al. 2025 MSR — MLE Scaled with RAASP. Official recipe per paper §4.

    Source: arxiv.org/abs/2502.09198 (Table 1, §3-4)
    - Matern-5/2 ARD kernel (default ScaleKernel + outputscale)
    - Initial lengthscale = sqrt(D)/10 BEFORE MLE fitting (no prior)
    - LogEI acquisition (Ament et al. 2024)
    - RAASP via BoTorch sample_around_best=True (the load-bearing detail)
    - Bounds [-1,1]^D for LassoBench (paper uses [0,1]^D for synthetic tasks)
    """
    import math
    import torch
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.optim import optimize_acqf
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.mlls import ExactMarginalLogLikelihood

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    cs_points = rng.uniform(-1, 1, size=(n_cold, D))
    cs_scores = np.array([bench.evaluate(x) for x in cs_points])

    train_X = torch.tensor(cs_points, dtype=torch.float64)
    train_Y = torch.tensor(-cs_scores, dtype=torch.float64).unsqueeze(-1)
    y_mean = train_Y.mean(); y_std = train_Y.std().clamp_min(1e-8)
    train_Y_std = (train_Y - y_mean) / y_std

    best_score = float(cs_scores.min())
    trace = [best_score]; all_scores = list(cs_scores)

    # MSR initial lengthscale: 0.1 * sqrt(D) (paper §4 "Experimental Setup")
    ls_init = math.sqrt(D) / 10.0
    bounds = torch.tensor([[-1.0] * D, [1.0] * D], dtype=torch.float64)

    for i in range(n_iter):
        try:
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=D)
            base_kernel.lengthscale = torch.full((1, D), ls_init, dtype=torch.float64)
            gp = SingleTaskGP(train_X, train_Y_std,
                              covar_module=ScaleKernel(base_kernel))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            acqf = qLogExpectedImprovement(model=gp, best_f=train_Y_std.max())
            candidate, _ = optimize_acqf(
                acqf, bounds=bounds, q=1,
                num_restarts=4, raw_samples=512,
                options={"sample_around_best": True,
                         "sample_around_best_sigma": 0.1,
                         "maxiter": 300, "batch_limit": 64},
            )
            x = candidate.detach().numpy().flatten()
        except (np.linalg.LinAlgError, RuntimeError, ValueError) as _gp_err:
            logger.warning(f"Papenmeier step failed: {type(_gp_err).__name__}: {_gp_err}")
            x = rng.uniform(-1, 1, size=D)

        score = bench.evaluate(x)
        all_scores.append(score)
        if score < best_score: best_score = score
        trace.append(float(best_score))
        train_X = torch.cat([train_X, torch.tensor(x, dtype=torch.float64).unsqueeze(0)])
        train_Y = torch.cat([train_Y, torch.tensor([[-score]], dtype=torch.float64)])
        y_mean = train_Y.mean(); y_std = train_Y.std().clamp_min(1e-8)
        train_Y_std = (train_Y - y_mean) / y_std

    return {"best_score": float(best_score), "best_trace": trace, "all_scores": all_scores}


def run_xu_standard_gp(bench, D: int, n_cold: int, n_iter: int, seed: int) -> dict:
    """Xu et al. 2025 "Standard GP is all you need" — official recipe.

    Source: https://github.com/XZT008/Standard-GP-is-all-you-need-for-HDBO
    baselines/GP.py ExactGPModel + baselines/BO_loop.py BO_loop_GP

    Spec:
    - Matern-5/2 ARD with ScaleKernel (outputscale fitted, no prior)
    - Initial lengthscale = c * sqrt(D), c=1 (per repo line 92)
    - No lengthscale prior, no constraint
    - Default GaussianLikelihood (no noise prior)
    - LogEI acquisition (paper §; analytic LogEI in their BO_loop)
    - num_restarts=10, raw_samples=1000 (per BO_loop.py line 49-50)
    - Bounds [-1,1]^D (LassoBench native)
    """
    import math
    import torch
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.optim import optimize_acqf
    from gpytorch.kernels import MaternKernel, ScaleKernel
    from gpytorch.mlls import ExactMarginalLogLikelihood

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    cs_points = rng.uniform(-1, 1, size=(n_cold, D))
    cs_scores = np.array([bench.evaluate(x) for x in cs_points])
    train_X = torch.tensor(cs_points, dtype=torch.float64)
    train_Y = torch.tensor(-cs_scores, dtype=torch.float64).unsqueeze(-1)
    y_mean = train_Y.mean(); y_std = train_Y.std().clamp_min(1e-8)
    train_Y_std = (train_Y - y_mean) / y_std

    best_score = float(cs_scores.min())
    trace = [best_score]; all_scores = list(cs_scores)
    ls_init = 1.0 * math.sqrt(D)
    bounds = torch.tensor([[-1.0] * D, [1.0] * D], dtype=torch.float64)

    for i in range(n_iter):
        try:
            base_kernel = MaternKernel(nu=2.5, ard_num_dims=D)
            base_kernel.lengthscale = torch.full((1, D), ls_init, dtype=torch.float64)
            gp = SingleTaskGP(train_X, train_Y_std,
                              covar_module=ScaleKernel(base_kernel))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            acqf = qLogExpectedImprovement(model=gp, best_f=train_Y_std.max())
            candidate, _ = optimize_acqf(
                acqf, bounds=bounds, q=1,
                num_restarts=10, raw_samples=1000,
            )
            x = candidate.detach().numpy().flatten()
        except (np.linalg.LinAlgError, RuntimeError, ValueError) as _gp_err:
            logger.warning(f"Xu step failed: {type(_gp_err).__name__}: {_gp_err}")
            x = rng.uniform(-1, 1, size=D)

        score = bench.evaluate(x)
        all_scores.append(score)
        if score < best_score: best_score = score
        trace.append(float(best_score))
        train_X = torch.cat([train_X, torch.tensor(x, dtype=torch.float64).unsqueeze(0)])
        train_Y = torch.cat([train_Y, torch.tensor([[-score]], dtype=torch.float64)])
        y_mean = train_Y.mean(); y_std = train_Y.std().clamp_min(1e-8)
        train_Y_std = (train_Y - y_mean) / y_std

    return {"best_score": float(best_score), "best_trace": trace, "all_scores": all_scores}


def run_linear_bo(bench, D: int, n_cold: int, n_iter: int, seed: int) -> dict:
    """Doumont et al. 2025 "Linear BO" — official recipe on LassoBench's native box.

    Source: arxiv.org/abs/2512.00170, github.com/colmont/linear-bo
    - SphericalLinearKernel: inverse stereographic projection of ARD-scaled inputs
      onto S^D, then a linear (dot-product) kernel with softmax weights on the
      constant and linear terms (their Eq. 3, Section 3.1 "Spherical Mapping")
    - ConstantMean, LogNormal(-4, 1) noise prior, GreaterThan(1e-4) noise constraint
    - Analytic LogExpectedImprovement (their src/optimize.py)
    - optimize_acqf: num_restarts=4, raw_samples=512, sample_around_best=True
    - Bounds [-1,1]^D: LassoBench's native domain, so this is the *published*
      recipe on its own support (no adaptation). Doumont et al. evaluate
      Lasso-DNA (180D) themselves.

    The GP is built by ``_make_linear_bo_gp`` from ``src.spherevo.gp_competitors``,
    the same constructor as the molecular Linear BO rows, so the two benchmarks
    share one definition of the recipe.
    """
    import torch
    from botorch.acquisition.analytic import LogExpectedImprovement
    from botorch.fit import fit_gpytorch_mll
    from botorch.optim import optimize_acqf
    from gpytorch.mlls import ExactMarginalLogLikelihood

    from src.spherevo.gp_competitors import _make_linear_bo_gp

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    cs_points = rng.uniform(-1, 1, size=(n_cold, D))
    cs_scores = np.array([bench.evaluate(x) for x in cs_points])

    train_X = torch.tensor(cs_points, dtype=torch.float64)
    train_Y = torch.tensor(-cs_scores, dtype=torch.float64).unsqueeze(-1)
    y_mean = train_Y.mean(); y_std = train_Y.std().clamp_min(1e-8)
    train_Y_std = (train_Y - y_mean) / y_std

    best_score = float(cs_scores.min())
    trace = [best_score]
    all_scores = list(cs_scores)

    bounds = torch.tensor([[-1.0] * D, [1.0] * D], dtype=torch.float64)
    n_fallback = 0

    for i in range(n_iter):
        try:
            # The kernel asserts its inputs lie within `bounds`; optimize_acqf can
            # return a coordinate a hair outside, so clamp before fitting.
            gp = _make_linear_bo_gp(
                train_X.clamp(-1.0, 1.0), train_Y_std,
                bounds=(-1.0, 1.0), device=train_X.device,
            )
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            gp.eval()
            acqf = LogExpectedImprovement(model=gp, best_f=float(train_Y_std.max()))
            candidate, _ = optimize_acqf(
                acqf, bounds=bounds, q=1,
                num_restarts=4, raw_samples=512,
                options={"sample_around_best": True,
                         "sample_around_best_sigma": 0.1,
                         "maxiter": 300, "batch_limit": 64},
            )
            if not torch.isfinite(candidate).all():
                raise ValueError("optimize_acqf returned non-finite candidate")
            x = candidate.detach().numpy().flatten().clip(-1.0, 1.0)
        except (np.linalg.LinAlgError, RuntimeError, ValueError, AssertionError) as _gp_err:
            n_fallback += 1
            logger.warning(f"LinearBO step failed: {type(_gp_err).__name__}: {_gp_err}")
            x = rng.uniform(-1, 1, size=D)

        score = bench.evaluate(x)
        all_scores.append(score)
        if score < best_score:
            best_score = score
        trace.append(float(best_score))
        train_X = torch.cat([train_X, torch.tensor(x, dtype=torch.float64).unsqueeze(0)])
        train_Y = torch.cat([train_Y, torch.tensor([[-score]], dtype=torch.float64)])
        y_mean = train_Y.mean(); y_std = train_Y.std().clamp_min(1e-8)
        train_Y_std = (train_Y - y_mean) / y_std

    return {"best_score": float(best_score), "best_trace": trace,
            "all_scores": all_scores, "surrogate_fallback_count": n_fallback}


def run_gp_bo(bench, D: int, d: int, n_cold: int, n_iter: int, seed: int) -> dict:
    """GP-guided BO in random subspace (to test if GP helps)."""
    import torch
    from botorch.models import SingleTaskGP
    from botorch.fit import fit_gpytorch_mll
    from gpytorch.mlls import ExactMarginalLogLikelihood
    from botorch.acquisition import qExpectedImprovement
    from botorch.optim import optimize_acqf

    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)

    # Cold start FIRST (before QR) so all methods get identical cold-start
    cs_points = rng.uniform(-1, 1, size=(n_cold, D))
    cs_scores = np.array([bench.evaluate(x) for x in cs_points])

    # QR projection (after cold-start to preserve RNG fairness)
    A, _ = np.linalg.qr(rng.randn(D, d))
    cs_proj = cs_points @ A

    best_idx = int(np.argmin(cs_scores))
    best_score = cs_scores[best_idx]

    # Negate scores for maximization (GP-BO maximizes)
    train_X = torch.tensor(cs_proj, dtype=torch.float64)
    train_Y = torch.tensor(-cs_scores, dtype=torch.float64).unsqueeze(-1)

    trace = [float(best_score)]
    all_scores = list(cs_scores)

    for i in range(n_iter):
        try:
            model = SingleTaskGP(train_X, train_Y)
            mll = ExactMarginalLogLikelihood(model.likelihood, model)
            fit_gpytorch_mll(mll)

            acqf = qExpectedImprovement(model, best_f=train_Y.max())
            bounds = torch.tensor([[-3.0] * d, [3.0] * d], dtype=torch.float64)
            candidate, _ = optimize_acqf(acqf, bounds=bounds, q=1,
                                         num_restarts=5, raw_samples=256)
            v = candidate.detach().numpy().flatten()
        except (np.linalg.LinAlgError, RuntimeError, ValueError) as _gp_err:
            logger.warning(f"Subspace GP step failed: {type(_gp_err).__name__}: {_gp_err}")
            v = rng.randn(d)

        u = (v.reshape(1, -1) @ A.T).flatten()
        x = np.clip(u, -1, 1)
        score = bench.evaluate(x)
        all_scores.append(score)

        if score < best_score:
            best_score = score
        trace.append(float(best_score))

        train_X = torch.cat([train_X, torch.tensor(v, dtype=torch.float64).unsqueeze(0)])
        train_Y = torch.cat([train_Y, torch.tensor([[-score]], dtype=torch.float64)])

    return {
        "best_score": float(best_score),
        "best_trace": trace,
        "all_scores": all_scores,
    }


# ---------------------------------------------------------------------------
# Published-recipe methods (camera-ready): the recipes of src/spherevo/published_recipes.py,
# unchanged, on LassoBench's own box [-1, 1]^D mapped affinely to the unit cube the recipes are
# defined on (x = 2u - 1). Cold start, budget and result format are this harness's (5 points
# U[-1, 1]^D from RandomState(seed), 25 steps). A numerical failure in a step gives a uniform
# point for that step and is recorded; it never ends the run (cf. ANA_runs.md 3.6).
# ---------------------------------------------------------------------------

PUBLISHED_LASSO_METHODS = {
    "hvarfner_published": "hvarfner",
    "papenmeier_published": "msr",
    "papenmeier_mle_scaled_published": "mle_scaled",
    "xu_published": "xu",
    "linear_bo_published": "doumont",
    # external baselines as released (src/spherevo/published_external.py)
    "turbo_published": "turbo1",        # TuRBO-1 on the FIXED box [-1, 1]^D (turbo_1.py)
    "baxus_official": "baxus",          # BAxUS, the authors' BoTorch tutorial, budget = this run's
    "cmaes_full_published": "cmaes",    # pycma, LassoBench settings: popsize 20, sigma0 0.1, x0 0;
                                        # f(x0) is evaluation 1, then n_cold + n_iter - 1 pycma samples
    # round 2, R2-6: MSR / MLE (scaled) with the Gaussian noise learned (--no-noiseless path)
    "papenmeier_published_learned": "msr_learned",
    "papenmeier_mle_scaled_published_learned": "mle_scaled_learned",
}
EXTERNAL_LASSO_METHODS = ("turbo1", "baxus", "cmaes")


def _run_external_lasso(recipe: str, bench, D: int, n_cold: int, n_iter: int, seed: int,
                        cs_points: np.ndarray, cs_scores: np.ndarray) -> dict:
    """TuRBO-1 / BAxUS / CMA-ES as released; minimisation scores, every evaluation counted."""
    from src.spherevo.published_external import (
        BaxusPublished,
        Turbo1Published,
        external_settings,
        run_cmaes_published,
    )

    record = {"recipe": recipe, "domain": "[-1, 1]^D (LassoBench box)",
              "settings": external_settings(recipe, dim=D), "steps": []}
    reasons: list[str] = []

    def loss_at(x):
        return float(bench.evaluate(np.clip(np.asarray(x, dtype=float).reshape(-1), -1.0, 1.0)))

    if recipe == "turbo1":
        # the 5 cold starts are TuRBO's initial design (TuRBO accepts any initial data)
        scores = list(cs_scores)
        turbo = Turbo1Published(dim=D, n_init=n_cold, batch_size=1)
        turbo.seed_data((cs_points + 1.0) / 2.0, cs_scores)
        for i in range(n_iter):
            u, info = turbo.ask()
            loss = loss_at(2.0 * u - 1.0)
            turbo.tell(u, loss)
            scores.append(loss)
            step = info.to_dict()
            step.update(step=i, failcount=int(turbo.failcount), restarts=int(turbo.n_restarts))
            record["steps"].append(step)
            if info.fallback:
                reasons.append(info.fallback)
    elif recipe == "baxus":
        # BAxUS cannot use points outside its embedding: it evaluates its own n_cold-point
        # Sobol design in the initial target space (as published), so the shared cold start is
        # not evaluated; budget = n_cold + n_iter evaluations, eval_budget = n_iter.
        record["cold_start"] = "own Sobol design in the target space (shared cold start unused)"
        scores = []
        bx = BaxusPublished(dim=D, n_init=n_cold, eval_budget=n_iter, seed=seed)
        for i in range(n_cold + n_iter):
            x, info = bx.ask()
            loss = loss_at(x)
            bx.tell(-loss)
            scores.append(loss)
            info.update(step=i)
            record["steps"].append(info)
            if info["fallback"]:
                reasons.append(info["fallback"])
        record["splits"] = bx.n_splits_done
    else:  # cmaes
        # Published LassoBench protocol (cma_example.py: no initial design, CMA-ES from x0 = 0,
        # sigma0 0.1; the shared cold start is not evaluated, round-1 C6). The published curves start
        # with f(x0) at x = 1, so x0 = 0 is evaluation 1 of the n_cold + n_iter, followed by
        # n_cold + n_iter - 1 pycma samples (round 2, R2-12): every arm keeps exactly 30 evaluations.
        record["cold_start"] = ("none (shared cold start unused): evaluation 1 is f(x0 = 0), the "
                                "published curve's x = 1 point, then n_cold + n_iter - 1 pycma "
                                "evaluations from x0 = 0 (popsize 20, sigma0 0.1); round 2, R2-12")
        record["x0_evaluation"] = "evaluation 1 of the budget (production); the harness keeps x0 outside"
        scores = [loss_at(np.zeros(D))]
        record["steps"].append({"fallback": None, "events": ["x0 = 0 (start point of pycma)"],
                                "generation": None, "step": 0})

        def on_eval(x, v, info):
            scores.append(v)
            record["steps"].append({**info, "step": len(scores) - 1})

        record["cma"] = run_cmaes_published(loss_at, D, n_cold + n_iter - 1, seed, on_eval=on_eval)
    trace = np.minimum.accumulate(np.asarray(scores, dtype=float)).tolist()
    if recipe == "cmaes":
        record["full_best_trace"] = trace   # running best from evaluation 1 = f(x0)
    first = n_cold - 1
    return {"best_score": float(min(scores)), "best_trace": [trace[first], *trace[first + 1:]],
            "all_scores": scores, "status": "completed",
            "surrogate_fallback_count": len(reasons), "surrogate_fallback_reasons": reasons,
            "published_recipe": record}


def _published_lasso_provenance() -> dict:
    """Commit, dirty flag and package versions of a published-method result (round-1 fix C9)."""
    import importlib.metadata as md

    from src.provenance import collect_provenance

    prov = collect_provenance()["provenance"]
    for pkg in ("torch", "botorch", "gpytorch", "linear_operator", "LassoBench", "cma",
                "scikit-learn", "numpy"):
        try:
            prov.setdefault("package_versions", {})[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    return prov


def _published_lasso_provenance_or_error() -> dict:
    """For the crash record only: provenance, or why it could not be collected (a failure here must
    not replace the crash being recorded, e.g. when ``src`` is not importable without PYTHONPATH)."""
    try:
        return _published_lasso_provenance()
    except Exception as e:  # noqa: BLE001
        return {"error": f"provenance unavailable: {type(e).__name__}: {e}"}


def run_published_lasso(method: str, bench, D: int, n_cold: int, n_iter: int, seed: int) -> dict:
    """Run one published recipe; minimisation scores in, the recipe maximises ``-loss``."""
    import torch

    from src.spherevo.published_recipes import (
        DTYPE,
        recipe_settings,
        run_loop,
        xu_recipe_defined,
    )

    recipe = PUBLISHED_LASSO_METHODS[method]
    rng = np.random.RandomState(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)       # TuRBO-1 and pycma draw from numpy's global RNG
    cs_points = rng.uniform(-1, 1, size=(n_cold, D))
    if recipe in ("baxus", "cmaes"):      # own design / no design: the cold start is not evaluated
        return {**_run_external_lasso(recipe, bench, D, n_cold, n_iter, seed, cs_points, None),
                "provenance": _published_lasso_provenance()}
    cs_scores = np.array([bench.evaluate(x) for x in cs_points])
    if recipe in EXTERNAL_LASSO_METHODS:
        return {**_run_external_lasso(recipe, bench, D, n_cold, n_iter, seed, cs_points, cs_scores),
                "provenance": _published_lasso_provenance()}

    record = {"recipe": recipe, "domain": "[-1, 1]^D -> [0, 1]^D (x = 2u - 1)",
              "settings": recipe_settings(recipe, D=D)}
    if recipe == "xu" and not xu_recipe_defined(D):
        return {"best_score": float("nan"), "best_trace": [], "all_scores": list(cs_scores),
                "status": "not_applicable",
                "error": (f"Xu SBO-Matern (RI) is undefined at D={D}: sqrt(D) lies outside the "
                          f"official lengthscale constraint Interval(1e-10, 30) (GP.py:196-214)"),
                "surrogate_fallback_count": 0, "surrogate_fallback_reasons": [],
                "published_recipe": record, "provenance": _published_lasso_provenance()}

    scores = list(cs_scores)

    def evaluate(x_unit):
        x = (2.0 * x_unit.detach().cpu().numpy().reshape(-1) - 1.0).clip(-1.0, 1.0)
        loss = float(bench.evaluate(x))
        scores.append(loss)
        return -loss

    X0 = torch.tensor((cs_points + 1.0) / 2.0, dtype=DTYPE)
    Y0 = torch.tensor(-cs_scores, dtype=DTYPE).reshape(-1, 1)
    out = run_loop(recipe, evaluate, X0, Y0, n_iter)
    trace = np.minimum.accumulate(np.asarray(scores, dtype=float)).tolist()
    record["steps"] = out["steps"]
    return {"best_score": float(min(scores)), "best_trace": [trace[n_cold - 1], *trace[n_cold:]],
            "all_scores": scores, "status": "completed",
            "surrogate_fallback_count": out["fallback_count"],
            "surrogate_fallback_reasons": out["fallback_reasons"],
            "published_recipe": record, "provenance": _published_lasso_provenance()}


def main():
    parser = argparse.ArgumentParser(description="LassoBench CMA-ES Subspace Benchmark")
    parser.add_argument("--tasks", type=str, default="synt_simple,synt_medium,synt_high,synt_hard",
                        help="Comma-separated tasks. Synthetic: synt_simple,synt_medium,synt_high,synt_hard. "
                             "Real: dna,leukemia")
    parser.add_argument("--real-only", action="store_true",
                        help="Run only real-world tasks (dna, leukemia)")
    parser.add_argument("--seeds", type=str, default="42-51")
    parser.add_argument("--methods", type=str, default="cmaes_sub,cmaes_full,random,gp_bo,turbo,baxus")
    parser.add_argument("--n-cold-start", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=25)
    parser.add_argument("--subspace-dim", type=int, default=16)
    parser.add_argument("--pop-size", type=int, default=5)
    parser.add_argument("--results-dir", type=str, default="results/lassobench")
    args = parser.parse_args()

    if args.real_only:
        tasks = list(REAL_TASKS.keys())
    else:
        tasks = args.tasks.split(",")
    methods = args.methods.split(",")

    # Parse seeds
    if "-" in args.seeds:
        start, end = map(int, args.seeds.split("-"))
        seeds = list(range(start, end + 1))
    else:
        seeds = [int(s) for s in args.seeds.split(",")]

    os.makedirs(args.results_dir, exist_ok=True)

    # ---- LP-PFN via solver reuse (SSOT: delegates to src/lppfn/solver.py) ----
    def run_lppfn_subspace(bench, D, d, n_cold, n_iter, pop_size, seed):
        """CMA-ES in QR subspace with LP-PFN surrogate via LPPFNSubspaceSolver.

        Wraps bench.evaluate (minimization) as a negated black_box (maximization)
        and delegates to the canonical solver implementation.
        """
        from pathlib import Path
        from src.lppfn.solver import LPPFNSubspaceSolver

        rng = np.random.RandomState(seed)
        cs_points = rng.uniform(-1, 1, size=(n_cold, D))
        cs_scores = np.array([bench.evaluate(x) for x in cs_points])

        # Negate: solver maximizes, LassoBench minimizes
        def negated_bb(z):
            scores = []
            for row in z:
                x = np.clip(row, -1, 1)
                scores.append(-bench.evaluate(x))
            return np.array(scores).reshape(-1, 1)

        y0_neg = -cs_scores.reshape(-1, 1)

        solver = LPPFNSubspaceSolver(
            black_box=negated_bb, x0=cs_points, y0=y0_neg,
            subspace_dim=d, population_size=pop_size,
            pfn_weights_path=Path("weights/lppfn/best.pt"),
        )
        solver.solve(n_iter)

        # Convert back to minimization scores
        all_scores_neg = solver.all_scores
        all_scores = [-s for s in all_scores_neg]
        best_score = -solver.best_score

        # Build trace (running min of all_scores)
        trace = []
        running_min = float("inf")
        for s in all_scores:
            running_min = min(running_min, s)
            trace.append(running_min)

        return {
            "best_score": best_score,
            "best_trace": trace,
            "all_scores": all_scores,
        }

    import LassoBench

    total = len(tasks) * len(seeds) * len(methods)
    done = 0

    for task_name in tasks:
        D = REAL_TASKS.get(task_name) or TASK_DIMS.get(task_name)
        if D is None:
            print(f"WARNING: Unknown task '{task_name}', skipping")
            continue
        d = min(args.subspace_dim, D)

        print(f"\n{'='*60}")
        print(f"Task: {task_name} (D={D}, d={d})")
        print(f"{'='*60}")

        # Compute oracle/reference once per task (not per method)
        if task_name in REAL_TASKS:
            ref_bench = LassoBench.RealBenchmark(pick_data=task_name)
        else:
            ref_bench = LassoBench.SyntheticBenchmark(pick_bench=task_name)
        if hasattr(ref_bench, 'loss_oracle'):
            oracle_val = float(ref_bench.loss_oracle)
        else:
            try:
                oracle_val = float(ref_bench.run_LASSOCV()["val_loss"])
            except (RuntimeError, ValueError, KeyError) as _oracle_err:
                logger.error(
                    f"LASSOCV oracle failed for reference bench: "
                    f"{type(_oracle_err).__name__}: {_oracle_err}"
                )
                oracle_val = None

        for seed in seeds:
            if task_name in REAL_TASKS:
                bench = LassoBench.RealBenchmark(pick_data=task_name)
            else:
                bench = LassoBench.SyntheticBenchmark(pick_bench=task_name)

            for method in methods:
                result_file = os.path.join(
                    args.results_dir,
                    f"{method}_{task_name}_s{seed}.json"
                )
                if os.path.exists(result_file):
                    done += 1
                    print(f"  [{done}/{total}] {method} s{seed}: SKIP (exists)")
                    continue

                t0 = time.time()

                # Dispatch: cmaes_sub_* variants use projection methods
                proj_prefix = "cmaes_sub_"
                if method == "cmaes_sub":
                    result = run_cmaes_subspace(
                        bench, D, d, args.n_cold_start, args.n_iter,
                        args.pop_size, seed, projection="random_qr"
                    )
                elif method == "lq_cmaes_sub":
                    result = run_cmaes_subspace(
                        bench, D, d, args.n_cold_start, args.n_iter,
                        args.pop_size, seed, projection="random_qr",
                        lq_K=3, lq_gate=0.3,
                    )
                elif method.startswith(proj_prefix) and method[len(proj_prefix):] in PROJECTION_METHODS:
                    proj = method[len(proj_prefix):]
                    result = run_cmaes_subspace(
                        bench, D, d, args.n_cold_start, args.n_iter,
                        args.pop_size, seed, projection=proj
                    )
                elif method == "cmaes_full":
                    try:
                        result = run_cmaes_full(
                            bench, D, args.n_cold_start, args.n_iter,
                            args.pop_size, seed
                        )
                    except Exception as e:
                        result = {
                            "best_score": float("inf"),
                            "best_trace": [],
                            "all_scores": [],
                            "error": str(e),
                        }
                elif method == "random":
                    result = run_random_search(
                        bench, D, args.n_cold_start + args.n_iter, seed
                    )
                elif method == "gp_bo":
                    result = run_gp_bo(
                        bench, D, d, args.n_cold_start, args.n_iter, seed
                    )
                elif method == "hvarfner_gp":
                    try:
                        result = run_hvarfner_gp(bench, D, args.n_cold_start, args.n_iter, seed)
                    except Exception as e:
                        result = {"best_score": float("inf"), "best_trace": [], "all_scores": [], "error": str(e)}
                elif method == "papenmeier_msr":
                    try:
                        result = run_papenmeier_msr(bench, D, args.n_cold_start, args.n_iter, seed)
                    except Exception as e:
                        result = {"best_score": float("inf"), "best_trace": [], "all_scores": [], "error": str(e)}
                elif method == "xu_standard_gp":
                    try:
                        result = run_xu_standard_gp(bench, D, args.n_cold_start, args.n_iter, seed)
                    except Exception as e:
                        result = {"best_score": float("inf"), "best_trace": [], "all_scores": [], "error": str(e)}
                elif method == "linear_bo":
                    try:
                        result = run_linear_bo(bench, D, args.n_cold_start, args.n_iter, seed)
                    except Exception as e:
                        result = {"best_score": float("inf"), "best_trace": [], "all_scores": [], "error": str(e)}
                elif method == "turbo":
                    try:
                        result = run_turbo(
                            bench, D, args.n_cold_start, args.n_iter, seed
                        )
                    except Exception as e:
                        result = {
                            "best_score": float("inf"),
                            "best_trace": [],
                            "all_scores": [],
                            "error": str(e),
                        }
                elif method == "baxus":
                    try:
                        result = run_baxus(
                            bench, D, d, args.n_cold_start, args.n_iter, seed
                        )
                    except Exception as e:
                        result = {
                            "best_score": float("inf"),
                            "best_trace": [],
                            "all_scores": [],
                            "error": str(e),
                        }
                elif method == "lppfn_sub":
                    result = run_lppfn_subspace(
                        bench, D, d, args.n_cold_start, args.n_iter,
                        args.pop_size, seed,
                    )
                elif method == "geowalk":
                    result = run_geowalk(
                        bench, D, args.n_cold_start, args.n_iter, seed,
                    )
                elif method in PUBLISHED_LASSO_METHODS:
                    # Numerical failures are handled per step inside the recipe; what reaches
                    # this guard is a programming error, recorded with its traceback.
                    try:
                        result = run_published_lasso(
                            method, bench, D, args.n_cold_start, args.n_iter, seed,
                        )
                    except Exception as e:
                        import traceback
                        result = {"best_score": float("inf"), "best_trace": [], "all_scores": [],
                                  "status": "crashed", "error": f"{type(e).__name__}: {e}",
                                  "traceback": traceback.format_exc(),
                                  "provenance": _published_lasso_provenance_or_error()}
                else:
                    raise ValueError(f"Unknown method: {method}")

                dt = time.time() - t0
                done += 1

                result.update({
                    "method": method,
                    "task": task_name,
                    "D": D,
                    "d": d,
                    "seed": seed,
                    "n_cold_start": args.n_cold_start,
                    "n_iter": args.n_iter,
                    "n_total": args.n_cold_start + args.n_iter,
                    "pop_size": args.pop_size if "cmaes" in method else None,
                    "projection": method[len("cmaes_sub_"):] if method.startswith("cmaes_sub_") else (
                        "random_qr" if method in ("cmaes_sub", "lq_cmaes_sub") else None),
                    "oracle": oracle_val,
                    "elapsed_s": dt,
                    "timestamp": datetime.now().isoformat(),
                })
                if method == "cmaes_full_published":
                    # round 2, R2-14: this arm runs pycma with popsize 20 (LassoBench settings); the CLI
                    # --pop-size belongs to the submission-era cmaes_full / cmaes_sub and is not used
                    result["pop_size_cli_argument_unused"] = args.pop_size
                    result["pop_size"] = ((result.get("published_recipe") or {}).get("cma") or {}).get(
                        "popsize")

                with open(result_file, "w") as f:
                    json.dump(result, f, indent=2)

                oracle_str = f"{oracle_val:.4f}" if oracle_val is not None else "N/A"
                print(f"  [{done}/{total}] {method} s{seed}: {result['best_score']:.4f} "
                      f"(oracle={oracle_str}) {dt:.1f}s")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    for task_name in tasks:
        D_sum = REAL_TASKS.get(task_name) or TASK_DIMS.get(task_name, "?")
        print(f"\n{task_name} (D={D_sum}):")
        for method in methods:
            scores = []
            for seed in seeds:
                f = os.path.join(args.results_dir, f"{method}_{task_name}_s{seed}.json")
                if os.path.exists(f):
                    with open(f) as fh:
                        d_loaded = json.load(fh)
                    scores.append(d_loaded["best_score"])
            if scores:
                finite = [s for s in scores if np.isfinite(s)]
                n_bad = len(scores) - len(finite)
                if n_bad > 0:
                    logger.warning(
                        f"{method}/{task_name}: {n_bad} non-finite scores excluded "
                        f"(likely from solver crash fallback producing best_score=inf)"
                    )
                if finite:
                    print(
                        f"  {method:15s}: {np.mean(finite):.4f} +/- {np.std(finite):.4f} "
                        f"(n={len(finite)}/{len(scores)})"
                    )


if __name__ == "__main__":
    main()
