#!/usr/bin/env python3
"""Benchmark GP-free spherical subspace BO against poli-baselines methods.

Uses poli-baselines SOLVER interface but our own GuacaMol oracle (avoids
poli's process isolation which hangs). The solvers only need a callable
black_box(z) -> score, which we provide by wrapping VAE + oracle.

Supports two modes:
  1. HDBO mode (--mode hdbo): Use HDBO benchmark's MLP VAE (128D) for fair
     comparison with their published results.
  2. Our VAE mode (--mode ours): Use our SELFIES Transformer VAE (256D) for comparison
     on our home turf.

Usage:
    # Single run (smoke test with our VAE)
    PYTHONPATH=/home/prusek/NeurIPS CUDA_VISIBLE_DEVICES=0 python scripts/run_benchmark.py \
        --mode ours --solver gp_free_spherical --task adip --seed 42 --max-iter 10 --n-cold-start 5

    # HDBO mode
    PYTHONPATH=/home/prusek/NeurIPS CUDA_VISIBLE_DEVICES=0 python scripts/run_benchmark.py \
        --mode hdbo --solver gp_free_spherical --task amlodipine_mpo --seed 42 --max-iter 10

    # poli-baselines CMA-ES on our VAE
    PYTHONPATH=/home/prusek/NeurIPS CUDA_VISIBLE_DEVICES=0 python scripts/run_benchmark.py \
        --mode ours --solver cma_es --task adip --seed 42 --max-iter 25 --n-cold-start 5

    # All 12 tasks with our method
    PYTHONPATH=/home/prusek/NeurIPS CUDA_VISIBLE_DEVICES=0 python scripts/run_benchmark.py \
        --mode ours --solver gp_free_spherical --task ours --seed 42 --max-iter 25 --n-cold-start 5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

from src.provenance import collect_provenance
from src.analysis.config import (
    CODEC_RESULT_ROOTS,
    EMPIRICAL_SHELL_MEAN_NORM,
    EMPIRICAL_SHELL_PCT_5_95,
    GUACAMOL_TO_POLI,
    POLI_TO_GUACAMOL,
    RESULT_DIRS as _RESULT_DIRS,
)
RESULTS_DIR = PROJECT_ROOT / _RESULT_DIRS["spherevo"]
GEOWALK_DIR = PROJECT_ROOT / _RESULT_DIRS["geowalk"]
GEOWALK_ABL_DIR = PROJECT_ROOT / "results/guacamol/geowalk_ablations"


def _results_dir_for(solver_name: str) -> Path:
    """Pick the output directory based on solver name.

    Most solvers go into the SphereVO directory; geowalk gets its own
    directory and GeoWalk ablation variants (geowalk_*) go to a separate
    directory so the main geowalk dir stays clean for Table~\\ref{tab:main}.
    Riemannian BO baselines (Jaquier et al.) each get their own directory
    so the paper's GP-on-sphere baseline rows stay separable.
    """
    if solver_name == "geowalk":
        return GEOWALK_DIR
    if solver_name.startswith("geowalk_"):
        return GEOWALK_ABL_DIR
    if solver_name in {"gabo", "hd_gabo_2", "hd_gabo_16", "matern_gabo"}:
        return PROJECT_ROOT / _RESULT_DIRS[solver_name]
    return RESULTS_DIR

# HDBO PMO tasks (full set of 25 from poli)
HDBO_PMO_TASKS = [
    "albuterol_similarity", "amlodipine_mpo", "celecoxib_rediscovery",
    "deco_hop", "drd2", "fexofenadine_mpo", "gsk3b", "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl", "jnk3", "median_1", "median_2",
    "mestranol_similarity", "osimetrinib_mpo", "perindopril_mpo",
    "ranolazine_mpo", "scaffold_hop", "sitagliptin_mpo",
    "thiothixene_rediscovery", "troglitazone_rediscovery",
    "valsartan_smarts", "zaleplon_mpo",
]

OUR_TASKS_POLI = list(GUACAMOL_TO_POLI.values())

# Our solvers (from src/spherevo/solver.py)
OUR_SOLVERS = [
    "cma_es_subspace",
    "lq_cma_es_subspace",
    "lppfn_subspace",
    "jacobian_subspace",
    "gp_free_spherical",
    "gp_guided_spherical",
    "noise_gp_spherical",
    "rose_subspace",
    "pace_subspace",
    "random_search",
    # Alternative-support baselines (reviewer Lhbx Q4: "beyond the hypersphere")
    "convex_hull_rs",
    "convex_hull_k1", "convex_hull_k2", "convex_hull_k4",
    "convex_hull_k8", "convex_hull_k16", "convex_hull_k64",
    "ellipsoid_rs",
    "pca_sphere_k16",
    "pca_sphere_k32",
    "pca_sphere_k64",
    "pca_sphere_k128",
    "cowboys",
    "rembo",
    "hesbo",
    "geowalk",
    "geowalk_srda",
    # GeoWalk ablation variants (Section 5)
    "geowalk_const035",    # constant angle 0.35 (no schedule)
    "geowalk_const060",    # constant angle 0.60
    "geowalk_const015",    # constant angle 0.15 (tight throughout)
    "geowalk_explore8",    # phase 1 = 8 random (17 refinement)
    "geowalk_explore24",   # phase 1 = 24 random (1 refinement)
    # Refinement angle adapted by a success rule (x1.2 / x0.85, start 0.30,
    # clamp [0.05, 0.50]).
    "geowalk_one_fifth",
    # Radius / support-geometry robustness (reviewers Lhbx Q1/Q2/Q3, 6qvY W2)
    "geowalk_r4", "geowalk_r6", "geowalk_r8", "geowalk_r10", "geowalk_r12", "geowalk_r16",
    "geowalk_shell10", "geowalk_shell20", "geowalk_rmedian", "geowalk_rpop", "geowalk_shellpct",
    "geowalk_flat",            # Euclidean ablation (no sphere)
    "geowalk_no_anchor",       # tangent walk from random sphere point (no memory)
    "geowalk_no_tangent",      # uniform sphere sampling at mean_norm (no tangent walk)
    # GP competitors (native implementations, Section 6 reviewer-response)
    "hvarfner_scaled_gp",     # Hvarfner 2024 LogNormal sqrt(D) prior
    "papenmeier_msr",         # Papenmeier 2025 MLE@sqrt(D) init
    "xu_standard_gp",         # Xu 2025 Matern-5/2 + c*sqrt(D) init
    # Box-supported variants reproducing the original papers' Sobol+L-BFGS
    # acquisition recipe (vs. our on-sphere candidate sampling).
    "hvarfner_scaled_gp_box",
    # Box-radius sweep variants (k = 1, 2, 4, 5; k=3 is the default _box solver)
    "hvarfner_scaled_gp_box1",
    "hvarfner_scaled_gp_box2",
    "hvarfner_scaled_gp_box4",
    "hvarfner_scaled_gp_box5",
    "hvarfner_scaled_gp_box_faithful",  # per-coord [0,1] rescaling per Hvarfner 2024
    "papenmeier_msr_box_faithful",      # same faithful domain, Papenmeier 0.1*sqrt(D)
    "xu_standard_gp_box_faithful",      # same faithful domain, Xu c*sqrt(D)
    "hvarfner_scaled_gp_box_projected",
    # Sphere-radius sweep variants (multiplier on empirical mean_norm)
    "hvarfner_scaled_gp_sphere05",
    "hvarfner_scaled_gp_sphere15",
    "hvarfner_scaled_gp_sphere20",
    "papenmeier_msr_box",
    "xu_standard_gp_box",
    # Faithful [0,1]^D normalized-box variants — reproduces the original
    # papers' protocol (input min-max normalize + bounds=[[0]*D,[1]*D]).
    "hvarfner_scaled_gp_norm_box",
    "hvarfner_scaled_gp_norm_box_argmax",     # min/max box + acquisition argmax
    "hvarfner_scaled_gp_norm_box_projected",  # min/max L-BFGS, proposal projected to the shell
    "papenmeier_msr_norm_box",
    "papenmeier_msr_raasp",   # MSR variant: 0.1sqrt(D) GP + trust region + TuRBO-style RAASP argmax (NOT the published call)
    # Published MSR acquisition maximiser (analytic LogEI + sample_around_best
    # sigma 1e-3, 5 restarts, 512 raw samples) on the two existing box domains.
    "papenmeier_msr_published",           # [p01, p99] box of *_box_faithful
    "papenmeier_msr_published_norm_box",  # [min, max] box of *_norm_box
    # TuRBO-1 (Eriksson et al. 2019) on the FIXED [p01, p99] box of
    # *_box_faithful. The submitted TuRBO row used the running bounding box of
    # the observed latents, i.e. a domain that moves after every evaluation;
    # see src/spherevo/turbo_box_faithful.py.
    "turbo_box_faithful",
    "xu_standard_gp_norm_box",
    # Doumont 2025 "Linear BO" (SphericalLinearKernel GP) — reviewer tPDn.
    # Fourth recipe for the controlled box-vs-sphere ablation.
    "linear_bo",         # faithful [0,1]^D box reproduction
    "linear_bo_sphere",  # our on-sphere adaptation
    "linear_bo_sphere_calib",  # same, with kernel bounds matched to the shell spread
    "tripp_tanimoto_gp",      # Tripp 2024 Tanimoto-GP on Morgan fingerprints
    "rf_bo",                  # Random Forest BO (non-Lipschitz ablation, Track D2)
    # Reviewer-response variants: same GP recipes, alternative candidate supports
    "hvarfner_scaled_gp_prior",        # candidates ~ N(0, I_D) (VAE prior)
    "hvarfner_scaled_gp_angular_box",  # uniform-in-box renormalized to mean_norm
    "papenmeier_msr_prior",
    "papenmeier_msr_angular_box",
    "xu_standard_gp_prior",
    "xu_standard_gp_angular_box",
    "random_search_prior",             # RS drawing z ~ N(0, I_D) directly
    "random_box_faithful",             # uniform per-coord in [p01,p99], no GP (ported from 4780475, A10 S2b)
    "hvarfner_scaled_gp_box_argmax",   # faithful [0,1]^D box, acquisition ARGMAX (missing 2x2 cell)
    # Control block of the camera-ready run rules (amendment of 2026-10-12):
    "hvarfner_scaled_gp_box_argmax_raw",  # C1: box_argmax with the GP fitted on raw latents
    "hvarfner_scaled_gp_boxdir_argmax",   # C2a: [p01,p99] draws renormalised to the shell radius, GP raw, argmax
    "random_boxdir",                      # C2b: the same draws, one per step, no surrogate
    "hvarfner_scaled_gp_box_faithful_projected",  # [p01,p99] L-BFGS, proposal projected to the shell
    "hvarfner_scaled_gp_sphere_continuous",  # acquisition maximised continuously ON the shell
    # Control C4 of the camera-ready run rules (amendment of 2026-10-14); the printed shell x continuous cell
    # of the factorial since the authors' decision of 2026-11-06:
    "hvarfner_scaled_gp_sphere_lbfgs",  # C4: the shell x continuous cell with the box arm's optimize_acqf (L-BFGS-B)
    "hvarfner_scaled_gp_filtered_box", # Hvarfner-on-box but reject ||x||>1.2*r̄
    "hvarfner_scaled_gp_normalized",   # Universal: sample from prior, project all inputs to r̄
    # Proximity selection — the null model for what the GP contributes at N=30.
    # Same on-shell candidate pool as the on-sphere GP, ranked by angle to the
    # incumbent instead of by an acquisition function. No surrogate is fitted.
    "proximity_q06",      # pick uniformly among the nearest 6% (the GP's measured quantile)
    "proximity_nearest",  # pick the single nearest of the 2048 candidates
    "proximity_far",      # falsification control: pick among the FARTHEST 6%
    "proximity_q06_e16",  # same, after 16 uniform exploration steps (GeoWalk-matched)
    "proximity_far_e16",
    # Riemannian BO (Jaquier et al.) — geometry-aware GP-on-sphere baselines.
    "gabo",         # CoRL19  Sphere Gaussian kernel + manifold acquisition
    "hd_gabo_2",    # NeurIPS20 nested-sphere learned latent S^2
    "hd_gabo_16",   # NeurIPS20 nested-sphere learned latent S^16
    "matern_gabo",  # CoRL21 Sphere Riemannian Matern 5/2 kernel
    # Geodesic-Slice GP — novel surrogate that fits 2D GP per great-circle
    # at the current best, bypassing concentration of measure in D=256.
    "geodesic_slice_gp",
    # Hvarfner-Ball — Hvarfner GP recipe + GeoWalk-style angular ball
    # candidate proposal (codec-independent novel combination).
    "hvarfner_scaled_gp_ball",
    # SpherePFN — codec-independent in-context surrogate (transformer
    # pretrained offline on synthetic Matern-on-sphere tasks).
    "sphere_pfn",
    # Published-recipe block (camera-ready): each published GP-BO recipe run unchanged
    # (src/spherevo/published_recipes.py) on a fixed box; see src/spherevo/published_arms.py.
    "hvarfner_published",               # Vanilla BO default model + Ax qLogNEI, [p01, p99] box
    "papenmeier_msr_reference",         # MSR reference GP/fit + published call, [p01, p99] box
    "papenmeier_mle_scaled_reference",  # MLE (scaled): the same without RAASP, [p01, p99] box
    "xu_published",                     # SBO-Matern (RI): GP_ARD_setls + UCB(1.5), [p01, p99] box
    "linear_bo_box8",                   # spherical linear kernel, Doumont's box [-8, 8]^256
    "turbo_published",                  # TuRBO-1 as released (turbo_1.py + gp.py), [p01, p99] box
    "hvarfner_published_allobs",        # hvarfner_published with every observation in the baseline
    "papenmeier_msr_reference_learned",         # papenmeier_msr_reference, noise learned
    "papenmeier_mle_scaled_reference_learned",  # papenmeier_mle_scaled_reference, noise learned
]

# External poli-baselines
POLI_BASELINE_SOLVERS = [
    "cma_es",
    "cma_es_sphere",
    "hill_climbing",
    "random_line_bo",
    "coordinate_line_bo",
    "turbo",
    "vanilla_bo_hvarfner",
    "saasbo",
]

SUPPORTED_SOLVERS = OUR_SOLVERS + POLI_BASELINE_SOLVERS + ["cmaes_full_rd", "cmaes_full_rd_pop5", "baxus_cr"]


# ---------------------------------------------------------------------------
# SAASBO Solver (direct BoTorch, no poli-baselines dependency)
# ---------------------------------------------------------------------------


class _SAASBOSolver:
    """SAASBO: Sparse Axis-Aligned Subspace BO (Eriksson & Jankowiak 2021).

    Uses BoTorch's SaasFullyBayesianSingleTaskGP with NUTS inference.
    Operates in normalized [0,1]^D space internally, maps to [-5,5]^D externally.
    Implements the same interface as StepByStepSolver for compatibility.
    """

    def __init__(self, black_box, x0, y0):
        self._black_box = black_box
        self._x0 = x0.copy()
        self._y0 = y0.copy()
        self._dim = x0.shape[1]
        self._history_x = [x0[i:i+1] for i in range(x0.shape[0])]
        self._history_y = [float(y0[i, 0]) for i in range(y0.shape[0])]
        self._best = float(np.max(y0))
        self.iteration = 0

    def _to_unit(self, x):
        """Map from [-5, 5]^D to [0, 1]^D."""
        return (x + 5.0) / 10.0

    def _from_unit(self, x):
        """Map from [0, 1]^D to [-5, 5]^D."""
        return x * 10.0 - 5.0

    def step(self):
        try:
            train_X = torch.tensor(
                np.vstack([h.flatten() for h in self._history_x]),
                dtype=torch.float64,
            )
            train_X_unit = torch.tensor(
                self._to_unit(train_X.numpy()), dtype=torch.float64,
            )
            train_Y = torch.tensor(
                [[y] for y in self._history_y], dtype=torch.float64,
            )

            from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
            from botorch.fit import fit_fully_bayesian_model_nuts
            from botorch.optim import optimize_acqf
            from botorch.models.transforms.outcome import Standardize

            # Use Standardize as recommended by BoTorch docs
            model = SaasFullyBayesianSingleTaskGP(
                train_X_unit, train_Y,
                outcome_transform=Standardize(m=1),
            )
            fit_fully_bayesian_model_nuts(
                model,
                warmup_steps=256,   # BoTorch minimum recommended
                num_samples=128,    # BoTorch minimum recommended
                thinning=16,        # BoTorch default
                disable_progbar=True,
            )
            model.eval()

            # Use qLogExpectedImprovement for numerical stability (Ament et al. 2023)
            try:
                from botorch.acquisition.logei import qLogExpectedImprovement
                acqf = qLogExpectedImprovement(model, best_f=train_Y.max())
            except ImportError:
                from botorch.acquisition import qExpectedImprovement
                acqf = qExpectedImprovement(model, best_f=train_Y.max())

            bounds = torch.stack([
                torch.zeros(self._dim), torch.ones(self._dim),
            ]).double()
            candidate, _ = optimize_acqf(
                acqf, bounds=bounds, q=1,
                num_restarts=10, raw_samples=1024,
            )
            x_next = self._from_unit(candidate.detach().numpy().flatten())
        except Exception:
            x_next = np.random.uniform(-5, 5, size=self._dim)

        x_next = x_next.reshape(1, self._dim)
        y_next = self._black_box(x_next)
        score = float(y_next.flatten()[0])

        self._history_x.append(x_next)
        self._history_y.append(score)
        self._best = max(self._best, score)
        self.iteration += 1
        return score

    def solve(self, max_iter=25, **kwargs):
        for _ in range(max_iter):
            self.step()

    def get_best_performance(self):
        return np.array([[self._best]])

    def get_history_as_arrays(self, penalize_nans_with=-1.0):
        x = np.vstack([h.flatten() for h in self._history_x])
        y = np.array(self._history_y).reshape(-1, 1)
        y = np.where(np.isnan(y), penalize_nans_with, y)
        return x, y


# ---------------------------------------------------------------------------
# Lightweight black box wrappers (no poli process isolation)
# ---------------------------------------------------------------------------


class LatentBlackBox:
    """Wraps a VAE decoder + molecular oracle as a callable for all solvers.

    All solvers call this object directly as ``bb(z)`` where z is a latent
    vector. Decodes to SMILES, scores via GuacaMol oracle, and returns scores.
    """

    _Info = type("Info", (), {"is_deterministic": True})

    def __init__(self, decode_fn, score_fn):
        """
        Args:
            decode_fn: Callable(np.ndarray[N, D]) -> list[str] (latent -> SMILES)
            score_fn: Callable(str) -> float (SMILES -> score)
        """
        self.decode_fn = decode_fn
        self.score_fn = score_fn
        self.n_calls = 0
        self.n_failures = 0

    def __call__(self, z: np.ndarray) -> np.ndarray:
        """Evaluate latent vectors. Returns [N, 1] scores.

        Failed evaluations return NaN (not 0.0) so CMA-ES can distinguish
        genuine zeros from scoring failures via penalize_nans_with.
        Decode + score batch is capped at 30s via SIGALRM to prevent hangs
        from very long autoregressive sequences on out-of-distribution
        z-vectors (the VAE decoder is the hang risk, not the oracle).
        """
        import signal

        def _timeout_handler(signum, frame):
            raise TimeoutError("Decode+score batch exceeded 30s")

        # Timeout covers both decode (the actual hang risk: O(N^3) for long
        # autoregressive sequences) and scoring (fast, but included for safety).
        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(30)
        try:
            smiles_list = self.decode_fn(z)
        except (TimeoutError, RuntimeError) as e:
            logger.warning(f"Decode failed/timed out for batch of {len(z)}: "
                           f"{type(e).__name__}: {e}")
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
            self.n_failures += len(z)
            return np.full((len(z), 1), float("nan"))
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        scores = []
        for smi in smiles_list:
            try:
                if smi:
                    s = self.score_fn(smi)
                else:
                    s = float("nan")
            except (ValueError, TypeError, KeyError, AttributeError,
                    RuntimeError, ZeroDivisionError) as e:
                logger.warning(
                    f"Oracle scoring failed for SMILES '{smi[:80] if smi else '<empty>'}': "
                    f"{type(e).__name__}: {e}"
                )
                s = float("nan")
                self.n_failures += 1
            scores.append(s)
            self.n_calls += 1
        return np.array(scores).reshape(-1, 1)

    # Attributes that poli solvers may check
    @property
    def info(self):
        return self._Info()

    @property
    def num_workers(self):
        return 1

    def terminate(self):
        pass


# ---------------------------------------------------------------------------
# Our VAE mode
# ---------------------------------------------------------------------------


def make_our_latent_bb(
    task_name: str,
    device: str = "cuda",
    decoder_weights_path: str | None = None,
    codec_type: str = "selfies",
) -> tuple:
    """Create latent black box using our SELFIES or SpCauchy Transformer VAE (256D).

    Returns (black_box, codec, latent_dim).
    """
    from src.oracle.guacamol import GuacaMolOracle

    # Map poli name to our short task ID
    our_task = POLI_TO_GUACAMOL.get(task_name, task_name)

    if codec_type == "spcauchy":
        from src.codec.spcauchy_codec import SpCauchyVAECodec
        codec = SpCauchyVAECodec.from_pretrained(device=device)
    elif codec_type == "chembed":
        from src.codec.chembed_codec import ChembedVAECodec
        codec = ChembedVAECodec.from_pretrained(device=device)
    elif codec_type == "power_spherical":
        # Power Spherical VAE codec — second hyperspherical-VAE point in the
        # codec-geometry ablation alongside spcauchy. See De Cao & Aziz 2020.
        from src.codec.power_spherical_codec import PowerSphericalVAECodec
        codec = PowerSphericalVAECodec.from_pretrained(device=device)
    elif codec_type == "vmf":
        # von Mises-Fisher VAE codec — third hyperspherical-VAE point.
        # Davidson et al. 2018 original construction with rejection-sampling
        # reparameterization. Tested empirically at D=256 despite Davidson's
        # caveats about D > 64 stability.
        from src.codec.vmf_codec import VMFVAECodec
        codec = VMFVAECodec.from_pretrained(device=device)
    else:
        from src.codec import SELFIESVAECodec
        codec = SELFIESVAECodec.from_pretrained(
            device=device, decoder_weights_path=decoder_weights_path
        )
    oracle = GuacaMolOracle(task_id=our_task)

    # Cap max_string_length to prevent O(N³) decoder hangs on OOD z-vectors.
    # COWBOYS uses 80; we match that. Without this, random z-vectors without
    # stop tokens cause 1024 autoregressive steps × quadratic attention.
    codec.model.max_string_length = 80

    def decode_fn(z: np.ndarray) -> list[str]:
        z_tensor = torch.tensor(z, dtype=torch.float32, device=device)
        return codec.decode(z_tensor)

    bb = LatentBlackBox(decode_fn=decode_fn, score_fn=oracle.score)
    return bb, codec, 256


def generate_cold_start_ours(
    codec, oracle_score_fn, n_cold_start: int, seed: int, device: str = "cuda"
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Generate cold start points by encoding random ZINC molecules.

    Cold-start molecules are scored DIRECTLY (not through VAE round-trip)
    because they are known molecules, not generated ones.

    Returns (z0, y0, cs_smiles) where y0 = direct oracle scores.
    """
    from src.oracle.guacamol import load_zinc_smiles

    zinc_smiles = load_zinc_smiles(n_samples=10000)
    rng = np.random.RandomState(seed)
    cs_indices = rng.choice(len(zinc_smiles), n_cold_start, replace=False)
    cs_smiles = [zinc_smiles[i] for i in cs_indices]
    z0 = codec.encode(cs_smiles).cpu().numpy()
    # Score directly — no VAE round-trip for known molecules
    y0 = np.array([oracle_score_fn(smi) for smi in cs_smiles]).reshape(-1, 1)
    return z0, y0, cs_smiles


def generate_cold_start_csv(
    codec, task_name: str, n_cold_start: int, device: str = "cuda"
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load cold start from GuacaMol CSV -- same as external baselines.

    Deterministic: always loads first n_cold_start rows from the CSV.
    This ensures poli solvers get the exact same cold start as
    the pre-existing external baseline result files (TuRBO, LOL-BO, etc.).
    """
    from src.oracle.guacamol import load_guacamol_data

    our_task = POLI_TO_GUACAMOL.get(task_name, task_name)
    smiles_list, scores, _ = load_guacamol_data(
        n_samples=n_cold_start, task_id=our_task
    )
    z0 = codec.encode(smiles_list).cpu().numpy()
    y0 = scores.numpy().reshape(-1, 1)
    return z0, y0, smiles_list


# ---------------------------------------------------------------------------
# HDBO VAE mode
# ---------------------------------------------------------------------------


def make_hdbo_latent_bb(task_name: str, latent_dim: int = 128) -> tuple:
    """Create latent black box using HDBO's MLP VAE.

    Returns (black_box, vae, latent_dim).
    """
    from src.oracle.guacamol import GuacaMolOracle

    our_task = POLI_TO_GUACAMOL.get(task_name, task_name)
    oracle = GuacaMolOracle(task_id=our_task)

    # Load HDBO VAE
    from hdbo_benchmark.generative_models.vae_factory import VAEFactory

    vae = VAEFactory().create(
        experiment_name="benchmark_on_pmo",
        latent_dim=latent_dim,
    )

    def decode_fn(z: np.ndarray) -> list[str]:
        selfies_strings = vae.decode_to_string_array(z)
        import selfies as sf

        smiles_list = []
        for s in selfies_strings:
            try:
                smi = sf.decoder(s)
                smiles_list.append(smi if smi else "")
            except (ValueError, TypeError, IndexError, KeyError) as e:
                logger.warning(f"SELFIES decode failed for '{s[:40]}': {e}")
                smiles_list.append("")
        return smiles_list

    bb = LatentBlackBox(decode_fn=decode_fn, score_fn=oracle.score)
    return bb, vae, latent_dim


def generate_cold_start_hdbo(
    latent_dim: int, n_initial_points: int, seed: int
) -> np.ndarray:
    """Generate Sobol initial points in latent space (same as HDBO benchmark)."""
    sobol = torch.quasirandom.SobolEngine(
        dimension=latent_dim, scramble=True, seed=seed
    )
    z0 = sobol.draw(n_initial_points).numpy().astype(np.float32)
    return z0


# ---------------------------------------------------------------------------
# Solver factory
# ---------------------------------------------------------------------------


def _create_solver(
    solver_name: str,
    black_box,
    x0: np.ndarray,
    y0: np.ndarray,
    n_dimensions: int,
    subspace_dim: int,
    n_subspaces: int,
    seed: int,
    n_initial_points: int,
    codec=None,
    norm_mode: str = "fixed",
    n_iterations: int | None = None,
    cs_smiles: list[str] | None = None,
):
    """Instantiate a solver by name.

    All solvers use the unified poli interface: (black_box, x0, y0).
    """
    # --- Our solvers (src/spherevo/solver.py) ---

    if solver_name == "cma_es_subspace":
        from src.spherevo.solver import CMAESSubspaceSolver

        return CMAESSubspaceSolver(
            black_box=black_box, x0=x0, y0=y0,
            subspace_dim=subspace_dim,
            population_size=5,
            norm_mode=norm_mode,
        )

    if solver_name == "lq_cma_es_subspace":
        from src.spherevo.solver import LQCMAESSubspaceSolver

        return LQCMAESSubspaceSolver(
            black_box=black_box, x0=x0, y0=y0,
            subspace_dim=subspace_dim,
            population_size=5,
        )

    if solver_name == "lppfn_subspace":
        # Not part of this release: the LP-PFN solver (package src/lppfn/) is
        # used by no result of the paper, so its code was removed. The name
        # stays in OUR_SOLVERS (content and order pinned by
        # tests/test_published_block.py), and this block has the line count of
        # the removed dispatch, because scripts/cr/analysis/appendixH_facts.csv
        # cites line numbers of this file.
        raise ValueError(
            "solver 'lppfn_subspace' is not part of this release: no result "
            "of the paper uses it, and its code (src/lppfn/) was removed")

    if solver_name == "jacobian_subspace":
        from src.spherevo.solver import JacobianSubspaceSolver

        return JacobianSubspaceSolver(
            black_box=black_box, x0=x0, y0=y0,
            subspace_dim=subspace_dim,
            population_size=5,
            codec=codec,
        )

    if solver_name == "gp_free_spherical":
        from src.spherevo.solver import GPFreeSphericalSolver

        return GPFreeSphericalSolver(
            black_box=black_box, x0=x0, y0=y0,
            n_subspaces=n_subspaces,
            subspace_dim=subspace_dim,
        )

    if solver_name == "gp_guided_spherical":
        from src.spherevo.solver import GPGuidedSphericalSolver

        return GPGuidedSphericalSolver(
            black_box=black_box, x0=x0, y0=y0,
            n_subspaces=n_subspaces,
            subspace_dim=subspace_dim,
        )

    if solver_name == "noise_gp_spherical":
        from src.spherevo.solver import NoiseGPSphericalSolver

        return NoiseGPSphericalSolver(
            black_box=black_box, x0=x0, y0=y0,
            n_subspaces=n_subspaces,
            subspace_dim=subspace_dim,
        )

    if solver_name == "rose_subspace":
        from src.spherevo.solver import ROSESubspaceSolver

        return ROSESubspaceSolver(
            black_box=black_box, x0=x0, y0=y0,
            subspace_dim=subspace_dim,
        )

    if solver_name == "pace_subspace":
        from src.spherevo.solver import PACESubspaceSolver

        return PACESubspaceSolver(
            black_box=black_box, x0=x0, y0=y0,
            subspace_dim=subspace_dim,
        )

    if solver_name == "random_search":
        from src.spherevo.solver import RandomSearchSolver

        return RandomSearchSolver(
            black_box=black_box, x0=x0, y0=y0,
        )

    if solver_name == "random_search_prior":
        from src.spherevo.solver import RandomSearchPriorSolver

        return RandomSearchPriorSolver(
            black_box=black_box, x0=x0, y0=y0,
        )

    if solver_name == "random_box_faithful":
        # Surrogate-free uniform draw in the faithful [p01,p99] box (same bounds
        # file + 1e-6 floor as the *_box_faithful / _box_argmax GP arms); ported
        # unchanged from p1-blocks commit 4780475 (A10 S2b). The shell
        # counterpart is "random_search". Codec dispatch as in the
        # *_box_faithful block below (SELFIES -> None -> the ZINC bounds file).
        from src.spherevo.solver import RandomBoxFaithfulSolver

        codec_class_name = type(codec).__name__ if codec is not None else ""
        if "Chembed" in codec_class_name:
            box_bounds = "results/precomputed/chembed_per_coord_bounds.json"
        elif "SpCauchy" in codec_class_name:
            box_bounds = "results/precomputed/spcauchy_per_coord_bounds.json"
        elif "PowerSpherical" in codec_class_name:
            box_bounds = "results/precomputed/power_spherical_per_coord_bounds.json"
        elif "VMF" in codec_class_name or "Vmf" in codec_class_name:
            box_bounds = "results/precomputed/vmf_per_coord_bounds.json"
        else:
            box_bounds = None
        return RandomBoxFaithfulSolver(
            black_box=black_box, x0=x0, y0=y0, bounds_path=box_bounds,
        )

    if solver_name == "random_boxdir":
        # Control block C2b (camera-ready run rules, amendment of 2026-10-12):
        # random_box_faithful's draw renormalised to the run's shell radius (the
        # cold-start mean norm), one draw per step, no surrogate; the
        # surrogate-free twin of hvarfner_scaled_gp_boxdir_argmax. Codec
        # dispatch as random_box_faithful above (SELFIES -> None -> the ZINC
        # bounds file).
        from src.spherevo.solver import RandomBoxDirSolver

        codec_class_name = type(codec).__name__ if codec is not None else ""
        if "Chembed" in codec_class_name:
            box_bounds = "results/precomputed/chembed_per_coord_bounds.json"
        elif "SpCauchy" in codec_class_name:
            box_bounds = "results/precomputed/spcauchy_per_coord_bounds.json"
        elif "PowerSpherical" in codec_class_name:
            box_bounds = "results/precomputed/power_spherical_per_coord_bounds.json"
        elif "VMF" in codec_class_name or "Vmf" in codec_class_name:
            box_bounds = "results/precomputed/vmf_per_coord_bounds.json"
        else:
            box_bounds = None
        return RandomBoxDirSolver(
            black_box=black_box, x0=x0, y0=y0, bounds_path=box_bounds,
        )

    if solver_name == "convex_hull_rs":
        from src.spherevo.solver import ConvexHullRandomSearchSolver
        return ConvexHullRandomSearchSolver(black_box=black_box, x0=x0, y0=y0)

    # Convex-hull norm sweep: k anchors, uniform Dirichlet -> candidate norm ~ r-bar/sqrt(k).
    # Interpolates the empirical shell (k=1, vertices) down to the centroid (k large),
    # answering Lhbx Q4 (convex hull) AND Q3 (other radii) in one family.
    _CONVEX_K = {"convex_hull_k1": 1, "convex_hull_k2": 2, "convex_hull_k4": 4,
                 "convex_hull_k8": 8, "convex_hull_k16": 16, "convex_hull_k64": 64}
    if solver_name in _CONVEX_K:
        from src.spherevo.solver import ConvexHullRandomSearchSolver
        return ConvexHullRandomSearchSolver(
            black_box=black_box, x0=x0, y0=y0, n_anchors=_CONVEX_K[solver_name],
        )

    if solver_name == "ellipsoid_rs":
        from src.spherevo.solver import EllipsoidRandomSearchSolver
        return EllipsoidRandomSearchSolver(black_box=black_box, x0=x0, y0=y0)

    # Sphere restricted to the top-K PCA subspace of the training latents: same
    # radius as the isotropic scaffold, but all norm on active directions.
    # Isolates DIRECTION from NORM (Lhbx W2).
    _PCA_K = {"pca_sphere_k16": 16, "pca_sphere_k32": 32,
              "pca_sphere_k64": 64, "pca_sphere_k128": 128}
    if solver_name in _PCA_K:
        from src.spherevo.solver import PCASubspaceSphereSolver
        return PCASubspaceSphereSolver(
            black_box=black_box, x0=x0, y0=y0, n_components=_PCA_K[solver_name],
        )

    if solver_name == "cowboys":
        from src.spherevo.cowboys_solver import COWBOYSSolver

        return COWBOYSSolver(
            black_box=black_box, x0=x0, y0=y0,
            codec=codec,
            n_candidates=20,
        )

    if solver_name == "rembo":
        from src.spherevo.solver import REMBOSolver

        return REMBOSolver(
            black_box=black_box, x0=x0, y0=y0,
            subspace_dim=subspace_dim,
        )

    if solver_name == "hesbo":
        from src.spherevo.solver import HeSBOSolver

        return HeSBOSolver(
            black_box=black_box, x0=x0, y0=y0,
            subspace_dim=subspace_dim,
        )

    if solver_name == "geowalk":
        from src.spherevo.solver import GeoWalkSolver

        # Pass total_budget so the angle schedule naturally scales with
        # larger budgets (Table~\ref{tab:budget}).
        return GeoWalkSolver(
            black_box=black_box, x0=x0, y0=y0,
            total_budget=n_iterations,
        )

    if solver_name == "geowalk_srda":
        from src.spherevo.solver import GeoWalkSolver

        return GeoWalkSolver(
            black_box=black_box, x0=x0, y0=y0,
            total_budget=n_iterations,
            direction_accumulation=True,
            srda_cc=0.30,
            srda_lambda=0.50,
        )

    # GeoWalk ablation variants (Section 5)
    _GEOWALK_ABLATIONS = {
        "geowalk_const035":     {"angle_init": 0.35, "angle_final": 0.35},
        "geowalk_const060":     {"angle_init": 0.60, "angle_final": 0.60},
        "geowalk_const015":     {"angle_init": 0.15, "angle_final": 0.15},
        "geowalk_explore8":     {"explore_steps": 8},
        "geowalk_explore24":    {"explore_steps": 24},
        # Radius / support-geometry robustness (reviewers Lhbx Q1/Q2/Q3, 6qvY W2).
        # Fixed-radius sweep (absolute r; SELFIES empirical r̄ ≈ 7.98).
        "geowalk_r4":       {"radius_override": 4.0},
        "geowalk_r6":       {"radius_override": 6.0},
        "geowalk_r8":       {"radius_override": 8.0},
        "geowalk_r10":      {"radius_override": 10.0},
        "geowalk_r12":      {"radius_override": 12.0},
        "geowalk_r16":      {"radius_override": 16.0},
        # Thick-shell / annulus sweep (jitter radius uniformly over ±eps around r̄).
        "geowalk_shell10":  {"shell_eps": 0.10},
        "geowalk_shell20":  {"shell_eps": 0.20},
        # Radius-estimator alternatives (Lhbx Q1, 6qvY 5-restart concern).
        "geowalk_rmedian":  {"radius_estimator": "median"},
        "geowalk_rpop":     {"radius_override": EMPIRICAL_SHELL_MEAN_NORM},  # fixed population r̄
        # Thick shell sampled uniformly over the empirical 5-95th percentile band
        "geowalk_shellpct": {"radius_range": EMPIRICAL_SHELL_PCT_5_95},
    }
    if solver_name in _GEOWALK_ABLATIONS:
        from src.spherevo.solver import GeoWalkSolver

        return GeoWalkSolver(
            black_box=black_box, x0=x0, y0=y0,
            total_budget=n_iterations,
            **_GEOWALK_ABLATIONS[solver_name],
        )

    if solver_name == "geowalk_one_fifth":
        # GeoWalk exploration + greedy geodesic refinement whose angle starts at
        # 0.30 rad and is multiplied by 1.2 after an improving refinement step
        # and by 0.85 otherwise, clamped to [0.05, 0.50] (see the class).
        from src.spherevo.solver import GeoWalkOneFifthSolver

        return GeoWalkOneFifthSolver(
            black_box=black_box, x0=x0, y0=y0,
            total_budget=n_iterations,
        )

    if solver_name == "geowalk_flat":
        from src.spherevo.solver import FlatGeoWalkSolver

        return FlatGeoWalkSolver(
            black_box=black_box, x0=x0, y0=y0,
            total_budget=n_iterations,
        )

    if solver_name == "geowalk_no_anchor":
        from src.spherevo.solver import GeoWalkNoAnchorSolver

        return GeoWalkNoAnchorSolver(
            black_box=black_box, x0=x0, y0=y0,
            total_budget=n_iterations,
        )

    if solver_name == "geowalk_no_tangent":
        from src.spherevo.solver import GeoWalkNoTangentSolver

        return GeoWalkNoTangentSolver(
            black_box=black_box, x0=x0, y0=y0,
        )

    # --- Riemannian BO (Jaquier et al.) ---
    # GaBO (CoRL19), HD-GaBO (NeurIPS20), MaternGaBO (CoRL21).
    # See ``src/spherevo/riemannian_bo.py`` for the wrapper details and
    # ``papers/riemannian_bo/{GaBOtorch,MaternGaBO}/`` for the upstream code.
    if solver_name == "gabo":
        from src.spherevo.riemannian_bo import GaBOSolver

        return GaBOSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "matern_gabo":
        from src.spherevo.riemannian_bo import MaternGaBOSolver

        return MaternGaBOSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name in {"hd_gabo_2", "hd_gabo_16"}:
        from src.spherevo.riemannian_bo import HDGaBOSolver

        latent_dim = 2 if solver_name == "hd_gabo_2" else 16
        return HDGaBOSolver(
            black_box=black_box, x0=x0, y0=y0, latent_dim=latent_dim,
        )

    # --- External poli-baselines ---

    if solver_name == "cma_es":
        from poli_baselines.solvers.evolutionary_strategies.cma_es import CMA_ES

        # Bounds must contain x0 (cold-start encoded latents). For SELFIES
        # x0 fits comfortably in (-5, 5); for Chembed's wider per-coordinate
        # spread (max abs ~10 over a ZINC sample), default bounds reject x0
        # in cma.transformations._inverse_i. Adapt: widen 50% past observed
        # max abs, with a (-5, 5) floor.
        x0_max_abs = float(np.max(np.abs(x0))) if x0 is not None and len(x0) > 0 else 5.0
        bound_mag = max(5.0, np.ceil(x0_max_abs * 1.5))
        bounds = (-bound_mag, bound_mag)
        return CMA_ES(
            black_box=black_box,
            x0=x0,
            y0=y0,
            initial_mean=np.random.randn(n_dimensions).reshape(1, -1).clip(*bounds),
            population_size=10,
            initial_sigma=1.0,
            bounds=bounds,
        )

    if solver_name == "cma_es_sphere":
        # Same as cma_es but candidates are normalized to S^{D-1} at the
        # cold-start mean norm before decoding. CMA-ES still operates in
        # the box (its mean / covariance live in R^D), but the score it
        # receives is from the sphere-projected version of each proposal.
        # The radial direction therefore becomes uninformative; CMA-ES
        # has to learn this by feedback.
        from poli_baselines.solvers.evolutionary_strategies.cma_es import CMA_ES

        cold_norms = np.linalg.norm(x0, axis=-1)
        mean_norm = float(cold_norms.mean()) if len(cold_norms) > 0 else 1.0

        class _SphereProjectedBlackBox:
            """Wraps black_box; normalizes each row of z to ``mean_norm`` before evaluation."""

            _Info = type("Info", (), {"is_deterministic": True})

            def __init__(self, inner, mean_norm):
                self.inner = inner
                self.mean_norm = mean_norm
                self.n_calls = 0
                self.n_failures = 0
                self.info = self._Info()

            def __call__(self, z):
                z = np.asarray(z)
                norms = np.linalg.norm(z, axis=-1, keepdims=True)
                z_sphere = z / np.maximum(norms, 1e-12) * self.mean_norm
                out = self.inner(z_sphere)
                # Mirror inner's call-counter accounting so caller sees a
                # consistent n_calls / n_failures.
                self.n_calls = self.inner.n_calls
                self.n_failures = self.inner.n_failures
                return out

        wrapped = _SphereProjectedBlackBox(black_box, mean_norm)
        x0_max_abs = float(np.max(np.abs(x0))) if x0 is not None and len(x0) > 0 else 5.0
        bound_mag = max(5.0, np.ceil(x0_max_abs * 1.5))
        bounds = (-bound_mag, bound_mag)
        return CMA_ES(
            black_box=wrapped,
            x0=x0,
            y0=y0,
            initial_mean=np.random.randn(n_dimensions).reshape(1, -1).clip(*bounds),
            population_size=10,
            initial_sigma=1.0,
            bounds=bounds,
        )

    if solver_name == "hill_climbing":
        from poli_baselines.solvers.simple.continuous_random_mutation import (
            ContinuousRandomMutation,
        )

        return ContinuousRandomMutation(
            black_box=black_box,
            x0=x0,
            y0=y0,
            std=0.25,
            bounds=(-5.0, 5.0),
        )

    if solver_name in ("random_line_bo", "coordinate_line_bo"):
        from poli_baselines.solvers.bayesian_optimization.line_bayesian_optimization import (
            LineBO,
        )

        line_type = "coordinate" if solver_name == "coordinate_line_bo" else "random"
        return LineBO(
            black_box=black_box,
            x0=x0,
            y0=y0,
            type_of_line=line_type,
            bounds=(-5.0, 5.0),
        )

    if solver_name == "turbo":
        from poli_baselines.solvers.bayesian_optimization.turbo.turbo_wrapper import (
            Turbo,
        )

        torch.set_default_dtype(torch.float64)
        # Adapt bounds to fit x0 envelope (Chembed's per-coord spread is wider
        # than SELFIES). Same logic as cma_es above.
        x0_max_abs = float(np.max(np.abs(x0))) if x0 is not None and len(x0) > 0 else 5.0
        bound_mag = max(5.0, np.ceil(x0_max_abs * 1.5))
        return Turbo(
            black_box=black_box,
            x0=x0,
            y0=y0,
            bounds=(-bound_mag, bound_mag),
        )

    if solver_name == "vanilla_bo_hvarfner":
        from poli_baselines.solvers.bayesian_optimization.vanilla_bayesian_optimization import (
            VanillaBayesianOptimization,
        )

        return VanillaBayesianOptimization(
            black_box=black_box,
            x0=x0,
            y0=y0,
            bounds=(-5.0, 5.0),
        )

    if solver_name == "saasbo":
        return _SAASBOSolver(black_box=black_box, x0=x0, y0=y0)

    # Native GP competitors (reviewer-response, Section 6)
    if solver_name == "hvarfner_scaled_gp":
        from src.spherevo.gp_competitors import HvarfnerScaledGPSolver
        return HvarfnerScaledGPSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "papenmeier_msr":
        from src.spherevo.gp_competitors import PapenmeierMSRSolver
        return PapenmeierMSRSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "xu_standard_gp":
        from src.spherevo.gp_competitors import XuStandardGPSolver
        return XuStandardGPSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "hvarfner_scaled_gp_box":
        from src.spherevo.gp_competitors import HvarfnerScaledGPBoxSolver
        return HvarfnerScaledGPBoxSolver(black_box=black_box, x0=x0, y0=y0)

    # Box-radius sweep variants (k = sigma multiplier; default Hvarfner-on-box
    # is k = 3, see HvarfnerScaledGPBoxSolver.box_half_width). Sweep tests
    # whether the box-vs-sphere effect is driven by the box width being
    # mis-calibrated relative to the empirical sphere radius.
    if solver_name == "hvarfner_scaled_gp_box1":
        from src.spherevo.gp_competitors import HvarfnerScaledGPBox1Solver
        return HvarfnerScaledGPBox1Solver(black_box=black_box, x0=x0, y0=y0)
    if solver_name == "hvarfner_scaled_gp_box2":
        from src.spherevo.gp_competitors import HvarfnerScaledGPBox2Solver
        return HvarfnerScaledGPBox2Solver(black_box=black_box, x0=x0, y0=y0)
    if solver_name == "hvarfner_scaled_gp_box4":
        from src.spherevo.gp_competitors import HvarfnerScaledGPBox4Solver
        return HvarfnerScaledGPBox4Solver(black_box=black_box, x0=x0, y0=y0)
    if solver_name == "hvarfner_scaled_gp_box5":
        from src.spherevo.gp_competitors import HvarfnerScaledGPBox5Solver
        return HvarfnerScaledGPBox5Solver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "hvarfner_scaled_gp_box_projected":
        # Mechanism diagnostic: Hvarfner GP + box-supported L-BFGS acquisition,
        # then post-hoc projection of the candidate to the empirical sphere.
        from src.spherevo.gp_competitors import HvarfnerScaledGPBoxProjectedSolver
        return HvarfnerScaledGPBoxProjectedSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name in (
        "hvarfner_scaled_gp_box_faithful",
        "papenmeier_msr_box_faithful",
        "xu_standard_gp_box_faithful",
        "papenmeier_msr_published",
        "turbo_box_faithful",
    ):
        # Faithful on-box variants: per-coord [p01, p99] rescaling to [0,1]^D so
        # each recipe's lengthscale prior/init operates on the unit-cube domain it
        # was calibrated for. Bounds are codec-specific because per-coord stats
        # differ across encoders (SELFIES mean width ≈ 1.43, Chembed ≈ 1.77).
        from src.spherevo.gp_competitors import (
            HvarfnerScaledGPBoxFaithfulSolver,
            PapenmeierMSRBoxFaithfulSolver,
            PapenmeierMSRPublishedSolver,
            XuStandardGPBoxFaithfulSolver,
        )
        from src.spherevo.turbo_box_faithful import TurboBoxFaithfulSolver
        faithful_cls = {
            "hvarfner_scaled_gp_box_faithful": HvarfnerScaledGPBoxFaithfulSolver,
            "papenmeier_msr_box_faithful": PapenmeierMSRBoxFaithfulSolver,
            "xu_standard_gp_box_faithful": XuStandardGPBoxFaithfulSolver,
            # Same box and GP as papenmeier_msr_box_faithful; acquisition step
            # replaced by the published MSR call (see gp_competitors.py).
            "papenmeier_msr_published": PapenmeierMSRPublishedSolver,
            # Same fixed [p01, p99] box and the same bounds file; the GP and the
            # acquisition step are TuRBO-1's (trust region + Thompson sampling).
            "turbo_box_faithful": TurboBoxFaithfulSolver,
        }[solver_name]
        codec_class_name = type(codec).__name__ if codec is not None else ""
        if "Chembed" in codec_class_name:
            bounds = "results/precomputed/chembed_per_coord_bounds.json"
        elif "SpCauchy" in codec_class_name:
            bounds = "results/precomputed/spcauchy_per_coord_bounds.json"
        elif "PowerSpherical" in codec_class_name:
            bounds = "results/precomputed/power_spherical_per_coord_bounds.json"
        elif "VMF" in codec_class_name or "Vmf" in codec_class_name:
            bounds = "results/precomputed/vmf_per_coord_bounds.json"
        else:
            bounds = None  # solver default = SELFIES bounds
        return faithful_cls(
            black_box=black_box, x0=x0, y0=y0, bounds_path=bounds,
        )

    # Sphere-radius sweep variants (multiplier on empirical mean_norm).
    # Tests whether r = mean_norm specifically matters or any sphere works.
    if solver_name == "hvarfner_scaled_gp_sphere05":
        from src.spherevo.gp_competitors import HvarfnerScaledGPSphere05Solver
        return HvarfnerScaledGPSphere05Solver(black_box=black_box, x0=x0, y0=y0)
    if solver_name == "hvarfner_scaled_gp_sphere15":
        from src.spherevo.gp_competitors import HvarfnerScaledGPSphere15Solver
        return HvarfnerScaledGPSphere15Solver(black_box=black_box, x0=x0, y0=y0)
    if solver_name == "hvarfner_scaled_gp_sphere20":
        from src.spherevo.gp_competitors import HvarfnerScaledGPSphere20Solver
        return HvarfnerScaledGPSphere20Solver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "hvarfner_scaled_gp_prior":
        # Reviewer-response: candidates drawn from N(0, I_D) (the VAE's prior).
        from src.spherevo.gp_competitors import HvarfnerScaledGPPriorSolver
        return HvarfnerScaledGPPriorSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "hvarfner_scaled_gp_angular_box":
        # Reviewer-response: uniform-in-box candidates renormalized to mean_norm.
        # Same radial distribution as on-sphere, only angular distribution differs.
        from src.spherevo.gp_competitors import HvarfnerScaledGPAngularBoxSolver
        return HvarfnerScaledGPAngularBoxSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "papenmeier_msr_prior":
        from src.spherevo.gp_competitors import PapenmeierMSRPriorSolver
        return PapenmeierMSRPriorSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "papenmeier_msr_angular_box":
        from src.spherevo.gp_competitors import PapenmeierMSRAngularBoxSolver
        return PapenmeierMSRAngularBoxSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "xu_standard_gp_prior":
        from src.spherevo.gp_competitors import XuStandardGPPriorSolver
        return XuStandardGPPriorSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "xu_standard_gp_angular_box":
        from src.spherevo.gp_competitors import XuStandardGPAngularBoxSolver
        return XuStandardGPAngularBoxSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "hvarfner_scaled_gp_box_faithful_projected":
        # Shell support reached by continuous L-BFGS over the SAME [p01,p99]
        # domain the box arm uses, so the factorial varies one factor at a time.
        from src.spherevo.gp_competitors import HvarfnerScaledGPBoxFaithfulProjectedSolver
        return HvarfnerScaledGPBoxFaithfulProjectedSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "hvarfner_scaled_gp_sphere_continuous":
        # Shell support with the acquisition maximised CONTINUOUSLY on the shell
        # itself, so the two continuous arms of the factorial differ only in
        # support. Neither the argmax arm nor the projected arm can do this.
        from src.spherevo.gp_competitors import HvarfnerScaledGPSphereContinuousSolver
        return HvarfnerScaledGPSphereContinuousSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "hvarfner_scaled_gp_sphere_lbfgs":
        # Control C4 (camera-ready run rules, amendment of 2026-10-14): the
        # shell x continuous cell with the box arm's maximiser, optimize_acqf
        # (10 restarts, 512 raw samples, L-BFGS-B) on x = r * v / ||v||, so the
        # two continuous cells of the factorial use the same optimiser. The printed
        # shell x continuous cell since 2026-11-06 (RUN_RULES 2026-11-06 16:30).
        from src.spherevo.gp_competitors import HvarfnerScaledGPSphereLBFGSSolver
        return HvarfnerScaledGPSphereLBFGSSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "hvarfner_scaled_gp_box_argmax":
        # Faithful box support, but the acquisition is maximised by argmax over a
        # uniform candidate set -- the cell that separates the support effect from
        # the acquisition-optimizer effect in the submitted ablation.
        from src.spherevo.gp_competitors import HvarfnerScaledGPBoxArgmaxSolver
        return HvarfnerScaledGPBoxArgmaxSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "hvarfner_scaled_gp_box_argmax_raw":
        # Control block C1 (camera-ready run rules, amendment of 2026-10-12):
        # hvarfner_scaled_gp_box_argmax with the GP fitted on raw latents, as the
        # shell arm fits it, instead of on per-coordinate rescaled inputs
        # (input-transform control). Same bounds as hvarfner_scaled_gp_box_argmax
        # (the class default, no codec dispatch).
        from src.spherevo.gp_competitors import HvarfnerScaledGPBoxArgmaxRawSolver
        return HvarfnerScaledGPBoxArgmaxRawSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "hvarfner_scaled_gp_boxdir_argmax":
        # Control block C2a: the [p01,p99]-box draws of the C1 arm, each
        # renormalised to the run's shell radius (cold-start mean norm); GP on
        # raw latents, qLogEI argmax (anisotropy control). Bounds as C1.
        from src.spherevo.gp_competitors import HvarfnerScaledGPBoxDirArgmaxSolver
        return HvarfnerScaledGPBoxDirArgmaxSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "hvarfner_scaled_gp_filtered_box":
        # Mechanism diagnostic: box-supported, REJECT candidates with ||x||>1.2*r̄.
        from src.spherevo.gp_competitors import HvarfnerScaledGPFilteredBoxSolver
        return HvarfnerScaledGPFilteredBoxSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "hvarfner_scaled_gp_normalized":
        # Universal recipe: sample from prior, normalize all inputs to ||x||=r̄.
        from src.spherevo.gp_competitors import HvarfnerScaledGPNormalizedSolver
        return HvarfnerScaledGPNormalizedSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name.startswith("proximity_"):
        # Surrogate-free null model for the on-sphere GP's acquisition argmax.
        # Same 2048-point on-shell candidate pool; selection by angle to the
        # incumbent. See ProximitySelectionSolver for the measured motivation.
        from src.spherevo.solver import ProximitySelectionSolver
        n_cand = 2048
        cfg = {
            "proximity_q06": dict(q_select=0.06, mode="near"),
            "proximity_nearest": dict(q_select=1.0 / n_cand, mode="near"),
            "proximity_far": dict(q_select=0.06, mode="far"),
            # Exploration-matched variants: the plain rules anchor on the best of
            # 5 cold-start points from step 1, while GeoWalk and the on-sphere GP
            # both spend their early budget away from the incumbent. Without this
            # control a null result could just mean "no exploration phase".
            "proximity_q06_e16": dict(q_select=0.06, mode="near", explore_steps=16),
            "proximity_far_e16": dict(q_select=0.06, mode="far", explore_steps=16),
        }[solver_name]
        return ProximitySelectionSolver(
            black_box=black_box, x0=x0, y0=y0, n_candidates=n_cand, **cfg
        )

    if solver_name == "geodesic_slice_gp":
        # Novel: 2D GP per great-circle at current best on S^{D-1}, K=8 directions.
        from src.spherevo.gp_competitors import GeodesicSliceGPSolver
        return GeodesicSliceGPSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "hvarfner_scaled_gp_ball":
        # Novel: Hvarfner GP + GeoWalk-style annealing angular ball proposal.
        # Codec-independent; depends only on ambient D and empirical mean_norm.
        from src.spherevo.gp_competitors import HvarfnerScaledGPBallSolver
        return HvarfnerScaledGPBallSolver(
            black_box=black_box, x0=x0, y0=y0,
            total_budget=int(n_iterations) if n_iterations else 25,
        )

    if solver_name == "sphere_pfn":
        # Not part of this release: SpherePFN (package src/spherevo/pfn/) is
        # used by no result of the paper, so its code was removed. The name
        # stays in OUR_SOLVERS (content and order pinned by
        # tests/test_published_block.py). The class SpherePFNSolver stays in
        # src/spherevo/gp_competitors.py, whose sha256 the same test pins; its
        # import of the removed package runs only when the class is
        # constructed, which this dispatch no longer does. Same line count as
        # the removed dispatch (see the lppfn_subspace block above).
        raise ValueError(
            "solver 'sphere_pfn' is not part of this release: no result of "
            "the paper uses it, and its code (src/spherevo/pfn/) was "
            "removed"
        )

    if solver_name == "papenmeier_msr_box":
        from src.spherevo.gp_competitors import PapenmeierMSRBoxSolver
        return PapenmeierMSRBoxSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "xu_standard_gp_box":
        from src.spherevo.gp_competitors import XuStandardGPBoxSolver
        return XuStandardGPBoxSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "hvarfner_scaled_gp_norm_box_argmax":
        from src.spherevo.gp_competitors import HvarfnerScaledGPNormBoxArgmaxSolver
        return HvarfnerScaledGPNormBoxArgmaxSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "hvarfner_scaled_gp_norm_box_projected":
        from src.spherevo.gp_competitors import HvarfnerScaledGPNormBoxProjectedSolver
        return HvarfnerScaledGPNormBoxProjectedSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "hvarfner_scaled_gp_norm_box":
        from src.spherevo.gp_competitors import HvarfnerScaledGPNormBoxSolver
        return HvarfnerScaledGPNormBoxSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "papenmeier_msr_norm_box":
        from src.spherevo.gp_competitors import PapenmeierMSRNormBoxSolver
        return PapenmeierMSRNormBoxSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "papenmeier_msr_raasp":
        # MSR variant: 0.1*sqrt(D) init + trust region + TuRBO-style RAASP
        # candidates + qLogEI argmax. NOT the published MSR acquisition call,
        # which is papenmeier_msr_published[_norm_box].
        from src.spherevo.gp_competitors import PapenmeierMSRRaaspSolver
        return PapenmeierMSRRaaspSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "papenmeier_msr_published_norm_box":
        # papenmeier_msr_norm_box with the published MSR acquisition call.
        from src.spherevo.gp_competitors import PapenmeierMSRPublishedNormBoxSolver
        return PapenmeierMSRPublishedNormBoxSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "xu_standard_gp_norm_box":
        from src.spherevo.gp_competitors import XuStandardGPNormBoxSolver
        return XuStandardGPNormBoxSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name in ("linear_bo", "linear_bo_sphere", "linear_bo_sphere_calib"):
        # Doumont et al. 2025 "Linear BO" (SphericalLinearKernel GP).
        codec_class_name = type(codec).__name__ if codec is not None else ""
        if "Chembed" in codec_class_name:
            bounds = "results/precomputed/chembed_per_coord_bounds.json"
        elif "SpCauchy" in codec_class_name:
            bounds = "results/precomputed/spcauchy_per_coord_bounds.json"
        elif "PowerSpherical" in codec_class_name:
            bounds = "results/precomputed/power_spherical_per_coord_bounds.json"
        elif "VMF" in codec_class_name or "Vmf" in codec_class_name:
            bounds = "results/precomputed/vmf_per_coord_bounds.json"
        else:
            bounds = None  # solver default = SELFIES bounds
        if solver_name == "linear_bo":
            from src.spherevo.gp_competitors import LinearBOBoxSolver
            return LinearBOBoxSolver(
                black_box=black_box, x0=x0, y0=y0, bounds_path=bounds,
            )
        if solver_name == "linear_bo_sphere_calib":
            # bounds = on-sphere coordinate spread, not the sphere's bounding
            # box; see LinearBOSphereCalibratedSolver for why that matters.
            # No bounds_path: this arm derives its bounds from mean_norm and D,
            # not from a per-coordinate bounds file.
            from src.spherevo.gp_competitors import LinearBOSphereCalibratedSolver
            return LinearBOSphereCalibratedSolver(
                black_box=black_box, x0=x0, y0=y0,
            )
        from src.spherevo.gp_competitors import LinearBOSphereSolver
        return LinearBOSphereSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "tripp_tanimoto_gp":
        from src.spherevo.gp_competitors import TrippTanimotoGPSolver
        if codec is None:
            raise ValueError("tripp_tanimoto_gp requires --mode ours with a codec")
        return TrippTanimotoGPSolver(
            black_box=black_box, x0=x0, y0=y0,
            codec=codec, cs_smiles=cs_smiles,
        )

    if solver_name == "rf_bo":
        from src.spherevo.gp_competitors import RandomForestBOSolver
        return RandomForestBOSolver(black_box=black_box, x0=x0, y0=y0)

    if solver_name == "bounce":
        # Bounce requires poli-baselines (Python 3.11+); not available on 3.9.
        # Use BAxUS results as the closest substitute.
        raise ValueError(
            "Bounce requires poli-baselines (Python 3.11+). "
            "Use BAxUS as the closest available comparison."
        )

    if solver_name in (
        "hvarfner_published",
        "papenmeier_msr_reference",
        "papenmeier_mle_scaled_reference",
        "xu_published",
        "linear_bo_box8",
        "turbo_published",
        "hvarfner_published_allobs",
        "papenmeier_msr_reference_learned",
        "papenmeier_mle_scaled_reference_learned",
    ):
        # Published-recipe block: src/spherevo/published_arms.py. The four [p01, p99] arms take
        # the same codec-specific bounds file as papenmeier_msr_published; linear_bo_box8 runs
        # in Doumont et al.'s fixed molecular box [-8, 8]^D and takes no bounds file.
        from src.spherevo.published_arms import PUBLISHED_ARM_CLASSES
        pub_cls = PUBLISHED_ARM_CLASSES[solver_name]
        if solver_name == "linear_bo_box8":
            return pub_cls(black_box=black_box, x0=x0, y0=y0)
        codec_class_name = type(codec).__name__ if codec is not None else ""
        if "Chembed" in codec_class_name:
            pub_bounds = "results/precomputed/chembed_per_coord_bounds.json"
        elif "SpCauchy" in codec_class_name:
            pub_bounds = "results/precomputed/spcauchy_per_coord_bounds.json"
        elif "PowerSpherical" in codec_class_name:
            pub_bounds = "results/precomputed/power_spherical_per_coord_bounds.json"
        elif "VMF" in codec_class_name or "Vmf" in codec_class_name:
            pub_bounds = "results/precomputed/vmf_per_coord_bounds.json"
        else:
            pub_bounds = None  # solver default = SELFIES bounds
        return pub_cls(black_box=black_box, x0=x0, y0=y0, bounds_path=pub_bounds)

    # Baselines of 2026-11-02. Kept here, after every other branch: OUR_SOLVERS is pinned by
    # tests/test_published_block.py and appendixH_facts.csv cites line numbers above this one.
    if solver_name in ("cmaes_full_rd", "cmaes_full_rd_pop5"):
        # x0 = best cold start, sigma0 = 0.5, no bounds; population = pycma's default for D
        # (20 at D = 256), or 5 for the _pop5 arm, the population GeoCMA uses.
        from src.spherevo.cmaes_full_rd import CMAES_FULL_RD_POP5, CMAESFullSolver

        return CMAESFullSolver(
            black_box=black_box, x0=x0, y0=y0, seed=seed, initial_sigma=0.5,
            population_size=(CMAES_FULL_RD_POP5 if solver_name.endswith("pop5") else None),
        )

    if solver_name == "baxus_cr":
        # The split schedule and failure tolerance derive from the proposal budget.
        if n_iterations is None:
            raise ValueError("baxus_cr needs n_iterations (its schedule comes from the budget)")
        from src.spherevo.published_arms import PUBLISHED_ARM_CLASSES

        return PUBLISHED_ARM_CLASSES["baxus_cr"](
            black_box=black_box, x0=x0, y0=y0, seed=seed, eval_budget=int(n_iterations),
        )

    raise ValueError(
        f"Unknown solver: {solver_name}. Supported: {SUPPORTED_SOLVERS}"
    )


# ---------------------------------------------------------------------------
# Main benchmark runner
# ---------------------------------------------------------------------------


def _surrogate_fallbacks(solver) -> tuple[int, list[str]]:
    """Fallback steps a solver recorded, whichever of the two counters it uses.

    The GP competitors (``src/spherevo/gp_competitors.py``, TuRBO, MSR) count
    in ``_fallback_count`` / ``_fallback_reasons``. The GaBO family
    (``riemannian_bo._RiemannianBOBase``: GaBO, Matern-GaBO, HD-GaBO), REMBO,
    HeSBO and the GP-guided spherical solver count in ``gp_fallback_count`` /
    ``gp_fallback_reasons``. No solver class uses both, so the sum never
    double counts. Until the camera-ready successor commit only the first pair
    reached the result file, so GaBO, Matern-GaBO, REMBO and HeSBO always
    reported ``surrogate_fallback_count = 0`` whatever happened. Reading the
    counters changes no computation.
    """
    count = int(getattr(solver, "_fallback_count", 0)) + int(
        getattr(solver, "gp_fallback_count", 0)
    )
    reasons = list(getattr(solver, "_fallback_reasons", [])) + list(
        getattr(solver, "gp_fallback_reasons", [])
    )
    return count, reasons


def run_benchmark(
    mode: str,
    solver_name: str,
    task_name: str,
    seed: int,
    max_iter: int,
    n_initial_points: int,
    n_cold_start: int,
    latent_dim: int,
    subspace_dim: int,
    n_subspaces: int,
    cold_start_source: str = "zinc",
    decoder_weights_path: str | None = None,
    norm_mode: str = "fixed",
    codec_type: str = "selfies",
) -> dict:
    """Run a single benchmark configuration."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    codec = None

    cs_smiles: list[str] | None = None
    if mode == "hdbo":
        bb, vae_or_codec, dim = make_hdbo_latent_bb(task_name, latent_dim)
        z0 = generate_cold_start_hdbo(dim, n_initial_points, seed)
        y0 = bb(z0)
        n_init = n_initial_points
        logger.info(
            f"[HDBO] solver={solver_name} task={task_name} dim={dim} "
            f"seed={seed} cold_start best={np.nanmax(y0):.4f}"
        )
    elif cold_start_source == "csv":
        bb, codec, dim = make_our_latent_bb(task_name, device, decoder_weights_path, codec_type=codec_type)
        z0, y0, cs_smiles = generate_cold_start_csv(
            codec, task_name, n_cold_start, device
        )
        n_init = n_cold_start
        our_task = POLI_TO_GUACAMOL.get(task_name, task_name)
        logger.info(
            f"[Ours/CSV] solver={solver_name} task={task_name} ({our_task}) "
            f"dim={dim} seed={seed} cold_start best={np.nanmax(y0):.4f} "
            f"(CSV cold start, same as external baselines)"
        )
    else:
        bb, codec, dim = make_our_latent_bb(task_name, device, decoder_weights_path, codec_type=codec_type)
        our_task = POLI_TO_GUACAMOL.get(task_name, task_name)
        from src.oracle.guacamol import GuacaMolOracle
        _oracle = GuacaMolOracle(task_id=our_task)
        z0, y0, cs_smiles = generate_cold_start_ours(
            codec, _oracle.score, n_cold_start, seed, device
        )
        n_init = n_cold_start
        logger.info(
            f"[Ours] solver={solver_name} task={task_name} ({our_task}) "
            f"dim={dim} seed={seed} cold_start best={np.nanmax(y0):.4f}"
        )

    solver = _create_solver(
        solver_name=solver_name,
        black_box=bb,
        x0=z0,
        y0=y0,
        n_dimensions=dim,
        subspace_dim=subspace_dim,
        n_subspaces=n_subspaces,
        seed=seed,
        n_initial_points=n_init,
        codec=codec,
        norm_mode=norm_mode,
        n_iterations=max_iter,
        cs_smiles=cs_smiles,
    )

    t0 = time.time()
    status = "completed"
    crash_info = None
    # poli's CMA_ES.step() consumes population_size oracle calls per BO
    # iteration, not 1. To enforce the same total oracle budget as every
    # other solver in the comparison (5 cold-start + 25 iterations = 30
    # evaluations), pass an effective max_iter that yields ~25 inner-loop
    # evaluations: ceil(max_iter / population_size). With population_size=10
    # and max_iter=25 this gives 3 BO iterations -> 30 oracle calls plus 5
    # cold-start = 35 (slight overshoot, the last 5 are dropped at scoring).
    # Without this, CMA-ES would run 250 oracle calls and dominate purely
    # on budget mismatch.
    effective_max_iter = max_iter
    if solver_name in ("cma_es", "cma_es_sphere"):
        try:
            pop_size = int(getattr(solver, "population_size", 10))
        except (TypeError, ValueError):
            pop_size = 10
        effective_max_iter = max(1, math.ceil(max_iter / pop_size))
        logger.info(
            f"[budget-fairness] {solver_name} pop_size={pop_size}: "
            f"max_iter {max_iter} -> {effective_max_iter} BO iterations "
            f"({effective_max_iter * pop_size} oracle calls expected)"
        )
    try:
        solver.solve(max_iter=effective_max_iter, verbose=True)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:
        logger.error(
            f"Solver crashed: {type(e).__name__}: {e}",
            exc_info=True,
        )
        status = "crashed"
        crash_info = f"{type(e).__name__}: {e}"
    elapsed = time.time() - t0

    try:
        best_y = solver.get_best_performance()
        _, y_hist = solver.get_history_as_arrays(penalize_nans_with=-1.0)
        best_trace = np.maximum.accumulate(y_hist.flatten())
    except Exception as e2:
        logger.error(f"Failed to extract results after {status}: {e2}")
        best_y = np.array([[float("nan")]])
        y_hist = np.array([[float("nan")]])
        best_trace = np.array([float("nan")])

    # Extract norm_trace if available (CMAESSubspaceSolver with adaptive mode)
    norm_trace = getattr(solver, "_norm_trace", None)
    fallback_count, fallback_reasons = _surrogate_fallbacks(solver)

    results = {
        "mode": mode,
        "solver": solver_name,
        "task": task_name,
        "our_task_id": POLI_TO_GUACAMOL.get(task_name, task_name),
        "latent_dim": dim,
        "subspace_dim": subspace_dim,
        "n_subspaces": n_subspaces,
        "norm_mode": norm_mode,
        "seed": seed,
        "max_iter": max_iter,
        "n_initial_points": n_init,
        "n_evaluated": len(y_hist),
        "best_score": float(best_y.flatten()[0]),
        "best_trace": best_trace.tolist(),
        "all_scores": y_hist.flatten().tolist(),
        "norm_trace": norm_trace if norm_trace else None,
        "n_oracle_failures": bb.n_failures,
        "surrogate_fallback_count": fallback_count,
        "surrogate_fallback_reasons": fallback_reasons,
        "status": status,
        "crash_info": crash_info,
        "elapsed_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
        # published-recipe arms only (absent for every other solver): settings + per-step record
        **({"published_recipe": solver.published_recipe_record}
           if hasattr(solver, "published_recipe_record") else {}),
        **collect_provenance(),
    }

    bb.terminate()
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark GP-free spherical subspace BO against poli-baselines"
    )
    parser.add_argument(
        "--mode",
        choices=["hdbo", "ours"],
        required=True,
        help="hdbo = HDBO's MLP VAE 128D; ours = our Transformer VAE 256D",
    )
    parser.add_argument(
        "--solver",
        type=str,
        default="gp_free_spherical",
        help=f"Solver name. Options: {SUPPORTED_SOLVERS}",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="amlodipine_mpo",
        help='PMO task name, or "all" for all 25 PMO, or "ours" for our 12 tasks',
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--n-initial-points", type=int, default=10)
    parser.add_argument("--n-cold-start", type=int, default=5)
    parser.add_argument(
        "--codec",
        choices=["selfies", "spcauchy", "chembed", "power_spherical", "vmf"],
        default="selfies",
        help="Molecular VAE codec (selfies=LOL-BO Transformer 256D, spcauchy=SpCauchy Transformer 256D, chembed=Chembed PubChem-trained Transformer 256D)",
    )
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--subspace-dim", type=int, default=16)
    parser.add_argument("--n-subspaces", type=int, default=16)
    parser.add_argument(
        "--cold-start-source",
        choices=["zinc", "csv"],
        default="zinc",
        help="Cold start source: zinc=random ZINC (default), csv=GuacaMol CSV (same as external baselines)",
    )
    parser.add_argument(
        "--decoder-weights",
        type=str,
        default=None,
        help="Path to fine-tuned decoder weights (subspace-denoised). "
        "Only used in --mode ours. Adds 'ft' tag to result filenames.",
    )
    parser.add_argument(
        "--norm-mode",
        choices=["fixed", "adaptive"],
        default="fixed",
        help="Norm mode for CMA-ES subspace: fixed=mean_norm (default), "
        "adaptive=co-optimise log(norm) as (d+1)-th CMA-ES dimension.",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Total eval budget. Overrides max-iter = budget - n-cold-start. "
        "Adds _b{N} suffix to filenames for N != 30.",
    )

    args = parser.parse_args()

    # Early solver validation (before expensive GPU init)
    if args.solver not in SUPPORTED_SOLVERS:
        parser.error(
            f"Unknown solver: {args.solver}. "
            f"Supported: {', '.join(SUPPORTED_SOLVERS)}"
        )

    # Budget override
    if args.budget is not None:
        args.max_iter = args.budget - args.n_cold_start
    budget = args.budget if args.budget else args.n_cold_start + args.max_iter

    # Resolve task list
    if args.task == "all":
        tasks = HDBO_PMO_TASKS
    elif args.task == "ours":
        tasks = OUR_TASKS_POLI
    else:
        # Accept both our short names and poli names
        if args.task in GUACAMOL_TO_POLI:
            tasks = [GUACAMOL_TO_POLI[args.task]]
        else:
            tasks = [args.task]

    out_dir = _results_dir_for(args.solver)
    # Codec-segregated result roots (SSOT in src/analysis/config.py).
    if args.codec in CODEC_RESULT_ROOTS:
        out_dir = PROJECT_ROOT / CODEC_RESULT_ROOTS[args.codec] / args.solver
    out_dir.mkdir(parents=True, exist_ok=True)

    cs_tag = "_csv" if args.cold_start_source == "csv" else ""
    ft_tag = "_ft" if args.decoder_weights else ""
    codec_tag = f"_{args.codec}" if args.codec != "selfies" else ""
    budget_tag = f"_b{budget}" if budget != 30 else ""
    sd_tag = f"_sd{args.subspace_dim}" if args.subspace_dim != 16 else ""
    na_tag = "_na" if args.norm_mode == "adaptive" else ""

    for task_name in tasks:
        result_path = (
            out_dir
            / f"{args.mode}{cs_tag}{ft_tag}{codec_tag}_{args.solver}_{task_name}_s{args.seed}_{args.latent_dim}d{budget_tag}{sd_tag}{na_tag}.json"
        )
        if result_path.exists():
            logger.info(f"Skipping {task_name} seed={args.seed} (exists)")
            continue

        try:
            results = run_benchmark(
                mode=args.mode,
                solver_name=args.solver,
                task_name=task_name,
                seed=args.seed,
                max_iter=args.max_iter,
                n_initial_points=args.n_initial_points,
                n_cold_start=args.n_cold_start,
                latent_dim=args.latent_dim,
                subspace_dim=args.subspace_dim,
                n_subspaces=args.n_subspaces,
                cold_start_source=args.cold_start_source,
                decoder_weights_path=args.decoder_weights,
                norm_mode=args.norm_mode,
                codec_type=args.codec,
            )

            results["budget"] = budget

            with open(result_path, "w") as fp:
                json.dump(results, fp, indent=2)

            logger.info(
                f"DONE {task_name}: best={results['best_score']:.4f} "
                f"n_eval={results['n_evaluated']} time={results['elapsed_seconds']:.1f}s "
                f"-> {result_path.name}"
            )
        except Exception as e:
            logger.error(f"FAILED {task_name}: {e}", exc_info=True)


if __name__ == "__main__":
    main()
