"""Single source of truth for benchmark configuration.

All benchmark constants (tasks, seeds, paths, evaluation parameters) live here.
Always import from this module -- do not hardcode in scripts.
"""

from __future__ import annotations

import re

# All 12 hard GuacaMol MPO tasks
BENCHMARK_TASKS = [
    "adip", "dhop", "fexo", "med1", "med2", "osmb",
    "pdop", "rano", "shop", "siga", "valt", "zale",
]

# Task-name mapping: our short codes <-> poli identifiers (SSOT)
# Note: poli spells osimertinib as "osimetrinib" (known typo).
GUACAMOL_TO_POLI = {
    "adip": "amlodipine_mpo",
    "dhop": "deco_hop",
    "fexo": "fexofenadine_mpo",
    "med1": "median_1",
    "med2": "median_2",
    "osmb": "osimetrinib_mpo",
    "pdop": "perindopril_mpo",
    "rano": "ranolazine_mpo",
    "shop": "scaffold_hop",
    "siga": "sitagliptin_mpo",
    "valt": "valsartan_smarts",
    "zale": "zaleplon_mpo",
}

POLI_TO_GUACAMOL = {v: k for k, v in GUACAMOL_TO_POLI.items()}

# Human-readable display names for plots/tables
TASK_DISPLAY_NAMES = {
    "amlodipine_mpo": "Amlodipine",
    "deco_hop": "Deco Hop",
    "fexofenadine_mpo": "Fexofenadine",
    "median_1": "Median 1",
    "median_2": "Median 2",
    "osimetrinib_mpo": "Osimertinib",
    "perindopril_mpo": "Perindopril",
    "ranolazine_mpo": "Ranolazine",
    "scaffold_hop": "Scaffold Hop",
    "sitagliptin_mpo": "Sitagliptin",
    "valsartan_smarts": "Valsartan",
    "zaleplon_mpo": "Zaleplon",
}

# Standard seed range for main benchmark (40 seeds: 42-81)
BENCHMARK_SEEDS = list(range(42, 82))

# Smaller seed range for ablations and diagnostics (10 seeds: 42-51)
ABLATION_SEEDS = list(range(42, 52))

# Result directory paths (relative to NeurIPS project root)
RESULT_DIRS = {
    "guacamol": "results/guacamol",
    "spherevo": "results/guacamol/spherevo",
    "geowalk": "results/guacamol/geowalk",
    "geowalk_ablations": "results/guacamol/geowalk_ablations",
    "ablations": "results/guacamol/ablations",
    "turbo": "results/guacamol/turbo",
    "baxus": "results/guacamol/baxus",
    "lolbo": "results/guacamol/lolbo",
    "invbo": "results/guacamol/invbo",
    "cmaes_full": "results/guacamol/cmaes_full",
    "random_search": "results/guacamol/random_search",
    "nfbo": "results/guacamol/nfbo",
    "lassobench": "results/lassobench",
    "precomputed": "results/precomputed",
}

# Small benchmark parameters (5 cold start + 25 iterations = 30 evaluations)
SMALL_N_COLD_START = 5
SMALL_N_ITERATIONS = 25
SMALL_N_EVALUATED = 30  # expected total
SMALL_N_EVALUATED_RANGE = (25, 35)  # tolerance for filtering incomplete runs

# Filename parsing patterns
SEED_REGEX = re.compile(r"_s(\d+)_")


def extract_seed(filename: str) -> int | None:
    """Extract seed from a result filename like 'method_task_s42_20260215.json'."""
    m = SEED_REGEX.search(filename)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Synthetic embedded benchmark (Branin/Hartmann in high-D)
# ---------------------------------------------------------------------------

SYNTHETIC_TASKS = {
    "branin_100d": {"function": "branin", "ambient_dim": 100, "effective_dim": 2},
    "branin_500d": {"function": "branin", "ambient_dim": 500, "effective_dim": 2},
    "hartmann6_100d": {"function": "hartmann6", "ambient_dim": 100, "effective_dim": 6},
    "hartmann6_500d": {"function": "hartmann6", "ambient_dim": 500, "effective_dim": 6},
}

SYNTHETIC_METHODS = ["cmaes_sub", "cmaes_full", "random", "turbo", "baxus", "saasbo"]

SYNTHETIC_DISPLAY_NAMES = {
    "branin_100d": "Branin-100D",
    "branin_500d": "Branin-500D",
    "hartmann6_100d": "Hartmann6-100D",
    "hartmann6_500d": "Hartmann6-500D",
}

# ---------------------------------------------------------------------------
# Design-Bench (TFBind8, 32D continuous relaxation)
# ---------------------------------------------------------------------------

DESIGNBENCH_TASKS = {
    "tfbind8": {"dim": 32, "oracle": "TFBind8-Exact-v0"},
}

DESIGNBENCH_METHODS = ["cmaes_sub", "cmaes_full", "random", "turbo", "gp_bo"]

DESIGNBENCH_DISPLAY_NAMES = {
    "tfbind8": "TFBind8",
}

# ---------------------------------------------------------------------------
# NAS-Bench-101 via arch2vec (16D Graph VAE)
# ---------------------------------------------------------------------------

NASBENCH_TASKS = {
    "cifar10": {"latent_dim": 16, "dataset": "cifar10"},
}

NASBENCH_METHODS = ["cmaes_sub", "cmaes_full", "random", "turbo", "gp_bo"]

NASBENCH_DISPLAY_NAMES = {
    "cifar10": "NAS-101 CIFAR-10",
}

# ---------------------------------------------------------------------------
# LassoBench tasks (cross-domain validation)
# ---------------------------------------------------------------------------

LASSOBENCH_TASKS = {
    "synt_simple": 60,
    "synt_medium": 100,
    "synt_high": 300,
    "synt_hard": 1000,
    "dna": 180,
    "leukemia": 7129,
}

LASSOBENCH_METHODS = [
    "cmaes_sub", "cmaes_full", "random", "gp_bo", "turbo", "baxus",
    "lppfn_sub",
]

# Extend result directories
RESULT_DIRS.update({
    "synthetic": "results/synthetic",
    "designbench": "results/designbench",
    "nasbench": "results/nasbench",
    # Riemannian BO baselines (Jaquier et al.) — see src/spherevo/riemannian_bo.py
    "gabo": "results/guacamol/gabo",
    "hd_gabo_2": "results/guacamol/hd_gabo_2",
    "hd_gabo_16": "results/guacamol/hd_gabo_16",
    "matern_gabo": "results/guacamol/matern_gabo",
})

# Codec-segregated result roots: Chembed and SpCauchy results live in dedicated
# subdirectories so they cannot collide with SELFIES VAE results that share the
# filename pattern. Read by `scripts/run_benchmark.py::_codec_results_root`.
CODEC_RESULT_ROOTS = {
    "chembed": "results/guacamol/chembed",
    "spcauchy": "results/guacamol/spcauchy_40seed",
    "power_spherical": "results/guacamol/power_spherical",
    "vmf": "results/guacamol/vmf",
}

# Budget-scaling experiment constants
BUDGET_SCALING_BUDGETS = [30, 50, 100, 200]
BUDGET_SCALING_METHODS = ["cma_es_subspace", "turbo", "random_search"]


# ---------------------------------------------------------------------------
# VAE manifold geometry (precomputed from full ZINC-250K encoding)
# Source: results/precomputed/zinc_mean_norm.json -> per_coord_std_mean
# Recompute via: PYTHONPATH=. python scripts/analysis/zinc_mean_norm.py
# ---------------------------------------------------------------------------

# Mean per-coordinate standard deviation across all 249,456 ZINC-250K molecules
# encoded through the SELFIES Transformer VAE (D=256). Used to size the box
# bounds for original-recipe Hvarfner/Papenmeier/Xu acquisition (Sobol+L-BFGS).
ZINC_PER_COORD_STD = 0.2937

# Number of standard deviations covered by the box half-width per coordinate.
# Hvarfner et al. 2024 use [0,1]^D for unit-variance functions; we apply the
# same 3-sigma per-coordinate coverage rule, isotropic across all D dims.
BOX_SIGMA_MULTIPLIER = 3

# Resulting half-width of the box used by *BoxSolver acquisition variants:
# bounds = [-BOX_HALF_WIDTH, +BOX_HALF_WIDTH]^D ~= [-0.881, +0.881]^256
BOX_HALF_WIDTH = BOX_SIGMA_MULTIPLIER * ZINC_PER_COORD_STD


# ---------------------------------------------------------------------------
# Per-coordinate ZINC bounds for original-recipe [0,1]^D normalized acquisition
# (faithful reproduction of Hvarfner/Papenmeier/Xu protocol with input normalize)
# Source: results/precomputed/zinc_per_coord_bounds.json -> per_coord.{p01,p99}
# Recompute via: PYTHONPATH=. python scripts/analysis/zinc_per_coord_minmax.py
# ---------------------------------------------------------------------------

ZINC_PER_COORD_BOUNDS_PATH = "results/precomputed/zinc_per_coord_bounds.json"

# Empirical latent shell of the SELFIES VAE (ZINC-250K encoder means, D=256).
# r̄ = mean latent norm; 5-95th percentile band. Used by radius-robustness
# ablations (reviewers Lhbx Q2/Q3, 6qvY W2). SSOT — do not hardcode elsewhere.
EMPIRICAL_SHELL_MEAN_NORM = 7.98
EMPIRICAL_SHELL_PCT_5_95 = (6.81, 9.14)

# Cached sample of encoded ZINC latents for the convex-hull / ellipsoid
# alternative-support baselines (reviewer Lhbx Q4). Built by
# scripts/analysis/precompute_zinc_latents.py.
ZINC_LATENTS_SAMPLE_PATH = "results/precomputed/zinc_latents_sample.npz"
