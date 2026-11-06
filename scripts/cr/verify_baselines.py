#!/usr/bin/env python
"""Verify a published baseline in its AUTHORS' setting, through the same code the arms run.

The builders of ``src/spherevo/published_recipes.py`` and the optimisers of
``src/spherevo/published_external.py`` (the code the GuacaMol arms and the LassoBench
``*_published`` methods run) are run here on the authors' benchmark, initial design and budget,
and compared with the published numbers of ``scripts/cr/baseline_targets.json`` (a byte copy of
the baseline audit's file, sha256 recorded in every output). Rule: ``docs``/RUN_RULES amendment
"Published-recipe block and baseline print rule", point (2).

One process per (recipe, benchmark, seed). The JSON written after EVERY evaluation holds the
per-evaluation values and the best-so-far trace, each step's fallback/events/timings, the
settings and the provenance (commit, dirty flag, package versions, host, GPU). GP recipes and the
random control resume from an ``.npz`` checkpoint (X and RNG states; ``resumed_at`` records each
resume); TuRBO, BAxUS and CMA-ES restart from scratch (their optimiser state is not checkpointed).

Recipes      hvarfner, msr, mle_scaled, xu, doumont   published_recipes.py (unit-cube builders)
             turbo_published, baxus, cmaes            published_external.py (released code)
             turbo                                    the repository's turbo_box_faithful TuRBO-1
             random                                   uniform draws (objective-wiring control)
Benchmarks   lasso_dna, lasso_synt_{simple,medium,high,hard}   LassoBench, x in [-1,1]^D, min loss
             guacamol:<code>   the repository's SELFIES VAE (``--vae-weights`` loads another state
                               dict, e.g. Doumont's AABO checkpoint) + GuacaMol oracle in [-8,8]^256
             ackley20          BoTorch TuRBO tutorial: Ackley-20 on [-5,10]^20, maximise -f

Usage (repository root):

    python scripts/cr/verify_baselines.py run --recipe hvarfner --benchmark lasso_dna \\
        --seed 0 --budget 1000 --out-dir results/verify_baselines
    python scripts/cr/verify_baselines.py summarize --out-dir results/verify_baselines \\
        --write results/verify_baselines/verdict.json

Pass rule (``summarize``), per target and per listed evaluation count n (plot x; compared at
n + x_eval_offset evaluations, skipped inside the initial design): our statistic over seeds
0 .. n_reps-1, computed as the target states it (mean of best-so-far, or LassoBench's running
minimum of the across-repetition mean of per-evaluation losses, compared at ``x_logged`` when
given) must lie within 2 combined standard errors of an exact value (table, released file, or a
vector-PDF mean whose band is a stated SE: ``vector_figure_exact``), 2·sqrt(SE_ours² + SE_pub²),
or within ``read_unc`` + 2·sqrt(SE_ours² + SE_pub²) of a figure reading, where SE_pub is the stated
SE and, when the source states none, SE_pub := SE_ours (``target_se_from:
"assumed_equal_to_ours"``; round 2, R2-3). On a LassoBench objective every tolerance is increased by
δ_b·|target| (objective-version allowance of the targets file, round 2, R2-4). Each row and verdict
also carries ``pass_strict``, the previous rule (δ = 0, SE_pub = 0 where the band is unstated). A recipe passes on a target only if every
gating row passes and all n_reps seeds are complete. Targets or rows with ``gate: false`` are
informative (computed, never a pass or fail).

``summarize`` refuses a verdict (``pass: None`` with the reasons in ``refusals``; rows are still
computed) when the runs of one (recipe, benchmark, n_init) differ in commit, are dirty or carry no
commit, when the targets file is not the shipped one (sha256), or when n_init / initial design /
budget / batch size / m_D differ from the target's protocol. Runs are grouped by
(recipe, benchmark, n_init), so the 30- and 41-point Hvarfner designs give two verdicts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

GP_RECIPES = ("hvarfner", "msr", "mle_scaled", "xu", "doumont")
#: MSR / MLE (scaled) with the noise LEARNED (understanding-hdbo ``--no-noiseless``), harness only:
#: the release's docstrings say real-world benchmarks used a learned noise while
#: ``scripts/reproduce_fig8.sh`` runs the default (frozen 1e-4); the verification decides.
NOISE_LEARNED = {"msr_noise_learned": True, "mle_scaled_noise_learned": False}
EXTERNAL = ("turbo_published", "baxus", "cmaes")
RECIPES = GP_RECIPES + tuple(NOISE_LEARNED) + EXTERNAL + ("turbo", "random")
TARGETS_FILE = REPO_ROOT / "scripts" / "cr" / "baseline_targets.json"
#: sha256 of the shipped targets file: the audit's file (a7d96029...aadc) with the pre-run corrections
#: of 2026-10-13 (its "amendments" list). Round 1 (FIXLIST_round1 A1-A6): Xu x_eval_offset 1, vector
#: means (and SE where the band is a stated SE) for Hvarfner / Papenmeier / Doumont / BAxUS, plot-x
#: offsets, CMA-ES DNA and x = 1 rows informative, sd_at_argmin for the LassoBench files.
#: Round 2 (orchestrator, part 1): R2-2 the four noisy synthetic CMA-ES entries informative; R2-3 the
#: acceptance rule for figure readings (conventions); R2-4 the objective-version allowance; R2-8 Xu Fig. 6
#: band = stated SD (SE_pub = half-width / sqrt(10) where separable); R2-9 Doumont Fig. 13 from the vector page;
#: R2-11 CMA-ES start-point rows (released argmin at x = 1) informative.
TARGETS_SHA256 = "8b0507645cf5141613190ec502a2d4e7a3f4587ef39c6782e44e8b48da47bfc1"
#: every earlier shipped sha256, oldest first (each is the previous_sha256 of an amendments entry)
TARGETS_SHA256_HISTORY = (
    "a7d960294ad785d935221a4df04456ffc96abaa3c38deb19cd017780ebe8aadc",   # the audit's file
    "c1c9031ab86418df1ac8419e9f57c0c8444d2a9cb3fee43272e0e57d66124601",   # pre-run corrections
    "272fc8d1ecf6059a4661475cfeb84b3e0b96f09eb005d74126800602b14c2baf",   # round 1 (A1-A6)
    "57083bf841ec8267291a19870ce82c79d1f6d51ab637a2780d9d0b42ef74a395",   # round 2, R2-2
    "e56e2c5d8ffa0a8a7965fb42fdb66ba1178517287ad7e9d1c5ee7b9f3eb122a9",   # round 2, R2-3
    "b190322f138584c76fd45b4a352cb0ec625c088ad74a5ada3ae04b75736f63b0",   # round 2, R2-4
    "b06f72305610ae3e335d6eecf41c94edf5ed9d175d9c169080e007d0aa2c3473",   # round 2, R2-8
    "5d72465bed09b703db9756aef7fb6cde97597a5c0f7b511e227ce51088f479a6",   # round 2, R2-9
)
TARGETS_SHA256_PREVIOUS = TARGETS_SHA256_HISTORY[-1]
TARGETS_SHA256_AUDIT_ORIGINAL = TARGETS_SHA256_HISTORY[0]
GUACAMOL_INVALID_SCORE = -0.01   # linear-bo guacamol_task.py:646-648
LASSO_SYNTH = ("synt_simple", "synt_medium", "synt_high", "synt_hard")
EXACT_KINDS = ("table_exact", "released_file_exact", "computed_from_release", "vector_figure_exact")
#: BAxUS Fig. 3 (arXiv 2304.11468, Sec. 4): m_D = 1000 with a budget of 2000 (round-1 fix B5)
BAXUS_M_D_DEFAULT = 1000
#: initial design type the harness generates per recipe (default_init), checked by summarize
EXPECTED_INIT_TYPE = {"hvarfner": "sobol", "doumont": "sobol", "msr": "sobol", "mle_scaled": "sobol",
                      "msr_noise_learned": "sobol", "mle_scaled_noise_learned": "sobol",
                      "xu": "lhs", "turbo": "sobol", "turbo_published": "turbo_lhs",
                      "baxus": "baxus_sobol", "cmaes": "none", "random": "none"}
#: Hvarfner Lasso-DNA: the paper text and the figure use 30 Sobol points, the released code
#: ceil(3 sqrt(D)) = 41 (conf.yaml:10-11, main.py:54-56); both designs are run (orchestrator, E)
HVARFNER_LASSO_N_INIT = (30, 41)


def sha256_file(path: Path) -> Optional[str]:
    try:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    except OSError:
        return None


def default_target_id(recipe: str, bench: str) -> Optional[str]:
    fam, _, rest = bench.partition(":")
    syn = bench[len("lasso_"):] if bench.startswith("lasso_synt_") else None
    table = {
        ("hvarfner", "lasso_dna"): "hvarfner_dsp__lasso_dna__fig7",
        ("msr", "lasso_dna"): "papenmeier_msr__lasso_dna__fig8",
        ("mle_scaled", "lasso_dna"): "papenmeier_mle_scaled__lasso_dna__fig8",
        ("msr_noise_learned", "lasso_dna"): "papenmeier_msr__lasso_dna__fig8",
        ("mle_scaled_noise_learned", "lasso_dna"): "papenmeier_mle_scaled__lasso_dna__fig8",
        ("xu", "lasso_dna"): "xu_sbo_matern_ri__lasso_dna__fig6",
        ("doumont", "lasso_dna"): "doumont_linear__lasso_dna__fig2",
        ("doumont", "guacamol"): "doumont_linear__guacamol_aabo__fig13",
        ("hvarfner", "guacamol"): "hvarfner_dsp__guacamol_aabo__doumont_fig13",
        ("turbo_published", "lasso_dna"): "lassobench_pkl__turbo__dna__real",
        ("cmaes", "lasso_dna"): "lassobench_pkl__cma-es__dna__real",
        ("random", "lasso_dna"): "lassobench_pkl__random_search__dna__real",
        ("baxus", "lasso_synt_hard"): "baxus__lasso_hard__fig3",
        ("baxus", "lasso_synt_high"): "baxus__lasso_high__fig3",
        # ("turbo", "ackley20") has no target in the shipped file (the BoTorch-tutorial trajectory is
        # only in baseline_targets.example.json): no mapping, summarize reports "no target" (R2-10)
    }
    if (recipe, fam) in table:
        return table[(recipe, fam)]
    if (recipe, bench) in table:
        return table[(recipe, bench)]
    if syn is not None:
        name = {"turbo_published": "turbo", "cmaes": "cma-es", "random": "random_search"}.get(recipe)
        if name:
            return f"lassobench_pkl__{name}__{syn}__noiseless"
    return None


# ---------------------------------------------------------------------------
# Benchmarks: unit cube -> objective value in the benchmark's own sense
# ---------------------------------------------------------------------------


class Benchmark:
    name: str
    dim: int
    sense: str
    lb: np.ndarray
    ub: np.ndarray

    def value(self, x_raw: np.ndarray) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def to_raw(self, u: np.ndarray) -> np.ndarray:
        return self.lb + np.asarray(u, dtype=np.float64).reshape(-1) * (self.ub - self.lb)

    def to_max(self, v: float) -> float:
        return -v if self.sense == "min" else v


class Lasso(Benchmark):
    def __init__(self, which: str):
        import LassoBench
        if which == "dna":
            self._bench = LassoBench.RealBenchmark(pick_data="dna")
        else:
            self._bench = LassoBench.SyntheticBenchmark(pick_bench=which, noise=False)
        self.name, self.sense = f"lasso_{which}", "min"
        self.dim = int(self._bench.n_features)
        self.lb, self.ub = -np.ones(self.dim), np.ones(self.dim)

    def value(self, x_raw):
        return float(self._bench.evaluate(np.clip(x_raw, -1.0, 1.0)))


class GuacaMolBox8(Benchmark):
    def __init__(self, code: str, device: str, vae_weights: Optional[str] = None):
        from scripts.run_benchmark import make_our_latent_bb
        from src.analysis.config import GUACAMOL_TO_POLI
        self._bb, _, self.dim = make_our_latent_bb(GUACAMOL_TO_POLI[code], device=device,
                                                   decoder_weights_path=vae_weights)
        self.name, self.sense = f"guacamol:{code}", "max"
        self.lb, self.ub = -8.0 * np.ones(self.dim), 8.0 * np.ones(self.dim)

    def value(self, x_raw):
        v = float(np.asarray(self._bb(np.asarray(x_raw, dtype=np.float64).reshape(1, -1))).item())
        return v if math.isfinite(v) else GUACAMOL_INVALID_SCORE


class Ackley20(Benchmark):
    def __init__(self):
        import torch
        from botorch.test_functions import Ackley
        self._f = Ackley(dim=20, negate=True).to(dtype=torch.float64)
        self.name, self.sense, self.dim = "ackley20", "max", 20
        self.lb, self.ub = -5.0 * np.ones(20), 10.0 * np.ones(20)

    def value(self, x_raw):
        import torch
        return float(self._f(torch.tensor(np.asarray(x_raw).reshape(1, -1))).item())


def make_benchmark(name: str, device: str, vae_weights: Optional[str] = None) -> Benchmark:
    if name == "lasso_dna":
        return Lasso("dna")
    if name.startswith("lasso_") and name[len("lasso_"):] in LASSO_SYNTH:
        return Lasso(name[len("lasso_"):])
    if name.startswith("guacamol:"):
        return GuacaMolBox8(name.split(":", 1)[1], device, vae_weights)
    if name == "ackley20":
        return Ackley20()
    raise SystemExit(f"unknown benchmark {name!r}")


# ---------------------------------------------------------------------------
# Initial designs (the authors')
# ---------------------------------------------------------------------------


def default_init(recipe: str, dim: int, bench: str = "") -> tuple[int, str]:
    if recipe in ("hvarfner", "doumont"):
        return 30, "sobol"                                     # papers: 30 Sobol
    if recipe in ("msr", "mle_scaled", *NOISE_LEARNED):
        return int(math.floor(3.0 * math.sqrt(dim))), "sobol"  # bo.py:125-126
    if recipe == "xu":
        return 20, "lhs"                                       # run_script.py:76-79
    if recipe == "turbo":
        return 2 * dim, "sobol"                                # BoTorch tutorial
    if recipe == "turbo_published":                           # LassoBench: 0.1 d LHS / 100 real
        return (100 if bench == "lasso_dna" else max(1, int(0.1 * dim))), "turbo_lhs"
    if recipe == "baxus":
        return 10, "baxus_sobol"                               # BAxUS Fig. 3: 10 initial points
    return 0, "none"                                           # cmaes (x0 = 0), random


def initial_design(n: int, dim: int, kind: str, seed: int) -> np.ndarray:
    if n <= 0:
        return np.zeros((0, dim))
    if kind == "sobol":
        from torch.quasirandom import SobolEngine
        return SobolEngine(dimension=dim, scramble=True, seed=seed).draw(n).double().numpy()
    if kind == "lhs":
        from scipy.stats import qmc
        return qmc.LatinHypercube(d=dim, seed=seed).random(n)
    raise SystemExit(f"initial design {kind!r} is generated by the optimiser itself")


# ---------------------------------------------------------------------------
# Provenance and checkpointing
# ---------------------------------------------------------------------------


def provenance(device: str) -> dict[str, Any]:
    import importlib.metadata as md

    import torch
    from src.provenance import collect_provenance

    prov = collect_provenance()["provenance"]
    prov["git_commit_full"] = git_commit_full()
    for pkg in ("botorch", "gpytorch", "linear_operator", "LassoBench", "cma", "scikit-learn"):
        try:
            prov.setdefault("package_versions", {})[pkg] = md.version(pkg)
        except Exception:  # noqa: BLE001
            pass
    prov["device"] = device
    prov["gpu"] = torch.cuda.get_device_name(0) if device.startswith("cuda") else None
    prov["hostname"] = socket.gethostname()
    prov["platform"] = platform.platform()
    prov["targets_sha256"] = sha256_file(TARGETS_FILE)
    return prov


def git_commit_full() -> str:
    """The 40-hex HEAD commit; RAISES when git fails (an empty commit is never recorded, B3)."""
    try:
        res = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=REPO_ROOT, timeout=30)
    except (OSError, subprocess.SubprocessError) as e:
        raise RuntimeError(f"git rev-parse HEAD failed: {type(e).__name__}: {e}") from e
    commit = res.stdout.strip()
    if res.returncode != 0 or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeError(f"git rev-parse HEAD failed (exit {res.returncode}): "
                           f"{res.stderr.strip()!r} / {commit!r}")
    return commit


def _atomic_write_json(path: Path, obj: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def _atomic_save_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def _rng_state() -> dict[str, np.ndarray]:
    import torch
    st = {"torch_cpu": torch.get_rng_state().numpy()}
    if torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state().numpy()
    return st


def _set_rng_state(npz) -> None:
    import torch
    if "torch_cpu" in npz:
        torch.set_rng_state(torch.from_numpy(np.asarray(npz["torch_cpu"])).to(torch.uint8))
    if "torch_cuda" in npz and torch.cuda.is_available():
        torch.cuda.set_rng_state(torch.from_numpy(np.asarray(npz["torch_cuda"])).to(torch.uint8))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------


def resume_mismatches(old: dict, new: dict) -> list[str]:
    """Why a checkpoint may NOT be continued by this process (B4): another commit, recipe settings,
    design, batch size, m_D, benchmark or VAE weights. The budget may grow."""
    out = []
    oc = (old.get("provenance") or {}).get("git_commit_full")
    nc = new["provenance"]["git_commit_full"]
    if oc != nc:
        out.append(f"commit {oc!r} != {nc!r}")
    for k in ("recipe", "benchmark", "dim", "seed", "n_init", "init_type", "batch_size", "m_d",
              "vae_weights", "sense"):
        if old.get(k) != new.get(k):
            out.append(f"{k} {old.get(k)!r} != {new.get(k)!r}")
    if json.dumps(old.get("settings"), sort_keys=True, default=str) != \
            json.dumps(new.get("settings"), sort_keys=True, default=str):
        out.append("recipe settings differ")
    return out


def run(args: argparse.Namespace) -> int:
    import torch

    from src.spherevo.published_recipes import DTYPE, propose, recipe_settings

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.batch_size != 1 and args.recipe != "turbo_published":
        raise SystemExit("--batch-size applies to turbo_published only")
    if args.m_d is not None and args.recipe != "baxus":
        raise SystemExit("--m-d applies to baxus only")

    bench = make_benchmark(args.benchmark, device, args.vae_weights)
    D = bench.dim
    n_init, init_type = default_init(args.recipe, D, bench.name)
    if args.n_init is not None:
        n_init = args.n_init
    # the job identifier carries n_init, so runs with different designs never share a file (B7)
    stem = f"{args.recipe}__{args.benchmark.replace(':', '-')}__n{n_init}__s{args.seed}"
    jpath, npath = out_dir / f"{stem}.json", out_dir / f"{stem}.npz"
    if jpath.exists() and not args.force:
        if json.loads(jpath.read_text()).get("status") == "completed":
            print(f"[skip] {jpath} is complete")
            return 0
    if args.budget < n_init:
        raise SystemExit(f"budget {args.budget} < n_init {n_init}")
    m_d = (args.m_d if args.m_d is not None else BAXUS_M_D_DEFAULT) if args.recipe == "baxus" \
        else None
    if m_d is not None and m_d <= n_init:
        raise SystemExit(f"--m-d {m_d} must exceed n_init {n_init}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.recipe in GP_RECIPES:
        settings = recipe_settings(args.recipe, D=D)
    elif args.recipe in NOISE_LEARNED:
        base = "msr" if NOISE_LEARNED[args.recipe] else "mle_scaled"
        settings = {**recipe_settings(base, D=D), "recipe": args.recipe,
                    "likelihood": "get_gaussian_likelihood_with_gamma_prior(), noise LEARNED "
                                  "(understanding-hdbo --no-noiseless)"}
    elif args.recipe in EXTERNAL:
        from src.spherevo.published_external import external_settings
        settings = external_settings({"turbo_published": "turbo1"}.get(args.recipe, args.recipe),
                                     dim=D, batch_size=args.batch_size)
        if m_d is not None:
            settings["m_d"] = m_d
            settings["eval_budget"] = m_d - n_init
    else:
        settings = {"recipe": args.recipe}
    record: dict[str, Any] = {
        "recipe": args.recipe, "benchmark": bench.name, "dim": D, "seed": args.seed,
        "n_init": n_init, "init_type": init_type, "budget": args.budget, "sense": bench.sense,
        "batch_size": args.batch_size if args.recipe == "turbo_published" else 1,
        "m_d": m_d, "vae_weights": args.vae_weights,
        "curves": "values = objective at each evaluation (benchmark sense, incl. init); "
                  "best_so_far = its running best",
        "settings": settings, "provenance": provenance(device), "status": "running",
        "resumed_at": [], "provenance_history": [], "started": datetime.now().isoformat(),
        "values": [], "best_so_far": [], "t_eval_s": [], "steps": [],
        "fallback_count": 0, "fallback_reasons": [],
    }
    U = np.zeros((0, D))
    resumable = args.recipe in GP_RECIPES + tuple(NOISE_LEARNED) + ("random",)
    if resumable and npath.exists() and jpath.exists() and not args.force:
        old = json.loads(jpath.read_text())
        mismatch = resume_mismatches(old, record)
        if mismatch:
            raise SystemExit(f"[refuse resume] {jpath}: {'; '.join(mismatch)} "
                             f"(start over with --force)")
        record["provenance_history"] = list(old.get("provenance_history", [])) + [
            {"provenance": old.get("provenance"), "budget": old.get("budget"),
             "evaluations_before_resume": len(old.get("values", []))}]
        ck = np.load(npath)
        U = np.asarray(ck["U"])
        for k in ("values", "best_so_far", "t_eval_s", "steps", "fallback_count",
                  "fallback_reasons", "resumed_at", "started"):
            record[k] = old.get(k, record[k])
        record["values"] = record["values"][:U.shape[0]]
        record["best_so_far"] = record["best_so_far"][:U.shape[0]]
        record["resumed_at"].append(int(U.shape[0]))
        _set_rng_state(ck)
        print(f"[resume] {stem} at evaluation {U.shape[0]}")

    best = min if bench.sense == "min" else max

    def record_eval(u_row, v: float, dt: float, step: Optional[dict]) -> None:
        nonlocal U
        U = np.vstack([U, np.asarray(u_row, dtype=np.float64).reshape(1, -1)])
        record["values"].append(float(v))
        prev = record["best_so_far"][-1] if record["best_so_far"] else v
        record["best_so_far"].append(float(best(prev, v)))
        record["t_eval_s"].append(round(dt, 4))
        if step is not None:
            record["steps"].append(step)
            if step.get("fallback"):
                record["fallback_count"] += 1
                record["fallback_reasons"].append(step["fallback"])
        if resumable:
            _atomic_save_npz(npath, U=U, **_rng_state())
        _atomic_write_json(jpath, record)

    def value_unit(u) -> float:
        return bench.value(bench.to_raw(np.clip(np.asarray(u, dtype=np.float64), 0.0, 1.0)))

    t_start = time.time()
    try:
        if args.recipe == "turbo":
            _run_turbo_box_faithful(args, bench, n_init, out_dir, record_eval, record)
        elif args.recipe == "turbo_published":
            from src.spherevo.published_external import Turbo1Published
            tb = Turbo1Published(dim=D, n_init=n_init, batch_size=args.batch_size)
            while len(record["values"]) < args.budget:
                t0 = time.time()
                U_b, info = tb.ask()                     # one init point, or a batch of q
                U_b = np.asarray(U_b).reshape(-1, D)
                t_ask = time.time() - t0
                k = min(len(U_b), args.budget - len(record["values"]))
                vals_b = []
                for j in range(k):
                    t1 = time.time()
                    v = value_unit(U_b[j])
                    vals_b.append(v)
                    if j == 0:
                        step = info.to_dict()
                        step.update(restarts=tb.n_restarts, batch_rows=int(len(U_b)))
                    else:
                        step = {"fallback": None, "events": [], "batch_member": j}
                    record_eval(U_b[j], v, time.time() - t1 + (t_ask if j == 0 else 0.0), step)
                tb.tell(U_b[:k], [-bench.to_max(v) for v in vals_b])   # TuRBO minimises
        elif args.recipe == "baxus":
            from src.spherevo.published_external import BaxusPublished
            # eval_budget from m_D (paper: 1000), independent of this run's budget (B5)
            bx = BaxusPublished(dim=D, n_init=n_init, eval_budget=m_d - n_init, seed=args.seed)
            while len(record["values"]) < args.budget:
                t0 = time.time()
                x, info = bx.ask()                       # in [-1, 1]^D
                u = (np.asarray(x).reshape(-1) + 1.0) / 2.0
                v = value_unit(u)
                bx.tell(bench.to_max(v))
                record_eval(u, v, time.time() - t0, info)
            record["splits"] = bx.n_splits_done
        elif args.recipe == "cmaes":
            from src.spherevo.published_external import CMAES_POPSIZE, run_cmaes_published
            t_last = [time.time()]

            def on_eval(x, f_min, info):
                u = (np.asarray(x).reshape(-1) + 1.0) / 2.0
                v = -f_min if bench.sense == "max" else f_min
                record_eval(u, v, time.time() - t_last[0], info)
                t_last[0] = time.time()

            def f_min(x):                               # pycma works on [-1, 1]^D
                return -bench.to_max(value_unit((np.asarray(x) + 1.0) / 2.0))

            # f(x0 = 0), evaluated once outside the CMA-ES budget: LassoBench's pycma logger holds
            # it at x = 1 (orchestrator check: released x = 1 = f(0) on the synthetic series), so
            # the running minimum is built over the same columns as the released curves (B1)
            record["f_x0"] = value_unit(np.full(D, 0.5))
            record["f_x0_note"] = ("objective at x0 = 0 (unit 0.5), benchmark sense, NOT counted in "
                                   "values / budget; placed at x = 1 of the logged CMA-ES curve")
            record["cma"] = run_cmaes_published(f_min, D, args.budget, args.seed, on_eval=on_eval)
            record["cma_popsize"] = CMAES_POPSIZE
        else:
            U0 = initial_design(n_init, D, init_type, args.seed)
            for i in range(U.shape[0], n_init):
                t0 = time.time()
                record_eval(U0[i], value_unit(U0[i]), time.time() - t0, None)
            while U.shape[0] < args.budget:
                t0 = time.time()
                if args.recipe == "random":
                    u = torch.rand(1, D, dtype=DTYPE).numpy()
                    step = {"fallback": None, "events": []}
                else:
                    X = torch.tensor(U, dtype=DTYPE, device=device)
                    Y = torch.tensor([bench.to_max(v) for v in record["values"]],
                                     dtype=DTYPE, device=device).reshape(-1, 1)
                    if args.recipe in NOISE_LEARNED:
                        from src.spherevo.published_recipes import _propose_papenmeier
                        x_next, rec = _propose_papenmeier(
                            X, Y, raasp=NOISE_LEARNED[args.recipe], noiseless=False)
                    else:
                        x_next, rec = propose(args.recipe, X, Y)
                    u = x_next.detach().cpu().numpy()
                    step = rec.to_dict(n=int(U.shape[0]) + 1)
                record_eval(u, value_unit(u), time.time() - t0, step)
        record["status"] = "completed"
    except Exception as e:  # noqa: BLE001 -- record and re-raise
        import traceback
        record["status"] = "failed"
        record["error"] = f"{type(e).__name__}: {e}"
        record["traceback"] = traceback.format_exc()
        _atomic_write_json(jpath, record)
        raise
    record["finished"] = datetime.now().isoformat()
    record["wall_s_this_process"] = round(time.time() - t_start, 2)
    _atomic_write_json(jpath, record)
    print(f"[done] {stem}: {len(record['values'])} evaluations, best {record['best_so_far'][-1]:.6g},"
          f" fallbacks {record['fallback_count']}")
    return 0


def _run_turbo_box_faithful(args, bench: Benchmark, n_init: int, out_dir: Path,
                            record_eval: Callable, record: dict) -> None:
    """The repository's TuRBO-1 (turbo_box_faithful.py) on the benchmark's own box."""
    from src.spherevo.turbo_box_faithful import TurboBoxFaithfulSolver

    D = bench.dim
    bounds_file = out_dir / f"bounds_{bench.name.replace(':', '-')}.json"
    _atomic_write_json(bounds_file, {"per_coord": {"p01": bench.lb.tolist(),
                                                   "p99": bench.ub.tolist()}})
    U0 = initial_design(n_init, D, "sobol", args.seed)
    X0 = np.vstack([bench.to_raw(u) for u in U0])
    vals = []
    for u, x in zip(U0, X0):
        t0 = time.time()
        vals.append(bench.value(x))
        record_eval(u, vals[-1], time.time() - t0, None)

    def black_box(z: np.ndarray) -> np.ndarray:
        return np.array([[bench.to_max(bench.value(row))] for row in np.asarray(z)])

    solver = TurboBoxFaithfulSolver(black_box=black_box, x0=X0,
                                    y0=np.array([[bench.to_max(v)] for v in vals]),
                                    bounds_path=str(bounds_file))
    record["settings"] = {"recipe": "turbo", "class": "TurboBoxFaithfulSolver",
                          "n_candidates": solver.n_candidates,
                          "failure_tolerance": solver.turbo_state.failure_tolerance,
                          "success_tolerance": solver.turbo_state.success_tolerance,
                          "batch_size": 1}
    while len(record["values"]) < args.budget:
        t0 = time.time()
        n_fb = solver._fallback_count
        solver.step()
        z = np.asarray(solver.history["x"][-1]).reshape(-1)
        y_max = float(np.asarray(solver.history["y"][-1]).reshape(-1)[0])
        step = {"fallback": solver._fallback_reasons[-1] if solver._fallback_count > n_fb else None,
                "events": [], "tr_length": float(solver.turbo_state.length)}
        record_eval((z - bench.lb) / (bench.ub - bench.lb),
                    -y_max if bench.sense == "min" else y_max, time.time() - t0, step)


# ---------------------------------------------------------------------------
# Targets (tolerant reader), statistics, verdict
# ---------------------------------------------------------------------------


def _num(x) -> Optional[float]:
    """A number, or a string that is only a number (optionally '>=' / '\u2265' before it, read as
    that number); anything else (free text such as 'not stated (10 in Table 3)') is None."""
    if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x):
        return float(x)
    if isinstance(x, str):
        m = re.fullmatch(r"\s*(?:>=|\u2265)?\s*(\d+(?:\.\d+)?)\s*", x)
        return float(m.group(1)) if m else None
    return None


def _records(rows) -> list[dict]:
    out = []
    if isinstance(rows, list):
        for r in rows:
            if isinstance(r, dict) and _num(r.get("n")) is not None and \
                    (_num(r.get("value")) is not None or _num(r.get("mean")) is not None):
                # SE_pub source: se, else the SD at the arg-min column of the running minimum
                # (sd_at_argmin, A6), else sd / sd_at_x
                sd_key = next((k for k in ("sd_at_argmin", "sd", "sd_at_x")
                               if _num(r.get(k)) is not None), None)
                out.append({"n": _num(r["n"]),
                            "value": _num(r.get("value", r.get("mean"))),
                            "read_unc": _num(r.get("read_unc")) or 0.0,
                            "se": _num(r.get("se")),
                            "sd": _num(r.get(sd_key)) if sd_key else None, "sd_key": sd_key,
                            "x_logged": _num(r.get("x_logged")),
                            "se_from": r.get("se_from") if isinstance(r.get("se_from"), str) else None,
                            "argmin_x": _num(r.get("argmin_x")),
                            "gate": r.get("gate") is not False})
    elif isinstance(rows, dict):
        for k, r in rows.items():
            if _num(k) is None or not isinstance(r, (dict, int, float)):
                return []
            r = r if isinstance(r, dict) else {"value": r}
            out.extend(_records([{**r, "n": k}]))
    return sorted(out, key=lambda r: r["n"])


def normalise_values(values) -> dict[Optional[str], list[dict]]:
    recs = _records(values)
    if recs:
        return {None: recs}
    if isinstance(values, dict):
        return {str(k): _records(v) for k, v in values.items() if _records(v)}
    return {}


def load_targets(path: Path) -> dict[str, dict]:
    """``{id: entry}`` from ``{"targets": [...]}``, a bare list, or ``{id: entry}``; each entry
    gets ``_norm = {task|None: [ {n, value, read_unc, se, sd, x_logged} ]}``."""
    raw = json.loads(Path(path).read_text())
    if isinstance(raw, dict) and "targets" in raw:
        entries = raw["targets"]
    elif isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        entries = [{**v, "id": v.get("id", k)} for k, v in raw.items() if isinstance(v, dict)]
    else:
        raise ValueError(f"unreadable targets file {path}")
    allowance = raw.get("objective_version_allowance", {}) if isinstance(raw, dict) else {}
    out = {}
    for i, e in enumerate(entries):
        if isinstance(e, dict):
            e = dict(e)
            e["_norm"] = normalise_values(e.get("values"))
            tid = str(e.get("id", f"entry_{i}"))
            key = (allowance.get("target_benchmark") or {}).get(tid)
            e["_objective_benchmark"] = key
            e["_objective_delta"] = float(allowance["delta"][key]) if key is not None else 0.0
            out[tid] = e
    return out


def statistic_kind(target: dict) -> str:
    stat = str(target.get("statistic", "")).lower()
    if "running minimum" in stat:
        return "lassobench_running_min_of_mean"
    if target.get("curve") == "value_at_eval":
        return "mean_value_at_eval"
    return "mean_best_so_far"


def _cma_logged_curve(run: dict) -> Optional[dict[int, float]]:
    """CMA-ES runs: the values LassoBench's pycma logger holds: f(x0) at x = 1 (``f_x0``, evaluated
    outside the budget, B1) and the best loss of each completed generation at x = g * popsize
    (logger column 'best', UNVERIFIED pycma semantics). None when the run has no ``f_x0``."""
    if _num(run.get("f_x0")) is None:
        return None
    pop = int(run.get("cma_popsize", 20))
    vals = run["values"]
    curve = {1: float(run["f_x0"])}
    sense_min = run.get("sense", "min") == "min"
    for g in range(len(vals) // pop):
        chunk = vals[g * pop:(g + 1) * pop]
        curve[(g + 1) * pop] = min(chunk) if sense_min else max(chunk)
    return curve


def our_statistic(runs: list[dict], kind: str, n: float, x_logged: Optional[float],
                  ) -> tuple[Optional[float], Optional[float]]:
    """(statistic, its standard error) over ``runs`` at evaluation count ``n``."""
    return our_statistic_detail(runs, kind, n, x_logged)[:2]


def our_statistic_detail(runs: list[dict], kind: str, n: float, x_logged: Optional[float],
                         ) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """(statistic, its standard error, the x of the arg-min column of the running minimum; None for
    the other statistics)."""
    k = len(runs)
    if k == 0:
        return None, None, None
    if kind == "lassobench_running_min_of_mean":
        x = int(x_logged if x_logged is not None else n)
        if all(r["recipe"] == "cmaes" for r in runs):
            if any(len(r["values"]) < x for r in runs):     # B2: never a truncated curve
                return None, None, None
            curves = [_cma_logged_curve(r) for r in runs]
            if any(c is None for c in curves):
                return None, None, None
            idx = sorted(i for i in curves[0] if i <= x and all(i in c for c in curves))
            M = np.array([[c[i] for i in idx] for c in curves])
        else:
            if any(len(r["values"]) < x for r in runs):
                return None, None, None
            M = np.array([r["values"][:x] for r in runs])
            idx = list(range(1, x + 1))
        if M.size == 0:
            return None, None, None
        mean = M.mean(axis=0)
        sense_min = runs[0].get("sense", "min") == "min"
        j = int(np.argmin(mean) if sense_min else np.argmax(mean))
        se = float(M[:, j].std(ddof=1) / math.sqrt(k)) if k > 1 else None
        return float(mean[j]), se, float(idx[j])
    key = "values" if kind == "mean_value_at_eval" else "best_so_far"
    if any(len(r[key]) < n for r in runs):
        return None, None, None
    vals = np.array([r[key][int(n) - 1] for r in runs])
    return float(vals.mean()), (float(vals.std(ddof=1) / math.sqrt(k)) if k > 1 else None), None


def verdict_for(target: dict, task: Optional[str], runs_by_seed: dict[int, dict],
                n_reps_override: Optional[int] = None,
                refusals: Optional[list[str]] = None) -> dict:
    """The pass rule on one target. ``refusals`` (from ``summarize``: provenance and protocol)
    turn the verdict into ``pass: None``; the rows are computed either way."""
    n_reps = float(n_reps_override) if n_reps_override else _num(target.get("n_reps"))
    if not n_reps:
        return {"target_id": target.get("id"), "task": task, "pass": None, "rows": [],
                "note": f"n_reps of the target is not a number ({target.get('n_reps')!r}); "
                        f"give it with --n-reps {target.get('id')}=N (fail closed)"}
    need = int(n_reps)
    seeds = list(range(need))
    runs = [runs_by_seed[s] for s in seeds if s in runs_by_seed]
    refusals = list(refusals or [])
    # x_eval_offset: the record's n is the plot x; compared at n + offset evaluations (1 when the
    # plot indexes evaluations from 0: Xu Fig. 6, Papenmeier Fig. 8, BAxUS Fig. 3)
    offset = _num(target.get("x_eval_offset")) or 0.0
    q_target = _num(target.get("batch_size"))
    q_runs = {int(r.get("batch_size", 1)) for r in runs}
    if q_target is not None and runs and q_runs != {int(q_target)}:
        refusals.append(f"batch size of the runs {sorted(q_runs)} != target batch_size "
                        f"{int(q_target)}; rerun with --batch-size {int(q_target)}")
    kind = statistic_kind(target)
    exact = str(target.get("value_kind", "")).startswith(EXACT_KINDS)
    doe = _num(target.get("doe_only_below")) or 0
    gate = target.get("gate") is not False
    delta = _num(target.get("_objective_delta")) or 0.0       # R2-4 (load_targets)
    missing = [s for s in seeds if s not in runs_by_seed]
    all_ok, all_ok_strict = need > 0, need > 0
    rows = []
    recs = target["_norm"].get(task) or target["_norm"].get(None) or []
    for r in recs:
        n_eval = r["n"] + offset
        if n_eval < doe:                   # inside the initial design (evaluation counts)
            continue
        x_log = None if r["x_logged"] is None else r["x_logged"] + offset
        mean, se, our_argmin = our_statistic_detail(runs, kind, n_eval, x_log)
        row = row_rule(r, mean, se, n_reps, exact, delta=delta)
        row_gate = r["gate"]
        if r["argmin_x"] == 1.0 and our_argmin == 1.0:
            # R2-11: published and our running minimum both sit at x = 1 = f(x0): f(x0) against f(x0)
            row_gate = False
            row["start_point"] = True
            row["gate_reason"] = ("start point: published and our running minimum both at x = 1 "
                                  "(f(x0) against f(x0)), not CMA-ES behaviour")
        if row_gate and row["pass"] is not None:
            all_ok = all_ok and bool(row["pass"])
            all_ok_strict = all_ok_strict and bool(row["pass_strict"])
        rows.append({"n": r["n"], "n_evaluations_compared": n_eval, "x_logged": r["x_logged"],
                     "target": r["value"], "target_read_unc": r["read_unc"], **row,
                     "ours": mean, "ours_se": se, "ours_argmin_x": our_argmin,
                     "target_argmin_x": r["argmin_x"], "gate": row_gate})
    gating_rows = [row for row in rows if row["gate"]]
    within = all_ok and bool(gating_rows)
    within_strict = all_ok_strict and bool(gating_rows)
    # R2-10 (m4): an incomplete verdict is "not verified" (None), never a fail
    incomplete = []
    if missing:
        incomplete.append(f"incomplete: seeds missing {missing}")
    no_stat = [row["n"] for row in gating_rows if row["pass"] is None]
    if no_stat:
        incomplete.append(f"no statistic at n = {', '.join(f'{n:g}' for n in no_stat)}")
    if not gating_rows:
        incomplete.append("no gating row")
    decided = gate and not refusals and not incomplete
    out = {"target_id": target.get("id"), "task": task, "statistic": kind,
           "value_kind": target.get("value_kind"), "n_reps_required": need,
           "objective_benchmark": target.get("_objective_benchmark"), "objective_delta": delta,
           "seeds_used": [s for s in seeds if s in runs_by_seed], "rows": rows,
           "gate": gate, "refusals": refusals, "incomplete": incomplete,
           "pass": within if decided else None,
           "pass_strict": within_strict if decided else None}
    fb = sum(1 for r in runs if (r.get("fallback_count") or 0) > 0)
    if fb:                                    # R2-10 (m7): recorded, not a refusal
        out["fallback_note"] = f"fallbacks in {fb} runs"
    if not gate:
        out["note"] = "informative target (gate: false): rows reported, never a pass or fail"
        out["informative_all_gating_rows_within_tolerance"] = within
    elif refusals:
        out["note"] = "verdict refused: " + "; ".join(refusals)
        out["rows_within_tolerance_if_not_refused"] = within
    elif incomplete:
        out["note"] = "not verified: " + "; ".join(incomplete)
    return out


def row_rule(r: dict, mean: Optional[float], se: Optional[float], n_reps: float, exact: bool,
             delta: float = 0.0) -> dict[str, Any]:
    """Tolerance and pass of one row (RUN_RULES (2) with round 2, R2-3 and R2-4).

    exact kinds: 2·sqrt(SE_ours² + SE_pub²), SE_pub = the target's own (se, else sd / sqrt(n_reps),
    else 0). Figure readings: read_unc + 2·sqrt(SE_ours² + SE_pub²), SE_pub = the stated SE, else
    SE_ours ("assumed_equal_to_ours"). Both + δ·|value| on a LassoBench objective (``delta``).
    ``pass_strict``: δ = 0 and SE_pub = 0 where the band is unstated.
    """
    if r["se"] is not None:
        stated, src = r["se"], (r.get("se_from") or "se")
    elif r["sd"] is not None and n_reps:
        stated, src = r["sd"] / math.sqrt(n_reps), r["sd_key"]
    else:
        stated, src = None, None
    out: dict[str, Any] = {"target_se": None, "target_se_from": src, "target_se_strict": stated or 0.0,
                           "objective_delta": delta, "objective_allowance": delta * abs(r["value"]),
                           "tolerance": None, "pass": None, "tolerance_strict": None,
                           "pass_strict": None}
    if exact:
        se_pub = stated if stated is not None else 0.0
        out["target_se_from"] = src or "none (0)"
    elif stated is not None:
        se_pub = stated
    else:
        se_pub = se                       # may be None when ours is not computable yet
        out["target_se_from"] = "assumed_equal_to_ours"
    out["target_se"] = se_pub
    if mean is None or se is None:
        return out
    read_unc = 0.0 if exact else r["read_unc"]
    diff = abs(mean - r["value"])
    out["tolerance"] = read_unc + 2.0 * math.sqrt(se ** 2 + se_pub ** 2) + out["objective_allowance"]
    out["pass"] = diff <= out["tolerance"]
    out["tolerance_strict"] = read_unc + 2.0 * math.sqrt(se ** 2 + out["target_se_strict"] ** 2)
    out["pass_strict"] = diff <= out["tolerance_strict"]
    return out


def expected_n_init(target: dict, recipe: str) -> Optional[set[int]]:
    if recipe == "cmaes":
        return {0}                                     # x0 = 0, no design (f(x0) outside)
    if target.get("id") == "hvarfner_dsp__lasso_dna__fig7":
        return set(HVARFNER_LASSO_N_INIT)
    n = _num(target.get("n_init"))
    return {int(n)} if n is not None else None


def protocol_refusals(target: dict, recipe: str, runs: list[dict]) -> list[str]:
    """n_init, initial design, budget and m_D of the runs against the target's protocol (B3)."""
    out = []
    exp_n = expected_n_init(target, recipe)
    n_runs = {r.get("n_init") for r in runs}
    if exp_n is None:
        out.append(f"target {target.get('id')} states no numeric n_init for recipe {recipe}")
    elif len(n_runs) != 1 or not n_runs <= exp_n:
        out.append(f"n_init of the runs {sorted(map(str, n_runs))} not one of {sorted(exp_n)}")
    if len(n_runs) > 1:
        out.append("runs with different n_init in one verdict (use --n-init-filter)")
    exp_t = EXPECTED_INIT_TYPE.get(recipe)
    t_runs = {r.get("init_type") for r in runs}
    if exp_t is None or t_runs != {exp_t}:
        out.append(f"initial design of the runs {sorted(map(str, t_runs))} != {exp_t!r}")
    budgets = {r.get("budget") for r in runs}
    if len(budgets) != 1:
        out.append(f"runs with different budgets {sorted(map(str, budgets))}")
    offset = _num(target.get("x_eval_offset")) or 0.0
    need = max([((r["x_logged"] if r["x_logged"] is not None else r["n"]) + offset)
                for recs in target.get("_norm", {}).values() for r in recs] or [0.0])
    if budgets and min(int(b or 0) for b in budgets) < need:
        out.append(f"budget {sorted(map(str, budgets))} < {int(need)} evaluations the target needs")
    if recipe == "baxus" and {r.get("m_d") for r in runs} != {BAXUS_M_D_DEFAULT}:
        out.append(f"BAxUS m_D of the runs {sorted(map(str, {r.get('m_d') for r in runs}))} "
                   f"!= {BAXUS_M_D_DEFAULT} (paper, Sec. 4)")
    return out


def provenance_refusals(runs: list[dict]) -> list[str]:
    """One commit, clean trees, a recorded 40-hex commit (B3)."""
    out = []
    commits = {(r.get("provenance") or {}).get("git_commit_full") for r in runs}
    bad = [c for c in commits if not (isinstance(c, str) and re.fullmatch(r"[0-9a-f]{40}", c))]
    if bad:
        out.append(f"runs without a recorded commit: {sorted(map(str, bad))}")
    if len(commits) > 1:
        out.append(f"runs from {len(commits)} commits: {sorted(map(str, commits))}")
    dirty = [r.get("seed") for r in runs if (r.get("provenance") or {}).get("git_dirty") is not False]
    if dirty:
        out.append(f"runs from a dirty or unknown tree (seeds {sorted(dirty)})")
    hist = [r.get("seed") for r in runs
            if any((h.get("provenance") or {}).get("git_commit_full")
                   != (r.get("provenance") or {}).get("git_commit_full")
                   for h in r.get("provenance_history", []))]
    if hist:
        out.append(f"runs resumed across commits (seeds {sorted(hist)})")
    return out


def summarize(args: argparse.Namespace) -> int:
    tpath = Path(args.targets)
    targets = load_targets(tpath)
    overrides = {}
    for item in args.n_reps or []:
        tid_, _, n_ = item.partition("=")
        overrides[tid_] = int(n_)
    groups: dict[tuple[str, str, Any], dict[int, dict]] = {}
    for p in sorted(Path(args.out_dir).glob("*.json")):
        try:
            r = json.loads(p.read_text())
        except ValueError:
            continue
        if not isinstance(r, dict) or "recipe" not in r or "values" not in r:
            continue
        if args.n_init_filter is not None and r.get("n_init") != args.n_init_filter:
            continue
        if r.get("status") == "completed":
            key = (r["recipe"], r["benchmark"], r.get("n_init"))
            groups.setdefault(key, {})[int(r["seed"])] = r
    tsha = sha256_file(tpath)
    out = {"generated": datetime.now().isoformat(), "targets_file": str(tpath),
           "n_reps_overrides": overrides, "n_init_filter": args.n_init_filter,
           "targets_sha256": tsha, "targets_sha256_expected": TARGETS_SHA256,
           "summarizer_provenance": provenance("cpu"), "verdicts": []}
    file_refusals = ([] if tsha == TARGETS_SHA256 else
                     [f"targets file sha256 {tsha} != shipped {TARGETS_SHA256}"])
    for (recipe, bench, n_init), runs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        tid = args.target_id or default_target_id(recipe, bench)
        tgt = targets.get(tid) if tid else None
        task = bench.split(":", 1)[1] if ":" in bench else None
        if tgt is None:
            out["verdicts"].append({"recipe": recipe, "benchmark": bench, "n_init": n_init,
                                    "target_id": tid, "pass": None, "note": "no target"})
            continue
        used = [runs[s] for s in sorted(runs)]
        refusals = file_refusals + provenance_refusals(used) + protocol_refusals(tgt, recipe, used)
        v = verdict_for(tgt, task, runs, n_reps_override=overrides.get(tid), refusals=refusals)
        v.update(recipe=recipe, benchmark=bench, n_init=n_init,
                 commits=sorted({str((r.get("provenance") or {}).get("git_commit_full"))
                                 for r in used}),
                 fallbacks=sum(r.get("fallback_count", 0) for r in runs.values()))
        out["verdicts"].append(v)
    if args.write:
        _atomic_write_json(Path(args.write), out)
    for v in out["verdicts"]:
        print(json.dumps({k: v.get(k) for k in ("recipe", "benchmark", "n_init", "target_id",
                                                 "pass", "seeds_used", "fallbacks", "refusals")}))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run one (recipe, benchmark, seed)")
    r.add_argument("--recipe", required=True, choices=RECIPES)
    r.add_argument("--benchmark", required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--budget", type=int, required=True, help="total evaluations incl. the init")
    r.add_argument("--n-init", type=int, default=None)
    r.add_argument("--batch-size", type=int, default=1,
                   help="turbo_published only: TuRBO batch q (LassoBench pickles: 10)")
    r.add_argument("--m-d", type=int, default=None,
                   help=f"baxus only: m_D, the evaluations until the input dimension is reached "
                        f"(default {BAXUS_M_D_DEFAULT}, BAxUS Fig. 3); eval_budget = m_D - n_init")
    r.add_argument("--vae-weights", default=None, help="guacamol: state dict to load instead")
    r.add_argument("--out-dir", default="results/verify_baselines")
    r.add_argument("--device", default="auto")
    r.add_argument("--force", action="store_true", help="start over even if a run exists")
    s = sub.add_parser("summarize", help="pass/fail of finished runs against the targets")
    s.add_argument("--out-dir", default="results/verify_baselines")
    s.add_argument("--targets", default=str(TARGETS_FILE))
    s.add_argument("--target-id", default=None, help="override the default target id")
    s.add_argument("--write", default=None, help="also write the verdict JSON here")
    s.add_argument("--n-reps", action="append", default=None, metavar="TARGET_ID=N",
                   help="repetitions for a target whose n_reps is not a number (recorded)")
    s.add_argument("--n-init-filter", type=int, default=None,
                   help="only runs with this n_init (e.g. Hvarfner 30 or 41)")
    return ap


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args) if args.cmd == "run" else summarize(args)


if __name__ == "__main__":
    sys.exit(main())
