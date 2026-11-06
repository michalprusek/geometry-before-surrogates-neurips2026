#!/usr/bin/env python
"""Camera-ready retrieval-floor analysis (read-only over existing artefacts).

What it does
------------
1. Loads the authors' floor module ``scripts/analysis/zinc_retrieval_floor.py`` from the
   clean ``e0-instrumentation`` worktree (by file path; no bytecode is written) and uses
   ITS order-statistics and null-test functions: ``_unique_cdf``, ``max_cdf``,
   ``emax_from_cdf``, ``exact_emax``, ``exact_emax_no_replacement``, ``exact_max_quantile``,
   ``null_mean_draws``, ``null_mean_test``, ``null_sign_test``, ``sample_max_from_cdf``,
   ``coldstart_best``, ``holm``.
2. Loads the cached per-molecule oracle scores ``scores_<task>.npy`` (249,455 ZINC-250k
   molecules, commit c33f51d, clean) and computes per task the retrieval floor
   E[max of 30 distinct molecules] (hypergeometric, WITHOUT replacement; the WITH-replacement
   value the September screen printed is reported alongside) and three kinds of uncertainty:
   (a) spread of max_30 for ONE random draw of 30 (exact SD and 5/50/95 % quantiles),
   (b) null spread of a 40-seed mean of max_30 (SD and 2.5/97.5 % quantiles, Monte Carlo
       from the exact CDF), which is what a 40-seed method mean is compared against,
   (c) bootstrap SE of the floor itself over the molecule library (multinomial resampling
       of the 249,455 molecules), i.e. how much E[max_30] would move with another
       ZINC-sized draw from the same population.
3. Loads the best@30 per (task, seed) of every requested arm from the stored result JSONs
   under two documented conventions (primary ITT main-file; secondary clean filter) and
   tests each 40-seed mean against the floor:
     * exact-null Monte Carlo test (the authors' compare test): H0 = every seed's best@30
       is the max of 30 molecules drawn uniformly from the library; two one-sided
       p-values p_above, p_below; null scheme ``norepl`` (primary) and ``replace``;
     * one-sample t-test of mean(best@30) - floor (two-sided; tests the mean only, with the
       arm's own variance) as a robustness check;
     * for the external baselines, which record fewer than 30 distinct oracle calls in
       some cells, a budget-matched null where seed s is the max of n_evaluated(s) draws.
   Verdicts: rule P = exact null, two-sided alpha 0.05, i.e. each one-sided p < 0.025 after
   Holm adjustment across the 11 tasks within an arm; rule T = one-sample t-test (mean
   only), two-sided, Holm across the 11 tasks, alpha 0.05; rule S (the September screen) =
   exact null, one-sided p < 0.05, uncorrected.
4. Cross-task: mean of per-task means vs the mean floor, with the null of the 11-task
   average obtained from independent per-task null draws; plus the task-level exact
   Wilcoxon signed-rank test of the 11 per-task differences (task = unit).
5. Cold-start check for every "ours" arm: first n_initial_points scores of each cell vs
   the floor vector at the benchmark's own cold-start indices (exact oracle agreement).
6. k = 1 convex-hull arm: also compared with its own-pool reference = E[max(cold-start
   best, max of 25 draws from the 3,000-molecule latent cache)] assuming perfect
   reconstruction, per seed.

Writes ONLY into --outdir (JSON + Markdown). Never writes into the repository: --outdir is
refused when it lies inside any git work tree (``git rev-parse --show-toplevel`` succeeds for
it or for its nearest existing parent) or inside one of the input directories (--repo,
--guacamol, --floor-dir, --p1); paths are compared by component (``os.path.commonpath``).

Provenance keys of cr_floor.json: ``floor_worktree``, ``floor_worktree_head``,
``floor_worktree_status_lines`` and ``floor_worktree_status_lines_after`` describe the work tree
that supplied the floor module. Until 2026-10-11 they were called ``e0_worktree_head``,
``e0_worktree_status_lines`` and ``e0_status_lines_after``; the JSON carries both spellings
(the old ones as aliases, listed under ``provenance_key_aliases``) and ``write_md`` renders a
JSON with either.

Every input is a command-line argument, so the script runs from any checkout on
any host (CPU only; one process, ~10 min at --reps 1e5)::

    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= nice -n 19 <python> -B \
      scripts/cr/analysis/retrieval_floor.py \
      --repo <NeurIPS repo> --floor-module <generation>/scripts/analysis/zinc_retrieval_floor.py \
      --floor-dir <generation>/results/e0_floor --outdir <writable dir outside the repo>

Optional: --guacamol (a different result root), --p1 (September P1 blocks; without it
the @P1 rows are dropped), --smiles, --latent-sample, --smiles-sha,
--floor-module-sha.
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # must precede every import of repo code

import argparse  # noqa: E402
import datetime  # noqa: E402
import glob  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402

import numpy as np  # noqa: E402

# Every input is a command-line argument (see main()); these module globals are
# filled in by ``_configure`` before anything reads them, so the script carries
# no machine-specific path. The defaults of the SHA guards are the values the
# rebuttal-era analysis verified; pass --smiles-sha / --floor-module-sha to run
# against a different file on purpose.
REPO = ""            # --repo        NeurIPS repository root (read only)
G = ""               # --guacamol    default <repo>/results/guacamol
P1 = ""              # --p1          September P1 block directory ("" = skip)
FLOOR_MODULE = ""    # --floor-module
FLOOR_MODULE_SHA = "d6cc4b3cd6489af5c580cd62e7bf0a62fec0c208713278aed7a6222a3f47edd3"
FLOOR_DIR = ""       # --floor-dir
SMILES = ""          # --smiles      default <repo>/data/zinc_250k.txt
SMILES_SHA = "df0aa677c770f15c9943684d6600cba8e7076c67be6a090c12f24da3aaa55b70"
LATENT_SAMPLE = ""   # --latent-sample
ARMS: list = []      # built by build_arms() once the directories are known

T11 = {"amlodipine_mpo": "adip", "deco_hop": "dhop", "fexofenadine_mpo": "fexo", "median_1": "med1",
       "median_2": "med2", "osimetrinib_mpo": "osmb", "perindopril_mpo": "pdop",
       "ranolazine_mpo": "rano", "scaffold_hop": "shop", "sitagliptin_mpo": "siga",
       "zaleplon_mpo": "zale"}
TASKS = sorted(T11.values())
SEEDS = list(range(42, 82))
B = 30

# (row id, label, kind, directory, solver, generation note)
def build_arms(G: str, P1: str) -> list:
    """The arm table, as a function of the two result roots.

    Rows whose directory is the September P1 block are dropped when ``--p1``
    was not given, so the script runs on a machine that has only the
    repository.
    """
    arms = [
    ("geowalk", "GeoWalk (paper runs)", "ours", f"{G}/geowalk", "geowalk", "paper"),
    ("cma_es_subspace", "GeoCMA = cma_es_subspace (paper runs)", "ours", f"{G}/spherevo", "cma_es_subspace", "paper"),
    ("random_search", "Random search on the shell (paper runs, March)", "ours", f"{G}/spherevo", "random_search", "paper"),
    ("random_search@P1", "Random search on the shell (Sept. clean rerun, P1 4780475)", "ours", P1, "random_search", "sept-clean"),
    ("hvarfner_scaled_gp", "Vanilla BO shell x argmax (paper runs, April)", "ours", f"{G}/spherevo", "hvarfner_scaled_gp", "paper"),
    ("hvarfner_scaled_gp@E1", "Vanilla BO shell x argmax (Sept. clean rerun, E-1 abbd61e)", "ours", f"{G}/em1_premise", "hvarfner_scaled_gp", "sept-clean"),
    ("hvarfner_scaled_gp_box_faithful", "Vanilla BO [p01,p99] box x L-BFGS (rebuttal runs, May)", "ours", f"{G}/spherevo", "hvarfner_scaled_gp_box_faithful", "rebuttal"),
    ("hvarfner_scaled_gp_box_faithful@E1", "Vanilla BO [p01,p99] box x L-BFGS (Sept. clean rerun, E-1)", "ours", f"{G}/em1_premise", "hvarfner_scaled_gp_box_faithful", "sept-clean"),
    ("linear_bo", "Linear BO, box (rebuttal runs, July)", "ours", f"{G}/spherevo", "linear_bo", "rebuttal"),
    ("linear_bo_sphere_calib", "Linear BO, shell, calibrated (rebuttal runs, July)", "ours", f"{G}/spherevo", "linear_bo_sphere_calib", "rebuttal"),
    ("linear_bo_sphere_calib@P1", "Linear BO, shell, calibrated (Sept. clean rerun, P1)", "ours", P1, "linear_bo_sphere_calib", "sept-clean"),
    ("papenmeier_msr", "Papenmeier MSR GP, shell (corrected init, July)", "ours", f"{G}/spherevo", "papenmeier_msr", "rebuttal"),
    ("papenmeier_msr_box_faithful", "Papenmeier MSR GP, [p01,p99] box x L-BFGS (July)", "ours", f"{G}/spherevo", "papenmeier_msr_box_faithful", "rebuttal"),
    ("xu_standard_gp", "Xu standard GP, shell (April)", "ours", f"{G}/spherevo", "xu_standard_gp", "paper"),
    ("xu_standard_gp_box_faithful", "Xu standard GP, [p01,p99] box x L-BFGS (July)", "ours", f"{G}/spherevo", "xu_standard_gp_box_faithful", "rebuttal"),
    ("turbo", "TuRBO (external)", "ext", f"{G}/turbo", "turbo", "paper"),
    ("baxus", "BAxUS (external)", "ext", f"{G}/baxus", "baxus", "paper"),
    ("lolbo", "LOL-BO (external)", "ext", f"{G}/lolbo", "lolbo", "paper"),
    ("invbo", "InvBO (external)", "ext", f"{G}/invbo", "invbo", "paper"),
    ("cmaes_full", "CMA-ES full-D (external)", "ext", f"{G}/cmaes_full", "cmaes_full", "paper"),
    ("convex_hull_k1", "Convex hull k = 1 = random encoded ZINC latent, decoded (July)", "ours", f"{G}/spherevo", "convex_hull_k1", "rebuttal"),
    ]
    if not P1:
        arms = [a for a in arms if not a[0].endswith("@P1")]
    return arms


# ── helpers ──────────────────────────────────────────────────────────────────

def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(cwd: str, *args: str) -> str:
    try:
        # --no-optional-locks: `git status` must not refresh/rewrite .git/index (strictly read-only)
        return subprocess.run(["git", "--no-optional-locks", "-C", cwd, *args],
                              capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def _inside(path: str, root: str) -> bool:
    """True when ``path`` is ``root`` or lies below it (path components, not string prefixes:
    ``/x/NeurIPS-cr`` is NOT inside ``/x/NeurIPS``)."""
    a, b = os.path.realpath(path), os.path.realpath(root)
    try:
        return os.path.commonpath([a, b]) == b
    except ValueError:          # e.g. different drives
        return False


def _git_toplevel(path: str) -> str:
    """Top level of the git work tree that contains ``path`` or its nearest existing parent ('' = none)."""
    p = os.path.realpath(path)
    while not os.path.isdir(p):
        parent = os.path.dirname(p)
        if parent == p:
            return ""
        p = parent
    top = git(p, "rev-parse", "--show-toplevel")
    return top if top and not top.startswith("ERR") else ""


def check_outdir(outdir: str, inputs: list) -> None:
    """Refuse an output directory inside an input directory or inside any git work tree."""
    for d in inputs:
        if d and _inside(outdir, d):
            raise SystemExit(f"refusing to write inside an input directory ({d})")
    top = _git_toplevel(outdir)
    if top:
        raise SystemExit(f"refusing to write inside a git work tree ({top}); choose an --outdir outside every repository")


# Provenance keys renamed on 2026-10-11 (the script became path-independent): new name -> old name.
PROVENANCE_ALIASES = {"floor_worktree_head": "e0_worktree_head",
                      "floor_worktree_status_lines": "e0_worktree_status_lines",
                      "floor_worktree_status_lines_after": "e0_status_lines_after"}


def _pv(pv: dict, key: str):
    """Provenance value under its current name, or under the name it had before the rename."""
    if key in pv:
        return pv[key]
    old = PROVENANCE_ALIASES.get(key)
    if old is not None and old in pv:
        return pv[old]
    raise KeyError(f"provenance has neither {key!r} nor its old name {old!r}")


def _add_aliases(pv: dict) -> None:
    """Write the old key names next to the new ones, so that consumers of either spelling keep working."""
    for new, old in PROVENANCE_ALIASES.items():
        if new in pv:
            pv[old] = pv[new]
    pv["provenance_key_aliases"] = dict(PROVENANCE_ALIASES)


def _worktree_of(path: str) -> str:
    """Top level of the git worktree containing ``path`` ('' when there is none)."""
    if not path:
        return ""
    start = path if os.path.isdir(path) else os.path.dirname(os.path.abspath(path))
    top = git(start, "rev-parse", "--show-toplevel")
    return top if top and not top.startswith("ERR") else start


def load_floor_module(path: str):
    got = sha256(path)
    if got != FLOOR_MODULE_SHA:
        raise SystemExit(f"floor module sha256 {got} != expected {FLOOR_MODULE_SHA}")
    spec = importlib.util.spec_from_file_location("zinc_retrieval_floor_e0", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def prov_of(d: dict):
    p = d.get("provenance")
    if isinstance(p, dict) and isinstance(p.get("provenance"), dict):
        p = p["provenance"]
    if not isinstance(p, dict):
        return (None, None)
    return (p.get("git_commit"), p.get("git_dirty"))


def load_ours(dirpath: str, solver: str):
    """Main budget-30 files only: ours_<solver>_<task>_s<seed>_<dim>d.json (no suffix)."""
    pat = re.compile(rf"^ours_{re.escape(solver)}_(.+)_s(\d+)_(\d+)d\.json$")
    cells = {}
    for f in sorted(glob.glob(os.path.join(dirpath, f"ours_{solver}_*.json"))):
        m = pat.match(os.path.basename(f))
        if not m:
            continue
        task, seed = m.group(1), int(m.group(2))
        if task not in T11 or seed not in SEEDS:
            continue
        d = json.load(open(f))
        if d.get("solver") != solver:
            continue
        key = (T11[task], seed)
        if key in cells:
            raise RuntimeError(f"duplicate main file for {solver} {key}: {cells[key]['file']} and {f}")
        a = np.asarray([np.nan if x is None else x for x in (d.get("all_scores") or [])], dtype=float)
        cells[key] = {
            "file": f, "all_scores": a, "status": d.get("status"), "budget": d.get("budget"),
            "fallbacks": int(d.get("surrogate_fallback_count") or 0),
            "best_trace_len": len(d.get("best_trace") or []), "n_init": d.get("n_initial_points"),
            "mode": d.get("mode"), "prov": prov_of(d), "timestamp": str(d.get("timestamp", ""))[:19],
            "mtime": os.path.getmtime(f),
        }
    return cells


def load_ext(dirpath: str):
    """The convention of the submitted version's loader of external results: max(history.best_score) else
    best_score."""
    cells = {}
    for f in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
        if os.path.basename(f).startswith("gpu"):
            continue
        try:
            d = json.load(open(f))
        except (OSError, json.JSONDecodeError):
            continue
        t, s = d.get("task_id"), d.get("seed")
        if t not in TASKS or s not in SEEDS:
            continue
        h = (d.get("history") or {}).get("best_score") or []
        score = max(h) if h else d.get("best_score")
        if score is None:
            continue
        key = (t, int(s))
        if key in cells:  # first file in sorted order wins, as in load_external_results
            continue
        cells[key] = {"file": f, "best": float(score), "n_evaluated": d.get("n_evaluated"),
                      "n_hist": len(h), "mtime": os.path.getmtime(f)}
    return cells


# ── main ─────────────────────────────────────────────────────────────────────

def _configure(args) -> None:
    """Fill the module globals from the command line (see their declaration)."""
    global REPO, G, P1, FLOOR_MODULE, FLOOR_MODULE_SHA, FLOOR_DIR
    global SMILES, SMILES_SHA, LATENT_SAMPLE, ARMS
    REPO = os.path.abspath(args.repo)
    G = os.path.abspath(args.guacamol) if args.guacamol else f"{REPO}/results/guacamol"
    P1 = os.path.abspath(args.p1) if args.p1 else ""
    FLOOR_MODULE = os.path.abspath(args.floor_module)
    FLOOR_MODULE_SHA = args.floor_module_sha
    FLOOR_DIR = os.path.abspath(args.floor_dir)
    SMILES = os.path.abspath(args.smiles) if args.smiles else f"{REPO}/data/zinc_250k.txt"
    SMILES_SHA = args.smiles_sha
    LATENT_SAMPLE = (
        os.path.abspath(args.latent_sample) if args.latent_sample
        else f"{REPO}/results/precomputed/zinc_latents_sample.npz"
    )
    ARMS = build_arms(G, P1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", required=True)
    # --- inputs (no path is baked into this script) ---
    ap.add_argument("--repo", required=True, help="NeurIPS repository root (read only)")
    ap.add_argument("--guacamol", default=None, help="result root (default <repo>/results/guacamol)")
    ap.add_argument("--p1", default="", help="September P1 block dir; omitted => the @P1 rows are dropped")
    ap.add_argument("--floor-module", required=True,
                    help="scripts/analysis/zinc_retrieval_floor.py of the generation that scored the library")
    ap.add_argument("--floor-module-sha", default=FLOOR_MODULE_SHA, help="sha256 guard for --floor-module")
    ap.add_argument("--floor-dir", required=True,
                    help="directory with dataset.json, zinc_floor_summary.json, scores_<task>.npy, meta_<task>.json")
    ap.add_argument("--smiles", default=None, help="SMILES library (default <repo>/data/zinc_250k.txt)")
    ap.add_argument("--smiles-sha", default=SMILES_SHA, help="sha256 guard for --smiles")
    ap.add_argument("--latent-sample", default=None,
                    help="default <repo>/results/precomputed/zinc_latents_sample.npz")
    ap.add_argument("--reps", type=int, default=100_000, help="null Monte Carlo replicates (authors' default 1e5)")
    ap.add_argument("--null-seed", type=int, default=0, help="seed of the null Monte Carlo (authors' compare default 0)")
    ap.add_argument("--boot", type=int, default=1000, help="bootstrap replicates for the library SE of the floor")
    ap.add_argument("--boot-seed", type=int, default=20261011)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()
    t_start = time.time()
    _configure(args)

    outdir = os.path.abspath(args.outdir)
    check_outdir(outdir, [REPO, G, FLOOR_DIR, P1])
    os.makedirs(outdir, exist_ok=True)

    zrf = load_floor_module(FLOOR_MODULE)
    from scipy import stats

    smiles_sha = sha256(SMILES)
    if smiles_sha != SMILES_SHA:
        raise SystemExit(f"zinc_250k.txt sha256 changed: {smiles_sha}")
    ds = json.load(open(os.path.join(FLOOR_DIR, "dataset.json")))
    summ = json.load(open(os.path.join(FLOOR_DIR, "zinc_floor_summary.json")))

    provenance = {
        "script": os.path.abspath(__file__), "script_sha256": sha256(os.path.abspath(__file__)),
        "argv": sys.argv, "host": socket.gethostname(), "python": platform.python_version(),
        "numpy": np.__version__, "scipy": __import__("scipy").__version__,
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "floor_module": FLOOR_MODULE, "floor_module_sha256": FLOOR_MODULE_SHA,
        # Worktree of whatever generation supplied the floor module / the P1 blocks.
        "floor_worktree": _worktree_of(FLOOR_MODULE),
        "floor_worktree_head": git(_worktree_of(FLOOR_MODULE), "rev-parse", "HEAD"),
        "floor_worktree_status_lines": len(
            git(_worktree_of(FLOOR_MODULE), "status", "--short").splitlines()
        ),
        "repo_head": git(REPO, "rev-parse", "HEAD"),
        "repo_status_lines": len(git(REPO, "status", "--short").splitlines()),
        "p1_worktree_head": git(_worktree_of(P1), "rev-parse", "HEAD") if P1 else "",
        "floor_dir": FLOOR_DIR, "floor_scoring_provenance": summ.get("provenance"),
        "dataset_json": ds, "smiles_sha256_now": smiles_sha,
    }

    # ── per-task floors ─────────────────────────────────────────────────────
    floors = {}
    task_cdf = {}
    boot_rng = np.random.default_rng(args.boot_seed)
    sample_idx = np.random.RandomState(0).choice(10_000, 3000, replace=False)  # precompute_zinc_latents.py
    for t in TASKS:
        npy = os.path.join(FLOOR_DIR, f"scores_{t}.npy")
        meta = json.load(open(os.path.join(FLOOR_DIR, f"meta_{t}.json")))
        s = np.load(npy)
        if s.shape != (249_455,) or not np.isfinite(s).all():
            raise SystemExit(f"{t}: unexpected score vector {s.shape}, finite={np.isfinite(s).all()}")
        values, cum, n = zrf._unique_cdf(s)
        Gr = zrf.max_cdf(values, cum, n, B, "replace")
        Gn = zrf.max_cdf(values, cum, n, B, "norepl")
        e_r = zrf.exact_emax(values, cum, n, B)
        e_n = zrf.exact_emax_no_replacement(values, cum, n, B)
        assert abs(zrf.emax_from_cdf(values, Gn) - e_n) < 1e-12
        # SD of a single max_30 (exact, from the CDF)
        def sd_of(G):
            p = G - np.concatenate(([0.0], G[:-1]))
            m1 = float((values * p).sum()); m2 = float((values ** 2 * p).sum())
            return math.sqrt(max(m2 - m1 * m1, 0.0))
        sd_n, sd_r = sd_of(Gn), sd_of(Gr)
        q = {}
        for qq in (0.05, 0.5, 0.95):
            j = int(np.searchsorted(Gn, qq, side="left")); q[f"q{int(qq*100):02d}"] = float(values[min(j, values.size - 1)])
        # (b) null of the 40-seed mean (norepl), authors' sampler
        nm = zrf.null_mean_draws(values, Gn, 40, reps=args.reps, seed=args.null_seed)
        # (c) library bootstrap of E[max_30] (with-replacement formula on each resample)
        srt = np.sort(s)
        boots = np.empty(args.boot)
        for b in range(args.boot):
            # exact multinomial resample of the n molecules: counts of n uniform index draws
            cnt = np.bincount(boot_rng.integers(0, srt.size, srt.size), minlength=srt.size)
            c = np.cumsum(cnt) / srt.size
            P = c ** B
            boots[b] = float((srt * (P - np.concatenate(([0.0], P[:-1])))).sum())
        # pools: 10k prefix (cold-start pool) and the 3,000-molecule latent cache (k = 1 anchors)
        pv, pc, pn = zrf._unique_cdf(s[:10_000])
        sv, sc_, sn = zrf._unique_cdf(s[sample_idx])
        floors[t] = {
            "n": int(n), "smiles_sha256": meta["smiles_sha256"], "npy_sha256": sha256(npy),
            "scored_commit": meta["git_commit"], "scored_dirty": meta["git_dirty"],
            "n_invalid": meta["n_invalid"], "n_exceptions": meta["n_exceptions"], "n_corrupt": meta["n_corrupt"],
            "n_component_warnings": meta["n_component_warnings"], "n_nonfinite": meta["n_nonfinite"],
            "floor_norepl": e_n, "floor_replace": e_r, "norepl_minus_replace": e_n - e_r,
            "sd_max30_norepl": sd_n, "sd_max30_replace": sd_r, "max30_quantiles_norepl": q,
            "null40_mean_sd": float(nm.std(ddof=1)), "null40_q025": float(np.quantile(nm, 0.025)),
            "null40_q975": float(np.quantile(nm, 0.975)), "null40_mean_of_draws": float(nm.mean()),
            "library_boot_se": float(boots.std(ddof=1)), "library_boot_q025": float(np.quantile(boots, 0.025)),
            "library_boot_q975": float(np.quantile(boots, 0.975)), "library_boot_reps": args.boot,
            "floor_10k_prefix_norepl": zrf.exact_emax_no_replacement(pv, pc, pn, B),
            "floor_3k_latent_pool_norepl": zrf.exact_emax_no_replacement(sv, sc_, sn, B),
            "september_summary_emax30_replace": summ["tasks"][t]["emax"]["30"],
            "september_summary_emax30_norepl": summ["tasks"][t]["emax_norepl"]["30"],
        }
        assert abs(summ["tasks"][t]["emax_norepl"]["30"] - e_n) < 1e-12, (t, summ["tasks"][t]["emax_norepl"]["30"], e_n)
        assert abs(summ["tasks"][t]["emax"]["30"] - e_r) < 1e-12, (t, summ["tasks"][t]["emax"]["30"], e_r)
        task_cdf[t] = (s, values, cum, n, Gn, Gr, (sv, sc_, sn))
        print(f"floor {t}: norepl {e_n:.6f} replace {e_r:.6f} sd30 {sd_n:.4f} null40 sd {nm.std(ddof=1):.5f} boot se {boots.std(ddof=1):.5f}", flush=True)
    mean_floor = float(np.mean([floors[t]["floor_norepl"] for t in TASKS]))
    mean_floor_r = float(np.mean([floors[t]["floor_replace"] for t in TASKS]))

    # ── null caches (arm-independent) ──────────────────────────────────────
    null_cache: dict = {}

    def null_draws(t: str, n_seeds: int, scheme: str, seed: int | None = None) -> np.ndarray:
        """Per-task tests use the authors' single null seed (shared by all arms on a task).
        The cross-task average needs INDEPENDENT draws across tasks, so it passes a
        task-specific seed (same uniform stream mapped through different CDFs would be
        comonotonic and inflate the variance of the average)."""
        sd = args.null_seed if seed is None else seed
        key = (t, n_seeds, scheme, sd)
        if key not in null_cache:
            s, values, cum, n, Gn, Gr, _ = task_cdf[t]
            null_cache[key] = zrf.null_mean_draws(values, Gn if scheme == "norepl" else Gr, n_seeds,
                                                  reps=args.reps, seed=sd)
        return null_cache[key]

    def null_draws_budget(t: str, ns: list[int], seed_off: int) -> np.ndarray:
        """Null of the mean over seeds when seed i is the max of ns[i] draws (norepl)."""
        s, values, cum, n, Gn, Gr, _ = task_cdf[t]
        rng = np.random.default_rng(args.null_seed + 1000 + seed_off)
        tot = np.zeros(args.reps)
        for N, k in sorted(Counter(ns).items()):
            GN = zrf.max_cdf(values, cum, n, int(N), "norepl")
            tot += zrf.sample_max_from_cdf(values, GN, (args.reps, k), rng).sum(axis=1)
        return tot / len(ns)

    # ── arms ────────────────────────────────────────────────────────────────
    results = {"provenance": provenance, "floors": floors, "mean_floor_norepl": mean_floor,
               "mean_floor_replace": mean_floor_r, "arms": {}}
    for ai, (rid, label, kind, dirpath, solver, gen) in enumerate(ARMS):
        if kind == "ours":
            cells = load_ours(dirpath, solver)
        else:
            cells = load_ext(dirpath)
        arm = {"label": label, "kind": kind, "dir": dirpath, "solver": solver, "generation": gen,
               "n_files": len(cells), "conventions": {}}
        # descriptive provenance of the loaded files
        if kind == "ours":
            arm["provenance_states"] = {f"{c}|{d}": k for (c, d), k in Counter(v["prov"] for v in cells.values()).items()}
            arm["status_counts"] = dict(Counter(str(v["status"]) for v in cells.values()))
            arm["fallback_cells"] = int(sum(1 for v in cells.values() if v["fallbacks"] > 0))
            arm["len_all_scores"] = dict(Counter(int(v["all_scores"].size) for v in cells.values()))
            mts = sorted(v["mtime"] for v in cells.values())
            arm["mtime_range"] = [datetime.datetime.fromtimestamp(mts[0]).isoformat(timespec="minutes"),
                                  datetime.datetime.fromtimestamp(mts[-1]).isoformat(timespec="minutes")] if mts else None
            # cold-start check
            cs_max, cs_bad, cs_checked, cs_unchecked = 0.0, 0, 0, 0
            for (t, sd), v in cells.items():
                k = int(v["n_init"] or 0)
                a = v["all_scores"]
                if v["mode"] not in (None, "ours") or k <= 0 or a.size < k or not np.isfinite(a[:k]).all():
                    cs_unchecked += 1
                    continue
                idx = np.random.RandomState(sd).choice(10_000, k, replace=False)
                dlt = float(np.max(np.abs(a[:k] - task_cdf[t][0][idx])))
                cs_checked += 1
                cs_max = max(cs_max, dlt)
                cs_bad += int(dlt > 1e-9)
            arm["coldstart_check"] = {"checked": cs_checked, "unchecked": cs_unchecked,
                                      "cells_mismatch_gt_1e-9": cs_bad, "max_abs_diff": cs_max}
        else:
            ne = [v["n_evaluated"] for v in cells.values()]
            arm["n_evaluated_counts"] = dict(sorted(Counter(ne).items(), key=lambda kv: (kv[0] is None, kv[0])))
            arm["cells_with_30_evaluations"] = int(sum(1 for x in ne if x == 30))
            mts = sorted(v["mtime"] for v in cells.values())
            arm["mtime_range"] = [datetime.datetime.fromtimestamp(mts[0]).isoformat(timespec="minutes"),
                                  datetime.datetime.fromtimestamp(mts[-1]).isoformat(timespec="minutes")] if mts else None

        conv_list = ["itt"] if kind == "ext" else ["itt", "clean"]
        for conv in conv_list:
            per_task = {}
            for t in TASKS:
                best, nevals, imputed, n_missing_in_b = [], [], [], 0
                for sd in SEEDS:
                    v = cells.get((t, sd))
                    if v is None:
                        continue
                    if kind == "ext":
                        best.append(v["best"])
                        nevals.append(int(v["n_evaluated"]) if v["n_evaluated"] is not None else B)
                        continue
                    a = v["all_scores"]
                    head = a[:B]
                    if conv == "clean":
                        if not (v["status"] == "completed" and v["fallbacks"] == 0 and v["budget"] in (None, B)
                                and v["best_trace_len"] >= B and a.size >= B and np.isfinite(head).any()):
                            continue
                    n_missing_in_b += max(0, B - a.size)
                    if head.size and np.isfinite(head).any():
                        best.append(float(np.nanmax(head)))
                    else:
                        k = int(v["n_init"] or 5)
                        best.append(zrf.coldstart_best(task_cdf[t][0], sd, k=k))
                        imputed.append(sd)
                x = np.asarray(best, dtype=float)
                nseed = int(x.size)
                if nseed == 0:
                    per_task[t] = {"n": 0}
                    continue
                fl = floors[t]["floor_norepl"]
                obs = float(x.mean())
                tn = zrf.null_mean_test(null_draws(t, nseed, "norepl"), obs, nseed, args.null_seed)
                tr = zrf.null_mean_test(null_draws(t, nseed, "replace"), obs, nseed, args.null_seed)
                s_, values, cum, n, Gn, Gr, _ = task_cdf[t]
                sign = zrf.null_sign_test(values, Gn, x)
                if nseed > 1 and x.std(ddof=1) > 0:
                    tt = stats.ttest_1samp(x, fl)
                    se = float(x.std(ddof=1) / math.sqrt(nseed))
                    tcrit = float(stats.t.ppf(0.975, nseed - 1))
                    t_res = {"t": float(tt.statistic), "p_two_sided": float(tt.pvalue),
                             "ci95": [obs - fl - tcrit * se, obs - fl + tcrit * se]}
                else:
                    t_res = None
                row = {"n": nseed, "mean": obs, "sd": float(x.std(ddof=1)) if nseed > 1 else None,
                       "floor": fl, "diff": obs - fl,
                       "p_above": tn["p_above"], "p_below": tn["p_below"],
                       "p_above_replace": tr["p_above"], "p_below_replace": tr["p_below"],
                       "null_q025": tn["null_q025"], "null_q975": tn["null_q975"], "null_sd": tn["null_sd"],
                       "sign_test": sign, "ttest": t_res, "imputed_seeds": imputed,
                       "n_missing_in_budget": int(n_missing_in_b)}
                if kind == "ext":
                    nd = null_draws_budget(t, nevals, ai * 100 + TASKS.index(t))
                    tb = zrf.null_mean_test(nd, obs, nseed, args.null_seed)
                    row["budget_matched"] = {"mean_n_evaluated": float(np.mean(nevals)),
                                             "floor_budget_matched": float(nd.mean()),
                                             "p_above": tb["p_above"], "p_below": tb["p_below"]}
                per_task[t] = row
            # verdicts
            tasks_ok = [t for t in TASKS if per_task[t].get("n", 0) > 0]
            hol_a = zrf.holm([per_task[t]["p_above"] for t in tasks_ok])
            hol_b = zrf.holm([per_task[t]["p_below"] for t in tasks_ok])
            hol_t = zrf.holm([per_task[t]["ttest"]["p_two_sided"] if per_task[t]["ttest"] else float("nan")
                              for t in tasks_ok])
            for t, ha, hb, ht in zip(tasks_ok, hol_a, hol_b, hol_t):
                r = per_task[t]
                r["p_above_holm"], r["p_below_holm"] = ha, hb
                r["t_p_holm"] = ht
                # rule T: mean-only one-sample t-test vs the floor, two-sided 0.05, Holm over the 11 tasks
                r["verdict_T"] = (None if r["ttest"] is None or math.isnan(ht) else
                                  ("above" if (ht < args.alpha and r["diff"] > 0) else
                                   ("below" if (ht < args.alpha and r["diff"] < 0) else "indist")))
                r["verdict_P"] = "above" if ha < args.alpha / 2 else ("below" if hb < args.alpha / 2 else "indist")
                r["verdict_S"] = "above" if r["p_above"] < args.alpha else ("below" if r["p_below"] < args.alpha else "indist")
                r["verdict_t"] = (None if r["ttest"] is None else
                                  ("above" if (r["ttest"]["p_two_sided"] < args.alpha and r["diff"] > 0) else
                                   ("below" if (r["ttest"]["p_two_sided"] < args.alpha and r["diff"] < 0) else "indist")))
                if "budget_matched" in r:
                    bm = r["budget_matched"]
                    r["verdict_S_budget_matched"] = "above" if bm["p_above"] < args.alpha else ("below" if bm["p_below"] < args.alpha else "indist")
            # cross-task mean vs mean floor
            if len(tasks_ok) == len(TASKS):
                grand = float(np.mean([per_task[t]["mean"] for t in TASKS]))
                nulls = np.mean([null_draws(t, per_task[t]["n"], "norepl", seed=args.null_seed + 101 + TASKS.index(t))
                                 for t in TASKS], axis=0)
                ge = int(np.count_nonzero(nulls >= grand - 1e-12)); le = int(np.count_nonzero(nulls <= grand + 1e-12))
                dts = np.array([per_task[t]["diff"] for t in TASKS])
                wil = stats.wilcoxon(dts, method="exact") if np.any(dts != 0) else None
                cross = {"mean_of_task_means": grand, "mean_floor": mean_floor, "diff": grand - mean_floor,
                         "p_above": (1 + ge) / (args.reps + 1), "p_below": (1 + le) / (args.reps + 1),
                         "null_q025": float(np.quantile(nulls, 0.025)), "null_q975": float(np.quantile(nulls, 0.975)),
                         "task_level_signs": {"above": int((dts > 0).sum()), "below": int((dts < 0).sum())},
                         "task_level_wilcoxon_exact_p_two_sided": float(wil.pvalue) if wil is not None else None,
                         "counts_P": dict(Counter(per_task[t]["verdict_P"] for t in TASKS)),
                         "counts_S": dict(Counter(per_task[t]["verdict_S"] for t in TASKS)),
                         "counts_T": dict(Counter(per_task[t]["verdict_T"] for t in TASKS)),
                         "n_cells": int(sum(per_task[t]["n"] for t in TASKS))}
            else:
                cross = {"note": "not all tasks present", "n_cells": int(sum(per_task[t].get("n", 0) for t in TASKS))}
            arm["conventions"][conv] = {"per_task": per_task, "cross_task": cross}
        # k = 1 own-pool reference
        if rid == "convex_hull_k1":
            ref = {}
            for t in TASKS:
                s_, values, cum, n, Gn, Gr, (sv, sc_, sn) = task_cdf[t]
                F = sc_ / sn
                P25 = F ** 25
                diffs, exps = [], []
                for sd in SEEDS:
                    v = cells.get((t, sd))
                    if v is None:
                        continue
                    c = zrf.coldstart_best(s_, sd, k=5)
                    # E[max(c, M25)] with M25 = max of 25 i.i.d. draws from the 3k pool
                    below = sv <= c
                    p_le_c = P25[below][-1] if below.any() else 0.0
                    Pprev = np.concatenate(([0.0], P25[:-1]))
                    e = c * p_le_c + float((sv[~below] * (P25[~below] - Pprev[~below])).sum())
                    exps.append(e)
                    diffs.append(float(np.nanmax(v["all_scores"][:B])) - e)
                d_ = np.asarray(diffs)
                tt = stats.ttest_1samp(d_, 0.0)
                ref[t] = {"n": int(d_.size), "own_pool_reference_mean": float(np.mean(exps)),
                          "arm_minus_reference": float(d_.mean()), "t_p_two_sided": float(tt.pvalue)}
            dd = [ref[t]["arm_minus_reference"] for t in TASKS]
            arm["own_pool_reference"] = {"per_task": ref, "mean_over_tasks_arm_minus_reference": float(np.mean(dd)),
                                         "mean_reference": float(np.mean([ref[t]["own_pool_reference_mean"] for t in TASKS])),
                                         "assumption": "decode(encode(x)) == x for the 25 anchors drawn i.i.d. from the 3,000 cached latents (RandomState(0) choice of 3,000 from the 10k prefix); cold start = exact best@5"}
        results["arms"][rid] = arm
        itt = arm["conventions"]["itt"]["cross_task"]
        print(f"arm {rid:38s} n={itt.get('n_cells')} mean={itt.get('mean_of_task_means', float('nan')):.4f} "
              f"P={itt.get('counts_P')} S={itt.get('counts_S')}", flush=True)

    results["provenance"]["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    results["provenance"]["elapsed_s"] = time.time() - t_start
    results["provenance"]["repo_status_lines_after"] = len(git(REPO, "status", "--short").splitlines())
    results["provenance"]["floor_worktree_status_lines_after"] = len(
        git(_worktree_of(FLOOR_MODULE), "status", "--short").splitlines()
    )
    _add_aliases(results["provenance"])

    with open(os.path.join(outdir, "cr_floor.json"), "w") as f:
        json.dump(results, f, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    write_md(results, os.path.join(outdir, "cr_floor.md"))
    print(f"wrote {outdir}/cr_floor.json and cr_floor.md in {time.time() - t_start:.1f}s")
    return 0


def write_md(R: dict, path: str) -> None:
    """Render cr_floor.md. Reads the provenance keys under their current names (``floor_worktree_*``) or
    under the names they had before the rename (``e0_*``), so both old and new JSONs render."""
    L = []
    pv = R["provenance"]
    L.append("# Retrieval floor, camera-ready recomputation\n")
    L.append(f"Generated {pv['finished']} on {pv['host']} by `{pv['script']}` (sha256 `{pv['script_sha256'][:16]}`), "
             f"python {pv['python']}, numpy {pv['numpy']}, scipy {pv['scipy']}. Floor functions imported from "
             f"`{pv['floor_module']}` (sha256 `{pv['floor_module_sha256'][:16]}`, e0 worktree HEAD `{_pv(pv, 'floor_worktree_head')[:7]}`, "
             f"{_pv(pv, 'floor_worktree_status_lines')} status lines). Score vectors: commit `{R['floors']['adip']['scored_commit']}`, "
             f"dirty={R['floors']['adip']['scored_dirty']}; dataset sha256 `{pv['smiles_sha256_now'][:16]}` (unchanged). "
             f"Repo `{pv['repo_head'][:7]}` status lines before/after: {pv['repo_status_lines']}/{pv['repo_status_lines_after']}.\n")
    L.append("## Floors (N = 30)\n")
    L.append("| task | E[max30] no-repl. | E[max30] repl. | SD of one max30 | q05 / q50 / q95 of max30 | null 95 % band of a 40-seed mean | library bootstrap SE | E[max30] 10k prefix | E[max30] 3k latent pool |")
    L.append("|---|---:|---:|---:|---|---|---:|---:|---:|")
    for t in TASKS:
        f = R["floors"][t]; q = f["max30_quantiles_norepl"]
        L.append(f"| {t} | {f['floor_norepl']:.4f} | {f['floor_replace']:.4f} | {f['sd_max30_norepl']:.4f} | "
                 f"{q['q05']:.4f} / {q['q50']:.4f} / {q['q95']:.4f} | [{f['null40_q025']:.4f}, {f['null40_q975']:.4f}] | "
                 f"{f['library_boot_se']:.5f} | {f['floor_10k_prefix_norepl']:.4f} | {f['floor_3k_latent_pool_norepl']:.4f} |")
    L.append(f"\nMean over the 11 tasks: no-repl. {R['mean_floor_norepl']:.4f}, repl. {R['mean_floor_replace']:.4f}.\n")
    sym = {"above": "A", "indist": "i", "below": "B", None: "-"}
    L.append("Verdict rules. **P** = exact-null Monte Carlo test of the authors' compare (H0: each seed's best@30 is "
             "the max of 30 molecules drawn uniformly from ZINC-250k, no replacement), two one-sided p-values, Holm over "
             "the 11 tasks within an arm, two-sided alpha 0.05 (each side 0.025). **T** = one-sample t-test of the "
             "40-seed mean against the floor (mean only, arm's own variance), two-sided, Holm over 11 tasks, alpha 0.05. "
             "**S** = the September screen rule (exact null, one-sided 0.05 each side, uncorrected). A cell shows one "
             "letter when P and T agree, `P|T` when they differ. A = above, i = indistinguishable, B = below.\n")
    for conv in ("itt", "clean"):
        L.append(f"## Verdicts, convention `{conv}`\n")
        L.append("| arm | n cells | " + " | ".join(TASKS) + " | A/i/B (P) | A/i/B (T) | A/i/B (S) | cross-task mean | minus floor | cross null 95 % band | exact-null p_below / p_above (cross) | task signs +/- ; task-level Wilcoxon p |")
        L.append("|---|---:|" + "---|" * len(TASKS) + "---|---|---|---:|---:|---|---|---|")
        for rid, arm in R["arms"].items():
            if conv not in arm["conventions"]:
                continue
            c = arm["conventions"][conv]
            pt, cr = c["per_task"], c["cross_task"]
            cells = []
            for t in TASKS:
                r = pt[t]
                if r.get("n", 0) == 0:
                    cells.append("-"); continue
                a, b = sym[r["verdict_P"]], sym[r["verdict_T"]]
                cells.append(a if a == b else f"{a}&#124;{b}")
            cp = cr.get("counts_P", {}); cs = cr.get("counts_S", {}); ct = cr.get("counts_T", {})
            sg = cr.get("task_level_signs", {})
            wp = cr.get("task_level_wilcoxon_exact_p_two_sided")
            L.append(f"| {rid} | {cr.get('n_cells')} | " + " | ".join(cells) +
                     f" | {cp.get('above',0)}/{cp.get('indist',0)}/{cp.get('below',0)} | {ct.get('above',0)}/{ct.get('indist',0)}/{ct.get('below',0)} | {cs.get('above',0)}/{cs.get('indist',0)}/{cs.get('below',0)} | "
                     f"{cr.get('mean_of_task_means', float('nan')):.4f} | {cr.get('diff', float('nan')):+.4f} | "
                     f"[{cr.get('null_q025', float('nan')):.4f}, {cr.get('null_q975', float('nan')):.4f}] | "
                     f"{cr.get('p_below', float('nan')):.3g} / {cr.get('p_above', float('nan')):.3g} | "
                     f"{sg.get('above','')}/{sg.get('below','')} ; {wp if wp is None else format(wp, '.3g')} |")
        L.append("")
    L.append("## Per-task numbers (ITT): mean best@30 minus floor, raw p_below / p_above (exact null, no-repl.), t-test p\n")
    for rid, arm in R["arms"].items():
        c = arm["conventions"]["itt"]["per_task"]
        L.append(f"**{rid}** — {arm['label']}; files {arm['n_files']}; " +
                 (f"provenance {arm.get('provenance_states')}; status {arm.get('status_counts')}; fallback cells {arm.get('fallback_cells')}; cold start {arm.get('coldstart_check')}"
                  if arm["kind"] == "ours" else f"n_evaluated {arm.get('n_evaluated_counts')}"))
        L.append("")
        L.append("| task | n | mean | SD | floor | diff | 95 % CI of diff (t) | p_below | p_above | p_below Holm | p_above Holm | t p | t p Holm | P | T | S | budget-matched S |")
        L.append("|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|---|---|")
        for t in TASKS:
            r = c[t]
            if r.get("n", 0) == 0:
                L.append(f"| {t} | 0 |" + " |" * 15); continue
            tp = r["ttest"]["p_two_sided"] if r["ttest"] else float("nan")
            ci = r["ttest"]["ci95"] if r["ttest"] else [float("nan")] * 2
            L.append(f"| {t} | {r['n']} | {r['mean']:.4f} | {r['sd']:.4f} | {r['floor']:.4f} | {r['diff']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | "
                     f"{r['p_below']:.3g} | {r['p_above']:.3g} | {r['p_below_holm']:.3g} | {r['p_above_holm']:.3g} | {tp:.3g} | {r['t_p_holm']:.3g} | "
                     f"{r['verdict_P']} | {r['verdict_T']} | {r['verdict_S']} | {r.get('verdict_S_budget_matched', '')} |")
        if "own_pool_reference" in arm:
            o = arm["own_pool_reference"]
            L.append(f"\nOwn-pool reference (perfect reconstruction): mean reference {o['mean_reference']:.4f}; arm minus reference, mean over tasks {o['mean_over_tasks_arm_minus_reference']:+.4f}. Per task: " +
                     "; ".join(f"{t} {o['per_task'][t]['arm_minus_reference']:+.4f} (p {o['per_task'][t]['t_p_two_sided']:.2g})" for t in TASKS))
        L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
