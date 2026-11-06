#!/usr/bin/env python
"""Run ONE camera-ready cell (solver x task x seed, plus run overrides) in this process.

Rule R-1 of ``docs/camera_ready/RUN_RULES.md``: every GuacaMol arm runs through
``scripts.run_benchmark.run_benchmark()``, one fresh process per cell, with the
CLI defaults that produced the existing files::

    mode=ours, max_iter=25, n_cold_start=5, n_initial_points=10, latent_dim=128,
    subspace_dim=16, n_subspaces=16, cold_start_source=zinc, norm_mode=fixed,
    codec_type=selfies

The only allowed deviation is ``--run-overrides '{"subspace_dim": 8}'`` (or 64),
the R-1 exception for ``cma_es_subspace_sd8`` / ``cma_es_subspace_sd64`` in the
amendment log. It tags the file name exactly as the CLI does (``_sd8``,
``scripts/run_benchmark.py`` ``sd_tag``) and is recorded in the file.

This is the committed equivalent of the preparation driver
``/data/prusek/repro_check/rb_driver.py``, which reproduced the July files bit
for bit; nothing here touches ``run_benchmark()`` or the random streams.
What it adds around the call:

* **R-5, no clobbering.** An existing output is never replaced: a completed one
  is refused (exit 75), a non-completed one too (exit 77; the queue archives
  earlier attempts to ``attempts/`` before it reruns a cell). A cell with a
  crash record ``<out>.crash.json`` is terminal (R-8) and refused as well
  (exit 77), so a manual run cannot rerun a crashed cell behind the queue's
  back; only ``cr_queue.py --retry-crashed --amendment`` reruns it, after
  archiving the record. The cell is
  claimed with an ``O_EXCL`` file ``<out>.running`` (another holder: exit 76),
  and the result is published with ``os.link``, which fails instead of
  overwriting if a file appeared meanwhile. Files are written with mode 0644.
* **R-2, provenance.** With ``--expect-commit`` the cell refuses to start
  unless ``HEAD`` is that commit and the tree is clean (exit 78). After the
  run it adds, under ``provenance``: the full commit hash, the interpreter,
  the arm id, the run overrides, the host class and FQDN, the expected
  commit, the thread environment and the versions of botorch, gpytorch,
  linear_operator, rdkit, guacamol and LassoBench. These keys are additive;
  the top-level key set is the July files'.
* **Seed range of the arm.** With ``--arm`` the cell must belong to that arm
  of the manifest (``--manifest``, default ``scripts/cr/manifest.json``):
  the arm exists, its solver and run overrides are the cell's, and the seed
  is in the arm's seed range, i.e. its ``seeds`` field, or the manifest's
  ``seeds`` (42-81) when it has none (``arm_seeds``). Otherwise the cell is
  refused before anything is written (exit 78): a seed-82-121 control arm
  cannot produce a seed-42 cell, nor the reverse.

Usage (the queue ``scripts/cr/cr_queue.py`` builds this command)::

    CUDA_VISIBLE_DEVICES=<k> OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \\
    OPENBLAS_NUM_THREADS=4 TMPDIR=/data/prusek/tmp \\
      <python> scripts/cr/run_cell.py <solver> <poli_task> <seed> <output.json> \\
        [--arm ID] [--manifest PATH] [--run-overrides JSON] [--host-class gpu|cpu] \\
        [--expect-commit <40-hex sha>]

The process changes into the repository root first, so a relative output path
is relative to the repository, and inputs and git provenance are always the
repository's own.

Exit codes: 0 = the cell ran and its JSON was written (``status`` is
``completed``, or ``crashed`` when the solver raised -- ``run_benchmark``
records solver crashes in the file); 1 = hard crash, no JSON (traceback on
stderr); 2 = usage error; 75 = a completed file exists (left alone);
76 = another process holds the cell's claim; 77 = a non-completed file or a
crash record ``<out>.crash.json`` exists (the queue must archive it first);
78 = precondition failed (HEAD is not the expected commit, git failed, the
tree is dirty, the file name is not the cell's, or the cell is not in the
seed range, solver or run overrides of its ``--arm``).
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # `python scripts/cr/run_cell.py` without PYTHONPATH
    sys.path.insert(0, str(REPO_ROOT))
#: The committed manifest (the default of ``--manifest``).
MANIFEST_PATH = REPO_ROOT / "scripts" / "cr" / "manifest.json"

# Rule R-1 CLI defaults, kept in one place so a reader can check them against
# the rules file and against the provenance of the July files.
RUN_DEFAULTS = dict(
    mode="ours",
    max_iter=25,
    n_initial_points=10,
    n_cold_start=5,
    latent_dim=128,
    subspace_dim=16,
    n_subspaces=16,
    cold_start_source="zinc",
    decoder_weights_path=None,
    norm_mode="fixed",
    codec_type="selfies",
)
#: Total oracle budget of a cell (5 cold start + 25 steps); written as "budget".
BUDGET = RUN_DEFAULTS["n_cold_start"] + RUN_DEFAULTS["max_iter"]

#: The only R-1 arguments an arm may override, with their allowed values
#: (amendment log of RUN_RULES.md: subspace_dim 8 and 64 for the D-tilde arms).
ALLOWED_RUN_OVERRIDES: dict[str, frozenset] = {"subspace_dim": frozenset({8, 64})}

EXIT_OK = 0
EXIT_COMPLETED_EXISTS = 75
EXIT_BUSY = 76
EXIT_EXISTS_NOT_COMPLETED = 77
EXIT_PRECONDITION = 78

#: Mode of every file this script publishes (the CLI and the batch runner write 0644).
RESULT_MODE = 0o644
#: Seconds a git call may take before it counts as a failure (load on panda can exceed 150).
GIT_TIMEOUT_S = 120
#: Packages whose versions are added to provenance.package_versions.
EXTRA_PACKAGES = ("botorch", "gpytorch", "linear_operator", "rdkit", "guacamol", "LassoBench")
THREAD_ENV = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "CUDA_VISIBLE_DEVICES",
)


class RunOverrideError(ValueError):
    """An arm asked for an R-1 override that the rules do not allow."""


def validate_run_overrides(overrides: dict) -> dict:
    """Return a normalised copy of ``overrides`` or raise :class:`RunOverrideError`.

    An override must name an allowed argument, use an allowed value and differ
    from the R-1 default (an "override" equal to the default would give the
    same computation two arm ids).
    """
    if not isinstance(overrides, dict):
        raise RunOverrideError(f"run overrides must be a JSON object, got {overrides!r}")
    clean = {}
    for key, value in overrides.items():
        if key not in ALLOWED_RUN_OVERRIDES:
            raise RunOverrideError(
                f"override {key!r} is not allowed (allowed: {sorted(ALLOWED_RUN_OVERRIDES)})"
            )
        if isinstance(value, bool) or not isinstance(value, int):
            raise RunOverrideError(f"override {key}={value!r} must be an integer")
        if value not in ALLOWED_RUN_OVERRIDES[key]:
            raise RunOverrideError(
                f"override {key}={value} is not allowed "
                f"(allowed: {sorted(ALLOWED_RUN_OVERRIDES[key])})"
            )
        if value == RUN_DEFAULTS[key]:
            raise RunOverrideError(f"override {key}={value} equals the R-1 default")
        clean[key] = value
    return clean


def arm_seeds(man: dict, arm: dict) -> list[int]:
    """The seed range of a manifest arm: its ``seeds`` field if it has one,
    else the manifest's ``seeds`` (the paper protocol's 42-81).

    Raises ``ValueError`` unless the range is a non-empty list of distinct
    integers. The queue builds an arm's cells only on these seeds, and
    ``run_cell --arm`` refuses any other seed.
    """
    seeds = arm["seeds"] if "seeds" in arm else man["seeds"]
    where = f"arm {arm.get('arm')!r}" if "seeds" in arm else "the manifest"
    if not isinstance(seeds, list) or not seeds:
        raise ValueError(f"seeds of {where} must be a non-empty list, got {seeds!r}")
    bad = [s for s in seeds if isinstance(s, bool) or not isinstance(s, int)]
    if bad:
        raise ValueError(f"seeds of {where} must be integers, got {bad!r}")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds of {where} repeat a seed")
    return list(seeds)


def manifest_cell_problems(manifest: Path, arm_id: str, solver: str, overrides: dict,
                           seed: int) -> list[str]:
    """Problems that keep (solver, overrides, seed) from being a cell of arm
    ``arm_id`` of ``manifest`` (empty = it is one). Fail closed: an unreadable
    manifest or an unknown arm is a problem."""
    try:
        with Path(manifest).open() as fh:
            man = json.load(fh)
        arms = {a["arm"]: a for a in man["arms"]}
    except (OSError, ValueError, KeyError, TypeError) as e:
        return [f"cannot read the manifest {manifest}: {type(e).__name__}: {e}"]
    arm = arms.get(arm_id)
    if arm is None:
        return [f"arm {arm_id!r} is not in the manifest {manifest}"]
    problems = []
    if arm.get("solver") != solver:
        problems.append(f"arm {arm_id!r} runs solver {arm.get('solver')!r}, not {solver!r}")
    try:
        arm_overrides = validate_run_overrides(arm.get("run_overrides", {}))
    except RunOverrideError as e:
        problems.append(f"arm {arm_id!r}: {e}")
    else:
        if arm_overrides != overrides:
            problems.append(f"arm {arm_id!r} has run overrides {arm_overrides!r}, "
                            f"not {overrides!r}")
    try:
        seeds = arm_seeds(man, arm)
    except (ValueError, KeyError) as e:
        problems.append(f"{type(e).__name__}: {e}")
    else:
        if seed not in seeds:
            problems.append(f"seed {seed} is outside the seed range of arm {arm_id!r} "
                            f"({min(seeds)}-{max(seeds)}, {len(seeds)} seeds)")
    return problems


def cell_filename(solver: str, task: str, seed: int, subspace_dim: int = 16) -> str:
    """The name ``scripts/run_benchmark.py``'s CLI gives this cell.

    ``f"{mode}_{solver}_{task}_s{seed}_{latent_dim}d{sd_tag}.json"`` with the CLI
    default ``--latent-dim 128`` and ``sd_tag = f"_sd{N}"`` when
    ``--subspace-dim N`` differs from 16 -- byte for byte the CLI rule, so a
    camera-ready file and its July reference compare name for name. (The batch
    runner writes ``_256d`` instead, the codec's real dimension; E-1 files are
    matched by (solver, task, seed).)
    """
    sd_tag = f"_sd{subspace_dim}" if subspace_dim != 16 else ""
    return (
        f"{RUN_DEFAULTS['mode']}_{solver}_{task}_s{seed}_"
        f"{RUN_DEFAULTS['latent_dim']}d{sd_tag}.json"
    )


def claim_path(out: Path) -> Path:
    """``<out>.running``: exclusive claim held while a process computes the cell."""
    return Path(f"{out}.running")


def crash_record_path(out: Path) -> Path:
    """``<out>.crash.json``: the queue's terminal crash record of the cell (R-8)."""
    return Path(f"{out}.crash.json")


def is_completed(path: Path) -> bool:
    """True if ``path`` holds a readable result whose status is ``completed``."""
    try:
        with path.open() as fh:
            return json.load(fh).get("status") == "completed"
    except (OSError, json.JSONDecodeError):
        return False


def write_result(path: Path, payload: dict, *, force: bool = False) -> None:
    """Publish ``payload`` at ``path`` atomically, never replacing an existing file.

    The JSON goes to a temp file in the SAME directory (mode 0644, fsync), which
    is then hard-linked to ``path``: ``os.link`` raises ``FileExistsError`` if
    a file appeared at ``path`` in the meantime, so a result can never be
    overwritten (R-5). In that case our payload is kept next to it as
    ``<path>.collision-<pid>.json`` for inspection. ``force=True`` replaces
    instead (tests only).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        os.fchmod(fd, RESULT_MODE)
        with os.fdopen(fd, "w") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        if force:
            os.replace(tmp, path)
            return
        try:
            os.link(tmp, path)
        except FileExistsError:
            os.replace(tmp, f"{path}.collision-{os.getpid()}.json")
            raise
        os.unlink(tmp)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def git_tree_problems(repo: Path, expect_commit: str) -> list[str]:
    """Fail-closed check that ``repo`` is at ``expect_commit`` with a clean tree.

    Returns a list of problems (empty = OK). A git error or timeout is a
    problem, never "clean".
    """
    problems = []
    for args, what in ((["rev-parse", "HEAD"], "head"), (["status", "--porcelain"], "status")):
        try:
            r = subprocess.run(
                ["git", "--no-optional-locks", "-C", str(repo), *args],
                capture_output=True, text=True, timeout=GIT_TIMEOUT_S,
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            problems.append(f"git {' '.join(args)} failed: {type(e).__name__}: {e}")
            continue
        if r.returncode != 0:
            problems.append(f"git {' '.join(args)} failed (rc={r.returncode}): {r.stderr.strip()}")
        elif what == "head" and r.stdout.strip() != expect_commit:
            problems.append(f"HEAD is {r.stdout.strip()}, expected {expect_commit}")
        elif what == "status" and r.stdout.strip():
            problems.append(f"tree is dirty:\n{r.stdout.strip()}")
    return problems


def git_head_full(repo: Path) -> str:
    """Full HEAD hash, or "" if git fails (the queue then flags the cell)."""
    try:
        r = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return r.stdout.strip() if r.returncode == 0 else ""


def _package_version(name: str) -> str | None:
    import importlib.metadata

    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def extra_provenance(*, arm: str | None, run_overrides: dict, host_class: str | None,
                     expect_commit: str | None) -> dict:
    """Provenance keys run_cell adds to the file (all under ``provenance``)."""
    return {
        "git_commit_full": git_head_full(REPO_ROOT),
        "expect_commit": expect_commit,
        "python_executable": sys.executable,
        "cr_arm": arm,
        "run_overrides": dict(run_overrides),
        "host_class": host_class,
        "host_fqdn": socket.getfqdn(),
        "thread_env": {k: os.environ.get(k) for k in THREAD_ENV},
        "cr_run_cell": "scripts/cr/run_cell.py",
    }


def _raise_on_signal(signum, _frame):
    """SIGTERM/SIGHUP -> SystemExit, so the claim's ``finally`` still runs."""
    raise SystemExit(128 + signum)


def run_cell(solver: str, task: str, seed: int, out: Path, *,
             arm: str | None = None, run_overrides: dict | None = None,
             host_class: str | None = None, expect_commit: str | None = None,
             force: bool = False, manifest: Path | None = None) -> int:
    overrides = validate_run_overrides(run_overrides or {})
    subspace_dim = overrides.get("subspace_dim", RUN_DEFAULTS["subspace_dim"])
    expected_name = cell_filename(solver, task, seed, subspace_dim)
    if out.name != expected_name:
        print(f"REFUSED: output file name {out.name!r} is not {expected_name!r}",
              file=sys.stderr, flush=True)
        return EXIT_PRECONDITION
    tag = f"{solver} {task} s{seed}" + (f" {overrides}" if overrides else "")
    if arm is not None and manifest is not None:
        problems = manifest_cell_problems(manifest, arm, solver, overrides, seed)
        if problems:
            print(f"REFUSED {tag} (arm {arm}): " + "; ".join(problems),
                  file=sys.stderr, flush=True)
            return EXIT_PRECONDITION

    if not force:
        if out.exists() and is_completed(out):
            print(f"SKIP {tag}: completed file exists -> {out}", flush=True)
            return EXIT_COMPLETED_EXISTS
        if crash_record_path(out).exists():
            print(f"REFUSED {tag}: a crash record exists at {crash_record_path(out)}; the "
                  f"cell is terminal (R-8) and is rerun only through cr_queue.py "
                  f"--retry-crashed --amendment, which archives the record first", flush=True)
            return EXIT_EXISTS_NOT_COMPLETED
        if out.exists():
            print(f"REFUSED {tag}: a non-completed file exists at {out}; earlier attempts "
                  f"are archived by the queue, never overwritten (R-8)", flush=True)
            return EXIT_EXISTS_NOT_COMPLETED

    claim = claim_path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, RESULT_MODE)
    except FileExistsError:
        print(f"BUSY {tag}: {claim} exists", flush=True)
        return EXIT_BUSY
    with os.fdopen(fd, "w") as fh:
        fh.write(f"{os.getpid()} {socket.gethostname()} "
                 f"{datetime.datetime.now().isoformat(timespec='seconds')}\n")
    try:
        if expect_commit is not None:
            problems = git_tree_problems(REPO_ROOT, expect_commit)
            if problems:
                print(f"REFUSED {tag}: " + "; ".join(problems), file=sys.stderr, flush=True)
                return EXIT_PRECONDITION

        from scripts.run_benchmark import run_benchmark

        kwargs = dict(RUN_DEFAULTS)
        kwargs.update(overrides)
        res = run_benchmark(solver_name=solver, task_name=task, seed=seed, **kwargs)
        res["budget"] = kwargs["n_cold_start"] + kwargs["max_iter"]
        prov = res.setdefault("provenance", {})
        prov.update(extra_provenance(arm=arm, run_overrides=overrides,
                                     host_class=host_class, expect_commit=expect_commit))
        prov.setdefault("package_versions", {}).update(
            {name: _package_version(name) for name in EXTRA_PACKAGES}
        )
        write_result(out, res, force=force)
        print(
            f"DONE {tag} best={res['best_score']:.4f} status={res['status']} "
            f"fallbacks={res['surrogate_fallback_count']} "
            f"elapsed={res['elapsed_seconds']:.1f}s -> {out}",
            flush=True,
        )
        return EXIT_OK
    finally:
        with contextlib.suppress(OSError):
            os.unlink(claim)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("solver", help="solver name accepted by run_benchmark._create_solver")
    ap.add_argument("task", help="poli task id, e.g. ranolazine_mpo")
    ap.add_argument("seed", type=int)
    ap.add_argument("out", type=Path, help="output JSON path (relative = to the repository)")
    ap.add_argument("--arm", default=None,
                    help="manifest arm id (recorded in provenance); the cell must be in the "
                         "arm's solver, run overrides and seed range")
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH,
                    help="manifest the --arm is checked against (default: the committed "
                         "scripts/cr/manifest.json)")
    ap.add_argument("--run-overrides", default="{}",
                    help='JSON object of R-1 overrides, e.g. \'{"subspace_dim": 8}\'')
    ap.add_argument("--host-class", choices=("gpu", "cpu"), default=None)
    ap.add_argument("--expect-commit", default=None,
                    help="full 40-hex commit; refuse to run unless HEAD is it and the tree is clean")
    ap.add_argument("--force", action="store_true",
                    help="replace an existing file (R-5 says never; for tests only)")
    args = ap.parse_args(argv)
    try:
        overrides = json.loads(args.run_overrides)
        validate_run_overrides(overrides)
    except (json.JSONDecodeError, RunOverrideError) as e:
        ap.error(f"--run-overrides: {e}")
    if args.expect_commit is not None and (
        len(args.expect_commit) != 40
        or any(c not in "0123456789abcdef" for c in args.expect_commit)
    ):
        ap.error("--expect-commit must be a full 40-character lowercase hex hash")
    return run_cell(
        args.solver, args.task, args.seed, args.out.resolve(),
        arm=args.arm, run_overrides=overrides, host_class=args.host_class,
        expect_commit=args.expect_commit, force=args.force,
        manifest=args.manifest.resolve(),
    )


if __name__ == "__main__":
    os.chdir(REPO_ROOT)  # inputs, relative paths and git provenance are the repo's own
    for _sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(_sig, _raise_on_signal)
    sys.exit(main())
