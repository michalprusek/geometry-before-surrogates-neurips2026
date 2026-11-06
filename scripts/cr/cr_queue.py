#!/usr/bin/env python
"""Camera-ready cell queue: run the missing cells of ``scripts/cr/manifest.json``.

One fresh process per cell via ``scripts/cr/run_cell.py`` (rule R-1), N worker
slots per GPU with ``CUDA_VISIBLE_DEVICES`` pinned and lean thread pools
(rule R-2: OMP/MKL/OpenBLAS and TensorFlow's intra/inter-op pools = 4), one
commit, a clean tree, and every result validated before it counts.

Examples (from anywhere; the queue changes into ``--repo`` first)::

    # the GPU campaign in one queue: wave 2 first, then 1, 3, 3b, 4; the arms of
    # a wave are interleaved, the waves run one after another
    python scripts/cr/cr_queue.py --waves 2,1,3,3b,4 --expect-commit <40-hex sha>

    # everything still missing in wave 1, 4 workers per GPU on both GPUs
    python scripts/cr/cr_queue.py --wave 1 --expect-commit <40-hex sha>

    # two arms, two seeds, into a smoke directory instead of cr2026
    python scripts/cr/cr_queue.py --arms turbo_box_faithful,cma_es_subspace_sd8 \\
        --tasks rano --seeds 42-43 --output-root results/guacamol/cr_smoke2 \\
        --expect-commit <sha>

    # the CPU-only arms (GaBO family) on kajman: no GPU, 1 process x 4 threads
    python scripts/cr/cr_queue.py --host-class cpu --cpu-workers 1 --expect-commit <sha>

    # the control block (amendment of 2026-10-12) from its own commit, into its
    # own root: wave c1 (C1, C2a, C2b on seeds 42-81), then wave c2 (the six
    # decomposition arms as <arm>__s82 on seeds 82-121)
    python scripts/cr/cr_queue.py --waves c1,c2 --gpus 1 --workers-per-gpu 1 \\
        --output-root results/guacamol/cr2026ctl --expect-commit <sha>

    # control C4 (amendment of 2026-10-14) from its own commit, into the same
    # root: wave c3 (hvarfner_scaled_gp_sphere_lbfgs on seeds 42-81)
    python scripts/cr/cr_queue.py --waves c3 --gpus 1 --workers-per-gpu 1 \\
        --output-root results/guacamol/cr2026ctl --expect-commit <sha>

    # preflight only: what would run, and would anything refuse?
    python scripts/cr/cr_queue.py --wave 2 --expect-commit <sha> --dry-run

Rules the queue enforces (``docs/camera_ready/RUN_RULES.md``):

* **R-2.** ``--expect-commit`` (full hash) is required; ``HEAD`` must equal it
  and ``git status --porcelain`` must be empty, before the first cell AND before
  every cell (a tree that turns dirty stops the queue). Git errors fail closed.
  GPU-class arms run only on the manifest's ``gpu_hosts``, CPU-class arms only
  on ``cpu_hosts`` and with no GPU visible.
* **R-5.** A completed, valid cell is never rerun or overwritten. A result
  counts as OK only if ``status == "completed"``, 30 evaluations, solver, task,
  seed, ``subspace_dim``, arm id and run overrides match the cell, and
  provenance records ``git_commit_full == --expect-commit`` and
  ``git_dirty is False``. A completed file that fails these checks is
  *foreign*: the queue reports it and refuses to start (it is never skipped,
  never rerun). One queue per output root (``flock`` on ``<root>/.queue.lock``),
  each cell claimed by its process (``<out>.running``), results published
  without clobbering (``os.link``), selections de-duplicated.
  Every arm runs its own seed range: its ``seeds`` field, else the
  manifest's ``seeds`` (42-81). ``--seeds`` can only narrow it: a seed
  outside the range of a selected arm refuses the selection, and every child
  gets ``--manifest`` and refuses a cell outside its arm's solver, run
  overrides or seed range (run_cell exit 78). Two arms may share a solver and
  run overrides only with disjoint seed ranges (the control block's
  ``<arm>__s82`` arms), so no cell is computed under two arm ids.
  A child that published a valid result counts as completed even if it then
  exited non-zero or outlived its timeout (a crash in interpreter teardown):
  the odd exit is logged in ``queue.log`` and ``cells.jsonl`` and no crash
  record is written. A crash record found next to a valid completed result
  (left by an older queue) does not make the cell crashed: it counts as
  completed, with a warning, and is never retried.
* **R-8.** A crash is terminal: a hard crash, a timeout, a crash recorded by
  ``run_benchmark`` in the file, an invalid result, or a claim left by a dead
  process all leave ``<out>.crash.json`` (traceback tail + log path), and the
  cell is not run again unless ``--retry-crashed --amendment "<date: reason>"``
  is given; the earlier output is then moved to ``<arm dir>/attempts/`` (never
  deleted) and the amendment is written to the new attempt's records. An
  existing crash record is archived, never overwritten.
  A cell whose child the queue itself killed because the queue was stopped by
  SIGTERM/SIGHUP/SIGINT is *interrupted*, not crashed: ``<out>.interrupted.json``
  records it, and the next queue archives that record to ``attempts/`` and
  runs the cell (logged as RESUME). A child that crashed or timed out on its
  own while the queue was stopping is a crash, as at any other time. Every
  cell is deterministic given (arm, task, seed, commit).
  The queue stops (no further cells start) when run_cell refuses a cell
  (usage error, precondition, unexpected existing file), when a result has
  invalid provenance, when the tree turns dirty, and after 3 failed cells in
  a row (hard crash, timeout, crash inside the solver, or failed start: a
  broken environment or a hung GPU, not bad cells).

Records: ``logs/cr/<timestamp>_<tag>/`` (gitignored) holds ``queue.log``,
``cells.jsonl`` (one line per finished cell, incl. oracle failures and the
fallback count), one log per cell and ``summary.json`` (argv, host, commit,
counts, ``nvidia-smi`` and ``who`` at start, load average, sha256 of the
inputs and of the manifest).

Reading results (for the table generator): the result of a cell is
``<root>/<arm id>/ours_<solver>_<poli task>_s<seed>_128d.json`` (``_sd8`` /
``_sd64`` before ``.json`` for the D-tilde arms), i.e. only files matching
``ours_*_128d(_sd[0-9]+)?.json`` at the top level of an arm directory. Skip
``*.crash.json``, ``*.interrupted.json``, ``*.collision-*.json``, the
``attempts/`` subdirectory and dot files. Group by ``provenance.cr_arm`` (=
the directory), not by ``solver`` (``random_search__s82/`` holds files whose
``solver`` is ``random_search``, seeds 82-121). A cell is completed iff its result file
passes the queue's validation (``validate_result``); a crash record next to a
valid result is not a crash. See ``docs/camera_ready/README.md``.

Exit codes: 0 = every selected cell ended OK (or nothing to do; cells found
busy -- claimed by a live process -- when the queue lists them are skipped and
logged and do not change the exit code); 1 = some cell crashed, timed out, was
invalid or refused, was not started, or became busy between the listing and
its start (run_cell exit 76); 2 = usage error;
10 = tree dirty or HEAD is not ``--expect-commit``; 11 = git failed;
12 = another queue holds the output root; 13 = host not allowed for the arm
class; 14 = foreign completed files in the selection; 15 = stale claims that
cannot be resolved (another host's); 16 = required inputs missing or TMPDIR
unsafe; 128 + signal number when stopped by a signal.

``--allow-dirty``, ``--allow-any-host`` and ``--allow-missing-inputs`` exist for
tests and deliberate experiments OUTSIDE the camera-ready generation; a run
using them records so in ``summary.json`` and must not produce printed cells.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import errno
import fcntl
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import Counter, deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # `python scripts/cr/cr_queue.py` without PYTHONPATH
    sys.path.insert(0, str(REPO_ROOT))

from scripts.cr.run_cell import (  # noqa: E402  (stdlib + json only; no torch)
    BUDGET,
    EXIT_BUSY,
    EXIT_COMPLETED_EXISTS,
    EXIT_EXISTS_NOT_COMPLETED,
    EXIT_PRECONDITION,
    RunOverrideError,
    arm_seeds,
    cell_filename,
    claim_path,
    crash_record_path,
    validate_run_overrides,
)

DEFAULT_MANIFEST = REPO_ROOT / "scripts" / "cr" / "manifest.json"
RUN_CELL = Path("scripts") / "cr" / "run_cell.py"
LOG_ROOT = Path("logs") / "cr"
LOCK_NAME = ".queue.lock"
ATTEMPTS_DIR = "attempts"
#: tail of the child's output kept in the crash sidecar (the full log is linked)
CRASH_TAIL_CHARS = 8000
GIT_TIMEOUT_S = 120
#: c1 and c2 are the control block of the 2026-10-12 amendment (run from its
#: own commit, tag cr-controls-2026-10-12, into its own output root); c3 is
#: control C4 of the 2026-10-14 amendment (tag cr-controls-lbfgs-2026-10-14,
#: the same root), the printed shell x continuous cell since 2026-11-06.
VALID_WAVES = ("1", "2", "3", "3b", "4", "c1", "c2", "c3")
HOST_CLASSES = ("gpu", "cpu")
ARM_ID_RE = re.compile(r"^[a-z0-9_]+$")
#: Inputs every cell reads; their sha256 goes into summary.json and queue.log.
REQUIRED_INPUTS = (
    "weights/SELFIES-VAE-state-dict.pt",
    "data/zinc_10k.txt",
    "results/precomputed/zinc_per_coord_bounds.json",
    "results/precomputed/zinc_latents_sample.npz",
)
#: Directory whose files are all hashed as well (codec bounds, mean norm, ...).
HASHED_INPUT_DIRS = ("results/precomputed",)

#: run_cell's argparse usage-error code (it never computes anything then).
EXIT_USAGE = 2
#: run_cell exit codes that mean "refused, computed nothing": a file found at
#: ``<out>`` after such an exit was not written by that child.
REFUSAL_CODES = frozenset({EXIT_USAGE, EXIT_COMPLETED_EXISTS, EXIT_BUSY,
                           EXIT_EXISTS_NOT_COMPLETED, EXIT_PRECONDITION})
#: Return codes of a child that died of the queue's own kill (SIGTERM, then
#: SIGKILL; run_cell turns SIGTERM into exit 128 + 15).
KILL_RETURNCODES = frozenset({-signal.SIGTERM, -signal.SIGKILL, 128 + signal.SIGTERM})
#: Stop the queue after this many failed cells in a row (a broken environment
#: or a hung GPU, not bad cells).
MAX_CONSECUTIVE_FAILURES = 3
#: Cell statuses that count towards the circuit breaker; any other status resets it.
BREAKER_STATUSES = frozenset({"crashed", "timeout", "crashed_in_solver", "failed_start"})

EXIT_OK = 0
EXIT_FAILURES = 1
EXIT_DIRTY = 10
EXIT_GIT = 11
EXIT_LOCKED = 12
EXIT_HOST = 13
EXIT_FOREIGN = 14
EXIT_STALE = 15
EXIT_INPUTS = 16


class Refused(Exception):
    """A precondition failed; carries the exit code."""

    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code


class GitError(RuntimeError):
    pass


class _Interrupted(BaseException):
    def __init__(self, signum: int):
        super().__init__(signum)
        self.signum = signum


class WorkList:
    """Thread-safe hand-out of the cells (a deque behind a lock)."""

    def __init__(self, cells: list[dict]):
        self._items = deque(cells)
        self._lock = threading.Lock()

    def pop(self) -> dict | None:
        with self._lock:
            return self._items.popleft() if self._items else None

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


# --------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------


class ManifestError(ValueError):
    pass


def validate_manifest(man: dict) -> None:
    """Raise :class:`ManifestError` unless the manifest is well formed.

    Arm ids are unique and usable as directory names; no two arms run the same
    (solver, overrides) computation on a common seed (two arms may share a
    solver and overrides only with disjoint seed ranges, as the control
    block's ``<arm>__s82`` arms do); waves, host classes, timeouts, run
    overrides and seed ranges are valid; tasks and seeds are unique and every
    task is mapped. An arm's seed range is its ``seeds`` field, else the
    manifest's ``seeds`` (``run_cell.arm_seeds``).
    """
    for key in ("tasks", "task_poli_ids", "seeds", "arms", "output_root",
                "gpu_hosts", "cpu_hosts"):
        if key not in man:
            raise ManifestError(f"manifest is missing '{key}'")
    if len(set(man["tasks"])) != len(man["tasks"]):
        raise ManifestError("duplicate task codes")
    if len(set(man["seeds"])) != len(man["seeds"]):
        raise ManifestError("duplicate seeds")
    unmapped = [t for t in man["tasks"] if t not in man["task_poli_ids"]]
    if unmapped:
        raise ManifestError(f"tasks without a poli id: {unmapped}")
    ids = set()
    computations: dict[tuple, list[tuple[str, set[int]]]] = {}
    for arm in man["arms"]:
        for key in ("arm", "solver", "run_overrides", "wave", "host_class", "timeout_s"):
            if key not in arm:
                raise ManifestError(f"arm {arm.get('arm', arm)!r} is missing '{key}'")
        arm_id = arm["arm"]
        if not ARM_ID_RE.match(arm_id):
            raise ManifestError(f"arm id {arm_id!r} must match {ARM_ID_RE.pattern}")
        if arm_id in ids:
            raise ManifestError(f"duplicate arm id {arm_id!r}")
        ids.add(arm_id)
        try:
            overrides = validate_run_overrides(arm["run_overrides"])
        except RunOverrideError as e:
            raise ManifestError(f"arm {arm_id!r}: {e}") from e
        try:
            seeds = set(arm_seeds(man, arm))
        except ValueError as e:
            raise ManifestError(f"arm {arm_id!r}: {e}") from e
        key = (arm["solver"], tuple(sorted(overrides.items())))
        for other_id, other_seeds in computations.get(key, []):
            common = seeds & other_seeds
            if common:
                raise ManifestError(
                    f"arm {arm_id!r} repeats the computation {key} of arm {other_id!r} "
                    f"on {len(common)} seed(s), e.g. {sorted(common)[:3]}")
        computations.setdefault(key, []).append((arm_id, seeds))
        if str(arm["wave"]) not in VALID_WAVES:
            raise ManifestError(f"arm {arm_id!r}: wave {arm['wave']!r} not in {VALID_WAVES}")
        if arm["host_class"] not in HOST_CLASSES:
            raise ManifestError(f"arm {arm_id!r}: host_class {arm['host_class']!r}")
        if not isinstance(arm["timeout_s"], int) or arm["timeout_s"] <= 0:
            raise ManifestError(f"arm {arm_id!r}: timeout_s must be a positive int")


def load_manifest(path: Path) -> dict:
    with path.open() as fh:
        man = json.load(fh)
    try:
        validate_manifest(man)
    except ManifestError as e:
        raise SystemExit(f"manifest {path}: {e}") from e
    return man


def dedupe(items):
    """Order-preserving de-duplication."""
    return list(dict.fromkeys(items))


def parse_int_list(spec: str | None, default: list[int]) -> list[int]:
    """``'42-81'``, ``'42,43'`` and ``'42-44,81'`` all work; None = default; de-duplicated."""
    if not spec:
        return dedupe(default)
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return dedupe(out)


def format_seeds(seeds) -> str:
    """``[42, 43, 44, 81]`` -> ``'42-44,81'`` (sorted, consecutive runs joined)."""
    runs: list[list[int]] = []
    for s in sorted(set(seeds)):
        if runs and s == runs[-1][1] + 1:
            runs[-1][1] = s
        else:
            runs.append([s, s])
    return ",".join(f"{a}-{b}" if a != b else f"{a}" for a, b in runs)


def seeds_for_arms(man: dict, arms: list[dict],
                   requested: list[int] | None) -> dict[str, list[int]]:
    """The seeds each arm runs: its own range (``run_cell.arm_seeds``: the arm's
    ``seeds`` field, else the manifest's 42-81), or the ``--seeds`` list.

    Fail closed: a requested seed outside a selected arm's range refuses the
    selection (``SystemExit``), so no queue can write a cell its arm does not
    own (``run_cell --arm`` would refuse it too). Arms of different seed
    ranges are therefore selected together only without ``--seeds``.
    """
    out: dict[str, list[int]] = {}
    for arm in dedupe_arms(arms):
        own = arm_seeds(man, arm)
        if requested is None:
            out[arm["arm"]] = own
            continue
        outside = [s for s in dedupe(requested) if s not in own]
        if outside:
            raise SystemExit(
                f"--seeds {format_seeds(outside)} outside the seed range of arm "
                f"{arm['arm']!r} ({format_seeds(own)}); select arms of one seed range, "
                "or drop --seeds to run every arm on its own range")
        out[arm["arm"]] = dedupe(requested)
    return out


def select_arms(man: dict, *, wave: str | None, arm_ids: list[str] | None,
                host_class: str, waves: list[str] | None = None) -> list[dict]:
    """Arms filtered by wave, explicit ids (validated, de-duplicated) and host class.

    ``waves`` (an ordered list, e.g. ``["2", "1", "3", "3b", "4"]``) selects the
    arms of those waves, wave by wave in that order and in manifest order
    within a wave.
    """
    arms = list(man["arms"])
    if waves:
        unknown = [w for w in waves if w not in VALID_WAVES]
        if unknown:
            raise SystemExit(f"unknown wave(s) {unknown}; valid: {', '.join(VALID_WAVES)}")
        arms = [a for w in dedupe(waves) for a in man["arms"] if str(a["wave"]) == w]
    if arm_ids:
        by_id = {a["arm"]: a for a in man["arms"]}
        wanted = dedupe(arm_ids)
        missing = [s for s in wanted if s not in by_id]
        if missing:
            raise SystemExit(f"not in the manifest: {', '.join(missing)}")
        arms = [by_id[s] for s in wanted]
        if host_class != "all":
            other = [a["arm"] for a in arms if a["host_class"] != host_class]
            if other:
                raise SystemExit(f"arm(s) {', '.join(other)} are not {host_class}-class; "
                                 f"select them with the matching --host-class")
    if wave is not None:
        arms = [a for a in arms if str(a["wave"]) == str(wave)]
    if host_class != "all":
        arms = [a for a in arms if a["host_class"] == host_class]
    return arms


def output_root_path(man: dict, output_root: str | None, repo: Path) -> Path:
    """Absolute output root; a relative one is relative to the repository."""
    root = Path(output_root or man["output_root"])
    return root if root.is_absolute() else (repo / root)


def cell_output(arm: dict, task: str, seed: int, man: dict, output_root: str | None,
                *, repo: Path = REPO_ROOT) -> Path:
    """``<root>/<arm id>/<CLI file name>``, absolute."""
    subspace_dim = arm["run_overrides"].get("subspace_dim", 16)
    poli = man["task_poli_ids"][task]
    return (output_root_path(man, output_root, repo) / arm["arm"]
            / cell_filename(arm["solver"], poli, seed, subspace_dim))


def sidecar_path(out: Path) -> Path:
    """``<out>.crash.json`` (the name run_cell checks before it runs a cell)."""
    return crash_record_path(out)


def interrupted_path(out: Path) -> Path:
    return Path(f"{out}.interrupted.json")


# --------------------------------------------------------------------------
# result validation and cell classification
# --------------------------------------------------------------------------


def validate_result(res: dict, cell: dict, *, expect_commit: str | None) -> list[str]:
    """Problems that keep a result file from counting as OK (empty = valid)."""
    problems = []
    if res.get("status") != "completed":
        problems.append(f"status={res.get('status')!r} crash_info={res.get('crash_info')!r}")
    if res.get("n_evaluated") != BUDGET or len(res.get("all_scores") or []) != BUDGET:
        problems.append(f"n_evaluated={res.get('n_evaluated')} "
                        f"len(all_scores)={len(res.get('all_scores') or [])}, expected {BUDGET}")
    if res.get("budget") != BUDGET:
        problems.append(f"budget={res.get('budget')}, expected {BUDGET}")
    got = (res.get("solver"), res.get("task"), res.get("seed"), res.get("subspace_dim"))
    want = (cell["solver"], cell["poli_task"], cell["seed"], cell["subspace_dim"])
    if got != want:
        problems.append(f"(solver, task, seed, subspace_dim) {got} != cell {want}")
    prov = res.get("provenance") or {}
    if prov.get("cr_arm") != cell["arm"]:
        problems.append(f"provenance.cr_arm={prov.get('cr_arm')!r} != arm {cell['arm']!r}")
    if prov.get("run_overrides") != cell["run_overrides"]:
        problems.append(f"provenance.run_overrides={prov.get('run_overrides')!r} "
                        f"!= {cell['run_overrides']!r}")
    if expect_commit is not None:
        short = prov.get("git_commit") or ""
        if prov.get("git_commit_full") != expect_commit or not short \
                or not expect_commit.startswith(short):
            problems.append(f"provenance commit {short!r}/{prov.get('git_commit_full')!r} "
                            f"!= expected {expect_commit}")
        if prov.get("git_dirty") is not False:
            problems.append(f"provenance git_dirty={prov.get('git_dirty')!r}")
    return problems


def read_json(path: Path) -> dict | None:
    try:
        with path.open() as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def pid_is_run_cell(pid: int) -> bool:
    """True if ``pid`` is alive (and, where /proc exists, is a run_cell process)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    cmdline = Path(f"/proc/{pid}/cmdline")
    if cmdline.exists():
        try:
            return b"run_cell.py" in cmdline.read_bytes()
        except OSError:
            return True
    return True


def claim_state(claim: Path) -> tuple[str, str]:
    """``("alive" | "stale" | "foreign_host", content)`` of a ``<out>.running`` claim."""
    try:
        content = claim.read_text().strip()
    except OSError:
        return "stale", ""
    parts = content.split()
    try:
        pid, host = int(parts[0]), parts[1]
    except (IndexError, ValueError):
        return "stale", content
    if host.split(".")[0] != socket.gethostname().split(".")[0]:
        return "foreign_host", content
    return ("alive" if pid_is_run_cell(pid) else "stale"), content


def build_cells(man: dict, arms: list[dict], *, tasks: list[str], seeds: list[int] | None,
                output_root: str | None, retry_crashed: bool, interleave: bool,
                expect_commit: str | None, repo: Path = REPO_ROOT,
                group_by_wave: bool = False) -> tuple[list[dict], dict]:
    """Cells to run, plus what was skipped and why. Read-only (no side effects).

    Seeds: each arm's own range, or ``seeds`` when given, which must lie in
    the range of every selected arm (``seeds_for_arms``, fail closed).

    Per cell, in this order: a live claim -> ``busy`` (skipped, logged); a claim
    whose process is gone -> ``stale`` (resolved by the queue under its lock);
    a completed file that validates -> ``completed`` (a crash record next to it
    is reported in ``completed_with_crash_record`` and changes nothing: the
    child published a valid result before it died); a completed file that
    fails validation -> ``foreign`` (the queue refuses to start), unless a
    crash record sits next to it; a crash record or a non-completed file ->
    ``crashed_kept`` (terminal) or ``crashed_retry`` with ``--retry-crashed``;
    an interrupted record only -> run again (``resume``); nothing -> run.

    Cell order: arm by arm without ``interleave``; with it, round-robin over
    the arms, or, with ``group_by_wave``, round-robin within each run of
    consecutive arms of one wave, those runs one after another (``--waves``).
    """
    stats: dict = {"completed": 0, "crashed_kept": 0, "crashed_retry": 0, "resume": 0,
                   "busy": [], "stale": [], "foreign_host_claims": [], "foreign": [],
                   "completed_with_crash_record": []}
    per_arm: list[list[dict]] = []
    arm_waves: list[str] = []
    seen_out: set[str] = set()
    per_arm_seeds = seeds_for_arms(man, arms, seeds)
    for arm in dedupe_arms(arms):
        cells = []
        for task in dedupe(tasks):
            for seed in per_arm_seeds[arm["arm"]]:
                out = cell_output(arm, task, seed, man, output_root, repo=repo)
                if str(out) in seen_out:  # pragma: no cover - guarded by the dedupes
                    raise AssertionError(f"duplicate cell output {out}")
                seen_out.add(str(out))
                cell = {
                    "arm": arm["arm"], "solver": arm["solver"],
                    "run_overrides": dict(arm["run_overrides"]),
                    "subspace_dim": arm["run_overrides"].get("subspace_dim", 16),
                    "task": task, "poli_task": man["task_poli_ids"][task], "seed": seed,
                    "out": out, "timeout_s": int(arm["timeout_s"]),
                    "host_class": arm["host_class"], "wave": arm["wave"],
                    "tag": f"{arm['arm']}__{task}__s{seed}",
                    "retry": False, "resume": False,
                }
                claim = claim_path(out)
                if claim.exists():
                    state, content = claim_state(claim)
                    key = {"alive": "busy", "stale": "stale",
                           "foreign_host": "foreign_host_claims"}[state]
                    stats[key].append({"out": str(out), "claim": content})
                    continue
                sidecar = sidecar_path(out)
                res = read_json(out) if out.exists() else None
                if res is not None and res.get("status") == "completed":
                    problems = validate_result(res, cell, expect_commit=expect_commit)
                    if not problems:
                        # R-5: a valid result is completed, whatever record a
                        # child that died after publishing it may have left.
                        stats["completed"] += 1
                        if sidecar.exists():
                            stats["completed_with_crash_record"].append(str(out))
                        continue
                    if not sidecar.exists():
                        stats["foreign"].append({"out": str(out), "problems": problems})
                        continue
                    # an invalid completed file with a crash record: terminal
                if sidecar.exists() or out.exists():
                    if not retry_crashed:
                        stats["crashed_kept"] += 1
                        continue
                    stats["crashed_retry"] += 1
                    cell["retry"] = True
                elif interrupted_path(out).exists():
                    stats["resume"] += 1
                    cell["resume"] = True
                cells.append(cell)
        per_arm.append(cells)
        arm_waves.append(str(arm["wave"]))

    if not interleave:
        return [c for cells in per_arm for c in cells], stats
    if not group_by_wave:
        return _round_robin(per_arm), stats
    mixed: list[dict] = []
    start = 0
    for end in range(1, len(per_arm) + 1):
        if end == len(per_arm) or arm_waves[end] != arm_waves[start]:
            mixed.extend(_round_robin(per_arm[start:end]))
            start = end
    return mixed, stats


def _round_robin(per_arm: list[list[dict]]) -> list[dict]:
    """Cell i of every arm, then cell i + 1, ...: a slow arm does not tail one GPU."""
    mixed: list[dict] = []
    for i in range(max((len(c) for c in per_arm), default=0)):
        for cells in per_arm:
            if i < len(cells):
                mixed.append(cells[i])
    return mixed


def dedupe_arms(arms: list[dict]) -> list[dict]:
    seen, out = set(), []
    for a in arms:
        if a["arm"] not in seen:
            seen.add(a["arm"])
            out.append(a)
    return out


def archive_attempt(out: Path) -> list[str]:
    """R-8: keep the earlier output. Move ``<out>`` and its crash, interrupted and
    claim records to ``<arm dir>/attempts/<name>.attempt<N><suffix>``.

    N is the first number for which none of the targets exists, so nothing is
    ever overwritten. Returns the new paths.
    """
    dest = out.parent / ATTEMPTS_DIR
    dest.mkdir(exist_ok=True)
    sources = _attempt_sources(out)
    n = _next_attempt_number(out)
    moved = []
    for src, suffix in sources:
        if src.exists():
            target = dest / f"{out.name}.attempt{n}{suffix}"
            os.rename(src, target)
            moved.append(str(target))
    return moved


def _attempt_sources(out: Path) -> tuple[tuple[Path, str], ...]:
    return (
        (out, ".json"),
        (sidecar_path(out), ".crash.json"),
        (interrupted_path(out), ".interrupted.json"),
        (claim_path(out), ".running"),
    )


def _next_attempt_number(out: Path) -> int:
    """First N for which no ``attempts/<name>.attempt<N>*`` target exists."""
    dest = out.parent / ATTEMPTS_DIR
    n = 1
    while any((dest / f"{out.name}.attempt{n}{s}").exists() for _, s in _attempt_sources(out)):
        n += 1
    return n


def archive_crash_record(out: Path) -> str:
    """Move an existing ``<out>.crash.json`` to ``attempts/`` (R-8: never overwritten)."""
    dest = out.parent / ATTEMPTS_DIR
    dest.mkdir(exist_ok=True)
    target = dest / f"{out.name}.attempt{_next_attempt_number(out)}.crash.json"
    os.rename(sidecar_path(out), target)
    return str(target)


# --------------------------------------------------------------------------
# environment, git, host, inputs
# --------------------------------------------------------------------------


def run_git(repo: Path, *args: str) -> str:
    """Run git in ``repo``; fail closed (GitError) on error or timeout."""
    try:
        r = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(repo), *args],
            capture_output=True, text=True, timeout=GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise GitError(f"git {' '.join(args)} in {repo}: {type(e).__name__}: {e}") from e
    if r.returncode != 0:
        raise GitError(f"git {' '.join(args)} in {repo} failed (rc={r.returncode}): "
                       f"{r.stderr.strip()}")
    return r.stdout.strip()


def git_porcelain(repo: Path) -> str:
    return run_git(repo, "status", "--porcelain")


def git_head(repo: Path) -> str:
    return run_git(repo, "rev-parse", "HEAD")


def _fail_closed(fn, *args):
    """Call a git helper; a git failure refuses the run (exit 11) instead of
    reading as an empty commit or a clean tree."""
    try:
        return fn(*args)
    except GitError as e:
        raise Refused(EXIT_GIT, str(e)) from e


def tree_problem(repo: Path, expect_commit: str | None) -> tuple[int, str] | None:
    """``(exit code, message)`` if the tree may not run cells, else None."""
    try:
        head = git_head(repo)
        dirty = git_porcelain(repo)
    except GitError as e:
        return EXIT_GIT, str(e)
    if expect_commit is not None and head != expect_commit:
        return EXIT_DIRTY, f"HEAD is {head}, expected {expect_commit}"
    if dirty:
        return EXIT_DIRTY, ("the tree is dirty, so provenance would record git_dirty=True "
                            f"(rule R-2):\n{dirty}")
    return None


def host_allowed(man: dict, host_class: str, host: str | None = None) -> bool:
    """True if ``host`` (default: this machine) may run arms of ``host_class``."""
    host = host or socket.gethostname()
    allowed = man["gpu_hosts"] if host_class == "gpu" else man["cpu_hosts"]
    return host.split(".")[0] in {h.split(".")[0] for h in allowed}


def child_env(*, repo: Path, cuda_devices: str, threads: int, tmpdir: str | None) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo)
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        # TensorFlow (imported by guacamol) ignores OMP_NUM_THREADS.
        env[var] = str(threads)
    if tmpdir:
        env["TMPDIR"] = tmpdir
    # CUDA's JIT cache defaults to $HOME/.nv, i.e. the NFS home on the cluster.
    env["CUDA_CACHE_PATH"] = os.path.join(tmpdir or str(repo / LOG_ROOT), "cuda_cache")
    # `collect_provenance` runs `git status` at the end of every cell; with
    # several workers in one worktree an index refresh could collide on
    # .git/index.lock and the provenance fields would come back empty.
    env["GIT_OPTIONAL_LOCKS"] = "0"
    return env


def tmpdir_problem(tmpdir: str | None) -> str | None:
    if not tmpdir:
        return "no TMPDIR: pass --tmpdir (never the system /tmp on a cluster node)"
    if os.path.realpath(tmpdir) in {"/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp"}:
        return f"TMPDIR {tmpdir} is the shared system temp root; use a data disk"
    return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_hashes(repo: Path) -> dict[str, str]:
    files = list(REQUIRED_INPUTS)
    for d in HASHED_INPUT_DIRS:
        p = repo / d
        if p.is_dir():
            files.extend(str(f.relative_to(repo)) for f in sorted(p.iterdir()) if f.is_file())
    return {rel: (sha256_file(repo / rel) if (repo / rel).is_file() else "MISSING")
            for rel in dedupe(files)}


def command_output(cmd: list[str]) -> str:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return (r.stdout + r.stderr).strip()
    except (OSError, subprocess.TimeoutExpired) as e:
        return f"unavailable: {type(e).__name__}: {e}"


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.parent / f".{path.name}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(payload, indent=2, default=str))
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)


def write_crash_sidecar(cell: dict, *, kind: str, returncode: int | None, output: str,
                        started: str, elapsed: float, cmd: list[str], cuda_devices: str,
                        log_path: Path | None, amendment: str | None = None,
                        problems: list[str] | None = None, extra: dict | None = None) -> Path:
    """``<out>.crash.json``: the terminal record of a failed cell (R-8)."""
    path = sidecar_path(cell["out"])
    payload = {
        "kind": kind, "arm": cell["arm"], "solver": cell["solver"],
        "run_overrides": cell["run_overrides"], "task": cell["task"],
        "poli_task": cell["poli_task"], "seed": cell["seed"], "wave": cell["wave"],
        "expected_output": str(cell["out"]), "returncode": returncode,
        "timed_out": kind == "timeout", "timeout_s": cell["timeout_s"],
        "elapsed_seconds": round(elapsed, 1), "started": started,
        "finished": datetime.datetime.now().isoformat(timespec="seconds"),
        "host": socket.gethostname(), "cuda_visible_devices": cuda_devices,
        "cmd": cmd, "log": str(log_path) if log_path else None,
        "amendment": amendment, "problems": problems or [],
        "output_tail": output[-CRASH_TAIL_CHARS:], **(extra or {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, payload)
    return path


def resolve_stale_claims(stale: list[dict], *, log) -> None:
    """Turn claims of dead processes into terminal crash records (R-8).

    A dead process left ``<out>.running``. If a valid result is nevertheless
    there (killed between publishing and removing the claim), the claim is
    archived and the result stands; otherwise the claim is archived and a
    ``stale_claim`` crash sidecar is written, so the cell is rerun only through
    ``--retry-crashed --amendment``. A crash record already there (e.g. from a
    manual run that died) is archived to ``attempts/`` first, so its traceback
    is kept (R-8).
    """
    for rec in stale:
        out = Path(rec["out"])
        res = read_json(out) if out.exists() else None
        if res is not None and res.get("status") == "completed":
            moved = archive_claim_only(out)
            log(f"STALE-CLAIM {out.name}: process gone, result present; claim archived "
                f"-> {moved}")
            continue
        previous = archive_crash_record(out) if sidecar_path(out).exists() else None
        dest = archive_claim_only(out)
        cell = {"arm": out.parent.name, "solver": None, "run_overrides": None,
                "task": None, "poli_task": None, "seed": None, "wave": None,
                "timeout_s": None, "out": out}
        write_crash_sidecar(
            cell, kind="stale_claim", returncode=None, output="", started="",
            elapsed=0.0, cmd=[], cuda_devices="", log_path=None,
            extra={"claim": rec["claim"], "archived_claim": dest,
                   "archived_previous_crash_record": previous},
        )
        log(f"STALE-CLAIM {out.name}: process gone, no result -> crash sidecar "
            f"(terminal; rerun only with --retry-crashed --amendment)"
            + (f"; the earlier crash record was archived -> {previous}" if previous else ""))


def archive_claim_only(out: Path) -> str:
    dest = out.parent / ATTEMPTS_DIR
    dest.mkdir(exist_ok=True)
    n = 1
    while (dest / f"{out.name}.stale{n}.running").exists():
        n += 1
    target = dest / f"{out.name}.stale{n}.running"
    os.rename(claim_path(out), target)
    return str(target)


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------


class Runner:
    def __init__(self, args, repo: Path, log_dir: Path, *, expect_commit: str | None,
                 manifest_path: Path | None = None):
        self.args = args
        self.repo = repo
        self.log_dir = log_dir
        self.expect_commit = expect_commit
        #: passed to every child (``run_cell --manifest``), which checks the cell
        #: against its arm's solver, run overrides and seed range
        self.manifest_path = manifest_path
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.stop_reason: str | None = None
        self.signal_received: int | None = None
        self.children: dict[int, tuple[subprocess.Popen, dict]] = {}
        #: pids of the children the signal handler killed (only those cells
        #: may be recorded as interrupted)
        self.killed_pids: set[int] = set()
        self.consecutive_failures = 0
        self.counts = Counter({k: 0 for k in (
            "ok", "skipped", "crashed", "crashed_in_solver", "invalid", "timeout",
            "busy", "refused", "interrupted", "failed_start", "not_started",
        )})

    # -- records ----------------------------------------------------------

    def log_line(self, text: str) -> None:
        stamp = datetime.datetime.now().strftime("%H:%M:%S")
        line = f"[{stamp}] {text}"
        with self.lock:
            print(line, flush=True)
            with (self.log_dir / "queue.log").open("a") as fh:
                fh.write(line + "\n")

    def record(self, rec: dict) -> None:
        with self.lock:
            with (self.log_dir / "cells.jsonl").open("a") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")

    def request_stop(self, reason: str) -> None:
        with self.lock:
            if self.stop_reason is None:
                self.stop_reason = reason
        self.stop.set()

    def bump(self, status: str, n: int = 1) -> None:
        with self.lock:
            self.counts[status] += n

    def breaker(self, status: str) -> None:
        """Circuit breaker: consecutive failed cells mean a broken environment
        (H's first smoke: every child died in ``import torch``) or a hung GPU
        (every running cell times out), not bad cells; stop the queue."""
        with self.lock:
            self.consecutive_failures = (
                self.consecutive_failures + 1 if status in BREAKER_STATUSES else 0
            )
            trip = self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES
        if trip:
            self.request_stop(f"{MAX_CONSECUTIVE_FAILURES} failed cells in a row "
                              f"(hard crash, timeout, crash in the solver or failed start); "
                              "inspect the cell logs before relaunching")

    # -- one cell -----------------------------------------------------------

    def command(self, cell: dict) -> list[str]:
        cmd = [
            self.args.python, str(self.repo / RUN_CELL), cell["solver"], cell["poli_task"],
            str(cell["seed"]), str(cell["out"]), "--arm", cell["arm"],
            "--run-overrides", json.dumps(cell["run_overrides"], sort_keys=True),
            "--host-class", cell["host_class"],
        ]
        if self.manifest_path is not None:
            cmd += ["--manifest", str(self.manifest_path)]
        if self.expect_commit:
            cmd += ["--expect-commit", self.expect_commit]
        return cmd

    def run_cell(self, cell: dict, slot: int, cuda_devices: str) -> None:
        out = Path(cell["out"])
        tag = cell["tag"]
        amendment = self.args.amendment if cell["retry"] else None
        if cell["retry"] or cell["resume"]:
            moved = archive_attempt(out)
            what = "RETRY" if cell["retry"] else "RESUME"
            self.log_line(f"{what} {tag}: earlier attempt archived -> {moved}"
                          + (f"; amendment: {amendment}" if amendment else ""))
        cmd = self.command(cell)
        log_path = self.log_dir / f"{tag}.log"
        env = child_env(repo=self.repo, cuda_devices=cuda_devices,
                        threads=self.args.threads, tmpdir=self.args.tmpdir)
        started = datetime.datetime.now().isoformat(timespec="seconds")
        t0 = time.time()
        self.log_line(f"START slot{slot} gpu={cuda_devices or 'none'} {tag}")

        timed_out = False
        rc = None
        with log_path.open("a") as log_fh:
            log_fh.write(f"# cmd: {' '.join(cmd)}\n# cwd: {self.repo}\n"
                         f"# CUDA_VISIBLE_DEVICES={cuda_devices!r} threads={self.args.threads}\n"
                         f"# started {started}\n")
            log_fh.flush()
            try:
                proc = subprocess.Popen(
                    cmd, cwd=str(self.repo), env=env, stdout=log_fh,
                    stderr=subprocess.STDOUT, start_new_session=True,
                )
            except OSError as e:
                self.bump("failed_start")
                self.log_line(f"FAILED-START {tag}: {e}")
                self.record({"cell": tag, "arm": cell["arm"], "status": "failed_start",
                             "error": str(e)})
                self.breaker("failed_start")
                return
            # Register the child under the lock terminate_all() snapshots under:
            # a child started after the signal is killed here, so none escapes.
            with self.lock:
                late = self.signal_received is not None
                if late:
                    self.killed_pids.add(proc.pid)
                else:
                    self.children[proc.pid] = (proc, cell)
            if late:
                self._kill(proc, cell)
                rc = proc.returncode
            else:
                try:
                    rc = proc.wait(timeout=cell["timeout_s"])
                except subprocess.TimeoutExpired:
                    timed_out = True
                    self._kill(proc, cell)
                    rc = proc.returncode
                finally:
                    with self.lock:
                        self.children.pop(proc.pid, None)

        with self.lock:
            killed = proc.pid in self.killed_pids
        elapsed = time.time() - t0
        output = log_path.read_text(errors="replace") if log_path.exists() else ""
        sidecar_kw = dict(output=output, started=started, elapsed=elapsed, cmd=cmd,
                          cuda_devices=cuda_devices, log_path=log_path, amendment=amendment)
        rec = {
            "cell": tag, "arm": cell["arm"], "solver": cell["solver"],
            "run_overrides": cell["run_overrides"], "task": cell["task"],
            "seed": cell["seed"], "wave": cell["wave"], "returncode": rc,
            "wall_seconds": round(elapsed, 1), "cuda_visible_devices": cuda_devices,
            "host": socket.gethostname(), "out": str(out), "log": str(log_path),
            "retry": cell["retry"], "resume": cell["resume"], "amendment": amendment,
        }

        # A result the child published: does it validate? (75-78 and 2 mean the
        # child refused and computed nothing, so a file there is not its result.)
        published = read_json(out) if out.exists() and rc not in REFUSAL_CODES else None
        published_problems = (validate_result(published, cell, expect_commit=self.expect_commit)
                              if published is not None else None)

        if (killed and not timed_out and rc in KILL_RETURNCODES and not out.exists()):
            # Only a child the queue's signal handler killed is interrupted; one
            # that crashed or timed out on its own meanwhile is a crash below.
            status = "interrupted"
            write_json_atomic(interrupted_path(out), {
                "tag": tag, "signal": self.signal_received, "returncode": rc,
                "started": started, "elapsed_seconds": round(elapsed, 1),
                "log": str(log_path), "host": socket.gethostname(),
                "finished": datetime.datetime.now().isoformat(timespec="seconds"),
            })
            self.log_line(f"INTERRUPTED {tag} (signal {self.signal_received}) "
                          f"-> {interrupted_path(out).name}")
        elif published is not None and not published_problems:
            # R-5: a valid result is completed even if the child then exited
            # non-zero (crash in interpreter teardown), outlived its timeout or
            # was killed by a queue stop. The odd exit is recorded; no sidecar.
            status = "ok"
            prov = published.get("provenance") or {}
            rec.update(self._result_fields(published, prov))
            rec["problems"] = []
            note = ""
            if rc != 0 or timed_out:
                rec["odd_exit"] = {"returncode": rc, "timed_out": timed_out,
                                   "killed_by_queue_stop": killed}
                why = f"rc={rc}"
                if timed_out:
                    why += f", killed after the {cell['timeout_s']} s timeout"
                if killed:
                    why += ", killed by the queue stop"
                note = (f" ODD-EXIT ({why}): the child had published a valid result; "
                        "counted completed, no crash record")
            self.log_line(f"OK {tag} {elapsed:.0f}s best={published.get('best_score')} "
                          f"fallbacks={published.get('surrogate_fallback_count')}{note}")
        elif timed_out:
            status = "timeout"
            side = write_crash_sidecar(cell, kind="timeout", returncode=rc, **sidecar_kw)
            self.log_line(f"TIMEOUT {tag} after {elapsed:.0f}s > {cell['timeout_s']}s "
                          f"-> {side.name} (terminal)")
        elif rc == EXIT_COMPLETED_EXISTS and out.exists():
            res = read_json(out) or {}
            problems = validate_result(res, cell, expect_commit=self.expect_commit)
            if problems:
                status = "invalid"
                self.request_stop(f"a foreign completed file appeared at {out}")
                self.log_line(f"FOREIGN {tag}: {problems} -> stopping the queue")
            else:
                status = "skipped"
                self.log_line(f"SKIP {tag} (a valid completed file appeared)")
        elif rc == EXIT_BUSY:
            status = "busy"
            self.log_line(f"BUSY {tag}: another process holds {claim_path(out).name}")
        elif rc in (EXIT_EXISTS_NOT_COMPLETED, EXIT_PRECONDITION, EXIT_USAGE):
            # The child refused before computing anything: nothing to archive,
            # and every later cell would be refused the same way.
            status = "refused"
            reason = {EXIT_EXISTS_NOT_COMPLETED: "a non-completed file or a crash record exists",
                      EXIT_PRECONDITION: "precondition (commit, tree or file name)",
                      EXIT_USAGE: "usage error in the run_cell command"}[rc]
            self.log_line(f"REFUSED {tag} rc={rc}: {reason} -> stopping the queue")
            self.request_stop(f"run_cell refused {tag}: {reason}")
        elif rc != 0 or not out.exists():
            status = "crashed"
            side = write_crash_sidecar(cell, kind="hard_crash", returncode=rc, **sidecar_kw)
            self.log_line(f"CRASH {tag} rc={rc} after {elapsed:.0f}s -> {side.name} (terminal)")
        else:
            # rc == 0 and a file that does not validate (a valid one is "ok"
            # above): unreadable or not completed -> crashed in the solver,
            # completed but wrong -> invalid
            res = published if published is not None else {}
            problems = validate_result(res, cell, expect_commit=self.expect_commit)
            prov = res.get("provenance") or {}
            rec.update(self._result_fields(res, prov))
            if res.get("status") != "completed":
                status = "crashed_in_solver"
                side = write_crash_sidecar(cell, kind="crashed_in_solver", returncode=rc,
                                           problems=problems, **sidecar_kw)
                self.log_line(f"CRASH-IN-SOLVER {tag}: {res.get('crash_info')} -> "
                              f"{side.name} (terminal)")
            else:
                status = "invalid"
                side = write_crash_sidecar(cell, kind="invalid", returncode=rc,
                                           problems=problems, **sidecar_kw)
                self.log_line(f"INVALID {tag}: {problems} -> {side.name} (terminal)")
                if any("provenance" in p for p in problems):
                    # every later cell would carry the same stain
                    self.request_stop(f"invalid provenance in {tag}")
            rec["problems"] = problems
        self.bump(status)
        rec["status"] = status
        self.record(rec)
        self.breaker(status)

    @staticmethod
    def _result_fields(res: dict, prov: dict) -> dict:
        """The fields of a result file that ``cells.jsonl`` carries."""
        return {
            "cell_status": res.get("status"), "best_score": res.get("best_score"),
            "elapsed_seconds": res.get("elapsed_seconds"),
            "n_evaluated": res.get("n_evaluated"),
            "n_oracle_failures": res.get("n_oracle_failures"),
            "surrogate_fallback_count": res.get("surrogate_fallback_count"),
            "git_commit": prov.get("git_commit"),
            "git_commit_full": prov.get("git_commit_full"),
            "git_dirty": prov.get("git_dirty"),
        }

    def _kill(self, proc: subprocess.Popen, cell: dict | None) -> None:
        """Kill the child's whole process group, then drop its claim (SIGKILL
        skips the child's own ``finally``)."""
        for sig, wait_s in ((signal.SIGTERM, 20), (signal.SIGKILL, 10)):
            try:
                os.killpg(os.getpgid(proc.pid), sig)
            except (ProcessLookupError, PermissionError):
                break
            try:
                proc.wait(timeout=wait_s)
                break
            except subprocess.TimeoutExpired:
                continue
        if cell is not None:
            claim = claim_path(Path(cell["out"]))
            with contextlib.suppress(OSError, ValueError, IndexError):
                if int(claim.read_text().split()[0]) == proc.pid:
                    claim.unlink()

    def precheck(self) -> str | None:
        """Re-check the tree before each cell (R-2); a message stops the queue."""
        if self.args.allow_dirty:
            return None
        problem = tree_problem(self.repo, self.expect_commit)
        return None if problem is None else problem[1]

    def worker(self, slot: int, cuda_devices: str, work: WorkList) -> None:
        while not self.stop.is_set():
            cell = work.pop()
            if cell is None:
                return
            problem = self.precheck()
            if problem:
                self.log_line(f"STOP before {cell['tag']}: {problem}")
                self.request_stop(problem)
                self.bump("not_started")
                return
            self.run_cell(cell, slot, cuda_devices)

    def terminate_all(self, signum: int) -> None:
        """Kill every running child (signal handler path).

        Only children still alive at this moment are marked as killed; a child
        that has already exited is left to its worker, which records whatever
        it did (a crash stays a crash). Children registered after this point
        see ``signal_received`` and are killed by their worker (``run_cell``).
        """
        self.request_stop(f"signal {signum}")
        with self.lock:
            self.signal_received = signum
            children = [(p, c) for p, c in self.children.values() if p.poll() is None]
            self.killed_pids.update(p.pid for p, _ in children)
        threads = [threading.Thread(target=self._kill, args=(p, c), daemon=True)
                   for p, c in children]
        for th in threads:
            th.start()
        for th in threads:
            th.join(timeout=40)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--repo", type=Path, default=REPO_ROOT,
                    help="repository whose run_cell.py runs every cell (the queue chdirs into it)")
    ap.add_argument("--python", default=sys.executable, help="interpreter for the children")
    ap.add_argument("--expect-commit", default=None,
                    help="full 40-hex commit HEAD must be at (required unless --allow-dirty)")
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--wave", default=None, choices=VALID_WAVES, help="restrict to one wave")
    sel.add_argument("--waves", default=None,
                     help="comma-separated waves in run order, e.g. 2,1,3,3b,4: the arms of "
                          "a wave are interleaved, the waves run one after another")
    sel.add_argument("--arms", default=None, help="comma-separated arm ids")
    ap.add_argument("--tasks", default=None, help="comma-separated short task codes (default: all 11)")
    ap.add_argument("--seeds", default=None,
                    help="e.g. 42-81 or 42,43 (default: every arm's own seed range, i.e. its "
                         "'seeds' field, else the manifest's 42-81); a seed outside the range "
                         "of a selected arm refuses the selection")
    ap.add_argument("--host-class", choices=HOST_CLASSES, default="gpu")
    ap.add_argument("--gpus", default="0,1", help="GPU ids to pin workers to")
    ap.add_argument("--workers-per-gpu", type=int, default=4)
    ap.add_argument("--cpu-workers", type=int, default=1,
                    help="worker count for --host-class cpu (children see no GPU); kajman "
                         "allows 2 processes, the queue parent included, so 1")
    ap.add_argument("--threads", type=int, default=4, help="BLAS/OMP/TF threads per child")
    ap.add_argument("--tmpdir", default=os.environ.get("TMPDIR") or None,
                    help="TMPDIR for the children (never the system /tmp on a cluster node)")
    ap.add_argument("--output-root", default=None,
                    help="write under this root instead of the manifest's; relative = to --repo")
    ap.add_argument("--timeout-s", type=int, default=None, help="override every arm's timeout")
    ap.add_argument("--stagger", type=float, default=2.0,
                    help="seconds between worker starts (spreads the per-cell git calls)")
    ap.add_argument("--retry-crashed", action="store_true",
                    help="rerun crashed cells (their files are archived); needs --amendment")
    ap.add_argument("--amendment", default=None,
                    help='"<date>: <reason>" of the RUN_RULES amendment-log row behind a retry')
    ap.add_argument("--no-interleave", action="store_true",
                    help="run arm by arm instead of round-robin")
    ap.add_argument("--limit", type=int, default=None, help="run at most N cells")
    ap.add_argument("--dry-run", action="store_true",
                    help="preflight and list the cells; no side effects")
    ap.add_argument("--log-dir", type=Path, default=None,
                    help="default <repo>/logs/cr/<timestamp>_<tag>")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="skip commit/tree checks (violates R-2; tests and experiments only)")
    ap.add_argument("--allow-any-host", action="store_true",
                    help="skip the gpu_hosts / cpu_hosts guard (tests only)")
    ap.add_argument("--allow-missing-inputs", action="store_true",
                    help="start even if required inputs are missing (tests only)")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = build_parser()
    args = ap.parse_args(argv)
    if args.retry_crashed and not (args.amendment or "").strip():
        ap.error("--retry-crashed requires --amendment '<date>: <reason>' (rule R-8)")
    if args.expect_commit is None and not args.allow_dirty:
        ap.error("--expect-commit <full sha> is required (or --allow-dirty outside the generation)")
    if args.expect_commit is not None and not re.fullmatch(r"[0-9a-f]{40}", args.expect_commit):
        ap.error("--expect-commit must be a full 40-character lowercase hex hash")
    if args.waves is not None:
        waves = dedupe(w.strip() for w in args.waves.split(",") if w.strip())
        bad = [w for w in waves if w not in VALID_WAVES]
        if not waves or bad:
            ap.error(f"--waves takes waves from {', '.join(VALID_WAVES)}, got {args.waves!r}")
        args.waves = waves

    # Absolute paths first, then work from the repository (rule: a queue launched
    # from another directory must see exactly what its children see).
    repo = args.repo.resolve()
    manifest_path = args.manifest.resolve()
    if args.log_dir is not None:
        args.log_dir = args.log_dir.resolve()
    if args.tmpdir:
        args.tmpdir = os.path.abspath(args.tmpdir)
    os.chdir(repo)
    try:
        return _main(args, repo, manifest_path)
    except Refused as e:
        print(f"REFUSED (exit {e.code}): {e}", file=sys.stderr, flush=True)
        return e.code


def _main(args, repo: Path, manifest_path: Path) -> int:
    man = load_manifest(manifest_path)
    tasks = dedupe(t.strip() for t in args.tasks.split(",")) if args.tasks else list(man["tasks"])
    unknown = [t for t in tasks if t not in man["task_poli_ids"]]
    if unknown:
        raise SystemExit(f"unknown task code(s): {', '.join(unknown)}")
    requested_seeds = parse_int_list(args.seeds, []) if args.seeds else None
    arm_ids = dedupe(s.strip() for s in args.arms.split(",")) if args.arms else None
    arms = select_arms(man, wave=args.wave, arm_ids=arm_ids, host_class=args.host_class,
                       waves=args.waves)
    if not arms:
        raise SystemExit("no arm matches the filters")
    # each arm's seeds (its own range, or --seeds inside every selected range)
    arm_seed_map = seeds_for_arms(man, arms, requested_seeds)
    seed_ranges = Counter(format_seeds(s) for s in arm_seed_map.values())
    seeds_text = (f"{len(next(iter(arm_seed_map.values())))} ({next(iter(seed_ranges))})"
                  if len(seed_ranges) == 1 else
                  "per arm: " + "; ".join(f"{r} ({n} arms)" for r, n in seed_ranges.items()))
    expect = None if args.allow_dirty else args.expect_commit

    # ---- preflight (read-only) -------------------------------------------
    host = socket.gethostname()
    if not args.allow_any_host and not host_allowed(man, args.host_class, host):
        raise Refused(EXIT_HOST, f"{args.host_class}-class arms may not run on {host} "
                                 f"(manifest {args.host_class}_hosts)")
    if not args.allow_dirty:
        problem = tree_problem(repo, expect)
        if problem:
            raise Refused(*problem)
    problem = tmpdir_problem(args.tmpdir)
    if problem:
        raise Refused(EXIT_INPUTS, problem)
    hashes = input_hashes(repo)
    missing = [rel for rel in REQUIRED_INPUTS if hashes.get(rel) == "MISSING"]
    if missing and not args.allow_missing_inputs:
        raise Refused(EXIT_INPUTS, f"required inputs missing: {missing}")

    head = "" if args.allow_dirty else _fail_closed(git_head, repo)
    root = output_root_path(man, args.output_root, repo)

    def select() -> tuple[list[dict], dict]:
        """Build the cell list and refuse on foreign files or unverifiable claims."""
        cells, stats = build_cells(
            man, arms, tasks=tasks, seeds=requested_seeds, output_root=args.output_root,
            retry_crashed=args.retry_crashed, interleave=not args.no_interleave,
            expect_commit=expect, repo=repo, group_by_wave=bool(args.waves),
        )
        if args.timeout_s:
            for c in cells:
                c["timeout_s"] = args.timeout_s
        if args.limit is not None:
            cells = cells[: args.limit]
        print(f"arms: {len(arms)}  tasks: {len(tasks)}  seeds: {seeds_text}  root: {root}")
        print(f"cells to run: {len(cells)}   completed (valid): {stats['completed']}"
              f"   crashed kept: {stats['crashed_kept']}   retried: {stats['crashed_retry']}"
              f"   resumed: {stats['resume']}   busy: {len(stats['busy'])}"
              f"   stale claims: {len(stats['stale'])}   foreign: {len(stats['foreign'])}")
        if args.waves:
            per_wave = Counter(str(c["wave"]) for c in cells)
            print("cells to run per wave, in run order: "
                  + ", ".join(f"wave {w}: {per_wave.get(w, 0)}" for w in args.waves))
        for rec in stats["busy"]:
            print(f"  BUSY {rec['out']} ({rec['claim']})")
        for out in stats["completed_with_crash_record"]:
            print(f"  WARNING {out}: a crash record sits next to this valid completed "
                  "result; the cell counts as completed and is never rerun (record kept)")
        if stats["foreign"]:
            for rec in stats["foreign"][:20]:
                print(f"  FOREIGN {rec['out']}: {rec['problems']}")
            raise Refused(EXIT_FOREIGN, f"{len(stats['foreign'])} completed file(s) in the "
                                        "selection fail validation (another commit, a dirty "
                                        "tree or a mismatch); move them out by hand")
        if stats["foreign_host_claims"]:
            for rec in stats["foreign_host_claims"]:
                print(f"  CLAIM FROM ANOTHER HOST {rec['out']} ({rec['claim']})")
            raise Refused(EXIT_STALE, "claims by another host cannot be checked from here")
        return cells, stats

    cells, stats = select()
    if args.dry_run:
        for rec in stats["stale"]:
            print(f"  STALE CLAIM {rec['out']} ({rec['claim']}) -> would become a crash record")
        for c in cells[:60]:
            flag = " (retry)" if c["retry"] else " (resume)" if c["resume"] else ""
            print(f"  {c['arm']:38s} {c['task']} s{c['seed']}{flag} -> {c['out']}")
        if len(cells) > 60:
            print(f"  ... and {len(cells) - 60} more")
        return EXIT_OK
    if not cells and not stats["stale"]:
        return EXIT_OK

    # ---- exclusive use of the output root ----------------------------------
    root.mkdir(parents=True, exist_ok=True)
    lock_fh = open(root / LOCK_NAME, "a+")
    try:
        fcntl.flock(lock_fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        if e.errno in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
            lock_fh.seek(0)
            holder = lock_fh.read().strip()
            lock_fh.close()
            raise Refused(EXIT_LOCKED, f"another queue holds {root / LOCK_NAME} ({holder}); "
                                       "use one queue per output root") from e
        raise
    lock_fh.seek(0)
    lock_fh.truncate()
    lock_fh.write(f"{os.getpid()} {host} {datetime.datetime.now().isoformat(timespec='seconds')}\n")
    lock_fh.flush()

    try:
        tag = args.wave or ("waves_" + "-".join(args.waves) if args.waves else None) or (
            arm_ids[0] if arm_ids else args.host_class)
        log_dir = args.log_dir or (
            repo / LOG_ROOT / f"{datetime.datetime.now():%Y%m%d_%H%M%S}_{tag}"
        )
        log_dir.mkdir(parents=True, exist_ok=True)
        runner = Runner(args, repo, log_dir, expect_commit=expect, manifest_path=manifest_path)

        if stats["stale"]:
            resolve_stale_claims(stats["stale"], log=runner.log_line)
            cells, stats = select()  # the resolved cells are now crash records

        if args.host_class == "cpu":
            slots = [""] * args.cpu_workers
        else:
            gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
            slots = [g for _ in range(args.workers_per_gpu) for g in gpus]

        t_start = time.time()
        summary = {
            "status": "running",
            "started": datetime.datetime.fromtimestamp(t_start).isoformat(timespec="seconds"),
            "argv": sys.argv, "python": sys.executable, "child_python": args.python,
            "host": host, "host_fqdn": socket.getfqdn(), "host_class": args.host_class,
            "repo": str(repo), "head": head, "expect_commit": args.expect_commit,
            "allow_dirty": args.allow_dirty, "allow_any_host": args.allow_any_host,
            "allow_missing_inputs": args.allow_missing_inputs, "amendment": args.amendment,
            "retry_crashed": args.retry_crashed, "output_root": str(root),
            "log_dir": str(log_dir), "n_cells": len(cells), "n_workers": len(slots),
            "slots": slots, "threads": args.threads, "tmpdir": args.tmpdir,
            "wave": args.wave, "waves": args.waves,
            "interleave": not args.no_interleave,
            "arms": [a["arm"] for a in arms], "tasks": tasks,
            "seeds": sorted({s for v in arm_seed_map.values() for s in v}),
            "seeds_requested": requested_seeds,
            "arm_seeds": {a: format_seeds(v) for a, v in arm_seed_map.items()},
            "selection": {k: (v if isinstance(v, int) else len(v)) for k, v in stats.items()},
            "manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
            "inputs_sha256": hashes, "loadavg_start": os.getloadavg(),
            "nvidia_smi_start": command_output(["nvidia-smi"]),
            "who_start": command_output(["who"]),
        }
        write_json_atomic(log_dir / "summary.json", summary)
        runner.log_line(
            f"queue start: {len(cells)} cells, {len(slots)} workers ({args.host_class}, "
            f"gpus={args.gpus if args.host_class != 'cpu' else 'none'}), threads={args.threads}, "
            f"repo={repo}, head={head or '(not checked: --allow-dirty)'}, root={root}, "
            f"logs={log_dir}" + (f", waves in order: {','.join(args.waves)}" if args.waves else "")
            + (f", amendment: {args.amendment}" if args.amendment else "")
        )
        runner.log_line("required inputs sha256 (all inputs in summary.json): "
                        + json.dumps({k: hashes[k] for k in REQUIRED_INPUTS}))
        runner.log_line("load average at start: " + " ".join(f"{x:.1f}" for x in os.getloadavg()))
        if not cells:
            runner.log_line("nothing to run")
            summary.update(status="done", counts=dict(runner.counts), wall_seconds=0.0)
            write_json_atomic(log_dir / "summary.json", summary)
            return EXIT_OK

        signalled: list[int] = []

        def _on_signal(signum, _frame):
            if signalled:          # a second signal during the shutdown is ignored
                return
            signalled.append(signum)
            raise _Interrupted(signum)

        old_handlers = {s: signal.signal(s, _on_signal)
                        for s in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT)}
        work = WorkList(cells)
        finished: list[threading.Event] = []

        def _worker(slot_i: int, gpu: str, done: threading.Event) -> None:
            try:
                runner.worker(slot_i, gpu, work)
            finally:
                done.set()

        def _all_finished() -> bool:
            return all(ev.is_set() for ev in finished)

        # The main thread waits by sleeping, never in Thread.join(): an exception
        # raised by a signal handler inside join() on a still-running thread
        # leaves CPython believing that thread has stopped (bpo-45274 fix
        # releases its tstate lock), so a later join() returns at once and the
        # interrupted cells would never be recorded.
        exit_code = None
        try:
            for slot_i, gpu in enumerate(slots):
                if runner.stop.is_set():
                    break
                done = threading.Event()
                finished.append(done)
                threading.Thread(target=_worker, args=(slot_i, gpu, done), daemon=True).start()
                if args.stagger > 0 and slot_i + 1 < len(slots):
                    time.sleep(args.stagger)
            while not _all_finished():
                time.sleep(0.5)
        except _Interrupted as e:
            runner.log_line(f"signal {e.signum}: terminating children; their cells are "
                            "recorded as interrupted")
            runner.terminate_all(e.signum)
            deadline = time.time() + 180
            while not _all_finished() and time.time() < deadline:
                time.sleep(0.2)
            exit_code = 128 + int(e.signum)
        finally:
            for s, h in old_handlers.items():
                signal.signal(s, h)

        runner.bump("not_started", len(work))
        wall = time.time() - t_start
        runner.log_line(
            f"queue done in {wall / 60:.1f} min: "
            + ", ".join(f"{k}={v}" for k, v in runner.counts.items() if v)
            + (f"; stopped: {runner.stop_reason}" if runner.stop_reason else "")
        )
        failures = sum(v for k, v in runner.counts.items() if k not in ("ok", "skipped"))
        if exit_code is None:
            exit_code = EXIT_OK if failures == 0 else EXIT_FAILURES
        summary.update(
            status="interrupted" if runner.signal_received else "done",
            finished=datetime.datetime.now().isoformat(timespec="seconds"),
            wall_seconds=round(wall, 1), counts=dict(runner.counts),
            stop_reason=runner.stop_reason, exit_code=exit_code,
            loadavg_end=os.getloadavg(),
        )
        write_json_atomic(log_dir / "summary.json", summary)
        return exit_code
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(lock_fh, fcntl.LOCK_UN)
            lock_fh.close()


if __name__ == "__main__":
    sys.exit(main())
