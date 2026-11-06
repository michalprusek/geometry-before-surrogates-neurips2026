"""Load every arm of a sources config, cell by cell, and fail loudly.

Fatal (raised as one LoaderError listing every problem of every arm):
  * missing arm        - required arm whose directory is absent or matches no file
  * missing cell       - expected (task, seed) cell without a file, unless declared in
                         ``known_missing``; a declared cell that does exist is also fatal
                         (stale declaration)
  * duplicate cell     - two files resolving to the same (task, seed)
  * provenance         - an arm with ``provenance: require`` whose file lacks the expected
                         commit (prefix match) or has git_dirty != False. The expected commit is
                         per ROOT: the main campaign (roots CR, CRCPU) and the control block (root
                         CRCTL) were run from two tagged commits, and a file is accepted only with
                         the commit of the root its arm reads (``expected_commit_for``)
  * content            - JSON unreadable, or solver/method, task or seed field contradicting
                         the file name

Never fatal, never dropped silently (counted per arm, listed in loader_report):
  * non-completed status (crashed, timeout, missing status field under require_field)
  * short traces (fewer than trace_len best-so-far values)
  * surrogate-fallback cells and steps (``surrogate_fallback_count``; for an arm with ``fallback_rule``
    ``recipe_step_fallback`` the steps of ``published_recipe.steps`` whose ``fallback`` entry is set, see
    ``fallback_of``)
  * files in the arm's directory that the anchored pattern excludes (budget suffixes,
    subspace ablations, other tasks), counted by suffix

Camera-ready queue sidecars (scripts/cr/cr_queue.py, run_cell.py) are recognised and never
read as results: ``<out>.crash.json`` (terminal crash record), ``<out>.interrupted.json``
(operator interrupt, resumed by the next queue), ``<out>.running`` (claim of a live cell),
``<out>.collision-<pid>.json`` (a second payload kept for inspection), ``.<out>.*.tmp``
(atomic-write temp files) and the ``attempts/`` directory (archived earlier attempts).
  * a crash sidecar WITHOUT a result file is a crashed cell (R-5: counted and reported,
    never eligible, not a missing cell); with a result file the result's status decides
  * an interrupted or running cell without a result is unresolved: fatal like a missing
    cell (resume or finish the queue first), unless declared missing
  * collision and temp files and attempts/ are counted in the report and ignored
Camera-ready arms may also declare ``cr_arm`` (the file's provenance.cr_arm must equal it:
files are grouped by arm id, not by solver) and ``expect`` (top-level fields that must hold
these values, e.g. subspace_dim 8; and ``provenance.run_overrides``) -- a mismatch is a
content problem (fatal).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from . import TASKS_T11
from .config import EXTRA_POLI, ArmSpec, SourcesConfig


class LoaderError(Exception):
    """One or more fatal loader problems. ``problems`` is a list of (kind, arm, message)."""

    def __init__(self, problems):
        self.problems = problems
        lines = [f"[{k}] {a}: {m}" for k, a, m in problems]
        super().__init__(f"{len(problems)} fatal loader problem(s):\n  " + "\n  ".join(lines))

    def kinds(self) -> set:
        return {k for k, _, _ in self.problems}


@dataclass
class Cell:
    task: str
    seed: int
    file: str
    sha256: str
    status: str
    status_ok: bool
    fallback_count: int
    fallback_recorded: bool
    budget: object
    trace: np.ndarray | None      # best-so-far per evaluation (index i = after i+1 evaluations)
    scores: np.ndarray | None
    value30: float                # best@30 under the arm's value rule
    n_evaluated: object
    commit: object
    dirty: object
    host: object
    elapsed: object
    timestamp: str
    history_len: int | None = None   # external format only
    extra: dict | None = None        # external format: per-step n_evaluated / current_score, n_cold_start

    def ok_len(self, trace_len: int) -> bool:
        return self.trace is not None and len(self.trace) >= trace_len


@dataclass
class ArmData:
    spec: ArmSpec
    cells: dict = field(default_factory=dict)       # (task, seed) -> Cell (every matched file)
    absent: bool = False
    report: dict = field(default_factory=dict)
    sidecar_crashes: dict = field(default_factory=dict)   # (task, seed) -> crash sidecar record (no result file)

    def n_not_completed(self) -> int:
        """Result files with a non-completed status plus crash-sidecar-only cells."""
        return sum(1 for c in self.cells.values() if not c.status_ok) + len(self.sidecar_crashes)

    def eligible(self, cell_filter: str) -> set:
        out = set()
        for key, c in self.cells.items():
            if not c.status_ok:
                continue
            if self.spec.fmt == "ours" and not c.ok_len(self.spec.trace_len):
                continue
            if cell_filter == "clean":
                if c.fallback_count > 0:
                    continue
                if c.budget not in (None, self.spec.budget):
                    continue
            out.add(key)
        return out

    def value(self, key, N: int = 30) -> float:
        c = self.cells[key]
        if N == 30 and self.spec.fmt == "external":
            return c.value30
        if c.trace is None or len(c.trace) < N:
            return float("nan")
        return float(c.trace[N - 1])


def _prov(d: dict):
    p = d.get("provenance")
    if isinstance(p, dict) and isinstance(p.get("provenance"), dict):
        p = p["provenance"]
    if not isinstance(p, dict):
        return None, None, None
    return p.get("git_commit"), p.get("git_dirty"), p.get("hostname")


def _arr(x) -> np.ndarray | None:
    if x is None:
        return None
    return np.asarray([np.nan if v is None else v for v in x], dtype=float)


def fallback_of(spec: ArmSpec, d: dict) -> tuple[int, bool]:
    """(fallback steps, recorded) of one 'ours' result file under the arm's ``fallback_rule``.

    ``surrogate_fallback_count`` (default): the file's field. ``recipe_step_fallback`` (the rerun arms of 2026-11-02,
    cmaes_full_rd, cmaes_full_rd_pop5 and baxus_cr): their ``surrogate_fallback_count`` also counts invalid molecules,
    which no other arm does, so the count is the number of steps of ``published_recipe.steps`` whose ``fallback`` entry
    is set (a step whose proposal was replaced by a uniform draw); a step without the entry (CMA-ES has no surrogate)
    counts as none. Without a ``published_recipe.steps`` list the file records no fallback."""
    if spec.fallback_rule == "recipe_step_fallback":
        rec = d.get("published_recipe")
        steps = rec.get("steps") if isinstance(rec, dict) else None
        if not isinstance(steps, list):
            return 0, False
        return sum(1 for st in steps if isinstance(st, dict) and st.get("fallback") is not None), True
    fb_raw = d.get("surrogate_fallback_count")
    return int(fb_raw or 0), fb_raw is not None


def expected_commit_for(spec: ArmSpec, expected) -> str | None:
    """The commit an arm's files must carry. ``expected``: None, one commit for every arm (str), or
    {root name: commit, "*": default commit}; an arm reads the entry of its ``commit_root``."""
    if expected is None or isinstance(expected, str):
        return expected
    return expected.get(spec.commit_root) or expected.get("*")


def commit_matches(file_commit, expected: str) -> bool:
    if not file_commit or not expected:
        return False
    a, b = str(file_commit).strip(), str(expected).strip()
    k = min(len(a), len(b))
    return k >= 7 and a[:k] == b[:k]


SIDECAR_RX = re.compile(r"^(?P<base>.+?\.json)(?P<kind>\.crash\.json|\.interrupted\.json|\.running|"
                        r"\.collision-\d+\.json)$")
TMP_RX = re.compile(r"^\..+\.tmp$")
ATTEMPTS_DIR = "attempts"


def split_sidecars(names: list[str]) -> tuple[list[str], dict]:
    """Separate queue sidecars from candidate result names.
    -> (other names, {"crash": [(name, base)], "interrupted": [...], "running": [...],
                      "collision": [...], "tmp": [names], "attempts_dir": bool})"""
    side = {"crash": [], "interrupted": [], "running": [], "collision": [], "tmp": [], "attempts_dir": False}
    rest = []
    for n in names:
        if n == ATTEMPTS_DIR:
            side["attempts_dir"] = True
            continue
        if TMP_RX.match(n):
            side["tmp"].append(n)
            continue
        m = SIDECAR_RX.match(n)
        if m:
            k = m.group("kind")
            kind = ("crash" if k == ".crash.json" else "interrupted" if k == ".interrupted.json"
                    else "running" if k == ".running" else "collision")
            side[kind].append((n, m.group("base")))
            continue
        rest.append(n)
    return rest, side


def _classify_excluded(spec: ArmSpec, names: list[str]) -> dict:
    """Files in the arm directory that share the pattern's literal prefix but not the pattern
    (queue sidecars are removed before this classification)."""
    pre = spec.prefix
    known_tokens = sorted(list(TASKS_T11.values()) + EXTRA_POLI + list(TASKS_T11.keys()) + ["valt"],
                          key=len, reverse=True)
    rx = re.compile("^" + re.escape(pre) + "(?P<task>" + "|".join(map(re.escape, known_tokens)) +
                    r")_s(?:eed)?(?P<seed>\d+)(?P<tail>.*)\.json$")
    suffix, other_task, foreign = Counter(), Counter(), 0
    for n in names:
        if spec.regex.match(n) or not n.startswith(pre):
            continue
        m = rx.match(n)
        if not m:
            foreign += 1
            continue
        tok = m.group("task")
        if tok in TASKS_T11.values() or tok in TASKS_T11:
            suffix[m.group("tail") or "(none)"] += 1
        else:
            other_task[tok] += 1
    return {"suffix_variants": dict(suffix), "other_tasks": dict(other_task),
            "foreign_same_prefix": foreign}


def load_arm(spec: ArmSpec, expected_commit, problems: list) -> ArmData:
    arm = ArmData(spec=spec)
    expected_commit = expected_commit_for(spec, expected_commit)
    rep = {"name": spec.name, "kind": spec.kind, "format": spec.fmt, "dir": spec.dir,
           "pattern": spec.pattern, "provenance_rule": spec.provenance, "commit_root": spec.commit_root,
           "expected_commit": expected_commit if spec.provenance == "require" else None,
           **({"fallback_rule": spec.fallback_rule} if spec.fallback_rule != "surrogate_fallback_count" else {}),
           "seeds": [min(spec.seeds), max(spec.seeds)] if spec.seeds else [], "n_seeds": len(spec.seeds)}
    arm.report = rep
    if not spec.seeds or not spec.tasks:
        arm.absent = True
        rep["absent"] = "outside the run's task/seed subset (--tasks / --seeds)"
        rep["outside_subset"] = True
        return arm
    if spec.provenance == "require" and not expected_commit:
        problems.append(("config", spec.name, f"no expected commit for root {spec.commit_root!r} "
                                              "(--expected-commit ROOT=SHA)"))
    if not os.path.isdir(spec.dir):
        arm.absent = True
        rep["absent"] = "directory does not exist"
        return arm
    all_names = sorted(os.listdir(spec.dir))
    names, side = split_sidecars(all_names)
    matched = [n for n in names if spec.regex.match(n)]
    rep["n_files_in_dir"] = len(all_names)
    rep["excluded"] = _classify_excluded(spec, names)
    in_scope, outside = [], 0
    for n in matched:
        m = spec.regex.match(n)
        t = spec.short_task(m.group("task"))
        s = int(m.group("seed"))
        if t in spec.tasks and s in spec.seeds:
            in_scope.append((n, t, s))
        else:
            outside += 1
    rep["n_files_matched"] = len(matched)
    rep["n_files_outside_run_subset"] = outside
    # ---- queue sidecars, keyed by the cell of their result name ---------------------------
    side_cells: dict = {"crash": {}, "interrupted": {}, "running": {}, "collision": {}}
    side_other = Counter()
    for kind in side_cells:
        for n, base in side[kind]:
            m = spec.regex.match(base)
            if not m:
                side_other[kind] += 1
                continue
            t, s = spec.short_task(m.group("task")), int(m.group("seed"))
            if t in spec.tasks and s in spec.seeds:
                side_cells[kind].setdefault((t, s), []).append(n)
            else:
                side_other[kind + "_outside_run_subset"] += 1
    rep["sidecars"] = {"crash": sum(len(v) for v in side_cells["crash"].values()),
                       "interrupted": sum(len(v) for v in side_cells["interrupted"].values()),
                       "running": sum(len(v) for v in side_cells["running"].values()),
                       "collision": sum(len(v) for v in side_cells["collision"].values()),
                       "tmp": len(side["tmp"]), "attempts_dir": side["attempts_dir"],
                       "not_matching_this_arm": dict(side_other)}
    if not in_scope and not side_cells["crash"]:
        arm.absent = True
        rep["absent"] = "no file matches the pattern inside the run's task/seed set"
        return arm
    seen: dict = {}
    for n, t, s in in_scope:
        key = (t, s)
        path = os.path.join(spec.dir, n)
        if key in seen:
            problems.append(("duplicate", spec.name, f"cell {t}:{s} has two files: {seen[key]} and {path}"))
            continue
        seen[key] = path
        try:
            raw = open(path, "rb").read()
            d = json.loads(raw)
        except (OSError, json.JSONDecodeError) as e:
            problems.append(("content", spec.name, f"unreadable {path}: {e}"))
            continue
        sha = hashlib.sha256(raw).hexdigest()
        if spec.fmt == "ours":
            if spec.solver_field is not None and d.get("solver") != spec.solver_field:
                problems.append(("content", spec.name, f"{path}: solver field {d.get('solver')!r} != {spec.solver_field!r}"))
                continue
            tf = d.get("task")
            if tf is not None and tf != TASKS_T11[t]:
                problems.append(("content", spec.name, f"{path}: task field {tf!r} != {TASKS_T11[t]!r}"))
                continue
        else:
            if spec.solver_field is not None and d.get("method") != spec.solver_field:
                problems.append(("content", spec.name, f"{path}: method field {d.get('method')!r} != {spec.solver_field!r}"))
                continue
            tf = d.get("task_id")
            if tf is not None and tf != t:
                problems.append(("content", spec.name, f"{path}: task_id {tf!r} != {t!r}"))
                continue
        sf = d.get("seed")
        if sf is not None and int(sf) != s:
            problems.append(("content", spec.name, f"{path}: seed field {sf!r} != {s}"))
            continue
        bad_field = None
        if spec.cr_arm:
            pv = d.get("provenance") if isinstance(d.get("provenance"), dict) else {}
            if pv.get("cr_arm") != spec.cr_arm:
                bad_field = f"provenance.cr_arm {pv.get('cr_arm')!r} != {spec.cr_arm!r}"
        for k, v in (spec.expect or {}).items():
            if k == "provenance.run_overrides":
                pv = d.get("provenance") if isinstance(d.get("provenance"), dict) else {}
                got = pv.get("run_overrides")
            else:
                got = d.get(k)
            if got != v:
                bad_field = f"{k} {got!r} != {v!r}"
        if bad_field:
            problems.append(("content", spec.name, f"{path}: {bad_field}"))
            continue
        raw_status = d.get("status")
        if raw_status is None:
            status = "absent"
            status_ok = spec.status_rule == "absent_is_completed"
        else:
            status = str(raw_status).split(":")[0].strip()
            status_ok = status == "completed"
        commit, dirty, host = _prov(d)
        if spec.fmt == "ours":
            trace = _arr(d.get("best_trace"))
            scores = _arr(d.get("all_scores"))
            v30 = float(trace[29]) if trace is not None and len(trace) >= 30 else float("nan")
            fb_n, fb_rec = fallback_of(spec, d)
            cell = Cell(task=t, seed=s, file=path, sha256=sha, status=status, status_ok=status_ok,
                        fallback_count=fb_n, fallback_recorded=fb_rec,
                        budget=d.get("budget"), trace=trace, scores=scores, value30=v30,
                        n_evaluated=d.get("n_evaluated"), commit=commit, dirty=dirty, host=host,
                        elapsed=d.get("elapsed_seconds"), timestamp=str(d.get("timestamp", ""))[:19])
        else:
            hist = (d.get("history") or {}).get("best_score") or []
            hist = [float("nan") if v is None else float(v) for v in hist]
            v30 = float(np.nanmax(hist)) if hist else (float(d["best_score"]) if d.get("best_score") is not None else float("nan"))
            h = d.get("history") or {}
            cell = Cell(task=t, seed=s, file=path, sha256=sha, status=status, status_ok=status_ok,
                        fallback_count=0, fallback_recorded=False, budget=None, trace=None,
                        scores=np.asarray(hist, dtype=float), value30=v30,
                        n_evaluated=d.get("n_evaluated"), commit=commit, dirty=dirty, host=host,
                        elapsed=d.get("wall_time_seconds"), timestamp=str(d.get("timestamp", ""))[:19],
                        history_len=len(hist),
                        extra={"h_neval": list(h.get("n_evaluated") or []), "h_current": list(h.get("current_score") or []),
                               "n_cold_start": (d.get("config") or {}).get("n_cold_start", 5)})
        arm.cells[key] = cell
    # ---- crash sidecars: a crashed cell without a result file is counted, not missing ------
    crash_with_result = []
    for key, sns in sorted(side_cells["crash"].items()):
        p = os.path.join(spec.dir, sns[0])
        try:
            raw = open(p, "rb").read()
            info = json.loads(raw)
        except (OSError, json.JSONDecodeError) as e:
            problems.append(("content", spec.name, f"unreadable crash sidecar {p}: {e}"))
            continue
        if key in seen:
            crash_with_result.append(f"{key[0]}:{key[1]}")
            continue
        arm.sidecar_crashes[key] = {"path": p, "sha256": hashlib.sha256(raw).hexdigest(),
                                    "kind": info.get("kind"), "returncode": info.get("returncode"),
                                    "host": info.get("host"), "finished": info.get("finished")}
    rep["crash_sidecar_cells"] = sorted(f"{t}:{s}={v['kind']}" for (t, s), v in arm.sidecar_crashes.items())
    rep["crash_sidecar_with_result"] = crash_with_result
    rep["collision_cells"] = sorted(f"{t}:{s}" for t, s in side_cells["collision"])
    # ---- missing cells against the declared set --------------------------------------
    expected = spec.expected_cells()
    present = set(seen) | set(arm.sidecar_crashes)
    missing = expected - present     # a present-but-invalid file is a content problem, not a missing cell
    undeclared = sorted(missing - spec.known_missing)
    stale = sorted(spec.known_missing - missing)
    rep["n_cells_expected"] = len(expected)
    rep["missing_cells"] = [f"{t}:{s}" for t, s in sorted(missing)]
    rep["known_missing"] = [f"{t}:{s}" for t, s in sorted(spec.known_missing)]
    unresolved = {c: k for k in ("interrupted", "running") for c in side_cells[k] if c in missing}
    rep["unresolved_cells"] = sorted(f"{t}:{s}={k}" for (t, s), k in unresolved.items())
    if undeclared:
        why = [f"{t}:{s}" + (f" ({unresolved[(t, s)]}: resume or finish the queue)" if (t, s) in unresolved else "")
               for t, s in undeclared]
        problems.append(("missing_cell", spec.name,
                         f"{len(undeclared)} undeclared missing cell(s): " + ", ".join(why[:40]) +
                         (" ..." if len(undeclared) > 40 else "")))
    if stale:
        problems.append(("missing_cell", spec.name,
                         "declared known_missing but present (a result file or a crash sidecar exists): "
                         + ", ".join(f"{t}:{s}" for t, s in stale)))
    # ---- provenance ------------------------------------------------------------------
    states = Counter((str(c.commit), str(c.dirty), str(c.host)) for c in arm.cells.values())
    rep["provenance_states"] = {"|".join(k): v for k, v in sorted(states.items())}
    if spec.provenance == "require":
        bad = [c for c in arm.cells.values()
               if not commit_matches(c.commit, expected_commit or "") or c.dirty is not False]
        if bad:
            problems.append(("provenance", spec.name,
                             f"{len(bad)} file(s) without commit {expected_commit} and git_dirty False, e.g. " +
                             "; ".join(f"{os.path.basename(c.file)} ({c.commit}, dirty={c.dirty})" for c in bad[:5])))
    # ---- non-fatal counts --------------------------------------------------------------
    sc = Counter(c.status for c in arm.cells.values())
    for v in arm.sidecar_crashes.values():
        sc[f"crash_sidecar:{v['kind']}"] += 1
    rep["status_counts"] = dict(sc)
    rep["noncompleted_cells"] = sorted(
        [f"{c.task}:{c.seed}={c.status}" for c in arm.cells.values() if not c.status_ok] +
        [f"{t}:{s}=crash_sidecar:{v['kind']}" for (t, s), v in arm.sidecar_crashes.items()])
    if spec.fmt == "ours":
        rep["short_trace_cells"] = sorted(f"{c.task}:{c.seed}" for c in arm.cells.values()
                                          if c.status_ok and not c.ok_len(spec.trace_len))
    rep["fallback_recorded_cells"] = sum(1 for c in arm.cells.values() if c.fallback_recorded)
    fb = [c for c in arm.cells.values() if c.fallback_count > 0]
    rep["fallback_cells"] = len(fb)
    rep["fallback_steps"] = int(sum(c.fallback_count for c in fb))
    rep["fallback_cells_list"] = sorted(f"{c.task}:{c.seed}({c.fallback_count})" for c in fb)
    ts = sorted(c.timestamp for c in arm.cells.values() if c.timestamp)
    rep["timestamp_range"] = [ts[0], ts[-1]] if ts else None
    rep["n_cells_loaded"] = len(arm.cells)
    rep["n_crash_sidecar_cells"] = len(arm.sidecar_crashes)
    rows = sorted([(c.file, c.sha256) for c in arm.cells.values()] +
                  [(v["path"], v["sha256"]) for v in arm.sidecar_crashes.values()])
    rep["files_list_sha256"] = hashlib.sha256("\n".join(p for p, _ in rows).encode()).hexdigest()
    rep["files_content_sha256"] = hashlib.sha256("\n".join(f"{p}\t{s}" for p, s in rows).encode()).hexdigest()
    return arm


def attach_external_traces(arms: dict[str, ArmData]) -> None:
    """External harness files store 25 best-so-far values after the cold start; the
    trajectory prepends the shared cold-start prefix best_trace[:5] of the same cell from
    the arm named in ``coldstart_from`` (same config namespace, hence same generation)."""
    for arm in arms.values():
        if arm.absent or arm.spec.fmt != "external":
            continue
        src = arms.get(arm.spec.coldstart_from)
        n_missing, n_mismatch = 0, 0
        for key, c in arm.cells.items():
            sc = src.cells.get(key) if src is not None and not src.absent else None
            if sc is None or sc.trace is None or len(sc.trace) < 5:
                n_missing += 1
                continue
            cs = list(sc.trace[:5])
            run, best = [], cs[-1]
            for v in c.scores:
                best = max(best, v)
                run.append(best)
            c.trace = np.asarray(cs + run, dtype=float)
            if len(c.trace) >= 30 and abs(c.trace[29] - c.value30) > 1e-12:
                n_mismatch += 1
        arm.report["coldstart_source"] = arm.spec.coldstart_from
        arm.report["cells_without_coldstart_prefix"] = n_missing
        arm.report["cells_trace30_ne_maxhistory"] = n_mismatch


def coldstart_consistency(arms: dict[str, ArmData], namespace: str = "") -> dict:
    """Do all 'ours' arms of a namespace share the first five scores in every cell?"""
    ref: dict = {}
    mism = Counter()
    checked = 0
    for name, arm in arms.items():
        if arm.absent or arm.spec.fmt != "ours" or arm.spec.namespace != namespace or arm.spec.trace_len != 30:
            continue
        for key, c in arm.cells.items():
            if c.scores is None or len(c.scores) < 5:
                continue
            head = tuple(np.round(c.scores[:5], 12))
            if key not in ref:
                ref[key] = (name, head)
                continue
            checked += 1
            if head != ref[key][1]:
                mism[name] += 1
    return {"namespace": namespace or "(main)", "cells_compared": checked,
            "mismatching_cells_by_arm": dict(mism)}


@dataclass
class Dataset:
    config: SourcesConfig
    arms: dict
    cell_filter: str
    expected_commit: object          # None | str | {root: commit}
    absent: dict
    report: dict

    def present(self, name: str) -> bool:
        return name in self.arms and not self.arms[name].absent

    def arm(self, name: str) -> ArmData:
        a = self.arms.get(name)
        if a is None or a.absent:
            raise KeyError(f"arm {name} is absent")
        return a


def load_all(config: SourcesConfig, expected_commit, allow_missing_arms: set | None = None,
             require_optional: bool = False, only: set | None = None, log=print) -> Dataset:
    """``expected_commit``: None (legacy), one commit (str) or {root: commit, "*": default} (see
    ``expected_commit_for``)."""
    allow_missing_arms = set(allow_missing_arms or ())
    problems: list = []
    arms: dict[str, ArmData] = {}
    for name, spec in config.arms.items():
        if only is not None and name not in only:
            continue
        a = load_arm(spec, expected_commit, problems)
        arms[name] = a
        if not a.absent:
            log(f"  loaded {name:48s} {len(a.cells):4d} cells  {a.report.get('status_counts')}")
    absent = {}
    for name, a in arms.items():
        if not a.absent:
            continue
        spec = a.spec
        why = a.report.get("absent")
        if a.report.get("outside_subset"):
            absent[name] = why
        elif name in allow_missing_arms:
            absent[name] = f"allowed by --allow-missing-arms ({why})"
        elif not spec.required and not require_optional:
            absent[name] = f"optional arm absent ({why})"
        else:
            problems.append(("missing_arm", name, f"{why}: {spec.dir}/{spec.pattern}"))
    unknown = allow_missing_arms - set(config.arms)
    if unknown:
        problems.append(("config", "--allow-missing-arms", f"names not in the config: {sorted(unknown)}"))
    if problems:
        raise LoaderError(problems)
    attach_external_traces(arms)
    namespaces = sorted({a.spec.namespace for a in arms.values()})
    report = {
        "config": config.path, "config_sha256": config.sha256, "includes": config.includes,
        "cell_filter": config.cell_filter, "expected_commit": expected_commit,
        "tasks": config.tasks, "seeds": [min(config.seeds), max(config.seeds)] if config.seeds else [],
        "n_seeds": len(config.seeds),
        "absent_arms": absent,
        "coldstart_consistency": [coldstart_consistency(arms, ns) for ns in namespaces],
        "arms": {n: a.report for n, a in arms.items()},
    }
    return Dataset(config=config, arms=arms, cell_filter=config.cell_filter,
                   expected_commit=expected_commit, absent=absent, report=report)
