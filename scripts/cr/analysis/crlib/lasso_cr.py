"""LassoBench camera-ready rerun: loader with provenance join and loud coverage reporting.

RUN_RULES amendments 2026-10-12 ("LassoBench is rerun in full" and "Papenmeier at D = 7129 is run on all 40
seeds"): every LassoBench arm is rerun from tag ``cr-runs-2026-10-12`` (commit 14481cb) on kajman's CPU, one
thread and one hour per run. The rerun lives in ``<lasso-cr-root>/`` (a byte copy of
``kajman:/data2/prusek/NeurIPS-cr-ana/results/lassobench_cr`` made by ``tools/sync_lassobench_cr.sh``, with
sha256 manifests) as three run folders:

  full_rerun/          ten methods x five tasks (D <= 1000) x seeds 42-81; Hvarfner GP and Papenmeier on
                       leukemia (D = 7129), seeds 42-81; the Xu leukemia pilot (seed 42).
                       Papenmeier on leukemia started as a one-seed pilot: its seed 42 is the pilot's job
                       ``pilot_papenmeier_leukemia_s42``, its seeds 43-81 are the 39 jobs
                       ``papenmeier_leukemia_s<seed>`` that were queued after the first 1,442 jobs (same
                       command, clone, environment, thread count and one-hour limit as the pilot)
  leukemia_rerun/      the six cheap arms (random, GeoCMA, GeoWalk, full-D CMA-ES, TuRBO, BAxUS) on leukemia
  linear_bo_leukemia/  the Linear BO leukemia pilot (seed 42)

each with ``results/<method>_<task>_s<seed>.json``, ``manifest.jsonl`` (one provenance record per job, appended
when the job ends), ``cmds/<label>.cmd`` (written when a job starts) and ``status/<label>.rc`` (return code,
written after the record).

The result JSONs carry NO provenance, so the loader joins every result file to the job record that lists it
(``result_files[].file``) and requires a CLEAN record:

  * commit == 14481cb..., tag == cr-runs-2026-10-12, porcelain_lines_after == 0 (clean tree),
  * runner_script_sha256 == the constant below and runner_script_is_committed_file,
  * rc == 0,
  * the file's sha256 == the sha256 the record stored when the job ended.

Fatal (one ``LassoLoadError`` listing every problem):
  * a result file whose job record is not clean (unless its name is declared in ``allow_unclean``),
  * a result file without any job record in a COMPLETE run folder (every expected job has a record),
  * a result file listed by a record but absent, or with a different sha256,
  * a result whose method / task / seed fields contradict its file name, an unreadable file,
  * the same (method, task, seed) cell in two run folders, two records with the same job label,
  * a status file without a job record, or with another return code than the record,
  * a manifest line that is not JSON.

Tolerated, never silent (counted per cell, listed in ``coverage`` and in the loud warning):
  * missing cells: expected (method, task, seed) cells without a result file -- ``timeout`` (job rc 124, or rc 137
    when ``timeout --kill-after`` had to kill the run after the limit: its record lasted at least ``timeout_s``),
    ``failed`` (another non-zero rc, e.g. 137 well before the limit: the out-of-memory killer), ``pending`` (no job record yet: the rerun is still running; ``running``
    marks the pending cells whose job has started, i.e. has a ``cmds`` file);
  * result files without a job record while the run folder is INCOMPLETE (their job had not ended when the
    set was copied): they are NOT used and are listed as ``unrecorded``;
  * non-finite losses (the runner writes best_score = inf with an ``error`` field when a model fit fails).

Expected design (the two amendments; ``expected_cells``, ``expected_jobs``, ``design_text``): methods x
{synt_simple 60, synt_medium 100, dna 180, synt_high 300, synt_hard 1000} x seeds 42-81 for all ten methods; at
leukemia (7129) seeds 42-81 for the six cheap arms, the Hvarfner GP and Papenmeier (eight 40-seed arms, each
treated alike: a seed that times out or ends non-finite is excluded and counted, the table prints the mean over
the completed seeds with their number); Xu and Linear BO have one pilot seed each and are reported as "did not
complete" when the pilot has no result. A pilot that does complete is labelled as a pilot (one seed), never as
full coverage. 2,322 runs in 1,481 + 40 + 1 jobs.

Finished or not: the rerun is FINISHED when every expected job of every run folder has a job record and no
expected run is pending or unrecorded (``summary['finished']``). A run that timed out, failed or ended
non-finite is a final outcome and does not keep the rerun unfinished; a job without a record does (on
2026-10-12: the 39 Papenmeier leukemia jobs until they end). Tables of an unfinished rerun are marked
"INCOMPLETE RERUN: not the final table", and ``load_rerun(..., require_finished=True)``
(``make_tables.py --lasso-require-finished``, passed by the final run) raises instead of loading it.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import datetime as _dt
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

LASSO_CR_COMMIT = "14481cb55d21a770da11c3d6f866fcae98891c18"
LASSO_CR_TAG = "cr-runs-2026-10-12"
RUNNER_SHA256 = "c876261591925d1607da7d0b186b1a2f1c3874ef17fcb5558613836fe611814b"
RUN_SETS = ("full_rerun", "leukemia_rerun", "linear_bo_leukemia")
SEEDS = list(range(42, 82))
TASKS_LOW = ["synt_simple", "synt_medium", "dna", "synt_high", "synt_hard"]
LEUKEMIA = "leukemia"
ALL_METHODS = ["random", "cmaes_sub", "geowalk", "cmaes_full", "turbo", "baxus", "hvarfner_gp", "papenmeier_msr",
               "xu_standard_gp", "linear_bo"]
CHEAP_LEUKEMIA = ["random", "cmaes_sub", "geowalk", "cmaes_full", "turbo", "baxus"]       # run folder leukemia_rerun
FULL_LEUKEMIA = ["hvarfner_gp", "papenmeier_msr"]      # 40-seed leukemia arms of run folder full_rerun
PILOT_LEUKEMIA = ["xu_standard_gp", "linear_bo"]       # one pilot seed each (Papenmeier's pilot completed in time,
PILOT_SEED = 42                                        # so the arm is run on all 40 seeds: amendment 2026-10-12)
PAPENMEIER_PILOT_LABEL = "pilot_papenmeier_leukemia_s42"       # the job that made seed 42 of the 40-seed arm
N_TOTAL = 30                      # evaluations per run (5 cold start + 25)
N_COLD = 5
FILE_RX = re.compile(r"^(?P<m>.+)_(?P<t>synt_simple|synt_medium|synt_high|synt_hard|dna|leukemia)_s(?P<s>\d+)\.json$")
# job groups of the full rerun (scripts of the queue: full_rerun_jobs.py)
FULL_GROUPS = [("cheap", ["random", "cmaes_sub", "geowalk", "cmaes_full"]), ("turbo", ["turbo"]), ("baxus", ["baxus"]),
               ("hvarfner", ["hvarfner_gp"]), ("papenmeier", ["papenmeier_msr"]), ("xu", ["xu_standard_gp"]),
               ("linbo", ["linear_bo"])]


class LassoLoadError(Exception):
    """Fatal problems of the rerun set. ``problems`` is a list of (kind, where, message)."""

    def __init__(self, problems):
        self.problems = problems
        super().__init__(f"{len(problems)} fatal LassoBench rerun problem(s):\n  "
                         + "\n  ".join(f"[{k}] {w}: {m}" for k, w, m in problems))

    def kinds(self) -> set:
        return {k for k, _, _ in self.problems}


def expected_cells() -> dict:
    """{(method, task): {"seeds": [...], "set": run folder, "pilot": bool}} -- the design of the amendment."""
    out = {}
    for m in ALL_METHODS:
        for t in TASKS_LOW:
            out[(m, t)] = {"seeds": list(SEEDS), "set": "full_rerun", "pilot": False}
    for m in CHEAP_LEUKEMIA:
        out[(m, LEUKEMIA)] = {"seeds": list(SEEDS), "set": "leukemia_rerun", "pilot": False}
    for m in FULL_LEUKEMIA:
        out[(m, LEUKEMIA)] = {"seeds": list(SEEDS), "set": "full_rerun", "pilot": False}
    out[("xu_standard_gp", LEUKEMIA)] = {"seeds": [PILOT_SEED], "set": "full_rerun", "pilot": True}
    out[("linear_bo", LEUKEMIA)] = {"seeds": [PILOT_SEED], "set": "linear_bo_leukemia", "pilot": True}
    return out


def design_text() -> str:
    """The expected design in words, counted from ``expected_cells`` (so the sentence cannot disagree with it):
    '10 methods x 5 tasks x 40 seeds; 8 arms x 40 seeds and 2 pilot seeds at D = 7129'."""
    ex = expected_cells()
    low = sorted({m for (m, t) in ex if t != LEUKEMIA})
    arms = [m for (m, t), e in ex.items() if t == LEUKEMIA and not e["pilot"]]
    pilots = sum(len(e["seeds"]) for (m, t), e in ex.items() if t == LEUKEMIA and e["pilot"])
    return (f"{len(low)} methods x {len(TASKS_LOW)} tasks x {len(SEEDS)} seeds; {len(arms)} arms x {len(SEEDS)} "
            f"seeds and {pilots} pilot seed{'' if pilots == 1 else 's'} at D = 7129")


def expected_jobs() -> dict:
    """{run folder: {job label: (methods, task, seed)}} -- 1,481 + 40 + 1 jobs. The 1,481 jobs of full_rerun are
    the 1,442 of queue/jobs_full_rerun.txt and the 39 Papenmeier leukemia jobs (seeds 43-81) of
    queue/jobs_papenmeier_leukemia.txt; seed 42 of that arm is the pilot's job and has no second job."""
    full = {PAPENMEIER_PILOT_LABEL: (["papenmeier_msr"], LEUKEMIA, PILOT_SEED),
            "pilot_xu_leukemia_s42": (["xu_standard_gp"], LEUKEMIA, PILOT_SEED)}
    for s in SEEDS:
        for t in TASKS_LOW:
            for name, ms in FULL_GROUPS:
                full[f"{name}_{t}_s{s}"] = (list(ms), t, s)
        full[f"hvarfner_leukemia_s{s}"] = (["hvarfner_gp"], LEUKEMIA, s)
        if s != PILOT_SEED:
            full[f"papenmeier_leukemia_s{s}"] = (["papenmeier_msr"], LEUKEMIA, s)
    leuk = {f"s{s}": (list(CHEAP_LEUKEMIA), LEUKEMIA, s) for s in SEEDS}
    lin = {f"s{PILOT_SEED}": (["linear_bo"], LEUKEMIA, PILOT_SEED)}
    return {"full_rerun": full, "leukemia_rerun": leuk, "linear_bo_leukemia": lin}


@dataclass(frozen=True)
class Design:
    """A LassoBench run set: the commit, tag and runner its job records must carry, its run folders, its expected
    design (which may follow from exit codes, e.g. pilots) and the checks of the result files themselves."""
    name: str
    generation: str
    commit: str
    tag: str
    runner_sha256: str
    run_sets: tuple
    methods: tuple
    cells: Callable            # (records_by_set) -> {(method, task): {"seeds", "set", "pilot"}}
    jobs: Callable             # (records_by_set) -> {run folder: {label: (methods, task, seed)}}
    result_problems: Callable | None = None     # (result dict) -> [problems] of the file's own provenance
    check_root: Callable | None = None          # (root, expected jobs) -> [(kind, where, message)]
    unread: frozenset = field(default_factory=frozenset)   # methods whose files are checked but never parsed
    record_checks: Callable | None = None       # (records_by_set) -> [(kind, where, message)] (fatal)


#: fields of the 40 Papenmeier leukemia job records that must equal the pilot's (RUN_RULES amendment 2026-10-12:
#: "the same command, clone, environment, thread count and one-hour limit as the pilot"; GEN v5 backlog item 7)
PAP_ENV_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "CUDA_VISIBLE_DEVICES", "PYTHONPATH", "nice",
                "python", "python_version", "freeze_md5")
PAP_FIELDS = {
    "command (seed masked)": lambda r: re.sub(r"--seeds \d+", "--seeds <seed>", str(r.get("command"))),
    "clone": lambda r: r.get("clone"), "cwd": lambda r: r.get("cwd"), "host": lambda r: r.get("host"),
    "timeout_s": lambda r: str(r.get("timeout_s")), "commit": lambda r: r.get("commit"), "tag": lambda r: r.get("tag"),
    "runner_script_sha256": lambda r: r.get("runner_script_sha256"),
    "env (threads, python, freeze md5)": lambda r: json.dumps({k: (r.get("env") or {}).get(k) for k in PAP_ENV_KEYS},
                                                              sort_keys=True),
}


def timed_out(rec: dict) -> bool:
    """rc 124 (``timeout`` sent SIGTERM at the limit), or rc 137 after at least the limit: ``timeout --kill-after=10``
    exits 137 when it had to send SIGKILL (review M4). A 137 well before the limit (the out-of-memory killer) or a
    137 whose record lacks start/end/timeout_s counts as failed."""
    rc = rec.get("rc")
    if rc == 124:
        return True
    if rc != 137:
        return False
    try:
        el = (_dt.datetime.fromisoformat(str(rec["end"])) - _dt.datetime.fromisoformat(str(rec["start"]))).total_seconds()
        return el >= float(rec["timeout_s"])
    except (KeyError, TypeError, ValueError):
        return False


def papenmeier_leukemia_check(records_by_set: dict) -> list:
    """Every Papenmeier leukemia job record of full_rerun must carry the pilot's command (seed masked), clone, cwd,
    host, time limit, commit, tag, runner and environment (threads, python, freeze md5). Fatal otherwise; the
    environment's freeze FILE name is informative only (the scratchpad helper of GEN v4, now a loader check)."""
    pap = {r.get("seed"): r for r in records_by_set.get("full_rerun", [])
           if r.get("methods") == ["papenmeier_msr"] and r.get("task") == LEUKEMIA}
    if not pap or PILOT_SEED not in pap:
        return []            # the pilot has no record yet: the unfinished-rerun logic reports it
    pilot = pap[PILOT_SEED]
    out = []
    for name, f in PAP_FIELDS.items():
        diff = sorted(sd for sd, r in pap.items() if f(r) != f(pilot))
        if diff:
            out.append(("provenance", "full_rerun papenmeier_msr leukemia",
                        f"{name} differs from the pilot's (seed {PILOT_SEED}) at seeds {compress(diff)}"))
    return out


RERUN = Design(name="rerun", generation="rerun", commit=LASSO_CR_COMMIT, tag=LASSO_CR_TAG,
               runner_sha256=RUNNER_SHA256, run_sets=RUN_SETS, methods=tuple(ALL_METHODS),
               cells=lambda _r: expected_cells(), jobs=lambda _r: expected_jobs(),
               record_checks=papenmeier_leukemia_check)

# ---- the LassoBench published methods (RUN_RULES 2026-10-12 (3), revision 1 (f); tag cr-verify-2026-10-13) --------
PUB_COMMIT = "963ef040cae648d4132e1d3330a2a9c06456e97d"
PUB_TAG = "cr-verify-2026-10-13"
PUB_RUNNER_SHA256 = "cbdacba5364b38490fcda03897976fe09823567c5bcdbf26903f291cdaf86d3e"
PUB_SET = "pub_methods"
PUB_METHODS = ("hvarfner_published", "papenmeier_published", "papenmeier_mle_scaled_published",
               "papenmeier_published_learned", "papenmeier_mle_scaled_published_learned",
               "xu_published", "linear_bo_published", "turbo_published", "baxus_official", "cmaes_full_published")
#: frozen-noise variants: the verification chose the learned variants (RUN_RULES 2026-10-19), so these are not printed
#: and their result files are never parsed (record and sha256 are still checked)
PUB_UNREAD = frozenset({"papenmeier_published", "papenmeier_mle_scaled_published"})
PUB_TIMEOUT = "3600"


def _pub_pilot_ok(records_by_set: dict) -> dict:
    """method -> True (pilot exited 0), False (other exit code), None (no record yet). Phase 2 (leukemia seeds
    43-81) ran exactly for the methods whose pilot exited 0 inside the hour (pub_queue.sh)."""
    rc = {r.get("label"): r.get("rc") for r in records_by_set.get(PUB_SET, [])}
    out = {}
    for mth in PUB_METHODS:
        lab = f"pilot_{mth}_leukemia_s{PILOT_SEED}"
        out[mth] = None if lab not in rc else rc[lab] == 0
    return out


def pub_expected_jobs(records_by_set: dict) -> dict:
    ok = _pub_pilot_ok(records_by_set)
    jobs = {f"pilot_{mth}_leukemia_s{PILOT_SEED}": ([mth], LEUKEMIA, PILOT_SEED) for mth in PUB_METHODS}
    for seed in SEEDS:
        for t in TASKS_LOW:
            for mth in PUB_METHODS:
                jobs[f"{mth}_{t}_s{seed}"] = ([mth], t, seed)
    for seed in SEEDS:
        if seed == PILOT_SEED:
            continue
        for mth in PUB_METHODS:
            if ok[mth]:
                jobs[f"{mth}_leukemia_s{seed}"] = ([mth], LEUKEMIA, seed)
    return {PUB_SET: jobs}


def pub_expected_cells(records_by_set: dict) -> dict:
    ok = _pub_pilot_ok(records_by_set)
    out = {}
    for mth in PUB_METHODS:
        for t in TASKS_LOW:
            out[(mth, t)] = {"seeds": list(SEEDS), "set": PUB_SET, "pilot": False}
        out[(mth, LEUKEMIA)] = ({"seeds": list(SEEDS), "set": PUB_SET, "pilot": False} if ok[mth]
                                else {"seeds": [PILOT_SEED], "set": PUB_SET, "pilot": True})
    return out


def pub_result_problems(d: dict) -> list[str]:
    """GEN v5 backlog 2026-10-13 22:00: a published-method file is admitted only if its own provenance names the
    tag commit and a clean tree (an empty commit or a missing dirty flag is refused, never guessed)."""
    p = d.get("provenance") if isinstance(d.get("provenance"), dict) else {}
    gc, dirty = str(p.get("git_commit") or ""), p.get("git_dirty")
    bad = []
    if len(gc) < 7 or not PUB_COMMIT.startswith(gc):
        bad.append(f"provenance.git_commit {gc!r} is not the tag commit {PUB_COMMIT[:7]}")
    if dirty is not False:
        bad.append(f"provenance.git_dirty {dirty!r} (must be False)")
    return bad


def pub_queue_dir(root: str) -> str:
    """The queue set of the published methods: <root>_queue/queue (the copy of tools/sync_lassobench_pub.sh, a
    sibling of <root> as lassobench_cr_queue is of lassobench_cr), else <root>/queue."""
    sib = os.path.join(os.path.normpath(root) + "_queue", "queue")
    return sib if os.path.isdir(sib) else os.path.join(root, "queue")


def pub_check_root(root: str, exp_jobs: dict) -> list:
    """The copied queue job lists (jobs_phase1.txt, jobs_phase2.txt of pub_queue_dir) must name exactly the expected
    jobs, with the same method, task, seed and one-hour limit (the job lines of the queue are the protocol)."""
    probs, got = [], {}
    qdir = pub_queue_dir(root)
    for f in ("jobs_phase1.txt", "jobs_phase2.txt"):
        p = os.path.join(qdir, f)
        if not os.path.exists(p):
            if f == "jobs_phase2.txt" and not any(lab.endswith("_leukemia_s43") for lab in exp_jobs[PUB_SET]):
                continue
            probs.append(("queue", f"queue/{f}", "job list absent from the copy (sync with tools/sync_lassobench_pub.sh)"))
            continue
        for i, line in enumerate(open(p), 1):
            tok = line.split()
            if not tok:
                continue
            if len(tok) != 10 or tok[2] != PUB_SET or tok[4] != "results":
                probs.append(("queue", f"queue/{f}:{i}", f"unexpected job line {line.strip()[:120]!r}"))
                continue
            lab, mth, task, seed, tmo = tok[3], tok[6], tok[7], int(tok[8]), tok[9]
            if lab in got:
                probs.append(("queue", f"queue/{f}:{i}", f"job {lab} listed twice"))
            got[lab] = ([mth], task, seed)
            if tmo != PUB_TIMEOUT:
                probs.append(("queue", f"queue/{f}:{i}", f"job {lab}: time limit {tmo} s, not {PUB_TIMEOUT}"))
    exp = exp_jobs[PUB_SET]
    if got or not probs:
        only_q = sorted(set(got) - set(exp))
        only_e = sorted(set(exp) - set(got))
        if only_q:
            probs.append(("queue", "queue job lists", f"{len(only_q)} jobs not in the expected design: {only_q[:5]}"))
        if only_e:
            probs.append(("queue", "queue job lists", f"{len(only_e)} expected jobs not in the queue lists: {only_e[:5]}"))
        diff = sorted(lab for lab in set(got) & set(exp) if got[lab] != (list(exp[lab][0]), exp[lab][1], exp[lab][2]))
        if diff:
            probs.append(("queue", "queue job lists", f"jobs whose method/task/seed differ from the design: {diff[:5]}"))
    return probs


#: fields of a published method's phase-2 leukemia records (seeds 43-81) that must equal its pilot's: RUN_RULES
#: 2026-10-12 (3), "at D = 7129 one pilot seed decides, as in the LassoBench rows above", i.e. the same command, clone,
#: environment, thread count and one-hour limit as the pilot (amendment 2026-10-12, Papenmeier). The host is not part
#: of that rule and is not compared (review M12).
PUB_PILOT_FIELDS = {k: f for k, f in PAP_FIELDS.items() if k != "host"}


def pub_pilot_check(records_by_set: dict) -> list:
    """Every phase-2 leukemia record of a published method must carry its pilot's command (seed masked), clone,
    cwd, time limit, commit, tag, runner and environment (threads, python, freeze md5). Fatal otherwise."""
    recs = records_by_set.get(PUB_SET, [])
    out = []
    for mth in PUB_METHODS:
        lab = f"pilot_{mth}_leukemia_s{PILOT_SEED}"
        pilot = next((r for r in recs if r.get("label") == lab), None)
        if pilot is None or pilot.get("rc") != 0:
            continue                       # no phase 2 (or no pilot record yet: the unfinished-set logic reports it)
        ph2 = {r.get("seed"): r for r in recs
               if r.get("methods") == [mth] and r.get("task") == LEUKEMIA and r.get("label") != lab}
        for name, f in PUB_PILOT_FIELDS.items():
            diff = sorted(sd for sd, r in ph2.items() if f(r) != f(pilot))
            if diff:
                out.append(("provenance", f"{PUB_SET} {mth} leukemia",
                            f"{name} differs from the pilot's (seed {PILOT_SEED}) at seeds {compress(diff)}"))
    return out


PUB = Design(name="published", generation="published", commit=PUB_COMMIT, tag=PUB_TAG,
             runner_sha256=PUB_RUNNER_SHA256, run_sets=(PUB_SET,), methods=PUB_METHODS, cells=pub_expected_cells,
             jobs=pub_expected_jobs, result_problems=pub_result_problems, check_root=pub_check_root, unread=PUB_UNREAD,
             record_checks=pub_pilot_check)


def pub_design_text(records_by_set: dict | None = None, L: dict | None = None) -> str:
    """'10 methods x 5 tasks x 40 seeds; 10 pilots at D = 7129, of which k ran on 40 seeds'."""
    if L is not None:
        k = sum(1 for (mth, t), c in L["coverage"].items() if t == LEUKEMIA and not c["pilot"])
    else:
        k = sum(1 for v in _pub_pilot_ok(records_by_set or {}).values() if v)
    return (f"{len(PUB_METHODS)} methods x {len(TASKS_LOW)} tasks x {len(SEEDS)} seeds; {len(PUB_METHODS)} pilot seeds "
            f"at D = 7129, of which {k} ran on all {len(SEEDS)} seeds (pilot inside the one-hour limit)")


def compress(seeds) -> str:
    """[42, 43, 44, 47] -> '42-44, 47'."""
    s = sorted(set(int(x) for x in seeds))
    if not s:
        return ""
    out, a, b = [], s[0], s[0]
    for x in s[1:]:
        if x == b + 1:
            b = x
            continue
        out.append(f"{a}-{b}" if b > a else f"{a}")
        a = b = x
    out.append(f"{a}-{b}" if b > a else f"{a}")
    return ", ".join(out)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _finite(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def record_problems(rec: dict, design: "Design | None" = None) -> list[str]:
    """Why a job record is not clean ([] = clean)."""
    D = design or RERUN
    bad = []
    if rec.get("commit") != D.commit:
        bad.append(f"commit {rec.get('commit')!r}")
    if rec.get("tag") != D.tag:
        bad.append(f"tag {rec.get('tag')!r}")
    if rec.get("porcelain_lines_after") != 0:
        bad.append(f"porcelain_lines_after {rec.get('porcelain_lines_after')!r}")
    if rec.get("runner_script_sha256") != D.runner_sha256:
        bad.append(f"runner_script_sha256 {str(rec.get('runner_script_sha256'))[:16]}")
    if rec.get("runner_script_is_committed_file") is not True:
        bad.append("runner script is not the committed file")
    if rec.get("rc") != 0:
        bad.append(f"rc {rec.get('rc')!r}")
    return bad


def load_rerun(root: str, allow_unclean=(), require_finished: bool = False, design: "Design | None" = None) -> dict:
    """Load the rerun set under ``root``. Returns

    {"cells": {method: {task: {seed: (loss, path)}}},     finite losses of clean result files
     "nonfinite": {(method, task): n},                    clean result files with a non-finite loss
     "files": {method: [paths]},                          every clean result file (finite or not)
     "dir": root, "generation": "rerun",
     "manifests": [manifest.jsonl paths],                 part of every number's file set
     "coverage": {(method, task): {...}}, "sets": {...}, "summary": {...}, "leukemia": {...}}

    and raises LassoLoadError on any fatal problem (module docstring). ``allow_unclean``: result file names
    (``<method>_<task>_s<seed>.json``) whose unclean job record (e.g. rc != 0 of a multi-method job) the
    caller has declared; they are used and listed in ``summary['declared_unclean']``.
    ``require_finished``: raise LassoLoadError (kind ``unfinished``) unless ``summary['finished']``, i.e. while
    an expected job of any run folder has no job record yet (or a run folder, or the whole root, is absent)."""
    D = design or RERUN
    allow_unclean = set(allow_unclean or ())
    problems: list = []
    records_by_set = {}
    for sname in D.run_sets:          # pre-pass: designs whose expected jobs follow from exit codes (pilots)
        mpath = os.path.join(root, sname, "manifest.jsonl")
        recs = []
        if os.path.exists(mpath):
            with open(mpath) as fh:
                for line in fh:
                    try:
                        recs.append(json.loads(line)) if line.strip() else None
                    except json.JSONDecodeError:
                        pass                      # reported below
        records_by_set[sname] = recs
    exp_cells = D.cells(records_by_set)
    exp_jobs = D.jobs(records_by_set)
    if D.check_root is not None:
        problems += D.check_root(root, exp_jobs)
    if D.record_checks is not None:
        problems += D.record_checks(records_by_set)
    cells = defaultdict(lambda: defaultdict(dict))
    nonfinite = Counter()
    files = defaultdict(list)
    manifests = []
    sets = {}
    cell_set: dict = {}                       # (m, t, s) -> run folder holding its result file
    cell_info: dict = {}                      # (m, t, s) -> {"path", "finite", "error", "elapsed", "n_scores", ...}
    job_of: dict = {}                         # (m, t, s) -> (set, record) for every job that names the cell
    unrecorded: dict = {}                     # (m, t, s) -> path (result without a record, incomplete set)
    running: dict = {}                        # run folder -> labels of jobs that started and have no record yet
    declared = []
    for sname in D.run_sets:
        sdir = os.path.join(root, sname)
        mpath = os.path.join(sdir, "manifest.jsonl")
        info = {"dir": sdir, "present": os.path.isdir(sdir), "manifest": mpath if os.path.exists(mpath) else None,
                "n_records": 0, "n_expected_jobs": len(exp_jobs[sname]), "n_result_files": 0,
                "rc_counts": {}, "labels_unexpected": [], "labels_missing": len(exp_jobs[sname]),
                "labels_missing_list": sorted(exp_jobs[sname]), "complete": False,
                "n_started": 0, "running_labels": []}
        sets[sname] = info
        if not info["present"]:
            continue
        records = []
        if info["manifest"]:
            manifests.append(mpath)
            with open(mpath) as fh:
                for i, line in enumerate(fh, 1):
                    if not line.strip():
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        problems.append(("manifest", sname, f"{mpath} line {i} is not JSON ({e}); a line cut in a "
                                                            f"copy made while the rerun was writing: sync again"))
        labels = Counter(r.get("label") for r in records)
        for lab, n in labels.items():
            if n > 1:
                problems.append(("manifest", sname, f"job label {lab!r} has {n} records"))
        info["n_records"] = len(records)
        by_label = {r.get("label"): r for r in records}
        stdir, cmdir = os.path.join(sdir, "status"), os.path.join(sdir, "cmds")
        started = {n[:-4] for n in os.listdir(cmdir) if n.endswith(".cmd")} if os.path.isdir(cmdir) else set()
        info["n_started"] = len(started)
        info["running_labels"] = sorted(started - set(by_label))        # started, no record yet
        running[sname] = set(info["running_labels"])
        for n in sorted(os.listdir(stdir)) if os.path.isdir(stdir) else []:
            if not n.endswith(".rc"):
                continue
            lab = n[:-3]
            try:
                rc = int(open(os.path.join(stdir, n)).read().split()[0])
            except (OSError, ValueError, IndexError):
                rc = None
            if lab not in by_label:
                problems.append(("provenance", f"{sname}/status/{n}", "a finished job (status file) without a "
                                                                    "manifest record"))
            elif rc != by_label[lab].get("rc"):
                problems.append(("manifest", f"{sname}/status/{n}", f"status rc {rc!r} != the job record's rc "
                                                                  f"{by_label[lab].get('rc')!r}"))
        info["rc_counts"] = dict(Counter(str(r.get("rc")) for r in records))
        info["labels_unexpected"] = sorted(set(labels) - set(exp_jobs[sname]))
        info["labels_missing_list"] = sorted(set(exp_jobs[sname]) - set(labels))
        info["labels_missing"] = len(info["labels_missing_list"])
        info["complete"] = info["labels_missing"] == 0
        listed: dict = {}                     # result file name -> (record, result_files entry)
        for r in records:
            for rf in r.get("result_files") or []:
                name = os.path.basename(str(rf.get("file", "")))
                m = FILE_RX.match(name)
                if not m:
                    problems.append(("manifest", sname, f"job {r.get('label')!r} lists an unparsable file {name!r}"))
                    continue
                key = (m["m"], m["t"], int(m["s"]))
                if key in job_of and job_of[key][0] != sname:
                    problems.append(("duplicate", f"{sname}/{name}",
                                     f"cell also named by a job of {job_of[key][0]}"))
                job_of[key] = (sname, r)
                if name in listed:
                    problems.append(("manifest", sname, f"result file {name} is listed by two job records"))
                listed[name] = (r, rf)
        rdir = os.path.join(sdir, "results")
        names = sorted(n for n in os.listdir(rdir) if n.endswith(".json")) if os.path.isdir(rdir) else []
        info["n_result_files"] = len(names)
        for name in names:
            path = os.path.join(rdir, name)
            m = FILE_RX.match(name)
            if not m:
                problems.append(("content", f"{sname}/{name}", "file name does not parse as <method>_<task>_s<seed>.json"))
                continue
            key = (m["m"], m["t"], int(m["s"]))
            if name not in listed:
                if info["complete"]:
                    problems.append(("provenance", f"{sname}/{name}", "no job record lists this result file "
                                                                      "(the run folder is complete)"))
                else:
                    unrecorded[key] = path
                continue
            rec, rf = listed[name]
            bad = record_problems(rec, D)
            if bad:
                if name in allow_unclean:
                    declared.append(f"{sname}/{name} ({'; '.join(bad)})")
                else:
                    problems.append(("provenance", f"{sname}/{name}", f"job {rec.get('label')!r} is not clean: "
                                     + "; ".join(bad)))
                    continue
            try:
                raw = open(path, "rb").read()
            except OSError as e:
                problems.append(("content", f"{sname}/{name}", f"unreadable: {e}"))
                continue
            sha = hashlib.sha256(raw).hexdigest()
            if rf.get("sha256") != sha:
                problems.append(("provenance", f"{sname}/{name}", f"sha256 {sha[:16]} != the job record's "
                                                                  f"{str(rf.get('sha256'))[:16]}"))
                continue
            if key[0] in D.unread:
                # a method that is not printed (e.g. a frozen-noise variant): its record and sha256 are checked,
                # its content is never parsed
                if key in cell_set:
                    problems.append(("duplicate", f"{sname}/{name}", f"cell already loaded from {cell_set[key]}"))
                    continue
                cell_set[key] = sname
                cell_info[key] = {"path": path, "set": sname, "finite": None, "error": None, "elapsed_s": None,
                                  "n_scores": None, "fallback": None, "label": rec.get("label"), "unread": True}
                continue
            try:
                d = json.loads(raw)
            except json.JSONDecodeError as e:
                problems.append(("content", f"{sname}/{name}", f"unreadable: {e}"))
                continue
            if D.result_problems is not None:
                bad_prov = D.result_problems(d)
                if bad_prov:
                    problems.append(("provenance", f"{sname}/{name}", "result file provenance: " + "; ".join(bad_prov)))
                    continue
            if (d.get("method") not in (None, key[0]) or d.get("task") not in (None, key[1])
                    or (d.get("seed") is not None and int(d["seed"]) != key[2])):
                problems.append(("content", f"{sname}/{name}", f"fields method/task/seed {d.get('method')!r}/"
                                 f"{d.get('task')!r}/{d.get('seed')!r} contradict the file name"))
                continue
            if key in cell_set:
                problems.append(("duplicate", f"{sname}/{name}", f"cell already loaded from {cell_set[key]}"))
                continue
            cell_set[key] = sname
            v = d.get("best_score")
            fin = _finite(v)
            sc = d.get("all_scores") or []
            cell_info[key] = {"path": path, "set": sname, "finite": fin, "error": d.get("error"),
                              "elapsed_s": d.get("elapsed_s"), "n_scores": len(sc),
                              # the five cold-start evaluations, for the objective check (review S7; never printed)
                              "cold": [float(v) for v in sc[:N_COLD]] if len(sc) >= N_COLD else None,
                              "fallback": d.get("surrogate_fallback_count"), "label": rec.get("label"),
                              "status": d.get("status"), "unread": False}
            files[key[0]].append(path)
            if fin:
                cells[key[0]][key[1]][key[2]] = (float(v), path)
            else:
                nonfinite[(key[0], key[1])] += 1
        for name, (rec, rf) in listed.items():
            if rf.get("exists") and name not in names:
                problems.append(("provenance", f"{sname}/{name}", f"listed by job {rec.get('label')!r} as written "
                                                                  f"but absent from results/ (incomplete copy)"))
    if problems:
        raise LassoLoadError(problems)
    # ---- coverage against the expected design ------------------------------------------------------
    label_of = {(m, t, s): (sn, lab) for sn, js in exp_jobs.items() for lab, (ms, t, s) in js.items() for m in ms}
    coverage = {}
    for (m, t), e in exp_cells.items():
        c = {"expected": len(e["seeds"]), "pilot": e["pilot"], "set": e["set"], "present": [], "finite": [],
             "nonfinite": [], "timeout": [], "failed": [], "pending": [], "running": [], "unrecorded": [],
             "errors": Counter(), "unread": [], "not_applicable": [],
             "short": [], "fallback_steps": 0, "fallback_recorded": 0}
        for s in e["seeds"]:
            k = (m, t, s)
            if k in cell_info:
                ci = cell_info[k]
                c["present"].append(s)
                if ci.get("unread"):
                    c["unread"].append(s)
                    continue
                (c["finite"] if ci["finite"] else c["nonfinite"]).append(s)
                if ci.get("status") == "not_applicable":
                    c["not_applicable"].append(s)
                if ci["error"]:
                    c["errors"][str(ci["error"])[:100]] += 1
                if ci["finite"] and ci["n_scores"] != N_TOTAL:
                    c["short"].append(s)
                if ci["fallback"] is not None:
                    c["fallback_recorded"] += 1
                    c["fallback_steps"] += int(ci["fallback"])
            elif k in unrecorded:
                c["unrecorded"].append(s)
            elif k in job_of:
                (c["timeout"] if timed_out(job_of[k][1]) else c["failed"]).append(s)
            else:
                c["pending"].append(s)
                sn, lab = label_of.get(k, (None, None))
                if sn is not None and lab in running.get(sn, ()):
                    c["running"].append(s)
        c["missing"] = sorted(set(e["seeds"]) - set(c["present"]))
        c["errors"] = dict(c["errors"])
        coverage[(m, t)] = c
    extra = sorted(f"{m}_{t}_s{s}" for (m, t, s) in cell_info
                   if (m, t) not in exp_cells or s not in exp_cells[(m, t)]["seeds"])
    n_exp = sum(c["expected"] for c in coverage.values())
    n_present = sum(len(c["present"]) for c in coverage.values())
    summary = {"expected": n_exp, "present": n_present, "missing": n_exp - n_present,
               "unread": sum(len(c["unread"]) for c in coverage.values()),
               "not_applicable": sum(len(c["not_applicable"]) for c in coverage.values()),
               "finite": sum(len(c["finite"]) for c in coverage.values()),
               "nonfinite": sum(len(c["nonfinite"]) for c in coverage.values()),
               "timeout": sum(len(c["timeout"]) for c in coverage.values()),
               "failed": sum(len(c["failed"]) for c in coverage.values()),
               "pending": sum(len(c["pending"]) for c in coverage.values()),
               "running": sum(len(c["running"]) for c in coverage.values()),
               "unrecorded": sum(len(c["unrecorded"]) for c in coverage.values()),
               "short": sum(len(c["short"]) for c in coverage.values()),
               "outside_design": extra, "declared_unclean": declared,
               "unexpected_job_labels": {k: v["labels_unexpected"] for k, v in sets.items() if v["labels_unexpected"]},
               "complete": n_present == n_exp,
               "run_folders_complete": {k: v["complete"] for k, v in sets.items()},
               "jobs_expected": sum(v["n_expected_jobs"] for v in sets.values()),
               "jobs_without_record": sum(v["labels_missing"] for v in sets.values())}
    # finished = every expected job has ended and left a record; a timed-out, failed or non-finite run is a
    # final outcome ("complete" above is stricter: a result file for every expected run)
    summary["finished"] = bool(all(v["complete"] for v in sets.values()) and not summary["pending"]
                               and not summary["unrecorded"])
    if require_finished and not summary["finished"]:
        unfinished = []
        for k, v in sets.items():
            if not v["complete"]:
                miss = v["labels_missing_list"]
                unfinished.append(("unfinished", k, ("run folder absent; " if not v["present"] else "")
                                   + f"{len(miss)} of {v['n_expected_jobs']} expected jobs have no job record: "
                                   + ", ".join(miss[:6]) + (f", ... ({len(miss) - 6} more)" if len(miss) > 6 else "")))
        for (m, t), c in sorted(coverage.items()):
            for key, lab in (("pending", "no job record yet"), ("unrecorded", "result without a job record")):
                if c[key]:
                    unfinished.append(("unfinished", f"{m} {t}", f"{lab}: seeds {compress(c[key])}"))
        raise LassoLoadError(unfinished)
    # ---- leukemia status per method ------------------------------------------------------------------
    leuk = {}
    for m in D.methods:
        c = coverage[(m, LEUKEMIA)]
        if c["unread"]:
            st, txt = "unread", "not printed (its result files are not read)"
        elif c["not_applicable"] and not c["finite"]:
            st, txt = "not_applicable", "the recipe is undefined at this dimension"
        elif c["finite"] and not c["pilot"]:
            st, txt = "run", f"{len(c['finite'])} of {c['expected']} seeds with a finite loss"
        elif c["pilot"] and c["finite"]:
            st, txt = "pilot_completed", "the pilot seed completed within the time limit (one seed only)"
        elif c["timeout"]:
            lim = None
            k = (m, LEUKEMIA, c["timeout"][0])
            if k in job_of:
                lim = job_of[k][1].get("timeout_s")
            st = "not_completed"
            txt = ("pilot exceeded the time limit" if c["pilot"] else "runs exceeded the time limit") + \
                  (f" of {int(lim) // 60} min" if lim and str(lim).isdigit() else "")
        elif c["nonfinite"] or c["failed"]:
            st, txt = "not_completed", "no run produced a finite loss"
        elif c["running"]:
            st, txt = "pending", "job running when the set was copied (no result yet)"
        else:
            st, txt = "pending", "not run yet (no job record)"
        leuk[m] = {"status": st, "text": txt, "pilot": c["pilot"], "finite": len(c["finite"]),
                   "expected": c["expected"]}
    return {"cells": cells, "nonfinite": dict(nonfinite), "files": files, "dir": root, "generation": D.generation,
            "manifests": manifests, "coverage": coverage, "sets": sets, "summary": summary, "leukemia": leuk,
            "commit": D.commit, "tag": D.tag, "runner_sha256": D.runner_sha256, "design": D.name,
            "cold": {k: ci["cold"] for k, ci in cell_info.items() if ci.get("cold") is not None}}


#: published methods whose first five evaluations are the shared cold start, U[-1, 1]^D from RandomState(seed), as
#: GeoWalk's in the rerun (runner of 963ef04, run_published_lasso; runner of 14481cb, run_geowalk). BAxUS evaluates
#: its own Sobol design and CMA-ES starts at x0 = 0: neither shares it.
COLD_SHARED = ("hvarfner_published", "papenmeier_published_learned", "papenmeier_mle_scaled_published_learned",
               "xu_published", "linear_bo_published", "turbo_published")
#: relative tolerance: the two sets ran under different freeze files, so the last bits may differ; an objective that
#: changed (the reason for the rerun) differs far beyond this
COLD_RTOL = 1e-6


def objective_consistency(rerun: dict, pub: dict) -> dict:
    """Review S7: the published methods are normalised by, and tested against, GeoWalk and random search of the rerun
    (another commit and freeze file). Where both evaluate the same five cold-start points -- every (task, seed) of
    the methods in COLD_SHARED -- the losses must agree. Counts only (no value is returned or printed)."""
    ref = {(t, s): v for (m, t, s), v in rerun.get("cold", {}).items() if m == "geowalk"}
    out = {"compared": 0, "exact": 0, "within_tol": 0, "beyond_tol": 0, "without_reference": 0, "beyond_by_method": {},
           "methods": list(COLD_SHARED), "rtol": COLD_RTOL}
    for (m, t, s), v in sorted(pub.get("cold", {}).items()):
        if m not in COLD_SHARED:
            continue
        r = ref.get((t, s))
        if r is None:
            out["without_reference"] += 1
            continue
        a, b = np.asarray(v, dtype=float), np.asarray(r, dtype=float)
        out["compared"] += 1
        if np.array_equal(a, b):
            out["exact"] += 1
        elif np.allclose(a, b, rtol=COLD_RTOL, atol=0.0):
            out["within_tol"] += 1
        else:
            out["beyond_tol"] += 1
            out["beyond_by_method"][m] = out["beyond_by_method"].get(m, 0) + 1
    return out


def coverage_is_clean(summary: dict) -> bool:
    """True only when every expected run has a result with a finite loss and nothing unusual was loaded
    (no result outside the design, no declared unclean record, no unexpected job, no short run). Anything
    else must be announced loudly by the caller."""
    return bool(summary["complete"] and not summary["nonfinite"] and not summary["unrecorded"]
                and not summary["short"] and not summary["outside_design"] and not summary["declared_unclean"]
                and not summary["unexpected_job_labels"])


def coverage_lines(L: dict, only_incomplete: bool = True) -> list[str]:
    """One line per (method, task) cell block that is not complete: what is missing, and why."""
    out = []
    for (m, t), c in sorted(L["coverage"].items()):
        parts = []
        for key, lab in (("timeout", "timed out"), ("failed", "failed"), ("pending", "no job record yet"),
                         ("unrecorded", "result without a job record, not used")):
            if c[key]:
                parts.append(f"{lab}: seeds {compress(c[key])}"
                             + (f" (job running: {compress(c['running'])})" if key == "pending" and c["running"] else ""))
        if c["nonfinite"]:
            parts.append(f"non-finite loss: seeds {compress(c['nonfinite'])}")
        if c["short"]:
            parts.append(f"fewer than {N_TOTAL} scores: seeds {compress(c['short'])}")
        if not parts and only_incomplete:
            continue
        out.append(f"{m} {t}: {len(c['present'])}/{c['expected']} result files, {len(c['finite'])} finite"
                   + ("; " + "; ".join(parts) if parts else ""))
    return out


def missing_cell_list(L: dict) -> list[str]:
    """Every missing (method, task, seed) cell, compressed to seed ranges per (method, task)."""
    return [f"{m} {t}: seeds {compress(c['missing'])}" for (m, t), c in sorted(L["coverage"].items()) if c["missing"]]


def coverage_report(L: dict) -> str:
    """Markdown coverage report (written next to the tables)."""
    s = L["summary"]
    pub = L.get("design", "rerun") == "published"
    dtxt = pub_design_text(L=L) if pub else design_text()
    out = ["# LassoBench " + ("published methods" if pub else "camera-ready rerun") + ": coverage", "",
           f"Root `{L['dir']}`; expected commit `{L['commit'][:7]}` (tag `{L['tag']}`), runner sha256 "
           f"`{L['runner_sha256'][:16]}`. Every result file used is listed by a clean job record of its run folder's "
           f"`manifest.jsonl` with the same sha256.", "",
           f"Cells expected {s['expected']}, result files present {s['present']} ({s['finite']} finite, "
           f"{s['nonfinite']} non-finite), missing {s['missing']} (timed out {s['timeout']}, failed {s['failed']}, "
           f"no job record yet {s['pending']}, of which {s['running']} with a job running when the set was copied; "
           f"result without a record {s['unrecorded']})."
           + (f" Result files of methods that are not printed (record and sha256 checked, content not read): "
              f"{s.get('unread', 0)}; runs where the recipe is undefined (status not_applicable): "
              f"{s.get('not_applicable', 0)}." if pub else ""), "",
           f"Expected design: {dtxt}; {s['jobs_expected']} jobs. {'Run set' if pub else 'Rerun'} finished (every expected job has a "
           f"job record): " + ("yes." if s["finished"] else
                               f"NO -- INCOMPLETE RERUN, {s['jobs_without_record']} expected job(s) without a job "
                               f"record; not the final coverage."), "",
           "| run folder | job records / expected | jobs started | result files | rc counts | complete |",
           "|---|---|---|---|---|---|"]
    for k, v in L["sets"].items():
        out.append(f"| {k} | {v['n_records']} / {v['n_expected_jobs']} | {v['n_started']} | {v['n_result_files']} | "
                   f"{v['rc_counts']} | {'yes' if v['complete'] else 'NO'} |")
        if v["running_labels"]:
            out.append(f"| | started, no record yet: {', '.join(v['running_labels'])} | | | | |")
    out += ["", "| method | task | result files / expected | finite | non-finite | timed out | failed | pending | "
                "unrecorded | surrogate-fallback steps (files that record them) | errors |",
            "|---|---|---|---|---|---|---|---|---|---|---|"]
    for (m, t), c in sorted(L["coverage"].items()):
        fb = f"{c['fallback_steps']} ({c['fallback_recorded']})" if c["fallback_recorded"] else "n.r."
        out.append(f"| {m} | {t} | {len(c['present'])} / {c['expected']}{' (pilot)' if c['pilot'] else ''}"
                   f"{' (not read)' if c.get('unread') else ''} | "
                   f"{len(c['finite'])} | {compress(c['nonfinite'])} | {compress(c['timeout'])} | "
                   f"{compress(c['failed'])} | {compress(c['pending'])} | {compress(c['unrecorded'])} | {fb} | "
                   f"{'; '.join(f'{k} x{v}' for k, v in c['errors'].items())} |")
    out += ["", "## Leukemia (D = 7129)", ""]
    for m, v in L["leukemia"].items():
        out.append(f"- {m}: {v['status']} -- {v['text']}")
    if s["outside_design"]:
        out += ["", "Result files outside the expected design (loaded, not expected): " + ", ".join(s["outside_design"])]
    if s["declared_unclean"]:
        out += ["", "Result files with a declared unclean job record (used): " + "; ".join(s["declared_unclean"])]
    if s["unexpected_job_labels"]:
        out += ["", f"Job records outside the expected job list: {s['unexpected_job_labels']}"]
    return "\n".join(out) + "\n"
