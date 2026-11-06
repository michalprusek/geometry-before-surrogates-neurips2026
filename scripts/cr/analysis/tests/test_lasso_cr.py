"""LassoBench camera-ready rerun: provenance join to manifest.jsonl, coverage, the Table 3 convention of the
committed lassobench.py, and the tables / figure in both modes (rerun in camera-ready mode, stored in legacy).
Design of the two RUN_RULES amendments of 2026-10-12: Papenmeier at D = 7129 is a 40-seed arm (seed 42 from the
pilot's job, seeds 43-81 from the 39 later jobs); Xu and Linear BO stay one-seed pilots; an expected job without a
job record keeps the rerun unfinished ("INCOMPLETE RERUN"), a timed-out or non-finite run does not."""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest
from scipy import stats as sps

from crlib import lasso_cr, stored
from crlib.context import VARIANTS, Ctx
from crlib.latex import PREAMBLE_CHECK
from crlib.numbers import Registry
from v5_fixtures import cold_scores

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
DIMS = dict(stored.LASSO_TASKS)


# ---- builders ------------------------------------------------------------------------------------------
class Rerun:
    """A synthetic rerun root: <root>/<set>/results/*.json + <root>/<set>/manifest.jsonl."""

    def __init__(self, root):
        self.root = str(root)

    def job(self, sname, label, methods, task, seed, losses=None, *, rc=0, record=True, timeout_s="3600",
            error=None, status=True, **over):
        """losses: {method: value or None (no result file)}; default 1.0 for every method. A job always
        leaves cmds/<label>.cmd; with a record also status/<label>.rc (unless status=False)."""
        rdir = os.path.join(self.root, sname, "results")
        os.makedirs(rdir, exist_ok=True)
        for sub in ("cmds", "status"):
            os.makedirs(os.path.join(self.root, sname, sub), exist_ok=True)
        open(os.path.join(self.root, sname, "cmds", f"{label}.cmd"), "w").write("env ... run_lassobench.py\n")
        files = []
        for m in methods:
            v = 1.0 if losses is None else losses.get(m)
            path = os.path.join(rdir, f"{m}_{task}_s{seed}.json")
            rec = {"method": m, "file": "/kajman/" + os.path.relpath(path, self.root), "exists": v is not None}
            if v is not None:
                d = {"best_score": v, "best_trace": [], "all_scores": cold_scores(task, seed) + [float(i) for i in range(5, 30)], "method": m,
                     "task": task, "D": DIMS[task], "seed": seed, "n_total": 30, "elapsed_s": 1.0}
                if not (isinstance(v, float) and math.isfinite(v)):
                    d.update(best_score=float("inf"), all_scores=[], error=error or "All attempts to fit the model have failed.")
                raw = json.dumps(d).encode()
                with open(path, "wb") as f:
                    f.write(raw)
                rec.update(sha256=hashlib.sha256(raw).hexdigest(), best_score=d["best_score"])
            files.append(rec)
        if record:
            r = {"group": sname, "label": label, "methods": list(methods), "task": task, "seed": seed, "rc": rc,
                 "commit": lasso_cr.LASSO_CR_COMMIT, "tag": lasso_cr.LASSO_CR_TAG, "porcelain_lines_after": 0,
                 "runner_script_sha256": lasso_cr.RUNNER_SHA256, "runner_script_is_committed_file": True,
                 "timeout_s": timeout_s, "result_files": files}
            r.update(over)
            with open(os.path.join(self.root, sname, "manifest.jsonl"), "a") as f:
                f.write(json.dumps(r) + "\n")
            if status:
                open(os.path.join(self.root, sname, "status", f"{label}.rc"), "w").write(f"{rc} t0 t1\n")

    def leukemia_complete(self, losses_of=None):
        """All 40 jobs of leukemia_rerun (a COMPLETE run folder)."""
        for s in lasso_cr.SEEDS:
            ls = losses_of(s) if losses_of else None
            self.job("leukemia_rerun", f"s{s}", lasso_cr.CHEAP_LEUKEMIA, "leukemia", s, ls)

    def cheap(self, task, seed, losses=None, **kw):
        self.job("full_rerun", f"cheap_{task}_s{seed}", ["random", "cmaes_sub", "geowalk", "cmaes_full"], task, seed,
                 losses, **kw)

    def papenmeier_leukemia(self, seeds, loss_of=lambda s: 1.0 + 0.01 * s, timeout=(), nonfinite=()):
        """Papenmeier at D = 7129 as the queue ran it: seed 42 is the PILOT's job, every other seed a job
        ``papenmeier_leukemia_s<seed>`` (labels typed here, not taken from lasso_cr.expected_jobs)."""
        for s in seeds:
            label = "pilot_papenmeier_leukemia_s42" if s == 42 else f"papenmeier_leukemia_s{s}"
            if s in timeout:
                self.job("full_rerun", label, ["papenmeier_msr"], "leukemia", s, {"papenmeier_msr": None}, rc=124)
            else:
                self.job("full_rerun", label, ["papenmeier_msr"], "leukemia", s,
                         {"papenmeier_msr": float("inf") if s in nonfinite else loss_of(s)})


# the queue's job groups at D <= 1000 (code/full_rerun_jobs.py), typed out independently of lasso_cr.FULL_GROUPS
QUEUE_GROUPS = [("cheap", ["random", "cmaes_sub", "geowalk", "cmaes_full"]), ("turbo", ["turbo"]), ("baxus", ["baxus"]),
                ("hvarfner", ["hvarfner_gp"]), ("papenmeier", ["papenmeier_msr"]), ("xu", ["xu_standard_gp"]),
                ("linbo", ["linear_bo"])]


def full_caption(t) -> str:
    """The caption of the full LassoBench table: since RUN_RULES 2026-10-29 the coverage and time-out sentences are in the
    repository record (records/tab_lassobench_full.tex), and the printed caption is short."""
    return t.record_tables[0].caption if getattr(t, "record_tables", None) else t.caption


def write_original_design(root, nonfinite_seed_81=()):
    """Every job of the rerun as queued BEFORE the Papenmeier amendment, all ended: the 1,442 jobs of
    queue/jobs_full_rerun.txt (1,400 at D <= 1000, 40 Hvarfner leukemia, the Papenmeier pilot (completed) and the
    Xu pilot (timed out)), the 40 jobs of leukemia_rerun and the Linear BO pilot (timed out). Methods named in
    ``nonfinite_seed_81`` end non-finite at seed 81 of every task with D <= 1000."""
    rng = np.random.default_rng(17)
    R = Rerun(root)
    R.leukemia_complete(lambda s: {m: float(rng.uniform(0.5, 2.0)) for m in lasso_cr.CHEAP_LEUKEMIA})
    for s in range(42, 82):
        for t in ("synt_simple", "synt_medium", "dna", "synt_high", "synt_hard"):
            for name, ms in QUEUE_GROUPS:
                R.job("full_rerun", f"{name}_{t}_s{s}", ms, t, s,
                      {m: float("inf") if (s == 81 and m in nonfinite_seed_81) else float(rng.uniform(0.5, 2.0))
                       for m in ms})
        R.job("full_rerun", f"hvarfner_leukemia_s{s}", ["hvarfner_gp"], "leukemia", s,
              {"hvarfner_gp": float(rng.uniform(0.5, 2.0))})
    R.papenmeier_leukemia([42], loss_of=lambda s: 1.7)                      # the pilot: completed in time
    R.job("full_rerun", "pilot_xu_leukemia_s42", ["xu_standard_gp"], "leukemia", 42, {"xu_standard_gp": None}, rc=124)
    R.job("linear_bo_leukemia", "s42", ["linear_bo"], "leukemia", 42, {"linear_bo": None}, rc=124, timeout_s="7200")
    return R


@pytest.fixture(scope="module")
def original_design(tmp_path_factory):
    """One copy of the original design per module (read-only: tests copy it before adding jobs)."""
    return write_original_design(tmp_path_factory.mktemp("original_design") / "lassobench_cr").root


def copy_of(root, dst):
    shutil.copytree(root, str(dst))
    return Rerun(dst)


def write_stored(d, cells):
    """A stored-format directory: <method>_<task>_s<seed>.json with best_score, D, method."""
    os.makedirs(d, exist_ok=True)
    for (m, t, s), v in cells.items():
        with open(os.path.join(d, f"{m}_{t}_s{s}.json"), "w") as f:
            json.dump({"best_score": v, "method": m, "task": t, "D": DIMS[t], "seed": s,
                       "all_scores": [float(i) for i in range(30)]}, f)


def synthetic_cells(rng, methods=("random", "cmaes_sub", "geowalk", "linear_bo"), seeds=range(42, 50)):
    cells = {}
    for t, _ in stored.LASSO_TASKS:
        for s in seeds:
            for i, m in enumerate(methods):
                cells[(m, t, s)] = float(rng.uniform(0.5, 2.0) * (1.0 + 0.3 * i))
    return cells


LAB = stored.lasso_method_tex          # printed method labels (print rule, crlib/labels.py)
PLAB = lambda meth: LAB(meth).replace(" (rerun)", "")   # the printed view, RUN_RULES 2026-11-02 (2)


def ctx_for(tmp_path, mode, stored_dir=None, rerun_dir=None):
    out = tmp_path / f"out_{mode}"
    out.mkdir(parents=True, exist_ok=True)
    extra = {"inputs_root": str(tmp_path), "lasso_dir": stored_dir, "_shared_cache": {}}
    if rerun_dir is not None:
        extra["lasso_cr_dir"] = rerun_dir
    return Ctx(ds=None, eng=None, reg=Registry("x", "0" * 64, "default"), var=VARIANTS["default"], outdir=str(out),
               script_commit="x", mode=mode, extra=extra)


# ---- the expected design ---------------------------------------------------------------------------------
def test_expected_design_and_job_labels():
    cells = lasso_cr.expected_cells()
    assert sum(len(v["seeds"]) for v in cells.values()) == 10 * 5 * 40 + 8 * 40 + 2 == 2322
    assert cells[("linear_bo", "leukemia")] == {"seeds": [42], "set": "linear_bo_leukemia", "pilot": True}
    assert cells[("xu_standard_gp", "leukemia")] == {"seeds": [42], "set": "full_rerun", "pilot": True}
    assert cells[("hvarfner_gp", "leukemia")]["set"] == "full_rerun" and not cells[("hvarfner_gp", "leukemia")]["pilot"]
    assert cells[("baxus", "leukemia")]["set"] == "leukemia_rerun"
    # RUN_RULES amendment 2026-10-12: Papenmeier at D = 7129 is a 40-seed arm of run folder full_rerun, like the
    # Hvarfner GP; only Xu and Linear BO stay one-seed pilots
    assert cells[("papenmeier_msr", "leukemia")] == {"seeds": list(range(42, 82)), "set": "full_rerun", "pilot": False}
    assert cells[("papenmeier_msr", "leukemia")] == cells[("hvarfner_gp", "leukemia")]
    assert sorted(m for (m, t), v in cells.items() if v["pilot"]) == ["linear_bo", "xu_standard_gp"]
    assert sorted(lasso_cr.PILOT_LEUKEMIA) == ["linear_bo", "xu_standard_gp"]
    assert sum(1 for (m, t), v in cells.items() if t == "leukemia" and not v["pilot"]) == 8
    assert lasso_cr.design_text() == ("10 methods x 5 tasks x 40 seeds; 8 arms x 40 seeds and 2 pilot seeds "
                                      "at D = 7129")
    jobs = lasso_cr.expected_jobs()
    assert {k: len(v) for k, v in jobs.items()} == {"full_rerun": 1442 + 39, "leukemia_rerun": 40,
                                                    "linear_bo_leukemia": 1}
    assert jobs["full_rerun"]["cheap_dna_s57"] == (["random", "cmaes_sub", "geowalk", "cmaes_full"], "dna", 57)
    assert jobs["full_rerun"]["pilot_xu_leukemia_s42"] == (["xu_standard_gp"], "leukemia", 42)
    # every expected cell is produced by exactly one expected job
    made = [(m, t, s) for js in jobs.values() for ms, t, s in js.values() for m in ms]
    assert len(made) == len(set(made)) == 2322
    assert lasso_cr.compress([44, 42, 43, 47, 50, 51]) == "42-44, 47, 50-51" and lasso_cr.compress([]) == ""


# the first and the last line of kajman:/data2/prusek/cr_analysis/queue/jobs_papenmeier_leukemia.txt (39 lines)
JOB_LINES = """\
bash /data2/prusek/cr_analysis/code/run_lasso.sh full_rerun papenmeier_leukemia_s43 results /data2/prusek/NeurIPS-cr-ana/scripts/run_lassobench.py papenmeier_msr leukemia 43 3600
bash /data2/prusek/cr_analysis/code/run_lasso.sh full_rerun papenmeier_leukemia_s81 results /data2/prusek/NeurIPS-cr-ana/scripts/run_lassobench.py papenmeier_msr leukemia 81 3600
"""


def test_papenmeier_leukemia_jobs_seed_42_from_the_pilot_label_and_43_to_81_from_the_new_labels(tmp_path):
    jobs = lasso_cr.expected_jobs()
    pap = {lab: v for lab, v in jobs["full_rerun"].items() if v[0] == ["papenmeier_msr"] and v[1] == "leukemia"}
    assert sorted(v[2] for v in pap.values()) == list(range(42, 82))              # one job per seed, 40 seeds
    assert pap["pilot_papenmeier_leukemia_s42"] == (["papenmeier_msr"], "leukemia", 42)
    assert "papenmeier_leukemia_s42" not in jobs["full_rerun"]                    # the pilot's file IS seed 42
    assert {lab for lab in pap if not lab.startswith("pilot_")} == {f"papenmeier_leukemia_s{s}" for s in range(43, 82)}
    assert not any("papenmeier" in lab and "leukemia" in lab for k in ("leukemia_rerun", "linear_bo_leukemia")
                   for lab in jobs[k])
    for line in JOB_LINES.splitlines():       # the queue's job lines: run_lasso.sh <run folder> <label> <results
        a = line.split()                      # subfolder> <runner> <method> <task> <seed> <time limit in s>
        assert len(a) == 10 and a[2] == "full_rerun" and a[4] == "results" and a[9] == "3600"
        assert jobs[a[2]][a[3]] == ([a[6]], a[7], int(a[8]))
    # loaded: seed 42 is the value the pilot's job wrote, seeds 43-81 those of the later jobs; nothing unexpected
    R = Rerun(tmp_path / "a")
    R.leukemia_complete()
    R.papenmeier_leukemia(range(42, 82), loss_of=lambda s: 1.7 if s == 42 else 2.0 + 0.01 * s)
    L = lasso_cr.load_rerun(R.root)
    got = L["cells"]["papenmeier_msr"]["leukemia"]
    assert sorted(got) == list(range(42, 82)) and got[42][0] == 1.7 and got[43][0] == pytest.approx(2.43)
    assert got[42][1].endswith(os.path.join("full_rerun", "results", "papenmeier_msr_leukemia_s42.json"))
    c = L["coverage"][("papenmeier_msr", "leukemia")]
    assert c["expected"] == 40 and not c["pilot"] and c["set"] == "full_rerun" and c["present"] == list(range(42, 82))
    assert c["missing"] == [] and L["summary"]["outside_design"] == [] and not L["summary"]["unexpected_job_labels"]
    assert L["leukemia"]["papenmeier_msr"] == {"status": "run", "text": "40 of 40 seeds with a finite loss",
                                               "pilot": False, "finite": 40, "expected": 40}
    # a pilot job that has not ended yet: seed 42 is pending and reported as running under the PILOT's label
    R = Rerun(tmp_path / "b")
    R.leukemia_complete()
    os.makedirs(os.path.join(R.root, "full_rerun", "cmds"))
    open(os.path.join(R.root, "full_rerun", "cmds", "pilot_papenmeier_leukemia_s42.cmd"), "w").write("env ...\n")
    open(os.path.join(R.root, "full_rerun", "cmds", "papenmeier_leukemia_s43.cmd"), "w").write("env ...\n")
    c = lasso_cr.load_rerun(R.root)["coverage"][("papenmeier_msr", "leukemia")]
    assert c["running"] == [42, 43] and c["pending"] == list(range(42, 82))
    # a second job for seed 42 under the new label scheme is not part of the design: with the pilot it is fatal
    R = Rerun(tmp_path / "c")
    R.leukemia_complete()
    R.papenmeier_leukemia([42, 43])
    R.job("full_rerun", "papenmeier_leukemia_s42", ["papenmeier_msr"], "leukemia", 42)
    with pytest.raises(lasso_cr.LassoLoadError) as e:
        lasso_cr.load_rerun(R.root)
    assert e.value.kinds() == {"manifest"} and "papenmeier_msr_leukemia_s42.json is listed by two job records" in str(e.value)
    # ... and alone (instead of the pilot) it is used but announced: an unexpected label, the pilot's job missing
    R = Rerun(tmp_path / "d")
    R.leukemia_complete()
    R.job("full_rerun", "papenmeier_leukemia_s42", ["papenmeier_msr"], "leukemia", 42)
    L = lasso_cr.load_rerun(R.root)
    assert L["summary"]["unexpected_job_labels"] == {"full_rerun": ["papenmeier_leukemia_s42"]}
    assert "pilot_papenmeier_leukemia_s42" in L["sets"]["full_rerun"]["labels_missing_list"]
    assert not lasso_cr.coverage_is_clean(L["summary"]) and not L["summary"]["finished"]


def test_papenmeier_leukemia_timed_out_and_non_finite_seeds_are_excluded_and_counted_like_any_other_arm(tmp_path):
    """The amendment: 'treated like every other arm of the rerun'. Papenmeier and the Hvarfner GP get the same
    outcomes at D = 7129 (seed 50 timed out, seed 60 non-finite, seeds 70-81 not run yet)."""
    rng = np.random.default_rng(23)
    R = Rerun(tmp_path / "lassobench_cr")
    R.leukemia_complete(lambda s: {m: float("inf") if (m == "cmaes_sub" and s == 44) else float(rng.uniform(0.5, 2.0))
                                   for m in lasso_cr.CHEAP_LEUKEMIA})
    val = {s: float(rng.uniform(0.5, 2.0)) for s in range(42, 70)}
    R.papenmeier_leukemia(range(42, 70), loss_of=val.get, timeout=[50], nonfinite=[60])
    for s in range(42, 70):
        v = None if s == 50 else float("inf") if s == 60 else val[s]
        R.job("full_rerun", f"hvarfner_leukemia_s{s}", ["hvarfner_gp"], "leukemia", s, {"hvarfner_gp": v},
              rc=124 if s == 50 else 0)
    L = lasso_cr.load_rerun(R.root)
    cp, ch = L["coverage"][("papenmeier_msr", "leukemia")], L["coverage"][("hvarfner_gp", "leukemia")]
    assert cp == ch                                                # identical treatment, field by field
    finite = [s for s in range(42, 70) if s not in (50, 60)]
    assert cp["expected"] == 40 and not cp["pilot"]
    assert cp["finite"] == finite and len(finite) == 26 and cp["present"] == sorted(finite + [60])
    assert cp["timeout"] == [50] and cp["nonfinite"] == [60] and cp["pending"] == list(range(70, 82))
    assert cp["missing"] == [50] + list(range(70, 82)) and cp["failed"] == []
    assert 50 not in L["cells"]["papenmeier_msr"]["leukemia"] and 60 not in L["cells"]["papenmeier_msr"]["leukemia"]
    assert L["nonfinite"][("papenmeier_msr", "leukemia")] == 1
    assert L["leukemia"]["papenmeier_msr"] == L["leukemia"]["hvarfner_gp"] == {
        "status": "run", "text": "26 of 40 seeds with a finite loss", "pilot": False, "finite": 26, "expected": 40}
    assert any(x.startswith("papenmeier_msr leukemia: 27/40 result files, 26 finite; timed out: seeds 50; "
                            "no job record yet: seeds 70-81; non-finite loss: seeds 60") for x in lasso_cr.coverage_lines(L))
    assert "papenmeier_msr leukemia: seeds 50, 70-81" in lasso_cr.missing_cell_list(L)
    md = lasso_cr.coverage_report(L)
    assert "| papenmeier_msr | leukemia | 27 / 40 | 26 | 60 | 50 |  | 70-81 |" in md
    assert "| hvarfner_gp | leukemia | 27 / 40 | 26 | 60 | 50 |  | 70-81 |" in md and "27 / 40 (pilot)" not in md
    assert "- papenmeier_msr: run -- 26 of 40 seeds with a finite loss" in md
    # the statistics: the mean over the completed seeds with their number; pairs = seeds completed by both arms
    S = stored.lasso_stats(L)
    norm = S["norm"]["leukemia"]
    x = S["per_d"][("papenmeier_msr", "leukemia")]
    assert x["n"] == 26 and x["mean"] == pytest.approx(np.mean([val[s] for s in finite]) / norm)
    assert S["per_d"][("hvarfner_gp", "leukemia")]["mean"] == pytest.approx(x["mean"])
    v = S["vs_geocma"]["papenmeier_msr"]["per_d"]["leukemia"]
    assert v["n"] == 25                                            # GeoCMA's seed 44 is non-finite: not a pair
    gc = L["cells"]["cmaes_sub"]["leukemia"]
    assert v["delta"] == pytest.approx(np.mean([(val[s] - gc[s][0]) / norm for s in finite if s != 44]))
    # the table: the number of completed seeds, never a pilot label
    ctx = ctx_for(tmp_path, "cr", None, R.root)
    t = stored.tab_lassobench(ctx)
    rows = {r[0]: r for r in t.rows if isinstance(r, list)}
    assert rows[PLAB("papenmeier_msr")][-1].endswith("\\,(26/40)") and rows[PLAB("hvarfner_gp")][-1].endswith("\\,(26/40)")
    assert "pilot" not in rows[PLAB("papenmeier_msr")][-1] and "ran one pilot seed only" not in full_caption(t)
    rec = ctx.reg.records
    assert rec["crLassoPapenmeierMsrDSevenOneTwoNineN"]["value"] == 26
    runs = rec["crLassoPapenmeierMsrDSevenOneTwoNineRuns"]
    assert runs["value"] == 27 and runs["n"] == 40 and runs["rule"].endswith("of 40 expected seed(s)")
    assert rec["crLassoPapenmeierMsrDSevenOneTwoNine"]["value"] == pytest.approx(x["mean"])
    assert "INCOMPLETE RERUN: not the final table." in t.caption              # seeds 70-81 have no job record


def test_rerun_is_incomplete_until_the_39_papenmeier_leukemia_jobs_have_ended(original_design, tmp_path, capsys):
    """Everything queued before the amendment has ended; the 39 Papenmeier leukemia jobs have not."""
    sd = str(tmp_path / "lassobench")
    write_stored(sd, synthetic_cells(np.random.default_rng(5), seeds=range(42, 48)))
    R = copy_of(original_design, tmp_path / "lassobench_cr")
    L = lasso_cr.load_rerun(R.root)
    s = L["summary"]
    assert s["expected"] == 2322 and s["present"] == 2000 + 240 + 40 + 1 == 2281 and s["nonfinite"] == 0
    assert s["pending"] == 39 and s["timeout"] == 2 and s["missing"] == 41 and s["unrecorded"] == 0
    assert s["jobs_expected"] == 1481 + 40 + 1 and s["jobs_without_record"] == 39
    assert not s["finished"] and not s["complete"]
    assert s["run_folders_complete"] == {"full_rerun": False, "leukemia_rerun": True, "linear_bo_leukemia": True}
    fr = L["sets"]["full_rerun"]
    assert fr["n_expected_jobs"] == 1481 and fr["n_records"] == 1442 and fr["labels_unexpected"] == []
    assert fr["labels_missing_list"] == [f"papenmeier_leukemia_s{x}" for x in range(43, 82)]
    c = L["coverage"][("papenmeier_msr", "leukemia")]
    assert c["present"] == [42] and c["pending"] == list(range(43, 82)) and c["expected"] == 40
    assert [k for k, v in L["coverage"].items() if v["pending"]] == [("papenmeier_msr", "leukemia")]
    assert L["leukemia"]["papenmeier_msr"]["text"] == "1 of 40 seeds with a finite loss"
    md = lasso_cr.coverage_report(L)
    assert "| full_rerun | 1442 / 1481 |" in md and "| papenmeier_msr | leukemia | 1 / 40 | 1 |" in md
    assert ("Expected design: 10 methods x 5 tasks x 40 seeds; 8 arms x 40 seeds and 2 pilot seeds at D = 7129; "
            "1522 jobs. Rerun finished (every expected job has a job record): NO -- INCOMPLETE RERUN, 39 expected "
            "job(s) without a job record") in md
    # the final run refuses it and names the jobs and the cells
    with pytest.raises(lasso_cr.LassoLoadError) as e:
        lasso_cr.load_rerun(R.root, require_finished=True)
    assert e.value.kinds() == {"unfinished"} and len(e.value.problems) == 2
    assert "[unfinished] full_rerun: 39 of 1481 expected jobs have no job record: papenmeier_leukemia_s43" in str(e.value)
    assert "[unfinished] papenmeier_msr leukemia: no job record yet: seeds 43-81" in str(e.value)
    # the tables: three incomplete cells only (fewer than the 12 that used to trigger the mark), still INCOMPLETE
    ctx = ctx_for(tmp_path, "cr", sd, R.root)
    t = stored.tab_lassobench(ctx)
    rec = ctx.reg.records
    rows = {r[0]: r for r in t.rows if isinstance(r, list)}
    assert rows[PLAB("papenmeier_msr")][-1].endswith("\\,(1/40)") and "pilot" not in rows[PLAB("papenmeier_msr")][-1]
    assert rows[PLAB("xu_standard_gp")][-1] == "n.c." and rows[PLAB("linear_bo")][-1] == "n.c."
    # caption sentences of one point each (RUN_RULES 2026-10-22, wording): one sentence per method
    assert (f"39 were not run yet. {LAB('linear_bo')} has fewer finite losses than expected at $D=7129$ (0/1). "
            f"{LAB('papenmeier_msr')} has fewer finite losses than expected at $D=7129$ (1/40). "
            f"{LAB('xu_standard_gp')} has fewer finite losses than expected at $D=7129$ (0/1). "
            "\\textbf{INCOMPLETE RERUN: not the final table.}") in full_caption(t)
    assert rec["crLassoCellsExpected"]["value"] == 2322 and rec["crLassoCellsMissing"]["value"] == 41
    assert rec["crLassoCellsExpected"]["rule"] == ("runs of the expected design (10 methods x 5 tasks x 40 seeds; "
                                                   "8 arms x 40 seeds and 2 pilot seeds at D = 7129)")
    assert rec["crLassoPapenmeierMsrDSevenOneTwoNineRuns"]["value"] == 1
    assert rec["crLassoPapenmeierMsrDSevenOneTwoNineRuns"]["n"] == 40
    assert rec["crLassoLeukemiaNotCompleted"]["value"] == "xu_standard_gp, linear_bo"
    assert rec["crLassoLeukemiaNotCompletedN"]["value"] == 2 and rec["crLassoLeukemiaPendingN"]["value"] == 0
    out = capsys.readouterr().out
    assert "THE RERUN IS NOT FINISHED (39 of 1522 expected jobs without a job record)" in out
    assert "papenmeier_msr leukemia: 1/40 result files, 1 finite; no job record yet: seeds 43-81" in out
    assert "NO -- INCOMPLETE RERUN" in open(os.path.join(ctx.outdir, "lassobench_cr_coverage.md")).read()
    assert "\\textbf{The rerun is incomplete} (2{,}281 of 2{,}322 expected runs)" in stored.tab_lassobench_correspondence(ctx).caption

    # ---- the 39 jobs end: one timed out, one non-finite. The rerun is FINISHED; those two seeds are final outcomes
    R.papenmeier_leukemia(range(43, 82), timeout=[50], nonfinite=[60])
    L = lasso_cr.load_rerun(R.root, require_finished=True)               # accepted by the final run
    s = L["summary"]
    assert s["finished"] and s["pending"] == 0 and s["jobs_without_record"] == 0 and s["timeout"] == 3
    assert s["present"] == 2281 + 38 and s["nonfinite"] == 1 and not s["complete"]       # 'complete' = no gap at all
    assert all(s["run_folders_complete"].values())
    c = L["coverage"][("papenmeier_msr", "leukemia")]
    assert len(c["finite"]) == 38 and c["timeout"] == [50] and c["nonfinite"] == [60] and c["pending"] == []
    assert "Rerun finished (every expected job has a job record): yes." in lasso_cr.coverage_report(L)
    ctx = ctx_for(tmp_path / "fin", "cr", sd, R.root)
    t = stored.tab_lassobench(ctx)
    rows = {r[0]: r for r in t.rows if isinstance(r, list)}
    assert rows[PLAB("papenmeier_msr")][-1].endswith("\\,(38/40)")
    assert "INCOMPLETE" not in t.caption and "0 were not run yet" in full_caption(t) and "3 timed out" in full_caption(t)
    # RUN_RULES 2026-10-29 (4): the printed caption is short; coverage and time-outs are in the full table (a record)
    assert "Coverage:" not in t.caption and "tab\\_lassobench\\_full.tex" in t.caption and "n.c.:" in t.caption
    assert (f"{LAB('linear_bo')} has fewer finite losses than expected at $D=7129$ (0/1). "
            f"{LAB('papenmeier_msr')} has fewer finite losses than expected at $D=7129$ (38/40). "
            f"{LAB('xu_standard_gp')} has fewer finite losses than expected at $D=7129$ (0/1).") in full_caption(t)
    # one sentence per stated cause (RUN_RULES 2026-10-22, wording)
    assert (f"At $D=7129$ (leukemia), {LAB('xu_standard_gp')} did not complete (n.c.; pilot exceeded the time limit "
            f"of 60 min). At $D=7129$ (leukemia), {LAB('linear_bo')} did not complete (n.c.; pilot exceeded the time "
            "limit of 120 min).") in full_caption(t)
    assert ctx.reg.records["crLassoPapenmeierMsrDSevenOneTwoNineN"]["value"] == 38
    assert "THE RERUN IS NOT FINISHED" not in capsys.readouterr().out
    assert "incomplete" not in stored.tab_lassobench_correspondence(ctx).caption


def test_finished_rerun_with_many_excluded_runs_is_not_marked_incomplete(tmp_path):
    """More than 12 method-dimension cells with fewer finite losses than expected, every job ended: the caption
    points to the coverage file and does NOT call the table provisional (timed-out and non-finite runs are final)."""
    R = write_original_design(tmp_path / "lassobench_cr", nonfinite_seed_81=("xu_standard_gp", "baxus", "papenmeier_msr"))
    R.papenmeier_leukemia(range(43, 82), timeout=[50])
    L = lasso_cr.load_rerun(R.root, require_finished=True)
    assert L["summary"]["finished"] and L["summary"]["nonfinite"] == 15 and L["summary"]["timeout"] == 3
    ctx = ctx_for(tmp_path, "cr", None, R.root)
    t = stored.tab_lassobench(ctx)
    # GEN v5: the rerun's TuRBO, full-D CMA-ES and BAxUS rows are printed only in the correspondence table
    # (RUN_RULES 2026-10-12 (5)), so the cells with fewer finite losses are counted over the printed rows
    assert (" 13 method--dimension cells have fewer finite losses than expected (listed in "
            "\\texttt{lassobench\\_cr\\_coverage.md}).") in full_caption(t)
    assert "INCOMPLETE" not in full_caption(t) and "are incomplete" not in full_caption(t)
    rows = {r[0]: r for r in t.rows if isinstance(r, list)}
    assert PLAB("baxus") not in rows and rows[PLAB("papenmeier_msr")][-1].endswith("\\,(39/40)")


def test_make_tables_refuses_an_unfinished_rerun_only_under_lasso_require_finished(original_design, tmp_path,
                                                                                     monkeypatch, capsys):
    sys.path.insert(0, ANALYSIS)
    import make_tables

    def stop(*a, **k):                    # nothing beyond the LassoBench check may run in this test
        raise make_tables.ConfigError("stopped by the test after the LassoBench check")
    monkeypatch.setattr(make_tables, "load_config", stop)

    def run(name, root, *flags):
        out = tmp_path / name
        rc = make_tables.main(["--expected-commit", "CR=" + "a" * 40, "--outdir", str(out), "--inputs-root",
                               str(tmp_path / "no_inputs"), "--lasso-cr-root", str(root), "--validation-run", *flags])
        p = out / "LOADER_FAILED.txt"
        return rc, (p.read_text() if p.exists() else None), capsys.readouterr().err

    # unfinished (the 39 Papenmeier leukemia jobs are pending): refused before anything else is loaded
    rc, failed, err = run("unfinished", original_design, "--lasso-require-finished")
    assert rc == 2 and "CONFIG ERROR" not in err
    assert failed.startswith("LASSOBENCH RERUN NOT USABLE FOR A FINAL RUN (--lasso-require-finished")
    assert "papenmeier_leukemia_s43" in failed and "papenmeier_msr leukemia: no job record yet: seeds 43-81" in failed
    # the same set without the flag passes the check (and is stopped later, by this test's config stub)
    rc, failed, err = run("tolerant", original_design)
    assert rc == 2 and failed is None and "CONFIG ERROR" in err
    # an absent rerun folder is refused as well
    rc, failed, err = run("absent", tmp_path / "nowhere", "--lasso-require-finished")
    assert rc == 2 and "run folder absent" in failed and "1481 of 1481 expected jobs have no job record" in failed
    # finished: accepted
    R = copy_of(original_design, tmp_path / "finished")
    R.papenmeier_leukemia(range(43, 82), timeout=[50])
    rc, failed, err = run("finished_out", R.root, "--lasso-require-finished")
    # GEN v5: the finished rerun is accepted; the next check (the published-method set, absent here) refuses
    assert rc == 2 and failed.startswith("LASSOBENCH PUBLISHED-METHOD SET NOT USABLE") and "CONFIG ERROR" not in err
    # --legacy never reads the rerun, so the flag has nothing to check there
    out = tmp_path / "legacy"
    assert make_tables.main(["--legacy", "--outdir", str(out), "--lasso-cr-root", str(original_design),
                             "--lasso-require-finished"]) == 2
    assert not (out / "LOADER_FAILED.txt").exists() and "CONFIG ERROR" in capsys.readouterr().err


# ---- loader: clean join, coverage, tolerated gaps -----------------------------------------------------------
def test_load_joins_results_to_clean_records_and_reports_coverage(tmp_path):
    R = Rerun(tmp_path)
    R.leukemia_complete(lambda s: {"random": 2.0, "cmaes_sub": 1.0, "geowalk": 1.5, "cmaes_full": 1.6, "turbo": 2.2,
                                   "baxus": float("inf") if s in (49, 51) else 2.4})
    R.cheap("dna", 42)
    R.cheap("dna", 43)
    R.job("full_rerun", "xu_dna_s42", ["xu_standard_gp"], "dna", 42, {"xu_standard_gp": float("inf")})
    R.job("full_rerun", "papenmeier_dna_s42", ["papenmeier_msr"], "dna", 42, {"papenmeier_msr": None}, rc=124)
    R.job("linear_bo_leukemia", "s42", ["linear_bo"], "leukemia", 42, {"linear_bo": None}, rc=124, timeout_s="7200")
    L = lasso_cr.load_rerun(str(tmp_path))
    assert set(L["cells"]["geowalk"]) == {"leukemia", "dna"} and len(L["cells"]["geowalk"]["leukemia"]) == 40
    assert L["cells"]["random"]["dna"][42][0] == 1.0
    assert L["sets"]["full_rerun"]["n_started"] == 4 and L["sets"]["full_rerun"]["running_labels"] == []
    assert L["nonfinite"] == {("baxus", "leukemia"): 2, ("xu_standard_gp", "dna"): 1}
    assert 49 not in L["cells"]["baxus"]["leukemia"]                      # a non-finite loss is never a value
    assert len(L["manifests"]) == 3
    s = L["summary"]
    assert s["expected"] == 2322 and s["present"] == 240 + 8 + 1 and s["nonfinite"] == 3
    assert s["timeout"] == 2 and s["failed"] == 0 and s["unrecorded"] == 0 and not s["complete"]
    assert s["pending"] == 2322 - 249 - 2 and not s["finished"]
    assert s["run_folders_complete"] == {"full_rerun": False, "leukemia_rerun": True, "linear_bo_leukemia": True}
    c = L["coverage"][("random", "dna")]
    assert c["present"] == [42, 43] and c["missing"] == list(range(44, 82)) and c["pending"] == list(range(44, 82))
    assert L["coverage"][("papenmeier_msr", "dna")]["timeout"] == [42]
    assert L["coverage"][("baxus", "leukemia")]["nonfinite"] == [49, 51]
    assert L["coverage"][("xu_standard_gp", "dna")]["errors"] == {"All attempts to fit the model have failed.": 1}
    lk = L["leukemia"]
    assert lk["geowalk"]["status"] == "run" and lk["baxus"]["finite"] == 38
    assert lk["linear_bo"]["status"] == "not_completed" and "120 min" in lk["linear_bo"]["text"]
    assert lk["hvarfner_gp"]["status"] == "pending" and lk["xu_standard_gp"]["status"] == "pending"
    assert lk["papenmeier_msr"] == {"status": "pending", "text": "not run yet (no job record)", "pilot": False,
                                    "finite": 0, "expected": 40}
    assert lk["xu_standard_gp"]["pilot"] and lk["linear_bo"]["pilot"] and lk["xu_standard_gp"]["expected"] == 1
    lines = lasso_cr.coverage_lines(L)
    assert any(x.startswith("random dna: 2/40") and "seeds 44-81" in x for x in lines)
    assert "random dna: seeds 44-81" in lasso_cr.missing_cell_list(L)
    assert "| leukemia_rerun | 40 / 40 |" in lasso_cr.coverage_report(L)


def test_result_without_record_is_excluded_while_running_and_fatal_when_complete(tmp_path):
    R = Rerun(tmp_path / "a")
    R.cheap("dna", 42)
    R.cheap("dna", 43, record=False)                 # job still running when the set was copied
    L = lasso_cr.load_rerun(R.root)
    assert 43 not in L["cells"]["random"]["dna"] and L["coverage"][("random", "dna")]["unrecorded"] == [43]
    assert L["summary"]["unrecorded"] == 4
    # a job that has started (cmds file) and has no record: its cells are pending, marked as running
    open(os.path.join(R.root, "full_rerun", "cmds", "turbo_dna_s42.cmd"), "w").write("env ...\n")
    L = lasso_cr.load_rerun(R.root)
    c = L["coverage"][("turbo", "dna")]
    assert c["running"] == [42] and c["pending"] == list(range(42, 82)) and L["summary"]["running"] == 1
    assert "turbo_dna_s42" in L["sets"]["full_rerun"]["running_labels"]
    assert any("job running: 42" in x for x in lasso_cr.coverage_lines(L))
    R2 = Rerun(tmp_path / "b")
    R2.leukemia_complete()
    R2.job("leukemia_rerun", "extra", ["hvarfner_gp"], "leukemia", 42, record=False)   # complete folder, stray file
    with pytest.raises(lasso_cr.LassoLoadError) as e:
        lasso_cr.load_rerun(R2.root)
    assert e.value.kinds() == {"provenance"} and "no job record lists this result file" in str(e.value)


@pytest.mark.parametrize("over, word", [({"rc": 1}, "rc 1"), ({"commit": "abbd61e"}, "commit"),
                                        ({"tag": "other"}, "tag"), ({"porcelain_lines_after": 2}, "porcelain"),
                                        ({"runner_script_sha256": "0" * 64}, "runner_script_sha256"),
                                        ({"runner_script_is_committed_file": False}, "committed file")])
def test_unclean_job_record_is_fatal_unless_declared(tmp_path, over, word):
    R = Rerun(tmp_path)
    R.cheap("dna", 42)
    R.cheap("dna", 43, **over)
    with pytest.raises(lasso_cr.LassoLoadError) as e:
        lasso_cr.load_rerun(R.root)
    assert e.value.kinds() == {"provenance"} and word in str(e.value) and len(e.value.problems) == 4
    names = [f"{m}_dna_s43.json" for m in ("random", "cmaes_sub", "geowalk", "cmaes_full")]
    L = lasso_cr.load_rerun(R.root, allow_unclean=names)
    assert 43 in L["cells"]["geowalk"]["dna"] and len(L["summary"]["declared_unclean"]) == 4


def test_job_that_failed_without_a_result_is_a_missing_cell_not_a_fatal_problem(tmp_path):
    R = Rerun(tmp_path)
    R.cheap("dna", 42)
    R.job("full_rerun", "turbo_dna_s42", ["turbo"], "dna", 42, {"turbo": None}, rc=1)
    L = lasso_cr.load_rerun(R.root)
    assert L["coverage"][("turbo", "dna")]["failed"] == [42] and L["summary"]["failed"] == 1


def test_fatal_content_problems(tmp_path):
    # sha256 of the file differs from the job record
    R = Rerun(tmp_path / "sha")
    R.cheap("dna", 42)
    p = os.path.join(R.root, "full_rerun", "results", "geowalk_dna_s42.json")
    d = json.load(open(p))
    d["best_score"] = 0.001
    json.dump(d, open(p, "w"))
    with pytest.raises(lasso_cr.LassoLoadError) as e:
        lasso_cr.load_rerun(R.root)
    assert "sha256" in str(e.value) and e.value.kinds() == {"provenance"}
    # the same cell in two run folders
    R = Rerun(tmp_path / "dup")
    R.job("full_rerun", "hvarfner_leukemia_s42", ["hvarfner_gp"], "leukemia", 42)
    R.job("leukemia_rerun", "s42", lasso_cr.CHEAP_LEUKEMIA + ["hvarfner_gp"], "leukemia", 42)
    with pytest.raises(lasso_cr.LassoLoadError) as e:
        lasso_cr.load_rerun(R.root)
    assert "duplicate" in e.value.kinds()
    # the same cell named by jobs of two run folders, only one of which left a result (a rerun of a failed pilot)
    R = Rerun(tmp_path / "dupjob")
    R.job("linear_bo_leukemia", "s42", ["linear_bo"], "leukemia", 42, {"linear_bo": None}, rc=124)
    R.job("full_rerun", "linbo_leukemia_s42", ["linear_bo"], "leukemia", 42)
    with pytest.raises(lasso_cr.LassoLoadError) as e:
        lasso_cr.load_rerun(R.root)
    assert e.value.kinds() == {"duplicate"} and "also named by a job of" in str(e.value)
    # fields contradict the file name
    R = Rerun(tmp_path / "fields")
    R.cheap("dna", 42)
    src = os.path.join(R.root, "full_rerun", "results", "geowalk_dna_s42.json")
    d = json.load(open(src))
    d["seed"] = 43
    raw = json.dumps(d).encode()
    open(src, "wb").write(raw)
    man = os.path.join(R.root, "full_rerun", "manifest.jsonl")
    rec = json.loads(open(man).read())
    for rf in rec["result_files"]:
        if rf["method"] == "geowalk":
            rf["sha256"] = hashlib.sha256(raw).hexdigest()
    open(man, "w").write(json.dumps(rec) + "\n")
    with pytest.raises(lasso_cr.LassoLoadError) as e:
        lasso_cr.load_rerun(R.root)
    assert e.value.kinds() == {"content"} and "contradict the file name" in str(e.value)
    # a manifest line cut by a copy made while the rerun was writing
    R = Rerun(tmp_path / "cut")
    R.cheap("dna", 42)
    with open(os.path.join(R.root, "full_rerun", "manifest.jsonl"), "a") as f:
        f.write('{"group": "full_rerun", "label": "cheap_dna_s43", "meth')
    with pytest.raises(lasso_cr.LassoLoadError) as e:
        lasso_cr.load_rerun(R.root)
    assert e.value.kinds() == {"manifest"}
    # a status file without a job record, and a status rc that contradicts the record
    R = Rerun(tmp_path / "status")
    R.cheap("dna", 42)
    open(os.path.join(R.root, "full_rerun", "status", "turbo_dna_s42.rc"), "w").write("0 t0 t1\n")
    with pytest.raises(lasso_cr.LassoLoadError) as e:
        lasso_cr.load_rerun(R.root)
    assert "without a manifest record" in str(e.value)
    os.remove(os.path.join(R.root, "full_rerun", "status", "turbo_dna_s42.rc"))
    open(os.path.join(R.root, "full_rerun", "status", "cheap_dna_s42.rc"), "w").write("124 t0 t1\n")
    with pytest.raises(lasso_cr.LassoLoadError) as e:
        lasso_cr.load_rerun(R.root)
    assert "status rc 124" in str(e.value)
    # a result the record lists as written is absent (incomplete copy)
    R = Rerun(tmp_path / "absent")
    R.cheap("dna", 42)
    os.remove(os.path.join(R.root, "full_rerun", "results", "random_dna_s42.json"))
    with pytest.raises(lasso_cr.LassoLoadError) as e:
        lasso_cr.load_rerun(R.root)
    assert "absent from results/" in str(e.value)


# ---- statistics: the convention of the committed lassobench.py -----------------------------------------
def test_table3_row_by_hand(tmp_path):
    rng = np.random.default_rng(3)
    cells = synthetic_cells(rng)
    del cells[("geowalk", "dna", 44)]                 # an unpaired cell
    cells[("cmaes_sub", "synt_hard", 45)] = float("inf")
    write_stored(str(tmp_path / "s"), cells)
    R = stored.lasso_stats(stored.load_lasso(str(tmp_path / "s")))
    norm = {t: np.mean([v for (m, tt, s), v in cells.items() if m == "random" and tt == t]) for t in DIMS}
    pairs = [(t, s) for t in DIMS for s in range(42, 50)
             if (("geowalk", t, s) in cells and math.isfinite(cells[("cmaes_sub", t, s)]))]
    x = np.asarray([cells[("geowalk", t, s)] / norm[t] for t, s in pairs])
    y = np.asarray([cells[("cmaes_sub", t, s)] / norm[t] for t, s in pairs])
    assert R["n"] == len(pairs) == 46
    assert R["gw"] == pytest.approx(x.mean()) and R["cma"] == pytest.approx(y.mean())
    assert R["delta"] == pytest.approx((x - y).mean())
    assert R["d"] == pytest.approx((x - y).mean() / (x - y).std(ddof=1))
    assert R["p"] == pytest.approx(sps.wilcoxon(x - y).pvalue, rel=1e-12)       # exact differences
    assert R["n_tasks"] == 6 and R["per_d"][("cmaes_sub", "synt_hard")]["n"] == 7
    assert R["per_d"][("random", "dna")]["mean"] == pytest.approx(1.0)
    assert R["generation"] == "stored" and R["fileset"]["count"] == sum(
        1 for (m, t, s), v in cells.items() if m in ("geowalk", "cmaes_sub", "random") and math.isfinite(v))


def test_generator_equals_the_committed_lassobench_script(tmp_path):
    """Table 3 row, task-level test and the per-dimension figure numbers against lassobench.py itself."""
    sys.path.insert(0, ANALYSIS)
    import lassobench as committed
    if committed.git_toplevel(str(tmp_path)):
        pytest.skip("pytest's temporary directory is inside a git work tree (lassobench.py refuses to write there)")
    rng = np.random.default_rng(11)
    cells = synthetic_cells(rng, methods=("random", "cmaes_sub", "geowalk", "linear_bo", "turbo"), seeds=range(42, 52))
    for k in [("geowalk", "dna", 44), ("cmaes_sub", "leukemia", 50), ("turbo", "synt_high", 42)]:
        del cells[k]
    cells[("geowalk", "synt_simple", 47)] = float("inf")
    d = str(tmp_path / "res")
    write_stored(d, cells)
    out = str(tmp_path / "out")
    r = subprocess.run([sys.executable, "-B", os.path.join(ANALYSIS, "lassobench.py"), "--results", d, "--outdir", out],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    ref = json.load(open(os.path.join(out, "cr_lasso.json")))
    row = ref["table3_lassobench_row"]["6 tasks, seeds 42-81 (current files)"]
    tm = ref["table3_lassobench_row"]["task_mean"]
    R = stored.lasso_stats(stored.load_lasso(d))
    assert R["n"] == row["n"]
    for mine, theirs in ((R["gw"], row["mean_a"]), (R["cma"], row["mean_b"]), (R["delta"], row["delta"]),
                         (R["d"], row["d"]), (R["p"], row["p"]), (R["p_task"], tm["task_level_exact_wilcoxon_p"])):
        assert mine == pytest.approx(theirs, rel=1e-12)
    assert R["tasks_cma_better"] == tm["signs_geocma_better"]
    checked = 0
    for m, per in ref["figure2"]["methods"].items():
        for D, v in per.items():
            task = {str(dd): t for t, dd in stored.LASSO_TASKS}[str(D)]
            x = R["per_d"].get((m, task))
            if not v.get("n"):
                assert x is None
                continue
            assert x["n"] == v["n"] and x["mean"] == pytest.approx(v["mean"]) and x["se0"] == pytest.approx(v["se"])
            checked += 1
    assert checked >= 24


def test_committed_script_write_guard(tmp_path):
    sys.path.insert(0, ANALYSIS)
    import lassobench as committed
    a = tmp_path / "NeurIPS"
    b = tmp_path / "NeurIPS-cr"
    a.mkdir()
    b.mkdir()
    assert committed.inside(str(a / "x" / "y"), str(a)) and committed.inside(str(a), str(a))
    assert not committed.inside(str(b / "out"), str(a))          # a common string prefix is not containment
    with pytest.raises(SystemExit):
        committed.check_outdir(str(a / "out"), [str(a)])
    if shutil.which("git") and not committed.git_toplevel(str(tmp_path)):
        committed.check_outdir(str(b / "out"), [str(a)])          # allowed
        repo = tmp_path / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        assert committed.git_toplevel(str(repo / "results" / "new")) != ""
        with pytest.raises(SystemExit) as e:
            committed.check_outdir(str(repo / "results" / "new"), [])
        assert "git work tree" in str(e.value)


# ---- tables and figure in both modes ---------------------------------------------------------------------
def _both_sets(tmp_path):
    rng = np.random.default_rng(5)
    sd = str(tmp_path / "lassobench")
    write_stored(sd, synthetic_cells(rng, methods=("random", "cmaes_sub", "geowalk", "linear_bo", "cmaes_full"),
                                     seeds=range(42, 48)))
    R = Rerun(tmp_path / "lassobench_cr")
    R.leukemia_complete(lambda s: {m: float(rng.uniform(0.5, 2.0)) for m in lasso_cr.CHEAP_LEUKEMIA})
    for t in lasso_cr.TASKS_LOW:
        for s in range(42, 46):
            R.cheap(t, s, {m: float(rng.uniform(0.5, 2.0)) for m in ("random", "cmaes_sub", "geowalk", "cmaes_full")})
            R.job("full_rerun", f"linbo_{t}_s{s}", ["linear_bo"], t, s, {"linear_bo": float(rng.uniform(0.5, 2.0))})
    R.job("linear_bo_leukemia", "s42", ["linear_bo"], "leukemia", 42, {"linear_bo": None}, rc=124, timeout_s="7200")
    return sd, R.root


def test_camera_ready_mode_prints_the_rerun_and_legacy_the_stored_set(tmp_path, capsys):
    sd, rd = _both_sets(tmp_path)
    ctx = ctx_for(tmp_path, "cr", sd, rd)
    t = stored.tab_lassobench(ctx)
    rec = ctx.reg.records
    Rr = stored.lasso_stats(lasso_cr.load_rerun(rd))
    Rs = stored.lasso_stats(stored.load_lasso(sd))
    assert t is not None and "ran in one environment" in t.caption and "INCOMPLETE RERUN" in t.caption
    assert "lassobench_correspondence" not in t.caption      # not printed (scope 2026-10-14)
    # RUN_RULES 2026-10-22 (format only): the paired blocks are a second float that refers back to the first
    if t.blocks:
        assert t.continued[1] == "tab:lassobench_family" and "\\ref{tab:lassobench}" in t.continued[0]
        tex = t.render("x")
        assert tex.count("\\begin{table}") == 2 and "\\label{tab:lassobench_family}" in tex
    assert rec["crLassoGeowalk"]["value"] == pytest.approx(Rr["gw"]) != pytest.approx(Rs["gw"])
    assert rec["crLassoN"]["value"] == 40 + 5 * 4
    assert rec["crNTasksLasso"]["value"] == Rr["n_tasks"] and rec["crNTasksLasso"]["text"] == str(Rr["n_tasks"])
    # every number carries a file set, and the rerun's manifests are part of it
    n_t3 = sum(len(Rr["loaded"]["cells"][m][tk]) for m in ("geowalk", "cmaes_sub", "random")
               for tk in Rr["loaded"]["cells"][m])
    assert rec["crLassoGeowalk"]["files"]["count"] == n_t3 + 3
    assert all(r["files"]["count"] > 0 for r in rec.values())
    assert rec["crLassoCellsExpected"]["value"] == 2322 and rec["crLassoCellsPresent"]["value"] == 240 + 5 * 4 * 5
    assert rec["crLassoLinearBoDSevenOneTwoNineRuns"]["value"] == 0
    assert rec["crLassoGeowalkDSixZeroRuns"]["value"] == 4 and rec["crLassoGeowalkDSixZeroN"]["value"] == 4
    assert rec["crLassoLeukemiaNotCompleted"]["value"] == "linear_bo"
    assert rec["crLassoLeukemiaNotCompleted"]["text"] == LAB("linear_bo")      # running text, no \\text box
    assert rec["crLassoLeukemiaNotCompletedN"]["value"] == 1 and rec["crLassoLeukemiaPendingN"]["value"] == 3
    rows = {r[0]: r for r in t.rows if isinstance(r, list)}
    assert rows[PLAB("linear_bo")][-1] == "n.c." and rows[PLAB("hvarfner_gp")][-1] == "pend."
    assert rows["\\geowalk{}"][1].endswith("(4/40)") and rows["\\geowalk{}"][-1].endswith("(40)")
    # loud: banner printed once, recorded in the run-wide warnings and in this variant's skip list
    out = capsys.readouterr().out
    assert out.count("!!! WARNING: LassoBench rerun coverage") == 1 and "geowalk dna: 4/40" in out
    rw = [w for w in ctx.cache["loud_warnings"] if "rerun coverage" in w]
    assert len(rw) == 1 and "seeds 46-81" in rw[0]
    assert [s["what"] for s in ctx.skipped] == ["rerun_coverage", "pub_absent"]       # GEN v5: no published set here
    assert os.path.exists(os.path.join(ctx.outdir, "lassobench_cr_coverage.md"))
    assert os.path.join(ctx.outdir, "lassobench_cr_coverage.md") in ctx.written
    # correspondence: stored values under their own prefix, the rerun unchanged
    c = stored.tab_lassobench_correspondence(ctx)
    assert c is not None and rec["crLassoStoredGeowalk"]["value"] == pytest.approx(Rs["gw"])
    assert rec["crLassoStoredN"]["value"] == Rs["n"] and "crLassoStoredLinearBoDSixZero" in rec
    assert [r[0] for r in c.rows if isinstance(r, list)] == ["Submitted (printed)", "Files of the submitted version, recomputed",
                                                             "Rerun for this version (printed)"]
    assert c.rows[0][1:5] == ["$0.790$", "$0.440$", "$0.350$", "$0.68$"]      # reference_values.json, ids S-t3-*
    # a second variant of the same run: no second banner, but its own skip record
    ctx2 = Ctx(ds=None, eng=None, reg=Registry("x", "0" * 64, "v2"), var=VARIANTS["R1_flat"], outdir=ctx.outdir,
               script_commit="x", mode="cr", extra=dict(ctx.extra))
    stored.tab_lassobench(ctx2)
    assert "!!!" not in capsys.readouterr().out and [s["what"] for s in ctx2.skipped] == ["rerun_coverage", "pub_absent"]
    assert len([w for w in ctx.cache["loud_warnings"] if "rerun coverage" in w]) == 1

    # --legacy: the stored set, whatever the rerun folder holds
    lg = ctx_for(tmp_path, "legacy", sd, rd)
    tl = stored.tab_lassobench(lg)
    assert tl is not None and "runs of the submitted version" in tl.caption and "rerun" not in tl.caption
    assert lg.reg.records["crLassoGeowalk"]["value"] == pytest.approx(Rs["gw"])
    assert "crLassoCellsExpected" not in lg.reg.records and not lg.cache.get("loud_warnings")
    assert stored.tab_lassobench_correspondence(lg) is None
    assert stored.lasso_analysis(lg)["generation"] == "stored"


def test_camera_ready_mode_without_the_rerun_falls_back_to_stored_loudly(tmp_path, capsys):
    sd, _ = _both_sets(tmp_path)
    ctx = ctx_for(tmp_path, "cr", sd, str(tmp_path / "nowhere"))
    t = stored.tab_lassobench(ctx)
    assert t is not None and "runs of the submitted version" in t.caption
    assert stored.lasso_analysis(ctx)["generation"] == "stored"
    assert "fall back to the STORED" in capsys.readouterr().out
    assert any("rerun folder" in w for w in ctx.cache["loud_warnings"])
    assert stored.tab_lassobench_correspondence(ctx) is None


def test_unclean_rerun_stops_the_table(tmp_path):
    sd, rd = _both_sets(tmp_path)
    Rerun(rd).cheap("dna", 60, rc=137)
    ctx = ctx_for(tmp_path, "cr", sd, rd)
    with pytest.raises(lasso_cr.LassoLoadError):
        stored.tab_lassobench(ctx)
    ctx = ctx_for(tmp_path, "cr", sd, rd)
    ctx.extra["lasso_cr_allow_unclean"] = [f"{m}_dna_s60.json" for m in ("random", "cmaes_sub", "geowalk", "cmaes_full")]
    assert stored.tab_lassobench(ctx) is not None


def test_result_outside_the_expected_design_is_used_but_loud(tmp_path, capsys):
    sd, rd = _both_sets(tmp_path)
    R = Rerun(rd)
    # Xu at D = 7129 is a one-seed pilot (seed 42): a seed-50 run is neither a cell nor a job of the design
    R.job("full_rerun", "extra_xu_leukemia_s50", ["xu_standard_gp"], "leukemia", 50)
    L = lasso_cr.load_rerun(rd)
    assert L["summary"]["outside_design"] == ["xu_standard_gp_leukemia_s50"]
    assert L["summary"]["unexpected_job_labels"] == {"full_rerun": ["extra_xu_leukemia_s50"]}
    ctx = ctx_for(tmp_path, "cr", sd, rd)
    stored.tab_lassobench(ctx)
    out = capsys.readouterr().out
    assert "OUTSIDE the expected design" in out and "extra_xu_leukemia_s50" in out
    assert "outside the expected design" in open(os.path.join(ctx.outdir, "lassobench_cr_coverage.md")).read()


def test_completed_pilot_is_labelled_as_a_pilot_not_as_full_coverage(tmp_path):
    """The generic mechanism, with an arm that is still a one-seed pilot (Xu; Papenmeier no longer is one)."""
    sd, rd = _both_sets(tmp_path)
    Rerun(rd).job("full_rerun", "pilot_xu_leukemia_s42", ["xu_standard_gp"], "leukemia", 42, {"xu_standard_gp": 1.7})
    ctx = ctx_for(tmp_path, "cr", sd, rd)
    t = stored.tab_lassobench(ctx)
    rows = {r[0]: r for r in t.rows if isinstance(r, list)}
    assert rows[PLAB("xu_standard_gp")][-1].endswith("(pilot, 1)")
    assert ("At $D=7129$, " + LAB("xu_standard_gp") + " ran one pilot seed only. The pilot finished within the time "
            "limit, but the method was not run on the 40 seeds.") in full_caption(t)
    L = stored.lasso_rerun_analysis(ctx)["loaded"]
    assert L["leukemia"]["xu_standard_gp"] == {"status": "pilot_completed", "pilot": True, "finite": 1, "expected": 1,
                                               "text": "the pilot seed completed within the time limit (one seed only)"}
    assert L["coverage"][("xu_standard_gp", "leukemia")]["expected"] == 1
    assert ctx.reg.records["crLassoXuStandardGpDSevenOneTwoNineN"]["value"] == 1
    assert ctx.reg.records["crLassoXuStandardGpDSevenOneTwoNineRuns"]["rule"].endswith("(pilot seed only)")
    assert ctx.reg.records["crLassoLeukemiaNotCompleted"]["value"] == "linear_bo"
    assert "| xu_standard_gp | leukemia | 1 / 1 (pilot) | 1 |" in lasso_cr.coverage_report(L)
    assert "(pilot, 1)" in "".join(r[-1] for r in stored.tab_lassobench_correspondence(ctx).blocks[-1][-1])
    # Papenmeier is no pilot any more: the same single completed seed 42 is one of 40 expected seeds
    Rerun(rd).papenmeier_leukemia([42], loss_of=lambda s: 1.7)
    ctx = ctx_for(tmp_path / "p", "cr", sd, rd)
    rows = {r[0]: r for r in stored.tab_lassobench(ctx).rows if isinstance(r, list)}
    assert rows[PLAB("papenmeier_msr")][-1].endswith("\\,(1/40)") and rows[PLAB("xu_standard_gp")][-1].endswith("(pilot, 1)")
    assert stored.lasso_rerun_analysis(ctx)["loaded"]["leukemia"]["papenmeier_msr"]["status"] == "run"


def test_coverage_is_clean_only_when_nothing_needs_announcing():
    clean = {"complete": True, "nonfinite": 0, "unrecorded": 0, "short": 0, "outside_design": [],
             "declared_unclean": [], "unexpected_job_labels": {}}
    assert lasso_cr.coverage_is_clean(clean)
    for key, val in (("complete", False), ("nonfinite", 2), ("unrecorded", 1), ("short", 1),
                     ("outside_design", ["xu_standard_gp_leukemia_s50"]), ("declared_unclean", ["x"]),
                     ("unexpected_job_labels", {"full_rerun": ["extra"]})):
        assert not lasso_cr.coverage_is_clean(dict(clean, **{key: val})), key


def test_ten_methods():
    assert [m for m, _ in stored.LASSO_METHODS] == ["cmaes_sub", "geowalk", "random", "hvarfner_gp", "papenmeier_msr",
                                                    "xu_standard_gp", "linear_bo", "cmaes_full", "turbo", "baxus"]
    assert sorted(m for m, _ in stored.LASSO_METHODS) == sorted(lasso_cr.ALL_METHODS)


@pytest.mark.skipif(not shutil.which("pdflatex"), reason="pdflatex not available")
@pytest.mark.parametrize("mode", ["cr", "legacy"])
def test_tables_compile_at_text_width_without_overfull_boxes(tmp_path, mode):
    sd, rd = _both_sets(tmp_path)
    ctx = ctx_for(tmp_path, mode, sd, rd)
    files = []
    for b in (stored.tab_lassobench, stored.tab_lassobench_correspondence):
        t = b(ctx)
        if t is not None:
            files.append(t.write(ctx.outdir, "test"))
    ctx.reg.write_tex(os.path.join(ctx.outdir, "numbers.tex"), "test")
    body = [PREAMBLE_CHECK, "\\input{numbers.tex}"] + [f"\\input{{{os.path.basename(f)}}}\\clearpage" for f in files]
    body += ["\\crLassoGeowalk{} " + ("\\crLassoLeukemiaNotCompleted{}" if mode == "cr" else ""), "\\end{document}"]
    with open(os.path.join(ctx.outdir, "check.tex"), "w") as f:
        f.write("\n".join(body))
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "check.tex"], cwd=ctx.outdir,
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:]
    log = open(os.path.join(ctx.outdir, "check.log"), errors="replace").read()
    assert "Overfull \\hbox" not in log
    assert len(files) == (2 if mode == "cr" else 1)


@pytest.mark.parametrize("mode", ["cr", "legacy"])
def test_figure_all_methods_truetype(tmp_path, mode):
    from crlib import figures
    sd, rd = _both_sets(tmp_path)
    ctx = ctx_for(tmp_path, mode, sd, rd)
    p = figures.fig_lassobench(ctx)
    assert p and os.path.getsize(p) > 1000
    # methods with >= 3 finite seeds at some dimension: rerun = random, GeoCMA, GeoWalk + Linear BO (the full-D CMA-ES,
    # TuRBO and BAxUS rows of the rerun are correspondence-only in camera-ready mode, RUN_RULES 2026-10-12 (5)); stored = 5
    assert ctx.reg.records["crFigLassobenchMethods"]["value"] == (4 if mode == "cr" else 5)
    if shutil.which("pdffonts"):
        out = subprocess.run(["pdffonts", p], capture_output=True, text=True).stdout
        assert "Type 3" not in out and "TrueType" in out
