import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from crlib import TASKS_T11  # noqa: E402


def ours_record(solver, task_short, seed, *, status="completed", fallback=0, commit="abc1234", dirty=False,
                scores=None, budget=30, task_field=True):
    scores = scores if scores is not None else [0.1 + 0.01 * i + 0.001 * seed for i in range(30)]
    best, trace = -1e9, []
    for s in scores:
        best = max(best, s)
        trace.append(best)
    d = {"mode": "ours", "solver": solver, "seed": seed, "best_score": trace[-1], "best_trace": trace,
         "all_scores": scores, "surrogate_fallback_count": fallback, "status": status, "budget": budget,
         "provenance": {"git_commit": commit, "git_dirty": dirty, "hostname": "testhost"}}
    if task_field:
        d["task"] = TASKS_T11[task_short]
    if status is None:
        del d["status"]
    return d


@pytest.fixture
def make_arm(tmp_path):
    """make_arm(dirname, solver, cells=[(task, seed)], **record_kwargs) -> directory path."""
    def _make(dirname, solver, cells, suffix="_128d", per_cell=None, **kw):
        d = tmp_path / dirname
        d.mkdir(parents=True, exist_ok=True)
        for t, s in cells:
            rec_kw = dict(kw)
            if per_cell and (t, s) in per_cell:
                rec_kw.update(per_cell[(t, s)])
            rec = ours_record(solver, t, s, **rec_kw)
            (d / f"ours_{solver}_{TASKS_T11[t]}_s{s}{suffix}.json").write_text(json.dumps(rec))
        return str(d)
    return _make


@pytest.fixture
def write_config(tmp_path):
    def _write(arms: dict, tasks=("adip", "rano"), seeds="42-44", cell_filter="completed", name="cfg.json"):
        cfg = {"schema": "cr-sources-v1", "name": "test", "roots": {"T": str(tmp_path)}, "tasks": list(tasks),
               "seeds": seeds, "cell_filter": cell_filter, "arms": arms}
        p = tmp_path / name
        p.write_text(json.dumps(cfg))
        return str(p)
    return _write


def arm_cfg(solver, kind="cr", **kw):
    a = {"label": solver, "kind": kind, "format": "ours", "dir": "${T}/" + solver,
         "pattern": f"ours_{solver}_{{task}}_s{{seed}}_128d.json", "solver_field": solver,
         "provenance": "require" if kind == "cr" else "record"}
    a.update(kw)
    return a
