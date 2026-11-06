"""Stored September blocks (crlib/september.py): the side loader fails loudly, the exact task-level test, the
three cell conventions, logP never pooled, planted signs and levels come back, every number has a file set."""
import itertools
import json
import os

import numpy as np
import pytest
from scipy import stats as sps

from crlib import TASKS_T11, stats
from crlib import september as sep
from crlib.config import load_config
from crlib.context import VARIANTS, Ctx
from crlib.engine import Engine
from crlib.loader import load_all
from crlib.numbers import Registry

SEEDS3 = [42, 43, 44]


# ---- helpers ---------------------------------------------------------------------------------------------
def _write(d, solver, task_short, seed, value, *, cold=0.05, status="completed", fallback=0, commit=None,
           dirty=False, n=30, name=None, solver_field=None, nested=True):
    """One cell: 5 cold-start scores `cold`, then the best-so-far jumps to `value` (a planted best@30)."""
    scores = ([cold] * 5 + [value] * 25)[:n]
    best, trace = -1e9, []
    for s in scores:
        best = max(best, s)
        trace.append(best)
    commit = commit or (sep.E1_COMMIT if solver in sep.E1_SOLVERS.values() else sep.P1_COMMIT)
    prov = {"git_commit": commit, "git_dirty": dirty, "hostname": "testhost"}
    rec = {"mode": "ours", "solver": solver_field or solver, "task": sep.ALL_TASKS[task_short], "seed": seed,
           "best_trace": trace, "all_scores": scores, "status": status, "surrogate_fallback_count": fallback,
           "provenance": {"provenance": prov} if nested else prov}
    p = os.path.join(d, name or f"ours_{solver}_{sep.ALL_TASKS[task_short]}_s{seed}_256d.json")
    with open(p, "w") as f:
        json.dump(rec, f)
    return p


def _e1_dir(tmp_path, level, tasks=None, seeds=SEEDS3, name="em1"):
    """level(role, task, seed) -> planted best@30."""
    d = tmp_path / name
    d.mkdir()
    for role, solver in sep.E1_SOLVERS.items():
        for t in (tasks or sep.ALL_TASKS):
            for s in seeds:
                _write(str(d), solver, t, s, level(role, t, s))
    return str(d)


def _p1_dir(tmp_path, level, seeds=SEEDS3, name="p1"):
    d = tmp_path / name
    d.mkdir()
    for role, solver in sep.P1_SOLVERS.items():
        for t in sep.T18:
            for s in seeds:
                _write(str(d), solver, t, s, level(role, t, s))
    return str(d)


def _ctx(tmp_path, write_config, extra, tasks=None, seeds="42-44", arms=None, variant="default"):
    arms = arms or {"stub": {"kind": "stored", "format": "ours", "dir": "${T}/absent",
                             "pattern": "ours_stub_{task}_s{seed}_128d.json", "required": False}}
    ds = load_all(load_config(write_config(arms, tasks=tuple(tasks or TASKS_T11), seeds=seeds)), None, log=lambda m: None)
    var = VARIANTS[variant]
    ex = {"_shared_cache": {}, "inputs_root": str(tmp_path)}
    ex.update(extra)
    return Ctx(ds=ds, eng=Engine(ds, cell_filter=var.cell_filter), reg=Registry("x", "0" * 64, var.name), var=var,
               outdir=str(tmp_path), script_commit="x", mode="legacy", extra=ex)


@pytest.fixture
def three_seeds(monkeypatch):
    monkeypatch.setattr(sep, "SEEDS", SEEDS3)


# ---- exact task-level test ---------------------------------------------------------------------------------
def test_exact_signed_rank_equals_scipy_on_tie_free_data():
    rng = np.random.default_rng(0)
    for n in (2, 5, 11, 18):
        for _ in range(5):
            d = rng.normal(0.3, 1.0, n)
            assert sep.exact_signed_rank_p(d) == pytest.approx(sps.wilcoxon(d, method="exact").pvalue, rel=1e-12, abs=1e-15)
    assert sep.exact_signed_rank_p(np.arange(1, 12) * 0.01) == pytest.approx(2 / 2048)      # 11 positive: 2/2^11
    assert np.isnan(sep.exact_signed_rank_p([0.0, 0.3]))                                    # one non-zero value


def test_exact_signed_rank_with_ties_and_zeros_equals_brute_force():
    d = np.array([0.5, -0.5, 0.5, 1.0, -0.25, 0.25, 0.0, 2.0, 2.0])
    nz = d[d != 0]
    ranks = sps.rankdata(np.abs(nz))
    w = ranks[nz > 0].sum()
    ws = [sum(r for r, s in zip(ranks, signs) if s) for signs in itertools.product((0, 1), repeat=nz.size)]
    ws = np.asarray(ws)
    ref = min(1.0, 2 * min((ws <= w + 1e-9).mean(), (ws >= w - 1e-9).mean()))
    assert sep.exact_signed_rank_p(d) == pytest.approx(ref, abs=1e-15)


# ---- the side loader fails loudly ----------------------------------------------------------------------------
def _small_block(tmp_path, name="blk"):
    d = tmp_path / name
    d.mkdir()
    for solver in sep.E1_SOLVERS.values():
        for t in ("adip", "logp"):
            for s in SEEDS3:
                _write(str(d), solver, t, s, 0.4)
    return str(d)


def test_load_block_reads_every_cell_with_a_hash(tmp_path):
    d = _small_block(tmp_path)
    blk = sep.load_block(d, sep.E1_SOLVERS, ["adip", "logp"], sep.E1_COMMIT, seeds=SEEDS3)
    assert set(blk) == set(sep.E1_SOLVERS) and all(len(v) == 6 for v in blk.values())
    c = blk["BA"][("logp", 43)]
    assert c.file.endswith("ours_hvarfner_scaled_gp_box_argmax_logp_s43_256d.json") and len(c.sha256) == 64
    assert c.complete and c.clean and c.cold5 == pytest.approx(0.05) and c.trace[29] == pytest.approx(0.4)


def test_load_block_missing_duplicate_contradictory_and_provenance(tmp_path):
    args = (sep.E1_SOLVERS, ["adip", "logp"], sep.E1_COMMIT)
    d = _small_block(tmp_path, "a")
    os.remove(os.path.join(d, "ours_hvarfner_scaled_gp_amlodipine_mpo_s43_256d.json"))
    with pytest.raises(sep.SeptemberError, match="missing cell hvarfner_scaled_gp adip:43"):
        sep.load_block(d, *args, seeds=SEEDS3)
    d = _small_block(tmp_path, "b")
    _write(d, "hvarfner_scaled_gp", "adip", 43, 0.4, name="ours_hvarfner_scaled_gp_amlodipine_mpo_s043_256d.json")
    with pytest.raises(sep.SeptemberError, match="duplicate cell"):
        sep.load_block(d, *args, seeds=SEEDS3)
    d = _small_block(tmp_path, "c")
    _write(d, "hvarfner_scaled_gp_box_argmax", "adip", 42, 0.4, solver_field="hvarfner_scaled_gp")
    with pytest.raises(sep.SeptemberError, match="contradict the file name"):
        sep.load_block(d, *args, seeds=SEEDS3)
    d = _small_block(tmp_path, "d")
    _write(d, "hvarfner_scaled_gp", "logp", 44, 0.4, commit="1234567")
    with pytest.raises(sep.SeptemberError, match="provenance commit"):
        sep.load_block(d, *args, seeds=SEEDS3)
    d = _small_block(tmp_path, "e")
    _write(d, "hvarfner_scaled_gp", "logp", 44, 0.4, dirty=True)
    with pytest.raises(sep.SeptemberError, match="clean tree"):
        sep.load_block(d, *args, seeds=SEEDS3)
    d = _small_block(tmp_path, "f")
    open(os.path.join(d, "ours_hvarfner_scaled_gp_logp_s44_256d.json"), "w").write("{not json")
    with pytest.raises(sep.SeptemberError, match="unreadable"):
        sep.load_block(d, *args, seeds=SEEDS3)
    with pytest.raises(sep.SeptemberError, match="does not exist"):
        sep.load_block(str(tmp_path / "nope"), *args, seeds=SEEDS3)


def test_solver_names_that_prefix_each_other_are_not_confused(tmp_path):
    d = _small_block(tmp_path)
    blk = sep.load_block(d, sep.E1_SOLVERS, ["adip"], sep.E1_COMMIT, seeds=SEEDS3)
    for role, solver in sep.E1_SOLVERS.items():
        assert all(os.path.basename(c.file).startswith(f"ours_{solver}_amlodipine_mpo_s") for c in blk[role].values())


# ---- cell conventions -------------------------------------------------------------------------------------------
def test_clean_completed_and_itt_conventions(tmp_path):
    d = _small_block(tmp_path)
    _write(d, "hvarfner_scaled_gp_box_faithful", "adip", 42, 0.3, status="crashed: RuntimeError: device", n=12)
    _write(d, "hvarfner_scaled_gp", "adip", 43, 0.4, fallback=3)
    _write(d, "hvarfner_scaled_gp_sphere_continuous", "adip", 44, 0.4, status="crashed: x", n=0)
    blk = sep.load_block(d, sep.E1_SOLVERS, ["adip"], sep.E1_COMMIT, seeds=SEEDS3)
    roles = list(sep.E1_SOLVERS)
    assert sep.eligible(blk, roles, {"adip"}, "clean") == set()
    assert sep.eligible(blk, roles, {"adip"}, "completed") == {("adip", 43)}
    assert sep.eligible(blk, roles, {"adip"}, "itt") == {("adip", s) for s in SEEDS3}
    assert sep.cell_value(blk, "BC", ("adip", 42), "itt") == pytest.approx(0.3)       # best so far of the 12 values
    assert np.isnan(sep.cell_value(blk, "BC", ("adip", 42), "completed"))
    assert sep.cell_value(blk, "SC", ("adip", 44), "itt") == pytest.approx(0.05)      # imputed: shared cold-start best@5


# ---- E-1 extra tasks -------------------------------------------------------------------------------------------
# planted per-task contrasts: (support SA - BA, interaction); logP huge and of the opposite sign
# (all |values| distinct, so the exact test has no ties to resolve)
SUP = {t: 0.02 + 0.001 * i for i, t in enumerate(sep.T11)}
SUP.update(albu=0.04, cele=0.0105, isoa=-0.07, isob=-0.15, mest=-0.045, thio=-0.0115, trog=0.0205, logp=-700.0, valt=0.0,
           qed=-0.0125)
INT = {t: 0.03 + 0.0011 * i for i, t in enumerate(sep.T11)}
INT.update(med1=-0.0105, albu=0.05, cele=0.001, isoa=0.052, isob=-0.0405, mest=-0.01, thio=-0.003, trog=0.02, logp=250.0,
           valt=0.0, qed=-0.0015)


def _e1_level(role, t, s):
    base = 1000.0 if t == "logp" else 0.3
    jitter = 0.001 * (s - 43)                       # the same in all four cells: contrasts stay exact per seed
    return {"SA": base + SUP[t], "BA": base, "BC": base, "SC": base + SUP[t] - INT[t]}[role] + jitter


def test_e1_extra_tasks_planted_signs_and_logp_never_pooled(tmp_path, write_config, three_seeds):
    d = _e1_dir(tmp_path, _e1_level)
    ctx = _ctx(tmp_path, write_config, {"e1_dir": d})
    t = sep.tab_e1_extra_tasks(ctx)
    R = ctx.reg.records
    assert t is not None and len([r for r in t.rows if isinstance(r, list)]) == 10          # 8 extra + 2 excluded
    assert R["crEoneExtraSupportPositive"]["text"] == "3/8"                                 # albu, cele, trog
    assert R["crEoneExtraInteractionPositive"]["text"] == "5/8"                             # + isoa, logp (sign only)
    assert R["crEoneTNineteenInteractionPositive"]["text"] == "15/19"                       # 10 of T11 + 5
    assert R["crEoneTNineteenSupportPositive"]["text"] == "14/19"
    # pooled T18: the mean of the 18 planted per-task contrasts; logP (|700|, |250|) would dominate if it were pooled
    assert R["crEoneTEighteenSupportDelta"]["value"] == pytest.approx(np.mean([SUP[k] for k in sep.T18]), abs=1e-9)
    assert R["crEoneTEighteenIntDelta"]["value"] == pytest.approx(np.mean([INT[k] for k in sep.T18]), abs=1e-9)
    assert R["crEoneTEighteenIntTasks"]["text"] == "14/18" and R["crEoneTEighteenSupportTasks"]["text"] == "14/18"
    assert R["crEoneTEighteenIntPTask"]["value"] == pytest.approx(
        sep.exact_signed_rank_p([INT[k] for k in sep.T18]))
    assert R["crEoneExtraSevenSupportTasks"]["text"] == "3/7" and R["crEoneTElevenIntTasks"]["text"] == "10/11"
    assert R["crEoneTEighteenN"]["value"] == 18 * 3 and R["crEoneTEighteenNItt"]["value"] == 18 * 3
    assert abs(R["crEoneTEighteenSupportDelta"]["value"]) < 0.05                           # no logP leak
    assert R["crEoneExtraLogpSupportSign"]["text"] == "-" and R["crEoneExtraLogpInteractionSign"]["text"] == "+"
    assert R["crEoneExtraIsoaShellArgmax"]["value"] == pytest.approx(0.3 - 0.07)
    lo, hi = R["crEoneTEighteenIntCiLo"]["value"], R["crEoneTEighteenIntCiHi"]["value"]
    assert (lo, hi) == pytest.approx(stats.task_bootstrap_ci({k: INT[k] for k in sep.T18}), abs=1e-9)
    for name, rec in R.items():
        assert rec["files"]["count"] > 0, name
    assert R["crEoneTEighteenIntDelta"]["files"]["count"] == 18 * 3 * 4
    assert ctx.extra["september_notes"]["e1_arm_check"].startswith("not checked")


def test_e1_clean_convention_drops_fallback_cells_and_itt_keeps_all(tmp_path, write_config, three_seeds):
    d = _e1_dir(tmp_path, _e1_level)
    _write(d, sep.E1_SOLVERS["SA"], "isoa", 42, 0.9, fallback=25)                 # flat cold start: fallback steps
    _write(d, sep.E1_SOLVERS["BC"], "albu", 44, 0.31, status="crashed: x", n=20)  # unfinished run
    ctx = _ctx(tmp_path, write_config, {"e1_dir": d})
    assert sep.tab_e1_extra_tasks(ctx) is not None
    R = ctx.reg.records
    assert (R["crEoneExtraIsoaN"]["value"], R["crEoneExtraIsoaNCompleted"]["value"]) == (2, 3)
    assert (R["crEoneExtraAlbuN"]["value"], R["crEoneExtraAlbuNCompleted"]["value"]) == (2, 2)
    assert (R["crEoneTEighteenN"]["value"], R["crEoneTEighteenNCompleted"]["value"], R["crEoneTEighteenNItt"]["value"]) \
        == (52, 53, 54)
    assert R["crEoneTEighteenSupportDeltaCompleted"]["value"] > R["crEoneTEighteenSupportDelta"]["value"]   # the 0.9 cell


def test_e1_table_agrees_with_the_e1_arms_or_raises(tmp_path, write_config, three_seeds):
    d = _e1_dir(tmp_path, _e1_level)
    arms = {f"e1_{s}": {"kind": "stored", "format": "ours", "dir": d, "pattern": f"ours_{s}_{{task}}_s{{seed}}_256d.json",
                        "solver_field": s, "provenance": "record"} for s in sep.E1_SOLVERS.values()}
    ctx = _ctx(tmp_path, write_config, {}, arms=arms)
    assert sep.e1_dir(ctx) == d                                           # taken from the e1_* arm of the config
    assert sep.tab_e1_extra_tasks(ctx) is not None
    assert ctx.extra["september_notes"]["e1_arm_check"] == "checked"
    A = sep.e1_analysis(ctx)
    A["conv"]["clean"]["pr"][("int", "T11")].task_diffs["adip"] += 0.5
    with pytest.raises(sep.SeptemberError, match="disagrees with the e1_"):
        sep._check_against_arms(ctx, A)


def test_tables_are_skipped_on_a_task_subset_and_when_inputs_are_absent(tmp_path, write_config, three_seeds, capsys):
    d = _e1_dir(tmp_path, _e1_level)
    ctx = _ctx(tmp_path, write_config, {"e1_dir": d}, tasks=("rano",))
    assert sep.tab_e1_extra_tasks(ctx) is None and sep.tab_selection_rule(ctx) is None
    assert {s["table"] for s in ctx.skipped} == {"tab_e1_extra_tasks", "tab_selection_rule"} and not ctx.reg.records
    ctx = _ctx(tmp_path, write_config, {"e1_dir": str(tmp_path / "nope"), "p1_sel_dir": str(tmp_path / "nope2")})
    assert sep.tab_e1_extra_tasks(ctx) is None and sep.tab_selection_rule(ctx) is None
    assert sep.tab_selection_rule(ctx) is None                           # second variant: no second banner
    out = capsys.readouterr().out
    assert out.count("!!! WARNING") == 2 and len(ctx.cache["loud_warnings"]) == 2 and not ctx.reg.records


# ---- selection rule ---------------------------------------------------------------------------------------------
P1_LEVEL = {"US": 0.41, "UB": 0.39, "TS": 0.408, "TB": 0.386, "LS": 0.417, "LB": 0.391, "LR": 0.414}


def _p1_level(role, t, s):
    shift = 0.1 * (list(sep.T18).index(t) % 3)          # task effect, cancels in every contrast
    return P1_LEVEL[role] + shift + 0.001 * (s - 43)


def test_selection_rule_planted_levels(tmp_path, write_config, three_seeds):
    e1 = _e1_dir(tmp_path, _e1_level)
    p1 = _p1_dir(tmp_path, _p1_level)
    ctx = _ctx(tmp_path, write_config, {"e1_dir": e1, "p1_sel_dir": p1})
    t = sep.tab_selection_rule(ctx)
    R = ctx.reg.records
    assert t is not None and sep.LABEL in t.caption and "4780475" in t.caption
    assert len([r for r in t.rows if isinstance(r, list)]) == 12 and len([r for r in t.rows if r[0] == "group"]) == 5
    assert R["crSelN"]["value"] == 33 and R["crSelTEighteenN"]["value"] == 54
    exp = {"crSelNoneSupport": 0.02, "crSelTsSupport": 0.022, "crSelTsSurrShell": -0.002, "crSelTsSurrBox": -0.004,
           "crSelLinSupport": 0.026, "crSelLinSurrShell": 0.007, "crSelLinSurrBox": 0.001,
           "crSelLinShippedSupport": 0.023, "crSelLinShippedSurrShell": 0.004}
    for k, v in exp.items():
        assert R[k + "Delta"]["value"] == pytest.approx(v, abs=1e-9), k
        assert R[k + "Tasks"]["text"] == ("11/11" if v > 0 else "0/11"), k
        assert R[k + "PTask"]["value"] == pytest.approx(2 / 2048) and R[k + "Delta"]["files"]["count"] == 66
        assert R[k.replace("crSel", "crSelTEighteen") + "Delta"]["value"] == pytest.approx(v, abs=1e-9)
    # qLogEI rows: E-1 argmax cells against the uniform arms of the p1 block (they cross the two sets)
    shift = np.mean([0.1 * (i % 3) for i in range(11)])
    assert R["crSelLogeiSurrShellDelta"]["value"] == pytest.approx(
        0.3 + np.mean([SUP[k] for k in sep.T11]) - 0.41 - shift, abs=1e-9)
    assert R["crSelLevelUniformShell"]["value"] == pytest.approx(0.41 + shift, abs=1e-9)
    assert R["crSelLevelTsShellFallbackCells"]["value"] == 0
    for name, rec in R.items():
        assert rec["files"]["count"] > 0, name


def test_selection_rule_without_e1_and_under_the_clean_variant(tmp_path, write_config, three_seeds):
    p1 = _p1_dir(tmp_path, _p1_level)
    _write(p1, sep.P1_SOLVERS["TS"], "siga", 44, 0.9, fallback=2)
    ctx = _ctx(tmp_path, write_config, {"e1_dir": str(tmp_path / "nope"), "p1_sel_dir": p1})
    t = sep.tab_selection_rule(ctx)
    assert t is not None and "crSelLogeiSupportDelta" not in ctx.reg.records          # qLogEI rows omitted, loudly
    assert any(s["what"] == "qLogEI rows" for s in ctx.skipped)
    assert ctx.reg.records["crSelN"]["value"] == 33 and ctx.reg.records["crSelLevelTsShellFallbackCells"]["value"] == 1
    ctx2 = _ctx(tmp_path, write_config, {"e1_dir": str(tmp_path / "nope"), "p1_sel_dir": p1}, variant="clean_nofallback")
    assert sep.tab_selection_rule(ctx2) is not None
    assert ctx2.reg.records["crSelN"]["value"] == 32                                  # the fallback cell leaves every arm
    assert ctx2.reg.records["crSelTsSupportDelta"]["value"] < ctx.reg.records["crSelTsSupportDelta"]["value"]


def test_small_task_level_p_is_printed_with_an_exponent():
    assert sep.pfmt_task(7.63e-6) == "7.6\\times10^{-6}" and sep.pfmt_task(0.0023) == "0.0023"
    assert sep.pfmt_task(0.44) == "0.44" and sep.pfmt_task(float("nan")) == "--"


def test_selection_rule_reports_arms_that_do_not_share_the_cold_start(tmp_path, write_config, three_seeds):
    p1 = _p1_dir(tmp_path, _p1_level)
    ctx = _ctx(tmp_path, write_config, {"e1_dir": str(tmp_path / "nope"), "p1_sel_dir": p1})
    t = sep.tab_selection_rule(ctx)
    assert ctx.reg.records["crSelColdStartMismatch"]["value"] == 0 and "same five cold-start molecules" in t.caption
    p1b = _p1_dir(tmp_path, _p1_level, name="p1b")
    _write(p1b, sep.P1_SOLVERS["LB"], "trog", 43, 0.6, cold=0.07)          # another cold start in one arm of one cell
    ctx = _ctx(tmp_path, write_config, {"e1_dir": str(tmp_path / "nope"), "p1_sel_dir": p1b})
    t = sep.tab_selection_rule(ctx)
    assert ctx.reg.records["crSelColdStartMismatch"]["value"] == 1 and "WARNING: best@5 differs" in t.caption
