"""GEN v3: per-arm seed ranges and per-root expected commits, the control block (C1-C3 readings computed
mechanically), the tie rule in the cell-level Wilcoxon, seed-block permutation and shared-cold-start correlation,
recovered fraction, stable Monte Carlo digits and the quadrature, the MSR guard, the anytime metrics, and an
end-to-end camera-ready run on synthetic files of every manifest arm."""
import json
import math
import os
import re
import shutil
import sys

import numpy as np
import pytest
from scipy import stats as sps

from conftest import arm_cfg, ours_record
from crlib import TASKS_T11, controls, extras, mc, stats
from crlib.config import load_config, parse_seeds
from crlib.context import VARIANTS, Ctx
from crlib.engine import Engine
from crlib.loader import LoaderError, expected_commit_for, load_all
from crlib.numbers import Registry

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.dirname(HERE)
T11 = sorted(TASKS_T11)
C_MAIN, C_CTL, C_PUB = "1111111aaaaaaa", "2222222bbbbbbb", "3333333ccccccc"
C_C4 = "2a899ad301d3cb0badd906adfb4d81ad0b50f168"       # control C4, root CRC4 (pinned in make_tables.py)
C_RD = "4444444ddddddd"            # root CRRD: the GuacaMol runs of 2026-11-02


def _write(d, solver, t, s, *, commit, cr_arm=None, scores=None, **kw):
    rec = ours_record(solver, t, s, commit=commit, scores=scores, **kw)
    if cr_arm:
        rec["provenance"].update(cr_arm=cr_arm, run_overrides={})
    d.mkdir(parents=True, exist_ok=True)
    (d / f"ours_{solver}_{TASKS_T11[t]}_s{s}_128d.json").write_text(json.dumps(rec))


def _ctx(ds, tmp_path, mode="cr"):
    return Ctx(ds=ds, eng=Engine(ds), reg=Registry("x", "0" * 64, "default"), var=VARIANTS["default"],
               outdir=str(tmp_path), script_commit="x", mode=mode, extra={"_shared_cache": {}})


# ---- per-arm seeds and per-root commits ---------------------------------------------------------------
def _two_root_config(tmp_path, write_config):
    arms = {"geowalk": dict(arm_cfg("geowalk"), dir="${CR}/geowalk", cr_arm="geowalk"),
            "geowalk__s82": dict(arm_cfg("geowalk"), dir="${CRCTL}/geowalk__s82", cr_arm="geowalk__s82", seeds="82-84")}
    p = write_config(arms, tasks=("adip", "rano"), seeds="42-44")
    cfg = json.load(open(p))
    cfg["roots"] = {"CR": str(tmp_path / "cr"), "CRCTL": str(tmp_path / "ctl")}
    open(p, "w").write(json.dumps(cfg))
    return p


def _fill_two_roots(tmp_path, c_main=C_MAIN, c_ctl=C_CTL):
    for t in ("adip", "rano"):
        for s in (42, 43, 44):
            _write(tmp_path / "cr" / "geowalk", "geowalk", t, s, commit=c_main, cr_arm="geowalk")
        for s in (82, 83, 84):
            _write(tmp_path / "ctl" / "geowalk__s82", "geowalk", t, s, commit=c_ctl, cr_arm="geowalk__s82")


def test_arm_seed_range_outside_the_default_and_cli_subset(tmp_path, write_config):
    p = _two_root_config(tmp_path, write_config)
    cfg = load_config(p)
    assert cfg.arms["geowalk"].seeds == [42, 43, 44] and cfg.arms["geowalk__s82"].seeds == [82, 83, 84]
    assert cfg.arms["geowalk__s82"].commit_root == "CRCTL" and cfg.arms["geowalk"].commit_root == "CR"
    sub = load_config(p, seeds_subset=parse_seeds("42-43,82"))
    assert sub.arms["geowalk"].seeds == [42, 43] and sub.arms["geowalk__s82"].seeds == [82]
    only_main = load_config(p, seeds_subset=parse_seeds("42-43"))
    assert only_main.arms["geowalk__s82"].seeds == []
    _fill_two_roots(tmp_path)
    ds = load_all(only_main, {"CR": C_MAIN, "CRCTL": C_CTL}, log=lambda m: None)
    assert "geowalk__s82" in ds.absent and "subset" in ds.absent["geowalk__s82"]      # not an error, reported


def test_expected_commit_is_per_root(tmp_path, write_config):
    p = _two_root_config(tmp_path, write_config)
    _fill_two_roots(tmp_path)
    cfg = load_config(p)
    ds = load_all(cfg, {"CR": C_MAIN, "CRCTL": C_CTL}, log=lambda m: None)
    assert len(ds.arms["geowalk"].cells) == 6 and len(ds.arms["geowalk__s82"].cells) == 6
    assert ds.arms["geowalk__s82"].report["expected_commit"] == C_CTL
    assert expected_commit_for(cfg.arms["geowalk__s82"], {"*": C_MAIN, "CRCTL": C_CTL}) == C_CTL
    assert expected_commit_for(cfg.arms["geowalk"], {"*": C_MAIN, "CRCTL": C_CTL}) == C_MAIN
    with pytest.raises(LoaderError) as e:                       # one commit for both roots: the control files fail
        load_all(cfg, C_MAIN, log=lambda m: None)
    assert e.value.kinds() == {"provenance"} and "geowalk__s82" in str(e.value)
    with pytest.raises(LoaderError) as e:                       # swapped
        load_all(cfg, {"CR": C_CTL, "CRCTL": C_MAIN}, log=lambda m: None)
    assert len(e.value.problems) == 2
    with pytest.raises(LoaderError) as e:                       # no commit for a root
        load_all(cfg, {"CR": C_MAIN}, log=lambda m: None)
    assert "config" in e.value.kinds() and "CRCTL" in str(e.value)


def test_s82_arm_is_grouped_by_cr_arm_and_never_pooled(tmp_path, write_config):
    p = _two_root_config(tmp_path, write_config)
    _fill_two_roots(tmp_path)
    ds = load_all(load_config(p), {"CR": C_MAIN, "CRCTL": C_CTL}, log=lambda m: None)
    eng = Engine(ds)
    assert eng.joint(["geowalk", "geowalk__s82"]) == set()        # disjoint seeds: nothing to pool
    with pytest.raises(RuntimeError):
        controls.assert_not_pooled(["geowalk", "geowalk__s82"])
    controls.assert_not_pooled(["geowalk__s82", "random_search__s82"])
    # a base-arm file (cr_arm 'geowalk') dropped into the __s82 directory is refused
    _write(tmp_path / "ctl" / "geowalk__s82", "geowalk", "adip", 82, commit=C_CTL, cr_arm="geowalk")
    with pytest.raises(LoaderError) as e:
        load_all(load_config(p), {"CR": C_MAIN, "CRCTL": C_CTL}, log=lambda m: None)
    assert e.value.kinds() == {"content"} and "cr_arm" in str(e.value)


# ---- mechanical readings ------------------------------------------------------------------------------------
class _PR:
    def __init__(self, diffs):
        self.task_diffs = dict(zip(T11[:len(diffs)], diffs))
        v = np.asarray(diffs, dtype=float)
        self.n_tasks = v.size
        self.delta_taskmean = self.delta = float(v.mean())
        self.p_task = stats.wilcoxon_exact_p(v)


def _d(n_pos, mag_pos=0.02, mag_neg=0.001, n=11):
    return [mag_pos * (1 + 0.1 * i) for i in range(n_pos)] + [-mag_neg * (1 + 0.1 * i) for i in range(n - n_pos)]


def test_criterion_boundaries():
    c = controls.criterion(_PR(_d(8)).task_diffs, _PR(_d(8)).p_task)
    assert c["tasks"] == 8 and c["evaluable"] and c["met"] and c["p_task"] < 0.05
    c7 = controls.criterion(_PR(_d(7)).task_diffs, 0.001)
    assert c7["tasks"] == 7 and not c7["met"]                       # 7/11: not met whatever p
    weak = _d(8, mag_pos=0.001, mag_neg=0.02)                      # 8 positive tasks, three large negatives
    pw = _PR(weak)
    assert pw.p_task >= 0.05 and not controls.criterion(pw.task_diffs, pw.p_task)["met"]     # p decides
    ten = _PR(_d(8, n=10))
    c10 = controls.criterion(ten.task_diffs, ten.p_task)
    assert not c10["evaluable"] and not c10["met"]                  # fewer than 11 tasks: not evaluated
    neg = _PR([-x for x in _d(9)])
    assert controls.criterion(neg.task_diffs, neg.p_task, sign=-1)["met"]
    assert not controls.criterion(neg.task_diffs, neg.p_task, sign=+1)["met"]


def test_reading_c1_c2_every_branch():
    yes, no = _PR(_d(9)), _PR(_d(6))
    assert controls.reading_c1(yes)["verdict"].startswith("the shell advantage under candidate-set argmax does not")
    r = controls.reading_c1(no)
    assert not r["met"] and r["verdict"].startswith("part of the posted support effect reflects the input rescaling")
    assert controls.reading_c1(_PR(_d(8, n=5)))["verdict"].startswith("n/a")
    assert controls.reading_c2(no, no)["outcome"] == "radius"
    assert "glossed as the radius" in controls.reading_c2(no, no)["verdict"]
    both = controls.reading_c2(yes, yes)
    assert both["outcome"] == "isotropic" and "radius alone is not called sufficient" in both["verdict"]
    for a, b in ((yes, no), (no, yes)):
        mixed = controls.reading_c2(a, b)
        assert mixed["outcome"] == "mixed" and "does not cover this case" in mixed["verdict"]
    assert controls.reading_c2(yes, _PR(_d(8, n=3)))["outcome"] == "na"


def test_reading_c3_same_sign_as_reference():
    pos, pos2 = _PR(_d(9)), _PR(_d(10))
    ref = {"SupportArgmax": pos, "IntMax": pos2}
    assert controls.reading_c3(ref, {"SupportArgmax": pos2, "IntMax": pos})["verdict"].startswith("replicated")
    r = controls.reading_c3(ref, {"SupportArgmax": pos, "IntMax": _PR(_d(6))})
    assert not r["met"] and r["verdict"] == "not replicated on new seeds"
    flipped = _PR([-x for x in _d(10)])                            # significant, but the opposite sign
    assert not controls.reading_c3(ref, {"SupportArgmax": flipped, "IntMax": pos})["met"]
    refneg = {"SupportArgmax": flipped, "IntMax": pos}
    assert controls.reading_c3(refneg, {"SupportArgmax": _PR([-x for x in _d(9)]), "IntMax": pos})["met"]
    assert controls.reading_c3(ref, {"SupportArgmax": _PR(_d(4, n=4)), "IntMax": pos})["verdict"].startswith("n/a")


# ---- control tables on synthetic arms ---------------------------------------------------------------------------
BASE = {"random_search": 0.40, "random_box_faithful": 0.37, "hvarfner_scaled_gp": 0.44,
        "hvarfner_scaled_gp_box_argmax": 0.385, "hvarfner_scaled_gp_sphere_continuous": 0.39,
        "hvarfner_scaled_gp_box_faithful": 0.395}
CTL = {"hvarfner_scaled_gp_box_argmax_raw": 0.39, "hvarfner_scaled_gp_boxdir_argmax": 0.435, "random_boxdir": 0.398}


def _synthetic_control_dataset(tmp_path, write_config, levels_s82=None, seeds=(42, 43, 44, 45), seeds2=(82, 83, 84, 85)):
    rng = np.random.default_rng(3)
    arms = {}

    def fill(dirname, solver, lev, ss, root, commit, cr_arm):
        for t in T11:
            for s in ss:
                cs = [0.2 + 0.001 * i + 0.0001 * s for i in range(5)]
                tail = list(np.sort(rng.uniform(0.25, lev - 0.01, 24))) + [lev + 0.002 * rng.standard_normal()]
                _write(tmp_path / root / dirname, solver, t, s, commit=commit, cr_arm=cr_arm, scores=cs + tail)

    for solver, lev in BASE.items():
        fill(solver, solver, lev, seeds, "cr", C_MAIN, solver)
        arms[solver] = dict(arm_cfg(solver), dir="${CR}/" + solver, cr_arm=solver)
        lev2 = (levels_s82 or BASE)[solver]
        fill(solver + "__s82", solver, lev2, seeds2, "ctl", C_CTL, solver + "__s82")
        arms[solver + "__s82"] = dict(arm_cfg(solver), dir="${CRCTL}/" + solver + "__s82", cr_arm=solver + "__s82",
                                      seeds=f"{seeds2[0]}-{seeds2[-1]}")
    for solver, lev in CTL.items():
        fill(solver, solver, lev, seeds, "ctl", C_CTL, solver)
        arms[solver] = dict(arm_cfg(solver), dir="${CRCTL}/" + solver, cr_arm=solver)
    p = write_config(arms, tasks=T11, seeds=f"{seeds[0]}-{seeds[-1]}")
    cfg = json.load(open(p))
    cfg["roots"] = {"CR": str(tmp_path / "cr"), "CRCTL": str(tmp_path / "ctl")}
    open(p, "w").write(json.dumps(cfg))
    return load_all(load_config(p), {"CR": C_MAIN, "CRCTL": C_CTL}, log=lambda m: None)


def test_tab_controls_and_replication_print_the_prescribed_verdicts(tmp_path, write_config):
    ds = _synthetic_control_dataset(tmp_path, write_config)
    ctx = _ctx(ds, tmp_path)
    from crlib import decomp
    assert decomp.tab_decomposition_contrasts(ctx) is not None
    t = controls.tab_controls(ctx)
    rec = ctx.reg.records
    assert t is not None and rec["crCtlN"]["value"] == 44.0
    # planted: shell 0.44 vs box_raw 0.39 -> C1 met; boxdir 0.435 ~ shell (no advantage) -> C2 'radius' for the GP,
    # and uniform shell 0.40 vs boxdir 0.398 -> no clear advantage
    assert rec["crCtlCOneVerdict"]["value"] == "independent of the input rescaling"
    assert rec["crCtlCOneShellMinusBoxRawTasks"]["text"] == "11/11"
    assert rec["crCtlCTwoVerdict"]["value"] in ("the radius", "mixed (rule does not cover this case)",
                                               "isotropic directions at the encoder-mean radius")
    assert "Reading C1: the shell advantage under candidate-set argmax does not depend" in t.notes
    assert "Reading C2:" in t.notes and "fallback" in t.caption.lower()
    assert rec["crCtlArmBoxArgmaxRawFallbackCells"]["value"] == 0.0
    r = controls.tab_replication(ctx)
    assert r is not None and rec["crRepN"]["value"] == 44.0
    assert rec["crRepVerdict"]["value"] == "replicated on new seeds"
    assert rec["crRepSupportArgmaxSameSign"]["text"] == "11/11"
    assert rec["crRepLevelShellArgmax"]["files"]["count"] == 44
    assert "never pooled" in r.caption
    for name, x in rec.items():
        assert x["files"]["count"] > 0, name


def test_replication_not_replicated_when_the_new_seeds_disagree(tmp_path, write_config):
    lv = dict(BASE, hvarfner_scaled_gp=0.385, hvarfner_scaled_gp_box_argmax=0.44)       # support effect reversed
    ds = _synthetic_control_dataset(tmp_path, write_config, levels_s82=lv)
    ctx = _ctx(ds, tmp_path)
    from crlib import decomp
    decomp.tab_decomposition_contrasts(ctx)
    r = controls.tab_replication(ctx)
    assert ctx.reg.records["crRepVerdict"]["value"] == "not replicated on new seeds"
    assert "Reading C3: not replicated on new seeds" in r.notes
    assert ctx.reg.records["crRepSupportArgmaxSameSign"]["text"] == "0/11"


def test_control_tables_skip_without_the_arms(tmp_path, make_arm, write_config):
    make_arm("geowalk", "geowalk", [("adip", 42), ("rano", 42)])
    ds = load_all(load_config(write_config({"geowalk": arm_cfg("geowalk")}, seeds="42-42")), "abc1234", log=lambda m: None)
    ctx = _ctx(ds, tmp_path)
    assert controls.tab_controls(ctx) is None and controls.tab_replication(ctx) is None
    assert {s["table"] for s in ctx.skipped} == {"tab_controls", "tab_replication"}


# ---- seed-block permutation, shared-cold-start correlation, recovered fraction -----------------------------
def test_seed_block_permutation_matches_the_september_rule_on_a_complete_matrix():
    rng = np.random.default_rng(1)
    D = rng.normal(0.01, 0.03, size=(11, 40))
    cd = {(T11[i], 42 + j): D[i, j] for i in range(11) for j in range(40)}
    got = stats.seed_block_permutation(cd)
    # gate0_premise.seed_block_permutation with R = 1 (two-sided), re-implemented row by row
    def stat(e):
        e = e[e != 0]
        r = sps.rankdata(np.abs(e))
        return abs(r[e > 0].sum() - e.size * (e.size + 1) / 4.0)
    obs = stat(D.mean(1))
    signs = np.random.default_rng(0).choice(np.array([-1.0, 1.0]), size=(10_000, 40))
    E = (signs @ D.T) / 40
    cnt = sum(1 for row in E if stat(row) >= obs - 1e-12)
    assert got["p"] == pytest.approx((1 + cnt) / 10_001) and got["complete"]
    rb = stats.shared_coldstart_rbar(cd)
    ref = np.mean([np.corrcoef(D[i], D[j])[0, 1] for i in range(11) for j in range(i + 1, 11)])
    assert rb["rbar"] == pytest.approx(ref) and rb["n_pairs"] == 55


def test_seed_block_permutation_with_missing_cells_and_dependence():
    rng = np.random.default_rng(2)
    shared = rng.normal(0, 0.05, 40)                        # a seed effect shared by every task
    D = 0.004 + shared[None, :] + rng.normal(0, 0.005, size=(11, 40))
    cd = {(T11[i], 42 + j): D[i, j] for i in range(11) for j in range(40) if not (i == 3 and j < 5)}
    sb = stats.seed_block_permutation(cd)
    assert not sb["complete"] and 0 < sb["p"] <= 1
    assert stats.shared_coldstart_rbar(cd)["rbar"] > 0.8    # the planted dependence is seen
    td = {t: np.mean([v for (tt, _), v in cd.items() if tt == t]) for t in T11}
    assert sb["p"] > stats.wilcoxon_exact_p(list(td.values()))   # the task-level test overstates the evidence here


def test_ratio_bootstrap_uses_the_same_tasks_in_both_terms():
    num = {t: 0.010 + 0.001 * i for i, t in enumerate(T11)}
    den = {t: 2 * v for t, v in num.items()}
    r = stats.ratio_bootstrap(num, den)
    assert r["ratio"] == pytest.approx(0.5) and r["lo"] == pytest.approx(0.5) and r["hi"] == pytest.approx(0.5)
    den2 = {t: 0.012 + 0.002 * ((-1) ** i) for i, t in enumerate(T11)}
    r2 = stats.ratio_bootstrap(num, den2)
    idx = np.random.default_rng(stats.BOOT_SEED).integers(0, 11, size=(stats.BOOT_B, 11))
    a, b = np.asarray([num[t] for t in T11]), np.asarray([den2[t] for t in T11])
    ref = np.quantile(a[idx].mean(1) / b[idx].mean(1), [0.025, 0.975])
    assert (r2["lo"], r2["hi"]) == pytest.approx(tuple(ref)) and r2["den_nonpositive"] == 0
    assert stats.ratio_bootstrap(num, {t: -0.001 for t in T11})["den_nonpositive"] == stats.BOOT_B


# ---- Monte Carlo digits and the quadrature ------------------------------------------------------------------
def test_printed_digits_are_established_or_dropped():
    assert mc.stable_text(5.834858, 2.0e-5, 2) == ("5.83", 2)        # the corrected convex k = 3 value
    assert mc.stable_text(5.835660, 5.6e-4, 2) == ("5.8", 1)         # v2's estimate: 1.2 SE from the boundary
    assert mc.stable_text(5.315045, 1.0e-5, 2) == ("5.32", 2) and mc.stable_text(5.315045, 2.0e-5, 2)[1] == 1
    assert mc.stable_text(11.1250241, 1e-9, 2) == ("11.13", 2)
    assert mc.stable_text(7.9834, 0.0, 2) == ("7.98", 2)             # exact values keep every digit
    assert mc.stable_text(1.0504, 0.0003, 3)[1] == 2                 # SE above 0.2 of the last unit
    assert not mc.digits_established(0.7305, 1e-6, 3) and mc.digits_established(0.7310, 1e-6, 3)


def test_quadrature_against_closed_forms_and_monte_carlo():
    D = 256
    r = mc.gaussian_exact(np.zeros(D), np.eye(D))
    assert r["mean"] == pytest.approx(mc.prior_exact(D), abs=1e-9) and r["se"] < 1e-8
    assert mc.box_exact(np.array([-1.0]), np.array([2.0]))["mean"] == pytest.approx(5.0 / 6.0, abs=1e-10)
    sq = mc.box_exact(np.zeros(2), np.ones(2))
    assert sq["mean"] == pytest.approx((math.sqrt(2) + math.asinh(1)) / 3.0, abs=1e-10)
    rng = np.random.default_rng(0)
    lo = rng.normal(size=64) * 0.3 - 0.2
    hi = lo + np.abs(rng.normal(size=64)) * 0.5 + 1e-3
    ex, m = mc.box_exact(lo, hi), mc.box_mc(lo, hi, 200_000, np.random.default_rng(1))
    assert abs(m["mean"] - ex["mean"]) < 4 * m["se"]
    assert ex["sd"] == pytest.approx(m["sd"], rel=0.02)
    A = rng.normal(size=(8, 8)) * 0.4
    mu = rng.normal(size=8)
    exg, mg = mc.gaussian_exact(mu, A), mc.gaussian_mc(mu, A, 200_000, np.random.default_rng(2))
    assert abs(mg["mean"] - exg["mean"]) < 4 * mg["se"]
    with pytest.raises(RuntimeError):                                   # a wrong 'exact' value is caught by its check
        mc._check("x", {"mean": ex["mean"] + 0.05, "method": "exact"}, m)


def test_convex_control_variate_estimator():
    rng = np.random.default_rng(7)
    Z = rng.normal(size=(5, 3))
    G = Z @ Z.T
    w = (np.arange(100000) + 0.5) / 100000
    brute = np.mean([(np.linalg.norm(np.outer(w, Z[i]) + np.outer(1 - w, Z[j]), axis=1) ** 2).mean()
                     for i in range(5) for j in range(5) if i != j])
    assert mc.convex_second_moment(G, 2) == pytest.approx(brute, rel=1e-8)
    Z = rng.normal(size=(200, 12)) + 0.4
    G = Z @ Z.T
    for k in (2, 3, 16):                                              # Gram path (k <= 8) and direct path
        acc = mc._CVAccumulator(mc.convex_second_moment(G, k))
        mc.convex_cv_batch(Z, G, k, 200_000, np.random.default_rng(5), acc)
        r = acc.result()
        plain = mc.convex_mc(Z, k, 200_000, np.random.default_rng(6))
        assert abs(plain["mean"] - r["mean"]) < 4 * plain["se"]
        assert r["se"] < 0.5 * plain["se"]                           # the control variate pays
        assert abs(r["second_moment_mc"] - r["second_moment"]) < 4 * r["second_moment_mc_se"]
    out = mc.convex_cv(Z, G, 2, np.random.default_rng(9), rbar=1.0, draws=50_000, max_draws=400_000)
    assert out["draws"] % 50_000 == 0 and out["draws"] <= 400_000 and "digits_established" in out


def test_floor_null_prefix_property():
    """The first 100,000 replicates of a 1,000,000-replicate null are the 100,000-replicate null (floor.py relies
    on it to report the earlier rule-P verdicts next to the new ones)."""
    from crlib.floor import load_floor_module
    zrf = load_floor_module()
    rng = np.random.default_rng(0)
    values, cum, n = zrf._unique_cdf(np.round(rng.beta(2, 5, 5000), 3))
    Gn = zrf.max_cdf(values, cum, n, 30, "norepl")
    for nseed in (40, 39, 33):
        small = zrf.null_mean_draws(values, Gn, nseed, reps=100_000, seed=0)
        big = zrf.null_mean_draws(values, Gn, nseed, reps=230_000, seed=0)
        assert np.array_equal(big[:100_000], small)


# ---- MSR guard -------------------------------------------------------------------------------------------------
def _msr_ds(tmp_path, write_config, make_arm, **over):
    make_arm("papenmeier_msr_published", over.pop("file_solver", "papenmeier_msr_published"), [("adip", 42)])
    a = arm_cfg("papenmeier_msr_published", cr_arm=None)
    a.update(over)
    ds = load_all(load_config(write_config({"papenmeier_msr_published": a}, tasks=("adip",), seeds="42-42")),
                  "abc1234", log=lambda m: None)
    return ds


def test_msr_guard(tmp_path, write_config, make_arm):
    from crlib.tables import MsrGuardError, msr_guard
    d = tmp_path / "papenmeier_msr_published"
    d.mkdir()
    rec = ours_record("papenmeier_msr_published", "adip", 42)
    rec["provenance"]["cr_arm"] = "papenmeier_msr_published"
    (d / "ours_papenmeier_msr_published_amlodipine_mpo_s42_128d.json").write_text(json.dumps(rec))
    good = arm_cfg("papenmeier_msr_published", cr_arm="papenmeier_msr_published")
    cfgp = write_config({"papenmeier_msr_published": good}, tasks=("adip",), seeds="42-42")
    ctx = _ctx(load_all(load_config(cfgp), "abc1234", log=lambda m: None), tmp_path)
    msr_guard(ctx, "papenmeier_msr_published")                       # the production arm passes
    with pytest.raises(MsrGuardError):
        msr_guard(ctx, "papenmeier_msr_box_faithful")                # the row may not resolve to another arm
    # a camera-ready config whose MSR arm points at substitute files is refused
    d2 = tmp_path / "raasp"
    d2.mkdir()
    (d2 / "ours_papenmeier_msr_raasp_amlodipine_mpo_s42_128d.json").write_text(
        json.dumps(ours_record("papenmeier_msr_raasp", "adip", 42)))
    sub = dict(arm_cfg("papenmeier_msr_raasp", kind="stored"), dir="${T}/raasp",
               display={"label": "Papenmeier RAASP trust-region variant", "support": "trust region"})
    cfgs = write_config({"papenmeier_msr_published": sub}, tasks=("adip",), seeds="42-42", name="sub.json")
    ds2 = load_all(load_config(cfgs), None, log=lambda m: None)
    with pytest.raises(MsrGuardError) as e:
        msr_guard(_ctx(ds2, tmp_path, mode="cr"), "papenmeier_msr_published")
    assert "display override" in str(e.value) and "not a camera-ready arm" in str(e.value)
    msr_guard(_ctx(ds2, tmp_path, mode="legacy"), "papenmeier_msr_published")   # legacy: allowed, labelled by display


# ---- anytime metrics -------------------------------------------------------------------------------------------
def test_cell_anytime_definitions():
    class C:
        pass
    c = C()
    scores = [0.1, 0.2, 0.15, 0.05, 0.12] + [0.3, -1.0, 0.25] + [0.2] * 22
    best, tr = -9, []
    for v in scores:
        best = max(best, v)
        tr.append(best)
    c.trace, c.scores = np.asarray(tr), np.asarray(scores)
    a = extras.cell_anytime(c, fstar=0.5)
    assert a["auc"] == pytest.approx(0.3)                                    # best-so-far is 0.3 from evaluation 6 on
    assert a["ncsr"] == pytest.approx((0.5 - 0.3) / 0.5)
    assert a["perquery_raw"] == pytest.approx((0.3 - 1.0 + 0.25 + 0.2 * 22) / 25)
    assert a["perquery"] == pytest.approx((0.3 + 0.0 + 0.25 + 0.2 * 22) / 25)   # empty decode counts as 0


def test_new_tables_on_synthetic_arms(tmp_path, write_config):
    ds = _synthetic_control_dataset(tmp_path, write_config)
    ctx = _ctx(ds, tmp_path)
    t = extras.tab_decomposition_robustness(ctx)
    assert t is not None and len([r for r in t.rows if isinstance(r, list)]) == 9
    assert 0 < ctx.reg.records["crDecIntMaxPSeedBlock"]["value"] <= 1
    assert extras.tab_anytime(ctx) is None                                   # no GeoWalk arm: skipped, recorded
    assert any(s["table"] == "tab_anytime" for s in ctx.skipped)


# ---- end to end: a camera-ready run over synthetic files of every manifest arm --------------------------------
def test_end_to_end_camera_ready_run_on_synthetic_files(tmp_path, monkeypatch):
    """GEN v5: every manifest arm and the published-recipe arms of wave p get synthetic files; the camera-ready
    scope (RUN_RULES 2026-10-14) loads only the printed arms and builds only the printed tables."""
    import make_tables
    from crlib import scope
    from v5_fixtures import write_verdicts
    cfg = load_config(os.path.join(ANALYSIS, "sources_cr.json"), load_includes=False)
    roots = {"CR": tmp_path / "cr", "CRCPU": tmp_path / "crcpu", "CRCTL": tmp_path / "ctl", "CRPUB": tmp_path / "pub",
             "CRRD": tmp_path / "rd", "CRC4": tmp_path / "ctl"}      # CRC4: the control block's directory, own commit
    commit = {"CR": C_MAIN, "CRCPU": C_MAIN, "CRCTL": C_CTL, "CRPUB": C_PUB, "CRRD": C_RD, "CRC4": C_C4}
    rng = np.random.default_rng(11)
    stored, n_cr = [], 0
    for name, spec in cfg.arms.items():
        if spec.kind != "cr":
            stored.append(name)
            continue
        n_cr += 1
        root = spec.commit_root
        seeds = (82, 83) if name.endswith("__s82") else (42, 43)
        lev = 0.36 + 0.08 * rng.random()
        for t in T11:
            for s in seeds:
                cs = [0.2 + 0.001 * i + 0.0001 * s + 0.001 * T11.index(t) for i in range(5)]
                rec = ours_record(spec.solver_field, t, s, commit=commit[root],
                                  scores=cs + list(np.sort(rng.uniform(0.22, lev, 25))))
                rec["provenance"].update(cr_arm=spec.cr_arm, run_overrides=spec.expect.get("provenance.run_overrides", {}))
                if "subspace_dim" in spec.expect:
                    rec["subspace_dim"] = spec.expect["subspace_dim"]
                d = roots[root] / name
                d.mkdir(parents=True, exist_ok=True)
                (d / spec.pattern.format(task=TASKS_T11[t], seed=s)).write_text(json.dumps(rec))
    assert n_cr == 73 + 7 + 3          # 73 manifest arms (control C4 included), the 7 published arms, the three arms of
    #                                    the GuacaMol runs of 2026-11-02 (root CRRD)
    vpath = tmp_path / "verification" / "verdicts.json"
    write_verdicts(vpath, monkeypatch=monkeypatch)
    out = tmp_path / "out"
    argv = ["--outdir", str(out), "--sources", os.path.join(ANALYSIS, "sources_cr.json"),
            "--expected-commit", f"CR={C_MAIN}", "--expected-commit", f"CRCPU={C_MAIN}",
            "--expected-commit", f"CRCTL={C_CTL}", "--expected-commit", f"CRPUB={C_PUB}",
            "--expected-commit", f"CRRD={C_RD}", "--expected-commit", f"CRC4={C_C4}", "--seeds", "42-43,82-83",
            "--inputs-root", str(tmp_path / "no_inputs"), "--geometry-root", str(tmp_path / "no_geometry"),
            "--verdicts", str(vpath),
            "--allow-missing-arms", ",".join(a for a in stored if a not in scope.NOT_LOADED_ARMS),
            "--variants", "default,clean_nofallback", "--no-figures", "--validation-run"]
    for k, v in roots.items():
        argv += ["--root", f"{k}={v}"]
    if shutil.which("pdflatex"):
        argv.append("--latex-check")
    rc = make_tables.main(argv)
    assert rc == 0
    man = json.load(open(out / "run_manifest.json"))
    written = set(man["outputs"]["default"])
    for tab in ("tab_main.tex", "tab_boxsphere.tex", "tab_published.tex", "tab_verification.tex",
                "tab_trajectories.tex", "tab_baselines_overview.tex", "tab_hyper.tex",
                # RUN_RULES 2026-10-24: the cards and the run status are repository records, with their Markdown
                "records/tab_arm_status.tex", "records/tab_baselines.tex", "records/BASELINE_CARDS.md",
                "records/RUN_STATUS.md", "records/SOURCE.json",
                # RUN_RULES 2026-10-29: the full Table 1 is a repository record, as LaTeX (old bytes) and Markdown
                "records/tab_main_full.tex", "records/tab_main_full_notes.tex", "records/TABLE1_FULL.md"):
        assert tab in written, tab
    assert "tab_baselines.tex" not in written and "tab_arm_status.tex" not in written
    from test_appendix_records import check_source_json
    src = check_source_json(str(out / "records"))
    assert set(src["outputs_sha256"]) == {"BASELINE_CARDS.md", "RUN_STATUS.md", "TABLE1_FULL.md", "tab_main_full.tex",
                                          "tab_main_full_notes.tex", "tab_convex_full.tex", "tab_boxsphere_full.tex",
                                          "tab_crossdomain_full.tex"}
    from test_appendix_records import markdown_tex_residue
    for md in ("BASELINE_CARDS.md", "RUN_STATUS.md", "TABLE1_FULL.md"):
        txt = open(out / "records" / md).read()
        assert not re.search(r"\\[A-Za-z]", txt) and markdown_tex_residue(txt) == [], md      # no TeX left
    for tab in ("tab_decomposition_contrasts.tex", "tab_controls.tex", "tab_replication.tex",
                "tab_decomposition_robustness.tex", "tab_anytime.tex", "tab_linear_bo.tex", "tab_equivalence.tex",
                "tab_factorial_details.tex", "tab_box_sensitivity.tex", "tab_gp_radius.tex",
                # RUN_RULES 2026-10-29: merged into tab_radius / tab_convex, or repository records
                "tab_estimators.tex", "tab_shells.tex", "tab_pca_directions.tex", "tab_compute.tex",
                "tab_radius_pertask.tex"):
        assert tab not in written, tab
    assert set(man["scope"]["arms_not_loaded"]) == {a for a in scope.NOT_LOADED_ARMS if a in cfg.arms}
    # RUN_RULES 2026-10-19 second set (c), review B3: every printed row label and card title carries the printed
    # label; an author's or method's name only inside a verified recipe's name, never a code name
    from crlib import published as pub
    from v5_fixtures import label_problems, printed_labels
    labs = printed_labels(out)
    assert {f for f, _ in labs} >= {"tab_main.tex", "tab_boxsphere.tex", "tab_published.tex",
                                     "records/tab_arm_status.tex", "records/tab_baselines.tex", "tab_trajectories.tex",
                                     "tab_baselines_overview.tex", "tab_hyper.tex"}
    assert sum(f == "records/tab_baselines.tex" for f, _ in labs) > 40
    assert sum(f == "records/tab_arm_status.tex" for f, _ in labs) > 50
    # 24 rows since RUN_RULES 2026-11-06 21:30 (the TuRBO-1 re-implementation is no longer a row of the overview)
    assert 24 <= sum(f == "tab_baselines_overview.tex" for f, _ in labs) <= 32
    verified_names = [pub.RECIPES[r]["name"] for r in ("hvarfner", "msr", "mle_scaled", "xu", "doumont")]
    # RUN_RULES 2026-11-01 (2): the printed Table 1 carries the method names of the submitted version (the caption says
    # that rows without the mark of a published recipe adapt the cited method); every other table keeps the rule
    assert label_problems([(f, lab) for f, lab in labs if f != "tab_main.tex"], verified_names) == []
    t1 = [lab for f, lab in labs if f == "tab_main.tex"]
    assert any(lab.startswith("TuRBO") for lab in t1) and all("$^{\\S}$" in lab for lab in t1
                                                             if lab.startswith(("Papenmeier MSR", "Xu Standard GP")))
    status = [lab for f, lab in labs if f == "records/tab_arm_status.tex"]
    md_rows = [x for x in open(out / "records" / "RUN_STATUS.md").read().splitlines() if x.startswith("| ")]
    assert len(md_rows) == 1 + len(status)               # header + one row per method of the LaTeX record
    assert len(status) == len(set(status))                      # every arm keeps a distinct printed label
    # plain config labels are escaped (a raw % would comment out the rest of the row, a raw ~ is a space)
    assert "GeoWalk shell +-10\\%" in status and "GeoCMA D\\textasciitilde{}=8" in status
    assert not any(("%" in lab and "\\%" not in lab) or "~" in lab for lab in status)
    # review S3: the success-rule arm is described by its constants, never as a 1/5 rule (clarification 2026-10-12)
    sens = [lab for f, lab in labs if f == "tab_sensitivity.tex"]
    assert any("start $0.30$" in x and "clamp $[0.05,0.50]$" in x for x in sens) and not any("1/5" in x for x in sens)
    assert any("start 0.30, x1.2 / x0.85, clamp [0.05, 0.50]" in x for x in status)
    cards = open(out / "records" / "tab_baselines.tex").read()
    assert cards.count("Code name") == cards.count("\\hspace{0pt}\\textbf{")     # one code-name field per card
    nums = json.load(open(out / "numbers_dossier.json"))["numbers"]
    assert "crRepVerdict" not in nums and "crCtlN" not in nums and "crDecN" not in nums
    assert nums["crFactorialInteractionReading"]["text"] == nums["crFactorialInteractionReading"]["value"]
    assert nums["crFactorialN"]["value"] == 22.0 and nums["crPubN"]["value"] == 22.0
    assert nums["crPubFamilySize"]["value"] == 12.0
    assert nums["crMainVanillaSphereReading"]["text"] == nums["crMainVanillaSphereReading"]["value"]
    assert nums["crPubHvarfnerChosen"]["value"] in ("hvarfner_published", "hvarfner_published_allobs")
    assert nums["crVerRecipesVerified"]["value"] == 5.0
    # the rules are applied as written: BH within the five factorial contrasts and within the 12 baseline tests,
    # readings chosen by the BH-corrected task-level p, the Hvarfner arm with the higher mean (rev. 1 (g))
    five = ["SupportCand", "SupportCont", "AcqShell", "AcqBox", "Interaction"]
    assert [nums[f"crFactorial{k}PTaskBH"]["value"] for k in five] == pytest.approx(
        stats.bh([nums[f"crFactorial{k}PTask"]["value"] for k in five]))
    pub = sorted(k[:-len("PTaskBH")] for k in nums if k.startswith("crPub") and k.endswith("PTaskBH"))
    assert len(pub) == 12
    assert [nums[k + "PTaskBH"]["value"] for k in pub] == pytest.approx(stats.bh([nums[k + "PTask"]["value"] for k in pub]))
    for k in pub:
        q, dl, word = nums[k + "PTaskBH"]["value"], nums[k + "Delta"]["value"], nums[k + "Verdict"]["value"]
        assert word == ("better" if q < 0.05 and dl < 0 else "worse" if q < 0.05 and dl > 0 else "not worse"), k
        assert "$p_{\\mathrm{BH}}=" + nums[k + "PTaskBH"]["text"] + "$" in nums[k + "Sentence"]["value"], k
    pt = nums["crMainVanillaSpherePTask"]["value"]
    assert ("no detected difference" in nums["crMainVanillaSphereReading"]["value"]) == (pt >= 0.05)
    for k in five:
        assert ("significant" in nums[f"crFactorial{k}Reading"]["value"]) == (nums[f"crFactorial{k}PTaskBH"]["value"] < 0.05)
    # RUN_RULES 2026-10-19 (b), (c): BH within the three recipe rows of Table 2 on the task-level p, and the
    # "smaller margin" sentence iff the shell leads significantly and by less than in the submitted Table 2
    rows = ["VanillaBO", "Papenmeier", "XuStandardGP"]
    assert [nums[f"crBox{k}PTaskBH"]["value"] for k in rows] == pytest.approx(
        stats.bh([nums[f"crBox{k}PTask"]["value"] for k in rows]))
    assert nums["crBoxPapenmeierSubmittedDelta"]["value"] == pytest.approx(0.0355)
    dl, q = nums["crBoxPapenmeierDelta"]["value"], nums["crBoxPapenmeierPTaskBH"]["value"]
    holds = dl > 0 and q < 0.05 and dl < 0.0355
    assert nums["crBoxPapenmeierSmallerMarginHolds"]["value"] == ("yes" if holds else "no")
    assert nums["crBoxPapenmeierSmallerMargin"]["text"] == (
        "the sphere-supported version still outperforms the corrected box baseline, but by a smaller margin"
        if holds else "")
    # running-text macros are written in text mode, so a sentence can break; the LaTeX check sets each of them
    # in a paragraph (overfull_hbox below stays empty only if they can)
    tex = open(out / "numbers.tex").read()
    for k in ("crBoxPapenmeierSmallerMargin", "crMainVanillaSphereReading", "crFactorialInteractionReading",
              "crPubMsrVsGeowalkSentence", "crVerMsrNFourZeroPass"):
        assert "\\newcommand{\\" + k + "}{" + nums[k]["text"] + "}\n" in tex, k
    assert "records/tab_changes_vs_submitted.tex" in written or "tab_changes_vs_submitted.tex" not in written
    hi = "hvarfner_published" if nums["crPubHvarfnerInboxMean"]["value"] >= nums["crPubHvarfnerAllobsMean"]["value"] \
        else "hvarfner_published_allobs"
    assert nums["crPubHvarfnerChosen"]["value"] == hi
    # second set (e), review S6: one choice for the run, recorded identically in every variant's dossier
    vnums = json.load(open(out / "variants" / "clean_nofallback" / "numbers_dossier.json"))["numbers"]
    assert vnums["crPubHvarfnerChosen"]["value"] == hi and "chosen once on the default" in vnums["crPubHvarfnerChosen"]["rule"]
    # second set (c): the printed text of the choice is the arm's descriptive label, the code name only its value
    from crlib import labels as LB
    assert nums["crPubHvarfnerChosen"]["text"] == LB.label(hi) and "\\_" not in nums["crPubHvarfnerChosen"]["text"]
    # review M9: the 440-cell check is loud and reaches the manifest (2 seeds here, so every arm is short)
    gc = man["guacamol_cells_check"]
    assert gc["arms_checked"] > 50 and len(gc["not_440_complete"]) == gc["arms_checked"]
    assert gc["not_440_complete"]["geowalk"] == "22 of 22"
    assert any("do not have 440 completed cells" in w for w in man["loud_warnings"])
    rep = json.load(open(out / "loader_report.json"))
    assert "random_search__s82" not in rep["arms"] and "hvarfner_scaled_gp_sphere05" not in rep["arms"]
    assert rep["arms"]["xu_published"]["expected_commit"] == C_PUB
    assert rep["arms"]["random_search"]["expected_commit"] == C_MAIN
    if man["latex_check"]:
        assert man["latex_check"]["ok"] and man["latex_check"]["overfull_hbox"] == []
        check = open(out / "_latex_check" / "check.tex").read()
        assert "\\crBoxPapenmeierSmallerMargin;" in check and "\\crMainVanillaSphereReading;" in check
    # the published files carry their own tag's commit: the main campaign's commit for that root fails
    rc2 = make_tables.main([a if not a.startswith("CRPUB=" + C_PUB) else f"CRPUB={C_MAIN}" for a in argv]
                           + ["--outdir", str(tmp_path / "out2")])
    assert rc2 == 2
    # RUN_RULES 2026-10-22 (generator edits after the final run: format, labels and wording only)
    import glob
    import re as _re
    printed = {os.path.basename(f): open(f).read() for f in glob.glob(str(out / "tab_*.tex"))}
    for f, txt in printed.items():
        for word in ("run rules", "camera-ready", "maximis", "normalis", "initialis", "optimis",
                     "relabelled", "favour", "not worse", "sidecar"):
            assert word not in txt, (f, word)
    for f, txt in list(printed.items()) + [("numbers.tex", tex)]:
        assert not _re.search(r"(?<![0-9])[-+]0\.0+(?![0-9])", txt), f       # no signed zero (N6)
    tm = printed["tab_main.tex"].splitlines()
    head = tm[tm.index("\\toprule") + 1]
    assert "\\begin{tabularx}{\\linewidth}" in printed["tab_main.tex"] and "\\resizebox" not in head
    assert "p_{\\mathrm{task}}" in head and "(cells)" not in head      # the printed p is the task-level test
    # RUN_RULES 2026-11-01: the printed Table 1 is a view (Method, Cross-mean, Delta, W/L, p_task, d) of the full table
    # in the layout of the submitted version; the full table keeps every row and column in the records
    full = open(out / "records" / "tab_main_full.tex").read()
    fh = full.splitlines()[full.splitlines().index("\\toprule") + 1]
    assert "(cells)" in fh and "W/L" in fh and "W/L" in head and "$n$" not in head
    # authors' decision of 2026-11-02: no LOL-BO and InvBO rows, and CMA-ES, TuRBO and BAxUS print runs of this version
    # (no dagger; BAxUS covers part of the matched set when one of its runs did not complete)
    for gone in ("LOL-BO", "InvBO", "\\dagger"):
        assert gone not in printed["tab_main.tex"], gone
    for lab in ("CMA-ES~\\citep{hansen2001cmaes,hansen2019pycma} &", "TuRBO$^{\\S}$~\\citep{eriksson2019turbo} &",
                "BAxUS~\\citep{papenmeier2022baxus} &", "Linear BO$^{\\S}$~\\citep{doumont2026still}",
                "Random Search (shell) &", "All other comparators propose off the shell."):
        assert lab in printed["tab_main.tex"], lab
    # authors' decision of 2026-11-06 (RUN_RULES 2026-11-06): the GP on the shell is not a printed row of Table 1; it is
    # printed in Table 2, and the notes to Table 1 name its implementation
    assert "GP on the shell~\\citep{hvarfner2024vanilla}" not in printed["tab_main.tex"]
    assert "The GP on the shell of Section~\\ref{sec:main_results}, which Table~\\ref{tab:boxsphere} prints" in printed["tab_main_notes.tex"]
    assert "candidate-set argmax''" in printed["tab_main_notes.tex"]
    # authors' decision of 2026-11-04 (RUN_RULES 2026-11-04 15:30): the Linear BO port on the shell is printed only in
    # the full table, and the caption no longer states the partial coverage of a row (the notes and Section 3.1 do)
    assert "Linear BO (shell)" not in printed["tab_main.tex"] and "calibrated bounds" in full
    assert " row covers $n=" not in printed["tab_main.tex"]
    assert "COWBOYS" not in printed["tab_main.tex"] and "Tanimoto" not in printed["tab_main.tex"]
    assert "Rows marked $^\\S$ run the authors' released code" in printed["tab_main.tex"]
    for lab in ("($\\ell_0=\\sqrt{D}$), qLogEI, candidate-set argmax", "($\\ell_0=0.1\\sqrt{D}$), qLogEI, L-BFGS-B"):
        assert lab in full and lab not in printed["tab_main.tex"], lab
    assert "\\label{tab:main}" in printed["tab_main.tex"]
    assert "The full version of this table is released with the code" in printed["tab_main_notes.tex"]
    assert "docs/baselines/TABLE1\\_FULL.md" in printed["tab_main_notes.tex"]
    md = open(out / "records" / "TABLE1_FULL.md").read()
    assert md.count("| yes |") + md.count("| no |") == sum(1 for x in full.splitlines() if x.endswith("\\\\")
                                                             and "&" in x and not x.startswith("\\multicolumn")) - 1
    # Table 1: a float page with a short caption; the definitions, notes and glossary are a generated text block
    notes = printed["tab_main_notes.tex"]
    assert "\\begin{table}[!t]" in printed["tab_main.tex"] and "Abbreviations" not in printed["tab_main.tex"]
    assert "on one matched set of $n=" + nums["crMainMatchedN"]["text"] + "$ (task, seed) cells" in printed["tab_main.tex"]
    assert notes.splitlines()[2].startswith("\\paragraph{Notes to Table~\\ref{tab:main}.} All rows are evaluated")
    assert "Abbreviations: DSP prior" in notes and "is a tie" in notes and "\\begin{table}" not in notes
    assert "is a tie" in open(out / "records" / "tab_main_full_notes.tex").read()
    assert ("Box rows use the $[p_{01},p_{99}]$ box, except Linear BO, which runs in $[-8,8]^D$. "
            "Table~\\ref{tab:baselines} gives the support of every row.") in notes
    assert "stored run (Section" not in notes and "10-seed" not in notes
    assert "tab_main_notes.tex" in written
    # generated italic subheaders set the small-caps method names upright (no T1/ptm/m/scit substitution)
    for f, txt in printed.items():
        assert not _re.search(r"\\textit\{[^{}]*\\geo(walk|cma)\{\}", txt), f
    assert "\\begin{longtable}" in open(out / "records" / "tab_arm_status.tex").read()
    # the overview keeps the label of the card table, so the cross-references of the paper resolve; the Table 1 notes
    # point to it and to the cards of the repository
    ov = printed["tab_baselines_overview.tex"]
    assert "\\label{tab:baselines}" in ov and "\\begin{longtable}" in ov and "docs/baselines/" in ov
    assert "ARD = one lengthscale per input dimension. MAP = maximum a posteriori fit" in ov and "Tanimoto" not in ov
    assert "RAASP = random axis-aligned subspace perturbations" in ov and "seeds" not in ov.split("caption{")[1][:2000]
    assert "summarized in Table~\\ref{tab:baselines}" in notes and "docs/baselines/" in notes
    assert "\\label{tab:hyper}" in printed["tab_hyper.tex"]
    # W8: the means of Table 2 carry no bold; bold only where the caption defines it
    rows2 = [l for l in printed["tab_boxsphere.tex"].splitlines() if l.count("&") == 4 and ", shell &" in l]
    assert rows2 and not any("mathbf" in "&".join(l.split("&")[2:4]) for l in rows2)
    assert "Bold, in both blocks" in printed["tab_boxsphere.tex"]
    # N10: Table 3 prints the task-level columns from the values the generator already registers
    # RUN_RULES 2026-11-01: Tables 2 and 3 in the layout of the submitted version; the full tables stay in the records
    assert "Candidate support" in printed["tab_boxsphere.tex"] and "95\\,\\% CI" not in printed["tab_boxsphere.tex"]
    for rec in ("tab_boxsphere_full.tex", "tab_crossdomain_full.tex"):
        assert "(cells)" in open(out / "records" / rec).read(), rec
    assert "Dir." in printed.get("tab_crossdomain.tex", "Dir.") and "(cells)" not in printed.get("tab_crossdomain.tex", "")
    if "tab_crossdomain.tex" in printed:
        assert "$p_{\\mathrm{task}}$" in printed["tab_crossdomain.tex"]
        assert "$" + nums["crCrossGuacamolPTask"]["text"] + "$" in printed["tab_crossdomain.tex"]
    # count-only task macros
    for k, r in nums.items():
        if "Tasks" in k and isinstance(r["text"], str) and _re.fullmatch(r"\d+/\d+", r["text"]):
            assert nums[k + "K"]["text"] == r["text"].split("/")[0], k
    assert nums["crNTasksGuacamol"]["value"] == 11.0
    # tab_published: one table, the two references side by side (RUN_RULES 2026-10-29)
    pubt = printed["tab_published.tex"]
    assert "Reference: GP on the shell" in pubt and "\\cmidrule" in pubt and pubt.count("\\begin{tabular") == 1
    assert pubt.count("MSR (Papenmeier et al.) &") == 1
    # RUN_RULES 2026-10-29 (2): the Appendix I tables with the same unit are blocks of one table; each block keeps
    # its own family, and the merged tables keep their numbers
    rad = printed["tab_radius.tex"]
    assert "Radius of \\geowalk{} proposals" in rad and "Radius estimator" in rad and "Thick shells" in rad
    assert "Against the fixed radius" not in rad and "crShellPctVsREightPBH" in nums and "crEstMedianPBH" in nums
    assert "\\label{tab:estimators}" not in "".join(printed.values()) and "Direction controls" in printed["tab_convex.tex"]
    for lab in ("tab:estimators", "tab:shells", "tab:radius_pertask", "tab:pca_directions", "tab:compute"):
        assert "\\ref{" + lab + "}" not in "".join(printed.values()), lab
    # RUN_RULES 2026-10-29 (3): the Linear BO trajectory row and its three levels
    assert "Linear BO (Doumont et al.)" in printed["tab_trajectories.tex"]
    assert {"crTrajPubLinearNOneZero", "crTrajPubLinearNTwoZero", "crTrajPubLinearNThreeZero"} <= set(nums)
    assert not any("frozen noise" in x for x in printed["tab_verification.tex"].splitlines() if x.endswith("\\\\"))
    assert "Read from" not in printed["tab_hyper.tex"] and "Uniform" in printed["tab_verification.tex"]
    # second pass of 2026-10-29: no caption sentence starts with "Its"; the Table 1 caption has no semicolon
    for f, txt in printed.items():
        assert ". Its " not in txt and "{Its " not in txt, f
    assert ";" not in printed["tab_main.tex"].split("\\caption{")[1].split("\\label")[0]
