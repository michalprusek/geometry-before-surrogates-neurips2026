"""GEN v2: W/L tie tolerance, task bootstrap, TOST, file-set unions, queue sidecars, cr_arm/expect checks,
Monte Carlo norms, and the decomposition contrasts on a synthetic dataset."""
import hashlib
import json
import math

import numpy as np
import pytest
from scipy import stats as sps

from conftest import arm_cfg, ours_record
from crlib import TASKS_T11, mc, stats
from crlib.config import load_config
from crlib.context import VARIANTS, Ctx
from crlib.engine import Engine, fileset_from_rows, fileset_union
from crlib.loader import LoaderError, load_all
from crlib.numbers import Registry

GRID = [(t, s) for t in ("adip", "rano") for s in (42, 43, 44)]


def _load(path, commit="abc1234", **kw):
    return load_all(load_config(path), commit, kw.pop("allow", None), log=lambda m: None, **kw)


# ---- W/L tie tolerance ------------------------------------------------------------------------
def test_wl_tie_tolerance_and_exact_counts():
    cells = [("adip", s) for s in range(42, 50)]
    b = {c: 0.5 for c in cells}
    diffs = [0.01, -0.02, 2.6e-8, -1.5e-9, 1e-16, 0.0, 0.03, 5e-7]
    a = {c: 0.5 + d for c, d in zip(cells, diffs)}
    pr = stats.paired(a, b, cells)
    assert (pr.wins, pr.losses, pr.ties) == (2, 1, 5)            # |d| <= 1e-6 are ties
    assert (pr.wins_exact, pr.losses_exact, pr.ties_exact) == (5, 2, 1)
    d = np.asarray([a[c] - b[c] for c in cells])
    assert pr.p_exact == pytest.approx(sps.wilcoxon(d).pvalue, rel=1e-12)   # the earlier rule: exact differences
    dz = np.where(np.abs(d) <= 1e-6, 0.0, d)                                # GEN v3: ties are zero differences
    assert pr.p == pytest.approx(sps.wilcoxon(dz).pvalue, rel=1e-12)
    assert pr.p != pr.p_exact


# ---- task bootstrap -------------------------------------------------------------------------------
def test_task_bootstrap_is_deterministic_and_matches_hand_rule():
    td = {t: v for t, v in zip(sorted(TASKS_T11), np.linspace(-0.01, 0.03, 11))}
    lo, hi = stats.task_bootstrap_ci(td)
    assert (lo, hi) == stats.task_bootstrap_ci(td)
    v = np.asarray([td[k] for k in sorted(td)])
    rng = np.random.default_rng(stats.BOOT_SEED)
    idx = rng.integers(0, 11, size=(stats.BOOT_B, 11))
    ref = np.quantile(v[idx].mean(1), [0.025, 0.975])
    assert lo == pytest.approx(ref[0]) and hi == pytest.approx(ref[1])
    assert lo < v.mean() < hi
    lo90, hi90 = stats.task_bootstrap_ci(td, level=0.90)
    assert lo < lo90 < hi90 < hi


def test_task_bootstrap_flat_weights():
    td = {"adip": 0.0, "rano": 1.0}
    lo, hi = stats.task_bootstrap_ci(td, {"adip": 99, "rano": 1}, level=0.5)
    assert hi <= 0.5          # cell-weighted: the 99-cell task dominates every resample that contains it


# ---- TOST ----------------------------------------------------------------------------------------------
def test_tost_equivalent_and_not():
    tasks = sorted(TASKS_T11)
    small = {t: 0.001 * ((-1) ** i) for i, t in enumerate(tasks)}
    T = stats.tost(small, 0.01)
    assert T["equivalent"] and T["p_t"] < 0.05 and T["p_w"] < 0.05
    big = {t: 0.02 + 0.001 * i for i, t in enumerate(tasks)}
    T2 = stats.tost(big, 0.01)
    assert not T2["equivalent"] and T2["p_t"] > 0.5
    v = np.asarray([small[t] for t in tasks])
    se = v.std(ddof=1) / math.sqrt(v.size)
    p_lo = sps.t.sf((v.mean() + 0.01) / se, 10)
    p_hi = sps.t.cdf((v.mean() - 0.01) / se, 10)
    assert T["p_t"] == pytest.approx(max(p_lo, p_hi))
    assert T["p_w"] == pytest.approx(1 / 2048)        # every d + delta > 0 and every d - delta < 0
    one = stats.tost({"rano": 0.001}, 0.01)
    assert one["verdict"] == "n/a" and not one["equivalent"]   # no verdict from a single task


# ---- file sets -----------------------------------------------------------------------------------------
def test_fileset_serialisation_and_union():
    rows1 = [("/a/x.json", "1" * 64), ("/a/y.json", "2" * 64)]
    rows2 = [("/a/y.json", "2" * 64), ("/b/z.npy", "3" * 64)]
    f1, f2 = fileset_from_rows(rows1, ["x"]), fileset_from_rows(rows2, ["z"])
    u = fileset_union(f1, f2)
    assert u["count"] == 3 and u["arms"] == ["x", "z"]
    paths = sorted({r[0] for r in rows1 + rows2})
    assert u["list_sha256"] == hashlib.sha256("\n".join(paths).encode()).hexdigest()
    rows = sorted(set(rows1 + rows2))
    assert u["content_sha256"] == hashlib.sha256("\n".join(f"{p}\t{s}" for p, s in rows).encode()).hexdigest()


# ---- queue sidecars ---------------------------------------------------------------------------------
def _side(d, t, s, solver, suffix, payload=None):
    name = f"ours_{solver}_{TASKS_T11[t]}_s{s}_128d.json{suffix}"
    (d / name).write_text(json.dumps(payload or {"kind": "hard_crash", "returncode": -9}))


def test_crash_sidecar_without_result_is_a_crashed_cell(tmp_path, make_arm, write_config):
    make_arm("geowalk", "geowalk", [c for c in GRID if c != ("rano", 43)])
    d = tmp_path / "geowalk"
    _side(d, "rano", 43, "geowalk", ".crash.json", {"kind": "timeout", "returncode": None})
    _side(d, "adip", 42, "geowalk", ".crash.json", {"kind": "crashed_in_solver"})   # result exists too
    _side(d, "adip", 43, "geowalk", ".collision-1234.json")
    (d / ".ours_geowalk_amlodipine_mpo_s44_128d.json.abc.tmp").write_text("{}")
    (d / "attempts").mkdir()
    ds = _load(write_config({"geowalk": arm_cfg("geowalk")}))
    a = ds.arms["geowalk"]
    assert ("rano", 43) in a.sidecar_crashes and a.sidecar_crashes[("rano", 43)]["kind"] == "timeout"
    assert a.n_not_completed() == 1 and ("rano", 43) not in a.eligible("completed")
    assert a.report["missing_cells"] == [] and a.report["crash_sidecar_with_result"] == ["adip:42"]
    assert a.report["sidecars"]["collision"] == 1 and a.report["sidecars"]["tmp"] == 1
    assert a.report["sidecars"]["attempts_dir"] is True
    assert a.report["excluded"]["suffix_variants"] == {}            # sidecars are not counted as suffix files
    eng = Engine(ds)
    fs = eng.fileset_all("geowalk")
    assert fs["count"] == 6                                           # 5 results + 1 crash record


def test_interrupted_or_running_cell_is_fatal_unless_declared(tmp_path, make_arm, write_config):
    make_arm("geowalk", "geowalk", [c for c in GRID if c != ("rano", 43)])
    _side(tmp_path / "geowalk", "rano", 43, "geowalk", ".interrupted.json")
    with pytest.raises(LoaderError) as e:
        _load(write_config({"geowalk": arm_cfg("geowalk")}))
    assert "interrupted: resume or finish the queue" in str(e.value)
    ds = _load(write_config({"geowalk": arm_cfg("geowalk", known_missing=["rano:43"])}))
    assert ds.arms["geowalk"].report["unresolved_cells"] == ["rano:43=interrupted"]


def test_declared_missing_but_crashed_is_stale(tmp_path, make_arm, write_config):
    make_arm("geowalk", "geowalk", [c for c in GRID if c != ("rano", 43)])
    _side(tmp_path / "geowalk", "rano", 43, "geowalk", ".crash.json")
    with pytest.raises(LoaderError) as e:
        _load(write_config({"geowalk": arm_cfg("geowalk", known_missing=["rano:43"])}))
    assert "crash sidecar" in str(e.value)


def test_cr_arm_and_expect_checks(tmp_path, make_arm, write_config):
    d = tmp_path / "cma_es_subspace_sd8"
    d.mkdir()
    for t, s in GRID:
        rec = ours_record("cma_es_subspace", t, s)
        rec["subspace_dim"] = 8
        rec["provenance"].update(cr_arm="cma_es_subspace_sd8", run_overrides={"subspace_dim": 8})
        (d / f"ours_cma_es_subspace_{TASKS_T11[t]}_s{s}_128d_sd8.json").write_text(json.dumps(rec))
    cfg = arm_cfg("cma_es_subspace", dir="${T}/cma_es_subspace_sd8",
                  pattern="ours_cma_es_subspace_{task}_s{seed}_128d_sd8.json", cr_arm="cma_es_subspace_sd8",
                  expect={"subspace_dim": 8, "provenance.run_overrides": {"subspace_dim": 8}})
    ds = _load(write_config({"cma_es_subspace_sd8": cfg}))
    assert len(ds.arms["cma_es_subspace_sd8"].cells) == 6
    bad = dict(cfg, cr_arm="cma_es_subspace")
    with pytest.raises(LoaderError) as e:
        _load(write_config({"cma_es_subspace_sd8": bad}))
    assert e.value.kinds() == {"content"} and "cr_arm" in str(e.value)
    bad2 = dict(cfg, expect={"subspace_dim": 64})
    with pytest.raises(LoaderError):
        _load(write_config({"cma_es_subspace_sd8": bad2}))


# ---- Monte Carlo -------------------------------------------------------------------------------------
def test_distinct_rows_and_box_mc():
    rng = np.random.default_rng(1)
    idx = mc._distinct_rows(rng, 3000, 64, 5000)       # the convex k = 64 case (P(no repeat) ~ 0.5 per row)
    assert all(len(set(r)) == 64 for r in idx)
    small = mc._distinct_rows(rng, 100, 10, 20000)
    assert all(len(set(r)) == 10 for r in small)
    counts = np.bincount(small.ravel(), minlength=100)    # uniform over indices
    assert counts.min() > 0.8 * counts.mean() and counts.max() < 1.2 * counts.mean()
    r1 = mc.box_mc(np.array([1.0]), np.array([3.0]), 200_000, np.random.default_rng(3))
    assert abs(r1["mean"] - 2.0) < 4 * r1["se"] and r1["se"] == pytest.approx(r1["sd"] / math.sqrt(200_000))
    r2 = mc.box_mc(np.array([1.0]), np.array([3.0]), 200_000, np.random.default_rng(3))
    assert r1 == r2                                    # fixed seed -> identical
    assert mc.prior_exact(256) == pytest.approx(15.9844, abs=1e-4)


def test_convex_mc_matches_bruteforce_for_k2():
    rng = np.random.default_rng(7)
    Z = rng.normal(size=(5, 3))
    res = mc.convex_mc(Z, 2, 400_000, np.random.default_rng(11))
    # exact: average over the 20 ordered pairs of the integral of ||w zi + (1-w) zj|| over w ~ U[0,1]
    w = (np.arange(20000) + 0.5) / 20000
    vals = [np.linalg.norm(np.outer(w, Z[i]) + np.outer(1 - w, Z[j]), axis=1).mean()
            for i in range(5) for j in range(5) if i != j]
    assert abs(res["mean"] - np.mean(vals)) < 5 * res["se"]


# ---- decomposition on a synthetic dataset --------------------------------------------------------------
def test_decomposition_contrasts_on_synthetic_arms(tmp_path, write_config):
    from crlib import decomp
    tasks = ("adip", "rano", "fexo")
    seeds = (42, 43, 44, 45)
    base = {"random_search": 0.40, "random_box_faithful": 0.38, "hvarfner_scaled_gp": 0.43,
            "hvarfner_scaled_gp_box_argmax": 0.385, "hvarfner_scaled_gp_sphere_continuous": 0.39,
            "hvarfner_scaled_gp_box_faithful": 0.395}
    arms = {}
    rng = np.random.default_rng(5)
    for solver, lev in base.items():
        d = tmp_path / solver
        d.mkdir()
        for t in tasks:
            for s in seeds:
                cs = [0.2 + 0.001 * i for i in range(5)]
                tail = list(np.sort(rng.uniform(0.25, lev, 25)))
                rec = ours_record(solver, t, s, scores=cs + tail)
                (d / f"ours_{solver}_{TASKS_T11[t]}_s{s}_128d.json").write_text(json.dumps(rec))
        arms[solver] = arm_cfg(solver)
    ds = load_all(load_config(write_config(arms, tasks=tasks, seeds="42-45")), "abc1234", log=lambda m: None)
    eng = Engine(ds)
    ctx = Ctx(ds=ds, eng=eng, reg=Registry("x", "0" * 64, "default"), var=VARIANTS["default"], outdir=str(tmp_path),
              script_commit="x", mode="cr", extra={"_shared_cache": {}})
    A, legacy, missing = decomp.decomposition_arms(ctx)
    assert A == decomp.DEC_ARMS and not legacy and not missing
    R = decomp.dec_compute(ctx, A)
    v = {k: eng.vals(a, R["cells"]) for k, a in A.items()}
    tm = {k: stats.task_means(v[k], R["cells"]) for k in A}
    lev = {k: np.mean(list(tm[k].values())) for k in A}
    assert R["pr"]["SupportArgmax"].delta == pytest.approx(lev["SA"] - lev["BA"])
    assert R["pr"]["IntMax"].delta == pytest.approx((lev["SA"] - lev["BA"]) - (lev["SC"] - lev["BC"]))
    assert R["pr"]["IntSurr"].delta == pytest.approx((lev["SA"] - lev["BA"]) - (lev["S0"] - lev["B0"]))
    assert R["pr"]["SurrBox"].delta == pytest.approx(lev["BA"] - lev["B0"])
    q = stats.bh([R["pr"][k].p_task for k, _, _, _ in decomp.DEC_CONTRASTS])
    assert [R["pr"][k].p_task_bh for k, _, _, _ in decomp.DEC_CONTRASTS] == pytest.approx(q, nan_ok=True)
    t = decomp.tab_decomposition(ctx)
    assert t is not None and "crDecLevelShellArgmax" in ctx.reg.records
    assert ctx.reg.records["crDecLevelShellArgmax"]["files"]["count"] == len(R["cells"])
    t2 = decomp.tab_decomposition_contrasts(ctx)
    assert t2 is not None and len([r for r in t2.rows if isinstance(r, list)]) == 9
    for name, rec in ctx.reg.records.items():
        assert rec["files"]["count"] > 0, name
