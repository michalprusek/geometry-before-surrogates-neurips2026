"""Statistics against scipy and hand-computed toy data; macro naming; query comparison."""
import math

import numpy as np
import pytest
from scipy import stats as sps

from crlib import numbers as N
from crlib import stats
from crlib.queries import compare

RNG = np.random.default_rng(20261011)


def toy(n_per_task, shift=0.01, ties=0):
    """Two arms on an unbalanced grid of tasks; returns (a, b, cells)."""
    a, b, cells = {}, {}, []
    for t, n in n_per_task.items():
        for s in range(n):
            k = (t, 42 + s)
            x = RNG.normal(0.4, 0.1)
            a[k] = x + shift + RNG.normal(0, 0.02)
            b[k] = x
            cells.append(k)
    for k in cells[:ties]:
        a[k] = b[k]
    return a, b, cells


def test_wilcoxon_matches_scipy_defaults():
    for n in (8, 40, 440):
        d = RNG.normal(0.01, 0.05, n)
        assert stats.wilcoxon_p(d) == pytest.approx(sps.wilcoxon(d).pvalue, rel=0, abs=0)


def test_wilcoxon_with_zero_differences_uses_wilcox_rule():
    d = np.concatenate([RNG.normal(0.01, 0.05, 100), np.zeros(17)])
    assert stats.wilcoxon_p(d) == sps.wilcoxon(d, zero_method="wilcox").pvalue


def test_all_ties_gives_p_one_and_nan_d():
    assert stats.wilcoxon_p(np.zeros(10)) == 1.0
    assert math.isnan(stats.cohens_d(np.zeros(10)))


def test_cohens_d_is_mean_over_sd_ddof1():
    d = RNG.normal(0.02, 0.07, 200)
    assert stats.cohens_d(d) == pytest.approx(d.mean() / d.std(ddof=1))


def test_exact_task_level_wilcoxon():
    td = RNG.normal(0.01, 0.01, 11)
    assert stats.wilcoxon_exact_p(td) == pytest.approx(sps.wilcoxon(td, method="exact").pvalue)
    assert stats.wilcoxon_exact_p(np.full(11, 0.01) * np.arange(1, 12)) == pytest.approx(2 / 2048)
    assert math.isnan(stats.wilcoxon_exact_p([0.01]))           # one task (smoke run): no test


def test_levels_taskmean_vs_flat_on_unbalanced_set():
    vals = {("a", 1): 1.0, ("a", 2): 1.0, ("a", 3): 1.0, ("b", 1): 0.0}
    cells = list(vals)
    assert stats.level(vals, cells, "flat") == pytest.approx(0.75)
    assert stats.level(vals, cells, "taskmean") == pytest.approx(0.5)


def test_paired_record():
    a, b, cells = toy({"t1": 40, "t2": 30, "t3": 35}, ties=5)
    pr = stats.paired(a, b, cells, "taskmean")
    d = np.asarray([a[k] - b[k] for k in sorted(cells)])
    assert pr.n == len(cells) and pr.ties == 5
    assert pr.wins == int((d > 0).sum()) and pr.losses == int((d < 0).sum())
    assert pr.delta_flat == pytest.approx(d.mean())
    assert pr.delta_taskmean == pytest.approx(stats.level(a, cells) - stats.level(b, cells))
    assert pr.d == pytest.approx(d.mean() / d.std(ddof=1))
    assert pr.p == pytest.approx(sps.wilcoxon(d).pvalue)
    tm = [np.mean([a[k] - b[k] for k in cells if k[0] == t]) for t in ("t1", "t2", "t3")]
    assert pr.tasks_won == sum(x > 0 for x in tm) and pr.n_tasks == 3
    assert pr.p_task == pytest.approx(sps.wilcoxon(tm, method="exact").pvalue)
    assert pr.delta == pr.delta_taskmean
    assert stats.paired(a, b, cells, "flat").delta == pr.delta_flat


def test_bh_against_scipy_reference_and_hand_values():
    p = [0.01, 0.04, 0.03, 0.005]
    assert stats.bh(p) == pytest.approx([0.02, 0.04, 0.04, 0.02])
    q = RNG.uniform(0, 0.2, 25)
    assert stats.bh(q) == pytest.approx(stats.bh_reference(q))
    assert stats.bh(q) == pytest.approx(list(sps.false_discovery_control(q, method="bh")))


def test_bh_skips_nan():
    out = stats.bh([0.01, float("nan"), 0.04])
    assert math.isnan(out[1]) and out[0] == pytest.approx(0.02) and out[2] == pytest.approx(0.04)


def test_holm():
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_macro_names_are_letters_only_and_unique():
    assert N.macro("Main", "turbo_box_faithful", "Delta") == "crMainTurboBoxFaithfulDelta"
    assert N.macro("Radius", "r16") == "crRadiusROneSix"
    assert N.macro("Floor", "med1") == "crFloorMedOne"
    reg = N.Registry("abc", "0" * 64, "default")
    fs = {"count": 1, "list_sha256": "a" * 64, "content_sha256": "b" * 64, "arms": []}
    reg.add("crX", 0.5, "0.5000", kind="level", table="t", fileset=fs)
    with pytest.raises(ValueError):
        reg.add("crX", 0.5, "0.5000", kind="level", table="t", fileset=fs)
    with pytest.raises(ValueError):
        reg.add("crXa", 0.5, "0.5000", kind="level", table="t", fileset=None)     # every number needs a file set
    with pytest.raises(ValueError):
        reg.add("crXb", 0.5, "0.5000", kind="level", table="t", fileset=dict(fs, count=0))
    with pytest.raises(ValueError):
        reg.add("crX1", 0.5, "0.5000", kind="level", table="t", fileset=fs)


def test_formatting():
    assert N.f4(0.42256) == "0.4226" and N.s4(0.0421) == "+0.0421" and N.s4(-0.0102) == "-0.0102"
    assert N.pfmt(3e-5) == "<0.001" and N.pfmt(0.0123) == "0.012" and N.pfmt(0.46) == "0.46"
    assert N.psig(2 / 2048) == "0.00098" and N.psig(0.0049) == "0.0049" and N.psig(0.64) == "0.64"
    assert N.intfmt(249455) == "249{,}455"


def test_reference_compare_rules():
    assert compare({"value": 0.4164, "printed": "0.4164", "cmp": "abs"}, 0.41644)[0] == "PASS"
    assert compare({"value": 0.4164, "printed": "0.4164", "cmp": "abs"}, 0.41656)[0] == "FAIL"
    assert compare({"value": 0.14, "printed": "0.14", "cmp": "abs", "known_diff": True}, 0.154)[0] == "KNOWN-DIFF"
    assert compare({"value": 3.4e-23, "printed": "3.4e-23", "cmp": "sig2"}, 3.43e-23)[0] == "PASS"
    assert compare({"value": 423, "printed": "423", "cmp": "exact"}, 423)[0] == "PASS"
    assert compare({"value": 1e-5, "printed": "<1e-5", "cmp": "lt"}, 2e-6)[0] == "PASS"
