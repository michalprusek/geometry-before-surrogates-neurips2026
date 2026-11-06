"""Loader failure modes: missing arm / cell, duplicate cell, provenance, content; counted (never
silently dropped) non-completed and fallback cells; excluded suffix files."""
import json

import numpy as np
import pytest

from conftest import arm_cfg, ours_record
from crlib import TASKS_T11
from crlib.config import ConfigError, load_config
from crlib.loader import LoaderError, load_all

GRID = [(t, s) for t in ("adip", "rano") for s in (42, 43, 44)]


def _load(path, commit="abc1234", allow=None, **kw):
    return load_all(load_config(path), commit, allow, log=lambda m: None, **kw)


def test_clean_load_counts_everything(make_arm, write_config):
    make_arm("geowalk", "geowalk", GRID, per_cell={("rano", 43): {"status": "crashed", "scores": [0.1] * 12},
                                                   ("adip", 44): {"fallback": 3}})
    ds = _load(write_config({"geowalk": arm_cfg("geowalk")}))
    a = ds.arms["geowalk"]
    assert len(a.cells) == 6                         # the crashed cell is loaded, not dropped
    assert a.report["noncompleted_cells"] == ["rano:43=crashed"]
    assert a.report["fallback_cells"] == 1 and a.report["fallback_steps"] == 3
    assert a.eligible("completed") == set(GRID) - {("rano", 43)}
    assert a.eligible("clean") == set(GRID) - {("rano", 43), ("adip", 44)}


def test_missing_arm_fails(write_config):
    with pytest.raises(LoaderError) as e:
        _load(write_config({"geowalk": arm_cfg("geowalk")}))
    assert e.value.kinds() == {"missing_arm"}


def test_missing_arm_allowed_is_reported(make_arm, write_config):
    make_arm("geowalk", "geowalk", GRID)
    ds = _load(write_config({"geowalk": arm_cfg("geowalk"), "gabo": arm_cfg("gabo")}), allow={"gabo"})
    assert "gabo" in ds.absent and "allowed" in ds.absent["gabo"]
    assert not ds.present("gabo")


def test_allow_missing_unknown_name_fails(make_arm, write_config):
    make_arm("geowalk", "geowalk", GRID)
    with pytest.raises(LoaderError) as e:
        _load(write_config({"geowalk": arm_cfg("geowalk")}), allow={"nonexistent"})
    assert "config" in e.value.kinds()


def test_optional_arm_absent_reported_not_fatal(make_arm, write_config):
    make_arm("geowalk", "geowalk", GRID)
    ds = _load(write_config({"geowalk": arm_cfg("geowalk"), "extra": arm_cfg("extra", required=False)}))
    assert "optional" in ds.absent["extra"]
    with pytest.raises(LoaderError):
        _load(write_config({"geowalk": arm_cfg("geowalk"), "extra": arm_cfg("extra", required=False)}),
              require_optional=True)


def test_missing_cell_fails_unless_declared(make_arm, write_config):
    make_arm("geowalk", "geowalk", [c for c in GRID if c != ("adip", 43)])
    with pytest.raises(LoaderError) as e:
        _load(write_config({"geowalk": arm_cfg("geowalk")}))
    assert e.value.kinds() == {"missing_cell"} and "adip:43" in str(e.value)
    ds = _load(write_config({"geowalk": arm_cfg("geowalk", known_missing=["adip:43"])}))
    assert ds.arms["geowalk"].report["missing_cells"] == ["adip:43"]


def test_stale_known_missing_fails(make_arm, write_config):
    make_arm("geowalk", "geowalk", GRID)
    with pytest.raises(LoaderError) as e:
        _load(write_config({"geowalk": arm_cfg("geowalk", known_missing=["adip:43"])}))
    assert "declared known_missing but present" in str(e.value)


def test_duplicate_cell_fails(make_arm, write_config):
    make_arm("geowalk", "geowalk", GRID, suffix="_128d")
    make_arm("geowalk", "geowalk", [("adip", 42)], suffix="_256d")
    cfg = arm_cfg("geowalk", pattern="ours_geowalk_{task}_s{seed}_{dim}d.json")
    with pytest.raises(LoaderError) as e:
        _load(write_config({"geowalk": cfg}))
    assert e.value.kinds() == {"duplicate"}


def test_suffix_files_are_excluded_and_counted(make_arm, write_config):
    make_arm("geowalk", "geowalk", GRID)
    make_arm("geowalk", "geowalk", GRID, suffix="_128d_b1000")
    make_arm("geowalk", "geowalk", [("adip", 42)], suffix="_128d_sd8")
    ds = _load(write_config({"geowalk": arm_cfg("geowalk")}))
    a = ds.arms["geowalk"]
    assert len(a.cells) == 6 and all(c.file.endswith("_128d.json") for c in a.cells.values())
    assert a.report["excluded"]["suffix_variants"] == {"_128d_b1000": 6, "_128d_sd8": 1}


def test_provenance_mismatch_fails(make_arm, write_config):
    make_arm("geowalk", "geowalk", GRID, per_cell={("rano", 44): {"commit": "deadbee"}})
    with pytest.raises(LoaderError) as e:
        _load(write_config({"geowalk": arm_cfg("geowalk")}), commit="abc1234")
    assert e.value.kinds() == {"provenance"}


def test_provenance_dirty_fails_and_prefix_match_passes(make_arm, write_config):
    make_arm("geowalk", "geowalk", GRID, per_cell={("adip", 42): {"dirty": True}})
    with pytest.raises(LoaderError):
        _load(write_config({"geowalk": arm_cfg("geowalk")}))
    make_arm("gw2", "gw2", GRID, commit="abc1234")
    ds = _load(write_config({"gw2": arm_cfg("gw2")}), commit="abc1234ffffffffffffffffffffffffffffffffff")
    assert ds.present("gw2")


def test_provenance_record_does_not_fail(make_arm, write_config):
    make_arm("old", "old", GRID, commit="0000000", dirty=True)
    ds = _load(write_config({"old": arm_cfg("old", kind="stored")}))
    assert ds.arms["old"].report["provenance_states"] == {"0000000|True|testhost": 6}


def test_cr_arm_must_require_provenance(write_config):
    with pytest.raises(ConfigError):
        load_config(write_config({"geowalk": arm_cfg("geowalk", provenance="record")}))


def test_solver_field_mismatch_fails(make_arm, write_config):
    make_arm("geowalk", "geowalk", GRID)
    cfg = arm_cfg("geowalk", solver_field="hvarfner_scaled_gp")
    with pytest.raises(LoaderError) as e:
        _load(write_config({"geowalk": cfg}))
    assert e.value.kinds() == {"content"}


def test_status_absent_rule(make_arm, write_config):
    make_arm("old", "old", GRID, status=None)
    ds = _load(write_config({"old": arm_cfg("old", kind="stored")}))
    assert ds.arms["old"].eligible("completed") == set()            # require_field: absent != completed
    assert ds.arms["old"].report["status_counts"] == {"absent": 6}
    ds = _load(write_config({"old": arm_cfg("old", kind="stored", status_rule="absent_is_completed")}))
    assert ds.arms["old"].eligible("completed") == set(GRID)


def test_subset_run_and_values(make_arm, write_config):
    make_arm("geowalk", "geowalk", GRID)
    cfg = load_config(write_config({"geowalk": arm_cfg("geowalk")}), tasks_subset=["rano"], seeds_subset=[42, 43])
    ds = load_all(cfg, "abc1234", log=lambda m: None)
    a = ds.arms["geowalk"]
    assert set(a.cells) == {("rano", 42), ("rano", 43)}
    assert a.report["n_files_outside_run_subset"] == 4
    rec = ours_record("geowalk", "rano", 42)
    assert a.value(("rano", 42)) == pytest.approx(rec["best_trace"][29])
    assert a.value(("rano", 42), 10) == pytest.approx(rec["best_trace"][9])


def test_external_format_and_coldstart_prefix(tmp_path, make_arm, write_config):
    make_arm("geowalk", "geowalk", GRID)
    ext = tmp_path / "turbo"
    ext.mkdir()
    for t, s in GRID:
        hist = [0.3 + 0.001 * i for i in range(25)]
        (ext / f"turbo_{t}_seed{s}.json").write_text(json.dumps(
            {"method": "turbo", "task_id": t, "seed": s, "best_score": hist[-1], "n_evaluated": 30,
             "history": {"best_score": hist}}))
    cfg = {"label": "TuRBO", "kind": "stored", "format": "external", "dir": "${T}/turbo",
           "pattern": "turbo_{task}_seed{seed}.json", "task_key": "short", "solver_field": "turbo",
           "status_rule": "absent_is_completed", "provenance": "record", "coldstart_from": "geowalk"}
    ds = _load(write_config({"geowalk": arm_cfg("geowalk"), "turbo": cfg}))
    tu, gw = ds.arms["turbo"], ds.arms["geowalk"]
    c = tu.cells[("adip", 42)]
    assert len(c.trace) == 30
    np.testing.assert_allclose(c.trace[:5], gw.cells[("adip", 42)].trace[:5])
    assert tu.value(("adip", 42)) == pytest.approx(0.3 + 0.024)
    assert tu.eligible("completed") == set(GRID)
