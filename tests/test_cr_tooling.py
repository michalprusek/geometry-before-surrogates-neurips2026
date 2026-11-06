"""Camera-ready run tooling: manifest, per-cell driver, queue.

These tests guard the rules that make the generation auditable
(``docs/camera_ready/RUN_RULES.md``):

* R-1 the per-cell driver calls ``run_benchmark`` with the CLI defaults that
  produced the existing files (only ``subspace_dim`` 8/64 may be overridden,
  per the amendment log) and writes the file names those files use;
* R-2 one commit, a clean tree, the right host class, inputs hashed;
* R-5 a completed cell is never overwritten or rerun, a result counts only
  after validation, one queue per output root, cells claimed exclusively;
* R-8 crashes, timeouts, invalid results and dead claims are terminal; a rerun
  needs ``--retry-crashed --amendment`` and archives the earlier attempt; a
  valid result published by a child that then died is completed, not crashed;
* the manifest covers every arm of the generation, including the 13 arms of
  the 2026-10-11 amendment and the 3 GP sphere-radius arms of 2026-10-12, with
  unique arm ids; CPU-class arms run on kajman only;
* per-arm seed ranges (the control block of 2026-10-12): an arm runs its
  ``seeds`` field, else the manifest's 42-81; two arms share a computation only
  on disjoint seed ranges; the queue and run_cell refuse a seed outside an
  arm's range.

The end-to-end tests run the real ``scripts/cr/cr_queue.py`` against a
throwaway git repository with a fake child interpreter (no torch, seconds).
"""
from __future__ import annotations

import fcntl
import json
import os
import signal
import socket
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from scripts.cr import cr_queue
from scripts.cr import run_cell as cr_run_cell

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "scripts" / "cr" / "manifest.json"
FAKE_SHA = "0123456789abcdef0123456789abcdef01234567"

AMENDMENT_ARMS = {
    # arm id: (solver, run_overrides, wave)
    "geowalk_const060": ("geowalk_const060", {}, "1"),
    "geowalk_const035": ("geowalk_const035", {}, "1"),
    "geowalk_const015": ("geowalk_const015", {}, "1"),
    "geowalk_explore8": ("geowalk_explore8", {}, "1"),
    "geowalk_explore24": ("geowalk_explore24", {}, "1"),
    "geowalk_one_fifth": ("geowalk_one_fifth", {}, "1"),
    "cma_es_subspace_sd8": ("cma_es_subspace", {"subspace_dim": 8}, "1"),
    "cma_es_subspace_sd64": ("cma_es_subspace", {"subspace_dim": 64}, "1"),
    "random_search_prior": ("random_search_prior", {}, "1"),
    "random_box_faithful": ("random_box_faithful", {}, "1"),
    "papenmeier_msr_raasp": ("papenmeier_msr_raasp", {}, "1"),
    "hvarfner_scaled_gp_norm_box_argmax": ("hvarfner_scaled_gp_norm_box_argmax", {}, "3b"),
    "papenmeier_msr_published_norm_box": ("papenmeier_msr_published_norm_box", {}, "4"),
    # amendment of 2026-10-12: GP radius dose-response (candidate-set argmax)
    "hvarfner_scaled_gp_sphere05": ("hvarfner_scaled_gp_sphere05", {}, "1"),
    "hvarfner_scaled_gp_sphere15": ("hvarfner_scaled_gp_sphere15", {}, "1"),
    "hvarfner_scaled_gp_sphere20": ("hvarfner_scaled_gp_sphere20", {}, "1"),
}
#: The control block of the 2026-10-12 amendment: arm id -> (solver, wave, seeds, timeout_s).
SEEDS_42 = list(range(42, 82))
SEEDS_82 = list(range(82, 122))
DECOMPOSITION_ARMS = (
    "random_search", "random_box_faithful", "hvarfner_scaled_gp",
    "hvarfner_scaled_gp_box_argmax", "hvarfner_scaled_gp_sphere_continuous",
    "hvarfner_scaled_gp_box_faithful",
)
CONTROL_ARMS = {
    "hvarfner_scaled_gp_box_argmax_raw": ("hvarfner_scaled_gp_box_argmax_raw", "c1", SEEDS_42, 7200),
    "hvarfner_scaled_gp_boxdir_argmax": ("hvarfner_scaled_gp_boxdir_argmax", "c1", SEEDS_42, 7200),
    "random_boxdir": ("random_boxdir", "c1", SEEDS_42, 3600),
    **{f"{a}__s82": (a, "c2", SEEDS_82, None) for a in DECOMPOSITION_ARMS},  # timeout: the base arm's
}
#: Control C4 of the 2026-10-14 amendment (wave c3, after the control block); the printed shell x continuous cell
#: of the factorial since 2026-11-06 (RUN_RULES 2026-11-06 16:30).
C4_ARMS = {
    "hvarfner_scaled_gp_sphere_lbfgs": ("hvarfner_scaled_gp_sphere_lbfgs", "c3", SEEDS_42, 14400),
}
N_PRODUCTION_ARMS = 66   # 60 at 20ad7d5, + 3 sphere-radius arms (14481cb),
#                          + cmaes_full_rd, cmaes_full_rd_pop5, baxus_cr (2026-11-02)
N_ARMS = N_PRODUCTION_ARMS + len(CONTROL_ARMS) + len(C4_ARMS)    # 76
N_GPU_ARMS = 64 + len(CONTROL_ARMS) + len(C4_ARMS)               # 74


@pytest.fixture(scope="module")
def manifest() -> dict:
    return cr_queue.load_manifest(MANIFEST)


# =========================================================================
# manifest
# =========================================================================


def test_manifest_tasks_and_seeds_are_the_paper_protocol(manifest):
    assert manifest["tasks"] == [
        "adip", "dhop", "fexo", "med1", "med2", "osmb",
        "pdop", "rano", "shop", "siga", "zale",
    ]
    assert manifest["seeds"] == list(range(42, 82))
    from src.analysis.config import GUACAMOL_TO_POLI

    for code in manifest["tasks"]:
        assert manifest["task_poli_ids"][code] == GUACAMOL_TO_POLI[code]


def test_every_manifest_solver_is_known_to_the_benchmark_runner(manifest):
    from scripts.run_benchmark import SUPPORTED_SOLVERS

    for arm in manifest["arms"]:
        assert arm["solver"] in SUPPORTED_SOLVERS, arm["arm"]


def test_manifest_arm_ids_and_computations_are_unique(manifest):
    cr_queue.validate_manifest(manifest)
    ids = [a["arm"] for a in manifest["arms"]]
    assert len(ids) == len(set(ids)) == N_ARMS
    # every (solver, overrides, seed) cell belongs to one arm only: arms that share
    # a solver and overrides (the control block's <arm>__s82) have disjoint seeds
    cells = [(a["solver"], json.dumps(a["run_overrides"], sort_keys=True), s)
             for a in manifest["arms"] for s in cr_run_cell.arm_seeds(manifest, a)]
    assert len(cells) == len(set(cells)) == N_ARMS * 40
    keys = [(a["solver"], json.dumps(a["run_overrides"], sort_keys=True))
            for a in manifest["arms"]]
    assert len(keys) - len(set(keys)) == len(DECOMPOSITION_ARMS)   # only the __s82 arms
    dirs = {cr_queue.cell_output(a, "rano", 42, manifest, None).parent for a in manifest["arms"]}
    assert len(dirs) == N_ARMS


def test_wave_counts_host_classes_and_hosts(manifest):
    waves = [str(a["wave"]) for a in manifest["arms"]]
    assert {w: waves.count(w) for w in cr_queue.VALID_WAVES} == {
        "1": 44, "2": 9, "3": 8, "3b": 4, "4": 1, "c1": 3, "c2": 6, "c3": 1,
    }
    assert {a["arm"] for a in manifest["arms"] if a["host_class"] == "cpu"} == {
        "gabo", "matern_gabo",
    }
    assert sum(a["host_class"] == "gpu" for a in manifest["arms"]) == N_GPU_ARMS
    assert manifest["gpu_hosts"] == ["panda.utia.cas.cz"]
    # R-2 amendment of 2026-10-12: the CPU-class arms run on kajman only
    assert manifest["cpu_hosts"] == ["kajman.utia.cas.cz"]


def test_the_production_selection_is_64_gpu_arms_x_440_cells(manifest):
    arms = cr_queue.select_arms(manifest, wave=None, arm_ids=None, host_class="gpu",
                                waves=["2", "1", "3", "3b", "4"])
    assert len(arms) == 64
    assert len(arms) * len(manifest["tasks"]) * len(manifest["seeds"]) == 28160
    # the production arms have no seeds field: all run the manifest's 42-81
    assert not any("seeds" in a for a in arms)
    per_arm = cr_queue.seeds_for_arms(manifest, arms, None)
    assert all(s == SEEDS_42 for s in per_arm.values())
    assert sum(len(s) for s in per_arm.values()) * len(manifest["tasks"]) == 28160
    # wave by wave in the given order, manifest order within a wave
    order = [str(a["wave"]) for a in arms]
    assert order == sorted(order, key=["2", "1", "3", "3b", "4"].index)
    assert [a["arm"] for a in arms if str(a["wave"]) == "1"] == [
        a["arm"] for a in manifest["arms"] if str(a["wave"]) == "1"]
    # the production arms are the first 66 entries; the control block follows them, then control C4
    n_ctl = N_PRODUCTION_ARMS + len(CONTROL_ARMS)
    assert [a["arm"] for a in manifest["arms"][N_PRODUCTION_ARMS:n_ctl]] == list(CONTROL_ARMS)
    assert [a["arm"] for a in manifest["arms"][n_ctl:]] == list(C4_ARMS)


def test_the_control_block_selection_is_9_gpu_arms_x_440_cells(manifest):
    arms = cr_queue.select_arms(manifest, wave=None, arm_ids=None, host_class="gpu",
                                waves=["c1", "c2"])
    assert [a["arm"] for a in arms] == list(CONTROL_ARMS)
    per_arm = cr_queue.seeds_for_arms(manifest, arms, None)
    assert {a: s for a, s in per_arm.items()} == {
        a: spec[2] for a, spec in CONTROL_ARMS.items()}
    assert sum(len(s) for s in per_arm.values()) * len(manifest["tasks"]) == 3960
    # a seed list that does not fit every selected range is refused, not filtered
    with pytest.raises(SystemExit, match="outside the seed range of arm 'random_search__s82'"):
        cr_queue.seeds_for_arms(manifest, arms, [42, 43])


def test_the_c4_selection_is_1_gpu_arm_x_440_cells(manifest):
    arms = cr_queue.select_arms(manifest, wave=None, arm_ids=None, host_class="gpu",
                                waves=["c3"])
    assert [a["arm"] for a in arms] == list(C4_ARMS)
    per_arm = cr_queue.seeds_for_arms(manifest, arms, None)
    assert per_arm == {a: spec[2] for a, spec in C4_ARMS.items()}
    assert sum(len(s) for s in per_arm.values()) * len(manifest["tasks"]) == 440
    by_id = {a["arm"]: a for a in manifest["arms"]}
    for arm_id, (solver, wave, seeds, timeout) in C4_ARMS.items():
        arm = by_id[arm_id]
        assert (arm["solver"], arm["run_overrides"], arm["wave"], arm["seeds"],
                arm["host_class"], arm["timeout_s"]) == (solver, {}, wave, seeds, "gpu", timeout)
    # C4 pairs with the decomposition arms of the main campaign: same seeds 42-81
    assert by_id["hvarfner_scaled_gp_sphere_continuous"]["timeout_s"] == 14400
    assert "seeds" not in by_id["hvarfner_scaled_gp_sphere_continuous"]
    assert manifest["seeds"] == SEEDS_42


def test_the_amendment_arms_are_in_the_manifest(manifest):
    by_id = {a["arm"]: a for a in manifest["arms"]}
    for arm_id, (solver, overrides, wave) in AMENDMENT_ARMS.items():
        arm = by_id[arm_id]
        assert (arm["solver"], arm["run_overrides"], str(arm["wave"])) == (
            solver, overrides, wave), arm_id
        assert arm["host_class"] == "gpu"
    assert {"papenmeier_msr_published", "turbo_box_faithful"} <= set(by_id)


def test_the_control_block_arms_are_in_the_manifest(manifest):
    """C1, C2a, C2b on seeds 42-81 (wave c1) and C3 = the six decomposition arms
    of the 2026-10-12 analysis plan on seeds 82-121 (wave c2), each C3 arm with
    its base arm's solver, overrides, host class and timeout."""
    by_id = {a["arm"]: a for a in manifest["arms"]}
    for arm_id, (solver, wave, seeds, timeout) in CONTROL_ARMS.items():
        arm = by_id[arm_id]
        assert (arm["solver"], arm["run_overrides"], arm["wave"], arm["seeds"]) == (
            solver, {}, wave, seeds), arm_id
        assert arm["host_class"] == "gpu"
        if wave == "c2":
            base = by_id[solver]
            assert arm_id == f"{base['arm']}__s82"
            assert (base["solver"], base["run_overrides"], base["host_class"],
                    base["timeout_s"]) == (arm["solver"], arm["run_overrides"],
                                           arm["host_class"], arm["timeout_s"])
            assert "seeds" not in base
        else:
            assert arm["timeout_s"] == timeout
    assert sorted(a[: -len("__s82")] for a in CONTROL_ARMS if a.endswith("__s82")) == sorted(
        DECOMPOSITION_ARMS)


def test_timeouts_follow_the_reviewed_classes(manifest):
    slowest = {"papenmeier_msr_published", "papenmeier_msr_published_norm_box",
               "turbo_box_faithful"}
    gp_continuous = {
        "hvarfner_scaled_gp_box_faithful", "hvarfner_scaled_gp_sphere_continuous",
        "hvarfner_scaled_gp_sphere_lbfgs",
        "hvarfner_scaled_gp_norm_box", "papenmeier_msr_box_faithful",
        "papenmeier_msr_norm_box", "xu_standard_gp_box_faithful", "xu_standard_gp_norm_box",
        "linear_bo", "rembo", "hesbo", "gabo", "matern_gabo",
        "hvarfner_scaled_gp_box", "papenmeier_msr_box", "xu_standard_gp_box",
    }
    candidate_set = {
        "hvarfner_scaled_gp", "papenmeier_msr", "xu_standard_gp", "linear_bo_sphere_calib",
        "hvarfner_scaled_gp_box_argmax", "hvarfner_scaled_gp_prior",
        "hvarfner_scaled_gp_angular_box", "papenmeier_msr_raasp",
        "hvarfner_scaled_gp_norm_box_argmax",
        "hvarfner_scaled_gp_sphere05", "hvarfner_scaled_gp_sphere15",
        "hvarfner_scaled_gp_sphere20",
        "hvarfner_scaled_gp_box_argmax_raw", "hvarfner_scaled_gp_boxdir_argmax",
        "baxus_cr",            # candidate-set Thompson sampling (2026-11-02)
    }
    for arm in manifest["arms"]:
        a = arm["arm"].removesuffix("__s82")      # a control-block C3 arm: its base arm's class
        expected = (21600 if a in slowest else 14400 if a in gp_continuous
                    else 7200 if a in candidate_set else 3600)
        assert arm["timeout_s"] == expected, a


@pytest.mark.parametrize("mutate,match", [
    (lambda m: m["arms"].append(dict(m["arms"][0])), "duplicate arm id"),
    (lambda m: m["arms"].append({**m["arms"][0], "arm": "geowalk_copy"}), "repeats the computation"),
    (lambda m: m["arms"][0].update(run_overrides={"max_iter": 50}), "not allowed"),
    (lambda m: m["arms"][0].update(run_overrides={"subspace_dim": 16}), "not allowed"),
    (lambda m: m["arms"][0].update(run_overrides={"subspace_dim": 32}), "not allowed"),
    (lambda m: m["arms"][0].update(wave=5), "wave"),
    (lambda m: m["arms"][0].update(host_class="tpu"), "host_class"),
    (lambda m: m["arms"][0].update(arm="Geo/Walk"), "must match"),
    (lambda m: m["arms"][0].update(timeout_s=0), "timeout_s"),
    (lambda m: m.pop("gpu_hosts"), "gpu_hosts"),
    # per-arm seed ranges
    (lambda m: m["arms"][0].update(seeds=[]), "non-empty list"),
    (lambda m: m["arms"][0].update(seeds="82-121"), "non-empty list"),
    (lambda m: m["arms"][0].update(seeds=[82, 82]), "repeat a seed"),
    (lambda m: m["arms"][0].update(seeds=["82"]), "integers"),
    (lambda m: m["arms"][0].update(seeds=[True]), "integers"),
    (lambda m: m["arms"][0].update(seeds=[82.0]), "integers"),
    (lambda m: m["arms"].append({**m["arms"][0], "arm": "geowalk__s81", "seeds": [81, 82]}),
     "repeats the computation"),                      # overlaps geowalk's 42-81 on seed 81
    (lambda m: m["arms"].append({**next(a for a in m["arms"] if a["arm"].endswith("__s82")), "arm": "x__s100",
                                 "seeds": [100, 200]}),
     "repeats the computation"),                      # overlaps a __s82 arm's 82-121 (the last arm is control C4)
    (lambda m: m["arms"][0].update(wave="c9"), "wave"),
])
def test_validate_manifest_rejects_malformed_manifests(manifest, mutate, match):
    bad = json.loads(json.dumps(manifest))
    mutate(bad)
    with pytest.raises(cr_queue.ManifestError, match=match):
        cr_queue.validate_manifest(bad)


def test_arm_seed_ranges_default_and_explicit(manifest):
    by_id = {a["arm"]: a for a in manifest["arms"]}
    assert cr_run_cell.arm_seeds(manifest, by_id["geowalk"]) == SEEDS_42          # default
    assert cr_run_cell.arm_seeds(manifest, by_id["random_boxdir"]) == SEEDS_42    # explicit
    assert cr_run_cell.arm_seeds(manifest, by_id["random_search__s82"]) == SEEDS_82
    # a disjoint copy of an arm is a new computation and is accepted
    ok = json.loads(json.dumps(manifest))
    ok["arms"].append({**ok["arms"][0], "arm": "geowalk__s200", "seeds": [200, 201]})
    cr_queue.validate_manifest(ok)
    assert cr_queue.format_seeds(SEEDS_82) == "82-121"
    assert cr_queue.format_seeds([45, 42, 43, 44, 81, 83]) == "42-45,81,83"


# =========================================================================
# per-cell driver (run_cell.py)
# =========================================================================


def test_run_defaults_are_the_r1_cli_defaults():
    assert cr_run_cell.RUN_DEFAULTS == {
        "mode": "ours", "max_iter": 25, "n_initial_points": 10, "n_cold_start": 5,
        "latent_dim": 128, "subspace_dim": 16, "n_subspaces": 16,
        "cold_start_source": "zinc", "decoder_weights_path": None,
        "norm_mode": "fixed", "codec_type": "selfies",
    }
    assert cr_run_cell.BUDGET == 30
    assert cr_run_cell.ALLOWED_RUN_OVERRIDES == {"subspace_dim": frozenset({8, 64})}


def test_cell_filename_matches_the_cli_names():
    assert cr_run_cell.cell_filename("linear_bo", "ranolazine_mpo", 42) == (
        "ours_linear_bo_ranolazine_mpo_s42_128d.json"
    )
    # the July D-tilde files: spherevo/ours_cma_es_subspace_amlodipine_mpo_s42_128d_sd8.json
    assert cr_run_cell.cell_filename("cma_es_subspace", "amlodipine_mpo", 42, 8) == (
        "ours_cma_es_subspace_amlodipine_mpo_s42_128d_sd8.json"
    )
    assert cr_run_cell.cell_filename("cma_es_subspace", "amlodipine_mpo", 42, 64).endswith(
        "_128d_sd64.json")
    assert cr_run_cell.cell_filename("cma_es_subspace", "amlodipine_mpo", 42, 16) == (
        "ours_cma_es_subspace_amlodipine_mpo_s42_128d.json"
    )


def test_cli_sd_tag_rule_is_the_same_expression():
    """run_cell's tag is byte for byte the CLI's (scripts/run_benchmark.py main)."""
    src = (REPO / "scripts" / "run_benchmark.py").read_text()
    assert 'sd_tag = f"_sd{args.subspace_dim}" if args.subspace_dim != 16 else ""' in src


@pytest.mark.parametrize("overrides,ok", [
    ({}, True), ({"subspace_dim": 8}, True), ({"subspace_dim": 64}, True),
    ({"subspace_dim": 16}, False), ({"subspace_dim": 32}, False),
    ({"subspace_dim": True}, False), ({"subspace_dim": "8"}, False),
    ({"max_iter": 50}, False), ([], False),
])
def test_validate_run_overrides(overrides, ok):
    if ok:
        assert cr_run_cell.validate_run_overrides(overrides) == overrides
    else:
        with pytest.raises(cr_run_cell.RunOverrideError):
            cr_run_cell.validate_run_overrides(overrides)


def _fake_run_benchmark(calls, *, status="completed"):
    def fake(**kwargs):
        calls.append(kwargs)
        return {"status": status, "best_score": 0.75, "elapsed_seconds": 1.0,
                "surrogate_fallback_count": 0, "subspace_dim": kwargs["subspace_dim"],
                "provenance": {"git_commit": "abc1234", "git_dirty": False,
                               "package_versions": {"torch": "x"}}}
    return fake


def test_refuses_to_overwrite_a_completed_cell(tmp_path, capsys):
    out = tmp_path / "ours_geowalk_ranolazine_mpo_s42_128d.json"
    out.write_text(json.dumps({"status": "completed", "best_score": 0.5}))

    rc = cr_run_cell.main(["geowalk", "ranolazine_mpo", "42", str(out)])

    assert rc == cr_run_cell.EXIT_COMPLETED_EXISTS == 75
    assert "SKIP" in capsys.readouterr().out
    assert json.loads(out.read_text())["best_score"] == 0.5  # untouched


def test_refuses_a_non_completed_file_too(tmp_path, monkeypatch):
    """R-8: an earlier attempt is archived by the queue, never overwritten here."""
    out = tmp_path / "ours_geowalk_ranolazine_mpo_s42_128d.json"
    out.write_text(json.dumps({"status": "crashed"}))
    calls = []
    import scripts.run_benchmark as rb
    monkeypatch.setattr(rb, "run_benchmark", _fake_run_benchmark(calls))

    rc = cr_run_cell.main(["geowalk", "ranolazine_mpo", "42", str(out)])

    assert rc == cr_run_cell.EXIT_EXISTS_NOT_COMPLETED == 77
    assert calls == []
    assert json.loads(out.read_text()) == {"status": "crashed"}


def test_a_held_claim_makes_the_cell_busy(tmp_path, monkeypatch):
    out = tmp_path / "ours_geowalk_ranolazine_mpo_s42_128d.json"
    claim = cr_run_cell.claim_path(out)
    claim.write_text("12345 otherhost 2026-10-11T00:00:00\n")
    calls = []
    import scripts.run_benchmark as rb
    monkeypatch.setattr(rb, "run_benchmark", _fake_run_benchmark(calls))

    rc = cr_run_cell.main(["geowalk", "ranolazine_mpo", "42", str(out)])

    assert rc == cr_run_cell.EXIT_BUSY == 76
    assert calls == [] and not out.exists()
    assert claim.read_text().startswith("12345 otherhost")  # not ours to remove


def test_a_file_name_that_does_not_match_the_overrides_is_refused(tmp_path):
    out = tmp_path / "ours_cma_es_subspace_ranolazine_mpo_s42_128d.json"  # missing _sd8
    rc = cr_run_cell.main(["cma_es_subspace", "ranolazine_mpo", "42", str(out),
                           "--run-overrides", '{"subspace_dim": 8}'])
    assert rc == cr_run_cell.EXIT_PRECONDITION == 78


def test_runs_with_overrides_and_records_the_provenance(tmp_path, monkeypatch, capsys):
    out = tmp_path / "arm" / "ours_cma_es_subspace_ranolazine_mpo_s42_128d_sd8.json"
    calls = []
    import scripts.run_benchmark as rb
    monkeypatch.setattr(rb, "run_benchmark", _fake_run_benchmark(calls))

    rc = cr_run_cell.main([
        "cma_es_subspace", "ranolazine_mpo", "42", str(out),
        "--arm", "cma_es_subspace_sd8", "--run-overrides", '{"subspace_dim": 8}',
        "--host-class", "gpu",
    ])

    assert rc == 0, capsys.readouterr()
    (kw,) = calls
    assert kw["solver_name"] == "cma_es_subspace"
    assert kw["task_name"] == "ranolazine_mpo" and kw["seed"] == 42
    assert kw["subspace_dim"] == 8
    for key, value in cr_run_cell.RUN_DEFAULTS.items():
        if key != "subspace_dim":
            assert kw[key] == value, key
    written = json.loads(out.read_text())
    assert written["budget"] == 30 and written["subspace_dim"] == 8
    prov = written["provenance"]
    assert prov["cr_arm"] == "cma_es_subspace_sd8"
    assert prov["run_overrides"] == {"subspace_dim": 8}
    assert prov["host_class"] == "gpu"
    assert prov["python_executable"] == sys.executable
    assert len(prov["git_commit_full"]) == 40
    assert prov["git_commit"] == "abc1234"  # run_benchmark's own keys are kept
    for pkg in cr_run_cell.EXTRA_PACKAGES:
        assert pkg in prov["package_versions"]
    assert prov["package_versions"]["torch"] == "x"
    assert stat.S_IMODE(out.stat().st_mode) == 0o644
    # no claim, no temp file left behind
    assert sorted(p.name for p in out.parent.iterdir()) == [out.name]


def test_the_claim_is_released_when_the_run_raises(tmp_path, monkeypatch):
    out = tmp_path / "ours_geowalk_ranolazine_mpo_s42_128d.json"

    def boom(**kwargs):
        raise RuntimeError("hard crash")

    import scripts.run_benchmark as rb
    monkeypatch.setattr(rb, "run_benchmark", boom)
    with pytest.raises(RuntimeError, match="hard crash"):
        cr_run_cell.main(["geowalk", "ranolazine_mpo", "42", str(out)])
    assert not out.exists() and not cr_run_cell.claim_path(out).exists()


def test_expect_commit_mismatch_refuses_before_running(tmp_path, monkeypatch):
    out = tmp_path / "ours_geowalk_ranolazine_mpo_s42_128d.json"
    calls = []
    import scripts.run_benchmark as rb
    monkeypatch.setattr(rb, "run_benchmark", _fake_run_benchmark(calls))
    rc = cr_run_cell.main(["geowalk", "ranolazine_mpo", "42", str(out),
                           "--expect-commit", FAKE_SHA])
    assert rc == cr_run_cell.EXIT_PRECONDITION
    assert calls == [] and not out.exists() and not cr_run_cell.claim_path(out).exists()


def _seed_range_manifest(tmp_path) -> Path:
    """geowalk on the manifest's seeds 42-43, geowalk__s82 on its own 82-83."""
    arms = [
        {"arm": "geowalk", "solver": "geowalk", "run_overrides": {}, "wave": 1,
         "host_class": "gpu", "timeout_s": 60},
        {"arm": "geowalk__s82", "solver": "geowalk", "run_overrides": {}, "wave": "c2",
         "host_class": "gpu", "timeout_s": 60, "seeds": [82, 83]},
    ]
    man = {"tasks": ["rano"], "task_poli_ids": {"rano": "ranolazine_mpo"}, "seeds": [42, 43],
           "output_root": "results/x", "gpu_hosts": ["h"], "cpu_hosts": ["h"], "arms": arms}
    cr_queue.validate_manifest(man)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(man))
    return path


@pytest.mark.parametrize("arm,solver,seed,overrides,needle", [
    ("geowalk__s82", "geowalk", 42, {}, "outside the seed range of arm 'geowalk__s82'"),
    ("geowalk", "geowalk", 82, {}, "outside the seed range of arm 'geowalk'"),
    ("no_such_arm", "geowalk", 42, {}, "is not in the manifest"),
    ("geowalk", "random_search", 42, {}, "runs solver 'geowalk'"),
    ("geowalk", "geowalk", 42, {"subspace_dim": 8}, "has run overrides"),
])
def test_run_cell_refuses_a_cell_outside_its_arm(tmp_path, monkeypatch, capsys,
                                                 arm, solver, seed, overrides, needle):
    manifest = _seed_range_manifest(tmp_path)
    out = tmp_path / arm / cr_run_cell.cell_filename(
        solver, "ranolazine_mpo", seed, overrides.get("subspace_dim", 16))
    calls = []
    import scripts.run_benchmark as rb
    monkeypatch.setattr(rb, "run_benchmark", _fake_run_benchmark(calls))

    rc = cr_run_cell.main([solver, "ranolazine_mpo", str(seed), str(out), "--arm", arm,
                           "--manifest", str(manifest),
                           "--run-overrides", json.dumps(overrides)])

    assert rc == cr_run_cell.EXIT_PRECONDITION == 78
    assert calls == [] and not out.exists() and not cr_run_cell.claim_path(out).exists()
    assert needle in capsys.readouterr().err


def test_run_cell_runs_a_cell_inside_its_arms_seed_range(tmp_path, monkeypatch):
    manifest = _seed_range_manifest(tmp_path)
    calls = []
    import scripts.run_benchmark as rb
    monkeypatch.setattr(rb, "run_benchmark", _fake_run_benchmark(calls))
    out = tmp_path / "geowalk__s82" / cr_run_cell.cell_filename("geowalk", "ranolazine_mpo", 82)
    rc = cr_run_cell.main(["geowalk", "ranolazine_mpo", "82", str(out), "--arm", "geowalk__s82",
                           "--manifest", str(manifest)])
    assert rc == 0 and [kw["seed"] for kw in calls] == [82]
    assert json.loads(out.read_text())["provenance"]["cr_arm"] == "geowalk__s82"
    # fail closed on an unreadable manifest
    out2 = tmp_path / "geowalk__s82" / cr_run_cell.cell_filename("geowalk", "ranolazine_mpo", 83)
    rc = cr_run_cell.main(["geowalk", "ranolazine_mpo", "83", str(out2), "--arm", "geowalk__s82",
                           "--manifest", str(tmp_path / "missing.json")])
    assert rc == cr_run_cell.EXIT_PRECONDITION and not out2.exists() and len(calls) == 1


def test_run_cell_checks_the_committed_manifest_by_default(tmp_path, monkeypatch, capsys):
    """Without --manifest the committed manifest's ranges apply: a seed-42 cell of a
    seed-82-121 control arm, or a seed-82 cell of a production arm, is refused."""
    calls = []
    import scripts.run_benchmark as rb
    monkeypatch.setattr(rb, "run_benchmark", _fake_run_benchmark(calls))
    for arm, solver, seed in (("random_search__s82", "random_search", 42),
                              ("random_search", "random_search", 82),
                              ("random_boxdir", "random_boxdir", 82)):
        out = tmp_path / arm / cr_run_cell.cell_filename(solver, "ranolazine_mpo", seed)
        rc = cr_run_cell.main([solver, "ranolazine_mpo", str(seed), str(out), "--arm", arm])
        assert rc == cr_run_cell.EXIT_PRECONDITION, arm
        assert "outside the seed range" in capsys.readouterr().err
    assert calls == []
    assert cr_run_cell.MANIFEST_PATH == MANIFEST


def test_the_queue_passes_its_manifest_to_every_child(tmp_path):
    from types import SimpleNamespace

    cell = {"solver": "geowalk", "poli_task": "ranolazine_mpo", "seed": 82,
            "out": tmp_path / "x.json", "arm": "geowalk__s82", "run_overrides": {},
            "host_class": "gpu"}
    with_manifest = cr_queue.Runner(SimpleNamespace(python="py"), tmp_path, tmp_path,
                                    expect_commit=FAKE_SHA, manifest_path=tmp_path / "m.json")
    cmd = with_manifest.command(cell)
    assert cmd[cmd.index("--manifest") + 1] == str(tmp_path / "m.json")
    assert cmd[cmd.index("--arm") + 1] == "geowalk__s82"
    assert cmd[cmd.index("--expect-commit") + 1] == FAKE_SHA
    without = cr_queue.Runner(SimpleNamespace(python="py"), tmp_path, tmp_path,
                              expect_commit=None)
    assert "--manifest" not in without.command(cell)


def test_write_result_never_clobbers(tmp_path):
    out = tmp_path / "sub" / "cell.json"
    cr_run_cell.write_result(out, {"status": "completed", "v": 1})
    assert json.loads(out.read_text())["v"] == 1
    assert stat.S_IMODE(out.stat().st_mode) == 0o644
    with pytest.raises(FileExistsError):
        cr_run_cell.write_result(out, {"status": "completed", "v": 2})
    assert json.loads(out.read_text())["v"] == 1                      # untouched
    (collision,) = [p for p in out.parent.iterdir() if ".collision-" in p.name]
    assert json.loads(collision.read_text())["v"] == 2                # kept for inspection
    assert not [p for p in out.parent.iterdir() if p.name.endswith(".tmp")]

    class Unserialisable:
        pass

    with pytest.raises(TypeError):
        cr_run_cell.write_result(tmp_path / "x.json", {"bad": Unserialisable()})
    assert not (tmp_path / "x.json").exists()
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]

    cr_run_cell.write_result(out, {"status": "completed", "v": 3}, force=True)  # tests only
    assert json.loads(out.read_text())["v"] == 3


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def fake_repo(tmp_path):
    """A clean throwaway git repo with dummy (gitignored) inputs."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".gitignore").write_text("logs/\nresults/\nweights/\ndata/\n")
    for rel in cr_queue.REQUIRED_INPUTS:
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(f"dummy {rel}\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".gitignore")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@example.invalid",
         "commit", "-q", "-m", "init")
    return repo, _git(repo, "rev-parse", "HEAD")


def test_git_tree_problems_fail_closed(fake_repo, tmp_path):
    repo, head = fake_repo
    assert cr_run_cell.git_tree_problems(repo, head) == []
    assert any("expected" in p for p in cr_run_cell.git_tree_problems(repo, FAKE_SHA))
    (repo / "stray.txt").write_text("x")
    assert any("dirty" in p for p in cr_run_cell.git_tree_problems(repo, head))
    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    problems = cr_run_cell.git_tree_problems(not_a_repo, head)
    assert problems and all("failed" in p for p in problems)


# =========================================================================
# queue units
# =========================================================================


def test_parse_int_list_and_dedupe():
    assert cr_queue.parse_int_list("42-44", []) == [42, 43, 44]
    assert cr_queue.parse_int_list("42,81", []) == [42, 81]
    assert cr_queue.parse_int_list("42-43,81", []) == [42, 43, 81]
    assert cr_queue.parse_int_list("42,42,42-43", []) == [42, 43]
    assert cr_queue.parse_int_list(None, [1, 2, 2]) == [1, 2]
    assert cr_queue.dedupe(["b", "a", "b"]) == ["b", "a"]


def test_selecting_waves_arms_and_host_classes(manifest):
    wave3 = cr_queue.select_arms(manifest, wave="3", arm_ids=None, host_class="gpu")
    assert {a["arm"] for a in wave3} == {"turbo_box_faithful", "rembo", "hesbo",
                                        "cmaes_full_rd", "cmaes_full_rd_pop5", "baxus_cr"}
    wave4 = cr_queue.select_arms(manifest, wave="4", arm_ids=None, host_class="gpu")
    assert [a["arm"] for a in wave4] == ["papenmeier_msr_published_norm_box"]
    cpu = cr_queue.select_arms(manifest, wave=None, arm_ids=None, host_class="cpu")
    assert {a["arm"] for a in cpu} == {"gabo", "matern_gabo"}
    twice = cr_queue.select_arms(manifest, wave=None, host_class="gpu",
                                 arm_ids=["geowalk", "geowalk", "cma_es_subspace_sd8"])
    assert [a["arm"] for a in twice] == ["geowalk", "cma_es_subspace_sd8"]
    with pytest.raises(SystemExit, match="not in the manifest"):
        cr_queue.select_arms(manifest, wave=None, arm_ids=["no_such_arm"], host_class="all")
    with pytest.raises(SystemExit, match="not gpu-class"):
        cr_queue.select_arms(manifest, wave=None, arm_ids=["gabo"], host_class="gpu")


def test_build_cells_runs_every_arm_on_its_own_seed_range(tmp_path):
    man = _mini_manifest([_mini_arm("a"), {**_mini_arm("a__s82"), "solver": "a",
                                           "seeds": [82, 83], "wave": "c2"}])
    cr_queue.validate_manifest(man)
    cells, _ = cr_queue.build_cells(
        man, man["arms"], tasks=["rano"], seeds=None, output_root=str(tmp_path),
        retry_crashed=False, interleave=False, expect_commit=FAKE_SHA)
    assert [(c["arm"], c["solver"], c["seed"]) for c in cells] == [
        ("a", "a", 42), ("a", "a", 43), ("a__s82", "a", 82), ("a__s82", "a", 83)]
    assert {c["out"].parent.name for c in cells} == {"a", "a__s82"}      # own directories
    assert cells[2]["out"].name == "ours_a_ranolazine_mpo_s82_128d.json"
    # --seeds narrows inside every selected range ...
    only, _ = cr_queue.build_cells(
        man, man["arms"][1:], tasks=["rano"], seeds=[83], output_root=str(tmp_path),
        retry_crashed=False, interleave=False, expect_commit=FAKE_SHA)
    assert [(c["arm"], c["seed"]) for c in only] == [("a__s82", 83)]
    # ... and is refused when a seed falls outside a selected arm's range
    with pytest.raises(SystemExit, match=r"--seeds 42 outside the seed range of arm 'a__s82'"):
        cr_queue.build_cells(
            man, man["arms"], tasks=["rano"], seeds=[42], output_root=str(tmp_path),
            retry_crashed=False, interleave=False, expect_commit=FAKE_SHA)


def test_cell_output_is_absolute_per_arm_and_tags_overrides(manifest, tmp_path):
    by_id = {a["arm"]: a for a in manifest["arms"]}
    default = cr_queue.cell_output(by_id["turbo_box_faithful"], "rano", 42, manifest, None)
    assert default == (REPO / "results/guacamol/cr2026/turbo_box_faithful/"
                       "ours_turbo_box_faithful_ranolazine_mpo_s42_128d.json")
    smoke = cr_queue.cell_output(by_id["cma_es_subspace_sd8"], "adip", 43, manifest,
                                 "results/guacamol/cr_smoke2")
    assert smoke == (REPO / "results/guacamol/cr_smoke2/cma_es_subspace_sd8/"
                     "ours_cma_es_subspace_amlodipine_mpo_s43_128d_sd8.json")
    absolute = cr_queue.cell_output(by_id["geowalk"], "rano", 42, manifest, str(tmp_path))
    assert absolute.parent == tmp_path / "geowalk"


def _cell(**kw):
    base = {"arm": "ok_a", "solver": "ok_a", "poli_task": "ranolazine_mpo", "seed": 42,
            "subspace_dim": 16, "run_overrides": {}}
    base.update(kw)
    return base


def _result(**kw):
    res = {"status": "completed", "n_evaluated": 30, "all_scores": [0.1] * 30, "budget": 30,
           "solver": "ok_a", "task": "ranolazine_mpo", "seed": 42, "subspace_dim": 16,
           "provenance": {"git_commit": FAKE_SHA[:7], "git_commit_full": FAKE_SHA,
                          "git_dirty": False, "cr_arm": "ok_a", "run_overrides": {}}}
    for k, v in kw.items():
        if k.startswith("prov_"):
            res["provenance"][k[5:]] = v
        else:
            res[k] = v
    return res


@pytest.mark.parametrize("res,needle", [
    (_result(status="crashed"), "status="),
    (_result(n_evaluated=18), "n_evaluated="),
    (_result(all_scores=[0.1] * 29), "len(all_scores)"),
    (_result(budget=1000), "budget="),
    (_result(seed=43), "(solver, task, seed, subspace_dim)"),
    (_result(subspace_dim=8), "(solver, task, seed, subspace_dim)"),
    (_result(prov_cr_arm="other"), "cr_arm"),
    (_result(prov_run_overrides={"subspace_dim": 8}), "run_overrides"),
    (_result(prov_git_commit_full="f" * 40), "commit"),
    (_result(prov_git_commit=""), "commit"),
    (_result(prov_git_dirty=True), "git_dirty"),
    (_result(prov_git_dirty=None), "git_dirty"),
])
def test_validate_result_catches_every_problem(res, needle):
    problems = cr_queue.validate_result(res, _cell(), expect_commit=FAKE_SHA)
    assert problems and any(needle in p for p in problems), problems


def test_validate_result_accepts_a_good_file_and_skips_commit_checks_when_dirty_allowed():
    assert cr_queue.validate_result(_result(), _cell(), expect_commit=FAKE_SHA) == []
    dirty = _result(prov_git_dirty=True, prov_git_commit_full="f" * 40)
    assert cr_queue.validate_result(dirty, _cell(), expect_commit=None) == []


def test_archive_attempt_never_overwrites(tmp_path):
    out = tmp_path / "arm" / "cell.json"
    out.parent.mkdir()
    for round_ in (1, 2):
        out.write_text(json.dumps({"status": "crashed", "round": round_}))
        cr_queue.sidecar_path(out).write_text(json.dumps({"round": round_}))
        moved = cr_queue.archive_attempt(out)
        assert sorted(Path(m).name for m in moved) == [
            f"cell.json.attempt{round_}.crash.json", f"cell.json.attempt{round_}.json",
        ]
    att = out.parent / "attempts"
    assert json.loads((att / "cell.json.attempt1.json").read_text())["round"] == 1
    assert json.loads((att / "cell.json.attempt2.json").read_text())["round"] == 2
    assert not out.exists() and not cr_queue.sidecar_path(out).exists()


def _mini_manifest(arms: list[dict]) -> dict:
    return {"tasks": ["rano"], "task_poli_ids": {"rano": "ranolazine_mpo"}, "seeds": [42, 43],
            "output_root": "results/x", "gpu_hosts": ["h"], "cpu_hosts": ["h"], "arms": arms}


def _mini_arm(arm: str, wave=1) -> dict:
    return {"arm": arm, "solver": arm, "run_overrides": {}, "wave": wave, "host_class": "gpu",
            "timeout_s": 60}


def _select(man, tmp_path, *, retry=False, seeds=(42,), group_by_wave=False, arms=None):
    return cr_queue.build_cells(
        man, arms or man["arms"], tasks=["rano"], seeds=list(seeds), output_root=str(tmp_path),
        retry_crashed=retry, interleave=True, expect_commit=FAKE_SHA, group_by_wave=group_by_wave)


@pytest.mark.parametrize("retry", [False, True])
def test_a_crash_record_next_to_a_valid_result_counts_as_completed(tmp_path, retry):
    """N1: a child that published a valid result and then died left a record;
    the cell is completed (with a warning) and never retried (R-5)."""
    man = _mini_manifest([_mini_arm("ok_a")])
    out = cr_queue.cell_output(man["arms"][0], "rano", 42, man, str(tmp_path))
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps(_result()))
    cr_queue.sidecar_path(out).write_text(json.dumps({"kind": "hard_crash"}))
    cells, stats = _select(man, tmp_path, retry=retry)
    assert cells == []
    assert stats["completed"] == 1 and stats["crashed_kept"] == stats["crashed_retry"] == 0
    assert stats["completed_with_crash_record"] == [str(out)]


@pytest.mark.parametrize("with_record,retry,expected", [
    (False, False, "foreign"), (False, True, "foreign"),
    (True, False, "crashed_kept"), (True, True, "crashed_retry"),
])
def test_an_invalid_completed_file_is_foreign_or_with_a_record_terminal(
        tmp_path, with_record, retry, expected):
    man = _mini_manifest([_mini_arm("ok_a")])
    out = cr_queue.cell_output(man["arms"][0], "rano", 42, man, str(tmp_path))
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps(_result(prov_git_dirty=True)))
    if with_record:
        cr_queue.sidecar_path(out).write_text(json.dumps({"kind": "invalid"}))
    cells, stats = _select(man, tmp_path, retry=retry)
    assert stats["completed"] == 0 and not stats["completed_with_crash_record"]
    if expected == "foreign":
        assert len(stats["foreign"]) == 1 and cells == []
    elif expected == "crashed_kept":
        assert stats["crashed_kept"] == 1 and cells == []
    else:
        assert stats["crashed_retry"] == 1 and [c["retry"] for c in cells] == [True]


def test_waves_run_in_the_given_order_and_interleave_within_a_wave(tmp_path):
    man = _mini_manifest([_mini_arm("a1", 1), _mini_arm("a2", 1), _mini_arm("b1", 2),
                          _mini_arm("b2", 2), _mini_arm("c1", "3b")])
    arms = cr_queue.select_arms(man, wave=None, arm_ids=None, host_class="gpu",
                                waves=["2", "1", "3b"])
    assert [a["arm"] for a in arms] == ["b1", "b2", "a1", "a2", "c1"]
    cells, _ = _select(man, tmp_path, seeds=(42, 43), group_by_wave=True, arms=arms)
    assert [(c["arm"], c["seed"]) for c in cells] == [
        ("b1", 42), ("b2", 42), ("b1", 43), ("b2", 43),
        ("a1", 42), ("a2", 42), ("a1", 43), ("a2", 43),
        ("c1", 42), ("c1", 43),
    ]
    mixed, _ = _select(man, tmp_path, seeds=(42, 43), group_by_wave=False, arms=arms)
    assert [c["arm"] for c in mixed[:5]] == ["b1", "b2", "a1", "a2", "c1"]   # one round-robin
    only = cr_queue.select_arms(man, wave=None, arm_ids=None, host_class="gpu", waves=["3b"])
    assert [a["arm"] for a in only] == ["c1"]
    with pytest.raises(SystemExit, match="unknown wave"):
        cr_queue.select_arms(man, wave=None, arm_ids=None, host_class="gpu", waves=["5"])


def test_the_circuit_breaker_counts_timeouts_and_solver_crashes(tmp_path):
    runner = cr_queue.Runner(None, tmp_path, tmp_path, expect_commit=None)
    for status in ("timeout", "crashed_in_solver", "ok", "crashed", "failed_start"):
        runner.breaker(status)
    assert not runner.stop.is_set()                         # "ok" reset the count
    for status in ("interrupted", "timeout", "crashed_in_solver"):
        runner.breaker(status)
    assert not runner.stop.is_set()
    runner.breaker("timeout")
    assert runner.stop.is_set() and "3 failed cells in a row" in runner.stop_reason
    assert cr_queue.BREAKER_STATUSES == {"crashed", "timeout", "crashed_in_solver",
                                         "failed_start"}


def test_run_cell_refuses_a_cell_with_a_crash_record(tmp_path, monkeypatch, capsys):
    """N6: a manual run cannot rerun a crashed cell behind the queue's back."""
    out = tmp_path / "ours_geowalk_ranolazine_mpo_s42_128d.json"
    record = cr_run_cell.crash_record_path(out)
    record.write_text(json.dumps({"kind": "hard_crash"}))
    assert cr_queue.sidecar_path(out) == record
    calls = []
    import scripts.run_benchmark as rb
    monkeypatch.setattr(rb, "run_benchmark", _fake_run_benchmark(calls))
    rc = cr_run_cell.main(["geowalk", "ranolazine_mpo", "42", str(out)])
    assert rc == cr_run_cell.EXIT_EXISTS_NOT_COMPLETED == 77
    assert calls == [] and not out.exists() and not cr_run_cell.claim_path(out).exists()
    assert "crash record" in capsys.readouterr().out
    assert json.loads(record.read_text()) == {"kind": "hard_crash"}


def test_a_stale_claim_never_overwrites_an_earlier_crash_record(tmp_path):
    out = tmp_path / "arm" / "ours_geowalk_ranolazine_mpo_s42_128d.json"
    out.parent.mkdir()
    cr_queue.sidecar_path(out).write_text(json.dumps({"kind": "hard_crash",
                                                      "output_tail": "first traceback"}))
    cr_run_cell.claim_path(out).write_text("999999 somehost then\n")
    logs: list[str] = []
    cr_queue.resolve_stale_claims([{"out": str(out), "claim": "999999 somehost then"}],
                                  log=logs.append)
    new = json.loads(cr_queue.sidecar_path(out).read_text())
    assert new["kind"] == "stale_claim"
    previous = Path(new["archived_previous_crash_record"])
    assert previous == out.parent / "attempts" / f"{out.name}.attempt1.crash.json"
    assert json.loads(previous.read_text())["output_tail"] == "first traceback"
    assert not cr_run_cell.claim_path(out).exists()
    assert "earlier crash record was archived" in logs[0]
    # a later retry archives the stale-claim record as attempt 2, never over attempt 1
    assert [Path(m).name for m in cr_queue.archive_attempt(out)] == [
        f"{out.name}.attempt2.crash.json"]


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_terminate_all_kills_and_marks_only_the_children_still_running(tmp_path):
    """N7: a child that already exited (e.g. crashed on its own) is not marked,
    so its worker records what it did; a running child is killed and marked."""
    runner = cr_queue.Runner(None, tmp_path, tmp_path, expect_commit=None)
    exited = subprocess.Popen([sys.executable, "-c", "import sys; sys.exit(1)"],
                              start_new_session=True)
    running = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                               start_new_session=True)
    try:
        exited.wait(timeout=30)
        runner.children = {exited.pid: (exited, None), running.pid: (running, None)}
        runner.terminate_all(signal.SIGTERM)
        assert runner.killed_pids == {running.pid}
        assert running.poll() == -signal.SIGTERM and exited.returncode == 1
        assert runner.signal_received == signal.SIGTERM and runner.stop.is_set()
        assert -signal.SIGTERM in cr_queue.KILL_RETURNCODES and 1 not in cr_queue.KILL_RETURNCODES
    finally:
        if running.poll() is None:
            running.kill()
            running.wait()


def test_child_environment_pins_gpu_threads_tmp_and_caches(tmp_path):
    env = cr_queue.child_env(repo=tmp_path, cuda_devices="1", threads=4,
                             tmpdir="/data/prusek/tmp")
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["PYTHONPATH"] == str(tmp_path)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        assert env[var] == "4", var
    assert env["TMPDIR"] == "/data/prusek/tmp"
    assert env["CUDA_CACHE_PATH"] == "/data/prusek/tmp/cuda_cache"  # not $HOME/.nv on NFS
    assert env["GIT_OPTIONAL_LOCKS"] == "0"
    cpu = cr_queue.child_env(repo=tmp_path, cuda_devices="", threads=4, tmpdir=None)
    assert cpu["CUDA_VISIBLE_DEVICES"] == ""
    assert cpu["CUDA_CACHE_PATH"].startswith(str(tmp_path))


def test_crash_sidecar_records_kind_log_amendment_and_traceback(tmp_path):
    cell = {"arm": "turbo_box_faithful", "solver": "turbo_box_faithful", "run_overrides": {},
            "task": "rano", "poli_task": "ranolazine_mpo", "seed": 42, "wave": 3,
            "timeout_s": 21600,
            "out": tmp_path / "ours_turbo_box_faithful_ranolazine_mpo_s42_128d.json"}
    side = cr_queue.write_crash_sidecar(
        cell, kind="hard_crash", returncode=1, output="Traceback...\nRuntimeError: boom",
        started="2026-10-11T19:00:00", elapsed=12.5, cmd=["python", "x"],
        cuda_devices="0", log_path=tmp_path / "x.log", amendment="2026-10-12: device bug",
        problems=["status='crashed'"],
    )
    rec = json.loads(side.read_text())
    assert side.name.endswith(".json.crash.json")
    assert rec["kind"] == "hard_crash" and rec["returncode"] == 1 and rec["timed_out"] is False
    assert "RuntimeError: boom" in rec["output_tail"]
    assert rec["log"] == str(tmp_path / "x.log")
    assert rec["amendment"] == "2026-10-12: device bug"
    assert rec["problems"] == ["status='crashed'"]
    assert stat.S_IMODE(side.stat().st_mode) == 0o644
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_host_guard(manifest):
    assert cr_queue.host_allowed(manifest, "gpu", "panda.utia.cas.cz")
    assert cr_queue.host_allowed(manifest, "gpu", "panda")
    assert not cr_queue.host_allowed(manifest, "gpu", "kajman.utia.cas.cz")
    assert not cr_queue.host_allowed(manifest, "gpu", "tulen.utia.cas.cz")
    assert cr_queue.host_allowed(manifest, "cpu", "kajman.utia.cas.cz")
    assert cr_queue.host_allowed(manifest, "cpu", "kajman")
    assert not cr_queue.host_allowed(manifest, "cpu", "panda.utia.cas.cz")   # R-2, 2026-10-12
    assert not cr_queue.host_allowed(manifest, "cpu", "tulen.utia.cas.cz")


def test_tmpdir_guard(tmp_path):
    assert cr_queue.tmpdir_problem(None)
    assert cr_queue.tmpdir_problem("/tmp")
    assert cr_queue.tmpdir_problem("/var/tmp")
    assert cr_queue.tmpdir_problem(str(tmp_path)) is None


def test_input_hashes_cover_the_required_inputs(fake_repo):
    repo, _ = fake_repo
    hashes = cr_queue.input_hashes(repo)
    for rel in cr_queue.REQUIRED_INPUTS:
        assert len(hashes[rel]) == 64
    (repo / "weights" / "SELFIES-VAE-state-dict.pt").unlink()
    assert cr_queue.input_hashes(repo)["weights/SELFIES-VAE-state-dict.pt"] == "MISSING"


def test_no_module_in_scripts_cr_shadows_an_importable_module():
    """The old ``scripts/cr/queue.py`` shadowed the standard library's ``queue``
    for every script launched from that directory (torch died in ``from queue
    import Queue``). No file there may share a name with an importable module."""
    import importlib.util

    cr_dir = REPO / "scripts" / "cr"
    saved = list(sys.path)
    try:
        sys.path[:] = [p for p in sys.path
                       if os.path.abspath(p or os.curdir) != str(cr_dir)]
        for path in cr_dir.glob("*.py"):
            assert importlib.util.find_spec(path.stem) is None, path.name
    finally:
        sys.path[:] = saved
    assert not (cr_dir / "queue.py").exists()


def test_logs_go_under_a_gitignored_directory():
    assert cr_queue.LOG_ROOT == Path("logs") / "cr"
    gitignore = (REPO / ".gitignore").read_text()
    assert "logs/" in gitignore.splitlines()
    # results/ is ignored as a directory, or by its contents (results/*) so that tracked
    # files below it can be re-included
    assert "results/" in gitignore.splitlines() or "results/*" in gitignore.splitlines()
    assert os.fspath(cr_queue.RUN_CELL) == "scripts/cr/run_cell.py"


def test_the_committed_rules_carry_the_amendment():
    rules = (REPO / "docs" / "camera_ready" / "RUN_RULES.md").read_text()
    assert "## 4. Amendment log" in rules
    for arm_id in AMENDMENT_ARMS:
        assert f"`{arm_id}`" in rules, arm_id
    assert "Exception to R-1 for `cma_es_subspace_sd8` and `cma_es_subspace_sd64`" in rules
    rows = [line for line in rules.splitlines() if line.startswith("| 2026-")]
    # GEN v5 (backlog items 6/10): the committed copy is the binding file byte for byte, 26 amendment rows up to the
    # row of 2026-11-01 (Tables 1 to 3 printed in the layout of the submitted version)
    assert len(rows) == 32 and sum(r.startswith("| 2026-10-12 |") for r in rows) == 9
    assert sum(r.startswith("| 2026-11-01 |") for r in rows) == 1 and sum(r.startswith("| 2026-11-03 |") for r in rows) == 1
    assert sum(r.startswith("| 2026-10-19 |") for r in rows) == 3 and sum(r.startswith("| 2026-10-22 |") for r in rows) == 2
    assert sum(r.startswith("| 2026-10-24 |") for r in rows) == 1 and sum(r.startswith("| 2026-10-29 |") for r in rows) == 1
    assert "first refinement angle 0.30 rad, ×1.2 after an improving step" in rules
    assert "CPU-class arms (`gabo`, `matern_gabo`) run only on kajman" in rules
    # the control block (seventh row): C1, C2a/C2b and the six C3 arms on seeds 82-121
    ctl = rows[6]
    assert ctl.startswith("| 2026-10-12 | **Control block**")
    for arm_id in ("hvarfner_scaled_gp_box_argmax_raw", "hvarfner_scaled_gp_boxdir_argmax",
                   "random_boxdir", *DECOMPOSITION_ARMS):
        assert f"`{arm_id}`" in ctl, arm_id
    assert "seeds 82–121" in ctl
    # the later rows of the published-recipe block, the scope and the end of the verification
    assert any(r.startswith("| 2026-10-12 | **Published-recipe block and baseline print rule**") for r in rows)
    assert any(r.startswith("| 2026-10-14 | **Scope of the camera-ready reduced to the promised content**") for r in rows)
    assert rows[-15].startswith("| 2026-10-19 | **Verification ends with the recipes whose runs are complete**")
    assert rows[-14].startswith("| 2026-10-19 | **Clarifications for the final generator run**")
    assert "but by a smaller margin" in rows[-14]
    assert rows[-13].startswith("| 2026-10-19 | **Clarifications for the final generator run, second set**")
    assert '"{reference} is ahead of {recipe}"' in rows[-13] and "rounded to the printed precision" in rows[-13]
    assert rows[-12].startswith("| 2026-10-22 | **Wording of the published-recipe sentences**")
    assert 'The words "not worse" are not printed.' in rows[-12]
    assert rows[-11].startswith("| 2026-10-22 | **Generator edits after the final run are limited to format, labels and wording**")
    assert "every numeric macro of the final run is byte-identical" in rows[-11]
    assert rows[-10].startswith("| 2026-10-24 | **Appendix H prints an overview; the baseline cards and the run-status "
                               "table become repository records; two descriptive tables are added**")
    assert "`docs/baselines/`" in rows[-10] and "`numbers.tex` except added macros" in rows[-10]
    assert "(4) Docstrings and comments of the released code are brought up to date with the paper" in rows[-10]
    assert rows[-9].startswith("| 2026-10-29 | **The printed tables are cut to what the text needs, one promised "
                               "trajectory is added, and captions are corrected**")
    assert "every numeric macro of the final run is byte-identical" in rows[-9]
    assert 'promise C-32: "add Linear BO and its full trajectory"' in rows[-9]
    assert rows[-8].startswith("| 2026-11-01 | **Tables 1 to 3 are printed in the layout of the submitted version**")
    assert "every numeric macro of the final run is byte-identical" in rows[-8]
    assert rows[-7].startswith("| 2026-11-02 | **The paper states no history")
    assert rows[-6].startswith("| 2026-11-03 | **A post-hoc task-level test of random search on the shell")
    assert "post_hoc_random_search" in rows[-6] and "No generated number changes" in rows[-6]
    assert rows[-5].startswith("| 2026-11-04 | **Control C4 is read, as its row of 2026-10-14 prescribes")
    assert "c4_reading" in rows[-5] and "No generated number changes" in rows[-5]
    assert rows[-4].startswith("| 2026-11-04 15:30 | **The Linear BO port on the shell leaves the printed Table 1")
    assert "C-33" in rows[-4] and "439 of the 440 cells" in rows[-4] and "No generated number changes" in rows[-4]
    assert rows[-3].startswith("| 2026-11-06 | **The GP on the shell leaves the printed Table 1**")
    assert "Table 2 beside its box version" in rows[-3] and "No generated number changes" in rows[-3]
    assert rows[-2].startswith("| 2026-11-06 16:30 | **L-BFGS-B is the continuous maximiser on both supports of the printed factorial**")
    assert "`hvarfner_scaled_gp_sphere_lbfgs`" in rows[-2] and "root `CRC4`" in rows[-2] and "2a899ad" in rows[-2]
    assert "cell 0.3999" in rows[-2] and "interaction +0.0294, 9 of 11, p = 0.0049" in rows[-2]
    assert rows[-1].startswith("| 2026-11-06 21:30 | **The TuRBO-1 re-implementation leaves the printed appendix**")
    assert "`turbo_box_faithful`" in rows[-1] and "GP in the box" in rows[-1] and "every printed value is unchanged" in rows[-1]
    import hashlib
    assert hashlib.sha256((REPO / "docs" / "camera_ready" / "RUN_RULES.md").read_bytes()).hexdigest() == (
        "f93380a6a1971ab981a4c8fba1a746bc56b64bee6ffd84274972074d29864907")


def test_the_generator_docs_carry_the_reading_rules():
    docs = (REPO / "docs" / "camera_ready" / "README.md").read_text()
    for needle in ("ours_*_128d(_sd[0-9]+)?.json", "*.crash.json", "*.interrupted.json",
                   "*.collision-*.json", "attempts/", "provenance.cr_arm"):
        assert needle in docs, needle
    assert "ours_*_128d(_sd[0-9]+)?.json" in cr_queue.__doc__
    # the control block: its own root, waves and the <arm>__s82 directories
    for needle in ("results/guacamol/cr2026ctl", "--waves c1,c2", "cr-controls-2026-10-12",
                   *CONTROL_ARMS):
        assert needle in docs, needle
    # control C4: its own tag and wave, the printed shell x continuous cell since 2026-11-06
    for needle in ("--waves c3", "cr-controls-lbfgs-2026-10-14", *C4_ARMS, "RUN_RULES 2026-11-06 16:30"):
        assert needle in docs, needle


# =========================================================================
# queue end to end (real cr_queue.py, throwaway repo, fake child)
# =========================================================================

FAKE_CHILD = r'''
import json, os, signal, socket, sys, time, pathlib
argv = sys.argv[1:]
_run_cell, solver, task, seed, out = argv[:5]
opts = dict(zip(argv[5::2], argv[6::2]))
out = pathlib.Path(out)
expect = opts.get("--expect-commit")
arm = opts["--arm"]
overrides = json.loads(opts["--run-overrides"])
claim = pathlib.Path(str(out) + ".running")
out.parent.mkdir(parents=True, exist_ok=True)
fd = os.open(claim, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
os.write(fd, f"{os.getpid()} {socket.gethostname()} now\n".encode())
os.close(fd)
try:
    if solver == "hardcrash" and not os.environ.get("FAKE_FIX"):
        raise RuntimeError(f"boom pid={os.getpid()}")
    if solver == "hang" or os.environ.get("FAKE_HANG"):
        time.sleep(600)
    if solver == "termcrash":
        # ignores the queue's SIGTERM and crashes on its own a little later
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        time.sleep(3)
        raise RuntimeError("crashed on its own while the queue was stopping")
    full = "f" * 40 if solver == "badprov" else (expect or os.environ.get("FAKE_HEAD", ""))
    res = {
        "status": "crashed" if solver == "solvercrash" else "completed",
        "crash_info": "RuntimeError: in solver" if solver == "solvercrash" else None,
        "solver": solver, "task": task, "seed": int(seed),
        "subspace_dim": overrides.get("subspace_dim", 16),
        "n_evaluated": 30, "all_scores": [0.1] * 30, "budget": 30, "best_score": 0.5,
        "elapsed_seconds": 0.1, "surrogate_fallback_count": 0, "n_oracle_failures": 0,
        "provenance": {"git_commit": full[:7], "git_commit_full": full, "git_dirty": False,
                       "cr_arm": arm, "run_overrides": overrides},
    }
    tmp = out.parent / (out.name + ".tmp")
    tmp.write_text(json.dumps(res))
    os.link(tmp, out)
    os.unlink(tmp)
    print("DONE fake", solver, task, seed, flush=True)
    if solver == "teardowncrash":          # published, then dies in "interpreter teardown"
        claim.unlink()
        os._exit(1)
    if solver == "teardownhang":           # published, then hangs past its timeout
        claim.unlink()
        time.sleep(600)
finally:
    try:
        claim.unlink()
    except OSError:
        pass
'''


def _test_manifest(tmp_path, *, gpu_hosts=None) -> Path:
    host = socket.gethostname()
    arms = [
        {"arm": a, "solver": a, "run_overrides": {}, "wave": 1, "host_class": "gpu",
         "timeout_s": 3 if a in ("hang", "teardownhang") else 120}
        for a in ("ok_a", "ok_b", "solvercrash", "hardcrash", "hang", "badprov",
                  "teardowncrash", "teardownhang", "termcrash")
    ]
    for arm in arms:
        if arm["arm"] in ("hang", "solvercrash"):   # the breaker tests run 4 seeds
            arm["seeds"] = [42, 43, 44, 45]
    arms.append({"arm": "ok_sd8", "solver": "ok_c", "run_overrides": {"subspace_dim": 8},
                 "wave": 1, "host_class": "gpu", "timeout_s": 120})
    # ok_a's computation on a disjoint seed range, in its own directory (control block style)
    arms.append({"arm": "ok_a__s82", "solver": "ok_a", "run_overrides": {}, "wave": "c2",
                 "host_class": "gpu", "timeout_s": 120, "seeds": [82, 83]})
    man = {"gpu_hosts": gpu_hosts or [host], "cpu_hosts": [host], "tasks": ["rano"],
           "task_poli_ids": {"rano": "ranolazine_mpo"}, "seeds": [42, 43],
           "output_root": "results/cr_test", "arms": arms}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(man))
    return path


@pytest.fixture
def harness(tmp_path, fake_repo):
    repo, head = fake_repo
    child = tmp_path / "fake_python.py"
    child.write_text("#!" + sys.executable + "\n" + FAKE_CHILD)
    child.chmod(0o755)
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()
    manifest_path = _test_manifest(tmp_path)
    runs = {"n": 0}

    def run(*extra, cwd=None, env=None, expect=head, popen=False, manifest_path=manifest_path):
        runs["n"] += 1
        log_dir = tmp_path / f"logs{runs['n']}"
        args = [sys.executable, str(REPO / "scripts" / "cr" / "cr_queue.py"),
                "--repo", str(repo), "--manifest", str(manifest_path),
                "--python", str(child), "--tmpdir", str(tmpdir), "--stagger", "0",
                "--gpus", "0", "--workers-per-gpu", "2", "--log-dir", str(log_dir),
                "--tasks", "rano"]
        if expect:
            args += ["--expect-commit", expect]
        args += list(extra)
        full_env = dict(os.environ)
        full_env.update(env or {})
        if popen:
            return subprocess.Popen(args, cwd=str(cwd or repo), env=full_env,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True), log_dir
        proc = subprocess.run(args, cwd=str(cwd or repo), env=full_env,
                              capture_output=True, text=True, timeout=300)
        return proc, log_dir

    root = repo / "results" / "cr_test"
    return {"run": run, "repo": repo, "head": head, "root": root, "tmp": tmp_path}


def _cells(log_dir: Path) -> list[dict]:
    path = log_dir / "cells.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def _out(root: Path, arm: str, solver: str, seed: int, sd: int = 16) -> Path:
    return root / arm / cr_run_cell.cell_filename(solver, "ranolazine_mpo", seed, sd)


def test_e2e_ok_cells_are_validated_logged_and_never_rerun(harness):
    proc, log_dir = harness["run"]("--arms", "ok_a,ok_sd8,ok_a", "--seeds", "42,43,42")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    root = harness["root"]
    for seed in (42, 43):
        assert _out(root, "ok_a", "ok_a", seed).exists()
        sd8 = _out(root, "ok_sd8", "ok_c", seed, 8)
        assert sd8.exists() and sd8.name.endswith("_128d_sd8.json")
    cells = _cells(log_dir)
    assert len(cells) == 4                                   # de-duplicated selection
    assert {c["status"] for c in cells} == {"ok"}
    assert {c["git_commit_full"] for c in cells} == {harness["head"]}
    summary = json.loads((log_dir / "summary.json").read_text())
    assert summary["status"] == "done" and summary["counts"]["ok"] == 4
    assert summary["expect_commit"] == summary["head"] == harness["head"]
    assert set(cr_queue.REQUIRED_INPUTS) <= set(summary["inputs_sha256"])
    assert "nvidia_smi_start" in summary and "who_start" in summary
    assert len(summary["manifest_sha256"]) == 64
    assert (log_dir / "ok_a__rano__s42.log").exists()
    assert not list(root.rglob("*.running"))

    again, _ = harness["run"]("--arms", "ok_a,ok_sd8", "--seeds", "42-43")
    assert again.returncode == 0
    assert "cells to run: 0" in again.stdout and "completed (valid): 4" in again.stdout


def test_e2e_crashes_and_timeouts_are_terminal_and_keep_the_first_traceback(harness):
    root = harness["root"]
    proc, log_dir = harness["run"]("--arms", "solvercrash,hardcrash,hang", "--seeds", "42")
    assert proc.returncode == cr_queue.EXIT_FAILURES, proc.stdout + proc.stderr
    kinds = {}
    for arm in ("solvercrash", "hardcrash", "hang"):
        side = cr_queue.sidecar_path(_out(root, arm, arm, 42))
        kinds[arm] = json.loads(side.read_text())
    assert kinds["solvercrash"]["kind"] == "crashed_in_solver"
    assert kinds["hardcrash"]["kind"] == "hard_crash"
    assert kinds["hang"]["kind"] == "timeout" and kinds["hang"]["timed_out"] is True
    first_tb = kinds["hardcrash"]["output_tail"]
    assert "RuntimeError: boom" in first_tb
    assert {c["status"] for c in _cells(log_dir)} == {"crashed_in_solver", "crashed", "timeout"}
    assert not list(root.rglob("*.running"))                 # the timeout kill dropped the claim

    again, log2 = harness["run"]("--arms", "solvercrash,hardcrash,hang", "--seeds", "42")
    assert again.returncode == 0, again.stdout + again.stderr
    assert "cells to run: 0" in again.stdout and "crashed kept: 3" in again.stdout
    side = cr_queue.sidecar_path(_out(root, "hardcrash", "hardcrash", 42))
    assert json.loads(side.read_text())["output_tail"] == first_tb   # not overwritten

    no_amendment, _ = harness["run"]("--arms", "hardcrash", "--seeds", "42", "--retry-crashed")
    assert no_amendment.returncode == 2 and "--amendment" in no_amendment.stderr

    retry, log3 = harness["run"]("--arms", "hardcrash", "--seeds", "42", "--retry-crashed",
                                 "--amendment", "2026-10-12: test retry",
                                 env={"FAKE_FIX": "1"})
    assert retry.returncode == 0, retry.stdout + retry.stderr
    assert _out(root, "hardcrash", "hardcrash", 42).exists()
    attempts = root / "hardcrash" / "attempts"
    archived = json.loads((attempts / (_out(root, "hardcrash", "hardcrash", 42).name
                                       + ".attempt1.crash.json")).read_text())
    assert archived["output_tail"] == first_tb
    (rec,) = _cells(log3)
    assert rec["status"] == "ok" and rec["retry"] is True
    assert rec["amendment"] == "2026-10-12: test retry"
    assert "amendment: 2026-10-12: test retry" in (log3 / "queue.log").read_text()


def test_e2e_invalid_provenance_is_terminal_and_stops_the_queue(harness):
    proc, log_dir = harness["run"]("--arms", "badprov", "--seeds", "42",
                                   "--workers-per-gpu", "1")
    assert proc.returncode == cr_queue.EXIT_FAILURES
    side = json.loads(cr_queue.sidecar_path(_out(harness["root"], "badprov", "badprov", 42))
                      .read_text())
    assert side["kind"] == "invalid" and any("commit" in p for p in side["problems"])
    summary = json.loads((log_dir / "summary.json").read_text())
    assert "invalid provenance" in summary["stop_reason"]


def test_e2e_a_foreign_completed_file_refuses_the_start(harness):
    out = _out(harness["root"], "ok_a", "ok_a", 42)
    out.parent.mkdir(parents=True)
    res = _result(prov_git_commit_full="f" * 40, prov_git_commit="fffffff")
    res["provenance"]["cr_arm"] = "ok_a"
    out.write_text(json.dumps(res))
    proc, _ = harness["run"]("--arms", "ok_a", "--seeds", "42")
    assert proc.returncode == cr_queue.EXIT_FOREIGN, proc.stdout + proc.stderr
    assert "FOREIGN" in proc.stdout
    assert json.loads(out.read_text())["provenance"]["git_commit"] == "fffffff"


def test_e2e_runs_identically_from_another_directory(harness, tmp_path):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    crashed = _out(harness["root"], "ok_b", "ok_b", 42)
    crashed.parent.mkdir(parents=True)
    crashed.write_text(json.dumps({"status": "crashed"}))
    proc, _ = harness["run"]("--arms", "ok_b", "--seeds", "42-43",
                             "--output-root", "results/cr_test", cwd=elsewhere)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "crashed kept: 1" in proc.stdout
    assert json.loads(crashed.read_text()) == {"status": "crashed"}      # kept, not rerun
    assert _out(harness["root"], "ok_b", "ok_b", 43).exists()            # inside the repo
    assert not list(elsewhere.rglob("*.json"))


def test_e2e_one_queue_per_output_root(harness):
    root = harness["root"]
    root.mkdir(parents=True)
    with open(root / cr_queue.LOCK_NAME, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fh.write("99999 otherqueue now\n")
        fh.flush()
        proc, _ = harness["run"]("--arms", "ok_a", "--seeds", "42")
    assert proc.returncode == cr_queue.EXIT_LOCKED, proc.stdout + proc.stderr
    assert not _out(root, "ok_a", "ok_a", 42).exists()


def test_e2e_refuses_a_dirty_tree_a_wrong_head_and_a_wrong_host(harness, tmp_path):
    proc, _ = harness["run"]("--arms", "ok_a", "--seeds", "42", expect=FAKE_SHA)
    assert proc.returncode == cr_queue.EXIT_DIRTY and "expected" in proc.stderr
    (harness["repo"] / "stray.py").write_text("x = 1\n")
    proc, _ = harness["run"]("--arms", "ok_a", "--seeds", "42")
    assert proc.returncode == cr_queue.EXIT_DIRTY and "dirty" in proc.stderr
    (harness["repo"] / "stray.py").unlink()
    m2 = tmp_path / "m2"
    m2.mkdir()
    other = _test_manifest(m2, gpu_hosts=["some-other-host.example"])
    proc, _ = harness["run"]("--arms", "ok_a", "--seeds", "42", manifest_path=other)
    assert proc.returncode == cr_queue.EXIT_HOST, proc.stdout + proc.stderr
    proc, _ = harness["run"]("--arms", "ok_a", "--seeds", "42", expect=None)
    assert proc.returncode == 2 and "--expect-commit" in proc.stderr


def test_e2e_a_dead_claim_becomes_a_terminal_crash_record(harness):
    out = _out(harness["root"], "ok_a", "ok_a", 42)
    out.parent.mkdir(parents=True)
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    cr_run_cell.claim_path(out).write_text(f"{dead.pid} {socket.gethostname()} then\n")

    dry, _ = harness["run"]("--arms", "ok_a", "--seeds", "42", "--dry-run")
    assert dry.returncode == 0 and "STALE CLAIM" in dry.stdout
    assert cr_run_cell.claim_path(out).exists()                        # dry run: untouched

    proc, log_dir = harness["run"]("--arms", "ok_a", "--seeds", "42")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    side = json.loads(cr_queue.sidecar_path(out).read_text())
    assert side["kind"] == "stale_claim" and str(dead.pid) in side["claim"]
    assert not cr_run_cell.claim_path(out).exists() and not out.exists()
    assert "STALE-CLAIM" in (log_dir / "queue.log").read_text()


def test_e2e_a_live_claim_is_busy_and_left_alone(harness):
    out = _out(harness["root"], "ok_a", "ok_a", 42)
    out.parent.mkdir(parents=True)
    holder = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)",
                               "run_cell.py"])
    try:
        cr_run_cell.claim_path(out).write_text(f"{holder.pid} {socket.gethostname()} now\n")
        proc, _ = harness["run"]("--arms", "ok_a", "--seeds", "42-43")
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "busy: 1" in proc.stdout and not out.exists()
        assert cr_run_cell.claim_path(out).exists()
        assert _out(harness["root"], "ok_a", "ok_a", 43).exists()
    finally:
        holder.kill()
        holder.wait()


def test_e2e_dry_run_has_no_side_effects(harness):
    proc, log_dir = harness["run"]("--arms", "ok_a,ok_b", "--seeds", "42-43", "--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "cells to run: 4" in proc.stdout
    assert not harness["root"].exists() and not log_dir.exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_e2e_sigterm_interrupts_the_children_and_the_next_queue_resumes(harness):
    root = harness["root"]
    out = _out(root, "ok_a", "ok_a", 42)
    proc, log_dir = harness["run"]("--arms", "ok_a", "--seeds", "42", popen=True,
                                   env={"FAKE_HANG": "1"})
    claim = cr_run_cell.claim_path(out)
    deadline = time.time() + 60
    while not claim.exists() and time.time() < deadline:
        time.sleep(0.1)
    assert claim.exists(), "the child never started"
    child_pid = int(claim.read_text().split()[0])
    proc.send_signal(signal.SIGTERM)
    queue_out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 128 + signal.SIGTERM, queue_out
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)                               # the orphan is gone
    assert not claim.exists() and not out.exists()
    assert cr_queue.interrupted_path(out).exists(), queue_out
    marker = json.loads(cr_queue.interrupted_path(out).read_text())
    assert marker["signal"] == signal.SIGTERM
    summary = json.loads((log_dir / "summary.json").read_text())
    assert summary["status"] == "interrupted" and summary["counts"]["interrupted"] == 1
    assert not cr_queue.sidecar_path(out).exists()          # interrupted is not a crash

    again, log2 = harness["run"]("--arms", "ok_a", "--seeds", "42")
    assert again.returncode == 0, again.stdout + again.stderr
    assert "resumed: 1" in again.stdout and out.exists()
    assert (root / "ok_a" / "attempts" / (out.name + ".attempt1.interrupted.json")).exists()
    assert "RESUME" in (log2 / "queue.log").read_text()


def test_e2e_a_valid_result_then_a_bad_exit_is_completed_not_crashed(harness):
    """N1: publish, then exit 1 (teardown crash) or hang past the timeout: the
    cell is OK with the odd exit recorded, no crash record, never rerun."""
    root = harness["root"]
    arms = "teardowncrash,teardownhang"
    proc, log_dir = harness["run"]("--arms", arms, "--seeds", "42")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    cells = {c["arm"]: c for c in _cells(log_dir)}
    for arm in ("teardowncrash", "teardownhang"):
        out = _out(root, arm, arm, 42)
        assert out.exists() and not cr_queue.sidecar_path(out).exists(), arm
        assert cells[arm]["status"] == "ok" and cells[arm]["problems"] == []
        assert cells[arm]["git_commit_full"] == harness["head"]
    assert cells["teardowncrash"]["odd_exit"] == {
        "returncode": 1, "timed_out": False, "killed_by_queue_stop": False}
    assert cells["teardownhang"]["odd_exit"]["timed_out"] is True
    assert cells["teardownhang"]["odd_exit"]["returncode"] != 0
    assert (log_dir / "queue.log").read_text().count("ODD-EXIT") == 2
    summary = json.loads((log_dir / "summary.json").read_text())
    assert summary["counts"]["ok"] == 2 and summary["exit_code"] == 0
    assert not list(root.rglob("*.running"))

    again, _ = harness["run"]("--arms", arms, "--seeds", "42", "--retry-crashed",
                              "--amendment", "2026-10-12: must not rerun a completed cell")
    assert again.returncode == 0, again.stdout + again.stderr
    assert "cells to run: 0" in again.stdout and "completed (valid): 2" in again.stdout
    assert not (root / "teardowncrash" / "attempts").exists()


def test_e2e_a_crash_record_next_to_a_valid_result_is_never_retried(harness):
    """A record an older queue wrote next to a valid result: completed, warned, kept."""
    out = _out(harness["root"], "ok_a", "ok_a", 42)
    out.parent.mkdir(parents=True)
    res = _result(prov_git_commit_full=harness["head"], prov_git_commit=harness["head"][:7])
    out.write_text(json.dumps(res))
    record = cr_queue.sidecar_path(out)
    record.write_text(json.dumps({"kind": "hard_crash"}))
    proc, _ = harness["run"]("--arms", "ok_a", "--seeds", "42", "--retry-crashed",
                             "--amendment", "2026-10-12: test")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "completed (valid): 1" in proc.stdout and "retried: 0" in proc.stdout
    assert "WARNING" in proc.stdout and str(out) in proc.stdout
    assert json.loads(out.read_text()) == res and record.exists()
    assert not (out.parent / "attempts").exists()


@pytest.mark.parametrize("arm,status", [("hang", "timeout"), ("solvercrash", "crashed_in_solver")])
def test_e2e_the_breaker_stops_after_three_timeouts_or_solver_crashes(harness, arm, status):
    """N4: a hung GPU times out every running cell; stop instead of writing a
    terminal record for every cell of the selection."""
    proc, log_dir = harness["run"]("--arms", arm, "--seeds", "42-45", "--workers-per-gpu", "1")
    assert proc.returncode == cr_queue.EXIT_FAILURES, proc.stdout + proc.stderr
    summary = json.loads((log_dir / "summary.json").read_text())
    assert summary["counts"][status] == 3 and summary["counts"]["not_started"] == 1
    assert "3 failed cells in a row" in summary["stop_reason"]
    assert not _out(harness["root"], arm, arm, 45).exists()
    assert not cr_queue.sidecar_path(_out(harness["root"], arm, arm, 45)).exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_e2e_a_child_that_crashes_on_its_own_during_a_stop_is_a_crash(harness):
    """N7: only a child the queue's kill ended is interrupted (and resumed
    without an amendment); one that crashed by itself meanwhile is terminal."""
    root = harness["root"]
    out = _out(root, "termcrash", "termcrash", 42)
    proc, log_dir = harness["run"]("--arms", "termcrash", "--seeds", "42", popen=True)
    claim = cr_run_cell.claim_path(out)
    deadline = time.time() + 60
    while not claim.exists() and time.time() < deadline:
        time.sleep(0.05)
    assert claim.exists(), "the child never started"
    proc.send_signal(signal.SIGTERM)
    queue_out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 128 + signal.SIGTERM, queue_out
    assert not cr_queue.interrupted_path(out).exists(), queue_out
    side = json.loads(cr_queue.sidecar_path(out).read_text())
    assert side["kind"] == "hard_crash" and side["returncode"] == 1
    assert "crashed on its own" in side["output_tail"]
    summary = json.loads((log_dir / "summary.json").read_text())
    assert summary["counts"]["crashed"] == 1 and summary["counts"]["interrupted"] == 0


def test_e2e_every_arm_runs_its_own_seed_range_into_its_own_directory(harness):
    root = harness["root"]
    proc, log_dir = harness["run"]("--arms", "ok_a,ok_a__s82")          # no --seeds
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "seeds: per arm: 42-43 (1 arms); 82-83 (1 arms)" in proc.stdout
    for arm, seed in (("ok_a", 42), ("ok_a", 43), ("ok_a__s82", 82), ("ok_a__s82", 83)):
        res = json.loads(_out(root, arm, "ok_a", seed).read_text())
        assert (res["solver"], res["seed"], res["provenance"]["cr_arm"]) == ("ok_a", seed, arm)
    assert not _out(root, "ok_a__s82", "ok_a", 42).exists()
    assert not _out(root, "ok_a", "ok_a", 82).exists()
    cells = _cells(log_dir)
    assert sorted((c["arm"], c["seed"]) for c in cells) == [
        ("ok_a", 42), ("ok_a", 43), ("ok_a__s82", 82), ("ok_a__s82", 83)]
    for c in cells:   # every child got the queue's manifest (run_cell checks the seed range)
        cmd_line = Path(c["log"]).read_text().splitlines()[0]
        assert "--manifest " + str(harness["tmp"] / "manifest.json") in cmd_line
    summary = json.loads((log_dir / "summary.json").read_text())
    assert summary["arm_seeds"] == {"ok_a": "42-43", "ok_a__s82": "82-83"}
    assert summary["seeds"] == [42, 43, 82, 83] and summary["seeds_requested"] is None
    # --waves c2 selects the control-style arm; a seed outside its range is refused
    dry, _ = harness["run"]("--waves", "c2", "--dry-run")
    assert dry.returncode == 0 and "cells to run: 0" in dry.stdout
    bad, bad_log = harness["run"]("--arms", "ok_a__s82", "--seeds", "42")
    assert bad.returncode == 1 and "outside the seed range of arm 'ok_a__s82'" in bad.stderr
    assert not bad_log.exists() and not _out(root, "ok_a__s82", "ok_a", 42).exists()
    mixed, _ = harness["run"]("--arms", "ok_a,ok_a__s82", "--seeds", "42", "--dry-run")
    assert mixed.returncode == 1 and "outside the seed range" in mixed.stderr


def test_e2e_waves_option(harness):
    proc, log_dir = harness["run"]("--waves", "1", "--seeds", "42", "--dry-run")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "cells to run: 10" in proc.stdout
    assert "cells to run per wave, in run order: wave 1: 10" in proc.stdout
    assert not harness["root"].exists() and not log_dir.exists()
    both, _ = harness["run"]("--waves", "1", "--arms", "ok_a", "--dry-run")
    assert both.returncode == 2 and "not allowed with" in both.stderr
    bad, _ = harness["run"]("--waves", "1,5", "--dry-run")
    assert bad.returncode == 2 and "--waves" in bad.stderr
