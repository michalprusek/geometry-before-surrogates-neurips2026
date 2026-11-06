"""Published-recipe block: GuacaMol arms, registration, manifest, queue driver, untouched old code.

* 25-step D = 256 runs of every new arm (synthetic black box, cold start partly outside the
  [p01, p99] box as with real ZINC encodings) on CPU and, when present, CUDA: 0 crashes,
  0 fallbacks, every proposal inside its box, one record per step.
* A fit failure forced in the middle of a GuacaMol run is caught in that step and recorded.
* The block is ADDITIVE: the code the running campaigns use is byte-identical to
  fd8a03c (sha256 pins; scripts/cr/manifest.json additionally carries the three
  baseline arms added on 2026-11-02, see the pin -- it is the one shared file whose
  content changed, and it changed by adding arms, never by editing one; src/spherevo/solver.py: its code, i.e. its syntax tree without docstrings),
  the existing solver names keep their order and classes.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

pytest.importorskip("gpytorch")
pytest.importorskip("botorch")

from botorch.exceptions.errors import ModelFittingError  # noqa: E402

from scripts.cr import cr_queue  # noqa: E402
from scripts.cr import cr_queue_pub  # noqa: E402
from scripts.cr import run_cell as cr_run_cell  # noqa: E402
from src.spherevo import published_recipes as pr  # noqa: E402
from src.spherevo.published_arms import (  # noqa: E402
    PUBLISHED_ARM_CLASSES,
    HvarfnerPublishedAllObsSolver,
    HvarfnerPublishedSolver,
    LinearBOBox8Solver,
    PapenmeierMLEScaledReferenceLearnedSolver,
    PapenmeierMLEScaledReferenceSolver,
    PapenmeierMSRReferenceLearnedSolver,
    PapenmeierMSRReferenceSolver,
    TurboPublishedSolver,
    XuPublishedSolver,
)

REPO = Path(__file__).resolve().parents[1]
N_STEPS = 25
NEW_ARMS = ("hvarfner_published", "papenmeier_msr_reference", "papenmeier_mle_scaled_reference",
            "xu_published", "linear_bo_box8", "turbo_published",
            "hvarfner_published_allobs",                        # round 2, R2-5
            "papenmeier_msr_reference_learned",                 # round 2, R2-6
            "papenmeier_mle_scaled_reference_learned")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# sha256 of the shared files at fd8a03c (the commit the running campaigns use)
#
# scripts/cr/manifest.json is NO LONGER byte-identical to fd8a03c.  On 2026-11-02 three
# baseline arms were added to it (cmaes_full_rd, cmaes_full_rd_pop5, baxus_cr), as the
# task that added them required, so the pin below is the sha256 AFTER that addition and
# the fd8a03c value is kept beside it.  The addition is confined to the blank line that
# separated wave 3 from wave 3b, so the file keeps its line count and every manifest line
# number cited by scripts/cr/analysis/appendixH_facts.csv still points at the same arm.
# Note for whoever reads this next: the repository's own way of adding a block of arms
# without touching this file is a separate manifest (scripts/cr/manifest_pub.json, wave
# "p", output root results/guacamol/cr2026pub).  If a campaign is reading
# scripts/cr/manifest.json right now, move the three arms to such a manifest instead.
# 2026-11-06 (RUN_RULES 2026-11-06 16:30): cr_queue.py, manifest.json and gp_competitors.py carry the control-C4 code of
# commit 2a899ad (tag cr-controls-lbfgs-2026-10-14): wave c3, the arm hvarfner_scaled_gp_sphere_lbfgs and its solver
# class. gp_competitors.py is byte-identical to that commit; the other two add its hunks to the versions pinned before.
PINNED_SHA256 = {
    "scripts/cr/manifest.json": "4bed1b08cb6bb749d2a644dd70b69f5e93cf703e2efea7784e31141d471b8b0b",      # 2026-11-02, +3 baseline arms; 2026-11-06, + control C4
    "scripts/cr/cr_queue.py": "8cb32b0d16b8e4e0b38ced4770c844790a0e54cd4b92614af29d2e925eb97418",
    "scripts/cr/run_cell.py": "e2dfef5af4405a9e6942898bb9a335ed7ead56887231a494c01747541716517a",
    "src/spherevo/gp_competitors.py":
        "7bfd322f8a974ac2f55a1243f879deedacd167cf6161c6eef94e77da4428049e",
    "src/spherevo/turbo_box_faithful.py":
        "75b7c1bcec31ae4e8c16495b2d6a7ddcadde6930f8e14e45c77ba87d3e9f8540",
    # sha256 of the file as released: the release corrects its licence header (patch
    # linear-bo-licence-header, docstring only; fd8a03c had c487bff3029de0765b8c76970f0438ff9439bd77efb402b562a02e181d435423)
    "src/spherevo/linear_bo_kernel.py":
        "4472e9ce1c57c9549281a80ee02968256b50fc357e96bf031f28ce95664de195",
    "src/spherevo/turbo_baseline.py":
        "da664e17523f119e8e7a1549c29b1b69cc393dae6957909dec8a6cb4b2823e89",
}
# src/spherevo/solver.py: its docstrings were updated on 2026-10-24 (RUN_RULES 2026-10-24, clause 4); the code is
# unchanged. The pin is therefore the sha256 of ast.dump() of the module with every docstring removed
# (_strip_docstrings), computed from `git show fd8a03c:src/spherevo/solver.py` with Python 3.12.14 (the
# interpreter of the runs; ast.dump text can differ between Python versions). The full-file sha256 at fd8a03c
# was fbef7ae235f54d7fe63510101edb42c11270d0553d6fc03f9698ff318a8ac412.
SOLVER_AST_PINNED = ("src/spherevo/solver.py",
                     "f156a380825ad644ba10b2639eaf941c62980125ff048a72ac3f63d1e98709d6")
#: sha256 of "\n".join(OUR_SOLVERS) at fd8a03c (106 names) plus hvarfner_scaled_gp_sphere_lbfgs (control C4 of commit
#: 2a899ad, inserted after hvarfner_scaled_gp_sphere_continuous as in that commit; 107 names). Before 2026-11-06 the
#: pinned block was the 106 names alone (sha256 88c6d31ad44105b60d12e9ad63fdc16e50833ab327be8a7773c14de52be31294).
OLD_OUR_SOLVERS_SHA256 = "293119ccc4de725a4810077d8b9c19be0cc88015fb535c4123fa4b0d34ab45b9"


# ---------------------------------------------------------------------------
# synthetic D = 256 problem
# ---------------------------------------------------------------------------


def _problem(D=256, seed=0):
    rng = np.random.RandomState(seed)
    mean, sd = rng.randn(D) * 0.3, 0.5 + rng.rand(D)
    x0 = mean + sd * rng.randn(5, D)
    target = mean + 0.5 * sd * rng.randn(D)

    def black_box(z):
        z = np.asarray(z, dtype=np.float64).reshape(-1, D)
        return (-np.mean(((z - target) / sd) ** 2, axis=1)).reshape(-1, 1)

    p01, p99 = mean - 1.6 * sd, mean + 1.6 * sd     # narrower than the cold start
    return black_box, x0, black_box(x0), {"p01": p01.tolist(), "p99": p99.tolist()}


def _make(name, black_box, x0, y0, bounds_file):
    cls = PUBLISHED_ARM_CLASSES[name]
    if cls is LinearBOBox8Solver:
        return cls(black_box=black_box, x0=x0, y0=y0)
    return cls(black_box=black_box, x0=x0, y0=y0, bounds_path=str(bounds_file))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", NEW_ARMS)
def test_new_arm_runs_25_steps_at_d256_without_crash_or_fallback(tmp_path, monkeypatch, name,
                                                                   device):
    if device == "cpu":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    black_box, x0, y0, box = _problem()
    bounds_file = tmp_path / "bounds256.json"
    bounds_file.write_text(json.dumps({"per_coord": box}))
    torch.manual_seed(0)
    np.random.seed(0)
    s = _make(name, black_box, x0, y0, bounds_file)
    assert s._device.type == device
    s.solve(max_iter=N_STEPS)
    assert s.iteration == N_STEPS and len(s.all_scores) == 5 + N_STEPS
    assert s._fallback_count == 0, s._fallback_reasons
    rec = s.published_recipe_record
    assert rec["recipe"] == PUBLISHED_ARM_CLASSES[name].recipe
    assert len(rec["steps"]) == N_STEPS and not any(st["fallback"] for st in rec["steps"])
    Z = np.vstack(s.history["x"][5:])
    if name == "linear_bo_box8":
        assert np.all(Z >= -8.0) and np.all(Z <= 8.0)
    else:
        lo, hi = np.array(box["p01"]), np.array(box["p99"])
        assert np.all(Z >= lo - 1e-9) and np.all(Z <= hi + 1e-9)
        # the cold start lies partly outside [p01, p99], as real ZINC encodings do
        if name == "turbo_published":
            assert rec["cold_start_coordinates_projected"] > 0
            assert all(st["kind"] == "bo" for st in rec["steps"])
        else:
            assert rec["steps"][0]["n_obs_outside_box"] > 0


@pytest.mark.parametrize("name", ["hvarfner_published", "xu_published", "linear_bo_box8"])
def test_forced_fit_failure_in_a_guacamol_run_is_caught_and_recorded(tmp_path, monkeypatch, name):
    black_box, x0, y0, box = _problem(D=32)
    bounds_file = tmp_path / "b.json"
    bounds_file.write_text(json.dumps({"per_coord": box}))
    real_fit = pr.fit_gpytorch_mll
    n = {"calls": 0}

    def fit_failing_on_third(mll, **kw):
        n["calls"] += 1
        if n["calls"] == 3:
            raise ModelFittingError("All attempts to fit the model have failed.")
        return real_fit(mll, **kw)

    monkeypatch.setattr(pr, "fit_gpytorch_mll", fit_failing_on_third)
    s = _make(name, black_box, x0, y0, bounds_file)
    s.solve(max_iter=5)
    assert s.iteration == 5
    assert s._fallback_count == 1 and "ModelFittingError" in s._fallback_reasons[0]
    steps = s.published_recipe_record["steps"]
    assert [st["iteration"] for st in steps if st["fallback"]] == [2]


def test_msr_reference_fit_failure_in_a_guacamol_run_uses_adam_not_a_fallback(tmp_path,
                                                                              monkeypatch):
    black_box, x0, y0, box = _problem(D=32)
    bounds_file = tmp_path / "b.json"
    bounds_file.write_text(json.dumps({"per_coord": box}))

    def failing(mll, **kw):
        raise ModelFittingError("All attempts to fit the model have failed.")

    monkeypatch.setattr(pr, "fit_gpytorch_mll", failing)
    monkeypatch.setattr(pr, "MSR_ADAM_STEPS", 20)   # speed only; the count is pinned elsewhere
    s = _make("papenmeier_msr_reference", black_box, x0, y0, bounds_file)
    s.solve(max_iter=2)
    assert s._fallback_count == 0
    assert all(any("Adam fallback" in e for e in st["events"])
               for st in s.published_recipe_record["steps"])


# ---------------------------------------------------------------------------
# registration in the benchmark runner (additive)
# ---------------------------------------------------------------------------


def test_new_solvers_are_appended_and_dispatch_to_their_classes(tmp_path, monkeypatch):
    import scripts.run_benchmark as rb

    assert list(rb.OUR_SOLVERS[-len(NEW_ARMS):]) == list(NEW_ARMS)
    old = rb.OUR_SOLVERS[:-len(NEW_ARMS)]
    assert hashlib.sha256("\n".join(old).encode()).hexdigest() == OLD_OUR_SOLVERS_SHA256
    black_box, x0, y0, box = _problem(D=32)
    bounds_file = tmp_path / "b.json"
    bounds_file.write_text(json.dumps({"per_coord": box}))
    from src.spherevo.gp_competitors import _BoxFaithfulMixin
    monkeypatch.setattr(_BoxFaithfulMixin, "default_bounds_path", str(bounds_file))
    common = dict(black_box=black_box, x0=x0, y0=y0, n_dimensions=32, subspace_dim=16,
                  n_subspaces=16, seed=42, n_initial_points=5, codec=None, n_iterations=N_STEPS)
    for name, cls in (("hvarfner_published", HvarfnerPublishedSolver),
                      ("papenmeier_msr_reference", PapenmeierMSRReferenceSolver),
                      ("papenmeier_mle_scaled_reference", PapenmeierMLEScaledReferenceSolver),
                      ("xu_published", XuPublishedSolver),
                      ("linear_bo_box8", LinearBOBox8Solver),
                      ("turbo_published", TurboPublishedSolver),
                      ("hvarfner_published_allobs", HvarfnerPublishedAllObsSolver),
                      ("papenmeier_msr_reference_learned", PapenmeierMSRReferenceLearnedSolver),
                      ("papenmeier_mle_scaled_reference_learned",
                       PapenmeierMLEScaledReferenceLearnedSolver)):
        assert name in rb.SUPPORTED_SOLVERS
        s = rb._create_solver(solver_name=name, **common)
        assert type(s) is cls
        if cls is not LinearBOBox8Solver:   # same bounds file and rescaling as the faithful arms
            assert np.allclose(s._lo.cpu().numpy(), box["p01"])
    # the existing arms still resolve to their own classes
    from src.spherevo.gp_competitors import (
        LinearBOBoxSolver,
        PapenmeierMSRBoxFaithfulSolver,
        PapenmeierMSRPublishedSolver,
        XuStandardGPBoxFaithfulSolver,
    )
    for name, cls in (("papenmeier_msr_published", PapenmeierMSRPublishedSolver),
                      ("papenmeier_msr_box_faithful", PapenmeierMSRBoxFaithfulSolver),
                      ("xu_standard_gp_box_faithful", XuStandardGPBoxFaithfulSolver),
                      ("linear_bo", LinearBOBoxSolver)):
        assert type(rb._create_solver(solver_name=name, **common)) is cls


def test_linear_bo_box8_maps_the_molecular_box_to_the_unit_cube():
    black_box, x0, y0, _ = _problem(D=32)
    s = LinearBOBox8Solver(black_box=black_box, x0=x0, y0=y0)
    z = torch.tensor([[-8.0] * 16 + [8.0] * 16], dtype=torch.float64)
    u = s._rescale_to_unit_cube(z.to(s._device))
    assert torch.equal(u.cpu(), torch.tensor([[0.0] * 16 + [1.0] * 16], dtype=torch.float64))
    assert torch.allclose(s._rescale_from_unit_cube(u).cpu(), z)


def test_result_record_carries_the_published_recipe_only_for_new_arms():
    src = (REPO / "scripts" / "run_benchmark.py").read_text()
    assert ('**({"published_recipe": solver.published_recipe_record}\n'
            '           if hasattr(solver, "published_recipe_record") else {}),') in src
    from src.spherevo.gp_competitors import PapenmeierMSRPublishedSolver
    assert not hasattr(PapenmeierMSRPublishedSolver, "published_recipe_record")


# ---------------------------------------------------------------------------
# the running campaigns' code and manifest are untouched
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rel", sorted(PINNED_SHA256))
def test_shared_file_is_byte_identical_to_fd8a03c(rel):
    assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == PINNED_SHA256[rel]


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    """Remove the docstring (first-statement string constant) of the module and of every class and function."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def test_solver_code_is_identical_to_fd8a03c():
    """solver.py: the syntax tree without docstrings equals that of fd8a03c (only docstrings and comments
    changed since; see SOLVER_AST_PINNED)."""
    rel, pinned = SOLVER_AST_PINNED
    dump = ast.dump(_strip_docstrings(ast.parse((REPO / rel).read_text(encoding="utf-8"))))
    assert hashlib.sha256(dump.encode("utf-8")).hexdigest() == pinned


# ---------------------------------------------------------------------------
# the block's manifest and queue driver
# ---------------------------------------------------------------------------


@pytest.fixture
def pub_manifest():
    return cr_queue_pub.load_pub_manifest()


def test_pub_manifest_is_the_gpu_arms_in_wave_p_with_its_own_root(pub_manifest):
    m = pub_manifest
    assert [a["arm"] for a in m["arms"]] == list(NEW_ARMS)
    assert all(a["solver"] == a["arm"] and a["run_overrides"] == {} and a["wave"] == "p"
               and a["host_class"] == "gpu" and "seeds" not in a for a in m["arms"])
    assert m["output_root"] == "results/guacamol/cr2026pub"
    assert m["seeds"] == list(range(42, 82)) and len(m["tasks"]) == 11
    main = json.loads((REPO / "scripts" / "cr" / "manifest.json").read_text())
    assert m["tasks"] == main["tasks"] and m["task_poli_ids"] == main["task_poli_ids"]
    assert m["gpu_hosts"] == main["gpu_hosts"]
    assert not {a["solver"] for a in main["arms"]} & set(NEW_ARMS)
    from scripts.run_benchmark import SUPPORTED_SOLVERS
    assert set(NEW_ARMS) <= set(SUPPORTED_SOLVERS)


def test_pub_selection_is_440_cells_per_arm_under_cr2026pub(pub_manifest):
    with cr_queue_pub.published_waves():
        arms = cr_queue.select_arms(pub_manifest, wave=None, arm_ids=None, host_class="gpu",
                                    waves=["p"])
        per_arm = cr_queue.seeds_for_arms(pub_manifest, arms, None)
    assert [a["arm"] for a in arms] == list(NEW_ARMS)
    assert sum(len(s) for s in per_arm.values()) * len(pub_manifest["tasks"]) == 440 * len(NEW_ARMS)
    assert 440 * len(NEW_ARMS) == 3960                     # 9 arms after round 2 (R2-5, R2-6)
    out = cr_queue.cell_output(arms[0], "rano", 42, pub_manifest, None, repo=REPO)
    assert out.parent == REPO / "results" / "guacamol" / "cr2026pub" / "hvarfner_published"


def test_wave_p_is_admitted_only_inside_the_driver():
    before = cr_queue.VALID_WAVES
    assert "p" not in before
    with cr_queue_pub.published_waves():
        assert "p" in cr_queue.VALID_WAVES
    assert cr_queue.VALID_WAVES is before
    with pytest.raises(SystemExit, match="wave"):
        cr_queue.load_manifest(cr_queue_pub.PUB_MANIFEST)    # plain cr_queue refuses wave p


def test_driver_refuses_another_output_root_or_foreign_arms(tmp_path):
    with pytest.raises(SystemExit, match="cr2026pub"):
        cr_queue_pub.main(["--output-root", "results/guacamol/cr2026", "--dry-run"])
    with pytest.raises(SystemExit, match="cr2026pub"):
        cr_queue_pub.main(["--output-root", "results/guacamol/cr2026ctl", "--dry-run"])
    bad = json.loads(cr_queue_pub.PUB_MANIFEST.read_text())
    bad["arms"].append({**bad["arms"][0], "arm": "geowalk", "solver": "geowalk"})
    p = tmp_path / "m.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(SystemExit, match="not a published-recipe solver"):
        cr_queue_pub.load_pub_manifest(p)


def test_run_cell_checks_cells_against_the_pub_manifest():
    path = cr_queue_pub.PUB_MANIFEST
    ok = cr_run_cell.manifest_cell_problems(path, "xu_published", "xu_published", {}, 42)
    assert not ok
    assert cr_run_cell.manifest_cell_problems(path, "xu_published", "xu_published", {}, 82)
    assert cr_run_cell.manifest_cell_problems(path, "xu_published", "linear_bo_box8", {}, 42)


def test_turbo_published_gp_failure_in_a_guacamol_run_is_caught_and_recorded(tmp_path,
                                                                            monkeypatch):
    from linear_operator.utils.errors import NotPSDError

    from src.spherevo import published_external as pe

    black_box, x0, y0, box = _problem(D=32)
    bounds_file = tmp_path / "b.json"
    bounds_file.write_text(json.dumps({"per_coord": box}))
    real = pe.turbo_train_gp
    n = {"calls": 0}

    def failing_on_third(*a, **k):
        n["calls"] += 1
        if n["calls"] == 3:
            raise NotPSDError("synthetic")
        return real(*a, **k)

    monkeypatch.setattr(pe, "turbo_train_gp", failing_on_third)
    s = _make("turbo_published", black_box, x0, y0, bounds_file)
    s.solve(max_iter=5)
    assert s.iteration == 5 and s._fallback_count == 1
    steps = s.published_recipe_record["steps"]
    assert [st["iteration"] for st in steps if st["fallback"]] == [2]
    assert len(s.turbo._X) == 5 + 5     # the fallback point is told like any evaluation


# ---------------------------------------------------------------------------
# Round 1: C4 (invalid molecules in linear_bo_box8), C10 (effective output root), D1 (equality
# with the official TuRBO-1 and the BoTorch BAxUS tutorial)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,gp_value", [("linear_bo_box8", -0.01), ("hvarfner_published", -1.0)])
def test_invalid_molecules_enter_the_gp_with_the_recipe_value(tmp_path, monkeypatch, name, gp_value):
    """C4: linear-bo guacamol_task.py:646-649 scores an invalid molecule -0.01; the other arms keep
    the repository's -1.0. Only the GP training value changes; the recorded score stays NaN, and
    n_nan_scores counts them."""
    black_box, x0, y0, box = _problem(D=32)
    calls = {"n": 0}

    def bb(z):
        calls["n"] += 1
        return np.full((1, 1), np.nan) if calls["n"] % 2 else black_box(z)

    y0 = y0.copy()
    y0[1, 0] = np.nan
    bounds_file = tmp_path / "b.json"
    bounds_file.write_text(json.dumps({"per_coord": box}))
    seen = []
    real = pr.propose

    def spy(recipe, X, Y):
        seen.append(Y.detach().cpu().numpy().ravel().copy())
        return real(recipe, X, Y)

    monkeypatch.setattr("src.spherevo.published_arms.propose", spy)
    torch.manual_seed(0)
    s = _make(name, bb, x0, y0, bounds_file)
    assert s.published_recipe_record["n_nan_scores"] == 1
    s.solve(max_iter=3)
    assert seen[0][1] == gp_value and seen[2][1] == gp_value and seen[2][5] == gp_value
    assert np.isnan(s.all_scores[1]) and np.isnan(s.all_scores[5])
    assert s.published_recipe_record["n_nan_scores"] == 3          # cold start + 2 of 3 steps
    assert s.published_recipe_record["nan_score_gp_value"] == gp_value


def test_driver_checks_the_effective_output_root():
    """C10 (H-M5): cr_queue's parser takes the LAST --output-root, and a smoke root must sit
    directly under results/guacamol. Both dry-runs the reviewer found are refused."""
    with pytest.raises(SystemExit, match="effective --output-root"):
        cr_queue_pub.main(["--output-root", "results/guacamol/cr2026pub",
                           "--output-root", "results/guacamol/cr2026", "--dry-run"])
    with pytest.raises(SystemExit, match="effective --output-root"):
        cr_queue_pub.main(["--output-root", "results/guacamol/cr2026/cr2026pub_smoke",
                           "--dry-run"])
    with pytest.raises(SystemExit, match="effective --output-root"):
        cr_queue_pub.main(["--output-root", "results/guacamol/cr2026pub/../cr2026", "--dry-run"])
    ok = cr_queue_pub.check_output_root(cr_queue_pub.effective_args(
        ["--output-root", "results/guacamol/cr2026", "--output-root",
         "results/guacamol/cr2026pub_smoke2"]))
    assert ok == (REPO / "results" / "guacamol" / "cr2026pub_smoke2").resolve()
    assert cr_queue_pub.check_output_root(cr_queue_pub.effective_args([])) == \
        (REPO / "results" / "guacamol" / "cr2026pub").resolve()


import os  # noqa: E402

#: local copies of the official sources (cr/baselines/src): TuRBO/ (uber-research/TuRBO) and
#: botorch/tutorials/baxus/baxus.ipynb; the two tests below skip when it is not set
BASELINES_SRC = os.environ.get("PUBLISHED_BASELINES_SRC")


def _src(*parts):
    if not BASELINES_SRC or not Path(BASELINES_SRC, *parts).exists():
        pytest.skip(f"PUBLISHED_BASELINES_SRC/{'/'.join(parts)} not available")
    return Path(BASELINES_SRC, *parts)


def _ackley(x):
    x = np.asarray(x)
    d = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt((x ** 2).sum() / d))
                 - np.exp(np.cos(2 * np.pi * x).sum() / d) + 20 + np.e)


@pytest.mark.parametrize("dim,n_init,max_evals,q", [(10, 5, 45, 1), (12, 4, 64, 10), (3, 4, 60, 1),
                                                   (40, 4, 30, 1), (64, 6, 46, 10)])
def test_turbo_port_is_bit_identical_to_the_official_turbo1(monkeypatch, dim, n_init, max_evals, q):
    """D1: official turbo/turbo_1.py (imported from the local copy) and Turbo1Published with the same
    seeds give the bit-identical sequence of evaluated points (q = 1 and q = 10; at dim 3 the
    failure tolerance 4 collapses the trust region, so the restart path is compared too; at
    dim 40 and 64 the perturbation probability min(20/d, 1) < 1, the mask path of production,
    round 2, R2-13)."""
    import sys
    from src.spherevo.published_external import Turbo1Published

    monkeypatch.syspath_prepend(str(_src("TuRBO")))
    for mod in [m for m in sys.modules if m == "turbo" or m.startswith("turbo.")]:
        monkeypatch.delitem(sys.modules, mod)
    from turbo import Turbo1

    lb, ub = -5 * np.ones(dim), 10 * np.ones(dim)
    np.random.seed(7)
    torch.manual_seed(7)
    t = Turbo1(f=_ackley, lb=lb, ub=ub, n_init=n_init, max_evals=max_evals, batch_size=q,
               verbose=False)
    t.optimize()
    X_off = (t.X - lb) / (ub - lb)

    np.random.seed(7)
    torch.manual_seed(7)
    port = Turbo1Published(dim=dim, n_init=n_init, batch_size=q)
    X_port = []
    while len(X_port) < max_evals:
        U, info = port.ask()
        U = np.asarray(U).reshape(-1, dim)
        assert info.fallback is None, info.fallback
        f = [_ackley(lb + u * (ub - lb)) for u in U]
        U = ((lb + U * (ub - lb)) - lb) / (ub - lb)       # the official from/to unit-cube round trip
        port.tell(U, f)
        X_port.extend(U)
    X_port = np.array(X_port)[:len(X_off)]
    assert len(X_off) == max_evals
    assert all(np.array_equal(a, b) for a, b in zip(X_off, X_port))


def test_baxus_port_is_bit_identical_to_the_botorch_tutorial():
    """D1: the tutorial's cells (2, 5, 8, 10, 12, 17, 19, 21) executed from baxus.ipynb and
    BaxusPublished with the same torch seed give the bit-identical evaluated points (D = 100,
    budget 40, n_init 5). Fails on the C5 mutation (candidates from the target dimension)."""
    import contextlib
    import io
    import sys
    import types

    from src.spherevo.published_external import BaxusPublished

    nb = json.loads(_src("botorch", "tutorials", "baxus", "baxus.ipynb").read_text())
    src = {i: "".join(c["source"]) for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code"}
    D, BUDGET, N_INIT, SEED = 100, 40, 5, 3

    mod = types.ModuleType("baxus_tutorial_cells")
    sys.modules[mod.__name__] = mod
    ns = mod.__dict__
    try:
        exec("\n".join(line for line in src[2].splitlines() if "matplotlib" not in line), ns)
        ns["device"] = torch.device("cpu")
        exec(src[5], ns)
        ns.update(dim=D, n_init=N_INIT, max_cholesky_size=float("inf"), SMOKE_TEST=None)
        for i in (8, 10, 12, 17, 19):
            exec(src[i], ns)
        code21 = src[21].replace("EVALUATION_BUDGET = 100 if not SMOKE_TEST else 10",
                                 f"EVALUATION_BUDGET = {BUDGET}")
        assert code21 != src[21]
        torch.manual_seed(SEED)
        with contextlib.redirect_stdout(io.StringIO()):
            exec(code21, ns)
        X_tut = ns["X_baxus_input"].numpy()
        f = ns["branin_emb"]
    finally:
        sys.modules.pop(mod.__name__, None)

    torch.manual_seed(SEED)
    bx = BaxusPublished(dim=D, n_init=N_INIT, eval_budget=BUDGET - N_INIT, seed=0)
    X_port = []
    for _ in range(BUDGET):
        x, info = bx.ask()
        assert info["fallback"] is None, info["fallback"]
        bx.tell(float(f(torch.tensor(x[0], dtype=torch.double))))
        X_port.append(x[0])
    assert len(X_tut) == BUDGET
    assert all(np.array_equal(a, b) for a, b in zip(X_tut, np.array(X_port)))
