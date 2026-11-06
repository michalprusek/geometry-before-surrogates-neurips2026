"""``random_box_faithful``: surrogate-free uniform draw in the faithful box.

Ported from p1-blocks commit 4780475 (``tests/test_paper1_controls.py``,
analysis-plan A10 block S2(b)); the tests that need that branch's step-logging
instrumentation are not ported. Added here: the proposal is exactly the box GP
arms' random-fallback formula, and the arm is a camera-ready manifest arm.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.spherevo.solver import (
    RandomBoxFaithfulSolver,
    RandomSearchSolver,
    load_faithful_box_bounds,
)

D = 32
DEAD_DIM = 7  # p01 == p99 on this coordinate -> exercises the 1e-6 floor


def _bounds_file(tmp_path):
    """Synthetic D=32 bounds with VARIED per-coordinate widths and one dead
    coordinate (p01 == p99), so a per-coordinate indexing bug is detectable
    and the dead-dimension floor is exercised."""
    rng = np.random.RandomState(0)
    p01 = rng.uniform(-3.0, -0.2, size=D)
    p99 = p01 + rng.uniform(0.05, 4.0, size=D)
    p01[DEAD_DIM] = p99[DEAD_DIM] = 0.123456
    p = tmp_path / "bounds32_varied.json"
    p.write_text(json.dumps({"per_coord": {
        "p01": p01.tolist(), "p99": p99.tolist(),
        "min": (p01 - 1).tolist(), "max": (p99 + 1).tolist(),
    }}))
    return str(p), p01, p99


def test_box_proposals_within_p01_p99_over_1000_draws(tmp_path, cold_start_data, mock_black_box):
    bp, p01, p99 = _bounds_file(tmp_path)
    x0, y0 = cold_start_data
    solver = RandomBoxFaithfulSolver(mock_black_box, x0, y0, bounds_path=bp)
    np.random.seed(0)
    Z = np.concatenate([solver.next_candidate() for _ in range(1000)], axis=0)
    assert Z.shape == (1000, D)
    live = np.ones(D, dtype=bool)
    live[DEAD_DIM] = False
    assert np.all(Z[:, live] >= p01[live]) and np.all(Z[:, live] <= p99[live])
    # dead coordinate: width floored at 1e-6 exactly as _BoxFaithfulMixin does
    assert np.all(Z[:, DEAD_DIM] >= p01[DEAD_DIM])
    assert np.all(Z[:, DEAD_DIM] <= p01[DEAD_DIM] + 1e-6)
    assert solver._width[DEAD_DIM] == 1e-6
    # the draw actually fills the box (not stuck at one edge) on live coords
    span = Z[:, live].max(0) - Z[:, live].min(0)
    assert np.all(span > 0.9 * (p99 - p01)[live])


def test_box_loader_matches_mixin_semantics(tmp_path):
    bp, p01, p99 = _bounds_file(tmp_path)
    lo, hi = load_faithful_box_bounds(bp, D)
    assert np.allclose(lo, p01) and np.allclose(hi, p99)
    with pytest.raises(ValueError):
        load_faithful_box_bounds(bp, D + 1)
    with pytest.raises(FileNotFoundError):
        load_faithful_box_bounds(str(tmp_path / "missing.json"), D)


def test_box_default_bounds_is_selfies_zinc_file():
    """Default path = the same file _BoxFaithfulMixin defaults to (D=256)."""
    from pathlib import Path

    from src.analysis.config import ZINC_PER_COORD_BOUNDS_PATH
    repo_root = Path(__file__).resolve().parent.parent
    if not (repo_root / ZINC_PER_COORD_BOUNDS_PATH).exists():
        pytest.skip("real 256-D bounds file not present in this tree")
    lo, hi = load_faithful_box_bounds(None, 256)
    assert lo.shape == (256,) and hi.shape == (256,) and np.all(hi >= lo)
    assert ZINC_PER_COORD_BOUNDS_PATH.endswith("zinc_per_coord_bounds.json")


def test_proposal_is_the_box_gp_arms_fallback_formula(tmp_path, cold_start_data, mock_black_box):
    """z = p01 + u * max(p99 - p01, 1e-6), as _BoxFaithfulMixin._rescale_from_unit_cube."""
    from src.spherevo.gp_competitors import HvarfnerScaledGPBoxFaithfulSolver

    bp, _, _ = _bounds_file(tmp_path)
    x0, y0 = cold_start_data
    box = RandomBoxFaithfulSolver(mock_black_box, x0, y0, bounds_path=bp)
    gp_arm = HvarfnerScaledGPBoxFaithfulSolver(
        black_box=mock_black_box, x0=x0, y0=y0, bounds_path=bp,
    )
    np.random.seed(5)
    z = box.next_candidate()
    np.random.seed(5)
    u = np.random.rand(1, D)
    u_t = torch.tensor(u, dtype=torch.float64, device=gp_arm._lo.device)
    ref = gp_arm._rescale_from_unit_cube(u_t).cpu().numpy()
    assert np.allclose(z, ref, rtol=0, atol=1e-12)


def test_runs_25_steps_and_records_no_fallback(tmp_path, cold_start_data, mock_black_box):
    bp, _, _ = _bounds_file(tmp_path)
    x0, y0 = cold_start_data
    solver = RandomBoxFaithfulSolver(mock_black_box, x0, y0, bounds_path=bp)
    np.random.seed(1)
    solver.solve(max_iter=25)
    assert len(solver.all_scores) == len(y0) + 25
    assert not hasattr(solver, "_fallback_count")
    from scripts.run_benchmark import _surrogate_fallbacks

    assert _surrogate_fallbacks(solver) == (0, [])


@pytest.mark.parametrize("name,cls", [
    ("random_search", RandomSearchSolver),
    ("random_box_faithful", RandomBoxFaithfulSolver),
])
def test_registry_resolves(tmp_path, monkeypatch, cold_start_data, mock_black_box, name, cls):
    from scripts.run_benchmark import OUR_SOLVERS, _create_solver

    assert name in OUR_SOLVERS
    bp, _, _ = _bounds_file(tmp_path)
    monkeypatch.setattr(RandomBoxFaithfulSolver, "default_bounds_path", bp)
    x0, y0 = cold_start_data
    solver = _create_solver(
        solver_name=name, black_box=mock_black_box, x0=x0, y0=y0,
        n_dimensions=D, subspace_dim=16, n_subspaces=16, seed=42,
        n_initial_points=len(y0), codec=None, norm_mode="fixed", n_iterations=25,
        cs_smiles=None,
    )
    assert isinstance(solver, cls)
    z = solver.next_candidate()
    assert z.shape == (1, D) and np.all(np.isfinite(z))


def test_registry_codec_bounds_dispatch(monkeypatch, cold_start_data, mock_black_box):
    """Non-SELFIES codecs get their own bounds file, as the *_box_faithful arms."""
    import scripts.run_benchmark as rb

    seen = {}

    def fake_loader(bounds_path, dim):
        seen["path"] = bounds_path
        return np.zeros(dim), np.ones(dim)

    import src.spherevo.solver as solver_mod
    monkeypatch.setattr(solver_mod, "load_faithful_box_bounds", fake_loader)

    class ChembedCodecX:  # the class name carries the dispatch key
        pass

    x0, y0 = cold_start_data
    for codec, expected in ((None, None), (ChembedCodecX(),
                            "results/precomputed/chembed_per_coord_bounds.json")):
        rb._create_solver(
            solver_name="random_box_faithful", black_box=mock_black_box, x0=x0, y0=y0,
            n_dimensions=D, subspace_dim=16, n_subspaces=16, seed=42,
            n_initial_points=len(y0), codec=codec, norm_mode="fixed", n_iterations=25,
            cs_smiles=None,
        )
        assert seen["path"] == expected
