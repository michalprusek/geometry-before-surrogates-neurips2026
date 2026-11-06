"""The three baseline arms added on 2026-11-02: cmaes_full_rd, cmaes_full_rd_pop5, baxus_cr.

Two of these tests are guards against a defect that really happened, and each was
mutation-tested (break the code, watch the test fail, revert):

* ``test_baxus_cr_does_not_collapse_onto_a_box_corner`` fails if the candidates
  collapse onto a single point.  The submitted BAxUS wrapper projected inputs into
  the target space with ``X @ S.T`` and no normalisation, which at ``target_dim = 1``
  put the five cold starts near ``|t| ~ 10``; the trust region then clamped every
  proposal onto one corner of the box (25/25 corners, 1 distinct corner --
  ``/data/prusek/cr_work/dagger_audit_2026-11-02/turbo_baxus/logs/verify_corner.log``).
  Mutation: drop the ``/ nnz`` in ``baxus_project_down``.
* ``test_cmaes_full_rd_population_is_the_arm_s_population`` fails if the population
  setting is ignored.  Mutations: drop ``opts["popsize"]`` in ``CMAESFullSolver``, and
  break the ``endswith("pop5")`` dispatch in ``_create_solver``.

Everything here runs on the CPU in seconds: the solvers are built directly, with a
stub black box, so neither the VAE nor the oracle is needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
D = 256
N_COLD_START = 5
#: The proposal budget of a camera-ready cell (R-1: 5 cold start + 25 proposals).
N_PROPOSALS = 25


class _StubBlackBox:
    """``black_box(z) -> scores``: deterministic, smooth, never NaN."""

    def __init__(self) -> None:
        self.n_calls = 0

    def __call__(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64).reshape(-1, D)
        self.n_calls += z.shape[0]
        return (1.0 / (1.0 + np.linalg.norm(z - 0.3, axis=1))).reshape(-1, 1)


@pytest.fixture(scope="module")
def bounds_file(tmp_path_factory) -> str:
    """A per-coordinate [p01, p99] bounds file of the real file's shape."""
    real = REPO / "results" / "precomputed" / "zinc_per_coord_bounds.json"
    if real.exists():
        return str(real)
    rng = np.random.RandomState(0)
    centre = rng.normal(0.0, 0.1, D)
    p01, p99 = centre - 0.7, centre + 0.7
    path = tmp_path_factory.mktemp("bounds") / "bounds.json"
    path.write_text(json.dumps({"per_coord": {"p01": p01.tolist(), "p99": p99.tolist()}}))
    return str(path)


def _cold_start(seed: int = 42, spread: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Cold-start latents of roughly the real encodings' scale, and their scores."""
    rng = np.random.RandomState(seed)
    x0 = rng.normal(0.0, 0.2937 * spread, (N_COLD_START, D))
    y0 = rng.uniform(0.2, 0.5, (N_COLD_START, 1))
    return x0, y0


def _baxus_solver(bounds_file: str, *, seed: int = 42, spread: float = 1.0):
    from src.spherevo.published_arms import BaxusCRSolver

    torch.manual_seed(seed)
    np.random.seed(seed)
    x0, y0 = _cold_start(seed, spread)
    return BaxusCRSolver(black_box=_StubBlackBox(), x0=x0, y0=y0,
                         bounds_path=bounds_file, seed=seed,
                         eval_budget=N_PROPOSALS), x0, y0


# =========================================================================
# BAxUS: the collapse guard
# =========================================================================

def test_baxus_cr_does_not_collapse_onto_a_box_corner(bounds_file):
    """GUARD. Over a full cell's worth of proposals the arm must propose many
    distinct latents and (practically) no exact corner of the box.

    Under the old defect this test sees 1 distinct latent and 25 corners.
    """
    solver, _, _ = _baxus_solver(bounds_file)
    for _ in range(N_PROPOSALS):
        solver.step()
    rec = solver.published_recipe_record

    assert len(rec["steps"]) == N_PROPOSALS
    assert rec["n_distinct_proposed_latents"] >= N_PROPOSALS - 2, (
        f"only {rec['n_distinct_proposed_latents']} distinct latents in "
        f"{N_PROPOSALS} proposals: the candidates collapsed"
    )
    assert rec["n_box_corner_proposals"] <= 1, (
        f"{rec['n_box_corner_proposals']} of {N_PROPOSALS} proposals are exact box "
        f"corners: the target-space projection is not normalised"
    )
    # the proposals must also be spread out, not merely unequal
    v = np.concatenate([solver._to_pm1_np(x) for x in solver.history["x"][N_COLD_START:]])
    assert v.shape == (N_PROPOSALS, D)
    assert float(np.abs(v).max()) <= 1.0 + 1e-12          # inside the mapped box
    # 0.01 per coordinate: a near-collapse with distinct but equal-looking floats fails too
    assert float(np.linalg.norm(v - v.mean(axis=0), axis=1).max()) > 0.01 * np.sqrt(D)


def test_baxus_cr_seeds_the_cold_start_inside_the_target_box(bounds_file):
    """The least-squares projection must put the cold starts inside [-1, 1].

    Without the normalisation they land at |t| ~ D / target_dim and are all clamped,
    which is the first half of the old defect.
    """
    solver, _, _ = _baxus_solver(bounds_file)
    rec = solver.published_recipe_record
    assert rec["n_seeded"] == N_COLD_START
    assert rec["target_dim_at_seeding"] == 1               # d_init at D = 256
    assert rec["cold_start_target_coordinates_clamped"] == 0, (
        "a cold start was clamped in the target space: the projection is not the "
        "least-squares one"
    )
    T = solver.baxus.X_target.detach().cpu().numpy()
    assert T.shape == (N_COLD_START, 1)
    assert np.abs(T).max() <= 1.0
    assert len(np.unique(T.round(12))) == N_COLD_START     # not collapsed either


def test_baxus_project_down_is_the_identity_on_the_optimisers_own_points():
    """Faithfulness: the projection changes nothing the tutorial itself produces.

    For a target point ``t`` the up-projection is ``x = (t S).clamp(-1, 1)``, and the
    projection of ``x`` back is ``t`` exactly; after a split it returns the parent
    coordinate, which is the value the tutorial copies into each child bin.
    """
    from src.spherevo.published_external import (
        baxus_embedding_matrix,
        baxus_increase_embedding_and_observations,
        baxus_project_down,
    )

    torch.manual_seed(7)
    S = baxus_embedding_matrix(64, 4, torch.device("cpu"), torch.double)
    t = (2 * torch.rand(6, 4, dtype=torch.double) - 1)
    x = (t @ S).clamp(-1.0, 1.0)
    assert torch.allclose(baxus_project_down(x, S), t, atol=1e-12)

    S2, t_copied = baxus_increase_embedding_and_observations(S, t, 3)
    t_reprojected = baxus_project_down(x, S2)
    assert t_reprojected.shape == t_copied.shape
    assert torch.allclose(t_reprojected, t_copied, atol=1e-12), (
        "re-projection after a split differs from the tutorial's column copy"
    )


def test_baxus_cr_records_a_step_failure_instead_of_swallowing_it(bounds_file):
    """A failed step must appear in the record and in surrogate_fallback_count."""
    from src.spherevo.published_recipes import ModelFittingError

    solver, _, _ = _baxus_solver(bounds_file)

    def boom(events=None):
        raise ModelFittingError("induced GP failure")

    solver.baxus._fit_and_propose = boom
    solver.step()
    step = solver.published_recipe_record["steps"][-1]
    assert step["fallback"] is not None and "induced GP failure" in step["fallback"]
    assert solver._fallback_count == 1
    assert solver._fallback_reasons and "induced GP failure" in solver._fallback_reasons[0]


def test_baxus_cr_reprojects_the_cold_starts_after_a_split(bounds_file):
    """GUARD. After a split every row, the seeded ones included, is the clamped
    least-squares projection under the NEW embedding, not the tutorial's column copy."""
    from src.spherevo.published_external import (
        baxus_increase_embedding_and_observations as increase, baxus_project_down)

    solver, _, _ = _baxus_solver(bounds_file)
    b = solver.baxus
    solver.step()
    copied_before = b.X_target.clone()
    b.state.length = b.state.length_min / 4          # the next failed step triggers the split
    b.state.best_value = float("inf")
    rng_state = torch.get_rng_state()
    solver.step()
    assert b.n_splits_done == 1 and b.S.shape[0] > 1
    assert solver.published_recipe_record["splits"][0]["target_dim_after"] == b.S.shape[0]
    assert torch.equal(b.X_target, baxus_project_down(b.X_input, b.S).clamp(-1.0, 1.0))
    # and that differs from the copy on the seeded rows, so the test can see the difference
    copied = b.X_target[:N_COLD_START, 0:1].expand(N_COLD_START, b.S.shape[0])
    assert not torch.allclose(b.X_target[:N_COLD_START],
                              copied_before[:N_COLD_START, 0:1].expand_as(copied))


def test_baxus_cr_puts_a_fit_fallback_and_an_invalid_molecule_in_the_fallback_fields(bounds_file):
    solver, _, _ = _baxus_solver(bounds_file)
    real = solver.baxus._fit_and_propose

    def adam(events=None):
        events.append("fit: ModelFittingError -> Adam fallback")
        return real(events=[])

    solver.baxus._fit_and_propose = adam
    solver.next_candidate()
    assert solver._fallback_count == 1 and "Adam" in solver._fallback_reasons[0]
    solver.post_update(np.zeros((1, D)), np.array([[np.nan]]))
    assert solver._fallback_count == 2 and "invalid molecule" in solver._fallback_reasons[1]
    assert float(solver.baxus.Y[-1]) == -1.0


def test_baxus_cr_maps_the_box_onto_the_tutorials_domain(bounds_file):
    """-1 is p01 and +1 is p99 in every coordinate, and the two maps are inverses."""
    solver, _, _ = _baxus_solver(bounds_file)
    lo = solver._lo.detach().cpu().numpy()
    hi = solver._hi.detach().cpu().numpy()
    assert np.allclose(solver._to_pm1_np(lo.reshape(1, -1)), -1.0)
    assert np.allclose(solver._to_pm1_np(hi.reshape(1, -1)), +1.0)
    v = np.linspace(-1.0, 1.0, D).reshape(1, -1)
    assert np.allclose(solver._to_pm1_np(solver._from_pm1(v)), v, atol=1e-10)
    # the real cold starts are NOT inside the box, as for every faithful-domain arm
    rec = solver.published_recipe_record
    assert rec["cold_start_input_coordinates_clamped"] >= 0
    assert rec["domain"].startswith("[p01, p99]")


def test_baxus_cr_lets_a_real_bug_propagate(bounds_file):
    """The counterpart: a bare RuntimeError is a bug, not a numerical failure, and
    must crash the cell (run_benchmark then records it in status / crash_info)
    instead of quietly becoming a random candidate."""
    solver, _, _ = _baxus_solver(bounds_file)

    def boom(events=None):
        raise RuntimeError("shape mismatch, i.e. a bug")

    solver.baxus._fit_and_propose = boom
    with pytest.raises(RuntimeError, match="a bug"):
        solver.next_candidate()


def test_baxus_cr_split_schedule_comes_from_the_proposal_budget(bounds_file):
    """eval_budget is the arm's 25 proposals, and the schedule follows from it."""
    solver, _, _ = _baxus_solver(bounds_file)
    st = solver.baxus.state
    assert st.eval_budget == N_PROPOSALS
    assert (st.dim, st.d_init, st.n_splits) == (D, 1, 4)
    assert st.failure_tolerance == 1
    s = solver.published_recipe_record["settings"]
    assert s["eval_budget"] == N_PROPOSALS
    assert (s["d_init"], s["n_splits"]) == (1, 4)


def test_baxus_published_keeps_its_verified_behaviour_without_seeding():
    """The LassoBench ``baxus_official`` path is untouched: without seed_external the
    class draws its own Sobol design and copies columns on a split."""
    from src.spherevo.published_external import BaxusPublished

    torch.manual_seed(3)
    b = BaxusPublished(dim=64, n_init=4, eval_budget=50, seed=3)
    assert b._external_seeding is False
    assert len(b._pending) == 4
    for i in range(4):
        v, info = b.ask()
        assert info["kind"] == "init"
        b.tell(float(i))
    assert b.X_target.shape == (4, b.state.d_init)
    assert b.X_input.shape == (4, 64)


# =========================================================================
# CMA-ES: the population guard
# =========================================================================

@pytest.mark.parametrize("solver_name,expected_popsize", [
    ("cmaes_full_rd", 20),          # pycma default 4 + 3 ln 256 = 20
    ("cmaes_full_rd_pop5", 5),      # the population GeoCMA uses
])
def test_cmaes_full_rd_population_is_the_arm_s_population(solver_name, expected_popsize):
    """GUARD. Each arm must really run its own population, in pycma and in the
    generation structure of its proposals.

    Built through ``_create_solver``, so the test fails both if the solver ignores
    ``population_size`` and if the dispatch stops distinguishing the two arms.
    """
    from scripts.run_benchmark import _create_solver

    x0, y0 = _cold_start(42)
    solver = _create_solver(
        solver_name=solver_name, black_box=_StubBlackBox(), x0=x0, y0=y0,
        n_dimensions=D, subspace_dim=16, n_subspaces=16, seed=42,
        n_initial_points=10, codec=None, norm_mode="fixed", n_iterations=N_PROPOSALS,
    )
    assert solver.popsize == expected_popsize
    assert int(solver._cma.popsize) == expected_popsize
    assert solver.published_recipe_record["settings"]["popsize"] == expected_popsize

    # The proposals are grouped into generations of exactly popsize points, and the
    # tell is deferred to the first ask of the next generation (the old wrapper's
    # pattern), so a last incomplete generation is evaluated and never told:
    # floor((N - 1) / popsize) generations are told out of N proposals.
    for _ in range(N_PROPOSALS):
        solver.step()
    assert len(solver.all_scores) == N_COLD_START + N_PROPOSALS
    told = solver.published_recipe_record["generations_told"]
    assert told == (N_PROPOSALS - 1) // expected_popsize, (
        f"{told} generations told for {N_PROPOSALS} proposals at popsize "
        f"{expected_popsize}: the population setting is not reaching pycma"
    )
    sizes = [s["index_in_generation"] for s in solver.published_recipe_record["steps"]]
    assert max(sizes) == min(expected_popsize, N_PROPOSALS) - 1
    # and the generation index really advances every popsize proposals
    gens = [s["generation"] for s in solver.published_recipe_record["steps"]]
    assert gens == [i // expected_popsize for i in range(N_PROPOSALS)]


def test_cmaes_full_rd_starts_at_the_best_cold_start_with_sigma0_half():
    """x0 = best cold-start latent, sigma0 = 0.5, no bounds (the old wrapper)."""
    from src.spherevo.cmaes_full_rd import CMAESFullSolver

    x0, y0 = _cold_start(42)
    solver = CMAESFullSolver(black_box=_StubBlackBox(), x0=x0, y0=y0, seed=42)
    best = x0[int(np.argmax(y0.flatten()))]
    assert np.allclose(np.asarray(solver._cma.x0, dtype=np.float64), best, atol=0)
    assert solver.sigma0 == 0.5
    assert solver._cma.boundary_handler.bounds in (None, [None, None])
    s = solver.published_recipe_record["settings"]
    assert (s["sigma0"], s["bounds"], s["x0"]) == (0.5, None, "best cold-start latent")


def test_cmaes_full_rd_feeds_a_nonfinite_score_as_zero_and_records_it():
    """A failed decode is not swallowed: it enters pycma as 0.0 (the old wrapper's
    value) and is counted in the record."""
    from src.spherevo.cmaes_full_rd import CMAESFullSolver

    x0, y0 = _cold_start(42)
    solver = CMAESFullSolver(black_box=_StubBlackBox(), x0=x0, y0=y0, seed=42,
                             population_size=5)
    solver.next_candidate()
    solver.post_update(np.zeros((1, D)), np.array([[np.nan]]))
    assert solver._fitnesses == [0.0]
    assert solver.published_recipe_record["n_nonfinite_scores_told"] == 1
    assert solver.published_recipe_record["steps"][-1]["nonfinite_score"] is True
    assert solver._fallback_count == 1 and "invalid molecule" in solver._fallback_reasons[0]


# =========================================================================
# manifest wiring of all three arms
# =========================================================================

def test_the_three_new_arms_are_wired_into_the_manifest_and_the_runner():
    from scripts.run_benchmark import SUPPORTED_SOLVERS

    man = json.loads((REPO / "scripts" / "cr" / "manifest.json").read_text())
    by_id = {a["arm"]: a for a in man["arms"]}
    expected = {"cmaes_full_rd": 3600, "cmaes_full_rd_pop5": 3600, "baxus_cr": 7200}
    for arm_id, timeout in expected.items():
        arm = by_id[arm_id]
        assert arm["solver"] == arm_id
        assert arm["run_overrides"] == {}
        assert str(arm["wave"]) == "3"
        assert arm["host_class"] == "gpu"
        assert arm["timeout_s"] == timeout
        assert "seeds" not in arm            # the production seed range 42-81
        assert arm_id in SUPPORTED_SOLVERS


def test_run_cell_accepts_each_new_arm_and_refuses_a_foreign_seed():
    from scripts.cr import run_cell as cr_run_cell

    man = REPO / "scripts" / "cr" / "manifest.json"
    for arm_id in ("cmaes_full_rd", "cmaes_full_rd_pop5", "baxus_cr"):
        assert cr_run_cell.manifest_cell_problems(man, arm_id, arm_id, {}, 42) == []
        assert cr_run_cell.manifest_cell_problems(man, arm_id, arm_id, {}, 82) != []
        assert cr_run_cell.manifest_cell_problems(man, arm_id, "geowalk", {}, 42) != []
        assert cr_run_cell.cell_filename(arm_id, "ranolazine_mpo", 42) == (
            f"ours_{arm_id}_ranolazine_mpo_s42_128d.json")
