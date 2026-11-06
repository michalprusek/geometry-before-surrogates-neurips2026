"""Control block of the camera-ready run rules (amendment of 2026-10-12): the
three new solver classes.

* C1 ``hvarfner_scaled_gp_box_argmax_raw``: ``hvarfner_scaled_gp_box_argmax``
  with the GP fitted on raw latents. Pinned here: the GP sees raw latents (the
  parent sees ``(x - p01) / width``); qLogEI scores the parent's box draws in
  raw coordinates; with identity bounds (p01 = 0, p99 = 1) the two arms are
  bit-identical, so the input transform is the only difference; the
  random-number use and the fallbacks are the parent's.
* C2a ``hvarfner_scaled_gp_boxdir_argmax``: the same box draws renormalised to
  the cold-start mean norm, GP on raw latents, qLogEI arg-max.
* C2b ``random_boxdir``: ``random_box_faithful``'s draw renormalised to the
  cold-start mean norm, NumPy's generator only, no surrogate.

The problems here are deterministic (the shared ``MockCodec`` / ``MockOracle``
hash strings, and Python randomises string hashes per process) and their
cold starts lie inside the box, as encoder means do: with the cold start far
outside a box with a 1e-6-wide coordinate, the rescaled GP inputs reach 1e6
and the first fit attempt of ``hvarfner_scaled_gp_box_argmax`` can fail, after
which BoTorch retries it with ``sample_all_priors`` and consumes random
numbers. The tests that reproduce a step's random draws assert that no retry
happened (``fit_retries``), so a failure says so instead of a draw mismatch.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from src.spherevo.gp_competitors import (
    HvarfnerScaledGPBoxArgmaxRawSolver,
    HvarfnerScaledGPBoxArgmaxSolver,
    HvarfnerScaledGPBoxDirArgmaxSolver,
    HvarfnerScaledGPSolver,
    _BoxFaithfulMixin,
)
from src.spherevo.solver import RandomBoxDirSolver, RandomBoxFaithfulSolver

D = 32
N0 = 5
DEAD_DIM = 7  # p01 == p99 on this coordinate -> the 1e-6 width floor

C1 = HvarfnerScaledGPBoxArgmaxRawSolver
C2A = HvarfnerScaledGPBoxDirArgmaxSolver
BOX_ARGMAX = HvarfnerScaledGPBoxArgmaxSolver

ARMS = {
    "hvarfner_scaled_gp_box_argmax_raw": C1,
    "hvarfner_scaled_gp_boxdir_argmax": C2A,
    "random_boxdir": RandomBoxDirSolver,
}


def black_box(z: np.ndarray) -> np.ndarray:
    """Deterministic, smooth, not flat."""
    z = np.atleast_2d(z)
    return np.sin(z[:, :4].sum(axis=1)).reshape(-1, 1)


class Problem:
    """Bounds file, cold start inside the box, and the black box."""

    def __init__(self, tmp_path, *, identity: bool = False):
        if identity:          # p01 = 0, p99 = 1: both rescalings are the identity
            p01, p99 = np.zeros(D), np.ones(D)
        else:                 # varied widths and one dead coordinate
            rng = np.random.RandomState(0)
            p01 = rng.uniform(-3.0, -0.2, size=D)
            p99 = p01 + rng.uniform(0.05, 4.0, size=D)
            p01[DEAD_DIM] = p99[DEAD_DIM] = 0.123456
        self.p01, self.p99 = p01, p99
        self.width = np.where(p99 - p01 > 1e-6, p99 - p01, 1e-6)
        self.path = tmp_path / ("bounds_identity.json" if identity else "bounds_varied.json")
        self.path.write_text(json.dumps({"per_coord": {
            "p01": p01.tolist(), "p99": p99.tolist(),
            "min": (p01 - 1).tolist(), "max": (p99 + 1).tolist(),
        }}))
        self.bp = str(self.path)
        self.x0 = p01 + np.random.RandomState(100).uniform(0.1, 0.9, size=(N0, D)) * self.width
        self.y0 = black_box(self.x0)
        self.bb = black_box

    def gp(self, cls, *, bb=None, y0=None):
        return cls(black_box=bb or self.bb, x0=self.x0,
                   y0=self.y0 if y0 is None else y0, bounds_path=self.bp)

    def in_box(self, z) -> bool:
        return bool(np.all(z >= self.p01) and np.all(z <= self.p01 + self.width))


@pytest.fixture
def problem(tmp_path):
    return Problem(tmp_path)


@pytest.fixture
def fit_retries(monkeypatch):
    """Calls of BoTorch's ``sample_all_priors`` (a fit retry), recorded."""
    import botorch.fit

    calls = []
    original = botorch.fit.sample_all_priors

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(botorch.fit, "sample_all_priors", spy)
    return calls


def _renorm(x: torch.Tensor, radius: float) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True) * radius


def _rng_states(device: torch.device) -> list[torch.Tensor]:
    states = [torch.get_rng_state()]
    if device.type == "cuda":
        states.append(torch.cuda.get_rng_state(device))
    return states


def _create(name, bb, x0, y0, codec=None):
    from scripts.run_benchmark import _create_solver

    return _create_solver(
        solver_name=name, black_box=bb, x0=x0, y0=y0,
        n_dimensions=x0.shape[1], subspace_dim=16, n_subspaces=16, seed=42,
        n_initial_points=len(y0), codec=codec, norm_mode="fixed", n_iterations=25,
        cs_smiles=None,
    )


# =========================================================================
# registration and construction
# =========================================================================


@pytest.mark.parametrize("name", sorted(ARMS))
def test_the_arm_resolves_to_its_class(name, problem, monkeypatch):
    from scripts.run_benchmark import OUR_SOLVERS, SUPPORTED_SOLVERS

    assert name in OUR_SOLVERS and name in SUPPORTED_SOLVERS
    monkeypatch.setattr(BOX_ARGMAX, "default_bounds_path", problem.bp)
    monkeypatch.setattr(RandomBoxFaithfulSolver, "default_bounds_path", problem.bp)
    solver = _create(name, problem.bb, problem.x0, problem.y0)
    assert type(solver) is ARMS[name]
    torch.manual_seed(0)
    np.random.seed(0)
    z = solver.next_candidate()
    assert z.shape == (1, D) and np.all(np.isfinite(z))


def test_c1_and_c2a_keep_the_box_argmax_constructor_gp_and_candidate_count(problem):
    for cls in (C1, C2A):
        assert issubclass(cls, BOX_ARGMAX)
        assert cls.__init__ is _BoxFaithfulMixin.__init__
        assert cls._make_gp is HvarfnerScaledGPSolver._make_gp
        assert cls._rescale_from_unit_cube is _BoxFaithfulMixin._rescale_from_unit_cube

    # C2a overrides only the candidate map; C1 adds it and overrides next_candidate
    def methods(cls):
        return {k for k, v in vars(cls).items() if callable(v) and not k.startswith("__")}

    assert methods(C2A) == {"_candidates_from_unit"}
    assert methods(C1) == {"_candidates_from_unit", "next_candidate"}
    assert C2A.next_candidate is C1.next_candidate
    ref = problem.gp(BOX_ARGMAX)
    for cls in (C1, C2A):
        s = problem.gp(cls)
        assert s.n_candidates == ref.n_candidates == 2048
        assert torch.equal(s._lo, ref._lo) and torch.equal(s._width, ref._width)
        assert s.mean_norm == ref.mean_norm


def test_with_the_real_bounds_every_box_arm_loads_the_same_box():
    """SELFIES (codec None) -> the same ZINC [p01, p99] file for box_argmax, C1, C2a
    (the class default, no codec dispatch) and random_box_faithful, C2b
    (codec dispatch, None -> the default)."""
    from pathlib import Path

    from src.analysis.config import ZINC_PER_COORD_BOUNDS_PATH

    if not (Path(__file__).resolve().parents[1] / ZINC_PER_COORD_BOUNDS_PATH).exists():
        pytest.skip("real 256-D bounds file not present in this tree")
    rng = np.random.RandomState(0)
    x0 = rng.randn(5, 256) * 0.5
    y0 = rng.rand(5, 1)

    def bb(z):
        return np.zeros((np.atleast_2d(z).shape[0], 1))

    names = ("hvarfner_scaled_gp_box_argmax", "hvarfner_scaled_gp_box_argmax_raw",
             "hvarfner_scaled_gp_boxdir_argmax", "random_box_faithful", "random_boxdir")
    solvers = {n: _create(n, bb, x0, y0) for n in names}
    ref = solvers["hvarfner_scaled_gp_box_argmax"]
    lo, width = ref._lo.cpu().numpy(), ref._width.cpu().numpy()
    assert lo.shape == (256,)
    for n, s in solvers.items():
        s_lo = s._lo.cpu().numpy() if torch.is_tensor(s._lo) else s._lo
        s_w = s._width.cpu().numpy() if torch.is_tensor(s._width) else s._width
        assert np.array_equal(s_lo, lo) and np.array_equal(s_w, width), n


# =========================================================================
# C1: only the GP's input transform differs from hvarfner_scaled_gp_box_argmax
# =========================================================================


def test_c1_equals_box_argmax_bit_for_bit_when_the_rescaling_is_the_identity(tmp_path):
    p = Problem(tmp_path, identity=True)
    runs = []
    for cls in (BOX_ARGMAX, C1):
        s = p.gp(cls)
        torch.manual_seed(3)
        np.random.seed(3)
        s.solve(max_iter=5)
        runs.append((np.concatenate(s.history["x"]), list(s.all_scores),
                     s._fallback_count, list(s._fallback_reasons)))
    (xa, ya, fa, ra), (xb, yb, fb, rb) = runs
    assert np.array_equal(xa, xb) and ya == yb and fa == fb == 0 and ra == rb == []
    assert len(ya) == N0 + 5
    assert len({tuple(x) for x in xa[N0:]}) == 5          # 5 distinct proposals


@pytest.mark.parametrize("cls,raw", [(BOX_ARGMAX, False), (C1, True), (C2A, True)])
def test_the_gp_inputs_are_raw_in_c1_and_c2a_and_rescaled_in_box_argmax(
        cls, raw, problem, monkeypatch):
    seen = []
    original = HvarfnerScaledGPSolver._make_gp

    def spy(self, X, Y):
        seen.append(X.detach().cpu().numpy().copy())
        return original(self, X, Y)

    monkeypatch.setattr(HvarfnerScaledGPSolver, "_make_gp", spy)
    s = problem.gp(cls)
    torch.manual_seed(0)
    s.solve(max_iter=2)
    assert len(seen) == 2
    x_hist = np.concatenate(s.history["x"])
    for step, X in enumerate(seen):
        hist = x_hist[: N0 + step]
        expected = hist if raw else (hist - problem.p01) / problem.width
        assert np.array_equal(X, expected), (cls.__name__, step)


@pytest.mark.parametrize("cls", [BOX_ARGMAX, C1, C2A])
def test_qlogei_scores_the_draws_in_the_gps_input_space(cls, problem, monkeypatch,
                                                        fit_retries):
    """box_argmax scores the unit-cube draw u (its GP's input space); C1 scores
    the same draw in raw coordinates p01 + u * width (its GP's input space);
    C2a scores those points renormalised to the mean norm."""
    import src.spherevo.gp_competitors as gpc

    seen = []

    class Spy(gpc.qLogExpectedImprovement):
        def forward(self, X):
            seen.append(X.detach().cpu().numpy().copy())
            return super().forward(X)

    monkeypatch.setattr(gpc, "qLogExpectedImprovement", Spy)
    s = problem.gp(cls)
    torch.manual_seed(21)
    s.next_candidate()
    assert fit_retries == [] and s._fallback_count == 0
    torch.manual_seed(21)
    u = torch.rand(s.n_candidates, D, dtype=torch.float64, device=s._device)
    expected = {BOX_ARGMAX: u, C1: s._lo + u * s._width,
                C2A: _renorm(s._lo + u * s._width, s.mean_norm)}[cls]
    (X,) = seen
    assert X.shape == (s.n_candidates, 1, D)
    assert np.array_equal(X[:, 0, :], expected.cpu().numpy())


@pytest.mark.parametrize("cls", [C1, C2A])
def test_the_proposal_is_one_of_the_box_argmax_draws_in_raw_coordinates(
        cls, problem, fit_retries):
    """The candidate set is the parent's ``torch.rand(2048, D)`` mapped to
    ``p01 + u * width`` (C1), renormalised to the mean norm (C2a)."""
    s = problem.gp(cls)
    for seed in (11, 12):
        torch.manual_seed(seed)
        z = s.next_candidate()
        assert fit_retries == [] and s._fallback_count == 0
        torch.manual_seed(seed)
        u = torch.rand(s.n_candidates, D, dtype=torch.float64, device=s._device)
        cand = s._lo + u * s._width
        if cls is C2A:
            cand = _renorm(cand, s.mean_norm)
        hits = np.flatnonzero((cand.cpu().numpy() == z).all(axis=1))
        assert len(hits) == 1, (cls.__name__, seed)
        if cls is C1:
            assert problem.in_box(z)
        else:
            assert np.linalg.norm(z) == pytest.approx(s.mean_norm, rel=1e-12)


def test_a_gp_step_uses_the_random_numbers_of_box_argmax(problem, fit_retries):
    """Same calls, same order: one torch.rand(2048, D) on the solver's device and
    BoTorch's lazy sampler seed (one torch.randint on the CPU generator); NumPy's
    generator untouched."""
    after = {}
    device = None
    for cls in (BOX_ARGMAX, C1, C2A):
        s = problem.gp(cls)
        device = s._device
        np.random.seed(0)
        np_before = np.random.get_state()
        torch.manual_seed(5)
        s.next_candidate()
        assert fit_retries == [] and s._fallback_count == 0, cls.__name__
        after[cls] = _rng_states(device)
        np_after = np.random.get_state()
        assert np_before[0] == np_after[0] and np.array_equal(np_before[1], np_after[1])
        assert np_before[2:] == np_after[2:]
    torch.manual_seed(5)
    torch.rand(2048, D, dtype=torch.float64, device=device)
    torch.randint(0, 1000000, (1,))
    ref = _rng_states(device)
    for cls, states in after.items():
        assert all(torch.equal(a, b) for a, b in zip(states, ref)), cls.__name__


# =========================================================================
# C2a: the same draws on the shell radius
# =========================================================================


def test_c2a_candidates_lie_on_the_shell_with_the_directions_of_box_draws(problem):
    c1 = problem.gp(C1)
    c2a = problem.gp(C2A)
    x0_64 = np.asarray(problem.x0, dtype=np.float64)
    assert c2a.mean_norm == float(np.mean(np.linalg.norm(x0_64, axis=1)))
    torch.manual_seed(0)
    u = torch.rand(256, D, dtype=torch.float64, device=c1._device)
    box = c1._candidates_from_unit(u)
    shell = c2a._candidates_from_unit(u)
    assert torch.equal(box, c1._lo + u * c1._width)
    norms = shell.norm(dim=-1).cpu().numpy()
    assert np.allclose(norms, c2a.mean_norm, rtol=1e-12, atol=0)
    cos = (box * shell).sum(-1) / (box.norm(dim=-1) * shell.norm(dim=-1))
    assert torch.allclose(cos, torch.ones_like(cos), rtol=0, atol=1e-12)
    # a solve stays on the shell
    torch.manual_seed(1)
    c2a.solve(max_iter=3)
    for z in c2a.history["x"][N0:]:
        assert np.linalg.norm(z) == pytest.approx(c2a.mean_norm, rel=1e-12)
    assert c2a._fallback_count == 0


def test_c2a_and_c2b_map_the_same_draw_to_the_same_point(problem):
    """C2b is C2a's candidate distribution: for one u the two maps agree."""
    c2a = problem.gp(C2A)
    c2b = RandomBoxDirSolver(problem.bb, problem.x0, problem.y0, bounds_path=problem.bp)
    assert c2a.mean_norm == c2b.mean_norm
    u = np.random.RandomState(4).rand(64, D)
    gp_side = c2a._candidates_from_unit(
        torch.tensor(u, dtype=torch.float64, device=c2a._device)).cpu().numpy()
    z = c2b._lo + u * c2b._width
    rs_side = z / np.linalg.norm(z, axis=1, keepdims=True) * c2b.mean_norm
    assert np.allclose(gp_side, rs_side, rtol=0, atol=1e-12)


# =========================================================================
# fallbacks (C1, C2a): recorded as in box_argmax, on the arm's own support
# =========================================================================


@pytest.mark.parametrize("cls", [C1, C2A])
def test_the_flat_signal_fallback_is_one_draw_on_the_arms_support(cls, problem):
    y0 = np.full((N0, 1), 0.5)

    def flat(z):
        return np.full((np.atleast_2d(z).shape[0], 1), 0.5)

    zs = {}
    for k in (BOX_ARGMAX, cls):
        s = problem.gp(k, bb=flat, y0=y0)
        torch.manual_seed(2)
        s.solve(max_iter=1)
        assert s._fallback_count == 1 and s._fallback_reasons[0].startswith("flat-signal")
        zs[k] = s.history["x"][-1]
    torch.manual_seed(2)
    u = torch.rand(1, D, dtype=torch.float64, device=s._device)
    expected = s._lo + u * s._width
    if cls is C2A:
        expected = _renorm(expected, s.mean_norm)
    assert np.array_equal(zs[cls], expected.cpu().numpy())
    if cls is C1:   # literally box_argmax's fallback point
        assert np.array_equal(zs[cls], zs[BOX_ARGMAX])


@pytest.mark.parametrize("cls", [C1, C2A])
def test_a_numerical_gp_failure_is_a_recorded_fallback_and_a_bug_propagates(
        cls, problem, monkeypatch):
    from linear_operator.utils.errors import NotPSDError

    from scripts.run_benchmark import _surrogate_fallbacks

    def not_psd(self, X, Y):
        raise NotPSDError("forced")

    monkeypatch.setattr(HvarfnerScaledGPSolver, "_make_gp", not_psd)
    s = problem.gp(cls)
    torch.manual_seed(0)
    s.solve(max_iter=2)
    assert _surrogate_fallbacks(s) == (2, ["NotPSDError: forced"] * 2)
    for z in s.history["x"][N0:]:
        if cls is C1:
            assert problem.in_box(z)
        else:
            assert np.linalg.norm(z) == pytest.approx(s.mean_norm, rel=1e-12)

    def bug(self, X, Y):
        raise TypeError("a programming error")

    monkeypatch.setattr(HvarfnerScaledGPSolver, "_make_gp", bug)
    with pytest.raises(TypeError, match="programming error"):
        problem.gp(cls).next_candidate()


# =========================================================================
# C2b random_boxdir
# =========================================================================


def _box_and_dir(problem):
    return (RandomBoxFaithfulSolver(problem.bb, problem.x0, problem.y0, bounds_path=problem.bp),
            RandomBoxDirSolver(problem.bb, problem.x0, problem.y0, bounds_path=problem.bp))


def test_random_boxdir_is_the_renormalised_random_box_faithful_draw(problem):
    box, bdir = _box_and_dir(problem)
    assert np.array_equal(bdir._lo, box._lo) and np.array_equal(bdir._width, box._width)
    assert bdir.mean_norm == box.mean_norm
    for seed in (0, 1, 2):
        np.random.seed(seed)
        zb = box.next_candidate()
        np.random.seed(seed)
        zd = bdir.next_candidate()
        assert zd.shape == (1, D)
        assert np.array_equal(zd, zb / np.linalg.norm(zb) * bdir.mean_norm)
        assert np.linalg.norm(zd) == pytest.approx(bdir.mean_norm, rel=1e-12)


def test_random_boxdir_draws_once_from_numpy_and_never_from_torch(problem):
    _, bdir = _box_and_dir(problem)
    torch.manual_seed(0)
    torch_before = torch.get_rng_state()
    np.random.seed(7)
    bdir.next_candidate()
    after = np.random.get_state()
    np.random.seed(7)
    np.random.rand(1, D)
    ref = np.random.get_state()
    assert after[0] == ref[0] and np.array_equal(after[1], ref[1]) and after[2:] == ref[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_random_boxdir_runs_25_steps_and_records_no_fallback(problem):
    from scripts.run_benchmark import _surrogate_fallbacks

    _, s = _box_and_dir(problem)
    np.random.seed(1)
    s.solve(max_iter=25)
    assert len(s.all_scores) == N0 + 25
    assert not hasattr(s, "_fallback_count")
    assert _surrogate_fallbacks(s) == (0, [])
    for z in s.history["x"][N0:]:
        assert np.linalg.norm(z) == pytest.approx(s.mean_norm, rel=1e-12)


def test_random_boxdir_codec_bounds_dispatch_is_random_box_faithfuls(problem, monkeypatch):
    import src.spherevo.solver as solver_mod

    seen = []

    def fake_loader(bounds_path, dim):
        seen.append(bounds_path)
        return np.zeros(dim), np.ones(dim)

    monkeypatch.setattr(solver_mod, "load_faithful_box_bounds", fake_loader)

    class ChembedCodecX:  # the class name carries the dispatch key
        pass

    for codec in (None, ChembedCodecX()):
        for name in ("random_box_faithful", "random_boxdir"):
            _create(name, problem.bb, problem.x0, problem.y0, codec=codec)
    assert seen == [None, None, "results/precomputed/chembed_per_coord_bounds.json",
                    "results/precomputed/chembed_per_coord_bounds.json"]


# =========================================================================
# end to end: run_benchmark() writes the same record as for the twin arm
# =========================================================================


class _BB:
    def __init__(self, inner):
        self._inner = inner
        self.n_failures = 0

    def __call__(self, z):
        return self._inner(z)

    def terminate(self):
        pass


@pytest.mark.parametrize("name,twin", [
    ("hvarfner_scaled_gp_box_argmax_raw", "hvarfner_scaled_gp_box_argmax"),
    ("hvarfner_scaled_gp_boxdir_argmax", "hvarfner_scaled_gp_box_argmax"),
    ("random_boxdir", "random_box_faithful"),
])
def test_run_benchmark_records_the_new_arm_like_its_twin(name, twin, problem, monkeypatch):
    import scripts.run_benchmark as rb
    import src.oracle.guacamol as guacamol_mod

    monkeypatch.setattr(BOX_ARGMAX, "default_bounds_path", problem.bp)
    monkeypatch.setattr(RandomBoxFaithfulSolver, "default_bounds_path", problem.bp)

    class StubOracle:
        def __init__(self, task_id):
            self.task_id = task_id

        def score(self, smiles):
            return 0.0

    monkeypatch.setattr(rb, "make_our_latent_bb",
                        lambda *a, **k: (_BB(problem.bb), None, D))
    monkeypatch.setattr(guacamol_mod, "GuacaMolOracle", StubOracle)
    monkeypatch.setattr(rb, "generate_cold_start_ours",
                        lambda *a, **k: (problem.x0.copy(), problem.y0.copy(), ["C"] * N0))
    out = {}
    for solver_name in (twin, name):
        out[solver_name] = rb.run_benchmark(
            mode="ours", solver_name=solver_name, task_name="ranolazine_mpo", seed=42,
            max_iter=3, n_initial_points=10, n_cold_start=N0, latent_dim=128,
            subspace_dim=16, n_subspaces=16,
        )
    res, ref = out[name], out[twin]
    assert set(res) == set(ref)
    assert set(res["provenance"]) == set(ref["provenance"])
    assert res["status"] == "completed", res["crash_info"]
    assert res["solver"] == name and res["seed"] == 42
    assert res["n_evaluated"] == N0 + 3 == len(res["all_scores"])
    assert res["all_scores"][:N0] == ref["all_scores"][:N0]       # the same cold start
    assert res["surrogate_fallback_count"] == 0 and res["surrogate_fallback_reasons"] == []
