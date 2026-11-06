"""Tests for the rebuttal solvers (Linear BO, RAASP, GeoWalk radius variants,
convex-hull / ellipsoid alternative supports). Guards the correctness of code
that produces paper numbers. dim=32 MockBlackBox unless a test needs 256-D.
"""
import math

import numpy as np
import pytest
import torch

from src.spherevo.solver import (
    ConvexHullRandomSearchSolver,
    EllipsoidRandomSearchSolver,
    GeoWalkSolver,
)


# --------------------------------------------------------------------------
# GeoWalk: default behaviour must be byte-identical to the pre-radius-kwarg code
# (every headline paper number was produced on that path).
# --------------------------------------------------------------------------
class TestGeoWalkDefaultInvariance:
    def test_default_base_radius_is_mean_norm(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        g = GeoWalkSolver(mock_black_box, x0, y0)
        assert g._base_radius == pytest.approx(g.mean_norm)

    def test_default_sample_radius_consumes_no_rng(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        g = GeoWalkSolver(mock_black_box, x0, y0)
        st = np.random.get_state()[1].copy()
        for _ in range(50):
            assert g._sample_radius() == g._base_radius
        assert np.array_equal(st, np.random.get_state()[1])  # RNG stream untouched

    def test_default_equals_explicit_default(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(0)
        a = GeoWalkSolver(mock_black_box, x0.copy(), y0.copy()); a.solve(15)
        np.random.seed(0)
        b = GeoWalkSolver(mock_black_box, x0.copy(), y0.copy(),
                          radius_scale=1.0, shell_eps=0.0); b.solve(15)
        assert np.allclose(a.all_scores, b.all_scores)

    def test_default_candidates_on_mean_norm_shell(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        np.random.seed(1)
        g = GeoWalkSolver(mock_black_box, x0, y0, total_budget=25)
        g.iteration = 0
        assert np.linalg.norm(g.next_candidate()) == pytest.approx(g.mean_norm, rel=1e-6)
        g.iteration = 20  # refinement phase
        assert np.linalg.norm(g.next_candidate()) == pytest.approx(g.mean_norm, rel=1e-6)


# --------------------------------------------------------------------------
# GeoWalk radius variants honor their configured radius.
# --------------------------------------------------------------------------
class TestGeoWalkRadiusVariants:
    def test_radius_override(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        g = GeoWalkSolver(mock_black_box, x0, y0, radius_override=3.0)
        assert g._base_radius == 3.0
        g.iteration = 0
        assert np.linalg.norm(g.next_candidate()) == pytest.approx(3.0, rel=1e-6)

    def test_radius_scale(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        g = GeoWalkSolver(mock_black_box, x0, y0, radius_scale=2.0)
        assert g._base_radius == pytest.approx(2.0 * g.mean_norm)

    def test_median_estimator(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        g = GeoWalkSolver(mock_black_box, x0, y0, radius_estimator="median")
        assert g._base_radius == pytest.approx(np.median(np.linalg.norm(x0, axis=1)))

    def test_shell_eps_band(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        g = GeoWalkSolver(mock_black_box, x0, y0, shell_eps=0.1)
        rs = [g._sample_radius() for _ in range(500)]
        assert all(g._base_radius * 0.9 <= r <= g._base_radius * 1.1 for r in rs)
        assert len(set(rs)) > 1  # jitter actually applied

    def test_radius_range_precedence(self, mock_black_box, cold_start_data):
        x0, y0 = cold_start_data
        g = GeoWalkSolver(mock_black_box, x0, y0, radius_range=(6.81, 9.14), shell_eps=0.5)
        rs = [g._sample_radius() for _ in range(500)]
        assert all(6.81 <= r <= 9.14 for r in rs)  # radius_range wins over shell_eps


# --------------------------------------------------------------------------
# Papenmeier MSR lengthscale fix: locks 0.1*sqrt(D) (was sqrt(D), 10x too large).
# --------------------------------------------------------------------------
def test_papenmeier_lengthscale_init_is_0p1_sqrt_d(mock_black_box, cold_start_data):
    from src.spherevo.gp_competitors import PapenmeierMSRSolver
    x0, y0 = cold_start_data
    D = x0.shape[1]
    s = PapenmeierMSRSolver(black_box=mock_black_box, x0=x0, y0=y0)
    X = torch.tensor(x0, dtype=torch.float64)
    Y = torch.tensor(y0, dtype=torch.float64)
    gp = s._make_gp(X, Y)
    ls = gp.covar_module.base_kernel.lengthscale.detach().flatten()
    assert torch.allclose(ls, torch.full_like(ls, 0.1 * math.sqrt(D)))


# --------------------------------------------------------------------------
# RAASP: trust-region state machine + sparse-perturbation contract.
# Tested on a bare object so no 256-D ZINC bounds are needed.
# --------------------------------------------------------------------------
def _bare_raasp(dim=256, L=0.8):
    from src.spherevo.gp_competitors import PapenmeierMSRRaaspSolver
    obj = object.__new__(PapenmeierMSRRaaspSolver)
    obj._dim = dim
    obj._L = L
    obj._succ = obj._fail = 0
    obj._SUCC_TOL = PapenmeierMSRRaaspSolver._SUCC_TOL
    obj._L0 = PapenmeierMSRRaaspSolver._L0
    obj._L_MIN = PapenmeierMSRRaaspSolver._L_MIN
    obj._L_MAX = PapenmeierMSRRaaspSolver._L_MAX
    obj._fail_tol = max(4, int(math.ceil(dim)))
    obj._tr_best = 0.0
    return obj


class TestRAASP:
    def test_tr_expands_after_successes(self):
        o = _bare_raasp()
        for i in range(o._SUCC_TOL):
            o.post_update(None, np.array([[float(i + 1)]]))  # strictly improving
        assert o._L == pytest.approx(min(o._L0 * 2, o._L_MAX))
        assert o._succ == 0

    def test_tr_shrinks_after_failures(self):
        o = _bare_raasp()
        o._tr_best = 10.0
        for _ in range(o._fail_tol):
            o.post_update(None, np.array([[1.0]]))  # non-improving
        assert o._L == pytest.approx(o._L0 / 2)
        assert o._fail == 0

    def test_success_streak_resets_on_failure(self):
        o = _bare_raasp()
        o.post_update(None, np.array([[1.0]]))
        o.post_update(None, np.array([[2.0]]))
        o.post_update(None, np.array([[1.5]]))  # failure
        assert o._succ == 0

    def test_L_clamped(self):
        o = _bare_raasp(L=1.5)
        for i in range(30):  # many successes
            o.post_update(None, np.array([[float(i + 1)]]))
        assert o._L <= o._L_MAX

    def test_raasp_perturbation_contract(self):
        o = _bare_raasp(dim=256)
        x_c = np.full(256, 0.5)
        cands = o._raasp_candidates(x_c, np.ones(256), 2000)
        assert cands.shape == (2000, 256)
        ndiff = (np.abs(cands - x_c) > 1e-12).sum(axis=1)
        assert 12 <= ndiff.mean() <= 28          # ~ min(1,20/256)*256 = 20
        assert (ndiff >= 1).all()                # no all-zero mask
        assert cands.min() >= 0.0 and cands.max() <= 1.0

    def test_raasp_full_perturb_low_dim(self):
        o = _bare_raasp(dim=10)
        cands = o._raasp_candidates(np.full(10, 0.5), np.ones(10), 200)
        # prob = min(1, 20/10) = 1 -> essentially all dims perturbed
        assert (np.abs(cands - 0.5) > 1e-12).sum(axis=1).mean() > 9


# --------------------------------------------------------------------------
# Linear BO on-sphere: end-to-end at dim=32, candidates on the shell.
# --------------------------------------------------------------------------
def test_linear_bo_sphere_end_to_end(mock_black_box, cold_start_data):
    from src.spherevo.gp_competitors import LinearBOSphereSolver
    x0, y0 = cold_start_data
    s = LinearBOSphereSolver(black_box=mock_black_box, x0=x0, y0=y0)
    s.solve(max_iter=3)
    assert len(s.all_scores) == len(y0) + 3
    assert s._fallback_count == 0  # clean GP fits on mock data
    # last proposed candidate is on the empirical shell
    c = s.next_candidate()
    assert np.linalg.norm(c) == pytest.approx(s.mean_norm, rel=1e-3)


# --------------------------------------------------------------------------
# Alternative supports enforce the SELFIES-256D contract at dim!=256.
# --------------------------------------------------------------------------
@pytest.mark.parametrize("cls", [ConvexHullRandomSearchSolver, EllipsoidRandomSearchSolver])
def test_alt_support_dim_guard(cls, mock_black_box, cold_start_data):
    cls._Z_CACHE = None
    cls._STATS_CACHE = None
    x0, y0 = cold_start_data  # dim=32
    with pytest.raises((ValueError, FileNotFoundError)):
        cls(black_box=mock_black_box, x0=x0, y0=y0)


# --------------------------------------------------------------------------
# ProximitySelectionSolver: the surrogate-free null model for what the on-sphere
# GP's acquisition argmax contributes at N=30. Its whole scientific value rests
# on the selection rule being exactly what it claims, so pin the contract.
# --------------------------------------------------------------------------
class TestProximitySelection:
    def _solver(self, bb, data, **kw):
        from src.spherevo.solver import ProximitySelectionSolver
        x0, y0 = data
        return ProximitySelectionSolver(bb, x0, y0, n_candidates=256, **kw)

    def test_candidates_lie_on_the_empirical_shell(self, mock_black_box, cold_start_data):
        s = self._solver(mock_black_box, cold_start_data)
        for _ in range(10):
            x = s.next_candidate()
            assert np.linalg.norm(x) == pytest.approx(s.mean_norm)

    def test_near_mode_selects_within_the_requested_quantile(
        self, mock_black_box, cold_start_data
    ):
        """Selection is uniform over the nearest q fraction, so the population
        quantile of the pick is ~Uniform[0, q]: mean q/2, and essentially never
        above q. Assert the induced distribution, not a per-draw bound -- with a
        finite pool the k-th order statistic overshoots q by O(sqrt(q/n)) and a
        per-draw assertion is flaky by construction.
        """
        q, n_draws = 0.10, 400
        s = self._solver(mock_black_box, cold_start_data, q_select=q)
        x_hist, y_hist = s.get_history_as_arrays(penalize_nans_with=-1.0)
        x_star = x_hist[int(np.argmax(y_hist.flatten()))]
        u_star = x_star / np.linalg.norm(x_star)
        ref = np.random.randn(8192, s._dim)
        ref /= np.linalg.norm(ref, axis=1, keepdims=True)
        ref_cos = np.sort(ref @ u_star)

        quantiles = np.array([
            1.0 - np.searchsorted(ref_cos, s.next_candidate().ravel() @ u_star / s.mean_norm)
            / len(ref_cos)
            for _ in range(n_draws)
        ])
        assert quantiles.mean() == pytest.approx(q / 2, abs=0.02)
        assert quantiles.max() < 2 * q
        # A uniform-random selector would sit at 0.5; the rule must be far from it.
        assert quantiles.mean() < 0.2

    def test_far_mode_is_the_mirror_image(self, mock_black_box, cold_start_data):
        near = self._solver(mock_black_box, cold_start_data, q_select=0.10, mode="near")
        far = self._solver(mock_black_box, cold_start_data, q_select=0.10, mode="far")
        x_hist, y_hist = near.get_history_as_arrays(penalize_nans_with=-1.0)
        x_star = x_hist[int(np.argmax(y_hist.flatten()))]
        u_star = x_star / np.linalg.norm(x_star)
        cos_near = [float(near.next_candidate().ravel() @ u_star / near.mean_norm) for _ in range(50)]
        cos_far = [float(far.next_candidate().ravel() @ u_star / far.mean_norm) for _ in range(50)]
        assert min(cos_near) > max(cos_far)

    def test_no_surrogate_is_fitted(self, mock_black_box, cold_start_data):
        """The claim is 'no model'; assert the object carries no fitted state."""
        s = self._solver(mock_black_box, cold_start_data)
        s.next_candidate()
        assert not any(
            k for k in vars(s) if "gp" in k.lower() or "model" in k.lower()
        )

    def test_rejects_invalid_configuration(self, mock_black_box, cold_start_data):
        with pytest.raises(ValueError):
            self._solver(mock_black_box, cold_start_data, q_select=0.0)
        with pytest.raises(ValueError):
            self._solver(mock_black_box, cold_start_data, mode="sideways")

    def test_end_to_end_runs_and_improves_over_cold_start(
        self, mock_black_box, cold_start_data
    ):
        s = self._solver(mock_black_box, cold_start_data)
        s.solve(max_iter=15)
        assert s.get_best_performance() >= float(np.max(cold_start_data[1]))


# --------------------------------------------------------------------------
# Sphere-continuous acquisition maximiser: the last cell of the support x
# acquisition-optimizer factorial. Its scientific value rests entirely on it
# being a genuine continuous maximiser ON the shell -- not an argmax with extra
# steps, and not a box optimiser with a projection bolted on -- so pin both.
# --------------------------------------------------------------------------
class TestSphereContinuous:
    def _solver(self, bb, data):
        from src.spherevo.gp_competitors import HvarfnerScaledGPSphereContinuousSolver
        x0, y0 = data
        return HvarfnerScaledGPSphereContinuousSolver(black_box=bb, x0=x0, y0=y0)

    def test_every_candidate_lands_exactly_on_the_shell(
        self, mock_black_box, cold_start_data,
    ):
        s = self._solver(mock_black_box, cold_start_data)
        for _ in range(3):
            c = s.next_candidate()
            assert np.linalg.norm(c) == pytest.approx(s.mean_norm, rel=1e-9)
            s.step()
        assert s._fallback_count == 0

    def test_ascent_never_returns_worse_than_its_best_start(
        self, mock_black_box, cold_start_data,
    ):
        """The arm is seeded from a raw-sample argmax, so if gradient ascent on
        the sphere ever returned a lower-acquisition point the cell would be a
        *worse* argmax rather than a better optimiser, and the factorial's
        optimizer axis would be mislabelled."""
        from botorch.acquisition import qLogExpectedImprovement
        from botorch.fit import fit_gpytorch_mll
        from gpytorch.mlls import ExactMarginalLogLikelihood

        s = self._solver(mock_black_box, cold_start_data)
        s.solve(max_iter=2)
        x_hist, y_hist = s.get_history_as_arrays(penalize_nans_with=-1.0)
        X = torch.tensor(x_hist, dtype=torch.float64, device=s._device)
        Y = torch.tensor(
            y_hist.flatten(), dtype=torch.float64, device=s._device,
        ).unsqueeze(-1)
        Y = (Y - Y.mean()) / Y.std()
        gp = s._make_gp(X, Y)
        fit_gpytorch_mll(ExactMarginalLogLikelihood(gp.likelihood, gp))
        gp.eval()
        best_y = float(Y.max())

        torch.manual_seed(0)
        cand = s._maximize_on_sphere(gp, best_y, X.shape[-1])
        acqf = qLogExpectedImprovement(model=gp, best_f=best_y)
        with torch.no_grad():
            got = float(acqf(torch.tensor(
                cand, dtype=torch.float64, device=s._device,
            ).unsqueeze(1)))
            u = torch.randn(
                s.n_raw_samples, X.shape[-1], dtype=torch.float64, device=s._device,
            )
            u = u / u.norm(dim=-1, keepdim=True) * s.mean_norm
            raw_best = float(acqf(u.unsqueeze(1)).max())
        # Ascent starts from the top of a draw of the same size, so it must at
        # least match a fresh independent draw's best.
        assert got >= raw_best - 1e-6

    def test_restart_budget_mirrors_the_box_arm(self, mock_black_box, cold_start_data):
        """optimize_acqf(num_restarts=10, raw_samples=512) in _BoxFaithfulMixin;
        if these drift apart the two continuous arms differ in effort as well as
        in support, which is the confound this cell exists to remove."""
        s = self._solver(mock_black_box, cold_start_data)
        assert (s.n_restarts, s.n_raw_samples) == (10, 512)
