"""Tests for spherical transforms.

Tests geodesic distance, exponential/logarithmic maps, spherical whitening,
and geodesic trust region sampling. All CPU-only, no external dependencies.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.spherevo.transforms import (
    GeodesicTrustRegion,
    SphericalWhitening,
    exponential_map,
    geodesic_distance,
    logarithmic_map,
)


# ---------------------------------------------------------------------------
# Helper: generate random unit vectors on CPU
# ---------------------------------------------------------------------------


def _random_unit_vectors(n: int, d: int, seed: int = 42) -> torch.Tensor:
    torch.manual_seed(seed)
    x = torch.randn(n, d)
    return F.normalize(x, p=2, dim=-1)


# ---------------------------------------------------------------------------
# _normalize (via solver module, but tested through transforms behavior)
# ---------------------------------------------------------------------------


class TestNormalizeOnSphere:
    """Points produced by transforms should lie on the unit sphere."""

    def test_whitening_produces_unit_vectors(self):
        X = _random_unit_vectors(20, 16)
        sw = SphericalWhitening(device="cpu")
        X_t = sw.fit_transform(X)
        norms = X_t.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones(20), atol=1e-5, rtol=1e-5)

    def test_geodesic_tr_produces_unit_vectors(self):
        center = _random_unit_vectors(1, 16).squeeze(0)
        gtr = GeodesicTrustRegion(max_angle=0.5, global_fraction=0.2, device="cpu")
        samples = gtr.sample(center, n_samples=100)
        norms = samples.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones(100), atol=1e-5, rtol=1e-5)

    def test_exp_map_stays_on_sphere(self):
        base = _random_unit_vectors(1, 16)
        # Create a tangent vector (orthogonal to base)
        raw = torch.randn(1, 16)
        tangent = raw - (raw * base).sum(dim=-1, keepdim=True) * base
        result = exponential_map(base, tangent, t=1.0)
        norms = result.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones(1), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# Geodesic distance
# ---------------------------------------------------------------------------


class TestGeodesicDistance:
    """Tests for geodesic_distance function."""

    def test_same_point_zero_distance(self):
        x = _random_unit_vectors(5, 16)
        dist = geodesic_distance(x, x)
        # The implementation clamps cos_sim to [-1+1e-6, 1-1e-6], so
        # arccos(1-1e-6) ~ 0.0014, not exactly 0.
        torch.testing.assert_close(dist, torch.zeros(5), atol=2e-3, rtol=1e-3)

    def test_symmetric(self):
        x = _random_unit_vectors(10, 16, seed=42)
        y = _random_unit_vectors(10, 16, seed=99)
        d_xy = geodesic_distance(x, y)
        d_yx = geodesic_distance(y, x)
        torch.testing.assert_close(d_xy, d_yx, atol=1e-6, rtol=1e-6)

    def test_non_negative(self):
        x = _random_unit_vectors(10, 16, seed=42)
        y = _random_unit_vectors(10, 16, seed=99)
        dist = geodesic_distance(x, y)
        assert (dist >= -1e-6).all(), f"Negative distances: {dist}"

    def test_antipodal_points(self):
        """Antipodal points should have distance close to pi."""
        x = _random_unit_vectors(5, 16)
        y = -x
        dist = geodesic_distance(x, y)
        expected = torch.full((5,), math.pi)
        torch.testing.assert_close(dist, expected, atol=1e-3, rtol=1e-3)

    def test_orthogonal_points(self):
        """Two orthogonal unit vectors should have distance pi/2."""
        x = torch.zeros(1, 8)
        x[0, 0] = 1.0
        y = torch.zeros(1, 8)
        y[0, 1] = 1.0
        dist = geodesic_distance(x, y)
        expected = torch.tensor([math.pi / 2])
        torch.testing.assert_close(dist, expected, atol=1e-4, rtol=1e-4)

    def test_known_angle(self):
        """Two vectors at 60 degrees should have distance pi/3."""
        x = torch.tensor([[1.0, 0.0]])
        y = torch.tensor([[0.5, math.sqrt(3) / 2]])
        dist = geodesic_distance(x, y)
        expected = torch.tensor([math.pi / 3])
        torch.testing.assert_close(dist, expected, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# Exponential and logarithmic maps
# ---------------------------------------------------------------------------


class TestExpLogMaps:
    """Tests for exp/log map consistency."""

    def test_exp_with_zero_tangent_stays_at_base(self):
        base = _random_unit_vectors(1, 16)
        tangent = torch.zeros(1, 16)
        result = exponential_map(base, tangent, t=1.0)
        torch.testing.assert_close(result, F.normalize(base, p=2, dim=-1), atol=1e-5, rtol=1e-5)

    def test_log_of_same_point_is_zero(self):
        base = _random_unit_vectors(1, 16)
        tangent = logarithmic_map(base, base)
        # The implementation clamps cos_sim to [-1+1e-6, 1-1e-6],
        # so arccos(1-1e-6) ~ 0.0014, not exactly 0.
        assert tangent.norm().item() < 2e-3, f"log_map(p, p) should be ~0, got norm={tangent.norm().item()}"

    def test_exp_log_roundtrip(self):
        """exp_base(log_base(target)) should recover target."""
        base = _random_unit_vectors(1, 16, seed=42)
        target = _random_unit_vectors(1, 16, seed=99)
        tangent = logarithmic_map(base, target)
        recovered = exponential_map(base, tangent, t=1.0)
        # Both should be on the sphere, so compare directions
        cos_sim = (recovered * F.normalize(target, p=2, dim=-1)).sum(dim=-1)
        assert cos_sim.item() > 0.99, f"Round-trip failed: cos_sim={cos_sim.item()}"

    def test_exp_distance_matches_tangent_norm(self):
        """geodesic_distance(base, exp(base, v)) should equal ||v||."""
        base = _random_unit_vectors(1, 16)
        raw = torch.randn(1, 16) * 0.3
        # Project to tangent space
        tangent = raw - (raw * base).sum(dim=-1, keepdim=True) * base
        result = exponential_map(base, tangent, t=1.0)
        dist = geodesic_distance(base, result)
        expected_dist = tangent.norm(dim=-1)
        torch.testing.assert_close(dist, expected_dist, atol=1e-3, rtol=1e-3)


# ---------------------------------------------------------------------------
# SphericalWhitening
# ---------------------------------------------------------------------------


class TestSphericalWhitening:
    """Tests for Householder-based spherical whitening."""

    def test_mean_direction_becomes_north_pole(self):
        X = _random_unit_vectors(50, 16)
        sw = SphericalWhitening(device="cpu")
        X_t = sw.fit_transform(X)
        # Mean of transformed data should point along e1
        mean_dir = F.normalize(X_t.mean(dim=0), p=2, dim=0)
        e1 = torch.zeros(16)
        e1[0] = 1.0
        cos_sim = (mean_dir * e1).sum()
        assert cos_sim.item() > 0.95, f"Mean not aligned with north pole: cos_sim={cos_sim.item()}"

    def test_inverse_transform_recovers_original(self):
        X = _random_unit_vectors(20, 16)
        sw = SphericalWhitening(device="cpu")
        X_t = sw.fit_transform(X)
        X_rec = sw.inverse_transform(X_t)
        # Householder is self-inverse, so inverse_transform(transform(X)) = X
        cos_sim = (F.normalize(X, p=2, dim=-1) * X_rec).sum(dim=-1)
        assert (cos_sim > 0.99).all(), "Round-trip through whitening failed"

    def test_householder_is_orthogonal(self):
        X = _random_unit_vectors(10, 8)
        sw = SphericalWhitening(device="cpu")
        sw.fit(X)
        H = sw.H
        # H @ H^T should be identity
        HHt = H @ H.T
        torch.testing.assert_close(HHt, torch.eye(8), atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# GeodesicTrustRegion
# ---------------------------------------------------------------------------


class TestGeodesicTrustRegion:
    """Tests for geodesic trust region sampling."""

    def test_samples_within_cap(self):
        """Local samples should be within max_angle of center."""
        center = _random_unit_vectors(1, 16).squeeze(0)
        max_angle = 0.5
        gtr = GeodesicTrustRegion(
            max_angle=max_angle, global_fraction=0.0, device="cpu"
        )
        samples = gtr.sample(center, n_samples=200)
        dists = geodesic_distance(
            center.unsqueeze(0).expand(200, -1), samples
        )
        # All local samples should be within max_angle (with small tolerance)
        assert (dists < max_angle + 0.05).all(), (
            f"Some samples outside cap: max dist = {dists.max().item():.4f}"
        )

    def test_correct_number_of_samples(self):
        center = _random_unit_vectors(1, 16).squeeze(0)
        gtr = GeodesicTrustRegion(max_angle=0.5, global_fraction=0.2, device="cpu")
        samples = gtr.sample(center, n_samples=100)
        assert samples.shape == (100, 16)

    def test_global_fraction_produces_distant_samples(self):
        """With 100% global fraction, samples should span the sphere."""
        center = _random_unit_vectors(1, 16).squeeze(0)
        gtr = GeodesicTrustRegion(
            max_angle=0.1, global_fraction=1.0, device="cpu"
        )
        samples = gtr.sample(center, n_samples=500)
        dists = geodesic_distance(
            center.unsqueeze(0).expand(500, -1), samples
        )
        # At least some samples should be far from center
        assert dists.max().item() > 0.5, (
            f"Global samples too close to center: max dist = {dists.max().item()}"
        )
