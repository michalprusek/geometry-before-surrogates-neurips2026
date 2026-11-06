"""Tests for the spherical Cauchy VAE implementation.

Verifies:
1. Mobius transform maps S^{d-1} to S^{d-1} (preserves sphere)
2. spCauchy samples concentrate around mu as rho -> 1
3. spCauchy samples are uniform when rho = 0
4. KL divergence is 0 when rho = 0 and increases with rho
5. Forward pass produces valid loss
6. Gradients flow through the reparameterized sample
7. Encoder outputs have correct shapes and constraints
"""

import math

import pytest
import torch
import torch.nn.functional as F

from src.codec.spcauchy_vae import (
    SpCauchyTransformerVAE,
    mobius_transform,
    sample_spcauchy,
    spcauchy_kl_analytic,
    spcauchy_kl_divergence,
    _log_C_d,
    _log_surface_area_sphere,
)


# ---------------------------------------------------------------------------
# Helper: deterministic seeding
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def seed_rng():
    torch.manual_seed(42)


# ---------------------------------------------------------------------------
# Tests for Mobius transform
# ---------------------------------------------------------------------------


class TestMobiusTransform:
    """Tests for the Mobius transformation mapping."""

    def test_preserves_sphere(self):
        """Mobius transform maps S^{d-1} to S^{d-1}."""
        d = 16
        batch = 100

        mu = F.normalize(torch.randn(batch, d), dim=-1)
        rho = torch.rand(batch, 1) * 0.95  # [0, 0.95)
        u = F.normalize(torch.randn(batch, d), dim=-1)

        z = mobius_transform(u, mu, rho)
        norms = z.norm(dim=-1)

        # Should be unit norm (after renormalization in mobius_transform)
        assert torch.allclose(norms, torch.ones(batch), atol=1e-5), (
            f"Mobius output norms: min={norms.min():.6f}, max={norms.max():.6f}"
        )

    def test_identity_at_rho_zero(self):
        """When rho=0, Mobius transform is identity (up to sign/reflection)."""
        d = 16
        batch = 50

        mu = F.normalize(torch.randn(batch, d), dim=-1)
        rho = torch.zeros(batch, 1)
        u = F.normalize(torch.randn(batch, d), dim=-1)

        z = mobius_transform(u, mu, rho)

        # With rho=0: T_0(u) = 0 + (1-0)*(u-0)/|u-0|^2 = u/|u|^2 = u (since |u|=1)
        assert torch.allclose(z, u, atol=1e-4), (
            f"Max diff from identity: {(z - u).abs().max():.6f}"
        )

    def test_high_rho_concentrates(self):
        """With high rho, samples concentrate near mu."""
        d = 32
        n_samples = 1000

        mu = F.normalize(torch.randn(1, d), dim=-1).expand(n_samples, -1)
        rho = torch.full((n_samples, 1), 0.99)
        u = F.normalize(torch.randn(n_samples, d), dim=-1)

        z = mobius_transform(u, mu, rho)

        # Cosine similarity to mu should be high
        cos_sim = (z * mu).sum(dim=-1)
        mean_cos = cos_sim.mean()

        assert mean_cos > 0.8, (
            f"Mean cosine similarity to mu at rho=0.99: {mean_cos:.4f} (expected > 0.8)"
        )

    def test_preserves_sphere_high_dim(self):
        """Test sphere preservation in 256D (actual VAE latent dim)."""
        d = 256
        batch = 50

        mu = F.normalize(torch.randn(batch, d), dim=-1)
        rho = torch.rand(batch, 1) * 0.95
        u = F.normalize(torch.randn(batch, d), dim=-1)

        z = mobius_transform(u, mu, rho)
        norms = z.norm(dim=-1)

        assert torch.allclose(norms, torch.ones(batch), atol=1e-4)


# ---------------------------------------------------------------------------
# Tests for spCauchy sampling
# ---------------------------------------------------------------------------


class TestSpCauchySampling:
    """Tests for the spCauchy sampling function."""

    def test_output_shape(self):
        """Sample shape is correct."""
        d = 16
        batch = 8

        mu = F.normalize(torch.randn(batch, d), dim=-1)
        rho = torch.rand(batch, 1) * 0.9

        # Single sample
        z1 = sample_spcauchy(mu, rho, n_samples=1)
        assert z1.shape == (batch, d), f"Expected ({batch}, {d}), got {z1.shape}"

        # Multiple samples
        z_multi = sample_spcauchy(mu, rho, n_samples=5)
        assert z_multi.shape == (5, batch, d), f"Expected (5, {batch}, {d}), got {z_multi.shape}"

    def test_samples_on_sphere(self):
        """All samples lie on S^{d-1}."""
        d = 32
        batch = 20

        mu = F.normalize(torch.randn(batch, d), dim=-1)
        rho = torch.rand(batch, 1) * 0.95

        z = sample_spcauchy(mu, rho, n_samples=1)
        norms = z.norm(dim=-1)

        assert torch.allclose(norms, torch.ones(batch), atol=1e-4)

    def test_uniform_at_rho_zero(self):
        """At rho=0, samples should be approximately uniform on S^{d-1}."""
        d = 8
        n_samples = 5000

        mu = F.normalize(torch.randn(1, d), dim=-1).expand(n_samples, -1)
        rho = torch.zeros(n_samples, 1)

        z = sample_spcauchy(mu, rho, n_samples=1)

        # For uniform samples on S^{d-1}, E[z_i] = 0 and Var(z_i) = 1/d
        mean = z.mean(dim=0)
        assert mean.abs().max() < 0.1, (
            f"Sample mean at rho=0 should be near zero, got max |mean|={mean.abs().max():.4f}"
        )

    def test_reparameterization_gradient(self):
        """Gradients flow through the sample (reparameterization trick)."""
        d = 16
        batch = 4

        mu = F.normalize(torch.randn(batch, d), dim=-1)
        mu.requires_grad_(True)
        rho = torch.rand(batch, 1) * 0.9
        rho.requires_grad_(True)

        z = sample_spcauchy(mu, rho, n_samples=1)
        loss = z.sum()
        loss.backward()

        assert mu.grad is not None, "No gradient for mu"
        assert rho.grad is not None, "No gradient for rho"
        assert not torch.all(mu.grad == 0), "Gradient for mu is all zeros"
        assert not torch.all(rho.grad == 0), "Gradient for rho is all zeros"


# ---------------------------------------------------------------------------
# Tests for KL divergence
# ---------------------------------------------------------------------------


class TestKLDivergence:
    """Tests for KL divergence computation."""

    def test_analytic_kl_zero_at_rho_zero(self):
        """KL_proxy should be 0 when rho=0 (uniform distribution)."""
        rho = torch.zeros(10, 1)
        kl = spcauchy_kl_analytic(rho, d=256)
        assert kl.item() < 1e-6, f"KL at rho=0: {kl.item():.6f} (expected ~0)"

    def test_analytic_kl_increases_with_rho(self):
        """KL should increase as rho increases (more concentrated)."""
        d = 256
        kl_values = []
        for rho_val in [0.0, 0.3, 0.6, 0.9, 0.99]:
            rho = torch.full((1, 1), rho_val)
            kl = spcauchy_kl_analytic(rho, d=d)
            kl_values.append(kl.item())

        for i in range(len(kl_values) - 1):
            assert kl_values[i] < kl_values[i + 1], (
                f"KL not increasing: rho={[0.0, 0.3, 0.6, 0.9, 0.99][i]:.2f} "
                f"-> KL={kl_values[i]:.4f} >= {kl_values[i+1]:.4f}"
            )

    def test_mc_kl_near_zero_at_rho_zero(self):
        """MC KL estimate should be near 0 when rho is near 0.

        At rho=0, spCauchy IS uniform, so KL to uniform should be 0.
        The normalization offset (log C_d + log A_d) is subtracted in the
        estimator so that KL(rho=0) = 0. MC variance is low near rho=0
        because samples are uniform and ||z - 0*mu|| = ||z|| = 1.
        """
        d = 16
        batch = 500

        mu = F.normalize(torch.randn(batch, d), dim=-1)
        rho = torch.full((batch, 1), 0.001)

        kl = spcauchy_kl_divergence(mu, rho, d)
        # Near rho=0, MC estimate has low variance and should be close to 0
        assert abs(kl.item()) < 1.0, f"MC KL at rho~0: {kl.item():.4f} (expected near 0)"

    def test_mc_kl_large_at_high_rho(self):
        """MC KL should be large and positive at high rho (concentrated).

        At high rho, spCauchy concentrates near mu, so KL to uniform
        should be large. We test rho=0.9 where the signal dominates MC noise.
        """
        d = 16
        batch = 500

        mu = F.normalize(torch.randn(batch, d), dim=-1)
        rho = torch.full((batch, 1), 0.9)

        kl = spcauchy_kl_divergence(mu, rho, d)
        assert kl.item() > 5.0, f"MC KL at rho=0.9: {kl.item():.4f} (expected >> 0)"

    def test_mc_kl_positive_at_high_rho(self):
        """MC KL should be positive at high rho."""
        d = 16
        batch = 100

        mu = F.normalize(torch.randn(batch, d), dim=-1)
        rho = torch.full((batch, 1), 0.9)

        kl = spcauchy_kl_divergence(mu, rho, d)
        assert kl.item() > 0, f"MC KL at rho=0.9: {kl.item():.4f} (expected > 0)"


# ---------------------------------------------------------------------------
# Tests for normalizing constant
# ---------------------------------------------------------------------------


class TestNormalizingConstant:
    """Tests for the spCauchy normalizing constant."""

    def test_log_C_d_values(self):
        """Verify log C_d = log(Gamma(d)/(2*pi^d)) for known dimensions."""
        # d=2: C_2 = Gamma(2)/(2*pi^2) = 1/(2*pi^2)
        log_c2 = _log_C_d(2)
        expected = math.log(1.0) - math.log(2.0) - 2 * math.log(math.pi)
        assert abs(log_c2 - expected) < 1e-10, (
            f"log C_2: got {log_c2:.6f}, expected {expected:.6f}"
        )

        # d=3: C_3 = Gamma(3)/(2*pi^3) = 2/(2*pi^3) = 1/pi^3
        log_c3 = _log_C_d(3)
        expected = math.log(2.0) - math.log(2.0) - 3 * math.log(math.pi)
        assert abs(log_c3 - expected) < 1e-10

    def test_surface_area_sphere(self):
        """Verify surface area of S^{d-1} for known dimensions."""
        # S^1 (circle): area = 2*pi
        log_a2 = _log_surface_area_sphere(2)
        assert abs(log_a2 - math.log(2 * math.pi)) < 1e-10

        # S^2 (sphere): area = 4*pi
        log_a3 = _log_surface_area_sphere(3)
        assert abs(log_a3 - math.log(4 * math.pi)) < 1e-10


# ---------------------------------------------------------------------------
# Tests for the full model
# ---------------------------------------------------------------------------


class TestSpCauchyVAE:
    """Tests for SpCauchyTransformerVAE model."""

    @pytest.fixture
    def model(self):
        """Small model for testing."""
        return SpCauchyTransformerVAE(
            vocab_size=20,
            bottleneck_size=2,
            d_model=16,
            kl_factor=0.1,
            encoder_nhead=2,
            encoder_dim_feedforward=32,
            encoder_num_layers=1,
            decoder_nhead=2,
            decoder_dim_feedforward=32,
            decoder_num_layers=1,
        )

    def test_encode_shapes(self, model):
        """Encoder outputs have correct shapes and constraints."""
        batch = 4
        seq_len = 10
        tokens = torch.randint(0, 20, (batch, seq_len))
        # Ensure at least one stop token (idx=1) per sequence
        tokens[:, -1] = 1

        mu, rho = model.encode(tokens)

        assert mu.shape == (batch, model.latent_dim), (
            f"mu shape: {mu.shape}, expected ({batch}, {model.latent_dim})"
        )
        assert rho.shape == (batch, 1), f"rho shape: {rho.shape}, expected ({batch}, 1)"

        # mu should be unit norm
        mu_norms = mu.norm(dim=-1)
        assert torch.allclose(mu_norms, torch.ones(batch), atol=1e-5), (
            f"mu norms: {mu_norms}"
        )

        # rho should be in [0, 0.99)
        assert (rho >= 0).all(), f"rho has negative values: min={rho.min():.4f}"
        assert (rho < 0.99).all(), f"rho >= 0.99: max={rho.max():.4f}"

    def test_forward_returns_valid_loss(self, model):
        """Forward pass produces finite loss and valid metrics."""
        batch = 4
        seq_len = 10
        tokens = torch.randint(0, 20, (batch, seq_len))
        tokens[:, -1] = 1

        result = model(tokens)

        assert torch.isfinite(result["loss"]), f"Loss is not finite: {result['loss']}"
        assert torch.isfinite(result["recon_loss"]), f"Recon loss not finite"
        assert torch.isfinite(result["kldiv"]), f"KL div not finite"
        assert 0 <= result["recon_token_acc"] <= 1, "Token acc out of range"
        assert 0 <= result["rho_mean"] < 1, "rho_mean out of range"

    def test_latent_on_sphere(self, model):
        """Latent z from forward pass should be on S^{d-1}."""
        batch = 4
        seq_len = 10
        tokens = torch.randint(0, 20, (batch, seq_len))
        tokens[:, -1] = 1

        result = model(tokens)
        z_flat = result["z_flat"]  # [batch, latent_dim]
        norms = z_flat.norm(dim=-1)

        assert torch.allclose(norms, torch.ones(batch), atol=1e-4), (
            f"z norms: {norms}"
        )

    def test_prior_samples_on_sphere(self, model):
        """Prior samples should be on S^{d-1}."""
        z = model.sample_prior(10)
        # z shape: [10, bottleneck_size, d_model]
        z_flat = z.reshape(10, -1)
        norms = z_flat.norm(dim=-1)

        assert torch.allclose(norms, torch.ones(10), atol=1e-5)

    def test_backward_pass(self, model):
        """Loss is differentiable and produces gradients."""
        batch = 4
        seq_len = 10
        tokens = torch.randint(0, 20, (batch, seq_len))
        tokens[:, -1] = 1

        result = model(tokens)
        result["loss"].backward()

        # Check that gradients exist for key parameters
        has_grads = sum(
            1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0
        )
        total_params = sum(1 for _ in model.parameters())
        assert has_grads > 0, "No parameters have gradients"

    def test_sample_generates_tokens(self, model):
        """Autoregressive sampling produces valid token sequences."""
        model.eval()
        tokens = model.sample(n=2)

        assert tokens.dim() == 2, f"Expected 2D tensor, got {tokens.dim()}D"
        assert tokens.shape[0] == 2, f"Expected batch=2, got {tokens.shape[0]}"
        assert tokens.shape[1] > 1, "Generated sequence too short"

    def test_init_from_gaussian_vae(self):
        """Transfer learning from Gaussian VAE works."""
        # Create a fake Gaussian VAE state dict with matching keys
        from src.codec.vae_model import InfoTransformerVAE

        gaussian_model = InfoTransformerVAE(
            vocab_size=20,
            bottleneck_size=2,
            d_model=16,
        )
        gaussian_state = gaussian_model.state_dict()

        # Create spCauchy model and transfer weights
        spcauchy_model = SpCauchyTransformerVAE(
            vocab_size=20,
            bottleneck_size=2,
            d_model=16,
        )
        spcauchy_model.init_from_gaussian_vae(gaussian_state)

        # Verify decoder weights were transferred
        for name in ["decoder_token_embedding.weight", "decoder_token_unembedding"]:
            if name in gaussian_state:
                assert torch.allclose(
                    spcauchy_model.state_dict()[name],
                    gaussian_state[name],
                ), f"Weight {name} not transferred correctly"
