"""von Mises-Fisher VAE — third hyperspherical-VAE baseline.

Identical architecture and training to PowerSphericalTransformerVAE; only
the posterior family changes from Power Spherical to von Mises-Fisher.
This is the *original* Davidson et al. 2018 hyperspherical-VAE construction
(arXiv:1804.00891), which we include as a third explicit-sphere-prior point
in the codec-geometry ablation.

The vMF posterior is reparameterized via Wood's 1994 rejection sampling +
Householder reflection (Davidson 2018, Algorithm 1). The implementation is
the official ``hyperspherical_vae`` package (Davidson, github:nicola-decao/
s-vae-pytorch). Davidson noted numerical instabilities for D > 64; we test
empirically at D = 256 and report whether training converges.

Refs:
- Davidson et al. 2018, "Hyperspherical Variational Auto-Encoders".
- Wood 1994, "Simulation of the von Mises Fisher distribution".
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from hyperspherical_vae.distributions import HypersphericalUniform, VonMisesFisher

from src.codec.spcauchy_vae import SpCauchyTransformerVAE


class VMFTransformerVAE(SpCauchyTransformerVAE):
    """von Mises-Fisher VAE; closed-form KL, rejection-sample reparameterization.

    Encoder outputs ``(mu, kappa)`` where ``mu in S^{D-1}`` and
    ``kappa >= 0`` is the concentration parameter. We re-purpose the spCauchy
    encoder's ``rho_head`` for log-kappa, identical to PowerSphericalVAE.

    Training caveats. Davidson 2018 reports numerical issues for D > 64; the
    Householder-reflection reparameterization can produce non-finite
    gradients when kappa is very large or very small. We clamp ``kappa`` to
    ``[1, kappa_max]`` (kappa = 0 collapses to uniform; kappa > 50000 hits
    Bessel-ratio overflow); and add a NaN guard in ``forward``.

    Args:
        kappa_max: Upper clamp for concentration. Default 5000 (lower than PS-VAE
            default of 10000 — vMF rejection sampling becomes unstable at very
            high kappa in high D, per Davidson 2018).
    """

    def __init__(
        self,
        *args,
        kappa_max: float = 5000.0,
        kappa_min: float = 100.0,
        **kwargs,
    ):
        # NOTE on kappa_min: in D=256 the vMF concentration scales as
        # E[cos(sample, mu)] ≈ kappa / D for small kappa. With kappa_min=1 (the
        # naive choice from Davidson 2018) the posterior is essentially
        # uniform-on-sphere and the encoder transmits no information. We
        # default to kappa_min=100 (≈ 0.4 mean cos) to keep the bottleneck
        # informative; the model can still raise kappa toward kappa_max=5000
        # as training progresses.
        super().__init__(*args, **kwargs)
        self.kappa_max = float(kappa_max)
        self.kappa_min = float(kappa_min)

    def encode(self, tokens, as_probs: bool = False):
        """Encode tokens to (mu in S^{D-1}, kappa >= kappa_min)."""
        if as_probs:
            embed = tokens @ self.encoder_token_embedding.weight
        else:
            embed = self.encoder_token_embedding(tokens)
        embed = self.encoder_position_encoding(embed)

        pad_mask = self.generate_pad_mask(tokens)
        encoding = self.encoder(embed, src_key_padding_mask=pad_mask)
        encoding = encoding[:, : self.bottleneck_size, :]
        mu_raw = encoding[..., : self.d_model]
        rho_input = encoding[..., self.d_model :]

        mu = mu_raw.reshape(-1, self.latent_dim)
        mu = F.normalize(mu, dim=-1)

        rho_features = rho_input.reshape(-1, self.latent_dim)
        log_kappa_raw = self.rho_head(rho_features)               # [B, 1]
        # Sigmoid scaling into [kappa_min, kappa_max] (Davidson 2018 alternate
        # parameterization, avoids the kappa-stuck-at-floor failure mode that
        # softplus + clamp produces from a Gaussian-initialized rho_head). At
        # init the encoder outputs log_kappa_raw ~ 0 → kappa ~ midpoint, far
        # from the floor, so the encoder has gradient signal to learn either
        # direction.
        kappa_norm = torch.sigmoid(log_kappa_raw)                  # (0, 1)
        kappa = self.kappa_min + kappa_norm * (self.kappa_max - self.kappa_min)
        return mu, kappa

    def sample_posterior(self, mu, kappa):
        """Rejection-sample reparameterized z ~ vMF(mu, kappa)."""
        vmf = VonMisesFisher(loc=mu, scale=kappa)
        return vmf.rsample()

    def forward(self, tokens):
        """Standard VAE ELBO with vMF posterior + uniform-on-sphere prior."""
        mu, kappa = self.encode(tokens)
        z_flat = self.sample_posterior(mu, kappa)
        z = z_flat.reshape(-1, self.bottleneck_size, self.d_model)
        logits = self.decode(z, tokens)

        recon_loss = F.cross_entropy(
            logits.permute(0, 2, 1), tokens, reduction="none",
        ).mean()

        # Closed-form KL(vMF(mu, kappa) || HypersphericalUniform(D)) per sample.
        vmf = VonMisesFisher(loc=mu, scale=kappa)
        prior = HypersphericalUniform(dim=self.latent_dim - 1, device=mu.device)
        kldiv_per_sample = torch.distributions.kl_divergence(vmf, prior)
        if kldiv_per_sample.ndim == 0:
            kldiv_per_sample = kldiv_per_sample.expand(mu.shape[0])
        if kldiv_per_sample.ndim == 2:
            kldiv_per_sample = kldiv_per_sample.squeeze(-1)
        # NaN guard: vMF Bessel-ratio occasionally produces NaN at extreme kappa
        # in high D (Davidson 2018). Replace NaN entries with the batch median
        # so a single bad sample doesn't poison the whole loss.
        if not torch.isfinite(kldiv_per_sample).all():
            finite = kldiv_per_sample[torch.isfinite(kldiv_per_sample)]
            replacement = finite.median() if len(finite) else torch.zeros((), device=mu.device)
            kldiv_per_sample = torch.where(
                torch.isfinite(kldiv_per_sample),
                kldiv_per_sample,
                replacement,
            )
        kldiv = kldiv_per_sample.mean()
        if self.free_bits > 0:
            kldiv_loss_term = kldiv_per_sample.clamp(min=self.free_bits).mean()
        else:
            kldiv_loss_term = kldiv

        primary_loss = recon_loss
        if self.kl_factor != 0:
            primary_loss = primary_loss + self.kl_factor * kldiv_loss_term

        return dict(
            loss=primary_loss,
            z=z, z_flat=z_flat,
            recon_loss=recon_loss,
            kldiv=kldiv,
            kldiv_loss_term=kldiv_loss_term,
            recon_token_acc=(logits.argmax(dim=-1) == tokens).float().mean(),
            recon_string_acc=(logits.argmax(dim=-1) == tokens).all(dim=1).float().mean(dim=0),
            rho_mean=kappa.mean(),
            mu_norm=mu.norm(dim=-1).mean(),
        )
