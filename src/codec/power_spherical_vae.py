"""Power Spherical VAE.

Drop-in replacement for SpCauchyTransformerVAE that uses the Power Spherical
distribution (De Cao & Aziz 2020, arXiv:2006.04437) as the posterior on
S^{D-1}. Power Spherical was designed precisely to address the two failure
modes of vMF VAEs (Davidson 2018) at high D:
1. Rejection-sampling reparameterization is unstable for D > 64.
2. Gradients through the sample acceptance step have high variance.

Power Spherical instead has a closed-form sampling procedure (no rejection),
closed-form KL to the uniform-on-sphere prior, and is verified stable up to
D >> 256. We reuse the SpCauchyTransformerVAE encoder/decoder architecture
unchanged; only the posterior parameterization and KL term are swapped.

This codec is the second hyperspherical-VAE point in our codec-geometry
ablation, complementing SpCauchyTransformerVAE. Both are explicit-spherical-
prior models; if both fail in benchmark, that is robust evidence against the
recipe rule "explicit sphere prior provides usable BO geometry".

Refs:
- De Cao & Aziz 2020, "The Power Spherical distribution".
- Davidson et al. 2018, "Hyperspherical VAEs". (vMF baseline.)
- Notin et al. 2021, applied Power Spherical to protein latents.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from power_spherical import HypersphericalUniform, PowerSpherical

from src.codec.spcauchy_vae import SpCauchyTransformerVAE


class PowerSphericalTransformerVAE(SpCauchyTransformerVAE):
    """Power Spherical VAE: same arch as SpCauchy but PS posterior + closed-form KL.

    Encoder outputs:
    - mu  : mean direction on S^{D-1}, ||mu|| = 1.
    - kappa : concentration in [0, kappa_max). High kappa = peaked posterior;
              kappa = 0 = uniform.

    The single behavioural change relative to SpCauchyTransformerVAE is in
    ``sample_posterior`` and ``forward`` (KL term). All embeddings, encoder,
    decoder, positional encoding, and rho_head (re-purposed for kappa) are
    inherited unchanged.

    Args:
        kappa_max: Upper clamp for concentration; large values approximate a
            Dirac delta on the sphere. Default 250 (well above empirical
            equilibrium for D=256 even under tight β).
    """

    def __init__(
        self,
        *args,
        kappa_max: float = 10000.0,
        **kwargs,
    ):
        # NOTE on kappa_max default: in D=256 the Power Spherical density
        # ``(1 + mu·x)^kappa`` requires kappa O(D) to keep samples within
        # ~5 deg of the posterior mean. Empirically ``kappa=250`` gives
        # ``E[cos(sample, mu)] ≈ 0.48`` (samples 60 deg from the mode), too
        # noisy for the decoder to reconstruct, while ``kappa=10000`` gives
        # ``E[cos] ≈ 0.975`` and ``kappa=5000`` gives ``≈ 0.95`` — both
        # tight enough for the SELFIES decoder to use the latent. We default
        # to 10000 with the option to lower for more uniform-on-sphere geometry.
        super().__init__(*args, **kwargs)
        self.kappa_max = float(kappa_max)

    # -- Encoder: re-purpose rho_head to emit log-kappa ----------------------

    def encode(self, tokens, as_probs: bool = False):
        """Same as SpCauchy encode but second head is interpreted as log-kappa.

        Returns (mu, kappa) where mu in S^{D-1} and kappa >= 0.
        """
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
        # softplus → kappa >= 0; clamp upper bound for numerical stability of
        # PowerSpherical.rsample (the ratio (alpha-beta)/(alpha+beta) maps to
        # the sphere; very large kappa concentrates near a single point).
        kappa = F.softplus(log_kappa_raw).clamp(max=self.kappa_max)
        return mu, kappa

    # -- Posterior sampling: PowerSpherical.rsample (closed-form) ------------

    def sample_posterior(self, mu, kappa):
        """Sample z ~ PowerSpherical(mu, kappa) using closed-form rsample.

        Power Spherical density: f(x; mu, kappa) ∝ (1 + mu·x)^kappa, so
        kappa = 0 is uniform and kappa → ∞ collapses on mu.

        Returns z: [B, latent_dim] on S^{D-1}.
        """
        # PowerSpherical expects scale as scalar per batch element.
        ps = PowerSpherical(loc=mu, scale=kappa.squeeze(-1))
        return ps.rsample()

    # -- forward: replace KL term with PS-vs-uniform closed form -------------

    def forward(self, tokens):
        """Same as SpCauchy forward but with PS-uniform KL closed form."""
        mu, kappa = self.encode(tokens)
        z_flat = self.sample_posterior(mu, kappa)
        z = z_flat.reshape(-1, self.bottleneck_size, self.d_model)
        logits = self.decode(z, tokens)

        recon_loss = F.cross_entropy(
            logits.permute(0, 2, 1), tokens, reduction="none",
        ).mean()

        # KL(PS(mu, kappa) || HypersphericalUniform(D)) closed form, per-sample.
        ps = PowerSpherical(loc=mu, scale=kappa.squeeze(-1))
        prior = HypersphericalUniform(dim=self.latent_dim, device=mu.device)
        kldiv_per_sample = torch.distributions.kl_divergence(ps, prior)  # [B]
        # Some torch versions return [] (scalar) when batched; handle both.
        if kldiv_per_sample.ndim == 0:
            kldiv_per_sample = kldiv_per_sample.expand(mu.shape[0])
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
            rho_mean=kappa.mean(),  # named for compatibility with training script
            mu_norm=mu.norm(dim=-1).mean(),
        )
