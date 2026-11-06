"""SELFIES Transformer VAE architecture.

VAE architecture from LOL-BO (Maus et al., 2022; MIT License), inlined for reproducibility.
Original source: lolbo/utils/mol_utils/selfies_vae/model_positional_unbounded.py

Architecture:
- Encoder: 6-layer Transformer (d_model=256, 8 heads, ff=512)
  Outputs mu/sigma via channel split of 2*d_model encoding.
- Decoder: 6-layer Transformer decoder (d_model=128, 8 heads, ff=256)
  Autoregressive with Gumbel-softmax sampling.
- Bottleneck: 2 tokens x 128 dims = 256D latent space.
"""

from math import log

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def gumbel_softmax(
    logits: Tensor,
    tau: float = 1,
    hard: bool = False,
    dim: int = -1,
    return_randoms: bool = False,
    randoms: Tensor = None,
) -> Tensor:
    """Gumbel-softmax sampling with optional straight-through estimator.

    Mostly from https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax
    """
    if randoms is None:
        randoms = (
            -torch.empty_like(
                logits, memory_format=torch.legacy_contiguous_format
            )
            .exponential_()
            .log()
        )  # ~Gumbel(0,1)
    gumbels = (logits + randoms) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft

    if return_randoms:
        return ret, randoms
    else:
        return ret


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5_000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:, : x.shape[1], :]
        return self.dropout(x)


class InfoTransformerVAE(nn.Module):
    """Transformer VAE for SELFIES molecular strings.

    Args:
        vocab_size: Number of tokens in the SELFIES vocabulary.
        bottleneck_size: Number of latent bottleneck tokens (default: 2).
        d_model: Decoder model dimension; encoder uses 2*d_model (default: 128).
        is_autoencoder: If True, use mu directly (no sampling). Default: False.
        kl_factor: KL divergence weight. Default: 0.1.
        min_posterior_std: Minimum posterior sigma. Default: 1e-4.
        encoder_nhead: Encoder attention heads. Default: 8.
        encoder_dim_feedforward: Encoder FFN dimension. Default: 512.
        encoder_dropout: Encoder dropout. Default: 0.1.
        encoder_num_layers: Encoder layers. Default: 6.
        decoder_nhead: Decoder attention heads. Default: 8.
        decoder_dim_feedforward: Decoder FFN dimension. Default: 256.
        decoder_dropout: Decoder dropout. Default: 0.1.
        decoder_num_layers: Decoder layers. Default: 6.
        max_string_length: Max generation length. Default: 256.
    """

    def __init__(
        self,
        vocab_size: int,
        bottleneck_size: int = 2,
        d_model: int = 128,
        is_autoencoder: bool = False,
        kl_factor: float = 0.1,
        min_posterior_std: float = 1e-4,
        encoder_nhead: int = 8,
        encoder_dim_feedforward: int = 512,
        encoder_dropout: float = 0.1,
        encoder_num_layers: int = 6,
        decoder_nhead: int = 8,
        decoder_dim_feedforward: int = 256,
        decoder_dropout: float = 0.1,
        decoder_num_layers: int = 6,
        max_string_length: int = 256,
    ):
        super().__init__()

        assert bottleneck_size is not None, (
            "bottleneck_size must not be None — unbounded sequences not supported"
        )

        self.max_string_length = max_string_length
        self.vocab_size = vocab_size
        self.bottleneck_size = bottleneck_size
        self.d_model = d_model
        self.is_autoencoder = is_autoencoder
        self.kl_factor = kl_factor
        self.min_posterior_std = min_posterior_std

        encoder_embedding_dim = 2 * d_model

        self.encoder_token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=encoder_embedding_dim
        )
        self.encoder_position_encoding = PositionalEncoding(
            encoder_embedding_dim, dropout=encoder_dropout, max_len=5_000
        )
        self.decoder_token_embedding = nn.Embedding(
            num_embeddings=self.vocab_size, embedding_dim=d_model
        )
        self.decoder_position_encoding = PositionalEncoding(
            d_model, dropout=decoder_dropout, max_len=5_000
        )
        self.decoder_token_unembedding = nn.Parameter(
            torch.randn(d_model, self.vocab_size)
        )
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=encoder_embedding_dim,
                nhead=encoder_nhead,
                dim_feedforward=encoder_dim_feedforward,
                dropout=encoder_dropout,
                activation="relu",
                batch_first=True,
            ),
            num_layers=encoder_num_layers,
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=decoder_nhead,
                dim_feedforward=decoder_dim_feedforward,
                dropout=decoder_dropout,
                activation="relu",
                batch_first=True,
            ),
            num_layers=decoder_num_layers,
        )

    def sample_prior(self, n: int) -> Tensor:
        sequence_length = self.bottleneck_size
        return torch.randn(n, sequence_length, self.d_model).to(
            next(self.parameters()).device
        )

    def sample_posterior(self, mu: Tensor, sigma: Tensor, n: int = None) -> Tensor:
        if n is not None:
            mu = mu.unsqueeze(0).expand(n, -1, -1, -1)
        return mu + torch.randn_like(mu) * sigma

    def generate_pad_mask(self, tokens: Tensor) -> Tensor:
        """Generate mask that tells encoder to ignore all but first stop token."""
        mask = tokens == 1
        inds = mask.float().argmax(dim=-1)
        mask[torch.arange(0, tokens.shape[0]), inds] = False
        return mask

    def encode(self, tokens: Tensor, as_probs: bool = False):
        if as_probs:
            embed = tokens @ self.encoder_token_embedding.weight
        else:
            embed = self.encoder_token_embedding(tokens)

        embed = self.encoder_position_encoding(embed)

        pad_mask = self.generate_pad_mask(tokens)
        encoding = self.encoder(embed, src_key_padding_mask=pad_mask)
        mu = encoding[..., : self.d_model]
        sigma = F.softplus(encoding[..., self.d_model :]) + self.min_posterior_std

        if self.bottleneck_size is not None:
            mu = mu[:, : self.bottleneck_size, :]
            sigma = sigma[:, : self.bottleneck_size, :]

        return mu, sigma

    def decode(self, z: Tensor, tokens: Tensor, as_probs: bool = False) -> Tensor:
        if as_probs:
            embed = tokens[:, :-1] @ self.decoder_token_embedding.weight
        else:
            embed = self.decoder_token_embedding(tokens[:, :-1])

        device = next(self.parameters()).device
        embed = torch.cat(
            [
                torch.zeros(embed.shape[0], 1, embed.shape[-1], device=device),
                embed,
            ],
            dim=1,
        )
        embed = self.decoder_position_encoding(embed)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            embed.shape[1]
        ).to(device)
        decoding = self.decoder(tgt=embed, memory=z, tgt_mask=tgt_mask)
        logits = decoding @ self.decoder_token_unembedding

        return logits

    @torch.no_grad()
    def sample(
        self,
        n: int = -1,
        z: Tensor = None,
        differentiable: bool = False,
        return_logits: bool = False,
        deterministic: bool = False,
    ):
        model_state = self.training
        self.eval()
        device = next(self.parameters()).device

        if z is None:
            z = self.sample_prior(n)
        else:
            n = z.shape[0]

        tokens = torch.zeros(n, 1, device=device).long()
        random_gumbels = torch.zeros(n, 0, self.vocab_size, device=device)
        while True:
            tgt = self.decoder_token_embedding(tokens)
            tgt = self.decoder_position_encoding(tgt)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                tokens.shape[-1]
            ).to(device)

            decoding = self.decoder(tgt=tgt, memory=z, tgt_mask=tgt_mask)
            logits = decoding @ self.decoder_token_unembedding
            # Deterministic: zero Gumbel noise → pure argmax decoding
            forced_randoms = torch.zeros_like(logits) if deterministic else None
            sample, randoms = gumbel_softmax(
                logits, dim=-1, hard=True, return_randoms=True,
                randoms=forced_randoms,
            )

            tokens = torch.cat(
                [tokens, sample[:, -1, :].argmax(dim=-1)[:, None]], dim=-1
            )
            random_gumbels = torch.cat([random_gumbels, randoms], dim=1)

            # 1 is the stop token. Check if all molecules have a stop token.
            if (
                torch.all((tokens == 1).sum(dim=-1) > 0).item()
                or tokens.shape[-1] > self.max_string_length
            ):
                break

        self.train(model_state)

        if not differentiable:
            sample = tokens

        if return_logits:
            return sample, logits
        else:
            return sample

    def forward(self, tokens: Tensor) -> dict:
        mu, sigma = self.encode(tokens)

        if self.is_autoencoder:
            z = mu
        else:
            z = self.sample_posterior(mu, sigma)

        logits = self.decode(z, tokens)

        recon_loss = F.cross_entropy(
            logits.permute(0, 2, 1), tokens, reduction="none"
        ).mean()

        sigma2 = sigma.pow(2)
        kldiv = 0.5 * (mu.pow(2) + sigma2 - sigma2.log() - 1).mean()

        primary_loss = recon_loss
        if self.kl_factor != 0:
            primary_loss = primary_loss + self.kl_factor * kldiv

        return dict(
            loss=primary_loss,
            z=z,
            recon_loss=recon_loss,
            kldiv=kldiv,
            recon_token_acc=(logits.argmax(dim=-1) == tokens).float().mean(),
            recon_string_acc=(logits.argmax(dim=-1) == tokens)
            .all(dim=1)
            .float()
            .mean(dim=0),
            sigma_mean=sigma.mean(),
        )
