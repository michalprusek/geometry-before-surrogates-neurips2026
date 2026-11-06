"""Gaussian-prior graph VAE over NAS-Bench-101 cells, with a generative decoder.

Why this exists (added during the review period). The paper's claim is
that candidate generation should respect the empirical latent support of the
generative model, and all of its decisive evidence comes from one codec: the
SELFIES molecular VAE. The question raised in review was whether the method
applies beyond the SELFIES VAE latent space.

This module supplies a second codec that shares with SELFIES exactly the two
properties our mechanism says are required -- a Gaussian prior (so the aggregate
posterior concentrates on a shell) and a generative decoder (so decoding degrades
off that shell) -- and shares nothing else:

  SELFIES VAE                          NAS-Bench-101 graph VAE
  -----------                          -----------------------
  discrete strings (SELFIES tokens)    discrete DAGs (7-node cells)
  autoregressive Transformer decoder   feed-forward structured decoder
  RDKit + GuacaMol oracle              exact table lookup over 423,624 cells
  validity = parseable SMILES          validity = the decoded spec exists
  D = 256                              D configurable (default 64)

The oracle is free (a dictionary lookup), so the paired comparisons here can run
at 200 seeds rather than 40.

Representation. Every NAS-Bench-101 cell is stored as a strictly upper-triangular
7x7 adjacency matrix (21 free bits) and a 7x5 one-hot operation matrix, with
smaller cells padded. We model exactly that: 21 Bernoulli edge variables and 7
categorical node variables. Decoding thresholds the edges at 0.5 and takes the
operation argmax, then looks the resulting spec up in the table; a spec that is
not in the table is invalid and scores NaN, exactly as an unparseable SMILES does
in the molecular pipeline.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_NODES = 7
N_OPS = 5
N_EDGES = N_NODES * (N_NODES - 1) // 2  # 21 strictly-upper-triangular entries
FEATURE_DIM = N_EDGES + N_NODES * N_OPS  # 21 + 35 = 56
TRIU = np.triu_indices(N_NODES, k=1)

DEFAULT_DATA = "/data2/prusek/NeurIPS/data/nasbench/data.json"
DEFAULT_WEIGHTS = "/data2/prusek/NeurIPS/weights/nas_vae/nas_vae.pt"


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------


def load_nasbench(path: str = DEFAULT_DATA) -> tuple[np.ndarray, dict, dict]:
    """Return (features [N, 56], spec_key -> record, index -> spec_key).

    ``spec_key`` is the exact discrete spec: the 21 upper-triangular edge bits
    followed by the 7 operation labels. Lookup is exact -- no isomorphism
    canonicalisation -- which is the conservative choice: an isomorphic relabelling
    counts as invalid rather than being silently accepted.
    """
    with open(path) as f:
        raw = json.load(f)

    feats = np.zeros((len(raw), FEATURE_DIM), dtype=np.float32)
    table: dict[tuple, dict] = {}
    keys: dict[int, tuple] = {}
    for i, (_, rec) in enumerate(sorted(raw.items(), key=lambda kv: int(kv[0]))):
        adj = np.asarray(rec["module_adjacency"], dtype=np.int64)
        ops = np.asarray(rec["module_operations"], dtype=np.int64).argmax(axis=1)
        edges = adj[TRIU]
        feats[i, :N_EDGES] = edges
        feats[i, N_EDGES:] = np.eye(N_OPS, dtype=np.float32)[ops].reshape(-1)
        key = (tuple(int(e) for e in edges), tuple(int(o) for o in ops))
        table[key] = rec
        keys[i] = key
    return feats, table, keys


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class NASGraphVAE(nn.Module):
    """Plain Gaussian-prior VAE. Deliberately unremarkable -- the point is that
    the geometry we describe is a property of the ELBO objective, not of any
    architectural choice specific to the molecular codec."""

    def __init__(self, latent_dim: int = 64, hidden: int = 512):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Linear(FEATURE_DIM, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.to_mu = nn.Linear(hidden, latent_dim)
        self.to_logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, FEATURE_DIM),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(x)
        return self.to_mu(h), self.to_logvar(h)

    def decode_logits(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.dec(z)
        return out[:, :N_EDGES], out[:, N_EDGES:].view(-1, N_NODES, N_OPS)

    def forward(self, x: torch.Tensor) -> tuple:
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        edge_logits, op_logits = self.decode_logits(z)
        return edge_logits, op_logits, mu, logvar


def elbo_loss(edge_logits, op_logits, x, mu, logvar, beta: float = 1.0):
    """Standard beta=1 ELBO: Bernoulli edges + categorical ops + KL to N(0, I)."""
    edges = x[:, :N_EDGES]
    ops = x[:, N_EDGES:].view(-1, N_NODES, N_OPS).argmax(dim=-1)
    rec_e = F.binary_cross_entropy_with_logits(edge_logits, edges, reduction="none").sum(1)
    rec_o = F.cross_entropy(
        op_logits.reshape(-1, N_OPS), ops.reshape(-1), reduction="none"
    ).view(-1, N_NODES).sum(1)
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(1)
    return (rec_e + rec_o + beta * kl).mean(), rec_e.mean(), rec_o.mean(), kl.mean()


# ---------------------------------------------------------------------------
# codec
# ---------------------------------------------------------------------------


class NASVAECodec:
    """encode(specs) -> latent means; decode(z) -> discrete specs. Deterministic.

    Mirrors the SELFIESVAECodec contract used by every solver, so the identical
    solver code paths run on this codec without modification.
    """

    def __init__(self, model: NASGraphVAE, table: dict, feats: np.ndarray,
                 device: str = "cuda"):
        self.model = model.to(device).eval()
        self.table = table
        self.feats = feats
        self.device = device

    @property
    def dim(self) -> int:
        return self.model.latent_dim

    @classmethod
    def from_pretrained(cls, weights: str = DEFAULT_WEIGHTS,
                        data: str = DEFAULT_DATA, device: str = "cuda") -> "NASVAECodec":
        ckpt = torch.load(weights, map_location=device)
        model = NASGraphVAE(latent_dim=ckpt["latent_dim"], hidden=ckpt["hidden"])
        model.load_state_dict(ckpt["state_dict"])
        feats, table, _ = load_nasbench(data)
        return cls(model, table, feats, device=device)

    @torch.no_grad()
    def encode(self, idx: np.ndarray) -> torch.Tensor:
        """Encoder MEANS for the given architecture indices (no posterior noise)."""
        x = torch.tensor(self.feats[idx], dtype=torch.float32, device=self.device)
        mu, _ = self.model.encode(x)
        return mu

    @torch.no_grad()
    def decode(self, z: torch.Tensor) -> list[Optional[tuple]]:
        """Latents -> discrete specs. Returns None where the spec is not a real cell."""
        if z.ndim == 1:
            z = z.unsqueeze(0)
        z = z.to(self.device, dtype=torch.float32)
        edge_logits, op_logits = self.model.decode_logits(z)
        edges = (torch.sigmoid(edge_logits) > 0.5).to(torch.int64).cpu().numpy()
        ops = op_logits.argmax(dim=-1).cpu().numpy()
        out: list[Optional[tuple]] = []
        for e, o in zip(edges, ops):
            key = (tuple(int(v) for v in e), tuple(int(v) for v in o))
            out.append(key if key in self.table else None)
        return out
