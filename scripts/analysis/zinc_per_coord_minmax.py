"""Compute per-coordinate min/max + percentile bounds from ZINC-250K encoded latents.

These define the natural [0,1]^D box for original-recipe Hvarfner / Papenmeier /
Xu acquisition. Output saved to results/precomputed/zinc_per_coord_bounds.json.

Stats per coordinate i in {0..255}:
  - min, max  (for strict [zmin, zmax] bounds)
  - p01, p99  (robust bounds; recommended)
  - mean, std (sanity check vs zinc_mean_norm.json)

Usage:
    PYTHONPATH=. python scripts/analysis/zinc_per_coord_minmax.py [--n-mols 250000]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import selfies as sf
import torch

from src.codec.selfies_vae import SELFIESVAECodec

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ZINC_PATH = Path("/path/to/NeurIPS/data/zinc_250k.txt")
OUT_PATH = Path("results/precomputed/zinc_per_coord_bounds.json")
BATCH_SIZE = 256


def smiles_to_token_ids(smiles, codec):
    try:
        selfies_str = sf.encoder(smiles)
    except sf.EncoderError:
        return None
    if selfies_str is None:
        return None
    tokens = list(sf.split_selfies(selfies_str))
    known = [t for t in tokens if t in codec.vocab.vocab2idx]
    if not known:
        return None
    return codec.vocab.encode(known)


def pad_batch(token_lists, pad_idx):
    max_len = max(t.shape[0] for t in token_lists)
    out = torch.full((len(token_lists), max_len), pad_idx, dtype=torch.long)
    for i, t in enumerate(token_lists):
        out[i, : t.shape[0]] = t
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-mols", type=int, default=250000,
                        help="Cap on number of SMILES to encode (default: all)")
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu (default: auto)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    smiles_list = [s.strip() for s in ZINC_PATH.read_text().splitlines() if s.strip()]
    if args.n_mols < len(smiles_list):
        smiles_list = smiles_list[: args.n_mols]
    logger.info("Loaded %d SMILES", len(smiles_list))

    codec = SELFIESVAECodec.from_pretrained(device=device)
    pad_idx = codec.vocab.vocab2idx.get("<stop>", 1)
    D = codec.EMBEDDING_DIM

    # Streaming stats: keep all encoded latents in a single ndarray for percentile.
    Z_all = []
    n_valid = 0
    t0 = time.time()
    n_batches = (len(smiles_list) + BATCH_SIZE - 1) // BATCH_SIZE

    for b_idx in range(n_batches):
        start = b_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(smiles_list))
        batch_smiles = smiles_list[start:end]

        token_lists = []
        for smi in batch_smiles:
            tk = smiles_to_token_ids(smi, codec)
            if tk is not None and tk.shape[0] <= codec.model.max_string_length:
                token_lists.append(tk)
        if not token_lists:
            continue

        padded = pad_batch(token_lists, pad_idx).to(device)
        with torch.no_grad():
            mu, _ = codec.model.encode(padded)
        z = mu.reshape(mu.shape[0], -1).cpu().numpy().astype(np.float32)
        Z_all.append(z)
        n_valid += z.shape[0]

        if (b_idx + 1) % 50 == 0 or b_idx == n_batches - 1:
            elapsed = time.time() - t0
            rate = ((b_idx + 1) * BATCH_SIZE) / max(elapsed, 1e-9)
            logger.info(
                "Batch %d/%d  valid=%d  rate=%.1f mol/s  elapsed=%.1fs",
                b_idx + 1, n_batches, n_valid, rate, elapsed,
            )

    Z = np.concatenate(Z_all, axis=0)  # [N, D]
    logger.info("Encoded shape: %s", Z.shape)

    coord_min = Z.min(axis=0)
    coord_max = Z.max(axis=0)
    coord_p01 = np.percentile(Z, 1.0, axis=0)
    coord_p99 = np.percentile(Z, 99.0, axis=0)
    coord_mean = Z.mean(axis=0)
    coord_std = Z.std(axis=0)

    summary = {
        "dataset": str(ZINC_PATH),
        "n_smiles": int(len(smiles_list)),
        "n_encoded": int(n_valid),
        "latent_dim": int(D),
        # Global aggregates (across all coordinates) for quick reporting
        "global": {
            "min_min": float(coord_min.min()),
            "min_max": float(coord_min.max()),
            "max_min": float(coord_max.min()),
            "max_max": float(coord_max.max()),
            "p01_min": float(coord_p01.min()),
            "p99_max": float(coord_p99.max()),
        },
        # Per-coordinate arrays — use these for [0,1]^D normalization
        "per_coord": {
            "min": coord_min.tolist(),
            "max": coord_max.tolist(),
            "p01": coord_p01.tolist(),
            "p99": coord_p99.tolist(),
            "mean": coord_mean.tolist(),
            "std": coord_std.tolist(),
        },
        "elapsed_seconds": time.time() - t0,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f)

    logger.info("Saved %s (%.1f KB)", OUT_PATH, OUT_PATH.stat().st_size / 1024)
    g = summary["global"]
    logger.info(
        "Global: min ∈ [%.3f, %.3f], max ∈ [%.3f, %.3f]; p01 from %.3f, p99 to %.3f",
        g["min_min"], g["min_max"], g["max_min"], g["max_max"],
        g["p01_min"], g["p99_max"],
    )


if __name__ == "__main__":
    main()
