"""Encode ZINC-250K via SELFIES VAE and compute per-molecule and per-coordinate
latent statistics. The output ``per_coord_std_mean`` field is the source of
``src.analysis.config.ZINC_PER_COORD_STD``, which feeds the box bounds used by
the Hvarfner/Papenmeier/Xu *BoxSolver acquisition variants.

Outputs:
  - Per-molecule ``\u2016z\u2016_2`` statistics (mean, std, median, percentiles)
  - Per-coordinate std (to confirm spherical-shell concentration)
  - Saves results to ``results/precomputed/zinc_mean_norm.json``

Usage:
    PYTHONPATH=. python scripts/analysis/zinc_mean_norm.py
"""

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
OUT_PATH = Path("results/precomputed/zinc_mean_norm.json")
BATCH_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def smiles_to_token_ids(smiles: str, codec: SELFIESVAECodec) -> torch.Tensor | None:
    """SMILES \u2192 SELFIES \u2192 token ids tensor (1D). None on a legitimate
    encoder failure (invalid SMILES) or vocab miss; other exceptions
    (MemoryError, KeyboardInterrupt, ImportError) propagate.
    """
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


def pad_batch(token_lists: list[torch.Tensor], pad_idx: int) -> torch.Tensor:
    """Pad a list of variable-length token tensors to a single [B, L] tensor."""
    max_len = max(t.shape[0] for t in token_lists)
    out = torch.full((len(token_lists), max_len), pad_idx, dtype=torch.long)
    for i, t in enumerate(token_lists):
        out[i, : t.shape[0]] = t
    return out


def main():
    smiles_list = [s.strip() for s in ZINC_PATH.read_text().splitlines() if s.strip()]
    logger.info("Loaded %d SMILES from %s", len(smiles_list), ZINC_PATH)

    logger.info("Loading SELFIES VAE codec on %s ...", DEVICE)
    codec = SELFIESVAECodec.from_pretrained(device=DEVICE)
    pad_idx = codec.vocab.vocab2idx.get("<stop>", 1)

    norms = np.empty(len(smiles_list), dtype=np.float64)
    norms.fill(np.nan)
    sum_z = np.zeros(codec.EMBEDDING_DIM, dtype=np.float64)
    sum_z_sq = np.zeros(codec.EMBEDDING_DIM, dtype=np.float64)
    n_valid = 0

    t0 = time.time()
    n_batches = (len(smiles_list) + BATCH_SIZE - 1) // BATCH_SIZE

    for b_idx in range(n_batches):
        start = b_idx * BATCH_SIZE
        end = min(start + BATCH_SIZE, len(smiles_list))
        batch_smiles = smiles_list[start:end]

        token_lists = []
        valid_indices = []
        for local_i, smi in enumerate(batch_smiles):
            tk = smiles_to_token_ids(smi, codec)
            if tk is not None and tk.shape[0] <= codec.model.max_string_length:
                token_lists.append(tk)
                valid_indices.append(local_i)

        if not token_lists:
            continue

        padded = pad_batch(token_lists, pad_idx).to(DEVICE)
        with torch.no_grad():
            mu, _ = codec.model.encode(padded)
        z = mu.reshape(mu.shape[0], -1).cpu().numpy().astype(np.float64)

        z_norms = np.linalg.norm(z, axis=1)
        for k, local_i in enumerate(valid_indices):
            norms[start + local_i] = z_norms[k]
        sum_z += z.sum(axis=0)
        sum_z_sq += (z ** 2).sum(axis=0)
        n_valid += z.shape[0]

        if (b_idx + 1) % 50 == 0 or b_idx == n_batches - 1:
            elapsed = time.time() - t0
            rate = ((b_idx + 1) * BATCH_SIZE) / max(elapsed, 1e-9)
            logger.info(
                "Batch %d/%d  valid=%d  rate=%.1f mol/s  elapsed=%.1fs",
                b_idx + 1, n_batches, n_valid, rate, elapsed,
            )

    valid_mask = ~np.isnan(norms)
    n_total = int(len(norms))
    n_used = int(valid_mask.sum())
    valid_norms = norms[valid_mask]

    mean_z = sum_z / n_valid
    var_z = sum_z_sq / n_valid - mean_z ** 2
    coord_std = np.sqrt(np.clip(var_z, 0.0, None))

    summary = {
        "dataset": str(ZINC_PATH),
        "n_smiles_total": n_total,
        "n_encoded": n_used,
        "n_failed": n_total - n_used,
        "latent_dim": int(codec.EMBEDDING_DIM),
        "norm_mean": float(np.mean(valid_norms)),
        "norm_std": float(np.std(valid_norms)),
        "norm_median": float(np.median(valid_norms)),
        "norm_p05": float(np.percentile(valid_norms, 5)),
        "norm_p25": float(np.percentile(valid_norms, 25)),
        "norm_p75": float(np.percentile(valid_norms, 75)),
        "norm_p95": float(np.percentile(valid_norms, 95)),
        "norm_min": float(np.min(valid_norms)),
        "norm_max": float(np.max(valid_norms)),
        "per_coord_std_mean": float(coord_std.mean()),
        "per_coord_std_min": float(coord_std.min()),
        "per_coord_std_max": float(coord_std.max()),
        "per_coord_mean_abs_mean": float(np.abs(mean_z).mean()),
        "elapsed_seconds": time.time() - t0,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Saved summary to %s", OUT_PATH)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
