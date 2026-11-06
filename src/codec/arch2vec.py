"""arch2vec codec: GIN VAE for NAS-Bench-101 architecture encoding.

Wraps the pretrained arch2vec model (Yan et al., 2020) for encoding/decoding
neural architectures. The 16D latent space represents the graph structure
(adjacency matrix + operations) of NAS-Bench-101 cells.

Requires pretrained weights: weights/arch2vec/
Download: see scripts/download_arch2vec.sh
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_WEIGHTS_DIR = PROJECT_ROOT / "weights" / "arch2vec"

# NAS-Bench-101 architecture representation:
# - adjacency matrix: 7x7 upper triangular (21 values)
# - operations: 5 interior nodes x 3 ops (one-hot) = 15 values
# Total discrete representation: 36 values
NASBENCH_ADJ_SIZE = 21  # upper triangle of 7x7
NASBENCH_OPS_SIZE = 15  # 5 nodes x 3 ops
NASBENCH_REPR_SIZE = NASBENCH_ADJ_SIZE + NASBENCH_OPS_SIZE  # 36


class Arch2VecCodec:
    """Codec wrapping arch2vec GIN VAE for NAS-Bench-101.

    Encodes architectures (adjacency + ops) -> 16D latent.
    Decodes 16D latent -> reconstructed (adjacency + ops).

    The arch2vec model is a Graph Isomorphism Network (GIN) VAE that
    processes the architecture graph directly, producing a 16D latent
    representation that captures structural similarity.
    """

    def __init__(self, latent_dim: int = 16, device: str = "cpu"):
        self.latent_dim = latent_dim
        self.device = device
        self._model = None
        self._arch_data = None
        self._cached_embeddings = None

    @property
    def dim(self) -> int:
        """Dimensionality of the latent space."""
        return self.latent_dim

    @classmethod
    def from_pretrained(
        cls, weights_dir: str | Path | None = None, device: str = "cpu"
    ) -> "Arch2VecCodec":
        """Load pretrained arch2vec model.

        Args:
            weights_dir: Directory containing arch2vec pretrained weights.
                Expected files: model.pt, arch2vec-nasbench101.pt
            device: torch device
        """
        codec = cls(latent_dim=16, device=device)
        weights_dir = Path(weights_dir) if weights_dir else DEFAULT_WEIGHTS_DIR

        if not weights_dir.exists():
            raise FileNotFoundError(
                f"arch2vec weights not found at {weights_dir}. "
                "Download with: bash scripts/download_arch2vec.sh"
            )

        # Load precomputed latent embeddings (faster than running GIN encoder)
        embeddings_path = weights_dir / "arch2vec-nasbench101.pt"
        if embeddings_path.exists():
            logger.info(f"Loading precomputed arch2vec embeddings from {embeddings_path}")
            data = torch.load(embeddings_path, map_location=device, weights_only=False)
            # data is a dict: {arch_hash: {"embedding": tensor_16d, ...}}
            codec._arch_data = data
            logger.info(f"Loaded {len(data)} architecture embeddings")
        else:
            raise FileNotFoundError(
                f"Precomputed embeddings not found at {embeddings_path}. "
                "Download with: bash scripts/download_arch2vec.sh"
            )

        return codec

    def get_all_embeddings(self) -> tuple[np.ndarray, list[str]]:
        """Return all precomputed embeddings and their architecture hashes.

        Cached after first call to avoid rebuilding the 423K-entry array
        on every nearest-neighbor lookup.

        Returns:
            (embeddings [N, 16], arch_hashes [N])
        """
        if self._cached_embeddings is not None:
            return self._cached_embeddings

        if self._arch_data is None:
            raise RuntimeError("Codec not loaded. Call from_pretrained() first.")

        hashes = sorted(self._arch_data.keys())
        embeddings = []
        for h in hashes:
            entry = self._arch_data[h]
            # arch2vec stores embeddings under "feature" or "embedding" key
            emb = entry.get("feature", entry.get("embedding"))
            if emb is None:
                raise KeyError(f"No 'feature' or 'embedding' key in arch data entry {h}. Keys: {list(entry.keys())}")
            if isinstance(emb, torch.Tensor):
                embeddings.append(emb.cpu().numpy().flatten())
            else:
                embeddings.append(np.array(emb).flatten())

        self._cached_embeddings = (np.array(embeddings, dtype=np.float32), hashes)
        return self._cached_embeddings

    def encode_by_hash(self, arch_hash) -> np.ndarray:
        """Get latent embedding for a known architecture key."""
        if self._arch_data is None:
            raise RuntimeError("Codec not loaded.")
        entry = self._arch_data[arch_hash]
        emb = entry.get("feature", entry.get("embedding"))
        if isinstance(emb, torch.Tensor):
            return emb.cpu().numpy().flatten()
        return np.array(emb).flatten()

    def find_nearest(self, z: np.ndarray, k: int = 1) -> list[tuple]:
        """Find nearest architecture(s) to latent vector z.

        Uses L2 distance in latent space. This is the "decode" operation:
        map continuous latent -> nearest known architecture.

        Args:
            z: latent vector [16] or [1, 16]
            k: number of nearest neighbors

        Returns:
            List of (arch_key, distance) tuples. Keys are returned as strings
            for compatibility with the NASBench101Oracle lookup table.
        """
        z = np.asarray(z).flatten()
        all_emb, hashes = self.get_all_embeddings()
        dists = np.linalg.norm(all_emb - z[None, :], axis=1)
        idx = np.argsort(dists)[:k]
        return [(str(hashes[i]), float(dists[i])) for i in idx]
