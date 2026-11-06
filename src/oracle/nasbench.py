"""NAS-Bench-101 oracle: lookup architecture performance from precomputed table.

Provides instant evaluation of neural architectures by table lookup in the
NAS-Bench-101 dataset (423K CIFAR-10 architectures with full training results).

The oracle returns validation accuracy (higher = better), which is the standard
metric for NAS-Bench-101 benchmarks.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "nasbench"


class NASBench101Oracle:
    """Table-lookup oracle for NAS-Bench-101 CIFAR-10.

    Looks up validation accuracy from the precomputed NAS-Bench-101 table.
    Returns mean validation accuracy across the 3 evaluation runs stored
    in the benchmark (at 108 epochs).

    Higher is better (validation accuracy in [0, 1]).
    """

    def __init__(self, data_dir: str | Path | None = None):
        self.data_dir = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
        self._lookup: dict[str, float] | None = None

    def load(self) -> None:
        """Load the NAS-Bench-101 lookup table."""
        lookup_path = self.data_dir / "nasbench101_accuracy.json"
        if not lookup_path.exists():
            raise FileNotFoundError(
                f"NAS-Bench-101 lookup table not found at {lookup_path}. "
                "Download with: bash scripts/download_nasbench.sh"
            )
        with open(lookup_path) as f:
            self._lookup = json.load(f)
        logger.info(f"Loaded NAS-Bench-101 oracle: {len(self._lookup)} architectures")

    def score(self, arch_hash: str) -> float:
        """Return validation accuracy for an architecture hash.

        Args:
            arch_hash: NAS-Bench-101 architecture hash string

        Returns:
            Mean validation accuracy at 108 epochs (in [0, 1])
        """
        if self._lookup is None:
            self.load()
        if arch_hash not in self._lookup:
            logger.warning(f"Architecture {arch_hash} not in NAS-Bench-101 table")
            return 0.0
        return self._lookup[arch_hash]

    def __call__(self, arch_hash: str) -> float:
        return self.score(arch_hash)
