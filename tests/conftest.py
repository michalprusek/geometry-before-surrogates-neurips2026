"""Shared fixtures and mocks for NeurIPS tests.

MockBlackBox wraps MockCodec+MockOracle into a single callable matching
the poli interface: black_box(z) -> scores.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


class MockCodec:
    """Identity-like codec: encode returns the input as a fixed-dim tensor,
    decode returns dummy SMILES strings.

    For SMILES input, encode hashes each string to a deterministic vector.
    """

    def __init__(self, dim: int = 32):
        self.dim = dim

    def encode(self, smiles_list: list[str]) -> torch.Tensor:
        """Deterministic encoding: hash SMILES to a fixed vector."""
        vectors = []
        for s in smiles_list:
            # Use hash of string to seed a deterministic vector
            rng = np.random.RandomState(hash(s) % (2**31))
            vectors.append(rng.randn(self.dim))
        return torch.tensor(np.array(vectors), dtype=torch.float32)

    def decode(self, embeddings: torch.Tensor) -> list[str]:
        """Return dummy SMILES strings derived from embedding values.

        Uses the first few embedding dimensions to generate a unique string
        per latent vector, so different latent vectors produce different SMILES.
        """
        results = []
        for i in range(embeddings.shape[0]):
            key = int(embeddings[i, :3].sum().item() * 1000) % 10000
            results.append(f"MOL{key:04d}")
        return results


class MockOracle:
    """Deterministic oracle: hashes SMILES to a score in [0, 1]."""

    def __call__(self, smiles: str) -> float:
        h = hash(smiles) % 10000
        return h / 10000.0


class MockBlackBox:
    """Poli-compatible black box wrapping MockCodec + MockOracle.

    Mirrors LatentBlackBox in run_benchmark.py: decode latent vectors
    to SMILES, then score them.
    """

    def __init__(self, codec: MockCodec, oracle: MockOracle):
        self.codec = codec
        self.oracle = oracle

    def __call__(self, z: np.ndarray) -> np.ndarray:
        z_tensor = torch.tensor(z, dtype=torch.float32)
        smiles_list = self.codec.decode(z_tensor)
        scores = []
        for smi in smiles_list:
            scores.append(self.oracle(smi) if smi else 0.0)
        return np.array(scores).reshape(-1, 1)


@pytest.fixture
def mock_codec():
    return MockCodec(dim=32)


@pytest.fixture
def mock_oracle():
    return MockOracle()


@pytest.fixture
def mock_black_box(mock_codec, mock_oracle):
    return MockBlackBox(mock_codec, mock_oracle)


@pytest.fixture
def cold_start_data(mock_codec, mock_oracle):
    """Return (x0, y0) cold-start arrays in poli format."""
    smiles = ["CCO", "CCCO", "CC=O", "C1CCCCC1", "c1ccccc1"]
    x0 = mock_codec.encode(smiles).numpy()
    y0 = np.array([mock_oracle(s) for s in smiles]).reshape(-1, 1)
    return x0, y0
