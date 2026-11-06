"""Tests for new benchmark infrastructure (synthetic, designbench, nasbench).

Tests verify:
  - Config SSOT has all new benchmark constants
  - Arch2Vec codec interface works with mock data
  - NAS-Bench oracle interface works with mock data
"""

import numpy as np
import pytest

from src.analysis.config import (
    DESIGNBENCH_TASKS,
    NASBENCH_TASKS,
    RESULT_DIRS,
    SYNTHETIC_TASKS,
)


# ---------------------------------------------------------------------------
# Config SSOT tests
# ---------------------------------------------------------------------------


class TestConfigSSoT:
    """Verify new benchmark constants in config.py."""

    def test_synthetic_tasks_defined(self):
        assert "branin_100d" in SYNTHETIC_TASKS
        assert "branin_500d" in SYNTHETIC_TASKS
        assert "hartmann6_100d" in SYNTHETIC_TASKS
        assert "hartmann6_500d" in SYNTHETIC_TASKS

    def test_synthetic_task_structure(self):
        for name, cfg in SYNTHETIC_TASKS.items():
            assert "function" in cfg
            assert "ambient_dim" in cfg
            assert "effective_dim" in cfg
            assert cfg["ambient_dim"] > cfg["effective_dim"]

    def test_designbench_tasks_defined(self):
        assert "tfbind8" in DESIGNBENCH_TASKS
        assert DESIGNBENCH_TASKS["tfbind8"]["dim"] == 32

    def test_nasbench_tasks_defined(self):
        assert "cifar10" in NASBENCH_TASKS
        assert NASBENCH_TASKS["cifar10"]["latent_dim"] == 16

    def test_result_dirs_extended(self):
        assert "synthetic" in RESULT_DIRS
        assert "designbench" in RESULT_DIRS
        assert "nasbench" in RESULT_DIRS


# ---------------------------------------------------------------------------
# Arch2Vec codec tests (mock data)
# ---------------------------------------------------------------------------


class TestArch2VecCodec:
    """Test arch2vec codec with mock data."""

    def test_find_nearest_returns_closest(self):
        """find_nearest should return the closest architecture by L2."""
        from src.codec.arch2vec import Arch2VecCodec

        codec = Arch2VecCodec(latent_dim=4, device="cpu")
        # Mock arch data
        import torch
        codec._arch_data = {
            "hash_a": {"embedding": torch.tensor([1.0, 0.0, 0.0, 0.0])},
            "hash_b": {"embedding": torch.tensor([0.0, 1.0, 0.0, 0.0])},
            "hash_c": {"embedding": torch.tensor([0.5, 0.5, 0.0, 0.0])},
        }

        # Query near hash_a
        results = codec.find_nearest(np.array([0.9, 0.1, 0.0, 0.0]), k=1)
        assert results[0][0] == "hash_a"

        # Query near hash_b
        results = codec.find_nearest(np.array([0.1, 0.9, 0.0, 0.0]), k=1)
        assert results[0][0] == "hash_b"

    def test_get_all_embeddings_shape(self):
        """get_all_embeddings should return (N, latent_dim) array."""
        from src.codec.arch2vec import Arch2VecCodec

        codec = Arch2VecCodec(latent_dim=4, device="cpu")
        import torch
        codec._arch_data = {
            f"hash_{i}": {"embedding": torch.randn(4)} for i in range(10)
        }

        embs, hashes = codec.get_all_embeddings()
        assert embs.shape == (10, 4)
        assert len(hashes) == 10

    def test_encode_by_hash(self):
        """encode_by_hash should return the correct embedding."""
        from src.codec.arch2vec import Arch2VecCodec

        codec = Arch2VecCodec(latent_dim=4, device="cpu")
        import torch
        expected = torch.tensor([1.0, 2.0, 3.0, 4.0])
        codec._arch_data = {"test_hash": {"embedding": expected}}

        result = codec.encode_by_hash("test_hash")
        np.testing.assert_allclose(result, expected.numpy(), atol=1e-6)

    def test_dim_property(self):
        from src.codec.arch2vec import Arch2VecCodec

        codec = Arch2VecCodec(latent_dim=16)
        assert codec.dim == 16


# ---------------------------------------------------------------------------
# NAS-Bench oracle tests (mock data)
# ---------------------------------------------------------------------------


class TestNASBenchOracle:
    """Test NAS-Bench-101 oracle with mock data."""

    def test_score_from_mock_lookup(self):
        """Oracle should return accuracy from lookup table."""
        from src.oracle.nasbench import NASBench101Oracle

        oracle = NASBench101Oracle()
        oracle._lookup = {"arch_123": 0.943, "arch_456": 0.912}

        assert oracle.score("arch_123") == 0.943
        assert oracle("arch_456") == 0.912

    def test_missing_hash_returns_zero(self):
        """Unknown architecture hash should return 0.0."""
        from src.oracle.nasbench import NASBench101Oracle

        oracle = NASBench101Oracle()
        oracle._lookup = {"known": 0.9}

        assert oracle.score("unknown") == 0.0


# ---------------------------------------------------------------------------
# COWBOYS TanimotoKernel tests
# ---------------------------------------------------------------------------


try:
    import gpytorch as _gpytorch  # noqa: F401
    _has_gpytorch = True
except ImportError:
    _has_gpytorch = False


@pytest.mark.skipif(not _has_gpytorch, reason="gpytorch not installed")
class TestCOWBOYSSolver:
    def test_tanimoto_kernel_diagonal_is_one(self):
        from src.spherevo.cowboys_solver import TanimotoKernel
        import torch
        k = TanimotoKernel()
        x = torch.tensor([[1,0,1,0],[0,1,1,0],[1,1,0,0]], dtype=torch.float64)
        K = k(x, x).evaluate()
        np.testing.assert_allclose(K.diag().numpy(), 1.0, atol=1e-6)

    def test_tanimoto_kernel_symmetric(self):
        from src.spherevo.cowboys_solver import TanimotoKernel
        import torch
        k = TanimotoKernel()
        x = torch.rand(5, 20, dtype=torch.float64)
        K = k(x, x).evaluate()
        np.testing.assert_allclose(K.numpy(), K.T.numpy(), atol=1e-10)

    def test_tanimoto_kernel_known_value(self):
        from src.spherevo.cowboys_solver import TanimotoKernel
        import torch
        k = TanimotoKernel()
        # [1,1,0] and [1,0,1]: intersection=1, union=3, tanimoto=1/3
        x = torch.tensor([[1,1,0],[1,0,1]], dtype=torch.float64)
        K = k(x, x).evaluate()
        assert abs(K[0,1].item() - 1/3) < 1e-6
