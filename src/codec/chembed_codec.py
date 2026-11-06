"""Chembed VAE molecular codec (256D, PubChem-trained Transformer VAE).

Wraps the `chembed` package (Talibart & Gilis 2025) as a `MolecularCodec`
implementation, so the existing `run_benchmark.py` solver pipeline can target
Chembed by simply switching the `--codec chembed` flag.

Chembed produces latents of shape ``(N, d_bottleneck, d_model)`` (default
``(N, 1, 256)``); we squeeze/unsqueeze to expose a flat ``[N, 256]`` tensor
matching the SELFIES interface.

The `chembed` package and its dependencies (chembl_structure_pipeline, rdkit,
fastparquet, selfies, huggingface_hub, safetensors) require Python 3.12.
On systems where it is not on the default path, set ``CHEMBED_REPO`` env var
to its repository root, or rely on the default ``/home/prusek/chembed_ref``.

Example::

    codec = ChembedVAECodec.from_pretrained()
    z = codec.encode(["CCO", "CCC", "c1ccccc1"])  # [3, 256]
    smiles = codec.decode(z)
"""
from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import torch

from src.codec.selfies_vae import MolecularCodec

logger = logging.getLogger(__name__)

DEFAULT_CHEMBED_REPO = os.environ.get("CHEMBED_REPO", "/home/prusek/chembed_ref")
DEFAULT_HF_REPO_ID = "3BioCompBio/chembed-default"


def _ensure_chembed_on_path() -> None:
    """Make sure `chembed` package is importable; insert repo root on sys.path
    if needed. Idempotent."""
    if DEFAULT_CHEMBED_REPO not in sys.path:
        sys.path.insert(0, DEFAULT_CHEMBED_REPO)


class _ChembedModelProxy:
    """Compatibility proxy exposing SELFIES-VAE-like attributes (`max_string_length`)
    over a Chembed VAE (`max_len`).

    `make_our_latent_bb` in `scripts/run_benchmark.py` sets
    ``codec.model.max_string_length = 80`` to cap decoder runtime on
    out-of-distribution candidates. Chembed's analog is ``vae.max_len``;
    this proxy forwards reads/writes between the two names.
    """

    __slots__ = ("_vae",)

    def __init__(self, vae):
        object.__setattr__(self, "_vae", vae)

    @property
    def max_string_length(self) -> int:
        return int(self._vae.max_len)

    @max_string_length.setter
    def max_string_length(self, value: int) -> None:
        self._vae.max_len = int(value)

    def __getattr__(self, name):
        return getattr(self._vae, name)


class ChembedVAECodec(MolecularCodec):
    """Chembed VAE codec exposing a flat 256-D latent.

    The Chembed package's encode/decode functions handle SMILES standardization
    and SELFIES tokenization internally. This adapter only handles the shape
    squeeze/unsqueeze and the `MolecularCodec` interface.

    Latent shape inside Chembed is ``(N, d_bottleneck, d_model)`` (default
    ``(N, 1, 256)``); we present ``[N, d_bottleneck * d_model]`` to callers.
    """

    def __init__(self, vae, device: str = "cuda"):
        _ensure_chembed_on_path()
        self.vae = vae
        self.device = torch.device(device)
        self.vae.eval()
        # Cache the flattened latent dim
        self._dim = int(self.vae.d_bottleneck * self.vae.d_model)
        # Compatibility proxy so `codec.model.max_string_length = N` works
        self.model = _ChembedModelProxy(self.vae)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def embedding_dim(self) -> int:
        """Alias for backward compatibility with SELFIESVAECodec."""
        return self._dim

    @classmethod
    def from_pretrained(
        cls,
        repo_id: Optional[str] = None,
        device: str = "cuda",
    ) -> "ChembedVAECodec":
        """Load Chembed VAE from HuggingFace Hub (default `3BioCompBio/chembed-default`)."""
        _ensure_chembed_on_path()
        from chembed import checkpoint_utils

        if repo_id is None:
            repo_id = DEFAULT_HF_REPO_ID

        torch_device = torch.device(device)
        vae = checkpoint_utils.load_vae_from_hub(device=torch_device, repo_id=repo_id)
        logger.info("Loaded Chembed VAE from HF hub repo=%s onto %s", repo_id, torch_device)
        return cls(vae, device=device)

    def encode(self, smiles_list: list[str]) -> torch.Tensor:
        """Encode SMILES to flat 256-D latent vectors. Returns tensor on `self.device`."""
        _ensure_chembed_on_path()
        from chembed.encode import encode_multiple_smiles

        z = encode_multiple_smiles(
            smiles_list,
            self.vae,
            batch_size=256,
            replace_if_not_in_vocab=True,
            use_sigma=False,
        )
        # z shape: (N, d_bottleneck, d_model) -> (N, d_bottleneck * d_model)
        return z.reshape(z.shape[0], -1).to(self.device)

    def decode(self, z: torch.Tensor) -> list[str]:
        """Decode flat 256-D latents to SMILES strings.

        Failures (invalid SELFIES, decoding errors) yield empty strings as
        per the `MolecularCodec` contract.
        """
        _ensure_chembed_on_path()
        from chembed.decode import decode_zs_to_smiles

        if not isinstance(z, torch.Tensor):
            z = torch.as_tensor(z, dtype=torch.float32)
        # Move to model device and unsqueeze bottleneck dim back: [N, d] -> [N, 1, d]
        z = z.to(self.vae.device, dtype=torch.float32)
        z3 = z.reshape(z.shape[0], self.vae.d_bottleneck, self.vae.d_model)
        try:
            return list(decode_zs_to_smiles(z3, self.vae))
        except (torch.cuda.OutOfMemoryError, MemoryError, KeyboardInterrupt):
            # Resource/interrupt errors must surface — silent retry on these
            # would mask GPU exhaustion and produce a benchmark of empty SMILES.
            raise
        except Exception as e:  # noqa: BLE001
            # Decoder-level failures (invalid SELFIES/vocab drift, tokenizer bug)
            # → return empty SMILES; oracle scores them 0.0 which is the correct
            # behavior for non-decodable latents. Log at error so failure rates
            # are auditable in the run log.
            logger.error(
                "Chembed decode failed for batch shape %s: %s: %s",
                tuple(z3.shape), type(e).__name__, e,
            )
            return ["" for _ in range(z3.shape[0])]
