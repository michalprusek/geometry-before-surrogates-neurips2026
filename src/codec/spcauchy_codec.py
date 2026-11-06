"""Spherical Cauchy VAE molecular codec.

Encodes SMILES strings to embeddings on S^{d-1} via a spCauchy Transformer
VAE and decodes them back. The latent space is natively spherical, so no
post-hoc normalization is needed for SphereVO.

Interface is identical to SELFIESVAECodec (implements MolecularCodec ABC).

Dependencies: torch, selfies, rdkit (for canonicalization only).
"""

import logging
from pathlib import Path
from typing import Optional

import selfies as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.codec.selfies_vae import MolecularCodec, _SELFIESVocab, DEFAULT_SELFIES_VOCAB
from src.codec.spcauchy_vae import SpCauchyTransformerVAE

logger = logging.getLogger(__name__)

# Default weights path: <project_root>/weights/spcauchy_vae/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SPCAUCHY_WEIGHTS_PATH = _PROJECT_ROOT / "weights" / "spcauchy_vae" / "best_model.pt"


class SpCauchyVAECodec(MolecularCodec):
    """Spherical Cauchy VAE codec for molecular embeddings on S^{d-1}.

    Latent embeddings live natively on the unit sphere S^{d-1} where
    d = bottleneck_size * d_model (default: 256).

    Unlike SELFIESVAECodec, encode() returns NORMALIZED vectors (unit norm),
    so downstream SphereVO solvers can skip post-hoc spherical normalization.

    Example::

        codec = SpCauchyVAECodec.from_pretrained()
        z = codec.encode(["CCO", "CCC", "c1ccccc1"])  # [3, 256], ||z||=1
        smiles = codec.decode(z)                        # list of SMILES
    """

    def __init__(
        self,
        model: SpCauchyTransformerVAE,
        vocab: _SELFIESVocab,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.vocab = vocab
        self.device = device
        self._embedding_dim = model.latent_dim

    @property
    def dim(self) -> int:
        return self._embedding_dim

    @property
    def embedding_dim(self) -> int:
        """Alias for backward compatibility."""
        return self._embedding_dim

    @classmethod
    def from_pretrained(
        cls,
        weights_path: Optional[str] = None,
        device: str = "cuda",
    ) -> "SpCauchyVAECodec":
        """Load codec from pretrained spCauchy VAE weights.

        Args:
            weights_path: Path to checkpoint ``.pt`` file.
                Defaults to ``<project_root>/weights/spcauchy_vae/best_model.pt``.
            device: Torch device string.
        """
        weights_path = Path(weights_path) if weights_path else DEFAULT_SPCAUCHY_WEIGHTS_PATH

        vocab = _SELFIESVocab()

        # Create model matching architecture defaults
        model = SpCauchyTransformerVAE(
            vocab_size=len(vocab),
            bottleneck_size=2,
            d_model=128,
        )
        model.max_string_length = 1024

        if weights_path.exists():
            checkpoint = torch.load(
                weights_path, map_location="cpu", weights_only=False
            )
            # Support both raw state_dict and checkpoint dict
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
            logger.info("Loaded spCauchy VAE weights from %s", weights_path)
        else:
            raise FileNotFoundError(
                f"spCauchy VAE weights not found at {weights_path}\n"
                f"The weights must be provided; the trainer is not part of this release."
            )

        return cls(model, vocab, device)

    @classmethod
    def from_gaussian_vae(
        cls,
        gaussian_weights_path: Optional[str] = None,
        device: str = "cuda",
    ) -> "SpCauchyVAECodec":
        """Create spCauchy codec initialized from pretrained Gaussian VAE.

        Transfers decoder weights for fine-tuning. The encoder's spCauchy
        heads (rho_head) are randomly initialized.

        Args:
            gaussian_weights_path: Path to Gaussian VAE state dict.
            device: Torch device string.
        """
        from src.codec.selfies_vae import DEFAULT_WEIGHTS_PATH

        gaussian_path = (
            Path(gaussian_weights_path) if gaussian_weights_path
            else DEFAULT_WEIGHTS_PATH
        )

        vocab = _SELFIESVocab()

        model = SpCauchyTransformerVAE(
            vocab_size=len(vocab),
            bottleneck_size=2,
            d_model=128,
        )
        model.max_string_length = 1024

        if gaussian_path.exists():
            gaussian_state = torch.load(
                gaussian_path, map_location="cpu", weights_only=False
            )
            model.init_from_gaussian_vae(gaussian_state)
            logger.info(
                "Initialized spCauchy VAE from Gaussian VAE at %s",
                gaussian_path,
            )
        else:
            raise FileNotFoundError(
                f"Gaussian VAE weights not found at {gaussian_path}"
            )

        return cls(model, vocab, device)

    # -- SELFIES helpers (identical to SELFIESVAECodec) ----------------------

    @staticmethod
    def _smiles_to_selfies(smiles: str) -> Optional[str]:
        """Convert SMILES to SELFIES. Returns None on failure."""
        try:
            return sf.encoder(smiles)
        except sf.EncoderError as e:
            logger.debug("SELFIES encoding failed for '%s...': %s", smiles[:50], e)
            return None
        except Exception as e:
            logger.warning(
                "Unexpected error encoding '%s...': %s: %s",
                smiles[:50],
                type(e).__name__,
                e,
            )
            return None

    @staticmethod
    def _selfies_to_smiles(selfies_str: str) -> Optional[str]:
        """Convert SELFIES to SMILES. Returns None on failure."""
        try:
            return sf.decoder(selfies_str)
        except sf.DecoderError as e:
            logger.debug("SELFIES decoding failed: %s", e)
            return None
        except Exception as e:
            logger.warning(
                "Unexpected error decoding SELFIES: %s: %s", type(e).__name__, e
            )
            return None

    def _tokenize_selfies(self, selfies_str: str) -> torch.Tensor:
        """Tokenize SELFIES string, filtering unknown tokens."""
        tokens = list(sf.split_selfies(selfies_str))
        known_tokens = [t for t in tokens if t in self.vocab.vocab2idx]
        if not known_tokens:
            known_tokens = ["[C]"]
        return self.vocab.encode(known_tokens)

    # -- Encode / Decode ---------------------------------------------------

    def encode(self, smiles_list: list[str]) -> torch.Tensor:
        """Encode SMILES strings to unit-norm embeddings on S^{d-1}.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            Tensor of shape ``[N, dim]`` with ``||z_i|| = 1``.
        """
        if not smiles_list:
            return torch.empty(0, self._embedding_dim, device=self.device)

        embeddings = []
        invalid_count = 0
        with torch.no_grad():
            for smiles in smiles_list:
                selfies_str = self._smiles_to_selfies(smiles)
                if selfies_str is None:
                    invalid_count += 1
                    # Return a random unit vector for invalid molecules
                    rand_z = torch.randn(self._embedding_dim, device=self.device)
                    rand_z = F.normalize(rand_z, dim=0)
                    embeddings.append(rand_z)
                    continue

                try:
                    tokens = (
                        self._tokenize_selfies(selfies_str)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    mu, _rho = self.model.encode(tokens)
                    # mu is already L2-normalized and on S^{d-1}
                    embeddings.append(mu[0])  # [latent_dim]
                except torch.cuda.OutOfMemoryError:
                    raise
                except RuntimeError as e:
                    logger.warning("VAE encode failed for '%s': %s", smiles[:50], e)
                    invalid_count += 1
                    rand_z = torch.randn(self._embedding_dim, device=self.device)
                    rand_z = F.normalize(rand_z, dim=0)
                    embeddings.append(rand_z)

        if invalid_count > 0:
            logger.warning(
                "Encoded %d/%d invalid molecules as random unit vectors",
                invalid_count,
                len(smiles_list),
            )

        return torch.stack(embeddings)

    def decode(self, z: torch.Tensor) -> list[str]:
        """Decode embeddings to SMILES strings.

        Input z is normalized to unit norm before decoding (spCauchy VAE
        expects inputs on S^{d-1}).

        Args:
            z: Tensor of shape ``[N, dim]`` or ``[dim]``.

        Returns:
            List of SMILES strings (empty string on failure).
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        # Normalize to sphere (in case caller passes non-unit vectors)
        z = F.normalize(z.float(), dim=-1)

        B = z.shape[0]
        smiles_list = []

        with torch.no_grad():
            for i in range(B):
                try:
                    # Reshape to [1, bottleneck_size, d_model]
                    z_i = z[i: i + 1].reshape(
                        1, self.model.bottleneck_size, self.model.d_model
                    ).to(self.device)
                    tokens = self.model.sample(z=z_i)
                    selfies_str = self.vocab.decode(tokens[0].cpu())
                    smiles = self._selfies_to_smiles(selfies_str)
                    smiles_list.append(smiles if smiles is not None else "")
                except torch.cuda.OutOfMemoryError:
                    raise
                except RuntimeError as e:
                    logger.warning("VAE decode failed for embedding %d: %s", i, e)
                    smiles_list.append("")

        return smiles_list

    def encode_batch(
        self, smiles_list: list[str], batch_size: int = 64
    ) -> torch.Tensor:
        """Encode SMILES strings in batches.

        Args:
            smiles_list: List of SMILES strings.
            batch_size: Batch size for encoding.

        Returns:
            Tensor of shape ``[N, dim]`` with unit norm.
        """
        all_embeddings = []
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i: i + batch_size]
            embeddings = self.encode(batch)
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0)
