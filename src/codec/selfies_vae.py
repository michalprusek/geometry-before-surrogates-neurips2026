"""SELFIES VAE molecular codec.

Encodes SMILES strings to 256D continuous embeddings via a SELFIES
Transformer VAE (LOL-BO, Maus et al., 2022; MIT) and decodes them back.

Dependencies: torch, selfies, rdkit (for canonicalization only).
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import selfies as sf
import torch
import torch.nn as nn

from src.codec.vae_model import InfoTransformerVAE

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default SELFIES vocabulary (LOL-BO's list: token set of the GuacaMol v1 training set)
# ---------------------------------------------------------------------------
DEFAULT_SELFIES_VOCAB = [
    "<start>",
    "<stop>",
    "[#Branch1]",
    "[#Branch2]",
    "[#C-1]",
    "[#C]",
    "[#N+1]",
    "[#N]",
    "[#O+1]",
    "[=B]",
    "[=Branch1]",
    "[=Branch2]",
    "[=C-1]",
    "[=C]",
    "[=N+1]",
    "[=N-1]",
    "[=NH1+1]",
    "[=NH2+1]",
    "[=N]",
    "[=O+1]",
    "[=OH1+1]",
    "[=O]",
    "[=PH1]",
    "[=P]",
    "[=Ring1]",
    "[=Ring2]",
    "[=S+1]",
    "[=SH1]",
    "[=S]",
    "[=Se+1]",
    "[=Se]",
    "[=Si]",
    "[B-1]",
    "[BH0]",
    "[BH1-1]",
    "[BH2-1]",
    "[BH3-1]",
    "[B]",
    "[Br+2]",
    "[Br-1]",
    "[Br]",
    "[Branch1]",
    "[Branch2]",
    "[C+1]",
    "[C-1]",
    "[CH1+1]",
    "[CH1-1]",
    "[CH1]",
    "[CH2+1]",
    "[CH2]",
    "[C]",
    "[Cl+1]",
    "[Cl+2]",
    "[Cl+3]",
    "[Cl-1]",
    "[Cl]",
    "[F+1]",
    "[F-1]",
    "[F]",
    "[H]",
    "[I+1]",
    "[I+2]",
    "[I+3]",
    "[I]",
    "[N+1]",
    "[N-1]",
    "[NH0]",
    "[NH1+1]",
    "[NH1-1]",
    "[NH1]",
    "[NH2+1]",
    "[NH3+1]",
    "[N]",
    "[O+1]",
    "[O-1]",
    "[OH0]",
    "[O]",
    "[P+1]",
    "[PH1]",
    "[PH2+1]",
    "[P]",
    "[Ring1]",
    "[Ring2]",
    "[S+1]",
    "[S-1]",
    "[SH1]",
    "[S]",
    "[Se+1]",
    "[Se-1]",
    "[SeH1]",
    "[SeH2]",
    "[Se]",
    "[Si-1]",
    "[SiH1-1]",
    "[SiH1]",
    "[SiH2]",
    "[Si]",
]


class _SELFIESVocab:
    """Lightweight vocabulary wrapper for SELFIES tokenization.

    Replaces the LOL-BO SELFIESDataset for inference — no data loading,
    just vocab lookup and token encode/decode.
    """

    def __init__(self, vocab: list[str] = None):
        self.vocab = vocab or DEFAULT_SELFIES_VOCAB
        self.vocab2idx: dict[str, int] = {v: i for i, v in enumerate(self.vocab)}

    def encode(self, tokens: list[str]) -> torch.Tensor:
        """Encode a list of SELFIES tokens to a LongTensor (appends <stop>)."""
        return torch.tensor([self.vocab2idx[t] for t in [*tokens, "<stop>"]])

    def decode(self, token_ids) -> str:
        """Decode token IDs back to a SELFIES string."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        dec = [self.vocab[t] for t in token_ids]
        # Chop out start token and everything past (and including) first stop
        stop = dec.index("<stop>") if "<stop>" in dec else None
        selfie = dec[0:stop]
        while "<start>" in selfie:
            start = 1 + dec.index("<start>")
            selfie = selfie[start:]
        return "".join(selfie)

    def __len__(self) -> int:
        return len(self.vocab)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class MolecularCodec(ABC):
    """Abstract base class for molecular encoders/decoders."""

    @property
    @abstractmethod
    def dim(self) -> int:
        """Dimensionality of the embedding space."""
        ...

    @abstractmethod
    def encode(self, smiles_list: list[str]) -> torch.Tensor:
        """Encode SMILES strings to embeddings.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            Tensor of shape ``[N, dim]``.
        """
        ...

    @abstractmethod
    def decode(self, z: torch.Tensor) -> list[str]:
        """Decode embeddings to SMILES strings.

        Args:
            z: Tensor of shape ``[N, dim]``.

        Returns:
            List of SMILES strings (empty string for decode failures).
        """
        ...


# ---------------------------------------------------------------------------
# SELFIES VAE Codec
# ---------------------------------------------------------------------------

# Default weights path: <project_root>/weights/SELFIES-VAE-state-dict.pt
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_WEIGHTS_PATH = _PROJECT_ROOT / "weights" / "SELFIES-VAE-state-dict.pt"


class SELFIESVAECodec(MolecularCodec):
    """SELFIES VAE codec for 256D molecular embeddings.

    Uses the SELFIES VAE from LOL-BO (Maus et al., 2022):
    - 256D latent space (2 bottleneck tokens x 128 d_model)
    - Transformer-based encoder/decoder
    - SELFIES encoding guarantees syntactically valid molecules

    Example::

        codec = SELFIESVAECodec.from_pretrained()
        z = codec.encode(["CCO", "CCC", "c1ccccc1"])  # [3, 256]
        smiles = codec.decode(z)                        # list of SMILES
    """

    EMBEDDING_DIM = 256  # 2 bottleneck x 128 d_model

    def __init__(
        self,
        model: nn.Module,
        vocab: _SELFIESVocab,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.vocab = vocab
        self.device = device

    @property
    def dim(self) -> int:
        return self.EMBEDDING_DIM

    # Keep the old name as an alias for backward compat
    @property
    def embedding_dim(self) -> int:
        return self.EMBEDDING_DIM

    @classmethod
    def from_pretrained(
        cls,
        weights_path: Optional[str] = None,
        decoder_weights_path: Optional[str] = None,
        device: str = "cuda",
    ) -> "SELFIESVAECodec":
        """Load codec from pretrained SELFIES VAE weights.

        Args:
            weights_path: Path to ``state_dict`` ``.pt`` file.
                Defaults to ``<project_root>/weights/SELFIES-VAE-state-dict.pt``.
            decoder_weights_path: Optional path to fine-tuned decoder weights.
                If provided, decoder parameters are overwritten after loading
                the base model.
            device: Torch device string.
        """
        weights_path = Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH

        vocab = _SELFIESVocab()

        # Create model with default architecture matching LOL-BO pretrained
        model = InfoTransformerVAE(
            vocab_size=len(vocab),
            bottleneck_size=2,
            d_model=128,
        )

        # Match LOL-BO's max_string_length (1024 vs default 256)
        model.max_string_length = 1024

        # Load weights
        if weights_path.exists():
            state_dict = torch.load(
                weights_path, map_location="cpu", weights_only=False
            )
            model.load_state_dict(state_dict)
            logger.info("Loaded SELFIES VAE weights from %s", weights_path)
        else:
            raise FileNotFoundError(
                f"Weights not found at {weights_path}\n"
                f"Expected: <project_root>/weights/SELFIES-VAE-state-dict.pt"
            )

        # Overlay fine-tuned decoder weights if provided
        if decoder_weights_path is not None:
            dec_path = Path(decoder_weights_path)
            if dec_path.exists():
                dec_state = torch.load(dec_path, map_location="cpu", weights_only=True)
                current = model.state_dict()
                current.update(dec_state)
                model.load_state_dict(current)
                logger.info(
                    "Loaded fine-tuned decoder weights from %s "
                    "(%d parameters overwritten)",
                    dec_path,
                    len(dec_state),
                )
            else:
                raise FileNotFoundError(
                    f"Decoder weights not found at {dec_path}"
                )

        return cls(model, vocab, device)

    # -- SELFIES helpers ---------------------------------------------------

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
        """Encode SMILES strings to 256D embeddings.

        Args:
            smiles_list: List of SMILES strings.

        Returns:
            Tensor of shape ``[N, 256]``.
        """
        if not smiles_list:
            return torch.empty(0, self.EMBEDDING_DIM, device=self.device)

        embeddings = []
        invalid_count = 0
        with torch.no_grad():
            for smiles in smiles_list:
                selfies_str = self._smiles_to_selfies(smiles)
                if selfies_str is None:
                    invalid_count += 1
                    embeddings.append(
                        torch.zeros(self.EMBEDDING_DIM, device=self.device)
                    )
                    continue

                try:
                    tokens = (
                        self._tokenize_selfies(selfies_str)
                        .unsqueeze(0)
                        .to(self.device)
                    )
                    mu, _sigma = self.model.encode(tokens)
                    # Flatten: [1, bottleneck_size, d_model] -> [1, 256]
                    z = mu.reshape(1, -1)
                    embeddings.append(z[0])
                except torch.cuda.OutOfMemoryError:
                    raise
                except RuntimeError as e:
                    logger.warning("VAE encode failed for '%s': %s", smiles[:50], e)
                    invalid_count += 1
                    embeddings.append(
                        torch.zeros(self.EMBEDDING_DIM, device=self.device)
                    )

        if invalid_count > 0:
            logger.warning(
                "Encoded %d/%d invalid molecules as zeros",
                invalid_count,
                len(smiles_list),
            )

        return torch.stack(embeddings)

    def decode(self, z: torch.Tensor, deterministic: bool = True) -> list[str]:
        """Decode embeddings to SMILES strings.

        Args:
            z: Tensor of shape ``[N, 256]`` or ``[256]``.
            deterministic: If True (default), use greedy argmax decoding
                (no Gumbel noise). If False, use stochastic Gumbel-softmax
                sampling as in the original LOL-BO training pipeline.

        Returns:
            List of SMILES strings (empty string on failure).
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)

        B = z.shape[0]
        smiles_list = []

        with torch.no_grad():
            for i in range(B):
                try:
                    # Reshape to [1, bottleneck_size, d_model]
                    z_i = z[i : i + 1].reshape(1, 2, 128).to(self.device)
                    tokens = self.model.sample(
                        z=z_i, deterministic=deterministic,
                    )
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
        """Encode SMILES strings in batches for efficiency.

        Args:
            smiles_list: List of SMILES strings.
            batch_size: Batch size for encoding.

        Returns:
            Tensor of shape ``[N, 256]``.
        """
        all_embeddings = []
        for i in range(0, len(smiles_list), batch_size):
            batch = smiles_list[i : i + batch_size]
            embeddings = self.encode(batch)
            all_embeddings.append(embeddings)
        return torch.cat(all_embeddings, dim=0)

    def test_roundtrip(
        self, smiles_list: list[str]
    ) -> tuple[torch.Tensor, list[str], list[bool]]:
        """Test encode-decode roundtrip quality.

        Returns:
            Tuple of (embeddings, decoded_smiles, match_flags).
        """
        embeddings = self.encode(smiles_list)
        decoded = self.decode(embeddings)
        matches = [orig == dec for orig, dec in zip(smiles_list, decoded)]
        return embeddings, decoded, matches
