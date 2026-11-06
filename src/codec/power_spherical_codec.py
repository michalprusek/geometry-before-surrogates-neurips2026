"""Codec wrapper around PowerSphericalTransformerVAE.

Mirrors SpCauchyVAECodec's MolecularCodec interface but for the Power
Spherical posterior model. The class inherits from SpCauchyVAECodec to share
the encode/decode token-level helpers (SELFIES tokenization, batching, etc.)
unchanged; only the model class and default weights path differ.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from src.codec.selfies_vae import _SELFIESVocab
from src.codec.spcauchy_codec import SpCauchyVAECodec
from src.codec.power_spherical_vae import PowerSphericalTransformerVAE

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PS_WEIGHTS_PATH = _PROJECT_ROOT / "weights" / "power_spherical_vae" / "best_model.pt"


class PowerSphericalVAECodec(SpCauchyVAECodec):
    """Power Spherical molecular codec.

    Identical interface to SpCauchyVAECodec; underlying model is a Power
    Spherical posterior VAE rather than a spCauchy posterior. Both are
    explicit-spherical-prior architectures, so this codec serves as a
    second hyperspherical-VAE point in the codec-geometry ablation.
    """

    @classmethod
    def from_pretrained(
        cls,
        weights_path: Optional[str] = None,
        device: str = "cuda",
    ) -> "PowerSphericalVAECodec":
        weights_path = Path(weights_path) if weights_path else DEFAULT_PS_WEIGHTS_PATH
        vocab = _SELFIESVocab()
        model = PowerSphericalTransformerVAE(
            vocab_size=len(vocab),
            bottleneck_size=2,
            d_model=128,
        )
        model.max_string_length = 1024

        if weights_path.exists():
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            logger.info("Loaded Power Spherical VAE weights from %s", weights_path)
        else:
            raise FileNotFoundError(
                f"Power Spherical VAE weights not found at {weights_path}\n"
                f"Train with: python scripts/train_power_spherical_vae.py"
            )

        return cls(model, vocab, device)
