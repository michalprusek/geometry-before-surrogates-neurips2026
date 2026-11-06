"""Codec wrapper for VMFTransformerVAE.

Mirrors PowerSphericalVAECodec; only the model class differs.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import torch

from src.codec.selfies_vae import _SELFIESVocab
from src.codec.spcauchy_codec import SpCauchyVAECodec
from src.codec.vmf_vae import VMFTransformerVAE

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_VMF_WEIGHTS_PATH = _PROJECT_ROOT / "weights" / "vmf_vae" / "best_model.pt"


class VMFVAECodec(SpCauchyVAECodec):
    """von Mises-Fisher molecular codec.

    Same MolecularCodec interface as SpCauchyVAECodec; underlying model is a
    vMF posterior VAE rather than spCauchy. Third explicit-spherical-prior
    point in the codec-geometry ablation.
    """

    @classmethod
    def from_pretrained(
        cls,
        weights_path: Optional[str] = None,
        device: str = "cuda",
    ) -> "VMFVAECodec":
        weights_path = Path(weights_path) if weights_path else DEFAULT_VMF_WEIGHTS_PATH
        vocab = _SELFIESVocab()
        model = VMFTransformerVAE(
            vocab_size=len(vocab),
            bottleneck_size=2,
            d_model=128,
        )
        model.max_string_length = 1024

        if weights_path.exists():
            ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model.load_state_dict(state_dict)
            logger.info("Loaded vMF VAE weights from %s", weights_path)
        else:
            raise FileNotFoundError(
                f"vMF VAE weights not found at {weights_path}\n"
                f"The weights must be provided; the trainer is not part of this release."
            )

        return cls(model, vocab, device)
