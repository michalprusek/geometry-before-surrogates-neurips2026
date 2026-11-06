"""Molecular codec package."""

from src.codec.selfies_vae import MolecularCodec, SELFIESVAECodec

# ChembedVAECodec is import-on-demand: requires Python 3.12 + chembed package
# which is only available in the chembed_venv. Importing eagerly would break
# regular SELFIES-only runs on Python <3.12.
__all__ = ["MolecularCodec", "SELFIESVAECodec"]
