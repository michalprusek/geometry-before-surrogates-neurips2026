#!/usr/bin/env bash
# Download the pretrained SELFIES Transformer VAE weights released with LOL-BO
# (Maus et al., NeurIPS 2022; https://github.com/nataliemaus/lolbo, MIT License)
# to weights/SELFIES-VAE-state-dict.pt and verify the file.
#
# The weights are not redistributed in this repository. The URL below is the
# Git-LFS object of the LOL-BO repository at a fixed commit.
#
# Usage:  bash scripts/download_weights.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/weights/SELFIES-VAE-state-dict.pt"
COMMIT=0876147812bc86511528d69d985af9712daa5175
URL="https://media.githubusercontent.com/media/nataliemaus/lolbo/$COMMIT/lolbo/utils/mol_utils/selfies_vae/state_dict/SELFIES-VAE-state-dict.pt"
SHA256=edffc4b4c2d5582a6efa372eddae8913b8a38ba18d96006a6c3cdcdffacb31f0
BYTES=25366351

sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

if [ ! -f "$OUT" ]; then
    mkdir -p "$(dirname "$OUT")"
    echo "Downloading $URL"
    if command -v curl >/dev/null; then curl -fL --retry 3 -o "$OUT.part" "$URL"; else wget -O "$OUT.part" "$URL"; fi
    mv "$OUT.part" "$OUT"
fi
GOT="$(sha "$OUT")"
if [ "$GOT" != "$SHA256" ]; then
    echo "ERROR: $OUT has sha256 $GOT, expected $SHA256 ($BYTES bytes)." >&2
    echo "A file of about 130 bytes is a Git-LFS pointer, not the weights. Delete it and retry, or" >&2
    echo "clone https://github.com/nataliemaus/lolbo with git-lfs and copy the file by hand." >&2
    exit 1
fi
echo "OK: $OUT ($BYTES bytes, sha256 verified)"
