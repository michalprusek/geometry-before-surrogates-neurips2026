#!/usr/bin/env bash
# Download the ZINC-250k reference set (the file distributed with chemical_vae,
# Gomez-Bombarelli et al. 2018; molecules from the ZINC database, Irwin et al. 2012)
# and derive data/zinc_250k.txt (one SMILES per line), verifying both files.
#
# Needed only by the geometry, retrieval-floor and bounds scripts. Running cells
# needs only data/zinc_10k.txt and results/precomputed/, which ship with the repository.
#
# Usage:  bash scripts/download_zinc250k.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RAW="$ROOT/data/zinc_250k_raw.csv"
TXT="$ROOT/data/zinc_250k.txt"
COMMIT=a509dc613c4c9be54d448a34829bab25bcf09a79
URL="https://raw.githubusercontent.com/the-matter-lab/chemical_vae/$COMMIT/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
RAW_SHA256=35e3f1a52b1badc0697e373d73a18ad773f415936ff992f4c6baa2e067b3e6ae   # 22,606,589 bytes
TXT_SHA256=df0aa677c770f15c9943684d6600cba8e7076c67be6a090c12f24da3aaa55b70   # 11,302,921 bytes, 249,455 SMILES

sha() { if command -v sha256sum >/dev/null; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }

mkdir -p "$ROOT/data"
if [ ! -f "$RAW" ]; then
    echo "Downloading $URL"
    if command -v curl >/dev/null; then curl -fL --retry 3 -o "$RAW.part" "$URL"; else wget -O "$RAW.part" "$URL"; fi
    mv "$RAW.part" "$RAW"
fi
[ "$(sha "$RAW")" = "$RAW_SHA256" ] || { echo "ERROR: $RAW does not have the expected sha256" >&2; exit 1; }

# SMILES column, surrounding whitespace removed, one molecule per line.
python3 - "$RAW" "$TXT" <<'PY'
import csv, sys
with open(sys.argv[1], newline="") as fh:
    rows = list(csv.reader(fh))
assert rows[0][0] == "smiles", rows[0]
with open(sys.argv[2], "w", newline="\n") as out:
    for r in rows[1:]:
        out.write(r[0].strip() + "\n")
PY
[ "$(sha "$TXT")" = "$TXT_SHA256" ] || { echo "ERROR: $TXT does not have the expected sha256" >&2; exit 1; }
echo "OK: $RAW and $TXT (sha256 verified)"
