#!/usr/bin/env python3
"""Cache a sample of encoded ZINC latents for the convex-hull support baseline
(reviewer Lhbx Q4). One-shot; writes results/precomputed/zinc_latents_sample.npz.
"""
import numpy as np
import torch

from src.codec.selfies_vae import SELFIESVAECodec
from src.oracle.guacamol import load_zinc_smiles

N = 3000
device = "cuda" if torch.cuda.is_available() else "cpu"
codec = SELFIESVAECodec.from_pretrained(device=device)
smiles = load_zinc_smiles(n_samples=10000)
rng = np.random.RandomState(0)
idx = rng.choice(len(smiles), N, replace=False)
sample = [smiles[i] for i in idx]

Z = []
B = 256
for i in range(0, N, B):
    z = codec.encode(sample[i:i + B]).cpu().numpy()
    Z.append(z)
Z = np.concatenate(Z, axis=0)
norms = np.linalg.norm(Z, axis=1)
print(f"Encoded {Z.shape} ; norm mean={norms.mean():.3f} std={norms.std():.3f} "
      f"5-95pct=[{np.percentile(norms,5):.2f},{np.percentile(norms,95):.2f}]")
out = "results/precomputed/zinc_latents_sample.npz"
np.savez_compressed(out, Z=Z.astype(np.float32),
                    mean=Z.mean(0).astype(np.float32))
print(f"Saved {out}")
