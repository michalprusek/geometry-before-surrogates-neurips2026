#!/usr/bin/env python
"""Train PowerSpherical VAE on ZINC molecules using the same recipe as spcauchy_v3.

Follows the spCauchy VAE trainer (not part of this release) with three
changes targeted at the diagnosed failure modes of the current spCauchy
checkpoint:

1. **β = 1.0 with linear warmup 0 → 1 over 20 epochs** (was 0.0002 max).
   The previous 500-epoch run had effectively zero pressure toward the
   uniform-on-sphere prior, allowing the posterior to over-concentrate on
   data clusters (1.47× theoretical angular spread, bimodal roundtrip drift).

2. **SMILES enumeration augmentation** (5× per epoch via RDKit randomization).
   Each canonical molecule yields 5 random equivalent SMILES strings, forcing
   the decoder to map each underlying molecule consistently from any of its
   surface forms; this is the direct fix for the 33% bimodal drift.

3. **Online angular-uniformity diagnostic** logged every epoch alongside ELBO.
   Tracks the running estimate of pairwise cos-similarity std on a 256-SMILES
   probe set; the target is to converge to ``1/√D ≈ 0.0625`` (ratio 1.0).

Usage::

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python \\
        scripts/train_power_spherical_vae.py --epochs 100 --batch-size 256 \\
        --output-dir <output dir>
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import selfies as sf
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.codec.selfies_vae import _SELFIESVocab
from src.codec.power_spherical_vae import PowerSphericalTransformerVAE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Dataset with SMILES enumeration augmentation
# ---------------------------------------------------------------------------


class AugmentedSELFIESDataset(Dataset):
    """SELFIES dataset with optional RDKit-based SMILES randomization.

    For each underlying molecule we precompute one canonical token sequence
    plus ``aug_factor - 1`` randomly enumerated alternatives. At ``__getitem__``
    we draw uniformly across the augmentations to expose the decoder to many
    equivalent surface forms of the same molecule (target: lower bimodal
    roundtrip drift via consistency).
    """

    def __init__(
        self,
        smiles_file: str,
        vocab: _SELFIESVocab,
        max_len: int = 256,
        aug_factor: int = 1,
    ):
        self.vocab = vocab
        self.max_len = max_len
        self.aug_factor = max(1, int(aug_factor))

        logger.info(f"Loading SMILES from {smiles_file}...")
        with open(smiles_file) as f:
            smiles_list = [line.strip() for line in f if line.strip()]
        logger.info(f"Read {len(smiles_list)} SMILES strings")

        # If aug_factor > 1, generate variants; else just canonical.
        if self.aug_factor > 1:
            from rdkit import Chem
            from rdkit import RDLogger
            RDLogger.DisableLog("rdApp.*")

        n_valid = 0
        n_failed_selfies = 0
        n_failed_rdkit = 0
        self.samples: list[list[str]] = []  # tokens per sample
        for smi in smiles_list:
            variants = [smi]
            if self.aug_factor > 1:
                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None:
                        n_failed_rdkit += 1
                    else:
                        for _ in range(self.aug_factor - 1):
                            variants.append(
                                Chem.MolToSmiles(mol, doRandom=True, canonical=False)
                            )
                except Exception:
                    n_failed_rdkit += 1
            for v in variants:
                try:
                    selfies_str = sf.encoder(v)
                    if selfies_str is None:
                        n_failed_selfies += 1
                        continue
                    tokens = list(sf.split_selfies(selfies_str))
                    known = [t for t in tokens if t in self.vocab.vocab2idx]
                    if not known:
                        n_failed_selfies += 1
                        continue
                    self.samples.append(known)
                    n_valid += 1
                except Exception:
                    n_failed_selfies += 1

        logger.info(
            f"Tokenized {n_valid} samples (aug_factor={self.aug_factor}), "
            f"{n_failed_selfies} SELFIES errors, {n_failed_rdkit} RDKit errors"
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.vocab.encode(self.samples[idx])


def collate_fn(batch):
    max_len = max(x.shape[0] for x in batch)
    return torch.stack(
        [F.pad(x, (0, max_len - x.shape[0]), value=1) for x in batch]
    )


# ---------------------------------------------------------------------------
# Online angular diagnostic
# ---------------------------------------------------------------------------


@torch.no_grad()
def angular_diagnostic(model, probe_tokens: torch.Tensor) -> dict:
    """Encode the probe batch, compute angular spread + sphere stats.

    Returns ratios + raw values in the form of the offline codec-geometry
    diagnostic (not part of this release). Cheap (one forward pass).
    """
    model.eval()
    device = next(model.parameters()).device
    probe_tokens = probe_tokens.to(device)
    out = model(probe_tokens)
    z_flat = out["z_flat"].detach()        # [B, D] on the sphere
    Z = z_flat.cpu().double().numpy()
    D = Z.shape[1]

    norms = np.linalg.norm(Z, axis=1)
    mean_norm = float(norms.mean())
    std_norm = float(norms.std())

    U = Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(1e-9)
    sim = U @ U.T
    np.fill_diagonal(sim, np.nan)
    sim_off = sim[~np.isnan(sim)]
    angular_std = float(np.nanstd(sim_off))
    theory = 1.0 / math.sqrt(D)
    angular_ratio = angular_std / theory

    Zc = Z - Z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Zc, compute_uv=False)
    pr = float((s ** 2).sum() ** 2 / (s ** 4).sum())

    sigma_coord = Z.std(axis=0)
    n_collapsed = int((sigma_coord < 0.05 * sigma_coord.mean()).sum())

    return {
        "mean_norm": mean_norm, "std_norm": std_norm,
        "shell_concentration": std_norm / mean_norm if mean_norm > 0 else float("nan"),
        "angular_std": angular_std, "angular_ratio_to_uniform": angular_ratio,
        "participation_ratio": pr, "n_collapsed_coords": n_collapsed,
    }


# ---------------------------------------------------------------------------
# Train / validate
# ---------------------------------------------------------------------------


def train_epoch(model, loader, optimizer, device, epoch, log_interval=200):
    model.train()
    totals = {"loss": 0, "recon_loss": 0, "kldiv": 0, "token_acc": 0, "rho_mean": 0}
    n_batches = 0
    for batch_idx, tokens in enumerate(loader):
        tokens = tokens.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        result = model(tokens)
        loss = result["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        for k in totals:
            totals[k] += result[k].item() if k != "token_acc" else result["recon_token_acc"].item()
        n_batches += 1
        if (batch_idx + 1) % log_interval == 0:
            logger.info(
                f"  ep{epoch} [{batch_idx+1}/{len(loader)}] "
                f"loss={result['loss'].item():.4f} recon={result['recon_loss'].item():.4f} "
                f"kl={result['kldiv'].item():.4f} rho={result['rho_mean'].item():.4f}"
            )
    return {k: v / n_batches for k, v in totals.items()}


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    totals = {"loss": 0, "recon_loss": 0, "kldiv": 0, "token_acc": 0,
              "string_acc": 0, "rho_mean": 0}
    n_batches = 0
    for tokens in loader:
        tokens = tokens.to(device, non_blocking=True)
        result = model(tokens)
        totals["loss"] += result["loss"].item()
        totals["recon_loss"] += result["recon_loss"].item()
        totals["kldiv"] += result["kldiv"].item()
        totals["token_acc"] += result["recon_token_acc"].item()
        totals["string_acc"] += result["recon_string_acc"].item()
        totals["rho_mean"] += result["rho_mean"].item()
        n_batches += 1
    return {k: v / n_batches for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="Retrain spCauchy VAE with codec-geometry recipe")

    p.add_argument("--train-data", default=str(PROJECT_ROOT / "data" / "zinc_250k.txt"))
    p.add_argument("--val-split", type=float, default=0.05)
    p.add_argument("--aug-factor", type=int, default=5,
                   help="SMILES enumeration factor (5x ⇒ 5 random SMILES per canonical)")
    p.add_argument("--bottleneck-size", type=int, default=2)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--kl-factor", type=float, default=1.0,
                   help="Target β after warmup (default 1.0; previous run used 0.0002)")
    p.add_argument("--kl-warmup-epochs", type=int, default=20)
    p.add_argument("--use-mc-kl", action="store_true")
    p.add_argument("--free-bits", type=float, default=0.0,
                   help="Per-sample KL floor (nat). Prevents posterior collapse "
                        "to rho=0. Typical 5-20 for d=256. 0 disables (legacy).")

    p.add_argument("--init-from-gaussian", action="store_true", default=True)
    p.add_argument("--gaussian-weights",
                   default=str(PROJECT_ROOT / "weights" / "SELFIES-VAE-state-dict.pt"))

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=4)

    p.add_argument("--output-dir", default="/path/to/power_spherical_vae_v1")
    p.add_argument("--probe-size", type=int, default=256,
                   help="Number of validation tokens for online angular diagnostic")
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--resume-from", default=None,
                   help="Path to checkpoint to resume from (loads model+optimizer state, "
                        "continues epoch counter and history). Survives server restarts.")

    args = p.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    vocab = _SELFIESVocab()
    dataset = AugmentedSELFIESDataset(
        args.train_data, vocab, aug_factor=args.aug_factor,
    )
    n_val = int(len(dataset) * args.val_split)
    n_train = len(dataset) - n_val
    train_set, val_set = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True,
    )
    logger.info(f"Train: {n_train}, Val: {n_val}")

    # Probe set for the online angular diagnostic — fixed across epochs
    # so the time series is comparable.
    probe_tokens = next(iter(DataLoader(
        val_set, batch_size=args.probe_size, shuffle=False, collate_fn=collate_fn,
    )))

    # Model
    model = PowerSphericalTransformerVAE(
        vocab_size=len(vocab),
        bottleneck_size=args.bottleneck_size,
        d_model=args.d_model,
        kl_factor=args.kl_factor,
        use_analytic_kl=not args.use_mc_kl,
        free_bits=args.free_bits,
    )
    model.max_string_length = 1024
    if args.init_from_gaussian:
        gpath = Path(args.gaussian_weights)
        if gpath.exists():
            model.init_from_gaussian_vae(
                torch.load(gpath, map_location="cpu", weights_only=False),
            )
            logger.info(f"Initialized from Gaussian VAE: {gpath}")
    model = model.to(device)
    logger.info(f"Model: {sum(p.numel() for p in model.parameters())} params, "
                f"latent_dim={model.latent_dim}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6,
    )

    # Resume support: load checkpoint + history if provided. Survives server
    # restarts. The resume path uses the saved epoch as starting point and
    # reloads the existing training_history.json from output_dir if present.
    start_epoch = 1
    best_val_loss = float("inf")
    no_improve = 0
    history = []
    if args.resume_from:
        resume_path = Path(args.resume_from)
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        # Load existing history so we keep the trajectory continuous.
        hist_path = output_dir / "training_history.json"
        if hist_path.exists():
            history = json.loads(hist_path.read_text())
            if history:
                # Truncate any history entries past the resume epoch (in case
                # we saved more after the saved checkpoint).
                history = [h for h in history if h["epoch"] < start_epoch]
            best_val_loss = min(
                (h["val_loss"] for h in history if "val_loss" in h),
                default=float("inf"),
            )
        # Step the LR scheduler forward to match the resume epoch.
        for _ in range(start_epoch - 1):
            scheduler.step()
        logger.info(
            f"RESUMED from {resume_path} at epoch {start_epoch} "
            f"(best_val_loss={best_val_loss:.4f}, history_len={len(history)})"
        )
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        # KL warmup: linear 0 → kl_factor over kl_warmup_epochs.
        beta = args.kl_factor * min(1.0, epoch / max(1, args.kl_warmup_epochs))
        model.kl_factor = beta

        train_m = train_epoch(model, train_loader, optimizer, device, epoch, args.log_interval)
        val_m = validate(model, val_loader, device)
        diag = angular_diagnostic(model, probe_tokens)
        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            f"E{epoch:>3d}/{args.epochs} ({elapsed:.0f}s) "
            f"β={beta:.3f} lr={optimizer.param_groups[0]['lr']:.2e} "
            f"| val: loss={val_m['loss']:.4f} recon={val_m['recon_loss']:.4f} "
            f"str_acc={val_m['string_acc']:.3f} rho={val_m['rho_mean']:.3f} "
            f"| GEOM: PR={diag['participation_ratio']:.1f} "
            f"angular_ratio={diag['angular_ratio_to_uniform']:.3f} "
            f"shell={diag['shell_concentration']:.4f} "
            f"collapsed={diag['n_collapsed_coords']}"
        )

        history.append({
            "epoch": epoch, "beta": beta,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"val_{k}": v for k, v in val_m.items()},
            **{f"diag_{k}": v for k, v in diag.items()},
        })

        is_best = val_m["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_m["loss"]
            no_improve = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss, "val_metrics": val_m,
                "diag": diag, "args": vars(args),
            }, output_dir / "best_model.pt")
            logger.info(f"  ** new best val_loss={best_val_loss:.4f}")
        else:
            no_improve += 1
            if no_improve >= args.patience:
                logger.info(f"Early stop after {args.patience} no-improve epochs")
                break

        if epoch % 10 == 0:
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "diag": diag},
                       output_dir / f"checkpoint_epoch{epoch}.pt")

        # Persist history every epoch so a kill mid-training keeps usable trace.
        (output_dir / "training_history.json").write_text(json.dumps(history, indent=2))

    # Final save -- guard against resume-with-no-epochs-to-run case where the
    # for-loop body never executes (start_epoch > args.epochs).
    if "val_m" in dir() and "diag" in dir():
        torch.save({
            "epoch": epoch, "model_state_dict": model.state_dict(),
            "val_metrics": val_m, "diag": diag, "args": vars(args),
        }, output_dir / "final_model.pt")
        logger.info(f"Done. Best val_loss={best_val_loss:.4f}. Outputs in {output_dir}")
    else:
        logger.warning(
            f"No epochs ran (start={start_epoch}, end={args.epochs}); skipping final save."
        )


if __name__ == "__main__":
    main()
