#!/usr/bin/env python
"""Full-set latent geometry of the SELFIES VAE on ZINC-250k (camera-ready, items 2.2 / 2.4).

Encodes every molecule of ``data/zinc_250k.txt`` with the repository's SELFIES VAE
(``src.codec.selfies_vae.SELFIESVAECodec``; same tokenisation as ``codec.encode``) and
stores, per molecule, the encoder mean mu(x) (256 floats), the per-dimension posterior
variance sigma^2(x) (the model returns sigma = softplus(.) + 1e-4; sigma^2 is stored) and
the line index into the SMILES file. It then writes ``geometry.json`` with

* ||mu||: mean, SD, 1/5/25/50/75/95/99 % percentiles, min, max;
* the aggregate covariance Sigma of the encoder means (ddof 0 and ddof 1): tr(Sigma),
  ||mu_bar||^2, tr(Sigma^2), mu_bar' Sigma mu_bar, lambda_max, the participation ratio
  (tr Sigma)^2 / tr(Sigma^2), the number of principal components for 50/80/90/95/99 %
  of the variance, the number of eigenvalues > 0.5, exp(spectral entropy);
* sum_i E_x sigma_i^2(x) (and the per-dimension E_x sigma_i^2(x));
* posterior samples z = mu + sigma * eps, one eps ~ N(0, I) per molecule with a fixed seed
  (eps of line i depends only on (eps_seed, i), never on batching): mean / SD /
  percentiles of ||z||, and the covariance of the z cloud;
* the model-based radius prediction for the encoder-mean cloud,
      m = tr Sigma + ||mu_bar||^2,
      Var(||z||^2) = 2 tr(Sigma^2) + 4 mu_bar' Sigma mu_bar   (Gaussian moments),
      r_pred = sqrt(m) - Var(||z||^2) / (8 m^1.5),
      width_pred = sqrt(Var(||z||^2)) / (2 sqrt(m))          (delta method),
  next to the observed mean / SD of ||mu|| and the posted in-sample formula
  sqrt(m) - sd(||mu||)^2 / (2 sqrt(m)) (the rebuttal's shell prediction);
  the same prediction for the posterior-sample cloud z;
* the same mu-cloud quantities on the 3,000-latent sample
  ``results/precomputed/zinc_latents_sample.npz`` (posted: 7.9859 predicted / 7.9846
  observed / participation ratio 63.6 / 66 PCs for 90 %), and a row-by-row check of that
  sample against the full-set encodings at the indices
  ``RandomState(0).choice(10000, 3000, replace=False)`` used by
  ``scripts/analysis/precompute_zinc_latents.py``;
* a per-molecule cross-check of the batched encoder against ``codec.encode`` on the first
  ``--check-n`` molecules (batched padding with <stop> + key-padding mask vs unpadded).

Vocabulary caveat (measured 2026-10-11): the codec's 97-token SELFIES vocabulary has no
stereo or cis/trans tokens, and ``SELFIESVAECodec._tokenize_selfies`` silently drops unknown
tokens, i.e. deletes those atoms. On the first 20,000 ZINC-250k lines 62.3 % of molecules
lose at least one token. The default reproduces exactly what the paper's numbers used
(``codec.encode``); ``--strip-stereo`` is the sensitivity variant that removes stereo with
RDKit first, after which every token is in the vocabulary.

Determinism: no stochastic ops in the encoder in eval mode; molecules are processed in a
fixed length-sorted order (stable argsort of token lengths) with a fixed batch size;
torch deterministic algorithms are requested; eps uses numpy PCG64 seeded per 4,096-line
chunk with SeedSequence([eps_seed, chunk]). Results are bit-reproducible on the same
hardware/software for the same --batch-size and --device; across devices only float
round-off differs (reported by the cross-checks).

It never writes into the repository: --outdir must lie outside --repo; bytecode writing is
disabled. Run it with ``python -B``.

Usage (GPU, full set; about 1-8 min on one GPU, see the report)::

    PYTHONDONTWRITEBYTECODE=1 python -B G_fullset_geometry.py --repo /path/to/NeurIPS \\
        --outdir /data/prusek/cr_geom_full --device cuda --batch-size 256

Smoke test (CPU, 200 molecules)::

    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= python -B G_fullset_geometry.py \\
        --repo /path/to/NeurIPS --outdir /data/prusek/cr_geom_test --device cpu --limit 200 --threads 4
"""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import os  # noqa: E402

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["GIT_OPTIONAL_LOCKS"] = "0"  # git status must not refresh the repository index

import argparse  # noqa: E402
import datetime  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import platform  # noqa: E402
import socket  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

D_LAT = 256
EPS_CHUNK = 4096
PCT = (1, 5, 25, 50, 75, 95, 99)
VAR_THRESH = (0.50, 0.80, 0.90, 0.95, 0.99)
POSTED = {"r_pred_posted": 7.9859, "r_obs_sample": 7.9846, "participation_ratio": 63.6, "n_pc_90": 66,
          "full_set_mean_norm_zinc_mean_norm_json": 7.983377, "n_encoded_zinc_mean_norm_json": 249456}


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def git(repo: Path, *args: str) -> str:
    try:
        r = subprocess.run(["git", "--no-optional-locks", "-C", str(repo), *args],
                           capture_output=True, text=True, timeout=60)
        return r.stdout.strip() if r.returncode == 0 else f"ERR({r.returncode}) {r.stderr.strip()[:200]}"
    except Exception as e:  # noqa: BLE001
        return f"ERR {type(e).__name__}: {e}"


def is_within(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


# ── statistics ───────────────────────────────────────────────────────────────

def cloud_stats(X: np.ndarray | None = None, *, mean: np.ndarray | None = None, cov0: np.ndarray | None = None,
                n: int | None = None, norms: np.ndarray | None = None) -> dict:
    """Moments, spectrum and radius predictions of a point cloud (rows = points).

    Either pass X (in memory) or the precomputed (mean, population covariance, n, norms).
    """
    if X is not None:
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        mean = X.mean(axis=0)
        Xc = X - mean
        cov0 = Xc.T @ Xc / n
        norms = np.linalg.norm(X, axis=1)
    out: dict = {"n": int(n)}
    nm = np.asarray(norms, dtype=np.float64)
    out["norm"] = {"mean": float(nm.mean()), "sd_ddof0": float(nm.std()), "sd_ddof1": float(nm.std(ddof=1)) if n > 1 else None,
                   "min": float(nm.min()), "max": float(nm.max()),
                   **{f"p{p:02d}": float(np.percentile(nm, p)) for p in PCT}}
    mu2 = float(mean @ mean)
    out["mean_vector_norm_sq"] = mu2
    out["mean_vector_norm"] = math.sqrt(mu2)
    out["mean_norm_sq_identity_check"] = {"mean_of_norm_sq": float((nm ** 2).mean()),
                                          "tr_cov_ddof0_plus_mu2": float(np.trace(cov0) + mu2)}
    for ddof in (0, 1):
        if ddof == 1 and n < 2:
            continue
        C = cov0 * (n / (n - ddof)) if ddof else cov0
        lam = np.clip(np.linalg.eigvalsh(C)[::-1], 0.0, None)
        tr, tr2 = float(lam.sum()), float((lam ** 2).sum())
        msm = float(mean @ C @ mean)
        m = tr + mu2
        var_r2 = 2.0 * tr2 + 4.0 * msm
        r_pred = math.sqrt(m) - var_r2 / (8.0 * m ** 1.5)
        width_pred = math.sqrt(var_r2) / (2.0 * math.sqrt(m))
        sd_obs = float(nm.std())  # numpy default ddof 0, as in the rebuttal's shell prediction
        r_posted_formula = math.sqrt(m) - sd_obs ** 2 / (2.0 * math.sqrt(m))
        cum = np.cumsum(lam) / tr if tr > 0 else np.zeros_like(lam)
        p = lam / tr if tr > 0 else lam
        ent = float(-(p[p > 0] * np.log(p[p > 0])).sum())
        out[f"cov_ddof{ddof}"] = {
            "tr_Sigma": tr, "tr_Sigma_sq": tr2, "mu_bar_Sigma_mu_bar": msm, "lambda_max": float(lam[0]),
            "participation_ratio": tr ** 2 / tr2 if tr2 > 0 else None,
            "n_pc": {f"{int(t * 100)}pct": int(np.searchsorted(cum, t - 1e-12) + 1) for t in VAR_THRESH},
            "cum_var_at_64_65_66": [float(cum[63]), float(cum[64]), float(cum[65])] if lam.size >= 66 else None,
            "n_eig_gt_0p5": int((lam > 0.5).sum()), "exp_spectral_entropy": math.exp(ent),
            "m_second_moment": m, "sqrt_m": math.sqrt(m), "var_norm_sq_gaussian": var_r2,
            "var_norm_sq_empirical": float((nm ** 2).var()),
            "r_pred_model": r_pred, "r_pred_model_rel_err": (r_pred - float(nm.mean())) / float(nm.mean()),
            "width_pred_model": width_pred, "width_obs_sd_ddof0": sd_obs,
            "r_pred_posted_in_sample_formula": r_posted_formula,
            "r_pred_posted_rel_err": (r_posted_formula - float(nm.mean())) / float(nm.mean()),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="NeurIPS repository root (read only)")
    ap.add_argument("--data", default=None, help="SMILES file (default <repo>/data/zinc_250k.txt)")
    ap.add_argument("--weights", default=None, help="VAE state dict (default: the codec's default path)")
    ap.add_argument("--sample-npz", default=None, help="default <repo>/results/precomputed/zinc_latents_sample.npz")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--device", default="auto", choices=("auto", "cuda", "cpu"))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = torch default)")
    ap.add_argument("--limit", type=int, default=None, help="encode only the first N molecules (smoke test)")
    ap.add_argument("--eps-seed", type=int, default=0)
    ap.add_argument("--check-n", type=int, default=64, help="molecules re-encoded one by one with codec.encode")
    ap.add_argument("--no-arrays", action="store_true", help="do not keep mu/sigma2 arrays (stats only)")
    ap.add_argument("--strip-stereo", action="store_true",
                    help="SENSITIVITY: feed RDKit MolToSmiles(isomericSmiles=False) to the codec. The codec's "
                         "97-token vocabulary has no stereo / cis-trans tokens ([C@@H1], [C@H1], [/C], ...), and "
                         "codec.encode silently DROPS unknown tokens, which deletes those atoms; the default "
                         "(off) reproduces what every paper number used")
    args = ap.parse_args()
    t0 = time.time()

    repo = Path(args.repo).resolve()
    outdir = Path(args.outdir).resolve()
    if is_within(outdir, repo):
        raise SystemExit(f"refusing to write inside the repository: {outdir}")
    outdir.mkdir(parents=True, exist_ok=True)
    data = Path(args.data).resolve() if args.data else repo / "data" / "zinc_250k.txt"
    sample_npz = Path(args.sample_npz).resolve() if args.sample_npz else repo / "results" / "precomputed" / "zinc_latents_sample.npz"
    sys.path.insert(0, str(repo))

    import torch  # noqa: E402
    import selfies as sf  # noqa: E402
    from src.codec import selfies_vae as sv  # noqa: E402

    if not is_within(Path(sv.__file__), repo):
        raise SystemExit(f"src.codec.selfies_vae imported from {sv.__file__}, not from {repo}")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    weights = Path(args.weights).resolve() if args.weights else Path(sv.DEFAULT_WEIGHTS_PATH)
    codec = sv.SELFIESVAECodec.from_pretrained(weights_path=str(weights), device=device)
    model = codec.model.eval()
    vocab2idx = codec.vocab.vocab2idx
    stop_idx = vocab2idx["<stop>"]

    meta = {
        "script": str(Path(__file__).resolve()), "script_sha256": sha256(Path(__file__).resolve()),
        "argv": sys.argv, "host": socket.gethostname(), "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
        "selfies": getattr(sf, "__version__", None), "device": device,
        "cuda_device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch_threads": torch.get_num_threads(), "batch_size": args.batch_size, "eps_seed": args.eps_seed,
        "eps_chunk": EPS_CHUNK, "repo": str(repo), "repo_head": git(repo, "rev-parse", "HEAD"),
        "repo_dirty_lines": len([x for x in git(repo, "status", "--porcelain", "--untracked-files=no").splitlines() if x.strip()]),
        "codec_module": sv.__file__, "weights": str(weights), "weights_sha256": sha256(weights),
        "data": str(data), "data_sha256": sha256(data), "sample_npz": str(sample_npz),
        "sample_npz_sha256": sha256(sample_npz) if sample_npz.exists() else None,
    }

    # ── SMILES (same alignment as load_zinc_smiles and the floor vectors) ───
    raw_lines = data.read_text().splitlines()
    smiles = [s.strip() for s in raw_lines if s.strip()]
    meta["n_raw_lines"] = len(raw_lines)
    meta["n_nonempty_lines"] = len(smiles)
    meta["n_duplicate_smiles"] = len(smiles) - len(set(smiles))
    N_all = len(smiles)
    N = min(N_all, args.limit) if args.limit else N_all
    print(f"[geom] {N_all} non-empty SMILES in {data}; encoding {N} on {device}", flush=True)

    # ── tokenisation exactly as SELFIESVAECodec.encode ─────────────────────
    t_tok = time.time()
    toks: list[list[int] | None] = []
    n_unknown_mol = 0
    n_unknown_tok = 0
    n_tok_total = 0
    fail_idx: list[int] = []
    inputs: list[str] = []          # the SMILES actually fed to the codec (needed for the cross-check)
    n_strip_changed = n_strip_fail = 0
    if args.strip_stereo:
        from rdkit import Chem, RDLogger  # noqa: E402
        RDLogger.DisableLog("rdApp.*")
    for i in range(N):
        s_in = smiles[i]
        if args.strip_stereo:
            mol = Chem.MolFromSmiles(s_in)
            if mol is None:
                n_strip_fail += 1
            else:
                s_ns = Chem.MolToSmiles(mol, isomericSmiles=False)
                n_strip_changed += int(s_ns != s_in)
                s_in = s_ns
        inputs.append(s_in)
        s_sf = codec._smiles_to_selfies(s_in)
        if s_sf is None:
            toks.append(None); fail_idx.append(i); continue
        parts = list(sf.split_selfies(s_sf))
        unk = sum(1 for p in parts if p not in vocab2idx)
        n_unknown_tok += unk
        n_tok_total += len(parts)
        if unk:
            n_unknown_mol += 1
        toks.append(codec._tokenize_selfies(s_sf).tolist())  # known tokens (+["[C]"] if none) + <stop>
    t_tok = time.time() - t_tok
    ok = np.array([i for i in range(N) if toks[i] is not None], dtype=np.int64)
    lens = np.array([len(toks[i]) for i in ok], dtype=np.int64)
    print(f"[geom] tokenised {N} in {t_tok:.1f}s; {len(fail_idx)} SELFIES failures; {n_unknown_mol} molecules with "
          f"tokens outside the vocabulary; token length min/median/max {lens.min()}/{int(np.median(lens))}/{lens.max()}", flush=True)

    # ── batched encoding in a fixed length-sorted order ─────────────────────
    M = ok.size
    if args.no_arrays:
        mu_arr = np.empty((M, D_LAT), dtype=np.float32)
        s2_arr = np.empty((M, D_LAT), dtype=np.float32)
    else:
        mu_arr = np.lib.format.open_memmap(outdir / "mu.npy", mode="w+", dtype=np.float32, shape=(M, D_LAT))
        s2_arr = np.lib.format.open_memmap(outdir / "sigma2.npy", mode="w+", dtype=np.float32, shape=(M, D_LAT))
        np.save(outdir / "index.npy", ok)
    order = np.argsort(lens, kind="stable")
    t_enc = time.time()
    with torch.inference_mode():
        for b0 in range(0, M, args.batch_size):
            rows = order[b0:b0 + args.batch_size]
            L = int(lens[rows].max())
            tk_np = np.full((rows.size, L), stop_idx, dtype=np.int64)  # pad with <stop>; the model masks all but the first
            for j, r in enumerate(rows):
                t = toks[ok[r]]
                tk_np[j, :len(t)] = t
            mu, sigma = model.encode(torch.from_numpy(tk_np).to(device))
            mu = mu.reshape(rows.size, -1).float().cpu().numpy()
            sg = sigma.reshape(rows.size, -1).double().cpu().numpy()
            mu_arr[rows] = mu
            s2_arr[rows] = (sg * sg).astype(np.float32)
            if (b0 // args.batch_size) % 100 == 0:
                print(f"[geom] encoded {min(b0 + args.batch_size, M)}/{M} ({time.time() - t_enc:.1f}s)", flush=True)
    t_enc = time.time() - t_enc
    if not args.no_arrays:
        mu_arr.flush(); s2_arr.flush()
    print(f"[geom] encoder pass {t_enc:.1f}s ({M / max(t_enc, 1e-9):.0f} mol/s)", flush=True)

    # ── cross-check vs codec.encode (unpadded, one molecule at a time) ──────
    k = int(min(args.check_n, M))
    zc = codec.encode([inputs[ok[i]] for i in range(k)]).float().cpu().numpy() if k else np.zeros((0, D_LAT))
    xchk = {"n": k, "max_abs_diff_mu": float(np.max(np.abs(zc - mu_arr[:k]))) if k else None,
            "max_abs_mu": float(np.max(np.abs(mu_arr[:k]))) if k else None}
    print(f"[geom] batched vs codec.encode on {k} molecules: max |dmu| = {xchk['max_abs_diff_mu']}", flush=True)

    # ── statistics of the encoder-mean cloud (float64, chunked, two-pass) ───
    CH = 16384
    s1 = np.zeros(D_LAT)
    for a in range(0, M, CH):
        s1 += mu_arr[a:a + CH].astype(np.float64).sum(axis=0)
    mean = s1 / M
    C0 = np.zeros((D_LAT, D_LAT))
    norms = np.empty(M)
    sig2_sum_dim = np.zeros(D_LAT)
    sig2_rowsum = np.empty(M)
    for a in range(0, M, CH):
        X = mu_arr[a:a + CH].astype(np.float64)
        Xc = X - mean
        C0 += Xc.T @ Xc
        norms[a:a + X.shape[0]] = np.linalg.norm(X, axis=1)
        S2 = s2_arr[a:a + CH].astype(np.float64)
        sig2_sum_dim += S2.sum(axis=0)
        sig2_rowsum[a:a + X.shape[0]] = S2.sum(axis=1)
    C0 /= M
    mu_stats = cloud_stats(mean=mean, cov0=C0, n=M, norms=norms)
    Esig2 = sig2_sum_dim / M
    post_var = {
        "sum_i_E_sigma2": float(Esig2.sum()),
        "E_sigma2_per_dim": [float(x) for x in Esig2],
        "n_dims_E_sigma2_gt_0p5": int((Esig2 > 0.5).sum()), "n_dims_E_sigma2_gt_0p9": int((Esig2 > 0.9).sum()),
        "n_dims_E_sigma2_lt_0p1": int((Esig2 < 0.1).sum()),
        "row_sum_sigma2": {"mean": float(sig2_rowsum.mean()), "sd": float(sig2_rowsum.std()),
                           "p05": float(np.percentile(sig2_rowsum, 5)), "p95": float(np.percentile(sig2_rowsum, 95))},
        "E_norm_z_sq_expected": float((norms ** 2).mean() + sig2_rowsum.mean()),
        "note": "E||z||^2 = E||mu||^2 + E tr diag sigma^2(x) for z = mu + sigma*eps; sqrt of it is the 'prior-like' radius",
    }

    # ── posterior samples z = mu + sigma * eps (eps of line i fixed by (seed, i)) ──
    zn = np.empty(M)
    zs1 = np.zeros(D_LAT)
    zC = np.zeros((D_LAT, D_LAT))
    line_chunks = np.unique(ok // EPS_CHUNK)
    # first pass: norms and mean; second pass: covariance (z regenerated identically)
    def eps_for_chunk(c: int) -> np.ndarray:
        g = np.random.Generator(np.random.PCG64(np.random.SeedSequence([args.eps_seed, int(c)])))
        return g.standard_normal((EPS_CHUNK, D_LAT))
    pos = {int(c): np.nonzero(ok // EPS_CHUNK == c)[0] for c in line_chunks}
    for pss in (1, 2):
        zmean = zs1 / M if pss == 2 else None
        for c in line_chunks:
            rws = pos[int(c)]
            E = eps_for_chunk(int(c))[ok[rws] - int(c) * EPS_CHUNK]
            Z = mu_arr[rws].astype(np.float64) + np.sqrt(s2_arr[rws].astype(np.float64)) * E
            if pss == 1:
                zn[rws] = np.linalg.norm(Z, axis=1)
                zs1 += Z.sum(axis=0)
            else:
                Zc = Z - zmean
                zC += Zc.T @ Zc
    zmean = zs1 / M
    zC /= M
    if not args.no_arrays:
        np.save(outdir / "mu_norm.npy", norms)
        np.save(outdir / "z_norm.npy", zn)
    z_stats = cloud_stats(mean=zmean, cov0=zC, n=M, norms=zn)
    # analytic covariance of the aggregate posterior with one sample per molecule
    Cz_analytic = C0 + np.diag(Esig2)
    z_stats["analytic_cov_prediction"] = cloud_stats(mean=mean, cov0=Cz_analytic, n=M, norms=zn)["cov_ddof0"]

    # ── 3,000-latent sample: posted numbers + row check against the full set ──
    sample = {"path": str(sample_npz), "present": sample_npz.exists()}
    if sample_npz.exists():
        Zs = np.load(sample_npz)["Z"].astype(np.float64)
        sample["stats"] = cloud_stats(Zs)
        sample["posted"] = POSTED
        idx3k = np.random.RandomState(0).choice(10_000, Zs.shape[0], replace=False)
        rowpos = {int(l): r for r, l in enumerate(ok)}
        have = [(j, rowpos[int(l)]) for j, l in enumerate(idx3k) if int(l) in rowpos]
        if have:
            jj = np.array([h[0] for h in have]); rr = np.array([h[1] for h in have])
            dd = np.abs(Zs[jj] - mu_arr[rr].astype(np.float64))
            sample["row_check_vs_full_set"] = {"n_rows_available": int(len(have)), "max_abs_diff": float(dd.max()),
                                               "median_row_max_abs_diff": float(np.median(dd.max(axis=1))),
                                               "frac_rows_max_abs_diff_lt_1e-4": float((dd.max(axis=1) < 1e-4).mean())}
            sample["posterior_terms_on_sample_rows"] = {
                "n": int(len(have)),
                "sum_i_E_sigma2": float(s2_arr[rr].astype(np.float64).sum(axis=1).mean()),
                "z_norm_mean": float(zn[rr].mean()), "z_norm_sd": float(zn[rr].std())}
        else:
            sample["row_check_vs_full_set"] = {"n_rows_available": 0,
                                               "note": "sample indices lie beyond --limit; run without --limit"}

    out = {
        "meta": {**meta, "n_encoded": int(M), "n_requested": int(N), "n_selfies_failures": len(fail_idx),
                 "selfies_failure_indices_head": fail_idx[:50], "n_molecules_with_unknown_tokens": n_unknown_mol,
                 "n_unknown_tokens_dropped": n_unknown_tok, "n_selfies_tokens_total": n_tok_total,
                 "strip_stereo": bool(args.strip_stereo), "n_smiles_changed_by_strip": n_strip_changed,
                 "n_rdkit_parse_failures_in_strip": n_strip_fail,
                 "token_length": {"min": int(lens.min()), "median": float(np.median(lens)), "max": int(lens.max())},
                 "timing_s": {"tokenise": t_tok, "encode": t_enc, "total": time.time() - t0},
                 "throughput_mol_per_s": {"tokenise": N / max(t_tok, 1e-9), "encode": M / max(t_enc, 1e-9)},
                 "extrapolated_full_set_s": (N_all * (t_tok / max(N, 1) + t_enc / max(M, 1))) if args.limit else None,
                 "finished": datetime.datetime.now().isoformat(timespec="seconds"),
                 "repo_dirty_lines_after": len([x for x in git(repo, "status", "--porcelain", "--untracked-files=no").splitlines() if x.strip()])},
        "batched_vs_codec_encode": xchk,
        "mu_cloud": mu_stats,
        "posterior_variance": post_var,
        "z_cloud": z_stats,
        "sample_3000": sample,
        "arrays": None if args.no_arrays else {
            "mu.npy": f"float32 [{M}, 256], rows aligned with index.npy",
            "sigma2.npy": f"float32 [{M}, 256], posterior variance sigma(x)^2 per dimension",
            "index.npy": "int64 [M], 0-based index of the non-empty, stripped line of --data",
            "mu_norm.npy": "float64 [M]", "z_norm.npy": "float64 [M], ||mu + sigma*eps||"},
    }
    with open(outdir / "geometry.json", "w") as f:
        json.dump(out, f, indent=1)
    c0, c1 = mu_stats["cov_ddof0"], mu_stats.get("cov_ddof1", {})
    print(f"[geom] n={M} ||mu|| mean {mu_stats['norm']['mean']:.4f} sd {mu_stats['norm']['sd_ddof0']:.4f}; "
          f"PR {c0['participation_ratio']:.2f}; PCs90 {c0['n_pc']['90pct']} PCs99 {c0['n_pc']['99pct']}; "
          f"r_pred {c0['r_pred_model']:.4f} width_pred {c0['width_pred_model']:.4f}; sum E sigma2 {post_var['sum_i_E_sigma2']:.2f}; "
          f"||z|| mean {z_stats['norm']['mean']:.4f} sd {z_stats['norm']['sd_ddof0']:.4f}", flush=True)
    if sample.get("stats"):
        s1_ = sample["stats"]["cov_ddof1"]
        print(f"[geom] sample 3000: obs {sample['stats']['norm']['mean']:.4f} posted-formula {s1_['r_pred_posted_in_sample_formula']:.4f} "
              f"model {s1_['r_pred_model']:.4f} PR {s1_['participation_ratio']:.2f} PCs90 {s1_['n_pc']['90pct']}; "
              f"row check {sample.get('row_check_vs_full_set')}", flush=True)
    print(f"[geom] wrote {outdir / 'geometry.json'} in {time.time() - t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
