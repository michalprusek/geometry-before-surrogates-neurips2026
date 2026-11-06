#!/usr/bin/env python3
"""Batch runner for the GeoWalk-family rebuttal sweeps (radius suite + convex
hull + ellipsoid). Loads the VAE codec ONCE per task and loops over
(variant, seed) in-process, which is ~10x faster than one subprocess per run
under GPU contention. Output JSONs are format-compatible with the existing
analysis loaders (best_trace / best_score / seed / task).

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/run_geowalk_family_batch.py \
        --solvers geowalk_r4,geowalk_r8 --seeds 42-51 --outdir geowalk_ablations
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.run_benchmark import (
    _create_solver, generate_cold_start_ours, make_our_latent_bb,
)
from src.analysis.config import GUACAMOL_TO_POLI, POLI_TO_GUACAMOL
from src.oracle.guacamol import GuacaMolOracle
from src.provenance import collect_provenance

TASKS = ["adip", "dhop", "fexo", "med1", "med2", "osmb",
         "pdop", "rano", "shop", "siga", "zale"]  # 11 non-degenerate


def parse_seeds(s):
    a, b = s.split("-")
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--solvers", required=True, help="comma-separated solver names")
    ap.add_argument("--tasks", default=",".join(TASKS),
                    help="comma-separated task ids (short GuacaMol ids like 'med1' "
                         "or PMO/extra ids like 'albuterol_similarity'); "
                         "default = the 11 non-degenerate GuacaMol tasks")
    ap.add_argument("--seeds", default="42-51")
    ap.add_argument("--outdir", default="geowalk_ablations",
                    help="subdir under results/guacamol/")
    ap.add_argument("--max-iter", type=int, default=25)
    ap.add_argument("--n-cold-start", type=int, default=5)
    args = ap.parse_args()

    solvers = args.solvers.split(",")
    tasks = args.tasks.split(",")
    seeds = parse_seeds(args.seeds)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outdir = Path("results/guacamol") / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    total = len(tasks) * len(solvers) * len(seeds)
    done = 0
    t_start = time.time()
    for task in tasks:
        # Tasks outside the 12 GuacaMol MPO set (PMO_EXTRA_TASKS, logp, qed)
        # have no poli alias; their id doubles as the filename token.
        poli_task = GUACAMOL_TO_POLI.get(task, task)
        # Codec + black box (oracle bound inside) loaded ONCE per task.
        bb, codec, dim = make_our_latent_bb(poli_task, device)
        oracle = GuacaMolOracle(task_id=task)
        for seed in seeds:
            np.random.seed(seed)
            torch.manual_seed(seed)
            z0, y0, _ = generate_cold_start_ours(
                codec, oracle.score, args.n_cold_start, seed, device
            )
            for solver_name in solvers:
                done += 1
                fname = f"ours_{solver_name}_{poli_task}_s{seed}_256d.json"
                fpath = outdir / fname
                # Skip if a complete run already exists (reuse prior seeds).
                if fpath.exists():
                    try:
                        prev = json.loads(fpath.read_text())
                        if len(prev.get("best_trace", [])) >= args.max_iter + args.n_cold_start:
                            print(f"[{done}/{total}] {solver_name} {task} s{seed} SKIP (exists)",
                                  flush=True)
                            continue
                    except (json.JSONDecodeError, OSError):
                        pass
                # Re-seed per solver so each variant sees the same RNG stream
                # as if launched independently (matches per-subprocess runs).
                np.random.seed(seed)
                torch.manual_seed(seed)
                solver = _create_solver(
                    solver_name=solver_name, black_box=bb, x0=z0, y0=y0,
                    n_dimensions=dim, subspace_dim=16, n_subspaces=16,
                    seed=seed, n_initial_points=args.n_cold_start, codec=codec,
                    norm_mode="fixed", n_iterations=args.max_iter, cs_smiles=None,
                )
                t0 = time.time()
                status = "completed"
                try:
                    solver.solve(max_iter=args.max_iter, verbose=False)
                except Exception as e:  # noqa: BLE001
                    status = f"crashed: {type(e).__name__}: {e}"
                all_scores = [float(s) for s in solver.all_scores]
                best_trace = np.maximum.accumulate(
                    np.nan_to_num(all_scores, nan=-1.0)
                ).tolist()
                rec = {
                    "mode": "ours", "solver": solver_name, "task": poli_task,
                    "our_task_id": task, "latent_dim": dim, "seed": seed,
                    "max_iter": args.max_iter, "n_initial_points": args.n_cold_start,
                    "n_evaluated": len(all_scores),
                    "best_score": float(np.nanmax(all_scores)),
                    "best_trace": best_trace, "all_scores": all_scores,
                    "status": status, "elapsed_seconds": time.time() - t0,
                    "surrogate_fallback_count": getattr(solver, "_fallback_count", 0),
                    # Every result file carries the code state that produced it
                    # (commit, resolved package versions, host) -- the batch
                    # path must match the per-subprocess path here.
                    "provenance": collect_provenance(),
                }
                fpath.write_text(json.dumps(rec))
                rate = (time.time() - t_start) / done
                eta = rate * (total - done) / 3600.0
                print(f"[{done}/{total}] {solver_name:16s} {task} s{seed} "
                      f"best={rec['best_score']:.4f} {rec['elapsed_seconds']:.1f}s "
                      f"ETA={eta:.1f}h", flush=True)
    print(f"DONE {done}/{total} in {(time.time()-t_start)/3600:.2f}h")


if __name__ == "__main__":
    main()
