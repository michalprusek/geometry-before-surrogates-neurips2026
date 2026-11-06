#!/usr/bin/env python3
"""Did the on-shell continuous optimizer actually maximise the acquisition?

Two very different worlds produce the same benchmark number for the
shell/continuous cell of the factorial:

  (A) Adam failed to climb, so the cell measures a WORSE argmax. Then a low
      score is a bug, not a finding, and the cell must be re-run with a proper
      Riemannian trust region before it means anything.
  (B) Adam climbed higher than the 2048-sample argmax and the objective still
      got worse. Then the acquisition surface is anti-correlated with outcome at
      N=30 -- consistent with the LOO R^2 ~ 0 we already report -- and the
      finding is about the acquisition, not about the support.

Nothing in a best-so-far trace distinguishes these. This script measures the
achieved qLogEI directly, on the same GP, at every refinement step.

Usage:
    PYTHONPATH=/path/to/NeurIPS python scripts/analysis/rebuttal_acq_attainment.py \
        --tasks adip dhop --seeds 42 43 44 --out results/precomputed/acq_attainment.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os

import numpy as np
import torch
from botorch.acquisition import qLogExpectedImprovement
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)


def _instrument(solver):
    """Wrap the solver so each refinement step records achieved acquisition."""
    records = []
    orig = solver._maximize_on_sphere

    def traced(gp, best_y, D):
        acqf = qLogExpectedImprovement(model=gp, best_f=best_y)
        dev = solver._device
        # The competing arm, reproduced exactly: 2048 uniform on-shell draws,
        # take the argmax. Same GP, same acquisition, same shell.
        u = torch.randn(2048, D, dtype=torch.float64, device=dev)
        u = u / u.norm(dim=-1, keepdim=True) * solver.mean_norm
        with torch.no_grad():
            argmax_best = float(acqf(u.unsqueeze(1)).max())

        cand = orig(gp, best_y, D)
        with torch.no_grad():
            got = float(acqf(torch.tensor(
                cand, dtype=torch.float64, device=dev).unsqueeze(1)))
        records.append({"continuous": got, "argmax2048": argmax_best})
        return cand

    solver._maximize_on_sphere = traced
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["adip", "dhop"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    ap.add_argument("--out", default="results/precomputed/acq_attainment.json")
    args = ap.parse_args()

    from scripts.run_benchmark import generate_cold_start_ours, make_our_latent_bb
    from src.analysis.config import GUACAMOL_TO_POLI
    from src.oracle.guacamol import GuacaMolOracle
    from src.spherevo.gp_competitors import HvarfnerScaledGPSphereContinuousSolver

    all_recs = []
    for task in args.tasks:
        bb, codec, _ = make_our_latent_bb(GUACAMOL_TO_POLI[task])
        oracle = GuacaMolOracle(task)
        for seed in args.seeds:
            x0, y0, _ = generate_cold_start_ours(codec, oracle.score, 5, seed)
            s = HvarfnerScaledGPSphereContinuousSolver(black_box=bb, x0=x0, y0=y0)
            recs = _instrument(s)
            s.solve(max_iter=25)
            for r in recs:
                r.update(task=task, seed=seed)
            all_recs.extend(recs)
            logger.info(f"{task} s{seed}: {len(recs)} steps")

    cont = np.array([r["continuous"] for r in all_recs])
    amax = np.array([r["argmax2048"] for r in all_recs])
    frac = float((cont > amax).mean())
    summary = {
        "n_steps": len(all_recs),
        "frac_continuous_beats_argmax": frac,
        "mean_continuous": float(cont.mean()),
        "mean_argmax2048": float(amax.mean()),
        "mean_gap": float((cont - amax).mean()),
        "median_gap": float(np.median(cont - amax)),
        # If this is well below 1.0 the cell measures optimizer failure (world A)
        # and its benchmark number must not be read as an acquisition-optimizer
        # effect. Near 1.0 puts us in world B.
        "verdict": "B: optimizer climbs higher" if frac > 0.9 else (
            "A: optimizer FAILS to beat plain argmax" if frac < 0.6 else "mixed"),
        "records": all_recs,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    for k, v in summary.items():
        if k != "records":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
