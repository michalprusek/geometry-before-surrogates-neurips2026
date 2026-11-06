"""Numbers read from stored non-arm artefacts: the 99.1 % acquisition-attainment record. (The
geometry numbers moved to crlib/geometry.py, which reads the --geometry-root.)"""
from __future__ import annotations

import json
import os

from .context import Ctx
from .engine import fileset_from_paths


def _fs(paths) -> dict:
    return fileset_from_paths(paths)


def acq_numbers(ctx: Ctx) -> None:
    p = os.path.join(ctx.extra.get("inputs_root", ""), "precomputed", "acq_attainment.json")
    if not os.path.exists(p):
        ctx.skip("numbers_acq", "acq_attainment.json", "input absent")
        return
    d = json.load(open(p))
    fs = _fs([p])
    n = int(d["n_steps"])
    wins = int(round(d["frac_continuous_beats_argmax"] * n))
    ctx.num("crAcqPct", 100 * d["frac_continuous_beats_argmax"], f"{100 * d['frac_continuous_beats_argmax']:.1f}",
            kind="percent", table="numbers_acq", fileset=fs, n=n,
            rule="steps where the continuous maximiser attains a higher acquisition value than a 2,048-candidate argmax "
                 "(9 instrumented runs: adip, dhop, fexo x seeds 42-44, July 2026)")
    ctx.num("crAcqSteps", n, str(n), kind="n", table="numbers_acq", fileset=fs, n=n, rule="instrumented steps")
    ctx.num("crAcqWins", wins, str(wins), kind="count", table="numbers_acq", fileset=fs, n=n, rule="steps won")
