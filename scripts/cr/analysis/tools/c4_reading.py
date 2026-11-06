#!/usr/bin/env python3
"""Control C4 (L-BFGS-B on the shell), read as the run rules fixed on 2026-10-14, before any result was seen.

The control replaces projected Adam by BoTorch's L-BFGS-B in the shell x continuous cell of the factorial
experiment of Table 2 (arm ``hvarfner_scaled_gp_sphere_lbfgs``, result set ``cr2026ctl``). The reading was
run on 2026-11-04 at the authors' request, after every final run had been read (RUN_RULES row 2026-11-04);
the reading itself (contrasts, test, thresholds) is the one of RUN_RULES row 2026-10-14, written before the
control ran. The printed factorial is unchanged.

Conventions of the generator: best@30 = best_trace[29] of the completed cells (n_evaluated >= 30) of the 11
protocol tasks; a level is the mean of per-task means; the task-level test is the exact two-sided Wilcoxon
signed-rank test on the 11 per-task mean differences (zero differences dropped); the cell-level test is the
Wilcoxon signed-rank test with |diff| <= 1e-6 as a tie; Cohen's d = mean / sd (ddof=1) of the cell
differences; W/L excludes ties; the share of distinct scores is the mean over cells of the share of distinct
values among the 30 scores. The first block reproduces the printed factorial as a check of the conventions.

    python scripts/cr/analysis/tools/c4_reading.py --bundle <results_bundle> [--markdown out.md]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

import numpy as np
from scipy.stats import wilcoxon

ARMS = {  # cell -> directory inside the bundle
    "SA": "cr2026/hvarfner_scaled_gp",                       # shell x candidate-set argmax
    "SC": "cr2026/hvarfner_scaled_gp_sphere_continuous",     # shell x projected Adam (printed)
    "SL": "cr2026ctl/hvarfner_scaled_gp_sphere_lbfgs",       # shell x L-BFGS-B (control C4)
    "BA": "cr2026/hvarfner_scaled_gp_box_argmax",            # box x candidate-set argmax
    "BC": "cr2026/hvarfner_scaled_gp_box_faithful",          # box x L-BFGS-B (printed)
}
LABEL = {"SA": "shell, candidate-set argmax", "SC": "shell, projected Adam", "SL": "shell, L-BFGS-B (C4)",
         "BA": "box, candidate-set argmax", "BC": "box, L-BFGS-B"}
TASKS = {"amlodipine_mpo", "fexofenadine_mpo", "osimetrinib_mpo", "perindopril_mpo", "ranolazine_mpo",
         "sitagliptin_mpo", "zaleplon_mpo", "deco_hop", "scaffold_hop", "median_1", "median_2"}
TIE_TOL = 1e-6
PRINTED = {  # Table 2 and numbers.tex of the paper (run 20261102_final2), for the convention check
    "SA": 0.4228, "SC": 0.3883, "BA": 0.3925, "BC": 0.3989,
    "support_cand": (+0.0303, 9, 0.014), "support_cont": (-0.0106, 5, 0.32),
    "acq_shell": (+0.0345, 9, 0.014), "acq_box": (-0.0064, 1, 0.0049), "interaction": (+0.0409, 10, 0.0029),
    "distinct_SC": 70.4, "distinct_other": (92.9, 99.3),
}


def load(bundle: str, arm: str) -> dict:
    """(task, seed) -> dict(best, distinct, fallback) for the completed cells of the 11 protocol tasks."""
    out = {}
    for fn in glob.glob(os.path.join(bundle, ARMS[arm], "*.json")):
        d = json.load(open(fn))
        if d["task"] not in TASKS or d.get("status", "completed") != "completed" or d.get("n_evaluated", 30) < 30:
            continue
        tr = d.get("best_trace") or []
        sc = [float(x) for x in (d.get("all_scores") or [])[:30]]
        out[(d["task"], d["seed"])] = dict(
            best=float(tr[29] if len(tr) >= 30 else d["best_score"]),
            distinct=len(set(sc)) / 30.0,
            fallback=int(d.get("surrogate_fallback_count") or 0))
    return out


def task_means(vals: dict, cells) -> dict:
    g = collections.defaultdict(list)
    for c in cells:
        g[c[0]].append(vals[c])
    return {t: float(np.mean(v)) for t, v in sorted(g.items())}


def level(vals: dict, cells) -> float:
    return float(np.mean(list(task_means(vals, cells).values())))


def p_task(td) -> float:
    d = np.asarray(td, dtype=float)
    d = d[d != 0]
    return float(wilcoxon(d, method="exact").pvalue) if d.size >= 2 else float("nan")


def p_cell(diffs) -> float:
    d = np.asarray(diffs, dtype=float).copy()
    d[np.abs(d) <= TIE_TOL] = 0.0
    return 1.0 if not np.any(d != 0) else float(wilcoxon(d).pvalue)


def contrast(name: str, diff_cells: dict, cells) -> dict:
    d = np.asarray([diff_cells[c] for c in cells], dtype=float)
    tm = task_means(diff_cells, cells)
    td = list(tm.values())
    w, l = int(np.sum(d > TIE_TOL)), int(np.sum(d < -TIE_TOL))
    return dict(name=name, n=len(cells), delta=float(np.mean(td)), tasks=sum(1 for x in td if x > 0),
                p_task=p_task(td), d=float(d.mean() / d.std(ddof=1)), p_cell=p_cell(d), wl=f"{w}/{l}",
                task_diffs=tm)


def bh(ps: list) -> list:
    order = np.argsort(ps)
    m = len(ps)
    out = [0.0] * m
    prev = 1.0
    for rank in range(m - 1, -1, -1):
        i = int(order[rank])
        prev = min(prev, ps[i] * m / (rank + 1))
        out[i] = prev
    return out


def row(r: dict, q: float | None = None) -> str:
    s = f"| {r['name']} | {r['n']} | {r['delta']:+.4f} | {r['tasks']}/11 | {r['p_task']:.4g} |"
    if q is not None:
        s += f" {q:.3g} |"
    return s + f" {r['d']:+.2f} | {r['p_cell']:.2g} | {r['wl']} |"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bundle", required=True, help="the unpacked results bundle")
    ap.add_argument("--markdown", help="write the record here as well")
    a = ap.parse_args()
    data = {k: load(a.bundle, k) for k in ARMS}
    best = {k: {c: v["best"] for c, v in data[k].items()} for k in ARMS}
    four = sorted(set(data["SA"]) & set(data["SC"]) & set(data["BA"]) & set(data["BC"]))
    five = sorted(set(four) & set(data["SL"]))
    lines: list[str] = []

    def P(s: str = "") -> None:
        print(s)
        lines.append(s)

    def diffs(x: str, y: str, cells) -> dict:
        return {c: best[x][c] - best[y][c] for c in cells}

    P("# Control C4: L-BFGS-B on the shell, read as the run rules of 2026-10-14 prescribe")
    P()
    P("Read on 2026-11-04 at the authors' request, after every final run had been read (RUN_RULES row 2026-11-04). "
      "The reading, its contrasts, test and thresholds are those of RUN_RULES row 2026-10-14, fixed before the "
      "control ran. The printed factorial (Table 2) is unchanged; this record is the only place of the result.")
    P()
    P("## Cells")
    P()
    for k in ARMS:
        P(f"- {LABEL[k]}: {len(data[k])} completed cells, {sum(v['fallback'] for v in data[k].values())} fallback step(s)")
    P(f"- matched set of the printed factorial: {len(four)} cells; with the control: {len(five)} cells")
    P()
    P("## Check: the printed factorial, as Table 2 prints it")
    P()
    P("| Cell | Level | Printed |")
    P("|---|---|---|")
    ok_all = True
    for k in ("SA", "SC", "BA", "BC"):
        lv = level(best[k], four)
        ok = abs(lv - PRINTED[k]) < 5e-5
        ok_all &= ok
        P(f"| {LABEL[k]} | {lv:.4f} | {PRINTED[k]:.4f} {'' if ok else '(MISMATCH)'} |")
    P()
    printed = {
        "support_cand": contrast("shell minus box, candidate-set argmax", diffs("SA", "BA", four), four),
        "support_cont": contrast("shell (Adam) minus box (L-BFGS-B)", diffs("SC", "BC", four), four),
        "acq_shell": contrast("candidate-set argmax minus projected Adam, shell", diffs("SA", "SC", four), four),
        "acq_box": contrast("candidate-set argmax minus L-BFGS-B, box", diffs("BA", "BC", four), four),
        "interaction": contrast("interaction I_A (Adam on the shell)",
                                {c: (best["SA"][c] - best["BA"][c]) - (best["SC"][c] - best["BC"][c]) for c in four}, four),
    }
    P("| Contrast | n | Delta | Tasks | p_task | d | p_cell | W/L | Printed (Delta, tasks, p_task) |")
    P("|---|---|---|---|---|---|---|---|---|")
    for key, r in printed.items():
        pd, pt, pp = PRINTED[key]
        ok = abs(r["delta"] - pd) < 5e-5 and r["tasks"] == pt and abs(r["p_task"] - pp) < 2e-3
        ok_all &= ok
        P(row(r) + f" {pd:+.4f}, {pt}/11, {pp} {'' if ok else '(MISMATCH)'} |")
    dist = {k: 100 * float(np.mean([data[k][c]["distinct"] for c in four])) for k in ARMS if k != "SL"}
    rest = [v for k, v in dist.items() if k != "SC"]
    P()
    P(f"Share of distinct scores, printed factorial: shell Adam {dist['SC']:.1f} % (printed {PRINTED['distinct_SC']}), "
      f"other cells {min(rest):.1f} to {max(rest):.1f} % (printed {PRINTED['distinct_other'][0]} to {PRINTED['distinct_other'][1]}).")
    P(f"Convention check: {'passed' if ok_all else 'FAILED'}.")
    P()
    P("## The control on the matched set of all five cells")
    P()
    P("| Cell | Level | Distinct scores |")
    P("|---|---|---|")
    for k in ("SA", "SL", "BC", "BA", "SC"):
        P(f"| {LABEL[k]} | {level(best[k], five):.4f} | {100 * float(np.mean([data[k][c]['distinct'] for c in five])):.1f} % |")
    P()
    P("### Pre-committed reading (RUN_RULES 2026-10-14): the interaction with L-BFGS-B in place of Adam")
    P()
    rA = contrast("I_A: (shell argmax - box argmax) - (shell Adam - box L-BFGS-B)",
                  {c: (best["SA"][c] - best["BA"][c]) - (best["SC"][c] - best["BC"][c]) for c in five}, five)
    rL = contrast("I_L: (shell argmax - box argmax) - (shell L-BFGS-B - box L-BFGS-B)",
                  {c: (best["SA"][c] - best["BA"][c]) - (best["SL"][c] - best["BC"][c]) for c in five}, five)
    P("| Interaction | n | Delta | Tasks > 0 | p_task | d | p_cell | W/L |")
    P("|---|---|---|---|---|---|---|---|")
    P(row(rA))
    P(row(rL))
    same = sum(1 for t in rA["task_diffs"] if (rA["task_diffs"][t] > 0) == (rL["task_diffs"][t] > 0))
    passes = same >= 8 and rL["p_task"] < 0.05
    P()
    P(f"I_L has the sign of I_A on {same} of the 11 tasks, and its task-level p is {rL['p_task']:.4g}. "
      f"The rule asks for at least 8 of 11 and p < 0.05: the control "
      f"{'passes, so the interaction does not depend on the optimiser of the shell continuous cell' if passes else 'fails, so the interaction is established only with projected Adam on the shell'}.")
    P()
    P("| Task | I_A | I_L |")
    P("|---|---|---|")
    for t in rA["task_diffs"]:
        P(f"| {t} | {rA['task_diffs'][t]:+.4f} | {rL['task_diffs'][t]:+.4f} |")
    P()
    P("### Descriptive contrasts of the rule (outside the Benjamini-Hochberg family)")
    P()
    P("| Contrast | n | Delta | Tasks | p_task | d | p_cell | W/L |")
    P("|---|---|---|---|---|---|---|---|")
    for r in (contrast("shell L-BFGS-B minus shell Adam (optimiser at fixed support)", diffs("SL", "SC", five), five),
              contrast("shell L-BFGS-B minus box L-BFGS-B (support under the same maximiser)", diffs("SL", "BC", five), five),
              contrast("shell argmax minus shell L-BFGS-B", diffs("SA", "SL", five), five)):
        P(row(r))
    P()
    P("### The factorial with L-BFGS-B as the continuous maximiser on both supports (not printed)")
    P()
    alt = [contrast("support under candidate-set argmax (shell - box)", diffs("SA", "BA", five), five),
           contrast("support under continuous maximisation (shell - box, L-BFGS-B)", diffs("SL", "BC", five), five),
           contrast("maximiser on the shell (argmax - L-BFGS-B)", diffs("SA", "SL", five), five),
           contrast("maximiser in the box (argmax - L-BFGS-B)", diffs("BA", "BC", five), five), rL]
    qs = bh([r["p_task"] for r in alt])
    P("| Contrast | n | Delta | Tasks | p_task | p_task_BH | d | p_cell | W/L |")
    P("|---|---|---|---|---|---|---|---|---|")
    for r, q in zip(alt, qs):
        P(row(r, q))
    if a.markdown:
        with open(a.markdown, "w") as f:
            f.write("\n".join(lines) + "\n")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
