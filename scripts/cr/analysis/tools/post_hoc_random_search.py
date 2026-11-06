#!/usr/bin/env python3
"""Post-hoc task-level test of random search on the shell against the Table 1 rows that propose off the shell.

Added on 2026-11-03 at the authors' request (RUN_RULES row 2026-11-03), after every final run had been read, so it is
not one of the pre-specified families of the paper. The test is the primary test of the paper: the exact two-sided
Wilcoxon signed-rank test on the 11 per-task mean differences of best@30 on the matched set of Table 1 (the BAxUS
row covers 439 of its 440 cells), with Benjamini-Hochberg correction over the seven rows. The first block reproduces
the GeoWalk column of Table 1 from the same files, as a check of the conventions.

    python scripts/cr/analysis/tools/post_hoc_random_search.py --bundle <results_bundle> [--markdown out.md]
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os

from scipy.stats import wilcoxon

ARMS = {  # arm -> (result set inside the bundle, directory)
    "geowalk": "cr2026/geowalk", "random_search": "cr2026/random_search",
    "cmaes_full_rd_pop5": "cr2026rd/cmaes_full_rd_pop5", "turbo_published": "cr2026pub/turbo_published",
    "baxus_cr": "cr2026rd/baxus_cr", "hvarfner_published": "cr2026pub/hvarfner_published",
    "papenmeier_msr_reference_learned": "cr2026pub/papenmeier_msr_reference_learned",
    "xu_published": "cr2026pub/xu_published", "linear_bo_box8": "cr2026pub/linear_bo_box8"}
LABEL = {"cmaes_full_rd_pop5": "CMA-ES", "turbo_published": "TuRBO", "baxus_cr": "BAxUS",
         "hvarfner_published": "Vanilla BO", "papenmeier_msr_reference_learned": "Papenmeier MSR",
         "xu_published": "Xu Standard GP", "linear_bo_box8": "Linear BO", "random_search": "Random Search (shell)"}
OFF_SHELL = ["cmaes_full_rd_pop5", "turbo_published", "baxus_cr", "hvarfner_published",
             "papenmeier_msr_reference_learned", "xu_published", "linear_bo_box8"]
TASKS = {"amlodipine_mpo", "fexofenadine_mpo", "osimetrinib_mpo", "perindopril_mpo", "ranolazine_mpo",
         "sitagliptin_mpo", "zaleplon_mpo", "deco_hop", "scaffold_hop", "median_1", "median_2"}


def load(bundle: str, arm: str) -> dict:
    """(task, seed) -> best score after 30 evaluations, for the completed cells of the 11 protocol tasks."""
    out = {}
    for fn in glob.glob(os.path.join(bundle, ARMS[arm], "*.json")):
        d = json.load(open(fn))
        if d["task"] not in TASKS or d.get("status", "completed") != "completed" or d.get("n_evaluated", 30) < 30:
            continue
        tr = d.get("best_trace") or []
        out[(d["task"], d["seed"])] = float(tr[29] if len(tr) >= 30 else d["best_score"])
    return out


def paired(data: dict, a: str, b: str):
    """n cells, task differences (a minus b, per-task means), mean of task differences, tasks a higher, p_task."""
    cells = sorted(set(data[a]) & set(data[b]))
    by_task = collections.defaultdict(list)
    for c in cells:
        by_task[c[0]].append(data[a][c] - data[b][c])
    td = [sum(v) / len(v) for _, v in sorted(by_task.items())]
    p = float(wilcoxon(td, alternative="two-sided", method="exact").pvalue)
    return len(cells), td, sum(td) / len(td), sum(1 for x in td if x > 0), p


def bh(ps: list) -> list:
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    out, prev = [0.0] * m, 1.0
    for rank, i in reversed(list(enumerate(order, 1))):
        prev = min(prev, ps[i] * m / rank)
        out[i] = prev
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, help="the results bundle (results_bundle/ of release v1.0.0)")
    ap.add_argument("--markdown", default=None, help="write the tables as Markdown to this file")
    args = ap.parse_args()
    data = {a: load(args.bundle, a) for a in ARMS}
    lines = ["# Random search on the shell against the rows of Table 1 that propose off the shell (post-hoc)", "",
             "Added on 2026-11-03 at the authors' request, after every final run had been read (RUN_RULES row "
             "2026-11-03); not a pre-specified family. Exact two-sided Wilcoxon signed-rank test on the 11 per-task "
             "mean differences of best@30 on the matched set of Table 1; Delta = mean of the per-task mean "
             "differences; Tasks = tasks on which the first method has the higher mean; p_BH = Benjamini-Hochberg over "
             "the seven rows of the second table.", "",
             "## Check: GeoWalk minus each row, as Table 1 prints it", "",
             "| Row | n | Delta | Tasks | p_task |", "|---|---|---|---|---|"]
    for b in OFF_SHELL + ["random_search"]:
        n, td, dl, ta, p = paired(data, "geowalk", b)
        lines.append(f"| {LABEL[b]} | {n} | {dl:+.4f} | {ta}/{len(td)} | {p:.4g} |")
    lines += ["", "## Random search on the shell minus each row that proposes off the shell", "",
              "| Row | n | Delta | Tasks | p_task | p_BH |", "|---|---|---|---|---|---|"]
    rows = [paired(data, "random_search", b) for b in OFF_SHELL]
    q = bh([r[4] for r in rows])
    for b, (n, td, dl, ta, p), pq in zip(OFF_SHELL, rows, q):
        lines.append(f"| {LABEL[b]} | {n} | {dl:+.4f} | {ta}/{len(td)} | {p:.4g} | {pq:.4g} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.markdown:
        with open(args.markdown, "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
