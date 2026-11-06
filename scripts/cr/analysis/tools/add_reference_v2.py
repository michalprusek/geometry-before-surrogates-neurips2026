"""Additions to reference_values.json for GEN v2 (2026-10-12; kept as the audit trail).

Usage: python tools/add_reference_v2.py reference_values.json
Appends the entries below (ids 'A2-...') to the committed reference file; existing entries are not touched,
and re-running is refused if an A2 entry already exists. Every expected value comes from an earlier,
independent computation:

  R04a2  = scratchpad/research/04_numbers_geometry.md, GP-based radius sweep table, and its raw output
           scratchpad/work/numbers-geo/ana1.out section (a2) (flat means over each arm's own clean cells;
           paired statistics on the 439 common cells; tasks favouring 1.0x and exact task-level p)
  WT     = scratchpad/work/numbers-core/worktrees.txt (p1_blocks random_box_faithful: T11clean = 440,
           flat = tmean = 0.3895)
  V2     = scratchpad/cr/gen/GEN_verify_2.md section 4 (2,000,000-draw Monte Carlo of the convex norms:
           k = 4 5.3155 +- 0.0005, Gaussian fit 7.9885 +- 0.0005; exact k = 1 mean 7.98462)
  E/R03  = the posted 2x2 (flat, 423-cell clean mask) and E-1 (task-mean, 426) values that the v1 checks
           already use, now read from the new tab_factorial_by_box production numbers
"""
import json
import sys

PATH = sys.argv[1]
ref = json.load(open(PATH))
if any(e["id"].startswith("A2-") for e in ref["entries"]):
    sys.exit("A2 entries already present; refusing to add them twice")
E = []


def add(id_, source, quantity, printed, value, query, cmp="abs", tol=None, note=""):
    e = {"id": id_, "group": "audit", "source": source, "quantity": quantity, "printed": printed, "value": value,
         "cmp": cmp, "query": query}
    if tol is not None:
        e["tol"] = tol
    if note:
        e["note"] = note
    E.append(e)


OWN = {"kind": "own"}
PAIR = {"kind": "pair"}
# ---- GP radius sweep (legacy files: spherevo, tulen/kajman April 2026) ------------------------------------
GP = [("sphere05", "0.5x", "0.4144", 0.4144, "+0.0089", 0.0089, "0.15", 0.149, "1.51e-01", 0.151, 6, "0.638", 0.638),
      ("sphere15", "1.5x", "0.4164", 0.4164, "+0.0069", 0.0069, "0.11", 0.113, "2.48e-04", 2.48e-4, 7, "0.320", 0.320),
      ("sphere20", "2.0x", "0.4062", 0.4062, "+0.0171", 0.0171, "0.24", 0.242, "7.84e-09", 7.84e-9, 9, "0.019", 0.019),
      ("prior", "N(0,I)", "0.4112", 0.4112, "+0.0121", 0.0121, "0.20", 0.197, "1.80e-07", 1.80e-7, 7, "0.083", 0.083)]
for key, lab, lp, lv, dp, dv, cp, cv, pp, pv, tw, ptp, ptv in GP:
    arm = f"hvarfner_scaled_gp_{key}"
    src = "R04a2 (research/04 GP radius table; work/numbers-geo/ana1.out (a2))"
    add(f"A2-gprad-{key}-level", src, f"GP argmax {lab}: flat mean, own clean cells", lp, lv,
        {"type": "level", "arm": arm, "mask": OWN, "conv": "flat"})
    q = {"type": "paired", "a": "hvarfner_scaled_gp", "b": arm, "mask": PAIR, "conv": "flat"}
    add(f"A2-gprad-{key}-delta", src, f"1.0x - {lab}: flat Delta on the common cells", dp, dv, dict(q, stat="delta_flat"))
    add(f"A2-gprad-{key}-d", src, f"1.0x - {lab}: Cohen's d", cp, cv, dict(q, stat="d"), tol=0.005)
    add(f"A2-gprad-{key}-p", src, f"1.0x - {lab}: cell-level Wilcoxon p", pp, pv, dict(q, stat="p"), cmp="sig2")
    add(f"A2-gprad-{key}-tasks", src, f"1.0x - {lab}: tasks favouring 1.0x", str(tw), tw, dict(q, stat="tasks_won"),
        cmp="exact")
    add(f"A2-gprad-{key}-ptask", src, f"1.0x - {lab}: exact task-level p", ptp, ptv, dict(q, stat="p_task"))
add("A2-gprad-ref-level", "R04a2", "GP argmax 1.0x: flat mean, own clean cells", "0.4223", 0.4223,
    {"type": "level", "arm": "hvarfner_scaled_gp", "mask": OWN, "conv": "flat"})
# ---- the September p1 box arm used by the legacy decomposition variant ----------------------------------------
add("A2-p1-b0-level", "WT (work/numbers-core/worktrees.txt: p1_blocks random_box_faithful T11clean=440)",
    "uniform [p01,p99] box (p1 blocks): flat mean, 440 cells", "0.3895", 0.3895,
    {"type": "level", "arm": "p1_random_box_faithful", "mask": OWN, "conv": "flat"})
add("A2-p1-b0-n", "WT", "uniform [p01,p99] box (p1 blocks): clean cells", "440", 440,
    {"type": "n", "mask": {"kind": "joint", "arms": ["p1_random_box_faithful"]}}, cmp="exact")
# ---- Monte Carlo norms (2,000,000 draws now; verifier 2's independent 2M-draw values) ------------------------
add("A2-mc-k1", "V2 section 4 (exact mean of the 3,000 cached latents)", "convex k = 1 mean norm (exact)", "7.98462",
    7.98462, {"type": "number", "name": "crConvexNormConvexHullKOne"}, tol=1e-5)
add("A2-mc-k4", "V2 section 4 (2M draws: 5.3155 +- 0.0005)", "convex k = 4 mean norm (Monte Carlo)", "5.3155", 5.3155,
    {"type": "number", "name": "crConvexNormConvexHullKFour"}, tol=0.0021,
    note="tolerance = 3 x the combined standard error of two independent 2M-draw estimates")
add("A2-mc-gauss", "V2 section 4 (2M draws: 7.9885 +- 0.0005)", "Gaussian-fit mean norm (Monte Carlo)", "7.9885", 7.9885,
    {"type": "number", "name": "crConvexNormEllipsoidRs"}, tol=0.0021,
    note="tolerance = 3 x the combined standard error of two independent 2M-draw estimates")
# ---- tab_factorial_by_box production numbers against the posted and E-1 values ---------------------------
for cell, val in (("ShellCand", 0.4164), ("ShellCont", 0.3830), ("BoxCand", 0.3862), ("BoxCont", 0.3931)):
    add(f"A2-fbb-posted-{cell}", "posted 6qvY rebuttal (Q1), R03 4.1", f"tab_factorial_by_box posted row {cell} (flat, 423)",
        f"{val:.4f}", val, {"type": "number", "name": f"crFbbPostedPct{cell}"})
add("A2-fbb-posted-n", "R03 4.1", "tab_factorial_by_box posted row n", "423", 423,
    {"type": "number", "name": "crFbbPostedPctN"}, cmp="exact")
add("A2-fbb-posted-inter", "GEN_build 4 (posted flat interaction +0.0404)", "posted row interaction (flat)", "+0.0404",
    0.0404, {"type": "number", "name": "crFbbPostedPctInteractionDelta"})
add("A2-fbb-posted-inter-ptask", "GEN_build 4", "posted row interaction exact task p", "0.0049", 0.0049,
    {"type": "number", "name": "crFbbPostedPctInteractionPTask"}, tol=0.00005)
for cell, val in (("ShellCand", 0.4223), ("ShellCont", 0.3884), ("BoxCand", 0.3913), ("BoxCont", 0.3973)):
    add(f"A2-fbb-e1-{cell}", "E-1 (GEN_build 4; tab_factorial_details)", f"tab_factorial_by_box E-1 row {cell} (task-mean, 426)",
        f"{val:.4f}", val, {"type": "number", "name": f"crFbbEonePct{cell}"})
add("A2-fbb-e1-n", "E-1", "tab_factorial_by_box E-1 row n", "426", 426, {"type": "number", "name": "crFbbEonePctN"},
    cmp="exact")
ref["entries"].extend(E)
with open(PATH, "w") as f:
    json.dump(ref, f, indent=1)
print(len(E), "A2 entries added;", len(ref["entries"]), "in total")
