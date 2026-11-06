"""Additions to reference_values.json for GEN v3 (2026-10-12; kept as the audit trail).

Usage: python tools/add_reference_v3.py reference_values.json [extra_entries.json ...]
Appends the entries below (ids 'A3-...' = legacy checks, 'P3-...' = correspondence rows of
tab_posted_vs_printed) and the entries of any extra JSON list given on the command line (the lists
proposed with the LassoBench and September modules) to the committed reference file. Existing entries
are not touched; re-running is refused if an A3 or P3 entry already exists.

Sources of the expected values (each an earlier, independent computation):

  V3   = scratchpad/cr/gen/GEN_v2_verify.md section 9 (verifier 3): the "precise value" of every support's
         mean candidate norm (exact; quadrature with error <= 4e-8 for the boxes, the cube and the
         Gaussian fit; control-variate Monte Carlo with 20M-100M draws for the convex samplers) and the
         "correct" printed digits of the norm (2 decimals) and of the ratio to r-bar (3 decimals)
  ANY  = results/precomputed/rebuttal_anytime.json written by the anytime script of the July rebuttal (July
         rebuttal, reviewer 6qvY W3): nCSR and per-query mean of GeoWalk and Vanilla BO on the shell
  R03  = the posted 2x2 cells (6qvY Q1) and scratchpad/work/september-digest/server/gate0_premise.md
         (E-1, 426 clean cells) for the interaction rows
  P1   = scratchpad/work/september-digest/server/p1_blocks_report.md section 6.2 (uniform shell - uniform
         box, native scale)
  GEO  = the posted shell prediction (Lhbx, AC): measured 7.9846, predicted 7.9859, 0.016 %, participation
         ratio 63.6, 66 components for 90 %
  LIN  = work log of the Linear BO shell port (uncalibrated arm linear_bo_sphere: 0.4130 on the tPDn mask)
"""
import json
import sys

PATH = sys.argv[1]
ref = json.load(open(PATH))
if any(e["id"].startswith(("A3-", "P3-")) for e in ref["entries"]):
    sys.exit("A3/P3 entries already present; refusing to add them twice")
E = []


def add(id_, source, quantity, printed, value, query, cmp="abs", tol=None, note="", group="audit", cr_number=None,
        reason=""):
    e = {"id": id_, "group": group, "source": source, "quantity": quantity, "printed": printed, "value": value,
         "cmp": cmp, "query": query}
    if tol is not None:
        e["tol"] = tol
    if note:
        e["note"] = note
    if cr_number is not None or group != "audit":
        e["cr_number"] = cr_number
    if reason:
        e["reason"] = reason
    E.append(e)


def num(name, field=None):
    q = {"type": "number", "name": name}
    if field:
        q["field"] = field
    return q


# ---- candidate norms: precise values and printed digits (verifier 3) ---------------------------------------
# support id, macro stem, precise value, tolerance, correct printed norm, correct printed ratio
SRC = "V3 (GEN_v2_verify section 9, precise values and correct digits)"
MC = [("shell", "Shell", 7.983371, 1e-6, "7.98", "1.000"),
      ("box_p01p99", "BoxPZeroOnePNineNine", 11.1250241, 1e-6, "11.13", "1.394"),
      ("box_minmax", "BoxMinmax", 19.8560777, 1e-6, "19.86", "2.487"),
      ("box_3sigma", "BoxThreeSigma", 8.1360677, 1e-6, "8.14", "1.019"),
      ("prior", "Prior", 15.9843827, 1e-6, "15.98", "2.002"),
      ("gaussian_fit", "GaussianFit", 7.9883605, 1e-6, "7.99", "1.001"),
      ("convex_k1", "ConvexKOne", 7.9846203, 1e-6, "7.98", "1.000"),
      # convex samplers: tolerance 4 x sqrt(SE_generator^2 + SE_verifier^2)
      ("convex_k2", "ConvexKTwo", 6.617561, 2.1e-4, "6.62", "0.829"),
      ("convex_k3", "ConvexKThree", 5.834858, 1.2e-4, "5.83", "0.731"),
      ("convex_k4", "ConvexKFour", 5.315045, 5.5e-5, "5.32", "0.666"),
      ("convex_k8", "ConvexKEight", 4.241765, 1.4e-4, "4.24", "0.531"),
      ("convex_k16", "ConvexKOneSix", 3.456331, 7e-5, "3.46", "0.433"),
      ("convex_k64", "ConvexKSixFour", 2.623723, 1.5e-5, "2.62", "0.329")]
for sid, stem, val, tol, tn, tr in MC:
    add(f"A3-mc-{sid}", SRC, f"mean candidate norm, {sid}", f"{val}", val, num(f"crSupp{stem}Norm"), tol=tol)
    add(f"A3-mcdigit-{sid}-norm", SRC, f"printed mean norm, {sid}", tn, tn, num(f"crSupp{stem}Norm", "text"), cmp="exact")
    add(f"A3-mcdigit-{sid}-ratio", SRC, f"printed norm / r-bar, {sid}", tr, tr, num(f"crSupp{stem}Ratio", "text"),
        cmp="exact")
add("A3-mcdigit-convex-k3-tabconvex", SRC, "printed mean norm of the convex k = 3 sampler in tab_convex", "5.83", "5.83",
    num("crConvexNormConvexHullRs", "text"), cmp="exact",
    note="GEN v2 printed 5.84 (plain Monte Carlo, 1.2 SE from the rounding boundary)")

# ---- anytime metrics of the July rebuttal ---------------------------------------------------------------------
# SRC is the source string recorded in reference_values.json (kept as written there); the script it names is
# not part of this release
SRC = "ANY (results/precomputed/rebuttal_anytime.json; scripts/analysis/rebuttal_anytime.py, July)"
add("A3-any-ncsr-geowalk", SRC, "GeoWalk nCSR (own clean cells, f* of the rebuttal)", "0.3161", 0.31606, num("crAnyLegacyGeowalkNcsr"),
    tol=5e-5)
add("A3-any-ncsr-vanilla", SRC, "Vanilla BO (shell) nCSR", "0.3059", 0.30586, num("crAnyLegacyVanillaSphereNcsr"), tol=5e-5)
add("A3-any-perquery-geowalk", SRC, "GeoWalk mean per-query score (empty decode = -1)", "0.2367", 0.23671,
    num("crAnyLegacyGeowalkPerQuery"), tol=5e-5)
add("A3-any-perquery-vanilla", SRC, "Vanilla BO (shell) mean per-query score", "0.2237", 0.22373,
    num("crAnyLegacyVanillaSpherePerQuery"), tol=5e-5)

# ---- correspondence rows of tab_posted_vs_printed (G14) --------------------------------------------------------
F4 = ["hvarfner_scaled_gp", "hvarfner_scaled_gp_sphere_continuous", "hvarfner_scaled_gp_box_argmax",
      "hvarfner_scaled_gp_box_faithful"]
E4 = ["e1_" + a for a in F4]
CR_SET = "camera-ready runs on the six-arm matched set of the decomposition, mean of per-task means (R-3, R-6)"
add("P3-2x2-interaction", "posted cells (their sum), 6qvY rebuttal Q1: implied by the four posted cells", "2x2 interaction (SA-BA)-(SC-BC), flat",
    "+0.0403", 0.0403,
    {"type": "interaction", "SA": F4[0], "SC": F4[1], "BA": F4[2], "BC": F4[3], "stat": "delta_flat", "conv": "flat",
     "mask": {"kind": "joint", "arms": F4, "filter": "clean"}},
    tol=1.5e-4, group="posted", cr_number="crDecIntMaxDelta", reason=CR_SET,
    note="the posted cells are rounded to 4 decimals; the files give +0.0404")
add("P3-2x2-interaction-tasks", "posted files (value not posted), recomputed from the posted-generation files", "2x2 interaction: tasks positive", "9", 9,
    {"type": "interaction", "SA": F4[0], "SC": F4[1], "BA": F4[2], "BC": F4[3], "stat": "tasks_won", "conv": "flat",
     "mask": {"kind": "joint", "arms": F4, "filter": "clean"}},
    cmp="exact", group="posted", cr_number="crDecIntMaxTasks", reason=CR_SET)
for k, a, pr_, role in (("SA", E4[0], "0.4223", "ShellArgmax"), ("SC", E4[1], "0.3884", "ShellCont"),
                        ("BA", E4[2], "0.3913", "BoxArgmax"), ("BC", E4[3], "0.3973", "BoxCont")):
    add(f"P3-e1-{k}", "E-1 rerun (not posted), September, commit abbd61e, gate0_premise.md",
        f"E-1 cell {k} (task-mean, 4-way clean mask, n = 426)", pr_, float(pr_),
        {"type": "level", "arm": a, "conv": "taskmean", "mask": {"kind": "joint", "arms": E4, "filter": "clean"}},
        group="posted", cr_number=f"crDecLevel{role}", reason=CR_SET)
add("P3-e1-interaction", "E-1 rerun (not posted), September, gate0_premise.md", "E-1 interaction (task-mean, n = 426)",
    "+0.0398", 0.0398,
    {"type": "interaction", "SA": E4[0], "SC": E4[1], "BA": E4[2], "BC": E4[3], "stat": "delta_taskmean",
     "conv": "taskmean", "mask": {"kind": "joint", "arms": E4, "filter": "clean"}},
    group="posted", cr_number="crDecIntMaxDelta", reason=CR_SET)
add("P3-e1-interaction-ptask", "E-1 rerun (not posted), September, gate0_premise.md", "E-1 interaction: exact task-level p",
    "0.0049", 0.0049,
    {"type": "interaction", "SA": E4[0], "SC": E4[1], "BA": E4[2], "BC": E4[3], "stat": "p_task", "conv": "taskmean",
     "mask": {"kind": "joint", "arms": E4, "filter": "clean"}},
    tol=5e-5, group="posted", cr_number="crDecIntMaxPTask", reason=CR_SET)
SEP = "p1 blocks (not posted), September, commit 4780475, p1_blocks_report.md 6.2"
for id_, q, pr_, v, crn in (("P3-p1-s0b0-delta", "uniform shell - uniform box, native scale", "+0.0223", 0.0223, "crDecSupportNoneDelta"),
                            ("P3-p1-s0b0-tasks", "uniform shell - uniform box: tasks positive", "9", 9, "crDecSupportNoneTasks"),
                            ("P3-p1-s0b0-ptask", "uniform shell - uniform box: native two-sided task-level p", "0.067", 0.067,
                             "crDecSupportNonePTask")):
    add(id_, SEP, q, pr_, v, None, group="posted", cr_number=crn,
        reason="camera-ready random_search and random_box_faithful (one commit, per-cell runner), six-arm matched set")
GEO = "posted (Lhbx W1; AC), shell prediction on the 3000-latent sample"
GJ = "geometry_cr/default/geometry.json"
for id_, q, pr_, v, path, crn, why, tol in (
        ("P3-geo-measured", "measured mean encoder norm (3,000 latents)", "7.9846", 7.9846,
         ["sample_3000", "stats", "norm", "mean"], "crGeoTabRbarFull", "full ZINC-250k set (249,455 molecules)", None),
        ("P3-geo-predicted", "predicted radius (in-sample formula)", "7.9859", 7.9859,
         ["sample_3000", "stats", "cov_ddof1", "r_pred_posted_in_sample_formula"], "crGeoTabRadiusModelFull",
         "model radius sqrt(m) - Var_G/(8 m^1.5) on the full set, covariance with ddof 1 (the posted formula uses the "
         "observed norm variance and is an in-sample identity)", None),
        ("P3-geo-participation", "participation ratio", "63.6", 63.6,
         ["sample_3000", "stats", "cov_ddof1", "participation_ratio"], "crGeoTabParticipationFull", "full set", None),
        ("P3-geo-pc90", "principal components for 90 % of the variance", "66", 66,
         ["sample_3000", "stats", "cov_ddof1", "n_pc", "90pct"], "crGeoTabPcNinetyFull", "full set", 0)):
    add(id_, GEO, q, pr_, v, {"type": "json", "file": GJ, "path": path}, tol=tol, group="posted", cr_number=crn, reason=why)
add("P3-geo-relerr", GEO, "relative error of the predicted radius, percent", "0.016", 0.016, None, group="posted",
    cr_number="crGeoTabRadiusModelRelErrPctFull",
    reason="model radius against the measured mean norm on the full set (percent); the posted 0.016 % compared the "
           "in-sample formula with the sample it was fitted on")
TP = ["hvarfner_scaled_gp", "geowalk", "linear_bo_sphere_calib", "random_search", "linear_bo",
      "hvarfner_scaled_gp_box_faithful"]
add("P3-linear-uncalibrated", "work log (not posted), July, first shell port of Linear BO",
    "Linear BO shell, uncalibrated kernel bound, best@30 (flat, tPDn 430-cell mask)", "0.4130", 0.4130,
    {"type": "level", "arm": "linear_bo_sphere", "conv": "flat",
     "mask": {"kind": "joint", "arms": TP + ["linear_bo_sphere"], "filter": "clean"}},
    group="posted", cr_number="crLinLinearCalibVsUncalibSecond",
    reason="stored July files (not rerun), mean of per-task means on the cells shared with the camera-ready calibrated "
           "arm; the posted Linear BO shell row (0.4176) is the calibrated arm")

for extra in sys.argv[2:]:
    for e in json.load(open(extra)):
        E.append(e)
ids = [e["id"] for e in ref["entries"]] + [e["id"] for e in E]
if len(ids) != len(set(ids)):
    dup = sorted({i for i in ids if ids.count(i) > 1})
    sys.exit(f"duplicate reference ids: {dup}")
ref["entries"].extend(E)
with open(PATH, "w") as f:
    json.dump(ref, f, indent=1)
print(f"added {len(E)} entries; {len(ref['entries'])} in total")
