"""Builder of reference_values.json (run once on 2026-10-11; kept as the audit trail).

Parses the submitted manuscript's tables (main.tex of the Overleaf base, 513 lines) and adds the
posted rebuttal values and the independently audited values with the query that recomputes each.
The committed JSON is the source of truth. Usage: python tools/build_reference.py <main.tex> <out.json>
writes reference_values.json (committed).

Sources of the expected values (all verified independently before this generator existed):
  R03 = scratchpad/research/03_numbers_core.md, R04 = research/04_numbers_geometry.md,
  G   = cr/G_floor_lasso_geometry.md, C = cr/C_budget_audit.md,
  E   = cr/E_thread_with_math.md (the posted thread with math restored), main.tex = submitted manuscript.
"""
import json
import re
import sys

MAIN = sys.argv[1]
OUT = sys.argv[2]
E = []
_DW = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def camel(part):
    out = []
    for tok in re.split(r"[^A-Za-z0-9]+", str(part)):
        tok = re.sub(r"\d", lambda m: "_" + _DW[m.group(0)] + "_", tok)
        out += [t[0].upper() + t[1:] for t in tok.split("_") if t]
    return "".join(out)


def add(id_, group, source, quantity, printed, value, query=None, cmp="abs", tol=None, cr=None, reason="",
        known_diff=False, note=""):
    e = {"id": id_, "group": group, "source": source, "quantity": quantity, "printed": printed, "value": value,
         "cmp": cmp}
    if tol is not None:
        e["tol"] = tol
    if query is not None:
        e["query"] = query
    if cr:
        e["cr_number"] = cr
    if reason:
        e["reason"] = reason
    if known_diff:
        e["known_diff"] = True
    if note:
        e["note"] = note
    E.append(e)


F4 = ["hvarfner_scaled_gp", "hvarfner_scaled_gp_sphere_continuous", "hvarfner_scaled_gp_box_argmax",
      "hvarfner_scaled_gp_box_faithful"]
E4 = ["e1_" + a for a in F4]
J4 = {"kind": "joint", "arms": F4}
JE = {"kind": "joint", "arms": E4}
TPDN = ["hvarfner_scaled_gp", "geowalk", "linear_bo_sphere_calib", "random_search", "linear_bo",
        "hvarfner_scaled_gp_box_faithful"]
JT = {"kind": "joint", "arms": TPDN}


def lev(arm, mask, conv="flat", N=30):
    return {"type": "level", "arm": arm, "mask": mask, "conv": conv, "N": N}


def pr(a, b, stat, mask=None, conv="flat", N=30):
    q = {"type": "paired", "a": a, "b": b, "stat": stat, "conv": conv, "N": N}
    if mask:
        q["mask"] = mask
    return q


# ---------------------------------------------------------------------------------------------
# Posted (OpenReview, 28 Jul 2026) -- 6qvY 2x2 (flat, 423 mask)
POSTED_6QVY = "posted 6qvY rebuttal (Q1)"
for k, a, v, cr in (("SA", F4[0], 0.4164, "crFactorialShellCand"), ("SC", F4[1], 0.3830, "crFactorialShellCont"),
                    ("BA", F4[2], 0.3862, "crFactorialBoxCand"), ("BC", F4[3], 0.3931, "crFactorialBoxCont")):
    add(f"P-2x2-{k}", "posted", POSTED_6QVY + ", R03 4.1", f"2x2 cell {k} (flat, 4-way clean mask)", f"{v:.4f}", v,
        lev(a, J4), cr=cr, reason="camera-ready runs (R-3), mean of per-task means, completed cells (R-1/R-5/R-6)")
add("P-2x2-n", "posted", POSTED_6QVY + ", R03 4.1", "2x2 matched cells", "423", 423, {"type": "n", "mask": J4},
    cmp="exact", cr="crFactorialN", reason="camera-ready matched set")
add("P-2x2-dcand", "posted", POSTED_6QVY + ", E #30", "support effect at candidate-set argmax, d (own 439 pairs)", "0.44",
    0.44, pr(F4[0], F4[2], "d"), cr="crFactorialSupportCandD", reason="camera-ready runs on the 4-way matched set")
add("P-2x2-dcont", "posted", POSTED_6QVY + ", E #31", "support effect at continuous max, d (423)", "-0.11", -0.11,
    pr(F4[1], F4[3], "d", J4), cr="crFactorialSupportContD", reason="camera-ready runs")
add("A-2x2-pcont", "audit", "R03 4.1 / audit CLAIM 8", "support at continuous max, raw p (423)", "0.138", 0.138,
    pr(F4[1], F4[3], "p", J4))
add("A-2x2-int-mean", "audit", "R03 5", "interaction, mean per cell (423, flat)", "+0.0404", 0.0404,
    {"type": "interaction", "SA": F4[0], "SC": F4[1], "BA": F4[2], "BC": F4[3], "stat": "delta_flat", "conv": "flat"})
add("A-2x2-int-d", "audit", "R03 5", "interaction, d (423)", "0.357", 0.357,
    {"type": "interaction", "SA": F4[0], "SC": F4[1], "BA": F4[2], "BC": F4[3], "stat": "d"})
add("A-2x2-int-tasks", "audit", "R03 5", "interaction, tasks positive", "9", 9,
    {"type": "interaction", "SA": F4[0], "SC": F4[1], "BA": F4[2], "BC": F4[3], "stat": "tasks_won"}, cmp="exact")
add("A-2x2-int-ptask", "audit", "R03 5", "interaction, exact task-level p", "0.0049", 0.0049,
    {"type": "interaction", "SA": F4[0], "SC": F4[1], "BA": F4[2], "BC": F4[3], "stat": "p_task"})
add("A-2x2-taskmean-SA", "audit", "R03 7.2", "rebuttal 2x2 in task-mean, SA (423)", "0.4222", 0.4222, lev(F4[0], J4, "taskmean"))
add("A-2x2-taskmean-BC", "audit", "R03 7.2", "rebuttal 2x2 in task-mean, BC (423)", "0.3983", 0.3983, lev(F4[3], J4, "taskmean"))
# E-1 (task-mean, 426)
for k, a, v in (("SA", E4[0], 0.4223), ("SC", E4[1], 0.3884), ("BA", E4[2], 0.3913), ("BC", E4[3], 0.3973)):
    add(f"A-E1-{k}", "audit", "R03 7.2 / plan R4", f"E-1 cell {k} (task-mean, 4-way clean mask)", f"{v:.4f}", v,
        lev(a, JE, "taskmean"))
add("A-E1-n", "audit", "R03 7.2", "E-1 matched cells", "426", 426, {"type": "n", "mask": JE}, cmp="exact")
add("A-E1-flat-SA", "audit", "R03 7.2", "E-1 SA flat (426)", "0.4220", 0.4220, lev(E4[0], JE))
add("A-E1-cont-delta", "audit", "R03 7.3", "E-1 support at continuous max, flat Delta", "-0.0088", -0.0088, pr(E4[1], E4[3], "delta_flat", JE))
add("A-E1-cont-p", "audit", "R03 7.3", "E-1 support at continuous max, p", "0.026", 0.026, pr(E4[1], E4[3], "p", JE))
add("A-E1-cand-d", "audit", "R03 7.3", "E-1 support at candidate-set max, d", "+0.43", 0.43, pr(E4[0], E4[2], "d", JE))
add("A-E1-int-d", "audit", "R03 5", "E-1 interaction d (426)", "+0.373", 0.373,
    {"type": "interaction", "SA": E4[0], "SC": E4[1], "BA": E4[2], "BC": E4[3], "stat": "d"})
add("A-E1-int-ptask", "audit", "R03 5", "E-1 interaction exact task-level p", "0.0049", 0.0049,
    {"type": "interaction", "SA": E4[0], "SC": E4[1], "BA": E4[2], "BC": E4[3], "stat": "p_task"})

# 6qvY trajectory table (task-mean, each arm on its own clean cells)
TRAJ = [("geowalk", "GeoWalk", (0.3610, 0.3977, 0.4226), "Geowalk"),
        ("hvarfner_scaled_gp", "Vanilla BO sphere", (0.3673, 0.4067, 0.4223), "VanillaSphere"),
        ("random_search", "RS sphere", (0.3613, 0.3965, 0.4099), "RandomSearch"),
        ("hvarfner_scaled_gp_box_faithful", "Vanilla BO box", (0.3454, 0.3787, 0.3986), "VanillaBox"),
        ("turbo", "TuRBO", (0.3393, 0.3665, 0.3804), "Turbo"),
        ("baxus", "BAxUS", (0.3169, 0.3404, 0.3579), "Baxus")]
WORD = {10: "OneZero", 20: "TwoZero", 30: "ThreeZero"}
for arm, lab, vals, key in TRAJ:
    for N, v in zip((10, 20, 30), vals):
        add(f"P-traj6-{arm}-{N}", "posted", "posted 6qvY rebuttal (W3) table, R03 4.1",
            f"{lab}, best@{N} (task-mean, own clean cells)", f"{v:.4f}", v,
            lev(arm, {"kind": "own"}, "taskmean", N), cr=f"crTraj{key}N{WORD[N]}",
            reason="one matched set for all rows (R-5), camera-ready runs (R-3)")
# tPDn 430-cell table (flat, 6-way clean joint)
TPDN_V = [("hvarfner_scaled_gp", "Vanilla BO sphere", (0.3651, 0.4050, 0.4205), "VanillaSphere"),
          ("geowalk", "GeoWalk", (0.3588, 0.3953, 0.4202), "Geowalk"),
          ("linear_bo_sphere_calib", "Linear BO sphere", (0.3661, 0.4050, 0.4176), "LinearSphere"),
          ("random_search", "RS sphere", (0.3593, 0.3940, 0.4076), "RandomSearch"),
          ("linear_bo", "Linear BO box", (0.3478, 0.3869, 0.4063), "LinearBox"),
          ("hvarfner_scaled_gp_box_faithful", "Vanilla BO box", (0.3437, 0.3769, 0.3968), "VanillaBox")]
for arm, lab, vals, key in TPDN_V:
    for N, v in zip((10, 20, 30), vals):
        add(f"P-tpdn-{arm}-{N}", "posted", "posted tPDn rebuttal table, R03 4.2",
            f"{lab}, best@{N} (flat, 430-cell joint mask)", f"{v:.4f}", v, lev(arm, JT, "flat", N),
            cr=f"crTraj{key}N{WORD[N]}", reason="mean of per-task means on the camera-ready matched set")
        add(f"A-prod-tpdn-{arm}-{N}", "audit", "R03 4.2 via the production tab_trajectories (variant R1_flat)",
            f"{lab}, best@{N} from crTraj{key}N{WORD[N]} (R1_flat)", f"{v:.4f}", v,
            {"type": "number", "name": f"crTraj{key}N{WORD[N]}", "variant": "R1_flat"})
add("P-tpdn-n", "posted", "posted tPDn rebuttal, R03 4.2", "tPDn matched cells", "430", 430, {"type": "n", "mask": JT},
    cmp="exact", cr="crTrajN", reason="camera-ready matched set")
add("A-prod-tpdn-n", "audit", "production tab_trajectories (R1_flat)", "joint n of the trajectory table", "430", 430,
    {"type": "number", "name": "crTrajN", "variant": "R1_flat"}, cmp="exact")
add("P-tpdn-linbox-rs-d", "posted", "posted tPDn rebuttal (Linear BO), E #35", "RS sphere - Linear BO box at N=30, d", "0.03",
    0.03, pr("random_search", "linear_bo", "d"), tol=0.008, cr="crMainLinearBoxD", reason="camera-ready runs")
add("P-tpdn-linbox-rs-p", "posted", "posted tPDn rebuttal (Linear BO), E #35", "RS sphere - Linear BO box at N=30, p", "0.14",
    0.14, pr("random_search", "linear_bo", "p"), known_diff=True, cr="crMainLinearBoxPBH",
    note="R03 4.2: not reproduced by any file state (0.154 on own pairs, 0.217 on the 430 mask)",
    reason="camera-ready runs")

# Radius estimator / sweep / shells / convex / PCA (posted; own cells, 440 each = flat = task-mean)
EST = [("geowalk", "mean-of-5 (default)", 0.4226, "crEstGeowalkRefMean"),
       ("geowalk_rmedian", "median-of-5", 0.4207, "crEstMedianMean"),
       ("geowalk_rpop", "all training latents (fixed 7.98)", 0.4223, "crEstPopMean")]
for arm, lab, v, cr in EST:
    add(f"P-est-{arm}", "posted", "posted 6qvY (W2), R03 4.1", f"radius estimator: {lab}", f"{v:.4f}", v,
        lev(arm, {"kind": "own"}), cr=cr, reason="camera-ready runs (wave 1)")
    add(f"A-prod-est-{arm}", "audit", "production tab_estimators", f"radius estimator {lab} ({cr})", f"{v:.4f}", v,
        {"type": "number", "name": cr})
RAD = [(4, 0.4124), (6, 0.4187), (8, 0.4227), (10, 0.4204), (12, 0.4199), (16, 0.4117)]
WN = {4: "Four", 6: "Six", 8: "Eight", 10: "OneZero", 12: "OneTwo", 16: "OneSix"}
for r, v in RAD:
    cr = "crRadiusGeowalkREightRefMean" if r == 8 else f"crRadiusR{WN[r]}Mean"
    add(f"P-radius-{r}", "posted", "posted Lhbx / 6qvY radius sweep, R04 a", f"GeoWalk r={r}", f"{v:.4f}", v,
        lev(f"geowalk_r{r}", {"kind": "own"}), cr=cr, reason="camera-ready runs (wave 1; July files had no provenance)")
    add(f"A-prod-radius-{r}", "audit", "production tab_radius", f"GeoWalk r={r} ({cr})", f"{v:.4f}", v,
        {"type": "number", "name": cr})
SH = [("geowalk_shell10", "+-10 %", 0.4183, "crShellTenMean"), ("geowalk_shell20", "+-20 %", 0.4195, "crShellTwentyMean"),
      ("geowalk_shellpct", "U[p05,p95]", 0.4204, "crShellPctMean")]
for arm, lab, v, cr in SH:
    add(f"P-shell-{arm}", "posted", "posted Lhbx thick shells, R04 b", f"thick shell {lab}", f"{v:.4f}", v,
        lev(arm, {"kind": "own"}), cr=cr, reason="camera-ready runs")
    add(f"A-prod-shell-{arm}", "audit", "production tab_shells", f"thick shell {lab} ({cr})", f"{v:.4f}", v,
        {"type": "number", "name": cr})
CV = [("convex_hull_k1", 1, 0.3838), ("convex_hull_k2", 2, 0.3888), ("convex_hull_rs", 3, 0.3909),
      ("convex_hull_k4", 4, 0.3829), ("convex_hull_k8", 8, 0.3893), ("convex_hull_k16", 16, 0.3855),
      ("convex_hull_k64", 64, 0.3823)]
KW = {1: "One", 2: "Two", 3: "Three", 4: "Four", 8: "Eight", 16: "OneSix", 64: "SixFour"}
for arm, k, v in CV:
    add(f"A-convex-{k}", "audit", "R04 d", f"convex k={k}", f"{v:.4f}", v, lev(arm, {"kind": "own"}),
        cr=f"crConvexK{KW[k]}Mean")
    add(f"A-prod-convex-{k}", "audit", "production tab_convex", f"convex k={k}", f"{v:.4f}", v,
        {"type": "number", "name": f"crConvexK{KW[k]}Mean"})
add("P-convex-min", "posted", "posted Lhbx convex combinations, R03 4.3", "convex range min (k in 1,2,4,8,16,64)", "0.3823",
    0.3823, {"type": "level_agg", "a_list": [a for a, k, _ in CV if k != 3], "agg": "min", "conv": "flat"}, cr="crConvexRangeMin", reason="camera-ready runs")
add("P-convex-max", "posted", "posted Lhbx convex combinations, R03 4.3", "convex range max (k in 1,2,4,8,16,64)", "0.3893",
    0.3893, {"type": "level_agg", "a_list": [a for a, k, _ in CV if k != 3], "agg": "max", "conv": "flat"}, cr="crConvexRangeMax", reason="camera-ready runs")
add("A-prod-convex-min", "audit", "production tab_convex", "convex range min (crConvexRangeMin)", "0.3823", 0.3823,
    {"type": "number", "name": "crConvexRangeMin"})
add("A-prod-convex-max", "audit", "production tab_convex", "convex range max (crConvexRangeMax)", "0.3893", 0.3893,
    {"type": "number", "name": "crConvexRangeMax"})
add("P-convex-rs", "posted", "posted Lhbx, R03 4.3", "uniform random search on the shell", "0.4099", 0.4099,
    lev("random_search", {"kind": "own"}), cr="crConvexRandomSearchRefMean",
    reason="camera-ready rerun of random search (deterministic decoder, 80-token cap)")
add("P-convex-p", "posted", "posted Lhbx (math restored), E #21", "max p of the six k vs RS (posted p < 1e-5)", "<1e-5",
    1e-5, {"type": "paired_agg", "a_list": [a for a, k, _ in CV if k != 3], "b": "random_search", "stat": "p", "agg": "max"},
    cmp="lt", known_diff=True, note="R04 d: k = 8 gives p = 1.3e-5, so the posted 'p < 1e-5' holds for 5 of 6 k",
    reason="camera-ready runs")
add("A-convex-pmax", "audit", "R04 d", "max p of the six k vs RS", "1.3e-05", 1.3e-5,
    {"type": "paired_agg", "a_list": [a for a, k, _ in CV if k != 3], "b": "random_search", "stat": "p", "agg": "max"},
    cmp="sig2")
add("P-pca", "posted", "posted Lhbx (W2), R03 4.3", "top-64 PCA sphere", "0.4069", 0.4069, lev("pca_sphere_k64", {"kind": "own"}),
    cr="crDirPcaVariantMean", reason="camera-ready runs")
add("P-pca-d", "posted", "posted Lhbx (math restored), E #17", "PCA-64 vs full sphere, d", "0.05", 0.05,
    pr("random_search", "pca_sphere_k64", "d"), cr="crDirPcaD", cmp="absval", reason="camera-ready runs (sign: variant - shell)")
add("A-pca-p", "audit", "R04 e", "PCA-64 vs RS, p", "0.048", 0.048, pr("random_search", "pca_sphere_k64", "p"))
add("A-angular", "audit", "R03 audit CLAIM 3", "Vanilla BO box directions at r (flat, 439)", "0.4226", 0.4226,
    lev("hvarfner_scaled_gp_angular_box", {"kind": "pair", }, "flat") | {"mask": {"kind": "joint", "arms": ["hvarfner_scaled_gp_angular_box", "hvarfner_scaled_gp"]}})
add("A-angular-shell", "audit", "R03 audit CLAIM 3", "Vanilla BO shell on the same 439 cells (flat)", "0.4233", 0.4233,
    lev("hvarfner_scaled_gp", {"kind": "joint", "arms": ["hvarfner_scaled_gp_angular_box", "hvarfner_scaled_gp"]}))
add("A-prior", "audit", "R03 audit CLAIM 2", "Vanilla BO candidates from N(0,I) (flat, 439)", "0.4112", 0.4112,
    lev("hvarfner_scaled_gp_prior", {"kind": "joint", "arms": ["hvarfner_scaled_gp_prior", "hvarfner_scaled_gp"]}))
add("A-prior-d", "audit", "R03 audit CLAIM 2", "shell - prior, d (439)", "+0.20", 0.20, pr("hvarfner_scaled_gp", "hvarfner_scaled_gp_prior", "d"))
add("A-rsprior", "audit", "R04 d", "random search from the prior", "0.4004", 0.4004, lev("random_search_prior", {"kind": "own"}))
add("A-ellipsoid", "audit", "R04 d", "ellipsoid N(mu, Sigma)", "0.3840", 0.3840, lev("ellipsoid_rs", {"kind": "own"}))

# Linear BO rows (R03 6.7), own cells and pairs with GeoWalk
add("A-linbox-flat", "audit", "R03 6.7", "Linear BO box, flat (own 439)", "0.4087", 0.4087, lev("linear_bo", {"kind": "own"}))
add("A-linbox-tm", "audit", "R03 6.7", "Linear BO box, task-mean (own 439)", "0.4081", 0.4081, lev("linear_bo", {"kind": "own"}, "taskmean"))
add("A-linbox-delta", "audit", "R03 6.7", "GeoWalk - Linear BO box, Delta", "+0.0147", 0.0147, pr("geowalk", "linear_bo", "delta_flat"))
add("A-linbox-d", "audit", "R03 6.7", "GeoWalk - Linear BO box, d", "0.21", 0.21, pr("geowalk", "linear_bo", "d"))
add("A-linsph-flat", "audit", "R03 6.7", "Linear BO sphere calibrated, flat (own 439)", "0.4201", 0.4201,
    lev("linear_bo_sphere_calib", {"kind": "own"}))
add("A-linsph-delta", "audit", "R03 6.7", "GeoWalk - Linear BO sphere, Delta", "+0.0034", 0.0034, pr("geowalk", "linear_bo_sphere_calib", "delta_flat"))
add("A-linsph-d", "audit", "R03 6.7", "GeoWalk - Linear BO sphere, d", "0.06", 0.06, pr("geowalk", "linear_bo_sphere_calib", "d"))
add("A-linsph-p", "audit", "R03 6.7", "GeoWalk - Linear BO sphere, p", "0.059", 0.059, pr("geowalk", "linear_bo_sphere_calib", "p"))
add("A-linsph-wl", "audit", "R03 6.7", "GeoWalk - Linear BO sphere, wins", "222", 222, pr("geowalk", "linear_bo_sphere_calib", "wins"), cmp="exact")

# Corrected Table 1 (R03 6.1; 11 tasks, per-row pairs with GeoWalk, clean)
T1 = [("lolbo", "LOL-BO", 0.3947, 0.0278, 292, 148, 0.33, "Lolbo"),
      ("cmaes_full", "CMA-ES", 0.3873, 0.0352, 295, 144, 0.39, "CmaesFull"),
      ("invbo", "InvBO", 0.3856, 0.0370, 305, 135, 0.40, "Invbo"),
      ("turbo", "TuRBO", 0.3804, 0.0421, 297, 139, 0.42, "Turbo"),
      ("baxus", "BAxUS", 0.3579, 0.0646, 347, 93, 0.58, "Baxus"),
      ("random_search", "Random Search", 0.4099, 0.0127, 207, 85, 0.28, "RandomSearch"),
      ("cma_es_subspace", "GeoCMA", 0.4126, 0.0099, 230, 164, 0.15, "Geocma"),
      ("gabo", "GaBO", 0.4159, 0.0066, 222, 176, 0.12, "Gabo"),
      ("matern_gabo", "Riemannian Matern", 0.4059, 0.0166, 232, 152, 0.26, "MaternGabo")]
for arm, lab, cm, dl, w, l_, d, key in T1:
    add(f"A-t1-{arm}-cm", "audit", "R03 6.1 (corrected 11-task values)", f"Table 1 {lab}: cross-mean", f"{cm:.4f}", cm,
        lev(arm, {"kind": "own"}))
    add(f"A-t1-{arm}-delta", "audit", "R03 6.1", f"Table 1 {lab}: Delta (GeoWalk - row)", f"+{dl:.4f}", dl,
        pr("geowalk", arm, "delta_flat"))
    add(f"A-t1-{arm}-wins", "audit", "R03 6.1", f"Table 1 {lab}: wins", str(w), w, pr("geowalk", arm, "wins"), cmp="exact")
    add(f"A-t1-{arm}-losses", "audit", "R03 6.1", f"Table 1 {lab}: losses", str(l_), l_, pr("geowalk", arm, "losses"), cmp="exact")
    add(f"A-t1-{arm}-d", "audit", "R03 6.1", f"Table 1 {lab}: d", f"{d:.2f}", d, pr("geowalk", arm, "d"),
        known_diff=(arm == "gabo"),
        note=("submitted 0.12 = d over the 398 non-tied cells (0.1206); the stated formula (src/analysis/stats.py "
              "cohens_d, all 440 cells, ddof 1) gives 0.1146 -> 0.11; Riemannian Matern 0.26 does use all cells "
              "(0.2569; non-tied 0.2763), so the submitted table mixes the two" if arm == "gabo" else ""))
add("A-t1-gabo-p", "audit", "R03 6.1", "Table 1 GaBO: raw p", "0.0051", 0.0051, pr("geowalk", "gabo", "p"))
add("A-t1-geowalk", "audit", "R03 6.1", "GeoWalk cross-mean (440)", "0.4226", 0.4226, lev("geowalk", {"kind": "own"}))
for arm, lab, n_, cm, dl, w, l_, d in (("saasbo", "SAASBO", 104, 0.3810, 0.0490, 72, 21, 0.53),
                                         ("rembo", "REMBO", 110, 0.4098, 0.0200, 67, 27, 0.35),
                                         ("hesbo", "HeSBO", 110, 0.4084, 0.0213, 63, 31, 0.34),
                                         ("cowboys", "COWBOYS", 110, 0.3353, 0.0945, 89, 6, 0.86)):
    add(f"A-t1-{arm}-cm", "audit", "R03 6.1 (10-seed rows)", f"{lab}: cross-mean (flat)", f"{cm:.4f}", cm, lev(arm, {"kind": "own"}))
    add(f"A-t1-{arm}-n", "audit", "R03 6.1", f"{lab}: pairs with GeoWalk", str(n_), n_, pr("geowalk", arm, "n"), cmp="exact")
    add(f"A-t1-{arm}-delta", "audit", "R03 6.1", f"{lab}: paired Delta", f"+{dl:.4f}", dl, pr("geowalk", arm, "delta_flat"))
    add(f"A-t1-{arm}-wl", "audit", "R03 6.1", f"{lab}: wins", str(w), w, pr("geowalk", arm, "wins"), cmp="exact")
    add(f"A-t1-{arm}-d", "audit", "R03 6.1", f"{lab}: d", f"{d:.2f}", d, pr("geowalk", arm, "d"))
add("A-t1-saasbo-tm", "audit", "R03 6.1", "SAASBO task-mean (104)", "0.3837", 0.3837, lev("saasbo", {"kind": "own"}, "taskmean"))
add("A-t1-vanbox-clean", "audit", "R03 6.1", "Vanilla [min,max] box clean (434): cross-mean flat", "0.3902", 0.3902,
    lev("hvarfner_scaled_gp_norm_box", {"kind": "own"}))
add("A-t1-vanbox-delta", "audit", "R03 6.1", "GeoWalk - Vanilla [min,max] clean: Delta", "+0.0332", 0.0332,
    pr("geowalk", "hvarfner_scaled_gp_norm_box", "delta_flat"))
add("A-t1-vanbox-d", "audit", "R03 6.1", "GeoWalk - Vanilla [min,max] clean: d", "0.43", 0.43, pr("geowalk", "hvarfner_scaled_gp_norm_box", "d"))
add("A-t1-xubox-delta", "audit", "R03 6.1", "GeoWalk - Xu [min,max] clean (439): Delta", "+0.0435", 0.0435,
    pr("geowalk", "xu_standard_gp_norm_box", "delta_flat"))
add("A-t1-xubox-d", "audit", "R03 6.1", "GeoWalk - Xu [min,max] clean: d", "0.52", 0.52, pr("geowalk", "xu_standard_gp_norm_box", "d"))
FAM18 = [["geowalk", a] for a in ("lolbo", "cmaes_full", "invbo", "turbo", "baxus", "saasbo", "rembo", "hesbo", "cowboys",
                                 "hvarfner_scaled_gp_norm_box", "papenmeier_msr_norm_box", "xu_standard_gp_norm_box",
                                 "gabo", "matern_gabo", "random_search", "cma_es_subspace", "linear_bo",
                                 "linear_bo_sphere_calib")]
for target, lab, v in ((17, "Linear BO sphere", 0.059), (12, "GaBO", 0.0054), (15, "GeoCMA", 0.0011)):
    add(f"A-t1bh18-{target}", "audit", "R03 6.7 (BH over the 18-row Table 1)", f"p_BH {lab}", str(v), v,
        {"type": "bh", "family": FAM18, "target": target}, tol=0.00051 if v < 0.01 else 0.0006)
# Submitted Table 2 (unfiltered n = 440)
UNF = {"kind": "own", "filter": "completed"}
for arm, lab, v in (("hvarfner_scaled_gp", "Vanilla sphere", 0.4223), ("hvarfner_scaled_gp_norm_box", "Vanilla box [min,max]", 0.3901),
                    ("papenmeier_msr_sqrtd", "Papenmeier sphere sqrt(D)", 0.4149), ("papenmeier_msr_norm_box", "Papenmeier box sqrt(D)", 0.3794),
                    ("xu_standard_gp", "Xu sphere", 0.4118), ("xu_standard_gp_norm_box", "Xu box [min,max]", 0.3794)):
    add(f"A-t2sub-{arm}", "audit", "R03 6.6 (submitted Table 2, unfiltered)", lab, f"{v:.4f}", v, lev(arm, UNF, "flat"))
for a, b, lab, dl, d in (("hvarfner_scaled_gp", "hvarfner_scaled_gp_norm_box", "Vanilla", 0.0322, 0.40),
                         ("papenmeier_msr_sqrtd", "papenmeier_msr_norm_box", "Papenmeier", 0.0355, 0.42),
                         ("xu_standard_gp", "xu_standard_gp_norm_box", "Xu", 0.0324, 0.39)):
    m_ = {"kind": "pair", "filter": "completed"}
    add(f"A-t2sub-{lab}-delta", "audit", "R03 6.6 (within recipe, unfiltered 440)", f"{lab} sphere - [min,max] box: Delta",
        f"+{dl:.4f}", dl, pr(a, b, "delta_flat", m_))
    add(f"A-t2sub-{lab}-d", "audit", "R03 6.6", f"{lab} sphere - [min,max] box: d", f"{d:.2f}", d, pr(a, b, "d", m_))
add("A-papmsr-box-faithful-d", "audit", "R03 6.3", "Papenmeier sphere - box_faithful (439): d", "0.22", 0.22,
    pr("papenmeier_msr", "papenmeier_msr_box_faithful", "d"))
add("A-papmsr-raasp-d", "audit", "R03 6.3", "Papenmeier sphere - RAASP (438): d", "0.42", 0.42,
    pr("papenmeier_msr", "papenmeier_msr_raasp", "d"))
add("A-papmsr-raasp-delta", "audit", "R03 6.3", "Papenmeier sphere - RAASP (438): Delta flat", "+0.0360", 0.0360,
    pr("papenmeier_msr", "papenmeier_msr_raasp", "delta_flat"))
# Appendix B (10 seeds) through the production tab_sensitivity
APPB = [("crSensGeowalkRefMean", "default", 0.4298), ("crSensConstSixtyMean", "const 0.60", 0.4237),
        ("crSensConstThirtyFiveMean", "const 0.35", 0.4233), ("crSensConstFifteenMean", "const 0.15", 0.4183),
        ("crSensExploreEightMean", "explore 8", 0.4191), ("crSensExploreTwentyFourMean", "explore 24", 0.4165)]
for name, lab, v in APPB:
    add(f"A-prod-appb-{lab}", "audit", "R03 6.8 via production tab_sensitivity", f"Appendix B {lab}", f"{v:.4f}", v,
        {"type": "number", "name": name})
for name, lab, v in (("crSensConstSixtyPBH", "const 0.60", 0.0019), ("crSensConstThirtyFivePBH", "const 0.35", 0.042),
                     ("crSensConstFifteenPBH", "const 0.15", 0.0010), ("crSensExploreEightPBH", "explore 8", 0.057),
                     ("crSensExploreTwentyFourPBH", "explore 24", 0.0002)):
    add(f"A-prod-appb-pbh-{lab}", "audit", "R03 6.8 (BH of the script's raw p)", f"Appendix B p_BH {lab}", str(v), v,
        {"type": "number", "name": name}, tol=max(0.00051, v * 0.03))
add("A-prod-appb-n", "audit", "production tab_sensitivity", "Appendix B matched cells (seeds 42-51)", "110", 110,
    {"type": "number", "name": "crSensMatchedN"}, cmp="exact")
for name, lab, v in (("crSensDtildeEightMean", "D~=8", 0.5986), ("crSensDtildeSixteenMean", "D~=16", 0.6032),
                     ("crSensDtildeSixtyFourMean", "D~=64", 0.6021)):
    add(f"A-prod-dtilde-{lab}", "audit", "map MC-5 (recomputed from the March files)", f"GeoCMA {lab}", f"{v:.4f}", v,
        {"type": "number", "name": name})
add("A-prod-dtilde-d8", "audit", "map MC-5", "GeoCMA D~=8 vs 16: d", "-0.059", -0.059, {"type": "number", "name": "crSensDtildeEightD"}, tol=0.0051)
# Floor (G 1.3, 1.5)
FLOORS = {"adip": 0.465939, "dhop": 0.568440, "fexo": 0.602554, "med1": 0.150132, "med2": 0.163296, "osmb": 0.687791,
          "pdop": 0.387062, "rano": 0.264543, "shop": 0.436934, "siga": 0.143071, "zale": 0.349098}
for t, v in FLOORS.items():
    add(f"A-floor-{t}", "audit", "G 1.3", f"retrieval floor {t} (no replacement)", f"{v:.6f}", v,
        {"type": "number", "name": "crFloor" + camel(t)})
add("A-floor-mean", "audit", "G 1.3", "mean floor over 11 tasks", "0.3835", 0.3835, {"type": "number", "name": "crFloorMean"})
add("A-floor-gw-P", "audit", "G 1.5", "GeoWalk A/i/B (rule P)", "5/3/3", "5/3/3", {"type": "number", "name": "crFloorArmGeowalkCountsP"}, cmp="exact")
add("A-floor-gw-T", "audit", "G 1.5", "GeoWalk A/i/B (rule T)", "5/3/3", "5/3/3", {"type": "number", "name": "crFloorArmGeowalkCountsT"}, cmp="exact")
add("A-floor-gw-diff", "audit", "G 1.5", "GeoWalk cross-task mean - floor", "+0.0390", 0.0390, {"type": "number", "name": "crFloorArmGeowalkDiff"})
add("A-floor-gw-ptask", "audit", "G 1.5", "GeoWalk task-level p", "0.278", 0.278, {"type": "number", "name": "crFloorArmGeowalkPTask"})
add("A-floor-baxus-ptask", "audit", "G 1.5", "BAxUS task-level p (the only significant arm)", "0.032", 0.032,
    {"type": "number", "name": "crFloorArmBaxusPTask"})
add("A-floor-baxus-P", "audit", "G 1.5", "BAxUS A/i/B (rule P)", "1/1/9", "1/1/9", {"type": "number", "name": "crFloorArmBaxusCountsP"}, cmp="exact")
add("A-floor-k1-diff", "audit", "G 1.7", "convex k=1 minus floor", "+0.0003", 0.0003, {"type": "number", "name": "crFloorArmConvexHullKOneDiff"})
# LassoBench (G 2.2)
for name, lab, v, tol in (("crLassoGeowalk", "GeoWalk normalised loss", 0.7950, None), ("crLassoGeocma", "GeoCMA normalised loss", 0.4389, None),
                          ("crLassoDelta", "Delta", 0.3561, None), ("crLassoD", "d", 0.591, None), ("crLassoN", "n", 240, None),
                          ("crLassoPTask", "task-level exact p", 0.031, None)):
    add(f"A-lasso-{name}", "audit", "G 2.2 (current files, 6 tasks, seeds 42-81)", f"LassoBench {lab}", str(v), v,
        {"type": "number", "name": name}, cmp="exact" if name == "crLassoN" else "abs", tol=tol)
add("A-lasso-p", "audit", "G 2.2", "LassoBench p", "3.4e-23", 3.4e-23, {"type": "number", "name": "crLassoP"}, cmp="sig2")
for meth, D, v, n_ in (("Cmaessub", 60, 0.471, 40), ("Linearbo", 100, 0.243, 40), ("Xustandardgp", 300, 0.825, 25),
                       ("Hvarfnergp", 7129, 0.879, 36), ("Geowalk", 7129, 0.947, 40)):
    pass
for key, D, v, n_ in (("CmaesSub", "SixZero", 0.471, 40), ("LinearBo", "OneZeroZero", 0.243, 40),
                      ("XuStandardGp", "ThreeZeroZero", 0.825, 25), ("HvarfnerGp", "SevenOneTwoNine", 0.879, 36),
                      ("Geowalk", "SevenOneTwoNine", 0.947, 40)):
    add(f"A-lasso-{key}-{D}", "audit", "G 2.2 (Figure 2 points)", f"LassoBench {key} at D={D}", f"{v:.3f}", v,
        {"type": "number", "name": f"crLasso{key}D{D}"})
    add(f"A-lasso-{key}-{D}-n", "audit", "G 2.2 (seed coverage)", f"LassoBench {key} at D={D}: seeds", str(n_), n_,
        {"type": "number", "name": f"crLasso{key}D{D}N"}, cmp="exact")
# LassoBench: GeoCMA against Linear BO (Related Work / Doumont) and per dimension (G 2.2, R04 j)
for name, lab, v, cmp_ in (("crLassoVsGeocmaLinearBoDelta", "Linear BO - GeoCMA pooled Delta", 0.060, "abs"),
                           ("crLassoVsGeocmaLinearBoD", "Linear BO - GeoCMA pooled d", 0.10, "abs"),
                           ("crLassoVsGeocmaLinearBoP", "Linear BO - GeoCMA pooled p", 0.33, "abs"),
                           ("crLassoVsGeocmaLinearBoN", "Linear BO - GeoCMA pooled n (5 tasks x 40)", 200, "exact"),
                           ("crLassoVsGeocmaLinearBoDOneZeroZeroD", "D=100: Linear BO - GeoCMA d (Linear BO better)", -0.53, "abs"),
                           ("crLassoVsGeocmaLinearBoDOneZeroZeroP", "D=100: Linear BO - GeoCMA p", 2.6e-4, "sig2"),
                           ("crLassoVsGeocmaHvarfnerGpDSixZeroP", "D=60: Hvarfner - GeoCMA p (the only significant one)", 8e-4, "sig2"),
                           ("crLassoVsGeocmaGeowalkDSixZeroP", "D=60: GeoWalk - GeoCMA p", 0.23, "abs"),
                           ("crLassoVsGeocmaLinearBoDOneZeroZeroZeroP", "D=1000: Linear BO - GeoCMA p", 0.014, "abs")):
    add(f"A-lassovs-{name}", "audit", "G 2.2 / R04 j", f"LassoBench {lab}", str(v), v, {"type": "number", "name": name},
        cmp=cmp_, tol=(0.006 if name.endswith("D") and cmp_ == "abs" else None))

# NAS-Bench (R04 j)
for name, lab, v in (("crNasGeowalk", "GeoWalk", 0.9372), ("crNasGeocma", "GeoCMA", 0.9415), ("crNasDelta", "Delta", 0.0044),
                     ("crNasD", "d", 0.94), ("crNasP", "p", 0.0195), ("crNasN", "n", 10),
                     ("crNasGeowalkVsRandomDelta", "GeoWalk - random", -0.0033), ("crNasGeowalkVsRandomP", "GeoWalk - random p", 0.037)):
    add(f"A-nas-{name}", "audit", "R04 j (seeds 42-51)", f"NAS-Bench {lab}", str(v), v, {"type": "number", "name": name},
        cmp="exact" if name == "crNasN" else "abs")
# Budget (R04 k): own 110 cells per row
BUD = [("b1000_geowalk", "GeoWalk", (0.424, 0.445, 0.463, 0.479, 0.497, 0.509)),
       ("b1000_hvarfner_scaled_gp", "Vanilla sphere", (0.426, 0.443, 0.458, 0.472, 0.489, 0.501)),
       ("b1000_cma_es_subspace", "GeoCMA", (0.414, 0.440, 0.462, 0.474, 0.488, 0.496)),
       ("b1000_random_search", "RS sphere", (0.410, 0.424, 0.444, 0.460, 0.478, 0.491)),
       ("b1000_vanilla_bo_hvarfner", "TS-GP cold-start box", (0.378, 0.393, 0.417, 0.427, 0.445, 0.456)),
       ("b1000_baxus", "BAxUS", (0.365, 0.379, 0.393, 0.404, 0.416, 0.424))]
for arm, lab, vals in BUD:
    for N, v in zip((30, 50, 100, 200, 500, 1000), vals):
        add(f"A-budget-{arm}-{N}", "audit", "R04 k (recomputed, own cells)", f"budget {lab} N={N}", f"{v:.3f}", v,
            lev(arm, {"kind": "own"}, "taskmean", N), known_diff=(arm == "b1000_vanilla_bo_hvarfner"),
            note=("the printed row used 20 older _128d_b1000 files for deco_hop and scaffold_hop, seeds 42-51 "
                  "(substituting them gives 0.3785/0.3928/0.4168/0.4267/0.4446/0.4563); the generator reads the one "
                  "complete generation (_256d_b1000), which differs in 13 of those 20 cells"
                  if arm == "b1000_vanilla_bo_hvarfner" else ""))
# Cross-codec (R04 i)
for a, b, lab, sa, sb, d, p, n_ in (("chembed_hvarfner_scaled_gp", "chembed_hvarfner_scaled_gp_box_faithful", "Chembed", 0.3870, 0.3835, 0.04, 0.38, 424),
                                    ("power_spherical_hvarfner_scaled_gp", "power_spherical_hvarfner_scaled_gp_box_faithful", "PowerSph", 0.3681, 0.3588, 0.11, 0.052, 413)):
    add(f"A-codec-{lab}-n", "audit", "R04 i", f"{lab} sphere vs box: n", str(n_), n_, pr(a, b, "n"), cmp="exact")
    add(f"A-codec-{lab}-sph", "audit", "R04 i", f"{lab} sphere (flat, pair)", f"{sa:.4f}", sa, pr(a, b, "level_a"))
    add(f"A-codec-{lab}-box", "audit", "R04 i", f"{lab} box (flat, pair)", f"{sb:.4f}", sb, pr(a, b, "level_b"))
    add(f"A-codec-{lab}-d", "audit", "R04 i", f"{lab} sphere - box: d", f"{d:.2f}", d, pr(a, b, "d"))
    add(f"A-codec-{lab}-p", "audit", "R04 i", f"{lab} sphere - box: p", str(p), p, pr(a, b, "p"))
add("A-codec-chembed-gw", "audit", "R04 i", "Chembed GeoWalk - Vanilla sphere: Delta", "-0.0094", -0.0094,
    pr("chembed_geowalk", "chembed_hvarfner_scaled_gp", "delta_flat"))
add("A-codec-chembed-gw-p", "audit", "R04 i", "Chembed GeoWalk - Vanilla sphere: p", "0.026", 0.026,
    pr("chembed_geowalk", "chembed_hvarfner_scaled_gp", "p"), tol=0.001,
    note="scipy 1.17.1 gives 0.0255 (58 zero differences, zero_method 'wilcox'); third-decimal difference only")
# Per-task table (submitted version)
PT = {"Amlodipine": "adip", "Deco Hop": "dhop", "Fexofenadine": "fexo", "Median 1": "med1", "Median 2": "med2",
      "Osimertinib": "osmb", "Perindopril": "pdop", "Ranolazine": "rano", "Scaffold Hop": "shop", "Sitagliptin": "siga",
      "Zaleplon": "zale"}
# Geometry (G 3.4) and acq attainment
for name, lab, v in (("crGeomRbarFour", "full-set mean norm", 7.9834), ("crGeomRbarSd", "SD of the norm", 0.71),
                     ("crGeomParticipation", "participation ratio", 64.7), ("crGeomPcNinety", "PCs for 90 %", 66),
                     ("crGeomPcNinetyNine", "PCs for 99 %", 92), ("crGeomRadiusModel", "model radius", 7.9860),
                     ("crGeomSumSigma", "sum E sigma^2", 181.6), ("crGeomZNorm", "posterior sample norm", 15.66),
                     ("crGeomTokenLossPct", "percent token-dropped", 63.1), ("crGeomNZinc", "molecules", 249455),
                     ("crGeomSampleMean", "3,000 sample: measured", 7.9846), ("crGeomSamplePred", "3,000 sample: predicted", 7.9859),
                     ("crGeomSampleParticipation", "3,000 sample: participation ratio", 63.6)):
    add(f"A-geom-{name}", "audit", "G 3.4 / posted Lhbx", f"geometry: {lab}", str(v), v, {"type": "number", "name": name},
        cmp="exact" if isinstance(v, int) else "abs")
add("A-geom-var64", "audit", "R04 e / posted Lhbx", "top-64 PCs carry this percent of variance (3,000 sample)", "89.2", 89.2,
    {"type": "number", "name": "crGeomSampleVarTopSixtyFour"})
add("A-p2-08-d", "audit", "R03 audit CLAIM 8 (map P2-08)", "shell/argmax - GaBO: d", "0.10", 0.10, {"type": "number", "name": "crShellArgmaxVsGaboD"},
    tol=0.0051)
add("A-p2-08-p", "audit", "R03 audit CLAIM 8 (map P2-08)", "shell/argmax - GaBO: p", "0.077", 0.077, {"type": "number", "name": "crShellArgmaxVsGaboP"})
add("A-acq", "audit", "R03 4.1 (posted 99.1 %)", "acquisition attainment, percent", "99.1", 99.1, {"type": "number", "name": "crAcqPct"})

# Budget fidelity (C 3): distinct new molecules of 25, repeats per cell (clean cells)
for key, v, what in (("Geowalk", 24.22, "Distinct"), ("Turbo", 23.20, "Distinct"), ("Invbo", 23.84, "Distinct"),
                     ("Baxus", 24.05, "Distinct"), ("Lolbo", 24.70, "Distinct"), ("CmaesFull", 24.76, "Distinct"),
                     ("RandomSearch", 24.93, "Distinct"), ("CmaEsSubspace", 24.48, "Distinct"),
                     ("Geowalk", 0.78, "Repeats"), ("Turbo", 1.41, "Empty"), ("Invbo", 0.85, "Empty"),
                     ("Baxus", 0.65, "Empty")):
    add(f"A-fid-{key}-{what}", "audit", "C 3 (score-equality proxy)", f"budget fidelity {key}: {what.lower()} per cell",
        f"{v:.2f}", v, {"type": "number", "name": f"crFid{key}{what}"}, tol=0.006)
add("A-fid-sphcont-repeats", "audit", "C 6.8", "shell x Adam: repeats per cell", "8.7", 8.7,
    {"type": "number", "name": "crFidHvarfnerScaledGpSphereContinuousRepeats"}, tol=0.06)

# ---------------------------------------------------------------------------------------------
# Submitted values (main.tex), with the camera-ready macro that replaces each
lines = open(MAIN).read().split("\n")


def body(label):
    st = next(i for i, ln in enumerate(lines) if "\\label{" + label + "}" in ln)
    en = next(i for i in range(st, len(lines)) if "\\bottomrule" in lines[i])
    return range(st, en)


def row_cells(i):
    s = lines[i].split("\\\\")[0]
    return [c.strip() for c in s.split("&")]


def clean(c):
    c = re.sub(r"\^\\(dagger|ddagger|S)", "", c)
    c = re.sub(r"\$\^\{?\\(dagger|ddagger|S)\}?\$", "", c)
    c = re.sub(r"\\textbf\{([^}]*)\}", r"\1", c)
    return c.replace("$", "").replace("\\!", "").strip()


def num(c):
    c = clean(c).replace("+", "")
    c = c.replace("−", "-")
    try:
        return float(c) if not c.startswith("<") else None
    except ValueError:
        return None


T1MAP = {"LOL-BO": "Lolbo", "CMA-ES": "CmaesFull", "InvBO": "Invbo", "TuRBO": "Turbo", "BAxUS": "Baxus", "SAASBO": "Saasbo",
         "REMBO": "Rembo", "HeSBO": "Hesbo", "COWBOYS": "Cowboys", "Vanilla BO": "VanillaBox", "Papenmeier MSR": "PapenmeierMsr",
         "Xu Standard GP": "XuBox", "GaBO": "Gabo", "Riemannian Mat": "MaternGabo", "Random Search": "RandomSearch",
         "\\geocma": "Geocma"}
REASON_T1 = {
    "Lolbo": "11 tasks (valsartan dropped from the paired statistics); GeoWalk from the camera-ready runs; one matched set (R-5)",
    "CmaesFull": "same as LOL-BO", "Invbo": "same as LOL-BO", "Turbo": "same as LOL-BO", "Baxus": "same as LOL-BO",
    "Saasbo": "printed row reproduced by no file state; stored files (104 cells) against camera-ready GeoWalk",
    "Rembo": "rerun at 40 seeds from the camera-ready commit; submitted Delta was unpaired (40 vs 10 seeds)",
    "Hesbo": "as REMBO", "Cowboys": "row dropped (RUN_RULES 3: not the published method); variant cowboys_kept",
    "VanillaBox": "box [p01,p99] instead of [min,max] (R2); device fix; camera-ready rerun; one matched set",
    "PapenmeierMsr": "submitted row used the sqrt(D) initialisation and was identical to Xu; now the published MSR maximiser on [p01,p99]",
    "XuBox": "box [p01,p99] (R2); camera-ready rerun", "Gabo": "camera-ready rerun", "MaternGabo": "camera-ready rerun",
    "RandomSearch": "camera-ready rerun (deterministic decoder, 80-token cap); 11 tasks",
    "Geocma": "camera-ready rerun (deterministic decoder, 80-token cap); 11 tasks"}
for i in body("tab:main"):
    if "&" not in lines[i] or "textbf{Method}" in lines[i]:
        continue
    c = row_cells(i)
    lab = clean(c[0])
    key = next((v for k, v in T1MAP.items() if lab.startswith(k) or k in c[0]), None)
    if key is None and "geowalk" in c[0]:
        add("S-t1-geowalk", "submitted", f"main.tex:{i + 1}", "Table 1 GeoWalk cross-mean", clean(c[1]), num(c[1]),
            cr="crMainGeowalkRefMean", reason="camera-ready rerun; one matched set")
        continue
    if key is None:
        continue
    reason = REASON_T1.get(key, "")
    for j, (col, suffix) in enumerate((("cross-mean", "Mean"), ("Delta", "Delta"), ("W/L", "WL"), ("p_BH", "PBH"), ("d", "D")), start=1):
        printed = clean(c[j])
        value = num(c[j]) if suffix not in ("WL",) else printed
        add(f"S-t1-{key}-{suffix}", "submitted", f"main.tex:{i + 1}", f"Table 1 {lab.split('~')[0]} {col}", printed, value,
            cr=f"crMain{key}{suffix}", reason=reason, cmp="exact" if suffix == "WL" else "abs")
T2 = [("Vanilla BO sphere", "crBoxVanillaBOSphereMean"), ("Vanilla BO box", "crBoxVanillaBOBoxMean"),
      ("Papenmeier sphere", "crBoxPapenmeierSphereMean"), ("Papenmeier box", "crBoxPapenmeierBoxMean"),
      ("Xu sphere", "crBoxXuStandardGPSphereMean"), ("Xu box", "crBoxXuStandardGPBoxMean")]
for i in body("tab:boxsphere"):
    for lab, cr in T2:
        if lines[i].strip().startswith(lab + " ") and "&" in lines[i]:
            c = row_cells(i)
            add(f"S-t2-{cr}", "submitted", f"main.tex:{i + 1}", f"Table 2 {lab}", clean(c[2]), num(c[2]), cr=cr,
                reason={"Vanilla BO box": "box [p01,p99] (R2); camera-ready rerun",
                        "Papenmeier sphere": "0.1 sqrt(D) initialisation (submitted sqrt(D)); camera-ready rerun",
                        "Papenmeier box": "published MSR maximiser on [p01,p99] (submitted: sqrt(D), identical to Xu)",
                        "Xu sphere": "current Xu GP likelihood; camera-ready rerun",
                        "Xu box": "box [p01,p99] (R2); camera-ready rerun"}.get(lab, "camera-ready rerun; one matched set"))
# Table 3
for i in body("tab:crossdomain"):
    ln = lines[i]
    if "&" not in ln:
        continue
    c = row_cells(i)
    if ln.startswith("GuacaMol"):
        for j, cr, q in ((2, "crCrossGuacamolGeowalk", "GeoWalk"), (3, "crCrossGuacamolGeocma", "GeoCMA"),
                         (4, "crCrossGuacamolDelta", "|Delta|"), (5, "crCrossGuacamolD", "|d|")):
            add(f"S-t3-{cr}", "submitted", f"main.tex:{i + 1}", f"Table 3 GuacaMol {q}", clean(c[j]), num(c[j]), cr=cr,
                cmp="absval" if j >= 4 else "abs", reason="camera-ready runs (GeoCMA: deterministic decoder)")
    if ln.startswith("NAS-Bench"):
        for j, cr, q in ((2, "crNasGeowalk", "GeoWalk"), (3, "crNasGeocma", "GeoCMA"), (4, "crNasDelta", "Delta"), (5, "crNasD", "|d|")):
            add(f"S-t3-{cr}", "submitted", f"main.tex:{i + 1}", f"Table 3 NAS {q}", clean(c[j]), num(c[j]), cr=cr,
                reason="stored files; printed to 4 decimals (plan 5.4)")
    if ln.startswith("LassoBench"):
        for j, cr, q in ((2, "crLassoGeowalk", "GeoWalk"), (3, "crLassoGeocma", "GeoCMA"), (4, "crLassoDelta", "Delta"), (5, "crLassoD", "|d|")):
            add(f"S-t3-{cr}", "submitted", f"main.tex:{i + 1}", f"Table 3 LassoBench {q}", clean(c[j]), num(c[j]), cr=cr,
                reason="current stored files, 40 seeds (printed |d| and p were the 10-seed subset)")
# Appendix B
SENS = {"const-$0.60$": "ConstSixty", "const-$0.35$": "ConstThirtyFive", "const-$0.15$": "ConstFifteen",
        "explore$_8$": "ExploreEight", "explore$_{24}$": "ExploreTwentyFour"}
for i in body("tab:sensitivity"):
    ln = lines[i]
    if "&" not in ln:
        continue
    c = row_cells(i)
    if "(default)" in c[0]:
        add("S-b-default", "submitted", f"main.tex:{i + 1}", "Appendix B GeoWalk default (10 seeds)", clean(c[2]), num(c[2]),
            cr="crSensGeowalkRefMean", reason="all 40 seeds of the camera-ready runs (R-5)")
    for k, key in SENS.items():
        if ln.strip().startswith(k):
            for j, suf, q in ((2, "Mean", "cross-mean"), (3, "Delta", "Delta"), (4, "PBH", "p_BH")):
                add(f"S-b-{key}-{suf}", "submitted", f"main.tex:{i + 1}", f"Appendix B {k} {q}", clean(c[j]), num(c[j]),
                    cr=f"crSens{key}{suf}", reason="camera-ready runs at 40 seeds (amendment arms); BH applied by the generator")
# Appendix C per task
for i in body("tab:pertask"):
    ln = lines[i]
    if "&" not in ln:
        continue
    c = row_cells(i)
    lab = clean(c[0])
    if lab in PT:
        t = PT[lab]
        T = camel(t)
        for j, suf, q in ((2, "Gw", "GeoWalk"), (3, "Vanilla", "Vanilla BO"), (4, "Delta", "Delta"), (5, "D", "d"), (6, "P", "p")):
            add(f"S-c-{t}-{suf}", "submitted", f"main.tex:{i + 1}", f"Appendix C {lab} {q}", clean(c[j]), num(c[j]),
                cr=f"crPertask{T}{suf}", reason="camera-ready runs; matched cells per task")
    if "Cross-mean" in c[0]:
        add("S-c-all-gw", "submitted", f"main.tex:{i + 1}", "Appendix C cross-mean GeoWalk", clean(c[2]), num(c[2]), cr="crPertaskAllGw",
            reason="camera-ready runs")
        add("S-c-all-van", "submitted", f"main.tex:{i + 1}", "Appendix C cross-mean Vanilla BO", clean(c[3]), num(c[3]),
            cr="crPertaskAllVanilla", reason="camera-ready runs")
        add("S-c-all-delta", "submitted", f"main.tex:{i + 1}", "Appendix C Delta", clean(c[4]), num(c[4]), cr="crPertaskAllDelta",
            reason="camera-ready runs")
# Appendix C full per task
FULLCOLS = ["Geowalk", "Geocma", "RandomSearch", "VanillaSphere", "VanillaBox"]
for i in body("tab:pertask_full"):
    ln = lines[i]
    if "&" not in ln:
        continue
    c = row_cells(i)
    lab = clean(c[0])
    if lab in PT or "Cross-mean" in c[0]:
        T = camel(PT[lab]) if lab in PT else "All"
        for j, col in enumerate(FULLCOLS, start=1):
            add(f"S-cf-{T}-{col}", "submitted", f"main.tex:{i + 1}", f"Appendix C full {lab} {col}", clean(c[j]), num(c[j]),
                cr=f"crFull{T}{col}", reason=("box [p01,p99] (R2) instead of [min,max]" if col == "VanillaBox" else
                                              "camera-ready runs; one matched set"))
# Appendix E means
for i, ln in enumerate(lines):
    for lab, key in (("SELFIES", "Selfies"), ("Chembed", "Chembed"), ("PowerSph", "Powersph")):
        if ln.strip().startswith(lab + " ") and "&" in ln and i > 440 and i < 460:
            c = row_cells(i)
            for j, col in ((1, "Gw"), (2, "Cma"), (3, "Sph"), (4, "Box")):
                add(f"S-e-{key}-{col}", "submitted", f"main.tex:{i + 1}", f"Appendix E {lab} {col}", clean(c[j]), num(c[j]),
                    cr=f"crCodec{key}{col}Mean", reason=("camera-ready SELFIES runs; box [p01,p99]" if key == "Selfies" else
                                                        "stored runs recomputed on the codec's matched clean cells"))
# Appendix F budget
BROW = {"\\geowalk{}": "Geowalk", "Vanilla BO on sphere": "VanillaSphere", "\\geocma{}": "Geocma",
        "Random Search (sphere)": "RandomSearch", "Vanilla BO on $[0,1]^D$": "TsGpBox", "BAxUS": "Baxus"}
NW = {30: "ThreeZero", 50: "FiveZero", 100: "OneZeroZero", 200: "TwoZeroZero", 500: "FiveZeroZero", 1000: "OneZeroZeroZero"}
for i in body("tab:budget"):
    ln = lines[i]
    if "&" not in ln:
        continue
    c = row_cells(i)
    lab = c[0].strip()
    if lab in BROW:
        for j, N in enumerate((30, 50, 100, 200, 500, 1000), start=1):
            v = clean(c[j])
            add(f"S-f-{BROW[lab]}-{N}", "submitted", f"main.tex:{i + 1}", f"Appendix F {clean(lab)} N={N}", v,
                float("0" + v) if v.startswith(".") else float(v), cr=f"crBudget{BROW[lab]}N{NW[N]}",
                reason="matched set of all six rows (109 cells, BAxUS zale s44 missing); relabelled box row")
# Inline numbers (text)
add("S-inline-e32-lo", "submitted", "main.tex:221", "sphere - box gain, lower end (Result 2)", "0.033", 0.033, cr="crBoxDeltaMin",
    reason="within-recipe camera-ready contrast on the [p01,p99] box; the printed range was the GeoWalk-vs-box rows")
add("S-inline-e32-hi", "submitted", "main.tex:221", "sphere - box gain, upper end", "0.043", 0.043, cr="crBoxDeltaMax", reason="as above")
add("S-inline-e32-dlo", "submitted", "main.tex:221", "paired d, lower end", "0.43", 0.43, cr="crBoxDMin", reason="as above")
add("S-inline-e32-dhi", "submitted", "main.tex:221", "paired d, upper end", "0.51", 0.51, cr="crBoxDMax", reason="as above")
add("S-inline-e21-dlo", "submitted", "main.tex:190", "Table 1 smallest d (GaBO)", "0.12", 0.12, cr="crMainDMin",
    reason="new rows (Linear BO, on-shell GP recipes) and camera-ready runs")
add("S-inline-e21-dhi", "submitted", "main.tex:190", "Table 1 largest d (COWBOYS)", "0.86", 0.86, cr="crMainDMax",
    reason="COWBOYS dropped (RUN_RULES 3)")
add("S-inline-e21-rows", "submitted", "main.tex:190", "number of Table 1 comparators", "14", 14, cr="crMainRows",
    reason="rows added (Linear BO x2, TuRBO-1, MSR, on-shell GP recipes), COWBOYS dropped", cmp="exact")
add("S-inline-rbar", "submitted", "main.tex:93", "r-bar", "7.98", 7.98, cr="crGeomRbar", reason="full set, 249,455 molecules")
add("S-inline-rbarsd", "submitted", "main.tex:93", "SD of the norm", "0.71", 0.71, cr="crGeomRbarSd", reason="full set")
add("S-inline-nzinc", "submitted", "main.tex:93", "ZINC-250K molecules", "249,456", 249456, cr="crGeomNZinc", cmp="exact",
    reason="the current data/zinc_250k.txt has 249,455 lines; 249,456 came from a since-deleted file")
add("S-inline-p05", "submitted", "main.tex:93", "5th percentile", "6.81", 6.81, cr="crGeomPFive", reason="full set")
add("S-inline-p95", "submitted", "main.tex:93", "95th percentile", "9.14", 9.14, cr="crGeomPNinetyFive", reason="full set")
add("S-inline-e33-delta", "submitted", "main.tex:221", "GeoWalk - Vanilla BO sphere, Delta", "+0.0002", 0.0002, cr="crPertaskAllDelta",
    reason="camera-ready runs")

with open(OUT, "w") as f:
    json.dump({"schema": "cr-reference-v1",
               "description": ("Earlier numbers (submitted manuscript, posted rebuttal, independent audits) with the query "
                               "that recomputes each from stored files and the camera-ready macro that replaces it. "
                               "Built once from main.tex and the audit reports listed in each 'source'; edit here, not in code."),
               "entries": E}, f, indent=1)
print(len(E), "entries;", {g: sum(1 for e in E if e["group"] == g) for g in ("posted", "audit", "submitted")})
