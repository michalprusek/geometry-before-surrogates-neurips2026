"""Builder of effective_support.json (run on 2026-10-12, rebuilt for GEN v3; kept as the audit trail).

Usage: python tools/build_effective_support.py <appendixH_facts.csv> <out.json>

The per-arm support definitions and maximiser settings of tab_effective_support. Short texts are written
here by hand from the Appendix H facts table (appendixH_facts.csv next to make_tables.py = the audited
B_appendixH_facts.csv plus the rows added for the camera-ready and control-block arms; columns 'acquisition
function and best_f', 'acquisition maximiser', 'candidate support / box definition and source', 'trust
region / RAASP'). Every arm of this file now has a facts row. The CSV's verbatim cells are copied into 'facts' so every short
text can be checked against its source; the CSV's sha256 is recorded. The control-block arms C1, C2a and C2b
have their own rows; the replication arms <arm>__s82 run the solver of <arm> and have no row of their own.

support_key: the Monte Carlo support of crlib/mc.py whose typical norm the arm's candidates have
(radius_multiple scales it for the GP radius arms; fixed_radius gives r for the fixed-radius GeoWalk arms);
null where candidates are anchored at the incumbent, lie in a trust region or in an embedding.
"""
import csv
import hashlib
import json
import sys

CSV, OUT = sys.argv[1], sys.argv[2]

ARGMAX = "qLogEI argmax over 2,048 fresh candidates per step"
LBFGS = "qLogEI, \\texttt{optimize\\_acqf} L-BFGS-B, 10 restarts, 512 raw"
NONE1 = "none: one uniform draw per step"

# (arm, group, label, support text, support_key, maximiser text, csv row or None, extra)
A = [
    # ---- this paper ----
    ("geowalk", "This paper (surrogate-free, on the shell)", "\\geowalk{}", "shell at $r$ (per run)", "shell",
     "none: 16 uniform shell draws, then 9 greedy geodesic steps from the incumbent (angle $0.60\\to0.15$ rad)",
     "geowalk", {}),
    ("cma_es_subspace", "This paper (surrogate-free, on the shell)", "\\geocma{}", "shell at $r$, CMA-ES in a 16-D subspace",
     None, "CMA-ES in $\\mathbb{R}^{16}$, popsize 5, $\\sigma_0=0.5$, one evaluation per step", "cma_es_subspace", {}),
    ("random_search", "This paper (surrogate-free, on the shell)", "Random search", "shell at $r$ (per run)", "shell",
     NONE1, "random_search", {}),
] + [
    (f"geowalk_r{r}", "\\geowalk{} and \\geocma{} variants", f"\\geowalk{{}}, $r={r}$", f"sphere of fixed radius {r}", None,
     "as \\geowalk{}", "geowalk_r{4,6,8,10,12,16}", {"fixed_radius": r}) for r in (4, 6, 8, 10, 12, 16)
] + [
    ("geowalk_shell10", "\\geowalk{} and \\geocma{} variants", "\\geowalk{}, shell $\\pm10\\,\\%$", "radius $\\bar r\\,U[0.9,1.1]$",
     "shell", "as \\geowalk{}", "geowalk_shell10 / shell20 / shellpct", {}),
    ("geowalk_shell20", "\\geowalk{} and \\geocma{} variants", "\\geowalk{}, shell $\\pm20\\,\\%$", "radius $\\bar r\\,U[0.8,1.2]$",
     "shell", "as \\geowalk{}", "geowalk_shell10 / shell20 / shellpct", {}),
    ("geowalk_shellpct", "\\geowalk{} and \\geocma{} variants", "\\geowalk{}, shell $[p_{05},p_{95}]$",
     "radius $U[6.81, 9.14]$", None, "as \\geowalk{}", "geowalk_shell10 / shell20 / shellpct", {}),
    ("geowalk_rmedian", "\\geowalk{} and \\geocma{} variants", "\\geowalk{}, median radius", "shell at the median cold-start norm",
     None, "as \\geowalk{}", "geowalk_rmedian / geowalk_rpop", {}),
    ("geowalk_rpop", "\\geowalk{} and \\geocma{} variants", "\\geowalk{}, $r=7.98$", "sphere of fixed radius 7.98", None,
     "as \\geowalk{}", "geowalk_rmedian / geowalk_rpop", {"fixed_radius": 7.98}),
] + [
    (f"geowalk_{v}", "\\geowalk{} and \\geocma{} variants", lab, "shell at $r$ (per run)", "shell", maxi,
     "geowalk_one_fifth" if v == "one_fifth" else "geowalk_const{060,035,015} / geowalk_explore{8,24}", {})
    for v, lab, maxi in (("const060", "\\geowalk{}, angle 0.60", "as \\geowalk{}, constant angle 0.60 rad"),
                         ("const035", "\\geowalk{}, angle 0.35", "as \\geowalk{}, constant angle 0.35 rad"),
                         ("const015", "\\geowalk{}, angle 0.15", "as \\geowalk{}, constant angle 0.15 rad"),
                         ("explore8", "\\geowalk{}, 8 exploration steps", "8 uniform shell draws, then 17 geodesic steps"),
                         ("explore24", "\\geowalk{}, 24 exploration steps", "24 uniform shell draws, then 1 geodesic step"),
                         ("one_fifth", "\\geowalk{}, success rule",
                          "geodesic steps with angle $\\times1.2$ after a success, $\\times0.85$ otherwise (start 0.30, "
                          "clamp $[0.05,0.50]$)"))
] + [
    (f"cma_es_subspace_sd{d}", "\\geowalk{} and \\geocma{} variants", f"\\geocma{{}}, $\\tilde D={d}$",
     f"shell at $r$, CMA-ES in an {d}-D subspace" if d == 8 else f"shell at $r$, CMA-ES in a {d}-D subspace", None,
     f"CMA-ES in $\\mathbb{{R}}^{{{d}}}$, popsize 5, one evaluation per step", "cma_es_subspace_sd{8,64}", {})
    for d in (8, 64)
] + [
    # ---- decomposition / factorial ----
    ("random_box_faithful", "Support $\\times$ selection decomposition (Vanilla BO GP)", "Random search, box",
     "box $[p_{01},p_{99}]$", "box_p01p99", NONE1, "random_box_faithful", {}),
    ("hvarfner_scaled_gp", "Support $\\times$ selection decomposition (Vanilla BO GP)", "Vanilla BO, shell",
     "shell at $r$", "shell", ARGMAX, "hvarfner_scaled_gp", {}),
    ("hvarfner_scaled_gp_box_argmax", "Support $\\times$ selection decomposition (Vanilla BO GP)", "Vanilla BO, box, argmax",
     "box $[p_{01},p_{99}]$", "box_p01p99", ARGMAX, "hvarfner_scaled_gp_box_argmax", {}),
    ("hvarfner_scaled_gp_sphere_continuous", "Support $\\times$ selection decomposition (Vanilla BO GP)",
     "Vanilla BO, shell, continuous", "shell at $r$", "shell",
     "qLogEI, projected Adam on the shell: 10 starts from 512 raw, 200 steps (lr 0.05)",
     "hvarfner_scaled_gp_sphere_continuous", {}),
    ("hvarfner_scaled_gp_sphere_lbfgs", "Support $\\times$ selection decomposition (Vanilla BO GP)",
     "Vanilla BO, shell, continuous", "shell at $r$", "shell",
     "qLogEI, L-BFGS-B on the shell over a free direction $v$ mapped to $r v/\\|v\\|$: 10 starts from 512 Gaussian raw samples (BoTorch optimize_acqf, as in the box arm)",
     "hvarfner_scaled_gp_sphere_lbfgs", {}),
    ("hvarfner_scaled_gp_box_faithful", "Support $\\times$ selection decomposition (Vanilla BO GP)", "Vanilla BO, box",
     "box $[p_{01},p_{99}]$", "box_p01p99", LBFGS, "hvarfner_scaled_gp_box_faithful", {}),
    ("hvarfner_scaled_gp_norm_box_argmax", "Support $\\times$ selection decomposition (Vanilla BO GP)",
     "Vanilla BO, box $[\\min,\\max]$, argmax", "box $[\\min,\\max]$", "box_minmax", ARGMAX,
     "hvarfner_scaled_gp_norm_box_argmax", {}),
    ("hvarfner_scaled_gp_norm_box", "Support $\\times$ selection decomposition (Vanilla BO GP)",
     "Vanilla BO, box $[\\min,\\max]$", "box $[\\min,\\max]$", "box_minmax", LBFGS, "hvarfner_scaled_gp_norm_box", {}),
    ("hvarfner_scaled_gp_box", "Support $\\times$ selection decomposition (Vanilla BO GP)", "Vanilla BO, cube $\\pm3\\bar\\sigma$",
     "isotropic cube $[-0.881,0.881]^D$", "box_3sigma", LBFGS, "hvarfner_scaled_gp_box", {}),
    # ---- single-factor controls ----
    ("hvarfner_scaled_gp_prior", "Support controls", "Vanilla BO, prior candidates", "$N(0,I_D)$", "prior",
     "qLogEI argmax over 2,048 draws from $N(0,I_D)$", "hvarfner_scaled_gp_prior", {}),
    ("hvarfner_scaled_gp_angular_box", "Support controls", "Vanilla BO, box directions", "cube directions renormalised to $r$",
     "angular_box", "qLogEI argmax over 2,048 cube points renormalised to $r$", "hvarfner_scaled_gp_angular_box", {}),
    ("hvarfner_scaled_gp_sphere05", "Support controls", "Vanilla BO, shell $0.5\\,r$", "sphere at $0.5\\,r$", "shell",
     ARGMAX, "hvarfner_scaled_gp_sphere{05,15,20}", {"radius_multiple": 0.5}),
    ("hvarfner_scaled_gp_sphere15", "Support controls", "Vanilla BO, shell $1.5\\,r$", "sphere at $1.5\\,r$", "shell",
     ARGMAX, "hvarfner_scaled_gp_sphere{05,15,20}", {"radius_multiple": 1.5}),
    ("hvarfner_scaled_gp_sphere20", "Support controls", "Vanilla BO, shell $2\\,r$", "sphere at $2\\,r$", "shell",
     ARGMAX, "hvarfner_scaled_gp_sphere{05,15,20}", {"radius_multiple": 2.0}),
    ("random_search_prior", "Support controls", "Random search, prior", "$N(0,I_D)$", "prior", NONE1,
     "random_search_prior", {}),
    ("pca_sphere_k64", "Support controls", "Random search, PCA-64", "top-64 PCA sphere at $r$", "pca64", NONE1,
     "pca_sphere_k64", {}),
    ("ellipsoid_rs", "Support controls", "Random search, Gaussian fit", "$N(\\hat\\mu,\\hat\\Sigma)$ of 3,000 latents",
     "gaussian_fit", NONE1, "ellipsoid_rs", {}),
] + [
    (f"convex_hull_{k}", "Support controls", f"Convex, $k={kk}$", f"convex hull of {kk} cached latent(s)",
     f"convex_k{kk}", "none: one Dirichlet(1) combination per step", "convex_hull_k{1,2,3,4,8,16,64}", {})
    for k, kk in (("k1", 1), ("k2", 2), ("rs", 3), ("k4", 4), ("k8", 8), ("k16", 16), ("k64", 64))
] + [
    # ---- control block (RUN_RULES amendment 2026-10-12): C1, C2a, C2b ----
    ("hvarfner_scaled_gp_box_argmax_raw", "Control block (C1, C2)", "Vanilla BO, box, argmax, GP on raw latents (C1)",
     "box $[p_{01},p_{99}]$", "box_p01p99", ARGMAX + ", GP fitted on raw latents",
     "hvarfner_scaled_gp_box_argmax_raw", {}),
    ("hvarfner_scaled_gp_boxdir_argmax", "Control block (C1, C2)", "Vanilla BO, box directions at $r$ (C2a)",
     "$[p_{01},p_{99}]$-box draws renormalised to $r$", "shell",
     "qLogEI argmax over 2,048 box draws renormalised to $r$, GP fitted on raw latents",
     "hvarfner_scaled_gp_boxdir_argmax", {}),
    ("random_boxdir", "Control block (C1, C2)", "Random search, box directions at $r$ (C2b)",
     "$[p_{01},p_{99}]$-box draws renormalised to $r$", "shell", NONE1, "random_boxdir", {}),
    # ---- GP recipes ----
    ("papenmeier_msr", "GP recipes", "Papenmeier MLE (scaled), shell", "shell at $r$", "shell", ARGMAX, "papenmeier_msr", {}),
    ("papenmeier_msr_box_faithful", "GP recipes", "Papenmeier MLE (scaled), box", "box $[p_{01},p_{99}]$", "box_p01p99",
     LBFGS + ", no RAASP", "papenmeier_msr_box_faithful", {}),
    ("papenmeier_msr_published", "GP recipes", "Papenmeier MSR (published), box", "box $[p_{01},p_{99}]$", "box_p01p99",
     "published MSR call: analytic LogEI, RAASP starts, 5 restarts, 512 raw", "papenmeier_msr_published", {}),
    ("papenmeier_msr_norm_box", "GP recipes", "Papenmeier MLE (scaled), box $[\\min,\\max]$", "box $[\\min,\\max]$",
     "box_minmax", LBFGS, "papenmeier_msr_norm_box", {}),
    ("papenmeier_msr_published_norm_box", "GP recipes", "Papenmeier MSR (published), box $[\\min,\\max]$",
     "box $[\\min,\\max]$", "box_minmax", "published MSR call: analytic LogEI, RAASP starts, 5 restarts, 512 raw",
     "papenmeier_msr_published_norm_box", {}),
    ("papenmeier_msr_raasp", "GP recipes", "Papenmeier RAASP trust-region variant", "trust region in the $[\\min,\\max]$ box",
     None, "qLogEI argmax over 5,000 RAASP candidates in a TuRBO-style trust region", "papenmeier_msr_raasp", {}),
    ("papenmeier_msr_box", "GP recipes", "Papenmeier MLE (scaled), cube $\\pm3\\bar\\sigma$", "isotropic cube $[-0.881,0.881]^D$",
     "box_3sigma", LBFGS, "papenmeier_msr_box", {}),
    ("xu_standard_gp", "GP recipes", "Xu Standard GP, shell", "shell at $r$", "shell", ARGMAX, "xu_standard_gp", {}),
    ("xu_standard_gp_box_faithful", "GP recipes", "Xu Standard GP, box", "box $[p_{01},p_{99}]$", "box_p01p99", LBFGS,
     "xu_standard_gp_box_faithful", {}),
    ("xu_standard_gp_norm_box", "GP recipes", "Xu Standard GP, box $[\\min,\\max]$", "box $[\\min,\\max]$", "box_minmax",
     LBFGS, "xu_standard_gp_norm_box", {}),
    ("xu_standard_gp_box", "GP recipes", "Xu Standard GP, cube $\\pm3\\bar\\sigma$", "isotropic cube $[-0.881,0.881]^D$",
     "box_3sigma", LBFGS, "xu_standard_gp_box", {}),
    ("linear_bo", "GP recipes", "Linear BO, box", "box $[p_{01},p_{99}]$", "box_p01p99",
     "qLogEI, \\texttt{optimize\\_acqf} L-BFGS-B, 4 restarts, 512 raw, \\texttt{sample\\_around\\_best}", "linear_bo", {}),
    ("linear_bo_sphere_calib", "GP recipes", "Linear BO, shell (calibrated)", "shell at $r$", "shell", ARGMAX,
     "linear_bo_sphere_calib", {}),
    # ---- other baselines ----
    ("turbo_box_faithful", "Other baselines", "TuRBO-1, fixed box", "trust region in the $[p_{01},p_{99}]$ box", None,
     "Thompson sampling over 5,000 perturbation candidates (TuRBO-1)", "turbo_box_faithful", {}),
    ("rembo", "Other baselines", "REMBO", "16-D embedding, lifted and renormalised to $r$", None,
     "Thompson sampling, argmax over 100 points in $[-4,4]^{16}$", "rembo", {}),
    ("hesbo", "Other baselines", "HeSBO", "16-D hashing embedding, renormalised to $r$", None,
     "Thompson sampling, argmax over 100 points in the low-$d$ box", "hesbo", {}),
    ("gabo", "Other baselines", "GaBO", "unit sphere, proposal scaled to $r$", "shell",
     "analytic EI, Riemannian trust-region optimiser (pymanopt, 5 restarts, 100 raw)", "gabo", {}),
    ("matern_gabo", "Other baselines", "Riemannian Mat\\'ern GP", "unit sphere, proposal scaled to $r$", "shell",
     "analytic EI, Riemannian trust-region optimiser (5 restarts, 100 raw)", "matern_gabo", {}),
    ("turbo", "Stored baselines (not rerun)", "TuRBO (submitted)", "trust region inside the box of the 5 cold-start latents",
     None, "Thompson sampling over 5,000 candidates", "turbo", {}),
    ("baxus", "Stored baselines (not rerun)", "BAxUS", "nested-subspace trust region in the cold-start box", None,
     "Thompson sampling over 5,000 candidates (\\texttt{eval\\_budget} 500)", "baxus", {}),
    ("lolbo", "Stored baselines (not rerun)", "LOL-BO", "latent trust region around the incumbent", None,
     "Thompson sampling (reference code)", "lolbo", {}),
    ("invbo", "Stored baselines (not rerun)", "InvBO", "latent trust region with inversion", None,
     "Thompson sampling (reference code)", "invbo", {}),
    ("cmaes_full", "Stored baselines (not rerun)", "CMA-ES (full-$D$)", "unbounded $\\mathbb{R}^D$", None,
     "CMA-ES in $\\mathbb{R}^{256}$, $\\sigma_0=0.5$, popsize 20", "cma_es (full-dimensional)", {}),
    ("saasbo", "Stored baselines (not rerun)", "SAASBO", "box $[-5,5]^D$", None,
     "qLogEI, \\texttt{optimize\\_acqf}, 10 restarts, 1,024 raw", "saasbo", {}),
    ("cowboys", "Stored baselines (not rerun)", "COWBOYS (as run)", "Gaussian around the cold-start centroid ($\\|z\\|\\approx64$)",
     None, "Thompson sampling over 20 decoded draws", "cowboys", {}),
]

rows = {r["arm"]: r for r in csv.DictReader(open(CSV))}
FACT_COLS = ["acquisition function and best_f", "acquisition maximiser", "candidate support / box definition and source",
             "trust region / RAASP"]
out = []
for arm, group, label, sup, skey, maxi, csv_row, extra in A:
    e = {"arm": arm, "group": group, "label": label, "support": sup, "support_key": skey, "maximiser": maxi}
    e.update({k: v for k, v in extra.items() if k in ("radius_multiple", "fixed_radius")})
    if csv_row:
        r = rows[csv_row]
        e["source"] = f"appendixH_facts.csv row '{csv_row}'"
        e["facts"] = {k: r[k] for k in FACT_COLS}
    else:
        e["source"] = extra.get("source", "")
    out.append(e)
sha = hashlib.sha256(open(CSV, "rb").read()).hexdigest()
json.dump({"schema": "cr-effective-support-v1", "source_csv": "appendixH_facts.csv", "source_csv_sha256": sha,
           "built": "2026-10-12", "arms": out}, open(OUT, "w"), indent=1)
print(len(out), "arms;", "csv sha256", sha)
