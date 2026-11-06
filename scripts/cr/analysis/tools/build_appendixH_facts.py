"""Builder of scripts/cr/analysis/appendixH_facts.csv (GEN v3, 2026-10-12; kept as the audit trail).

Usage: python build_appendixH_facts.py <B_appendixH_facts.csv> <out appendixH_facts.csv>

* The 20 columns of the audited facts table (B_appendixH_facts.csv, sha256 d9220956...) are copied
  VERBATIM, cell by cell, for its 54 rows (the script fails if the source differs from the audited file).
* Seven columns are appended:
    run arms                    generator arm ids the row covers (';'-separated; empty = not a GuacaMol arm)
    lengthscale initialisation  short text DERIVED from the verbatim kernel cell (old rows) or the code (new rows)
    lengthscale prior           the same
    camera-ready overrides      JSON {field: text}: cells whose audited text describes the STORED files only
                                and is known to differ from the code at tag cr-controls-2026-10-12
    row origin                  'B audit 2026-10-11 (verbatim)' | 'GEN v3 2026-10-12 (...)'
    needs proofreading          'all' or a ';'-separated list of column names
    table group                 heading under which tab_baselines prints the row's card
* Thirteen rows are added for the arms the audit did not cover (camera-ready and control-block arms). Every
  file:line in them refers to the code at tag cr-controls-2026-10-12 (fd8a03c), read with
  ``git show cr-controls-2026-10-12:<path>``; facts come from that code and from F_msr_published.md,
  H_commit.md, H2_successor.md, H3_final.md, CTL_build.md and CTL_check.md.
The committed CSV is the source of truth: edit it directly (then update ORIGINAL_BLOCK_SHA256 in the test
only if an AUDITED cell really has to change, which should not happen).
"""
import csv
import hashlib
import json
import sys

SRC, OUT = sys.argv[1], sys.argv[2]
SRC_SHA256 = "d9220956ae334d1ce043797154dd0672cec322a6215cfbb3aeba9f0f1a7f99fc"
TAG = "cr-controls-2026-10-12"

raw = open(SRC, "rb").read()
got = hashlib.sha256(raw).hexdigest()
if got != SRC_SHA256:
    raise SystemExit(f"source CSV sha256 {got} != audited {SRC_SHA256}")
rows = list(csv.reader(open(SRC, newline="", encoding="utf-8")))
HEADER, OLD = rows[0], rows[1:]
assert len(HEADER) == 20 and len(OLD) == 54, (len(HEADER), len(OLD))
C = {name: i for i, name in enumerate(HEADER)}
NEW_COLS = ["run arms", "lengthscale initialisation", "lengthscale prior", "camera-ready overrides", "row origin",
            "needs proofreading", "table group"]
ORIGIN_OLD = "B audit 2026-10-11 (verbatim)"
ORIGIN_NEW = f"GEN v3 2026-10-12 (new row; code read at {TAG})"


def block_sha(header, body) -> str:
    """sha256 of the audited block: cells joined by US (0x1f), rows by RS (0x1e), header first."""
    return hashlib.sha256("\x1e".join("\x1f".join(r) for r in [header] + body).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------------------------
# appended cells of the 54 audited rows
# ------------------------------------------------------------------------------------------------
S82 = {"random_search", "hvarfner_scaled_gp", "hvarfner_scaled_gp_box_argmax", "hvarfner_scaled_gp_sphere_continuous",
       "hvarfner_scaled_gp_box_faithful"}          # base arms of the C3 replication block (+ random_box_faithful, new row)
RUN_ARMS = {
    "geowalk_r{4,6,8,10,12,16}": [f"geowalk_r{r}" for r in (4, 6, 8, 10, 12, 16)],
    "geowalk_shell10 / shell20 / shellpct": ["geowalk_shell10", "geowalk_shell20", "geowalk_shellpct"],
    "geowalk_rmedian / geowalk_rpop": ["geowalk_rmedian", "geowalk_rpop"],
    "convex_hull_k{1,2,3,4,8,16,64}": ["convex_hull_k1", "convex_hull_k2", "convex_hull_rs", "convex_hull_k4",
                                       "convex_hull_k8", "convex_hull_k16", "convex_hull_k64"],
    "cma_es (full-dimensional)": ["cmaes_full"],
    "vanilla_bo_hvarfner (budget appendix)": ["b1000_vanilla_bo_hvarfner"],
}
NA = "n/a (no surrogate)"
HV_INIT = "not set in the code: gpytorch's default raw value 0, i.e. softplus(0) = 0.693 in every dimension"
HV_PRIOR = ("LogNormal(loc = sqrt(2) + 0.5 ln D = 4.187, scale = sqrt(3)) on every ARD lengthscale (Hvarfner et al. "
            "2024, Eq. 3); constraint Positive (gpytorch default); the prior acts through the MAP fit")
PAP_PRIOR = "none (maximum likelihood); constraint Interval(0.01, 100)"
PAP_INIT = "0.1 sqrt(D) = 1.6 in every dimension"
PAP_INIT_OLD = ("sqrt(D) = 16 in every dimension in the stored files (pre-fix value, 10x the published "
                "0.1 sqrt(D))")
XU_INIT = "c sqrt(D) = 16 (c = 1) in every dimension"
LIN_INIT = ("ARD lengthscales at the prior mode 0.205 (GreaterThan(0.025, transform=None)); learnable global "
            "lengthscale sigmoid(raw_glob_ls)")
LIN_PRIOR = "LogNormal(sqrt(2), sqrt(3)) ('dsp_unscaled': not scaled by D)"
UNV = "not recorded in the audit (see the kernel cell)"


def derived(arm: str, kernel: str) -> tuple[str, str]:
    if kernel.startswith("none"):
        return NA, NA
    if arm.startswith("hvarfner_scaled_gp"):
        return HV_INIT, HV_PRIOR
    if arm in ("papenmeier_msr", "papenmeier_msr_box_faithful", "papenmeier_msr_raasp"):
        return PAP_INIT, PAP_PRIOR
    if arm in ("papenmeier_msr_norm_box", "papenmeier_msr_box"):
        return PAP_INIT_OLD, PAP_PRIOR
    if arm.startswith("xu_standard_gp"):
        return XU_INIT, PAP_PRIOR
    if arm == "linear_bo":
        return LIN_INIT, LIN_PRIOR
    if arm in ("linear_bo_sphere", "linear_bo_sphere_calib"):
        return "as linear_bo: " + LIN_INIT, "as linear_bo: " + LIN_PRIOR
    if arm == "turbo":
        return ("not recorded in the audit (no explicit initialisation named in the kernel cell)",
                "none (no explicit priors); constraint Interval(0.005, 4)")
    if arm in ("baxus", "lolbo", "invbo"):
        return "UNVERIFIED (see the kernel cell)", "UNVERIFIED (see the kernel cell)"
    if arm == "saasbo":
        return ("n/a: fully Bayesian, lengthscales are NUTS samples (no point initialisation)",
                "the SAAS prior of BoTorch's SaasFullyBayesianSingleTaskGP (not restated in the audit)")
    if arm == "cowboys":
        return "n/a (the Tanimoto kernel has no lengthscale)", "n/a (the Tanimoto kernel has no lengthscale)"
    if arm in ("rembo", "hesbo"):
        return ("default of the installed BoTorch's SingleTaskGP (version UNVERIFIED)",
                "default of the installed BoTorch's SingleTaskGP (version UNVERIFIED)")
    if arm == "gabo":
        return ("not recorded in the audit (kernel parameter beta, beta_min = 0.21)",
                "none recorded for the kernel parameter (outputscale Gamma(2, 0.15))")
    if arm == "matern_gabo":
        return "not recorded in the audit", "no lengthscale prior; nu prior Gamma(2, 0.1)"
    if arm.startswith("vanilla_bo_hvarfner"):
        return "gpytorch defaults (no explicit initialisation)", "none (gpytorch defaults, no explicit priors)"
    if arm == "lassobench:hvarfner_gp":
        return "not recorded in the audit", "LogNormal(1.4 + 0.5 ln D, sqrt(3)); constraint GreaterThan(1e-4)"
    if arm == "lassobench:papenmeier_msr":
        return "sqrt(D)/10", "none; no constraint"
    if arm == "lassobench:xu_standard_gp":
        return "sqrt(D)", "none; no constraint"
    if arm == "lassobench:linear_bo":
        return "as GuacaMol linear_bo: " + LIN_INIT, "as GuacaMol linear_bo: " + LIN_PRIOR
    if arm in ("nasbench:turbo", "nasbench:gp_bo"):
        return "not audited (see implementation)", "not audited (see implementation)"
    raise SystemExit(f"no derived lengthscale cells for row {arm!r} (kernel: {kernel[:60]})")


G_PAPER = "This paper"
G_GW = "GeoWalk variants"
G_VBO = "Vanilla BO GP: support x selection cells and controls"
G_SUP = "Surrogate-free support controls"
G_PAP = "Papenmeier et al. recipes"
G_XU = "Xu et al. Standard GP"
G_LIN = "Linear BO (Doumont et al.)"
G_OTHER = "Other baselines, in-repo harness"
G_STORED = "Stored baselines (not rerun)"
G_LASSO = "LassoBench (audited description of the stored runs)"
G_NAS = "NAS-Bench-101 (stored runs)"


def group_of(arm: str) -> str:
    if arm in ("geowalk", "cma_es_subspace", "random_search"):
        return G_PAPER
    if arm.startswith("geowalk_") or arm.startswith("cma_es_subspace_sd"):
        return G_GW
    if arm.startswith("hvarfner_scaled_gp") or arm in ("random_box_faithful", "random_boxdir", "random_search_prior"):
        return G_VBO
    if arm.startswith(("pca_sphere", "convex_hull", "ellipsoid")):
        return G_SUP
    if arm.startswith("papenmeier"):
        return G_PAP
    if arm.startswith("xu_"):
        return G_XU
    if arm.startswith("linear_bo"):
        return G_LIN
    if arm in ("turbo_box_faithful", "rembo", "hesbo", "gabo", "matern_gabo"):
        return G_OTHER
    if arm.startswith("lassobench:"):
        return G_LASSO
    if arm.startswith("nasbench:"):
        return G_NAS
    if arm in ("turbo", "baxus", "lolbo", "invbo", "cma_es (full-dimensional)", "saasbo", "cowboys",
               "vanilla_bo_hvarfner (budget appendix)"):
        return G_STORED
    raise SystemExit(f"no table group for row {arm!r}")


CR_OVERRIDES = {
    arm: {
        "kernel": ("as the stored description, except the lengthscale initialisation: at the camera-ready commit the "
                   "arm runs PapenmeierMSRSolver._make_gp with ls_init = 0.1*sqrt(D) = 1.6 "
                   f"(src/spherevo/gp_competitors.py:571-601 at {TAG})"),
        "lengthscale initialisation": (f"0.1 sqrt(D) = 1.6 in every dimension (src/spherevo/gp_competitors.py:577-586 "
                                       f"at {TAG}); the sqrt(D) value describes the stored files only"),
        "deviations": ("the stored files' deviation (lengthscale initialised to sqrt(D)) does not apply to the "
                       "camera-ready run; otherwise as papenmeier_msr_box_faithful ('MLE (scaled)': no RAASP, 10 "
                       "restarts, no outputscale prior) on this row's box"),
    } for arm in ("papenmeier_msr_norm_box", "papenmeier_msr_box")
}

out_rows = []
for r in OLD:
    arm, kernel = r[C["arm"]], r[C["kernel, priors, constraints, mean, fitting"]]
    if arm.startswith(("lassobench:", "nasbench:")):
        run = []
    else:
        run = RUN_ARMS.get(arm, [arm])
        if arm in S82:
            run = run + [arm + "__s82"]
    init, prior = derived(arm, kernel)
    ov = CR_OVERRIDES.get(arm)
    proof = ["lengthscale initialisation", "lengthscale prior"] + (["camera-ready overrides"] if ov else [])
    out_rows.append(list(r) + [";".join(run), init, prior, json.dumps(ov, sort_keys=True) if ov else "", ORIGIN_OLD,
                               ";".join(proof), group_of(arm)])

# ------------------------------------------------------------------------------------------------
# new rows (camera-ready and control-block arms)
# ------------------------------------------------------------------------------------------------
GP = "src/spherevo/gp_competitors.py"
SV = "src/spherevo/solver.py"
RB = "scripts/run_benchmark.py"
TB = "src/spherevo/turbo_box_faithful.py"
MAN = "scripts/cr/manifest.json"
AT = f"all at {TAG}"

COLD = ("5 ZINC molecules drawn by RandomState(seed).choice from the 10,000 SMILES of data/zinc_10k.txt (no replacement), "
        "encoded with the VAE posterior mean, scored directly by the oracle (no decode round trip); same 5 molecules "
        f"for every task and every solver at a given seed ({RB}:497-516)")
BUDGET = ("5 cold-start + 25 proposals = 30 oracle calls; no duplicate filtering (a re-proposed molecule is decoded and "
          "scored again and counts); NaN scores enter the history as -1; the queue accepts a cell only with "
          "n_evaluated = 30, 30 scores and budget = 30 (H2_successor D7)")
ENV_GPU = ("camera-ready runs: CUDA (panda, 2x L40S; RUN_RULES R-2), environment /scratch/prusek/envs/neurips: python "
           "3.12.14, torch 2.11.0 (CUDA 13.0), botorch 0.17.2, gpytorch 1.15.2, linear_operator 0.6.1, numpy 2.4.4, scipy "
           "1.17.1, cma 4.4.4, selfies 2.2.0, rdkit 2026.3.1, guacamol 0.5.2 (provenance.package_versions of the smoke "
           "cells at 20ad7d5 and fd8a03c)")
ENV_NP = ("the proposal step uses numpy only (decoding on GPU); " + ENV_GPU)
NO_STORED = "; no stored generation"


def stored_env(text: str) -> str:
    return "; stored generation: " + text
VERS = ("python_version, cuda_version and package_versions = numpy, torch, cma, scipy, selfies, botorch, gpytorch, "
        "linear_operator, rdkit, guacamol, LassoBench (scripts/cr/run_cell.py provenance, H2_successor D7 #13)")
GIT_CR = ("campaign cells: 14481cb (tag cr-runs-2026-10-12), git_dirty False, enforced by the queue (--expect-commit); "
          "smoke cells: 20ad7d5 clean")
GIT_CTL = "control-block cells: fd8a03c (tag cr-controls-2026-10-12), git_dirty False, enforced by the queue"
HV_KERNEL = ("SingleTaskGP, ConstantMean, ScaleKernel(Matern-5/2 ARD, D=256); lengthscale prior LogNormal(loc=sqrt(2)+0.5*ln D "
             "= 4.187, scale=sqrt(3)), constraint gpytorch default Positive, initial value softplus(0)=0.693 (prior acts "
             "through MAP only); outputscale: no prior; noise prior HalfNormal(1.0), GreaterThan(1e-5) "
             f"(HvarfnerScaledGPSolver._make_gp, {GP}:466-491, inherited unchanged; the priors are the device-safe "
             "subclasses :424-429, same distributions); fit_gpytorch_mll")
STD_OUT = ("outputs standardised by _standardize (mean, sd floored at 1e-8) and again by the default Standardize outcome "
           "transform of BoTorch 0.17.2 SingleTaskGP")
QLOGEI = "qLogExpectedImprovement (MC; BoTorch default sampler), best_f = max of standardised observations"
BOX_SRC = "results/precomputed/zinc_per_coord_bounds.json (per-coordinate 1st/99th percentiles of 50,000 encoded ZINC-250k molecules)"

NEW = [
    {
        "arm": "papenmeier_msr_published",
        "paper role / label": ("Papenmeier MSR with the PUBLISHED acquisition maximiser on the [p01,p99] box (RUN_RULES R-4; "
                               "the MSR row of Table 2, printed next to papenmeier_msr_box_faithful = 'MLE (scaled)')"),
        "implementing code (abbd61e unless stated)": (
            f"PapenmeierMSRPublishedSolver(_BoxFaithfulMixin, PapenmeierMSRSolver), {GP}:3321-3386; "
            "_propose_msr_published_on_unit_box :3257-3318; settings MSR_PUBLISHED_* :3240-3244, options :3247-3254; "
            "GP PapenmeierMSRSolver._make_gp :571-601; box _BoxFaithfulMixin.__init__ :1230-1273; registration "
            f"{RB}:173 (OUR_SOLVERS) and :1097-1139 (_create_solver); {MAN}:100 ({AT}; class added in 68292ee)"),
        "kernel, priors, constraints, mean, fitting": (
            "SingleTaskGP, ConstantMean, ScaleKernel(Matern-5/2 ARD); lengthscale initialised to 0.1*sqrt(D)=1.6 in all "
            "dims, Interval(0.01, 100), no lengthscale prior (MLE); outputscale: no prior; noise prior Gamma(1.1, 0.05), "
            f"GreaterThan(1e-5) ({GP}:571-601: the GP of papenmeier_msr_box_faithful, shared by identity); "
            "fit_gpytorch_mll with the default 3 Cholesky tries and no Adam fallback (F_msr_published D11)"),
        "input and outcome transforms": (
            f"inputs: per-coordinate (x - p01)/(p99 - p01) (width floor 1e-6), not clamped ({GP}:1275-1277, :3349-3350); "
            + STD_OUT + " (F_msr_published D10)"),
        "acquisition function and best_f": (
            "analytic LogExpectedImprovement, best_f = max of standardised observations, evaluated under "
            f"gpytorch.settings.cholesky_max_tries(9) ({GP}:3299-3300, :3371)"),
        "acquisition maximiser": (
            "optimize_acqf(q=1, num_restarts=5, raw_samples=512, options={batch_limit: 1, sample_around_best: True, "
            f"sample_around_best_sigma: 1e-3, maxiter: 2000}}) on [0,1]^256 ({GP}:3303-3310, :3240-3254): 512 scrambled "
            "Sobol points + 512 RAASP points, 5 starts by Boltzmann sampling on LogEI, L-BFGS-B from each "
            "(F_msr_published section 1)"),
        "candidate support / box definition and source": (
            f"[p01, p99] box: {BOX_SRC}, the box of papenmeier_msr_box_faithful ({GP}:1242-1273; codec dispatch "
            f"{RB}:1126-1139)"),
        "trust region / RAASP": (
            "no trust region; RAASP = BoTorch sample_around_best, sigma 1e-3: 256 points perturb every dimension, 256 "
            "perturb each dimension with probability min(1, 20/D); centres = the observations projected (clamped) onto "
            f"the box, acqf.X_baseline = X_unit.clamp(0, 1) ({GP}:3301), ranked by the GP posterior mean at the projected "
            "points; 'top 5 %' = max(1, round(0.05 n)) = 1 centre for every step of this protocol (n <= 29) "
            "(F_msr_published D4-D6)"),
        "cold start": COLD,
        "budget accounting": BUDGET,
        "fallback on GP failure": (
            "flat signal (sd < 1e-6) or _BOX_FALLBACK_ERRORS (GP_FIT_ERRORS + torch LinAlgError; a RuntimeError of "
            "optimize_acqf and a non-finite candidate are raised as ModelFittingError) -> uniform random point in the "
            f"same box, counted in surrogate_fallback_count ({GP}:3355-3364, :3375-3384, :91-104)"),
        "device and software environment": ENV_GPU + NO_STORED,
        "result files, seeds, status": (
            "camera-ready campaign only: results/guacamol/cr2026/papenmeier_msr_published/, 11 tasks x seeds 42-81 = 440 "
            "cells (manifest wave 2, timeout 21600 s); no pre-camera-ready generation exists (the legacy configuration "
            "points this arm at the July papenmeier_msr_raasp files and labels them as the RAASP trust-region variant)"),
        "versions recorded in provenance": VERS,
        "git_commit / git_dirty groups": GIT_CR,
        "original repository": (
            "github.com/lpapenme/understanding-hdbo-code @292e7251f1 (src/understanding_hdbo/bo.py:170-190; pins "
            "botorch==0.9.5); Papenmeier, Poloczek & Nardi, arXiv:2502.09198v2, App. B; not vendored"),
        "deviations from the published method": (
            "acquisition call as published; remaining differences (F_msr_published section 5, D1-D13): initial design "
            "of 5 encoded ZINC molecules instead of floor(3 sqrt(d)) = 48 Sobol points; 25 steps instead of 1000; domain "
            "= a rescaled [p01,p99] latent box that observations can lie outside of (191 of the 200 cold-start encodings "
            "of seeds 42-81), so the RAASP centres are projected onto the box and ranked at the projected points; a "
            "single RAASP centre at every step; GP: lengthscale Interval(0.01, 100) instead of Positive, no outputscale "
            "prior (reference Gamma(2, 0.15)), noise GreaterThan(1e-5) (reference GreaterThan(1e-4)), default "
            "Standardize outcome transform, fit with 3 Cholesky tries and no Adam fallback; fallback only on the "
            "numerical error set; botorch 0.17.2 / gpytorch 1.15.2 / torch 2.11.0 instead of 0.9.5 / 1.11 / 2.3.0"),
        "rerun at abbd61e reproduces existing files? (engineering item 2)": (
            "n/a: the class does not exist at abbd61e (added in 68292ee as an add-only patch; no stored files)"),
        "evidence": (
            "F_msr_published.md (patch, 7 tests, 6 mutations, BoTorch 0.9.5 vs 0.17.2 table); H_review_solvers.md section "
            "3.5 (the projection changed the RAASP centre in 3 of 109 step-1 cold starts); tests/test_msr_published.py; "
            f"code read at {TAG}"),
        "_run": ["papenmeier_msr_published"],
        "_init": f"0.1 sqrt(D) = 1.6 in every dimension ({GP}:577-586)",
        "_prior": f"none (maximum likelihood); constraint Interval(0.01, 100) ({GP}:582)",
    },
    {
        "arm": "turbo_box_faithful",
        "paper role / label": ("TuRBO-1 on the FIXED [p01,p99] box (RUN_RULES R-4; printed next to the submitted TuRBO row, "
                               "whose box is the running box of the observed latents)"),
        "implementing code (abbd61e unless stated)": (
            f"TurboBoxFaithfulSolver(_BoxFaithfulMixin, HvarfnerScaledGPSolver), {TB}:217-457 (settings :178-196, _make_gp "
            ":317-341, _tr_centre :345-348, trust_region_bounds :350-363, generate_candidates :365-394, thompson_select "
            ":396-403, next_candidate :410-457); TurboState imported from src/spherevo/turbo_baseline.py (:279); "
            f"registration {RB}:179 and :1097-1139; {MAN}:106 ({AT}; module added in 68292ee, docstring revised in "
            "20ad7d5 without a code change)"),
        "kernel, priors, constraints, mean, fitting": (
            "SingleTaskGP, ScaleKernel(Matern-5/2 ARD); lengthscale Interval(0.005, 4.0), outputscale Interval(0.05, 20.0), "
            f"noise Interval(1e-8, 1e-3); no priors ({TB}:190-192, :317-341): the submitted wrapper's GP "
            "(turbo_baseline.py:232-240), kept unchanged (decision D2 of H2_successor); fit_gpytorch_mll inside "
            f"gpytorch.settings.max_cholesky_size(inf) ({TB}:434-436, choice C5)"),
        "input and outcome transforms": (
            "inputs: per-coordinate (x - p01)/(p99 - p01) (width floor 1e-6), GP training inputs not clamped (choice C2); "
            + STD_OUT + " (choice C3)"),
        "acquisition function and best_f": (
            f"Thompson sampling: MaxPosteriorSampling(replacement=False), one posterior sample per step ({TB}:396-403)"),
        "acquisition maximiser": (
            f"argmax of the posterior sample over min(100 D, 5000) = 5000 candidates ({TB}:199-205, :365-394)"),
        "candidate support / box definition and source": (
            f"trust region inside the fixed [p01, p99] box of {BOX_SRC} (the unit cube of _BoxFaithfulMixin): each "
            "candidate is the clamped incumbent with every coordinate replaced, with probability min(1, 20/D) and at "
            "least one coordinate, by a scrambled-Sobol value inside the lengthscale-weighted trust region clamped to "
            f"[0,1] ({TB}:345-394)"),
        "trust region / RAASP": (
            "TuRBO-1: length 0.8, length_min 0.5^7, length_max 1.6, success tolerance 3, failure tolerance "
            f"ceil(max(4/b, D/b)) = 256 with b = 1 ({TB}:178-185, :208-214); doubling after 3 consecutive improvements, "
            "halving after 256 consecutive failures, restart below length_min, so the failure and restart branches are "
            "unreachable in 25 steps (choice C7); centre = incumbent clamped into the box (choice C1); perturbation "
            "probability min(1, 20/D) = 0.078; no RAASP in the MSR sense"),
        "cold start": COLD,
        "budget accounting": (
            "5 cold-start + 25 proposals = 30 oracle calls; every proposal is scored and counted through "
            "StepByStepSolver.step: a repeated molecule is re-scored (oracle cache), enters the GP data and counts as a "
            "trust-region failure, whereas the submitted wrapper skipped such a step without an oracle call; a NaN score "
            "enters the history and the trust-region state as -1 (choice C9)"),
        "fallback on GP failure": (
            "flat signal (sd < 1e-6), _BOX_FALLBACK_ERRORS or a non-finite Thompson candidate -> uniform random point "
            f"in the fixed box, counted in surrogate_fallback_count ({TB}:405-408, :423-431, :441-457; choice C10)"),
        "device and software environment": ENV_GPU + NO_STORED,
        "result files, seeds, status": (
            "camera-ready campaign only: results/guacamol/cr2026/turbo_box_faithful/, 11 tasks x seeds 42-81 = 440 cells "
            "(manifest wave 3, timeout 21600 s); no pre-camera-ready generation"),
        "versions recorded in provenance": VERS,
        "git_commit / git_dirty groups": GIT_CR,
        "original repository": (
            "Eriksson et al., NeurIPS 2019 (arXiv:1910.01739); github.com/uber-research/TuRBO (turbo/turbo_1.py, "
            "turbo/gp.py); BoTorch TuRBO-1 tutorial (v0.17.2); re-implemented in the in-repo harness, not vendored"),
        "deviations from the published method": (
            "GP constraints are the submitted wrapper's mix: lengthscale Interval(0.005, 4.0) and noise Interval(1e-8, "
            "1e-3) from the BoTorch tutorial, outputscale Interval(0.05, 20.0) from the paper; the paper uses lengthscale "
            "[0.005, 2.0] and noise [5e-4, 0.1]; initial design of 5 encoded ZINC molecules and 25 steps instead of the "
            "published design and budget; domain = fixed [p01,p99] latent box with the trust-region centre clamped into "
            "it (C1); shrink and restart are inactive at this budget (C7); a restart, if it fired, would keep the "
            "history (C8); unseeded scrambled Sobol from the global torch stream (C4). Against the submitted TuRBO row: "
            "fixed box instead of the running box of the observed latents, and duplicates are evaluated and counted"),
        "rerun at abbd61e reproduces existing files? (engineering item 2)": (
            "n/a: the module does not exist at abbd61e (added in 68292ee); its smoke cells are bit-identical between "
            "68292ee and 20ad7d5 (H2_successor D2)"),
        "evidence": (
            "H_commit.md section 3 (choices C1-C10, 21 tests, real-data diagnostic rano s42: 4/256 incumbent coordinates "
            "outside the box, about 20 perturbed coordinates per candidate); H2_successor.md D2; H_review_solvers.md "
            f"section 4.4; tests/test_turbo_box_faithful.py; code read at {TAG}"),
        "_run": ["turbo_box_faithful"],
        "_init": (f"not set in the code ({TB}:329-336): gpytorch's default raw value 0 under Interval(0.005, 4.0), i.e. "
                  "2.0025 (derived from the constraint transform, not measured)"),
        "_prior": f"none; constraint Interval(0.005, 4.0) ({TB}:190, :333)",
    },
    {
        "arm": "hvarfner_scaled_gp_box_argmax_raw",
        "paper role / label": ("Control block C1 (RUN_RULES amendment 2026-10-12): the box x candidate-set cell with the GP "
                               "fitted on RAW latents (input-transform control)"),
        "implementing code (abbd61e unless stated)": (
            f"HvarfnerScaledGPBoxArgmaxRawSolver(HvarfnerScaledGPBoxArgmaxSolver), {GP}:1405-1498 (_candidates_from_unit "
            ":1459-1461, next_candidate :1463-1498); the parent step _BoxFaithfulArgmaxMixin.next_candidate :1354-1390; "
            f"GP HvarfnerScaledGPSolver._make_gp :466-491; registration {RB}:199 and :1200-1207; {MAN}:119 ({AT}; class "
            "added in fd8a03c)"),
        "kernel, priors, constraints, mean, fitting": HV_KERNEL,
        "input and outcome transforms": (
            f"inputs: NONE, the GP is fitted on raw latents as in the shell arm hvarfner_scaled_gp ({GP}:1466, :1481); "
            "the parent arm hvarfner_scaled_gp_box_argmax fits on (x - p01)/(p99 - p01); " + STD_OUT),
        "acquisition function and best_f": QLOGEI + f" ({GP}:1488)",
        "acquisition maximiser": (
            "argmax over 2048 fresh candidates per step: u = torch.rand(2048, D) mapped to raw coordinates p01 + u * width "
            f"before qLogEI scores them ({GP}:1484-1492): the same points of the latent space the parent arm scores in "
            "unit-cube coordinates"),
        "candidate support / box definition and source": (
            f"uniform in the [p01, p99] box of {BOX_SRC}, width floor 1e-6; class default bounds, no codec dispatch "
            f"({GP}:1242-1273; {RB}:1200-1207)"),
        "trust region / RAASP": "none",
        "cold start": COLD,
        "budget accounting": BUDGET,
        "fallback on GP failure": (
            "flat signal (sd < 1e-6) or _BOX_FALLBACK_ERRORS -> one uniform point in the box, "
            f"_candidates_from_unit(torch.rand(1, D)), counted in surrogate_fallback_count ({GP}:1472-1479, :1493-1496)"),
        "device and software environment": ENV_GPU + NO_STORED,
        "result files, seeds, status": (
            "control block only: results/guacamol/cr2026ctl/hvarfner_scaled_gp_box_argmax_raw/ in the control clone, 11 "
            "tasks x seeds 42-81 = 440 cells (manifest wave c1, timeout 7200 s)"),
        "versions recorded in provenance": VERS,
        "git_commit / git_dirty groups": GIT_CTL,
        "original repository": "ours (control; no published counterpart)",
        "deviations from the published method": (
            "not a published recipe: finite-candidate argmax in the box with the GP on raw latents; it differs from "
            "hvarfner_scaled_gp_box_argmax only in the GP's input coordinates (with p01 = 0 and p99 = 1 the two arms "
            "compute bit-identical trajectories, tests/test_control_block.py)"),
        "rerun at abbd61e reproduces existing files? (engineering item 2)": (
            "n/a: class added in fd8a03c; the code shared with the main campaign is bit-identical at fd8a03c (12/12 "
            "cell comparisons against 20ad7d5 / 14481cb, CTL_build section 5)"),
        "evidence": (
            "CTL_build.md section 2.1 (source diff of the two next_candidate methods, 15 mutations); CTL_check.md "
            f"section 2 (spies on the real 256-D box, 11 mutations); tests/test_control_block.py; code read at {TAG}"),
        "_run": ["hvarfner_scaled_gp_box_argmax_raw"],
        "_init": HV_INIT + f" ({GP}:472-476)",
        "_prior": HV_PRIOR + f" ({GP}:468-470)",
    },
    {
        "arm": "hvarfner_scaled_gp_boxdir_argmax",
        "paper role / label": ("Control block C2a (RUN_RULES amendment 2026-10-12): [p01,p99]-box draws renormalised to the "
                               "run's shell radius, GP on raw latents, qLogEI argmax (anisotropy control, GP arm)"),
        "implementing code (abbd61e unless stated)": (
            f"HvarfnerScaledGPBoxDirArgmaxSolver(HvarfnerScaledGPBoxArgmaxRawSolver), {GP}:1501-1530; the only "
            "overridden method is _candidates_from_unit :1527-1530; the step is C1's next_candidate :1463-1498; "
            f"registration {RB}:200 and :1209-1214; {MAN}:120 ({AT}; class added in fd8a03c)"),
        "kernel, priors, constraints, mean, fitting": HV_KERNEL,
        "input and outcome transforms": (
            f"inputs: none (raw latents, as C1 and the shell arm; {GP}:1466, :1481); " + STD_OUT),
        "acquisition function and best_f": QLOGEI + f" ({GP}:1488)",
        "acquisition maximiser": (
            "argmax over 2048 fresh candidates per step: the C1 draw p01 + u * width, each row renormalised to the "
            f"shell radius ({GP}:1484-1492, :1527-1530)"),
        "candidate support / box definition and source": (
            "[p01, p99]-box directions at the run's shell radius: x = p01 + u * width, then x / ||x|| * mean_norm, with "
            f"mean_norm the mean L2 norm of the run's 5 cold-start latents ({GP}:1527-1530; {SV}:78-80), i.e. the norm of "
            "the shell arm and the direction of a box draw (anisotropic; not the isotropic cube of "
            "hvarfner_scaled_gp_angular_box)"),
        "trust region / RAASP": "none",
        "cold start": COLD,
        "budget accounting": BUDGET,
        "fallback on GP failure": (
            "flat signal (sd < 1e-6) or _BOX_FALLBACK_ERRORS -> one renormalised box draw (the fallback stays on this "
            f"arm's support), counted in surrogate_fallback_count ({GP}:1472-1479, :1493-1496, :1527-1530)"),
        "device and software environment": ENV_GPU + NO_STORED,
        "result files, seeds, status": (
            "control block only: results/guacamol/cr2026ctl/hvarfner_scaled_gp_boxdir_argmax/ in the control clone, 11 "
            "tasks x seeds 42-81 = 440 cells (manifest wave c1, timeout 7200 s)"),
        "versions recorded in provenance": VERS,
        "git_commit / git_dirty groups": GIT_CTL,
        "original repository": "ours (control; no published counterpart)",
        "deviations from the published method": (
            "not a published recipe (anisotropy control); a zero-norm draw has probability zero and is not "
            "special-cased"),
        "rerun at abbd61e reproduces existing files? (engineering item 2)": (
            "n/a: class added in fd8a03c; determinism rerun of the smoke cell identical (CTL_build section 5)"),
        "evidence": (
            "CTL_build.md section 2.2; CTL_check.md section 2 (proposal norms of the shell arm and of this arm agree to "
            f"1 ulp); tests/test_control_block.py; code read at {TAG}"),
        "_run": ["hvarfner_scaled_gp_boxdir_argmax"],
        "_init": HV_INIT + f" ({GP}:472-476)",
        "_prior": HV_PRIOR + f" ({GP}:468-470)",
    },
    {
        "arm": "random_boxdir",
        "paper role / label": ("Control block C2b (RUN_RULES amendment 2026-10-12): [p01,p99]-box directions at the run's "
                               "shell radius, no surrogate (anisotropy control, surrogate-free twin of C2a)"),
        "implementing code (abbd61e unless stated)": (
            f"RandomBoxDirSolver(RandomBoxFaithfulSolver), {SV}:1654-1679 (next_candidate :1677-1679); registration "
            f"{RB}:201 and :737-759; {MAN}:121 ({AT}; class added in fd8a03c)"),
        "kernel, priors, constraints, mean, fitting": "none (surrogate-free)",
        "input and outcome transforms": "none",
        "acquisition function and best_f": (
            "none: one draw per step, z = p01 + u * max(p99 - p01, 1e-6) with u ~ U[0,1)^D (np.random.rand(1, D)), "
            f"renormalised to z / ||z|| * mean_norm ({SV}:1649-1651, :1677-1679)"),
        "acquisition maximiser": "n/a",
        "candidate support / box definition and source": (
            "[p01, p99]-box directions at the run's shell radius (the candidate distribution of "
            f"hvarfner_scaled_gp_boxdir_argmax, one draw per step); bounds {BOX_SRC}, codec dispatch as "
            f"random_box_faithful ({RB}:746-759)"),
        "trust region / RAASP": "none",
        "cold start": COLD,
        "budget accounting": BUDGET,
        "fallback on GP failure": "n/a (no GP; the result file records surrogate_fallback_count = 0)",
        "device and software environment": ENV_NP + NO_STORED,
        "result files, seeds, status": (
            "control block only: results/guacamol/cr2026ctl/random_boxdir/ in the control clone, 11 tasks x seeds 42-81 = "
            "440 cells (manifest wave c1, timeout 3600 s)"),
        "versions recorded in provenance": VERS,
        "git_commit / git_dirty groups": GIT_CTL,
        "original repository": "ours (control; no published counterpart)",
        "deviations from the published method": "n/a",
        "rerun at abbd61e reproduces existing files? (engineering item 2)": (
            "n/a: class added in fd8a03c; under the same NumPy seed every proposal equals the renormalised "
            "random_box_faithful proposal (tested, CTL_build section 2.3)"),
        "evidence": f"CTL_build.md section 2.3; CTL_check.md section 2; tests/test_control_block.py; code read at {TAG}",
        "_run": ["random_boxdir"],
        "_init": NA,
        "_prior": NA,
    },
    {
        "arm": "random_box_faithful",
        "paper role / label": ("Surrogate-free box control: uniform in the [p01,p99] box (decomposition cell box x no "
                               "surrogate; RUN_RULES amendment 2026-10-11)"),
        "implementing code (abbd61e unless stated)": (
            f"RandomBoxFaithfulSolver, {SV}:1596-1651 (next_candidate :1649-1651; load_faithful_box_bounds :1551-1593); "
            f"registration {RB}:196 and :714-735; {MAN}:90 ({AT}; ported in 20ad7d5 from p1-blocks commit 4780475, code "
            "AST-identical)"),
        "kernel, priors, constraints, mean, fitting": "none (surrogate-free)",
        "input and outcome transforms": "none",
        "acquisition function and best_f": (
            "none: one uniform draw per step, z = p01 + u * max(p99 - p01, 1e-6) with u ~ U[0,1)^D (np.random.rand(1, D)): "
            f"the box GP arms' fallback formula taken on every step ({SV}:1649-1651)"),
        "acquisition maximiser": "n/a",
        "candidate support / box definition and source": (
            f"uniform in the [p01, p99] box of {BOX_SRC}, width floor 1e-6 ({SV}:1642-1647): the same bounds file and "
            f"loader semantics as _BoxFaithfulMixin; codec dispatch as the *_box_faithful arms ({RB}:722-735)"),
        "trust region / RAASP": "none",
        "cold start": COLD,
        "budget accounting": BUDGET,
        "fallback on GP failure": "n/a (no GP; the result file records surrogate_fallback_count = 0)",
        "device and software environment": ENV_NP + stored_env(
            "September p1_blocks files (kajman, batch runner, 4780475 clean); their environment was not audited here"),
        "result files, seeds, status": (
            "camera-ready campaign: results/guacamol/cr2026/random_box_faithful/, 11 tasks x seeds 42-81 = 440 cells "
            "(manifest wave 1, timeout 3600 s); control block C3: results/guacamol/cr2026ctl/random_box_faithful__s82/, "
            "seeds 82-121, 440 cells (wave c2); stored September generation: "
            "/data2/prusek/NeurIPS-p1/results/guacamol/p1_blocks/ours_random_box_faithful_*_256d.json (batch runner, "
            "4780475 clean, kajman; 21 tasks x 40 seeds)"),
        "versions recorded in provenance": VERS,
        "git_commit / git_dirty groups": GIT_CR + "; C3 cells: fd8a03c clean; stored September files: 4780475 clean",
        "original repository": "ours (control)",
        "deviations from the published method": "n/a",
        "rerun at abbd61e reproduces existing files? (engineering item 2)": (
            "the class does not exist at abbd61e; at 20ad7d5 the per-cell smoke cells (rano s42, s43) are identical to "
            "the September p1_blocks files (all 30 scores; H2_successor section 4.4), which is valid across runners "
            "because the arm draws only from NumPy's global generator"),
        "evidence": (
            "H2_successor.md D6 and section 4.4 (AST comparison with 4780475, 2/2 cells identical, determinism rerun); "
            f"tests/test_random_box_faithful.py; code read at {TAG}"),
        "_run": ["random_box_faithful", "random_box_faithful__s82", "p1_random_box_faithful"],
        "_init": NA,
        "_prior": NA,
    },
    {
        "arm": "geowalk_one_fifth",
        "paper role / label": ("GeoWalk with a success-based angle rule in the refinement phase (the adaptive-step sentence "
                               "of Section 3.3; RUN_RULES amendments 2026-10-11 and 2026-10-12)"),
        "implementing code (abbd61e unless stated)": (
            f"GeoWalkOneFifthSolver(GeoWalkSolver), {SV}:1837-1946 (_angle :1926-1929, post_update :1931-1946); "
            f"GeoWalkSolver :1687-1834; registration {RB}:135 and :870-879; {MAN}:86 ({AT}; class added in 20ad7d5)"),
        "kernel, priors, constraints, mean, fitting": "none (surrogate-free)",
        "input and outcome transforms": "none",
        "acquisition function and best_f": (
            "none: steps 0-15 uniform on the shell (GeoWalk's exploration, int(0.64 * 25) = 16 steps); steps 16-24 greedy "
            "geodesic step from the incumbent, random tangent direction, with a success-based angle: first refinement "
            "angle 0.30 rad, multiplied by 1.2 after a refinement step that improved the incumbent (score > best finite "
            "score before it + 1e-10) and by 0.85 otherwise, clamped to [0.05, 0.50] rad; a NaN score is a failure "
            f"({SV}:1903-1908, :1931-1946; choices O1-O6 in the class docstring)"),
        "acquisition maximiser": "n/a",
        "candidate support / box definition and source": (
            "uniform directions on S^255 scaled to r = mean L2 norm of THIS run's 5 cold-start latents "
            f"({SV}:78-80), as geowalk"),
        "trust region / RAASP": "none",
        "cold start": COLD,
        "budget accounting": BUDGET,
        "fallback on GP failure": "n/a",
        "device and software environment": ENV_NP + NO_STORED,
        "result files, seeds, status": (
            "camera-ready campaign only: results/guacamol/cr2026/geowalk_one_fifth/, 11 tasks x seeds 42-81 = 440 cells "
            "(manifest wave 1, timeout 3600 s); no earlier run in this harness"),
        "versions recorded in provenance": VERS,
        "git_commit / git_dirty groups": GIT_CR,
        "original repository": "ours",
        "deviations from the published method": (
            "n/a (ours). Despite the registry name it is not a 1/5 rule: with x1.2 / x0.85 the angle is stationary at a "
            "success rate of ln(1/0.85)/ln(1.2/0.85) = 0.47 (O6)"),
        "rerun at abbd61e reproduces existing files? (engineering item 2)": (
            "n/a: class added in 20ad7d5, no stored files; determinism rerun of rano s42 identical (H2_successor "
            "section 4.6); cold start and exploration are bit-identical to geowalk under the same seed (tested)"),
        "evidence": f"H2_successor.md D6; tests/test_geowalk_one_fifth.py (7 tests); code read at {TAG}",
        "_run": ["geowalk_one_fifth"],
        "_init": NA,
        "_prior": NA,
    },
    {
        "arm": "hvarfner_scaled_gp_sphere{05,15,20}",
        "paper role / label": ("GP radius dose-response: Vanilla BO GP with candidate-set argmax on spheres of 0.5, 1.5 and "
                               "2 times the cold-start radius (RUN_RULES amendment 2026-10-12)"),
        "implementing code (abbd61e unless stated)": (
            f"_HvarfnerScaledGPSphereScaledSolver(HvarfnerScaledGPSolver), {GP}:1701-1745; HvarfnerScaledGPSphere05Solver "
            ":1769-1771, Sphere15 :1774-1776, Sphere20 :1779-1781 (sphere_radius_multiplier 0.5 / 1.5 / 2.0); "
            f"_propose_qlogei_on_sphere :288-312; registration {RB}:159-161 and :1143-1151; {MAN}:92-94 ({AT}; manifest "
            "arms added in 14481cb)"),
        "kernel, priors, constraints, mean, fitting": HV_KERNEL,
        "input and outcome transforms": f"inputs: none (raw latents; {GP}:1711, :1727); " + STD_OUT,
        "acquisition function and best_f": QLOGEI + f" ({GP}:308)",
        "acquisition maximiser": (
            f"argmax over 2048 fresh uniform points on the scaled sphere per step, no gradient step ({GP}:288-312, "
            ":1732-1734)"),
        "candidate support / box definition and source": (
            "uniform directions on S^255 scaled to radius = multiplier x mean_norm, with mean_norm the mean L2 norm of the "
            f"run's 5 cold-start latents and multiplier 0.5 / 1.5 / 2.0 ({GP}:1709, :1732-1734; {SV}:78-80); the "
            "multiplier is the class attribute encoded in the solver name, not a field of the result file"),
        "trust region / RAASP": "none",
        "cold start": COLD,
        "budget accounting": BUDGET,
        "fallback on GP failure": (
            "flat signal (sd < 1e-6) or GP_FIT_ERRORS (NotPSDError, NanError, ModelFittingError, numpy LinAlgError) -> "
            f"one uniform random point on the same scaled sphere, counted in surrogate_fallback_count ({GP}:1716-1724, "
            ":1735-1743)"),
        "device and software environment": ENV_GPU + stored_env(
            "tulen (A100 / RTX A5000), 2026-04-29/30, selfies 2.1.2, Python 3.12.12 (H3_final section 3.4); other "
            "versions not audited here"),
        "result files, seeds, status": (
            "camera-ready campaign: results/guacamol/cr2026/hvarfner_scaled_gp_sphere{05,15,20}/, 11 tasks x seeds 42-81 "
            "= 440 cells each (manifest wave 1, timeout 7200 s); stored generation: "
            "results/guacamol/spherevo/ours_hvarfner_scaled_gp_sphere{05,15,20}_*_128d.json, 440 files each, tulen, "
            "2026-04-29/30"),
        "versions recorded in provenance": VERS + "; stored files: selfies 2.1.2, Python 3.12.12 (H3_final section 3.4)",
        "git_commit / git_dirty groups": GIT_CR + "; stored files: f299e5e dirty (tulen), one sphere05 file at 5d71a49 dirty "
                                                  "(GEN_v2_verify section 2)",
        "original repository": "github.com/hvarfner/vanilla_bo_in_highdim (not vendored); the radius multiples are ours",
        "deviations from the published method": (
            "not a published recipe: the Vanilla BO shell arm (see hvarfner_scaled_gp for its deviations from Hvarfner "
            "et al.) with the candidate sphere at 0.5, 1.5 or 2 times the cold-start radius"),
        "rerun at abbd61e reproduces existing files? (engineering item 2)": (
            "partly (at 14481cb, H3_final section 3.4): rano s42: sphere15 and sphere20 identical to the stored files (30/30 "
            "scores), sphere05 first differs at evaluation 19 with the same best at 30; rano s43: the three arms first "
            "differ at evaluations 10, 14 and 9; all six cells share evaluations 1-8; the late divergence is not "
            "attributed (tulen GPUs, selfies 2.1.2 or the dirty tree)"),
        "evidence": f"H3_final.md section 3.4 and item j; tests/test_sphere_radius_arms.py; code read at {TAG}",
        "_run": ["hvarfner_scaled_gp_sphere05", "hvarfner_scaled_gp_sphere15", "hvarfner_scaled_gp_sphere20"],
        "_init": HV_INIT + f" ({GP}:472-476)",
        "_prior": HV_PRIOR + f" ({GP}:468-470)",
    },
]

# ---- further camera-ready arms the audit did not cover (amendments of 2026-10-11) -------------------------
GW_BASE = ("uniform directions on S^255 scaled to r = mean L2 norm of THIS run's 5 cold-start latents "
           f"({SV}:78-80), as geowalk")
NEW += [
    {
        "arm": "geowalk_const{060,035,015} / geowalk_explore{8,24}",
        "paper role / label": ("GeoWalk schedule variants: constant angle 0.60 / 0.35 / 0.15, and 8 or 24 exploration "
                               "steps (Appendix B and the constant-angle sentence of Section 3.3; RUN_RULES amendment "
                               "2026-10-11)"),
        "implementing code (abbd61e unless stated)": (
            f"GeoWalkSolver, {SV}:1687-1834 (_angle :1760-1767), with the keyword arguments of _GEOWALK_ABLATIONS, "
            f"{RB}:838-868 (constant angle: angle_init = angle_final; exploration: explore_steps = 8 or 24); "
            f"registration {RB}:128-132 (OUR_SOLVERS) and :861-868; {MAN}:81-85 ({AT})"),
        "kernel, priors, constraints, mean, fitting": "none (surrogate-free)",
        "input and outcome transforms": "none",
        "acquisition function and best_f": (
            "none: GeoWalk (uniform shell exploration, then greedy geodesic steps from the incumbent with a random "
            "tangent direction). const060 / const035 / const015: 16 exploration steps, then a constant angle of 0.60 / "
            "0.35 / 0.15 rad; explore8: 8 exploration and 17 refinement steps, angle 0.60 -> 0.15 rad linearly; "
            f"explore24: 24 exploration steps and one refinement step at 0.60 rad ({RB}:838-843; {SV}:1760-1767)"),
        "acquisition maximiser": "n/a",
        "candidate support / box definition and source": GW_BASE,
        "trust region / RAASP": "none",
        "cold start": COLD,
        "budget accounting": BUDGET,
        "fallback on GP failure": "n/a",
        "device and software environment": ENV_NP + stored_env(
            "geowalk_ablations files (kajman, 6244f20 dirty); their environment was not audited here"),
        "result files, seeds, status": (
            "camera-ready campaign: results/guacamol/cr2026/<arm>/, 11 tasks x seeds 42-81 = 440 cells per arm "
            "(manifest wave 1, timeout 3600 s); stored generation: results/guacamol/geowalk_ablations/ (6244f20 dirty, "
            "kajman)"),
        "versions recorded in provenance": VERS,
        "git_commit / git_dirty groups": GIT_CR + "; stored files: 6244f20 dirty",
        "original repository": "ours",
        "deviations from the published method": "n/a",
        "rerun at abbd61e reproduces existing files? (engineering item 2)": (
            "at 20ad7d5 (rano s42, s43): 10/10 cells differ from the stored geowalk_ablations files only by oracle "
            "round-off (2-6 of 30 scores differ by at most 5.6e-17, best at 30 equal; H2_successor section 4.4)"),
        "evidence": f"H2_successor.md D6 and section 4.4; code read at {TAG}",
        "_run": ["geowalk_const060", "geowalk_const035", "geowalk_const015", "geowalk_explore8", "geowalk_explore24"],
        "_init": NA,
        "_prior": NA,
    },
    {
        "arm": "cma_es_subspace_sd{8,64}",
        "paper role / label": ("GeoCMA with subspace dimension 8 or 64 instead of 16 (the D-tilde paragraph; RUN_RULES "
                               "amendment 2026-10-11, exception to R-1)"),
        "implementing code (abbd61e unless stated)": (
            f"CMAESSubspaceSolver, {SV}:271 ff., created by {RB}:617-625 with subspace_dim taken from the run arguments; "
            f"the arms run solver cma_es_subspace with run_overrides {{subspace_dim: 8}} / {{subspace_dim: 64}} "
            f"({MAN}:87-88; allowed overrides scripts/cr/run_cell.py:109) ({AT})"),
        "kernel, priors, constraints, mean, fitting": "none (CMA-ES)",
        "input and outcome transforms": "none",
        "acquisition function and best_f": (
            "none: as cma_es_subspace (see that row), with the CMA-ES search space R^8 or R^64 instead of R^16"),
        "acquisition maximiser": "n/a",
        "candidate support / box definition and source": (
            "as cma_es_subspace: a random linear subspace of dimension 8 or 64 mapped to the shell at the run's "
            "cold-start radius (see the cma_es_subspace row for the mapping)"),
        "trust region / RAASP": "none",
        "cold start": COLD,
        "budget accounting": BUDGET,
        "fallback on GP failure": "n/a",
        "device and software environment": ENV_NP + stored_env(
            "March spherevo files (9c042c2 dirty); their environment was not audited here"),
        "result files, seeds, status": (
            "camera-ready campaign: results/guacamol/cr2026/cma_es_subspace_sd8/ and .../cma_es_subspace_sd64/, files "
            "ours_cma_es_subspace_<task>_s<seed>_128d_sd8.json / _sd64.json, 440 cells each (manifest wave 1); the "
            "files record subspace_dim and provenance.run_overrides; stored generation: results/guacamol/spherevo/ "
            "(9c042c2 dirty, March 19; 4 tasks x 10 seeds, stochastic decoding)"),
        "versions recorded in provenance": VERS,
        "git_commit / git_dirty groups": GIT_CR + "; stored files: 9c042c2 dirty",
        "original repository": "ours (CMA-ES: pycma)",
        "deviations from the published method": "n/a",
        "rerun at abbd61e reproduces existing files? (engineering item 2)": (
            "NO, as expected: at 20ad7d5 8/8 cells differ from the March files from evaluation 1-3 on (the March files "
            "predate the deterministic decoder and the 80-token cap; H2_successor section 4.4)"),
        "evidence": f"H2_successor.md D6 and section 4.4; RUN_RULES amendment log; code read at {TAG}",
        "_run": ["cma_es_subspace_sd8", "cma_es_subspace_sd64"],
        "_init": NA,
        "_prior": NA,
    },
    {
        "arm": "random_search_prior",
        "paper role / label": "Surrogate-free control: candidates from the VAE prior N(0, I_D) (RUN_RULES amendment 2026-10-11)",
        "implementing code (abbd61e unless stated)": (
            f"RandomSearchPriorSolver, {SV}:1524-1542 (next_candidate :1541-1542); registration {RB}:195 and :707-712; "
            f"{MAN}:89 ({AT})"),
        "kernel, priors, constraints, mean, fitting": "none (surrogate-free)",
        "input and outcome transforms": "none",
        "acquisition function and best_f": f"none: one draw z ~ N(0, I_D) per step, np.random.randn(1, D), not normalised ({SV}:1541-1542)",
        "acquisition maximiser": "n/a",
        "candidate support / box definition and source": "the VAE prior N(0, I_256): norms concentrate near sqrt(D), not at the shell radius",
        "trust region / RAASP": "none",
        "cold start": COLD,
        "budget accounting": BUDGET,
        "fallback on GP failure": "n/a (no GP)",
        "device and software environment": ENV_NP + stored_env(
            "spherevo files (kajman, f299e5e dirty); their environment was not audited here"),
        "result files, seeds, status": (
            "camera-ready campaign: results/guacamol/cr2026/random_search_prior/, 440 cells (manifest wave 1, timeout "
            "3600 s); stored generation: results/guacamol/spherevo/ (f299e5e dirty, kajman)"),
        "versions recorded in provenance": VERS,
        "git_commit / git_dirty groups": GIT_CR + "; stored files: f299e5e dirty",
        "original repository": "ours (control)",
        "deviations from the published method": "n/a",
        "rerun at abbd61e reproduces existing files? (engineering item 2)": (
            "at 20ad7d5: 2/2 smoke cells (rano s42, s43) identical to the stored spherevo files (H2_successor "
            "section 4.4)"),
        "evidence": f"H2_successor.md section 4.4; code read at {TAG}",
        "_run": ["random_search_prior"],
        "_init": NA,
        "_prior": NA,
    },
    {
        "arm": "hvarfner_scaled_gp_norm_box_argmax",
        "paper role / label": ("[min,max] twin of the factorial's box x candidate-set cell: Vanilla BO GP, qLogEI argmax "
                               "over uniform draws in the [min,max] box (RUN_RULES amendment 2026-10-11)"),
        "implementing code (abbd61e unless stated)": (
            f"HvarfnerScaledGPNormBoxArgmaxSolver(HvarfnerScaledGPSolver), {GP}:1931-1935 -> "
            f"_norm_box_argmax_next_candidate :1885-1928; bounds _load_zinc_per_coord_bounds(mode='minmax') :183-225; "
            f"GP HvarfnerScaledGPSolver._make_gp :466-491; registration {RB}:167 and :1284-1286; {MAN}:115 ({AT})"),
        "kernel, priors, constraints, mean, fitting": HV_KERNEL,
        "input and outcome transforms": (
            f"inputs: per-coordinate (x - min)/(max - min), spans below 1e-3 widened to 1e-3 around the midpoint "
            f"({GP}:1894-1898, :180, :215-223); " + STD_OUT),
        "acquisition function and best_f": QLOGEI + f" ({GP}:1918)",
        "acquisition maximiser": f"argmax over 2048 fresh uniform points in [0,1]^256 per step ({GP}:1915-1922)",
        "candidate support / box definition and source": (
            f"[min, max] box: per-coordinate min and max of {BOX_SRC.replace('per-coordinate 1st/99th percentiles', 'min / max')}; "
            "always the SELFIES bounds (no codec dispatch)"),
        "trust region / RAASP": "none",
        "cold start": COLD,
        "budget accounting": BUDGET,
        "fallback on GP failure": (
            "flat signal (sd < 1e-6) or _BOX_FALLBACK_ERRORS -> uniform random point in the same box "
            f"(_random_in_normalized_box), counted in surrogate_fallback_count ({GP}:1904-1910, :1925-1928)"),
        "device and software environment": ENV_GPU + NO_STORED,
        "result files, seeds, status": (
            "camera-ready campaign only: results/guacamol/cr2026/hvarfner_scaled_gp_norm_box_argmax/, 440 cells "
            "(manifest wave 3b, timeout 7200 s); no stored generation"),
        "versions recorded in provenance": VERS,
        "git_commit / git_dirty groups": GIT_CR,
        "original repository": "ours (factorial control)",
        "deviations from the published method": "not a published recipe: finite-candidate argmax on the [min,max] box",
        "rerun at abbd61e reproduces existing files? (engineering item 2)": (
            "n/a: no stored files; determinism rerun of rano s42 identical (H2_successor section 4.6)"),
        "evidence": f"H2_successor.md D6 and section 4.6; code read at {TAG}",
        "_run": ["hvarfner_scaled_gp_norm_box_argmax"],
        "_init": HV_INIT + f" ({GP}:472-476)",
        "_prior": HV_PRIOR + f" ({GP}:468-470)",
    },
    {
        "arm": "papenmeier_msr_published_norm_box",
        "paper role / label": ("Papenmeier MSR with the published acquisition maximiser on the [min,max] box (manifest "
                               "wave 4: run only if capacity remains; RUN_RULES amendment 2026-10-11)"),
        "implementing code (abbd61e unless stated)": (
            f"PapenmeierMSRPublishedNormBoxSolver(PapenmeierMSRSolver), {GP}:3455-3465 -> "
            "_norm_box_msr_published_next_candidate :3389-3452 -> _propose_msr_published_on_unit_box :3257-3318; "
            f"registration {RB}:174 and :1307-1310; {MAN}:117 ({AT}; class added in 68292ee)"),
        "kernel, priors, constraints, mean, fitting": (
            "as papenmeier_msr_published: SingleTaskGP, ConstantMean, ScaleKernel(Matern-5/2 ARD); lengthscale initialised "
            "to 0.1*sqrt(D)=1.6 in all dims, Interval(0.01, 100), no lengthscale prior (MLE); outputscale: no prior; noise "
            f"prior Gamma(1.1, 0.05), GreaterThan(1e-5) ({GP}:571-601); fit_gpytorch_mll"),
        "input and outcome transforms": (
            f"inputs: per-coordinate (x - min)/(max - min), spans below 1e-3 widened to 1e-3 ({GP}:3404-3418, :180, :215-223); "
            + STD_OUT),
        "acquisition function and best_f": (
            "analytic LogExpectedImprovement, best_f = max of standardised observations, under "
            f"gpytorch.settings.cholesky_max_tries(9) ({GP}:3299-3300, :3438-3439)"),
        "acquisition maximiser": (
            "the published MSR call of papenmeier_msr_published: optimize_acqf(q=1, num_restarts=5, raw_samples=512, "
            f"options={{batch_limit: 1, sample_around_best: True, sample_around_best_sigma: 1e-3, maxiter: 2000}}) on "
            f"[0,1]^256 ({GP}:3303-3310)"),
        "candidate support / box definition and source": (
            "[min, max] box of papenmeier_msr_norm_box: per-coordinate min and max of 50,000 encoded ZINC-250k molecules "
            f"(results/precomputed/zinc_per_coord_bounds.json), always the SELFIES bounds ({GP}:3404)"),
        "trust region / RAASP": (
            "as papenmeier_msr_published (no trust region; RAASP = sample_around_best, sigma 1e-3, centres projected onto "
            "the box); on this box the projection is nearly the identity: 2 of 200 cold-start points lie outside, by at "
            "most 6.2e-7 in unit coordinates (F_msr_published, result in brief)"),
        "cold start": COLD,
        "budget accounting": BUDGET,
        "fallback on GP failure": (
            "flat signal (sd < 1e-6) or _BOX_FALLBACK_ERRORS -> uniform random point in the same box "
            f"(_random_in_normalized_box), counted in surrogate_fallback_count ({GP}:3423-3431, :3444-3452)"),
        "device and software environment": ENV_GPU + NO_STORED,
        "result files, seeds, status": (
            "camera-ready campaign, optional: results/guacamol/cr2026/papenmeier_msr_published_norm_box/, 440 cells if "
            "run (manifest wave 4, timeout 21600 s); no stored generation"),
        "versions recorded in provenance": VERS,
        "git_commit / git_dirty groups": GIT_CR,
        "original repository": (
            "github.com/lpapenme/understanding-hdbo-code @292e7251f1 (src/understanding_hdbo/bo.py:170-190); not vendored"),
        "deviations from the published method": (
            "as papenmeier_msr_published (F_msr_published section 5), with the [min,max] box instead of the [p01,p99] box"),
        "rerun at abbd61e reproduces existing files? (engineering item 2)": (
            "n/a: class added in 68292ee, no stored files; determinism rerun of rano s42 identical (H2_successor "
            "section 4.6)"),
        "evidence": f"F_msr_published.md; H2_successor.md section 4.6; tests/test_msr_published.py; code read at {TAG}",
        "_run": ["papenmeier_msr_published_norm_box"],
        "_init": f"0.1 sqrt(D) = 1.6 in every dimension ({GP}:577-586)",
        "_prior": f"none (maximum likelihood); constraint Interval(0.01, 100) ({GP}:582)",
    },
]

for n in NEW:
    extra = {k: n.pop(k) for k in ("_run", "_init", "_prior")}
    missing = [h for h in HEADER if h not in n]
    unknown = [k for k in n if k not in C]
    if missing or unknown:
        raise SystemExit(f"{n.get('arm')}: missing {missing}, unknown {unknown}")
    out_rows.append([n[h] for h in HEADER] + [";".join(extra["_run"]), extra["_init"], extra["_prior"], "", ORIGIN_NEW,
                                              "all", group_of(n["arm"])])

for r in out_rows:
    for cell in r:
        if "\n" in cell or "\r" in cell:
            raise SystemExit(f"newline in a cell of row {r[0]}")
        cell.encode("ascii")           # the audited table is pure ASCII; keep it so
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, lineterminator="\n")
    w.writerow(HEADER + NEW_COLS)
    w.writerows(out_rows)
print(f"{len(out_rows)} rows ({len(OLD)} audited + {len(NEW)} new), {len(HEADER) + len(NEW_COLS)} columns")
print("audited block sha256 (US/RS serialisation):", block_sha(HEADER, OLD))
print("output sha256:", hashlib.sha256(open(OUT, "rb").read()).hexdigest())
