"""Printed names of the baseline arms (RUN_RULES 2026-10-12, published-recipe block, item (1) "print rule").

A row carries a published method's name only if its implementation follows the method's released code in every
setting tabulated in PUB_build.md AND the recipe passed the verification (RUN_RULES 2026-10-13 revision 1 and
2026-10-19). Every other GP or external arm -- the submission-era arms ``hvarfner_scaled_gp*``, ``papenmeier_msr*``,
``xu_standard_gp*``, the LassoBench rerun arms ``hvarfner_gp`` / ``papenmeier_msr`` / ``xu_standard_gp``, the stored
external runs and the published recipes that were not verified -- is printed under a descriptive label that names
kernel, acquisition and maximiser (the support is the table's own column or a suffix), never under the authors'
names. The published names are attached in ``crlib/published.py`` from the verdict file; this module holds the
descriptive labels only, so that every table prints the same words for the same arm.

The wording is a proposal for the authors (GEN v5, decisions); changing it here changes every table and figure.
"""
from __future__ import annotations

# kernel phrases (LaTeX); the lengthscale treatment distinguishes the three submission-era recipes
_K_DSP = "Mat\\'ern-5/2 GP (DSP prior)"                 # LogNormal(sqrt 2 + 0.5 ln D, sqrt 3) lengthscale prior
_K_P01 = "Mat\\'ern-5/2 GP ($\\ell_0=0.1\\sqrt{D}$)"     # lengthscale initialised at 0.1 sqrt(D), MLE
_K_X1 = "Mat\\'ern-5/2 GP ($\\ell_0=\\sqrt{D}$)"         # lengthscale initialised at sqrt(D), MLE
_K_LIN = "spherical linear-kernel GP"
_ARGMAX = "qLogEI, candidate-set argmax"

#: arm -> (LaTeX label, plain-text label for figure legends). The support is printed separately.
DESCRIPTIVE = {
    # GuacaMol, submission-era GP recipes (camera-ready reruns)
    "hvarfner_scaled_gp": (f"{_K_DSP}, {_ARGMAX}", "Matern GP (DSP prior), qLogEI, argmax"),
    "hvarfner_scaled_gp_sphere_continuous": (f"{_K_DSP}, qLogEI, projected Adam", "Matern GP (DSP prior), qLogEI, Adam"),
    # control C4 (RUN_RULES 2026-10-14), the shell x continuous cell of the printed factorial since 2026-11-06
    "hvarfner_scaled_gp_sphere_lbfgs": (f"{_K_DSP}, qLogEI, L-BFGS-B", "Matern GP (DSP prior), qLogEI, L-BFGS-B"),
    "hvarfner_scaled_gp_box_argmax": (f"{_K_DSP}, {_ARGMAX}", "Matern GP (DSP prior), qLogEI, argmax"),
    "hvarfner_scaled_gp_box_faithful": (f"{_K_DSP}, qLogEI, L-BFGS-B", "Matern GP (DSP prior), qLogEI, L-BFGS-B"),
    "hvarfner_scaled_gp_box": (f"{_K_DSP}, qLogEI, L-BFGS-B", "Matern GP (DSP prior), qLogEI, L-BFGS-B"),
    "hvarfner_scaled_gp_angular_box": (f"{_K_DSP}, qLogEI, argmax over box directions at $\\bar r$",
                                       "Matern GP (DSP prior), argmax over box directions"),
    "papenmeier_msr": (f"{_K_P01}, {_ARGMAX}", "Matern GP (l0 = 0.1 sqrt D), qLogEI, argmax"),
    "papenmeier_msr_box_faithful": (f"{_K_P01}, qLogEI, L-BFGS-B", "Matern GP (l0 = 0.1 sqrt D), qLogEI, L-BFGS-B"),
    "papenmeier_msr_published": (f"{_K_P01}, LogEI, L-BFGS-B with sample-around-best",
                                 "Matern GP (l0 = 0.1 sqrt D), LogEI, L-BFGS-B + sample-around-best"),
    "xu_standard_gp": (f"{_K_X1}, {_ARGMAX}", "Matern GP (l0 = sqrt D), qLogEI, argmax"),
    "xu_standard_gp_box_faithful": (f"{_K_X1}, qLogEI, L-BFGS-B", "Matern GP (l0 = sqrt D), qLogEI, L-BFGS-B"),
    # the same recipes on the [min,max] box (legacy variant R2 only; not loaded in camera-ready mode)
    "hvarfner_scaled_gp_norm_box": (f"{_K_DSP}, qLogEI, L-BFGS-B", "Matern GP (DSP prior), qLogEI, L-BFGS-B"),
    "hvarfner_scaled_gp_norm_box_argmax": (f"{_K_DSP}, {_ARGMAX}", "Matern GP (DSP prior), qLogEI, argmax"),
    "papenmeier_msr_norm_box": (f"{_K_P01}, qLogEI, L-BFGS-B", "Matern GP (l0 = 0.1 sqrt D), qLogEI, L-BFGS-B"),
    "papenmeier_msr_published_norm_box": (f"{_K_P01}, LogEI, L-BFGS-B with sample-around-best",
                                          "Matern GP (l0 = 0.1 sqrt D), LogEI, L-BFGS-B + sample-around-best"),
    "xu_standard_gp_norm_box": (f"{_K_X1}, qLogEI, L-BFGS-B", "Matern GP (l0 = sqrt D), qLogEI, L-BFGS-B"),
    "linear_bo": (f"{_K_LIN}, LogEI, L-BFGS-B with sample-around-best",
                  "spherical linear-kernel GP, LogEI, L-BFGS-B + sample-around-best"),
    "linear_bo_sphere_calib": (f"{_K_LIN} (calibrated bounds), {_ARGMAX}",
                               "spherical linear-kernel GP, qLogEI, argmax"),
    # GuacaMol, other in-repo arms (camera-ready reruns)
    "turbo_box_faithful": ("Mat\\'ern-5/2 GP, Thompson sampling in a trust region", "Matern GP, Thompson sampling, trust region"),
    "rembo": ("GP in a random linear embedding, Thompson sampling", "GP in a random embedding, Thompson sampling"),
    "hesbo": ("GP in a hashed embedding, Thompson sampling", "GP in a hashed embedding, Thompson sampling"),
    "gabo": ("sphere Gaussian-kernel GP, EI, Riemannian trust-region maximizer", "sphere Gaussian-kernel GP, EI"),
    "matern_gabo": ("Riemannian Mat\\'ern GP, EI, Riemannian trust-region maximizer", "Riemannian Matern GP, EI"),
    # GuacaMol, the rerun of 2026-11-02 (root CRRD): the full-dimensional CMA-ES of the submitted version ported to the
    # current harness (two populations), and the BAxUS port of the published-recipe block on the [p01, p99] box
    "cmaes_full_rd": ("CMA-ES in $\\mathbb{R}^D$, $\\sigma_0=0.5$, population 20", "CMA-ES in R^D, population 20"),
    "cmaes_full_rd_pop5": ("CMA-ES in $\\mathbb{R}^D$, $\\sigma_0=0.5$, population 5", "CMA-ES in R^D, population 5"),
    "baxus_cr": ("GP in nested count-sketch subspaces, Thompson sampling in a trust region",
                 "GP in nested subspaces, TS, trust region"),
    # GuacaMol, stored external runs (not rerun; RUN_RULES section 3)
    "turbo": ("Mat\\'ern-5/2 GP, Thompson sampling in a trust region (submitted version)",
              "Matern GP, Thompson sampling, trust region (submitted version)"),
    "baxus": ("GP in nested random subspaces, Thompson sampling (submitted version)", "GP in nested subspaces, TS (submitted version)"),
    "lolbo": ("latent trust-region BO with VAE updates (submitted version)", "latent trust-region BO, VAE updates (submitted version)"),
    "invbo": ("latent trust-region BO with latent inversion (submitted version)", "latent trust-region BO, inversion (submitted version)"),
    "cmaes_full": ("CMA-ES in $\\mathbb{R}^D$, $\\sigma_0=0.5$ (submitted version)", "CMA-ES in R^D (submitted version)"),
    "saasbo": ("SAAS fully Bayesian GP, qLogEI, L-BFGS-B (submitted version)", "SAAS GP, qLogEI (submitted version)"),
    "cowboys": ("Tanimoto GP on fingerprints, Thompson sampling (submitted version)", "Tanimoto GP, TS (submitted version)"),
    # budget appendix (stored N = 1000 runs)
    "b1000_hvarfner_scaled_gp": (f"{_K_DSP}, {_ARGMAX}", "Matern GP (DSP prior), qLogEI, argmax"),
    "b1000_vanilla_bo_hvarfner": ("Mat\\'ern-5/2 GP, Thompson sampling", "Matern GP, Thompson sampling"),
    "b1000_baxus": ("GP in nested random subspaces, Thompson sampling", "GP in nested subspaces, TS"),
    # published recipes that were NOT verified (RUN_RULES 2026-10-19): descriptive, marked by crlib/published.py
    "turbo_published": ("Mat\\'ern-5/2 GP, Thompson sampling of the predictive in a trust region",
                        "Matern GP, Thompson sampling, trust region"),
    # verified recipes in their descriptive form (Appendix H uses both forms)
    "hvarfner_published": ("RBF GP (DSP prior), qLogNEI, L-BFGS-B with sample-around-best",
                           "RBF GP (DSP prior), qLogNEI, L-BFGS-B"),
    "hvarfner_published_allobs": ("RBF GP (DSP prior), qLogNEI (all observations), L-BFGS-B with sample-around-best",
                                  "RBF GP (DSP prior), qLogNEI (all obs.), L-BFGS-B"),
    "papenmeier_msr_reference_learned": (f"{_K_P01}, learned noise, LogEI, L-BFGS-B with RAASP",
                                         "Matern GP (l0 = 0.1 sqrt D), LogEI, RAASP"),
    "papenmeier_mle_scaled_reference_learned": (f"{_K_P01}, learned noise, LogEI, L-BFGS-B",
                                                "Matern GP (l0 = 0.1 sqrt D), LogEI, L-BFGS-B"),
    "xu_published": (f"{_K_X1} (uniform prior), UCB, L-BFGS-B", "Matern GP (l0 = sqrt D), UCB, L-BFGS-B"),
    "linear_bo_box8": (f"{_K_LIN}, LogEI, L-BFGS-B with sample-around-best, box $[-8,8]^D$",
                       "spherical linear-kernel GP, LogEI, box [-8, 8]^D"),
}

#: LassoBench methods (rerun at tag cr-runs-2026-10-12 and published methods at tag cr-verify-2026-10-13).
LASSO_DESCRIPTIVE = {
    "hvarfner_gp": ("RBF GP (DSP prior), qLogNEI, L-BFGS-B (rerun)", "RBF GP (DSP prior), qLogNEI"),
    "papenmeier_msr": ("Mat\\'ern-5/2 GP ($\\ell_0=0.1\\sqrt{D}$), qLogEI, L-BFGS-B (rerun)",
                       "Matern GP (l0 = 0.1 sqrt D), qLogEI"),
    "xu_standard_gp": ("Mat\\'ern-5/2 GP ($\\ell_0=\\sqrt{D}$), qLogEI, L-BFGS-B (rerun)", "Matern GP (l0 = sqrt D), qLogEI"),
    "linear_bo": ("spherical linear-kernel GP, LogEI, L-BFGS-B (rerun)", "spherical linear-kernel GP, LogEI"),
    "cmaes_full": ("CMA-ES in the box, $\\sigma_0=0.5$ (rerun)", "CMA-ES (rerun)"),
    "turbo": ("Mat\\'ern-5/2 GP, Thompson sampling in a trust region (rerun)", "Matern GP, TS, trust region (rerun)"),
    "baxus": ("GP in a fixed 16-D subspace, Thompson sampling (rerun)", "GP in a 16-D subspace, TS (rerun)"),
    # published methods that were not verified
    "turbo_published": ("Mat\\'ern-5/2 GP, Thompson sampling of the predictive in a trust region",
                        "Matern GP, TS, trust region"),
    "baxus_official": ("GP in nested count-sketch subspaces, Thompson sampling in a trust region",
                       "GP in nested subspaces, TS, trust region"),
    "cmaes_full_published": ("CMA-ES, popsize 20, $\\sigma_0=0.1$, $x_0=0$", "CMA-ES, popsize 20"),
    # verified methods in their descriptive form
    "hvarfner_published": ("RBF GP (DSP prior), qLogNEI, L-BFGS-B with sample-around-best", "RBF GP (DSP prior), qLogNEI"),
    "papenmeier_published_learned": ("Mat\\'ern-5/2 GP ($\\ell_0=0.1\\sqrt{D}$), learned noise, LogEI with RAASP",
                                     "Matern GP, LogEI, RAASP"),
    "papenmeier_mle_scaled_published_learned": ("Mat\\'ern-5/2 GP ($\\ell_0=0.1\\sqrt{D}$), learned noise, LogEI",
                                                "Matern GP, LogEI"),
    "xu_published": ("Mat\\'ern-5/2 GP ($\\ell_0=\\sqrt{D}$, uniform prior), UCB", "Matern GP, UCB"),
    "linear_bo_published": ("spherical linear-kernel GP, LogEI, L-BFGS-B with sample-around-best",
                            "spherical linear-kernel GP, LogEI"),
}


_BOX = "box $[p_{01},p_{99}]$"
#: support of a GuacaMol GP or external arm, appended where its label stands alone (Appendix H card titles, the
#: arm-status table; RUN_RULES 2026-10-19 second set (c)); Table 1 and Table 2 print it in their own column
SUPPORT = {
    "hvarfner_scaled_gp": "shell", "hvarfner_scaled_gp_sphere_continuous": "shell",
    "hvarfner_scaled_gp_sphere_lbfgs": "shell",
    "hvarfner_scaled_gp_box_argmax": _BOX, "hvarfner_scaled_gp_box_faithful": _BOX,
    "hvarfner_scaled_gp_box": "box $[-3\\sigma,3\\sigma]$", "papenmeier_msr": "shell",
    "papenmeier_msr_box_faithful": _BOX, "papenmeier_msr_published": _BOX, "xu_standard_gp": "shell",
    "xu_standard_gp_box_faithful": _BOX, "linear_bo": _BOX, "linear_bo_sphere_calib": "shell",
    "turbo_box_faithful": "fixed " + _BOX, "rembo": "embedding lifted to $\\bar r$", "hesbo": "embedding lifted to $\\bar r$",
    "gabo": "$S^{D-1}$", "matern_gabo": "$S^{D-1}$", "turbo": "box of the observed latents",
    "baxus": "nested subspaces of a box", "cmaes_full": None, "saasbo": "box $[-5,5]^D$", "lolbo": None, "invbo": None,
    "cowboys": None, "b1000_hvarfner_scaled_gp": "shell", "b1000_vanilla_bo_hvarfner": "cold-start bounding box",
    "b1000_baxus": None, "cmaes_full_rd": None, "cmaes_full_rd_pop5": None, "baxus_cr": _BOX,
}
#: stored arms named by a prefix and the arm they repeat (budget appendix, codec appendix)
STORED_PREFIX = (("b1000_", "$N=1000$ run"), ("chembed_", "Chembed codec"), ("power_spherical_", "Power-Spherical codec"))
#: printed labels of the stored LassoBench and NAS-Bench-101 runs (Appendix H cards of the other benchmarks)
OTHER_STORED = {
    "lassobench": {"geowalk": "\\geowalk{}", "cmaes_sub": "\\geocma{}", "random": "Random search",
                   "hvarfner_gp": "RBF GP (DSP prior), qLogNEI, L-BFGS-B",
                   "papenmeier_msr": "Mat\\'ern-5/2 GP ($\\ell_0=0.1\\sqrt{D}$), qLogEI, L-BFGS-B",
                   "xu_standard_gp": "Mat\\'ern-5/2 GP ($\\ell_0=\\sqrt{D}$), qLogEI, L-BFGS-B",
                   "linear_bo": "spherical linear-kernel GP, LogEI, L-BFGS-B"},
    "nasbench": {"geowalk": "\\geowalk{}", "cmaes_sub": "\\geocma{}", "cmaes_full": "CMA-ES in $\\mathbb{R}^D$",
                 "random": "Random search", "turbo": "Mat\\'ern-5/2 GP, Thompson sampling in a trust region",
                 "gp_bo": "GP (BoTorch defaults), qEI, L-BFGS-B"},
}
#: surnames and method names that a descriptive label must not contain (the test of every printed label source)
AUTHOR_TOKENS = ("Hvarfner", "Papenmeier", "Xu", "Doumont", "Eriksson", "TuRBO", "BAxUS", "SAASBO", "REMBO", "HeSBO",
                 "GaBO", "LOL-BO", "InvBO", "COWBOYS", "Vanilla BO", "Linear BO", "Standard GP", "MSR")


def label(arm: str, default: str | None = None) -> str:
    """LaTeX label of an arm (descriptive unless crlib/published.py attaches a verified name)."""
    if arm in DESCRIPTIVE:
        return DESCRIPTIVE[arm][0]
    if default is None:
        raise KeyError(f"no printed label for arm {arm!r}")
    return default


def plain(arm: str, default: str | None = None) -> str:
    """Plain-text label (figure legends)."""
    if arm in DESCRIPTIVE:
        return DESCRIPTIVE[arm][1]
    if default is None:
        raise KeyError(f"no printed label for arm {arm!r}")
    return default


def lasso_label(method: str, default: str) -> str:
    return LASSO_DESCRIPTIVE[method][0] if method in LASSO_DESCRIPTIVE else default


def lasso_plain(method: str, default: str) -> str:
    return LASSO_DESCRIPTIVE[method][1] if method in LASSO_DESCRIPTIVE else default
