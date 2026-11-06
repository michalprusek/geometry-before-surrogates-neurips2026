"""Builder of sources_cr.json and sources_legacy.json (run on 2026-10-11, updated 2026-10-12 for the production
manifest: 60 arms of 20ad7d5 + the three GP radius arms of the 2026-10-12 amendment = 63 arm ids, and again for
the control block of tag cr-controls-2026-10-12 (fd8a03c): 3 new arms + 6 replication arms on seeds 82-121 = 72
arm ids; kept as the audit trail).

The committed JSON files are the source of truth: edit them directly. Re-running this script
overwrites them. Usage: python tools/build_sources.py scripts/cr/analysis
writes sources_cr.json and sources_legacy.json.

The committed JSON files are the source of truth; this script only avoids typing the shared
stored-arm block twice.
"""
import json
import os
import sys

OUT = sys.argv[1]

T4 = ["adip", "fexo", "osmb", "rano"]

# ---- manifest arms (wave) and their legacy (pre-camera-ready) file locations --------------
# name: (wave, label, legacy_dir, legacy_pattern_suffix_dim, legacy_extra)
MANIFEST = [
    ("geowalk", 1, "GeoWalk"), ("cma_es_subspace", 1, "GeoCMA"), ("random_search", 1, "Random search (sphere)"),
    ("hvarfner_scaled_gp", 1, "Vanilla BO (sphere)"), ("papenmeier_msr", 1, "Papenmeier MLE (scaled), sphere"),
    ("xu_standard_gp", 1, "Xu Standard GP (sphere)"), ("linear_bo_sphere_calib", 1, "Linear BO (sphere)"),
    ("hvarfner_scaled_gp_box_argmax", 1, "Vanilla BO, box, candidate-set argmax"),
    ("hvarfner_scaled_gp_prior", 1, "Vanilla BO, prior N(0,I) candidates"),
    ("hvarfner_scaled_gp_angular_box", 1, "Vanilla BO, box directions at r"),
    ("pca_sphere_k64", 1, "RS on top-64 PCA sphere"),
    ("geowalk_r4", 1, "GeoWalk r=4"), ("geowalk_r6", 1, "GeoWalk r=6"), ("geowalk_r8", 1, "GeoWalk r=8"),
    ("geowalk_r10", 1, "GeoWalk r=10"), ("geowalk_r12", 1, "GeoWalk r=12"), ("geowalk_r16", 1, "GeoWalk r=16"),
    ("geowalk_shell10", 1, "GeoWalk shell +-10%"), ("geowalk_shell20", 1, "GeoWalk shell +-20%"),
    ("geowalk_shellpct", 1, "GeoWalk shell U[p05,p95]"), ("geowalk_rmedian", 1, "GeoWalk median radius"),
    ("geowalk_rpop", 1, "GeoWalk fixed radius 7.98"),
    ("convex_hull_k1", 1, "Convex k=1"), ("convex_hull_k2", 1, "Convex k=2"), ("convex_hull_rs", 1, "Convex k=3"),
    ("convex_hull_k4", 1, "Convex k=4"), ("convex_hull_k8", 1, "Convex k=8"), ("convex_hull_k16", 1, "Convex k=16"),
    ("convex_hull_k64", 1, "Convex k=64"), ("ellipsoid_rs", 1, "Ellipsoid"),
    ("hvarfner_scaled_gp_sphere05", 1, "Vanilla BO, shell at 0.5 r"),
    ("hvarfner_scaled_gp_sphere15", 1, "Vanilla BO, shell at 1.5 r"),
    ("hvarfner_scaled_gp_sphere20", 1, "Vanilla BO, shell at 2 r"),
    ("hvarfner_scaled_gp_box_faithful", 2, "Vanilla BO (box [p01,p99])"),
    ("hvarfner_scaled_gp_sphere_continuous", 2, "Vanilla BO, shell, continuous (Adam)"),
    ("hvarfner_scaled_gp_norm_box", 2, "Vanilla BO (box [min,max])"),
    ("papenmeier_msr_box_faithful", 2, "Papenmeier MLE (scaled), box [p01,p99]"),
    ("papenmeier_msr_published", 2, "Papenmeier MSR (published), box [p01,p99]"),
    ("papenmeier_msr_norm_box", 2, "Papenmeier MLE (scaled), box [min,max]"),
    ("xu_standard_gp_box_faithful", 2, "Xu Standard GP (box [p01,p99])"),
    ("xu_standard_gp_norm_box", 2, "Xu Standard GP (box [min,max])"),
    ("linear_bo", 2, "Linear BO (box [p01,p99])"),
    ("turbo_box_faithful", 3, "TuRBO-1 (fixed [p01,p99] box)"), ("rembo", 3, "REMBO"), ("hesbo", 3, "HeSBO"),
    ("gabo", 3, "GaBO"), ("matern_gabo", 3, "Riemannian Matern GP"),
    ("hvarfner_scaled_gp_box", "3b", "Vanilla BO (box +-3 sigma)"), ("papenmeier_msr_box", "3b", "Papenmeier MLE (scaled), box +-3 sigma"),
    ("xu_standard_gp_box", "3b", "Xu Standard GP (box +-3 sigma)"),
]
# Arms added by the RUN_RULES amendment log of 2026-10-11 (not in manifest.json at 68292ee).
AMEND = [
    ("geowalk_const060", "GeoWalk const 0.60", 1), ("geowalk_const035", "GeoWalk const 0.35", 1),
    ("geowalk_const015", "GeoWalk const 0.15", 1), ("geowalk_explore8", "GeoWalk explore 8", 1),
    ("geowalk_explore24", "GeoWalk explore 24", 1), ("geowalk_one_fifth", "GeoWalk success rule (x1.2 / x0.85)", 1),
    ("random_search_prior", "RS from the prior N(0,I)", 1), ("random_box_faithful", "RS in the [p01,p99] box", 1),
    ("papenmeier_msr_raasp", "Papenmeier RAASP trust-region variant, [min,max]", 1),
    ("hvarfner_scaled_gp_norm_box_argmax", "Vanilla BO, box [min,max], candidate-set argmax", "3b"),
    ("papenmeier_msr_published_norm_box", "Papenmeier MSR (published), box [min,max]", 4),
]


CPU_ARMS = {"gabo", "matern_gabo"}   # manifest host_class 'cpu': run on kajman's CPU (R-2)


def cr_arm(name, label, wave, solver=None, overrides=None, root=None, seeds=None):
    """A production arm: directory = arm id (<root>/<arm id>/), file name = run_cell.cell_filename(solver, ...),
    provenance.cr_arm = arm id, provenance.run_overrides = the manifest's (R-1 exception only for D~).
    root: CR (main campaign, panda GPU), CRCPU (main campaign, kajman CPU) or CRCTL (control block, its own tag);
    seeds: the arm's own seed range (the replication arms run 82-121)."""
    root = "${" + root + "}/" if root else ("${CRCPU}/" if name in CPU_ARMS else "${CR}/")
    solver = solver or name
    overrides = overrides or {}
    sd = overrides.get("subspace_dim")
    tag = f"_sd{sd}" if sd else ""
    a = {"label": label, "kind": "cr", "format": "ours", "dir": root + name,
         "pattern": f"ours_{solver}_{{task}}_s{{seed}}_128d{tag}.json", "solver_field": solver,
         "status_rule": "require_field", "provenance": "require",
         "required": wave != 4,
         "cr_arm": name, "expect": {"provenance.run_overrides": overrides}, "wave": wave,
         "note": f"manifest wave {wave}" + (" (run only if capacity remains: optional)" if wave == 4 else "")}
    if sd:
        a["expect"]["subspace_dim"] = sd
    if seeds:
        a["seeds"] = seeds
    return a


G = "${G}"
SV = G + "/spherevo"


def ours(dir_, solver, dim="128d", fname_solver=None, **kw):
    fs = fname_solver or solver
    a = {"kind": "stored", "format": "ours", "dir": dir_, "pattern": f"ours_{fs}_{{task}}_s{{seed}}_{dim}.json",
         "solver_field": solver, "status_rule": "require_field", "provenance": "record"}
    a.update(kw)
    return a


LEGACY_MAP = {
    "geowalk": ours(G + "/geowalk", "geowalk", note="April 2026 (6244f20 dirty)"),
    "cma_es_subspace": ours(SV, "cma_es_subspace", status_rule="absent_is_completed",
                            note="March 2026; seeds 42-51 old schema without status"),
    "random_search": ours(SV, "random_search", note="March 2026 (9c042c2/6244f20 dirty); the copy in random_search/ is not used"),
    "hvarfner_scaled_gp": ours(SV, "hvarfner_scaled_gp", note="April 2026 (295a7f6 dirty)"),
    "papenmeier_msr": ours(SV, "papenmeier_msr", note="July 2026, 0.1 sqrt(D) (ec93427 dirty)"),
    "xu_standard_gp": ours(SV, "xu_standard_gp", note="April 2026, four commit states"),
    "linear_bo_sphere_calib": ours(SV, "linear_bo_sphere_calib", note="July 2026 (6b48a28)"),
    "hvarfner_scaled_gp_box_argmax": ours(SV, "hvarfner_scaled_gp_box_argmax", "256d", note="July 2026 batch runner"),
    "hvarfner_scaled_gp_prior": ours(SV, "hvarfner_scaled_gp_prior", note="April 2026 (f299e5e dirty)"),
    "hvarfner_scaled_gp_angular_box": ours(SV, "hvarfner_scaled_gp_angular_box", note="April 2026"),
    "pca_sphere_k64": ours(SV, "pca_sphere_k64", note="July 2026"),
    "hvarfner_scaled_gp_box_faithful": ours(SV, "hvarfner_scaled_gp_box_faithful", note="May 2026 (8c9bef1 dirty)"),
    "hvarfner_scaled_gp_sphere_continuous": ours(SV, "hvarfner_scaled_gp_sphere_continuous", note="July 2026 (433 files; 7 cells never written)", known_missing=["osmb:60", "osmb:67", "osmb:71", "osmb:75", "osmb:79", "pdop:43", "pdop:47"]),
    "hvarfner_scaled_gp_norm_box": ours(SV, "hvarfner_scaled_gp_norm_box", note="May 2026 (submitted [min,max] row)"),
    "papenmeier_msr_box_faithful": ours(SV, "papenmeier_msr_box_faithful", note="July 2026"),
    "papenmeier_msr_published": ours(SV, "papenmeier_msr_raasp", known_missing=["siga:79"], note="NO pre-CR generation of the published-MSR arm: legacy maps it to the July trust-region RAASP arm (papenmeier_msr_raasp, [min,max])",
                                     display={"label": "Papenmeier RAASP trust-region variant, $[\\min,\\max]$",
                                              "support": "$[\\min,\\max]$, trust region"}),
    "papenmeier_msr_norm_box": ours(G + "/_archive_papenmeier_sqrtD_bug", "papenmeier_msr_norm_box", note="submitted sqrt(D) [min,max] row (archive)",
                                    display={"label": "Papenmeier, $\\sqrt{D}$ archive (identical to Xu)",
                                             "support": "$[\\min,\\max]$"}),
    "xu_standard_gp_box_faithful": ours(SV, "xu_standard_gp_box_faithful", note="July 2026"),
    "xu_standard_gp_norm_box": ours(SV, "xu_standard_gp_norm_box", note="May 2026 (submitted row)"),
    "linear_bo": ours(SV, "linear_bo", note="July 2026"),
    "rembo": ours(SV, "rembo", seeds="42-51", note="March 2026, 10 seeds"),
    "hesbo": ours(SV, "hesbo", seeds="42-51", note="March 2026, 10 seeds"),
    "gabo": ours(G + "/gabo", "gabo", note="April 2026 (f299e5e dirty)"),
    "matern_gabo": ours(G + "/matern_gabo", "matern_gabo", note="April 2026"),
    "hvarfner_scaled_gp_box": ours(SV, "hvarfner_scaled_gp_box", note="April 2026 (+-3 sigma)"),
    "papenmeier_msr_box": ours(SV, "papenmeier_msr_box", note="April 2026, pre-fix sqrt(D); 4 cells never written", known_missing=["osmb:52", "pdop:43", "pdop:52", "rano:52"]),
    "xu_standard_gp_box": ours(SV, "xu_standard_gp_box", note="April 2026"),
}
for r in (4, 6, 8, 10, 12, 16):
    LEGACY_MAP[f"geowalk_r{r}"] = ours(G + "/geowalk_ablations", f"geowalk_r{r}", "256d", note="July 2026 batch runner")
for s in ("shell10", "shell20", "shellpct", "rmedian", "rpop"):
    LEGACY_MAP[f"geowalk_{s}"] = ours(G + "/geowalk_ablations", f"geowalk_{s}", "256d", note="July 2026 batch runner")
for k in ("k1", "k2", "rs", "k4", "k8", "k16", "k64"):
    LEGACY_MAP[f"convex_hull_{k}"] = ours(SV, f"convex_hull_{k}", "256d", note="July 2026 batch runner")
LEGACY_MAP["ellipsoid_rs"] = ours(SV, "ellipsoid_rs", "256d", note="July 2026 batch runner")
for v in ("const060", "const035", "const015", "explore8", "explore24"):
    LEGACY_MAP[f"geowalk_{v}"] = ours(G + "/geowalk_ablations", f"geowalk_{v}", seeds="42-51", note="April 2026 (6244f20 dirty), 10 seeds")
LEGACY_MAP["random_search_prior"] = ours(SV, "random_search_prior", note="April 2026 (f299e5e dirty)")
for c in ("05", "15", "20"):
    LEGACY_MAP[f"hvarfner_scaled_gp_sphere{c}"] = ours(SV, f"hvarfner_scaled_gp_sphere{c}",
                                                       note="April 2026 (tulen, f299e5e dirty); lab notebook sweep")
LEGACY_MAP["papenmeier_msr_raasp"] = ours(SV, "papenmeier_msr_raasp", note="July 2026", known_missing=["siga:79"])
for name in ("turbo_box_faithful", "geowalk_one_fifth", "random_box_faithful",
             "hvarfner_scaled_gp_norm_box_argmax", "papenmeier_msr_published_norm_box"):
    LEGACY_MAP[name] = ours(G + "/__no_pre_camera_ready_generation__/" + name, name, required=False,
                            note="no stored pre-camera-ready generation readable from panda")
# GeoCMA subspace-dimension variants (D~ paragraph): March 2026, 4 tasks x seeds 42-51
for sd in (8, 64):
    LEGACY_MAP[f"cma_es_subspace_sd{sd}"] = {
        "kind": "stored", "format": "ours", "dir": SV, "pattern": f"ours_cma_es_subspace_{{task}}_s{{seed}}_128d_sd{sd}.json",
        "solver_field": "cma_es_subspace", "status_rule": "absent_is_completed", "provenance": "record",
        "seeds": "42-51", "tasks": T4, "note": "March 2026 stochastic decoding, 4 tasks x 10 seeds"}
LEGACY_ONLY = {
    "papenmeier_msr_sqrtd": ours(G + "/_archive_papenmeier_sqrtD_bug", "papenmeier_msr",
                                 note="submitted sqrt(D) sphere row (archive)"),
    # September p1 blocks (4780475 clean, batch runner, kajman GPU): the only stored per-cell generation of
    # uniform box sampling. Used ONLY by the clearly labelled legacy variant of the decomposition tables.
    "p1_random_box_faithful": ours("${INPUTS}/p1_blocks", "random_box_faithful", "256d", required=False,
                                   note="September p1 blocks @4780475 (clean, batch runner), copied read-only from "
                                        "kajman /data2/prusek/NeurIPS-p1/results/guacamol/p1_blocks with a sha256 "
                                        "manifest; legacy decomposition variant only"),
}


def ext(m, prefix=None, method=None):
    p = prefix or m
    return {"kind": "stored", "format": "external", "dir": G + "/" + m, "pattern": f"{p}_{{task}}_seed{{seed}}.json",
            "task_key": "short", "solver_field": method or m, "status_rule": "absent_is_completed",
            "provenance": "record", "coldstart_from": "geowalk",
            "note": "external harness (src/benchmark/runner.py, deleted in 68ed3dc); not rerunnable (RUN_RULES 3)"}


STORED = {
    "turbo": dict(ext("turbo"), label="TuRBO (submitted harness)"),
    "baxus": dict(ext("baxus"), label="BAxUS"),
    "lolbo": dict(ext("lolbo"), label="LOL-BO"),
    "invbo": dict(ext("invbo"), label="InvBO"),
    "cmaes_full": dict(ext("cmaes_full", prefix="cmaes", method="cmaes"), label="CMA-ES (full-D)"),
    "saasbo": dict(ours(SV, "saasbo", seeds="42-51"), label="SAASBO",
                   known_missing=["dhop:46", "osmb:47", "shop:48", "shop:50", "shop:51", "siga:42"],
                   note="April 2026, 10 seeds; 6 cells were never written; not rerun (RUN_RULES 3)"),
    "cowboys": dict(ours(SV, "cowboys", seeds="42-51"), label="COWBOYS",
                    note="March 2026, 10 seeds, stochastic decoding; not rerun, recommended drop (RUN_RULES 3)"),
}
STORED["linear_bo_sphere"] = dict(
    ours(SV, "linear_bo_sphere"), label="Linear BO (shell, uncalibrated; stored)",
    note="July 2026, Linear BO on the shell before the kernel bound was calibrated (audit CLAIM 1); stored, not "
         "rerun: printed only as a disclosure row next to the calibrated arm linear_bo_sphere_calib")
for s in ("hvarfner_scaled_gp", "hvarfner_scaled_gp_sphere_continuous", "hvarfner_scaled_gp_box_argmax",
          "hvarfner_scaled_gp_box_faithful"):
    STORED["e1_" + s] = dict(ours(G + "/em1_premise", s, "256d"), label="E-1 " + s,
                             note="E-1 (em1_premise, abbd61e clean, batch runner), kept by R-7")
B1000 = {
    "geowalk": (G + "/geowalk", "geowalk", "128d_b1000", "geowalk"),
    "hvarfner_scaled_gp": (SV, "hvarfner_scaled_gp", "128d_b1000", "hvarfner_scaled_gp"),
    "cma_es_subspace": (SV, "cma_es_subspace", "128d_b1000", "cma_es_subspace"),
    "random_search": (SV, "random_search", "128d_b1000", "random_search"),
    "vanilla_bo_hvarfner": (SV, "vanilla_bo_hvarfner", "256d_b1000", None),
    "baxus": (SV, "baxus", "256d_b1000", None),
}
for k, (d, fs, dim, sf) in B1000.items():
    STORED["b1000_" + k] = {"label": "N=1000 " + k, "kind": "stored", "format": "ours", "dir": d,
                            "pattern": f"ours_{fs}_{{task}}_s{{seed}}_{dim}.json", "solver_field": sf,
                            "status_rule": "absent_is_completed", "provenance": "record", "seeds": "42-51",
                            "trace_len": 1000, "budget": 1000,
                            "known_missing": (["zale:44"] if k == "baxus" else []),
                            "note": "budget-scaling runs (Appendix F), relabelled, not rerun (RUN_RULES 3)"}
for codec, pre in (("chembed", "chembed"), ("power_spherical", "power_spherical")):
    sols = (["geowalk", "cma_es_subspace", "hvarfner_scaled_gp", "hvarfner_scaled_gp_box_faithful", "random_search"]
            if codec == "chembed" else
            ["geowalk", "cma_es_subspace", "hvarfner_scaled_gp", "hvarfner_scaled_gp_box_faithful", "hvarfner_scaled_gp_box"])
    for s in sols:
        STORED[f"{codec}_{s}"] = {"label": f"{codec} {s}", "kind": "stored", "format": "ours",
                                  "dir": "${INPUTS}/" + codec + "/" + s,
                                  "pattern": f"ours_{pre}_{s}_{{task}}_s{{seed}}_128d.json", "solver_field": s,
                                  "status_rule": "require_field", "provenance": "record",
                                  "note": f"{codec} codec runs (tulen, Apr/May 2026); no environment loads the codec (RUN_RULES 3); copy in /data/prusek/cr_inputs/{codec} (sha256 manifest)"}

ROOTS_LEGACY = {"G": "/home/prusek/NeurIPS/results/guacamol", "INPUTS": "/data/prusek/cr_inputs"}
ROOTS_CR = {"CR": "/data/prusek/NeurIPS-cr/results/guacamol/cr2026",
            # GaBO / Riemannian Matern run on kajman's CPU (R-2, amendment 2026-10-12); their production files are
            # copied to panda later. Default = the planned copy location; override with --root CRCPU=<dir>.
            "CRCPU": "/data/prusek/cr_inputs/cr2026_kajman",
            # control block (tag cr-controls-2026-10-12 = fd8a03c; its own expected commit)
            "CRCTL": "/data/prusek/NeurIPS-cr-ctl/results/guacamol/cr2026ctl",
            # control C4 (tag cr-controls-lbfgs-2026-10-14 = 2a899ad): the same directory, its own expected commit
            "CRC4": "/data/prusek/NeurIPS-cr-ctl/results/guacamol/cr2026ctl",
            "G": "/home/prusek/NeurIPS/results/guacamol", "INPUTS": "/data/prusek/cr_inputs"}

legacy_arms = {}
for name, wave, label in MANIFEST:
    a = dict(LEGACY_MAP[name])
    a["label"] = label
    if "display" in a:      # legacy substitute files: the config label names what the files ARE
        a["label"] = (a["display"]["label"].replace("$", "").replace("\\", "")
                      + f" (legacy substitute for {name})")
    legacy_arms[name] = a
for name, label, _wave in AMEND:
    a = dict(LEGACY_MAP[name])
    a["label"] = label
    a.setdefault("required", True)
    legacy_arms[name] = a
for sd in (8, 64):
    a = dict(LEGACY_MAP[f"cma_es_subspace_sd{sd}"])
    a["label"] = f"GeoCMA D~={sd}"
    legacy_arms[f"cma_es_subspace_sd{sd}"] = a
for name, a in LEGACY_ONLY.items():
    a = dict(a)
    a["label"] = name
    legacy_arms[name] = a
legacy_arms.update(STORED)

cr_arms = {}
for name, wave, label in MANIFEST:
    cr_arms[name] = cr_arm(name, label, wave)
for name, label, wave in AMEND:
    cr_arms[name] = cr_arm(name, label, wave)
    cr_arms[name]["note"] += "; RUN_RULES amendment 2026-10-11"
for sd in (8, 64):
    a = cr_arm(f"cma_es_subspace_sd{sd}", f"GeoCMA D~={sd}", 1, solver="cma_es_subspace",
               overrides={"subspace_dim": sd})
    a["note"] += (f"; RUN_RULES amendment 2026-10-11 (R-1 exception): run_cell --run-overrides subspace_dim={sd} "
                  f"writes the _sd{sd} suffix (run_cell.cell_filename)")
    cr_arms[f"cma_es_subspace_sd{sd}"] = a
for c in ("05", "15", "20"):
    cr_arms[f"hvarfner_scaled_gp_sphere{c}"]["note"] += "; RUN_RULES amendment 2026-10-12 (GP radius dose-response)"
# ---- control block (RUN_RULES amendment 2026-10-12; manifest waves c1, c2 at fd8a03c; output root cr2026ctl) --
CONTROL_NEW = [
    ("hvarfner_scaled_gp_box_argmax_raw",
     "C1: Vanilla BO, box [p01,p99], candidate-set argmax, GP fitted on raw latents (input-transform control)"),
    ("hvarfner_scaled_gp_boxdir_argmax",
     "C2a: Vanilla BO, box draws renormalised to the shell radius, candidate-set argmax, GP on raw latents"),
    ("random_boxdir", "C2b: uniform box draws renormalised to the shell radius, no surrogate"),
]
REPLICATION_BASE = ["random_search", "random_box_faithful", "hvarfner_scaled_gp", "hvarfner_scaled_gp_box_argmax",
                    "hvarfner_scaled_gp_sphere_continuous", "hvarfner_scaled_gp_box_faithful"]
for name, label in CONTROL_NEW:
    cr_arms[name] = cr_arm(name, label, "c1", root="CRCTL")
    cr_arms[name]["note"] = ("control block, manifest wave c1 (tag cr-controls-2026-10-12 = fd8a03c, root CRCTL); "
                             "RUN_RULES amendment 2026-10-12; seeds 42-81")
for base in REPLICATION_BASE:
    name = base + "__s82"
    cr_arms[name] = cr_arm(name, f"C3 replication on seeds 82-121: {base}", "c2", solver=base, root="CRCTL",
                           seeds="82-121")
    cr_arms[name]["note"] = ("control block, manifest wave c2 (tag cr-controls-2026-10-12 = fd8a03c, root CRCTL): "
                             f"the solver of {base} on seeds 82-121 (new cold starts); directory and "
                             f"provenance.cr_arm = {name}; never pooled with {base} (seeds 42-81)")
# ---- control C4 (RUN_RULES amendment 2026-10-14; manifest wave c3 at 2a899ad, tag cr-controls-lbfgs-2026-10-14;
# output root cr2026ctl, read through root CRC4 with its own expected commit). Since RUN_RULES 2026-11-06 16:30 the
# shell x continuous cell of the printed factorial (in place of hvarfner_scaled_gp_sphere_continuous) -------------
cr_arms["hvarfner_scaled_gp_sphere_lbfgs"] = cr_arm(
    "hvarfner_scaled_gp_sphere_lbfgs",
    "Factorial cell shell x continuous: L-BFGS-B on the shell (control C4; the printed cell since 2026-11-06)", "c3",
    root="CRC4")
cr_arms["hvarfner_scaled_gp_sphere_lbfgs"]["note"] = (
    "control C4, manifest wave c3 (tag cr-controls-lbfgs-2026-10-14 = 2a899ad, run into the control block's directory "
    "cr2026ctl; root CRC4 names that directory with this tag's commit as the expected commit); RUN_RULES amendment "
    "2026-10-14; seeds 42-81, paired with the main campaign's cold starts. Since RUN_RULES 2026-11-06 16:30 the shell x "
    "continuous cell of the printed factorial, in place of hvarfner_scaled_gp_sphere_continuous")
cr_arms.update(STORED)
N_MANIFEST = sum(1 for a in cr_arms.values() if a["kind"] == "cr")
assert N_MANIFEST == 73, N_MANIFEST

common = {"schema": "cr-sources-v1", "tasks": ["adip", "dhop", "fexo", "med1", "med2", "osmb", "pdop", "rano",
                                                "shop", "siga", "zale"], "seeds": "42-81"}
cr = dict(common, name="camera-ready", description=(
    "Camera-ready sources: every manifest arm (73 arm ids: the 60 of 20ad7d5, the three GP radius arms of the "
    "2026-10-12 amendment, and the nine control-block arms of tag cr-controls-2026-10-12, which read "
    "${CRCTL}/<arm id>/ = results/guacamol/cr2026ctl of the control clone and must carry that tag's commit; the six "
    "'<arm>__s82' replication arms hold base-solver files on seeds 82-121 and are grouped by provenance.cr_arm, "
    "never pooled with '<arm>') reads ONLY its directory ${CR}/<arm id>/ (results/guacamol/cr2026 of the production "
    "clone, written by scripts/cr/run_cell.py through cr_queue.py at the tagged commit); file names are "
    "run_cell.cell_filename (ours_<solver>_<poli task>_s<seed>_128d.json, _sd8/_sd64 for the GeoCMA D~ arms); "
    "provenance required: git_commit = the --expected-commit of the arm's root, git_dirty False, provenance.cr_arm = the arm id and "
    "provenance.run_overrides = the manifest's. Queue sidecars (crash/interrupted/running/collision/tmp, attempts/) "
    "are recognised by the loader. GaBO and Riemannian Matern GP run on kajman's CPU and are read from ${CRCPU} "
    "(a copy of kajman's production files; override with --root CRCPU=...). Wave 4 (papenmeier_msr_published_norm_box) "
    "is optional. Non-rerunnable arms (RUN_RULES 3) "
    "read their stored directories with anchored file names. The posted (rebuttal-era) generation is "
    "loaded from sources_legacy.json under the namespace 'posted:' for the correspondence tables only; that "
    "include is optional (an absent old arm is reported and its correspondence rows show n/a, a missing cell "
    "inside a present old arm still fails)."),
    roots=ROOTS_CR, cell_filter="completed",
    include=[{"namespace": "posted", "file": "sources_legacy.json", "optional": True}],
    arms=cr_arms)
legacy = dict(common, name="legacy", description=(
    "Legacy sources: the camera-ready arm names point at the OLD stored files (submitted / July / "
    "rebuttal / E-1 generations) so that the generator can be validated against numbers that earlier "
    "audits verified independently. cell_filter 'clean' = the audits' convention (status completed, "
    "no surrogate fallback, budget 30, 30 values)."),
    roots=ROOTS_LEGACY, cell_filter="clean", arms=legacy_arms)
os.makedirs(OUT, exist_ok=True)
for fn, obj in (("sources_cr.json", cr), ("sources_legacy.json", legacy)):
    with open(os.path.join(OUT, fn), "w") as f:
        json.dump(obj, f, indent=1)
    print(fn, len(obj["arms"]))
