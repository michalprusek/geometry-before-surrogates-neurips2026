"""Candidate supports: tab_support_geometry (typical candidate norm per support next to best@30 of the arms
that draw from it) and tab_effective_support (one reporting row per printed arm).

The norms come from crlib/mc.py: exact where a closed form exists, a one-dimensional quadrature for the boxes,
the cube and the Gaussian fit, and a control-variate Monte Carlo (fixed seed, standard error printed) for the
convex combinations; a digit is printed only when it is established (SE <= 0.2 of the last unit and >= 4 SE from
a rounding boundary). The arm facts (support definition, maximiser type and
settings) come from ``effective_support.json`` next to make_tables.py, which tools/build_effective_support.py
derived from the Appendix H facts table (B_appendixH_facts.csv, sha256 recorded in the JSON) and the manifest
notes of the camera-ready commit.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import numbers as N
from .context import Ctx
from .fidelity import cell_fidelity
from .latex import Table, m
from .mc import SUPPORT_DEF, compute_supports
from .numbers import camel
from .tables import conv_text, filt_text

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EFF_JSON = os.path.join(HERE, "effective_support.json")


def get_supports(ctx: Ctx) -> dict:
    """Monte Carlo of every support, computed once per run (shared by all variants)."""
    c = ctx.cache
    if "supports" in c:
        return c["supports"]
    from .geometry import geometry_inputs
    G = geometry_inputs(ctx)
    lat = ctx.extra.get("latent_sample")
    bounds = os.path.join(ctx.extra.get("inputs_root", ""), "precomputed", "zinc_per_coord_bounds.json")
    if "default" not in G or not lat or not os.path.exists(lat) or not os.path.exists(bounds):
        msg = "support Monte Carlo skipped: geometry default/, the latent cache or the bounds file is missing"
        print("!!! " + msg, flush=True)
        c.setdefault("loud_warnings", []).append(msg)
        c["supports"] = {}
        return c["supports"]
    inputs = [lat, bounds, G["default"]["json"]] + ([G["default"]["mu_norm"]] if G["default"]["mu_norm"] else [])
    cache = ctx.extra.get("support_cache")
    key = _support_cache_key(inputs, ctx.extra["repo_root"])
    if cache and os.path.exists(cache):
        d = json.load(open(cache))
        if d.get("key") == key:
            print(f"  candidate norms of every support: read from the cache {cache} (key {key[:12]})", flush=True)
            c["supports"] = d["supports"]
            c.setdefault("notes", []).append(f"support norms read from cache {cache}")
            return c["supports"]
    print("  candidate norms of every support (exact / quadrature / control-variate Monte Carlo) ...", flush=True)
    c["supports"] = compute_supports(lat, bounds, G["default"]["json"], G["default"]["mu_norm"],
                                     ctx.extra["repo_root"], log=lambda s: print(s, flush=True))
    if cache:
        os.makedirs(os.path.dirname(os.path.abspath(cache)), exist_ok=True)
        with open(cache, "w") as f:
            json.dump({"key": key, "supports": c["supports"]}, f, indent=1)
    return c["supports"]


def _support_cache_key(inputs: list, repo_root: str) -> str:
    """sha256 over the content hashes of every input of the support norms and of the code that computes them
    (crlib/mc.py). The optional cache (--support-cache, a development convenience: the norms depend on nothing
    else) is used only when this key matches."""
    import hashlib
    from .engine import file_sha256
    from .mc import box_half_width
    paths = sorted(inputs) + [box_half_width(repo_root)[1], os.path.join(os.path.dirname(__file__), "mc.py")]
    h = hashlib.sha256()
    for p in paths:
        h.update(f"{os.path.basename(p)}\t{file_sha256(p)}\n".encode())
    return h.hexdigest()


def how_text(v: dict) -> str:
    """How a support's mean norm was obtained (dossier rule text)."""
    meth = v.get("method")
    if meth == "exact":
        return "exact"
    if meth == "quadrature":
        return (f"exact up to quadrature error {v.get('quadrature_error', 0.0):.1e} (Laplace-transform integral; "
                f"plain Monte Carlo check z = {v.get('mc_check', {}).get('z', float('nan')):+.2f})")
    return (f"control-variate Monte Carlo, {v['draws']:,} draws, fixed seed (stream {v['seed_index']}), "
            f"SE {v['se']:.1e}")


def se_text(se: float) -> str:
    """Two significant digits of a standard error / error estimate in LaTeX math ('0' when exact)."""
    return N.sci(se)


# support id -> (label, {selection layer: arm})
SUPPORT_ROWS = [
    ("shell", "Empirical shell ($r$ per run)", {"none": "random_search", "argmax": "hvarfner_scaled_gp",
                                                "cont": "hvarfner_scaled_gp_sphere_continuous"}),
    ("box_p01p99", "Box $[p_{01},p_{99}]$", {"none": "random_box_faithful", "argmax": "hvarfner_scaled_gp_box_argmax",
                                             "cont": "hvarfner_scaled_gp_box_faithful"}),
    ("box_minmax", "Box $[\\min,\\max]$", {"argmax": "hvarfner_scaled_gp_norm_box_argmax",
                                          "cont": "hvarfner_scaled_gp_norm_box"}),
    ("box_3sigma", "Cube $\\pm3\\bar\\sigma$", {"cont": "hvarfner_scaled_gp_box"}),
    ("prior", "Prior $N(0,I_D)$", {"none": "random_search_prior", "argmax": "hvarfner_scaled_gp_prior"}),
    ("gaussian_fit", "Gaussian fit $N(\\hat\\mu,\\hat\\Sigma)$", {"none": "ellipsoid_rs"}),
    ("convex_k1", "Convex, $k=1$", {"none": "convex_hull_k1"}),
    ("convex_k2", "Convex, $k=2$", {"none": "convex_hull_k2"}),
    ("convex_k3", "Convex, $k=3$", {"none": "convex_hull_rs"}),
    ("convex_k4", "Convex, $k=4$", {"none": "convex_hull_k4"}),
    ("convex_k8", "Convex, $k=8$", {"none": "convex_hull_k8"}),
    ("convex_k16", "Convex, $k=16$", {"none": "convex_hull_k16"}),
    ("convex_k64", "Convex, $k=64$", {"none": "convex_hull_k64"}),
    ("pca64", "Top-64 PCA sphere at $r$", {"none": "pca_sphere_k64"}),
    ("angular_box", "Box directions at $r$", {"argmax": "hvarfner_scaled_gp_angular_box"}),
]
LAYERS = [("none", "No surrogate"), ("argmax", "GP + argmax"), ("cont", "GP + continuous")]


def tab_support_geometry(ctx: Ctx) -> Table | None:
    table = "tab_support_geometry"
    S = get_supports(ctx)
    if not S:
        ctx.skip(table, "supports", "Monte Carlo inputs absent (table omitted)")
        return None
    present = sorted({a for _, _, d in SUPPORT_ROWS for a in d.values() if ctx.has(a)})
    for _, _, d in SUPPORT_ROWS:
        for a in d.values():
            if not ctx.has(a):
                ctx.skip(table, a, "arm absent (cell shows --)")
    cells = ctx.eng.joint(present) if present else set()
    meta = S["_meta"]
    hdr = ["Support", "Mean $\\|x\\|$", "SE", "$\\|x\\|/\\bar r$", "No surrogate", "GP + argmax", "GP + continuous"]
    t = Table("support_geometry", "tab:support_geometry", "", "lrrrrrr", hdr, size="\\footnotesize", wide=True)
    for sid, lab, d in SUPPORT_ROWS:
        v = S.get(sid)
        if v is None:
            ctx.skip(table, sid, "support Monte Carlo absent (row omitted)")
            continue
        fs = ctx.fs_paths(v["inputs"])
        k = camel(sid)
        how = how_text(v)
        ctx.num(f"crSupp{k}Norm", v["mean"], v["text_norm"], kind="norm", table=table, fileset=fs,
                rule=f"mean candidate norm, {how}: {SUPPORT_DEF[sid]}")
        ctx.num(f"crSupp{k}NormSe", v["se"], se_text(v["se"]), kind="mc_se", table=table, fileset=fs,
                rule="standard error of the mean norm (Monte Carlo) or quadrature error estimate (0 = exact)")
        ctx.num(f"crSupp{k}Ratio", v["ratio"], v["text_ratio"], kind="ratio", table=table, fileset=fs,
                rule=f"mean norm / r-bar (r-bar = {meta['rbar']:.4f}, full ZINC-250k set)")
        ctx.num(f"crSupp{k}RatioSe", v["ratio_se"], se_text(v["ratio_se"]), kind="mc_se", table=table, fileset=fs,
                rule="standard error of the ratio")
        row = [lab, m(v["text_norm"]), ("exact" if v["exact"] else m(se_text(v["se"]))), m(v["text_ratio"])]
        for layer, _ in LAYERS:
            a = d.get(layer)
            if not a or not ctx.has(a) or not cells:
                row.append("--")
                continue
            lv = ctx.eng.level(a, cells)
            ctx.num(f"crSupp{k}{camel(layer)}Mean", lv, N.f4(lv), kind="level", table=table, fileset=ctx.fs({a: cells}),
                    n=len(cells), rule=f"{conv_text(ctx)} of best@30 of {a}, joint matched set of the table")
            row.append(m(N.f4(lv)))
        t.add(row)
    if cells:
        ctx.num("crSuppN", len(cells), N.intfmt(len(cells)), kind="n", table=table, fileset=ctx.fs({a: cells for a in present}),
                n=len(cells), rule="joint matched set of every arm in the table")
    mcs = [v for k_, v in S.items() if not k_.startswith("_") and not v["exact"]]
    se_max = max((v["se"] for v in mcs), default=0.0)
    dr = sorted({v["draws"] for v in mcs})
    unest = meta.get("unestablished") or []
    cap = ("\\textbf{Typical candidate norm of every support, next to the best@30 of the arms that draw from it.} "
           "$\\|x\\|$: norm of one candidate. Exact (column SE) for the shell rows (every candidate sits at the "
           "run's radius $r$, the mean norm of its 5 cold-start encodings, whose expectation is $\\bar r$ when the "
           "cold-start pool is approximated by the full set), for $N(0,I_D)$ ($\\sqrt2\\,\\Gamma(\\tfrac{D+1}{2})/"
           "\\Gamma(\\tfrac D2)$), for $k=1$ (mean norm of the 3,000 cached latents the convex samplers use) and, "
           "up to a quadrature error below $10^{-8}$, for the boxes, the cube and the Gaussian fit (independent "
           "coordinates: a one-dimensional integral of the Laplace transform of $\\|x\\|^2$). Convex combinations "
           "with $k\\ge2$: Monte Carlo with $\\|x\\|^2$ as a control variate and a fixed seed")
    if dr:
        cap += f", {N.intfmt(min(dr))}--{N.intfmt(max(dr))} draws (SE $\\le{N.sci(se_max)}$)"
    t.caption = (cap + "; "
                 + "a digit is printed only if the SE is at most 0.2 of its unit and the value lies at least 4 SE "
                 "from a rounding boundary" + (f" (not established within the draw cap, printed with fewer digits: "
                                               f"{', '.join(x.replace('_', ' ') for x in unest)})" if unest else "") + ". "
                 f"$\\bar r={meta['rbar']:.4f}$, the mean encoder-mean norm of the full ZINC-250k set. Boxes: "
                 "per-coordinate bounds of 50,000 encoded ZINC molecules, as the solvers read them. Result files store "
                 "no candidate norms, so these describe the candidate distributions, not the proposals. Best@30: "
                 + (f"{conv_text(ctx)} on one matched set of all arms, $n={len(cells)}$ ({filt_text(ctx)}); "
                    if cells else "no arm present; ")
                 + "-- = no arm with that support and selection layer in this run.")
    return t


# =========================================================================================
# per-arm reporting card
# =========================================================================================
def _eff_facts() -> dict:
    d = json.load(open(EFF_JSON))
    return d


def _harness(ctx: Ctx, arm: str) -> str:
    sp = ctx.ds.arm(arm).spec
    if sp.fmt == "external":
        return "external, stored"
    if sp.kind == "cr":
        return "in-repo, CR run"
    return "in-repo, stored"


def tab_effective_support(ctx: Ctx) -> Table | None:
    table = "tab_effective_support"
    if not os.path.exists(EFF_JSON):
        ctx.skip(table, "effective_support.json", "absent")
        return None
    F = _eff_facts()
    S = get_supports(ctx)
    G_meta = S.get("_meta", {})
    eff_fs = ctx.fs_paths([EFF_JSON])
    from .decomp import stack
    hdr = ["Arm", "Support", "$\\|x\\|/\\bar r$", "Maximiser (type, settings)", stack("Distinct", "of 25"),
           stack("Not", "compl."), stack("Fallback", "cells", "(steps)"), "Harness"]
    t = Table("effective_support", "tab:effective_support", "",
              "p{2.1cm}p{2.2cm}rp{3.0cm}rrrp{1.3cm}", hdr, size="\\scriptsize", long=True, tabcolsep="3pt")
    n = 0
    cur = None
    for e in F["arms"]:
        arm = e["arm"]
        if not ctx.has(arm):
            ctx.skip(table, arm, "arm absent (row omitted)")
            continue
        if e["group"] != cur:
            cur = e["group"]
            t.group(cur)
        A = ctx.ds.arm(arm)
        role = camel(arm)
        cells = sorted(ctx.eng.elig(arm))
        fs_arm = ctx.eng.fileset_all(arm)
        # typical norm / r-bar
        rt_txt = "--"
        if e.get("support_key") and e["support_key"] in S:
            v = S[e["support_key"]]
            mult = float(e.get("radius_multiple") or 1.0)
            rt = v["ratio"] * mult
            ctx.num(f"crEff{role}NormRatio", rt, f"{rt:.2f}", kind="ratio", table=table,
                    fileset=ctx.fs_union(ctx.fs_paths(v["inputs"]), eff_fs),
                    rule=f"typical candidate norm / r-bar of support {e['support_key']}"
                         + (f" x {mult:g}" if mult != 1.0 else "") + " (Monte Carlo of tab_support_geometry)")
            rt_txt = m(f"{rt:.2f}")
        elif e.get("fixed_radius") and G_meta.get("rbar"):
            rt = float(e["fixed_radius"]) / G_meta["rbar"]
            ctx.num(f"crEff{role}NormRatio", rt, f"{rt:.2f}", kind="ratio", table=table,
                    fileset=ctx.fs_union(ctx.fs_paths([G_meta["rbar_source"]]), eff_fs),
                    rule=f"fixed radius {e['fixed_radius']} / r-bar")
            rt_txt = m(f"{rt:.2f}")
        # distinct proposals
        dist_txt = "--"
        if cells:
            dist = float(np.mean([25 - sum(cell_fidelity(A, A.cells[c])) for c in cells]))
            ctx.num(f"crEff{role}Distinct", dist, f"{dist:.1f}", kind="fidelity",
                    table=table, fileset=ctx.fs({arm: set(cells)}), n=len(cells),
                    rule="mean distinct proposals of 25 over the arm's eligible cells (score-equality proxy)")
            dist_txt = m(f"{dist:.1f}")
        nonc = A.n_not_completed()
        r = A.report
        ctx.num(f"crEff{role}NotCompleted", nonc, str(nonc), kind="count", table=table, fileset=fs_arm,
                rule="result files not completed plus crash-sidecar cells")
        ctx.num(f"crEff{role}FallbackCells", r["fallback_cells"], str(r["fallback_cells"]), kind="count", table=table,
                fileset=fs_arm, rule="cells with >= 1 surrogate-fallback step")
        fb = (f"{r['fallback_cells']} ({r['fallback_steps']})" if r.get("fallback_recorded_cells", 0)
              else "n.r.")
        label = ctx.label(arm, e["label"])
        sup = ctx.support(arm, e["support"])
        t.add([label, sup, rt_txt, e["maximiser"], dist_txt, str(nonc), fb, _harness(ctx, arm)])
        n += 1
    if not n:
        return None
    t.caption = ("\\textbf{Effective support and selection of every printed GuacaMol arm, as run.} Support and "
                 "maximiser from the code (Appendix H facts table and the camera-ready manifest); $\\|x\\|/\\bar r$ "
                 "from Table~\\ref{tab:support_geometry} (-- where candidates are anchored at the incumbent, lie in a "
                 "trust region or an embedding, so no fixed candidate distribution exists); distinct of 25 = "
                 "score-equality proxy of Table~\\ref{tab:distinct} over the arm's own "
                 f"{filt_text(ctx)}; not completed = result files not completed plus queue crash records; fallback = "
                 "cells with at least one surrogate-fallback step (steps in parentheses; n.r. = the files carry no "
                 "fallback field). Harness: in-repo = \\texttt{scripts/run\\_benchmark.py}, either a camera-ready run "
                 "of the tagged commit (CR run) or a stored, not rerun generation; external = the deleted NLP harness "
                 "(\\texttt{src/benchmark/runner.py}, section 3 of the run rules released with the code). Rows are omitted for arms absent from this run.")
    return t
