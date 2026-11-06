"""Shell geometry of the encoder means: tab_geometry, fig_shells and the inline geometry numbers.

Inputs: a geometry root (``--geometry-root``) with up to three subfolders written by
``scripts/cr/analysis/fullset_geometry.py`` (the committed copy of G_fullset_geometry.py):

  default/          the full ZINC-250k reference set as the codec encodes it
  strip/            the same set with stereo tokens stripped before encoding
  guacamol_train/   the VAE's own training corpus (GuacaMol v1 train, about 1.27 M molecules;
                    RUN_RULES amendment 2026-10-12: printed whatever it shows)

each with ``geometry.json`` and the per-molecule arrays ``mu_norm.npy`` (||mu(x)||) and
``z_norm.npy`` (||mu(x) + sigma(x) eps||, one eps per molecule). A missing subfolder prints
"n/a" in its column with a loud warning in the log, the run manifest and the caption; a
missing default/ skips the table. Every number's file set is the geometry.json (and the
arrays) it was read from.
"""
from __future__ import annotations

import math
import os

import numpy as np

from . import numbers as N
from . import scope
from .context import Ctx
from .latex import Table, m, WRAP, stack

GEOM_COLS = [("default", "Full", "ZINC-250k (full set)"),
             ("strip", "Strip", "ZINC-250k, stereo stripped"),
             ("guacamol_train", "Guacamol", "GuacaMol training set")]


def _scoped(ctx: Ctx) -> bool:
    """Camera-ready mode prints the ZINC-250k geometry only (RUN_RULES 2026-10-14, scope item 4: no training-corpus
    geometry and no posterior-sample shell); legacy mode keeps every column and row."""
    return ctx.mode == "cr"


def _cols(ctx: Ctx) -> list:
    return [c for c in GEOM_COLS if not _scoped(ctx) or c[0] in scope.GEOMETRY_COLUMNS]


def geometry_inputs(ctx: Ctx) -> dict:
    """{col: {"dir", "json", "mu_norm", "z_norm", "g"}} for the subfolders that exist (cached)."""
    c = ctx.cache
    if "geometry_inputs" in c:
        return c["geometry_inputs"]
    import json
    root = ctx.extra.get("geometry_root") or ""
    out = {}
    for col, _, _ in _cols(ctx):          # camera-ready mode never opens the training-corpus geometry
        d = os.path.join(root, col)
        p = os.path.join(d, "geometry.json")
        if not os.path.exists(p):
            continue
        rec = {"dir": d, "json": p, "g": json.load(open(p))}
        for arr in ("mu_norm", "z_norm"):
            q = os.path.join(d, f"{arr}.npy")
            rec[arr] = q if os.path.exists(q) else None
        out[col] = rec
    c["geometry_inputs"] = out
    return out


def geometry_numbers(ctx: Ctx) -> None:
    """The inline geometry macros (unchanged names; the legacy checks read them)."""
    G = geometry_inputs(ctx)
    if "default" not in G:
        ctx.skip("numbers_geometry", "geometry default/", "input absent")
        return
    p = G["default"]["json"]
    ps = G["strip"]["json"] if "strip" in G else None
    g = G["default"]["g"]
    fs = ctx.fs_paths([p] + ([ps] if ps else []))
    mu = g["mu_cloud"]
    c1 = mu["cov_ddof1"]
    T = "numbers_geometry"

    def add(name, v, text, rule):
        ctx.num(name, v, text, kind="geometry", table=T, fileset=fs, n=mu["n"], rule=rule)

    add("crGeomNZinc", mu["n"], N.intfmt(mu["n"]), "encoded ZINC-250k molecules (data/zinc_250k.txt, 249,455 lines)")
    add("crGeomRbar", mu["norm"]["mean"], N.f2(mu["norm"]["mean"]), "mean ||mu(x)|| over all encoded molecules")
    add("crGeomRbarFour", mu["norm"]["mean"], N.f4(mu["norm"]["mean"]), "mean ||mu(x)||, 4 decimals")
    add("crGeomRbarSd", mu["norm"]["sd_ddof1"], N.f2(mu["norm"]["sd_ddof1"]), "SD of ||mu(x)|| (ddof 1)")
    add("crGeomPFive", mu["norm"]["p05"], N.f2(mu["norm"]["p05"]), "5th percentile of ||mu||")
    add("crGeomPNinetyFive", mu["norm"]["p95"], N.f2(mu["norm"]["p95"]), "95th percentile of ||mu||")
    add("crGeomParticipation", c1["participation_ratio"], f"{c1['participation_ratio']:.1f}",
        "participation ratio tr(S)^2/tr(S^2) of the encoder means")
    add("crGeomPcNinety", c1["n_pc"]["90pct"], str(c1["n_pc"]["90pct"]), "principal components for 90 % of the variance")
    add("crGeomPcNinetyNine", c1["n_pc"]["99pct"], str(c1["n_pc"]["99pct"]), "principal components for 99 %")
    add("crGeomRadiusModel", c1["r_pred_model"], N.f4(c1["r_pred_model"]), "model radius sqrt(m) - Var/(8 m^1.5)")
    add("crGeomRadiusModelRelErrPct", 100 * c1["r_pred_model_rel_err"], f"{100 * c1['r_pred_model_rel_err']:.3f}",
        "relative error of the model radius, percent")
    add("crGeomWidthModel", c1["width_pred_model"], N.f3(c1["width_pred_model"]), "predicted SD of the norm")
    if not _scoped(ctx):          # posterior variance and posterior-sample shell: not printed (scope item 4)
        pv = g["posterior_variance"]
        add("crGeomSumSigma", pv["sum_i_E_sigma2"], f"{pv['sum_i_E_sigma2']:.1f}", "sum over dimensions of E sigma_i^2")
        z = g["z_cloud"]["norm"]
        add("crGeomZNorm", z["mean"], N.f2(z["mean"]), "mean ||mu + sigma eps|| (posterior samples)")
        add("crGeomZNormSd", z["sd_ddof1"], N.f2(z["sd_ddof1"]), "SD of the posterior-sample norm")
    meta = g["meta"]
    add("crGeomTokenLossN", meta["n_molecules_with_unknown_tokens"], N.intfmt(meta["n_molecules_with_unknown_tokens"]),
        "molecules encoded after dropping unknown (stereo) tokens")
    pct = 100.0 * meta["n_molecules_with_unknown_tokens"] / meta["n_encoded"]
    add("crGeomTokenLossPct", pct, f"{pct:.1f}", "percent of molecules with dropped tokens")
    s = g["sample_3000"]["stats"]
    add("crGeomSampleMean", s["norm"]["mean"], N.f4(s["norm"]["mean"]), "3,000-latent sample: mean norm (posted 'measured')")
    add("crGeomSamplePred", s["cov_ddof1"]["r_pred_posted_in_sample_formula"],
        N.f4(s["cov_ddof1"]["r_pred_posted_in_sample_formula"]), "3,000-latent sample: posted in-sample prediction")
    add("crGeomSampleParticipation", s["cov_ddof1"]["participation_ratio"], f"{s['cov_ddof1']['participation_ratio']:.1f}",
        "3,000-latent sample: participation ratio (posted 63.6)")
    v64 = 100.0 * s["cov_ddof1"]["cum_var_at_64_65_66"][0]
    add("crGeomSampleVarTopSixtyFour", v64, f"{v64:.1f}", "3,000-latent sample: percent of variance in the top-64 PCs "
        "(the PCA-sphere control's subspace)")
    f64 = 100.0 * c1["cum_var_at_64_65_66"][0]
    add("crGeomVarTopSixtyFour", f64, f"{f64:.1f}", "full set: percent of variance in the top-64 PCs")
    if ps:
        gs = G["strip"]["g"]
        shift = 100.0 * (gs["mu_cloud"]["norm"]["mean"] / mu["norm"]["mean"] - 1.0)
        add("crGeomStripShiftPct", shift, f"{shift:+.2f}", "radius change when stereo is stripped before encoding, percent")


def _col_values(g: dict, D: int) -> dict:
    """Row values of one geometry.json (covariance with ddof 1)."""
    mu = g["mu_cloud"]
    c1 = mu["cov_ddof1"]
    R0 = c1["participation_ratio"]
    rb = mu["norm"]["mean"]
    out = {"N": (mu["n"], N.intfmt(mu["n"])),
           "Rbar": (rb, N.f4(rb)), "RbarSd": (mu["norm"]["sd_ddof1"], N.f3(mu["norm"]["sd_ddof1"])),
           "PFive": (mu["norm"]["p05"], N.f2(mu["norm"]["p05"])),
           "PNinetyFive": (mu["norm"]["p95"], N.f2(mu["norm"]["p95"])),
           "Participation": (R0, f"{R0:.1f}"),
           "PcNinety": (c1["n_pc"]["90pct"], str(c1["n_pc"]["90pct"])),
           "PcNinetyNine": (c1["n_pc"]["99pct"], str(c1["n_pc"]["99pct"])),
           "RadiusModel": (c1["r_pred_model"], N.f4(c1["r_pred_model"])),
           "WidthModel": (c1["width_pred_model"], N.f3(c1["width_pred_model"])),
           "RelWidth": (mu["norm"]["sd_ddof1"] / rb, f"{mu['norm']['sd_ddof1'] / rb:.4f}"),
           "RelWidthModel": (1.0 / math.sqrt(2.0 * R0), f"{1.0 / math.sqrt(2.0 * R0):.4f}"),
           "RbarOverPrior": (rb / math.sqrt(D - 0.5), f"{rb / math.sqrt(D - 0.5):.3f}")}
    out["TrSigma"] = (c1["tr_Sigma"], f"{c1['tr_Sigma']:.2f}")
    out["MuBarSq"] = (mu["mean_vector_norm_sq"], f"{mu['mean_vector_norm_sq']:.2f}")
    out["RadiusModelRelErrPct"] = (100 * c1["r_pred_model_rel_err"], f"{100 * c1['r_pred_model_rel_err']:+.2f}")
    meta = g.get("meta") or {}
    if meta.get("n_encoded"):
        nd = meta.get("n_molecules_with_unknown_tokens", 0)
        out["TokenLossN"] = (nd, N.intfmt(nd))
        out["TokenLossPct"] = (100.0 * nd / meta["n_encoded"], f"{100.0 * nd / meta['n_encoded']:.1f}\\,\\%")
    pv = g.get("posterior_variance") or {}
    if "sum_i_E_sigma2" in pv:
        out["SumSigma"] = (pv["sum_i_E_sigma2"], f"{pv['sum_i_E_sigma2']:.1f}")
    if "n_dims_E_sigma2_gt_0p9" in pv:
        out["DimsSigmaHigh"] = (pv["n_dims_E_sigma2_gt_0p9"], str(pv["n_dims_E_sigma2_gt_0p9"]))
    zc = (g.get("z_cloud") or {}).get("norm")
    if zc:
        out["ZNorm"] = (zc["mean"], N.f2(zc["mean"]))
        out["ZNormSd"] = (zc["sd_ddof1"], N.f2(zc["sd_ddof1"]))
    return out


def _posted_values(g: dict, D: int) -> dict:
    """The 3,000-latent sample column: the four posted values verbatim, the rest recomputed on the
    same 3,000 latents (geometry.json sample_3000)."""
    s3 = g.get("sample_3000") or {}
    st, po, pt = s3.get("stats") or {}, s3.get("posted") or {}, s3.get("posterior_terms_on_sample_rows") or {}
    if not st:
        return {}
    c1 = st["cov_ddof1"]
    rb = po.get("r_obs_sample", st["norm"]["mean"])
    R0 = po.get("participation_ratio", c1["participation_ratio"])
    out = {"N": (st["n"], N.intfmt(st["n"])), "Rbar": (rb, N.f4(rb)),
           "RbarSd": (st["norm"]["sd_ddof1"], N.f3(st["norm"]["sd_ddof1"])),
           "PFive": (st["norm"]["p05"], N.f2(st["norm"]["p05"])),
           "PNinetyFive": (st["norm"]["p95"], N.f2(st["norm"]["p95"])),
           "Participation": (R0, f"{R0:.1f}"),
           "RadiusModel": (po.get("r_pred_posted", float("nan")), N.f4(po.get("r_pred_posted", float("nan")))),
           "WidthModel": (c1["width_pred_model"], N.f3(c1["width_pred_model"])),
           "RelWidth": (st["norm"]["sd_ddof1"] / rb, f"{st['norm']['sd_ddof1'] / rb:.4f}"),
           "RelWidthModel": (1.0 / math.sqrt(2.0 * R0), f"{1.0 / math.sqrt(2.0 * R0):.4f}"),
           "RbarOverPrior": (rb / math.sqrt(D - 0.5), f"{rb / math.sqrt(D - 0.5):.3f}")}
    if "n_pc_90" in po:
        out["PcNinety"] = (po["n_pc_90"], str(po["n_pc_90"]))
    elif "n_pc" in c1:
        out["PcNinety"] = (c1["n_pc"]["90pct"], str(c1["n_pc"]["90pct"]))
    if isinstance(c1.get("n_pc"), dict) and "99pct" in c1["n_pc"]:
        out["PcNinetyNine"] = (c1["n_pc"]["99pct"], str(c1["n_pc"]["99pct"]))
    if "sum_i_E_sigma2" in pt:
        out["SumSigma"] = (pt["sum_i_E_sigma2"], f"{pt['sum_i_E_sigma2']:.1f}")
    if "z_norm_mean" in pt:
        out["ZNorm"] = (pt["z_norm_mean"], N.f2(pt["z_norm_mean"]))
        out["ZNormSd"] = (pt["z_norm_sd"], N.f2(pt["z_norm_sd"]))
    return out


GEOM_ROWS = [("N", None, "Molecules encoded"),
             ("TokenLossN", "TokenLossPct", "Molecules with dropped tokens (count, share)"),
             ("Rbar", "RbarSd", "$\\bar r$ = mean $\\|\\mu(x)\\|$ $\\pm$ SD"),
             ("PFive", "PNinetyFive", "5th and 95th percentile of $\\|\\mu\\|$"),
             ("TrSigma", "MuBarSq", "$\\mathrm{tr}\\,\\Sigma$ and $\\|\\bar\\mu\\|^2$ (two values)"),
             ("Participation", None, "Participation ratio $R_0=\\mathrm{tr}(\\Sigma)^2/\\mathrm{tr}(\\Sigma^2)$"),
             ("PcNinety", "PcNinetyNine", "Principal components for 90\\,\\% and 99\\,\\% of the variance"),
             ("RadiusModel", "RadiusModelRelErrPct", "Model radius $\\sqrt{m}-\\mathrm{Var}(\\|\\mu\\|^2)/(8m^{3/2})$ (error vs $\\bar r$, \\%)"),
             ("WidthModel", None, "Model width $\\sqrt{\\mathrm{Var}(\\|\\mu\\|^2)}/(2\\sqrt{m})$"),
             ("RelWidth", "RelWidthModel", "Relative width SD$/\\bar r$ vs $1/\\sqrt{2R_0}$"),
             ("SumSigma", "DimsSigmaHigh", "$\\sum_i \\mathbb{E}\\,\\sigma_i^2(x)$ / dimensions with $\\mathbb{E}\\,\\sigma_i^2>0.9$"),
             ("ZNorm", "ZNormSd", "Posterior-sample norm $\\|\\mu+\\sigma\\epsilon\\|$ $\\pm$ SD"),
             ("RbarOverPrior", None, "$\\bar r/\\sqrt{D-1/2}$")]


def tab_geometry(ctx: Ctx) -> Table | None:
    table = "tab_geometry"
    G = geometry_inputs(ctx)
    if "default" not in G:
        ctx.skip(table, "geometry default/", "input absent (table omitted)")
        return None
    D = 256
    cols = []
    for col, key, lab in _cols(ctx):
        if col in G:
            cols.append((col, key, lab, _col_values(G[col]["g"], D), ctx.fs_paths(
                [G[col]["json"]] + [G[col][a] for a in ("mu_norm", "z_norm") if G[col][a]])))
        else:
            cols.append((col, key, lab, None, None))
            msg = (f"WARNING: geometry input {col}/ is missing under {ctx.extra.get('geometry_root')}: "
                   f"its column prints n/a")
            ctx.skip(table, col, msg)
            if not ctx.cache.get(f"warned_{col}"):
                ctx.cache[f"warned_{col}"] = True
                print("!" * 100 + "\n!!! " + msg + "\n" + "!" * 100, flush=True)
                ctx.cache.setdefault("loud_warnings", []).append(msg)
    # RUN_RULES 2026-11-02 (2): the paper states no history, so the column of the posted sample is not printed
    # legibility (RUN_RULES 2026-10-22): the quantity column wraps and the column headers stack, nothing is scaled
    hdr = ["Quantity"] + [stack(*lab.replace(" (", "|(").replace(", ", ",|").split("|"))
                          for _, _, lab, _, _ in cols]
    t = Table("geometry", "tab:geometry", "", WRAP + "r" * len(cols), hdr, size="\\footnotesize", wide=True,
              tabcolsep="4pt")
    for a, b, lab in GEOM_ROWS:
        if _scoped(ctx) and a in scope.GEOMETRY_ROWS_DROPPED:
            continue
        row = [lab]
        for col, key, _, vals, fs in cols:
            if vals is None or a not in vals:
                row.append("n/a" if vals is None else "--")
                continue
            va, ta = vals[a]
            ctx.num(f"crGeoTab{a}{key}", va, ta, kind="geometry", table=table, fileset=fs, n=None,
                    rule=f"{lab} ({col})")
            txt = ta
            if b and b in vals:
                vb, tb = vals[b]
                ctx.num(f"crGeoTab{b}{key}", vb, tb, kind="geometry", table=table, fileset=fs, n=None,
                        rule=f"{lab}, second value ({col})")
                # two values, not a ratio (RUN_RULES 2026-10-29 (4)): a semicolon between them
                sep = " \\pm " if b in ("RbarSd", "ZNormSd") else (" \\text{ vs } " if b == "RelWidthModel" else ";\\ ")
                txt = f"{ta}{sep}{tb}"
                if b in ("RadiusModelRelErrPct", "TokenLossPct"):
                    txt = f"{ta}\\;({tb})"
            row.append(m(txt))
        t.add(row)
    missing = [lab for col, _, lab, vals, _ in cols if vals is None]
    t.caption = ("\\textbf{Shell geometry of the encoder means $\\mu(x)$ of the VAE ($D=256$)"
                 + (", on ZINC-250k" if _scoped(ctx) else "") + ".} $\\Sigma$ is the covariance of $\\mu(x)$ and "
                 "$m=\\mathrm{tr}\\,\\Sigma+\\|\\bar\\mu\\|^2$, both with the $n-1$ normalization. SD is the "
                 "standard deviation of $\\|\\mu\\|$, also with $n-1$. The model radius and width are second-order "
                 "(Gaussian) delta-method values, with $\\mathrm{Var}(\\|\\mu\\|^2)=2\\,\\mathrm{tr}(\\Sigma^2)"
                 "+4\\bar\\mu^\\top\\Sigma\\bar\\mu$. They are computed on the same encodings, so they describe "
                 "the shell and do not predict it. "
                 + ("" if _scoped(ctx) else "Posterior samples $z=\\mu+\\sigma\\epsilon$ use one $\\epsilon$ per "
                                            "molecule. ")
                 + "Dropped tokens: SELFIES tokens outside the vocabulary of the VAE "
                 "(stereo tokens) are removed before encoding."
                 + ("" if _scoped(ctx) else " The GuacaMol training set is the corpus the VAE was trained on (analysis "
                                            "plan of 2026-10-12: printed whatever it shows).")
                 + (f" Not available in this run: {', '.join(missing)} (n/a)." if missing else ""))
    return t


def _shell_panel(ax, G: dict, cols: list, D: int, title: str, posterior: bool = True) -> tuple[list, int]:
    """Histograms of ||mu|| and ||z|| of the corpora in ``cols`` (first = filled, others = outlines), with marks
    at the first corpus's r-bar and at sqrt(D - 1/2). Returns (input paths, molecules of the first corpus)."""
    from .figures import plt      # noqa: F401  (rcParams: pdf.fonttype 42, DejaVu Sans)
    bins = np.linspace(4.0, 20.0, 321)
    styles = {"default": "#2471a3", "strip": "#1abc9c", "guacamol_train": "#c0392b"}
    labels = {"default": "ZINC-250k", "strip": "ZINC-250k, stereo stripped", "guacamol_train": "GuacaMol train"}
    paths, rbar, nmol = [], None, 0
    for i, col in enumerate(cols):
        mu = np.load(G[col]["mu_norm"])
        z = np.load(G[col]["z_norm"]) if posterior else None
        paths += [G[col]["mu_norm"], G[col]["json"]] + ([G[col]["z_norm"]] if posterior else [])
        if i == 0:
            rbar, nmol = float(mu.mean()), int(mu.size)
            ax.hist(mu, bins=bins, density=True, color=styles[col], alpha=0.55, label=f"$\\|\\mu(x)\\|$, {labels[col]}")
            if posterior:
                ax.hist(z, bins=bins, density=True, color="#7f8c8d", alpha=0.45,
                        label=f"$\\|\\mu+\\sigma\\epsilon\\|$, {labels[col]}")
        else:
            ax.hist(mu, bins=bins, density=True, histtype="step", color=styles[col], lw=1.0, ls="--",
                    label=f"$\\|\\mu(x)\\|$, {labels[col]}")
            if posterior:
                ax.hist(z, bins=bins, density=True, histtype="step", color=styles[col], lw=0.8, ls="--")
    # the two vertical marks are named in the legend (below the axes), so no text sits on the histograms
    ax.axvline(rbar, color=styles[cols[0]], lw=1.0, label=f"mean $\\|\\mu(x)\\|$: $\\bar r$ = {rbar:.2f}")
    ax.axvline(math.sqrt(D - 0.5), color="#333333", lw=1.0, ls=":",
               label=f"$\\sqrt{{D-1/2}}$ = {math.sqrt(D - 0.5):.2f}")
    ax.set_xlim(4, 20)
    ax.set_xlabel("latent norm")
    ax.set_title(f"{title} (n = {nmol:,})", fontsize=7)
    ax.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=1)
    return paths, nmol


def fig_shells(ctx: Ctx) -> str | None:
    """Histograms of ||mu|| and ||z|| with marks at r-bar and sqrt(D - 1/2) (TrueType fonts): fig_shells.pdf
    with one panel per corpus (ZINC-250k; GuacaMol training set when its geometry run is present), and the
    single-panel files fig_shells_zinc.pdf and fig_shells_guacamol.pdf."""
    from .figures import plt     # rcParams: pdf.fonttype 42, DejaVu Sans
    name = "fig_shells"
    G = geometry_inputs(ctx)

    post = not _scoped(ctx)       # camera-ready: ||mu|| of ZINC-250k only (scope item 4)

    def ok(col):
        return col in G and G[col]["mu_norm"] and (G[col]["z_norm"] or not post)

    if not ok("default"):
        ctx.skip(name, "geometry default/ arrays", "mu_norm.npy / z_norm.npy absent (figure omitted)")
        return None
    D = 256
    zinc_cols = ["default"] + (["strip"] if ok("strip") else [])
    panels = [("zinc", zinc_cols, "ZINC-250k", "crFigShellsN")]
    if _scoped(ctx):
        ctx.skip(name, "guacamol_train", "training-corpus panel not printed (RUN_RULES 2026-10-14, scope item 4)")
    elif ok("guacamol_train"):
        panels.append(("guacamol", ["guacamol_train"], "GuacaMol training set", "crFigShellsGuacamolN"))
    else:
        ctx.skip(name, "guacamol_train", "geometry of the GuacaMol training set absent (panel omitted)")
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 if len(panels) == 2 else 3.4, 3.3), squeeze=False)
    for ax, (key, cols, title, macro) in zip(axes[0], panels):
        paths, nmol = _shell_panel(ax, G, cols, D, title, posterior=post)
        ctx.num(macro, nmol, N.intfmt(nmol), kind="n", table=name, fileset=ctx.fs_paths(paths), n=nmol,
                rule=f"molecules in the filled histograms of the panel '{title}'")
    axes[0][0].set_ylabel("density")
    fig.tight_layout()
    path = os.path.join(ctx.outdir, f"{name}.pdf")
    fig.savefig(path)
    plt.close(fig)
    for key, cols, title, _ in panels:           # single-panel versions
        f1, ax = plt.subplots(figsize=(3.6, 3.3))
        _shell_panel(ax, G, cols, D, title, posterior=post)
        ax.set_ylabel("density")
        f1.tight_layout()
        p1 = os.path.join(ctx.outdir, f"{name}_{key}.pdf")
        f1.savefig(p1)
        plt.close(f1)
        ctx.written.append(p1)
    return path
