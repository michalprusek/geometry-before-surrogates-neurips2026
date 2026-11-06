#!/usr/bin/env python
"""Camera-ready generator: every table, figure and printed number from raw result JSONs.

NeurIPS 2026 submission 24168. Rules: docs/camera_ready/RUN_RULES.md (R-3: print the
camera-ready generation; R-5: matched sets; R-6: analysis defaults).

Camera-ready run (production; the exact command with its preflight is /data/prusek/cr_work/run_generator_final.sh).
The main campaign (roots CR = panda GPU arms, CRCPU = copy of kajman's GaBO / Riemannian Matern cells) and the
control block (root CRCTL) were run from two tagged commits, so every root has its own expected commit::

    python scripts/cr/analysis/make_tables.py \
        --expected-commit CR=<sha of cr-runs-2026-10-12> --expected-commit CRCPU=<the same sha> \
        --expected-commit CRCTL=<sha of cr-controls-2026-10-12> --expected-commit CRPUB=<sha of cr-verify-2026-10-13> \
        --expected-commit CRRD=<sha of the rerun of 2026-11-02, 1ea2f7d> \
        --expected-commit CRC4=<sha of cr-controls-lbfgs-2026-10-14, 2a899ad> \
        --root CR=<clone>/results/guacamol/cr2026 --root CRCPU=<copy of kajman's cr2026> \
        --root CRCTL=<control clone>/results/guacamol/cr2026ctl --root CRPUB=<published clone>/results/guacamol/cr2026pub \
        --root CRC4=<control clone>/results/guacamol/cr2026ctl \
        --root CRRD=<rerun clone>/results/guacamol/cr2026rd \
        --inputs-root <cr_inputs> --geometry-root <cr_inputs>/geometry_cr --lasso-cr-root <cr_inputs>/lassobench_cr \
        --lasso-pub-root <cr_inputs>/lassobench_pub --verdicts <cr_inputs>/verification/verdicts.json \
        --lasso-require-finished --p1-sel-root <cr_inputs>/p1_blocks_sel --outdir <dir outside the repo> --latex-check

Camera-ready mode prints only the promised content (RUN_RULES amendment 2026-10-14; crlib/scope.py): the arms of
unprinted analyses (the control block, the GP radius arms, the stopped waves 3b and 4, ...) are removed from the
configuration before any file is opened, the unprinted tables are not built, the variants R2_minmax and R4_E1 are
not run, and the posted generation is not loaded. --legacy keeps every builder.
--lasso-require-finished also requires the LassoBench published-method set to be finished, and the run needs the
pinned verdict file (crlib/published.py).

--lasso-require-finished (passed by the final run, not by the smoke run) stops the run with exit 2 before anything
is loaded while the LassoBench rerun is unfinished, i.e. while an expected job has no job record (on 2026-10-12:
the 39 Papenmeier leukemia jobs, seeds 43-81) or the rerun folder is absent. Without it an unfinished rerun is
tabulated with loud warnings and "INCOMPLETE RERUN: not the final table" in the captions.

Legacy validation (the camera-ready arm names pointed at the OLD stored files; compares the
generator with numbers that earlier audits verified independently)::

    python scripts/cr/analysis/make_tables.py --legacy --inputs-root <cr_inputs> --outdir <dir> --latex-check

Smoke run (task rano, seeds 42-43 and, for the replication arms, seed 82; arms or cells not run yet must be
declared explicitly; an arm whose seed range lies outside --seeds is reported as absent, not as an error)::

    python scripts/cr/analysis/make_tables.py --expected-commit CR=20ad7d5029a488bc223e95d22ce5f1f217aa4420 \
        --expected-commit CRCPU=20ad7d5029a488bc223e95d22ce5f1f217aa4420 \
        --expected-commit CRCTL=fd8a03c5a75475e79ed8e9f460c38514a7e9717d --outdir <dir> --inputs-root <cr_inputs> \
        --root CR=<clone>/results/guacamol/cr_smoke2 --root CRCPU=<clone>/results/guacamol/cr_smoke2 \
        --root CRCTL=<control clone>/results/guacamol/ctl_smoke \
        --tasks rano --seeds 42-43,82 --allow-missing-cells gabo=rano:43,matern_gabo=rano:43 \
        --allow-missing-arms hvarfner_scaled_gp_sphere05,hvarfner_scaled_gp_sphere15,hvarfner_scaled_gp_sphere20,\
random_search__s82,random_box_faithful__s82,hvarfner_scaled_gp__s82,\
hvarfner_scaled_gp_sphere_continuous__s82,hvarfner_scaled_gp_box_faithful__s82

Outputs (in --outdir; the variants in variants/<name>/):
    tab_<name>.tex, fig_<name>.pdf, numbers.tex, numbers_dossier.{json,md},
    loader_report.{json,md}, run_manifest.json, legacy_check.{json,md} (--legacy only).
    records/ (camera-ready mode): repository records that the paper does not print -- tab_baselines.tex and
    tab_arm_status.tex (the Appendix H cards and the run status, RUN_RULES 2026-10-24), BASELINE_CARDS.md,
    RUN_STATUS.md and SOURCE.json (committed as docs/baselines/), tab_changes_vs_submitted.tex; since RUN_RULES
    2026-10-29 also the full Table 1 (tab_main_full.tex, tab_main_full_notes.tex, TABLE1_FULL.md) and the full
    LassoBench table (tab_lassobench_full.tex), committed as docs/baselines/ too, the compute table (tab_compute.tex),
    the per-task radius table (tab_radius_pertask.tex) and the figures fig_trajectories.pdf and fig_lassobench.pdf.
    tab_estimators and tab_shells are blocks of tab_radius.tex, tab_pca_directions the last block of tab_convex.tex
    (crlib/scope.py MERGED_TABLES); their numbers keep their names and their place in numbers.tex.
    lassobench_cr_coverage.md, lassobench_pub_coverage.md (camera-ready mode: coverage of the LassoBench rerun and
    of the LassoBench published methods).
The loader fails (exit 2) on a missing arm, a missing or duplicate cell, a provenance mismatch
or contradictory file content; non-completed and fallback cells are counted, never dropped silently.
A broken stored input set (September blocks, LassoBench rerun provenance), an unfinished LassoBench rerun
under --lasso-require-finished or a tripped guard (the Table 2 MSR row) also ends the run with exit 2.
Exit codes: 0 ok, 1 a legacy check failed, 2 loader / inputs / configuration, 3 LaTeX error or Type 3 font.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from crlib import (baselines, compute, controls, decomp, extras, fidelity, figures, floor, geometry,  # noqa: E402
                   hyper, lasso_cr, misc, overview, published, records, reference, scope, september, stored,
                   supports, tables)
from crlib.config import ConfigError, load_config, parse_seeds  # noqa: E402
from crlib.context import VARIANTS, Ctx  # noqa: E402
from crlib.engine import Engine  # noqa: E402
from crlib.latex import PREAMBLE_CHECK  # noqa: E402
from crlib.loader import LoaderError, load_all  # noqa: E402
from crlib.numbers import Registry, text_macros  # noqa: E402

TABLE_BUILDERS = [
    tables.tab_main, tables.tab_boxsphere, tables.tab_factorial_details, tables.tab_trajectories,
    stored.tab_crossdomain, tables.tab_sensitivity, tables.tab_pertask, tables.tab_pertask_full,
    stored.tab_budget, stored.tab_crosscodec, tables.tab_radius, tables.tab_radius_pertask, tables.tab_shells,
    tables.tab_estimators, tables.tab_convex, tables.tab_pca_directions, tables.tab_shell_prior,
    floor.tab_floor, tables.tab_box_sensitivity, stored.tab_lassobench, stored.tab_lassobench_correspondence,
    tables.tab_arm_status,
    fidelity.tab_budget_fidelity,
    # camera-ready analysis plan of 2026-10-12 (GEN v2)
    decomp.tab_decomposition, decomp.tab_decomposition_contrasts, decomp.tab_decomposition_contrasts_byN,
    decomp.tab_decomposition_pertask, decomp.tab_factorial_by_box, decomp.tab_equivalence, decomp.tab_distinct,
    decomp.tab_gp_radius, geometry.tab_geometry, supports.tab_support_geometry, supports.tab_effective_support,
    baselines.tab_baselines,          # Appendix H: the baseline cards, rendered from appendixH_facts.csv
    # GEN v5: published-recipe block (RUN_RULES 2026-10-12, 2026-10-13 revision 1, 2026-10-19)
    published.tab_published, published.tab_verification,
    # GEN v3: control block (RUN_RULES amendment 2026-10-12, tag cr-controls-2026-10-12) and synthesis additions
    controls.tab_controls, controls.tab_replication, extras.tab_decomposition_robustness, extras.tab_anytime,
    extras.tab_linear_bo,
    # stored September runs (E-1 on its extra tasks; p1 blocks: selection-rule robustness)
    september.tab_e1_extra_tasks, september.tab_selection_rule,
    # RUN_RULES 2026-10-24: Appendix A settings table, Appendix K compute table, and the Appendix H overview (last:
    # it summarizes the cards tab_baselines printed; tab_baselines and tab_arm_status are repository records)
    hyper.tab_hyper, compute.tab_compute, overview.tab_baselines_overview,
]
# failures of stored input sets and guards that stop a run like a loader problem (exit 2)
#: review S4: the input commits of a camera-ready run, pinned in code (tag cr-runs-2026-10-12 for the GuacaMol runs on
#: panda and the CPU arms copied from kajman; tag cr-verify-2026-10-13 for the published-recipe arms, as the verdicts;
#: commit 1ea2f7d for the rerun of 2026-11-02, root CRRD: cmaes_full_rd, cmaes_full_rd_pop5 and baxus_cr)
RD_COMMIT = "1ea2f7d8fdbec151c59c91e68bc15f45bc79379e"
#: commit 2a899ad (tag cr-controls-lbfgs-2026-10-14) for control C4, root CRC4: hvarfner_scaled_gp_sphere_lbfgs, the shell x
#: continuous cell of the printed factorial since RUN_RULES 2026-11-06 16:30
C4_COMMIT = "2a899ad301d3cb0badd906adfb4d81ad0b50f168"
PINNED_COMMITS = {"CR": lasso_cr.LASSO_CR_COMMIT, "CRCPU": lasso_cr.LASSO_CR_COMMIT, "CRPUB": published.VERDICTS_TAG_COMMIT,
                  "CRRD": RD_COMMIT, "CRC4": C4_COMMIT}
INPUT_ERRORS = (september.SeptemberError, tables.MsrGuardError, lasso_cr.LassoLoadError, published.PublishedInputError,
                tables.TrajectoryGuardError)
FIGURE_BUILDERS = [figures.fig_convergence_grid, figures.fig_trajectories, figures.fig_lassobench, geometry.fig_shells]


def git(*args, cwd=HERE) -> str:
    try:
        return subprocess.run(["git", "--no-optional-locks", *args], cwd=cwd, capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"ERR {e}"


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_loader_report(ds, outdir: str) -> None:
    with open(os.path.join(outdir, "loader_report.json"), "w") as f:
        json.dump(ds.report, f, indent=1, default=str)
    L = ["# Loader report", "", f"Config `{ds.report['config']}` (sha256 `{ds.report['config_sha256'][:16]}`), "
         f"cell filter `{ds.report['cell_filter']}`, expected commit `{ds.report['expected_commit']}`, "
         f"tasks {ds.report['tasks']}, seeds {ds.report['seeds']}.", ""]
    if ds.absent:
        L += ["## Absent arms (allowed)", ""] + [f"- `{k}`: {v}" for k, v in sorted(ds.absent.items())] + [""]
    if ds.report.get("allow_missing_cells_cli"):
        L += ["## Missing cells declared on the command line", ""] + [f"- `{x}`" for x in ds.report["allow_missing_cells_cli"]] + [""]
    L += ["## Cold-start consistency (first five scores identical across 'ours' arms)", ""]
    for c in ds.report["coldstart_consistency"]:
        L.append(f"- {c['namespace']}: {c['cells_compared']} cell comparisons, mismatches {c['mismatching_cells_by_arm'] or 'none'}")
    L += ["", "## Arms", "",
          "| arm | kind | cells | status | not completed | fallback cells/steps | missing (declared) | provenance states | excluded files |",
          "|---|---|---|---|---|---|---|---|---|"]
    for n, r in sorted(ds.report["arms"].items()):
        if r.get("absent"):
            L.append(f"| `{n}` | {r['kind']} | ABSENT | {r['absent']} | | | | | |")
            continue
        ex = r.get("excluded", {})
        exs = ", ".join(f"{k}:{v}" for k, v in ex.get("suffix_variants", {}).items())
        if ex.get("other_tasks"):
            exs += f"; other tasks {sum(ex['other_tasks'].values())}"
        prov = "; ".join(f"{k} x{v}" for k, v in list(r["provenance_states"].items())[:4])
        if len(r["provenance_states"]) > 4:
            prov += f"; ... ({len(r['provenance_states'])} states)"
        L.append(f"| `{n}` | {r['kind']} | {r['n_cells_loaded']} | {r['status_counts']} | "
                 f"{', '.join(r['noncompleted_cells'][:6])}{' ...' if len(r['noncompleted_cells']) > 6 else ''} | "
                 f"{r['fallback_cells']}/{r['fallback_steps']}{'' if r['fallback_recorded_cells'] else ' (not recorded)'} | "
                 f"{len(r['missing_cells'])} ({', '.join(r['known_missing'][:6])}) | {prov} | {exs} |")
    with open(os.path.join(outdir, "loader_report.md"), "w") as f:
        f.write("\n".join(L) + "\n")


NUMBER_FUNCTIONS = [geometry.geometry_numbers, extras.anytime_legacy_numbers, misc.acq_numbers, fidelity.extra_numbers]


def run_variant(ds, var, outdir: str, mode: str, script_commit: str, extra: dict, ref: dict | None,
                registries: dict, do_figures: bool, log) -> Ctx:
    eng = Engine(ds, conv=var.conv, cell_filter=var.cell_filter)
    reg = Registry(script_commit, ds.config.sha256, var.name)
    ctx = Ctx(ds=ds, eng=eng, reg=reg, var=var, outdir=outdir, script_commit=script_commit, mode=mode,
              extra=dict(extra))
    os.makedirs(outdir, exist_ok=True)
    scoped = mode == "cr"          # camera-ready mode prints only the promised content (crlib/scope.py)
    rdir = os.path.join(outdir, "records")
    record_tables: dict = {}
    record_extra: dict = {}        # further repository records (RUN_RULES 2026-10-29): {file name: path}
    hosts = {h for h, _ in scope.MERGED_TABLES.values()} if scoped else set()
    merged: dict = {}              # host builder -> [(builder, Table)] (RUN_RULES 2026-10-29 (2))
    deferred: dict = {}            # host builder -> its Table, written once its blocks are built

    def write_printed(name: str, t) -> None:
        ctx.written.append(t.write(outdir, ctx.prov_comment(name)))
        ctx.written += t.extra_paths                # e.g. tab_main_notes.tex (RUN_RULES 2026-10-22, format only)
        if t.record_tables or t.record_files:
            # RUN_RULES 2026-10-29: the full version of a printed table is a repository record (same provenance line,
            # so its bytes are those of the table printed before the cut)
            os.makedirs(rdir, exist_ok=True)
            for rt in t.record_tables:
                p = rt.write(rdir, ctx.prov_comment(name))
                ctx.written += [p] + rt.extra_paths
                for q in [p] + rt.extra_paths:
                    record_extra[os.path.basename(q)] = q
            for fn, text in t.record_files.items():
                p = os.path.join(rdir, fn)
                with open(p, "w", encoding="utf-8") as f:
                    f.write(text)
                ctx.written.append(p)
                record_extra[fn] = p

    for b in TABLE_BUILDERS:
        if scoped and b.__name__ in scope.NOT_PRINTED_TABLES:
            ctx.not_built.append({"what": b.__name__, "why": scope.NOT_PRINTED_TABLES[b.__name__]})
            log(f"    [{var.name}] {b.__name__:28s} not built (scope: {scope.NOT_PRINTED_TABLES[b.__name__]})")
            continue
        t0 = time.time()
        t = b(ctx)
        where = "written"
        if t is not None and scoped and b.__name__ in scope.RECORD_ONLY_TABLES:
            # RUN_RULES 2026-10-24: a repository record, written to records/ with the same bytes as a printed table
            os.makedirs(rdir, exist_ok=True)
            ctx.written.append(t.write(rdir, ctx.prov_comment(b.__name__)))
            ctx.written += t.extra_paths
            ctx.not_built.append({"what": b.__name__, "why": "record only: " + scope.RECORD_ONLY_TABLES[b.__name__]})
            record_tables[b.__name__] = t
            where = "written to records/"
        elif t is not None and scoped and b.__name__ in scope.MERGED_TABLES:
            # RUN_RULES 2026-10-29 (2): its numbers are registered under its own name; its blocks join the host table
            host, why = scope.MERGED_TABLES[b.__name__]
            merged.setdefault(host, []).append((b.__name__, t))
            ctx.not_built.append({"what": b.__name__, "why": f"merged into {host}: {why}"})
            where = f"merged into {host}"
        elif t is not None and b.__name__ in hosts:
            deferred[b.__name__] = t
            where = "built (written with its merged blocks)"
        elif t is not None:
            write_printed(b.__name__, t)
        log(f"    [{var.name}] {b.__name__:28s} {where if t is not None else 'SKIPPED'} ({time.time() - t0:.1f}s)")
    for host in sorted(hosts):
        parts = merged.get(host, [])
        if host in deferred:
            write_printed(host, tables.merge_blocks(ctx, host, deferred[host], parts))
            log(f"    [{var.name}] {host:28s} written with the blocks of {', '.join(n for n, _ in parts) or 'none'}")
        else:                   # no host table: the merged tables are printed on their own
            for name, t in parts:
                write_printed(name, t)
    if record_tables:
        # the cards and the run status as Markdown for the repository (docs/baselines/), with SOURCE.json
        ctx.written += records.write_records(ctx, rdir, record_tables, record_extra)
    for f in NUMBER_FUNCTIONS:
        if scoped and f.__name__ in scope.NOT_PRINTED_NUMBERS:
            ctx.not_built.append({"what": f.__name__, "why": scope.NOT_PRINTED_NUMBERS[f.__name__]})
            continue
        f(ctx)
    if do_figures:
        for fb in FIGURE_BUILDERS:
            p = fb(ctx)
            where = "written"
            if p and scoped and fb.__name__ in scope.RECORD_ONLY_FIGURES:
                # RUN_RULES 2026-10-29 (2): a repository record, moved to records/ with the same bytes
                os.makedirs(rdir, exist_ok=True)
                q = os.path.join(rdir, os.path.basename(p))
                os.replace(p, q)
                p, where = q, "written to records/"
                ctx.not_built.append({"what": fb.__name__,
                                      "why": "record only: " + scope.RECORD_ONLY_FIGURES[fb.__name__]})
            if p:
                ctx.written.append(p)
            log(f"    [{var.name}] {fb.__name__:28s} {where if p else 'SKIPPED'}")
    registries[var.name] = reg
    if ref is not None and var.name == "default":
        for b in (reference.tab_posted_vs_printed, reference.tab_changes_vs_submitted):
            if scoped and b.__name__ in scope.NOT_PRINTED_TABLES:
                ctx.not_built.append({"what": b.__name__, "why": scope.NOT_PRINTED_TABLES[b.__name__]})
                continue
            t = b(ctx, ref, registries)
            if t is not None and scoped and b.__name__ in scope.RECORD_ONLY_TABLES:
                rdir = os.path.join(outdir, "records")        # a repository record, not a printed table
                os.makedirs(rdir, exist_ok=True)
                ctx.written.append(t.write(rdir, ctx.prov_comment(b.__name__) + "; NOT PRINTED (repository record)"))
                ctx.not_built.append({"what": b.__name__, "why": "record only: " + scope.RECORD_ONLY_TABLES[b.__name__]})
            elif t is not None:
                ctx.written.append(t.write(outdir, ctx.prov_comment(b.__name__)))
            log(f"    [{var.name}] {b.__name__:28s} {'written' if t is not None else 'SKIPPED'}")
    header = f"Camera-ready numbers ({mode} mode), scripts/cr/analysis/make_tables.py"
    reg.write_tex(os.path.join(outdir, "numbers.tex"), header)
    reg.write_dossier(os.path.join(outdir, "numbers_dossier.json"), os.path.join(outdir, "numbers_dossier.md"),
                      {"mode": mode, "variant": var.name, "variant_description": var.description,
                       "script_commit": script_commit, "config": ds.config.path, "config_sha256": ds.config.sha256,
                       "cell_filter": eng.cell_filter, "levels": var.conv, "skipped": ctx.skipped,
                       "not_built": ctx.not_built})
    ctx.written += [os.path.join(outdir, f) for f in ("numbers.tex", "numbers_dossier.json", "numbers_dossier.md")]
    return ctx


def latex_check(outdir: str, files: list[str], log) -> dict:
    """Compile every table (and numbers.tex) in one throw-away document."""
    if not shutil.which("pdflatex"):
        return {"ok": None, "note": "pdflatex not available"}
    d = os.path.join(outdir, "_latex_check")
    os.makedirs(d, exist_ok=True)
    body = [PREAMBLE_CHECK, "\\input{../numbers.tex}"]
    nt = os.path.join(outdir, "numbers.tex")
    names = text_macros(nt) if os.path.exists(nt) else []
    if names:   # every running-text macro set in a paragraph: a sentence that cannot break shows up as an overfull line
        body.append("Running text: " + "; ".join(f"\\{n}" for n in names) + ".\\clearpage")
    for f in sorted(files):
        if os.path.basename(f).startswith("tab_") and f.endswith(".tex") and os.path.dirname(f) == outdir:
            body.append(f"\\input{{../{os.path.basename(f)}}}\\clearpage")
    body.append("\\end{document}")
    with open(os.path.join(d, "check.tex"), "w") as fh:
        fh.write("\n".join(body))
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "check.tex"], cwd=d,
                       capture_output=True, text=True, timeout=300)
    ok = r.returncode == 0
    errs = [ln for ln in r.stdout.splitlines() if ln.startswith("!")][:10]
    log_txt = open(os.path.join(d, "check.log"), errors="replace").read() if os.path.exists(os.path.join(d, "check.log")) else ""
    over = [ln for ln in log_txt.splitlines() if "Overfull \\hbox" in ln][:40]
    log(f"  latex check: {'OK' if ok else 'FAILED'} {errs}")
    return {"ok": ok, "errors": errs, "textwidth": "5.5in (NeurIPS)", "overfull_hbox": over,
            "pdf": os.path.join(d, "check.pdf") if ok else None}


def pdffonts_check(paths: list[str]) -> dict:
    out = {}
    exe = shutil.which("pdffonts")
    for p in paths:
        if not p.endswith(".pdf"):
            continue
        if not exe:
            out[p] = {"ok": None, "note": "pdffonts not available"}
            continue
        r = subprocess.run([exe, p], capture_output=True, text=True)
        lines = r.stdout.splitlines()[2:]
        out[p] = {"ok": ("Type 3" not in r.stdout) and bool(lines), "pdffonts": lines}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", default=None, help="sources config (default sources_cr.json, or sources_legacy.json with --legacy)")
    ap.add_argument("--legacy", action="store_true", help="validate against the old stored files (sources_legacy.json)")
    ap.add_argument("--expected-commit", action="append", default=[],
                    help="commit the camera-ready files must carry (prefix match). Repeatable: ROOT=SHA sets the "
                         "commit of one root (CR, CRCPU = the main campaign; CRCTL = the control block; CRPUB = the "
                         "published-recipe block; CRRD = the rerun of 2026-11-02), a bare SHA is the default for "
                         "roots without their own entry")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--root", action="append", default=[], help="override a config root, NAME=PATH (repeatable)")
    ap.add_argument("--tasks", default=None, help="comma-separated short task ids (subset)")
    ap.add_argument("--seeds", default=None, help="seed subset, e.g. 42-43")
    ap.add_argument("--allow-missing-arms", default="", help="comma-separated arms that may be absent (echoed in the report)")
    ap.add_argument("--allow-missing-cells", default="",
                    help="extra declared missing cells, comma-separated ARM=TASK:SEED (added to the arm's known_missing; echoed)")
    ap.add_argument("--require-optional", action="store_true", help="treat optional (amendment) arms as required")
    ap.add_argument("--variants", default="all", help="'all' or comma-separated variant names (default always included)")
    ap.add_argument("--reference", default=os.path.join(HERE, "reference_values.json"))
    ap.add_argument("--inputs-root", default=os.environ.get("CR_INPUTS_ROOT") or None,
                    help="folder with the generator inputs, cr_inputs/ of the results bundle (precomputed/, "
                         "lassobench*/, nasbench/, p1_blocks*/, e0_floor/, geometry_cr/, verification/, ...). "
                         "Required; without the option the environment variable CR_INPUTS_ROOT is used")
    ap.add_argument("--geometry-root", default=None,
                    help="folder with default/, strip/ and guacamol_train/ (each geometry.json + mu_norm.npy + "
                         "z_norm.npy); default <inputs-root>/geometry_cr (the runs of the tagged commit), else "
                         "<inputs-root>/geometry_dev. A missing subfolder prints n/a with a loud warning")
    ap.add_argument("--lasso-cr-root", default=None,
                    help="folder with the LassoBench full rerun (full_rerun/, leukemia_rerun/ ...; default "
                         "<inputs-root>/lassobench_cr)")
    ap.add_argument("--lasso-pub-root", default=None,
                    help="folder with the LassoBench published methods (pub_methods/ and queue/, a byte copy made by "
                         "tools/sync_lassobench_pub.sh; default <inputs-root>/lassobench_pub)")
    ap.add_argument("--verdicts", default=None,
                    help="verification verdicts (verdicts.json of the summarize pass of 2026-10-19, sha256 pinned in "
                         "crlib/published.py; default <inputs-root>/verification/verdicts.json)")
    ap.add_argument("--lasso-allow-unclean", default="",
                    help="comma-separated LassoBench result file names (<method>_<task>_s<seed>.json) whose job record "
                         "is not clean (e.g. rc != 0 of a multi-method job) and which are used anyway (echoed)")
    ap.add_argument("--validation-run", action="store_true",
                    help="smoke runs, tests and mutation copies only: camera-ready mode WITHOUT the clean-tree "
                         "requirement and WITHOUT the code pins of the camera-ready input commits (PINNED_COMMITS). "
                         "Recorded in the run manifest; the final wrapper never passes it")
    ap.add_argument("--lasso-require-finished", action="store_true",
                    help="camera-ready mode: exit 2 unless the LassoBench rerun is finished (every expected job of "
                         "every run folder has a job record; timed-out and non-finite runs are final outcomes). "
                         "The final run passes it; without it an unfinished rerun is tabulated and marked INCOMPLETE")
    ap.add_argument("--p1-sel-root", default=None,
                    help="folder with the September p1 blocks of all seven solvers (default <inputs-root>/p1_blocks_sel)")
    ap.add_argument("--support-cache", default=None,
                    help="JSON file that caches the candidate norms of the supports (development convenience; the "
                         "cache is used only if its key = hash of every input and of crlib/mc.py matches)")
    ap.add_argument("--floor-reps", type=int, default=None,
                    help="Monte Carlo replicates of the floor's exact null (default 1,000,000; smaller only for tests)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--latex-check", action="store_true")
    args = ap.parse_args(argv)

    t_start = time.time()
    mode = "legacy" if args.legacy else "cr"
    src = args.sources or os.path.join(HERE, "sources_legacy.json" if args.legacy else "sources_cr.json")
    outdir = os.path.abspath(args.outdir)
    repo = git("rev-parse", "--show-toplevel")
    if repo and not repo.startswith("ERR") and os.path.realpath(outdir).startswith(os.path.realpath(repo) + os.sep):
        print("refusing to write inside the repository", file=sys.stderr)
        return 2
    if mode == "cr" and not args.expected_commit:
        print("--expected-commit is required for a camera-ready run", file=sys.stderr)
        return 2
    no_inputs_root = "--inputs-root is required (or set the environment variable CR_INPUTS_ROOT)"
    if mode == "cr" and not args.inputs_root:
        print(no_inputs_root, file=sys.stderr)
        return 2
    expected: dict = {}
    for item in args.expected_commit:
        root, _, sha = item.rpartition("=")
        key = root or "*"
        if key in expected and expected[key] != sha:
            print(f"--expected-commit: two different commits for {key}", file=sys.stderr)
            return 2
        if len(sha) < 7:
            print(f"--expected-commit: {item!r} is not a commit (at least 7 hex digits)", file=sys.stderr)
            return 2
        expected[key] = sha
    names = list(VARIANTS) if args.variants == "all" else ["default"] + [v for v in args.variants.split(",") if v != "default"]
    unknown = [v for v in names if v not in VARIANTS]
    if unknown:
        print(f"--variants: unknown {unknown}", file=sys.stderr)
        return 2
    if mode == "cr":
        asked = [v for v in names if v in scope.NOT_RUN_VARIANTS and args.variants != "all"]
        if asked:
            print(f"--variants: {asked} are not run in camera-ready mode (crlib/scope.py)", file=sys.stderr)
            return 2
        names = [v for v in names if v not in scope.NOT_RUN_VARIANTS]
    os.makedirs(outdir, exist_ok=True)
    if args.lasso_require_finished and mode == "cr":
        # the final run: an unfinished LassoBench rerun stops everything before a table is written
        lasso_root = args.lasso_cr_root or os.path.join(args.inputs_root, "lassobench_cr")
        pub_root = args.lasso_pub_root or os.path.join(args.inputs_root, "lassobench_pub")
        sets = {}
        for what, root, kw in (("RERUN", lasso_root, {"allow_unclean": [x for x in args.lasso_allow_unclean.split(",") if x]}),
                               ("PUBLISHED-METHOD SET", pub_root, {"design": lasso_cr.PUB})):
            try:
                sets[what] = lasso_cr.load_rerun(root, require_finished=True, **kw)
            except lasso_cr.LassoLoadError as e:
                msg = (f"LASSOBENCH {what} NOT USABLE FOR A FINAL RUN (--lasso-require-finished, root {root}): "
                       f"{type(e).__name__}: {e}")
                print(msg, file=sys.stderr)
                with open(os.path.join(outdir, "LOADER_FAILED.txt"), "w") as f:
                    f.write(msg + "\n")
                return 2
        # review S7: one objective for both sets (shared cold starts; counts only) before any table is written
        chk = lasso_cr.objective_consistency(sets["RERUN"], sets["PUBLISHED-METHOD SET"])
        if chk["beyond_tol"] or not chk["compared"]:
            msg = (f"LASSOBENCH OBJECTIVE CHECK FAILED (--lasso-require-finished): {chk['beyond_tol']} of "
                   f"{chk['compared']} published-method runs evaluate the shared cold start to other losses than the "
                   f"rerun's GeoWalk (relative tolerance {chk['rtol']:g}; by method {chk['beyond_by_method']})")
            print(msg, file=sys.stderr)
            with open(os.path.join(outdir, "LOADER_FAILED.txt"), "w") as f:
                f.write(msg + "\n")
            return 2
        try:
            published.load_verdicts(args.verdicts or os.path.join(args.inputs_root, "verification", "verdicts.json"))
        except published.PublishedInputError as e:
            msg = f"VERIFICATION VERDICTS NOT USABLE FOR A FINAL RUN: {e}"
            print(msg, file=sys.stderr)
            with open(os.path.join(outdir, "LOADER_FAILED.txt"), "w") as f:
                f.write(msg + "\n")
            return 2
    log_lines = []

    def log(msg):
        print(msg, flush=True)
        log_lines.append(msg)

    script_commit = git("rev-parse", "HEAD")
    dirty = git("status", "--porcelain", "--", HERE)
    script_state = script_commit[:12] + ("+dirty" if dirty else "")
    if mode == "cr" and not args.validation_run:
        # review S4: a camera-ready run comes from a committed, clean generator and from the pinned input commits
        if not re.fullmatch(r"[0-9a-f]{40}", script_commit) or dirty:
            print(f"camera-ready run refused: the generator tree {HERE} is "
                  + ("not a git checkout" if not re.fullmatch(r"[0-9a-f]{40}", script_commit) else
                     f"dirty ({len(dirty.splitlines())} paths)") + " (--validation-run is for smoke runs and tests)",
                  file=sys.stderr)
            return 2
        wrong = [f"{k}={expected.get(k)} (pinned {v[:12]})" for k, v in PINNED_COMMITS.items()
                 if not (expected.get(k) and v.startswith(expected[k]))]
        if wrong:
            print("camera-ready run refused: --expected-commit differs from the commits pinned in the generator: "
                  + ", ".join(wrong), file=sys.stderr)
            return 2
    roots = dict(r.split("=", 1) for r in args.root)
    try:
        cfg = load_config(src, roots, args.tasks.split(",") if args.tasks else None,
                          parse_seeds(args.seeds) if args.seeds else None, load_includes=(mode != "cr"))
    except ConfigError as e:
        print(f"CONFIG ERROR: {e}", file=sys.stderr)
        return 2
    if not args.inputs_root:                 # --legacy (camera-ready mode checked it above)
        print(no_inputs_root, file=sys.stderr)
        return 2
    not_loaded = {}
    if mode == "cr":
        # scope (RUN_RULES 2026-10-14): unprinted arms are removed BEFORE the loader opens a file
        bad = scope.check_scope(set(cfg.arms), {b.__name__ for b in TABLE_BUILDERS}
                                | {"tab_posted_vs_printed", "tab_changes_vs_submitted"}, set(VARIANTS),
                                {f.__name__ for f in NUMBER_FUNCTIONS}, {f.__name__ for f in FIGURE_BUILDERS})
        if bad:
            print("SCOPE ERROR: names in crlib/scope.py that do not exist: " + ", ".join(bad), file=sys.stderr)
            return 2
        for a in sorted(scope.NOT_LOADED_ARMS):
            if a in cfg.arms:
                not_loaded[a] = scope.NOT_LOADED_ARMS[a]
                del cfg.arms[a]
        dangling = sorted(f"{s.name} -> {s.coldstart_from}" for s in cfg.arms.values()
                          if s.coldstart_from and s.coldstart_from not in cfg.arms)
        if dangling:
            print("SCOPE ERROR: a loaded arm takes its cold start from an arm that is not loaded: " + ", ".join(dangling),
                  file=sys.stderr)
            return 2
    log(f"mode {mode}; config {cfg.path}; {len(cfg.arms)} arms; tasks {cfg.tasks}; "
        f"{len(cfg.seeds)} seeds; script {script_state}")
    if not_loaded:
        log(f"SCOPE ({scope.RULE}): {len(not_loaded)} arms not loaded: " + ", ".join(sorted(not_loaded)))
    allow = {a for a in args.allow_missing_arms.split(",") if a}
    extra_missing = []
    for item in [x for x in args.allow_missing_cells.split(",") if x]:
        arm, cell = item.split("=")
        t, s_ = cell.split(":")
        if arm not in cfg.arms:
            print(f"--allow-missing-cells: unknown arm {arm}", file=sys.stderr)
            return 2
        if t in cfg.arms[arm].tasks and int(s_) in cfg.arms[arm].seeds:
            cfg.arms[arm].known_missing.add((t, int(s_)))
            extra_missing.append(item)
    if extra_missing:
        log("DECLARED MISSING CELLS (command line): " + ", ".join(extra_missing))
    try:
        ds = load_all(cfg, expected if mode == "cr" else None, allow, args.require_optional, log=log)
    except LoaderError as e:
        print(f"LOADER FAILED:\n{e}", file=sys.stderr)
        with open(os.path.join(outdir, "LOADER_FAILED.txt"), "w") as f:
            f.write(str(e) + "\n")
        return 2
    ds.report["allow_missing_cells_cli"] = extra_missing
    write_loader_report(ds, outdir)
    if ds.absent:
        log("ABSENT ARMS (allowed): " + "; ".join(f"{k}: {v}" for k, v in sorted(ds.absent.items())))
    ref = reference.load_reference(args.reference) if os.path.exists(args.reference) else None
    geometry_root = args.geometry_root or os.path.join(
        args.inputs_root, "geometry_cr" if os.path.isdir(os.path.join(args.inputs_root, "geometry_cr")) else "geometry_dev")
    extra = {"inputs_root": args.inputs_root,
             "lasso_dir": os.path.join(args.inputs_root, "lassobench"),
             "nas_dir": os.path.join(args.inputs_root, "nasbench"),
             "floor_dir": os.path.join(args.inputs_root, "e0_floor"),
             "latent_sample": os.path.join(args.inputs_root, "precomputed", "zinc_latents_sample.npz"),
             "geometry_root": geometry_root,
             "lasso_cr_dir": args.lasso_cr_root or os.path.join(args.inputs_root, "lassobench_cr"),
             "lasso_cr_allow_unclean": [x for x in args.lasso_allow_unclean.split(",") if x],
             "lasso_pub_dir": args.lasso_pub_root or os.path.join(args.inputs_root, "lassobench_pub"),
             "verdicts_path": args.verdicts or os.path.join(args.inputs_root, "verification", "verdicts.json"),
             "p1_sel_dir": args.p1_sel_root or os.path.join(args.inputs_root, "p1_blocks_sel"),
             "floor_reps": args.floor_reps, "support_cache": args.support_cache,
             "repo_root": repo if repo and not repo.startswith("ERR") else os.path.dirname(os.path.dirname(os.path.dirname(HERE))),
             "_shared_cache": {}}       # Monte Carlo and geometry arrays: computed once, shared by every variant
    for col in (scope.GEOMETRY_COLUMNS if mode == "cr" else ("default", "strip", "guacamol_train")):
        if not os.path.exists(os.path.join(geometry_root, col, "geometry.json")):
            msg = f"WARNING: geometry input {col}/ missing under {geometry_root}"
            log("!" * 100 + "\n!!! " + msg + ("  -> tab_geometry prints n/a in its column" if col != "default"
                                                  else "  -> tab_geometry, fig_shells and the support Monte Carlo are skipped")
                + "\n" + "!" * 100)
            extra["_shared_cache"].setdefault("loud_warnings", []).append(msg)
    if mode == "cr":
        # review M9: every printed camera-ready GuacaMol arm should have all its cells (11 tasks x 40 seeds = 440 in
        # the final run), so that the matched sets of all tables coincide; loud, counts only
        short = {}
        for n_, a_ in sorted(ds.arms.items()):
            if a_.absent or a_.spec.namespace or a_.spec.kind != "cr":
                continue
            want = len(a_.spec.tasks) * len(a_.spec.seeds)
            got = sum(1 for c in a_.cells.values() if c.status_ok)
            if got < want or want != 440:
                short[n_] = f"{got} of {want}"
        n_cr = sum(1 for a_ in ds.arms.values() if not a_.absent and not a_.spec.namespace and a_.spec.kind == "cr")
        extra["_shared_cache"]["guacamol_cells_check"] = {"arms_checked": n_cr, "not_440_complete": short}
        if short:
            msg = (f"WARNING: {len(short)} of {n_cr} camera-ready GuacaMol arms do not have 440 completed cells "
                   "(the matched sets of the tables may differ): " + ", ".join(f"{k} {v}" for k, v in short.items()))
            log("!" * 100 + "\n!!! " + msg + "\n" + "!" * 100)
            extra["_shared_cache"].setdefault("loud_warnings", []).append(msg)
        else:
            log(f"GuacaMol cells: all {n_cr} camera-ready arms have 440 completed cells")
    registries: dict = {}
    ctxs = {}
    for name in names:
        var = VARIANTS[name]
        vdir = outdir if name == "default" else os.path.join(outdir, "variants", name)
        log(f"  variant {name}: {var.description}")
        try:
            ctxs[name] = run_variant(ds, var, vdir, mode, script_state, extra, ref, registries,
                                     do_figures=not args.no_figures, log=log)
        except INPUT_ERRORS as e:      # a broken stored input set or a guard: fatal like a loader problem
            msg = f"INPUT OR GUARD FAILURE in variant {name}: {type(e).__name__}: {e}"
            print(msg, file=sys.stderr)
            with open(os.path.join(outdir, "LOADER_FAILED.txt"), "w") as f:
                f.write(msg + "\n")
            return 2
    default = ctxs["default"]
    checks = None
    counts = None
    if mode == "legacy" and ref is not None:
        checks = reference.run_checks(default, ref, registries, ns="")
        counts = reference.write_legacy_report(checks, outdir, {"script_commit": script_state, "config": cfg.path})
        log(f"  legacy checks: {counts}")
    lc = latex_check(outdir, default.written, log) if args.latex_check else None
    fonts = pdffonts_check([p for c in ctxs.values() for p in c.written])
    if any(v.get("ok") is False for v in fonts.values()):
        log("  pdffonts: TYPE 3 FONT FOUND " + str([p for p, v in fonts.items() if v.get("ok") is False]))
    manifest = {
        "mode": mode, "argv": sys.argv, "started": dt.datetime.fromtimestamp(t_start).isoformat(timespec="seconds"),
        "elapsed_s": round(time.time() - t_start, 1), "host": socket.gethostname(), "python": platform.python_version(),
        "script_commit": script_commit, "script_dirty_paths": dirty.splitlines(), "config": cfg.path,
        "validation_run": bool(args.validation_run), "pinned_commits": PINNED_COMMITS,
        "config_sha256": cfg.sha256, "includes": cfg.includes, "expected_commit": expected,
        "allow_missing_arms": sorted(allow), "allow_missing_cells": extra_missing, "absent_arms": ds.absent, "cell_filter": cfg.cell_filter,
        "scope": ({"rule": scope.RULE, "arms_not_loaded": not_loaded,
                   "variants_not_run": {v: scope.NOT_RUN_VARIANTS[v] for v in scope.NOT_RUN_VARIANTS},
                   "includes_not_loaded": scope.SKIP_INCLUDES,
                   "not_built": {n: c.not_built for n, c in ctxs.items()}} if mode == "cr" else None),
        "reference": args.reference if ref is not None else None,
        "reference_sha256": sha256(args.reference) if ref is not None else None,
        "inputs_root": args.inputs_root, "variants": {n: VARIANTS[n].description for n in names},
        "skipped": {n: c.skipped for n, c in ctxs.items()},
        "outputs": {n: sorted(os.path.relpath(p, outdir) for p in c.written) for n, c in ctxs.items()},
        "output_sha256": {os.path.relpath(p, outdir): sha256(p) for c in ctxs.values() for p in c.written},
        "latex_check": lc, "pdffonts": {os.path.relpath(k, outdir): v for k, v in fonts.items()},
        "legacy_counts": counts,
        "numbers": {n: len(r.records) for n, r in registries.items()},
        "geometry_root": geometry_root, "support_cache": args.support_cache,
        "notes": extra["_shared_cache"].get("notes", []),
        "lassobench_rerun": ((extra["_shared_cache"].get("lasso_rerun") or {}).get("loaded") or {}).get("summary"),
        "lassobench_published": ((extra["_shared_cache"].get("lasso_pub") or {}).get("loaded") or {}).get("summary"),
        "lassobench_objective_check": (extra["_shared_cache"].get("lasso_pub") or {}).get("objective_check"),
        "guacamol_cells_check": extra["_shared_cache"].get("guacamol_cells_check"),
        "verdicts": ({"path": extra["verdicts_path"], "sha256": published.VERDICTS_SHA256,
                      "verified": {r: v["verified"] for r, v in extra["_shared_cache"]["verdicts"].items()}}
                     if extra["_shared_cache"].get("verdicts") else None),
        "lasso_pub_root": extra["lasso_pub_dir"],
        "lasso_require_finished": bool(args.lasso_require_finished),
        "september_e1_arm_check": extra["_shared_cache"].get("september_e1_arm_check"),
        "loud_warnings": extra["_shared_cache"].get("loud_warnings", []),
        "monte_carlo": {k: {kk: vv for kk, vv in v.items() if kk != "inputs"} if isinstance(v, dict) else v
                        for k, v in (extra["_shared_cache"].get("supports") or {}).items()},
    }
    with open(os.path.join(outdir, "run_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1, default=str)
    with open(os.path.join(outdir, "run.log"), "w") as f:
        f.write("\n".join(log_lines) + "\n")
    log(f"done in {time.time() - t_start:.0f}s: {len(default.written)} default outputs, "
        f"{sum(len(c.written) for c in ctxs.values())} in total")
    bad = counts and (counts.get("FAIL", 0) or counts.get("ERROR", 0))
    if lc and lc.get("ok") is False:
        return 3
    if any(v.get("ok") is False for v in fonts.values()):
        return 3
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
