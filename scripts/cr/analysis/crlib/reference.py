"""Earlier (submitted / posted / audited) numbers: legacy checks and the correspondence tables.

reference_values.json holds one entry per earlier number:
  id, group ('submitted' | 'posted' | 'audit'), source, quantity, printed, value, cmp, tol,
  query (how the earlier number is computed from stored files, see queries.py),
  cr_number (the camera-ready macro that replaces it), reason, known_diff (+ note).
"""
from __future__ import annotations

import json
import math
import os

from . import numbers as N
from .context import Ctx
from .latex import Table
from .queries import QueryError, compare, evaluate, fmt_value


def load_reference(path: str) -> dict:
    with open(path) as f:
        ref = json.load(f)
    ids = [e["id"] for e in ref["entries"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate reference ids")
    return ref


def run_checks(ctx: Ctx, ref: dict, registries: dict, ns: str = "") -> list[dict]:
    out = []
    for e in ref["entries"]:
        q = e.get("query")
        if q is None:
            continue
        try:
            val, n = evaluate(ctx, q, ns, registries)
            status, detail = compare(e, val)
        except (QueryError, KeyError, ValueError) as ex:
            val, n, status, detail = None, None, "ERROR", f"{type(ex).__name__}: {ex}"
        out.append({"id": e["id"], "group": e["group"], "quantity": e["quantity"], "source": e["source"],
                    "printed": e.get("printed"), "expected": e.get("value"), "computed": val, "n": n,
                    "status": status, "detail": detail, "note": e.get("note", ""), "query": q})
    return out


def write_legacy_report(results: list[dict], outdir: str, header: dict) -> dict:
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    with open(os.path.join(outdir, "legacy_check.json"), "w") as f:
        json.dump({"header": header, "counts": counts, "checks": results}, f, indent=1,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    lines = ["# Legacy validation: the generator against independently verified earlier numbers", "",
             f"Mode legacy (sources_legacy.json), script commit `{header.get('script_commit')}`. "
             f"Counts: {counts}.", "",
             "PASS = reproduced within half a unit of the last printed digit (or the stated tolerance); "
             "KNOWN-DIFF = an earlier number that earlier audits already found NOT reproducible from any file state "
             "(listed for completeness, not a generator failure); FAIL = generator disagrees.", "",
             "| id | group | quantity | printed | computed | n | status | source / note |",
             "|---|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['id']} | {r['group']} | {r['quantity']} | {r['printed']} | {fmt_value(r['computed'])} | "
                     f"{'' if r['n'] is None else r['n']} | {r['status']} | {r['source']}"
                     f"{(' -- ' + r['note']) if r['note'] else ''} |")
    with open(os.path.join(outdir, "legacy_check.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return counts


def _cr_value(reg, name):
    if not name:
        return None, "--"
    r = reg.records.get(name)
    if r is None:
        return None, "(not printed)"
    return r["value"], r["text"]


def _tex(s: str) -> str:
    return (str(s).replace("^", "").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")
            .replace("<", "$<$").replace(">", "$>$"))


def _stack(*lines: str) -> str:
    return "\\begin{tabular}[b]{@{}c@{}}" + "\\\\".join(lines) + "\\end{tabular}"


def _exact_wl(reg, name: str) -> str:
    """The camera-ready value under the OLD tie rule (exact float differences): W/L counts and, since GEN v3,
    cell-level Wilcoxon p values (macros <name>Exact; BH-corrected p where the family registers one)."""
    if name and name.endswith(("WL", "P", "PBH")) and (name + "Exact") in reg.records:
        t = reg.records[name + "Exact"]["text"]
        return t if name.endswith("WL") or t in ("--", "") else f"${t}$"
    return ""


def tab_posted_vs_printed(ctx: Ctx, ref: dict, registries: dict) -> Table | None:
    table = "tab_posted_vs_printed"
    reg = registries["default"]
    ns = "posted" if ctx.mode == "cr" else ""
    hdr = ["Posted number", "Post", "Posted", "Recomputed", _stack("Printed", "(CR)"), _stack("CR, old", "tie rule"),
           "Why it differs"]
    t = Table("posted_vs_printed", "tab:posted_vs_printed", "", "p{2.5cm}p{1.2cm}rrrrp{2.9cm}", hdr, size="\\scriptsize",
              long=True, tabcolsep="4pt")
    n = 0
    for e in ref["entries"]:
        if e["group"] != "posted":
            continue
        rec, st = "--", ""
        if e.get("query"):
            try:
                v, _ = evaluate(ctx, e["query"], ns, registries)
                status, _ = compare(e, v)
                rec = fmt_value(v) + ("" if status == "PASS" else ("$^*$" if status == "KNOWN-DIFF" else "$^!$"))
                st = status
            except (QueryError, KeyError, ValueError):
                rec = "n/a"
        cv, ct = _cr_value(reg, e.get("cr_number"))
        nm = e.get("cr_number") or ""
        if nm:
            reg.attach_old(nm, {"kind": "posted", "value": e.get("printed"), "where": e["source"], "id": e["id"]})
        t.add([_tex(e["quantity"]), _tex(e["source"].split(",")[0]), _tex(e.get("printed", "")), _tex(rec),
               ct if ct.startswith("$") or ct in ("--",) else f"${ct}$" if ct and ct[0] in "+-0123456789<" else _tex(ct),
               _exact_wl(reg, nm), _tex(e.get("reason", ""))])
        ctx.extra.setdefault("posted_rows", []).append({"id": e["id"], "posted": e.get("printed"), "recomputed": rec,
                                                        "status": st, "cr_number": nm, "cr_text": ct})
        n += 1
    if not n:
        return None
    t.caption = ("\\textbf{Numbers posted in the rebuttal versus the numbers this version prints.} Posted = value in "
                 "the OpenReview thread; Recomputed = the same quantity recomputed by this generator from the stored "
                 "posted-generation files under the posted convention (cell set and averaging as posted; $^*$ = a "
                 "posted value that no file state reproduces, $^!$ = mismatch); Printed = the camera-ready value "
                 "(rule R-3 of the run rules released with the code: camera-ready runs, mean of per-task means, one matched set per table); CR, old "
                 "tie rule = for W/L counts and cell-level Wilcoxon $p$ values, the camera-ready value under the rule "
                 "of the posted numbers (exact float differences) instead of the printed rule (a cell with "
                 "$|\\mathrm{diff}|\\le10^{-6}$ is a tie: excluded from W/L, a zero difference in the test). Rows whose Post column "
                 "names a September rerun (E-1, p1 blocks), the posted-generation files or a work log were not posted; "
                 "they are listed because the rebuttal-period text rested on them (-- in Recomputed: the stored files "
                 "of that row are not read here; see Tables~\\ref{tab:e1_extra_tasks} and \\ref{tab:selection_rule}).")
    return t


def tab_changes_vs_submitted(ctx: Ctx, ref: dict, registries: dict) -> Table | None:
    reg = registries["default"]
    hdr = ["Submitted number", "Where", "Submitted", _stack("Camera-", "ready"), _stack("CR, old", "tie rule"), "Reason"]
    t = Table("changes_vs_submitted", "tab:changes_vs_submitted", "", "p{2.6cm}lrrrp{3.6cm}", hdr, size="\\scriptsize",
              long=True, tabcolsep="4pt")
    changed = unchanged = 0
    used = []
    for e in ref["entries"]:
        if e["group"] != "submitted":
            continue
        cv, ct = _cr_value(reg, e.get("cr_number"))
        nm = e.get("cr_number") or ""
        if nm:
            reg.attach_old(nm, {"kind": "submitted", "value": e.get("printed"), "where": e["source"], "id": e["id"]})
            if nm in reg.filesets:
                used.append(nm)
        same = False
        if isinstance(cv, (int, float)) and not (isinstance(cv, float) and math.isnan(cv)) and e.get("value") is not None:
            dec = len(str(e["printed"]).split(".")[1]) if "." in str(e["printed"]) else 0
            same = round(abs(float(cv)) if e.get("cmp") == "absval" else float(cv), dec) == round(float(e["value"]), dec)
        elif isinstance(cv, str) and cv == e.get("printed"):
            same = True
        elif e.get("value") is None and ct.replace("$", "") == str(e.get("printed", "")).replace("$", ""):
            same = True
        if same:
            unchanged += 1
            continue
        changed += 1
        t.add([_tex(e["quantity"]), _tex(e["source"]), _tex(e.get("printed", "")),
               (f"${ct}$" if ct and ct[0] in "+-0123456789<" else _tex(ct)), _exact_wl(reg, nm),
               _tex(e.get("reason", ""))])
    if used:
        fs = ctx.fs_union(*[reg.filesets[n] for n in sorted(set(used))])
        ctx.num("crChangesChanged", changed, str(changed), kind="count", table="tab_changes_vs_submitted", fileset=fs,
                rule="submitted numbers whose camera-ready value differs at the printed precision (file set: union of "
                     "the mapped camera-ready numbers' sets)")
        ctx.num("crChangesUnchanged", unchanged, str(unchanged), kind="count", table="tab_changes_vs_submitted",
                fileset=fs, rule="submitted numbers unchanged at the printed precision")
    t.caption = ("\\textbf{Changes relative to the submitted version.} Every submitted number whose camera-ready value "
                 f"differs at the printed precision ({changed} of {changed + unchanged} mapped numbers; the other "
                 f"{unchanged} are unchanged). Camera-ready values come from this generator (rules R-3 and R-6 of the run rules released with the code). "
                 "CR, old tie rule: for W/L counts and cell-level Wilcoxon $p$ values, the camera-ready value under the "
                 "submitted rule (exact float differences); the printed values treat $|\\mathrm{diff}|\\le10^{-6}$ as "
                 "a tie (excluded from W/L, a zero difference in the test).")
    return t if changed else None
