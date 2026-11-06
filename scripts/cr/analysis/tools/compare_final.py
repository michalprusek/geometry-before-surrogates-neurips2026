"""Byte comparison of two generator output folders (RUN_RULES 2026-10-22, generator edits after the final run).

usage: compare_final.py <old output folder> <new output folder> [--report <file.md>]

The row of 2026-10-22 allows format, label and wording edits only, and requires that every numeric macro of the final
run stays byte-identical. This script checks that and lists every difference:

1. numbers.tex: every macro of the old folder must exist in the new one with a byte-identical body, except
   - the running-text macros whose wording the row changes: the published-recipe sentences and verdict texts
     (\\crPub*Sentence, \\crPub*Verdict, \\crLassoPub*Sentence, \\crLassoPub*Verdict) and the factorial readings
     (\\crFactorial*Reading, American spelling); and text macros that print a row label;
   - a value that rounded to a signed zero ("-0.00", "+0.00") and is now printed without the sign.
   Any other difference is a FAILURE. New macros must be count-only reprints (\\cr...TasksK, \\cr...TasksN) or the task
   totals (\\crNTasksGuacamol, \\crNTasksLasso).
2. Tables: per tab_*.tex, the multiset of numeric tokens in the table cells (every data row without its first cell,
   the row label) must be equal, except for the zero-sign change and the new task-level columns of Table 3.
3. Figures: listed as changed or unchanged (the figure code changed labels, legends and layout only).
4. Byte comparison (RUN_RULES 2026-10-24): every table, numbers.tex, every figure and every coverage report of the
   old folder (top level and variants/*) is compared byte for byte with the new one, after the commit stamp of the
   generator is replaced by a placeholder and the PDF dates are removed. Allowed differences, and only these:
   - the card table and the run-status table (MOVED_TO_RECORDS) are absent at the top level and present in records/
     with the same bytes, except the corrected licence of the Linear BO card (LICENCE_FIX);
   - tab_main_notes.tex differs by the one sentence that points to the overview and the cards (NOTES_SENTENCE);
   - numbers.tex differs by its macro count (line 2) and the added macros only;
   - the new outputs of the row (OVERVIEW_NEW) and records/ are new.
   Run-dependent files (manifest, log, loader report, dossier) are not compared; in the LassoBench coverage reports
   the folder the inputs were read from is masked.
Exit 0 if nothing outside the allowed changes differs, else 1.

usage: compare_final.py <old output folder> <new output folder> --rules 2026-10-29 [--report <file.md>]
       compare_final.py <old output folder> <new output folder> --rules numbers-identical [--report <file.md>]

RUN_RULES 2026-10-29 (printed tables cut to what the text needs, the Linear BO trajectory added, captions corrected):
every macro of the old numbers.tex is byte-identical and in the same place in the new one, and the only added macros
are the three levels of the Linear BO trajectory row (CUT_NEW_MACROS). Per output folder (top level and every variant):
the full Table 1, the full LassoBench table, the compute table and the per-task radius table are in records/ with the
old bytes (commit stamp normalized); the printed Table 1 is a view of the full table (every kept cell equal to the old
cell of the same column, only the five GP-recipe rows of CUT_MAIN_DROPPED left out); the printed LassoBench table has
the old blocks without "each method minus GeoCMA"; tab_radius has the rows of tab_radius, tab_estimators and the first
block of tab_shells, tab_convex the rows of tab_pca_directions as its last block; tab_published has the two old blocks
side by side; tab_verification loses only its frozen-noise rows; tab_hyper loses only its "Read from" column;
tab_trajectories gains one row whose values are the three new macros; every other table has the old data cells apart
from the wording edits of CUT_CELL_EDITS; the two record figures are in records/ unchanged apart from the PDF dates.
Captions may change and are listed. The second pass of 2026-10-29 (captions and labels) may reword table cells and row
labels: those tables must keep their numbers row by row, and every reworded cell is listed; in the overview only the
descriptive cells of OVERVIEW_NUMBER_EDIT_ROWS may change numbers; the stored BAxUS card (CARD_EDITS) is corrected in
the card records. --rules numbers-identical checks that numbers.tex of every folder is byte-identical to an earlier
run of the same rules (commit stamp masked).
"""
from __future__ import annotations

import collections
import hashlib
import os
import re
import sys

MACRO = re.compile(r"^\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}$")
WORDING = re.compile(r"^cr(Pub|LassoPub)[A-Za-z]*(Sentence|Verdict)$|^crFactorial[A-Za-z]*Reading$")
LABEL_TEXT = {"crLassoLeukemiaNotCompleted", "crPubHvarfnerChosen"}     # text macros that print row labels
NEW_OK = re.compile(r"^cr[A-Za-z]*Tasks[A-Za-z]*(K|N)$|^crNTasks(Guacamol|Lasso)$")
#: Appendix H proofreading (fork B, branch cr-facts-proof, merged at 17/n): the card counts change and the GaBO-type
#: card prints its fallback share (audit N4); every other number of tab_baselines is the cards' text
FORKB_CHANGED = {"crBaselinesProofCells", "crBaselinesMarkedCells", "crBaselinesComputedCells"}
FORKB_NEW = {"crBaselinesGaboFallbackSteps", "crBaselinesGaboFallbackCells"}
FORKB_REASON = "Appendix H proofreading (fork B)"
#: RUN_RULES 2026-10-24: Appendix H prints an overview; the cards and the run status become repository records
MOVED_TO_RECORDS = ("tab_baselines.tex", "tab_arm_status.tex")
OVERVIEW_NEW = ("tab_baselines_overview.tex", "tab_hyper.tex", "tab_compute.tex")
#: the fifth, crComputeStoredPrintedHours (the stored runs that printed tables read, root G and the stored folders
#: under the inputs root), was added on the authors' decision of 2026-10-24, after the run of 15:23
COMPUTE_NEW = {"crComputeGpuServerHours", "crComputeCpuServerHours", "crComputeVerifyHours", "crComputeStoredHours",
               "crComputeStoredPrintedHours"}
#: the corrected licence of the Linear BO card (the released file header and THIRD_PARTY_NOTICES.md name MIT)
LICENCE_FIX = (
    "upstream commit 6ee3bb6; the kernel is copied as src/\\allowbreak{}spherevo/\\allowbreak{}linear\\_\\allowbreak{}bo"
    "\\_\\allowbreak{}kernel.\\allowbreak{}py with cosmetic edits only; the copied file's header names CC-BY 4.0, while "
    "the upstream repository is MIT-licensed",
    "upstream commit 6ee3bb6, MIT license; the kernel is copied as src/\\allowbreak{}spherevo/\\allowbreak{}linear\\_"
    "\\allowbreak{}bo\\_\\allowbreak{}kernel.\\allowbreak{}py with cosmetic edits only")
NOTES_SENTENCE = ("Their deviations from the published methods are listed in the cards of Table~\\ref{tab:baselines}. ",
                  "Their deviations from the published methods are summarized in Table~\\ref{tab:baselines} and listed in "
                  "full in the method cards released with the code (\\texttt{docs/baselines/}). ")
STAMP = re.compile(r"(commit )[0-9a-f]{7,40}(\+dirty)?")
RUN_DEPENDENT = re.compile(r"^(run_manifest\.json|run\.log|loader_report\.(json|md)|numbers_dossier\.(json|md)|"
                           r"RUN_COMMAND\.txt|COMPARE_.*\.md|LOADER_FAILED\.txt)$")
SIGNED_ZERO = re.compile(r"^(\\ensuremath\{)?[-+](0\.0+)(\})?$")
NUM = re.compile(r"(?<![A-Za-z_{.\d])[-+]?\d+(?:\{,\}\d{3})*(?:\.\d+)?(?:\\times10\^\{-?\d+\})?(?:/\d+)?")


def macros(path: str) -> dict[str, str]:
    out = {}
    for line in open(path):
        mo = MACRO.match(line.rstrip("\n"))
        if mo:
            out[mo.group(1)] = mo.group(2)
    return out


def unsigned_zero(old: str, new: str) -> bool:
    mo = SIGNED_ZERO.match(old)
    return bool(mo) and new == old.replace("-" + mo.group(2), mo.group(2)).replace("+" + mo.group(2), mo.group(2))


def cell_tokens(path: str, text: str | None = None) -> collections.Counter:
    """Numeric tokens of the data cells of every tabular in a generated table file (row labels and headers excluded)."""
    c = collections.Counter()
    prev = ""
    for line in (text.splitlines(True) if text is not None else open(path)):
        line = line.rstrip("\n")
        header, prev = prev.strip() == "\\toprule", line
        end = next((e for e in ("\\\\", "\\tabularnewline", "\\tabularnewline*") if line.endswith(e)), None)
        if header or end is None or "&" not in line or line.lstrip().startswith(("%", "\\multicolumn")):
            continue
        cells = line[:-len(end)].split("&")[1:]
        for cell in cells:
            cell = re.sub(r"\\mathbf\{([^{}]*)\}", r"\1", cell)
            for tok in NUM.findall(cell):
                c[tok.lstrip("+-") if re.fullmatch(r"[-+]0\.0+", tok) else tok] += 1
    return c


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def sha_no_dates(path: str) -> str:
    """sha256 of a PDF without its creation and modification dates (matplotlib writes the time of the run)."""
    b = re.sub(rb"/(CreationDate|ModDate) \(D:[^)]*\)", b"", open(path, "rb").read())
    return hashlib.sha256(b).hexdigest()


def _norm(path: str) -> bytes:
    """Bytes of a generator output with the generator's commit stamp replaced (tables, numbers.tex, reports) or the
    PDF dates removed (figures)."""
    b = open(path, "rb").read()
    if path.endswith(".pdf"):
        return re.sub(rb"/(CreationDate|ModDate) \(D:[^)]*\)", b"", b)
    t = b.decode("utf-8")
    head = t.split("\n", 3)
    head[:3] = [STAMP.sub(r"\1<commit>", x) for x in head[:3]]
    t = "\n".join(head)
    if path.endswith("_coverage.md"):           # the folder the LassoBench inputs were read from
        t = re.sub(r"^Root `[^`]*`;", "Root `<inputs>`;", t, flags=re.M)
    return t.encode("utf-8")


def _numbers_equal_but_added(po: str, pn: str) -> tuple[bool, list[str]]:
    """numbers.tex: equal after the count of line 2 is masked and the added macro lines are removed."""
    old = _norm(po).decode().splitlines()
    new = _norm(pn).decode().splitlines()
    mask = re.compile(r"; \d+ numbers\.")
    old[1], new[1] = mask.sub("; <n> numbers.", old[1]), mask.sub("; <n> numbers.", new[1])
    oldm = {MACRO.match(x).group(1) for x in old if MACRO.match(x)}
    added = [x for x in new if MACRO.match(x) and MACRO.match(x).group(1) not in oldm]
    rest = [x for x in new if x not in added]
    # a table heading line ("% --- tab_compute") that only the added macros need is part of the addition
    heads = [x for x in rest if x.startswith("% --- ") and x not in old]
    rest = [x for x in rest if x not in heads]
    return rest == old, [MACRO.match(x).group(1) for x in added] + heads


def byte_compare(old: str, new: str) -> tuple[list[str], int]:
    lines = ["", "## Byte comparison (commit stamp and PDF dates normalized; RUN_RULES 2026-10-24)", "",
             "| folder | file | result |", "|---|---|---|"]
    fail = 0
    folders = [""] + sorted(os.path.join("variants", v) for v in (os.listdir(os.path.join(old, "variants"))
                                                                  if os.path.isdir(os.path.join(old, "variants")) else []))
    n_same = 0
    for sub in folders:
        o, n = os.path.join(old, sub), os.path.join(new, sub)
        for f in sorted(x for x in os.listdir(o) if os.path.isfile(os.path.join(o, x))):
            if RUN_DEPENDENT.match(f):
                continue
            po, pn = os.path.join(o, f), os.path.join(n, f)
            label = sub or "."
            if f in MOVED_TO_RECORDS:
                pr = os.path.join(n, "records", f)
                if os.path.exists(pn):
                    lines.append(f"| {label} | {f} | FAIL: still at the top level |")
                    fail += 1
                    continue
                if not os.path.exists(pr):
                    lines.append(f"| {label} | {f} | FAIL: not in records/ |")
                    fail += 1
                    continue
                a, b = _norm(po), _norm(pr)
                if a == b:
                    lines.append(f"| {label} | {f} | moved to records/, identical |")
                elif a.decode().replace(LICENCE_FIX[0], LICENCE_FIX[1]).encode() == b and LICENCE_FIX[0] in a.decode():
                    lines.append(f"| {label} | {f} | moved to records/, identical apart from the corrected Linear BO "
                                 "licence |")
                else:
                    lines.append(f"| {label} | {f} | FAIL: records/{f} differs beyond the allowed licence fix |")
                    fail += 1
                continue
            if not os.path.exists(pn):
                lines.append(f"| {label} | {f} | FAIL: absent |")
                fail += 1
                continue
            if f == "numbers.tex":
                ok, added = _numbers_equal_but_added(po, pn)
                lines.append(f"| {label} | {f} | " + ("identical apart from the count and the added lines "
                                                       f"{', '.join(added) or 'none'} |" if ok else
                                                       "FAIL: differs beyond added macros |"))
                fail += not ok
                continue
            a, b = _norm(po), _norm(pn)
            if a == b:
                n_same += 1
                continue
            if f == "tab_main_notes.tex" and a.decode().replace(NOTES_SENTENCE[0], NOTES_SENTENCE[1]).encode() == b \
                    and NOTES_SENTENCE[0] in a.decode():
                lines.append(f"| {label} | {f} | identical apart from the sentence that points to the overview and "
                             "the cards |")
                continue
            lines.append(f"| {label} | {f} | FAIL: differs |")
            fail += 1
        for f in sorted(x for x in os.listdir(n) if os.path.isfile(os.path.join(n, x))):
            if not os.path.exists(os.path.join(o, f)) and not RUN_DEPENDENT.match(f):
                ok = f in OVERVIEW_NEW
                lines.append(f"| {sub or '.'} | {f} | {'new (allowed)' if ok else 'FAIL: new file'} |")
                fail += not ok
        rn = os.path.join(n, "records")
        if os.path.isdir(rn):
            lines.append(f"| {sub or '.'} | records/ | {', '.join(sorted(os.listdir(rn)))} |")
    lines.append(f"\n{n_same} files byte-identical after normalization.")
    return lines, fail


# =====================================================================================================================
# RUN_RULES 2026-10-29: the printed tables are cut to what the text needs (--rules 2026-10-29)
# =====================================================================================================================
#: the only macros the row may add: the three levels of the Linear BO trajectory row (promise C-32)
CUT_NEW_MACROS = {"crTrajPubLinearNOneZero", "crTrajPubLinearNTwoZero", "crTrajPubLinearNThreeZero"}
#: printed files that became repository records with the same bytes (new name in records/)
CUT_MOVED = {"tab_compute.tex": "tab_compute.tex", "tab_radius_pertask.tex": "tab_radius_pertask.tex",
             "tab_main.tex": "tab_main_full.tex", "tab_main_notes.tex": "tab_main_full_notes.tex",
             "tab_lassobench.tex": "tab_lassobench_full.tex"}
CUT_MOVED_FIGURES = ("fig_trajectories.pdf", "fig_lassobench.pdf")
#: the full version of a printed table, written to records/ with the bytes of the old printed file (second pass)
CUT_FULL_RECORDS = {"tab_convex.tex": "tab_convex_full.tex"}
#: tables whose row labels may change (one name per row, 2026-10-29 second pass); numbers must stay row by row
RELABELED = ("tab_budget.tex", "tab_geometry.tex")
#: printed tables merged into a host table (their file disappears; the host has their rows)
CUT_MERGED = {"tab_estimators.tex": "tab_radius.tex", "tab_shells.tex": "tab_radius.tex",
              "tab_pca_directions.tex": "tab_convex.tex"}
#: Table 1: the five GP-recipe rows that only the full table has (labels without marks)
CUT_MAIN_DROPPED = {
    "Mat\\'ern-5/2 GP ($\\ell_0=0.1\\sqrt{D}$), LogEI, L-BFGS-B with sample-around-best",
    "Mat\\'ern-5/2 GP ($\\ell_0=0.1\\sqrt{D}$), qLogEI, L-BFGS-B",
    "Mat\\'ern-5/2 GP ($\\ell_0=\\sqrt{D}$), qLogEI, L-BFGS-B",
    "Mat\\'ern-5/2 GP ($\\ell_0=0.1\\sqrt{D}$), qLogEI, candidate-set argmax",
    "Mat\\'ern-5/2 GP ($\\ell_0=\\sqrt{D}$), qLogEI, candidate-set argmax"}
#: names of the columns of the full Table 1, in order (report only)
MAIN_COLUMN_NAMES = ["Method", "Support", "n", "Mean", "Delta", "W/L", "d", "p_BH (cells)", "Tasks", "p_task"]
_MARKS = re.compile(r"\$\^(\\dagger|\{\\dagger\}|\{\\dagger\\ddagger\}|\{\\ddagger\}|\\ddagger)\$")


def _strip_bold(cell: str) -> str:
    return re.sub(r"\\mathbf\{([^{}]*)\}", r"\1", cell.strip())


def _label(cell: str) -> str:
    return _MARKS.sub("", cell.strip())


def tabulars(text: str) -> list[dict]:
    """Every tabular of a generated table file: {'header': [cells], 'rows': [[cells]], 'groups': [text]}. The header is
    the last '&' line before the first \\midrule (a grouped header line above it is skipped); a data row is an '&' line
    after it that ends a row and does not start with \\multicolumn (group lines)."""
    out, cur, state, longt = [], None, None, False
    for line in text.splitlines():
        s = line.strip()
        if (s.startswith("\\begin{tabular") and not s.startswith("\\begin{tabular}[b]")) or \
                s.startswith("\\begin{longtable}"):
            cur, state, longt = {"header": None, "rows": [], "groups": []}, "head", s.startswith("\\begin{longtable}")
            continue
        if cur is None:
            continue
        if (s.startswith("\\end{tabular") and not s.startswith("\\end{tabular}\\\\")) or s.startswith("\\end{longtable}"):
            out.append(cur)
            cur = None
            continue
        if state == "head" and ((s == "\\midrule" and not longt) or (s == "\\endfoot" and longt)):
            state = "body"            # a longtable repeats its header; its body starts after \endfoot
            continue
        end = next((e for e in ("\\\\", "\\tabularnewline") if s.endswith(e)), None)
        if "&" not in s or end is None:
            continue
        cells = [c.strip() for c in _split_cells(s[:-len(end)])]
        if state == "head":
            if not s.startswith("\\multicolumn"):
                cur["header"] = cells
        elif s.startswith("\\multicolumn"):
            cur["groups"].append(s)
        else:
            cur["rows"].append(cells)
    return out


def _split_cells(s: str) -> list[str]:
    """Split a row at the '&' that are not inside braces (a stacked header or cell holds a nested tabular)."""
    cells, depth, cur = [], 0, []
    for ch in s:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "&" and depth == 0:
            cells.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    cells.append("".join(cur))
    return cells


def _caption(text: str) -> str:
    """The caption(s) of a table file (brace-matched)."""
    out, i = [], text.find("\\caption{")
    while i >= 0:
        j, depth = i + len("\\caption"), 0
        for k in range(j, len(text)):
            depth += {"{": 1, "}": -1}.get(text[k], 0)
            if depth == 0:
                out.append(text[j + 1:k])
                break
        i = text.find("\\caption{", i + 1)
    return "\n".join(out)


#: wording edits of 2026-10-29 inside table cells (old text, new text): G5 of the authors' list
CUT_CELL_EDITS = (("box directions rescaled to", "cube directions rescaled to"), ("box directions at", "cube directions at"),
                  ("convex hulls", "convex combinations of latents"), ("x0 = 0", "$x_0=0$"),
                  ("decoding and scoring of a proposal capped at 30 s",                 # G9: read from the code
                   "decoding of a proposal capped at 30 s (scoring is not time-limited)"))


def _edited(rows: list[list[str]]) -> list[list[str]]:
    out = []
    for r in rows:
        rr = []
        for c in r:
            for a, b in CUT_CELL_EDITS:
                c = c.replace(a, b)
            rr.append(c)
        out.append(rr)
    return out


def _num_tokens(rows: list[list[str]]) -> collections.Counter:
    c = collections.Counter()
    for r in rows:
        for cell in r[1:]:
            for tok in NUM.findall(_strip_bold(cell)):
                c[tok] += 1
    return c


def _same_numbers(rows_a: list, rows_b: list) -> bool:
    """Same number of data rows and, row by row, the same multiset of numeric tokens (row label excluded)."""
    return len(rows_a) == len(rows_b) and all(_num_tokens([x]) == _num_tokens([y]) for x, y in zip(rows_a, rows_b))


def _reworded(rows_a: list, rows_b: list) -> list[str]:
    """The cells whose text changed (second pass of 2026-10-29: wording only), as 'row: old -> new'."""
    out = []
    for x, y in zip(rows_a, rows_b):
        for i, (c, d) in enumerate(zip(x, y)):
            if _strip_bold(c) != _strip_bold(d):
                out.append(f"{_label(x[0])[:40]} [col {i}]: {c[:60]!r} -> {d[:60]!r}")
    return out


#: rows of the overview whose descriptive cells may change their numbers (the stored BAxUS row: its split schedule,
#: corrected from the code and the recorded trust-region lengths on 2026-10-29)
OVERVIEW_NUMBER_EDIT_ROWS = ("nested random subspaces",)
#: card lines that the second pass of 2026-10-29 corrects (the stored BAxUS card, C4): an old line must hold one of
#: the first phrases and the new line one of the second
CARD_EDITS = (("most likely never grows", "failure tolerance from a split budget"),
              ("splits in every run", "the failure tolerance is 1"))


def _card_lines_ok(old: str, new: str) -> tuple[bool, int]:
    import difflib
    a, b = old.splitlines(), new.splitlines()
    bad, n = False, 0
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        n += 1
        if tag != "replace" or not all(any(k in x for k in CARD_EDITS[0]) for x in a[i1:i2]) \
                or not all(any(k in x for k in CARD_EDITS[1]) for x in b[j1:j2]):
            bad = True
    return not bad, n


def cut_table_checks(o: str, n: str, mn: dict) -> tuple[list[str], int]:
    """The per-table relations of the row of 2026-10-29 for one output folder (top level or a variant)."""
    L, fail = [], 0

    def rd(*p):
        q = os.path.join(*p)
        return open(q).read() if os.path.exists(q) else None

    def ok(name, cond, what_ok, what_fail):
        nonlocal fail
        L.append(f"| {name} | {what_ok if cond else 'FAIL: ' + what_fail} |")
        fail += not cond

    old_tabs = sorted(x for x in os.listdir(o) if x.startswith("tab_") and x.endswith(".tex"))
    for f in old_tabs:
        ot, nt = rd(o, f), rd(n, f)
        cap = "" if nt is None or _caption(ot) == _caption(nt) else "; caption changed"
        if f in ("tab_main.tex", "tab_lassobench.tex", "tab_main_notes.tex", "tab_compute.tex",
                 "tab_radius_pertask.tex"):
            continue                               # checked below (moved, and printed as a view)
        if f in CUT_MERGED:
            ok(f, nt is None, f"merged into {CUT_MERGED[f]} (no file of its own)", "still printed on its own")
            continue
        if nt is None:
            ok(f, False, "", "absent")
            continue
        A, B = tabulars(ot), tabulars(nt)
        if f == "tab_radius.tex":
            sh = tabulars(rd(o, "tab_shells.tex"))
            exp = A[0]["rows"] + tabulars(rd(o, "tab_estimators.tex"))[0]["rows"] + sh[0]["rows"]
            same_hdr = len(B) == 1 and B[0]["header"] == A[0]["header"]
            ok(f, same_hdr and B[0]["rows"] == exp,
               f"the rows of tab_radius, tab_estimators and tab_shells, cell for cell, in one tabular "
               f"({len(exp)} rows); the block 'against r = 8' of tab_shells ({len(sh[1]['rows']) if len(sh) > 1 else 0} "
               f"rows) is not printed{cap}", "rows differ from the three former tables")
            continue
        if f == "tab_convex.tex":
            P = tabulars(rd(o, "tab_pca_directions.tex"))
            ok(f, len(B) == 2 and B[0] == A[0] and B[1]["rows"] == _edited(P[0]["rows"]) and B[1]["header"] == P[0]["header"],
               f"tab_convex unchanged and tab_pca_directions as its last block, cell for cell{cap}",
               "rows differ from tab_convex + tab_pca_directions")
            continue
        if f == "tab_published.tex":
            exp = []
            for r0 in A[0]["rows"]:
                r1 = next((r for r in A[1]["rows"] if r[0] == r0[0]), None) if len(A) > 1 else None
                exp.append(r0 + (r1[1:] if r1 else []))
            rw = _reworded(exp, B[0]["rows"]) if len(B) == 1 else []
            ok(f, len(B) == 1 and _same_numbers(exp, B[0]["rows"]) and [_label(r[0]) for r in exp] ==
               [_label(r[0]) for r in B[0]["rows"]],
               f"the two blocks side by side, numbers identical row by row ({len(exp)} rows)"
               + (f"; reworded cells: {'; '.join(rw)}" if rw else "") + cap, "rows differ from the two blocks")
            continue
        if f == "tab_verification.tex":
            exp = [r for r in A[0]["rows"] if "frozen noise" not in r[0]]
            lost = [r[0] for r in A[0]["rows"] if "frozen noise" in r[0]]
            hd = [f"{x!r} -> {y!r}" for x, y in zip(A[0]["header"], B[0]["header"]) if x != y] if len(B) == 1 else []
            ok(f, len(B) == 1 and B[0]["rows"] == exp,
               f"identical apart from the {len(lost)} frozen-noise rows ({'; '.join(lost)})"
               + (f"; header {'; '.join(hd)}" if hd else "") + cap, "rows differ beyond the frozen-noise rows")
            continue
        if f == "tab_trajectories.tex":
            vals = sorted(mn.get(k, "?").replace("\\ensuremath{", "").rstrip("}") for k in CUT_NEW_MACROS)
            new_rows = [r for r in B[0]["rows"] if r[0] == "Linear BO (Doumont et al.)"]
            kept = [r for r in B[0]["rows"] if r not in new_rows]
            added_ok = (len(new_rows) == 1 and sorted(_strip_bold(c).strip("$") for c in new_rows[0][2:]) == vals)
            same = _same_numbers(A[0]["rows"], kept) and [_label(r[0]) for r in kept] == [_label(r[0]) for r in A[0]["rows"]]
            bold = [r[0] for r, q in zip(kept, A[0]["rows"]) if ["mathbf" in c for c in r] != ["mathbf" in c for c in q]]
            rw = _reworded(A[0]["rows"], kept)
            ok(f, added_ok and same and B[1:] == A[1:],
               f"identical numbers apart from the added row {new_rows[0][0] if new_rows else ''!r} with the values of "
               f"{', '.join(sorted(CUT_NEW_MACROS))} ({', '.join(vals)})"
               + (f"; bold moved in the rows {bold}" if bold else "; no bold moved")
               + (f"; reworded cells: {'; '.join(rw)}" if rw else "") + cap,
               f"added rows {new_rows}, other rows with the same numbers: {same}")
            continue
        if f == "tab_hyper.tex":
            old2 = _edited([r[:2] for r in A[0]["rows"]])
            exact = len(B) == 1 and B[0]["rows"] == old2
            split = [r[0] for r in B[0]["rows"] if r not in old2] if len(B) == 1 else []
            ok(f, len(B) == 1 and B[0]["header"] == A[0]["header"][:2] and (exact or (
                _num_tokens(B[0]["rows"]) == _num_tokens(old2) and len(B[0]["rows"]) == len(old2) + 1
                and all(r in B[0]["rows"] for r in old2 if r[0] != "Decoding"))),
               f"identical apart from the dropped column {A[0]['header'][2:]}"
               + ("" if exact else f"; the decoding row split into the rows {split}, same numbers") + cap,
               "rows differ beyond the dropped column and the split decoding row")
            continue
        if f == "tab_baselines_overview.tex":
            ea, eb = _edited(A[0]["rows"]), B[0]["rows"] if len(B) == 1 else []
            labels_ok = len(ea) == len(eb) and all(_label(x[0]) == _label(y[0]) for x, y in zip(ea, eb))
            numdiff = [(_label(x[0]), dict(_num_tokens([x]) - _num_tokens([y])), dict(_num_tokens([y]) - _num_tokens([x])))
                       for x, y in zip(ea, eb) if _num_tokens([x]) != _num_tokens([y])] if labels_ok else []
            allowed = all(any(k in lab for k in OVERVIEW_NUMBER_EDIT_ROWS) for lab, _, _ in numdiff)
            rw = _reworded(A[0]["rows"], eb) if labels_ok else []
            ok(f, len(B) == 1 and labels_ok and allowed,
               f"same rows; {len(rw)} cells reworded"
               + (f"; numbers changed only in the descriptive cells of {'; '.join(f'{l} (removed {r}, added {a_})' for l, r, a_ in numdiff)}"
                  if numdiff else "; numeric tokens identical") + cap,
               f"rows differ: labels the same {labels_ok}, number changes {numdiff}")
            continue
        if B == A:
            ok(f, True, "data cells identical" + cap, "")
            continue
        same = len(A) == len(B) and all(_same_numbers(x["rows"], y["rows"]) and [_label(r[0]) for r in x["rows"]] ==
                                         [_label(r[0]) for r in y["rows"]] for x, y in zip(A, B))
        relabel = len(A) == len(B) and all(_same_numbers(x["rows"], y["rows"]) for x, y in zip(A, B)) and f in RELABELED
        rw = [w for x, y in zip(A, B) for w in _reworded(x["rows"], y["rows"])] if len(A) == len(B) else []
        hd = [f"{u!r} -> {v!r}" for x, y in zip(A, B) for u, v in zip(x["header"] or [], y["header"] or []) if u != v]
        ok(f, same or relabel, "numbers identical row by row"
           + (f"; reworded cells: {'; '.join(rw)}" if rw else "") + (f"; header {'; '.join(hd)}" if hd else "") + cap,
           "data cells differ")
    # ---- Table 1: the full table is a record with the old bytes; the printed table is a view of it
    for f in ("tab_main.tex", "tab_main_notes.tex", "tab_lassobench.tex", "tab_compute.tex", "tab_radius_pertask.tex"):
        if not os.path.exists(os.path.join(o, f)):
            continue
        rec = os.path.join(n, "records", CUT_MOVED[f])
        ok(f"records/{CUT_MOVED[f]}", os.path.exists(rec) and _norm(os.path.join(o, f)) == _norm(rec),
           f"the old {f}, byte for byte (commit stamp normalized)", f"differs from the old {f} or is absent")
    for f, rec in CUT_FULL_RECORDS.items():
        if os.path.exists(os.path.join(o, f)) and os.path.exists(os.path.join(n, "records", rec)):
            ok(f"records/{rec}", _norm(os.path.join(o, f)) == _norm(os.path.join(n, "records", rec)),
               f"the old {f}, byte for byte (commit stamp normalized)", f"differs from the old {f}")
    ot, nt = rd(o, "tab_main.tex"), rd(n, "tab_main.tex")
    if ot and nt:
        A, B = tabulars(ot)[0], tabulars(nt)[0]
        cols = [A["header"].index(h) for h in B["header"]] if all(h in A["header"] for h in B["header"]) else None
        olds = {(_label(r[0]), r[1]): r for r in A["rows"]}        # label and support (two rows share a label)
        bad, kept = [], []
        for r in B["rows"]:
            src = olds.get((_label(r[0]), r[1]))
            if cols is None or src is None or [_strip_bold(r[i]) for i in range(1, len(r))] != \
                    [_strip_bold(src[j]) for j in cols[1:]]:
                bad.append(_label(r[0]))
            kept.append((_label(r[0]), r[1]))
        dropped = sorted(lab for lab, _ in set(olds) - set(kept))
        ok("tab_main.tex", cols is not None and not bad and len(dropped) == len(CUT_MAIN_DROPPED)
           and set(dropped) == CUT_MAIN_DROPPED,
           f"a view of the full table: columns {', '.join(MAIN_COLUMN_NAMES[j] for j in cols) if cols else ''} (old "
           f"columns {cols}), every kept row cell for cell ({len(B['rows'])} rows); the {len(dropped)} rows "
           "left out are the five GP-recipe rows" + ("; caption changed" if _caption(ot) != _caption(nt) else ""),
           f"rows differing {bad}, rows left out {dropped}")
    ot, nt = rd(o, "tab_lassobench.tex"), rd(n, "tab_lassobench.tex")
    if ot and nt:
        A, B = tabulars(ot), tabulars(nt)
        strip_s = lambda rows: [_label(r[0]).replace("$^{\\S}$", "") for r in rows]       # noqa: E731
        ok("tab_lassobench.tex", len(A) == 3 and len(B) == 2 and B[0]["rows"] == A[0]["rows"]
           and _same_numbers(A[2]["rows"], B[1]["rows"]) and strip_s(A[2]["rows"]) == strip_s(B[1]["rows"]),
           "the main block cell for cell and the block of the published methods with the same numbers (LassoBench-only "
           "rows marked); the block 'each method minus GeoCMA' is only in the record; captions shortened",
           "blocks differ")
    ot, nt = rd(o, "tab_main_notes.tex"), rd(n, "tab_main_notes.tex")
    if ot and nt:
        ok("tab_main_notes.tex", sum(cell_tokens("", nt).values()) == 0, "text changed (column definitions of the "
           "printed columns, the pointer to the full table, the box sentence of the authors)", "has table cells")
    # ---- figures
    for f in sorted(x for x in os.listdir(o) if x.startswith("fig_") and x.endswith(".pdf")):
        po = os.path.join(o, f)
        if f in CUT_MOVED_FIGURES:
            pr = os.path.join(n, "records", f)
            ok(f"records/{f}", os.path.exists(pr) and not os.path.exists(os.path.join(n, f))
               and sha_no_dates(po) == sha_no_dates(pr), "moved to records/, identical apart from the PDF dates",
               "not moved or changed")
        elif f == "fig_convergence_grid.pdf":
            pn = os.path.join(n, f)
            ok(f, os.path.exists(pn), "redrawn with the added Linear BO curve (crFigConvergenceN = "
               f"{mn.get('crFigConvergenceN', '?')})", "absent")
        else:
            pn = os.path.join(n, f)
            ok(f, os.path.exists(pn) and sha_no_dates(po) == sha_no_dates(pn), "unchanged apart from the PDF dates",
               "changed or absent")
    # ---- coverage reports and other files
    for f in sorted(x for x in os.listdir(o) if x.endswith("_coverage.md")):
        ok(f, os.path.exists(os.path.join(n, f)) and _norm(os.path.join(o, f)) == _norm(os.path.join(n, f)),
           "identical (input folder masked)", "differs")
    new_top = sorted(x for x in os.listdir(n) if os.path.isfile(os.path.join(n, x)) and not RUN_DEPENDENT.match(x)
                     and not os.path.exists(os.path.join(o, x)))
    ok("new files at the top level", not new_top, "none", ", ".join(new_top))
    for f in sorted(os.listdir(os.path.join(o, "records"))) if os.path.isdir(os.path.join(o, "records")) else []:
        po, pn = os.path.join(o, "records", f), os.path.join(n, "records", f)
        if f == "SOURCE.json":
            continue                    # names the generator commit and the outputs of the run
        if os.path.exists(pn) and _norm(po) != _norm(pn) and f in ("tab_baselines.tex", "BASELINE_CARDS.md"):
            good, k = _card_lines_ok(_norm(po).decode(), _norm(pn).decode())
            ok(f"records/{f}", good, f"identical apart from {k} corrected passage(s) of the stored BAxUS card",
               "differs beyond the corrected BAxUS card")
            continue
        ok(f"records/{f}", os.path.exists(pn) and _norm(po) == _norm(pn), "identical (commit stamp normalized)",
           "differs or is absent")
    if os.path.isdir(os.path.join(n, "records")):
        L.append(f"| records/ (new) | {', '.join(sorted(os.listdir(os.path.join(n, 'records'))))} |")
    return L, fail


def main_cut(old: str, new: str, report: str | None) -> int:
    """RUN_RULES 2026-10-29: every old macro byte-identical and in the same place, new macros only CUT_NEW_MACROS; the
    tables, figures and records in the relations of cut_table_checks, in the top folder and in every variant."""
    lines, fail = [f"# Comparison {os.path.basename(old.rstrip('/'))} -> {os.path.basename(new.rstrip('/'))} "
                   "(RUN_RULES 2026-10-29: printed tables cut, Linear BO trajectory added, captions corrected)", ""], 0
    folders = [""] + sorted(os.path.join("variants", v) for v in (os.listdir(os.path.join(old, "variants"))
                                                                   if os.path.isdir(os.path.join(old, "variants")) else []))
    for sub in folders:
        o, n = os.path.join(old, sub), os.path.join(new, sub)
        label = sub or "."
        if not os.path.isdir(n):
            lines.append(f"## {label}: FAIL, absent")
            fail += 1
            continue
        mo, mn = macros(os.path.join(o, "numbers.tex")), macros(os.path.join(n, "numbers.tex"))
        changed = sorted(k for k in set(mo) & set(mn) if mo[k] != mn[k])
        missing = sorted(set(mo) - set(mn))
        added = sorted(set(mn) - set(mo))
        bad_new = [k for k in added if k not in CUT_NEW_MACROS]
        same, extra = _numbers_equal_but_added(os.path.join(o, "numbers.tex"), os.path.join(n, "numbers.tex"))
        f_n = len(changed) + len(missing) + len(bad_new) + (not same)
        fail += f_n
        lines += [f"## {label}", "",
                  f"numbers.tex: {len(mo)} old macros, {len(mn)} new; {len(set(mo) & set(mn)) - len(changed)} "
                  f"byte-identical; {len(changed)} changed; {len(missing)} missing; {len(added)} added "
                  f"({len(bad_new)} not allowed); file identical apart from the count and the added lines: "
                  f"{'yes' if same else 'NO'}." + (" **FAIL**" if f_n else ""), ""]
        for title, ks in (("CHANGED (FAIL)", changed), ("MISSING (FAIL)", missing), ("ADDED, NOT ALLOWED (FAIL)", bad_new)):
            if ks:
                lines += [f"{title}: " + ", ".join(f"`\\{k}`" for k in ks), ""]
        if added:
            lines += ["Added macros: " + ", ".join(f"`\\{k}` = `{mn[k]}`" for k in added), ""]
        tl, tf = cut_table_checks(o, n, mn)
        lines += ["| file | result |", "|---|---|"] + tl + [""]
        fail += tf
    lines += [f"**Result: {'PASS' if not fail else f'FAIL ({fail})'}**"]
    txt = "\n".join(lines) + "\n"
    if report:
        open(report, "w").write(txt)
    print(txt)
    return 0 if not fail else 1


def main_numbers(old: str, new: str, report: str | None) -> int:
    """numbers.tex of every output folder (top level and variants) byte-identical after the generator's commit stamp
    is masked: a later wording pass must not change one number (second pass of 2026-10-29)."""
    lines, fail = [f"# numbers.tex {os.path.basename(old.rstrip('/'))} -> {os.path.basename(new.rstrip('/'))}", ""], 0
    folders = [""] + sorted(os.path.join("variants", v) for v in (os.listdir(os.path.join(old, "variants"))
                                                                   if os.path.isdir(os.path.join(old, "variants")) else []))
    for sub in folders:
        po, pn = os.path.join(old, sub, "numbers.tex"), os.path.join(new, sub, "numbers.tex")
        same = os.path.exists(pn) and _norm(po) == _norm(pn)
        fail += not same
        lines.append(f"- {sub or '.'}: {len(macros(po))} macros; "
                     + ("byte-identical (commit stamp masked)" if same else "FAIL: differs or is absent"))
    lines += ["", f"**Result: {'PASS' if not fail else f'FAIL ({fail})'}**"]
    txt = "\n".join(lines) + "\n"
    if report:
        open(report, "w").write(txt)
    print(txt)
    return 0 if not fail else 1


def main(argv: list[str]) -> int:
    old, new = argv[0], argv[1]
    report = argv[argv.index("--report") + 1] if "--report" in argv else None
    if "--rules" in argv:
        rules = argv[argv.index("--rules") + 1]
        if rules == "numbers-identical":
            return main_numbers(old, new, report)
        if rules != "2026-10-29":
            print(f"--rules: unknown {rules!r}", file=sys.stderr)
            return 2
        return main_cut(old, new, report)
    lines, fail = [], 0
    mo, mn = macros(os.path.join(old, "numbers.tex")), macros(os.path.join(new, "numbers.tex"))
    missing = sorted(set(mo) - set(mn))
    wording, zeros, other, forkb = [], [], [], []
    for k in sorted(set(mo) & set(mn)):
        if mo[k] == mn[k]:
            continue
        if k in FORKB_CHANGED:
            forkb.append(k)
        elif WORDING.match(k) or k in LABEL_TEXT:
            wording.append(k)
        elif unsigned_zero(mo[k], mn[k]):
            zeros.append(k)
        else:
            other.append(k)
    added = sorted(set(mn) - set(mo))
    forkb += [k for k in added if k in FORKB_NEW]
    compute_new = [k for k in added if k in COMPUTE_NEW]
    bad_new = [k for k in added if not NEW_OK.match(k) and k not in FORKB_NEW and k not in COMPUTE_NEW]
    same = sum(1 for k in set(mo) & set(mn) if mo[k] == mn[k])
    lines += [f"# Byte comparison {os.path.basename(old.rstrip('/'))} -> {os.path.basename(new.rstrip('/'))}", "",
              f"numbers.tex: {len(mo)} old macros, {len(mn)} new; {same} byte-identical; "
              f"{len(wording)} wording macros changed (allowed); {len(zeros)} signed zeros now unsigned (allowed); "
              f"{len(forkb)} changed or added by the {FORKB_REASON} (allowed); "
              f"{len(other)} other changes; {len(missing)} missing; {len(added)} added ({len(bad_new)} not count-only "
              "or proofreading)."]
    for title, ks, show in (("Wording macros changed (allowed)", wording, True), ("Signed zero -> unsigned (allowed)", zeros, True),
                            (f"{FORKB_REASON} (allowed)", forkb, True),
                            ("OTHER CHANGES (FAIL)", other, True), ("MISSING (FAIL)", missing, False),
                            ("Added, not count-only (FAIL)", bad_new, False)):
        if ks:
            lines += ["", f"## {title}", ""]
            for k in ks:
                lines.append(f"- `\\{k}`: `{mo.get(k, '')}` -> `{mn.get(k, '')}`" if show else f"- `\\{k}`")
    lines += ["", "## Added count-only macros", "", ", ".join(f"`\\{k}`" for k in added if NEW_OK.match(k)) or "none"]
    lines += ["", "## Added compute macros (RUN_RULES 2026-10-24)", "",
              ", ".join(f"`\\{k}` = `{mn[k]}`" for k in compute_new) or "none"]
    fail += len(other) + len(missing) + len(bad_new)
    # ---- tables
    lines += ["", "## Tables (numeric tokens of the data cells)", "", "| table | old tokens | new tokens | result |",
              "|---|---|---|---|"]
    t3_new = collections.Counter()
    for f in sorted(x for x in os.listdir(old) if x.startswith("tab_") and x.endswith(".tex")):
        po, pn = os.path.join(old, f), os.path.join(new, f)
        if f in MOVED_TO_RECORDS and not os.path.exists(pn):
            pn = os.path.join(new, "records", f)          # RUN_RULES 2026-10-24: a repository record
        if not os.path.exists(pn):
            lines.append(f"| {f} | | absent | FAIL |")
            fail += 1
            continue
        co, cn = cell_tokens(po), cell_tokens(pn)
        if f == "tab_baselines.tex" and LICENCE_FIX[0] in open(po).read():
            # the corrected Linear BO licence (its old text held the number of "CC-BY 4.0")
            co = cell_tokens(po, open(po).read().replace(LICENCE_FIX[0], LICENCE_FIX[1]))
        if co == cn:
            lines.append(f"| {f} | {sum(co.values())} | {sum(cn.values())} | identical |")
            continue
        extra, lost = cn - co, co - cn
        if f == "tab_crossdomain.tex" and not lost:
            t3_new = extra
            lines.append(f"| {f} | {sum(co.values())} | {sum(cn.values())} | allowed: new task-level columns "
                         f"{dict(extra)} |")
            continue
        lines.append(f"| {f} | {sum(co.values())} | {sum(cn.values())} | FAIL: added {dict(extra)}, lost {dict(lost)} |")
        fail += 1
    if t3_new:
        exp = {mo.get("crCrossGuacamolTasks", "").replace("\\ensuremath{", "").rstrip("}"),
               mo.get("crCrossGuacamolPTask", "").replace("\\ensuremath{", "").rstrip("}"),
               mo.get("crLassoPTask", "").replace("\\ensuremath{", "").rstrip("}"),
               mo.get("crLassoTasksGeocma", "").replace("\\ensuremath{", "").rstrip("}")}
        lines.append(f"\nTable 3 new tokens {sorted(t3_new)}; values of the existing macros crCrossGuacamolTasks, "
                     f"crCrossGuacamolPTask, crLassoTasksGeocma, crLassoPTask: {sorted(exp)}"
                     + (" (the same)." if set(t3_new) == exp else " (NOT the same: FAIL)."))
        fail += set(t3_new) != exp
    # ---- figures
    lines += ["", "## Figures", ""]
    for f in sorted(x for x in os.listdir(old) if x.startswith("fig_") and x.endswith(".pdf")):
        pn = os.path.join(new, f)
        po = os.path.join(old, f)
        state = "absent (FAIL)" if not os.path.exists(pn) else (
            "unchanged" if sha(po) == sha(pn) else "unchanged apart from the PDF dates" if sha_no_dates(po) == sha_no_dates(pn)
            else "figure changed: labels only (legend, title, layout)")
        lines.append(f"- {f}: {state}")
        fail += not os.path.exists(pn)
    # ---- byte comparison (RUN_RULES 2026-10-24)
    blines, bfail = byte_compare(old, new)
    lines += blines
    fail += bfail
    lines += ["", f"**Result: {'PASS' if not fail else f'FAIL ({fail})'}**"]
    txt = "\n".join(lines) + "\n"
    if report:
        open(report, "w").write(txt)
    print(txt)
    return 0 if not fail else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
