"""Repository records of Appendix H: the baseline cards and the run status as Markdown, with a source manifest.

Since the amendment of 2026-10-24 to the run rules, Appendix H prints the overview (crlib/overview.py); the card
table (``tab_baselines``) and the run-status table (``tab_arm_status``) are written to ``records/`` of the output
folder, byte for byte as they were printed, and as Markdown for the repository (committed as ``docs/baselines/``):

  records/tab_baselines.tex, records/tab_arm_status.tex   the LaTeX tables (no longer in the paper)
  records/BASELINE_CARDS.md                               every card with its ten fields and its code name
  records/RUN_STATUS.md                                   result files, completed, non-completed and fallback cells
                                                          and commits of every loaded method, with its code name
  records/SOURCE.json                                     sha256 of the inputs (facts table, proofread text,
                                                          sources configuration, loaded result files) and of the two
                                                          Markdown files, and the generator commit

The Markdown is converted from the table rows by ``tex_to_md``, a closed whitelist: the text commands the two tables
use (``\\texttt``, ``\\textbf``, ``\\emph``, ``\\textit``, ``\\textup``, ``\\allowbreak``, escaped specials, ``\\'e``,
``\\geowalk``, ``\\geocma``, ``\\dots``, ``Table~\\ref`` of the tables named in ``REF_NAMES``), the mathematics of
``MATH_MD``, the thousands separator ``{,}`` (also escaped, ``\\{,\\}``) as a comma, and outside code ``---`` and
``--`` as an em and an en dash. Anything else raises ``RecordError``, so no LaTeX reaches the repository documents
unconverted. The Markdown carries no commit (it is reproduced byte for byte by a later run); SOURCE.json names the
generator commit.
"""
from __future__ import annotations

import hashlib
import json
import os
import re

from .context import Ctx

CARDS_MD, STATUS_MD, SOURCE_JSON = "BASELINE_CARDS.md", "RUN_STATUS.md", "SOURCE.json"
#: the paper's tables that the cards cite, named for a reader of the repository (the table numbers of the paper
#: are not stable across versions, the names are)
REF_NAMES = {"tab:main": "Table 1 of the paper",
             "tab:verification": "the verification table of the paper",
             "tab:lassobench": "the LassoBench table of the paper",
             "tab:budget": "the budget table of the paper",
             "tab:crosscodec_means": "the codec table of the paper"}
#: mathematics of the two tables, as Markdown text (closed list)
MATH_MD = {"\\times": "×", "^{\\ddagger}": "‡", "^{\\dagger}": "†", "^{*}": "\\*", "[p_{01},p_{99}]": "[p01, p99]",
           "\\bar r": "r̄", "\\ell_0=0.1\\sqrt{D}": "ℓ0 = 0.1√D", "\\ell_0=\\sqrt{D}": "ℓ0 = √D",
           "0.1\\sqrt{D}": "0.1√D", "\\sqrt{D}": "√D", "S^{D-1}": "S^(D−1)", "[-5,5]^D": "[−5, 5]^D",
           "[-3\\sigma,3\\sigma]": "[−3σ, 3σ]", "\\mathbb{R}^D": "ℝ^D", "\\sigma_0=0.1": "σ0 = 0.1",
           "\\sigma_0=0.5": "σ0 = 0.5", "N=1000": "N = 1000", "x_0=0": "x0 = 0", "[-8,8]^D": "[−8, 8]^D"}
_SIMPLE = {"\\{,\\}": ",", "{,}": ",", "\\allowbreak{}": "", "\\_": "_", "\\^{}": "^", "\\%": "%", "\\&": "&", "\\#": "#", "\\$": "$",
           "\\{": "{", "\\}": "}", "\\textgreater{}": ">", "\\textless{}": "<", "\\textbar{}": "|",
           "\\textasciitilde{}": "~", "\\textasciigrave{}": "`", "\\textbackslash{}": "\\", "\\dots{}": "…",
           "\\geowalk{}": "GeoWalk", "\\geocma{}": "GeoCMA", "\\'e": "é", "-{}-": "--", "\\hspace{0pt}": "",
           "\\raggedright": ""}
_WRAP = {"\\texttt": ("`", "`"), "\\textbf": ("**", "**"), "\\emph": ("*", "*"), "\\textit": ("*", "*"),
         "\\textup": ("", "")}


#: characters that Markdown would read as markup outside a code span (escaped with a backslash)
_MD_SPECIAL = "|*_<>$~[]"


class RecordError(RuntimeError):
    """A table cell holds LaTeX outside the whitelist of tex_to_md, or a record does not match its source."""


def _group(s: str, i: int) -> int:
    """Index after the brace group that opens at s[i] == '{'."""
    depth = 0
    for j in range(i, len(s)):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return j + 1
    raise RecordError(f"unbalanced braces in {s!r}")


def tex_to_md(s: str, code: bool = False, math: dict | None = None) -> str:
    """Markdown of one LaTeX cell of the two tables (closed whitelist; see the module docstring). ``math`` replaces
    the mathematics whitelist (``MATH_MD``) for another table (the full Table 1: ``TABLE1_MATH_MD``)."""
    math = MATH_MD if math is None else math
    out, i = [], 0
    while i < len(s):
        if s.startswith("Table~\\ref{", i):
            j = _group(s, i + len("Table~\\ref"))
            lab = s[i + len("Table~\\ref{"):j - 1]
            if lab not in REF_NAMES:
                raise RecordError(f"reference to {lab!r}, which REF_NAMES does not name")
            out.append(REF_NAMES[lab])
            i = j
            continue
        if s[i] == "$":
            j = s.find("$", i + 1)
            if j < 0:
                raise RecordError(f"unclosed math in {s!r}")
            frag = s[i + 1:j]
            if frag not in math:
                raise RecordError(f"mathematics {frag!r} is not in MATH_MD (or the table's mathematics whitelist)")
            out.append(math[frag])
            i = j + 1
            continue
        hit = next((k for k in _SIMPLE if s.startswith(k, i)), None)
        if hit is not None:
            val = _SIMPLE[hit]
            out.append("\\" + val if (not code and len(val) == 1 and val in _MD_SPECIAL) else val)
            i += len(hit)
            continue
        w = next((k for k in _WRAP if s.startswith(k + "{", i)), None)
        if w is not None:
            j = _group(s, i + len(w))
            inner = tex_to_md(s[i + len(w) + 1:j - 1], code=code or w == "\\texttt", math=math)
            a, b = _WRAP[w]
            out.append(a + inner + b)
            i = j
            continue
        if not code and s.startswith("---", i):
            out.append("—")
            i += 3
            continue
        if not code and s.startswith("--", i):
            out.append("–")
            i += 2
            continue
        if s.startswith("``", i) or s.startswith("''", i):
            out.append('"')
            i += 2
            continue
        ch = s[i]
        if ch == "\\":
            raise RecordError(f"LaTeX command {s[i:i + 20]!r} is not in the whitelist of tex_to_md")
        if ch in "{}":
            raise RecordError(f"bare brace in {s!r}")
        if ch == "~":
            out.append(" ")
        elif ch == "`":
            out.append("'")
        elif not code and ch in _MD_SPECIAL:
            out.append("\\" + ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out)


def _sha(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


HEAD = ("Generated by `scripts/cr/analysis/make_tables.py` from the result files of the run; do not edit by hand. "
        "`SOURCE.json` holds the sha256 of every input and of this file, and the generator commit.")


def cards_markdown(t) -> str:
    """BASELINE_CARDS.md from the rows of the card table (crlib/baselines.py CardTable)."""
    L = ["# Baseline cards", "", HEAD, "", tex_to_md(t.caption), ""]
    for r in t.rows:
        if r[0] == "group":
            L += [f"## {tex_to_md(r[1])}", ""]
        elif r[0] == "card":
            title = r[1]
            if title.startswith("\\textbf{") and title.endswith("}"):
                title = title[len("\\textbf{"):-1]
            if L[-1] != "":
                L.append("")
            L += [f"### {tex_to_md(title)}", ""]
        else:
            L.append(f"- **{tex_to_md(r[1])}:** {tex_to_md(r[2])}")
    return "\n".join(L).rstrip("\n") + "\n"


STATUS_HEADER = ["Method", "Code name", "Run", "Files", "Completed", "Not completed", "Fallback cells",
                 "Fallback steps", "Commit(s)"]


STATUS_CODE_NAMES = ("Their code names are in the Appendix H cards.",
                     "Their code names are in the second column and in BASELINE_CARDS.md.")


def arm_status_markdown(t) -> str:
    """RUN_STATUS.md from the rows of the run-status table (crlib/tables.py tab_arm_status) and its code names."""
    rows = [r for r in t.rows if isinstance(r, list)]
    if len(rows) != len(getattr(t, "row_arms", [])):
        raise RecordError(f"run-status table: {len(rows)} rows, {len(getattr(t, 'row_arms', []))} code names")
    cap = tex_to_md(t.caption)
    if STATUS_CODE_NAMES[0] in cap:              # the Markdown table carries the code names itself
        cap = cap.replace(STATUS_CODE_NAMES[0], STATUS_CODE_NAMES[1])
    L = ["# Run status of every loaded method", "", HEAD, "", cap, "",
         "| " + " | ".join(STATUS_HEADER) + " |", "|" + "-|" * len(STATUS_HEADER)]
    for r, arm in zip(rows, t.row_arms):
        cells = [tex_to_md(r[0]), f"`{arm}`"] + [tex_to_md(c) for c in r[1:]]
        L.append("| " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


#: the full Table 1 (RUN_RULES 2026-10-29 (1)): mathematics of its cells as Markdown text (closed list)
TABLE1_MATH_MD = dict(MATH_MD, **{"^\\dagger": "†", "[0,1]^D": "[0, 1]^D", "\\|z\\|\\approx 63": "‖z‖ ≈ 63",
                                  "n": "n", "\\Delta": "Δ", "d": "d",
                                  "[\\min,\\max]": "[min, max]"})  # legacy (submitted-version) row labels
_NUM_CELL = re.compile(r"^\$(\\mathbf\{)?([-+<]?\d[\d.]*(?:e[-+]\d+)?)(\})?\$$")
TABLE1_HEADER_MD = {"Method": "Method", "Support": "Support", "$n$": "n", "Mean": "Mean", "$\\Delta$": "Δ",
                    "W/L": "W/L", "$d$": "d"}


def _cell_md(s: str) -> str:
    """Markdown of a cell of the full Table 1: a number (bold kept) or text through tex_to_md."""
    mo = _NUM_CELL.match(s)
    if mo:
        return f"**{mo.group(2)}**" if mo.group(1) else mo.group(2)
    return tex_to_md(s, math=TABLE1_MATH_MD)


def _header_md(h: str) -> str:
    if h in TABLE1_HEADER_MD:
        return TABLE1_HEADER_MD[h]
    plain = re.sub(r"\\begin\{tabular\}\[b\]\{@\{\}c@\{\}\}|\\end\{tabular\}", "", h).replace("\\\\", " ")
    return {"$p_{\\mathrm{BH}}$ (cells)": "p_BH (cells)", "Tasks \\geowalk{} higher": "Tasks GeoWalk higher",
            "$p_{\\mathrm{task}}$ (uncorr.)": "p_task (uncorr.)"}[plain]


def table1_markdown(t, printed: list[bool]) -> str:
    """TABLE1_FULL.md: the full Table 1 (every comparator row and every column of the final run) as Markdown, with
    the column definitions and a column that says whether the paper prints the row (RUN_RULES 2026-10-29 (1))."""
    data = [r for r in t.rows if isinstance(r, list)]
    if len(data) != len(printed):
        raise RecordError(f"full Table 1: {len(data)} rows, {len(printed)} print flags")
    hdr = [_header_md(h) for h in t.header] + ["In the paper"]
    L = ["# Table 1 of the paper, full version", "", HEAD, "",
         "The paper prints a view of this table: the columns Method, Support, Mean, Δ, Tasks and p_task, and the rows "
         "marked \"yes\" in the last column. Every value below is computed on the one matched set of the final run, "
         "over all rows, so the printed values are the same as here.", "",
         "Columns: n = (task, seed) cells of the row inside the matched set; Mean = mean of per-task means of the best "
         "score after 30 evaluations (bold: the highest); Δ = GeoWalk minus the row on the same cells, as a mean of "
         "per-task means of the paired differences; W/L = cells won/lost by GeoWalk (a cell with |diff| ≤ 1e-6 is a "
         "tie, excluded from W/L and a zero difference in the cell-level test); d = mean/sd of the cell differences; "
         "p_BH (cells) = two-sided Wilcoxon signed-rank test over the cells, Benjamini–Hochberg corrected over the "
         f"{len(data) - 1} comparator rows; Tasks = tasks on which GeoWalk has the higher mean; p_task (uncorr.) = "
         "exact two-sided Wilcoxon test on the per-task mean differences, the primary test of the paper, without "
         "correction. † = run from the submitted version, not rerun. The notes on the rows and the abbreviations of "
         "the labels are those of the notes to Table 1 in the paper (Appendix H); tab_main_full_notes.tex holds the "
         "notes of this full version.", "",
         "| " + " | ".join(hdr) + " |", "|" + "-|" * len(hdr)]
    it = iter(printed)
    for r in t.rows:
        if isinstance(r, tuple):
            if r[0] == "group":
                L.append("| *" + tex_to_md(r[1], math=TABLE1_MATH_MD) + "* |" + " |" * (len(hdr) - 1))
            continue
        L.append("| " + " | ".join(_cell_md(c) for c in r) + f" | {'yes' if next(it) else 'no'} |")
    return "\n".join(L) + "\n"


def write_records(ctx: Ctx, rdir: str, built: dict, extra: dict | None = None) -> list[str]:
    """Write the Markdown records and SOURCE.json next to the record tables; returns the paths written. ``extra``
    ({file name: path} in ``rdir``) are further repository records of the run (RUN_RULES 2026-10-29: the full Table 1
    and the full LassoBench table), hashed into SOURCE.json."""
    from .baselines import FACTS_CSV, PRINTED_CSV
    from .overview import OVERVIEW_CSV
    paths = []
    cards, status = built.get("tab_baselines"), built.get("tab_arm_status")
    if cards is None and status is None:
        return paths
    os.makedirs(rdir, exist_ok=True)
    out = {name: _sha(p) for name, p in sorted((extra or {}).items())}
    for name, t, fn in ((CARDS_MD, cards, cards_markdown), (STATUS_MD, status, arm_status_markdown)):
        if t is None:
            continue
        p = os.path.join(rdir, name)
        with open(p, "w", encoding="utf-8") as f:
            f.write(fn(t))
        out[name] = _sha(p)
        paths.append(p)
    # the loaded result files of the cards, independent of where the folders are: sha256 of their sorted sha256s
    shas = sorted(c.sha256 for card in ctx.extra.get("baseline_cards", []) for a in card["present"]
                  for c in ctx.ds.arm(a).cells.values())
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    inputs = {os.path.join("scripts/cr/analysis", os.path.relpath(p, here)): _sha(p)
              for p in (FACTS_CSV, PRINTED_CSV, OVERVIEW_CSV) if os.path.exists(p)}
    src = {
        "generated_by": "scripts/cr/analysis/make_tables.py",
        "generator_commit": ctx.script_commit,
        "mode": ctx.mode, "variant": ctx.var.name,
        "sources_config": os.path.basename(ctx.ds.config.path), "sources_config_sha256": ctx.ds.config.sha256,
        "inputs_sha256": inputs,
        "result_files_of_the_cards": {"count": len(shas),
                                      "sha256_of_sorted_sha256s": hashlib.sha256("\n".join(shas).encode()).hexdigest()},
        "outputs_sha256": out,
        "tables": {n: f"records/tab_{t.name}.tex" for n, t in built.items() if t is not None},
    }
    p = os.path.join(rdir, SOURCE_JSON)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(src, f, indent=1, sort_keys=True)
        f.write("\n")
    paths.append(p)
    return paths
