"""Appendix H: the baseline table, rendered from the facts table ``appendixH_facts.csv``.

``appendixH_facts.csv`` (next to make_tables.py) is the audited Appendix-H facts table
(B_appendixH_facts.csv of 2026-10-11, 54 rows x 20 columns, copied cell by cell) extended by
eight rows for the arms the audit did not cover (published MSR, fixed-box TuRBO-1, the
control-block arms C1 / C2a / C2b, the uniform-box control, the success-rule GeoWalk and
the GP radius arms; every file:line in them refers to tag ``cr-controls-2026-10-12``) and
by seven appended columns (run arms, lengthscale initialisation, lengthscale prior,
camera-ready overrides, row origin, needs proofreading, table group). The audited block is
protected by a sha256 (``AUDITED_BLOCK_SHA256``): loading fails if an audited cell changed.

``tab_baselines`` prints one card per facts row whose arm is in the run, with the ten
fields promised for the appendix:

  field                       source
  --------------------------  -----------------------------------------------------------
  Software version            camera-ready arms: provenance of the loaded result files
                              (python, CUDA, package versions, host; computed);
                              stored arms: column 'device and software environment', followed
                              by what the loaded files themselves record'
  Kernel                      column 'kernel, priors, constraints, mean, fitting'
  Acquisition                 columns 'acquisition function and best_f' + 'acquisition maximiser'
  Lengthscale initialisation  appended column (derived from the kernel cell or the code)
  Lengthscale prior           appended column (derived)
  Candidate support           column 'candidate support / box definition and source'
  Trust region / RAASP        column 'trust region / RAASP'
  Seed count                  computed from the loaded files (seeds, tasks, cells, completed);
                              stored arms also print column 'result files, seeds, status'
  Repository                  column 'original repository'
  Deviations                  column 'deviations from the published method'

Cells are the facts table's text, LaTeX-escaped and otherwise unchanged. Two markers:

  dagger         the text needs author proofreading: every cell of a row written for GEN v3,
                 every derived cell (lengthscale initialisation / prior), every camera-ready
                 override, every cell whose text says it is unverified, not recorded, not
                 checked, assumed or derived, and the software / seed cells of the LassoBench
                 rows (they describe the stored runs; LassoBench is rerun in full)
  double dagger  computed by this generator from the loaded result files (no text to
                 proofread; the numbers are in the dossier). A cell that holds both computed
                 content and facts-table text gets the dagger if that text needs proofreading.

An arm that reads SUBSTITUTE files (a config ``display`` override, no ``cr_arm``, and files that
belong to the solver of another facts row, e.g. the legacy stand-in of the published MSR arm) is
not described by its own row and is left out of the card, with a skip record.

The facts table describes the code audited at ``abbd61e`` and the STORED files. For an arm
loaded as a camera-ready run (``spec.kind == 'cr'``) the software version and the seed count
therefore come from the loaded files, and a 'camera-ready overrides' entry replaces a cell
whose audited text is known to describe the stored files only.

Proofread text (2026-10-22, RUN_RULES row of that date: card wording and facts may change, no value
may): ``appendixH_printed.csv`` (next to the facts table) holds, for every printed field of every
card, the text checked against the code and the runs the card describes, in long format::

  row      facts-table arm of the card
  field    'title' (other-benchmark cards only; raw LaTeX) or a field key of FIELDS
  verdict  confirmed | corrected | rewritten | unknown | omitted
  text     the printed text (plain text, escaped here); for 'software' and 'seeds' of a card with
           loaded files it replaces only the facts-table part, the computed part stays
  source   where the text was checked (file and function at a tag, result-file fields, ...)

A field with a proofread entry is printed from it and carries no dagger; the facts table is kept
unchanged as the audit record (its sha256 is still checked). 'unknown' text states what is not known
and why; 'omitted' (software and seeds only) drops the facts-table part. The loader rejects text
with internal paths, codes, review marks, file line numbers or British spellings (``PRINTED_FORBIDDEN``).
A field text may contain the tokens ``{fallback_steps}``, ``{fallback_cells}`` and ``{cells}``: the
uniform-fallback steps, the cells with at least one, and the loaded cells of the card's arms, counted
by the loader (the counts of the arm-status table).
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from collections import Counter

from . import numbers as N
from .context import Ctx
from .published import other_row_label, printed_label
from .latex import Table, upright_names
from .numbers import camel

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACTS_CSV = os.path.join(HERE, "appendixH_facts.csv")
PRINTED_CSV = os.path.join(HERE, "appendixH_printed.csv")

AUDITED_COLUMNS = [
    "arm", "paper role / label", "implementing code (abbd61e unless stated)",
    "kernel, priors, constraints, mean, fitting", "input and outcome transforms",
    "acquisition function and best_f", "acquisition maximiser",
    "candidate support / box definition and source", "trust region / RAASP", "cold start",
    "budget accounting", "fallback on GP failure", "device and software environment",
    "result files, seeds, status", "versions recorded in provenance", "git_commit / git_dirty groups",
    "original repository", "deviations from the published method",
    "rerun at abbd61e reproduces existing files? (engineering item 2)", "evidence",
]
APPENDED_COLUMNS = ["run arms", "lengthscale initialisation", "lengthscale prior", "camera-ready overrides",
                    "row origin", "needs proofreading", "table group"]
N_AUDITED_ROWS = 54
#: sha256 of the audited block (header + 54 rows of the 20 audited columns; cells joined by 0x1f, rows by
#: 0x1e), computed from B_appendixH_facts.csv (file sha256 d9220956...99fc) when the CSV was built.
AUDITED_BLOCK_SHA256 = "2519bfbb63f00ca054a5b827f82d131b058696341138a3ca5458f703b983a266"
ORIGIN_AUDIT = "B audit"
ORIGIN_NEW = "GEN v3"

K_SOFT, K_KERNEL, K_ACQ, K_MAX = (AUDITED_COLUMNS[12], AUDITED_COLUMNS[3], AUDITED_COLUMNS[5], AUDITED_COLUMNS[6])
K_SUPPORT, K_TR, K_FILES, K_REPO, K_DEV = (AUDITED_COLUMNS[7], AUDITED_COLUMNS[8], AUDITED_COLUMNS[13],
                                           AUDITED_COLUMNS[16], AUDITED_COLUMNS[17])
K_INIT, K_PRIOR = "lengthscale initialisation", "lengthscale prior"

#: (field key, printed name, facts columns the field is taken from). The field key is also the key of a
#: 'camera-ready overrides' entry.
FIELDS = [
    ("software", "Software version", [K_SOFT]),
    ("kernel", "Kernel", [K_KERNEL]),
    ("acquisition", "Acquisition", [K_ACQ, K_MAX]),
    ("lengthscale initialisation", "Lengthscale initialization", [K_INIT]),
    ("lengthscale prior", "Lengthscale prior", [K_PRIOR]),
    ("support", "Candidate support", [K_SUPPORT]),
    ("trust region", "Trust region / RAASP", [K_TR]),
    ("seeds", "Seed count", [K_FILES]),
    ("repository", "Repository", [K_REPO]),
    ("deviations", "Deviations", [K_DEV]),
]
OVERRIDE_KEYS = {k for k, _, _ in FIELDS}
GROUP_ORDER = [
    "This paper", "GeoWalk variants", "Vanilla BO GP: support x selection cells and controls",
    "Surrogate-free support controls", "Papenmeier et al. recipes", "Xu et al. Standard GP",
    "Linear BO (Doumont et al.)", "Published recipes (released-code settings)", "Other baselines, in-repo harness",
    "Stored baselines (not rerun)", "LassoBench published methods (released-code settings)",
    "LassoBench (audited description of the stored runs)", "NAS-Bench-101 (stored runs)",
]
#: The printed group headings (the 'table group' column keeps the internal key). A heading names a published
#: method only where every card of the group carries a verified recipe's name or the group says so.
GROUP_HEADINGS = {           # raw LaTeX
    "This paper": "Methods of this paper",
    "GeoWalk variants": "Variants of \\geowalk{} and \\geocma{}",
    "Vanilla BO GP: support x selection cells and controls":
        "Mat\\'ern-5/2 GP with the dimension-scaled lengthscale prior: support, maximizer and controls",
    "Surrogate-free support controls": "Surrogate-free sampling of other supports",
    "Papenmeier et al. recipes": "Mat\\'ern-5/2 GP with initial lengthscale $0.1\\sqrt{D}$",
    "Xu et al. Standard GP": "Mat\\'ern-5/2 GP with initial lengthscale $\\sqrt{D}$",
    "Linear BO (Doumont et al.)": "GP with the spherical linear kernel",
    "Published recipes (released-code settings)":
        "Published recipes with the settings of the released code (authors' names only where verified)",
    "Other baselines, in-repo harness": "Other baselines implemented in this repository",
    "Stored baselines (not rerun)": "Runs from the submitted version (not rerun)",
    "LassoBench published methods (released-code settings)":
        "LassoBench: published methods with the settings of the released code (authors' names only where "
        "verified)",
    "LassoBench (audited description of the stored runs)": "LassoBench: methods rerun for this version",
    "NAS-Bench-101 (stored runs)": "NAS-Bench-101: runs from the submitted version (not rerun)",
}
#: Rows of other benchmarks (no GuacaMol arm): printed as a second block from the facts text alone.
OTHER_BENCHMARKS = True
#: Runner of the other-benchmark rows (the code-name field points into it).
OTHER_RUNNERS = {"lassobench": "scripts/run_lassobench.py", "lassobench_pub": "scripts/run_lassobench.py",
                 "nasbench": "scripts/run_nasbench.py"}

#: GEN v5 backlog item 1: the printed table names machines by their hardware, not by host name (the host names stay
#: in the dossier and in the provenance of every result file).
HOSTS = {"panda": "server A (2$\\times$ Intel Xeon Gold 6542Y, 2$\\times$ NVIDIA L40S)",
         # review M11: the rules state only kajman's CPU model, and every kajman run is CPU only
         # 2026-10-22: no 'no GPU used' here: true of the camera-ready CPU arms only, not of the stored runs that
         # decoded on a GPU; each card says whether its runs used one
         "kajman": "server B (2$\\times$ Intel Xeon Gold 6542Y)",
         "tulen": "server C (2$\\times$ Intel Xeon Gold 5315Y, NVIDIA A100 80GB and RTX A5000)"}
_HOST_RX = re.compile(r"\b(panda|kajman|tulen)(?:\.utia\.cas\.cz)?\b")


def neutral_hosts(latex: str) -> str:
    """Replace host names in a printed (already escaped) cell by the hardware description."""
    def rep(mo):
        return HOSTS[mo.group(1)]
    # host names may be escaped with discretionary breaks (\allowbreak{}) after their dots
    latex = re.sub(r"\b(panda|kajman|tulen)\.(?:\\allowbreak\{\})?utia\.(?:\\allowbreak\{\})?cas\.(?:\\allowbreak\{\})?cz",
                   lambda mo: HOSTS[mo.group(1)], latex)
    return _HOST_RX.sub(rep, latex)


def reader_facing(latex: str) -> str:
    """GEN v5 backlog item 3: the internal file name of the rules never reaches the page."""
    return latex.replace("RUN\\_RULES", "the run rules")


DAGGER = "$^{\\dagger}$"
DDAGGER = "$^{\\ddagger}$"
#: A cell whose text contains one of these needs proofreading whatever its origin.
UNCERTAIN_RX = re.compile(r"unverified|not verified|unknown|not recorded|not checked|not audited|not measured|"
                          r"not attributed|not restated|assum|derived|most likely|see implementation", re.I)
PKG_ORDER = ["torch", "botorch", "gpytorch", "linear_operator", "numpy", "scipy", "cma", "selfies", "rdkit",
             "guacamol"]
MAX_ENVS_LISTED = 4      # environments spelled out in a software cell (stored arms can have many states)


# =========================================================================================
# facts table
# =========================================================================================
def audited_block_sha256(header: list[str], rows: list[list[str]]) -> str:
    return hashlib.sha256("\x1e".join("\x1f".join(r) for r in [header] + rows).encode("utf-8")).hexdigest()


def load_facts(path: str = FACTS_CSV) -> list[dict]:
    """The rows of the facts table as dicts (+ parsed '_run', '_overrides', '_proof', '_new').
    Raises RuntimeError if the layout is wrong or an audited cell differs from the audit."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, body = rows[0], rows[1:]
    if header[:len(AUDITED_COLUMNS)] != AUDITED_COLUMNS:
        raise RuntimeError(f"{path}: the first {len(AUDITED_COLUMNS)} columns are not the audited columns")
    if header[len(AUDITED_COLUMNS):] != APPENDED_COLUMNS:
        raise RuntimeError(f"{path}: appended columns {header[len(AUDITED_COLUMNS):]} != {APPENDED_COLUMNS}")
    if any(len(r) != len(header) for r in body):
        raise RuntimeError(f"{path}: ragged row")
    k = len(AUDITED_COLUMNS)
    got = audited_block_sha256(header[:k], [r[:k] for r in body[:N_AUDITED_ROWS]])
    if got != AUDITED_BLOCK_SHA256:
        raise RuntimeError(f"{path}: the audited block (first {N_AUDITED_ROWS} rows x {k} columns) has sha256 {got}, "
                           f"expected {AUDITED_BLOCK_SHA256}: an audited cell was changed")
    out = []
    for i, r in enumerate(body):
        d = dict(zip(header, r))
        d["_new"] = not d["row origin"].startswith(ORIGIN_AUDIT)
        if (i < N_AUDITED_ROWS) == d["_new"]:
            raise RuntimeError(f"{path}: row {d['arm']!r}: 'row origin' contradicts its position")
        if d["_new"] and d["needs proofreading"] != "all":
            raise RuntimeError(f"{path}: new row {d['arm']!r} must have needs proofreading = all")
        d["_run"] = [a for a in d["run arms"].split(";") if a]
        d["_overrides"] = json.loads(d["camera-ready overrides"]) if d["camera-ready overrides"] else {}
        bad = set(d["_overrides"]) - OVERRIDE_KEYS
        if bad:
            raise RuntimeError(f"{path}: row {d['arm']!r}: unknown override fields {sorted(bad)}")
        d["_proof"] = set(d["needs proofreading"].split(";")) if d["needs proofreading"] else set()
        if d["table group"] not in GROUP_ORDER:
            raise RuntimeError(f"{path}: row {d['arm']!r}: unknown table group {d['table group']!r}")
        out.append(d)
    arms = [a for d in out for a in d["_run"]]
    if len(arms) != len(set(arms)):
        dup = sorted(a for a, c in Counter(arms).items() if c > 1)
        raise RuntimeError(f"{path}: run arms listed in two rows: {dup}")
    return out


PRINTED_COLUMNS = ["row", "field", "verdict", "text", "source"]
PRINTED_VERDICTS = ("confirmed", "corrected", "rewritten", "unknown", "omitted")
PRINTED_FIELDS = {"title"} | OVERRIDE_KEYS
#: Text a reader of the paper must never see in a card: machine paths, internal file and review codes,
#: proofreading marks, line numbers of the audited code (the released snapshot has other line numbers),
#: British spellings (the paper is in American English).
PRINTED_FORBIDDEN = re.compile(
    r"/data2?/|/home/|/scratch/|\.venv|UNVERIFIED|NOT recorded|Stanton|learning_rte|\bD-[A-Z]?\d|H2_successor|"
    r"\bE-m\d|\bF_msr|\breport 0\d|\(0\d [A-Z]\.\d|NLP repo|uv\.lock|\.md\b|\b[CO]\d+-[CO]?\d\b|\bchoice [CO]\d|"
    r"\(C\d\)|\(O\d\)|\bR2-\d|\.py:\d|\bline \d|\blines \d|initialis|maximis|minimis|standardis|normalis|"
    r"optimis|parametris|behaviour|centre|favour|\bmodell")


def load_printed(path: str = PRINTED_CSV, rows: list[dict] | None = None) -> dict:
    """{(row arm, field): {'verdict', 'text', 'source'}} of the proofread text; RuntimeError on any
    layout problem, unknown row or field, duplicate, empty text or forbidden text."""
    with open(path, newline="", encoding="utf-8") as f:
        data = list(csv.reader(f))
    if not data or data[0] != PRINTED_COLUMNS:
        raise RuntimeError(f"{path}: header must be {PRINTED_COLUMNS}")
    known = {r["arm"] for r in rows} if rows is not None else None
    out: dict = {}
    for i, r in enumerate(data[1:], start=2):
        if len(r) != len(PRINTED_COLUMNS):
            raise RuntimeError(f"{path}:{i}: {len(r)} cells")
        row, field, verdict, text, source = r
        if known is not None and row not in known:
            raise RuntimeError(f"{path}:{i}: unknown facts row {row!r}")
        if field not in PRINTED_FIELDS:
            raise RuntimeError(f"{path}:{i}: unknown field {field!r}")
        if verdict not in PRINTED_VERDICTS:
            raise RuntimeError(f"{path}:{i}: unknown verdict {verdict!r}")
        if verdict == "omitted":
            if field not in ("software", "seeds") or text:
                raise RuntimeError(f"{path}:{i}: 'omitted' is for an empty software or seeds text only")
        elif not text.strip():
            raise RuntimeError(f"{path}:{i}: empty text")
        if not source.strip():
            raise RuntimeError(f"{path}:{i}: no source")
        if field != "title" and PRINTED_FORBIDDEN.search(text):
            raise RuntimeError(f"{path}:{i}: forbidden text {PRINTED_FORBIDDEN.search(text).group(0)!r} in "
                               f"{row}/{field}")
        if (row, field) in out:
            raise RuntimeError(f"{path}:{i}: duplicate entry for {row}/{field}")
        out[(row, field)] = {"verdict": verdict, "text": text, "source": source}
    return out


# =========================================================================================
# LaTeX
# =========================================================================================
_ESC = {"\\": "\\textbackslash{}", "{": "\\{", "}": "\\}", "$": "\\$", "&": "\\&", "%": "\\%", "#": "\\#",
        "_": "\\_", "^": "\\^{}", "~": "\\textasciitilde{}", "<": "\\textless{}", ">": "\\textgreater{}",
        "|": "\\textbar{}", '"': "''", "`": "\\textasciigrave{}"}
_BREAK_AFTER = set("/_.:,=()[]*+;")
LONG_TOKEN = 20          # tokens longer than this get break points after separators


def tex_escape(text: str) -> str:
    """The cell text for a ragged-right p column: LaTeX specials escaped, '--' kept as two hyphens,
    and discretionary break points after separators inside long tokens (paths, calls, file patterns)."""
    out = []
    for tok in re.split(r"( +)", text):
        if not tok or tok.isspace():
            out.append(" " if tok else "")
            continue
        brk = len(tok) > LONG_TOKEN
        piece = []
        for i, ch in enumerate(tok):
            piece.append(_ESC.get(ch, ch))
            if ch == "-" and i + 1 < len(tok) and tok[i + 1] == "-":
                piece.append("{}")                       # no en-dash ligature
            elif brk and ch in _BREAK_AFTER and i + 1 < len(tok):
                piece.append("\\allowbreak{}")
        out.append("".join(piece))
    return "".join(out)


def longest_unbreakable(text: str) -> int:
    """Length of the longest stretch of the cell that TeX cannot break (used by the tests)."""
    best = 0
    for tok in text.split():
        if len(tok) <= LONG_TOKEN:
            best = max(best, len(tok))
            continue
        cur = 0
        for i, ch in enumerate(tok):
            cur += 1
            if (ch in _BREAK_AFTER or ch == "-") and i + 1 < len(tok):
                best = max(best, cur)
                cur = 0
        best = max(best, cur)
    return best


class CardTable(Table):
    """A two-column longtable of cards: ('group', title) | ('card', title) | ('field', name, text)."""

    WIDTH_NAME, WIDTH_TEXT, WIDTH_FULL = "2.25cm", "11.0cm", "13.4cm"

    def __init__(self, name: str, label: str, caption: str = "", printed: bool = False):
        super().__init__(name, label, caption, "", ["Field", "Facts table text"], size="\\scriptsize", long=True,
                         tabcolsep="3pt")
        self.printed = printed

    def group(self, text: str) -> None:
        self.rows.append(("group", text))

    def card(self, title: str, row_arm: str = "") -> None:
        self.rows.append(("card", title, row_arm))

    def field(self, name: str, text: str) -> None:
        self.rows.append(("field", name, text))

    def render(self, provenance_comment: str) -> str:
        full = f"\\multicolumn{{2}}{{@{{}}p{{{self.WIDTH_FULL}}}@{{}}}}"
        head = ("\\textbf{Field} & \\textbf{" + ("Description" if self.printed else "Text of the facts table")
                + "} \\tabularnewline")
        L = [f"% {provenance_comment}",
             "% Generated file, do not edit by hand. Requires \\usepackage{longtable,booktabs}.",
             "{" + self.size + f"\\setlength{{\\tabcolsep}}{{{self.tabcolsep}}}",
             f"\\begin{{longtable}}{{@{{}}p{{{self.WIDTH_NAME}}}p{{{self.WIDTH_TEXT}}}@{{}}}}",
             f"\\caption{{{self.caption}}}\\label{{{self.label}}}\\tabularnewline", "\\toprule", head, "\\midrule",
             "\\endfirsthead", "\\toprule", head, "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"]
        first = True
        for r in self.rows:
            if r[0] == "group":
                if not first:
                    L.append("\\midrule")
                L.append(f"{full}{{\\raggedright\\hspace{{0pt}}\\textit{{{upright_names(r[1])}}}}} \\tabularnewline*")
                L.append("\\midrule")
            elif r[0] == "card":
                if not first and L[-1] != "\\midrule":
                    L.append("\\addlinespace")
                L.append(f"{full}{{\\raggedright\\hspace{{0pt}}{r[1]}}} \\tabularnewline*")
            else:
                L.append(f"\\raggedright\\hspace{{0pt}}{r[1]} & \\raggedright\\hspace{{0pt}}{r[2]} \\tabularnewline")
            first = False
        L.append("\\end{longtable}}")
        return "\n".join(L) + "\n"


# =========================================================================================
# cells computed from the loaded files
# =========================================================================================
def _file_env(path: str):
    """(python, cuda, ((package, version), ...), host) of one result file, or None if not recorded."""
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    p = d.get("provenance")
    if isinstance(p, dict) and isinstance(p.get("provenance"), dict):
        p = p["provenance"]
    if not isinstance(p, dict) or not isinstance(p.get("package_versions"), dict):
        return None
    pk = tuple(sorted((str(k), str(v)) for k, v in p["package_versions"].items()))
    return (str(p.get("python_version")), str(p.get("cuda_version")), pk, str(p.get("hostname")))


def arm_envs(ctx: Ctx, arm: str) -> Counter:
    """Counter of the software environments recorded in the arm's loaded result files (cached per run)."""
    store = ctx.cache.setdefault("baselines_env", {})
    if arm not in store:
        store[arm] = Counter(_file_env(c.file) for c in ctx.ds.arm(arm).cells.values())
    return store[arm]


def env_text(env) -> str:
    if env is None:
        return "no package versions recorded in the file"
    py, cuda, pk, host = env
    d = dict(pk)
    names = [k for k in PKG_ORDER if k in d] + sorted(k for k in d if k not in PKG_ORDER)
    parts = [f"python {py}"] + [f"{k} {d[k]}" + (f" (CUDA {cuda})" if k == "torch" and cuda not in ("None", "") else "")
                                 for k in names]
    return ", ".join(parts) + f"; host {host}"


def _span(seeds) -> str:
    seeds = sorted(seeds)
    if not seeds:
        return "none"
    if seeds == list(range(seeds[0], seeds[-1] + 1)):
        return f"{seeds[0]}--{seeds[-1]}" if len(seeds) > 1 else str(seeds[0])
    return ", ".join(str(s) for s in seeds)


def _software_cell(ctx: Ctx, table: str, cr_arms: list[str]) -> str:
    """Software versions recorded in the provenance of the loaded result files of the given arms."""
    total = Counter()
    for a in cr_arms:
        c = arm_envs(ctx, a)
        total.update(c)
        ctx.num(f"crBaselines{camel(a)}EnvSets", len(c), str(len(c)), kind="count", table=table,
                fileset=ctx.eng.fileset_all(a), n=sum(c.values()),
                rule="distinct software environments (python, CUDA, package versions, host) recorded in the "
                     "provenance of the arm's loaded result files")
    n = sum(total.values())
    of = f" of {len(cr_arms)} variants" if len(cr_arms) > 1 else ""
    if len(total) == 1:
        (env, k), = total.items()
        return tex_escape(f"{env_text(env)} ({k} of {n} result file{'s' if n != 1 else ''}{of})")
    parts = [f"({chr(97 + i)}) {env_text(env)} ({k} file{'s' if k != 1 else ''})"
             for i, (env, k) in enumerate(total.most_common(MAX_ENVS_LISTED))]
    more = len(total) - len(parts)
    return tex_escape(f"{len(total)} environments in {n} result files{of}: " + "; ".join(parts)
                      + (f"; and {more} more" if more > 0 else ""))


def _tag(i: int) -> str:
    """(a), (b), ...: how the cells of a card with several arms point into its code-name field."""
    return f"({chr(97 + i)})"


def code_name_cell(row_arm: str, arms: list[str]) -> str:
    """The card's one code-name field (RUN_RULES 2026-10-19 second set (c)): the repository identifier of each
    arm, with the tag the other cells use in a card with several arms. A card of another benchmark (no loaded
    arm) names its method in that benchmark's runner."""
    if not arms:
        bench, _, mth = row_arm.partition(":")
        if bench in OTHER_RUNNERS:
            return (f"\\texttt{{{tex_escape(mth)}}} in \\texttt{{{tex_escape(OTHER_RUNNERS[bench])}}}")
        return "\\texttt{" + tex_escape(row_arm) + "}"
    if len(arms) == 1:
        return "\\texttt{" + tex_escape(arms[0]) + "}"
    return "; ".join(f"{_tag(i)} \\texttt{{{tex_escape(a)}}}" for i, a in enumerate(arms))


def _seed_cell(ctx: Ctx, table: str, arms: list[str]) -> str:
    """Seeds, tasks, cells and completed cells of every arm of a card, counted in the loaded files."""
    parts = []
    for a in arms:
        A = ctx.ds.arm(a)
        seeds = {s for _, s in A.cells} | {s for _, s in A.sidecar_crashes}
        tasks = {t for t, _ in A.cells} | {t for t, _ in A.sidecar_crashes}
        done = sum(1 for c in A.cells.values() if c.status_ok)
        fs = ctx.eng.fileset_all(a)
        key = f"crBaselines{camel(a)}"
        ctx.num(key + "Seeds", len(seeds), str(len(seeds)), kind="count", table=table, fileset=fs, n=len(A.cells),
                rule="distinct seeds among the arm's loaded cells (result files and crash records)")
        ctx.num(key + "Cells", len(A.cells), N.intfmt(len(A.cells)), kind="count", table=table, fileset=fs,
                n=len(A.cells), rule="result files loaded for the arm")
        ctx.num(key + "Completed", done, N.intfmt(done), kind="count", table=table, fileset=fs, n=len(A.cells),
                rule="loaded result files with status completed")
        txt = (f"{len(seeds)} seed{'s' if len(seeds) != 1 else ''} ({_span(seeds)}) $\\times$ {len(tasks)} "
               f"task{'s' if len(tasks) != 1 else ''}: {N.intfmt(len(A.cells))} result "
               f"file{'s' if len(A.cells) != 1 else ''}, {N.intfmt(done)} completed")
        if A.sidecar_crashes:
            txt += f", {len(A.sidecar_crashes)} crash record{'s' if len(A.sidecar_crashes) != 1 else ''} without a result"
        parts.append((f"{_tag(arms.index(a))} " if len(arms) > 1 else "") + txt)
    return "; ".join(parts)


# =========================================================================================
# the table
# =========================================================================================
def _needs_proof(row: dict, cols: list[str], text: str) -> bool:
    return ("all" in row["_proof"] or any(c in row["_proof"] for c in cols) or bool(UNCERTAIN_RX.search(text)))


TOKEN_RX = re.compile(r"\{(fallback_steps|fallback_cells|cells)\}")


REF_RX = re.compile(r"\{Table:([a-z]+:[a-z_]+)\}")


def _subst(text: str, subst: dict | None) -> str:
    """Replace the count tokens of a proofread text; a token without a count is an error."""
    for tok, val in (subst or {}).items():
        text = text.replace(tok, val)
    if TOKEN_RX.search(text):
        raise RuntimeError(f"count token {TOKEN_RX.search(text).group(0)} in a card without loaded files: {text!r}")
    return text


def tex_printed(text: str, subst: dict | None = None) -> str:
    """LaTeX of a proofread text: count tokens replaced, the text escaped, and each ``{Table:<label>}`` turned into
    ``Table~\\ref{<label>}`` (a cross-reference the plain text cannot carry through the escaping)."""
    text = _subst(text, subst)
    parts = REF_RX.split(text)
    out = []
    for i, part in enumerate(parts):
        out.append(tex_escape(part) if i % 2 == 0 else f"Table~\\ref{{{part}}}")
    return "".join(out)


def _facts_field(row: dict, key: str, cols: list[str], use_override: bool,
                 printed: dict | None = None, subst: dict | None = None) -> tuple[str, bool]:
    """(LaTeX text, needs proofreading) of a field taken from the facts table, or from its proofread entry."""
    if printed is not None and (row["arm"], key) in printed:
        return tex_printed(printed[(row["arm"], key)]["text"], subst), False
    if key == "acquisition":
        verb = f"{row[K_ACQ]}; maximiser: {row[K_MAX]}"
    else:
        verb = row[cols[0]]
    if use_override and key in row["_overrides"]:
        txt = (tex_escape(row["_overrides"][key]) + " \\emph{Audited description of the stored files:} "
               + tex_escape(verb))
        return txt, True
    return tex_escape(verb), _needs_proof(row, cols, verb)


def _fallback_tokens(ctx: Ctx, table: str, row_arm: str, arms: list[str]) -> dict:
    """{fallback_steps}, {fallback_cells}, {cells} of a card's arms, counted by the loader (the counts of the
    arm-status table), registered once per card."""
    cells = [c for a in arms for c in ctx.ds.arm(a).cells.values()]
    fb = [c for c in cells if c.fallback_count > 0]
    steps = int(sum(c.fallback_count for c in fb))
    key = f"crBaselines{camel(arms[0] if len(arms) == 1 else row_arm)}"
    fs = ctx.fs_union(*[ctx.eng.fileset_all(a) for a in arms])
    ctx.num(key + "FallbackSteps", steps, N.intfmt(steps), kind="count", table=table, fileset=fs, n=len(cells),
            rule="proposal steps replaced by a uniform draw (surrogate_fallback_count summed over the loaded cells)")
    ctx.num(key + "FallbackCells", len(fb), N.intfmt(len(fb)), kind="count", table=table, fileset=fs,
            n=len(cells), rule="loaded cells with at least one fallback step")
    return {"{fallback_steps}": N.intfmt(steps), "{fallback_cells}": N.intfmt(len(fb)),
            "{cells}": N.intfmt(len(cells))}


def tab_baselines(ctx: Ctx) -> Table | None:
    table = "tab_baselines"
    if not os.path.exists(FACTS_CSV):
        ctx.skip(table, "appendixH_facts.csv", "absent")
        return None
    rows = load_facts(FACTS_CSV)
    facts_fs = ctx.fs_paths([FACTS_CSV])
    printed = load_printed(PRINTED_CSV, rows) if os.path.exists(PRINTED_CSV) else None
    t = CardTable("baselines", "tab:baselines", printed=printed is not None)
    n_cards = n_guacamol = n_dag = n_ddag = n_cells = 0
    used_fs = [facts_fs] + ([ctx.fs_paths([PRINTED_CSV])] if printed is not None else [])
    covered: set = set()
    row_of = {a: r["arm"] for r in rows for a in r["_run"]}
    substituted: set = set()

    def put(name: str, text: str, mark: str) -> None:
        nonlocal n_dag, n_ddag, n_cells
        n_cells += 1
        n_dag += mark == DAGGER
        n_ddag += mark == DDAGGER
        t.field(name, reader_facing(neutral_hosts(text)) + mark)

    def pr(row: dict, key: str) -> dict | None:
        return printed.get((row["arm"], key)) if printed is not None else None

    any_cr = False
    for group in GROUP_ORDER:
        head_done = False
        for row in rows:
            if row["table group"] != group:
                continue
            other = not row["_run"]
            if other and not OTHER_BENCHMARKS:
                continue
            present = [a for a in row["_run"] if ctx.has(a)]
            for a in list(present):
                sp = ctx.ds.arm(a).spec
                # substitute files: a display override, no camera-ready arm id, and the files' solver belongs
                # to ANOTHER facts row (a replication arm <arm>__s82 reads its own row's solver)
                if sp.display and not sp.cr_arm and row_of.get(sp.solver_field, row["arm"]) != row["arm"]:
                    present.remove(a)       # substitute files of another row's solver: not this row's method
                    substituted.add(a)
                    ctx.skip(table, a, f"reads substitute files of solver {sp.solver_field} (display override): "
                                       "not described by its own facts row, left out of the card")
            for a in row["_run"]:
                if not ctx.has(a):
                    ctx.skip(table, a, "arm absent" + (" (card omitted)" if not present else " (not in the card)"))
            if not other and not present:
                continue
            covered.update(present)
            cr = [a for a in present if ctx.ds.arm(a).spec.kind == "cr"]
            stored = [a for a in present if a not in cr]
            any_cr = any_cr or bool(cr)
            if not head_done:
                t.group(GROUP_HEADINGS[group] if printed is not None else tex_escape(group))
                head_done = True
            # RUN_RULES 2026-10-19 second set (c): the title carries the printed label (authors' name only for a
            # verified recipe), the code name appears only in the card's code-name field
            labels = []
            for a in present:
                lab = printed_label(ctx, a) if ctx.mode == "cr" else ctx.label(a, ctx.ds.arm(a).spec.label)
                lab = lab if "$" in lab or "\\" in lab else tex_escape(lab)
                if lab not in labels:
                    labels.append(lab)
            if other:
                ti = pr(row, "title")
                labels = [ti["text"] if ti is not None else
                          (other_row_label(ctx, row["arm"]) if ctx.mode == "cr" else tex_escape(row["arm"]))]
            title = "\\textbf{" + "; ".join(labels[:8]) + ("; \\dots" if len(labels) > 8 else "") + "}"
            if row["_new"] and printed is None:
                title += " (row written for this version" + DAGGER + ")"
            t.card(title, row["arm"])
            # the printed cards, for the Appendix H overview (crlib/overview.py) built after every other table
            ctx.extra.setdefault("baseline_cards", []).append(
                {"row": row["arm"], "group": group, "present": list(present), "other": other, "labels": list(labels),
                 "title": title})
            t.field("Code name", code_name_cell(row["arm"], present) if printed is not None else
                    _legacy_code_name_cell(row["arm"], present))
            n_cards += 1
            n_guacamol += not other
            used_fs += [ctx.eng.fileset_all(a) for a in present]
            subst = _fallback_tokens(ctx, table, row["arm"], present) if present and printed is not None and any(
                TOKEN_RX.search(e["text"]) for (r_, _), e in printed.items() if r_ == row["arm"]) else None
            for key, name, cols in FIELDS:
                ent = pr(row, key)
                if key == "software" and present:
                    # camera-ready arms: what their files record. Stored arms: the checked text, then what
                    # their files record.
                    parts, proof = [], False
                    if cr:
                        parts.append(_software_cell(ctx, table, cr))
                    if stored:
                        if ent is not None:
                            verb, proof = ent["text"], False
                        else:
                            verb = row[K_SOFT]
                            proof = _needs_proof(row, cols, verb)
                        who = (" " + ", ".join(_tag(present.index(a)) for a in stored)) if cr and len(present) > 1 else ""
                        lead = ((f"\\emph{{Runs from the submitted version{who}:}} " if printed is not None else
                                 f"\\emph{{Stored arms{who}, facts table:}} ") if cr else "")
                        rec_lab = ("\\emph{Recorded in the result files:} " if printed is not None else
                                   "\\emph{Recorded in the loaded files:} ")
                        vt = (tex_printed(verb, subst) if ent is not None else tex_escape(verb)) if verb else ""
                        if vt and ent is not None and not vt.rstrip().endswith("."):
                            vt += "."
                        parts.append(lead + (vt + " " if vt else "") + rec_lab
                                     + _software_cell(ctx, table, stored))
                    put(name, " ".join(parts), DAGGER if proof else DDAGGER)
                elif key == "seeds" and present:
                    txt, proof = _seed_cell(ctx, table, present), False
                    if stored:
                        if ent is not None:
                            if ent["text"]:
                                txt += ". " + tex_printed(ent["text"], subst)
                        else:
                            txt += " \\emph{Audit of the stored files:} " + tex_escape(row[K_FILES])
                            proof = _needs_proof(row, cols, row[K_FILES])
                    put(name, txt, DAGGER if proof else DDAGGER)
                else:
                    txt, proof = _facts_field(row, key, cols, use_override=bool(cr), printed=printed, subst=subst)
                    if other and ent is None and row["arm"].startswith("lassobench:") and key in ("software", "seeds"):
                        proof = True          # describes the stored runs; LassoBench is rerun in full
                    put(name, txt, DAGGER if proof else "")
    if not n_guacamol:
        ctx.skip(table, "GuacaMol arms", "no arm with a facts row is present (table omitted)")
        return None
    # arms of the run that no facts row covers (reported, never silently dropped)
    main = [n for n, a in ctx.ds.arms.items() if not a.absent and not a.spec.namespace]
    no_row = sorted(set(main) - covered - substituted)
    for a in no_row:
        ctx.skip(table, a, "no Appendix-H facts row for this arm (not in the table)")
    fs_all = ctx.fs_union(*used_fs)
    for nm, v, rule in (
            ("crBaselinesCards", n_cards, "cards (facts rows) printed in the baseline table"),
            ("crBaselinesCells", n_cells, "field cells printed (ten per card)"),
            ("crBaselinesProofCells", n_dag, "cells marked with a dagger: text not checked against the code"),
            ("crBaselinesComputedCells", n_ddag, "cells marked with a double dagger: computed from the loaded files"),
            ("crBaselinesMarkedCells", n_dag + n_ddag, "cells carrying either marker")):
        ctx.num(nm, v, N.intfmt(v), kind="count", table=table, fileset=fs_all, rule=rule)
    ctx.num("crBaselinesFactsRows", len(rows), str(len(rows)), kind="count", table=table, fileset=facts_fs,
            rule="rows of appendixH_facts.csv (54 audited + rows added for the camera-ready and control arms)")
    ctx.num("crBaselinesArmsWithoutRow", len(no_row), str(len(no_row)), kind="count", table=table,
            fileset=ctx.fs_union(facts_fs, *[ctx.eng.fileset_all(a) for a in no_row]),
            rule="loaded arms of the run that no facts row covers (listed under 'skipped' in the run manifest)")
    if printed is not None:
        t.caption = _reader_caption(n_dag, n_ddag, n_cells, no_row, any_cr)
    else:
        t.caption = _legacy_caption(n_dag, n_ddag, n_cells, no_row, any_cr)
    return t


def _reader_caption(n_dag: int, n_ddag: int, n_cells: int, no_row: list, any_cr: bool) -> str:
    """Caption of the proofread table, written for a reader of the paper."""
    gen = ("" if any_cr else "No run of this version is loaded here: every card reads the stored result files. ")
    return (
        "\\textbf{Implementation of every method in the GuacaMol, LassoBench and NAS-Bench-101 tables.} There is one "
        "card per method, with ten fields. They are: software version, kernel, acquisition function and its maximizer, lengthscale "
        "initialization, lengthscale prior, candidate support, trust region and RAASP (random axis-aligned subspace "
        "perturbation) settings, seed count, repository of the published method, and deviations from it. A card "
        "is titled by the row label of its method. The authors of a published method are named only for a recipe "
        "that reproduced a published result (Table~\\ref{tab:verification}). The code-name field gives the "
        "identifier of the method in the released code. In a card with several methods, the tags (a), (b), "
        "\\dots{} point from the counted fields to that list. Each field was checked against the code that "
        "produced the runs. For the runs of this version that is tag \\texttt{cr-runs-2026-10-12}, and for the "
        "published recipes tag \\texttt{cr-verify-2026-10-13}. For the runs from the submitted version it is the "
        "code their result files record. Machines are named by their hardware. " + gen
        + f"{DDAGGER}~counted from the loaded result files ({n_ddag} fields). "
        + (f"{DAGGER}~text not yet checked against the code: {n_dag} of {n_cells} fields. " if n_dag else "")
        + _no_card_sentence(no_row))


#: Loaded arms without a facts row that repeat a carded method in another setting: (prefix, setting, label).
REPEATED_SETTINGS = (("b1000_", "at the larger budgets of Table~\\ref{tab:budget}", "tab:budget"),
                     ("chembed_", "with the other codecs of Table~\\ref{tab:crosscodec_means}", "tab:crosscodec_means"),
                     ("power_spherical_", "with the other codecs of Table~\\ref{tab:crosscodec_means}",
                      "tab:crosscodec_means"))


def _no_card_sentence(no_row: list) -> str:
    """The loaded arms without a card: the stored budget and codec runs repeat carded methods in another setting;
    any other arm is named as uncovered."""
    if not no_row:
        return ""
    settings, other = [], []
    for a in no_row:
        hit = next((st for pre, st, _ in REPEATED_SETTINGS if a.startswith(pre)), None)
        if hit is None:
            other.append(a)
        elif hit not in settings:
            settings.append(hit)
    out = ""
    n_rep = len(no_row) - len(other)
    if n_rep:
        out += (f"The {n_rep} runs of the submitted version {' and '.join(settings)} repeat methods of these cards in "
                f"another setting. They have no card of their own; the run manifest of the released code lists them. ")
    if other:
        out += (f"{len(other)} further loaded run{'s' if len(other) != 1 else ''} "
                f"{'have' if len(other) != 1 else 'has'} no card (listed in the run manifest). ")
    return out


def _legacy_code_name_cell(row_arm: str, arms: list[str]) -> str:
    """Code-name field of the facts-table rendering (no proofread text): facts row and arms."""
    if len(arms) <= 1:
        ids = arms or [row_arm]
        return "\\texttt{" + tex_escape(ids[0]) + "}" + ("" if ids[0] == row_arm else
                                                          f" (facts row \\texttt{{{tex_escape(row_arm)}}})")
    return (f"facts row \\texttt{{{tex_escape(row_arm)}}}; arms "
            + "; ".join(f"{_tag(i)} \\texttt{{{tex_escape(a)}}}" for i, a in enumerate(arms)))


def _legacy_caption(n_dag: int, n_ddag: int, n_cells: int, no_row: list, any_cr: bool) -> str:
    gen = ("Arms loaded as camera-ready runs take the software version and the seed count from their result files; "
           "their other cells are the audited description of the same solver code. " if any_cr else
           "No camera-ready run is loaded here (legacy mode): every arm reads its stored files. ")
    return (
        "\\textbf{Implementation of every GuacaMol arm and baseline (the facts table behind Appendix~H).} One card per "
        "row of the facts table whose arm is in this run, with the ten fields: software version, kernel, acquisition, "
        "lengthscale initialisation, lengthscale prior, candidate support, trust-region and RAASP settings, seed count, "
        "repository, and deviations from the published method. A card is titled by the printed label of its arms (the "
        "authors' name only for a verified recipe, otherwise the descriptive label) and opens with its code name, the "
        "identifier in the repository; in a card with several arms the tags (a), (b), \\dots{} point from the "
        "computed cells to that list. The text is that of the facts table (an audit of the "
        "code at commit \\texttt{abbd61e} and of the stored result files, 2026-10-11; rows marked as written for this "
        "version describe the code at the tags \\texttt{cr-controls-2026-10-12} and, for the published recipes, "
        "\\texttt{cr-verify-2026-10-13}). Machines are named by their hardware. " + gen +
        f"{DAGGER}~text that needs author proofreading (rows written for this version, derived cells such as the two "
        "lengthscale fields, camera-ready overrides of a stored-file description, and cells that say they are "
        f"unverified, not recorded or derived): {n_dag} of {n_cells} cells. {DDAGGER}~computed by the generator from "
        f"the loaded result files: {n_ddag} cells (a cell that also holds facts-table text needing proofreading "
        f"carries {DAGGER}). Software version of a stored arm: the audited text, then what its loaded files record. "
        "Seed count: distinct seeds $\\times$ tasks among the loaded cells; for stored arms the audit's own file "
        "count follows. "
        + (f"{len(no_row)} loaded arm{'s' if len(no_row) != 1 else ''} of this run "
           f"{'have' if len(no_row) != 1 else 'has'} no facts row and no card (listed in the run manifest). "
           if no_row else "")
        + ("The groups without a GuacaMol arm describe, from the facts table alone, the LassoBench published methods "
           "and the stored LassoBench and NAS-Bench-101 runs; LassoBench is rerun in full for this version, so its "
           "software version and seed count are those of the LassoBench table, not of these cards." if OTHER_BENCHMARKS else ""))
