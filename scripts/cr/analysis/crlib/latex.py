"""Booktabs table writer. Every table: caption above the tabular, a label, and the matched n."""
from __future__ import annotations

import os


#: Reader-facing name of the binding rules file (docs/camera_ready/RUN_RULES.md). Since 2026-10-22 (RUN_RULES row of
#: that date on generator edits after the final run) printed captions no longer cite rule numbers; they say what the
#: rule does (``COMPLETED``, ``SUBMITTED``). ``rules()`` stays for the tables that are not printed.
RULES = "the run rules released with the code"

#: reader phrases printed instead of rule numbers (Table 1 defines both in full; other tables use the short forms)
COMPLETED_FULL = ("completed cells (runs that reached all 30 evaluations; runs in which a surrogate fit failed and "
                  "a step fell back to a uniform proposal are included)")
COMPLETED_SHORT = "completed cells (as in Table~\\ref{tab:main})"
SUBMITTED_FULL = ("run from the submitted version (not rerun for this version: it ran through a separate harness or "
                  "no current environment reproduces it)")
SUBMITTED_SHORT = "run from the submitted version, not rerun"


def rules(item: str = "") -> str:
    """'rule R-5 of the run rules released with the code', 'section 3 of ...', 'the amendment of 2026-10-12 to ...'."""
    if not item:
        return RULES
    if item.startswith("R-"):
        return f"rule {item} of {RULES}"
    if item[:4].isdigit():
        return f"the amendment of {item} to {RULES}"
    return f"{item} of {RULES}"


def stack(*lines: str, align: str = "c") -> str:
    """A header cell set on several lines (narrow numeric columns with a descriptive header)."""
    if len(lines) == 1:
        return lines[0]
    return f"\\begin{{tabular}}[b]{{@{{}}{align}@{{}}}}" + "\\\\".join(lines) + "\\end{tabular}"


#: a wrapped, left-aligned column that takes the width the other columns leave (tabularx)
WRAP = ">{\\raggedright\\arraybackslash}X"


def wrapp(width: str) -> str:
    """A wrapped, left-aligned column of fixed width."""
    return f">{{\\raggedright\\arraybackslash}}p{{{width}}}"


def upright_names(text: str) -> str:
    """\\geowalk{} / \\geocma{} upright inside an italic subheader (small caps have no italic shape in Times: the
    T1/ptm/m/scit substitution warning)."""
    return text.replace("\\geowalk{}", "\\textup{\\geowalk{}}").replace("\\geocma{}", "\\textup{\\geocma{}}")


def m(text: str) -> str:
    """Math-mode cell ('--' stays text)."""
    if text in ("--", ""):
        return text
    return f"${text}$"


def mb(text: str, bold: bool) -> str:
    if text in ("--", "") or not bold:
        return m(text)
    return f"$\\mathbf{{{text}}}$"


class Table:
    """rows: list of list[str] | ('rule',) | ('group', text) | ('raw', latex)."""

    def __init__(self, name: str, label: str, caption: str, colspec: str, header: list[str],
                 size: str = "\\small", wide: bool = False, notes: str = "", long: bool = False,
                 tabcolsep: str | None = None):
        self.name, self.label, self.caption = name, label, caption
        self.colspec, self.header, self.size, self.wide, self.notes = colspec, header, size, wide, notes
        self.long = long
        self.tabcolsep = tabcolsep        # e.g. "4pt": longtables cannot be scaled to the line width
        self.rows: list = []
        self.blocks: list = []   # extra (header, colspec, rows, title) blocks for multi-block tables
        #: (caption, label): set the extra blocks as a second float right after the first (a table too tall for one
        #: page at a legible size; RUN_RULES 2026-10-22, format only)
        self.continued: tuple[str, str] | None = None
        #: float placement of the first float ("t"; Table 1 is a float page, "p")
        self.placement = "t"
        #: extra generated files written next to the table, {file name: LaTeX text} (Table 1 notes)
        self.extra: dict[str, str] = {}
        self.extra_paths: list[str] = []
        #: raw header lines set between \toprule and the column header (a grouped header; RUN_RULES 2026-10-29)
        self.header_pre: list[str] = []
        #: repository records of this printed table (RUN_RULES 2026-10-29): full versions written to records/ by
        #: make_tables.run_variant -- Table objects (written with their own name) and {file name: text} files
        self.record_tables: list = []
        self.record_files: dict[str, str] = {}

    def add(self, row) -> None:
        self.rows.append(row)

    def rule(self) -> None:
        self.rows.append(("rule",))

    def group(self, text: str) -> None:
        self.rows.append(("group", text))

    def add_block(self, title: str, colspec: str, header: list[str], rows: list) -> None:
        self.blocks.append((title, colspec, header, rows))

    @staticmethod
    def _tabular(colspec: str, header: list[str], rows: list, wide: bool, pre: list[str] | None = None) -> list[str]:
        """A tabular scaled down to the line width only when it is wider (graphicx only: the tabular is
        set in a save box, measured, and resized only if its natural width exceeds the line width)."""
        ncol = len(header)
        # a colspec with an X column is set as tabularx at the line width: the X column wraps and takes the width the
        # other columns leave, so the table is not scaled (legibility; RUN_RULES 2026-10-22)
        env = "tabularx" if "X" in colspec else "tabular"
        begin = (f"\\begin{{tabularx}}{{\\linewidth}}{{{colspec}}}" if env == "tabularx"
                 else f"\\begin{{tabular}}{{{colspec}}}")
        out = ["\\begin{lrbox}{\\crtabbox}", begin, "\\toprule"]
        out += list(pre or [])
        out.append(" & ".join(header) + " \\\\")
        out.append("\\midrule")
        for r in rows:
            if isinstance(r, tuple):
                if r[0] == "rule":
                    out.append("\\midrule")
                elif r[0] == "group":
                    out.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{upright_names(r[1])}}}}} \\\\")
                elif r[0] == "raw":
                    out.append(r[1])
                continue
            if len(r) != ncol:
                raise ValueError(f"row has {len(r)} cells, header {ncol}: {r}")
            # a row that starts with '[' would be read as the optional argument of the previous line break
            out.append(("{}" if str(r[0]).lstrip().startswith("[") else "") + " & ".join(r) + " \\\\")
        out.append("\\bottomrule")
        out += [f"\\end{{{env}}}", "\\end{lrbox}",
                "\\ifdim\\wd\\crtabbox>\\linewidth\\resizebox{\\linewidth}{!}{\\usebox{\\crtabbox}}"
                "\\else\\usebox{\\crtabbox}\\fi"]
        return out

    def render_long(self, provenance_comment: str) -> str:
        """Multi-page table (longtable): caption on the first page, header repeated."""
        ncol = len(self.header)
        hdr = " & ".join(self.header) + " \\\\"
        lines = [f"% {provenance_comment}", "% Generated file, do not edit by hand. Requires \\usepackage{longtable,booktabs}.",
                 "{" + self.size + (f"\\setlength{{\\tabcolsep}}{{{self.tabcolsep}}}" if self.tabcolsep else ""),
                 f"\\begin{{longtable}}{{{self.colspec}}}",
                 f"\\caption{{{self.caption}}}\\label{{{self.label}}}\\\\", "\\toprule", hdr, "\\midrule",
                 "\\endfirsthead", "\\toprule", hdr, "\\midrule", "\\endhead", "\\bottomrule", "\\endfoot"]
        for r in self.rows:
            if isinstance(r, tuple):
                if r[0] == "rule":
                    lines.append("\\midrule")
                elif r[0] == "group":
                    lines.append(f"\\multicolumn{{{ncol}}}{{@{{}}l}}{{\\textit{{{upright_names(r[1])}}}}} \\\\")
                continue
            lines.append(" & ".join(r) + " \\\\")
        lines += ["\\end{longtable}}"]
        return "\n".join(lines) + "\n"

    def render(self, provenance_comment: str) -> str:
        if self.long:
            return self.render_long(provenance_comment)
        lines = [f"% {provenance_comment}", "% Generated file, do not edit by hand. Requires \\usepackage{booktabs,graphicx,array,tabularx}.",
                 "\\ifdefined\\crtabbox\\else\\newsavebox{\\crtabbox}\\fi",
                 f"\\begin{{table}}[{self.placement}]", "\\centering", f"\\caption{{{self.caption}}}",
                 f"\\label{{{self.label}}}", self.size]
        if self.tabcolsep:
            lines.append(f"\\setlength{{\\tabcolsep}}{{{self.tabcolsep}}}")
        lines += self._tabular(self.colspec, self.header, self.rows, self.wide, self.header_pre)
        if self.continued and self.blocks:
            lines += ["\\end{table}", "\\begin{table}[t]", "\\centering", f"\\caption{{{self.continued[0]}}}",
                      f"\\label{{{self.continued[1]}}}", self.size]
            if self.tabcolsep:
                lines.append(f"\\setlength{{\\tabcolsep}}{{{self.tabcolsep}}}")
        for i, (title, colspec, header, rows) in enumerate(self.blocks):
            if i or not self.continued:
                lines.append("\\par\\medskip")
            if title:
                lines.append(f"{{\\footnotesize\\textit{{{upright_names(title)}}}}}\\par\\smallskip")
            lines += self._tabular(colspec, header, rows, self.wide)
        if self.notes:
            lines += ["\\par\\smallskip", f"\\begin{{minipage}}{{0.97\\linewidth}}\\footnotesize {self.notes}\\end{{minipage}}"]
        lines.append("\\end{table}")
        return "\n".join(lines) + "\n"

    def write(self, outdir: str, provenance_comment: str) -> str:
        path = os.path.join(outdir, f"tab_{self.name}.tex")
        with open(path, "w") as f:
            f.write(self.render(provenance_comment))
        self.extra_paths = []
        for fname, text in self.extra.items():
            p = os.path.join(outdir, fname)
            with open(p, "w") as f:
                f.write(f"% {provenance_comment}\n% Generated file, do not edit by hand.\n{text}\n")
            self.extra_paths.append(p)
        return path


PREAMBLE_CHECK = r"""\documentclass{article}
\usepackage[textwidth=5.5in,paperwidth=8.5in,paperheight=40in,top=1in]{geometry}
\usepackage{booktabs,amsmath,amssymb,graphicx,longtable,array,tabularx}
\usepackage{natbib}
\newcommand{\geowalk}{GeoWalk}
\newcommand{\geocma}{GeoCMA}
\begin{document}
"""
