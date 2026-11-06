"""Registry of every printed number: numbers.tex (one \\newcommand each) and the dossier."""
from __future__ import annotations

import json
import math
import re

_DIGITS = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five",
           "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}


def camel(part: str) -> str:
    """'turbo_box_faithful' -> 'TurboBoxFaithful'; digits become words ('r16' -> 'RSixteen'
    is not attempted: each digit becomes its own word, 'r16' -> 'ROneSix')."""
    out = []
    for tok in re.split(r"[^A-Za-z0-9]+", str(part)):
        if not tok:
            continue
        tok = re.sub(r"\d", lambda m: "_" + _DIGITS[m.group(0)] + "_", tok)
        for t in tok.split("_"):
            if t:
                out.append(t[0].upper() + t[1:])
    return "".join(out)


def macro(*parts) -> str:
    name = "cr" + "".join(camel(p) for p in parts)
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(f"bad macro name {name!r}")
    return name


def text_macros(numbers_tex: str) -> list[str]:
    """Names of the running-text macros in a written numbers.tex (kind "text": defined without \\ensuremath)."""
    pat = re.compile(r"\\newcommand\{\\([A-Za-z]+)\}\{(?!\\ensuremath\{)")
    with open(numbers_tex) as f:
        return [mo.group(1) for mo in map(pat.match, f) if mo]


# ---- formatting ------------------------------------------------------------------------
def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _nz(text: str) -> str:
    """A value that rounds to zero is printed without a sign: '0.00', never '-0.00' or '+0.00' (RUN_RULES 2026-10-22,
    generator edits after the final run: format only)."""
    if text[:1] in "+-" and float(text) == 0.0:
        return text[1:]
    return text


def f4(x) -> str:
    return "--" if _nan(x) else _nz(f"{x:.4f}")


def f3(x) -> str:
    return "--" if _nan(x) else _nz(f"{x:.3f}")


def f2(x) -> str:
    return "--" if _nan(x) else _nz(f"{x:.2f}")


def s4(x) -> str:
    return "--" if _nan(x) else _nz(f"{x:+.4f}")


def s3(x) -> str:
    return "--" if _nan(x) else _nz(f"{x:+.3f}")


def d2(x) -> str:
    return "--" if _nan(x) else _nz(f"{x:+.2f}")


def pfmt(p) -> str:
    """Cell-level p: <0.001 below 1e-3, else 3 decimals (2 above 0.1)."""
    if _nan(p):
        return "--"
    if p < 1e-3:
        return "<0.001"
    if p < 0.1:
        return f"{p:.3f}"
    return f"{p:.2f}"


def psig(p) -> str:
    """Exact task-level p (11 tasks: minimum 2/2048): two significant digits."""
    if _nan(p):
        return "--"
    if p >= 0.1:
        return f"{p:.2f}"
    return f"{p:.2g}" if p >= 1e-4 else f"{p:.1e}"


def pexp(p) -> str:
    """p with exponent, for sentences: 3.4e-23 -> 3.4\\times10^{-23}."""
    if _nan(p):
        return "--"
    if p >= 1e-3:
        return pfmt(p)
    m, e = f"{p:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def sci(x) -> str:
    """Two significant digits in LaTeX math: 5.0e-05 -> 5.0\\times10^{-5}; 0 -> 0 (exact)."""
    if _nan(x):
        return "--"
    if x == 0:
        return "0"
    m, e = f"{x:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def intfmt(n) -> str:
    if _nan(n):
        return "--"
    s = f"{int(n):,}"
    return s.replace(",", "{,}")


def math_text(text: str) -> str:
    """Text of a macro / table cell in math mode with proper minus and '<'."""
    if text in ("--", ""):
        return text
    return text


class Registry:
    """Every printed number. Each one carries the file set it was computed from (count, sha256
    of the sorted path list, sha256 of the sorted (path, content sha256) list): result files,
    stored inputs (floor score arrays, geometry arrays, LassoBench files, latent cache, bounds
    file) or, for a derived number (min/max/count over other numbers), the union of the file
    sets of the numbers it is derived from. A number without a file set is a programming error."""

    def __init__(self, script_commit: str, config_sha256: str, variant: str):
        self.script_commit = script_commit
        self.config_sha256 = config_sha256
        self.variant = variant
        self.records: dict[str, dict] = {}
        self.filesets: dict[str, dict] = {}     # name -> file set incl. its rows (in memory, for unions)

    def add(self, name: str, value, text: str, *, kind: str, table: str, fileset: dict | None,
            n=None, rule: str = "", arms=None, note: str = "") -> str:
        if name in self.records:
            raise ValueError(f"duplicate number name {name}")
        if not re.fullmatch(r"cr[A-Za-z]+", name):
            raise ValueError(f"bad number name {name}")
        if not fileset or "count" not in fileset or "list_sha256" not in fileset:
            raise ValueError(f"number {name} has no file set (every dossier number must carry one)")
        if fileset["count"] == 0:
            raise ValueError(f"number {name} has an empty file set")
        rec = {
            "name": name, "value": (None if _nan(value) else (float(value) if not isinstance(value, (str, list, dict)) else value)),
            "text": text, "kind": kind, "table": table, "n": n, "rule": rule,
            "arms": list(arms or (fileset or {}).get("arms", [])),
            "files": {k: fileset[k] for k in ("count", "list_sha256", "content_sha256")},
            "variant": self.variant, "script_commit": self.script_commit,
            "config_sha256": self.config_sha256, "note": note, "old": [],
        }
        self.records[name] = rec
        self.filesets[name] = fileset
        return text

    def attach_old(self, name: str, entry: dict) -> bool:
        if name not in self.records:
            return False
        self.records[name]["old"].append(entry)
        return True

    def value(self, name: str):
        return self.records[name]["value"]

    # ---- writers ------------------------------------------------------------------------
    def write_tex(self, path: str, header: str) -> None:
        lines = [f"% {header}", f"% variant: {self.variant}; script commit {self.script_commit}; "
                 f"{len(self.records)} numbers. Generated file, do not edit by hand.", ""]
        cur = None
        for name in sorted(self.records, key=lambda n: (self.records[n]["table"], n)):
            r = self.records[name]
            if r["table"] != cur:
                cur = r["table"]
                lines.append(f"% --- {cur}")
            txt = r["text"]
            if r["kind"] == "text":
                # running text (verdicts, sentences): no \ensuremath and no \text box, so a sentence can break
                # across lines in a paragraph; an empty text stays empty (a sentence that is not printed)
                lines.append(f"\\newcommand{{\\{name}}}{{{txt}}}")
                continue
            body = "--" if txt in ("--", "") else txt
            lines.append(f"\\newcommand{{\\{name}}}{{\\ensuremath{{{body}}}}}")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")

    def write_dossier(self, json_path: str, md_path: str, header: dict) -> None:
        with open(json_path, "w") as f:
            json.dump({"header": header, "numbers": self.records}, f, indent=1, sort_keys=False,
                      default=lambda o: o.item() if hasattr(o, "item") else str(o))
        lines = ["# Numbers dossier", "",
                 f"Variant `{self.variant}`, script commit `{self.script_commit}`, sources config sha256 "
                 f"`{self.config_sha256[:16]}`. {len(self.records)} printed numbers.", "",
                 "Columns: macro, printed text, value, table, matched n, aggregation rule, arms, file set "
                 "(count, sha256 of the sorted file list), old printed/posted value(s).", "",
                 "| macro | text | value | table | n | rule | arms | files | old |",
                 "|---|---|---|---|---|---|---|---|---|"]
        for name in sorted(self.records, key=lambda n: (self.records[n]["table"], n)):
            r = self.records[name]
            v = r["value"]
            vs = "" if v is None else (f"{v:.6g}" if isinstance(v, float) else str(v))
            files = "" if not r["files"] else f"{r['files']['count']} / `{r['files']['list_sha256'][:12]}`"
            old = "; ".join(f"{o.get('kind')}: {o.get('value')} ({o.get('where', '')})" for o in r["old"])
            arms = ", ".join(r["arms"][:6]) + (" ..." if len(r["arms"]) > 6 else "")
            lines.append(f"| `\\{name}` | `{r['text']}` | {vs} | {r['table']} | {r['n'] if r['n'] is not None else ''} "
                         f"| {r['rule']} | {arms} | {files} | {old} |")
        with open(md_path, "w") as f:
            f.write("\n".join(lines) + "\n")
