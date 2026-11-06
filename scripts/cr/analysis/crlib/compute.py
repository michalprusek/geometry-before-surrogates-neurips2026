"""Appendix K: the compute of the project, from the run times recorded in the result files.

``tab_compute`` prints, for every run block of the results bundle, the machine it ran on, the number of result
files, the process-hours and the median time of one run, and the totals per machine. The time of a run is
``run_time`` (the rule of ``tools/compute_summary.py``, which imports it from here): the first of the keys
``elapsed_seconds``, ``elapsed_s``, ``wall_seconds``, ``elapsed`` and ``wall_s_this_process`` that the file has; the
verification runs record only per-evaluation times, so for them it is the sum of ``t_eval_s``. Process-hours are
sums of these per-run wall-clock times; runs shared their machine with other runs, so they are not exclusive GPU or
CPU hours.

Run blocks (the folders the generator reads, i.e. the directories of the results bundle):

  block      folder                                   runs
  ---------  ---------------------------------------  ----------------------------------------------------------
  CR         root CR                                  GuacaMol runs of this version (GPU server)
  CRPUB      root CRPUB                               GuacaMol runs of the published recipes
  CRRD       root CRRD                                GuacaMol runs of 2026-11-02: full-dimensional CMA-ES (two
                                                      populations) and BAxUS
  CRCTL      root CRCTL                               GuacaMol control runs
  CRCPU      root CRCPU                               GuacaMol runs of the two sphere-manifold GPs (CPU server)
  LASSO_CR   --lasso-cr-root                          LassoBench rerun
  LASSO_PUB  --lasso-pub-root                         LassoBench runs of the published methods
  VERIFY     <inputs-root>/verification_runs          verification of the published recipes in the authors' settings
  STORED     root G                                   result files of the submitted version (only if a stored arm
                                                      is loaded)
  STORED_*   <inputs-root>/chembed, power_spherical,  the stored runs under the inputs root that printed tables
             nasbench                                 read (STORED_PRINTED): the codec comparison and NAS-Bench-101

The machine of a run is the host its result file records (``provenance.hostname``). The LassoBench rerun files
record no host; the rerun records it once, in the first line of ``env_provenance.txt`` of its folder, which is used
for the files of that block that record none. Hosts are named by their hardware (``crlib/baselines.py`` HOSTS).

Five totals are registered as macros (rounded to whole hours, as the paper's checklist prints them): the runs of this
version on the machine of block CR (``crComputeGpuServerHours``) and on the machine of block CRCPU
(``crComputeCpuServerHours``), the verification runs (``crComputeVerifyHours``), the runs of the submitted version
under root G (``crComputeStoredHours``), and those together with the STORED_* blocks
(``crComputeStoredPrintedHours``). The stored folders under the inputs root that no printed table reads
(NOT_COUNTED) are named in the caption.
"""
from __future__ import annotations

import os
import re
import statistics

from . import numbers as N
from .baselines import HOSTS
from .context import Ctx
from .engine import fileset_from_rows
from .latex import Table, stack
from .runtime import KEYS, folder_host, recorded_host, run_time, scan  # noqa: F401 (re-exported)

#: (block, reader label). The folder of each block is found by ``block_roots``.
BLOCKS = [
    ("CR", "GuacaMol, main runs"),
    ("CRPUB", "GuacaMol, published recipes"),
    ("CRRD", "GuacaMol, full-dimensional CMA-ES and BAxUS"),
    ("CRCTL", "GuacaMol, control runs"),
    ("CRCPU", "GuacaMol, the two sphere-manifold GPs"),
    ("LASSO_CR", "LassoBench rerun"),
    ("LASSO_PUB", "LassoBench, published methods"),
    ("VERIFY", "Verification of the published recipes"),
]
STORED = "STORED"
#: machines whose caption entry is not the hardware string of HOSTS (crlib/baselines.py; the cards print HOSTS and
#: stay as they are). Server B (kajman) has the CPUs and GPUs of server A (nvidia-smi -L and lscpu, 2026-10-24);
#: RUN_RULES R-2: the CPU-only arms and LassoBench ran on its CPUs with the GPUs hidden.
MACHINE_CAPTION = {"B": "the same CPUs and GPUs as A"}
#: stored folders under the inputs root that a printed table reads (block, folder, row label): the codec comparison
#: (tab_crosscodec, the chembed_* and power_spherical_* arms of sources_cr.json) and NAS-Bench-101 (tab_crossdomain;
#: crlib/stored.py nas_analysis reads every file of the folder). Every result file of the folder is counted, as for
#: root G.
STORED_PRINTED = (("STORED_CHEMBED", "chembed", "Codec comparison: Chembed"),
                  ("STORED_POWER", "power_spherical", "Codec comparison: Power-Spherical VAE"),
                  ("STORED_NAS", "nasbench", "NAS-Bench-101"))
#: stored folders under the inputs root that no printed table reads (folders, reader name), named in the caption:
#: the March/April LassoBench files (only tab_lassobench_correspondence reads them in camera-ready mode), the
#: September p1 blocks (crlib/september.py, not built) and the oracle scores of the retrieval floor (crlib/floor.py,
#: not built)
NOT_COUNTED = ((("lassobench",), "the stored LassoBench runs"),
               (("p1_blocks", "p1_blocks_sel"), "the selection-rule runs"),
               (("e0_floor",), "the retrieval-floor scores"))


def block_roots(ctx: Ctx) -> dict:
    """{block: folder} of the blocks whose folder exists. STORED only if a stored GuacaMol arm is loaded."""
    roots = dict(getattr(ctx.ds.config, "roots", {}) or {})
    inputs = ctx.extra.get("inputs_root") or ""
    cand = {"CR": roots.get("CR"), "CRPUB": roots.get("CRPUB"), "CRRD": roots.get("CRRD"), "CRCTL": roots.get("CRCTL"),
            "CRCPU": roots.get("CRCPU"), "LASSO_CR": ctx.extra.get("lasso_cr_dir"),
            "LASSO_PUB": ctx.extra.get("lasso_pub_dir"),
            "VERIFY": os.path.join(inputs, "verification_runs") if inputs else None}
    if any(not a.absent and a.spec.kind == "stored" and not a.spec.namespace for a in ctx.ds.arms.values()):
        cand[STORED] = roots.get("G")
    for b, d, _ in STORED_PRINTED:
        cand[b] = os.path.join(inputs, d) if inputs else None
    return {k: v for k, v in cand.items() if v and os.path.isdir(v)}


def tex_path(name: str) -> str:
    """A folder name in \\texttt (underscores escaped)."""
    return name.replace("_", "\\_")


def _join(names: list) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _stored_names(stored_dir: str, folders: list) -> list:
    """Caption parts naming the stored folders of the submitted-version rows and what each holds."""
    out = [f"\\texttt{{{tex_path(stored_dir)}}} (GuacaMol, the rows without a benchmark name)"] if stored_dir else []
    codec = [d for d in folders if d in ("chembed", "power_spherical")]
    if codec:
        out.append(" and ".join(f"\\texttt{{{tex_path(d)}}}" for d in codec)
                   + " (codec comparison, Table~\\ref{tab:crosscodec_means})")
    if "nasbench" in folders:
        out.append("\\texttt{nasbench} (NAS-Bench-101, Table~\\ref{tab:crossdomain})")
    return out


def _short(host) -> str | None:
    return host.split(".")[0] if host else None


def machine_letter(host) -> str:
    """'A', 'B', 'C' (crlib/baselines.py HOSTS) or 'not recorded' / 'unknown'."""
    if host is None:
        return "not recorded"
    h = HOSTS.get(_short(host))
    mo = re.match(r"server ([A-Z])", h or "")
    return mo.group(1) if mo else "unknown"


def _hours(times) -> float:
    return sum(times) / 3600.0


def _fmt1(x: float) -> str:
    """One decimal, thousands separated as in the other tables (4{,}184.8)."""
    return f"{x:,.1f}".replace(",", "{,}")


def _fmt_h(h: float) -> str:
    return _fmt1(h)


def _fmt_s(times) -> str:
    return _fmt1(statistics.median(times)) if times else "--"


def scans(ctx: Ctx) -> dict:
    """{block: scan} (cached for the run: every variant prints the same table)."""
    store = ctx.cache.setdefault("compute_scans", {})
    roots = block_roots(ctx)
    key = tuple(sorted(roots.items()))
    if key not in store:
        store[key] = {b: scan(r) for b, r in roots.items()}
    return store[key]


def tab_compute(ctx: Ctx) -> Table | None:
    table = "tab_compute"
    S = scans(ctx)
    if not any(S[b]["timed"] for b in S):
        ctx.skip(table, "run blocks", "no result file with a recorded run time (table omitted)")
        return None
    hdr = ["Runs", "Machine", stack("Result", "files"), stack("With a", "time"), stack("Process-", "hours"),
           stack("Median s", "per run")]
    t = Table("compute", "tab:compute", "", ">{\\raggedright\\arraybackslash}Xlrrrr", hdr, size="\\footnotesize",
              tabcolsep="4pt")
    by_machine: dict = {}
    unknown = []
    t.group("Runs of this version")
    for b, lab in BLOCKS:
        if b not in S:
            ctx.skip(table, b, "folder absent (row omitted)")
            continue
        sc = S[b]
        for host, e in sorted(S[b]["hosts"].items(), key=lambda kv: (machine_letter(kv[0]), str(kv[0]))):
            m = machine_letter(host)
            if m == "unknown":
                unknown.append(f"{b}: {host}")
            by_machine.setdefault(m, []).extend(e["times"])
            t.add([lab + ("" if len(sc["hosts"]) == 1 else f" ({m})"), m, N.intfmt(e["files"]),
                   N.intfmt(len(e["times"])), _fmt_h(_hours(e["times"])), _fmt_s(e["times"])])
    stored_untimed = 0
    printed = [(b, d, lab) for b, d, lab in STORED_PRINTED if b in S]
    if STORED in S or printed:
        t.group("Runs of the submitted version (results bundle)")
    if STORED in S:
        sc = S[STORED]
        # one row per recorded host over the files with a time, and one row for the files without a time
        for host, e in sorted(sc["hosts"].items(), key=lambda kv: (kv[0] is None, machine_letter(kv[0]))):
            m = machine_letter(host)
            if m == "unknown":
                unknown.append(f"{STORED}: {host}")
            stored_untimed += e["files"] - len(e["times"])
            if not e["times"]:
                continue
            what = "Host not recorded" if host is None else f"Recorded on server {m}"
            t.add([what, m if host is not None else "--", N.intfmt(len(e["times"])), N.intfmt(len(e["times"])),
                   _fmt_h(_hours(e["times"])), _fmt_s(e["times"])])
        if stored_untimed:
            t.add(["No time recorded", "--", N.intfmt(stored_untimed), "0", "--", "--"])
    for b, _, lab in printed:
        # one row per folder: its machine(s), every result file, those with a time, their hours and median
        sc = S[b]
        times = [x for e in sc["hosts"].values() for x in e["times"]]
        for h in sc["hosts"]:
            if machine_letter(h) == "unknown":
                unknown.append(f"{b}: {h}")
        ms = sorted({machine_letter(h) for h in sc["hosts"]} & {"A", "B", "C"})
        stored_untimed += sc["files"] - len(times)
        t.add([lab, ", ".join(ms) or "--", N.intfmt(sc["files"]), N.intfmt(len(times)), _fmt_h(_hours(times)),
               _fmt_s(times)])
    # ---- totals per machine
    def block_machine(b):
        hs = list(S[b]["hosts"]) if b in S else []
        ms = {machine_letter(h) for h in hs}
        return ms.pop() if len(ms) == 1 else None

    gpu_m, cpu_m = block_machine("CR"), block_machine("CRCPU")
    t.group("Totals")
    totals = {}
    for m in sorted(by_machine):
        times = by_machine[m]
        totals[m] = times
        files = sum(e["files"] for b, _ in BLOCKS if b in S for h, e in S[b]["hosts"].items() if machine_letter(h) == m)
        t.add([f"Runs of this version on server {m}" if len(m) == 1 else f"Runs of this version, machine {m}", m,
               N.intfmt(files), N.intfmt(len(times)), _fmt_h(_hours(times)), _fmt_s(times)])
    stored_blocks = ([STORED] if STORED in S else []) + [b for b, _, _ in printed]
    if STORED in S:
        times = [x for e in S[STORED]["hosts"].values() for x in e["times"]]
        t.add(["Runs of the submitted version" + (", GuacaMol" if printed else ""), "",
               N.intfmt(S[STORED]["files"]), N.intfmt(len(times)), _fmt_h(_hours(times)), _fmt_s(times)])
    if printed:
        times = [x for b in stored_blocks for e in S[b]["hosts"].values() for x in e["times"]]
        t.add(["Runs of the submitted version, all folders above", "",
               N.intfmt(sum(S[b]["files"] for b in stored_blocks)), N.intfmt(len(times)), _fmt_h(_hours(times)),
               _fmt_s(times)])
    if unknown:
        msg = f"WARNING: {table}: hosts not in crlib/baselines.py HOSTS: {unknown}"
        ctx.cache.setdefault("loud_warnings", []).append(msg)

    # ---- macros (the totals of the checklist), rounded to whole hours
    def fs_of(blocks):
        rows = []
        for b in blocks:
            if b in S:
                rows += S[b]["rows"]
        return fileset_from_rows(rows)

    def reg(name, hours, blocks, rule):
        ctx.num(name, hours, N.intfmt(round(hours)), kind="count", table=table, fileset=fs_of(blocks), rule=rule)

    this = [b for b, _ in BLOCKS if b in S]
    if gpu_m is not None and gpu_m in totals:
        bl = [b for b in this if block_machine(b) == gpu_m]
        reg("crComputeGpuServerHours", _hours(totals[gpu_m]), bl,
            f"process-hours of the runs of this version on server {gpu_m} (the machine of block CR): sum of the "
            "per-run wall-clock times recorded in the result files (crlib/compute.py run_time), rounded to hours")
    if cpu_m is not None and cpu_m in totals:
        bl = [b for b in this if block_machine(b) == cpu_m]
        reg("crComputeCpuServerHours", _hours(totals[cpu_m]), bl,
            f"process-hours of the runs of this version on server {cpu_m} (the machine of block CRCPU), rounded")
    if "VERIFY" in S and S["VERIFY"]["timed"]:
        reg("crComputeVerifyHours", _hours([x for e in S["VERIFY"]["hosts"].values() for x in e["times"]]),
            ["VERIFY"], "process-hours of the verification runs (sum of their per-evaluation times), rounded")
    if STORED in S and S[STORED]["timed"]:
        reg("crComputeStoredHours", _hours([x for e in S[STORED]["hosts"].values() for x in e["times"]]),
            [STORED], "process-hours of every result file (*.json) under root G that records a run time (with the "
                      "results bundle: its stored GuacaMol result files, stored_guacamol), rounded")
    if stored_blocks and any(S[b]["timed"] for b in stored_blocks):
        reg("crComputeStoredPrintedHours",
            _hours([x for b in stored_blocks for e in S[b]["hosts"].values() for x in e["times"]]), stored_blocks,
            "process-hours of every result file (*.json) that records a run time in the stored folders that printed "
            "tables read: " + ", ".join(([f"root G ({os.path.basename(os.path.normpath(block_roots(ctx)[STORED]))})"]
                                         if STORED in S else [])
                                        + [f"<inputs-root>/{d}" for _, d, _ in printed]) + ", rounded")
    letters = sorted({machine_letter(h) for sc in S.values() for h in sc["hosts"]} & {"A", "B", "C"})
    hw = "; ".join(f"{L} = {MACHINE_CAPTION.get(L) or HOSTS[k][len('server ' + L + ' ('):-1]}" for L in letters
                   for k in HOSTS if HOSTS[k].startswith(f"server {L} "))
    if "B" in letters:
        hw += (". The runs of this version on B used only its CPUs, with the GPUs hidden; of the submitted version, "
               "no result file with a run time records the device")
    inputs = ctx.extra.get("inputs_root") or ""
    skipped = [(name, [d for d in dirs if os.path.isdir(os.path.join(inputs, d))]) for dirs, name in NOT_COUNTED
               if inputs and any(os.path.isdir(os.path.join(inputs, d)) for d in dirs)]
    skipped_txt = [f"{name} (" + ", ".join("\\texttt{" + tex_path(d) + "}" for d in dirs) + ")"
                   for name, dirs in skipped]
    inputs_dir = tex_path(os.path.basename(os.path.normpath(inputs))) if inputs else ""
    stored_dir = os.path.basename(os.path.normpath(block_roots(ctx)[STORED])) if STORED in S else ""
    t.caption = (
        "\\textbf{Compute of the project, from the run times recorded in the result files.} Process-hours are "
        "sums of the wall-clock time that each result file records for its run; runs shared their machine with "
        "other runs, so these are not exclusive GPU or CPU hours. The verification runs record per-evaluation times, "
        "which are summed. Median = median time of one run. The machine of a run is the host its result file "
        "records; the LassoBench rerun records its host once, in the environment record of the rerun. Machines: "
        + hw + ". "
        + (("The runs of the submitted version are every result file of the stored folders that printed tables "
            "read: " + "; ".join(_stored_names(stored_dir if STORED in S else "", [d for _, d, _ in printed]))
            + ". " + (f"{N.intfmt(stored_untimed)} of these files record no time and are not counted. "
                      if stored_untimed else "")) if (STORED in S or printed) else "")
        + (f"In \\texttt{{{inputs_dir}/}}, " + _join(skipped_txt) + " are not counted, as no printed table reads them. "
           if skipped and (STORED in S or printed) else "")
        + "The totals of the checklist are rounded to whole hours.")
    return t
