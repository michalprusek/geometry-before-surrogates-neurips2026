"""Synthetic inputs of the GEN v5 tests: a verdict file in the layout of verdicts.json (the real file's sha256 is
pinned in crlib/published.py, so tests point the pin at the synthetic file), and a LassoBench published-method set
in the layout of kajman's pub_methods group with its queue job lists."""
import hashlib
import json
import math
import os
import re

from crlib import lasso_cr, published, stored
from crlib.labels import AUTHOR_TOKENS

DIMS = dict(stored.LASSO_TASKS)

#: (recipe, n_init) -> (protocol pass, strict pass, fit-skipped steps) as in the summarize pass of 2026-10-19
REAL_LIKE = {("doumont", 30): (True, True, 0), ("hvarfner", 30): (True, False, 0), ("hvarfner", 41): (False, False, 0),
             ("mle_scaled_noise_learned", 40): (True, True, 82), ("msr_noise_learned", 40): (True, True, 27),
             ("xu", 20): (True, True, 0)}


def write_verdicts(path, verdicts=None, extra=(), commit=published.VERDICTS_TAG_COMMIT, monkeypatch=None,
                   summarizer_commit=None, summarizer_dirty=False):
    """Write a verdict file and (with monkeypatch) pin crlib/published.py to its sha256."""
    verdicts = dict(REAL_LIKE if verdicts is None else verdicts)
    rows = []
    for (recipe, n_init), (ok, strict, fb) in list(verdicts.items()) + list(extra):
        rows.append({"recipe": recipe, "benchmark": "lasso_dna", "n_init": n_init, "target_id": f"t_{recipe}",
                     "pass": ok, "pass_strict": strict, "seeds_used": list(range(10)), "fallbacks": fb,
                     "refusals": [], "commits": [commit], "gate": True,
                     "rows": [{"n": 100.0, "gate": True, "pass": True, "pass_strict": True},
                              {"n": 500.0, "gate": True, "pass": ok, "pass_strict": strict}]})
    d = {"verdicts": rows, "targets_sha256": published.TARGETS_SHA256,
         "targets_sha256_expected": published.TARGETS_SHA256,
         "summarizer_provenance": {"git_commit_full": summarizer_commit or commit, "git_dirty": summarizer_dirty}}
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    if monkeypatch is not None:
        monkeypatch.setattr(published, "VERDICTS_SHA256", sha)
    return sha


def cold_scores(task, seed) -> list:
    """The five cold-start losses of (task, seed): the same in the rerun and the published set (one objective, the
    shared cold start of every method in lasso_cr.COLD_SHARED; review S7)."""
    return [1.0 + 0.01 * i + 0.001 * seed + 0.1 * list(DIMS).index(task) for i in range(lasso_cr.N_COLD)]


class PubSet:
    """<root>/pub_methods/{results,status,cmds,manifest.jsonl} and <root>_queue/queue/jobs_phase{1,2}.txt."""

    def __init__(self, root):
        self.root = str(root)
        self.g = os.path.join(self.root, lasso_cr.PUB_SET)
        self.q = os.path.join(self.root + "_queue", "queue")
        for d in ("results", "status", "cmds"):
            os.makedirs(os.path.join(self.g, d), exist_ok=True)
        os.makedirs(self.q, exist_ok=True)

    def job(self, label, method, task, seed, loss=1.0, *, rc=0, status_field="completed", prov=None, raw=None,
            error=None, timeout_s="3600"):
        path = os.path.join(self.g, "results", f"{method}_{task}_s{seed}.json")
        rf = {"method": method, "file": "/kajman/" + os.path.relpath(path, self.root), "exists": loss is not None}
        if loss is not None:
            if raw is None:
                d = {"best_score": loss if math.isfinite(loss) else float("inf"), "best_trace": [],
                     "all_scores": cold_scores(task, seed) + [0.0] * (25 if status_field == "completed" else 0),
                     "method": method, "task": task,
                     "D": DIMS[task], "seed": seed, "status": status_field, "surrogate_fallback_count": 0,
                     "provenance": prov if prov is not None else {"git_commit": "963ef04", "git_dirty": False}}
                if error:
                    d["error"] = error
                raw = json.dumps(d).encode()
            with open(path, "wb") as f:
                f.write(raw)
            rf["sha256"] = hashlib.sha256(raw).hexdigest()
        r = {"group": lasso_cr.PUB_SET, "label": label, "methods": [method], "task": task, "seed": seed, "rc": rc,
             "commit": lasso_cr.PUB_COMMIT, "tag": lasso_cr.PUB_TAG, "porcelain_lines_after": 0,
             "runner_script_sha256": lasso_cr.PUB_RUNNER_SHA256, "runner_script_is_committed_file": True,
             "timeout_s": timeout_s, "result_files": [rf]}
        with open(os.path.join(self.g, "manifest.jsonl"), "a") as f:
            f.write(json.dumps(r) + "\n")
        open(os.path.join(self.g, "status", f"{label}.rc"), "w").write(f"{rc} t0 t1\n")
        open(os.path.join(self.g, "cmds", f"{label}.cmd"), "w").write("env ... run_lassobench.py\n")

    def queue_line(self, f, label, method, task, seed, tmo="3600"):
        with open(os.path.join(self.q, f), "a") as fh:
            fh.write(f"bash /c/run_lasso.sh {lasso_cr.PUB_SET} {label} results /r/run_lassobench.py {method} {task} "
                     f"{seed} {tmo}\n")


def write_pub_design(root, pilot_rc=None, loss=lambda m, t, s: 1.0, skip=(), timeout=(), not_applicable=("xu_published",),
                     queue=True, frozen_raw=b"{not json at all", failed=()):
    """The whole published-method design as the queue ran it: pilots first (rc from ``pilot_rc``, default 0 except
    the learned Papenmeier variants and Linear BO, which exceeded the hour as on kajman), the five tasks, then
    phase 2 for each method whose pilot exited 0. ``skip``: (method, task, seed) without a record; ``timeout``:
    (method, task, seed) with rc 124; ``not_applicable``: methods undefined above D = 900 (status not_applicable,
    non-finite). The frozen-noise variants' result files hold ``frozen_raw`` (they must never be parsed)."""
    P = PubSet(root)
    rc = {m: 0 for m in lasso_cr.PUB_METHODS}
    rc.update({"papenmeier_published_learned": 124, "papenmeier_mle_scaled_published_learned": 124,
               "linear_bo_published": 124})
    rc.update(pilot_rc or {})

    def one(label, m, t, s, f):
        if queue:
            P.queue_line(f, label, m, t, s)
        if (m, t, s) in skip:
            return
        if (m, t, s) in timeout:
            P.job(label, m, t, s, None, rc=124)
            return
        if (m, t, s) in failed:                   # another exit code, no result file
            P.job(label, m, t, s, None, rc=1)
            return
        na = m in not_applicable and DIMS[t] > 900
        raw = frozen_raw if m in lasso_cr.PUB_UNREAD else None
        P.job(label, m, t, s, float("inf") if na else loss(m, t, s), status_field="not_applicable" if na else "completed",
              error="undefined" if na else None, raw=raw)

    for m in lasso_cr.PUB_METHODS:
        lab = f"pilot_{m}_leukemia_s42"
        if queue:
            P.queue_line("jobs_phase1.txt", lab, m, "leukemia", 42)
        if ("pilot", m) in skip:
            continue
        if rc[m] != 0:
            P.job(lab, m, "leukemia", 42, None, rc=rc[m])
        else:
            na = m in not_applicable
            P.job(lab, m, "leukemia", 42, float("inf") if na else loss(m, "leukemia", 42),
                  status_field="not_applicable" if na else "completed", error="undefined" if na else None,
                  raw=frozen_raw if m in lasso_cr.PUB_UNREAD else None)
    for s in lasso_cr.SEEDS:
        for t in lasso_cr.TASKS_LOW:
            for m in lasso_cr.PUB_METHODS:
                one(f"{m}_{t}_s{s}", m, t, s, "jobs_phase1.txt")
    for s in lasso_cr.SEEDS:
        if s == 42:
            continue
        for m in lasso_cr.PUB_METHODS:
            if rc[m] == 0:
                one(f"{m}_leukemia_s{s}", m, "leukemia", s, "jobs_phase2.txt")
    return P


# ---- printed labels (RUN_RULES 2026-10-19 second set (c), review B3) --------------------------------------------
_CARD_TITLE = re.compile(r"\\hspace\{0pt\}\\textbf\{(.*)\}(?: \(row written for this version.*\))?\} \\tabularnewline\*$")
_CODE = re.compile(r"[a-z0-9]\\?_[a-z0-9]")         # a code name such as hvarfner\_scaled\_gp


def printed_labels(outdir) -> list[tuple[str, str]]:
    """(file, label) of every row label of the printed tables (first cell of a data row) and every Appendix H card
    title (tab_baselines), from a generator output folder. Since RUN_RULES 2026-10-24 the cards and the run-status
    table are repository records (records/tab_baselines.tex, records/tab_arm_status.tex): their labels are listed
    under those names."""
    out = []
    rdir = os.path.join(outdir, "records")
    files = sorted(f for f in os.listdir(outdir) if f.startswith("tab_") and f.endswith(".tex"))
    files += sorted(os.path.join("records", f) for f in (os.listdir(rdir) if os.path.isdir(rdir) else [])
                    if f in ("tab_baselines.tex", "tab_arm_status.tex"))
    for f in files:
        prev = ""
        for line in open(os.path.join(outdir, f)):
            line = line.rstrip("\n")
            header, prev = prev.strip() == "\\toprule", line
            if header:                  # the header row (repeated on every page of a longtable) is not a label
                continue
            if os.path.basename(f) == "tab_baselines.tex":
                mo = _CARD_TITLE.search(line)
                if mo:
                    out.append((f, mo.group(1)))
                continue
            if line.endswith("\\\\") and "&" in line and not line.lstrip().startswith(("%", "\\multicolumn")):
                out.append((f, line.split("&")[0].strip()))
    return out


def label_problems(labels_, verified_names) -> list[str]:
    """A label may carry an author's or method's name only inside a verified recipe's printed name, and never a
    code name or \\texttt."""
    bad = []
    for f, lab in labels_:
        rest = lab
        for nm in verified_names:
            rest = rest.replace(nm, "")
        toks = [t for t in AUTHOR_TOKENS if re.search(r"(?<![A-Za-z])" + re.escape(t) + r"(?![A-Za-z])", rest)]
        if toks or "\\texttt" in lab or _CODE.search(re.sub(r"\$[^$]*\$", "", rest)):
            bad.append(f"{f}: {lab!r} ({toks or 'code name'})")
    return bad
