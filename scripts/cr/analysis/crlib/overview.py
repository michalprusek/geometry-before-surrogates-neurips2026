"""Appendix H: the overview of every printed method (``tab_baselines_overview``, label ``tab:baselines``).

One row per printed method; cards that describe the same recipe (a method and its variants, a recipe on GuacaMol
and on LassoBench, the GP constructor shared by several supports and maximizers) are merged into one row. The full
card of every method stays in the repository (``records/BASELINE_CARDS.md`` of the generator, committed as
``docs/baselines/``).

Columns and where they come from:

  column                        source
  ----------------------------  ------------------------------------------------------------------------------------
  Method                        computed: the printed label of the row's card (Table 1 form; published recipes carry
                                the authors' name only if verified), or the written ``label`` of a merged row
  Implementation                written (``appendixH_overview.csv``)
  Surrogate                     written
  Acquisition and maximizer     written
  Support and trust region      written
  Published result reproduced   computed from the verdict file (crlib/published.py): yes / no (not verified) for the
  (header "Repr.")              published recipes, n/a for the methods of this paper and the adapted rows, -- for
                                the runs of the submitted version (not checked)

``appendixH_overview.csv`` (next to the facts table) has the columns ``row, group, cards, label, implementation,
surrogate, acquisition, support, source``. ``cards`` lists the facts rows of appendixH_facts.csv that the row merges
(``;``-separated; ``<facts row>@<arm>`` takes one arm of a card with several, e.g. the two runs of the Hvarfner
recipe). ``source`` lists the proofread entries ``<facts row>|<field>`` of appendixH_printed.csv that the written
cells condense. ``load_overview`` refuses the file unless

  * every facts row is covered exactly once (whole, or by all of its run arms through ``@`` entries), or is listed
    in ``OVERVIEW_OMITTED`` with a reason, and no row is both;
  * every written cell is free of the ``PRINTED_FORBIDDEN`` patterns and at most ``MAX_CELL`` characters long (a
    token counts as the text it prints);
  * every cited entry exists and belongs to a card of the row;
  * every number and every keyword of ``KEYWORDS`` in a written cell occurs in the cited proofread text.

Written cells are plain text, escaped like the cards; ``{r}`` prints the mean cold-start norm $\\bar r$, and a closed
list of plain notations (``MATH``) prints as mathematics. At build time every card that tab_baselines printed must be
in a row or in ``OVERVIEW_OMITTED``.
"""
from __future__ import annotations

import csv
import os
import re

from . import labels, published
from .baselines import (FACTS_CSV, PRINTED_CSV, PRINTED_FORBIDDEN, load_facts, load_printed, neutral_hosts,
                        reader_facing, tex_escape)
from .context import Ctx
from .latex import Table, stack, wrapp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OVERVIEW_CSV = os.path.join(HERE, "appendixH_overview.csv")
OVERVIEW_COLUMNS = ["row", "group", "cards", "label", "implementation", "surrogate", "acquisition", "support",
                    "source"]
WRITTEN = ("implementation", "surrogate", "acquisition", "support")
#: at most this many visible characters per written cell (the columns wrap; raised from 65 on 2026-10-29 for the
#: support cell "convex combinations of latents", RUN_RULES 2026-10-29 (4))
MAX_CELL = 85

#: overview groups, in print order, with their headings (raw LaTeX)
GROUPS = {
    "paper": "Methods of this paper (GuacaMol, LassoBench and NAS-Bench-101)",
    "adapted": "GP recipes adapted in this repository",
    "published": "Published recipes: the authors' released code (BAxUS: the BoTorch tutorial)",
    "inrepo": "Other baselines implemented in this repository",
    "submitted": "GuacaMol stored runs",
}
_NOT_LOADED = "not loaded in a camera-ready run (crlib/scope.py NOT_LOADED_ARMS): no printed table reads it"
#: facts rows without an overview row, with the reason (their cards stay in the repository record)
OVERVIEW_OMITTED = {
    "hvarfner_scaled_gp_norm_box": _NOT_LOADED, "hvarfner_scaled_gp_box": _NOT_LOADED,
    "hvarfner_scaled_gp_prior": _NOT_LOADED, "papenmeier_msr_raasp": _NOT_LOADED,
    "papenmeier_msr_norm_box": _NOT_LOADED, "papenmeier_msr_box": _NOT_LOADED, "xu_standard_gp_norm_box": _NOT_LOADED,
    "xu_standard_gp_box": _NOT_LOADED, "linear_bo_sphere": _NOT_LOADED,
    "hvarfner_scaled_gp_box_argmax_raw": _NOT_LOADED, "hvarfner_scaled_gp_boxdir_argmax": _NOT_LOADED,
    "random_boxdir": _NOT_LOADED, "random_box_faithful": _NOT_LOADED,
    "hvarfner_scaled_gp_sphere{05,15,20}": _NOT_LOADED, "random_search_prior": _NOT_LOADED,
    "hvarfner_scaled_gp_norm_box_argmax": _NOT_LOADED, "papenmeier_msr_published_norm_box": _NOT_LOADED,
    # RUN_RULES 2026-11-06 16:30: the printed shell x continuous cell is L-BFGS-B on the shell; the projected-Adam arm
    # stays loaded as a repository record (card, run status) and is read by no printed table
    "hvarfner_scaled_gp_sphere_continuous": "shell x continuous cell with projected Adam, read by no printed table since "
                                            "2026-11-06 (L-BFGS-B on both supports); card and run status kept",
    "nasbench:cmaes_full": "NAS-Bench-101 run of the submitted version that no printed table or sentence reads",
    "nasbench:turbo": "NAS-Bench-101 run of the submitted version that no printed table or sentence reads",
    "nasbench:gp_bo": "NAS-Bench-101 run of the submitted version that no printed table or sentence reads",
    "cowboys": "not printed: the COWBOYS row of the submitted Table 1 was dropped; Appendix H gives the reason",
    # authors' decision of 2026-11-02: Table 1 and the trajectory table no longer print these stored runs (LOL-BO and
    # InvBO update the VAE, which the paper keeps fixed; CMA-ES, TuRBO and BAxUS print runs of this version instead);
    # their cards stay in the repository record and their rows in the full Table 1
    "lolbo": "stored run no longer printed in Table 1 (authors' decision of 2026-11-02); card and full-table row kept",
    "invbo": "stored run no longer printed in Table 1 (authors' decision of 2026-11-02); card and full-table row kept",
    "turbo": "stored run replaced in Table 1 and the trajectory table by turbo_published (2026-11-02); card kept",
    "baxus": "stored run replaced in Table 1 and the trajectory table by baxus_cr (2026-11-02); card kept",
    "cma_es (full-dimensional)": "stored run replaced in Table 1 by cmaes_full_rd_pop5 (2026-11-02); card kept",
    # RUN_RULES 2026-11-06 21:30: the re-implementation of TuRBO-1 leaves Table 9, Figure 1 and this overview; the
    # published TuRBO run is printed; the card and the full Table 1 keep the re-implementation
    "turbo_box_faithful": "TuRBO-1 re-implementation, no longer printed (2026-11-06; the published TuRBO run is); card kept",
}
#: GuacaMol arms of an overview row that run a published recipe's port without being an arm of crlib/published.py
#: RECIPES (whose GuacaMol arms form the published-recipe block): the BAxUS rerun of 2026-11-02 runs the BAxUS port of
#: that block; its Repr. cell is the verification status of that recipe
OVERVIEW_RECIPE = {"baxus_cr": "baxus"}
#: plain notations of the written cells that print as mathematics (closed list; longest first)
MATH = [("[-1, 1]^D", "$[-1,1]^D$"), ("[-8, 8]^256", "$[-8,8]^{256}$"), ("[-5, 5]^256", "$[-5,5]^{256}$"),
        ("[-4, 4]^16", "$[-4,4]^{16}$"), ("[p01, p99]", "$[p_{01},p_{99}]$"), ("sqrt(D)/10", "$\\sqrt{D}/10$"),
        ("0.1 sqrt(D)", "$0.1\\sqrt{D}$"), ("sqrt(D)", "$\\sqrt{D}$"), ("sqrt(2)", "$\\sqrt{2}$"),
        ("sqrt(3)", "$\\sqrt{3}$"), ("S^255", "$S^{255}$"), ("R^256", "$\\mathbb{R}^{256}$"), ("ln D", "$\\ln D$"),
        ("N(m, C)", "$N(\\hat\\mu,\\hat\\Sigma)$"), ("x0 = 0", "$x_0=0$"), ("{r}", "$\\bar r$")]
TOKENS = {"{r}": "r"}           # visible text of a token (cell length)
#: keyword (as written in a cell) -> spellings, one of which the cited proofread text must contain
KEYWORDS = {
    "Matern": ("Matern",), "RBF": ("RBF",), "Tanimoto": ("Tanimoto",), "linear kernel": ("linear kernel",),
    "Gaussian kernel": ("Gaussian kernel",), "Riemannian": ("Riemannian",), "SAAS": ("SAAS",), "NUTS": ("NUTS",),
    "PPGPR": ("PPGPR",), "ExactGP": ("ExactGP",), "BoTorch": ("BoTorch",), "gpytorch": ("gpytorch",),
    "pycma": ("pycma",), "Ax": ("Ax",), "qLogEI": ("qLogEI", "qLogExpectedImprovement"),
    "LogEI": ("LogEI", "LogExpectedImprovement"), "qLogNEI": ("qLogNEI", "qLogNoisyExpectedImprovement"),
    "EI": ("EI", "ExpectedImprovement"), "UCB": ("UCB", "UpperConfidenceBound"), "Thompson": ("Thompson",),
    "L-BFGS-B": ("L-BFGS-B",), "Adam": ("Adam",), "argmax": ("argmax",), "RAASP": ("RAASP",), "Sobol": ("Sobol",),
    "CMA-ES": ("CMA-ES",), "trust region": ("trust region",), "TuRBO-1": ("TuRBO-1",), "BAxUS": ("BAxUS",),
    "count-sketch": ("count-sketch",), "MCMC": ("MCMC",), "Morgan": ("Morgan",), "LogNormal": ("LogNormal",),
    "Gamma": ("Gamma",), "Uniform": ("Uniform",), "geodesic": ("geodesic",), "QR": ("QR",),
    "MAP": ("maximum a posteriori",), "ARD": ("one lengthscale per dimension",), "DSP": ("dimension-scaled",),
    "PCA": ("principal components",), "learned noise": ("noise learned", "learned noise"),
    "pymanopt": ("pymanopt",), "embedding": ("embedding",), "fingerprints": ("fingerprints",),
    "outputscale": ("outputscale",), "no priors": ("no priors",), "nu": ("nu",), "potential-aware": ("potential-aware",),
}
NUM_RX = re.compile(r"(?<![A-Za-z0-9_.])\d+(?:,\d{3})*(?:\.\d+)?")
_TOKEN_RX = re.compile(r"\{[a-z_]+\}")


class OverviewError(RuntimeError):
    """appendixH_overview.csv breaks one of the rules of the module docstring."""


def visible_length(text: str) -> int:
    for tok, vis in TOKENS.items():
        text = text.replace(tok, vis)
    return len(text)


def numbers_of(text: str) -> set:
    return {float(m.replace(",", "")) for m in NUM_RX.findall(_TOKEN_RX.sub(" ", text))}


def _has(word: str, text: str, ignore_case: bool, plural: bool = False) -> bool:
    tail = r"s?(?![A-Za-z])" if plural else r"(?![A-Za-z])"
    return re.search(r"(?<![A-Za-z])" + re.escape(word) + tail, text, re.I if ignore_case else 0) is not None


def keywords_of(text: str) -> list[str]:
    return [k for k in KEYWORDS if _has(k, text, ignore_case=False)]


def unsupported(cell: str, source_text: str) -> list[str]:
    """Numbers and keywords of a written cell that the cited proofread text does not contain."""
    miss = [f"number {x:g}" for x in sorted(numbers_of(cell) - numbers_of(source_text))]
    miss += [f"keyword {k!r}" for k in keywords_of(cell)
             if not any(_has(s, source_text, ignore_case=True, plural=True) for s in KEYWORDS[k])]
    return miss


def _card_and_arm(entry: str) -> tuple[str, str | None]:
    card, _, arm = entry.partition("@")
    return card, (arm or None)


def load_overview(path: str = OVERVIEW_CSV, facts: list[dict] | None = None, printed: dict | None = None) -> list[dict]:
    """The rows of the overview as dicts (+ '_cards': [(facts row, arm or None)], '_source': [(row, field)]).
    Raises OverviewError on any rule of the module docstring."""
    facts = facts if facts is not None else load_facts(FACTS_CSV)
    printed = printed if printed is not None else load_printed(PRINTED_CSV, facts)
    with open(path, newline="", encoding="utf-8") as f:
        data = list(csv.reader(f))
    if not data or data[0] != OVERVIEW_COLUMNS:
        raise OverviewError(f"{path}: header must be {OVERVIEW_COLUMNS}")
    by_arm = {r["arm"]: r for r in facts}
    rows, seen_ids = [], set()
    whole: dict = {}
    split: dict = {}
    for i, r in enumerate(data[1:], start=2):
        if len(r) != len(OVERVIEW_COLUMNS):
            raise OverviewError(f"{path}:{i}: {len(r)} cells")
        d = dict(zip(OVERVIEW_COLUMNS, r))
        if not d["row"] or d["row"] in seen_ids:
            raise OverviewError(f"{path}:{i}: empty or duplicate row id {d['row']!r}")
        seen_ids.add(d["row"])
        if d["group"] not in GROUPS:
            raise OverviewError(f"{path}:{i}: unknown group {d['group']!r}")
        cards = [_card_and_arm(c) for c in d["cards"].split(";") if c]
        if not cards:
            raise OverviewError(f"{path}:{i}: row {d['row']} merges no card")
        for card, arm in cards:
            if card not in by_arm:
                raise OverviewError(f"{path}:{i}: unknown facts row {card!r}")
            if arm is None:
                if card in whole or card in split:
                    raise OverviewError(f"{path}:{i}: facts row {card!r} is covered twice")
                whole[card] = d["row"]
            else:
                if card in whole:
                    raise OverviewError(f"{path}:{i}: facts row {card!r} is covered twice")
                if arm not in by_arm[card]["_run"]:
                    raise OverviewError(f"{path}:{i}: {arm!r} is not a run arm of facts row {card!r}")
                if arm in split.setdefault(card, {}):
                    raise OverviewError(f"{path}:{i}: arm {arm!r} of {card!r} is covered twice")
                split[card][arm] = d["row"]
        src = []
        for e in [x for x in d["source"].split(";") if x]:
            card, sep, field = e.partition("|")
            if not sep or (card, field) not in printed:
                raise OverviewError(f"{path}:{i}: cited entry {e!r} is not in appendixH_printed.csv")
            if card not in {c for c, _ in cards}:
                raise OverviewError(f"{path}:{i}: cited entry {e!r} is not a card of row {d['row']}")
            src.append((card, field))
        source_text = "\n".join(printed[k]["text"] for k in src)
        for col in WRITTEN + ("label",):
            cell = d[col]
            if col != "label" and not cell.strip():
                raise OverviewError(f"{path}:{i}: row {d['row']}: empty {col}")
            if PRINTED_FORBIDDEN.search(cell):
                raise OverviewError(f"{path}:{i}: row {d['row']}/{col}: forbidden text "
                                    f"{PRINTED_FORBIDDEN.search(cell).group(0)!r}")
            if col == "label":
                continue
            if visible_length(cell) > MAX_CELL:
                raise OverviewError(f"{path}:{i}: row {d['row']}/{col}: {visible_length(cell)} characters "
                                    f"(at most {MAX_CELL})")
            if _TOKEN_RX.sub("", cell).count("{") or set(_TOKEN_RX.findall(cell)) - set(TOKENS):
                raise OverviewError(f"{path}:{i}: row {d['row']}/{col}: unknown token in {cell!r}")
            miss = unsupported(cell, source_text)
            if miss:
                raise OverviewError(f"{path}:{i}: row {d['row']}/{col}: {', '.join(miss)} not in the cited text "
                                    f"({d['source']})")
        d["_cards"], d["_source"] = cards, src
        rows.append(d)
    for card, arms in split.items():
        if set(arms) != set(by_arm[card]["_run"]):
            raise OverviewError(f"{path}: facts row {card!r} is split by arm, but its arms "
                                f"{sorted(set(by_arm[card]['_run']) - set(arms))} have no row")
    covered = set(whole) | set(split)
    both = covered & set(OVERVIEW_OMITTED)
    if both:
        raise OverviewError(f"{path}: facts rows both covered and omitted: {sorted(both)}")
    missing = set(by_arm) - covered - set(OVERVIEW_OMITTED)
    if missing:
        raise OverviewError(f"{path}: facts rows neither covered nor omitted: {sorted(missing)}")
    unknown = set(OVERVIEW_OMITTED) - set(by_arm)
    if unknown:
        raise OverviewError(f"OVERVIEW_OMITTED names unknown facts rows {sorted(unknown)}")
    order = list(GROUPS)
    if [order.index(r["group"]) for r in rows] != sorted(order.index(r["group"]) for r in rows):
        raise OverviewError(f"{path}: rows are not in the group order {order}")
    return rows


def render_cell(text: str) -> str:
    """LaTeX of a written cell: escaped as the cards are, with the notations of ``MATH`` set as mathematics."""
    keys = {}
    for i, (plain, _) in enumerate(MATH):
        mark = f"\x00{i}\x00"
        if plain in text:
            text = text.replace(plain, mark)
            keys[mark] = MATH[i][1]
    out = tex_escape(text)
    for mark, tex in keys.items():
        out = out.replace(mark, tex)
    return re.sub(r"Matern(?![A-Za-z])", lambda _: "Mat\\'ern", out)       # not inside a name (MaternGaBO)


def overview_label(ctx: Ctx, arm: str) -> str:
    """Label of a GuacaMol arm in the Table 1 form (the support is a column of its own)."""
    r = published.recipe_of(arm)
    if r == "hvarfner" and arm != published.hvarfner_arm(ctx):
        return labels.label(arm) + " (the other run of the same recipe)"
    if r is not None:
        return published.name_or_label(ctx, arm)
    if arm in labels.DESCRIPTIVE and not any(arm.startswith(p) for p, _ in labels.STORED_PREFIX):
        return labels.label(arm)
    return published.printed_label(ctx, arm)


def _recipe(card: str, arms: list[str]) -> str | None:
    if card.startswith("lassobench_pub:"):
        return published.recipe_of(card.partition(":")[2], "lasso")
    rs = {published.recipe_of(a) or OVERVIEW_RECIPE.get(a) for a in arms} - {None}
    if len(rs) > 1:
        raise OverviewError(f"card {card} mixes recipes {sorted(rs)}")
    return rs.pop() if rs else None


def reproduced_cell(ctx: Ctx, group: str, recipes: set, arms: list | None = None) -> str:
    """The verification status of the row (the words of Tables 14 and 15, crlib/published.py): a row whose GuacaMol
    arms are all variants that were not verified themselves (the all-observations Hvarfner run) says 'no'."""
    if len(recipes) > 1:
        raise OverviewError(f"an overview row mixes recipes {sorted(recipes)}")
    if recipes:
        r = next(iter(recipes))
        guac = [a for a in (arms or []) if published.recipe_of(a) == r]
        if guac and all(published.verification_status(ctx, a) == published.STATUS_NO for a in guac):
            return published.STATUS_NO
        return published.recipe_status(ctx, r)
    return {"paper": "n/a", "adapted": "n/a", "inrepo": "n/a", "submitted": "--"}[group]


def _short(label: str) -> str:
    """The first phrase of a label (caption mentions)."""
    return label.split(",")[0].split(" (")[0]


def tab_baselines_overview(ctx: Ctx) -> Table | None:
    table = "tab_baselines_overview"
    cards = ctx.extra.get("baseline_cards")
    if not cards or not os.path.exists(OVERVIEW_CSV):
        ctx.skip(table, "baseline cards", "tab_baselines printed no card or appendixH_overview.csv is absent")
        return None
    facts = load_facts(FACTS_CSV)
    printed = load_printed(PRINTED_CSV, facts)
    rows = load_overview(OVERVIEW_CSV, facts, printed)
    by_card = {c["row"]: c for c in cards}
    in_rows = {card for r in rows for card, _ in r["_cards"]}
    stray = sorted(set(by_card) - in_rows - set(OVERVIEW_OMITTED))
    if stray:
        raise OverviewError(f"printed cards without an overview row: {stray}")
    hdr = ["Method", "Implementation", "Surrogate", "Acquisition, maximizer", "Support, trust region", "Repr."]
    colspec = ("@{}" + wrapp("2.7cm") + wrapp("2.0cm") + wrapp("2.4cm") + wrapp("2.4cm") + wrapp("2.4cm")
               + wrapp("1.2cm") + "@{}")
    t = Table("baselines_overview", "tab:baselines", "", colspec, hdr, size="\\scriptsize", long=True,
              tabcolsep="2pt")
    t.row_ids = []
    cur = None
    for r in rows:
        present_cards = [(c, a) for c, a in r["_cards"] if c in by_card]
        arms = []
        for c, a in present_cards:
            pa = by_card[c]["present"]
            arms += [x for x in pa if a is None or x == a]
        if not present_cards or (not arms and not any(by_card[c]["other"] for c, _ in present_cards)):
            ctx.skip(table, r["row"], "no card of the row is printed (row omitted)")
            continue
        if arms == ["saasbo"]:   # RUN_RULES 2026-11-02 (1): the 10-seed row is printed only in the full table
            ctx.skip(table, r["row"], "10-seed row, only in the full table of the repository")
            continue
        if r["group"] != cur:
            t.group(GROUPS[r["group"]])
            cur = r["group"]
        if r["label"]:
            lab = r["label"]
        elif arms:
            lab = overview_label(ctx, arms[0]) if ctx.mode == "cr" else by_card[present_cards[0][0]]["labels"][0]
        else:
            lab = by_card[present_cards[0][0]]["title"]
            lab = lab[len("\\textbf{"):-1] if lab.startswith("\\textbf{") and lab.endswith("}") else lab
        recipes = set()
        for c, a in present_cards:
            rc = _recipe(c, [x for x in by_card[c]["present"] if a is None or x == a])
            if rc:
                recipes.add(rc)
        cells = [reader_facing(neutral_hosts(render_cell(r[col]))) for col in WRITTEN]
        # RUN_RULES 2026-11-02 (3): the last column of this table states the verification
        t.add([lab.replace(" (not verified)", "").replace(" (submitted version)", "").replace(" (rerun)", "")] + cells + [reproduced_cell(ctx, r["group"], recipes, arms)])
        t.row_ids.append(r["row"])
    t.caption = _caption(ctx)
    return t


def _macro(name: str) -> str:
    return "\\" + name + "{}"


def _caption(ctx: Ctx) -> str:
    """Caption: what the columns mean, the fallback draws of the sphere Gaussian-kernel GP, and where the full cards
    are. The seeds and the machines are not printed (RUN_RULES 2026-10-29 (4)); the cards in the repository give them."""
    gabo = ""
    if "crBaselinesGaboFallbackSteps" in ctx.reg.records and "crBaselinesGaboCells" in ctx.reg.records:
        gabo = (f"The sphere Gaussian-kernel GP fell back to a uniform draw in {_macro('crBaselinesGaboFallbackSteps')} "
                f"proposals, in {_macro('crBaselinesGaboFallbackCells')} of its {_macro('crBaselinesGaboCells')} runs. ")
    return (
        "\\textbf{Overview of every printed method, one row per method and its variants.} Implementation: where the "
        "code comes from. Surrogate: model and fit. Acquisition, maximizer: how a proposal is chosen. Support, trust "
        "region: where the candidates are drawn ($\\bar r$: mean norm of the run's five cold-start latents). Repr. "
        "gives the result of the verification in Table~\\ref{tab:verification}. The entry ``yes'' means that a "
        "published result was reproduced. The entry ``yes, with allowance'' means that it was reproduced only with "
        "the allowances of the protocol. The entry ``no'' means not verified. Either the verification runs were not "
        "complete when the verification ended, or the row is a variant whose settings were not verified. The entry "
        "n/a means not applicable (methods of this paper and adapted recipes). A dash marks a stored run, not "
        "checked. Abbreviations: ARD = one lengthscale "
        "per input dimension. MAP = maximum a posteriori fit of the hyperparameters. RAASP = random axis-aligned "
        "subspace perturbations. qLogNEI = Monte Carlo log noisy expected improvement. UCB = upper confidence bound. "
        "NUTS = the No-U-Turn sampler. PPGPR = parametric predictive GP regression. PCA sphere = the sphere of radius "
        "$\\bar r$ in the span of the top 64 principal components of 3,000 encoded ZINC latents. Our adapted GP uses "
        "the DSP prior with location $\\sqrt{2}+0.5\\ln D$, as in the paper of Hvarfner et al. Their released "
        "configuration, which the published recipe follows, uses 1.4 in place of $\\sqrt{2}$. " + gabo
        + "The full card of every method and its run status are released with the code (\\texttt{docs/baselines/}).")
