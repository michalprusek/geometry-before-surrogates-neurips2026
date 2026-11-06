"""Full-dimensional CMA-ES in R^D (``cmaes_full_rd``, ``cmaes_full_rd_pop5``).

Current-code port of the deleted March baseline ``src/benchmark/baselines/cmaes_full.py``
(removed in 68ed3dc), recovered inside the camera-ready harness so that the only
difference against the other camera-ready rows is the algorithm itself -- not the
runner, the cold start or the decoder.  The port and its validation are the audit
of 2026-11-02 (``/data/prusek/cr_work/dagger_audit_2026-11-02/cmaes_shift``,
``out/audit_summary.json``): its population-20 variant reproduced the old code's
behaviour under the current decoder on 44 drawn cells.

Algorithm, exactly the old wrapper's:

* ``x0`` = the best cold-start latent, ``sigma0`` = 0.5, no bounds;
* pycma's default population for D = 256 (4 + 3 ln 256 = 20) for ``cmaes_full_rd``,
  and 5 -- the population GeoCMA uses -- for ``cmaes_full_rd_pop5``;
* deferred tell: one oracle call per step, ``tell`` after a full generation
  (the old wrapper's ask/tell pattern, same order of candidates);
* pycma options of the old code: verbose -9, maxiter 10000, tolx/tolfun 1e-12;
* CMA-ES minimises, so the fitness is ``-score``.

Deliberate deviations from the old wrapper, both inherited from the current harness
and shared with every other camera-ready arm:

1. no duplicate suppression -- the old wrapper replaced a repeated or undecodable
   molecule's fitness by 0.0 and did not count it in ``n_evaluated``; here every one
   of the 25 proposals is an evaluation, as for the other arms.  A non-finite score
   (failed decode) is fed to pycma as 0.0, which is the old wrapper's value for that
   case.
2. the black box is ``LatentBlackBox``: 80-token decoder cap, 30 s decode timeout,
   invalid -> NaN (recorded as -1.0 by ``run_benchmark``).

A failed step is never swallowed: a non-finite score goes to the fallback fields
(``surrogate_fallback_count`` / ``surrogate_fallback_reasons``), and an exception of pycma
ends the cell, which ``run_benchmark`` records in ``status`` and ``crash_info``.
"""
from __future__ import annotations

from typing import Any

import cma
import numpy as np

from src.spherevo.solver import StepByStepSolver

#: ``sigma0`` of the old wrapper.
CMAES_FULL_RD_SIGMA0 = 0.5
#: Population of the ``_pop5`` variant (GeoCMA's population).
CMAES_FULL_RD_POP5 = 5


class CMAESFullSolver(StepByStepSolver):
    """CMA-ES in the full ambient latent space R^D (no subspace, no sphere)."""

    def __init__(
        self,
        black_box: Any,
        x0: np.ndarray,
        y0: np.ndarray,
        *,
        seed: int,
        initial_sigma: float = CMAES_FULL_RD_SIGMA0,
        population_size: int | None = None,
    ) -> None:
        super().__init__(black_box, x0, y0)

        best_idx = int(np.argmax(self._y0.flatten()))
        x_start = self._x0[best_idx]

        opts: dict = {"seed": seed, "verbose": -9, "maxiter": 10000,
                      "tolx": 1e-12, "tolfun": 1e-12}
        if population_size is not None:
            opts["popsize"] = population_size
        self._cma = cma.CMAEvolutionStrategy(x_start.tolist(), initial_sigma, opts)

        self._pending: list | None = None
        self._pending_idx = 0
        self._fitnesses: list[float] = []
        self.popsize = int(self._cma.popsize)
        self.sigma0 = float(initial_sigma)
        self._n_nonfinite_told = 0
        self._fallback_count = 0
        self._fallback_reasons: list[str] = []
        self._generations_told = 0
        self.published_recipe_record: dict[str, Any] = {
            "recipe": "cmaes_full_rd",
            "domain": "R^D, unbounded (x0 = best cold-start latent)",
            "settings": {
                "code": "current-code port of src/benchmark/baselines/cmaes_full.py "
                        "(deleted in 68ed3dc); audit of 2026-11-02",
                "popsize": self.popsize,
                "popsize_source": ("pycma default for D (4 + 3 ln D)"
                                   if population_size is None
                                   else f"explicit population_size={population_size}"),
                "sigma0": self.sigma0,
                "x0": "best cold-start latent",
                "bounds": None,
                "cma_seed": int(seed),
                "cma_version": cma.__version__,
                "cma_options": {"verbose": -9, "maxiter": 10000,
                                "tolx": 1e-12, "tolfun": 1e-12},
                "tell": "deferred: one oracle call per step, tell after a full generation",
                "fitness": "-score (CMA-ES minimises); a non-finite score enters as 0.0, "
                           "the old wrapper's value for an unusable molecule",
                "duplicates": "every proposal is evaluated and counted (harness convention); "
                              "the old wrapper replaced it by 0.0 and did not count it",
            },
            "deviations_ours": [
                "no duplicate or undecodable suppression: a repeated or undecodable "
                "proposal spends one of the 25 proposals, as in every other arm",
                "black box = LatentBlackBox (80-token decoder cap, 30 s decode timeout)",
            ],
            "deviations_inherited": [
                "the cold start, budget and decoder of every camera-ready arm (R-1)",
            ],
            "last_generation": "a last incomplete generation is evaluated and never told; "
                               "20 of the 25 proposals at population 20 inform exactly one update",
            "generations_told": 0,
            "n_nonfinite_scores_told": 0,
            "steps": [],
        }

    def next_candidate(self) -> np.ndarray:
        if self._pending is None or self._pending_idx >= len(self._pending):
            if self._pending is not None:
                self._cma.tell(self._pending, self._fitnesses)
                self._generations_told += 1
                self.published_recipe_record["generations_told"] = self._generations_told
            self._pending = self._cma.ask()
            self._pending_idx = 0
            self._fitnesses = []
        z = self._pending[self._pending_idx]
        self._pending_idx += 1
        self.published_recipe_record["steps"].append({
            "iteration": int(self.iteration),
            "generation": int(self._generations_told),
            "index_in_generation": int(self._pending_idx - 1),
            "sigma": float(self._cma.sigma),
        })
        return np.asarray(z, dtype=np.float64).reshape(1, -1)

    def post_update(self, x: np.ndarray, y: np.ndarray) -> None:
        score = float(np.asarray(y).flatten()[0])
        # old wrapper: a molecule it could not use entered CMA-ES as 0.0
        if not np.isfinite(score):
            self._n_nonfinite_told += 1
            self.published_recipe_record["n_nonfinite_scores_told"] = self._n_nonfinite_told
            self.published_recipe_record["steps"][-1]["nonfinite_score"] = True
            self._fallback_count += 1
            self._fallback_reasons.append(
                f"iter {self.iteration}: non-finite score (invalid molecule) entered "
                f"CMA-ES as fitness 0.0")
        self._fitnesses.append(-score if np.isfinite(score) else 0.0)
