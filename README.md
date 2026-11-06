# Match the Geometry Before the Surrogate: Extreme Low-Budget Optimization in High Dimensions

Michal Průšek, Adam Novozámský, Filip Šroubek · NeurIPS 2026 · [paper on OpenReview](https://openreview.net/forum?id=2BB0oOmLs4) · contact: prusek@utia.cas.cz

In the latent space of a SELFIES VAE with a Gaussian prior, encoder means of drug-like molecules lie near a thin
spherical shell, while many high-dimensional Bayesian optimisers draw candidates from boxes. At 30 evaluations on 11
GuacaMol tasks, the paper studies two minimal optimisers that keep every candidate on the shell (GeoWalk, geodesic
local search without a surrogate; GeoCMA, CMA-ES in a lifted random subspace) and a factorial experiment crossing
candidate support with acquisition maximiser. This repository holds the code of every method and baseline, the run
configuration and the generator of every table, figure and number of the paper.

## Installation and quickstart

```bash
git clone https://github.com/michalprusek/geometry-before-surrogates-neurips2026 repo && cd repo
uv venv --python 3.12 .venv                       # or: python3.12 -m venv .venv
uv pip install --no-deps --python .venv/bin/python -r environment/requirements-frozen.txt
source .venv/bin/activate
curl -L -o ../results_bundle.tar.gz https://github.com/michalprusek/geometry-before-surrogates-neurips2026/releases/download/v1.0.0/results_bundle.tar.gz
tar -xzf ../results_bundle.tar.gz -C .. && (cd ../results_bundle && sha256sum -c --quiet SHA256SUMS)
bash scripts/reproduce_tables.sh ../results_bundle ../tables
```

- `--no-deps` is required: `requirements-frozen.txt` lists all 114 packages of the runs (Python 3.12.14; system and
  hardware in `environment/README.md`) and `lassobench` declares requirements the runs never had. `requirements.txt`
  and `pyproject.toml` give lower bounds only.
- `reproduce_tables.sh` calls `scripts/cr/analysis/make_tables.py` with the arguments that produced the paper's tables
  (about 75 s on the authors' server). It needs a git clone whose generator (`scripts/cr/analysis/`) is unmodified and
  an output directory outside the repository, and stops (exit 2) on a missing or unfinished cell or one from another commit. Every table and `numbers.tex` of the
  paper come out byte for byte, apart from the commit stamp in their first lines.

## Paper item -> generated file -> how to rebuild

`scripts/reproduce_tables.sh` writes every file below and `run_manifest.json` (sha256 of all inputs and outputs). The
last column gives the bundle directories read (`docs/camera_ready/RESULTS.md`) and the rerun script, or "stored".

| Paper item | Generated file | How to rebuild |
|---|---|---|
| Table 1 (Sec. 4.1), its notes (App. H) | `tab_main.tex`, `tab_main_notes.tex` | `cr2026/`, `cr2026_kajman/` [`cr_queue.py`], `cr2026rd/` [`cr_queue.py`], `cr2026pub/` [`cr_queue_pub.py`], `stored_guacamol/` [stored] |
| Table 2 (Sec. 4.3): the three recipe rows | `tab_boxsphere.tex` | `cr2026/` [`cr_queue.py`] |
| Table 3 (Sec. 4.3): the 2x2 cell means of the factorial, split from the generated `tab_boxsphere.tex` by hand (`docs/camera_ready/PAPER_TABLE_EDITS.patch`) | `tab_factorial.tex` (in the patch; the generator writes these cells as the middle block of `tab_boxsphere.tex`) | `cr2026/` [`cr_queue.py`] |
| Table 4 (Sec. 4.4) | `tab_crossdomain.tex` | `cr2026/` [`cr_queue.py`], `cr_inputs/lassobench_cr/` [`run_lassobench.py`], `cr_inputs/nasbench/` [stored] |
| Table 5 (App. A) | `tab_hyper.tex` | the settings of GeoWalk and GeoCMA; no result files |
| Tables 6 (App. B), 7, 8 (App. C) | `tab_sensitivity.tex`, `tab_pertask.tex`, `tab_pertask_full.tex` | `cr2026/` [`cr_queue.py`] |
| Figure 1, Table 9 (App. D) | `fig_convergence_grid.pdf`, `tab_trajectories.tex` | `cr2026/` [`cr_queue.py`], `cr2026pub/` [`cr_queue_pub.py`], `stored_guacamol/` [stored] |
| Table 10 (App. E) | `tab_crosscodec.tex` | `cr2026/` [`cr_queue.py`], `cr_inputs/chembed/`, `cr_inputs/power_spherical/` [stored] |
| Table 11 (App. F) | `tab_budget.tex` | `stored_guacamol/` [stored, N = 1000] |
| Tables 12, 13 (App. G) | `tab_lassobench.tex` (both tables) | `cr_inputs/lassobench_cr/`, `cr_inputs/lassobench_pub/` [`run_lassobench.py`] |
| Table 14 (App. H); the full method cards and the run status (result files, completed and fallback cells per method) | `tab_baselines_overview.tex`; `records/BASELINE_CARDS.md`, `records/RUN_STATUS.md` (committed as `docs/baselines/`) | the card texts in `scripts/cr/analysis/appendixH_*.csv` and the result directories of the printed arms |
| Table 15 (App. H), and the verification of the published recipes, which the paper does not print | `tab_published.tex`, `tab_verification.tex` | `cr_inputs/verification/verdicts.json` [`verify_baselines.py`]; for `tab_published.tex` also `cr2026/` [`cr_queue.py`], `cr2026pub/` [`cr_queue_pub.py`] |
| Figure 2, Tables 16 to 18 (App. I) | `fig_shells.pdf`, `tab_geometry.tex`, `tab_radius.tex` (three blocks), `tab_convex.tex` (with the direction controls) | `cr2026/` [`cr_queue.py`], `cr_inputs/geometry_cr/` [`scripts/cr/analysis/fullset_geometry.py`, needs ZINC-250k and the weights] |
| Table 19 (App. J): the five factorial contrasts, split from the generated `tab_boxsphere.tex` by hand (in the patch) | `tab_factorial_contrasts.tex` (in the patch; the generator writes these rows as the contrast block of `tab_boxsphere.tex`) | `cr2026/` [`cr_queue.py`] |
| Repository records (not printed): the full Tables 1, 2 and 4 with every row and the cell-level statistics (the full Table 2 also holds the cells of Table 3 and the contrasts of Table 19 with their intervals), the compute per block of runs (App. K totals), the per-task radius table, the aggregated trajectory figure and the LassoBench figure | `records/tab_main_full.tex`, `records/TABLE1_FULL.md`, `records/tab_boxsphere_full.tex`, `records/tab_crossdomain_full.tex`, `records/tab_compute.tex`, `records/tab_radius_pertask.tex`, `records/fig_trajectories.pdf`, `records/fig_lassobench.pdf` | as for the printed tables above |
| Every number in the text | `numbers.tex` (one macro per number) | all of the above |

## Reproducibility by benchmark

| Benchmark | Bundle directory | Status |
|---|---|---|
| GuacaMol: camera-ready campaign, control block, published recipes | `cr2026/`, `cr2026_kajman/`, `cr2026ctl/`, `cr2026pub/`, `cr2026rd/` | rerunnable from this code (`scripts/cr/cr_queue.py`, `scripts/cr/cr_queue_pub.py`); bit-for-bit agreement with the published result files was checked on the authors' hardware and environment only |
| LassoBench: rerun and published methods | `cr_inputs/lassobench_cr/`, `cr_inputs/lassobench_pub/` | rerunnable from this code: one call of `scripts/run_lassobench.py --methods <method> --tasks <task> --seeds <seed>` per run, one CPU thread; design in `expected_jobs()` of `scripts/cr/analysis/crlib/lasso_cr.py`, launchers in `scripts/cr/lasso/` |
| Published recipes in their authors' settings | `cr_inputs/verification_runs/`, `cr_inputs/verification/` | run by `scripts/cr/verify_baselines.py` (launchers in `scripts/cr/verify_queue/`) |
| NAS-Bench-101 | `cr_inputs/nasbench/` | recomputed from the stored per-seed results; the benchmark data and VAE weights of these runs are not available |
| Stored runs of the submitted version (external baselines, SAASBO, N = 1000, other codecs) | `stored_guacamol/`, `cr_inputs/` | recomputed from the stored result files, not rerun; section 3 of `docs/camera_ready/RUN_RULES.md` gives the reason for each. The harness that ran TuRBO, BAxUS, LOL-BO, InvBO and full-dimensional CMA-ES for the submitted version no longer exists and is not part of this repository |

## Run one cell

| what | how to get it | licence |
|---|---|---|
| SELFIES Transformer VAE weights of LOL-BO (25,366,351 bytes), trained by its authors on the GuacaMol training set (about 1.27 million molecules), not on ZINC-250k | `bash scripts/download_weights.sh` (fixed URL, sha256 checked) | MIT (LOL-BO repository) |
| `data/zinc_10k.txt`, the cold-start pool (first 10,000 molecules of ZINC-250k, a reference set of drug-like molecules) | in the repository | see `THIRD_PARTY_NOTICES.md` |
| `results/precomputed/zinc_per_coord_bounds.json`, `zinc_latents_sample.npz`, `zinc_mean_norm.json` (box bounds, a latent sample and the mean latent norm, computed by the authors from ZINC-250k and the weights) | in the repository | as this repository |
| ZINC-250k, all 249,455 molecules; only for the geometry, floor and bounds scripts | `bash scripts/download_zinc250k.sh` (fixed URL, sha256 checked) | see `THIRD_PARTY_NOTICES.md` |
| GuacaMol oracles | the `guacamol` package of the environment | MIT |

```bash
python scripts/cr/run_cell.py geowalk ranolazine_mpo 42 \
    results/guacamol/demo/geowalk/ours_geowalk_ranolazine_mpo_s42_128d.json --arm geowalk --host-class gpu
```

Arguments: solver, task (`task_poli_ids` of `scripts/cr/manifest.json`), seed, output file (the cell's own name, never
overwritten; it holds the 30 scores, the running best and a `provenance` block). Median `elapsed_seconds` (440 cells,
one shared NVIDIA L40S): `geowalk` 5.4 s, `cma_es_subspace` 5.3 s, `hvarfner_scaled_gp` 12 s, `linear_bo` 139 s.

## Run the campaign

```bash
python scripts/cr/cr_queue.py --host-class gpu --waves 2,1,3,3b,4 --gpus 0,1 --workers-per-gpu 4 --threads 4 \
    --expect-commit "$(git rev-parse HEAD)" --output-root results/guacamol/cr2026 --allow-any-host --dry-run
```

`scripts/cr/manifest.json` lists every arm (solver, wave, host class, time limit), each on 11 tasks and seeds 42-81
(the replication arms of the control block on seeds 82-121). Without `--dry-run` the queue runs the missing cells and
never deletes or reruns a completed one; `--expect-commit` refuses a dirty tree or another commit, and
`--allow-any-host` admits hosts that the manifest does not name (it names the authors' machines). `cr_queue_pub.py`
runs the published-recipe block with the same flags; `docs/camera_ready/README.md` covers the output layout, the
control block and the CPU-class arms, which ran in one queue with one worker because GaBO depends on the CPU
microarchitecture (`scripts/cr/manifest.json`).

## Tests

```bash
python -m pytest tests -q                         # solvers, queue and run tooling
python -m pytest scripts/cr/analysis/tests -q     # table generator
python scripts/cr/analysis/make_tables.py --legacy --root G=../results_bundle/stored_guacamol \
    --root INPUTS=../results_bundle/cr_inputs --inputs-root ../results_bundle/cr_inputs --outdir ../legacy --latex-check
```

At the snapshot commit: `tests`: 644 passed and 33 skipped: 27 need a CUDA device (`tests/test_device_fix.py`), and 6 compare the ports with the official TuRBO code and the BoTorch BAxUS tutorial, which are not distributed here (set `PUBLISHED_BASELINES_SRC` to local copies to run them). `scripts/cr/analysis/tests`: 202 passed. Measured on panda (CPU, Python 3.12.14) on 2026-11-06 from a clean clone of the release. The third command checks the generator against the audited numbers of
earlier versions: 564 `PASS`, 9 documented `KNOWN-DIFF`, no `FAIL` and no `ERROR`.

## Repository layout

| path | content |
|---|---|
| `src/` | GeoWalk and GeoCMA (`spherevo/solver.py`), the GP and published-recipe baselines (`spherevo/`), codecs (`codec/`), oracles (`oracle/`), vendored GaBO code (`external/`), configuration and statistics (`analysis/`) |
| `scripts/` | the two entry points `run_benchmark.py` (one GuacaMol run) and `run_lassobench.py` (one LassoBench run); `cr/`, the camera-ready campaign, with the table generator in `cr/analysis/`; `reproduce_tables.sh` and the two download scripts. The other scripts in `scripts/` and `scripts/analysis/` record how the stored runs and the precomputed inputs were produced (machine paths replaced by `/path/to`) |
| `tests/` | tests of the solvers, the queue and the run tooling (the generator's tests are in `scripts/cr/analysis/tests/`) |
| `docs/` | `camera_ready/`: run rules with their dated amendments, output layout, results bundle, and `RELEASE_NOTES.md`, the provenance of this snapshot of the authors' development repository (it replaces the anonymous supplement of the submitted version); `baselines/`: method cards and run status |
| `environment/` | the frozen package list, the licence of every package, the system and hardware of the runs |
| `data/`, `results/precomputed/` | the cold-start pool; the precomputed box bounds, latent sample and mean latent norm |
| `LICENSES/` | licence texts of the third-party components |
| `CHANGES.md` | what changed between the submitted and the camera-ready version of the paper, and why |

## Licence and third-party material

The authors' code is released under the licence in `LICENSE`. Third-party code and data in this repository keep their
own licences: `THIRD_PARTY_NOTICES.md` lists every component with its version, licence and source, and `LICENSES/`
holds the licence texts. One file is not under the MIT licence: `src/spherevo/published_external.py` contains a port
of the official TuRBO code and is distributed under the TuRBO licence (Uber Non-Commercial License,
`LICENSES/turbo-LICENSE.txt`), for non-commercial purposes only. The `turbo_published` result files of the results
bundle are outputs of that port and, under the same licence, may not be used for commercial purposes.

## Citation

```bibtex
@inproceedings{prusek2026match,
  title     = {Match the Geometry Before the Surrogate: Extreme Low-Budget Optimization in High Dimensions},
  author    = {Pr{\r{u}}{\v{s}}ek, Michal and Novoz{\'a}msk{\'y}, Adam and {\v{S}}roubek, Filip},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://openreview.net/forum?id=2BB0oOmLs4}
}
```
