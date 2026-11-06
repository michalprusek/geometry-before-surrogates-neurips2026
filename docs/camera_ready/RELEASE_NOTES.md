# Release notes: how this snapshot was made

The latest commit of this repository is a snapshot of the authors' private development repository.
It was produced mechanically by `make_release.sh` from the lists in `release_files/` (kept by the authors).

| item | value |
|---|---|
| source commit (private repository) | `f811db2b0dba17d9bd1f459745da25ef5cea2706` (2026-11-06T22:40:29+01:00) |
| commit of the main campaign (tag `cr-runs-2026-10-12`), recorded in the result files | `14481cb55d21a770da11c3d6f866fcae98891c18` |
| commit of the control block (tag `cr-controls-2026-10-12`) | `fd8a03c5a75475e79ed8e9f460c38514a7e9717d` |
| commit of the published-recipe block, recorded in its result files | `963ef040cae648d4132e1d3330a2a9c06456e97d` |
| tracked paths at the source commit | 1958 |
| not shipped | 1643 |
| shipped byte-identical (same git blob id) | 269 |
| shipped with a documented change | 46 |
| files added by the release | 25 |
| `docs/camera_ready/RUN_RULES.md` | identical to the binding file (sha256 f93380a6a1971ab9) |

## Relation to the commits recorded in the result files

- tag `cr-runs-2026-10-12` = `14481cb55d21a770da11c3d6f866fcae98891c18` is an ancestor of the source commit
- tag `cr-controls-2026-10-12` = `fd8a03c5a75475e79ed8e9f460c38514a7e9717d` is an ancestor of the source commit
- the commit of the published-recipe block = `963ef040cae648d4132e1d3330a2a9c06456e97d` is an ancestor of the source commit
- Of the 315 shipped paths, 170 have at the source commit the same content as at the main-campaign commit, 21 differ and 124 did not exist there. Outside the table generator (`scripts/cr/analysis/`) the differing or new paths are: `.gitignore`, `docs/baselines/BASELINE_CARDS.md`, `docs/baselines/RUN_STATUS.md`, `docs/baselines/SOURCE.json`, `docs/baselines/TABLE1_FULL.md`, `docs/baselines/c4_reading.md`, `docs/baselines/post_hoc_random_search.md`, `docs/baselines/tab_boxsphere_full.tex`, `docs/baselines/tab_convex_full.tex`, `docs/baselines/tab_crossdomain_full.tex`, `docs/baselines/tab_lassobench_full.tex`, `docs/baselines/tab_main_full.tex`, `docs/baselines/tab_main_full_notes.tex`, `docs/camera_ready/README.md`, `docs/camera_ready/RUN_RULES.md`, `pyproject.toml`, `requirements.txt`, `scripts/cr/baseline_targets.example.json`, `scripts/cr/baseline_targets.json`, `scripts/cr/cr_queue.py`, `scripts/cr/cr_queue_pub.py`, `scripts/cr/lasso/README.md`, `scripts/cr/lasso/published/common.sh`, `scripts/cr/lasso/published/jobs_phase1.txt`, `scripts/cr/lasso/published/jobs_phase2.txt`, `scripts/cr/lasso/published/manifest_append.py`, `scripts/cr/lasso/published/pub_jobs.py`, `scripts/cr/lasso/published/pub_queue.sh`, `scripts/cr/lasso/published/pub_smoke.sh`, `scripts/cr/lasso/published/run_lasso.sh`, `scripts/cr/lasso/rerun/common.sh`, `scripts/cr/lasso/rerun/full_rerun_jobs.py`, `scripts/cr/lasso/rerun/full_rerun_tmux.sh`, `scripts/cr/lasso/rerun/jobs_full_rerun.txt`, `scripts/cr/lasso/rerun/jobs_papenmeier_leukemia.txt`, `scripts/cr/lasso/rerun/manifest_append.py`, `scripts/cr/lasso/rerun/papenmeier_leukemia_tmux.sh`, `scripts/cr/lasso/rerun/run_lasso.sh`, `scripts/cr/lasso/rerun/with_slot.sh`, `scripts/cr/manifest.json`, `scripts/cr/manifest_pub.json`, `scripts/cr/run_cell.py`, `scripts/cr/verify_baselines.py`, `scripts/cr/verify_queue/check_records.py`, `scripts/cr/verify_queue/common.sh`, `scripts/cr/verify_queue/jobs_primary_r1.txt`, `scripts/cr/verify_queue/jobs_smoke_r1.txt`, `scripts/cr/verify_queue/make_jobs.py`, `scripts/cr/verify_queue/not_run_2026-10-19.txt`, `scripts/cr/verify_queue/queue.sh`, `scripts/cr/verify_queue/run_job.sh`, `scripts/cr/verify_queue/run_job_until_2026-10-19.sh`, `scripts/cr/verify_queue/target_needs.py`, `scripts/reproduce_tables.sh`, `scripts/run_benchmark.py`, `scripts/run_lassobench.py`, `scripts/train_power_spherical_vae.py`, `src/codec/spcauchy_codec.py`, `src/codec/vmf_codec.py`, `src/spherevo/cmaes_full_rd.py`, `src/spherevo/gp_competitors.py`, `src/spherevo/published_arms.py`, `src/spherevo/published_external.py`, `src/spherevo/published_recipes.py`, `src/spherevo/solver.py`, `tests/test_control_block.py`, `tests/test_control_c4.py`, `tests/test_cr_cmaes_baxus.py`, `tests/test_cr_tooling.py`, `tests/test_geowalk_one_fifth.py`, `tests/test_new_benchmarks.py`, `tests/test_published_block.py`, `tests/test_published_recipes.py`.
- Of the 315 shipped paths, 172 have at the source commit the same content as at the control-block commit, 20 differ and 123 did not exist there. Outside the table generator (`scripts/cr/analysis/`) the differing or new paths are: `.gitignore`, `docs/baselines/BASELINE_CARDS.md`, `docs/baselines/RUN_STATUS.md`, `docs/baselines/SOURCE.json`, `docs/baselines/TABLE1_FULL.md`, `docs/baselines/c4_reading.md`, `docs/baselines/post_hoc_random_search.md`, `docs/baselines/tab_boxsphere_full.tex`, `docs/baselines/tab_convex_full.tex`, `docs/baselines/tab_crossdomain_full.tex`, `docs/baselines/tab_lassobench_full.tex`, `docs/baselines/tab_main_full.tex`, `docs/baselines/tab_main_full_notes.tex`, `docs/camera_ready/README.md`, `docs/camera_ready/RUN_RULES.md`, `pyproject.toml`, `requirements.txt`, `scripts/cr/baseline_targets.example.json`, `scripts/cr/baseline_targets.json`, `scripts/cr/cr_queue.py`, `scripts/cr/cr_queue_pub.py`, `scripts/cr/lasso/README.md`, `scripts/cr/lasso/published/common.sh`, `scripts/cr/lasso/published/jobs_phase1.txt`, `scripts/cr/lasso/published/jobs_phase2.txt`, `scripts/cr/lasso/published/manifest_append.py`, `scripts/cr/lasso/published/pub_jobs.py`, `scripts/cr/lasso/published/pub_queue.sh`, `scripts/cr/lasso/published/pub_smoke.sh`, `scripts/cr/lasso/published/run_lasso.sh`, `scripts/cr/lasso/rerun/common.sh`, `scripts/cr/lasso/rerun/full_rerun_jobs.py`, `scripts/cr/lasso/rerun/full_rerun_tmux.sh`, `scripts/cr/lasso/rerun/jobs_full_rerun.txt`, `scripts/cr/lasso/rerun/jobs_papenmeier_leukemia.txt`, `scripts/cr/lasso/rerun/manifest_append.py`, `scripts/cr/lasso/rerun/papenmeier_leukemia_tmux.sh`, `scripts/cr/lasso/rerun/run_lasso.sh`, `scripts/cr/lasso/rerun/with_slot.sh`, `scripts/cr/manifest.json`, `scripts/cr/manifest_pub.json`, `scripts/cr/verify_baselines.py`, `scripts/cr/verify_queue/check_records.py`, `scripts/cr/verify_queue/common.sh`, `scripts/cr/verify_queue/jobs_primary_r1.txt`, `scripts/cr/verify_queue/jobs_smoke_r1.txt`, `scripts/cr/verify_queue/make_jobs.py`, `scripts/cr/verify_queue/not_run_2026-10-19.txt`, `scripts/cr/verify_queue/queue.sh`, `scripts/cr/verify_queue/run_job.sh`, `scripts/cr/verify_queue/run_job_until_2026-10-19.sh`, `scripts/cr/verify_queue/target_needs.py`, `scripts/reproduce_tables.sh`, `scripts/run_benchmark.py`, `scripts/run_lassobench.py`, `scripts/train_power_spherical_vae.py`, `src/codec/spcauchy_codec.py`, `src/codec/vmf_codec.py`, `src/spherevo/cmaes_full_rd.py`, `src/spherevo/gp_competitors.py`, `src/spherevo/published_arms.py`, `src/spherevo/published_external.py`, `src/spherevo/published_recipes.py`, `src/spherevo/solver.py`, `tests/test_control_c4.py`, `tests/test_cr_cmaes_baxus.py`, `tests/test_cr_tooling.py`, `tests/test_geowalk_one_fifth.py`, `tests/test_new_benchmarks.py`, `tests/test_published_block.py`, `tests/test_published_recipes.py`.
- Of the 315 shipped paths, 180 have at the source commit the same content as at the published-recipe commit, 22 differ and 113 did not exist there. Outside the table generator (`scripts/cr/analysis/`) the differing or new paths are: `.gitignore`, `docs/baselines/BASELINE_CARDS.md`, `docs/baselines/RUN_STATUS.md`, `docs/baselines/SOURCE.json`, `docs/baselines/TABLE1_FULL.md`, `docs/baselines/c4_reading.md`, `docs/baselines/post_hoc_random_search.md`, `docs/baselines/tab_boxsphere_full.tex`, `docs/baselines/tab_convex_full.tex`, `docs/baselines/tab_crossdomain_full.tex`, `docs/baselines/tab_lassobench_full.tex`, `docs/baselines/tab_main_full.tex`, `docs/baselines/tab_main_full_notes.tex`, `docs/camera_ready/README.md`, `docs/camera_ready/RUN_RULES.md`, `pyproject.toml`, `requirements.txt`, `scripts/cr/cr_queue.py`, `scripts/cr/lasso/README.md`, `scripts/cr/lasso/published/common.sh`, `scripts/cr/lasso/published/jobs_phase1.txt`, `scripts/cr/lasso/published/jobs_phase2.txt`, `scripts/cr/lasso/published/manifest_append.py`, `scripts/cr/lasso/published/pub_jobs.py`, `scripts/cr/lasso/published/pub_queue.sh`, `scripts/cr/lasso/published/pub_smoke.sh`, `scripts/cr/lasso/published/run_lasso.sh`, `scripts/cr/lasso/rerun/common.sh`, `scripts/cr/lasso/rerun/full_rerun_jobs.py`, `scripts/cr/lasso/rerun/full_rerun_tmux.sh`, `scripts/cr/lasso/rerun/jobs_full_rerun.txt`, `scripts/cr/lasso/rerun/jobs_papenmeier_leukemia.txt`, `scripts/cr/lasso/rerun/manifest_append.py`, `scripts/cr/lasso/rerun/papenmeier_leukemia_tmux.sh`, `scripts/cr/lasso/rerun/run_lasso.sh`, `scripts/cr/lasso/rerun/with_slot.sh`, `scripts/cr/manifest.json`, `scripts/cr/verify_queue/check_records.py`, `scripts/cr/verify_queue/common.sh`, `scripts/cr/verify_queue/jobs_primary_r1.txt`, `scripts/cr/verify_queue/jobs_smoke_r1.txt`, `scripts/cr/verify_queue/make_jobs.py`, `scripts/cr/verify_queue/not_run_2026-10-19.txt`, `scripts/cr/verify_queue/queue.sh`, `scripts/cr/verify_queue/run_job.sh`, `scripts/cr/verify_queue/run_job_until_2026-10-19.sh`, `scripts/cr/verify_queue/target_needs.py`, `scripts/reproduce_tables.sh`, `scripts/run_benchmark.py`, `scripts/train_power_spherical_vae.py`, `src/codec/spcauchy_codec.py`, `src/codec/vmf_codec.py`, `src/spherevo/cmaes_full_rd.py`, `src/spherevo/gp_competitors.py`, `src/spherevo/published_arms.py`, `src/spherevo/published_external.py`, `src/spherevo/solver.py`, `tests/test_control_c4.py`, `tests/test_cr_cmaes_baxus.py`, `tests/test_cr_tooling.py`, `tests/test_geowalk_one_fifth.py`, `tests/test_new_benchmarks.py`, `tests/test_published_block.py`.

`SOURCE_MAP.tsv` lists every shipped path with its git blob id at the three run commits, at the source
commit and in this snapshot. Equal ids mean byte-identical files, so the code that produced a result file
(the commit recorded in its `provenance`) can be compared with what is published here, file by file.

## Files changed with respect to the source commit

| file | change |
|---|---|
| `.gitignore` | replaced by a new file written for the release |
| `README.md` | replaced by a new file written for the release |
| `pyproject.toml` | The declared build backend (setuptools.backends._legacy:_Backend) does not exist; replaced by setuptools.build_meta. Package description updated to the paper title (it named a superseded method). The package name 'spherevo' is unchanged. Author field filled in (was 'Anonymous'). |
| `scripts/download_weights.sh` | replaced by a new file written for the release |
| `scripts/train_arch2vec.py` | Notice, required by Apache-2.0, that the inlined arch2vec code was modified. absolute paths of the authors' machines replaced by `/path/to` |
| `src/codec/nas_vae.py` | A docstring quoted a sentence from a review; replaced by a paraphrase. |
| `src/codec/selfies_vae.py` | LOL-BO was attributed to the wrong authors; it is Maus et al. (NeurIPS 2022), MIT. Same correction in the class docstring. A comment called the vocabulary 'ZINC 250K'; it is LOL-BO's vocabulary, the token set of the GuacaMol v1 training set. |
| `src/codec/vae_model.py` | LOL-BO was attributed to the wrong authors; the licence of the upstream file (MIT) is now named. |
| `src/spherevo/linear_bo_kernel.py` | The header named the wrong licence (CC-BY 4.0). Upstream is MIT, Copyright (c) 2025 Colin Doumont; the header now says so and lists the changes from upstream. |
| `src/spherevo/published_external.py` | The TuRBO-1 part is a port of the official TuRBO code (Uber non-commercial licence). The file now carries the Uber copyright notice, states that it is distributed under that licence for non-commercial purposes only, and lists the changes and their date. |
| `src/spherevo/published_recipes.py` | A docstring named a local scratch directory of the authors; the clause is removed. |
| `tests/test_published_block.py` | A test pins the vendored Linear BO kernel by its sha256. The release corrects that file's licence header, so the test now pins the corrected file and names the original hash in a comment. |
| 34 scripts of the stored generation under `scripts/` (status `modified` in `SOURCE_MAP.tsv` and not listed above) | absolute paths of the authors' machines replaced by `/path/to`, in place; nothing else changed |

## Source files changed after the published-recipe commit

The published-recipe commit (`963ef040cae648d4132e1d3330a2a9c06456e97d`) is recorded in the result files of its block and descends from the other two run commits.
Before this snapshot was made, the development repository changed the files below; the last column compares
each file at the source commit with its version at the published-recipe commit, checked when this snapshot
was built. Comments are not part of a syntax tree, so a file whose trees are equal once docstrings
are removed runs the same code.

| file | change | check against the published-recipe commit |
|---|---|---|
| `scripts/run_benchmark.py` | The two dispatch blocks of the solvers `lppfn_subspace` and `sphere_pfn`, whose code (`src/lppfn/`, `src/spherevo/pfn/`) no result of the paper uses and the repository no longer contains, now raise an error that says so; one comment was shortened. The line count is unchanged, so the line numbers that `scripts/cr/analysis/appendixH_facts.csv` cites still hold. Every other solver is dispatched as before. | code differs (1746 lines before, 1780 after) |
| `src/spherevo/solver.py` | Docstrings and one comment brought up to date with the paper (the defaults of GeoWalk, the step-size rule of `GeoWalkOneFifthSolver`, a reference to an analysis script that is not part of this release). The code is unchanged. | code unchanged: the syntax trees are equal once docstrings are removed (2325 lines before, 2325 after) |
| `src/codec/spcauchy_codec.py` | The error message for missing weights no longer names a training script that is not part of this release; it says that the weights must be provided. | only string constants differ: the syntax trees are equal once every string is blanked (311 lines before, 311 after) |
| `src/codec/vmf_codec.py` | The same change of the error message for missing weights. | only string constants differ: the syntax trees are equal once every string is blanked (57 lines before, 57 after) |
| `scripts/train_power_spherical_vae.py` | Docstrings only: the usage example names this script, without machine paths, and two references to files that are not part of this release were reworded. The code is unchanged. | code unchanged: the syntax trees are equal once docstrings are removed (451 lines before, 452 after) |
| `pyproject.toml` | The optional dependency group `designbench` was removed together with the Design-Bench runner. (The release then corrects the metadata, see the table above.) | text differs (47 lines before, 46 after) |
| `requirements.txt` | The header comment now says that the file gives lower bounds only and that the runs used the frozen environment; the listed requirements are unchanged. | text differs (42 lines before, 43 after) |

## Files added by the release

- `CHANGES.md`
- `LICENSE`
- `LICENSES/GaBOtorch-MIT.txt`
- `LICENSES/MaternGaBO-MIT.txt`
- `LICENSES/arch2vec-Apache-2.0.txt`
- `LICENSES/botorch-MIT.txt`
- `LICENSES/chemical_vae-Apache-2.0.txt`
- `LICENSES/linear-bo-MIT.txt`
- `LICENSES/lolbo-MIT.txt`
- `LICENSES/pymanopt-fork-BSD-3-Clause.txt`
- `LICENSES/turbo-LICENSE.txt`
- `THIRD_PARTY_NOTICES.md`
- `docs/camera_ready/PAPER_TABLE_EDITS.patch`
- `docs/camera_ready/RESULTS.md`
- `environment/PACKAGE_LICENCES.tsv`
- `environment/README.md`
- `environment/requirements-frozen.txt`
- `results/precomputed/zinc_latents_sample.npz`
- `results/precomputed/zinc_mean_norm.json`
- `results/precomputed/zinc_per_coord_bounds.json`
- `scripts/download_zinc250k.sh`
- `src/external/gabotorch_bm/LICENSE`
- `src/external/gabotorch_bm/pymanopt_addons/LICENSE`
- `src/external/matern_gabo_bm/LICENSE`
- `src/external/matern_gabo_bm/pymanopt_addons/LICENSE`

## What is not in this snapshot

Internal working notes and drafts, the sources of the anonymous submitted manuscript, the archive
of the anonymous supplement, 2 nested clones of other repositories, a stale partial copy
of result files (the complete result files are in the results bundle) and launchers of a baseline
harness that no longer exists.

Code that the paper does not use was removed from the development repository after the
published-recipe commit; this includes the LP-PFN and SpherePFN experiments (17 files), the Design-Bench runner (1 file), analysis and plotting scripts that the paper does not use (69 files) and legacy launchers and training scripts (33 files); the dispatch blocks of `scripts/run_benchmark.py` for the removed solvers now raise an error.

The git history of the private repository is not published.
