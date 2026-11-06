# The results bundle

The result files are not tracked in this repository. They are published as one archive, attached to the
release of this repository: https://github.com/michalprusek/geometry-before-surrogates-neurips2026/releases/download/v1.0.0/results_bundle.tar.gz

Unpacked, the bundle has this layout; the right-hand column is the generator option that points at it
(`scripts/cr/analysis/make_tables.py`; `bash scripts/reproduce_tables.sh <bundle_dir> <out_dir>` runs it with all of
them, see the top-level README).

| directory | content | generator option |
|---|---|---|
| `cr2026/` | the main camera-ready campaign: one directory per GPU-class arm, one JSON per cell | `--root CR=` |
| `cr2026_kajman/` | the two CPU-class arms (`gabo`, `matern_gabo`) | `--root CRCPU=` |
| `cr2026ctl/` | the control block: C1, C2, C3 (the replication arms on seeds 82-121) and C4; since 2026-11-06 C4 (`hvarfner_scaled_gp_sphere_lbfgs/`, commit 2a899ad) is the shell x continuous cell of the printed factorial and is read through its own root | `--root CRCTL=` and `--root CRC4=` (the same directory) |
| `cr2026pub/` | the published-recipe block: the GP recipes, Linear BO in its authors' box and the TuRBO port, in the settings of their released code | `--root CRPUB=` |
| `cr2026rd/` | the GuacaMol runs of commit `1ea2f7d`: the full-dimensional CMA-ES with population 20 (`cmaes_full_rd`) and 5 (`cmaes_full_rd_pop5`), and the BAxUS port of the published-recipe block on the `[p01, p99]` box (`baxus_cr`; one of its 440 runs ended in a solver crash and is kept as a result file with status `crashed` and its crash record) | `--root CRRD=` |
| `stored_guacamol/` | the stored generation (submitted version, rebuttal, September reruns), only the arm directories the generator reads: `_archive_papenmeier_sqrtD_bug`, `baxus`, `cmaes_full`, `em1_premise`, `gabo`, `geowalk`, `geowalk_ablations`, `invbo`, `lolbo`, `matern_gabo`, `spherevo`, `turbo` | `--root G=` |
| `cr_inputs/` | everything else the generator reads: `chembed/`, `power_spherical/`, `p1_blocks/`, `p1_blocks_sel/` (stored result sets), `lassobench/` (stored), `lassobench_cr/` and `lassobench_cr_queue/` (camera-ready rerun with its job records), `lassobench_pub/` and `lassobench_pub_queue/` (published LassoBench methods with their job records), `verification/` (the verdicts of the verification of the published recipes), `nasbench/`, `e0_floor/`, `geometry_cr/`, `precomputed/` | `--root INPUTS=`, `--inputs-root` and the options `--geometry-root`, `--lasso-cr-root`, `--lasso-pub-root`, `--verdicts`, `--p1-sel-root` |
| `cr_inputs/verification_runs/` | the 112 run files of the verification of the published recipes; `verification/verdicts.json` summarises the 80 complete ones | not read by the generator |
| `SHA256SUMS` | checksum of every file of the bundle | — |

The camera-ready directories hold every run of the campaigns, including arms that the paper does not print.
Since the scope of the camera-ready was reduced on 2026-10-14 (amendment log of
`docs/camera_ready/RUN_RULES.md`), the generator does not read these arms (`scripts/cr/analysis/crlib/scope.py`,
`NOT_LOADED_ARMS`); its run manifest lists them under `scope`.

Reading rules for the camera-ready directories (which files are results, what counts as completed, how
crash records are counted) are in `docs/camera_ready/README.md`. Every result file of the camera-ready
generation carries a `provenance` block with the commit, the arm, the host and the package versions.

The `turbo_published` result files (in `cr2026pub/` and `cr_inputs/lassobench_pub/`) are outputs of a port of
the official TuRBO code. Under its licence (`LICENSES/turbo-LICENSE.txt`) they may not be used for commercial
purposes.

To check a download: `cd results_bundle && sha256sum -c SHA256SUMS`.
