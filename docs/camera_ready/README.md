# Camera-ready GuacaMol generation: tooling and outputs

`RUN_RULES.md` in this directory is the binding rule set (a byte copy of the rules file, including its
amendment log). This page says where the generation writes its results and how to read them.

## Tooling

| file | role |
|---|---|
| `scripts/cr/manifest.json` | every arm of the generation: unique arm id (= output directory), solver, run overrides, wave, host class, per-cell timeout, and an optional seed range `seeds` (default: the manifest's 42-81) |
| `scripts/cr/run_cell.py` | runs one (solver, task, seed) cell in a fresh process through `run_benchmark()` with the R-1 defaults (rule R-1) |
| `scripts/cr/cr_queue.py` | runs the missing cells of a selection, validates every result, writes crash records and the queue records |

GPU-class arms run on panda (2x L40S) in one queue over both GPUs, wave 2 first
(`--waves 2,1,3,3b,4`). The CPU-class arms (`gabo`, `matern_gabo`) run only on kajman, in one queue with
one worker (`cpu_hosts` in the manifest; amendment of 2026-10-12).

## Output layout

```
results/guacamol/cr2026/
  .queue.lock                                   one queue per output root (flock)
  <arm id>/
    ours_<solver>_<poli task>_s<seed>_128d.json            the result of a cell
    ours_<solver>_<poli task>_s<seed>_128d_sd8.json        (_sd8 / _sd64: cma_es_subspace_sd8 / _sd64)
    <result name>.crash.json                    terminal crash record (R-8)
    <result name>.interrupted.json              the queue was stopped while the cell ran
    <result name>.collision-<pid>.json          a second result that lost the publish race (R-5)
    <result name>.running                       claim of the process computing the cell
    attempts/                                   earlier attempts, archived, never deleted
```

Queue records (`queue.log`, `cells.jsonl`, `summary.json`, one log per cell) go to
`logs/cr/<timestamp>_<tag>/` in the clone or to the `--log-dir` the launch script gives.

## Reading results (table generator)

* Read only files whose name matches `ours_*_128d(_sd[0-9]+)?.json` at the top level of an arm
  directory. Skip `*.crash.json`, `*.interrupted.json`, `*.collision-*.json`, `*.running`, the
  `attempts/` subdirectory and dot files.
* Group by `provenance.cr_arm` (= the arm directory), not by `solver`: `cma_es_subspace_sd8/` and
  `cma_es_subspace_sd64/` hold files whose `solver` is `cma_es_subspace`; they record `subspace_dim`
  8 or 64 and `provenance.run_overrides`.
* A cell is completed if and only if its result file passes the queue's validation
  (`cr_queue.validate_result`): `status == "completed"`, `n_evaluated == len(all_scores) == budget ==
  30`, solver, task, seed and `subspace_dim` of the cell, `provenance.cr_arm` and
  `provenance.run_overrides` of the arm, `provenance.git_commit_full` equal to the campaign commit and
  `provenance.git_dirty is False`.
* A crash record next to a valid result is not a crash: the child published its result and then died
  (for example in interpreter teardown). The queue counts such a cell as completed and never reruns it.
  For the crash counts of rule R-5, count the cells that have a crash record and no valid result.
* An `.interrupted.json` record alone is not a result; the next queue reruns that cell.
* Host: `provenance.host_fqdn` should be uniform within an arm (panda for GPU-class arms, kajman for
  `gabo` and `matern_gabo`); check it.

## Arms whose name does not say everything

* `geowalk_one_fifth`: GeoWalk with a success-based angle rule in the refinement phase. The first
  refinement angle is 0.30 rad; after every refinement step the angle is multiplied by 1.2 if that step
  improved the incumbent (by more than 1e-10) and by 0.85 otherwise, then clamped to [0.05, 0.50] rad.
  With these constants the angle is stationary at a success rate of ln(1/0.85) / ln(1.2/0.85), about
  0.47, not 1/5; a printed description gives the constants (amendment of 2026-10-12).
* `hvarfner_scaled_gp_sphere05`, `_sphere15`, `_sphere20`: the `hvarfner_scaled_gp` arm (Hvarfner GP,
  qLogEI argmax over 2048 uniform candidates on a sphere) with the sphere radius set to 0.5, 1.5 and 2.0
  times the cold-start mean latent norm. The multiplier is the class attribute
  `sphere_radius_multiplier` and is encoded in the solver name (`05` = 0.5, `15` = 1.5, `20` = 2.0); it
  is not a field of the result file.

## Control block (amendment of 2026-10-12)

The control block runs from its own commit (tag `cr-controls-2026-10-12`, branch `camera-ready-controls`,
parent `cr-runs-2026-10-12`) in its own clone, into its own output root `results/guacamol/cr2026ctl`,
with `--waves c1,c2`. That commit only adds the three solver classes below, the per-arm seed range and
the arms of the block; the code it shares with the main campaign is unchanged.

* Wave `c1`, seeds 42-81 (the main campaign's cold starts):
  * `hvarfner_scaled_gp_box_argmax_raw` (C1): `hvarfner_scaled_gp_box_argmax` with the GP fitted on raw
    latents, as the shell arm `hvarfner_scaled_gp` fits it, instead of on per-coordinate rescaled inputs
    `(x - p01) / width`. Same GP constructor, the same 2048 uniform [p01, p99]-box candidates (scored in
    raw coordinates), qLogEI argmax, the same fallbacks and random-number use.
  * `hvarfner_scaled_gp_boxdir_argmax` (C2a): the same box draws, each renormalised to the run's shell
    radius (the cold-start mean latent norm), GP on raw latents, qLogEI argmax.
  * `random_boxdir` (C2b): the same candidate distribution with no surrogate, one draw per step
    (`random_box_faithful`'s draw renormalised to the shell radius).
* Wave `c2`, seeds 82-121 (new cold starts): the six decomposition arms of the 2026-10-12 analysis plan
  under the ids `<arm>__s82` (`random_search__s82`, `random_box_faithful__s82`, `hvarfner_scaled_gp__s82`,
  `hvarfner_scaled_gp_box_argmax__s82`, `hvarfner_scaled_gp_sphere_continuous__s82`,
  `hvarfner_scaled_gp_box_faithful__s82`). Each has its base arm's solver, overrides and timeout; its
  directory `<root>/<arm>__s82/` holds files whose `solver` is the base solver, whose seed is in 82-121
  and whose `provenance.cr_arm` is `<arm>__s82`. Group by `provenance.cr_arm`, as everywhere.

An arm runs only its own seed range: the queue builds no other cell (`--seeds` may only narrow the
range; a seed outside it refuses the selection), and `run_cell.py --arm` refuses any other seed.

## Control C4 (amendment of 2026-10-14), the printed shell x continuous cell since 2026-11-06

Control C4 runs from a third commit on `camera-ready-controls` (tag `cr-controls-lbfgs-2026-10-14`,
parent `cr-controls-2026-10-12`) into the control block's root `results/guacamol/cr2026ctl`, with
`--waves c3`. That commit only adds one solver class, its registration, the arm and tests; the code it
shares with the main campaign and the control block is unchanged. This repository carries that code, so
the printed cell can be rerun from it.

* Wave `c3`, seeds 42-81 (the main campaign's cold starts, so its cells pair with the decomposition
  arms of the main campaign): `hvarfner_scaled_gp_sphere_lbfgs`, the shell x continuous cell with the
  maximiser of `hvarfner_scaled_gp_box_faithful`. It is `hvarfner_scaled_gp_sphere_continuous` (same
  shell, GP on raw latents, qLogEI, flat-signal rule, fallback on the shell) with projected Adam
  replaced by `optimize_acqf(q=1, num_restarts=10, raw_samples=512)` on x = r * v / ||v||: v unbounded,
  512 Gaussian raw samples through BoTorch's `gen_batch_initial_conditions`, Boltzmann selection of the
  10 starts and BoTorch's batched L-BFGS-B, as in the box arm. Its files are in
  `<root>/hvarfner_scaled_gp_sphere_lbfgs/`; the generator reads them through the root `CRC4` (the same
  directory as `CRCTL`, with the commit of this tag as its expected commit).
* Since the authors' decision of 2026-11-06 (RUN_RULES 2026-11-06 16:30) this arm is the shell x
  continuous cell of the printed factorial (Table 3 and the contrasts of Appendix J), in place of
  `hvarfner_scaled_gp_sphere_continuous`, so that both supports use the same continuous maximiser.
