# LassoBench and verification launchers (as run)

Byte copies of the launchers that produced the LassoBench runs and the baseline verification of the camera-ready
(release blocker P3 of the release audit). They ran on the CPU server, outside this repository; the paths in them
are those of that server (the release build replaces machine paths). Each job wrote one provenance record to the
`manifest.jsonl` of its run folder, which the generator joins to every result file (`scripts/cr/analysis/crlib/lasso_cr.py`).

| folder | what | rules |
|---|---|---|
| `rerun/` | LassoBench full rerun from tag `cr-runs-2026-10-12`: `full_rerun_jobs.py` (1,442 jobs) and `jobs_full_rerun.txt`, the 39 Papenmeier leukemia jobs `jobs_papenmeier_leukemia.txt`, the job wrapper `run_lasso.sh` with `manifest_append.py`, `common.sh`, `with_slot.sh`, the tmux launchers | amendments of 2026-10-12 ("LassoBench is rerun in full", "Papenmeier at D = 7129 is run on all 40 seeds") |
| `published/` | LassoBench published methods from tag `cr-verify-2026-10-13`: `pub_jobs.py` (phase 1: ten leukemia pilots and 2,000 jobs on five tasks; phase 2: leukemia seeds 43-81 for each method whose pilot exited 0 inside the hour), `pub_queue.sh`, `pub_smoke.sh`, `run_lasso.sh`, `manifest_append.py`, `common.sh`, and the job lists as run (`jobs_phase1.txt`, `jobs_phase2.txt`) | published-recipe block of 2026-10-12 (3), revision 1 (f) of 2026-10-13 |
| `../verify_queue/` | verification queue (`queue.sh`, `run_job.sh`, `common.sh`, `make_jobs.py`, `target_needs.py`, `check_records.py`, job lists `jobs_primary_r1.txt`, `jobs_smoke_r1.txt`); `run_job_until_2026-10-19.sh` is the wrapper as it ran until 2026-10-19 08:45, `run_job.sh` the wrapper that skipped the BAxUS, TuRBO and CMA-ES jobs from then on, `not_run_2026-10-19.txt` the 238 jobs it skipped | verification protocol revision 1 (2026-10-13); end of the verification (2026-10-19) |

The binding rules are `docs/camera_ready/RUN_RULES.md`.
