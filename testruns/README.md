# Testrun Ledger

This directory is the experiment/tuning ledger for sonar training runs.

Use this naming format for each run folder:

`testrun_<run_id>_<short_commit>_<YYYYMMDD_HHMM>`

Example:

`testrun_0001_88b210c_20260210_2145`

## Required files per run

- `run.md`: objective, hypothesis, decisions, execution notes, verdict, next action.
- `config.env`: full resolved config/env for the run.
- `metrics.json`: summary metrics and key checkpoints.
- `artifacts/`: optional outputs (plots, screenshots, mesh summaries).

## Workflow

1. Copy templates from `testruns/templates/` into a new run folder.
2. Fill `run.md` and `config.env` before starting the run.
3. Update `metrics.json` and `run.md` during/after run.
4. Add a one-row summary to `testruns/INDEX.md`.

Git tracks code state; this folder tracks tuning state.
