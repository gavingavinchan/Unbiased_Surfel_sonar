# Snapshots

This folder holds lightweight, time-stamped “state snapshots” so you can track progress without having to reconstruct context from commits and scattered notes.

## Naming

Use the same convention as plans, but with `SNAPSHOT_`:

- `snapshots/SNAPSHOT_<NAME>_<YYYY-MM-DD>.md`

Examples:

- `snapshots/SNAPSHOT_SINGLE_FRAME_GUI_2026-01-26.md`
- `snapshots/SNAPSHOT_R2_SCALE_STATUS_2026-01-26.md`

## Suggested Content

```markdown
# Snapshot: <Title>

**Date/Time:** YYYY-MM-DD HH:MM:SS <TZ>
**Git Commit:** <commit hash>
**Branch:** <branch name>

## What Works (Verified)
- Command(s) run:
  - `<command>`
- Expected outputs:
  - `<path/to/output>`

## What’s Broken / Unclear
- <symptom> (pointer: `<file>`)

## Current Knobs
- Env vars used:
  - `FOO=...`

## Next Actions
1. <next step>
```

Keep snapshots factual and reproducible (commands + outputs + commit/branch). Put longer narratives in `plans/`.
