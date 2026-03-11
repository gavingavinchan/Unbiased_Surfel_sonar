# Submodule Workflow

Use this workflow whenever a git submodule is dirty or needs source changes.

## Core Rules

- Treat each submodule as its own git repo.
- Do not assume a dirty submodule is safe to reset; parent-repo code may already depend on it.
- Distinguish tracked source edits from ignored build artifacts before deciding what to do.
- Only fork/publish submodules that have meaningful local commits you want to keep.

## Recommended Commit Order

1. Commit the source change inside the submodule.
2. Put that submodule commit on a real branch (submodules often end up on detached `HEAD`).
3. Push that branch to your fork/remote.
4. Commit the updated submodule pointer in the parent repo.

## Recommended Remote Setup

- Keep upstream as `origin` when possible.
- Add your own fork as a second remote, e.g. `gavin`.
- Push your working branch to your fork, not to someone else's repo.

## Practical Notes

- After checking out older parent commits, re-check the submodule branch before editing.
- Build outputs inside submodules should usually stay ignored and uncommitted.
- Only modified submodules need this workflow; unchanged submodules do not need their own fork.
- A fresh `git submodule update --init` fetches from the URL in `.gitmodules` (upstream) and lands on upstream's default branch, not your fork's working branch. After cloning, check out the correct branch inside modified submodules manually.
