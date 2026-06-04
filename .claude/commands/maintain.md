---
description: Review code for readability, maintainability, and sync with CLAUDE.md
argument-hint: [file or path — defaults to the whole project]
---

Review the target for **readability**, **maintainability**, and **sync with
CLAUDE.md**. This is an analysis-and-cleanup pass, not a bug hunt. **Never
commit** — leave any changes in the working tree for the user to review.

## Target

$ARGUMENTS

If no target is given above, review the project's primary source (e.g.
`conseq.py`) together with `CLAUDE.md`.

## Steps

1. Read the target source file(s) and `CLAUDE.md`.

2. **Verify CLAUDE.md against the code, in both directions.** Check every
   constraint and claim — data-structure shapes (e.g. the voice-key triple),
   tie/grace semantics, function return shapes, the documented test count and
   coverage, etc. Flag any drift: the doc claiming something the code no longer
   does, *or* the code doing something the doc should mention but doesn't.

3. **Run the test suite exactly as CLAUDE.md documents it** and confirm the
   documented pass count and coverage still hold. Quote the real numbers; if
   they differ from the doc, that itself is a sync finding.

4. Smoke-test the CLI end to end if it's relevant to the change under review.

5. **Report findings**, grouped as:
   - readability / maintainability nits,
   - CLAUDE.md / README sync issues,
   - robustness notes that are out of scope but worth flagging.

   Keep every finding concrete with `file:line` references. Also call out things
   you checked that were *correct*, so the review reflects real coverage rather
   than reading as a list of complaints. Prefer a few high-confidence findings
   over an exhaustive nitpick dump.

## Applying fixes

- Only after the user asks (or approves), make **low-risk changes only**:
  comments, docstrings, naming, dead-code removal. Do not change behavior.
- Re-run the test suite after editing and confirm it stays green; note any line
  shifts in the coverage report.
- **Do not commit.** Stop once the changes are in the working tree.
