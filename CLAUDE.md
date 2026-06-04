# conseq — Claude Code notes

## Project overview

Single-file Python CLI (`conseq.py`) that detects consecutive perfect intervals (parallel fifths/octaves) in MusicXML files and colorizes the offending notes. Stdlib-only; no third-party runtime dependencies.

## Running

```bash
python conseq.py [--interval {fifths,octaves,both}] [--annotate] input.xml output.xml
# Use '-' for stdin/stdout
python conseq.py - - < input.xml > output.xml
```

`--annotate` (off by default) inserts one navigation marker per violation group — a `<direction>`/`<rehearsal>` reading `‖N`, color-matched to that group's notes — so highlights are findable in large scores. A **rehearsal mark** (not plain `<words>`) is used deliberately: it appears in MuseScore's Timeline panel as a clickable jump target. The marker is inserted as a `<measure>` child immediately before the group's earliest-onset note, so it renders at that beat. Only `color` is set — prominence is left to the renderer's rehearsal-mark style (bold/boxed). The `‖` glyph distinguishes our marks from the composer's rehearsal letters, but note they share the Timeline's rehearsal-mark lane and would be caught by MuseScore's "resequence rehearsal marks."

## Tests

Requires `pytest` and `pytest-cov`. Create a venv if not already present:

```bash
python -m venv /tmp/conseq-venv && /tmp/conseq-venv/bin/pip install pytest pytest-cov
/tmp/conseq-venv/bin/pytest tests/ -v --cov=conseq --cov-report=term-missing
```

`pyproject.toml` sets `pythonpath = ["."]` so `import conseq` works from the repo root.

174 tests, 99% coverage as of the last commit.

## Key constraints

- **Stdlib only.** All XML is handled with `xml.etree.ElementTree`. Do not add runtime dependencies.
- Voice key `(part_id, staff_id, voice_id)` — three components, not two. `staff_id` keeps grand-staff staves separate while still detecting cross-staff intervals.
- Tie-stop notes are not attacks; tie-start notes are not releases.
- Grace notes are excluded entirely from interval detection.
- Same-voice octave doublings (semitones == 0, same vk) are not violations.
- `find_note_colors` returns `(color_dict, n_groups)` — `n_groups` counts independent violation groups before palette modulo.
- `find_violation_groups` returns `list[ViolationGroup]` in score order (earliest onset first); colors match `find_note_colors` exactly (shared `_build_violation_uf` + `_assign_colors`), but group *numbers* run in score order. Used by `--annotate` and as `main`'s single detection pass.
- `PALETTE` has 8 colors; a warning is emitted if there are more than 8 independent groups.

## Pitfall: cross-part compound fifths in tests

When combining notes from multiple parts at the same timestamps, C4+G4 in one part and C4+G4 in another will form cross-part compound fifths and merge their violation groups. Use **time-offset notes** for multi-violation test scenarios (stagger each group to non-overlapping time ranges).
