# conseq — Claude Code notes

## Project overview

Single-file Python CLI (`conseq.py`) that detects consecutive perfect intervals (parallel fifths/octaves) in MusicXML files and colorizes the offending notes. Stdlib-only; no third-party runtime dependencies.

## Running

```bash
python conseq.py [--interval {fifths,octaves,both}] input.xml output.xml
# Use '-' for stdin/stdout
python conseq.py - - < input.xml > output.xml
```

## Tests

Requires `pytest` and `pytest-cov`. Create a venv if not already present:

```bash
python -m venv /tmp/conseq-venv && /tmp/conseq-venv/bin/pip install pytest pytest-cov
/tmp/conseq-venv/bin/pytest tests/ -v --cov=conseq --cov-report=term-missing
```

`pyproject.toml` sets `pythonpath = ["."]` so `import conseq` works from the repo root.

160 tests, 99% coverage as of the last commit.

## Key constraints

- **Stdlib only.** All XML is handled with `xml.etree.ElementTree`. Do not add runtime dependencies.
- Voice key `(part_id, staff_id, voice_id)` — three components, not two. `staff_id` keeps grand-staff staves separate while still detecting cross-staff intervals.
- Tie-stop notes are not attacks; tie-start notes are not releases.
- Grace notes are excluded entirely from interval detection.
- Same-voice octave doublings (semitones == 0, same vk) are not violations.
- `find_note_colors` returns `(color_dict, n_groups)` — `n_groups` counts independent violation groups before palette modulo.
- `PALETTE` has 8 colors; a warning is emitted if there are more than 8 independent groups.

## Pitfall: cross-part compound fifths in tests

When combining notes from multiple parts at the same timestamps, C4+G4 in one part and C4+G4 in another will form cross-part compound fifths and merge their violation groups. Use **time-offset notes** for multi-violation test scenarios (stagger each group to non-overlapping time ranges).
