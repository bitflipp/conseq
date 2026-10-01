<p align="center"><img src="icon.svg" alt="conseq icon" width="128" height="128"></p>

# conseq

Detect consecutive perfect intervals (parallel fifths, parallel octaves) in MusicXML scores and highlight where they occur.

Consecutive perfect intervals are a classic voice-leading concern in counterpoint. `conseq` finds them and marks them with color so you can review them in your notation software and decide what to do.

## Features

- Detects **parallel fifths**, **parallel octaves**, or **both**
- Handles **rest look-through**: parallels across a shared rest gap are still detected even if both voices briefly fall silent
- Works across **multiple parts and staves** (grand staff, SATB, etc.)
- **Independent groups** of parallels get distinct colors (up to 8; uses a matplotlib tab10 palette)
- Detections that share a note element are merged into one color group via union-find
- Excludes grace notes (ornamental) and same-voice octave doublings (piano reinforcement)
- Optional **navigation markers** (`--annotate`): one numbered, color-matched rehearsal mark per detected group
- Preserves the original XML prologue verbatim (declaration, DOCTYPE)
- Reads/writes **stdin and stdout** with `-`
- **No runtime dependencies** — stdlib only

## Usage

```
python conseq.py [--interval {fifths,octaves,both}] [--annotate] input.xml output.xml
```

| Argument | Description |
|---|---|
| `input.xml` | Input MusicXML file. Use `-` to read from stdin. |
| `output.xml` | Output MusicXML file. Use `-` to write to stdout. |
| `--interval` | Which interval type to detect: `fifths` (default), `octaves`, or `both`. |
| `--annotate` | Insert a numbered, color-matched rehearsal mark (`‖1`, `‖2`, …) above the first note of each detected group, to locate highlights in large scores. In MuseScore these appear in the Timeline panel as clickable jump targets. |

After processing, a summary line is printed to stderr:

```
4 note(s) in 2 consecutive-fifth group(s) colorized.
```

With `--annotate` it ends `colorized and marked.` instead.

## Examples

Check for parallel fifths and write a colorized copy:

```bash
python conseq.py piece.xml piece_colored.xml
```

Check for both parallel fifths and octaves:

```bash
python conseq.py --interval both piece.xml piece_colored.xml
```

Pipe through stdin/stdout:

```bash
python conseq.py - - < piece.xml > piece_colored.xml
```

Open the output in any MusicXML-aware renderer (MuseScore, Sibelius, Dorico, Finale, Flat.io, etc.) to see the colorized notes.

## How it works

1. **Collect notes** — Traverse the MusicXML tree, converting each pitched non-grace note into a `NoteInfo` with onset/offset times (as exact `Fraction` values), MIDI pitch, and a voice key `(part_id, staff_id, voice_id)`.

2. **Build a voice index** — Group notes by attack time and release time per voice. Precompute sorted structures for efficient lookback queries.

3. **Detect parallels** — For each time point where a perfect interval begins, check two cases:
   - *Direct boundary*: the same voice pair ended the same interval type immediately before.
   - *Rest look-through*: both voices previously sounded that interval, then both fell silent without any intervening attack in either voice, and both re-enter with the same interval.

4. **Merge groups** — Notes that share an element across detections are unioned via union-find so they receive the same color.

5. **Colorize** — Set the `color` attribute on each detected `<note>` element (and its `<stem>`/`<notehead>` children if present) and write the modified XML.

## Requirements

- Python 3.8+ (uses the walrus operator)
- No third-party packages

## Running the tests

```bash
pip install pytest pytest-cov
pytest tests/ -v --cov=conseq --cov-report=term-missing
```

177 tests, 99% coverage.

## License

MIT
