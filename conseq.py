#!/usr/bin/env python3
"""
Detect consecutive perfect intervals in MusicXML and colorize the involved notes.

A consecutive violation occurs when:
  - Direct boundary: the same interval between the same voice pair ends at time t
    and a new instance begins at time t.
  - Rest look-through: the two voices previously sounded that interval together
    (their notes overlapping in time), then both fell silent without any
    intervening attack in either voice, and both re-enter with the same interval.

Independent violations receive distinct palette colors.  Violations that share
a note element are merged via union-find and receive the same color.

Octave doublings within a single voice stream (piano reinforcement) are excluded.
Grace notes are excluded; ornamental notes are treated as outside the structural
voice leading.

Usage: python conseq.py [--interval {fifths,octaves,both}] <input.xml> <output.xml>
Use '-' as either filename to read from stdin or write to stdout.
"""

import argparse
import re
import sys
import warnings
from bisect import bisect_left, bisect_right
from collections import defaultdict
from fractions import Fraction
from typing import NamedTuple
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STEP_SEMITONE = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}

# Matplotlib tab10 palette — designed for visual distinctness.
PALETTE = [
    '#D62728',  # red
    '#FF7F0E',  # orange
    '#1F77B4',  # blue
    '#2CA02C',  # green
    '#9467BD',  # purple
    '#8C564B',  # brown
    '#E377C2',  # pink
    '#17BECF',  # teal
]

# Semitone distance (mod 12) for each named interval type.
INTERVAL_SEMITONES = {'fifths': 7, 'octaves': 0}

# Sentinel for bisect tuple keys.  When bisecting a list of tuples like
# `(end, start, midi, elem)` for "the last entry with end <= t", we use
# `bisect_left(lst, (t, _BISECT_SENTINEL))`: this sorts strictly after every
# real `(t, *, *, *)` entry because every real start time is a Fraction,
# and Fraction < float('inf').  Cleaner than adding a tiny epsilon to t.
_BISECT_SENTINEL = float('inf')


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

class NoteInfo(NamedTuple):
    """All information about a single pitched note, as collected from MusicXML."""
    start: Fraction   # onset time in quarter-note units
    end:   Fraction   # release time (exclusive)
    midi:  int        # MIDI pitch number
    elem:  object     # the raw ET.Element, used as an identity token
    vk:    tuple      # voice key = (part_id, staff_id, voice_id)


class IntervalPair(NamedTuple):
    """Two notes that form a perfect interval at some time point."""
    note_a: object    # ET.Element
    note_b: object    # ET.Element
    voices: frozenset # frozenset of the two voice keys (vk_a, vk_b)


# ---------------------------------------------------------------------------
# Pitch helpers
# ---------------------------------------------------------------------------

def pitch_to_midi(note_elem):
    """Return the MIDI pitch number for a <note> element, or None for rests.

    Also returns None for <unpitched> notes (percussion): they have no
    <pitch> child, so they're correctly excluded from interval detection.
    """
    pitch = note_elem.find('pitch')
    if pitch is None:
        return None
    step   = pitch.find('step').text
    octave = int(pitch.find('octave').text)
    alter_elem = pitch.find('alter')
    alter  = round(float(alter_elem.text)) if alter_elem is not None else 0
    return (octave + 1) * 12 + STEP_SEMITONE[step] + alter


def forms_interval(midi_a, midi_b, semitones):
    """Return True if the two pitches form the target interval type.

    semitones=0 matches any non-unison octave (a non-zero multiple of 12).
    semitones=N matches any interval whose size mod 12 equals N.
    """
    diff = abs(midi_a - midi_b)
    if semitones == 0:
        return diff > 0 and diff % 12 == 0
    return diff % 12 == semitones


# ---------------------------------------------------------------------------
# MusicXML traversal
# ---------------------------------------------------------------------------

def collect_notes(root):
    """Return a list of NoteInfo for every pitched non-grace note in the score.

    voice_key = (part_id, staff_id, voice_id) identifies which voice stream a
    note belongs to.  staff_id is included so that notes on different staves of
    the same part (e.g. piano grand staff) remain in separate streams; their
    cross-staff interactions are still detected but same-voice octave doublings
    on the same staff are excluded downstream.

    Grace notes (<grace/>) are excluded from interval detection.  This follows
    standard counterpoint pedagogy, which treats grace notes as ornamental and
    outside the structural voice leading.  All grace-note variants (acciaccatura,
    appoggiatura, before-grace, after-grace, grace chords) are skipped uniformly.
    Parallels that exist only through grace-note participation will not be flagged.
    """
    notes = []

    for part in root.iter('part'):
        part_id         = part.get('id', '')
        cursor          = Fraction(0)
        divisions       = 1
        last_note_start = Fraction(0)

        for measure in part.findall('measure'):
            for child in measure:
                tag = child.tag

                if tag == 'attributes':
                    d = child.find('divisions')
                    if d is not None:
                        divisions = int(d.text)

                elif tag == 'note':
                    if child.find('grace') is not None:
                        continue
                    dur_elem = child.find('duration')
                    if dur_elem is None:
                        continue
                    dur      = Fraction(int(dur_elem.text), divisions)
                    is_chord = child.find('chord') is not None

                    if is_chord:
                        start = last_note_start
                    else:
                        start           = cursor
                        last_note_start = cursor
                        cursor         += dur

                    midi = pitch_to_midi(child)
                    if midi is not None:
                        v  = child.find('voice')
                        s  = child.find('staff')
                        vk = (
                            part_id,
                            s.text if s is not None else '1',
                            v.text if v is not None else '1',
                        )
                        notes.append(NoteInfo(start, start + dur, midi, child, vk))

                elif tag == 'forward':
                    d = child.find('duration')
                    if d is not None:
                        cursor += Fraction(int(d.text), divisions)

                elif tag == 'backup':
                    d = child.find('duration')
                    if d is not None:
                        cursor -= Fraction(int(d.text), divisions)

    return notes


# ---------------------------------------------------------------------------
# Tie helpers
# ---------------------------------------------------------------------------

def _is_tie_start(elem):
    return any(t.get('type') == 'start' for t in elem.findall('tie'))

def _is_tie_stop(elem):
    return any(t.get('type') == 'stop'  for t in elem.findall('tie'))


# ---------------------------------------------------------------------------
# Voice index — built once, queried many times
# ---------------------------------------------------------------------------

class VoiceIndex:
    """Efficient per-voice lookback structures derived from a list of NoteInfo.

    Attributes
    ----------
    by_start : dict[Fraction, list[(midi, elem, vk)]]
        All real attacks (tie-stop continuations excluded) grouped by onset.
    by_end : dict[Fraction, list[(midi, elem, vk)]]
        All real releases (tie-start notes excluded) grouped by offset.
    note_to_vk : dict[int, tuple]
        Maps id(elem) -> voice key for O(1) lookup.
    note_start : dict[int, Fraction]
        Maps id(elem) -> start time, used for the simultaneity overlap check.
    note_end : dict[int, Fraction]
        Maps id(elem) -> end time, used for the simultaneity overlap check.
    _by_end_per_voice : dict[tuple, list[(end, start, midi, elem)]]
        Per-voice list sorted by end time, for last_sounding_at_or_before.
    _attacks_per_voice : dict[tuple, list[Fraction]]
        Per-voice sorted attack times, for had_attack_between.
    _chords_per_voice : dict[tuple, list[(start, end, members)]]
        Per-voice sorted chord list, for last_chord_at_or_before.
    """

    def __init__(self, notes):
        self.by_start        = defaultdict(list)
        self.by_end          = defaultdict(list)
        self.note_to_vk      = {}
        self.note_start      = {}
        self.note_end        = {}
        self._by_end_per_voice  = defaultdict(list)
        self._attacks_per_voice = defaultdict(list)
        # Temporary nested dict: vk -> start -> end -> [(midi, elem)]
        _chords_raw          = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for note in notes:
            s, e, m, elem, vk = note
            nid = id(elem)
            self.note_to_vk[nid] = vk
            self.note_start[nid] = s
            self.note_end[nid]   = e
            self._by_end_per_voice[vk].append((e, s, m, elem))
            _chords_raw[vk][s][e].append((m, elem))

            if not _is_tie_stop(elem):
                self.by_start[s].append((m, elem, vk))
                self._attacks_per_voice[vk].append(s)

            if not _is_tie_start(elem):
                self.by_end[e].append((m, elem, vk))

        for lst in self._by_end_per_voice.values():
            lst.sort()
        for lst in self._attacks_per_voice.values():
            lst.sort()

        self._chords_per_voice = {
            vk: sorted(
                (
                    (e, s, members)
                    for s, by_e in by_s.items()
                    for e, members in by_e.items()
                ),
                key=lambda chord: (chord[0], chord[1]),
            )
            for vk, by_s in _chords_raw.items()
        }

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def last_sounding_at_or_before(self, vk, t):
        """Return (end, midi, elem) of the most recent non-tie-continued note
        in voice `vk` whose sound ended at or before `t`, or None."""
        nl  = self._by_end_per_voice.get(vk)
        if not nl:
            return None
        idx = bisect_left(nl, (t, _BISECT_SENTINEL)) - 1
        while idx >= 0:
            e, s, m, elem = nl[idx]
            if not _is_tie_start(elem):
                return (e, m, elem)
            idx -= 1
        return None

    def last_chord_at_or_before(self, vk, t):
        """Return (start, end, members) of the most recent non-tie-continued
        chord in voice `vk` ending at or before `t`, or None.

        The chord list is sorted by end time (primary key), so we bisect to
        find the boundary directly.  `_BISECT_SENTINEL` is any value that
        sorts strictly after every realistic start time, so `(t, sentinel)`
        sorts after every chord with end == t.
        """
        cl = self._chords_per_voice.get(vk)
        if not cl:
            return None
        idx = bisect_left(cl, (t, _BISECT_SENTINEL)) - 1
        while idx >= 0:
            ce, cs, members = cl[idx]
            if not any(_is_tie_start(el) for _, el in members):
                return (cs, ce, members)
            idx -= 1
        return None

    def had_attack_between(self, vk, after, before):
        """True if voice `vk` has any real attack strictly between `after` and `before`."""
        attacks = self._attacks_per_voice.get(vk)
        if not attacks:
            return False
        # bisect_right(attacks, after) gives the first index strictly after `after`.
        # bisect_left(attacks, before) gives the first index >= `before`.
        # Any index in [lo, hi) is an attack in (after, before).
        lo = bisect_right(attacks, after)
        hi = bisect_left(attacks, before)
        return lo < hi

    def were_simultaneous(self, elem_a, elem_b):
        """True if the two note elements overlapped in time.

        The two notes must have been simultaneously sounding for them to have
        jointly formed an interval.  Notes that never overlapped can only be
        connected by a coincidental pitch-class match across a long voice
        absence — not a genuine consecutive interval.
        """
        nid_a, nid_b = id(elem_a), id(elem_b)
        return (self.note_start[nid_a] < self.note_end[nid_b] and
                self.note_start[nid_b] < self.note_end[nid_a])


# ---------------------------------------------------------------------------
# Interval pair helpers
# ---------------------------------------------------------------------------

def _interval_pairs(group, semitones):
    """Return all IntervalPairs in `group` forming the target interval.

    group is a list of (midi, elem, vk) entries all sounding at the same time.
    Same-voice octave doublings (e.g. piano reinforcement) are excluded.
    """
    pairs = []
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            midi_a, elem_a, vk_a = group[i]
            midi_b, elem_b, vk_b = group[j]
            if not forms_interval(midi_a, midi_b, semitones):
                continue
            if semitones == 0 and vk_a == vk_b:
                continue   # same-voice octave doubling — not a violation
            pairs.append(IntervalPair(elem_a, elem_b, frozenset({vk_a, vk_b})))
    return pairs


# ---------------------------------------------------------------------------
# Union-find
# ---------------------------------------------------------------------------

class _UnionFind:
    """Iterative path-compressed union-find over arbitrary hashable ids."""

    def __init__(self):
        self._parent = {}

    def find(self, x):
        if x not in self._parent:
            self._parent[x] = x
        # Two-pass iterative path compression.
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px != py:
            self._parent[px] = py

    def flag(self, elems):
        """Register all elements and union them into one component."""
        nids = [id(e) for e in elems]
        for nid in nids:
            self.find(nid)           # register if new
        for nid in nids[1:]:
            self.union(nids[0], nid)

    def components(self):
        """Return {id: root_id} for every registered id."""
        return {x: self.find(x) for x in self._parent}


# ---------------------------------------------------------------------------
# Violation detection
# ---------------------------------------------------------------------------

def _check_rest_lookthrough_inter(pair, t2, semitones, idx, uf):
    """Rest look-through for a pair whose notes are in different voice streams.

    Flags the previous + new notes if:
      - the previous notes in each voice formed the same interval,
      - those previous notes actually overlapped in time,
      - neither voice attacked anything between its previous note and t2.
    """
    vk_a = idx.note_to_vk[id(pair.note_a)]
    vk_b = idx.note_to_vk[id(pair.note_b)]

    prev_a = idx.last_sounding_at_or_before(vk_a, t2)
    prev_b = idx.last_sounding_at_or_before(vk_b, t2)
    if prev_a is None or prev_b is None:
        return

    end_a, midi_a, elem_a = prev_a
    end_b, midi_b, elem_b = prev_b

    # The direct-boundary path handles the case where both ended exactly at t2.
    if end_a == t2 and end_b == t2:
        return

    if not forms_interval(midi_a, midi_b, semitones):
        return

    # The two previous notes must have sounded simultaneously — otherwise the
    # "previous interval" is a phantom created by a long voice absence.
    if not idx.were_simultaneous(elem_a, elem_b):
        return

    # No intervening attack in either voice: any note between the previous
    # release and the new attack breaks the consecutive relationship.
    if idx.had_attack_between(vk_a, end_a, t2):
        return
    if idx.had_attack_between(vk_b, end_b, t2):
        return

    uf.flag([elem_a, elem_b, pair.note_a, pair.note_b])


def _check_rest_lookthrough_intra(pair, t2, semitones, vk, idx, uf):
    """Rest look-through for a pair whose notes are chord members in one voice.

    When both notes of an interval pair share a voice key (they are simultaneous
    chord members), the standard per-note lookback collapses to a single note.
    Instead we look back at the previous chord as a unit and search it for a
    matching interval pair.
    """
    prev = idx.last_chord_at_or_before(vk, t2)
    if prev is None:
        return

    prev_start, prev_end, members = prev

    # The direct-boundary case is already handled; skip if no actual gap.
    if prev_end == t2:
        return

    # No intervening attack in this voice.
    if idx.had_attack_between(vk, prev_end, t2):
        return

    # Same voice → the previous chord and new chord are in the same stream,
    # so simultaneity is guaranteed; no overlap check needed.
    #
    # No same-voice octave-doubling guard is needed here (unlike _interval_pairs):
    # this path is only reached when the new pair shares a voice key, and
    # _interval_pairs filters same-voice octave pairs out of `starting`, so the
    # intra path runs for fifths only.
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            midi_i, elem_i = members[i]
            midi_j, elem_j = members[j]
            if not forms_interval(midi_i, midi_j, semitones):
                continue
            uf.flag([elem_i, elem_j, pair.note_a, pair.note_b])


def find_note_colors(notes, intervals=('fifths',)):
    """Return ({id(note_elem): color_hex}, n_groups) for every note involved in a consecutive violation.

    Note on identity: id(elem) is used as a note identity token throughout.
    This is safe because all note elements remain alive for the full duration
    of this function (they are referenced in `notes`), so no two live elements
    can share an id.
    """
    idx = VoiceIndex(notes)
    uf  = _UnionFind()

    for interval_name in intervals:
        semitones = INTERVAL_SEMITONES[interval_name]

        # Pre-compute which time points have interval pairs ending / starting.
        ending  = {t: ps for t, g in idx.by_end.items()
                   if (ps := _interval_pairs(g, semitones))}
        starting = {t: ps for t, g in idx.by_start.items()
                   if (ps := _interval_pairs(g, semitones))}

        # --- Direct boundary -------------------------------------------------
        # The same interval (same voice pair) ends at t and begins again at t.
        for t in set(ending) & set(starting):
            for end_pair in ending[t]:
                for start_pair in starting[t]:
                    if end_pair.voices == start_pair.voices:
                        uf.flag([end_pair.note_a, end_pair.note_b,
                                 start_pair.note_a, start_pair.note_b])

        # --- Rest look-through -----------------------------------------------
        for t2, pairs in starting.items():
            for pair in pairs:
                vk_a = idx.note_to_vk[id(pair.note_a)]
                vk_b = idx.note_to_vk[id(pair.note_b)]

                if vk_a != vk_b:
                    _check_rest_lookthrough_inter(pair, t2, semitones, idx, uf)
                else:
                    _check_rest_lookthrough_intra(pair, t2, semitones, vk_a, idx, uf)

    # Assign one palette color per connected component.
    component_color   = {}
    palette_exhausted = False
    result            = {}
    for nid, root_id in uf.components().items():
        if root_id not in component_color:
            color_idx = len(component_color)
            if color_idx >= len(PALETTE) and not palette_exhausted:
                warnings.warn(
                    f'More than {len(PALETTE)} independent violation groups found; '
                    f'palette colors will repeat and groups may be visually '
                    f'indistinguishable.',
                    stacklevel=2,
                )
                palette_exhausted = True
            component_color[root_id] = PALETTE[color_idx % len(PALETTE)]
        result[nid] = component_color[root_id]

    return result, len(component_color)


# ---------------------------------------------------------------------------
# Colorization
# ---------------------------------------------------------------------------

def colorize(note_elem, color):
    """Apply `color` to a note element and its explicit child stem/notehead.

    The `color` attribute on <note> (MusicXML 3.1 §6.7.1) is the primary
    mechanism: renderers apply it to the entire notehead regardless of whether
    a <notehead> child is present.  This reliably covers chord members that
    carry no <stem> of their own.

    <stem> and <notehead> children are also colored when present, for renderers
    that only honour child-element attributes.  A synthetic <notehead> is NOT
    inserted when absent: doing so confuses some renderers and produces
    visually isolated (singleton-looking) chord members.
    """
    note_elem.set('color', color)

    stem = note_elem.find('stem')
    if stem is not None:
        stem.set('color', color)

    notehead = note_elem.find('notehead')
    if notehead is not None:
        notehead.set('color', color)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Detect consecutive perfect intervals in MusicXML and colorize them.',
    )
    parser.add_argument(
        'input',
        metavar='input.xml',
        help="Input MusicXML file, or '-' for stdin.",
    )
    parser.add_argument(
        'output',
        metavar='output.xml',
        help="Output MusicXML file, or '-' for stdout.",
    )
    parser.add_argument(
        '--interval',
        choices=['fifths', 'octaves', 'both'],
        default='fifths',
        help='Which consecutive intervals to detect (default: fifths)',
    )
    args = parser.parse_args()

    intervals = ['fifths', 'octaves'] if args.interval == 'both' else [args.interval]

    if args.input == '-':
        raw = sys.stdin.read()
    else:
        with open(args.input, encoding='utf-8', errors='replace') as f:
            raw = f.read()

    # Preserve the original XML prologue (declaration, DOCTYPE, etc.) verbatim.
    # ElementTree drops or rewrites it, which can corrupt downstream tooling.
    m = re.search(r'<score-(partwise|timewise)\b', raw)
    prologue = raw[:m.start()] if m else '<?xml version="1.0" encoding="UTF-8"?>\n'

    # Parse from the in-memory string so stdin reads once and works the same
    # as a file path.  ET.parse(path) would re-read the file, which fails for '-'.
    root = ET.fromstring(raw)

    notes       = collect_notes(root)
    note_colors, n_groups = find_note_colors(notes, intervals=intervals)

    for note in notes:
        color = note_colors.get(id(note.elem))
        if color:
            colorize(note.elem, color)

    body = ET.tostring(root, encoding='unicode', xml_declaration=False)
    if args.output == '-':
        sys.stdout.write(prologue)
        sys.stdout.write(body)
    else:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(prologue)
            f.write(body)

    # Progress message goes to stderr so it doesn't pollute stdout when the
    # output XML is piped to another tool via `-`.
    n_notes  = len(note_colors)
    label    = {'fifths': 'fifth', 'octaves': 'octave', 'both': 'fifth/octave'}[args.interval]
    print(
        f'{n_notes} note(s) in {n_groups} consecutive-{label} group(s) colorized.',
        file=sys.stderr,
    )


if __name__ == '__main__':
    main()
