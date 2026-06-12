"""Comprehensive test suite for conseq.py."""

import io
import sys
import warnings
import xml.etree.ElementTree as ET
from fractions import Fraction

import pytest

import conseq
from conseq import (
    NoteInfo,
    VoiceIndex,
    _UnionFind,
    _check_rest_lookthrough_inter,
    _check_rest_lookthrough_intra,
    _interval_pairs,
    _is_tie_start,
    _is_tie_stop,
    annotate_groups,
    build_note_locations,
    collect_notes,
    colorize,
    find_note_colors,
    find_violation_groups,
    forms_interval,
    make_navigation_marker,
    pitch_to_midi,
)

# ---------------------------------------------------------------------------
# MusicXML builder helpers
# ---------------------------------------------------------------------------

def make_note(step=None, octave=4, alter=None, duration=1, voice='1',
              staff=None, grace=False, chord=False, tie_types=()):
    """Build a <note> element. step=None → rest."""
    n = ET.Element('note')
    if grace:
        ET.SubElement(n, 'grace')
    if chord:
        ET.SubElement(n, 'chord')
    if step is not None:
        p = ET.SubElement(n, 'pitch')
        ET.SubElement(p, 'step').text = step
        ET.SubElement(p, 'octave').text = str(octave)
        if alter is not None:
            ET.SubElement(p, 'alter').text = str(alter)
    else:
        ET.SubElement(n, 'rest')
    ET.SubElement(n, 'duration').text = str(duration)
    ET.SubElement(n, 'voice').text = str(voice)
    if staff is not None:
        ET.SubElement(n, 'staff').text = str(staff)
    for ty in tie_types:
        t = ET.SubElement(n, 'tie')
        t.set('type', ty)
    return n


def make_measure(number='1', divisions=1, notes=(), forward=None, backup=None):
    """Build a <measure> element."""
    m = ET.Element('measure')
    m.set('number', str(number))
    attrs = ET.SubElement(m, 'attributes')
    ET.SubElement(attrs, 'divisions').text = str(divisions)
    for n in notes:
        m.append(n)
    if forward is not None:
        fw = ET.SubElement(m, 'forward')
        ET.SubElement(fw, 'duration').text = str(forward)
    if backup is not None:
        bk = ET.SubElement(m, 'backup')
        ET.SubElement(bk, 'duration').text = str(backup)
    return m


def make_score(parts):
    """Build <score-partwise>. parts = [(part_id, [measure, ...]), ...]"""
    root = ET.Element('score-partwise')
    for pid, measures in parts:
        part = ET.SubElement(root, 'part')
        part.set('id', pid)
        for m in measures:
            part.append(m)
    return root


def two_voice_score(voice1_notes, voice2_notes, divisions=1, part_id='P1'):
    """Single-part, single-measure score with two voices separated by <backup>."""
    measure = ET.Element('measure')
    measure.set('number', '1')
    attrs = ET.SubElement(measure, 'attributes')
    ET.SubElement(attrs, 'divisions').text = str(divisions)
    total_dur = sum(
        int(n.find('duration').text)
        for n in voice1_notes
        if n.find('chord') is None
    )
    for n in voice1_notes:
        measure.append(n)
    bk = ET.SubElement(measure, 'backup')
    ET.SubElement(bk, 'duration').text = str(total_dur)
    for n in voice2_notes:
        measure.append(n)
    root = ET.Element('score-partwise')
    part = ET.SubElement(root, 'part')
    part.set('id', part_id)
    part.append(measure)
    return root


def score_to_xml(root, prologue='<?xml version="1.0" encoding="UTF-8"?>\n'):
    """Serialise an ET root to a full XML string."""
    return prologue + ET.tostring(root, encoding='unicode')


def parallel_fifths_score():
    """C4(60)/G4(67) → D4(62)/A4(69) in two voices — direct boundary fifth violation."""
    v1 = [make_note('C', 4, voice='1', duration=1),
          make_note('D', 4, voice='1', duration=1)]
    v2 = [make_note('G', 4, voice='2', duration=1),
          make_note('A', 4, voice='2', duration=1)]
    return two_voice_score(v1, v2)


def parallel_octaves_score():
    """C4/C5 → D4/D5 in two voices — direct boundary octave violation."""
    v1 = [make_note('C', 4, voice='1', duration=1),
          make_note('D', 4, voice='1', duration=1)]
    v2 = [make_note('C', 5, voice='2', duration=1),
          make_note('D', 5, voice='2', duration=1)]
    return two_voice_score(v1, v2)


def _run_main(monkeypatch, capsys, argv, stdin_text=None):
    """Run conseq.main() with patched argv/stdin; return (out, err)."""
    monkeypatch.setattr(sys, 'argv', ['conseq.py'] + argv)
    if stdin_text is not None:
        monkeypatch.setattr(sys, 'stdin', io.StringIO(stdin_text))
    conseq.main()
    return capsys.readouterr()


# ---------------------------------------------------------------------------
# TestPitchToMidi
# ---------------------------------------------------------------------------

class TestPitchToMidi:
    def _note_with_pitch(self, step, octave, alter=None):
        n = ET.Element('note')
        p = ET.SubElement(n, 'pitch')
        ET.SubElement(p, 'step').text = step
        ET.SubElement(p, 'octave').text = str(octave)
        if alter is not None:
            ET.SubElement(p, 'alter').text = str(alter)
        return n

    def _rest_note(self):
        n = ET.Element('note')
        ET.SubElement(n, 'rest')
        return n

    def _empty_note(self):
        return ET.Element('note')

    def test_rest_returns_none(self):
        assert pitch_to_midi(self._rest_note()) is None

    def test_unpitched_percussion_returns_none(self):
        assert pitch_to_midi(self._empty_note()) is None

    def test_c4(self):
        assert pitch_to_midi(self._note_with_pitch('C', 4)) == 60

    def test_a4(self):
        assert pitch_to_midi(self._note_with_pitch('A', 4)) == 69

    def test_g_sharp_5(self):
        # G5 = (5+1)*12 + 7 = 79; +1 semitone = 80
        assert pitch_to_midi(self._note_with_pitch('G', 5, alter=1)) == 80

    def test_c_flat_4(self):
        assert pitch_to_midi(self._note_with_pitch('C', 4, alter=-1)) == 59

    def test_alter_absent_defaults_zero(self):
        assert pitch_to_midi(self._note_with_pitch('D', 4)) == 62

    def test_alter_rounds_positive_float(self):
        # alter=0.6 → round(0.6)=1; C4(60)+1=61
        assert pitch_to_midi(self._note_with_pitch('C', 4, alter=0.6)) == 61

    def test_alter_rounds_negative_float(self):
        # alter=-0.6 → round(-0.6)=-1; C4(60)-1=59
        assert pitch_to_midi(self._note_with_pitch('C', 4, alter=-0.6)) == 59

    def test_c0(self):
        assert pitch_to_midi(self._note_with_pitch('C', 0)) == 12

    def test_all_natural_steps_octave4(self):
        expected = {'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71}
        for step, midi in expected.items():
            assert pitch_to_midi(self._note_with_pitch(step, 4)) == midi

    def test_b4_sharp_is_c5(self):
        assert pitch_to_midi(self._note_with_pitch('B', 4, alter=1)) == 72


# ---------------------------------------------------------------------------
# TestFormsInterval
# ---------------------------------------------------------------------------

class TestFormsInterval:
    def test_perfect_fifth_above(self):
        assert forms_interval(60, 67, 7) is True

    def test_perfect_fifth_below(self):
        assert forms_interval(67, 60, 7) is True

    def test_compound_fifth(self):
        # C4=60, G5=79 → diff=19, 19%12=7
        assert forms_interval(60, 79, 7) is True

    def test_non_fifth(self):
        assert forms_interval(60, 69, 7) is False   # major sixth

    def test_octave_above(self):
        assert forms_interval(60, 72, 0) is True

    def test_double_octave(self):
        assert forms_interval(60, 84, 0) is True

    def test_unison_excluded_from_octave(self):
        assert forms_interval(60, 60, 0) is False   # diff==0

    def test_non_octave_multiple(self):
        assert forms_interval(60, 74, 0) is False   # diff=14, 14%12=2

    def test_fifths_semitone_unison_not_matched(self):
        assert forms_interval(60, 60, 7) is False   # diff%12==0 ≠ 7


# ---------------------------------------------------------------------------
# TestTieHelpers
# ---------------------------------------------------------------------------

class TestTieHelpers:
    def _note_with_ties(self, *types):
        n = ET.Element('note')
        for ty in types:
            t = ET.SubElement(n, 'tie')
            t.set('type', ty)
        return n

    def test_is_tie_start_true(self):
        assert _is_tie_start(self._note_with_ties('start')) is True

    def test_is_tie_start_false_no_ties(self):
        assert _is_tie_start(self._note_with_ties()) is False

    def test_is_tie_start_false_only_stop(self):
        assert _is_tie_start(self._note_with_ties('stop')) is False

    def test_is_tie_stop_true(self):
        assert _is_tie_stop(self._note_with_ties('stop')) is True

    def test_is_tie_stop_false_only_start(self):
        assert _is_tie_stop(self._note_with_ties('start')) is False

    def test_both_start_and_stop(self):
        n = self._note_with_ties('start', 'stop')
        assert _is_tie_start(n) is True
        assert _is_tie_stop(n) is True

    def test_unrecognized_type(self):
        n = self._note_with_ties('continue')
        assert _is_tie_start(n) is False
        assert _is_tie_stop(n) is False


# ---------------------------------------------------------------------------
# TestCollectNotes
# ---------------------------------------------------------------------------

class TestCollectNotes:
    def _score_with_notes(self, *note_elems, divisions=1, part_id='P1'):
        return make_score([(part_id, [make_measure(divisions=divisions, notes=note_elems)])])

    def test_single_pitched_note_fields(self):
        root = self._score_with_notes(make_note('C', 4, duration=1))
        notes = collect_notes(root)
        assert len(notes) == 1
        n = notes[0]
        assert n.start == Fraction(0)
        assert n.end == Fraction(1)
        assert n.midi == 60
        assert n.vk == ('P1', '1', '1')

    def test_rest_excluded(self):
        root = self._score_with_notes(make_note(None, duration=1))
        assert collect_notes(root) == []

    def test_grace_note_excluded(self):
        root = self._score_with_notes(make_note('C', 4, grace=True))
        assert collect_notes(root) == []

    def test_note_without_duration_excluded(self):
        n = ET.Element('note')
        p = ET.SubElement(n, 'pitch')
        ET.SubElement(p, 'step').text = 'C'
        ET.SubElement(p, 'octave').text = '4'
        # no <duration>
        root = self._score_with_notes(n)
        assert collect_notes(root) == []

    def test_cursor_advances_for_sequential_notes(self):
        root = self._score_with_notes(
            make_note('C', 4, duration=1),
            make_note('D', 4, duration=1),
        )
        notes = collect_notes(root)
        assert notes[0].start == Fraction(0)
        assert notes[1].start == Fraction(1)

    def test_chord_member_shares_start(self):
        root = self._score_with_notes(
            make_note('C', 4, duration=1, voice='1'),
            make_note('E', 4, duration=1, voice='1', chord=True),
        )
        notes = collect_notes(root)
        assert len(notes) == 2
        assert notes[0].start == notes[1].start == Fraction(0)

    def test_chord_cursor_not_advanced(self):
        root = self._score_with_notes(
            make_note('C', 4, duration=1, voice='1'),
            make_note('E', 4, duration=1, voice='1', chord=True),
            make_note('G', 4, duration=1, voice='1'),
        )
        notes = collect_notes(root)
        # Third note starts at 1, not 2
        assert notes[2].start == Fraction(1)

    def test_voice_key_defaults_to_one(self):
        n = ET.Element('note')
        p = ET.SubElement(n, 'pitch')
        ET.SubElement(p, 'step').text = 'C'
        ET.SubElement(p, 'octave').text = '4'
        ET.SubElement(n, 'duration').text = '1'
        root = self._score_with_notes(n)
        notes = collect_notes(root)
        assert notes[0].vk == ('P1', '1', '1')

    def test_voice_key_with_staff(self):
        root = self._score_with_notes(make_note('C', 4, staff='2'))
        notes = collect_notes(root)
        assert notes[0].vk[1] == '2'

    def test_voice_key_with_voice(self):
        root = self._score_with_notes(make_note('C', 4, voice='3'))
        notes = collect_notes(root)
        assert notes[0].vk[2] == '3'

    def test_divisions_changes_duration_scaling(self):
        # With divisions=2, duration=1 → Fraction(1, 2) quarter notes
        root = self._score_with_notes(
            make_note('C', 4, duration=2),
            make_note('D', 4, duration=2),
            divisions=2,
        )
        notes = collect_notes(root)
        assert notes[0].end == Fraction(1)
        assert notes[1].start == Fraction(1)

    def test_forward_advances_cursor(self):
        m = ET.Element('measure')
        m.set('number', '1')
        attrs = ET.SubElement(m, 'attributes')
        ET.SubElement(attrs, 'divisions').text = '1'
        fwd = ET.SubElement(m, 'forward')
        ET.SubElement(fwd, 'duration').text = '2'
        m.append(make_note('C', 4, duration=1))
        root = make_score([('P1', [m])])
        notes = collect_notes(root)
        assert notes[0].start == Fraction(2)

    def test_backup_retreats_cursor(self):
        root = two_voice_score(
            [make_note('C', 4, duration=2, voice='1')],
            [make_note('G', 4, duration=1, voice='2')],
        )
        notes = collect_notes(root)
        v2_note = next(n for n in notes if n.vk[2] == '2')
        assert v2_note.start == Fraction(0)

    def test_two_part_score_part_ids(self):
        root = make_score([
            ('P1', [make_measure(notes=[make_note('C', 4)])]),
            ('P2', [make_measure(notes=[make_note('G', 4)])]),
        ])
        notes = collect_notes(root)
        part_ids = {n.vk[0] for n in notes}
        assert part_ids == {'P1', 'P2'}

    def test_multiple_measures_cursor_continuity(self):
        m1 = make_measure(number='1', notes=[make_note('C', 4, duration=4)])
        m2 = make_measure(number='2', notes=[make_note('D', 4, duration=4)])
        root = make_score([('P1', [m1, m2])])
        notes = collect_notes(root)
        assert notes[0].start == Fraction(0)
        assert notes[1].start == Fraction(4)

    def test_fraction_arithmetic_non_unit_duration(self):
        root = self._score_with_notes(make_note('C', 4, duration=3), divisions=2)
        notes = collect_notes(root)
        assert notes[0].end - notes[0].start == Fraction(3, 2)

    def test_alter_sharp_midi(self):
        root = self._score_with_notes(make_note('F', 4, alter=1))
        notes = collect_notes(root)
        assert notes[0].midi == 66   # F#4


# ---------------------------------------------------------------------------
# Helpers shared by VoiceIndex / RestLookthrough tests
# ---------------------------------------------------------------------------

def _make_noteinfo(start, end, midi, voice='1', part='P1', staff='1', **tie_types):
    """Build a NoteInfo with a real ET.Element that carries optional tie markup."""
    elem = make_note(step='C', octave=4, duration=1,
                     tie_types=tie_types.get('tie_types', ()))
    vk = (part, staff, voice)
    return NoteInfo(Fraction(start), Fraction(end), midi, elem, vk)


def _build_index(*noteinfos):
    return VoiceIndex(list(noteinfos))


# ---------------------------------------------------------------------------
# TestVoiceIndexConstruction
# ---------------------------------------------------------------------------

class TestVoiceIndexConstruction:
    def test_empty_notes(self):
        idx = VoiceIndex([])
        assert idx.by_start == {}
        assert idx.by_end == {}

    def test_normal_note_in_by_start_and_by_end(self):
        ni = _make_noteinfo(0, 1, 60)
        idx = _build_index(ni)
        assert Fraction(0) in idx.by_start
        assert Fraction(1) in idx.by_end

    def test_tie_stop_excluded_from_by_start(self):
        elem = make_note('C', 4, tie_types=('stop',))
        ni = NoteInfo(Fraction(0), Fraction(1), 60, elem, ('P1', '1', '1'))
        idx = VoiceIndex([ni])
        assert Fraction(0) not in idx.by_start

    def test_tie_start_excluded_from_by_end(self):
        elem = make_note('C', 4, tie_types=('start',))
        ni = NoteInfo(Fraction(0), Fraction(1), 60, elem, ('P1', '1', '1'))
        idx = VoiceIndex([ni])
        assert Fraction(1) not in idx.by_end

    def test_note_to_vk_populated(self):
        ni = _make_noteinfo(0, 1, 60, voice='2')
        idx = _build_index(ni)
        assert idx.note_to_vk[id(ni.elem)] == ('P1', '1', '2')

    def test_note_start_end_populated(self):
        ni = _make_noteinfo(2, 5, 60)
        idx = _build_index(ni)
        assert idx.note_start[id(ni.elem)] == Fraction(2)
        assert idx.note_end[id(ni.elem)] == Fraction(5)

    def test_attacks_per_voice_excludes_tie_stop(self):
        elem = make_note('C', 4, tie_types=('stop',))
        ni = NoteInfo(Fraction(1), Fraction(2), 60, elem, ('P1', '1', '1'))
        idx = VoiceIndex([ni])
        assert not idx.had_attack_between(('P1', '1', '1'), Fraction(0), Fraction(3))


# ---------------------------------------------------------------------------
# TestLastSoundingAtOrBefore
# ---------------------------------------------------------------------------

class TestLastSoundingAtOrBefore:
    def test_unknown_vk_returns_none(self):
        idx = VoiceIndex([])
        assert idx.last_sounding_at_or_before(('P1', '1', '1'), Fraction(5)) is None

    def test_note_ending_at_t_found(self):
        ni = _make_noteinfo(0, 1, 60)
        idx = _build_index(ni)
        result = idx.last_sounding_at_or_before(('P1', '1', '1'), Fraction(1))
        assert result is not None
        assert result[0] == Fraction(1)
        assert result[1] == 60

    def test_note_ending_after_t_not_found(self):
        ni = _make_noteinfo(0, 3, 60)
        idx = _build_index(ni)
        assert idx.last_sounding_at_or_before(('P1', '1', '1'), Fraction(1)) is None

    def test_tie_start_note_skipped(self):
        elem_tie = make_note('C', 4, tie_types=('start',))
        elem_plain = make_note('G', 4)
        vk = ('P1', '1', '1')
        ni_plain = NoteInfo(Fraction(0), Fraction(1), 67, elem_plain, vk)
        ni_tie   = NoteInfo(Fraction(1), Fraction(2), 60, elem_tie,   vk)
        idx = VoiceIndex([ni_plain, ni_tie])
        result = idx.last_sounding_at_or_before(vk, Fraction(2))
        assert result[1] == 67   # tie-start skipped; plain note returned

    def test_all_notes_tie_start_returns_none(self):
        elem = make_note('C', 4, tie_types=('start',))
        vk = ('P1', '1', '1')
        ni = NoteInfo(Fraction(0), Fraction(1), 60, elem, vk)
        idx = VoiceIndex([ni])
        assert idx.last_sounding_at_or_before(vk, Fraction(1)) is None

    def test_most_recent_of_two_notes(self):
        vk = ('P1', '1', '1')
        ni1 = _make_noteinfo(0, 1, 60)
        ni2 = NoteInfo(Fraction(1), Fraction(2), 62, make_note('D', 4), vk)
        idx = VoiceIndex([ni1, ni2])
        result = idx.last_sounding_at_or_before(vk, Fraction(3))
        assert result[1] == 62

    def test_exact_boundary_included(self):
        ni = _make_noteinfo(0, 2, 60)
        idx = _build_index(ni)
        result = idx.last_sounding_at_or_before(('P1', '1', '1'), Fraction(2))
        assert result is not None


# ---------------------------------------------------------------------------
# TestLastChordAtOrBefore
# ---------------------------------------------------------------------------

class TestLastChordAtOrBefore:
    def test_unknown_vk_returns_none(self):
        idx = VoiceIndex([])
        assert idx.last_chord_at_or_before(('P1', '1', '1'), Fraction(5)) is None

    def test_chord_ending_at_t_found(self):
        vk = ('P1', '1', '1')
        ni1 = NoteInfo(Fraction(0), Fraction(1), 60, make_note('C', 4), vk)
        ni2 = NoteInfo(Fraction(0), Fraction(1), 67, make_note('G', 4), vk)
        idx = VoiceIndex([ni1, ni2])
        result = idx.last_chord_at_or_before(vk, Fraction(1))
        assert result is not None
        _, ce, members = result
        assert ce == Fraction(1)
        assert len(members) == 2

    def test_chord_ending_after_t_not_returned(self):
        vk = ('P1', '1', '1')
        ni = NoteInfo(Fraction(0), Fraction(3), 60, make_note('C', 4), vk)
        idx = VoiceIndex([ni])
        assert idx.last_chord_at_or_before(vk, Fraction(1)) is None

    def test_tie_start_chord_skipped(self):
        vk = ('P1', '1', '1')
        e1 = make_note('C', 4, tie_types=('start',))
        e2 = make_note('G', 4)
        ni_tie   = NoteInfo(Fraction(1), Fraction(2), 60, e1, vk)
        ni_plain = NoteInfo(Fraction(0), Fraction(1), 67, e2, vk)
        idx = VoiceIndex([ni_plain, ni_tie])
        result = idx.last_chord_at_or_before(vk, Fraction(2))
        # tie-start chord skipped; plain chord at [0,1) returned
        assert result is not None
        _, ce, _ = result
        assert ce == Fraction(1)

    def test_correct_members_returned(self):
        vk = ('P1', '1', '1')
        e1 = make_note('C', 4)
        e2 = make_note('E', 4)
        ni1 = NoteInfo(Fraction(0), Fraction(1), 60, e1, vk)
        ni2 = NoteInfo(Fraction(0), Fraction(1), 64, e2, vk)
        idx = VoiceIndex([ni1, ni2])
        _, _, members = idx.last_chord_at_or_before(vk, Fraction(1))
        midis = {m for m, _ in members}
        assert midis == {60, 64}


# ---------------------------------------------------------------------------
# TestHadAttackBetween
# ---------------------------------------------------------------------------

class TestHadAttackBetween:
    def _idx_with_attacks(self, *starts):
        vk = ('P1', '1', '1')
        notes = [NoteInfo(Fraction(s), Fraction(s + 1), 60, make_note('C', 4), vk)
                 for s in starts]
        return VoiceIndex(notes), vk

    def test_no_attacks_false(self):
        idx, vk = self._idx_with_attacks()
        assert idx.had_attack_between(vk, Fraction(0), Fraction(5)) is False

    def test_attack_strictly_inside_true(self):
        idx, vk = self._idx_with_attacks(1)
        assert idx.had_attack_between(vk, Fraction(0), Fraction(2)) is True

    def test_attack_at_after_boundary_false(self):
        idx, vk = self._idx_with_attacks(1)
        assert idx.had_attack_between(vk, Fraction(1), Fraction(3)) is False

    def test_attack_at_before_boundary_false(self):
        idx, vk = self._idx_with_attacks(3)
        assert idx.had_attack_between(vk, Fraction(1), Fraction(3)) is False

    def test_no_attack_in_range_false(self):
        idx, vk = self._idx_with_attacks(0, 4)
        assert idx.had_attack_between(vk, Fraction(1), Fraction(3)) is False

    def test_tie_stop_not_counted_as_attack(self):
        vk = ('P1', '1', '1')
        elem = make_note('C', 4, tie_types=('stop',))
        ni = NoteInfo(Fraction(1), Fraction(2), 60, elem, vk)
        idx = VoiceIndex([ni])
        assert idx.had_attack_between(vk, Fraction(0), Fraction(3)) is False


# ---------------------------------------------------------------------------
# TestWereSimultaneous
# ---------------------------------------------------------------------------

class TestWereSimultaneous:
    def _two_notes(self, s1, e1, s2, e2):
        vk = ('P1', '1', '1')
        a = NoteInfo(Fraction(s1), Fraction(e1), 60, make_note('C', 4), vk)
        b = NoteInfo(Fraction(s2), Fraction(e2), 67, make_note('G', 4), vk)
        idx = VoiceIndex([a, b])
        return idx, a.elem, b.elem

    def test_overlapping_true(self):
        idx, a, b = self._two_notes(0, 2, 1, 3)
        assert idx.were_simultaneous(a, b) is True

    def test_adjacent_not_overlapping(self):
        idx, a, b = self._two_notes(0, 1, 1, 2)
        assert idx.were_simultaneous(a, b) is False

    def test_gap_not_overlapping(self):
        idx, a, b = self._two_notes(0, 1, 2, 3)
        assert idx.were_simultaneous(a, b) is False

    def test_simultaneous_start(self):
        idx, a, b = self._two_notes(0, 2, 0, 1)
        assert idx.were_simultaneous(a, b) is True

    def test_contained_note(self):
        idx, a, b = self._two_notes(0, 4, 1, 2)
        assert idx.were_simultaneous(a, b) is True


# ---------------------------------------------------------------------------
# TestIntervalPairs
# ---------------------------------------------------------------------------

class TestIntervalPairs:
    def _group(self, *entries):
        # entries: (midi, vk_suffix) where vk = ('P1', '1', vk_suffix)
        return [(m, make_note('C', 4), ('P1', '1', str(v))) for m, v in entries]

    def test_empty_group(self):
        assert _interval_pairs([], 7) == []

    def test_single_entry(self):
        g = self._group((60, '1'))
        assert _interval_pairs(g, 7) == []

    def test_fifth_pair_detected(self):
        g = self._group((60, '1'), (67, '2'))
        pairs = _interval_pairs(g, 7)
        assert len(pairs) == 1

    def test_no_fifth_in_group(self):
        g = self._group((60, '1'), (62, '2'))   # major second
        assert _interval_pairs(g, 7) == []

    def test_same_vk_octave_excluded(self):
        g = self._group((60, '1'), (72, '1'))   # same voice, octave
        assert _interval_pairs(g, 0) == []

    def test_different_vk_octave_included(self):
        g = self._group((60, '1'), (72, '2'))
        pairs = _interval_pairs(g, 0)
        assert len(pairs) == 1

    def test_same_vk_fifth_not_excluded(self):
        # Only same-voice octave doublings are excluded; fifths are not
        g = self._group((60, '1'), (67, '1'))
        pairs = _interval_pairs(g, 7)
        assert len(pairs) == 1

    def test_compound_fifth_found(self):
        # C4=60, G5=79 → diff=19, 19%12=7
        g = self._group((60, '1'), (79, '2'))
        assert len(_interval_pairs(g, 7)) == 1

    def test_voices_is_frozenset(self):
        g = self._group((60, '1'), (67, '2'))
        pair = _interval_pairs(g, 7)[0]
        assert isinstance(pair.voices, frozenset)

    def test_multiple_pairs(self):
        # C4(60), G4(67), D5(74): C4-G4=7✓, G4-D5=7✓, C4-D5=14%12=2✗ → 2 fifths
        g = self._group((60, '1'), (67, '2'), (74, '3'))
        fifth_pairs = _interval_pairs(g, 7)
        assert len(fifth_pairs) == 2


# ---------------------------------------------------------------------------
# TestUnionFind
# ---------------------------------------------------------------------------

class TestUnionFind:
    def test_find_registers_new_element(self):
        uf = _UnionFind()
        root = uf.find('a')
        assert root == 'a'
        assert 'a' in uf._parent

    def test_find_idempotent(self):
        uf = _UnionFind()
        assert uf.find('x') == uf.find('x')

    def test_union_merges_two_singletons(self):
        uf = _UnionFind()
        uf.union('a', 'b')
        assert uf.find('a') == uf.find('b')

    def test_union_same_component_no_op(self):
        uf = _UnionFind()
        uf.union('a', 'b')
        uf.union('a', 'b')   # should not raise
        assert uf.find('a') == uf.find('b')

    def test_path_compression(self):
        uf = _UnionFind()
        # Build chain a→b→c→d manually
        uf._parent = {'a': 'b', 'b': 'c', 'c': 'd', 'd': 'd'}
        root = uf.find('a')
        assert root == 'd'
        assert uf._parent['a'] == 'd'   # path compressed

    def test_flag_single_element(self):
        uf = _UnionFind()
        e = object()
        uf.flag([e])
        assert id(e) in uf._parent

    def test_flag_multiple_elements(self):
        uf = _UnionFind()
        a, b, c = object(), object(), object()
        uf.flag([a, b, c])
        assert uf.find(id(a)) == uf.find(id(b)) == uf.find(id(c))

    def test_flag_empty_list_no_error(self):
        uf = _UnionFind()
        uf.flag([])   # must not raise

    def test_components_returns_all_registered(self):
        uf = _UnionFind()
        uf.find('x')
        uf.find('y')
        comps = uf.components()
        assert 'x' in comps and 'y' in comps

    def test_components_two_distinct_groups(self):
        uf = _UnionFind()
        uf.union('a', 'b')
        uf.find('c')
        comps = uf.components()
        assert comps['a'] == comps['b']
        assert comps['a'] != comps['c']

    def test_transitive_union(self):
        uf = _UnionFind()
        uf.union('a', 'b')
        uf.union('b', 'c')
        assert uf.find('a') == uf.find('b') == uf.find('c')


# ---------------------------------------------------------------------------
# Helpers for rest-look-through unit tests
# ---------------------------------------------------------------------------

def _build_inter_scenario(
    prev_midi_a=60, prev_midi_b=67,   # C4, G4 — a fifth
    prev_start_a=0, prev_end_a=1,
    prev_start_b=0, prev_end_b=1,
    attack_a_at=None,                  # if set, adds an attack in voice_a
    attack_b_at=None,
    new_start=2,
    new_midi_a=62, new_midi_b=69,      # D4, A4 — another fifth
    semitones=7,
    prev_interval_override=None,       # override prev_midi_b to break interval
):
    """Build VoiceIndex, UnionFind, and a current IntervalPair for inter-voice tests."""
    vk_a = ('P1', '1', '1')
    vk_b = ('P1', '1', '2')

    elem_prev_a = make_note('C', 4)
    elem_prev_b = make_note('G', 3)
    elem_new_a  = make_note('D', 4)
    elem_new_b  = make_note('A', 3)

    if prev_interval_override is not None:
        prev_midi_b = prev_interval_override

    notes = [
        NoteInfo(Fraction(prev_start_a), Fraction(prev_end_a), prev_midi_a, elem_prev_a, vk_a),
        NoteInfo(Fraction(prev_start_b), Fraction(prev_end_b), prev_midi_b, elem_prev_b, vk_b),
        NoteInfo(Fraction(new_start),    Fraction(new_start+1), new_midi_a,  elem_new_a,  vk_a),
        NoteInfo(Fraction(new_start),    Fraction(new_start+1), new_midi_b,  elem_new_b,  vk_b),
    ]
    if attack_a_at is not None:
        e = make_note('E', 4)
        notes.append(NoteInfo(Fraction(attack_a_at), Fraction(attack_a_at+1), 64, e, vk_a))
    if attack_b_at is not None:
        e = make_note('F', 4)
        notes.append(NoteInfo(Fraction(attack_b_at), Fraction(attack_b_at+1), 65, e, vk_b))

    from conseq import IntervalPair
    pair = IntervalPair(elem_new_a, elem_new_b, frozenset({vk_a, vk_b}))
    idx = VoiceIndex(notes)
    uf  = _UnionFind()
    return pair, Fraction(new_start), semitones, idx, uf, elem_prev_a, elem_prev_b, elem_new_a, elem_new_b


# ---------------------------------------------------------------------------
# TestRestLookthroughInter
# ---------------------------------------------------------------------------

class TestRestLookthroughInter:
    def test_canonical_violation_flagged(self):
        pair, t2, sem, idx, uf, pa, pb, na, nb = _build_inter_scenario()
        _check_rest_lookthrough_inter(pair, t2, sem, idx, uf)
        comps = uf.components()
        roots = {comps[id(e)] for e in (pa, pb, na, nb)}
        assert len(roots) == 1   # all in same component

    def test_prev_a_none_no_flag(self):
        # voice_a has no previous note (only current note)
        vk_a = ('P1', '1', '1')
        vk_b = ('P1', '1', '2')
        elem_new_a = make_note('D', 4)
        elem_new_b = make_note('A', 3)
        elem_prev_b = make_note('G', 3)
        from conseq import IntervalPair
        notes = [
            NoteInfo(Fraction(0), Fraction(1), 67, elem_prev_b, vk_b),
            NoteInfo(Fraction(2), Fraction(3), 62, elem_new_a,  vk_a),
            NoteInfo(Fraction(2), Fraction(3), 69, elem_new_b,  vk_b),
        ]
        pair = IntervalPair(elem_new_a, elem_new_b, frozenset({vk_a, vk_b}))
        idx = VoiceIndex(notes)
        uf  = _UnionFind()
        _check_rest_lookthrough_inter(pair, Fraction(2), 7, idx, uf)
        assert uf.components() == {}

    def test_prev_b_none_no_flag(self):
        vk_a = ('P1', '1', '1')
        vk_b = ('P1', '1', '2')
        elem_new_a  = make_note('D', 4)
        elem_new_b  = make_note('A', 3)
        elem_prev_a = make_note('C', 4)
        from conseq import IntervalPair
        notes = [
            NoteInfo(Fraction(0), Fraction(1), 60, elem_prev_a, vk_a),
            NoteInfo(Fraction(2), Fraction(3), 62, elem_new_a,  vk_a),
            NoteInfo(Fraction(2), Fraction(3), 69, elem_new_b,  vk_b),
        ]
        pair = IntervalPair(elem_new_a, elem_new_b, frozenset({vk_a, vk_b}))
        idx = VoiceIndex(notes)
        uf  = _UnionFind()
        _check_rest_lookthrough_inter(pair, Fraction(2), 7, idx, uf)
        assert uf.components() == {}

    def test_both_end_at_t2_no_flag(self):
        # Direct-boundary case: both previous notes end at t2 → handled elsewhere
        pair, t2, sem, idx, uf, *_ = _build_inter_scenario(
            prev_end_a=2, prev_end_b=2, new_start=2)
        _check_rest_lookthrough_inter(pair, t2, sem, idx, uf)
        assert uf.components() == {}

    def test_only_one_ends_at_t2_still_processed(self):
        # end_a==t2 but end_b < t2 → not skipped; violation should still fire
        pair, t2, sem, idx, uf, pa, pb, na, nb = _build_inter_scenario(
            prev_end_a=2, prev_end_b=1, new_start=2)
        _check_rest_lookthrough_inter(pair, t2, sem, idx, uf)
        comps = uf.components()
        assert len(comps) > 0

    def test_prev_interval_differs_no_flag(self):
        # prev notes form a major third (64 semitones), not a fifth
        pair, t2, sem, idx, uf, *_ = _build_inter_scenario(
            prev_interval_override=64)   # C4 + E4 → major third
        _check_rest_lookthrough_inter(pair, t2, sem, idx, uf)
        assert uf.components() == {}

    def test_not_simultaneous_no_flag(self):
        # prev_a [0,1) and prev_b [2,3) — never overlap
        pair, t2, sem, idx, uf, *_ = _build_inter_scenario(
            prev_start_a=0, prev_end_a=1,
            prev_start_b=2, prev_end_b=3,
            new_start=4,
            new_midi_a=62, new_midi_b=69,
        )
        _check_rest_lookthrough_inter(pair, t2, sem, idx, uf)
        assert uf.components() == {}

    def test_intervening_attack_in_a_no_flag(self):
        pair, t2, sem, idx, uf, *_ = _build_inter_scenario(attack_a_at=Fraction(3, 2))
        _check_rest_lookthrough_inter(pair, t2, sem, idx, uf)
        assert uf.components() == {}

    def test_intervening_attack_in_b_no_flag(self):
        pair, t2, sem, idx, uf, *_ = _build_inter_scenario(attack_b_at=Fraction(3, 2))
        _check_rest_lookthrough_inter(pair, t2, sem, idx, uf)
        assert uf.components() == {}

    def test_grace_note_does_not_block(self):
        # Grace notes are excluded from collect_notes so never enter by_start/attacks.
        # Simulated by simply not adding a grace note to the note list — the
        # canonical scenario already has no intervening attack; just verify flagged.
        pair, t2, sem, idx, uf, pa, pb, na, nb = _build_inter_scenario()
        _check_rest_lookthrough_inter(pair, t2, sem, idx, uf)
        comps = uf.components()
        assert id(na) in comps


# ---------------------------------------------------------------------------
# TestRestLookthroughIntra
# ---------------------------------------------------------------------------

class TestRestLookthroughIntra:
    def _intra_scenario(self, prev_end=1, new_start=2, attack_at=None,
                        prev_midi_a=60, prev_midi_b=67):
        """Single-voice chord rest look-through setup."""
        from conseq import IntervalPair
        vk = ('P1', '1', '1')

        elem_prev_a = make_note('C', 4)
        elem_prev_b = make_note('G', 4)
        elem_new_a  = make_note('C', 5)
        elem_new_b  = make_note('G', 5)

        notes = [
            NoteInfo(Fraction(0),         Fraction(prev_end),  prev_midi_a, elem_prev_a, vk),
            NoteInfo(Fraction(0),         Fraction(prev_end),  prev_midi_b, elem_prev_b, vk),
            NoteInfo(Fraction(new_start), Fraction(new_start+1), 72, elem_new_a, vk),
            NoteInfo(Fraction(new_start), Fraction(new_start+1), 79, elem_new_b, vk),
        ]
        if attack_at is not None:
            e = make_note('E', 4)
            notes.append(NoteInfo(Fraction(attack_at), Fraction(attack_at+1), 64, e, vk))

        pair = IntervalPair(elem_new_a, elem_new_b, frozenset({vk}))
        idx = VoiceIndex(notes)
        uf  = _UnionFind()
        return pair, Fraction(new_start), 7, vk, idx, uf, elem_prev_a, elem_prev_b

    def test_canonical_intra_violation_flagged(self):
        pair, t2, sem, vk, idx, uf, pa, pb = self._intra_scenario()
        _check_rest_lookthrough_intra(pair, t2, sem, vk, idx, uf)
        comps = uf.components()
        assert len(comps) > 0
        assert id(pa) in comps or id(pb) in comps

    def test_no_prev_chord_no_flag(self):
        from conseq import IntervalPair
        vk = ('P1', '1', '1')
        ea = make_note('C', 5)
        eb = make_note('G', 5)
        notes = [
            NoteInfo(Fraction(0), Fraction(1), 72, ea, vk),
            NoteInfo(Fraction(0), Fraction(1), 79, eb, vk),
        ]
        pair = IntervalPair(ea, eb, frozenset({vk}))
        idx = VoiceIndex(notes)
        uf  = _UnionFind()
        _check_rest_lookthrough_intra(pair, Fraction(0), 7, vk, idx, uf)
        assert uf.components() == {}

    def test_prev_chord_ends_at_t2_no_flag(self):
        # prev_end == new_start → direct boundary, skipped
        pair, t2, sem, vk, idx, uf, *_ = self._intra_scenario(prev_end=2, new_start=2)
        _check_rest_lookthrough_intra(pair, t2, sem, vk, idx, uf)
        assert uf.components() == {}

    def test_intervening_attack_no_flag(self):
        pair, t2, sem, vk, idx, uf, *_ = self._intra_scenario(attack_at=Fraction(3, 2))
        _check_rest_lookthrough_intra(pair, t2, sem, vk, idx, uf)
        assert uf.components() == {}

    def test_prev_chord_no_matching_interval_no_flag(self):
        # prev chord is C4+D4 — a major second, not a fifth
        pair, t2, sem, vk, idx, uf, *_ = self._intra_scenario(
            prev_midi_a=60, prev_midi_b=62)
        _check_rest_lookthrough_intra(pair, t2, sem, vk, idx, uf)
        assert uf.components() == {}

    def test_all_matching_pairs_in_prev_chord_flagged(self):
        from conseq import IntervalPair
        vk = ('P1', '1', '1')
        # prev chord: C4(60), G4(67), C5(72) — two fifths: C4-G4 and G4-C5(mod12)
        ea, eb, ec = make_note('C', 4), make_note('G', 4), make_note('C', 5)
        en = make_note('D', 4)  # new attack note (any pitch, just needs pair)
        en2 = make_note('A', 4)
        notes = [
            NoteInfo(Fraction(0), Fraction(1), 60, ea, vk),
            NoteInfo(Fraction(0), Fraction(1), 67, eb, vk),
            NoteInfo(Fraction(0), Fraction(1), 72, ec, vk),
            NoteInfo(Fraction(2), Fraction(3), 62, en,  vk),
            NoteInfo(Fraction(2), Fraction(3), 69, en2, vk),
        ]
        pair = IntervalPair(en, en2, frozenset({vk}))
        idx = VoiceIndex(notes)
        uf  = _UnionFind()
        _check_rest_lookthrough_intra(pair, Fraction(2), 7, vk, idx, uf)
        comps = uf.components()
        # ea-eb form a fifth and eb-ec form a fifth; at least one prev pair flagged
        assert len(comps) > 0


# ---------------------------------------------------------------------------
# find_note_colors helpers
# ---------------------------------------------------------------------------

def _colors(root, intervals=('fifths',)):
    notes = collect_notes(root)
    return find_note_colors(notes, intervals=intervals)


def _colored_elems(root, intervals=('fifths',)):
    color_map, n_groups = _colors(root, intervals)
    return color_map, n_groups


# ---------------------------------------------------------------------------
# TestFindNoteColorsDirect
# ---------------------------------------------------------------------------

class TestFindNoteColorsDirect:
    def test_direct_boundary_fifths(self):
        root = parallel_fifths_score()
        color_map, n_groups = _colored_elems(root)
        assert n_groups == 1
        assert len(color_map) == 4   # two pairs × 2 notes each

    def test_direct_boundary_octaves(self):
        root = parallel_octaves_score()
        color_map, n_groups = _colored_elems(root, intervals=('octaves',))
        assert n_groups == 1
        assert len(color_map) == 4

    def test_no_violation_voice_pair_mismatch(self):
        # At t=0, vk1-vk2 form a fifth (C4=60, F3=53, diff=7).
        # At t=1, vk1-vk3 form a fifth (D4=62, G3=55, diff=7).
        # Ending pair {vk1,vk2} differs from starting pair {vk1,vk3} → no direct boundary.
        # Rest look-through: vk3 has no previous note → no flag.
        vk1 = ('P1', '1', '1')
        vk2 = ('P1', '1', '2')
        vk3 = ('P1', '1', '3')
        ea = make_note('C', 4); eb = make_note('F', 3)
        ec = make_note('D', 4); ed = make_note('G', 3)
        notes = [
            NoteInfo(Fraction(0), Fraction(1), 60, ea, vk1),
            NoteInfo(Fraction(0), Fraction(1), 53, eb, vk2),  # C4-F3 diff=7 fifth
            NoteInfo(Fraction(1), Fraction(2), 62, ec, vk1),
            NoteInfo(Fraction(1), Fraction(2), 55, ed, vk3),  # D4-G3 diff=7 fifth
        ]
        color_map, n_groups = find_note_colors(notes, intervals=('fifths',))
        assert n_groups == 0

    def test_no_violation_interval_mismatch_at_t(self):
        # C4/G4 fifth ends at t=1; D4/F4 (minor third) starts at t=1 in same voices
        v1 = [make_note('C', 4, voice='1', duration=1),
              make_note('D', 4, voice='1', duration=1)]
        v2 = [make_note('G', 4, voice='2', duration=1),
              make_note('F', 4, voice='2', duration=1)]  # D4-F4 = minor third, not fifth
        root = two_voice_score(v1, v2)
        color_map, n_groups = _colored_elems(root)
        assert n_groups == 0

    def test_both_intervals_in_same_run(self):
        # Fifth violation at t=0-1; octave violation at t=2-3 — no shared timestamps.
        # Separate time ranges prevent cross-pair compound-interval false positives.
        vk_f1 = ('P1', '1', '1'); vk_f2 = ('P1', '1', '2')
        vk_o1 = ('P2', '1', '1'); vk_o2 = ('P2', '1', '2')
        e = [make_note('C', 4), make_note('D', 4),
             make_note('G', 4), make_note('A', 4),
             make_note('C', 4), make_note('D', 4),
             make_note('C', 5), make_note('D', 5)]
        notes = [
            NoteInfo(Fraction(0), Fraction(1), 60, e[0], vk_f1),
            NoteInfo(Fraction(0), Fraction(1), 67, e[2], vk_f2),  # C4-G4 fifth
            NoteInfo(Fraction(1), Fraction(2), 62, e[1], vk_f1),
            NoteInfo(Fraction(1), Fraction(2), 69, e[3], vk_f2),  # D4-A4 fifth
            NoteInfo(Fraction(2), Fraction(3), 60, e[4], vk_o1),
            NoteInfo(Fraction(2), Fraction(3), 72, e[6], vk_o2),  # C4-C5 octave
            NoteInfo(Fraction(3), Fraction(4), 62, e[5], vk_o1),
            NoteInfo(Fraction(3), Fraction(4), 74, e[7], vk_o2),  # D4-D5 octave
        ]
        color_map, n_groups = find_note_colors(notes, intervals=('fifths', 'octaves'))
        assert n_groups == 2

    def test_two_independent_violations_two_groups(self):
        # Two fifth violations at non-overlapping time ranges → no cross-pair interference.
        vk_a1 = ('P1', '1', '1'); vk_a2 = ('P1', '1', '2')
        vk_b1 = ('P2', '1', '1'); vk_b2 = ('P2', '1', '2')
        ea = [make_note('C', 4), make_note('D', 4),
              make_note('G', 4), make_note('A', 4)]
        eb = [make_note('C', 4), make_note('D', 4),
              make_note('G', 4), make_note('A', 4)]
        notes = [
            NoteInfo(Fraction(0), Fraction(1), 60, ea[0], vk_a1),
            NoteInfo(Fraction(0), Fraction(1), 67, ea[2], vk_a2),
            NoteInfo(Fraction(1), Fraction(2), 62, ea[1], vk_a1),
            NoteInfo(Fraction(1), Fraction(2), 69, ea[3], vk_a2),
            NoteInfo(Fraction(2), Fraction(3), 60, eb[0], vk_b1),
            NoteInfo(Fraction(2), Fraction(3), 67, eb[2], vk_b2),
            NoteInfo(Fraction(3), Fraction(4), 62, eb[1], vk_b1),
            NoteInfo(Fraction(3), Fraction(4), 69, eb[3], vk_b2),
        ]
        color_map, n_groups = find_note_colors(notes, intervals=('fifths',))
        assert n_groups == 2

    def test_no_violations_empty_result(self):
        root = make_score([('P1', [make_measure(notes=[make_note('C', 4), make_note('D', 4)])])])
        color_map, n_groups = _colored_elems(root)
        assert color_map == {}
        assert n_groups == 0

    def test_single_note_no_violation(self):
        root = make_score([('P1', [make_measure(notes=[make_note('C', 4)])])])
        color_map, n_groups = _colored_elems(root)
        assert color_map == {}
        assert n_groups == 0


# ---------------------------------------------------------------------------
# TestFindNoteColorsRestLookthrough
# ---------------------------------------------------------------------------

class TestFindNoteColorsRestLookthrough:
    def _rest_lookthrough_score(self, attack_v1_at=None, attack_v2_at=None,
                                 prev_simultaneous=True):
        """Build: voice1 C4, voice2 G4 (simultaneous), both rest, voice1 D4, voice2 A4.

        attack_v1_at / attack_v2_at: add an extra note in that voice during the gap.
        prev_simultaneous=False: offset voice2 so prev notes never overlap voice1.
        """
        measure = ET.Element('measure')
        measure.set('number', '1')
        attrs = ET.SubElement(measure, 'attributes')
        ET.SubElement(attrs, 'divisions').text = '4'  # quarter = 4 ticks

        # Voice 1: C4 for 4 ticks, rest 8 ticks, D4 for 4 ticks
        v1_prev = make_note('C', 4, voice='1', duration=4)
        v1_rest = make_note(None,   voice='1', duration=8)
        v1_new  = make_note('D', 4, voice='1', duration=4)
        for n in (v1_prev, v1_rest, v1_new):
            measure.append(n)

        # Backup to start
        bk = ET.SubElement(measure, 'backup')
        ET.SubElement(bk, 'duration').text = '16'

        # Voice 2: G4 at same time as voice1 C4 (or offset if not simultaneous)
        if prev_simultaneous:
            v2_prev = make_note('G', 4, voice='2', duration=4)
            v2_rest = make_note(None,  voice='2', duration=8)
        else:
            # Voice 2 starts after voice 1 ends
            v2_padding = make_note(None, voice='2', duration=8)
            v2_prev    = make_note('G', 4, voice='2', duration=4)
            v2_rest    = make_note(None, voice='2', duration=4)
            measure.append(v2_padding)
        v2_new = make_note('A', 4, voice='2', duration=4)
        for n in (v2_prev, v2_rest, v2_new) if prev_simultaneous else (v2_prev, v2_rest, v2_new):
            measure.append(n)

        # Extra attacks during gap
        if attack_v1_at is not None or attack_v2_at is not None:
            bk2 = ET.SubElement(measure, 'backup')
            ET.SubElement(bk2, 'duration').text = '16'
            if attack_v1_at is not None:
                fwd1 = ET.SubElement(measure, 'forward')
                ET.SubElement(fwd1, 'duration').text = str(attack_v1_at)
                measure.append(make_note('E', 4, voice='1', duration=1))
            if attack_v2_at is not None:
                bk3 = ET.SubElement(measure, 'backup')
                ET.SubElement(bk3, 'duration').text = '16'
                fwd2 = ET.SubElement(measure, 'forward')
                ET.SubElement(fwd2, 'duration').text = str(attack_v2_at)
                measure.append(make_note('F', 4, voice='2', duration=1))

        root = ET.Element('score-partwise')
        part = ET.SubElement(root, 'part')
        part.set('id', 'P1')
        part.append(measure)
        return root

    def test_rest_lookthrough_basic(self):
        root = self._rest_lookthrough_score()
        color_map, n_groups = _colored_elems(root)
        assert n_groups == 1

    def test_blocked_by_attack_in_voice_a(self):
        root = self._rest_lookthrough_score(attack_v1_at=6)
        color_map, n_groups = _colored_elems(root)
        assert n_groups == 0

    def test_blocked_by_attack_in_voice_b(self):
        # Build NoteInfo directly: voice2 attacks F4 during the gap → blocked
        vk1 = ('P1', '1', '1')
        vk2 = ('P1', '1', '2')
        prev_a = make_note('C', 4);  prev_b = make_note('G', 4)
        interv = make_note('F', 4)   # intervening voice2 attack
        new_a  = make_note('D', 4);  new_b  = make_note('A', 4)
        notes = [
            NoteInfo(Fraction(0), Fraction(1), 60, prev_a, vk1),
            NoteInfo(Fraction(0), Fraction(1), 67, prev_b, vk2),  # C4-G4 fifth
            NoteInfo(Fraction(Fraction(3,2)), Fraction(2), 65, interv, vk2),
            NoteInfo(Fraction(3), Fraction(4), 62, new_a, vk1),
            NoteInfo(Fraction(3), Fraction(4), 69, new_b, vk2),   # D4-A4 fifth
        ]
        color_map, n_groups = find_note_colors(notes, intervals=('fifths',))
        assert n_groups == 0

    def test_blocked_by_non_simultaneous_prev_notes(self):
        root = self._rest_lookthrough_score(prev_simultaneous=False)
        color_map, n_groups = _colored_elems(root)
        assert n_groups == 0

    def test_blocked_by_prev_interval_mismatch(self):
        # Replace voice2's prev note with a note that forms a third, not a fifth
        measure = ET.Element('measure')
        measure.set('number', '1')
        attrs = ET.SubElement(measure, 'attributes')
        ET.SubElement(attrs, 'divisions').text = '4'
        # Voice 1: C4(4) rest(8) D4(4)
        for n in (make_note('C', 4, voice='1', duration=4),
                  make_note(None, voice='1', duration=8),
                  make_note('D', 4, voice='1', duration=4)):
            measure.append(n)
        bk = ET.SubElement(measure, 'backup')
        ET.SubElement(bk, 'duration').text = '16'
        # Voice 2: E4(4) rest(8) A4(4) — C4/E4 is a major third (not a fifth)
        for n in (make_note('E', 4, voice='2', duration=4),
                  make_note(None, voice='2', duration=8),
                  make_note('A', 4, voice='2', duration=4)):
            measure.append(n)
        root = ET.Element('score-partwise')
        part = ET.SubElement(root, 'part')
        part.set('id', 'P1')
        part.append(measure)
        color_map, n_groups = _colored_elems(root)
        assert n_groups == 0

    def test_tie_stop_in_gap_is_not_attack(self):
        # The re-entering D4 in voice1 is a tie-stop — it is not a new attack.
        # Build: C4(start) → tied continuation → D4/A4 rest look-through should
        # still fire if the tie-stop is correctly ignored.
        # Simpler: verify that if one of the new notes is tie-stop it still gets
        # detected via the *other* note's vk check (tie-stop is not in by_start
        # so direct-boundary fires on the non-tie-stop side only).
        # Instead, verify a tie-stop mid-gap doesn't block the violation.
        vk1 = ('P1', '1', '1')
        vk2 = ('P1', '1', '2')
        prev_a = make_note('C', 4)
        prev_b = make_note('G', 4)  # C4(60)+G4(67)=diff 7 = fifth
        # A tie-stop note in voice1 during the gap: not an attack
        tie_cont = make_note('C', 4, tie_types=('stop', 'start'))
        new_a = make_note('D', 4)
        new_b = make_note('A', 4)  # D4(62)+A4(69)=diff 7 = fifth
        notes = [
            NoteInfo(Fraction(0), Fraction(1), 60, prev_a,   vk1),
            NoteInfo(Fraction(0), Fraction(1), 67, prev_b,   vk2),
            NoteInfo(Fraction(1), Fraction(Fraction(3, 2)), 60, tie_cont, vk1),  # tie, not attack
            NoteInfo(Fraction(2), Fraction(3), 62, new_a,   vk1),
            NoteInfo(Fraction(2), Fraction(3), 69, new_b,   vk2),
        ]
        color_map, n_groups = find_note_colors(notes, intervals=('fifths',))
        assert n_groups == 1


# ---------------------------------------------------------------------------
# TestFindNoteColorsIntra
# ---------------------------------------------------------------------------

class TestFindNoteColorsIntra:
    def _intra_score(self, with_attack=False, change_prev_interval=False):
        """Single voice: chord C4+G4, rest, chord C4+G4 again."""
        vk = ('P1', '1', '1')
        prev_a = make_note('C', 4)
        prev_b = make_note('G', 4) if not change_prev_interval else make_note('D', 4)
        new_a  = make_note('C', 5)
        new_b  = make_note('G', 5)
        prev_midi_b = 67 if not change_prev_interval else 62
        notes = [
            NoteInfo(Fraction(0), Fraction(1), 60, prev_a, vk),
            NoteInfo(Fraction(0), Fraction(1), prev_midi_b, prev_b, vk),
            NoteInfo(Fraction(2), Fraction(3), 72, new_a,  vk),
            NoteInfo(Fraction(2), Fraction(3), 79, new_b,  vk),
        ]
        if with_attack:
            e = make_note('E', 4)
            notes.insert(2, NoteInfo(Fraction(Fraction(3, 2)), Fraction(2), 64, e, vk))
        return notes

    def test_intra_voice_chord_rest_lookthrough(self):
        notes = self._intra_score()
        color_map, n_groups = find_note_colors(notes, intervals=('fifths',))
        assert n_groups == 1

    def test_blocked_by_intervening_attack(self):
        notes = self._intra_score(with_attack=True)
        color_map, n_groups = find_note_colors(notes, intervals=('fifths',))
        assert n_groups == 0

    def test_prev_chord_no_matching_interval(self):
        notes = self._intra_score(change_prev_interval=True)
        color_map, n_groups = find_note_colors(notes, intervals=('fifths',))
        assert n_groups == 0


# ---------------------------------------------------------------------------
# TestFindNoteColorsExclusions
# ---------------------------------------------------------------------------

class TestFindNoteColorsExclusions:
    def test_same_voice_octave_doubling_excluded(self):
        # Single voice plays C4+C5 → C5+C6; both chords are same-voice octave doublings
        vk = ('P1', '1', '1')
        e1, e2, e3, e4 = (make_note('C', o) for o in (4, 5, 5, 6))
        notes = [
            NoteInfo(Fraction(0), Fraction(1), 60, e1, vk),
            NoteInfo(Fraction(0), Fraction(1), 72, e2, vk),
            NoteInfo(Fraction(1), Fraction(2), 72, e3, vk),
            NoteInfo(Fraction(1), Fraction(2), 84, e4, vk),
        ]
        color_map, n_groups = find_note_colors(notes, intervals=('octaves',))
        assert n_groups == 0

    def test_grace_note_participation_excluded(self):
        # Grace notes never reach collect_notes, so they can't form pairs.
        # Verify a score with only grace + one real note has no violations.
        root = make_score([('P1', [make_measure(notes=[
            make_note('C', 4, grace=True),
            make_note('G', 4),
        ])])])
        notes = collect_notes(root)
        assert len(notes) == 1   # grace excluded at collection time
        color_map, n_groups = find_note_colors(notes)
        assert n_groups == 0

    def test_tie_stop_not_counted_as_new_attack_in_direct_boundary(self):
        # voice1: C4(tie-start) → C4(tie-stop) at t=1; voice2: G4 at 0, A4 at 1.
        # The tie-stop C4 should NOT appear in by_start, so no direct-boundary
        # violation fires between it and voice2's A4 (they don't form a fifth anyway,
        # but more importantly the tie-stop is absent from starting pairs).
        vk1 = ('P1', '1', '1')
        vk2 = ('P1', '1', '2')
        ts = make_note('C', 4, tie_types=('start',))
        tc = make_note('C', 4, tie_types=('stop',))
        g4 = make_note('G', 3)
        a3 = make_note('A', 3)
        notes = [
            NoteInfo(Fraction(0), Fraction(1), 60, ts, vk1),
            NoteInfo(Fraction(1), Fraction(2), 60, tc, vk1),
            NoteInfo(Fraction(0), Fraction(1), 55, g4, vk2),
            NoteInfo(Fraction(1), Fraction(2), 57, a3, vk2),
        ]
        color_map, n_groups = find_note_colors(notes, intervals=('fifths',))
        # No real attack at t=1 in vk1 → no direct boundary → no violation
        assert n_groups == 0

    def test_tie_start_not_in_by_end(self):
        # A tie-start note should not appear in by_end, so its end time is
        # absent from by_end and won't feed ending pairs.
        vk1 = ('P1', '1', '1')
        vk2 = ('P1', '1', '2')
        ts = make_note('C', 4, tie_types=('start',))
        g4 = make_note('G', 3, tie_types=('start',))
        d4 = make_note('D', 4)
        a3 = make_note('A', 3)
        notes = [
            NoteInfo(Fraction(0), Fraction(1), 60, ts, vk1),
            NoteInfo(Fraction(0), Fraction(1), 55, g4, vk2),
            NoteInfo(Fraction(1), Fraction(2), 62, d4, vk1),
            NoteInfo(Fraction(1), Fraction(2), 57, a3, vk2),
        ]
        idx = VoiceIndex(notes)
        # Tie-start notes must not be in by_end at t=1
        assert Fraction(1) not in idx.by_end


# ---------------------------------------------------------------------------
# TestFindNoteColorsMerge
# ---------------------------------------------------------------------------

class TestFindNoteColorsMerge:
    def test_shared_note_merges_groups(self):
        # voice1-voice2 fifth AND voice1-voice3 fifth at same time,
        # sharing voice1's notes → one merged group
        vk1 = ('P1', '1', '1')
        vk2 = ('P1', '1', '2')
        vk3 = ('P1', '1', '3')
        a1, b1 = make_note('C', 4), make_note('C', 4)   # voice1 notes (diff elems)
        a2, b2 = make_note('F', 3), make_note('G', 3)   # voice2 notes
        a3, b3 = make_note('G', 4), make_note('A', 4)   # voice3 notes
        notes = [
            NoteInfo(Fraction(0), Fraction(1), 60, a1, vk1),
            NoteInfo(Fraction(0), Fraction(1), 53, a2, vk2),  # C4(60)-F3(53) diff=7 fifth
            NoteInfo(Fraction(0), Fraction(1), 67, a3, vk3),  # C4(60)-G4(67) diff=7 fifth
            NoteInfo(Fraction(1), Fraction(2), 62, b1, vk1),
            NoteInfo(Fraction(1), Fraction(2), 55, b2, vk2),  # D4(62)-G3(55) diff=7 fifth
            NoteInfo(Fraction(1), Fraction(2), 69, b3, vk3),  # D4(62)-A4(69) diff=7 fifth
        ]
        color_map, n_groups = find_note_colors(notes, intervals=('fifths',))
        # vk1-vk2 and vk1-vk3 violations both share a1 and b1 → merged into one group
        assert n_groups == 1

    def test_independent_violations_distinct_colors(self):
        # Two violations at non-overlapping time ranges — no cross-pair interference
        vk_a1 = ('A', '1', '1'); vk_a2 = ('A', '1', '2')
        vk_b1 = ('B', '1', '1'); vk_b2 = ('B', '1', '2')
        ea = [make_note('C', 4), make_note('D', 4),
              make_note('G', 4), make_note('A', 4)]
        eb = [make_note('C', 4), make_note('D', 4),
              make_note('G', 4), make_note('A', 4)]
        notes = [
            NoteInfo(Fraction(0), Fraction(1), 60, ea[0], vk_a1),
            NoteInfo(Fraction(0), Fraction(1), 67, ea[2], vk_a2),
            NoteInfo(Fraction(1), Fraction(2), 62, ea[1], vk_a1),
            NoteInfo(Fraction(1), Fraction(2), 69, ea[3], vk_a2),
            NoteInfo(Fraction(2), Fraction(3), 60, eb[0], vk_b1),
            NoteInfo(Fraction(2), Fraction(3), 67, eb[2], vk_b2),
            NoteInfo(Fraction(3), Fraction(4), 62, eb[1], vk_b1),
            NoteInfo(Fraction(3), Fraction(4), 69, eb[3], vk_b2),
        ]
        color_map, n_groups = find_note_colors(notes, intervals=('fifths',))
        assert n_groups == 2
        # First group's note color must differ from second group's note color
        group_colors = {id(ea[0]): color_map[id(ea[0])],
                        id(eb[0]): color_map[id(eb[0])]}
        assert group_colors[id(ea[0])] != group_colors[id(eb[0])]

    def _make_staggered_violation(self, i, part_id):
        """Build 4 NoteInfo forming a direct-boundary fifth violation at time [2i, 2i+2)."""
        vk1 = (part_id, '1', '1')
        vk2 = (part_id, '1', '2')
        e1 = make_note('C', 4); e2 = make_note('D', 4)
        e3 = make_note('G', 4); e4 = make_note('A', 4)
        s = Fraction(2 * i)
        return [
            NoteInfo(s,     s + 1, 60, e1, vk1),
            NoteInfo(s,     s + 1, 67, e3, vk2),
            NoteInfo(s + 1, s + 2, 62, e2, vk1),
            NoteInfo(s + 1, s + 2, 69, e4, vk2),
        ]

    def test_palette_wraps_after_8_groups(self):
        # 9 violations staggered in time so no cross-group interaction
        all_notes = [n for i in range(9) for n in self._make_staggered_violation(i, f'P{i}')]
        with warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            color_map, n_groups = find_note_colors(all_notes, intervals=('fifths',))
        assert n_groups == 9

    def test_palette_warning_emitted_once(self):
        # 10 violations staggered in time
        all_notes = [n for i in range(10) for n in self._make_staggered_violation(i, f'P{i}')]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            find_note_colors(all_notes, intervals=('fifths',))
        palette_warns = [x for x in w if 'palette' in str(x.message).lower()]
        assert len(palette_warns) == 1

    def test_n_groups_reflects_actual_count(self):
        # Two violations staggered in time → exactly n_groups==2
        notes = (self._make_staggered_violation(0, 'A') +
                 self._make_staggered_violation(1, 'B'))
        _, n_groups = find_note_colors(notes, intervals=('fifths',))
        assert n_groups == 2


# ---------------------------------------------------------------------------
# TestFindNoteColorsMultiPart
# ---------------------------------------------------------------------------

class TestFindNoteColorsMultiPart:
    def test_two_parts_flagged(self):
        # Each part has one voice; together they form parallel fifths
        m1 = make_measure(notes=[make_note('C', 4, voice='1', duration=1),
                                  make_note('D', 4, voice='1', duration=1)])
        m2 = make_measure(notes=[make_note('G', 4, voice='1', duration=1),
                                  make_note('A', 4, voice='1', duration=1)])
        root = make_score([('P1', [m1]), ('P2', [m2])])
        color_map, n_groups = _colored_elems(root)
        assert n_groups == 1

    def test_cross_staff_same_part_flagged(self):
        # Two staves in the same part: staff=1 voice=1 and staff=2 voice=1
        measure = ET.Element('measure')
        measure.set('number', '1')
        attrs = ET.SubElement(measure, 'attributes')
        ET.SubElement(attrs, 'divisions').text = '1'
        total_dur = 2
        for n in (make_note('C', 4, voice='1', staff='1', duration=1),
                  make_note('D', 4, voice='1', staff='1', duration=1)):
            measure.append(n)
        bk = ET.SubElement(measure, 'backup')
        ET.SubElement(bk, 'duration').text = str(total_dur)
        for n in (make_note('G', 4, voice='1', staff='2', duration=1),
                  make_note('A', 4, voice='1', staff='2', duration=1)):
            measure.append(n)
        root = make_score([('P1', [measure])])
        color_map, n_groups = _colored_elems(root)
        assert n_groups == 1


# ---------------------------------------------------------------------------
# TestColorize
# ---------------------------------------------------------------------------

class TestColorize:
    def _plain_note(self):
        return ET.Element('note')

    def _note_with_stem(self):
        n = ET.Element('note')
        ET.SubElement(n, 'stem').text = 'up'
        return n

    def _note_with_notehead(self):
        n = ET.Element('note')
        ET.SubElement(n, 'notehead').text = 'normal'
        return n

    def _note_with_both(self):
        n = ET.Element('note')
        ET.SubElement(n, 'stem').text = 'up'
        ET.SubElement(n, 'notehead').text = 'normal'
        return n

    def test_color_set_on_note(self):
        n = self._plain_note()
        colorize(n, '#FF0000')
        assert n.get('color') == '#FF0000'

    def test_correct_hex_value_stored(self):
        n = self._plain_note()
        colorize(n, '#D62728')
        assert n.get('color') == '#D62728'

    def test_stem_colored_when_present(self):
        n = self._note_with_stem()
        colorize(n, '#FF0000')
        assert n.find('stem').get('color') == '#FF0000'

    def test_no_error_when_stem_absent(self):
        n = self._plain_note()
        colorize(n, '#FF0000')   # must not raise
        assert n.find('stem') is None

    def test_notehead_colored_when_present(self):
        n = self._note_with_notehead()
        colorize(n, '#FF0000')
        assert n.find('notehead').get('color') == '#FF0000'

    def test_no_error_when_notehead_absent(self):
        n = self._plain_note()
        colorize(n, '#FF0000')   # must not raise

    def test_no_synthetic_notehead_inserted(self):
        n = self._plain_note()
        colorize(n, '#FF0000')
        assert n.find('notehead') is None

    def test_both_stem_and_notehead_colored(self):
        n = self._note_with_both()
        colorize(n, '#1F77B4')
        assert n.get('color') == '#1F77B4'
        assert n.find('stem').get('color') == '#1F77B4'
        assert n.find('notehead').get('color') == '#1F77B4'


# ---------------------------------------------------------------------------
# TestMain
# ---------------------------------------------------------------------------

MINIMAL_XML = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<score-partwise><part id="P1">'
    '<measure number="1">'
    '<attributes><divisions>1</divisions></attributes>'
    '<note><pitch><step>C</step><octave>4</octave></pitch>'
    '<duration>1</duration><voice>1</voice></note>'
    '</measure></part></score-partwise>'
)


def _fifths_xml():
    return score_to_xml(parallel_fifths_score())


def _octaves_xml():
    return score_to_xml(parallel_octaves_score())


class TestMain:
    def test_stdin_stdout(self, monkeypatch, capsys):
        out, err = _run_main(monkeypatch, capsys, ['-', '-'], MINIMAL_XML)
        assert '<score-partwise' in out
        assert 'colorized' in err

    def test_file_input_output(self, tmp_path, monkeypatch, capsys):
        infile = tmp_path / 'in.xml'
        outfile = tmp_path / 'out.xml'
        infile.write_text(MINIMAL_XML)
        _run_main(monkeypatch, capsys, [str(infile), str(outfile)])
        assert outfile.exists()
        ET.fromstring(outfile.read_text().split('\n', 1)[-1])   # valid XML after prologue

    def test_interval_fifths_default(self, monkeypatch, capsys):
        _, err = _run_main(monkeypatch, capsys, ['-', '-'], _fifths_xml())
        assert 'consecutive-fifth' in err

    def test_interval_octaves(self, monkeypatch, capsys):
        _, err = _run_main(monkeypatch, capsys, ['--interval', 'octaves', '-', '-'], _octaves_xml())
        assert 'consecutive-octave' in err

    def test_interval_both(self, monkeypatch, capsys):
        _, err = _run_main(monkeypatch, capsys, ['--interval', 'both', '-', '-'], _fifths_xml())
        assert 'consecutive-fifth/octave' in err

    def test_progress_on_stderr_not_stdout(self, monkeypatch, capsys):
        out, err = _run_main(monkeypatch, capsys, ['-', '-'], _fifths_xml())
        assert 'colorized' not in out
        assert 'colorized' in err

    def test_xml_declaration_preserved(self, monkeypatch, capsys):
        xml = '<?xml version="1.0" encoding="UTF-8"?>\n' + \
              '<score-partwise><part id="P1"><measure number="1">' \
              '<attributes><divisions>1</divisions></attributes>' \
              '<note><pitch><step>C</step><octave>4</octave></pitch>' \
              '<duration>1</duration><voice>1</voice></note>' \
              '</measure></part></score-partwise>'
        out, _ = _run_main(monkeypatch, capsys, ['-', '-'], xml)
        assert out.startswith('<?xml version="1.0" encoding="UTF-8"?>')

    def test_doctype_preserved(self, monkeypatch, capsys):
        doctype = '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">'
        xml = ('<?xml version="1.0" encoding="UTF-8"?>\n' + doctype + '\n' +
               '<score-partwise><part id="P1"><measure number="1">'
               '<attributes><divisions>1</divisions></attributes>'
               '<note><pitch><step>C</step><octave>4</octave></pitch>'
               '<duration>1</duration><voice>1</voice></note>'
               '</measure></part></score-partwise>')
        out, _ = _run_main(monkeypatch, capsys, ['-', '-'], xml)
        assert doctype in out

    def test_no_declaration_empty_prologue(self, monkeypatch, capsys):
        # When input has no XML declaration, prologue = raw[:0] = '' (empty).
        # The default <?xml...?> prologue is only used when <score-partwise> is absent.
        xml = ('<score-partwise><part id="P1"><measure number="1">'
               '<attributes><divisions>1</divisions></attributes>'
               '<note><pitch><step>C</step><octave>4</octave></pitch>'
               '<duration>1</duration><voice>1</voice></note>'
               '</measure></part></score-partwise>')
        out, _ = _run_main(monkeypatch, capsys, ['-', '-'], xml)
        assert out.startswith('<score-partwise')

    def test_no_violations_message(self, monkeypatch, capsys):
        _, err = _run_main(monkeypatch, capsys, ['-', '-'], MINIMAL_XML)
        assert '0 note(s) in 0 consecutive-fifth group(s) colorized.' in err

    def test_violation_colorized_in_output(self, monkeypatch, capsys):
        out, _ = _run_main(monkeypatch, capsys, ['-', '-'], _fifths_xml())
        # Strip prologue line and parse remaining XML
        body = '\n'.join(out.split('\n')[1:]) if out.startswith('<?xml') else out
        root = ET.fromstring(body)
        colored = [n for n in root.iter('note') if n.get('color')]
        assert len(colored) > 0

    def test_missing_args_exits_nonzero(self, monkeypatch):
        monkeypatch.setattr(sys, 'argv', ['conseq.py'])
        with pytest.raises(SystemExit) as exc:
            conseq.main()
        assert exc.value.code != 0


# ---------------------------------------------------------------------------
# TestNavigationAnnotations
# ---------------------------------------------------------------------------

def _staggered_fifth(i):
    """4 NoteInfo forming a direct-boundary fifth at time [2i, 2i+2), part P{i}."""
    vk1 = (f'P{i}', '1', '1')
    vk2 = (f'P{i}', '1', '2')
    e1, e2 = make_note('C', 4), make_note('D', 4)
    e3, e4 = make_note('G', 4), make_note('A', 4)
    s = Fraction(2 * i)
    return [
        NoteInfo(s,     s + 1, 60, e1, vk1),
        NoteInfo(s,     s + 1, 67, e3, vk2),
        NoteInfo(s + 1, s + 2, 62, e2, vk1),
        NoteInfo(s + 1, s + 2, 69, e4, vk2),
    ]


class TestBuildNoteLocations:
    def test_maps_each_note_to_its_measure(self):
        root  = parallel_fifths_score()
        notes = collect_notes(root)
        locs  = build_note_locations(root)
        for n in notes:
            assert n.elem in list(locs[id(n.elem)])

    def test_distinct_measures_reported(self):
        m1 = make_measure('1', notes=[make_note('C', 4)])
        m2 = make_measure('2', notes=[make_note('D', 4)])
        root = make_score([('P1', [m1, m2])])
        locs = build_note_locations(root)
        numbers = {measure.get('number') for measure in locs.values()}
        assert numbers == {'1', '2'}


class TestFindViolationGroups:
    def test_single_group_basics(self):
        root   = parallel_fifths_score()
        notes  = collect_notes(root)
        groups = find_violation_groups(notes)
        assert len(groups) == 1
        g = groups[0]
        assert g.number == 1
        assert g.color == conseq.PALETTE[0]
        assert len(g.members) == 4

    def test_anchor_is_earliest_onset_note(self):
        root   = parallel_fifths_score()
        notes  = collect_notes(root)
        starts = {id(n.elem): n.start for n in notes}
        g      = find_violation_groups(notes)[0]
        assert starts[id(g.anchor)] == min(starts[id(m)] for m in g.members)

    def test_numbering_follows_score_order(self):
        notes  = _staggered_fifth(0) + _staggered_fifth(1)
        groups = find_violation_groups(notes)
        assert [g.number for g in groups] == [1, 2]
        starts = {id(n.elem): n.start for n in notes}
        anchor_starts = [starts[id(g.anchor)] for g in groups]
        assert anchor_starts == sorted(anchor_starts)

    def test_colors_match_find_note_colors(self):
        notes        = _staggered_fifth(0) + _staggered_fifth(1)
        color_map, _ = find_note_colors(notes)
        for g in find_violation_groups(notes):
            for member in g.members:
                assert color_map[id(member)] == g.color

    def test_no_violations_returns_empty(self):
        root  = parallel_fifths_score()
        notes = collect_notes(root)
        assert find_violation_groups(notes, intervals=('octaves',)) == []


class TestMakeNavigationMarker:
    def test_structure_and_content(self):
        d = make_navigation_marker(3, '#D62728')
        assert d.tag == 'direction'
        assert d.get('placement') == 'above'
        rehearsal = d.find('direction-type/rehearsal')
        assert rehearsal is not None
        assert rehearsal.get('color') == '#D62728'
        assert rehearsal.text == '‖3'

    def test_color_only_no_forced_font(self):
        # Prominence is left to the renderer's rehearsal-mark style.
        rehearsal = make_navigation_marker(1, '#000000').find('direction-type/rehearsal')
        assert rehearsal.get('font-size') is None
        assert rehearsal.get('font-weight') is None


class TestAnnotateGroups:
    def test_one_marker_inserted_before_anchor(self):
        root   = parallel_fifths_score()
        notes  = collect_notes(root)
        groups = find_violation_groups(notes)
        annotate_groups(groups, build_note_locations(root))

        directions = list(root.iter('direction'))
        assert len(directions) == 1
        measure = root.find('part/measure')
        kids    = list(measure)
        assert kids.index(directions[0]) == kids.index(groups[0].anchor) - 1

    def test_marker_color_matches_group(self):
        root   = parallel_fifths_score()
        notes  = collect_notes(root)
        groups = find_violation_groups(notes)
        annotate_groups(groups, build_note_locations(root))
        rehearsal = root.find('part/measure/direction/direction-type/rehearsal')
        assert rehearsal.get('color') == groups[0].color

    def test_missing_location_skipped(self):
        root   = parallel_fifths_score()
        notes  = collect_notes(root)
        groups = find_violation_groups(notes)
        annotate_groups(groups, {})   # no locations → nothing inserted
        assert list(root.iter('direction')) == []


class TestCliAnnotate:
    def _markers(self, out):
        body = '\n'.join(out.split('\n')[1:]) if out.startswith('<?xml') else out
        root = ET.fromstring(body)
        return [r for r in root.iter('rehearsal')
                if r.text and r.text.startswith('‖')], root

    def test_annotate_inserts_marker(self, monkeypatch, capsys):
        out, err = _run_main(monkeypatch, capsys, ['--annotate', '-', '-'], _fifths_xml())
        marks, _ = self._markers(out)
        assert len(marks) == 1
        assert 'and marked' in err

    def test_marker_color_matches_colored_note(self, monkeypatch, capsys):
        out, _ = _run_main(monkeypatch, capsys, ['--annotate', '-', '-'], _fifths_xml())
        marks, root = self._markers(out)
        note_colors = {n.get('color') for n in root.iter('note') if n.get('color')}
        assert marks[0].get('color') in note_colors

    def test_no_marker_without_flag(self, monkeypatch, capsys):
        out, err = _run_main(monkeypatch, capsys, ['-', '-'], _fifths_xml())
        marks, _ = self._markers(out)
        assert marks == []
        assert 'and marked' not in err
