from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

from file_model.SCORE import SCORE
from utils.CONSTANT import QUARTER_NOTE_UNIT


DIVISIONS_PER_QUARTER = 64
_UNITS_PER_DIVISION = float(QUARTER_NOTE_UNIT) / float(DIVISIONS_PER_QUARTER)
_EPS = 1e-6


@dataclass
class MeasureInfo:
    index: int
    start_units: float
    end_units: float
    numerator: int
    denominator: int


@dataclass
class NoteEvent:
    note_id: int
    start_units: float
    end_units: float
    pitch: int
    acc: int
    hand: str
    voice: int


@dataclass
class NoteChunk:
    measure_index: int
    start_units: float
    end_units: float
    pitch: int
    acc: int
    hand: str
    voice: int
    tie_start: bool
    tie_stop: bool


def _units_to_divisions(units: float) -> int:
    return int(round(float(units) / float(_UNITS_PER_DIVISION)))


def _duration_type_and_dots(duration_divisions: int) -> tuple[Optional[str], int]:
    if duration_divisions <= 0:
        return None, 0
    q = float(duration_divisions) / float(DIVISIONS_PER_QUARTER)
    values = [
        (4.0, "whole"),
        (2.0, "half"),
        (1.0, "quarter"),
        (0.5, "eighth"),
        (0.25, "16th"),
        (0.125, "32nd"),
        (0.0625, "64th"),
        (0.03125, "128th"),
    ]
    for dots in (0, 1, 2):
        dot_scale = 1.0 + (0.5 if dots >= 1 else 0.0) + (0.25 if dots >= 2 else 0.0)
        for base, type_name in values:
            if abs((base * dot_scale) - q) < 1e-6:
                return type_name, dots
    return None, 0


def _is_white_key_midi(midi_pitch: int) -> bool:
    return int(midi_pitch) % 12 in (0, 2, 4, 5, 7, 9, 11)


def _midi_to_step_alter_octave(midi_pitch: int) -> tuple[str, int, int]:
    semitone = int(midi_pitch) % 12
    octave = (int(midi_pitch) // 12) - 1
    mapping = {
        0: ("C", 0),
        1: ("C", 1),
        2: ("D", 0),
        3: ("D", 1),
        4: ("E", 0),
        5: ("F", 0),
        6: ("F", 1),
        7: ("G", 0),
        8: ("G", 1),
        9: ("A", 0),
        10: ("A", 1),
        11: ("B", 0),
    }
    step, alter = mapping[semitone]
    return step, alter, octave


def _midi_natural_to_step_octave(midi_pitch: int) -> tuple[str, int]:
    semitone = int(midi_pitch) % 12
    octave = (int(midi_pitch) // 12) - 1
    mapping = {
        0: "C",
        2: "D",
        4: "E",
        5: "F",
        7: "G",
        9: "A",
        11: "B",
    }
    step = mapping.get(semitone)
    if step is None:
        raise ValueError("Expected a natural MIDI pitch for natural spelling")
    return step, octave


def _accidental_text_from_alter(alter: int) -> Optional[str]:
    return {
        -2: "flat-flat",
        -1: "flat",
        0: "natural",
        1: "sharp",
        2: "double-sharp",
    }.get(int(alter))


def _note_pitch_xml(note_pitch: int, note_acc: int) -> tuple[str, int, int, Optional[str]]:
    midi_pitch = int(note_pitch) + 20
    acc = int(note_acc)

    # keyTAB accidental model: base natural is (pitch + acc), sounding pitch is pitch.
    # Therefore MusicXML alter is -acc when explicit accidental is present.
    if acc != 0:
        base_midi = midi_pitch + acc
        alter = -acc
        if _is_white_key_midi(base_midi) and -2 <= alter <= 2:
            step, octave = _midi_natural_to_step_octave(base_midi)
            return step, int(alter), octave, _accidental_text_from_alter(alter)

    step, alter, octave = _midi_to_step_alter_octave(midi_pitch)
    accidental_text = _accidental_text_from_alter(alter) if acc != 0 else None
    return step, alter, octave, accidental_text


def _group_notes_by_onset(notes: list[NoteEvent]) -> list[list[NoteEvent]]:
    grouped: list[list[NoteEvent]] = []
    for n in sorted(notes, key=lambda x: (x.start_units, x.pitch, x.end_units)):
        if not grouped:
            grouped.append([n])
            continue
        last_start = grouped[-1][0].start_units
        if abs(n.start_units - last_start) <= _EPS:
            grouped[-1].append(n)
        else:
            grouped.append([n])
    return grouped


def _assign_voices_for_hand(notes: list[NoteEvent]) -> dict[int, int]:
    if not notes:
        return {}

    voice_end: list[float] = []
    voice_last_start: list[float] = []
    voice_last_dur: list[float] = []
    mapping: dict[int, int] = {}

    for onset_group in _group_notes_by_onset(notes):
        for n in sorted(onset_group, key=lambda x: (-((x.end_units - x.start_units)), x.pitch)):
            start = float(n.start_units)
            dur = max(_EPS, float(n.end_units - n.start_units))
            end = float(n.end_units)

            selected = -1

            # Keep chord notes in the same voice when onset and duration match.
            for idx in range(len(voice_end)):
                if abs(voice_last_start[idx] - start) <= _EPS and abs(voice_last_dur[idx] - dur) <= _EPS:
                    selected = idx
                    break

            if selected < 0:
                candidates = [
                    idx for idx in range(len(voice_end))
                    if voice_end[idx] <= (start + _EPS)
                ]
                if candidates:
                    selected = max(candidates, key=lambda i: voice_end[i])

            if selected < 0:
                voice_end.append(end)
                voice_last_start.append(start)
                voice_last_dur.append(dur)
                selected = len(voice_end) - 1
            else:
                voice_end[selected] = max(voice_end[selected], end)
                voice_last_start[selected] = start
                voice_last_dur[selected] = dur

            mapping[int(n.note_id)] = int(selected + 1)

    return mapping


def _extract_note_events(score: SCORE) -> list[NoteEvent]:
    events: list[NoteEvent] = []
    for n in getattr(getattr(score, "events", None), "note", []) or []:
        try:
            start = float(getattr(n, "time", 0.0) or 0.0)
            dur = float(getattr(n, "duration", 0.0) or 0.0)
            if dur <= _EPS:
                continue
            events.append(
                NoteEvent(
                    note_id=int(getattr(n, "_id", 0) or 0),
                    start_units=start,
                    end_units=start + dur,
                    pitch=int(getattr(n, "pitch", 0) or 0),
                    acc=int(getattr(n, "acc", 0) or 0),
                    hand=str(getattr(n, "hand", "l") or "l"),
                    voice=1,
                )
            )
        except Exception:
            continue

    by_hand: dict[str, list[NoteEvent]] = {"l": [], "r": []}
    for n in events:
        hand = "r" if str(n.hand) == "r" else "l"
        by_hand[hand].append(n)

    for hand in ("l", "r"):
        voice_map = _assign_voices_for_hand(by_hand[hand])
        for n in by_hand[hand]:
            n.voice = int(voice_map.get(int(n.note_id), 1))

    return events


def _measure_length_units(numerator: int, denominator: int) -> float:
    return float(max(1, int(numerator))) * (4.0 / float(max(1, int(denominator)))) * float(QUARTER_NOTE_UNIT)


def _build_measures(score: SCORE, max_end_units: float) -> list[MeasureInfo]:
    measures: list[MeasureInfo] = []
    current_start = 0.0

    base_grid = list(getattr(score, "base_grid", []) or [])
    if not base_grid:
        base_grid = [type("_Grid", (), {"numerator": 4, "denominator": 4, "measure_amount": 1})()]

    idx = 0
    for segment in base_grid:
        numer = max(1, int(getattr(segment, "numerator", 4) or 4))
        denom = max(1, int(getattr(segment, "denominator", 4) or 4))
        amount = max(1, int(getattr(segment, "measure_amount", 1) or 1))
        mlen = _measure_length_units(numer, denom)
        for _ in range(amount):
            measures.append(
                MeasureInfo(
                    index=idx,
                    start_units=current_start,
                    end_units=current_start + mlen,
                    numerator=numer,
                    denominator=denom,
                )
            )
            current_start += mlen
            idx += 1

    if not measures:
        mlen = _measure_length_units(4, 4)
        measures.append(MeasureInfo(index=0, start_units=0.0, end_units=mlen, numerator=4, denominator=4))

    while measures[-1].end_units < (float(max_end_units) - _EPS):
        last = measures[-1]
        mlen = _measure_length_units(last.numerator, last.denominator)
        measures.append(
            MeasureInfo(
                index=last.index + 1,
                start_units=last.end_units,
                end_units=last.end_units + mlen,
                numerator=last.numerator,
                denominator=last.denominator,
            )
        )

    return measures


def _split_notes_into_measures(notes: list[NoteEvent], measures: list[MeasureInfo]) -> list[NoteChunk]:
    chunks: list[NoteChunk] = []
    if not measures:
        return chunks

    m_idx = 0
    for n in sorted(notes, key=lambda x: (x.start_units, x.pitch)):
        start = float(n.start_units)
        end = float(n.end_units)
        if end <= start + _EPS:
            continue

        while m_idx < len(measures) and measures[m_idx].end_units <= start + _EPS:
            m_idx += 1

        local_idx = min(max(m_idx, 0), len(measures) - 1)
        cursor = start
        while cursor < end - _EPS and local_idx < len(measures):
            measure = measures[local_idx]
            if cursor < measure.start_units - _EPS:
                cursor = measure.start_units
            piece_end = min(end, measure.end_units)
            if piece_end <= cursor + _EPS:
                local_idx += 1
                continue
            chunks.append(
                NoteChunk(
                    measure_index=measure.index,
                    start_units=cursor,
                    end_units=piece_end,
                    pitch=n.pitch,
                    acc=n.acc,
                    hand=n.hand,
                    voice=n.voice,
                    tie_start=cursor > start + _EPS,
                    tie_stop=end > piece_end + _EPS,
                )
            )
            cursor = piece_end
            local_idx += 1

    return chunks


def _tempo_beat_unit_and_quarter_bpm(duration_units: float, tempo: int) -> tuple[str, Optional[str], int, float]:
    """Map keyTAB tempo event (duration_units, tempo) to MusicXML beat-unit.

    keyTAB model: ``tempo`` units of ``duration_units`` ticks happen per minute.
    MusicXML <sound tempo> is always in quarter-note BPM.

    Returns (beat_unit, optional_dot, per_minute, quarter_bpm).
    """
    dur = float(duration_units) if float(duration_units) > 0 else float(QUARTER_NOTE_UNIT)
    q = float(QUARTER_NOTE_UNIT)
    quarter_bpm = float(max(1, int(tempo))) * (dur / q)

    # Standard note durations (in QUARTER_NOTE_UNIT ticks) -> (type_name, dotted)
    standard: list[tuple[float, str, bool]] = [
        (q * 4,        "whole",   False),
        (q * 3,        "half",    True),
        (q * 2,        "half",    False),
        (q * 1.5,      "quarter", True),
        (q * 1,        "quarter", False),
        (q * 0.75,     "eighth",  True),
        (q * 0.5,      "eighth",  False),
        (q * 0.375,    "16th",    True),
        (q * 0.25,     "16th",    False),
        (q * 0.1875,   "32nd",    True),
        (q * 0.125,    "32nd",    False),
        (q * 0.09375,  "64th",    True),
        (q * 0.0625,   "64th",    False),
        (q * 0.046875, "128th",   True),
        (q * 0.03125,  "128th",   False),
    ]
    for std_dur, type_name, dotted in standard:
        if abs(dur - std_dur) < 0.5:          # within half a tick
            per_minute = int(round(float(max(1, int(tempo)))))
            dot = "yes" if dotted else None
            return type_name, dot, per_minute, round(quarter_bpm, 6)

    # Non-standard duration: keep quarter as beat unit, convert BPM
    return "quarter", None, max(1, int(round(quarter_bpm))), round(quarter_bpm, 6)


def _append_note(
    measure_el: ET.Element,
    *,
    duration_div: int,
    voice: int,
    staff: int,
    pitch: Optional[tuple[str, int, int]] = None,
    accidental_text: Optional[str] = None,
    chord: bool = False,
    tie_start: bool = False,
    tie_stop: bool = False,
) -> None:
    note_el = ET.SubElement(measure_el, "note")
    if chord:
        ET.SubElement(note_el, "chord")

    if pitch is None:
        ET.SubElement(note_el, "rest")
    else:
        step, alter, octave = pitch
        pitch_el = ET.SubElement(note_el, "pitch")
        ET.SubElement(pitch_el, "step").text = str(step)
        if int(alter) != 0:
            ET.SubElement(pitch_el, "alter").text = str(int(alter))
        ET.SubElement(pitch_el, "octave").text = str(int(octave))

    ET.SubElement(note_el, "duration").text = str(max(1, int(duration_div)))
    ET.SubElement(note_el, "voice").text = str(int(voice))

    type_name, dots = _duration_type_and_dots(int(duration_div))
    if type_name:
        ET.SubElement(note_el, "type").text = type_name
        for _ in range(dots):
            ET.SubElement(note_el, "dot")

    if accidental_text and pitch is not None:
        ET.SubElement(note_el, "accidental").text = accidental_text

    if tie_start:
        ET.SubElement(note_el, "tie", {"type": "start"})
    if tie_stop:
        ET.SubElement(note_el, "tie", {"type": "stop"})

    if tie_start or tie_stop:
        notations = ET.SubElement(note_el, "notations")
        if tie_start:
            ET.SubElement(notations, "tied", {"type": "start"})
        if tie_stop:
            ET.SubElement(notations, "tied", {"type": "stop"})

    ET.SubElement(note_el, "staff").text = str(int(staff))


def _append_voice_material(
    measure_el: ET.Element,
    measure: MeasureInfo,
    chunks: list[NoteChunk],
    voice: int,
    staff: int,
) -> None:
    measure_start_div = _units_to_divisions(measure.start_units)
    measure_end_div = _units_to_divisions(measure.end_units)
    measure_len_div = max(1, measure_end_div - measure_start_div)

    cursor = 0
    chunks_sorted = sorted(chunks, key=lambda c: (c.start_units, c.pitch, c.end_units))

    i = 0
    while i < len(chunks_sorted):
        base = chunks_sorted[i]
        same_group = [base]
        i += 1
        while i < len(chunks_sorted):
            nxt = chunks_sorted[i]
            if abs(nxt.start_units - base.start_units) <= _EPS and abs(nxt.end_units - base.end_units) <= _EPS:
                same_group.append(nxt)
                i += 1
            else:
                break

        start_div = _units_to_divisions(base.start_units) - measure_start_div
        start_div = max(0, min(measure_len_div, start_div))
        if start_div > cursor:
            _append_note(
                measure_el,
                duration_div=(start_div - cursor),
                voice=voice,
                staff=staff,
                pitch=None,
            )

        dur_div = _units_to_divisions(base.end_units) - _units_to_divisions(base.start_units)
        dur_div = max(1, dur_div)

        for j, c in enumerate(sorted(same_group, key=lambda x: x.pitch)):
            step, alter, octave, acc_text = _note_pitch_xml(c.pitch, c.acc)
            _append_note(
                measure_el,
                duration_div=dur_div,
                voice=voice,
                staff=staff,
                pitch=(step, alter, octave),
                accidental_text=acc_text,
                chord=(j > 0),
                tie_start=bool(c.tie_stop),
                tie_stop=bool(c.tie_start),
            )

        cursor = min(measure_len_div, start_div + dur_div)

    if cursor < measure_len_div:
        _append_note(
            measure_el,
            duration_div=(measure_len_div - cursor),
            voice=voice,
            staff=staff,
            pitch=None,
        )


def export_score_to_musicxml(score: SCORE, output_path: Path) -> dict[str, int]:
    notes = _extract_note_events(score)
    max_note_end = max((float(n.end_units) for n in notes), default=0.0)
    max_tempo_time = max((float(getattr(t, "time", 0.0) or 0.0) for t in (getattr(score.events, "tempo", []) or [])), default=0.0)
    measures = _build_measures(score, max(max_note_end, max_tempo_time))
    chunks = _split_notes_into_measures(notes, measures)

    root = ET.Element("score-partwise", {"version": "4.0"})

    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = str(getattr(getattr(score, "info", None), "title", "") or "Untitled")

    ident = ET.SubElement(root, "identification")
    composer = str(getattr(getattr(score, "info", None), "composer", "") or "").strip()
    if composer:
        ET.SubElement(ident, "creator", {"type": "composer"}).text = composer

    part_list = ET.SubElement(root, "part-list")
    score_part = ET.SubElement(part_list, "score-part", {"id": "P1"})
    ET.SubElement(score_part, "part-name").text = "Piano"

    part = ET.SubElement(root, "part", {"id": "P1"})

    tempo_events = sorted((getattr(score.events, "tempo", []) or []), key=lambda t: float(getattr(t, "time", 0.0) or 0.0))
    prev_sig: tuple[int, int] | None = None

    chunks_by_measure: dict[int, list[NoteChunk]] = {}
    for c in chunks:
        chunks_by_measure.setdefault(int(c.measure_index), []).append(c)

    for m in measures:
        measure_el = ET.SubElement(part, "measure", {"number": str(m.index + 1)})

        sig = (int(m.numerator), int(m.denominator))
        if m.index == 0 or sig != prev_sig:
            attrs = ET.SubElement(measure_el, "attributes")
            ET.SubElement(attrs, "divisions").text = str(int(DIVISIONS_PER_QUARTER))
            key_el = ET.SubElement(attrs, "key")
            ET.SubElement(key_el, "fifths").text = "0"
            time_el = ET.SubElement(attrs, "time")
            ET.SubElement(time_el, "beats").text = str(int(m.numerator))
            ET.SubElement(time_el, "beat-type").text = str(int(m.denominator))
            ET.SubElement(attrs, "staves").text = "2"
            clef1 = ET.SubElement(attrs, "clef", {"number": "1"})
            ET.SubElement(clef1, "sign").text = "G"
            ET.SubElement(clef1, "line").text = "2"
            clef2 = ET.SubElement(attrs, "clef", {"number": "2"})
            ET.SubElement(clef2, "sign").text = "F"
            ET.SubElement(clef2, "line").text = "4"
            prev_sig = sig

        for t in tempo_events:
            t_units = float(getattr(t, "time", 0.0) or 0.0)
            if t_units < m.start_units - _EPS or t_units >= m.end_units - _EPS:
                continue
            bpm = int(getattr(t, "tempo", 120) or 120)
            dur_units = float(getattr(t, "duration", QUARTER_NOTE_UNIT) or QUARTER_NOTE_UNIT)
            beat_unit, dot, per_minute, quarter_bpm = _tempo_beat_unit_and_quarter_bpm(dur_units, bpm)
            direction = ET.SubElement(measure_el, "direction", {"placement": "above"})
            direction_type = ET.SubElement(direction, "direction-type")
            metronome = ET.SubElement(direction_type, "metronome")
            ET.SubElement(metronome, "beat-unit").text = beat_unit
            if dot:
                ET.SubElement(metronome, "beat-unit-dot")
            ET.SubElement(metronome, "per-minute").text = str(per_minute)
            # <sound tempo> must always be in quarter-note BPM
            sound_bpm = max(1, int(round(quarter_bpm)))
            ET.SubElement(direction, "sound", {"tempo": str(sound_bpm)})

        measure_chunks = chunks_by_measure.get(m.index, [])

        # Staff 1: right hand; Staff 2: left hand
        for staff, hand in ((1, "r"), (2, "l")):
            if staff == 2:
                backup = ET.SubElement(measure_el, "backup")
                ET.SubElement(backup, "duration").text = str(max(1, _units_to_divisions(m.end_units - m.start_units)))
            hand_chunks = [c for c in measure_chunks if str(c.hand) == hand]
            voices = sorted(set(int(c.voice) for c in hand_chunks))
            if not voices:
                # Keep classic grand-staff structure explicit in every measure.
                _append_note(
                    measure_el,
                    duration_div=max(1, _units_to_divisions(m.end_units - m.start_units)),
                    voice=1,
                    staff=staff,
                    pitch=None,
                )
                continue

            for vi, voice in enumerate(voices):
                if vi > 0:
                    backup = ET.SubElement(measure_el, "backup")
                    ET.SubElement(backup, "duration").text = str(max(1, _units_to_divisions(m.end_units - m.start_units)))
                voice_chunks = [c for c in hand_chunks if int(c.voice) == int(voice)]
                _append_voice_material(measure_el, m, voice_chunks, voice=int(voice), staff=staff)

    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="  ")
    except Exception:
        pass

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output_path), encoding="utf-8", xml_declaration=True)

    max_voice_l = max((n.voice for n in notes if n.hand == "l"), default=0)
    max_voice_r = max((n.voice for n in notes if n.hand == "r"), default=0)
    return {
        "notes": len(notes),
        "measures": len(measures),
        "voices_left": int(max_voice_l),
        "voices_right": int(max_voice_r),
    }


def convert_piano_to_musicxml(input_path: Path, output_path: Path) -> dict[str, int]:
    score = SCORE().load(str(input_path))
    return export_score_to_musicxml(score, output_path)


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".musicxml")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert keyTAB .piano files to MusicXML (.musicxml).",
    )
    parser.add_argument("input", type=Path, help="Input .piano file path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .musicxml path (default: same name as input)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 2

    output_path = Path(args.output).expanduser().resolve() if args.output else _default_output_path(input_path)
    if str(output_path.suffix or "").lower() not in (".musicxml", ".xml"):
        output_path = output_path.with_suffix(".musicxml")

    try:
        stats = convert_piano_to_musicxml(input_path, output_path)
    except Exception as exc:
        print(f"Conversion failed: {exc}")
        return 1

    print(
        f"Converted {input_path.name} -> {output_path}\n"
        f"notes={stats['notes']} measures={stats['measures']} "
        f"voices_left={stats['voices_left']} voices_right={stats['voices_right']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
