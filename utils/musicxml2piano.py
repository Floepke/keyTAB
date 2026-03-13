from __future__ import annotations

import sys
import os

_THIS_FILE = os.path.abspath(__file__)
_UTILS_DIR = os.path.dirname(_THIS_FILE)
_PROJECT_ROOT = os.path.dirname(_UTILS_DIR)
if _UTILS_DIR in sys.path:
    sys.path.remove(_UTILS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import io
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from file_model.SCORE import SCORE
from file_model.base_grid import BaseGrid
from utils.CONSTANT import QUARTER_NOTE_UNIT


@dataclass
class ParsedNote:
    midi_pitch: int
    time_units: float
    duration_units: float
    stave_key: tuple[int, int]


@dataclass
class ParsedGrace:
    midi_pitch: int
    anchor_time_units: float
    stave_key: tuple[int, int]


@dataclass
class ParsedTempo:
    time_units: float
    bpm: float
    beat_duration_units: float


def _local_name(tag: str) -> str:
    if not isinstance(tag, str):
        return ""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def _find_child(node: ET.Element, name: str) -> Optional[ET.Element]:
    for child in list(node):
        if _local_name(child.tag) == name:
            return child
    return None


def _find_children(node: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in list(node) if _local_name(child.tag) == name]


def _text_of(node: ET.Element, name: str, default: str = "") -> str:
    child = _find_child(node, name)
    if child is None or child.text is None:
        return default
    return str(child.text).strip()


def _int_text(node: ET.Element, name: str, default: int) -> int:
    txt = _text_of(node, name, "")
    try:
        return int(txt)
    except Exception:
        return int(default)


def _float_text(node: ET.Element, name: str, default: float) -> float:
    txt = _text_of(node, name, "")
    try:
        return float(txt)
    except Exception:
        return float(default)


def _open_musicxml_text(path: Path) -> str:
    if path.suffix.lower() != ".mxl":
        return path.read_text(encoding="utf-8")

    with zipfile.ZipFile(path, "r") as archive:
        names = archive.namelist()
        if "META-INF/container.xml" in names:
            container_xml = archive.read("META-INF/container.xml")
            croot = ET.fromstring(container_xml)
            rootfiles = [
                elem
                for elem in croot.iter()
                if _local_name(elem.tag) == "rootfile"
            ]
            for rf in rootfiles:
                full_path = str(rf.attrib.get("full-path", "") or "")
                if full_path and full_path in names and full_path.lower().endswith((".xml", ".musicxml")):
                    return archive.read(full_path).decode("utf-8", errors="replace")

        xml_candidates = [n for n in names if n.lower().endswith((".xml", ".musicxml"))]
        if not xml_candidates:
            raise ValueError(f"No MusicXML file found in {path}")
        best = sorted(xml_candidates, key=lambda n: ("container" in n.lower(), len(n)))[0]
        return archive.read(best).decode("utf-8", errors="replace")


def _step_alter_octave_to_midi(step: str, alter: int, octave: int) -> int:
    semis = {
        "C": 0,
        "D": 2,
        "E": 4,
        "F": 5,
        "G": 7,
        "A": 9,
        "B": 11,
    }
    base = semis.get(step.upper(), 0)
    midi = (int(octave) + 1) * 12 + base + int(alter)
    return max(0, min(127, int(midi)))


def _beat_unit_to_quarters(beat_unit: str) -> float:
    token = str(beat_unit or "quarter").strip().lower()
    dotted = token.startswith("dotted-")
    token = token.replace("dotted-", "")
    base = {
        "whole": 4.0,
        "half": 2.0,
        "quarter": 1.0,
        "eighth": 0.5,
        "16th": 0.25,
        "32nd": 0.125,
        "64th": 0.0625,
    }.get(token, 1.0)
    if dotted:
        base *= 1.5
    return float(base)


def _grouping_for_signature(numer: int, denom: int) -> list[int]:
    n = max(1, int(numer))
    if denom in (8, 16) and n in (6, 7):
        starts = [1, 4]
    elif denom in (8, 16) and n == 9:
        starts = [1, 4, 8]
    else:
        starts = [1]
    start_set = set(starts)
    seq: list[int] = []
    counter = 0
    for beat in range(1, n + 1):
        if beat in start_set:
            counter = 1
        else:
            counter += 1
        seq.append(counter)
    return seq


def _note_tie_types(note_node: ET.Element) -> set[str]:
    tie_types: set[str] = set()
    for t in _find_children(note_node, "tie"):
        tie_type = str(t.attrib.get("type", "") or "").strip().lower()
        if tie_type in ("start", "stop"):
            tie_types.add(tie_type)

    notations = _find_child(note_node, "notations")
    if notations is not None:
        for tied in _find_children(notations, "tied"):
            tie_type = str(tied.attrib.get("type", "") or "").strip().lower()
            if tie_type in ("start", "stop"):
                tie_types.add(tie_type)
    return tie_types


def _sanitize_musicxml(raw: str) -> str:
    txt = str(raw or "")
    txt = re.sub(r"<!DOCTYPE[^>]*>", "", txt, flags=re.IGNORECASE)
    return txt


def parse_musicxml(path: Path) -> tuple[SCORE, dict[str, int]]:
    xml_text = _sanitize_musicxml(_open_musicxml_text(path))
    root = ET.parse(io.StringIO(xml_text)).getroot()

    if _local_name(root.tag) not in ("score-partwise", "score-timewise"):
        raise ValueError("Unsupported MusicXML root element. Expected score-partwise or score-timewise.")

    score = SCORE().new()
    score.events.note = []
    score.events.grace_note = []
    score.events.tempo = []

    title_node = next((n for n in root.iter() if _local_name(n.tag) == "work-title"), None)
    if title_node is not None and title_node.text:
        score.info.title = str(title_node.text).strip() or score.info.title

    creators = [n for n in root.iter() if _local_name(n.tag) == "creator"]
    composer = ""
    for c in creators:
        ctype = str(c.attrib.get("type", "") or "").strip().lower()
        if ctype == "composer" and c.text:
            composer = str(c.text).strip()
            break
    if not composer:
        for c in creators:
            if c.text and str(c.text).strip():
                composer = str(c.text).strip()
                break
    if composer:
        score.info.composer = composer

    part_nodes = [n for n in root.iter() if _local_name(n.tag) == "part"]
    notes: list[ParsedNote] = []
    graces: list[ParsedGrace] = []
    tempos: list[ParsedTempo] = []
    stave_pitch_acc: dict[tuple[int, int], list[int]] = {}

    signature_by_measure: list[tuple[int, int]] = []
    sig_current_num = 4
    sig_current_den = 4

    for part_index, part in enumerate(part_nodes):
        part_time_units = 0.0
        divisions = 1
        active_ties: dict[tuple[int, str, int, int], ParsedNote] = {}

        measures = [m for m in list(part) if _local_name(m.tag) == "measure"]
        for m_idx, measure in enumerate(measures):
            attrs = _find_child(measure, "attributes")
            if attrs is not None:
                new_div = _int_text(attrs, "divisions", divisions)
                divisions = max(1, int(new_div))

                time_node = _find_child(attrs, "time")
                if time_node is not None:
                    sig_current_num = max(1, _int_text(time_node, "beats", sig_current_num))
                    sig_current_den = max(1, _int_text(time_node, "beat-type", sig_current_den))

            if part_index == 0:
                signature_by_measure.append((int(sig_current_num), int(sig_current_den)))

            measure_start = float(part_time_units)
            measure_pos_div = 0.0
            last_non_chord_start_div = 0.0
            pending_grace_for_voice: dict[str, list[ParsedGrace]] = {}

            for item in list(measure):
                tag = _local_name(item.tag)

                if tag == "direction":
                    offset_div = _float_text(item, "offset", 0.0)
                    dir_pos_div = max(0.0, measure_pos_div + offset_div)
                    dir_time_units = measure_start + (dir_pos_div / float(divisions)) * float(QUARTER_NOTE_UNIT)

                    bpm = None
                    beat_quarters = 1.0

                    sound = _find_child(item, "sound")
                    if sound is not None:
                        tempo_attr = str(sound.attrib.get("tempo", "") or "").strip()
                        try:
                            bpm = float(tempo_attr)
                        except Exception:
                            bpm = None

                    d_type = _find_child(item, "direction-type")
                    if d_type is not None:
                        metr = _find_child(d_type, "metronome")
                        if metr is not None:
                            bu = _text_of(metr, "beat-unit", "quarter")
                            beat_quarters = _beat_unit_to_quarters(bu)
                            per_min = _float_text(metr, "per-minute", 0.0)
                            if per_min > 0:
                                bpm = float(per_min)

                    if bpm is not None and bpm > 0:
                        beat_duration_units = float(QUARTER_NOTE_UNIT) * float(beat_quarters)
                        tempos.append(
                            ParsedTempo(
                                time_units=float(dir_time_units),
                                bpm=float(bpm),
                                beat_duration_units=float(max(1.0, beat_duration_units)),
                            )
                        )

                elif tag == "backup":
                    dur_div = max(0.0, _float_text(item, "duration", 0.0))
                    measure_pos_div = max(0.0, measure_pos_div - dur_div)

                elif tag == "forward":
                    dur_div = max(0.0, _float_text(item, "duration", 0.0))
                    measure_pos_div += dur_div

                elif tag == "note":
                    if _find_child(item, "rest") is not None:
                        if _find_child(item, "grace") is None and _find_child(item, "chord") is None:
                            measure_pos_div += max(0.0, _float_text(item, "duration", 0.0))
                        continue

                    voice = _text_of(item, "voice", "1") or "1"
                    staff_no = _int_text(item, "staff", 1)
                    stave_key = (part_index, int(staff_no))

                    pitch_node = _find_child(item, "pitch")
                    if pitch_node is None:
                        continue
                    step = _text_of(pitch_node, "step", "C")
                    alter = _int_text(pitch_node, "alter", 0)
                    octave = _int_text(pitch_node, "octave", 4)
                    midi_pitch = _step_alter_octave_to_midi(step, alter, octave)

                    is_grace = _find_child(item, "grace") is not None
                    is_chord = _find_child(item, "chord") is not None
                    dur_div = max(0.0, _float_text(item, "duration", 0.0))

                    if is_grace:
                        g = ParsedGrace(
                            midi_pitch=int(midi_pitch),
                            anchor_time_units=measure_start + (measure_pos_div / float(divisions)) * float(QUARTER_NOTE_UNIT),
                            stave_key=stave_key,
                        )
                        pending_grace_for_voice.setdefault(voice, []).append(g)
                        stave_pitch_acc.setdefault(stave_key, []).append(int(midi_pitch))
                        continue

                    start_div = last_non_chord_start_div if is_chord else measure_pos_div
                    start_units = measure_start + (start_div / float(divisions)) * float(QUARTER_NOTE_UNIT)
                    dur_units = (dur_div / float(divisions)) * float(QUARTER_NOTE_UNIT)

                    tie_types = _note_tie_types(item)
                    has_tie_start = "start" in tie_types
                    has_tie_stop = "stop" in tie_types
                    tie_key = (part_index, str(voice), int(staff_no), int(midi_pitch))

                    active = active_ties.get(tie_key)
                    if has_tie_stop:
                        if active is None:
                            active = ParsedNote(
                                midi_pitch=int(midi_pitch),
                                time_units=float(start_units),
                                duration_units=float(max(0.0, dur_units)),
                                stave_key=stave_key,
                            )
                            active_ties[tie_key] = active
                        else:
                            active.duration_units += float(max(0.0, dur_units))

                    if has_tie_start:
                        if active is None:
                            active_ties[tie_key] = ParsedNote(
                                midi_pitch=int(midi_pitch),
                                time_units=float(start_units),
                                duration_units=float(max(0.0, dur_units)),
                                stave_key=stave_key,
                            )
                        # middle tie segment (stop+start) remains active; do not append yet
                    else:
                        if has_tie_stop:
                            finished = active_ties.pop(tie_key, None)
                            if finished is not None:
                                notes.append(finished)
                            else:
                                notes.append(
                                    ParsedNote(
                                        midi_pitch=int(midi_pitch),
                                        time_units=float(start_units),
                                        duration_units=float(max(0.0, dur_units)),
                                        stave_key=stave_key,
                                    )
                                )
                        else:
                            # Untied standalone note
                            if active is not None:
                                # Malformed dangling tie chain: flush it before standalone note.
                                dangling = active_ties.pop(tie_key, None)
                                if dangling is not None:
                                    notes.append(dangling)
                            notes.append(
                                ParsedNote(
                                    midi_pitch=int(midi_pitch),
                                    time_units=float(start_units),
                                    duration_units=float(max(0.0, dur_units)),
                                    stave_key=stave_key,
                                )
                            )

                    stave_pitch_acc.setdefault(stave_key, []).append(int(midi_pitch))

                    if voice in pending_grace_for_voice and pending_grace_for_voice[voice]:
                        grace_batch = pending_grace_for_voice.pop(voice)
                        count = len(grace_batch)
                        for gi, g in enumerate(grace_batch):
                            offset = float((count - gi) * 0.5)
                            graces.append(
                                ParsedGrace(
                                    midi_pitch=int(g.midi_pitch),
                                    anchor_time_units=max(0.0, float(start_units) - offset),
                                    stave_key=g.stave_key,
                                )
                            )

                    if not is_chord:
                        last_non_chord_start_div = start_div
                        measure_pos_div += dur_div

            while len(signature_by_measure) < (m_idx + 1) and part_index == 0:
                signature_by_measure.append((int(sig_current_num), int(sig_current_den)))

            measure_quarters = float(sig_current_num) * (4.0 / float(max(1, sig_current_den)))
            part_time_units += measure_quarters * float(QUARTER_NOTE_UNIT)

        # Flush any unfinished ties at part end.
        for pending in active_ties.values():
            notes.append(pending)

    staff_avg: dict[tuple[int, int], float] = {}
    for key, pitches in stave_pitch_acc.items():
        if pitches:
            staff_avg[key] = float(sum(pitches) / len(pitches))

    if staff_avg:
        right_stave = max(staff_avg.items(), key=lambda it: it[1])[0]
    else:
        right_stave = (0, 1)

    for n in sorted(notes, key=lambda it: (it.time_units, it.midi_pitch)):
        app_pitch = int(n.midi_pitch) - 20
        hand = ">" if n.stave_key == right_stave else "<"
        score.new_note(
            pitch=int(app_pitch),
            time=float(n.time_units),
            duration=float(max(1.0, n.duration_units)),
            hand=hand,
            velocity=64,
        )

    for g in sorted(graces, key=lambda it: (it.anchor_time_units, it.midi_pitch)):
        app_pitch = int(g.midi_pitch) - 20
        score.new_grace_note(
            pitch=int(app_pitch),
            time=float(max(0.0, g.anchor_time_units)),
        )

    tempos_sorted = sorted(tempos, key=lambda t: t.time_units)
    tempo_unique: list[ParsedTempo] = []
    seen_tempo_times: set[int] = set()
    for t in tempos_sorted:
        key = int(round(float(t.time_units) * 1000.0))
        if key in seen_tempo_times:
            continue
        seen_tempo_times.add(key)
        tempo_unique.append(t)

    score.events.tempo = []
    if tempo_unique:
        if float(tempo_unique[0].time_units) > 0.0:
            score.new_tempo(
                time=0.0,
                duration=float(max(1.0, tempo_unique[0].beat_duration_units)),
                tempo=int(round(tempo_unique[0].bpm)),
            )
        for t in tempo_unique:
            score.new_tempo(
                time=float(t.time_units),
                duration=float(max(1.0, t.beat_duration_units)),
                tempo=int(round(t.bpm)),
            )
    else:
        score.new_tempo(
            time=0.0,
            duration=float(QUARTER_NOTE_UNIT),
            tempo=60,
        )

    if signature_by_measure:
        merged: list[BaseGrid] = []
        cur_num, cur_den = signature_by_measure[0]
        count = 1
        for numer, denom in signature_by_measure[1:]:
            if numer == cur_num and denom == cur_den:
                count += 1
            else:
                merged.append(
                    BaseGrid(
                        numerator=int(cur_num),
                        denominator=int(cur_den),
                        beat_grouping=_grouping_for_signature(cur_num, cur_den),
                        measure_amount=int(count),
                        indicator_enabled=True,
                    )
                )
                cur_num, cur_den = numer, denom
                count = 1
        merged.append(
            BaseGrid(
                numerator=int(cur_num),
                denominator=int(cur_den),
                beat_grouping=_grouping_for_signature(cur_num, cur_den),
                measure_amount=int(count),
                indicator_enabled=True,
            )
        )
        score.base_grid = merged

    try:
        score.apply_quick_line_breaks([6])
    except Exception:
        pass

    return score, {
        "notes": len(score.events.note),
        "grace_notes": len(score.events.grace_note),
        "tempo_events": len(score.events.tempo),
        "time_signature_segments": len(score.base_grid),
    }


def convert_musicxml_to_piano(input_path: Path, output_path: Path) -> dict[str, int]:
    score, stats = parse_musicxml(input_path)
    score.save(str(output_path))
    return stats


def _default_output_path(input_path: Path) -> Path:
    return input_path.with_suffix(".piano")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert MusicXML (.musicxml/.xml/.mxl) to keyTAB .piano format.",
    )
    parser.add_argument("input", type=Path, help="Input MusicXML file path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output .piano path (default: same name as input)",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}", file=sys.stderr)
        return 2

    if input_path.suffix.lower() not in (".xml", ".musicxml", ".mxl"):
        print("Warning: input does not look like MusicXML (.xml/.musicxml/.mxl); attempting parse anyway.", file=sys.stderr)

    output_path = Path(args.output).expanduser().resolve() if args.output else _default_output_path(input_path)
    if output_path.suffix.lower() != ".piano":
        output_path = output_path.with_suffix(".piano")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        stats = convert_musicxml_to_piano(input_path, output_path)
    except Exception as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 1

    print(
        f"Converted {input_path.name} -> {output_path}\n"
        f"notes={stats['notes']} grace_notes={stats['grace_notes']} "
        f"tempo_events={stats['tempo_events']} time_signature_segments={stats['time_signature_segments']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
