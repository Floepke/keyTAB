from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import math

import pretty_midi
import mido  # fallback parser

from file_model.SCORE import SCORE
from utils.CONSTANT import QUARTER_NOTE_UNIT, GRACENOTE_THRESHOLD
from file_model.base_grid import BaseGrid


def _seconds_to_units(pm: pretty_midi.PrettyMIDI, t_seconds: float) -> float:
    """Convert an absolute time in seconds to app time units (QUARTER_NOTE_UNIT based).

    Uses PrettyMIDI's time->tick mapping to respect tempo changes, then maps
    ticks to quarter notes via pm.resolution.
    """
    try:
        ticks = pm.time_to_tick(float(t_seconds))
        quarters = float(ticks) / float(pm.resolution or 480)
        return quarters * QUARTER_NOTE_UNIT
    except Exception:
        # Fallback: assume constant tempo estimated by PrettyMIDI
        bpm = float(pm.estimate_tempo() or 120.0)
        quarter_sec = 60.0 / bpm
        return (t_seconds / quarter_sec) * QUARTER_NOTE_UNIT


def midi_load(path: str) -> SCORE:
    """Load a MIDI file and convert it into a SCORE model.

    - Notes mapped to SCORE.events.note with pitch/time/duration
    - Time mapping respects tempo changes via PrettyMIDI's tick conversion
    - Imports tempo changes as SCORE tempo markers
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MIDI file not found: {path}")

    # Try PrettyMIDI first; if it fails, fallback to mido
    try:
        pm = pretty_midi.PrettyMIDI(str(p))
    except Exception as exc:
        if mido is None:
            raise
        # Fallback: parse with mido
        return _midi_load_with_mido(str(p))

    score = SCORE().new()
    # Replace SCORE.new() default tempo with imported tempo markers.
    score.events.tempo = []
    score.info.title = str(p.stem or score.info.title)

    # Tempo: map all tempo changes to tempo markers (fixed duration = one quarter unit)
    times_arr, tempi_arr = pm.get_tempo_changes()
    times = [float(t) for t in list(times_arr) if times_arr is not None] if times_arr is not None else []
    tempi = [float(tp) for tp in list(tempi_arr) if tempi_arr is not None] if tempi_arr is not None else []
    if len(tempi) > 0:
        tempo_points = sorted(
            [(float(t_sec), int(round(float(bpm_val)))) for t_sec, bpm_val in zip(times, tempi)],
            key=lambda it: it[0],
        )
        if tempo_points and float(tempo_points[0][0]) > 0.0:
            tempo_points.insert(0, (0.0, int(tempo_points[0][1])))
        seen_times: set[int] = set()
        for t_sec, bpm_val in tempo_points:
            start_units = _seconds_to_units(pm, float(t_sec))
            key = int(round(float(start_units) * 1000.0))
            if key in seen_times:
                continue
            seen_times.add(key)
            score.new_tempo(time=float(start_units), duration=float(QUARTER_NOTE_UNIT), tempo=int(bpm_val))
    else:
        bpm_guess = int(round(float(pm.estimate_tempo() or 120.0)))
        score.new_tempo(time=0.0, duration=float(QUARTER_NOTE_UNIT), tempo=bpm_guess)

    # Map notes from all non-drum instruments (convert MIDI pitch to app pitch)
    for inst in pm.instruments:
        if getattr(inst, 'is_drum', False):
            continue
        for n in inst.notes:
            start_units = _seconds_to_units(pm, float(n.start))
            end_units = _seconds_to_units(pm, float(n.end))
            duration_units = max(0.0, float(end_units) - float(start_units))
            # MIDI A4 (69) -> app A4 (49): subtract 20
            app_pitch = int(n.pitch) - 20
            # Simple left/right hand heuristic around app middle C (~40)
            hand = '<' if int(app_pitch) < 40 else '>'
            vel = int(getattr(n, 'velocity', 64) or 64)
            vel = max(0, min(127, vel))
            if duration_units < float(GRACENOTE_THRESHOLD):
                score.new_grace_note(pitch=int(app_pitch), time=float(start_units))
            else:
                score.new_note(pitch=int(app_pitch), time=float(start_units), duration=float(duration_units), velocity=int(vel), hand=hand)

    # Build base_grid from time signature changes
    ts_changes = list(getattr(pm, 'time_signature_changes', []) or [])

    def _grid_positions_for(numer: int, denom: int) -> List[int]:
        # Return per-beat grouping sequence (one digit per beat)
        starts: List[int]
        if denom in (8, 16) and numer in (6, 7):
            starts = [1, 4]
        elif denom in (8, 16) and numer == 9:
            starts = [1, 4, 8]
        else:
            starts = [1]
        # Build sequence by counting within groups
        seq: List[int] = []
        count = 0
        start_set = set(starts)
        for beat in range(1, max(1, int(numer)) + 1):
            if beat in start_set:
                count = 1
            else:
                count += 1
            seq.append(count)
        return seq

    end_units_total = _seconds_to_units(pm, float(getattr(pm, 'get_end_time', lambda: 0.0)()))
    segments: List[Tuple[float, int, int]] = []  # (start_units, numer, denom)
    for ts in ts_changes:
        start_u = _seconds_to_units(pm, float(getattr(ts, 'time', 0.0) or 0.0))
        numer = int(getattr(ts, 'numerator', 4) or 4)
        denom = int(getattr(ts, 'denominator', 4) or 4)
        segments.append((start_u, numer, denom))
    # If no time signatures present, infer a single segment from defaults
    if not segments:
        segments = [(0.0, 4, 4)]

    # Sort segments by start_units
    segments.sort(key=lambda x: x[0])
    # Construct BaseGrid entries with measure_amount per segment
    base_grid_list: List[BaseGrid] = []
    for idx, (start_u, numer, denom) in enumerate(segments):
        next_start = segments[idx + 1][0] if idx + 1 < len(segments) else end_units_total
        dur_u = max(0.0, float(next_start) - float(start_u))
        # Measure length in app units: quarters_per_measure * QUARTER_NOTE_UNIT
        quarters_per_measure = (float(numer) * (4.0 / float(max(1, denom))))
        measure_units = quarters_per_measure * float(QUARTER_NOTE_UNIT)
        measures = int(max(1, math.ceil(dur_u / measure_units))) if measure_units > 0 else 1
        gp = _grid_positions_for(numer, denom)
        base_grid_list.append(BaseGrid(numerator=numer, denominator=denom, beat_grouping=gp, measure_amount=measures))

    # Apply
    score.base_grid = base_grid_list

    # Distribute initial line breaks in groups of six measures across the import
    score.apply_quick_line_breaks([6])

    return score


def _midi_load_with_mido(path: str) -> SCORE:
    """Fallback MIDI loader using mido; pairs note on/off and computes seconds across tempo changes."""
    mid = mido.MidiFile(filename=path)
    tpq = int(getattr(mid, 'ticks_per_beat', 480) or 480)
    # Default tempo in microseconds per beat (500k = 120 BPM)
    default_tempo = 500_000

    # Collect tempo changes from track 0 or meta across all tracks
    tempo_events: List[Tuple[int, int]] = []  # (abs_ticks, tempo)
    ts_events: List[Tuple[int, int, int]] = []  # (abs_ticks, numer, denom)
    for i, track in enumerate(mid.tracks):
        abs_ticks = 0
        for msg in track:
            abs_ticks += int(getattr(msg, 'time', 0) or 0)
            if msg.type == 'set_tempo':
                tempo_events.append((abs_ticks, int(msg.tempo)))
            elif msg.type == 'time_signature':
                numer = int(getattr(msg, 'numerator', 4) or 4)
                denom = int(getattr(msg, 'denominator', 4) or 4)
                ts_events.append((abs_ticks, numer, denom))
    tempo_events.sort(key=lambda x: x[0])
    ts_events.sort(key=lambda x: x[0])

    # Build function to convert delta ticks at a given segment tempo to seconds
    def ticks_to_seconds(delta_ticks: int, tempo_us: int) -> float:
        # seconds = delta_ticks * (tempo_us / 1e6) / tpq
        return (float(delta_ticks) * (float(tempo_us) / 1_000_000.0)) / float(tpq)

    # Helper: cumulative ticks->seconds at an absolute tick
    def ticks_to_seconds_at(abs_ticks_query: int) -> float:
        seconds = 0.0
        prev_ticks = 0
        cur_tempo = default_tempo
        for t, tempo in tempo_events:
            if t > abs_ticks_query:
                break
            seconds += ticks_to_seconds(t - prev_ticks, cur_tempo)
            prev_ticks = t
            cur_tempo = tempo
        seconds += ticks_to_seconds(abs_ticks_query - prev_ticks, cur_tempo)
        return seconds

    def ticks_to_units(abs_ticks: int) -> float:
        return (float(abs_ticks) / float(tpq)) * float(QUARTER_NOTE_UNIT)

    # Iterate messages to build absolute seconds timeline per track and pair notes
    score = SCORE().new()
    # Replace SCORE.new() default tempo with imported tempo markers.
    score.events.tempo = []
    score.info.title = str(Path(path).stem or score.info.title)

    bpm0 = 60.0 / ((tempo_events[0][1] if tempo_events else default_tempo) / 1_000_000.0)

    # Helper: get current tempo at given absolute ticks
    def tempo_at_ticks(abs_ticks: int) -> int:
        cur = default_tempo
        for t, tempo in tempo_events:
            if abs_ticks >= t:
                cur = tempo
            else:
                break
        return cur

    # Tempo markers: one per tempo change, fixed duration = one quarter unit
    if tempo_events:
        tempo_by_tick: Dict[int, int] = {}
        if int(tempo_events[0][0]) > 0:
            tempo_by_tick[0] = int(default_tempo)
        for tick_pos, tempo_us in tempo_events:
            tempo_by_tick[int(max(0, tick_pos))] = int(tempo_us)
        for tick_pos, tempo_us in sorted(tempo_by_tick.items(), key=lambda it: it[0]):
            bpm = int(round(60_000_000.0 / float(tempo_us)))
            t_units = ticks_to_units(tick_pos)
            score.new_tempo(time=float(t_units), duration=float(QUARTER_NOTE_UNIT), tempo=bpm)
    else:
        bpm_default = int(round(60.0 / (default_tempo / 1_000_000.0)))
        score.new_tempo(time=0.0, duration=float(QUARTER_NOTE_UNIT), tempo=bpm_default)

    total_end_seconds = 0.0
    # For each track, track note stacks by (channel, pitch)
    for track in mid.tracks:
        abs_ticks = 0
        # Stack entries: (velocity, start_seconds)
        note_stack: Dict[Tuple[int, int], List[Tuple[int, float]]] = {}
        abs_seconds = 0.0
        for msg in track:
            dt_ticks = int(getattr(msg, 'time', 0) or 0)
            if dt_ticks:
                # Advance absolute seconds using tempo at start of segment
                tempo_us = tempo_at_ticks(abs_ticks)
                abs_seconds += ticks_to_seconds(dt_ticks, tempo_us)
                abs_ticks += dt_ticks
            if msg.type == 'note_on' and msg.velocity > 0:
                key = (getattr(msg, 'channel', 0), msg.note)
                note_stack.setdefault(key, []).append((int(msg.velocity), abs_seconds))
            elif msg.type in ('note_off', 'note_on'):
                # Treat note_on with velocity 0 as off
                if msg.type == 'note_on' and msg.velocity > 0:
                    continue
                key = (getattr(msg, 'channel', 0), msg.note)
                lst = note_stack.get(key)
                if lst:
                    vel, start_sec = lst.pop()  # last started
                    end_sec = abs_seconds
                    start_units = (start_sec / (60.0 / bpm0)) * QUARTER_NOTE_UNIT
                    end_units = (end_sec / (60.0 / bpm0)) * QUARTER_NOTE_UNIT
                    duration_units = max(0.0, float(end_units) - float(start_units))
                    app_pitch = int(msg.note) - 20
                    hand = '<' if int(app_pitch) < 40 else '>'
                    if duration_units < float(GRACENOTE_THRESHOLD):
                        score.new_grace_note(pitch=int(app_pitch), time=float(start_units))
                    else:
                        score.new_note(pitch=int(app_pitch), time=float(start_units), duration=float(duration_units), velocity=int(max(0, min(127, vel))), hand=hand)
                    total_end_seconds = max(total_end_seconds, end_sec)
        # For any unmatched note_on left, close them with short duration
        for (ch, pitch), lst in note_stack.items():
            for _, start_sec in lst:
                end_sec = start_sec + 0.05
                start_units = (start_sec / (60.0 / bpm0)) * QUARTER_NOTE_UNIT
                end_units = (end_sec / (60.0 / bpm0)) * QUARTER_NOTE_UNIT
                duration_units = max(0.0, float(end_units) - float(start_units))
                app_pitch2 = int(pitch) - 20
                hand = '<' if int(app_pitch2) < 40 else '>'
            # If velocity not retained (should be), default to 64
            default_vel = 64
            if duration_units < float(GRACENOTE_THRESHOLD):
                score.new_grace_note(pitch=int(app_pitch2), time=float(start_units))
            else:
                score.new_note(pitch=int(app_pitch2), time=float(start_units), duration=float(duration_units), velocity=default_vel, hand=hand)
            total_end_seconds = max(total_end_seconds, end_sec)

    # Build base_grid from mido time signatures
    def _grid_positions_for(numer: int, denom: int) -> List[int]:
        # Return per-beat grouping sequence (one digit per beat)
        starts: List[int]
        if denom == 8 and numer in (6, 7):
            starts = [1, 4]
        elif denom == 8 and numer == 9:
            starts = [1, 4, 8]
        else:
            starts = [1]
        seq: List[int] = []
        count = 0
        start_set = set(starts)
        for beat in range(1, max(1, int(numer)) + 1):
            if beat in start_set:
                count = 1
            else:
                count += 1
            seq.append(count)
        return seq

    end_units_total = (total_end_seconds / (60.0 / bpm0)) * QUARTER_NOTE_UNIT
    segments: List[Tuple[float, int, int]] = []
    for abs_t, numer, denom in ts_events:
        t_sec = ticks_to_seconds_at(abs_t)
        t_units = (t_sec / (60.0 / bpm0)) * QUARTER_NOTE_UNIT
        segments.append((t_units, int(numer), int(denom)))
    if not segments:
        segments = [(0.0, 4, 4)]
    segments.sort(key=lambda x: x[0])
    base_grid_list: List[BaseGrid] = []
    for idx, (start_u, numer, denom) in enumerate(segments):
        next_start = segments[idx + 1][0] if idx + 1 < len(segments) else end_units_total
        dur_u = max(0.0, float(next_start) - float(start_u))
        quarters_per_measure = (float(numer) * (4.0 / float(max(1, denom))))
        measure_units = quarters_per_measure * float(QUARTER_NOTE_UNIT)
        measures = int(max(1, math.ceil(dur_u / measure_units))) if measure_units > 0 else 1
        gp = _grid_positions_for(numer, denom)
        base_grid_list.append(BaseGrid(numerator=numer, denominator=denom, beat_grouping=gp, measure_amount=measures))

    score.base_grid = base_grid_list

    return score
