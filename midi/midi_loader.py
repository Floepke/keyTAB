from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import math
import struct

from file_model.SCORE import SCORE
from utils.CONSTANT import QUARTER_NOTE_UNIT, GRACENOTE_THRESHOLD
from file_model.base_grid import BaseGrid

# ---------------------------------------------------------------------------
# Pure-Python byte-level MIDI parser
# Handles corrupt/truncated chunks, missing meta fields, and running status.
# Never raises on malformed content inside a track - just skips the bad data.
# ---------------------------------------------------------------------------

class _MidiParseError(Exception):
    pass


class _Buf:
    """Positional reader over a bytes slice that silently returns None on overread."""
    __slots__ = ("_data", "pos", "end")

    def __init__(self, data: bytes, start: int = 0, end: int | None = None) -> None:
        self._data = data
        self.pos = start
        self.end = len(data) if end is None else end

    def remaining(self) -> int:
        return max(0, self.end - self.pos)

    def read(self, n: int) -> bytes:
        start = self.pos
        self.pos = min(self.end, self.pos + n)
        return self._data[start : self.pos]

    def read_u8(self) -> int | None:
        if self.pos >= self.end:
            return None
        v = self._data[self.pos]
        self.pos += 1
        return v

    def read_u16_be(self) -> int | None:
        raw = self.read(2)
        return struct.unpack(">H", raw)[0] if len(raw) == 2 else None

    def read_u32_be(self) -> int | None:
        raw = self.read(4)
        return struct.unpack(">I", raw)[0] if len(raw) == 4 else None

    def read_vlq(self) -> int | None:
        """Variable-length quantity (up to 4 bytes). Returns None on EOF."""
        value = 0
        for _ in range(4):
            b = self.read_u8()
            if b is None:
                return None
            value = (value << 7) | (b & 0x7F)
            if not (b & 0x80):
                return value
        return value  # truncated VLQ – return what we have


def _parse_midi_file(data: bytes) -> Tuple[int, List[List]]:
    """Parse raw MIDI bytes into (ticks_per_quarter, tracks).

    Each track is a list of event tuples:
      ('note_on',  abs_tick, channel, pitch, velocity)
      ('note_off', abs_tick, channel, pitch, velocity)
      ('tempo',    abs_tick, tempo_us)
      ('time_sig', abs_tick, numerator, denominator)
    """
    buf = _Buf(data)

    # -- MThd header --
    if buf.read(4) != b'MThd':
        raise _MidiParseError("Not a MIDI file: missing MThd marker")
    hdr_len = buf.read_u32_be()
    if hdr_len is None:
        raise _MidiParseError("Truncated MIDI header")
    hdr = _Buf(buf.read(int(hdr_len)))
    hdr.read_u16_be()  # format (0/1/2) – not needed for import
    hdr.read_u16_be()  # num_tracks – we read until EOF instead
    division = hdr.read_u16_be() or 480
    if division & 0x8000:
        # SMPTE timecode: approximate tpq from fps × subframes
        fps = (division >> 8) & 0x7F
        subs = division & 0xFF
        tpq = max(1, int(fps * subs // 4))
    else:
        tpq = max(1, int(division))

    # -- MTrk chunks --
    tracks: List[List] = []
    while buf.remaining() >= 8:
        chunk_id = buf.read(4)
        chunk_len_raw = buf.read_u32_be()
        if chunk_len_raw is None:
            break
        chunk_len = int(chunk_len_raw)
        chunk_data = buf.read(chunk_len)
        if chunk_id != b'MTrk':
            continue  # skip RIFF or other foreign chunks

        events: List = []
        tb = _Buf(chunk_data)
        abs_tick = 0
        running_status: int | None = None

        while tb.remaining() > 0:
            delta = tb.read_vlq()
            if delta is None:
                break
            abs_tick += int(delta)

            b = tb.read_u8()
            if b is None:
                break

            # -- Meta event --
            if b == 0xFF:
                meta_type = tb.read_u8()
                meta_len = tb.read_vlq()
                if meta_type is None or meta_len is None:
                    break
                meta_bytes = tb.read(int(meta_len))
                # meta events do NOT update running_status
                if meta_type in (0x03, 0x04):
                    # Track Name / Instrument Name
                    try:
                        events.append(('name', abs_tick, meta_bytes.decode('utf-8', errors='replace')))
                    except Exception:
                        pass
                elif meta_type == 0x51 and len(meta_bytes) >= 3:
                    # Set Tempo
                    us = (meta_bytes[0] << 16) | (meta_bytes[1] << 8) | meta_bytes[2]
                    events.append(('tempo', abs_tick, max(1, us)))
                elif meta_type == 0x58:
                    # Time Signature – all fields optional/defaulted if truncated
                    numer    = meta_bytes[0] if len(meta_bytes) > 0 else 4
                    denom_pw = meta_bytes[1] if len(meta_bytes) > 1 else 2
                    events.append(('time_sig', abs_tick, max(1, int(numer)), 2 ** max(0, int(denom_pw))))
                elif meta_type == 0x2F:
                    break  # end of track
                continue

            # -- SysEx --
            if b in (0xF0, 0xF7):
                slen = tb.read_vlq()
                if slen is not None:
                    tb.read(int(slen))
                running_status = None  # SysEx cancels running status
                continue

            # -- Channel messages --
            if b & 0x80:
                # New status byte; next byte is first data byte
                running_status = b
                d1 = tb.read_u8()
            else:
                # Running status – b is the first data byte
                if running_status is None:
                    continue  # can't decode without a status – skip
                d1 = b

            if running_status is None:
                continue

            status_nibble = (running_status >> 4) & 0x0F
            channel       = running_status & 0x0F

            # Program change (0xC) and channel pressure (0xD) are 1-data-byte messages
            if status_nibble in (0x0C, 0x0D):
                if d1 is None:
                    break
                continue  # not used for note import

            # All remaining channel messages have 2 data bytes
            d2 = tb.read_u8()
            if d1 is None or d2 is None:
                break

            if status_nibble == 0x09:
                if d2 == 0:
                    events.append(('note_off', abs_tick, channel, d1, 0))
                else:
                    events.append(('note_on', abs_tick, channel, d1, d2))
            elif status_nibble == 0x08:
                events.append(('note_off', abs_tick, channel, d1, d2))
            # CC, pitch-bend, poly-pressure – ignored for now

        tracks.append(events)

    return tpq, tracks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_tempo_map(tracks: List[List], default_us: int = 500_000) -> List[Tuple[int, int]]:
    """Merge tempo events from all tracks into a sorted [(abs_tick, tempo_us)] list."""
    tempos: List[Tuple[int, int]] = []
    for track in tracks:
        for ev in track:
            if ev[0] == 'tempo':
                tempos.append((int(ev[1]), int(ev[2])))
    tempos.sort(key=lambda x: x[0])
    if not tempos or tempos[0][0] > 0:
        tempos.insert(0, (0, default_us))
    return tempos


def _ticks_to_units(abs_ticks: int, tpq: int) -> float:
    """Convert absolute MIDI ticks to app units.

    App units are quarter-note based: QUARTER_NOTE_UNIT per quarter note.
    This conversion is purely tick/tpq and is independent of tempo.
    """
    return (float(abs_ticks) / float(tpq)) * float(QUARTER_NOTE_UNIT)


def _grid_positions_for(numer: int, denom: int) -> List[int]:
    if denom in (8, 16) and numer in (6, 7):
        starts = [1, 4]
    elif denom in (8, 16) and numer == 9:
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


def _emit_note(score: SCORE, tpq: int, on_tick: int, off_tick: int, pitch: int, vel: int, hand: str | None = None) -> None:
    start_units  = _ticks_to_units(on_tick,  tpq)
    end_units    = _ticks_to_units(off_tick, tpq)
    dur_units    = max(0.0, end_units - start_units)
    app_pitch    = int(pitch) - 20  # MIDI A4=69 -> app A4=49
    if hand is None:
        hand = 'l' if app_pitch < 40 else 'r'
    vel          = max(0, min(127, int(vel)))
    if dur_units < float(GRACENOTE_THRESHOLD):
        score.new_grace_note(pitch=int(app_pitch), time=float(start_units))
    else:
        score.new_note(
            pitch=int(app_pitch),
            time=float(start_units),
            duration=float(dur_units),
            velocity=vel,
            hand=hand,
        )


def _midi_pitch_to_name(midi_pitch: int) -> str:
    """Convert a MIDI pitch number to a human-readable note name (e.g. 60 → 'C4')."""
    names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (int(midi_pitch) // 12) - 1
    return f"{names[int(midi_pitch) % 12]}{octave}"


def midi_analyze_tracks(path: str) -> List[dict]:
    """Analyze a MIDI file and return per-track metadata.

    Returns a list of dicts (one per non-empty, non-drum-only track) with:
        index        – 0-based track index in the raw MIDI file
        name         – track name from meta 0x03/0x04 event, or 'Track N'
        note_count   – number of non-drum note_on events with velocity > 0
        min_pitch    – lowest MIDI pitch seen (as note name string)
        max_pitch    – highest MIDI pitch seen (as note name string)
        default_hand – 'l', 'r', or 'skip' (skip for drum-only tracks)
    Tracks that contain no notes at all are omitted.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MIDI file not found: {path}")
    _tpq, tracks = _parse_midi_file(p.read_bytes())

    result: List[dict] = []
    for idx, track in enumerate(tracks):
        # Collect name from first name meta event
        name = ""
        for ev in track:
            if ev[0] == 'name':
                name = str(ev[2]).strip()
                if name:
                    break
        if not name:
            name = f"Track {idx + 1}"

        # Count notes and collect pitch statistics
        note_pitches: List[int] = []
        all_drum = True
        has_non_drum = False
        for ev in track:
            if ev[0] == 'note_on' and int(ev[4]) > 0:  # velocity > 0
                ch = int(ev[2])
                pitch = int(ev[3])
                if ch == 9:
                    continue  # drum — don't count
                has_non_drum = True
                all_drum = False
                note_pitches.append(pitch)
            elif ev[0] in ('note_on', 'note_off'):
                if int(ev[2]) != 9:
                    all_drum = False

        if not note_pitches:
            continue  # skip empty / drum-only tracks

        avg_pitch = sum(note_pitches) / len(note_pitches)
        default_hand = 'l' if avg_pitch < 60 else 'r'

        result.append({
            'index': idx,
            'name': name,
            'note_count': len(note_pitches),
            'min_pitch': _midi_pitch_to_name(min(note_pitches)),
            'max_pitch': _midi_pitch_to_name(max(note_pitches)),
            'default_hand': default_hand,
        })

    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def midi_load(path: str, track_assignments: Dict[int, str] | None = None) -> SCORE:
    """Load a MIDI file and convert it to a SCORE model.

    Uses a pure-Python byte-level parser with no dependency on mido or
    pretty_midi for parsing.  Malformed/truncated data is silently skipped
    rather than raising an exception.

    Args:
        path:             Path to the MIDI file.
        track_assignments: Optional dict mapping 0-based track index to 'l'
                           (left hand), 'r' (right hand), or 'skip'. When
                           provided, the assigned hand overrides the default
                           pitch-based heuristic; 'skip' excludes all notes
                           from that track.  When None, the original
                           pitch-based heuristic is used for every note.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"MIDI file not found: {path}")

    raw = p.read_bytes()
    tpq, tracks = _parse_midi_file(raw)

    # ---- Tempo map ------------------------------------------------
    DEFAULT_US = 500_000
    tempo_map = _build_tempo_map(tracks, DEFAULT_US)

    # ---- Time signature events ------------------------------------
    ts_events: List[Tuple[int, int, int]] = []  # (abs_tick, numer, denom)
    seen_ts: set[int] = set()
    for track in tracks:
        for ev in track:
            if ev[0] == 'time_sig':
                tick = int(ev[1])
                if tick not in seen_ts:
                    seen_ts.add(tick)
                    ts_events.append((tick, int(ev[2]), int(ev[3])))
    ts_events.sort(key=lambda x: x[0])

    # ---- Build SCORE ----------------------------------------------
    score = SCORE().new()
    score.tempo = []
    score.info.title = str(p.stem or score.info.title)

    # Tempo markers
    seen_tick: set[int] = set()
    for tick_pos, tempo_us in tempo_map:
        if tick_pos in seen_tick:
            continue
        seen_tick.add(tick_pos)
        bpm = max(1, min(999, int(round(60_000_000.0 / float(tempo_us)))))
        score.new_tempo(
            time=float(_ticks_to_units(tick_pos, tpq)),
            duration=float(QUARTER_NOTE_UNIT),
            tempo=bpm,
        )

    # Notes – pair note_on / note_off per (channel, pitch) within each track.
    # Drum channel (MIDI ch 9) is skipped.
    max_end_tick = 0
    for track_idx, track in enumerate(tracks):
        # Determine hand override for this track (if assignments provided)
        if track_assignments is not None:
            track_hand = track_assignments.get(track_idx)
            if track_hand == 'skip':
                continue  # exclude this track entirely
            if track_hand not in ('l', 'r'):
                track_hand = None  # fall back to pitch heuristic
        else:
            track_hand = None  # always use pitch heuristic

        open_notes: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        for ev in track:
            kind = ev[0]
            if kind == 'note_on':
                _, tick, ch, pitch, vel = ev
                if ch == 9:  # drum channel
                    continue
                open_notes.setdefault((ch, pitch), []).append((int(vel), int(tick)))
            elif kind == 'note_off':
                _, tick, ch, pitch, _vel = ev
                if ch == 9:
                    continue
                lst = open_notes.get((ch, pitch))
                if lst:
                    on_vel, on_tick = lst.pop()
                    _emit_note(score, tpq, on_tick, int(tick), pitch, on_vel, hand=track_hand)
                    max_end_tick = max(max_end_tick, int(tick))

        # Close any dangling note_ons with a 1/32-note fallback duration
        for (ch, pitch), lst in open_notes.items():
            if ch == 9:
                continue
            for vel, on_tick in lst:
                off_tick = on_tick + max(1, tpq // 8)
                _emit_note(score, tpq, on_tick, off_tick, pitch, vel, hand=track_hand)
                max_end_tick = max(max_end_tick, off_tick)

    # ---- Base grid from time signatures ---------------------------
    end_units = _ticks_to_units(max_end_tick, tpq)
    segments: List[Tuple[float, int, int]] = [
        (_ticks_to_units(t, tpq), n, d) for t, n, d in ts_events
    ]
    if not segments:
        segments = [(0.0, 4, 4)]
    segments.sort(key=lambda x: x[0])

    base_grid_list: List[BaseGrid] = []
    for idx, (start_u, numer, denom) in enumerate(segments):
        next_start = segments[idx + 1][0] if idx + 1 < len(segments) else end_units
        dur_u = max(0.0, float(next_start) - float(start_u))
        quarters_per_measure = float(numer) * (4.0 / float(max(1, denom)))
        measure_units = quarters_per_measure * float(QUARTER_NOTE_UNIT)
        measures = int(max(1, math.ceil(dur_u / measure_units))) if measure_units > 0 else 1
        gp = _grid_positions_for(numer, denom)
        base_grid_list.append(
            BaseGrid(numerator=numer, denominator=denom, beat_grouping=gp, measure_amount=measures)
        )

    score.base_grid = base_grid_list
    score.apply_quick_line_breaks([6])
    return score


