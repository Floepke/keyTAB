from __future__ import annotations

import io
import struct
from pathlib import Path
from typing import List, Tuple

from file_model.SCORE import SCORE
from utils.CONSTANT import QUARTER_NOTE_UNIT

# ---------------------------------------------------------------------------
# Pure-Python MIDI writer (SMF format 1, no external dependencies)
# ---------------------------------------------------------------------------

_TPB = 480  # ticks per beat (quarter note)


# -- Low-level encoding helpers --------------------------------------------

def _vlq(value: int) -> bytes:
    """Encode a non-negative integer as a MIDI variable-length quantity."""
    value = max(0, int(value))
    buf = bytearray()
    buf.append(value & 0x7F)
    value >>= 7
    while value:
        buf.append((value & 0x7F) | 0x80)
        value >>= 7
    buf.reverse()
    return bytes(buf)


def _u16be(v: int) -> bytes:
    return struct.pack(">H", max(0, min(0xFFFF, int(v))))


def _u32be(v: int) -> bytes:
    return struct.pack(">I", max(0, min(0xFFFFFFFF, int(v))))


# -- Track builder ---------------------------------------------------------

def _make_track(events: List[Tuple[int, bytes]]) -> bytes:
    """Build a complete MTrk chunk from a list of (abs_tick, raw_message_bytes)."""
    events_sorted = sorted(events, key=lambda e: e[0])
    buf = io.BytesIO()
    last_tick = 0
    for abs_tick, msg_bytes in events_sorted:
        delta = max(0, int(abs_tick) - last_tick)
        buf.write(_vlq(delta))
        buf.write(msg_bytes)
        last_tick = int(abs_tick)
    # End-of-track meta message
    buf.write(_vlq(0))
    buf.write(b'\xFF\x2F\x00')
    payload = buf.getvalue()
    return b'MTrk' + _u32be(len(payload)) + payload


# -- Meta messages ---------------------------------------------------------

def _meta_set_tempo(tempo_us: int) -> bytes:
    us = max(1, min(0xFFFFFF, int(tempo_us)))
    data = bytes([(us >> 16) & 0xFF, (us >> 8) & 0xFF, us & 0xFF])
    return b'\xFF\x51' + _vlq(len(data)) + data


def _meta_time_signature(numer: int, denom: int) -> bytes:
    # denom must be a power of 2; encode as its log2
    denom = max(1, int(denom))
    denom_pw = 0
    d = denom
    while d > 1:
        denom_pw += 1
        d >>= 1
    data = bytes([max(1, int(numer)), denom_pw, 24, 8])  # 24 clocks/click, 8 32nds/quarter
    return b'\xFF\x58' + _vlq(len(data)) + data


# -- Channel messages ------------------------------------------------------

def _note_on(channel: int, pitch: int, velocity: int) -> bytes:
    return bytes([0x90 | (channel & 0x0F), pitch & 0x7F, velocity & 0x7F])


def _note_off(channel: int, pitch: int) -> bytes:
    return bytes([0x80 | (channel & 0x0F), pitch & 0x7F, 0x00])


# -- Unit conversion -------------------------------------------------------

def _units_to_ticks(units: float) -> int:
    return max(0, int(round((float(units) / float(QUARTER_NOTE_UNIT)) * float(_TPB))))


def _bpm_to_tempo_us(bpm: float) -> int:
    return max(1, int(round(60_000_000.0 / max(1.0, float(bpm)))))


def _tempo_marker_to_bpm(tp) -> float:
    marker_tempo = max(1.0, float(getattr(tp, "tempo", 120.0) or 120.0))
    marker_dur   = float(getattr(tp, "duration", float(QUARTER_NOTE_UNIT)) or float(QUARTER_NOTE_UNIT))
    marker_dur   = max(1e-6, marker_dur)
    return marker_tempo * (marker_dur / float(QUARTER_NOTE_UNIT))


def _closest_note_velocity(score: SCORE, t_units: float) -> int:
    notes = list(getattr(getattr(score, "events", None), "note", []) or [])
    if not notes:
        return 64
    closest = min(notes, key=lambda n: abs(float(getattr(n, "time", 0.0) or 0.0) - float(t_units)))
    return max(0, min(127, int(getattr(closest, "velocity", 64) or 64)))


# -- Public API ------------------------------------------------------------

def export_score_to_midi(score: SCORE, path: str | Path) -> None:
    """Export a SCORE to a Standard MIDI File (format 1, two tracks)."""

    # ---- Tempo track (track 0) ----------------------------------------
    tempos = sorted(
        list(getattr(getattr(score, "events", None), "tempo", []) or []),
        key=lambda t: float(getattr(t, "time", 0.0) or 0.0),
    )
    if not tempos or float(getattr(tempos[0], "time", 0.0) or 0.0) > 0.0:
        first_bpm = _tempo_marker_to_bpm(tempos[0]) if tempos else 120.0

        class _FakeTempo:
            time = 0.0
            tempo = first_bpm
            duration = float(QUARTER_NOTE_UNIT)

        tempos = [_FakeTempo()] + list(tempos)

    tempo_events: List[Tuple[int, bytes]] = []
    seen_ticks: set = set()
    for tp in tempos:
        tick = _units_to_ticks(float(getattr(tp, "time", 0.0) or 0.0))
        if tick in seen_ticks:
            continue
        seen_ticks.add(tick)
        bpm = max(1.0, _tempo_marker_to_bpm(tp))
        tempo_events.append((tick, _meta_set_tempo(_bpm_to_tempo_us(bpm))))

    # Time signature from the first base_grid entry (if present)
    base_grid = list(getattr(score, "base_grid", []) or [])
    if base_grid:
        bg = base_grid[0]
        numer = max(1, int(getattr(bg, "numerator",  4) or 4))
        denom = max(1, int(getattr(bg, "denominator", 4) or 4))
        tempo_events.append((0, _meta_time_signature(numer, denom)))

    tempo_track = _make_track(tempo_events)

    # ---- Note tracks --------------------------------------------------
    # Track 1 = left hand (channel 0), Track 2 = right hand (channel 1)
    right_events: List[Tuple[int, bytes]] = []
    left_events:  List[Tuple[int, bytes]] = []

    grace_dur_u = float(QUARTER_NOTE_UNIT) / 8.0

    for n in list(getattr(getattr(score, "events", None), "note", []) or []):
        start_u  = float(getattr(n, "time",     0.0) or 0.0)
        dur_u    = float(getattr(n, "duration", 0.0) or 0.0)
        end_u    = max(start_u, start_u + max(0.0, dur_u))
        on_tick  = _units_to_ticks(start_u)
        off_tick = max(on_tick + 1, _units_to_ticks(end_u))
        pitch    = max(0, min(127, int(getattr(n, "pitch", 40) or 40) + 20))
        vel      = max(1, min(127, int(getattr(n, "velocity", 64) or 64)))
        hand     = str(getattr(n, "hand", "r") or "r").strip().lower()
        if hand == "l":
            left_events.append((on_tick,  _note_on(0, pitch, vel)))
            left_events.append((off_tick, _note_off(0, pitch)))
        else:
            right_events.append((on_tick,  _note_on(1, pitch, vel)))
            right_events.append((off_tick, _note_off(1, pitch)))

    for g in list(getattr(getattr(score, "events", None), "grace_note", []) or []):
        start_u  = float(getattr(g, "time", 0.0) or 0.0)
        end_u    = start_u + grace_dur_u
        on_tick  = _units_to_ticks(start_u)
        off_tick = max(on_tick + 1, _units_to_ticks(end_u))
        pitch    = max(0, min(127, int(getattr(g, "pitch", 40) or 40) + 20))
        vel      = _closest_note_velocity(score, start_u)
        hand     = str(getattr(g, "hand", "r") or "r").strip().lower()
        if hand == "l":
            left_events.append((on_tick,  _note_on(0, pitch, vel)))
            left_events.append((off_tick, _note_off(0, pitch)))
        else:
            right_events.append((on_tick,  _note_on(1, pitch, vel)))
            right_events.append((off_tick, _note_off(1, pitch)))

    right_track = _make_track(right_events)
    left_track  = _make_track(left_events)

    # ---- Assemble SMF format-1 file -----------------------------------
    # Track 0: tempo/time-sig, Track 1: right hand, Track 2: left hand
    num_tracks = 3
    header = b'MThd' + _u32be(6) + _u16be(1) + _u16be(num_tracks) + _u16be(_TPB)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(header + tempo_track + right_track + left_track)
