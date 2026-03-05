from __future__ import annotations

from pathlib import Path

import mido

from file_model.SCORE import SCORE
from utils.CONSTANT import QUARTER_NOTE_UNIT


def _units_to_ticks(units: float, ticks_per_beat: int) -> int:
    return int(round((float(units) / float(QUARTER_NOTE_UNIT)) * float(ticks_per_beat)))


def _tempo_marker_to_quarter_bpm(tp) -> float:
    marker_tempo = max(1.0, float(getattr(tp, "tempo", 120.0) or 120.0))
    marker_duration = float(getattr(tp, "duration", float(QUARTER_NOTE_UNIT)) or float(QUARTER_NOTE_UNIT))
    marker_duration = max(1e-6, marker_duration)
    return marker_tempo * (marker_duration / float(QUARTER_NOTE_UNIT))


def _closest_note_velocity(score: SCORE, t_units: float) -> int:
    notes = list(getattr(getattr(score, "events", None), "note", []) or [])
    if not notes:
        return 64
    closest = min(notes, key=lambda n: abs(float(getattr(n, "time", 0.0) or 0.0) - float(t_units)))
    v = int(getattr(closest, "velocity", 64) or 64)
    return max(0, min(127, v))


def export_score_to_midi(score: SCORE, path: str | Path) -> None:
    ticks_per_beat = 480
    mid = mido.MidiFile(ticks_per_beat=ticks_per_beat)

    tempo_track = mido.MidiTrack()
    note_track = mido.MidiTrack()
    mid.tracks.append(tempo_track)
    mid.tracks.append(note_track)

    tempos = sorted(
        list(getattr(getattr(score, "events", None), "tempo", []) or []),
        key=lambda t: float(getattr(t, "time", 0.0) or 0.0),
    )
    if tempos:
        if float(getattr(tempos[0], "time", 0.0) or 0.0) > 0.0:
            first_qpm = _tempo_marker_to_quarter_bpm(tempos[0])
            tempos = [
                type(
                    "_TmpTempo",
                    (),
                    {"time": 0.0, "tempo": first_qpm, "duration": float(QUARTER_NOTE_UNIT)},
                )()
            ] + tempos
    else:
        tempos = [
            type(
                "_TmpTempo",
                (),
                {"time": 0.0, "tempo": 120.0, "duration": float(QUARTER_NOTE_UNIT)},
            )()
        ]

    tempo_events = []
    for tp in tempos:
        t_units = float(getattr(tp, "time", 0.0) or 0.0)
        quarter_bpm = max(1.0, _tempo_marker_to_quarter_bpm(tp))
        tempo_events.append((_units_to_ticks(t_units, ticks_per_beat), float(quarter_bpm)))

    deduped_tempos = {}
    for tick, bpm in tempo_events:
        deduped_tempos[int(max(0, tick))] = float(bpm)
    sorted_tempos = sorted(deduped_tempos.items(), key=lambda x: x[0])

    last_tick = 0
    for tick, bpm in sorted_tempos:
        delta = max(0, int(tick) - int(last_tick))
        tempo_track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(float(max(1.0, bpm))), time=delta))
        last_tick = int(tick)
    tempo_track.append(mido.MetaMessage("end_of_track", time=0))

    note_events = []
    for n in list(getattr(score.events, "note", []) or []):
        start_u = float(getattr(n, "time", 0.0) or 0.0)
        dur_u = float(getattr(n, "duration", 0.0) or 0.0)
        end_u = max(start_u, start_u + max(0.0, dur_u))
        start_tick = max(0, _units_to_ticks(start_u, ticks_per_beat))
        end_tick = max(start_tick + 1, _units_to_ticks(end_u, ticks_per_beat))
        pitch = max(0, min(127, int(getattr(n, "pitch", 40) or 40) + 20))
        vel = max(0, min(127, int(getattr(n, "velocity", 64) or 64)))
        note_events.append((start_tick, 0, mido.Message("note_on", note=pitch, velocity=vel, time=0)))
        note_events.append((end_tick, 1, mido.Message("note_off", note=pitch, velocity=0, time=0)))

    grace_dur_u = float(QUARTER_NOTE_UNIT) / 8.0
    for g in list(getattr(score.events, "grace_note", []) or []):
        start_u = float(getattr(g, "time", 0.0) or 0.0)
        end_u = start_u + grace_dur_u
        start_tick = max(0, _units_to_ticks(start_u, ticks_per_beat))
        end_tick = max(start_tick + 1, _units_to_ticks(end_u, ticks_per_beat))
        pitch = max(0, min(127, int(getattr(g, "pitch", 40) or 40) + 20))
        vel = _closest_note_velocity(score, start_u)
        note_events.append((start_tick, 0, mido.Message("note_on", note=pitch, velocity=vel, time=0)))
        note_events.append((end_tick, 1, mido.Message("note_off", note=pitch, velocity=0, time=0)))

    note_events.sort(key=lambda e: (int(e[0]), int(e[1])))
    last_tick = 0
    for tick, _prio, msg in note_events:
        msg.time = max(0, int(tick) - int(last_tick))
        note_track.append(msg)
        last_tick = int(tick)
    note_track.append(mido.MetaMessage("end_of_track", time=0))

    mid.save(str(Path(path)))
