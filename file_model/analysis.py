from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any


@dataclass
class Analysis:
    notes: int = 0
    grace_notes: int = 0
    pages: int = 0
    measures: int = 0
    lines: int = 0
    avg_frequency_hz: float = 0.0
    pitch_range_low: int = 0
    pitch_range_high: int = 0
    most_used_pitch: int = 0
    left_hand_notes: int = 0
    right_hand_notes: int = 0

    @staticmethod
    def _list_from_events(events: Any, name: str) -> list:
        try:
            if isinstance(events, dict):
                value = events.get(name, []) or []
            else:
                value = getattr(events, name, []) if hasattr(events, name) else []
        except Exception:
            value = []
        if isinstance(value, list):
            return value
        try:
            return list(value)
        except Exception:
            return []

    @staticmethod
    def _measure_count(base_grid: Any) -> int:
        total = 0
        try:
            iterable = list(base_grid or [])
        except Exception:
            iterable = []
        for bg in iterable:
            try:
                if isinstance(bg, dict):
                    measures = int(bg.get("measure_amount", 0) or 0)
                else:
                    measures = int(getattr(bg, "measure_amount", 0) or 0)
            except Exception:
                measures = 0
            total += max(0, measures)
        return total

    @staticmethod
    def _avg_frequency(note_list: list) -> float:
        freqs = []
        for n in note_list:
            try:
                p = int(n.get('pitch', 0) if isinstance(n, dict) else getattr(n, 'pitch', 0))
            except Exception:
                continue
            midi = 20 + p
            freqs.append(440.0 * (2.0 ** ((midi - 69) / 12.0)))
        return sum(freqs) / len(freqs) if freqs else 0.0

    @staticmethod
    def _hand_counts(note_list: list) -> tuple[int, int]:
        left = 0
        right = 0
        for n in note_list:
            try:
                hand = str(n.get('hand', '') if isinstance(n, dict) else getattr(n, 'hand', '')).strip().lower()
            except Exception:
                hand = ''
            if hand == 'l':
                left += 1
            elif hand == 'r':
                right += 1
        return (left, right)

    @staticmethod
    def _pitch_values(note_list: list) -> list[int]:
        pitches: list[int] = []
        for n in note_list:
            try:
                p = int(n.get('pitch', 0) if isinstance(n, dict) else getattr(n, 'pitch', 0))
            except Exception:
                continue
            if p > 0:
                pitches.append(p)
        return pitches

    @classmethod
    def _pitch_range(cls, note_list: list) -> tuple[int, int]:
        pitches = cls._pitch_values(note_list)
        if not pitches:
            return (0, 0)
        return (min(pitches), max(pitches))

    @classmethod
    def _most_used_pitch(cls, note_list: list) -> int:
        pitches = cls._pitch_values(note_list)
        if not pitches:
            return 0
        counts = Counter(pitches)
        return int(counts.most_common(1)[0][0])

    @classmethod
    def compute(cls, score: Any, *, lines_count: int | None = None, pages_count: int | None = None) -> "Analysis":
        if isinstance(score, dict):
            events = score.get("events", {}) or {}
            base_grid = score.get("base_grid", []) or []
        else:
            events = getattr(score, "events", None) or {}
            base_grid = getattr(score, "base_grid", []) or []

        note_list = cls._list_from_events(events, "note")
        notes = len(note_list)
        grace_notes = len(cls._list_from_events(events, "grace_note"))
        derived_lines = lines_count if lines_count is not None else len(cls._list_from_events(events, "line_break"))
        derived_pages = pages_count if pages_count is not None else 0
        measures = cls._measure_count(base_grid)
        pitch_range_low, pitch_range_high = cls._pitch_range(note_list)
        left_hand_notes, right_hand_notes = cls._hand_counts(note_list)

        return cls(
            notes=notes,
            grace_notes=grace_notes,
            lines=derived_lines,
            measures=measures,
            pages=derived_pages,
            avg_frequency_hz=cls._avg_frequency(note_list),
            pitch_range_low=pitch_range_low,
            pitch_range_high=pitch_range_high,
            most_used_pitch=cls._most_used_pitch(note_list),
            left_hand_notes=left_hand_notes,
            right_hand_notes=right_hand_notes,
        )
