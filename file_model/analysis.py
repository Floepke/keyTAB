from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Analysis:
    notes: int = 0
    grace_notes: int = 0
    pages: int = 0
    measures: int = 0
    lines: int = 0

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

    @classmethod
    def compute(cls, score: Any, *, lines_count: int | None = None, pages_count: int | None = None) -> "Analysis":
        if isinstance(score, dict):
            events = score.get("events", {}) or {}
            base_grid = score.get("base_grid", []) or []
        else:
            events = getattr(score, "events", None) or {}
            base_grid = getattr(score, "base_grid", []) or []

        notes = len(cls._list_from_events(events, "note"))
        grace_notes = len(cls._list_from_events(events, "grace_note"))
        derived_lines = lines_count if lines_count is not None else len(cls._list_from_events(events, "line_break"))
        derived_pages = pages_count if pages_count is not None else 0
        measures = cls._measure_count(base_grid)

        return cls(
            notes=notes,
            grace_notes=grace_notes,
            lines=derived_lines,
            measures=measures,
            pages=derived_pages,
        )
