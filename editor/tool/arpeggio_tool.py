from __future__ import annotations

from PySide6 import QtCore

from editor.tool.base_tool import BaseTool
from file_model.SCORE import SCORE
from file_model.events.arpeggio import Arpeggio
from file_model.events.note import Note
from utils.CONSTANT import SHORTEST_DURATION
from utils.operator import Operator


class ArpeggioTool(BaseTool):
    TOOL_NAME = "arpeggio"

    def __init__(self) -> None:
        super().__init__()
        self._dragging: bool = False
        self._drag_target: Arpeggio | None = None
        self._drag_handle: str = ""

    def toolbar_spec(self) -> list[dict]:
        return []

    def on_toolbar_button(self, name: str) -> None:
        return

    def _request_refresh(self) -> None:
        if self._editor is None:
            return
        if hasattr(self._editor, "force_redraw_from_model"):
            self._editor.force_redraw_from_model()
        else:
            self._editor.draw_frame()

    def _cursor_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        return self._editor.widget_px_to_page_mm(float(x_px), float(y_px))

    def _hit_note(self, score: SCORE, x_px: float, y_px: float) -> Note | None:
        x_mm, y_mm = self._cursor_mm(x_px, y_px)
        hit = self._editor.hit_test_hit_rect(x_mm, y_mm, "note")
        if hit is None:
            return None
        note_id = int(hit.get("_id", -1) or -1)
        found = self._editor.get_note_by_id(note_id)
        return found if isinstance(found, Note) else None

    def _hit_arpeggio(self, x_px: float, y_px: float) -> tuple[Arpeggio | None, str]:
        if self._editor is None:
            return (None, "")
        score = self._editor.current_score()
        if score is None:
            return (None, "")
        events = self._editor.current_events(score)
        if events is None:
            return (None, "")
        x_mm, y_mm = self._cursor_mm(x_px, y_px)
        hit = self._editor.hit_test_hit_rect(x_mm, y_mm, "arpeggio")
        if hit is None:
            return (None, "")
        arp_id = int(hit.get("_id", -1) or -1)
        handle = str(hit.get("handle", "") or "")
        for arp in getattr(events, "arpeggio", []) or []:
            if int(getattr(arp, "_id", -1) or -1) == arp_id:
                return (arp, handle)
        if handle in ("low", "high"):
            raw_pitches = hit.get("note_pitches")
            try:
                pitches = [int(p) for p in (raw_pitches or []) if int(p) > 0]
            except Exception:
                pitches = []
            try:
                base_time = float(hit.get("base_time", 0.0) or 0.0)
            except Exception:
                base_time = 0.0
            target = self._ensure_arpeggio_for_chord(score, pitches, base_time)
            if target is not None:
                return (target, handle)
        return (None, "")

    def _chord_pitches(self, score: SCORE, seed: Note) -> list[int]:
        """Return sorted pitches of all notes in the same chord (time+hand) as seed."""
        base_time = float(getattr(seed, "time", 0.0) or 0.0)
        hand = str(getattr(seed, "hand", "l") or "l")
        op = Operator(float(SHORTEST_DURATION))
        pitches: list[int] = []

        cache = getattr(self._editor, "_draw_cache", None) or {}
        notes_view = list(cache.get("notes_view") or [])
        note_sources = [notes_view]
        if not notes_view:
            events = self._editor.current_events(score)
            note_sources = [list(getattr(events, "note", []) or [])] if events is not None else [[]]
        else:
            events = self._editor.current_events(score)
            note_sources.append(list(getattr(events, "note", []) or []) if events is not None else [])

        for source_notes in note_sources:
            pitches.clear()
            for note in source_notes:
                if str(getattr(note, "hand", "l") or "l") != hand:
                    continue
                if not op.eq(float(getattr(note, "time", 0.0) or 0.0), base_time):
                    continue
                p = int(getattr(note, "pitch", 0) or 0)
                if p > 0:
                    pitches.append(p)
            if len(pitches) >= 2:
                break

        return sorted(set(pitches))

    def _find_matching_arpeggio(self, score: SCORE, pitches: list[int], base_time: float) -> Arpeggio | None:
        key = tuple(sorted(int(p) for p in pitches if int(p) > 0))
        op = Operator(float(SHORTEST_DURATION))
        events = self._editor.current_events(score)
        if events is None:
            return None
        for arp in getattr(events, "arpeggio", []) or []:
            arp_p = tuple(sorted(int(p) for p in (getattr(arp, "note_pitches", []) or []) if int(p) > 0))
            if arp_p != key:
                continue
            if op.eq(float(getattr(arp, "time", 0.0) or 0.0), float(base_time)):
                return arp
        return None

    def _ensure_arpeggio_for_chord(self, score: SCORE, pitches: list[int], base_time: float) -> Arpeggio | None:
        """Create/find arpeggio for a chord only when user edits its handles."""
        clean = sorted(set(int(p) for p in pitches if int(p) > 0))
        if len(clean) < 2:
            return None
        arp = self._find_matching_arpeggio(score, clean, base_time)
        if arp is not None:
            return arp
        return score.new_arpeggio(time=base_time, rtime1=0.0, rtime2=0.0, note_pitches=clean)

    def _ensure_arpeggio_for_note(self, score: SCORE, seed: Note) -> bool:
        pitches = self._chord_pitches(score, seed)
        if len(pitches) < 2:
            return False
        base_time = float(getattr(seed, "time", 0.0) or 0.0)
        arp = self._find_matching_arpeggio(score, pitches, base_time)
        if arp is None:
            default_r2 = float(max(1.0, min(32.0, getattr(self._editor, "snap_size_units", 32.0) or 32.0)))
            score.new_arpeggio(time=base_time, rtime1=0.0, rtime2=default_r2, note_pitches=pitches)
            return True
        changed = False
        if list(getattr(arp, "note_pitches", []) or []) != pitches:
            arp.note_pitches = pitches
            changed = True
        if not Operator(float(SHORTEST_DURATION)).eq(float(getattr(arp, "time", 0.0) or 0.0), base_time):
            arp.time = base_time
            changed = True
        return changed

    def on_left_press(self, x: float, y: float) -> None:
        super().on_left_press(x, y)
        score = self._editor.current_score()
        if score is None:
            return
        target, handle = self._hit_arpeggio(x, y)
        if target is not None and handle in ("low", "high"):
            self._dragging = True
            self._drag_target = target
            self._drag_handle = str(handle)
            self._request_refresh()
            return
        return

    def on_left_drag(self, x: float, y: float, dx: float, dy: float) -> None:
        super().on_left_drag(x, y, dx, dy)
        if not self._dragging or self._drag_target is None:
            return
        base_time = float(getattr(self._drag_target, "time", 0.0) or 0.0)
        current = float(self._editor.snap_time(float(self._editor.widget_px_to_time(x, y))))
        relative = float(current - base_time)
        if self._drag_handle == "low":
            self._drag_target.rtime1 = relative
        elif self._drag_handle == "high":
            self._drag_target.rtime2 = relative
        
        # Validate constraint: cannot both be on the same side (both past or both future)
        rtime1 = float(getattr(self._drag_target, "rtime1", 0.0) or 0.0)
        rtime2 = float(getattr(self._drag_target, "rtime2", 0.0) or 0.0)
        if (rtime1 < 0.0 and rtime2 < 0.0) or (rtime1 > 0.0 and rtime2 > 0.0):
            # Both on same side is invalid; clamp the handle being dragged to 0
            if self._drag_handle == "low":
                self._drag_target.rtime1 = 0.0
            elif self._drag_handle == "high":
                self._drag_target.rtime2 = 0.0
        
        self._request_refresh()

    def on_left_drag_end(self, x: float, y: float) -> None:
        super().on_left_drag_end(x, y)
        if self._dragging and self._drag_target is not None:
            # Check if both handles are zero; if so, delete the arpeggio (revert to normal chord)
            rtime1 = float(getattr(self._drag_target, "rtime1", 0.0) or 0.0)
            rtime2 = float(getattr(self._drag_target, "rtime2", 0.0) or 0.0)
            if rtime1 == 0.0 and rtime2 == 0.0:
                score = self._editor.current_score()
                if score is not None:
                    events = self._editor.current_events(score)
                    if events is not None:
                        arps = list(getattr(events, "arpeggio", []) or [])
                        events.arpeggio = [arp for arp in arps if arp is not self._drag_target]
            
            if hasattr(self._editor, "_snapshot_if_changed"):
                self._editor._snapshot_if_changed(coalesce=True, label="arpeggio_drag")
        self._dragging = False
        self._drag_target = None
        self._drag_handle = ""

    def on_left_unpress(self, x: float, y: float) -> None:
        super().on_left_unpress(x, y)
        self._dragging = False
        self._drag_target = None
        self._drag_handle = ""

    def on_right_click(self, x: float, y: float) -> bool:
        super().on_right_click(x, y)
        score = self._editor.current_score()
        if score is None:
            return False
        target, _handle = self._hit_arpeggio(x, y)
        if target is None:
            hit_note = self._hit_note(score, x, y)
            if hit_note is not None:
                note_pitches = set(self._chord_pitches(score, hit_note))
                events = self._editor.current_events(score)
                for arp in (getattr(events, "arpeggio", []) or []) if events is not None else []:
                    arp_p = set(int(p) for p in (getattr(arp, "note_pitches", []) or []) if int(p) > 0)
                    if note_pitches and arp_p == note_pitches:
                        target = arp
                        break
        if target is None:
            return False
        events = self._editor.current_events(score)
        if events is None:
            return False
        arps = list(getattr(events, "arpeggio", []) or [])
        events.arpeggio = [arp for arp in arps if arp is not target]
        self._editor.update_score_length()
        self._request_refresh()
        return True
