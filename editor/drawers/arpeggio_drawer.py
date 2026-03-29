from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import SHORTEST_DURATION
from utils.operator import Operator

if TYPE_CHECKING:
    from editor.editor import Editor
    from file_model.events.arpeggio import Arpeggio


class ArpeggioDrawerMixin:
    _arp_time_op: Operator = Operator(float(SHORTEST_DURATION))

    def _resolve_arpeggio_notes(self, arp: "Arpeggio", notes_all: list, op: Operator) -> list:
        # Match notes at the arpeggio time whose pitches are listed in arp.notes
        base_time = float(getattr(arp, "time", 0.0) or 0.0)
        target_pitches = list(getattr(arp, "notes", []) or [])
        if len(target_pitches) < 2:
            return []
        matches = [n for n in notes_all if op.eq(float(getattr(n, "time", 0.0) or 0.0), base_time)]
        remaining = list(int(p) for p in target_pitches)
        resolved = []
        for n in sorted(matches, key=lambda m: int(getattr(m, "pitch", 0) or 0)):
            p = int(getattr(n, "pitch", 0) or 0)
            if p in remaining:
                resolved.append(n)
                remaining.remove(p)
        return resolved

    def _arpeggio_times(self, base_time: float, duration: float, kind: str) -> tuple[float, float]:
        end_type = str(kind or "").endswith("ending")
        if end_type:
            start_t = max(0.0, float(base_time) - max(0.0, duration))
            return (start_t, float(base_time))
        return (float(base_time), float(base_time) + max(0.0, duration))

    def draw_arpeggio(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        score = self.current_score()
        if score is None:
            return

        arps = list(getattr(score.events, "arpeggio", []) or [])
        if not arps:
            return

        cache = getattr(self, "_draw_cache", {}) or {}
        notes_sorted = cache.get("notes_sorted") or list(getattr(score.events, "note", []) or [])

        semi = float(self.semitone_dist or 0.5)
        op = self._arp_time_op

        for arp in arps:
            try:
                base_time = float(getattr(arp, "time", 0.0) or 0.0)
                dur = max(0.0, float(getattr(arp, "duration", 32.0) or 32.0))
                kind = str(getattr(arp, "type", "up/starting") or "up/starting")
                arp_id = int(getattr(arp, "_id", 0) or 0)
            except Exception:
                continue

            chord_notes = self._resolve_arpeggio_notes(arp, notes_sorted, op)
            if len(chord_notes) < 2:
                continue

            start_t, end_t = self._arpeggio_times(base_time, dur, kind)
            up = kind.startswith("up")

            # Sort by pitch to define ordering along the diagonal stem
            chord_sorted = sorted(chord_notes, key=lambda n: int(getattr(n, "pitch", 0) or 0), reverse=not up)
            steps = max(1, len(chord_sorted) - 1)
            span = float(max(0.0, end_t - start_t))
            if op.eq(span, 0.0):
                span = float(max(1.0, dur))
                end_t = start_t + span

            chord_xs = [float(self.pitch_to_x(int(getattr(n, "pitch", 0) or 0))) for n in chord_sorted]
            if not chord_xs:
                continue
            x0 = min(chord_xs) - semi * 0.35
            x1 = max(chord_xs) + semi * 0.35
            y0 = float(self.time_to_mm(start_t))
            y1 = float(self.time_to_mm(end_t))

            def _lerp(a: float, b: float, r: float) -> float:
                return float(a + (b - a) * r)

            # Points along the diagonal stem and corresponding notehead anchors
            note_positions: list[tuple[float, float, object]] = []
            for idx, note_obj in enumerate(chord_sorted):
                x_note = float(self.pitch_to_x(int(getattr(note_obj, "pitch", 0) or 0)))
                if op.eq(float(x1 - x0), 0.0):
                    ratio = 0.0
                else:
                    ratio = max(0.0, min(1.0, (x_note - x0) / float(x1 - x0)))
                y_note = _lerp(y0, y1, ratio)
                note_positions.append((x_note, y_note, note_obj))

            start_pt = (x0, y0)
            end_pt = (x1, y1)
            du.add_polyline(
                [start_pt, end_pt],
                stroke_color=self.accent_color,
                stroke_width_mm=0.55,
                id=arp_id,
                tags=["arpeggio"],
                dash_pattern=None,
            )

            # Render full note visuals along the diagonal stem using the same helpers as note drawer
            for x_note, y_note, note_obj in note_positions:
                # Align head center on the diagonal, then let notehead helper adjust for black-note rules
                w = float(self.semitone_dist or 0.5)
                y_head_top = float(y_note - w)
                # Midi body spans the arpeggio duration
                self._draw_midinote(du, note_obj, x_note, y0, y1, draw_mode="note")
                self._draw_notehead(du, note_obj, x_note, y_head_top, draw_mode="note")
                self._draw_left_dot(du, note_obj, x_note, y_head_top, draw_mode="note")

            # Accent handle at the terminal end of the diagonal (used for resizing)
            handle_x, handle_y = end_pt
            r = semi * 0.8
            du.add_oval(
                handle_x - r,
                handle_y - r,
                handle_x + r,
                handle_y + r,
                stroke_color=None,
                fill_color=self.accent_color,
                id=arp_id,
                tags=["arpeggio_handle"],
            )
            try:
                self.register_hit_rect(
                    'arpeggio', arp_id,
                    handle_x - r * 1.2, handle_y - r * 1.2,
                    handle_x + r * 1.2, handle_y + r * 1.2,
                )
            except Exception:
                pass

            # Direction marker arrow at the start of the stroke
            arrow_w = semi * 0.8
            arrow_h = semi * 1.0
            if up:
                tri = [
                    (start_pt[0] - arrow_w * 0.5, start_pt[1] + arrow_h * 0.4),
                    (start_pt[0] + arrow_w * 0.5, start_pt[1] + arrow_h * 0.4),
                    (start_pt[0], start_pt[1] - arrow_h * 0.6),
                ]
            else:
                tri = [
                    (start_pt[0] - arrow_w * 0.5, start_pt[1] - arrow_h * 0.4),
                    (start_pt[0] + arrow_w * 0.5, start_pt[1] - arrow_h * 0.4),
                    (start_pt[0], start_pt[1] + arrow_h * 0.6),
                ]
            du.add_polygon(tri, stroke_color=None, fill_color=self.accent_color, id=arp_id, tags=["arpeggio_arrow"])
