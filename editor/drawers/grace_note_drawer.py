from __future__ import annotations
from typing import TYPE_CHECKING, cast
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import BLACK_KEYS, QUARTER_NOTE_UNIT

if TYPE_CHECKING:
    from editor.editor import Editor


class GraceNoteDrawerMixin:
    def draw_grace_note(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        if getattr(self, 'is_tiny_mode', None) and self.is_tiny_mode():
            return
        score = self.current_score()
        if score is None:
            return
        layout = getattr(score, 'layout', None)
        if layout is None or not getattr(layout, 'grace_note_visible', True):
            return

        margin = float(getattr(self, 'margin', 0.0) or 0.0)
        try:
            zpq = float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0) or 25.0)
        except Exception:
            zpq = 1.0

        def time_to_mm(ticks: float) -> float:
            return margin + (float(ticks) / float(QUARTER_NOTE_UNIT)) * zpq

        # Visible window with bleed similar to note drawer
        top_mm = float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
        vp_h_mm = float(getattr(self, '_viewport_h_mm', 0.0) or 0.0)
        bottom_mm = top_mm + vp_h_mm
        bleed_mm = max(2.0, zpq * 0.25)
        time_begin = float(self.mm_to_time(top_mm - bleed_mm))
        time_end = float(self.mm_to_time(bottom_mm + bleed_mm))

        notes = list(getattr(score.events, 'grace_note', []) or [])
        if not notes:
            return

        # Use the same thresholded comparator as note drawer
        from utils.operator import Operator

        op = Operator(7)
        semitone_dist = float(getattr(self, 'semitone_dist', 0.5) or 0.5)
        notation_color = getattr(self, 'notation_color', (0, 0, 0, 1))
        scale = float(getattr(layout, 'grace_note_scale', 0.75) or 0.75)
        outline_w = float(
            getattr(layout, 'grace_note_outline_width_mm', getattr(layout, 'grace_note_outline_width', 0.3))
            or 0.3
        )

        for g in notes:
            t = float(getattr(g, 'time', 0.0) or 0.0)
            if op.gt(t, time_end) or op.lt(t, time_begin):
                continue
            pitch = int(getattr(g, 'pitch', 40) or 40)
            x = float(self.pitch_to_x(pitch))
            y_top = float(time_to_mm(t))

            base_w = semitone_dist * scale
            top = y_top
            bottom = y_top + (base_w * 2.0)
            left = x - base_w
            right = x + base_w

            y_center = (top + bottom) * 0.5

            if pitch in BLACK_KEYS:
                du.add_oval(
                    left,
                    top,
                    right,
                    bottom,
                    stroke_color=notation_color,
                    stroke_width_mm=0.0,
                    fill_color=notation_color,
                    id=getattr(g, '_id', 0),
                    tags=["grace_note_black"],
                )
            else:
                # Draw outline inward: paint outer with notation color, then inner white.
                du.add_oval(
                    left,
                    top,
                    right,
                    bottom,
                    stroke_color=None,
                    fill_color=notation_color,
                    id=getattr(g, '_id', 0),
                    tags=["grace_note_white_outline"],
                )
                # Halve the inset so the visible outline matches the configured width.
                inset = outline_w * 0.25
                inner_left = left + inset
                inner_right = right - inset
                inner_top = top + inset
                inner_bottom = bottom - inset
                if inner_right <= inner_left:
                    mid_x = (left + right) * 0.5
                    inner_left = inner_right = mid_x
                if inner_bottom <= inner_top:
                    mid_y = (top + bottom) * 0.5
                    inner_top = inner_bottom = mid_y
                du.add_oval(
                    inner_left,
                    inner_top,
                    inner_right,
                    inner_bottom,
                    stroke_color=None,
                    fill_color=(1.0, 1.0, 1.0, 1.0),
                    id=getattr(g, '_id', 0),
                    tags=["grace_note_white_fill"],
                )

            # Hit rectangle uses unscaled notehead size for predictable picking
            hit_w = semitone_dist
            self.register_note_hit_rect(
                int(getattr(g, '_id', 0) or 0),
                float(x - hit_w),
                float(y_center - hit_w),
                float(x + hit_w),
                float(y_center + hit_w),
            )
