from __future__ import annotations
from typing import TYPE_CHECKING, cast
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import QUARTER_NOTE_UNIT
from symbol_design.noteheads import Notehead
from ui.style import Style

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
        score_events = self.current_events(score)
        layout = getattr(score, 'layout', None)
        if layout is None:
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

        notes = list(getattr(score_events, 'grace_note', []) or [])
        if not notes:
            return

        # Use the same thresholded comparator as note drawer
        from utils.operator import Operator

        op = Operator(7)
        semitone_dist = float(getattr(self, 'semitone_dist', 0.5) or 0.5)
        notation_color = getattr(self, 'notation_color', (0, 0, 0, 1))
        grace_scale = float(getattr(layout, 'grace_note_scale', 0.75) or 0.75)
        semitone_scaled = semitone_dist * max(0.05, grace_scale)
        style_scale = float(getattr(layout, 'scale', 1.0) or 1.0)
        outline_w = float(getattr(layout, 'note_stem_thickness_mm', 0.5) or 0.5) * style_scale * 2
        paper_r, paper_g, paper_b = Style.get_named_rgb('paper', (255, 255, 255))
        paper_color = (paper_r / 255.0, paper_g / 255.0, paper_b / 255.0, 1.0)
        grace_layout = self._grace_layout_no_tilt(layout)

        for g in notes:
            t = float(getattr(g, 'time', 0.0) or 0.0)
            if op.gt(t, time_end) or op.lt(t, time_begin):
                continue
            pitch = int(getattr(g, 'pitch', 40) or 40)
            x = float(self.pitch_to_x(pitch))
            y_top = float(time_to_mm(t))
            notehead = Notehead.from_note(
                x_mm=float(x),
                y_mm=float(y_top),
                note=g,
                layout=grace_layout,
                semitone_space_mm=float(semitone_scaled),
                notation_color=notation_color,
                paper_color=paper_color,
                default_black_above=False,
                outline_width_mm_override=float(outline_w),
            )
            tag = "grace_note_black" if bool(getattr(notehead, 'filled', False)) else "grace_note_white"
            notehead.draw_notehead(du, item_id=int(getattr(g, '_id', 0) or 0), tags=[tag], use_custom_color=False)

            # Hit rectangle uses notehead size scaled by notehead_height_scaling for predictable picking
            hit_w = semitone_dist
            notehead_height_scaling = float(getattr(layout, 'notehead_height_scaling', 1.2) or 1.2)
            hit_h = semitone_dist * notehead_height_scaling
            y_center = float(y_top + semitone_scaled)
            self.register_hit_rect(
                'note', int(getattr(g, '_id', 0) or 0),
                float(x - hit_w), float(y_center - hit_h),
                float(x + hit_w), float(y_center + hit_h),
            )

    def _grace_layout_no_tilt(self, layout):
        """Return layout copy with notehead_tilt set to 0 for grace notes visual contrast."""
        import copy
        layout_copy = copy.copy(layout)
        layout_copy.notehead_tilt = 0.0
        return layout_copy
