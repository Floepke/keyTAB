from __future__ import annotations
from typing import TYPE_CHECKING, cast
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import QUARTER_NOTE_UNIT

if TYPE_CHECKING:
    from editor.editor import Editor


class TempoDrawerMixin:
    def draw_tempo(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        tool_name = getattr(getattr(self, "_tool", None), "TOOL_NAME", "")
        if tool_name != "tempo":
            return
        score = self.current_score()
        if score is None:
            return

        # Layout anchors
        margin = float(self.margin or 0.0)
        # Draw on the outer right side of the editor page
        page_w_mm, _ = du.current_page_size_mm()

        # Iterate tempo events
        events = list(getattr(score.events, 'tempo', []) or [])
        if not events:
            return

        # Font setup: reuse C059 if available
        try:
            from fonts import register_font_from_bytes
            font_family = register_font_from_bytes('C059') or 'C059'
        except Exception:
            font_family = 'C059'

        font_size_pt = 24.0
        font_italic = False
        font_bold = True
        text_padding_mm = 1.0
        rect_width_mm = 12.0  # fixed width to avoid wobble based on glyph height

        for tp in events:
            try:
                t0 = float(getattr(tp, 'time', 0.0) or 0.0)
                du_ticks = float(getattr(tp, 'duration', 0.0) or 0.0)
                tempo_val = int(getattr(tp, 'tempo', 60) or 60)
            except Exception:
                continue
            if du_ticks <= 0.0:
                continue
            # Positions in mm
            y0 = float(self.time_to_mm(t0))
            y1 = float(self.time_to_mm(t0 + du_ticks))
            y3 = float(y0 + 50.0)
            if y1 < y0:
                y0, y1 = y1, y0
            text = str(tempo_val)
            angle_deg = 90.0  # rotate clockwise

            # Fixed-width lane; center text inside
            rect_w = rect_width_mm
            x_left = float(page_w_mm) - rect_w - margin * 0.35
            x_center = x_left + rect_w * 0.5
            y_center = (y0 + y1) * 0.5

            # Measure rotated text using shared text helpers
            try:
                w_mm, h_mm, _offset_down, rot_corners, rot_poly = self._text_bbox(
                    du,
                    text,
                    font_family,
                    font_size_pt,
                    font_italic,
                    font_bold,
                    angle_deg,
                    text_padding_mm,
                )
            except Exception:
                # Fallback to simple extents if helper fails
                _xb, _yb, w_mm, h_mm = du._get_text_extents_mm(text, font_family, font_size_pt, font_italic, font_bold)
                rot_poly = [(-w_mm * 0.5, -h_mm * 0.5), (w_mm * 0.5, -h_mm * 0.5), (w_mm * 0.5, h_mm * 0.5), (-w_mm * 0.5, h_mm * 0.5)]

            # Rotated polygon bounds for grey underlay height
            poly_abs = [(x_center + dx, y_center + dy) for (dx, dy) in rot_poly]
            min_y = min(p[1] for p in poly_abs)
            max_y = max(p[1] for p in poly_abs)
            text_height_rot = max(0.0, max_y - min_y)

            # Grey underlay sized to text height, fixed width lane
            du.add_rectangle(
                x_left,
                y_center - text_height_rot * 0.5,
                x_left + rect_w,
                y_center + text_height_rot * 0.5,
                stroke_color=None,
                fill_color=(0.7, 0.7, 0.7, 1),
                id=0,
                tags=["tempo_under"],
                dash_pattern=None,
            )

            # Black duration bar sized by tempo duration
            du.add_rectangle(
                x_left,
                y0,
                x_left + rect_w,
                y1,
                stroke_color=None,
                fill_color=(0, 0, 0, 1),
                id=0,
                tags=["tempo_bg"],
                dash_pattern=None,
            )
            try:
                self.register_tempo_hit_rect(int(getattr(tp, '_id', 0) or 0), x_left, min(y0, y1), x_left + rect_w, max(y0, y1))
            except Exception:
                pass
            # add guide start line (black)
            du.add_line(page_w_mm - margin - self.semitone_dist * 2, y0, x_left, y0,
                        color=(0, 0, 0, 1), width_mm=0.25, id=0,
                        tags=["tempo_guide_line"], dash_pattern=[0,1])
            # add guide line end (black)
            du.add_line(page_w_mm - margin - self.semitone_dist * 2, y1, x_left, y1,
                        color=(0, 0, 0, 1), width_mm=0.25, id=0,
                        tags=["tempo_guide_line"], dash_pattern=[0,1])
            # Rotated white text centered inside the black bar
            du.add_text(
                x_center,
                y_center,
                text,
                family=font_family,
                size_pt=font_size_pt,
                italic=font_italic,
                bold=font_bold,
                color=(1, 1, 1, 1),
                anchor='center',
                id=0,
                tags=["tempo_text"],
                hit_rect_mm=None,
                angle_deg=angle_deg,
            )
