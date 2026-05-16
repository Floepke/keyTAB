"""TextDrawer: renders text annotations, tempo markings, and labels."""

from utils.CONSTANT import SHORTEST_DURATION, ENGRAVER_FRACTIONAL_TEXT_SCALING_CORRECTION
from utils.operator import Operator
from engraver.helpers import time_to_y


class TextDrawer:
    """Draw text elements."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
        self.layout_data = context.layout_data
        self.notation_color = self.layout_data.get('notation_color')
        self.paper_color = self.layout_data.get('paper_color', (1.0, 1.0, 1.0, 1.0))
        self.scale = float(self.layout_data.get('scale', 1.0) or 1.0)
        self.op = Operator(SHORTEST_DURATION)
    
    def draw(self) -> None:
        """Draw all text elements for the current page and line."""
        score = self.context.score or {}
        events = dict(score.get('events', {}) or {})
        texts = list(events.get('text', []) or [])
        tempos = list(events.get('tempo', []) or [])
        if not texts and not tempos:
            return

        layout = self.layout_data.get('layout', {})
        semitone_mm = float(self.layout_data.get('semitone_mm', 2.0) or 2.0)
        key_to_x_map = self.layout_data.get('pitch_to_x_for_line', {})

        font_default = dict(layout.get('font_text', {}) or {})
        fam = str(font_default.get('family', 'Edwin') or 'Edwin')
        size = float(font_default.get('size_pt', 12.0) or 12.0) * ENGRAVER_FRACTIONAL_TEXT_SCALING_CORRECTION * (self.scale / 0.3333333333333333)
        italic = bool(font_default.get('italic', False))
        bold = bool(font_default.get('bold', False))
        underline = bool(font_default.get('underline', False))
        pad_mm = float(layout.get('text_background_padding_mm', 0.0) or 0.0) * self.scale

        for line_index, line in enumerate(self.layout_data.get('page_lines', []) or []):
            key_to_x = key_to_x_map.get(line_index) if isinstance(key_to_x_map, dict) else None
            if not callable(key_to_x):
                continue
            base_x_c4 = float(key_to_x(40))
            lt0 = float(line.get('time_start', 0.0) or 0.0)
            lt1 = float(line.get('time_end', lt0) or lt0)

            def _rpitch_to_x(rp: float) -> float:
                return base_x_c4 + (float(rp) * semitone_mm)

            if bool(layout.get('text_visible', True)):
                for tx in texts:
                    t = float(tx.get('time', 0.0) or 0.0)
                    if self.op.lt(t, lt0) or self.op.gt(t, lt1):
                        continue
                    txt = str(tx.get('text', '') or '')
                    if not txt:
                        continue
                    x = _rpitch_to_x(float(tx.get('x_rpitch', 0.0) or 0.0)) + float(tx.get('x_offset_mm', 0.0) or 0.0)
                    y = float(time_to_y(line, t)) + float(tx.get('y_offset_mm', 0.0) or 0.0)
                    angle = float(tx.get('rotation', 0.0) or 0.0)
                    alignment = str(tx.get('alignment', 'left') or 'left').lower()
                    xb, yb, w, h = self.du._get_text_extents_mm(txt, fam, size, italic, bold)
                    item_id = int(tx.get('_id', tx.get('id', 0)) or 0)
                    self.du.add_rectangle(
                        x - pad_mm,
                        y - pad_mm,
                        x + float(w) + pad_mm,
                        y + float(h) + pad_mm,
                        stroke_color=None,
                        fill_color=self.paper_color,
                        id=item_id,
                        tags=['text_bg'],
                    )
                    anchor = 'center' if alignment == 'center' else ('e' if alignment == 'right' else None)
                    self.du.add_text(
                        x,
                        y,
                        txt,
                        family=fam,
                        size_pt=size,
                        italic=italic,
                        bold=bold,
                        color=self.notation_color,
                        anchor=anchor,
                        angle_deg=angle,
                        id=item_id,
                        tags=['text'],
                    )
                    if underline:
                        uy = y + max(0.2, size * 0.025)
                        self.du.add_line(x, uy, x + float(w), uy, color=self.notation_color, width_mm=max(0.2, size * (0.04 if bold else 0.02)), id=item_id, tags=['text_underline'])

            if bool(layout.get('tempo_indicator_visible', True)):
                for tp in tempos:
                    t0 = float(tp.get('time', 0.0) or 0.0)
                    t1 = float(tp.get('end', t0 + float(tp.get('duration', 0.0) or 0.0)) or t0)
                    if self.op.le(t1, t0):
                        continue
                    seg0 = max(lt0, t0)
                    seg1 = min(lt1, t1)
                    if self.op.le(seg1, seg0):
                        continue
                    tempo_val = int(tp.get('tempo', 60) or 60)
                    tempo_txt = str(tempo_val)
                    x_left = float(key_to_x(int(line.get('natural_bound_right', line.get('bound_right', 88))))) + (semitone_mm * 2.0)
                    y0 = time_to_y(line, seg0)
                    y1 = time_to_y(line, seg1)
                    if y1 < y0:
                        y0, y1 = y1, y0
                    x_right = x_left + max(6.0 * self.scale, float(self.du._get_text_extents_mm(tempo_txt, 'Edwin', 32.0 * self.scale, False, True)[2]) + (2.0 * self.scale))
                    item_id = int(tp.get('_id', tp.get('id', 0)) or 0)
                    self.du.add_line(x_left, y0, x_right, y0, color=self.notation_color, width_mm=0.25, dash_pattern=[0.5, 1.0], id=item_id, tags=['tempo_bg'])
                    self.du.add_line(x_right, y0, x_right, y1, color=self.notation_color, width_mm=0.25, dash_pattern=[0.5, 1.0], id=item_id, tags=['tempo_bg'])
                    self.du.add_line(x_left, y1, x_right, y1, color=self.notation_color, width_mm=0.25, dash_pattern=[0.5, 1.0], id=item_id, tags=['tempo_bg'])
                    self.du.add_text(
                        (x_left + x_right) * 0.5,
                        (y0 + y1) * 0.5,
                        tempo_txt,
                        family='Edwin',
                        size_pt=32.0 * self.scale,
                        bold=True,
                        italic=False,
                        color=self.notation_color,
                        anchor='center',
                        id=item_id,
                        tags=['tempo_text'],
                    )
