"""DynamicDrawer: renders dynamic symbols, hairpins (crescendo/decrescendo)."""

import math

from utils.CONSTANT import SHORTEST_DURATION
from utils.operator import Operator
from engraver.helpers import time_to_y


class DynamicDrawer:
    """Draw dynamic symbols and hairpins."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
        self.layout_data = context.layout_data
        self.notation_color = self.layout_data.get('notation_color')
        self.paper_color = self.layout_data.get('paper_color', (1.0, 1.0, 1.0, 1.0))
        self.scale = float(self.layout_data.get('scale', 1.0) or 1.0)
        self.op = Operator(SHORTEST_DURATION)
    
    def draw(self) -> None:
        """Draw all dynamics and hairpins for the current page and line."""
        score = self.context.score or {}
        events = dict(score.get('events', {}) or {})
        cresc = list(events.get('crescendo', []) or [])
        decresc = list(events.get('decrescendo', []) or [])
        dyn_symbols = list(events.get('dynamic_symbol', []) or [])
        if not cresc and not decresc and not dyn_symbols:
            return

        layout = self.layout_data.get('layout', {})
        semitone_mm = float(self.layout_data.get('semitone_mm', 2.0) or 2.0)
        key_to_x_map = self.layout_data.get('pitch_to_x_for_line', {})

        hairpin_w = float(layout.get('hairpin_line_width_mm', 0.5) or 0.5) * self.scale
        hairpin_spread = float(layout.get('hairpin_width_mm', 5.0) or 5.0) * self.scale
        symbol_size = float(layout.get('dynamic_symbol_font_size_pt', 12.0) or 12.0) * self.scale
        symbol_pad = float(layout.get('dynamic_symbol_background_padding_mm', 0.5) or 0.5) * self.scale
        rot_default = float(layout.get('dynamic_rotation', 0.0) or 0.0)

        for line_index, line in enumerate(self.layout_data.get('page_lines', []) or []):
            key_to_x = key_to_x_map.get(line_index) if isinstance(key_to_x_map, dict) else None
            if not callable(key_to_x):
                continue
            base_x_c4 = float(key_to_x(40))
            lt0 = float(line.get('time_start', 0.0) or 0.0)
            lt1 = float(line.get('time_end', lt0) or lt0)

            def _rpitch_to_x(rp: float) -> float:
                return base_x_c4 + (float(rp) * semitone_mm)

            def _draw_hairpin(ev: dict, crescendo: bool):
                t0 = float(ev.get('time', 0.0) or 0.0)
                t1 = float(ev.get('end', t0 + float(ev.get('duration', 0.0) or 0.0)) or t0)
                if t1 <= t0:
                    return
                seg0 = max(lt0, t0)
                seg1 = min(lt1, t1)
                if seg1 <= seg0:
                    return
                dur = max(1e-6, t1 - t0)
                p0 = max(0.0, min(1.0, (seg0 - t0) / dur))
                p1 = max(0.0, min(1.0, (seg1 - t0) / dur))
                x = _rpitch_to_x(float(ev.get('x_rpitch', 0.0) or 0.0))
                y0 = time_to_y(line, seg0)
                y1 = time_to_y(line, seg1)
                half = hairpin_spread * 0.5
                if crescendo:
                    h0 = half * p0
                    h1 = half * p1
                else:
                    h0 = half * (1.0 - p0)
                    h1 = half * (1.0 - p1)
                tag_id = int(ev.get('_id', ev.get('id', 0)) or 0)
                self.du.add_line(x - h0, y0, x - h1, y1, color=self.notation_color, width_mm=hairpin_w, id=tag_id, tags=['hairpin'])
                self.du.add_line(x + h0, y0, x + h1, y1, color=self.notation_color, width_mm=hairpin_w, id=tag_id, tags=['hairpin'])

            if bool(layout.get('hairpin_visible', True)):
                for ev in cresc:
                    _draw_hairpin(ev, True)
                for ev in decresc:
                    _draw_hairpin(ev, False)

            if bool(layout.get('dynamic_symbol_visible', True)):
                for ds in dyn_symbols:
                    t = float(ds.get('time', 0.0) or 0.0)
                    if self.op.lt(t, lt0) or self.op.gt(t, lt1):
                        continue
                    glyph = str(ds.get('symbol', '') or '')
                    if not glyph:
                        continue
                    x = _rpitch_to_x(float(ds.get('x_rpitch', 0.0) or 0.0))
                    y = float(time_to_y(line, t))
                    raw_rot = ds.get('rotation', None)
                    angle = float(rot_default if raw_rot is None else raw_rot)
                    xb, yb, w, h = self.du._get_text_extents_mm(glyph, 'LelandText', symbol_size, False, False)
                    cx = x
                    cy = y
                    hw = max(0.5, float(w) * 0.5) + symbol_pad
                    hh = max(0.5, float(h) * 0.5) + symbol_pad
                    ang = math.radians(angle)
                    sin_a = math.sin(ang)
                    cos_a = math.cos(ang)
                    corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
                    poly = [
                        (
                            cx + (lx * cos_a) - (ly * sin_a),
                            cy + (lx * sin_a) + (ly * cos_a),
                        )
                        for (lx, ly) in corners
                    ]
                    item_id = int(ds.get('_id', ds.get('id', 0)) or 0)
                    self.du.add_polygon(poly, stroke_color=None, fill_color=self.paper_color, id=item_id, tags=['dynamic_symbol_bg'])
                    self.du.add_text(
                        cx - (float(xb) + float(w) * 0.5),
                        cy - (float(yb) + float(h) * 0.5),
                        glyph,
                        family='LelandText',
                        size_pt=symbol_size,
                        italic=False,
                        bold=False,
                        color=self.notation_color,
                        angle_deg=angle,
                        id=item_id,
                        tags=['dynamic_symbol_text'],
                    )
