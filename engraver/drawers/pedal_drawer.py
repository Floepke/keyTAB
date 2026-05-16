"""PedalDrawer: renders pedal symbols."""

from utils.CONSTANT import SHORTEST_DURATION
from utils.operator import Operator
from symbol_design.pedal import draw_pedal_symbol


class PedalDrawer:
    """Draw pedal symbols."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
        self.layout_data = context.layout_data
        self.op = Operator(SHORTEST_DURATION)
    
    def draw(self) -> None:
        """Draw all pedal symbols for the current page and line."""
        score = self.context.score or {}
        events = dict(score.get('events', {}) or {})
        pedals = list(events.get('pedal', []) or [])
        if not pedals:
            return

        layout = self.layout_data.get('layout', {})
        semitone_mm = float(self.layout_data.get('semitone_mm', 2.0) or 2.0)
        scale = float(self.layout_data.get('scale', 1.0) or 1.0)
        notation_color = self.layout_data.get('notation_color')
        paper_color = self.layout_data.get('paper_color', (1.0, 1.0, 1.0, 1.0))
        pedal_thickness_mm = float(layout.get('pedal_symbol_thickness_mm', 0.3) or 0.3) * scale

        key_to_x_map = self.layout_data.get('pitch_to_x_for_line', {})
        for line_index, line in enumerate(self.layout_data.get('page_lines', []) or []):
            key_to_x = key_to_x_map.get(line_index) if isinstance(key_to_x_map, dict) else None
            if not callable(key_to_x):
                continue
            lt0 = float(line.get('time_start', 0.0) or 0.0)
            lt1 = float(line.get('time_end', lt0) or lt0)

            def _time_to_y_local(t: float) -> float:
                denom = max(1e-6, float(lt1 - lt0))
                rel = max(0.0, min(1.0, (float(t) - float(lt0)) / denom))
                y0 = float(line.get('y_top', 0.0) or 0.0)
                y1 = float(line.get('y_bottom', y0) or y0)
                return y0 + ((y1 - y0) * rel)

            def _rpitch_to_x_local(rpitch_val: int) -> float:
                base_x_c4 = float(key_to_x(40))
                return base_x_c4 + (float(rpitch_val) * semitone_mm)

            for pedal_ev in pedals:
                p_t = float(pedal_ev.get('time', 0.0) or 0.0)
                p_symbol = str(pedal_ev.get('symbol', '') or '')
                is_up_symbol = p_symbol in ('up_keytab', 'up_klavarskribo')
                if is_up_symbol:
                    if self.op.le(p_t, lt0) or self.op.gt(p_t, lt1):
                        continue
                else:
                    if self.op.lt(p_t, lt0) or self.op.ge(p_t, lt1):
                        continue
                if bool(pedal_ev.get('invisible', False)):
                    continue
                try:
                    draw_pedal_symbol(
                        self.du,
                        pedal_ev,
                        time_to_y_mm=_time_to_y_local,
                        rpitch_to_x_mm=_rpitch_to_x_local,
                        color=notation_color,
                        background_color=paper_color,
                        width_mm=pedal_thickness_mm,
                        semitone_space_mm=semitone_mm,
                        layout=layout,
                        id=int(pedal_ev.get('_id', pedal_ev.get('id', 0)) or 0),
                        tags=['pedal_symbol'],
                    )
                except Exception:
                    continue
