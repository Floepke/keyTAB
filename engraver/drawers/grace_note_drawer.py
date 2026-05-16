"""GraceNoteDrawer: renders grace notes (small decorative notes)."""

from utils.CONSTANT import SHORTEST_DURATION, PIANO_KEY_AMOUNT
from utils.operator import Operator
from symbol_design.noteheads import Notehead
from engraver.helpers import time_to_y


class GraceNoteDrawer:
    """Draw grace notes."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
        self.layout_data = context.layout_data
        self.notation_color = self.layout_data.get('notation_color')
        self.paper_color = self.layout_data.get('paper_color', (1.0, 1.0, 1.0, 1.0))
        self.op = Operator(SHORTEST_DURATION)
    
    def draw(self) -> None:
        """Draw all grace notes for the current page and line."""
        score = self.context.score or {}
        events = dict(score.get('events', {}) or {})
        grace_notes = list(events.get('grace_note', []) or [])
        if not grace_notes:
            return

        layout = self.layout_data.get('layout', {})
        if not bool(layout.get('grace_note_visible', True)):
            return

        semitone_mm = float(self.layout_data.get('semitone_mm', 2.0) or 2.0)
        scale = float(self.layout_data.get('scale', 1.0) or 1.0)
        g_scale = float(layout.get('grace_note_scale', 0.75) or 0.75)
        g_outline = float(layout.get('grace_note_outline_width_mm', layout.get('grace_note_outline_width', 0.3)) or 0.3) * scale
        key_to_x_map = self.layout_data.get('pitch_to_x_for_line', {})

        grace_norm = []
        for idx, g in enumerate(grace_notes):
            if not isinstance(g, dict):
                continue
            grace_norm.append(
                {
                    'time': float(g.get('time', 0.0) or 0.0),
                    'pitch': int(g.get('pitch', 0) or 0),
                    'id': int(g.get('_id', 0) or 0),
                    'idx': int(idx),
                    'raw': g,
                }
            )

        grace_layout = dict(layout)
        grace_layout['notehead_tilt'] = 0.0

        for line_index, line in enumerate(self.layout_data.get('page_lines', []) or []):
            key_to_x = key_to_x_map.get(line_index) if isinstance(key_to_x_map, dict) else None
            if not callable(key_to_x):
                continue
            lt0 = float(line.get('time_start', 0.0) or 0.0)
            lt1 = float(line.get('time_end', lt0) or lt0)
            for item in grace_norm:
                g_t = float(item.get('time', 0.0) or 0.0)
                p = int(item.get('pitch', 0) or 0)
                if self.op.lt(g_t, lt0) or self.op.ge(g_t, lt1):
                    continue
                if p < 1 or p > PIANO_KEY_AMOUNT:
                    continue
                x = float(key_to_x(p))
                y = float(time_to_y(line, g_t))
                notehead = Notehead.from_note(
                    x_mm=x,
                    y_mm=y,
                    note=item.get('raw', {}) or {},
                    layout=grace_layout,
                    semitone_space_mm=float(semitone_mm * g_scale),
                    notation_color=self.notation_color,
                    paper_color=self.paper_color,
                    default_black_above=False,
                    outline_width_mm_override=float(g_outline),
                )
                tag = 'grace_note_black' if bool(getattr(notehead, 'filled', False)) else 'grace_note_white'
                notehead.draw_notehead(self.du, item_id=int(item.get('id', 0) or 0), tags=[tag])
