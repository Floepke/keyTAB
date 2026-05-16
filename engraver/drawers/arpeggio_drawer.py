"""ArpeggioDrawer: renders arpeggio stems."""

from utils.CONSTANT import SHORTEST_DURATION
from utils.operator import Operator
from engraver.helpers import time_to_y


class ArpeggioDrawer:
    """Draw arpeggio stems."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
        self.layout_data = context.layout_data
        self.notation_color = self.layout_data.get('notation_color')
        self.scale = float(self.layout_data.get('scale', 1.0) or 1.0)
        self.op = Operator(SHORTEST_DURATION)
    
    def draw(self) -> None:
        """Draw all arpeggios for the current page and line."""
        score = self.context.score or {}
        events = dict(score.get('events', {}) or {})
        arpeggios = list(events.get('arpeggio', []) or [])
        notes = list(events.get('note', []) or [])
        if not arpeggios or not notes:
            return
        layout = self.layout_data.get('layout', {})
        if not bool(layout.get('chord_connect_visible', True)):
            return

        semitone_mm = float(self.layout_data.get('semitone_mm', 2.0) or 2.0)
        stem_w = float(layout.get('note_stem_thickness_mm', 0.5) or 0.5) * self.scale
        stem_len_mm = float(layout.get('note_stem_length_semitone', 3) or 3) * semitone_mm
        key_to_x_map = self.layout_data.get('pitch_to_x_for_line', {})

        note_by_time_pitch = {}
        for idx, n in enumerate(notes):
            if not isinstance(n, dict):
                continue
            t = float(n.get('time', 0.0) or 0.0)
            p = int(n.get('pitch', 0) or 0)
            note_by_time_pitch[(int(round(t)), p)] = {
                'idx': idx,
                'time': t,
                'pitch': p,
                'hand': str(n.get('hand', 'l') or 'l'),
            }

        for line_index, line in enumerate(self.layout_data.get('page_lines', []) or []):
            key_to_x = key_to_x_map.get(line_index) if isinstance(key_to_x_map, dict) else None
            if not callable(key_to_x):
                continue
            lt0 = float(line.get('time_start', 0.0) or 0.0)
            lt1 = float(line.get('time_end', lt0) or lt0)

            for arp in arpeggios:
                base_t = float(arp.get('time', 0.0) or 0.0)
                if self.op.lt(base_t, lt0) or self.op.gt(base_t, lt1):
                    continue
                pitches = tuple(sorted({int(p) for p in (arp.get('note_pitches', []) or []) if int(p) > 0}))
                if len(pitches) < 2:
                    continue
                chord_notes = [note_by_time_pitch.get((int(round(base_t)), p)) for p in pitches]
                chord_notes = [n for n in chord_notes if isinstance(n, dict)]
                if len(chord_notes) < 2:
                    continue
                chord_sorted = sorted(chord_notes, key=lambda n: int(n.get('pitch', 0) or 0))
                hand = str(chord_sorted[0].get('hand', 'l') or 'l')
                t1 = float(base_t + float(arp.get('rtime1', 0.0) or 0.0))
                t2 = float(base_t + float(arp.get('rtime2', 0.0) or 0.0))
                y1 = float(time_to_y(line, t1))
                y2 = float(time_to_y(line, t2))

                if hand == 'l':
                    tip_pitch = int(chord_sorted[0].get('pitch', 0) or 0)
                    x = float(key_to_x(tip_pitch)) - stem_len_mm
                else:
                    tip_pitch = int(chord_sorted[-1].get('pitch', 0) or 0)
                    x = float(key_to_x(tip_pitch)) + stem_len_mm

                self.du.add_line(
                    x,
                    y1,
                    x,
                    y2,
                    color=self.notation_color,
                    width_mm=max(0.15, stem_w),
                    id=int(arp.get('_id', arp.get('id', 0)) or 0),
                    tags=['chord_connect'],
                )
