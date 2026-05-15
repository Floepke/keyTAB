"""MiniPianoDrawer: renders mini piano keyboard visualization at system end."""
from engraver.helpers import normalize_hex_color
from utils.CONSTANT import hex_to_rgba, PIANO_KEY_AMOUNT, BLACK_KEYS
from ui.widgets.draw_util import DrawUtil


class MiniPianoDrawer:
    """Draw mini piano keyboard."""
    
    def __init__(self, context):
        self.context = context
        self.du: DrawUtil = context.du
    
    def draw(self) -> None:
        """Draw mini piano for the current page and line."""
        layout_data = self.context.layout_data
        page_lines = layout_data.get('page_lines', [])
        key_to_x_map = layout_data.get('key_to_x_for_line', {})
        semitone_mm = float(layout_data.get('semitone_mm', 1.0) or 1.0)
        scale = layout_data.get('scale', 1.0)
        notation_color = layout_data.get('notation_color')
        layout = layout_data.get('layout', {})

        for line_index, line in enumerate(page_lines):
            if not bool(line.get('mini_piano_visible', False)):
                continue

            natural_bound_left = int(line.get('natural_bound_left', 1))
            natural_bound_right = int(line.get('natural_bound_right', 88))
            y2_draw = float(line.get('y_bottom', 0.0))
            kb_y1 = float(line.get('mini_piano_y_top', y2_draw) or y2_draw)
            kb_y2 = float(line.get('mini_piano_y_bottom', kb_y1) or kb_y1)
            if kb_y2 <= kb_y1:
                continue

            key_to_x = key_to_x_map.get(line_index)
            if not callable(key_to_x):
                continue
            kb_x1 = float(key_to_x(int(natural_bound_left))) - float(semitone_mm)
            kb_x2 = float(key_to_x(int(natural_bound_right))) + float(semitone_mm)
            if kb_x2 <= kb_x1:
                continue

            bar_width_mm = max(0.01, float(1.125 * scale))
            key_len_mm = float(semitone_mm) * 4.0
            black_key_width_mm = float(semitone_mm)
            black_key_set = set(BLACK_KEYS)
            read_direction = str(layout.get('read_direction', 'vertical') or 'vertical').strip().lower()

            def _octave_number(key: int) -> int:
                midi_note = 20 + int(key)
                return (midi_note // 12) - 1

            grey_octave_spans = [
                (4, 15),
                (28, 39),
                (52, 63),
                (76, 87),
            ]
            mini_piano_color = str(layout.get('mini_piano_color', '#ccc') or '#ccc')
            mr, mg, mb, _ = hex_to_rgba(normalize_hex_color(mini_piano_color) or '#ccc', 1.0)
            grey_fill = (float(mr) / 255.0, float(mg) / 255.0, float(mb) / 255.0, 1.0)

            kb_fill_x1 = float(kb_x1 - semitone_mm)
            kb_fill_x2 = float(kb_x2 + semitone_mm)
            visible_spans: list[tuple[float, float]] = []
            for span_start, span_end in grey_octave_spans:
                raw_x1 = float(key_to_x(int(span_start))) - semitone_mm
                raw_x2 = float(key_to_x(int(span_end))) + semitone_mm
                if min(raw_x2, kb_fill_x2) > max(raw_x1, kb_fill_x1):
                    visible_spans.append((raw_x1, raw_x2))

            if visible_spans:
                for idx, (raw_x1, raw_x2) in enumerate(visible_spans):
                    gx1 = float(raw_x1)
                    gx2 = float(raw_x2)
                    if idx + 1 == 0:
                        gx1 = kb_fill_x1
                    if idx == len(visible_spans):
                        gx2 = kb_fill_x2
                    gx1 = max(kb_fill_x1, gx1)
                    gx2 = min(kb_fill_x2, gx2)
                    if gx2 > gx1:
                        self.du.add_rectangle(
                            gx1,
                            kb_y1,
                            gx2,
                            kb_y2,
                            stroke_color=None,
                            fill_color=grey_fill,
                            corner_radius=1.0,
                            id=0,
                            tags=['piano_octave_band'],
                        )

            for key in range(max(1, int(natural_bound_left)), min(PIANO_KEY_AMOUNT, int(natural_bound_right)) + 1):
                if key not in black_key_set:
                    continue
                x_pos = float(key_to_x(key))
                if not (kb_x1 <= x_pos <= kb_x2):
                    continue
                dash = [0.5, 1.0] if key in (41, 43) else None
                self.du.add_line(
                    x_pos,
                    kb_y1,
                    x_pos,
                    kb_y1 + key_len_mm,
                    color=notation_color,
                    dash_pattern=dash,
                    dash_offset_mm=0.4,
                    width_mm=black_key_width_mm,
                    id=0,
                    tags=['piano_black_key'],
                )

            if bool(layout.get('mini_piano_octave_numbering', True)):
                octave_label_keys = [key for key in range(4, PIANO_KEY_AMOUNT + 1, 12)]
                for key in octave_label_keys:
                    x_pos = float(key_to_x(key))
                    if not (kb_x1 <= x_pos <= kb_x2):
                        continue
                    self.du.add_text(
                        x_pos if read_direction == 'vertical' else x_pos + semitone_mm,
                        kb_y1 + semitone_mm * 5.5 if read_direction == 'vertical' else kb_y1 + key_len_mm + semitone_mm * 1.75,
                        str(_octave_number(key)),
                        family='Edwin',
                        color=notation_color,
                        anchor='center',
                        size_pt=16.0 * scale,
                        angle_deg=90.0 if read_direction == 'horizontal' else 0.0,
                        id=0,
                        tags=['piano_octave_number'],
                    )

            self.du.add_rectangle(
                kb_x1 - semitone_mm,
                kb_y1,
                kb_x2 + semitone_mm,
                kb_y2,
                stroke_color=notation_color,
                stroke_width_mm=bar_width_mm,
                corner_radius=0.75 * scale,
                fill_color=None,
                id=0,
                tags=['piano_outline'],
            )
