from __future__ import annotations

from typing import TYPE_CHECKING, cast

from editor.editor_defaults import SCALE
from ui.style import Style
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import SHORTEST_DURATION
from utils.operator import Operator

if TYPE_CHECKING:
    from editor.editor import Editor


class ArticulationDrawerMixin:
    def _beamed_articulation_anchors(self, cache: dict, stem_len_mm: float) -> dict[int, tuple[float, float]]:
        self = cast("Editor", self)
        anchors: dict[int, tuple[float, float]] = {}
        op: Operator = cache.get('op') or Operator(float(SHORTEST_DURATION))
        groups_by_hand = dict(cache.get('beam_groups_by_hand') or {})
        windows_by_hand = dict(cache.get('beam_windows_by_hand') or {})
        stem_metrics = dict(cache.get('note_stem_metrics_by_id') or {})
        semitone_mm = float(self.semitone_dist or 0.5)

        for hand in ('l', 'r'):
            direction = -1.0 if hand == 'l' else 1.0
            groups = list(groups_by_hand.get(hand) or [])
            windows = list(windows_by_hand.get(hand) or [])
            for index, group in enumerate(groups):
                group = list(group or [])
                if len(group) < 2:
                    continue
                times = sorted(float(getattr(note, 'time', 0.0) or 0.0) for note in group)
                if not times or op.eq(times[0], times[-1]):
                    continue

                t0, t1 = windows[index] if index < len(windows) else (times[0], times[-1])
                members = [
                    note for note in group
                    if op.ge(float(getattr(note, 'time', 0.0) or 0.0), float(t0))
                    and op.lt(float(getattr(note, 'time', 0.0) or 0.0), float(t1))
                ]
                if len(members) < 2:
                    continue

                first_time = min(float(getattr(note, 'time', 0.0) or 0.0) for note in members)
                last_time = max(float(getattr(note, 'time', 0.0) or 0.0) for note in members)
                if op.eq(first_time, last_time):
                    continue

                edge_note = min(members, key=lambda note: int(getattr(note, 'pitch', 0) or 0)) if hand == 'l' else max(members, key=lambda note: int(getattr(note, 'pitch', 0) or 0))
                edge_id = int(getattr(edge_note, '_id', 0) or 0)
                edge_metric = stem_metrics.get(edge_id) or {}
                edge_x = float(edge_metric.get('x_tip')) if 'x_tip' in edge_metric else float(self.pitch_to_x(int(getattr(edge_note, 'pitch', 0) or 0))) + (direction * stem_len_mm)
                beam_x0 = edge_x
                beam_x1 = edge_x + (direction * semitone_mm)
                beam_y0 = float(self.time_to_mm(first_time))
                beam_y1 = float(self.time_to_mm(last_time))

                for note in members:
                    note_id = int(getattr(note, '_id', 0) or 0)
                    if note_id <= 0:
                        continue
                    metric = stem_metrics.get(note_id) or {}
                    y = float(metric.get('y')) if 'y' in metric else float(self.time_to_mm(float(getattr(note, 'time', 0.0) or 0.0)))
                    ratio = 0.0 if abs(beam_y1 - beam_y0) <= 1e-6 else (y - beam_y0) / (beam_y1 - beam_y0)
                    anchors[note_id] = (float(beam_x0 + (ratio * (beam_x1 - beam_x0))), y)
        return anchors

    def draw_articulation(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        if getattr(self, 'is_tiny_mode', None) and self.is_tiny_mode():
            return

        score = self.current_score()
        cache = getattr(self, '_draw_cache', None) or {}
        if score is None or not cache:
            return

        layout = score.layout
        notes = list(cache.get('notes_view') or [])
        if not notes:
            return

        stem_len_mm = float(getattr(layout, 'note_stem_length_semitone', 3.0) or 3.0) * float(self.semitone_dist or 0.5)
        gap_mm = max(0.0, float(getattr(layout, 'articulation_gap_mm', 1.0))) * SCALE
        dot_diameter_mm = float(getattr(layout, 'articulation_staccato_diameter_mm', 1.6) or 1.6) * SCALE
        dot_radius_mm = max(0.1, dot_diameter_mm * 0.5)
        beam_half_width_mm = max(0.0, float(getattr(layout, 'beam_thickness_mm', 1.0) or 1.0) * SCALE * 0.5)
        articulation_rgb = Style.get_named_rgb('accent_color2', (128, 0, 0))
        articulation_color = (articulation_rgb[0] / 255.0, articulation_rgb[1] / 255.0, articulation_rgb[2] / 255.0, 1.0)
        beam_anchors = self._beamed_articulation_anchors(cache, stem_len_mm)
        stem_metrics = dict(cache.get('note_stem_metrics_by_id') or {})

        for note in notes:
            articulation = getattr(note, 'articulation', None)
            if not bool(getattr(articulation, 'staccato', False)):
                continue

            hand = 'l' if str(getattr(note, 'hand', 'l') or 'l') == 'l' else 'r'
            direction = -1.0 if hand == 'l' else 1.0
            note_id = int(getattr(note, '_id', 0) or 0)
            anchor = beam_anchors.get(note_id)
            if anchor is None:
                metric = stem_metrics.get(note_id) or {}
                x_tip = float(metric.get('x_tip')) if 'x_tip' in metric else float(self.pitch_to_x(int(getattr(note, 'pitch', 0) or 0))) + (direction * stem_len_mm)
                y = float(metric.get('y')) if 'y' in metric else float(self.time_to_mm(float(getattr(note, 'time', 0.0) or 0.0)))
                anchor = (x_tip, y)

            x, y = anchor
            if note_id in beam_anchors:
                x += direction * (beam_half_width_mm + dot_radius_mm + gap_mm)
            else:
                x += direction * gap_mm
            du.add_oval(
                x - dot_radius_mm,
                y - dot_radius_mm,
                x + dot_radius_mm,
                y + dot_radius_mm,
                stroke_color=None,
                fill_color=articulation_color,
                id=note_id,
                tags=['articulation', 'articulation_staccato'],
            )