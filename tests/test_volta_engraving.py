import os
import unittest
from unittest.mock import patch

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from PySide6.QtWidgets import QApplication

from engraver.engraver import do_engrave
from editor.drawers.volta_drawer import VoltaDrawerMixin
from file_model.SCORE import SCORE
from ui.widgets.draw_util import DrawUtil


class VoltaEngravingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def score_data(self, direction='vertical'):
        score = SCORE().new()
        score.new_volta(time=0.0, duration=1024.0, number=12)
        data = score.get_dict()
        data['staves'] = []
        data['events']['line_break'][0]['stave_range'] = [35, 45]
        data['layout']['read_direction'] = direction
        return data

    def engrave(self, data, pdf_export=False):
        du = DrawUtil()
        do_engrave(data, du, pdf_export=pdf_export)
        return du, [item for page in du._pages for item in page.items]

    def bracket(self, items):
        return [item for item in items if 'volta_bracket' in item.tags]

    def test_solid_bracket_and_number_inside_in_both_directions(self):
        for direction in ('vertical', 'horizontal'):
            with self.subTest(direction=direction):
                data = self.score_data(direction)
                du, items = self.engrave(data)
                top, spine, bottom = self.bracket(items)
                self.assertEqual(top.y1_mm, top.y2_mm)
                self.assertEqual(spine.x1_mm, spine.x2_mm)
                self.assertEqual(bottom.y1_mm, bottom.y2_mm)
                self.assertEqual(top.y1_mm, spine.y1_mm)
                self.assertEqual(bottom.y1_mm, spine.y2_mm)
                self.assertLess(top.x1_mm, top.x2_mm)
                self.assertIsNone(spine.stroke.dash_pattern_mm)
                self.assertAlmostEqual(
                    spine.stroke.width_mm,
                    data['layout']['volta_thickness_mm'] * data['layout']['scale'],
                )
                text = next(item for item in items if 'volta_text' in item.tags)
                self.assertEqual(text.text, '12.')
                xb, yb, w, h = du._get_text_extents_mm(
                    text.text, text.family, text.size_pt, text.italic, text.bold,
                )
                self.assertEqual(text.anchor, 'center')
                center_x = text.x_mm + xb + w * 0.5
                center_y = text.y_mm + yb + h * 0.5
                self.assertAlmostEqual(center_x, (top.x1_mm + top.x2_mm) * 0.5)
                self.assertAlmostEqual(center_y, (spine.y1_mm + spine.y2_mm) * 0.5)
                span_x, span_y = (h, w) if direction == 'horizontal' else (w, h)
                left, right = center_x - span_x * 0.5, center_x + span_x * 0.5
                ink_top = center_y - span_y * 0.5
                self.assertGreater(left, top.x1_mm)
                self.assertLess(right, spine.x1_mm)
                self.assertGreater(ink_top, top.y1_mm)
                self.assertLess(ink_top, bottom.y1_mm)

    def test_editor_number_is_centered_in_both_directions(self):
        score = SCORE().new()
        score.new_volta(time=128.0, duration=512.0, number=12)

        class EditorStub(VoltaDrawerMixin):
            margin = 35.0
            _viewport_h_mm = 600.0

            def current_score(self):
                return score

            def time_to_mm(self, time):
                return time * 0.1

            def register_hit_rect(self, *args):
                pass

        for direction in ('vertical', 'horizontal'):
            with self.subTest(direction=direction):
                du = DrawUtil()
                du.new_page(210.0, 297.0)
                with patch('editor.drawers.volta_drawer.get_preferences',
                           return_value={'editor_orientation': direction}):
                    EditorStub().draw_volta(du)
                items = du._pages[0].items
                top, spine, _bottom = self.bracket(items)
                text = next(it for it in items if 'volta_text' in it.tags)
                xb, yb, w, h = du._get_text_extents_mm(
                    text.text, text.family, text.size_pt, text.italic, text.bold,
                )
                self.assertEqual(text.anchor, 'center')
                self.assertEqual(text.text, '12.')
                self.assertAlmostEqual(text.x_mm + xb + w * 0.5,
                                       (top.x1_mm + top.x2_mm) * 0.5)
                self.assertAlmostEqual(text.y_mm + yb + h * 0.5,
                                       (spine.y1_mm + spine.y2_mm) * 0.5)
                self.assertEqual(text.angle_deg, 90.0 if direction == 'horizontal' else 0.0)

    def test_ducks_past_notes_and_beams_inside_the_span(self):
        for direction in ('vertical', 'horizontal'):
            for beamed in (False, True):
                with self.subTest(direction=direction, beamed=beamed):
                    data = self.score_data(direction)
                    _, baseline = self.engrave(data)
                    baseline_left = self.bracket(baseline)[0].x1_mm
                    data['events']['note'] = [
                        dict(pitch=60, time=128, duration=128, hand='r'),
                        dict(pitch=60, time=512, duration=128, hand='r'),
                    ]
                    if beamed:
                        data['events']['beam'] = [dict(time=128, duration=512, hand='r')]
                    _, items = self.engrave(data)
                    top = self.bracket(items)[0]
                    self.assertGreater(top.x1_mm, baseline_left)
                    beam_stems = [item for item in items if 'beam_stem' in item.tags]
                    self.assertEqual(bool(beam_stems), beamed)
                    for stem in beam_stems:
                        self.assertGreater(
                            top.x1_mm - top.stroke.width_mm * 0.5,
                            max(stem.x1_mm, stem.x2_mm) + stem.stroke.width_mm * 0.5,
                        )

    def test_offset_translates_bracket_and_number_in_both_directions(self):
        for direction in ('vertical', 'horizontal'):
            for offset in (-3.0, 7.0):
                with self.subTest(direction=direction, offset=offset):
                    data = self.score_data(direction)
                    _, before = self.engrave(data)
                    data['volta'][0]['x_offset'] = offset
                    _, after = self.engrave(data)
                    before = [it for it in before if any(t.startswith('volta_') for t in it.tags)]
                    after = [it for it in after if any(t.startswith('volta_') for t in it.tags)]
                    for a, b in zip(before, after):
                        ax = a.x1_mm if 'volta_bracket' in a.tags else a.x_mm
                        bx = b.x1_mm if 'volta_bracket' in b.tags else b.x_mm
                        self.assertAlmostEqual(bx - ax, offset * data['layout']['scale'])

    def test_visibility_and_nonpositive_durations(self):
        for duration, visible in ((0.0, True), (-128.0, True), (1024.0, False)):
            with self.subTest(duration=duration, visible=visible):
                data = self.score_data()
                data['volta'][0]['duration'] = duration
                data['layout']['volta_visible'] = visible
                _, items = self.engrave(data)
                self.assertFalse(self.bracket(items))

    def test_continues_across_system_and_page_breaks(self):
        for page_break in (False, True):
            with self.subTest(page_break=page_break):
                data = self.score_data()
                data['events']['line_break'].append(dict(
                    time=512, margin_mm=[5.0, 5.0],
                    stave_range=[35, 45], page_break=page_break,
                ))
                _, items = self.engrave(data, pdf_export=True)
                self.assertEqual(len(self.bracket(items)), 6)
                self.assertEqual(sum('volta_text' in it.tags for it in items), 2)

    def test_tempo_geometry_is_unchanged(self):
        data = self.score_data()
        _, with_volta = self.engrave(data)
        data['volta'] = []
        _, without_volta = self.engrave(data)
        self.assertEqual(
            [it for it in with_volta if any(t.startswith('tempo_') for t in it.tags)],
            [it for it in without_volta if any(t.startswith('tempo_') for t in it.tags)],
        )

    def test_initial_anchor_matches_tempo(self):
        for direction in ('vertical', 'horizontal'):
            with self.subTest(direction=direction):
                du, items = self.engrave(self.score_data(direction))
                tempo = next(it for it in items if 'tempo_text' in it.tags)
                xb, _, w, h = du._get_text_extents_mm(
                    tempo.text, tempo.family, tempo.size_pt, tempo.italic, tempo.bold,
                )
                center_x = tempo.x_mm + xb + w * 0.5
                tempo_left = center_x - (h if direction == 'horizontal' else w) * 0.5
                self.assertAlmostEqual(self.bracket(items)[0].x1_mm, tempo_left)

    def test_uses_outer_stave_and_its_scale(self):
        for direction, stave_scale in (('vertical', 0.5), ('horizontal', 1.0)):
            with self.subTest(direction=direction):
                data = self.score_data(direction)
                data['staves'] = [
                    dict(enabled=True, scale=1.0, events=data['events']),
                    dict(enabled=True, scale=0.5, events=data['events']),
                ]
                _, items = self.engrave(data)
                self.assertEqual(len(self.bracket(items)), 3)
                self.assertAlmostEqual(
                    self.bracket(items)[0].stroke.width_mm,
                    data['layout']['volta_thickness_mm'] * data['layout']['scale'] * stave_scale,
                )
                text = next(it for it in items if 'volta_text' in it.tags)
                self.assertAlmostEqual(
                    text.size_pt,
                    data['layout']['volta_font']['size_pt'] * data['layout']['scale'] * stave_scale,
                )

    def test_midmeasure_and_short_brackets_without_tempo_or_measure_numbers(self):
        data = self.score_data()
        data['tempo'] = []
        data['layout']['measure_numbers_visible'] = False
        data['layout']['measure_numbering_guide_visible'] = False
        data['volta'][0].update(time=128.0, duration=32.0)
        _, items = self.engrave(data)
        top, spine, bottom = self.bracket(items)
        self.assertGreater(spine.y2_mm, spine.y1_mm)
        self.assertGreater(top.x2_mm, top.x1_mm)
        self.assertEqual(spine.y2_mm, bottom.y1_mm)


if __name__ == '__main__':
    unittest.main()
