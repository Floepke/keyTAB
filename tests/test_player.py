import os
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from midi.player import Player
from utils.CONSTANT import QUARTER_NOTE_UNIT


class MockBackend:
    def __init__(self, soundfont_path=None):
        self.program_select = Mock()
        self.set_gain = Mock()
        self.note_on = Mock()
        self.note_off = Mock()
        self.all_notes_off = Mock()


class PlayerRepeatTests(unittest.TestCase):
    def setUp(self):
        with patch('midi.player._FluidsynthBackend', MockBackend), \
                patch('midi.player.sys.platform', 'linux'), \
                patch.object(Player, '_load_gain_from_appdata'), \
                patch.object(Player, '_load_reverb_settings_from_appdata'):
            self.player = Player()
        self.backend = self.player._backend
        self.now = 100.0
        self.notes = []
        self.backend.note_on.side_effect = self.note_on
        self.on_note = lambda: None
        self.on_sleep = lambda: None

    def note_on(self, pitch, velocity):
        self.notes.append((pitch, self.now))
        self.on_note()

    def sleep(self, seconds):
        self.now += seconds
        self.on_sleep()

    def score(self, notes=None, start_repeat=(), end_repeat=(), tempo=()):
        if notes is None:
            notes = [(0.0, QUARTER_NOTE_UNIT, 40)]
        return SimpleNamespace(
            staves=[SimpleNamespace(
                enabled=True,
                events=SimpleNamespace(note=[
                    SimpleNamespace(time=t, duration=d, pitch=p, velocity=80, _id=i)
                    for i, (t, d, p) in enumerate(notes)
                ]),
            )],
            events=SimpleNamespace(
                start_repeat=[SimpleNamespace(time=t) for t in start_repeat],
                end_repeat=[SimpleNamespace(time=t) for t in end_repeat],
            ),
            tempo=list(tempo),
        )

    def run_playback(self, score, start_units=None):
        with patch('midi.player.threading.Thread') as thread, \
                patch('midi.player.time.time', side_effect=lambda: self.now), \
                patch('midi.player.time.sleep', side_effect=self.sleep):
            thread.return_value.is_alive.return_value = False
            if start_units is None:
                self.player.play_score(score)
            else:
                self.player.play_from_time_cursor(start_units, score)
            thread.call_args.kwargs['target']()
        self.assertFalse(self.player.is_playing())
        self.backend.all_notes_off.assert_called()

    def disable_after(self, count):
        def on_note():
            if len(self.notes) == count:
                self.player.set_repeat_song(False)
        self.on_note = on_note

    def test_repeat_is_off_by_default(self):
        self.run_playback(self.score())
        self.assertEqual([pitch for pitch, _ in self.notes], [60])

    def test_repeats_whole_song_and_resets_playhead(self):
        self.player.set_repeat_song(True)
        playheads = []
        score = self.score()

        def on_note():
            self.assertTrue(self.player.is_playing())
            playheads.append(self.player.get_playhead_time(score))
            if len(self.notes) == 3:
                self.player.set_repeat_song(False)
        self.on_note = on_note
        self.run_playback(score)
        self.assertEqual([pitch for pitch, _ in self.notes], [60, 60, 60])
        self.assertEqual(playheads, [0.0, 0.0, 0.0])
        self.assertAlmostEqual(self.notes[1][1] - self.notes[0][1], 0.5)
        self.assertAlmostEqual(self.notes[2][1] - self.notes[1][1], 0.5)

    def test_enabling_repeat_during_playback(self):
        def on_note():
            self.player.set_repeat_song(len(self.notes) == 1)
        self.on_note = on_note
        self.run_playback(self.score())
        self.assertEqual(len(self.notes), 2)

    def test_cursor_playback_repeats_from_song_beginning(self):
        self.player.set_repeat_song(True)
        self.disable_after(3)
        q = QUARTER_NOTE_UNIT
        self.run_playback(self.score(notes=[(0, q, 40), (q, q, 42)]), start_units=q)
        self.assertEqual([pitch for pitch, _ in self.notes], [62, 60, 62])
        self.assertAlmostEqual(self.notes[1][1] - self.notes[0][1], 0.5)
        self.assertEqual(self.player._start_units, 0.0)

    def test_score_repeat_symbols_are_preserved_in_every_loop(self):
        self.player.set_repeat_song(True)
        self.disable_after(4)
        self.run_playback(self.score(end_repeat=[QUARTER_NOTE_UNIT * 2]))
        self.assertEqual(len(self.notes), 4)
        self.assertAlmostEqual(self.notes[1][1] - self.notes[0][1], 1.0)
        self.assertAlmostEqual(self.notes[2][1] - self.notes[0][1], 2.0)

    def test_loop_duration_follows_tempo_changes(self):
        self.player.set_repeat_song(True)
        self.disable_after(3)
        q = QUARTER_NOTE_UNIT
        tempo = [
            SimpleNamespace(time=0, tempo=120, duration=q),
            SimpleNamespace(time=q, tempo=60, duration=q),
        ]
        self.run_playback(self.score(notes=[(0, 2 * q, 40)], tempo=tempo))
        self.assertAlmostEqual(self.notes[1][1] - self.notes[0][1], 1.5)
        self.assertAlmostEqual(self.notes[2][1] - self.notes[1][1], 1.5)

    def test_empty_score_does_not_loop(self):
        self.player.set_repeat_song(True)
        self.run_playback(self.score(notes=[]))
        self.assertEqual(self.notes, [])

    def test_cursor_after_last_note_does_not_loop(self):
        self.player.set_repeat_song(True)
        self.run_playback(self.score(), start_units=QUARTER_NOTE_UNIT * 2)
        self.assertEqual(self.notes, [])

    def test_stop_during_long_rest_does_not_restart(self):
        self.player.set_repeat_song(True)
        self.on_sleep = self.player.stop
        self.run_playback(self.score(notes=[(QUARTER_NOTE_UNIT * 100, QUARTER_NOTE_UNIT, 40)]))
        self.assertEqual(self.notes, [])
        self.assertLess(self.now, 100.02)

    def test_disabling_during_final_rest_prevents_next_loop(self):
        self.player.set_repeat_song(True)
        self.on_sleep = lambda: self.player.set_repeat_song(self.now < 100.75)
        self.run_playback(self.score(end_repeat=[QUARTER_NOTE_UNIT * 2]))
        # Two notes come from the score's repeat symbol, not whole-song looping.
        self.assertEqual(len(self.notes), 2)

    def test_real_worker_stops_after_repeating(self):
        repeated = threading.Event()
        count = 0

        def on_note(pitch, velocity):
            nonlocal count
            count += 1
            if count >= 3:
                repeated.set()

        self.backend.note_on.side_effect = on_note
        self.player.set_repeat_song(True)
        self.player.play_score(self.score(notes=[(0, QUARTER_NOTE_UNIT / 16, 40)]))
        worker = self.player._thread
        try:
            self.assertTrue(repeated.wait(timeout=2.0))
        finally:
            self.player.stop()
        self.assertFalse(worker.is_alive())
        self.assertFalse(self.player.is_playing())


class RepeatMenuTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
        from PySide6.QtWidgets import QApplication
        from ui.main_window import MainWindow
        cls.app = QApplication.instance() or QApplication([])
        cls.window_type = MainWindow

    def setUp(self):
        from PySide6.QtGui import QAction
        self.action = QAction('Repeat song')
        self.action.setCheckable(True)
        self.window = SimpleNamespace(
            player=None,
            _player_config=None,
            _repeat_song_action=self.action,
            _get_playback_mode_from_appdata=lambda: 'system',
            _get_midi_out_port_from_appdata=lambda: '',
            _get_soundfont_path_from_appdata=lambda: None,
        )
        self.action.toggled.connect(
            lambda checked: self.window_type._set_repeat_song(self.window, checked)
        )

    def test_toggle_updates_existing_player_and_saves_setting(self):
        self.window.player = Mock()
        with patch('ui.main_window.get_appdata_manager') as get_appdata:
            for checked in (True, False):
                self.action.setChecked(checked)
                self.window.player.set_repeat_song.assert_called_with(checked)
                get_appdata.return_value.set.assert_called_with('repeat_song', checked)
            self.assertEqual(get_appdata.return_value.save.call_count, 2)

    def test_toggle_before_player_creation_and_backend_recreation(self):
        with patch('ui.main_window.get_appdata_manager'), \
                patch('midi.player.Player') as player_type:
            self.action.setChecked(True)
            self.assertIsNone(self.window.player)
            self.window_type._ensure_player(self.window)
            player_type.return_value.set_repeat_song.assert_called_with(True)
            # Recreating the output backend must not lose the menu setting.
            self.window.player = None
            self.window_type._ensure_player(self.window)
            self.assertEqual(player_type.call_count, 2)
            player_type.return_value.set_repeat_song.assert_called_with(True)


if __name__ == '__main__':
    unittest.main()
