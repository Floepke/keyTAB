from PySide6 import QtCore
from datetime import datetime
import bisect, math
import multiprocessing as mp
import traceback
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import BE_KEYS, QUARTER_NOTE_UNIT, PIANO_KEY_AMOUNT, SHORTEST_DURATION, hex_to_rgba, BLACK_KEYS, ENGRAVER_FRACTIONAL_TEXT_SCALING_CORRECTION, SLUR_SEGMENT_COUNT
from utils.tiny_tool import key_class_filter
from utils.operator import Operator
from file_model.SCORE import SCORE
from file_model.layout import Layout
from file_model.base_grid import resolve_grid_layer_offsets
from file_model.info import Info
from file_model.analysis import Analysis
from ui.style import Style
from symbol_design.noteheads import (
    Notehead,
    normalize_notehead_literal,
    resolve_notehead_spec,
    sheared_notehead_outline_points,
    support_point_from_outline_points,
    sheared_notehead_support_v,
)
from symbol_design.pedal import draw_pedal_symbol
from file_model.events.note import Note
from engraver.helpers import (
    allow_font_registry as _allow_font_registry,
    black_note_above_stem as _black_note_above_stem,
    build_grid_band_dark_intervals as _build_grid_band_dark_intervals,
    group_by_beam_markers as _group_by_beam_markers,
    is_light_paper as _is_light_paper,
    normalize_hex_color as _normalize_hex_color,
    resolve_font_family_name as _resolve_font_family,
    scaled_dash_pattern_with_default as _scaled_dash_pattern_with_default,
    should_tune_under_stem_black_width as _should_tune_under_stem_black_width,
)

_MP_CONTEXT = mp.get_context("spawn")

def _grace_layout_no_tilt(layout: dict) -> dict:
    """Return layout dict copy with notehead_tilt set to 0 for grace notes visual contrast."""
    layout_copy = dict(layout)
    layout_copy['notehead_tilt'] = 0.0
    return layout_copy

def do_engrave(score: SCORE, du: DrawUtil, pageno: int = 0, pdf_export: bool = False) -> None:
    """Compute a full print layout and draw commands into DrawUtil.

    Problem solved: the engraver must be deterministic and thread-safe.
    It converts the score model into page/line geometry without any Qt
    rendering calls, then records only DrawUtil primitives.
    """
    score = score or {}
    layout: dict = dict(score.get('layout', {}) or {})
    base_grid: list = list(score.get('base_grid', []) or [])
    staves_raw: list = list(score.get('staves', []) or [])
    root_events: dict = dict(score.get('events', {}) or {})

    def _page_dimensions() -> tuple[float, float, float, float, float, float]:
        orientation = str(layout.get('page_orientation', 'portrait') or 'portrait').strip().lower()
        page_w = float(layout.get('page_width_mm', 210.0) or 210.0)
        page_h = float(layout.get('page_height_mm', 297.0) or 297.0)
        if orientation == 'landscape':
            page_w, page_h = page_h, page_w
        page_left = float(layout.get('page_left_margin_mm', 10.0) or 10.0)
        page_right = float(layout.get('page_right_margin_mm', 10.0) or 10.0)
        page_top = float(layout.get('page_top_margin_mm', 10.0) or 10.0)
        page_bottom = float(layout.get('page_bottom_margin_mm', 10.0) or 10.0)
        return page_w, page_h, page_left, page_right, page_top, page_bottom

    def _total_ticks(enabled_staves: list[dict]) -> float:
        total = 0.0
        for bg in base_grid:
            numer = int(bg.get('numerator', 4) or 4)
            denom = int(bg.get('denominator', 4) or 4)
            measures = int(bg.get('measure_amount', 1) or 1)
            measure_ticks = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
            total += measure_ticks * float(max(0, measures))
        if total > 0.0:
            return total

        max_end = 0.0
        for st in enabled_staves:
            for n in list((st.get('events', {}) or {}).get('note', []) or []):
                if not isinstance(n, dict):
                    continue
                t0 = float(n.get('time', 0.0) or 0.0)
                dur = float(n.get('duration', 0.0) or 0.0)
                max_end = max(max_end, t0 + dur)
        for n in list(root_events.get('note', []) or []):
            if not isinstance(n, dict):
                continue
            t0 = float(n.get('time', 0.0) or 0.0)
            dur = float(n.get('duration', 0.0) or 0.0)
            max_end = max(max_end, t0 + dur)
        return max(max_end, float(QUARTER_NOTE_UNIT))

    def _collect_enabled_staves() -> list[dict]:
        enabled: list[dict] = []
        for idx, st in enumerate(staves_raw):
            if not isinstance(st, dict):
                continue
            if not bool(st.get('enabled', True)):
                continue
            st_events = st.get('events', None)
            if not isinstance(st_events, dict):
                st_events = {}
            enabled.append({'index': int(idx), 'stave': st, 'events': st_events})
        if not enabled:
            enabled.append({'index': 0, 'stave': {}, 'events': root_events})
        return enabled

    def _pre_calculate() -> dict:
        page_w, page_h, page_left, page_right, page_top, page_bottom = _page_dimensions()
        semitone_mm = float(layout.get('semitone_mm', 2.5) or 2.5)
        stem_length_mm = float(layout.get('stem_length_mm', 6.0) or 6.0)
        beam_thickness_mm = float(layout.get('beam_thickness_mm', 1.0) or 1.0)

        enabled_staves = _collect_enabled_staves()
        total_ticks = _total_ticks(enabled_staves)
        draw_top = page_top
        draw_bottom = max(draw_top + 1.0, page_h - page_bottom)
        draw_h = max(1.0, draw_bottom - draw_top)

        def _time_to_y(tick_time: float) -> float:
            frac = 0.0 if total_ticks <= 0.0 else max(0.0, min(1.0, float(tick_time) / total_ticks))
            return draw_top + (frac * draw_h)

        stave_precalc: list[dict] = []
        cursor_x = page_left
        for st_info in enabled_staves:
            st = st_info.get('stave', {}) or {}
            st_events = st_info.get('events', {}) or {}
            notes = [n for n in list(st_events.get('note', []) or []) if isinstance(n, dict)]

            pitches = [int(n.get('pitch', 0) or 0) for n in notes if 1 <= int(n.get('pitch', 0) or 0) <= PIANO_KEY_AMOUNT]
            if pitches:
                key_min = int(min(pitches))
                key_max = int(max(pitches))
            else:
                key_range = list(st.get('key_range', []) or [])
                if len(key_range) >= 2:
                    key_min = int(key_range[0])
                    key_max = int(key_range[1])
                    if key_max < key_min:
                        key_min, key_max = key_max, key_min
                else:
                    key_min, key_max = 1, PIANO_KEY_AMOUNT

            note_width_mm = max(semitone_mm, float((key_max - key_min + 1)) * semitone_mm)
            margin_left_mm = float(st.get('margin_left_mm', 0.0) or 0.0)
            margin_right_mm = float(st.get('margin_right_mm', 0.0) or 0.0)

            content_left = cursor_x + margin_left_mm
            content_right = content_left + note_width_mm

            stave_precalc.append(
                {
                    'stave_index': int(st_info.get('index', 0) or 0),
                    'events': st_events,
                    'notes': notes,
                    'key_min': key_min,
                    'key_max': key_max,
                    'content_left_x_mm': float(content_left),
                    'content_right_x_mm': float(content_right),
                    'outer_left_x_mm': float(content_left),
                    'outer_right_x_mm': float(content_right),
                    'margin_left_mm': float(margin_left_mm),
                    'margin_right_mm': float(margin_right_mm),
                    'stave_width_mm': float(note_width_mm),
                    'beam_segments': [],
                }
            )
            cursor_x = content_right + margin_right_mm

        # Beam segments are pre-calculated as concrete draw coordinates.
        for stv in stave_precalc:
            notes = list(stv.get('notes', []) or [])
            notes_by_time = sorted(notes, key=lambda n: float(n.get('time', 0.0) or 0.0))
            beam_markers = [b for b in list((stv.get('events', {}) or {}).get('beam', []) or []) if isinstance(b, dict)]

            def _key_to_x(pitch: int) -> float:
                pitch_clamped = max(1, min(PIANO_KEY_AMOUNT, int(pitch)))
                return float(stv['content_left_x_mm']) + float(pitch_clamped - int(stv['key_min'])) * semitone_mm

            for marker in beam_markers:
                t0 = float(marker.get('time', 0.0) or 0.0)
                dur = float(marker.get('duration', 0.0) or 0.0)
                t1 = t0 + max(0.0, dur)
                hand = 'r' if str(marker.get('hand', 'l') or 'l') == 'r' else 'l'

                group = [
                    n for n in notes_by_time
                    if float(n.get('time', 0.0) or 0.0) >= t0 and float(n.get('time', 0.0) or 0.0) < t1
                ]
                if len(group) < 2:
                    continue

                first_note = group[0]
                last_note = group[-1]
                if hand == 'r':
                    ref_first = max(group, key=lambda n: int(n.get('pitch', 0) or 0))
                    ref_last = ref_first
                    x1 = _key_to_x(int(ref_first.get('pitch', 0) or 0)) + stem_length_mm
                    x2 = _key_to_x(int(ref_last.get('pitch', 0) or 0)) + stem_length_mm
                else:
                    ref_first = min(group, key=lambda n: int(n.get('pitch', 0) or 0))
                    ref_last = ref_first
                    x1 = _key_to_x(int(ref_first.get('pitch', 0) or 0)) - stem_length_mm
                    x2 = _key_to_x(int(ref_last.get('pitch', 0) or 0)) - stem_length_mm

                y1 = _time_to_y(float(first_note.get('time', 0.0) or 0.0))
                y2 = _time_to_y(float(last_note.get('time', 0.0) or 0.0))
                seg = {
                    'stave_index': int(stv.get('stave_index', 0) or 0),
                    'hand': hand,
                    'x1_mm': float(x1),
                    'y1_mm': float(y1),
                    'x2_mm': float(x2),
                    'y2_mm': float(y2),
                    'time_start': float(t0),
                    'time_end': float(t1),
                }
                stv['beam_segments'].append(seg)

                seg_left = min(float(x1), float(x2)) - beam_thickness_mm * 0.5
                seg_right = max(float(x1), float(x2)) + beam_thickness_mm * 0.5
                stv['outer_left_x_mm'] = min(float(stv['outer_left_x_mm']), seg_left)
                stv['outer_right_x_mm'] = max(float(stv['outer_right_x_mm']), seg_right)

            stv['stave_outer_width_mm'] = float(stv['outer_right_x_mm']) - float(stv['outer_left_x_mm'])

        if stave_precalc:
            system_left = min(float(st['outer_left_x_mm']) for st in stave_precalc)
            system_right = max(float(st['outer_right_x_mm']) for st in stave_precalc)
        else:
            system_left = page_left
            system_right = page_left

        return {
            'page': {
                'width_mm': float(page_w),
                'height_mm': float(page_h),
                'left_margin_mm': float(page_left),
                'right_margin_mm': float(page_right),
                'top_margin_mm': float(page_top),
                'bottom_margin_mm': float(page_bottom),
            },
            'timeline': {
                'total_ticks': float(total_ticks),
                'draw_top_mm': float(draw_top),
                'draw_bottom_mm': float(draw_bottom),
            },
            'staves': stave_precalc,
            'system': {
                'left_x_mm': float(system_left),
                'right_x_mm': float(system_right),
                'width_mm': max(0.0, float(system_right) - float(system_left)),
            },
        }

    def _draw(precalc: dict) -> None:
        page = precalc.get('page', {}) or {}
        system = precalc.get('system', {}) or {}
        timeline = precalc.get('timeline', {}) or {}

        page_w = float(page.get('width_mm', 210.0) or 210.0)
        page_h = float(page.get('height_mm', 297.0) or 297.0)
        y_top = float(timeline.get('draw_top_mm', 10.0) or 10.0)
        y_bottom = float(timeline.get('draw_bottom_mm', page_h - 10.0) or (page_h - 10.0))
        x_left = float(system.get('left_x_mm', 10.0) or 10.0)
        x_right = float(system.get('right_x_mm', x_left) or x_left)

        du._pages = []
        du._current_index = -1
        du.new_page(page_w, page_h)

        du.add_rectangle(
            x_left,
            y_top,
            x_right,
            y_bottom,
            stroke_color=(0.85, 0.2, 0.2, 0.95),
            stroke_width_mm=0.5,
            fill_color=None,
            dash_pattern=[2.5, 1.5],
            tags=['system-debug-rect'],
        )

    pre_calculated = _pre_calculate()
    _draw(pre_calculated)


def _engrave_worker(score: dict, request_id: int, pageno: int, out_conn) -> None:
    """Worker entry point to build DrawUtil in a separate process.

    Problem solved: isolate heavy engraving work from the UI thread.
    """
    try:
        local_du = DrawUtil()
        do_engrave(score, local_du, pageno=pageno)
        out_conn.send(('ok', int(request_id), local_du))
    except Exception as exc:
        try:
            out_conn.send(('error', int(request_id), str(exc), traceback.format_exc()))
        except Exception:
            pass
    finally:
        try:
            out_conn.close()
        except Exception:
            pass


class Engraver(QtCore.QObject):
    """Convenient engraver API ensuring single-run with latest-request semantics.

    - Call engrave(score) to request an engraving.
    - If one is running, stores the latest pending request and runs it next.
    - Skips intermediate requests; never runs two tasks at the same time.
    """

    engraved = QtCore.Signal()
    failed = QtCore.Signal(str, str)

    def __init__(self, draw_util: DrawUtil, parent=None):
        super().__init__(parent)
        self._du: DrawUtil = draw_util
        self._mp_ctx = _MP_CONTEXT
        self._result_recv = None
        self._result_send = None
        self._proc: mp.Process | None = None
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._poll_results)
        self._running: bool = False
        self._pending_score: dict | None = None
        self._pending_pageno: int | None = None
        self._pending_request_id: int | None = None
        self._latest_request_id: int = 0
        self._min_interval_ms: int = 500
        self._last_start_ms: int = -500
        self._elapsed = QtCore.QElapsedTimer()
        self._elapsed.start()
        self._delay_timer = QtCore.QTimer(self)
        self._delay_timer.setSingleShot(True)
        self._delay_timer.timeout.connect(self._maybe_start_pending)
        self.analysis: Analysis | None = None

    def _close_result_pipe(self) -> None:
        if self._result_send is not None:
            self._result_send.close()
            self._result_send = None
        if self._result_recv is not None:
            self._result_recv.close()
            self._result_recv = None

    def engrave(self, score: dict, pageno: int | None = None) -> None:
        """Request an engraving; coalesce to the most recent request.

        Problem solved: avoid a backlog of obsolete renders during edits.
        """
        if pageno is None:
            try:
                pageno = int(self._du.current_page_index())
            except Exception:
                pageno = 0
        self._latest_request_id += 1
        req_id = int(self._latest_request_id)
        # If currently running, just replace the pending request
        if self._running:
            self._pending_score = dict(score or {})
            self._pending_pageno = int(pageno)
            self._pending_request_id = req_id
            return
        self._pending_score = dict(score or {})
        self._pending_pageno = int(pageno)
        self._pending_request_id = req_id
        self._maybe_start_pending()

    def _maybe_start_pending(self) -> None:
        """Start a pending request if throttling allows it.

        Problem solved: rate-limit engraving so rapid edits do not spawn too
        many processes.
        """
        if self._running:
            return
        if self._pending_score is None:
            return
        if self._pending_pageno is None:
            return
        if self._pending_request_id is None:
            return
        elapsed_ms = int(self._elapsed.elapsed())
        since_last = elapsed_ms - int(self._last_start_ms)
        if since_last >= self._min_interval_ms:
            next_score = self._pending_score
            next_pageno = int(self._pending_pageno)
            next_req_id = int(self._pending_request_id)
            self._pending_score = None
            self._pending_pageno = None
            self._pending_request_id = None
            self._start_task(next_score, next_pageno, next_req_id)
            return
        delay_ms = max(1, int(self._min_interval_ms - since_last))
        if self._delay_timer.isActive():
            self._delay_timer.stop()
        self._delay_timer.start(delay_ms)

    def _start_task(self, score: dict, pageno: int, request_id: int) -> None:
        """Start a new process to engrave the given score.

        Problem solved: terminate stale workers before launching a new one.
        """
        self._running = True
        self._last_start_ms = int(self._elapsed.elapsed())
        if self._proc is not None:
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc.join(timeout=0.1)
            self._proc = None
        self._close_result_pipe()
        self._result_recv, self._result_send = self._mp_ctx.Pipe(duplex=False)
        self._proc = self._mp_ctx.Process(
            target=_engrave_worker,
            args=(score, request_id, pageno, self._result_send),
            daemon=True,
        )
        self._proc.start()
        if self._result_send is not None:
            try:
                self._result_send.close()
            except Exception:
                pass
            self._result_send = None
        if not self._poll_timer.isActive():
            self._poll_timer.start()

    def _poll_results(self) -> None:
        """Drain worker results and advance the state machine.

        Problem solved: process can exit without a result; this keeps the
        state machine moving and restarts pending work.
        """
        got_result = False
        if self._result_recv is not None:
            try:
                has_result = bool(self._result_recv.poll())
            except (EOFError, OSError):
                has_result = False
            if has_result:
                try:
                    payload = self._result_recv.recv()
                except (EOFError, OSError):
                    pass
                else:
                    got_result = True
                    self._close_result_pipe()
                    kind = str(payload[0]) if isinstance(payload, tuple) and payload else 'ok'
                    if kind == 'ok':
                        _kind, req_id, result_du = payload
                        self._on_finished(req_id, result_du)
                    else:
                        _kind, req_id, error_text, error_details = payload
                        self._on_failed(req_id, str(error_text), str(error_details))

        if self._proc is not None and not self._proc.is_alive():
            self._proc.join(timeout=0.1)
            self._proc = None
            self._close_result_pipe()
            if self._running and not got_result:
                self._running = False
                if self._pending_score is None:
                    self.failed.emit(
                        'Engraving failed',
                        'The engraver worker exited without returning a result.',
                    )
                if self._pending_score is not None:
                    self._maybe_start_pending()
            if not self._running:
                self._poll_timer.stop()

    def shutdown(self) -> None:
        """Stop timers and terminate the worker process if it is still running.

        Problem solved: prevent orphan processes on app shutdown.
        """
        if self._poll_timer.isActive():
            self._poll_timer.stop()
        if self._delay_timer.isActive():
            self._delay_timer.stop()
        if self._proc is not None:
            if self._proc.is_alive():
                self._proc.terminate()
            self._proc.join(timeout=0.1)
            self._proc = None
        self._close_result_pipe()
        self._running = False
        self._pending_score = None
        self._pending_pageno = None
        self._pending_request_id = None

    @QtCore.Slot(int, object)
    def _on_finished(self, request_id: int, result_du: DrawUtil) -> None:
        # Called on worker completion; schedule next or emit signal
        self._running = False
        if self._pending_score is not None:
            # Grab and clear the latest pending, then run it
            self._maybe_start_pending()
            return
        # No pending: notify listeners (e.g., to request render)
        if int(request_id) == int(self._latest_request_id):
            self._du._pages = list(result_du._pages)
            self._du._current_index = int(result_du._current_index)
            self.analysis = getattr(result_du, 'analysis', None)
            self._du.analysis = self.analysis
            self._du.print_time_map = getattr(result_du, 'print_time_map', [])
            self.engraved.emit()

    @QtCore.Slot(int, str, str)
    def _on_failed(self, request_id: int, error_text: str, error_details: str) -> None:
        self._running = False
        if self._pending_score is not None:
            self._maybe_start_pending()
        if int(request_id) == int(self._latest_request_id):
            self.failed.emit(str(error_text or 'Engraving failed'), str(error_details or ''))
