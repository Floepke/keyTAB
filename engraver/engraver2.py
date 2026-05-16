"""New engraver2 with drawer-based architecture.

Renders notation using a stateless pipeline:
1. Normalize score (parse events, organize by hand)
2. Compute layout (geometry maps, x/y converters, line windows)
3. Instantiate drawers with shared EngravingContext
4. Each drawer renders its notation element independently

Currently delegates to legacy engraver for A/B testing. Gradual porting via Help menu toggle.
"""

from __future__ import annotations

from dataclasses import dataclass
import bisect
import traceback
from PySide6 import QtCore

from file_model.SCORE import SCORE
from file_model.analysis import Analysis
from file_model.events.line_break import LineBreak
from ui.style import Style
from ui.widgets.draw_util import DrawUtil

from utils.CONSTANT import PIANO_KEY_AMOUNT, BE_KEYS, BLACK_KEYS, SHORTEST_DURATION
from utils.operator import Operator

from engraver.engraver import Engraver as _LegacyEngraver
from engraver.engraver import do_engrave as _legacy_do_engrave

# Import all drawer classes
from engraver.drawers.paper_drawer import PaperDrawer
from engraver.drawers.stave_drawer import StaveDrawer
from engraver.drawers.grid_drawer import GridDrawer
from engraver.drawers.note_drawer import NoteheadDrawer
from engraver.drawers.beam_drawer import BeamDrawer
from engraver.drawers.arpeggio_drawer import ArpeggioDrawer
from engraver.drawers.slur_drawer import SlurDrawer
from engraver.drawers.dynamic_drawer import DynamicDrawer
from engraver.drawers.text_drawer import TextDrawer
from engraver.drawers.repeat_drawer import RepeatDrawer
from engraver.drawers.pedal_drawer import PedalDrawer
from engraver.drawers.time_signature_drawer import TimeSignatureDrawer
from engraver.drawers.grace_note_drawer import GraceNoteDrawer
from engraver.drawers.grid_band_drawer import GridBandDrawer
from engraver.drawers.mini_piano_drawer import MiniPianoDrawer
from engraver.drawers.header_footer_drawer import HeaderFooterDrawer
from engraver.drawers.count_line_drawer import CountLineDrawer


# Drawer pipeline order (independent, can be parallelized later)
DRAWER_PIPELINE = [
    PaperDrawer,             # Paper background
    GridBandDrawer,          # Background first
    GridDrawer,              # Grid structure
    StaveDrawer,             # Stave lines (vertical per key)
    TimeSignatureDrawer,     # Time signatures
    HeaderFooterDrawer,      # Header/footer
    CountLineDrawer,         # Count guides

    NoteheadDrawer,          # Notes (with per-note ledgers)
    # BeamDrawer,           # Beams (disabled for now)
    # ArpeggioDrawer,       # Arpeggios (disabled for now)
    # GraceNoteDrawer,      # Grace notes (disabled for now)

    # SlurDrawer,           # Slurs (disabled for now)
    # DynamicDrawer,        # Dynamics and hairpins (disabled for now)
    # PedalDrawer,          # Pedal symbols (disabled for now)
    # TextDrawer,           # Text annotations (disabled for now)

    MiniPianoDrawer,         # Mini piano (last, on top)
]


@dataclass
class EngravingContext:
    """Immutable shared state for the drawer pipeline."""

    score: SCORE
    normalized: dict
    layout_data: dict
    pageno: int
    pdf_export: bool
    du: DrawUtil
    drawer_caches: dict


def _reset_drawutil_pages(target: DrawUtil, source: DrawUtil) -> None:
    """Rebuild target pages from source dimensions, without copying draw items."""
    target._pages = []
    target._current_index = -1
    for page in source._pages:
        target.new_page(float(page.width_mm), float(page.height_mm))
        target.set_current_page_rotation_deg(float(getattr(page, 'rotation_deg', 0.0) or 0.0))


def _build_stave_layout_data(score: SCORE, page_lines_map: list[dict], notation_color: tuple[float, float, float, float], page_index: int) -> dict:
    op = Operator(SHORTEST_DURATION)
    """Build layout payload required by PaperDrawer and StaveDrawer."""
    score = score or {}
    layout = dict((score or {}).get('layout', {}) or {})
    events = dict((score or {}).get('events', {}) or {})
    line_breaks = list(events.get('line_break', []) or [])
    notes = list(events.get('note', []) or [])

    try:
        layout_scale = float(layout.get('scale', 1.0) or 1.0)
    except Exception:
        layout_scale = 1.0
    if layout_scale <= 0.0:
        layout_scale = 1.0

    page_orientation = str(layout.get('page_orientation', 'portrait') or 'portrait').strip().lower()
    if page_orientation == 'vertical':
        page_orientation = 'portrait'
    elif page_orientation == 'horizontal':
        page_orientation = 'landscape'

    read_direction = str(layout.get('read_direction', 'vertical') or 'vertical').strip().lower()
    horizontal_read_direction = read_direction == 'horizontal'
    landscape_page_orientation = page_orientation == 'landscape'
    raw_page_w = float(layout.get('page_width_mm', 210.0) or 210.0)
    raw_page_h = float(layout.get('page_height_mm', 297.0) or 297.0)
    swap_page_axes = landscape_page_orientation != horizontal_read_direction
    if swap_page_axes:
        page_w = raw_page_h
        page_h = raw_page_w
    else:
        page_w = raw_page_w
        page_h = raw_page_h

    semitone_mm = 2.0 * layout_scale

    def _build_key_positions(start_key: int, end_key: int, semitone_mm_local: float) -> dict[int, float]:
        positions: dict[int, float] = {}
        x = 0.0
        prev = None
        for key in range(start_key, end_key + 1):
            if prev is not None and prev in BE_KEYS:
                x += semitone_mm_local
            x += semitone_mm_local
            positions[key] = x
            prev = key
        return positions

    def _sanitize_range(rng) -> list[int]:
        if not isinstance(rng, list) or len(rng) < 2:
            return [1, PIANO_KEY_AMOUNT]
        lo = int(rng[0])
        hi = int(rng[1])
        lo = max(1, min(PIANO_KEY_AMOUNT, lo))
        hi = max(1, min(PIANO_KEY_AMOUNT, hi))
        if hi < lo:
            lo, hi = hi, lo
        return [lo, hi]

    def _pc_char(key: int) -> str:
        pc = (int(key) - 1) % 12
        if pc in (0, 2, 3, 5, 7, 8, 10):
            return {0: 'a', 2: 'b', 3: 'c', 5: 'd', 7: 'e', 8: 'f', 10: 'g'}[pc]
        return {1: 'A', 4: 'C', 6: 'D', 9: 'F', 11: 'G'}[pc]

    def _build_line_groups() -> list[dict]:
        groups: list[dict] = []
        used: set[int] = set()

        def _next_index(start: int, pc_target: str) -> int | None:
            for j in range(start + 1, len(BLACK_KEYS)):
                if j in used:
                    continue
                if _pc_char(BLACK_KEYS[j]) == pc_target:
                    return j
            return None

        for i, key in enumerate(BLACK_KEYS):
            if i in used:
                continue
            pc = _pc_char(key)
            if pc == 'C':
                keys = [key]
                j = _next_index(i, 'D')
                if j is not None:
                    keys.append(BLACK_KEYS[j])
                    used.add(j)
                used.add(i)
                groups.append({'keys': keys})
            elif pc == 'F':
                keys = [key]
                j = _next_index(i, 'G')
                if j is not None:
                    keys.append(BLACK_KEYS[j])
                    used.add(j)
                    k = _next_index(j, 'A')
                    if k is not None:
                        keys.append(BLACK_KEYS[k])
                        used.add(k)
                used.add(i)
                groups.append({'keys': keys})

        groups.sort(key=lambda g: g['keys'][0])
        for i, grp in enumerate(groups):
            first = grp['keys'][0]
            last = grp['keys'][-1]
            if i == 0:
                low = 1
            else:
                prev_last = groups[i - 1]['keys'][-1]
                low = int((prev_last + first) // 2) + 1
            if i == len(groups) - 1:
                high = PIANO_KEY_AMOUNT
            else:
                next_first = groups[i + 1]['keys'][0]
                high = int((last + next_first) // 2)
            grp['range_low'] = int(max(1, low))
            grp['range_high'] = int(min(PIANO_KEY_AMOUNT, high))
            if 41 in grp['keys'] and 43 in grp['keys']:
                grp['pattern'] = 'c' # clef group (c#4/db4 & d#4/eb4 clef lines | central lines)
            elif len(grp['keys']) == 2:
                grp['pattern'] = '2' # group of 2
            else:
                grp['pattern'] = '3' # group of 3
        return groups

    stave_line_groups = _build_line_groups()
    if not stave_line_groups:
        stave_line_groups = [{'keys': [41, 43], 'range_low': 1, 'range_high': PIANO_KEY_AMOUNT, 'pattern': 'c'}]

    clef_group_index = 0
    for i, grp in enumerate(stave_line_groups):
        if 41 in grp['keys'] and 43 in grp['keys']:
            clef_group_index = i
            break

    def _group_index_for_key(key: int) -> int:
        for i, grp in enumerate(stave_line_groups):
            if int(grp['range_low']) <= int(key) <= int(grp['range_high']):
                return i
        return 0 if int(key) <= int(stave_line_groups[0]['range_low']) else len(stave_line_groups) - 1

    def _visible_line_groups_for_range(lo: int, hi: int, include_clef: bool = True) -> list[dict]:
        lo = int(max(1, min(PIANO_KEY_AMOUNT, lo)))
        hi = int(max(1, min(PIANO_KEY_AMOUNT, hi)))
        if hi < lo:
            lo, hi = hi, lo
        min_group = _group_index_for_key(lo)
        max_group = _group_index_for_key(hi)
        if include_clef:
            if clef_group_index < min_group:
                min_group = clef_group_index
            if clef_group_index > max_group:
                max_group = clef_group_index
        return [stave_line_groups[gi] for gi in range(min_group, max_group + 1)]

    line_breaks = sorted(line_breaks, key=lambda lb: float(lb.get('time', 0.0) or 0.0))
    line_break_times: list[float] = [float(lb.get('time', 0.0) or 0.0) for lb in line_breaks]

    def _line_break_for_time(ticks: float) -> dict:
        """Return active line-break settings at time using thresholded comparison (Operator)."""
        idx = 0
        for i, t in enumerate(line_break_times):
            if op.greater(ticks, t):
                idx = i
            elif op.equal(ticks, t):
                idx = i
            else:
                break
        if idx < 0:
            idx = 0
        if idx >= len(line_breaks):
            idx = len(line_breaks) - 1
        lb = line_breaks[idx]
        return lb

    norm_notes: list[dict] = []
    for idx, n in enumerate(notes):
        if not isinstance(n, dict):
            continue
        n_t = float(n.get('time', 0.0) or 0.0)
        n_d = float(n.get('duration', 0.0) or 0.0)
        p = int(n.get('pitch', 0) or 0)
        hand_raw = str(n.get('hand', 'l') or 'l')
        hand_key = 'l' if hand_raw == 'l' else 'r'
        norm_notes.append({
            'time': n_t,
            'end': n_t + n_d,
            'duration': n_d,
            'pitch': p,
            'hand': hand_key,
            'id': int(n.get('_id', 0) or 0),
            'idx': int(idx),
            'raw': n,
        })

    key_positions = _build_key_positions(1, PIANO_KEY_AMOUNT, semitone_mm)
    page_lines: list[dict] = []
    key_to_x_for_line: dict[int, object] = {}
    pitch_to_x_for_line: dict[int, object] = {}
    rpitch_to_x_for_line: dict[int, object] = {}
    time_to_y_for_line: dict[int, object] = {}

    for line_index, line in enumerate(page_lines_map or []):
        x_start = float(line.get('x_start', 0.0) or 0.0)
        y_top = float(line.get('y_top', 0.0) or 0.0)
        y_bottom = float(line.get('y_bottom', y_top) or y_top)
        t0 = float(line.get('time_start', 0.0) or 0.0)
        t1 = float(line.get('time_end', t0) or t0)

        lb = _line_break_for_time(t0)
        stave_range = lb.get('stave_range', 'auto')
        if stave_range is True:
            stave_range = 'auto'
        if isinstance(stave_range, list) and len(stave_range) >= 2:
            r0 = int(stave_range[0])
            r1 = int(stave_range[1])
            if (r0 == 0 and r1 == 0) or (r0 == 1 and r1 == 1):
                stave_range = 'auto'

        def _note_range_for_window(w0: float, w1: float) -> tuple[int | None, int | None]:
            lo = None
            hi = None
            for item in norm_notes:
                n_t = float(item.get('time', 0.0) or 0.0)
                n_end = float(item.get('end', 0.0) or 0.0)
                p = int(item.get('pitch', 0) or 0)
                # Use Operator for time window overlap
                if op.less(n_t, w1) and op.greater(n_end, w0):
                    if p < 1 or p > PIANO_KEY_AMOUNT:
                        continue
                    lo = p if lo is None else min(lo, p)
                    hi = p if hi is None else max(hi, p)
            return lo, hi

        def _auto_line_keys_and_bounds(w0: float, w1: float) -> tuple[list[int], int, int]:
            lo, hi = _note_range_for_window(w0, w1)
            if lo is None or hi is None:
                grp = stave_line_groups[clef_group_index]
                keys = list(grp['keys'])
                return keys, int(keys[0]), int(keys[-1])
            groups = _visible_line_groups_for_range(int(lo), int(hi), include_clef=True)
            if not groups:
                grp = stave_line_groups[clef_group_index]
                keys = list(grp['keys'])
                return keys, int(keys[0]), int(keys[-1])
            keys: list[int] = []
            for grp in groups:
                keys.extend(grp['keys'])
            return keys, int(keys[0]), int(keys[-1])

        requested_lo = 1
        if stave_range == 'auto':
            visible_keys, bound_left, bound_right = _auto_line_keys_and_bounds(t0, t1)
        else:
            manual = _sanitize_range(stave_range)
            requested_lo = int(manual[0])
            groups = _visible_line_groups_for_range(manual[0], manual[1], include_clef=False)
            if not groups:
                groups = [stave_line_groups[clef_group_index]]
            visible_keys = []
            for grp in groups:
                visible_keys.extend(grp['keys'])
            bound_left = int(visible_keys[0])
            bound_right = int(visible_keys[-1])

        natural_bound_left = int(bound_left)
        natural_bound_right = int(bound_right)
        low_key_present = bool(bound_left <= 2 or (stave_range != 'auto' and int(requested_lo) <= 2))
        for item in norm_notes:
            n_t = float(item.get('time', 0.0) or 0.0)
            n_end = float(item.get('end', 0.0) or 0.0)
            p = int(item.get('pitch', 0) or 0)
            if n_t >= t1 or n_end <= t0:
                continue
            if p in (1, 2, 3):
                low_key_present = True
                break
        a0_ledger_mode = bool(low_key_present and int(natural_bound_left) > 2)

        line_notes: list[dict] = []
        for item in norm_notes:
            n_t = float(item.get('time', 0.0) or 0.0)
            n_end = float(item.get('end', 0.0) or 0.0)
            p = int(item.get('pitch', 0) or 0)
            # Use Operator for time window overlap
            if op.greater_or_equal(n_t, t1) or op.less_or_equal(n_end, t0):
                continue
            if p < 1 or p > PIANO_KEY_AMOUNT:
                continue
            line_notes.append(item)

        origin = float(key_positions.get(bound_left, 0.0))

        # --- Mini piano visibility logic (match legacy engraver.py) ---
        visual_first_line = (line_index == len(page_lines_map) - 1) if horizontal_read_direction else (line_index == 0)
        mini_piano_enabled = bool(layout.get('mini_piano_visible', True)) and page_index == 0 and visual_first_line
        mini_piano_height_mm = (7.0 * float(semitone_mm)) if mini_piano_enabled else 0.0
        # `y_bottom` in `page_lines_map` already comes from legacy page geometry.
        # Do not shorten again here, otherwise first-line stave and mini piano shift up.
        mini_piano_y_top = float(y_bottom)
        mini_piano_y_bottom = float(
            line.get(
                'mini_piano_y_bottom',
                (mini_piano_y_top + mini_piano_height_mm) if mini_piano_enabled else mini_piano_y_top,
            )
            or ((mini_piano_y_top + mini_piano_height_mm) if mini_piano_enabled else mini_piano_y_top)
        )
        page_lines.append({
            'stave_visible': True,
            'y_top': y_top,
            'y_bottom': y_bottom,
            'mini_piano_visible': bool(mini_piano_enabled),
            'mini_piano_height_mm': float(mini_piano_height_mm),
            'mini_piano_y_top': mini_piano_y_top,
            'mini_piano_y_bottom': mini_piano_y_bottom,
            'line_x_start': x_start,
            'time_start': float(t0),
            'time_end': float(t1),
            'range': [int(bound_left), int(bound_right)],
            'bound_left': int(bound_left),
            'bound_right': int(bound_right),
            'visible_keys': list(visible_keys),
            'natural_bound_left': int(natural_bound_left),
            'natural_bound_right': int(natural_bound_right),
            'low_key_left': bool(low_key_present),
            'a0_ledger_mode': bool(a0_ledger_mode),
            'stave_range': stave_range,
            'notes': line_notes,
        })

        key_to_x_for_line[line_index] = (
            lambda key, _x_start=x_start, _origin=origin:
            _x_start + (float(key_positions.get(int(key), 0.0)) - _origin)
        )
        pitch_to_x_for_line[line_index] = key_to_x_for_line[line_index]
        rpitch_to_x_for_line[line_index] = (
            lambda rpitch, _pitch_to_x=key_to_x_for_line[line_index], _semitone_mm=semitone_mm:
            _pitch_to_x(40) + (float(rpitch) * _semitone_mm)
        )
        time_to_y_for_line[line_index] = (
            lambda ticks, _t0=t0, _t1=t1, _y0=y_top, _y1=y_bottom:
            _y0 + ((_y1 - _y0) * max(0.0, min(1.0, (float(ticks) - float(_t0)) / max(1e-6, float(_t1) - float(_t0)))))
        )

    return {
        'page_lines': page_lines,
        'stave_line_groups': stave_line_groups,
        'key_to_x_for_line': key_to_x_for_line,
        'pitch_to_x_for_line': pitch_to_x_for_line,
        'rpitch_to_x_for_line': rpitch_to_x_for_line,
        'time_to_y_for_line': time_to_y_for_line,
        'key_positions': key_positions,
        'line_keys': list(BLACK_KEYS),
        'semitone_mm': semitone_mm,
        'page_width_mm': float(page_w),
        'page_height_mm': float(page_h),
        'layout': layout,
        'notation_color': notation_color,
        'paper_color': (1.0, 1.0, 1.0, 1.0),
        'scale': layout_scale,
    }


def do_engrave(score: SCORE, du: DrawUtil, pageno: int = 0, pdf_export: bool = False) -> None:
    """New engraver entry point with drawer-based pipeline.

    This is intentionally API-compatible with engraver.engraver.do_engrave so
    callers can switch backends at runtime while the new pipeline is migrated.
    
    For the first migration step this runs the StaveDrawer pipeline on top of
    legacy-computed page geometry so the new backend can be tested incrementally.
    """
    scratch_du = DrawUtil()
    _legacy_do_engrave(score, scratch_du, pageno=pageno, pdf_export=pdf_export)

    if scratch_du.page_count() <= 0:
        _legacy_do_engrave(score, du, pageno=pageno, pdf_export=pdf_export)
        return

    _reset_drawutil_pages(du, scratch_du)
    du.print_time_map = getattr(scratch_du, 'print_time_map', [])
    du.total_height_mm = float(getattr(scratch_du, 'total_height_mm', 0.0) or 0.0)
    du.max_stave_width_mm = float(getattr(scratch_du, 'max_stave_width_mm', 0.0) or 0.0)
    du._stave_time_spans_by_page = getattr(scratch_du, '_stave_time_spans_by_page', [])

    notation_rgb = Style.get_notation_color()
    paper_rgb = Style.get_paper_color()
    notation_color = (
        notation_rgb[0] / 255.0,
        notation_rgb[1] / 255.0,
        notation_rgb[2] / 255.0,
        1.0,
    )
    paper_color = (
        paper_rgb[0] / 255.0,
        paper_rgb[1] / 255.0,
        paper_rgb[2] / 255.0,
        1.0,
    )
    if pdf_export:
        notation_color = (0.0, 0.0, 0.0, 1.0)
        paper_color = (1.0, 1.0, 1.0, 1.0)

    print_time_map = list(getattr(scratch_du, 'print_time_map', []) or [])
    for page_index, page_lines_map in enumerate(print_time_map):
        du.set_current_page(int(page_index))
        layout_data = _build_stave_layout_data(score or {}, list(page_lines_map or []), notation_color, page_index)
        layout_data['paper_color'] = paper_color
        context = EngravingContext(
            score=score or {},
            normalized={},
            layout_data=layout_data,
            pageno=int(page_index),
            pdf_export=bool(pdf_export),
            du=du,
            drawer_caches={},
        )
        for Drawer in DRAWER_PIPELINE:
            Drawer(context).draw()

    if du.page_count() > 0:
        target_index = max(0, min(int(pageno), du.page_count() - 1))
        du.set_current_page(target_index)


def _engrave_worker(score: dict, request_id: int, pageno: int, out_conn) -> None:
    """Worker entry point for engraver2 backend."""
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


class Engraver(_LegacyEngraver):
    """Process-managed engraver facade for the new backend.

    Professional entry-point guideline:
    - Keep one stable class contract (`engrave`, `shutdown`, signals)
    - Keep one stable function contract (`do_engrave`)
    - Internals can evolve without changing UI wiring

    For now this reuses the legacy worker/state machine while routing through
    the new module boundary, allowing safe A/B switching from the UI.
    """

    engraved = QtCore.Signal()
    failed = QtCore.Signal(str, str)

    def __init__(self, draw_util: DrawUtil, parent=None):
        super().__init__(draw_util, parent)
        self.analysis: Analysis | None = None

    def _start_task(self, score: dict, pageno: int, request_id: int) -> None:
        """Start worker process using engraver2.do_engrave."""
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
