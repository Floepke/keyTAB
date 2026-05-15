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
import traceback
from PySide6 import QtCore

from file_model.SCORE import SCORE
from file_model.analysis import Analysis
from ui.style import Style
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import PIANO_KEY_AMOUNT

from engraver.engraver import Engraver as _LegacyEngraver
from engraver.engraver import do_engrave as _legacy_do_engrave

# Import all drawer classes
from engraver.drawers.stave_drawer import StaveDrawer
from engraver.drawers.barline_drawer import BarlineDrawer
from engraver.drawers.notehead_drawer import NoteheadDrawer
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
    GridBandDrawer,          # Background first
    BarlineDrawer,           # Grid structure
    StaveDrawer,             # Stave lines (vertical per key)
    TimeSignatureDrawer,     # Time signatures
    HeaderFooterDrawer,      # Header/footer
    CountLineDrawer,         # Count guides
    
    NoteheadDrawer,          # Notes (with per-note ledgers)
    BeamDrawer,              # Beams
    ArpeggioDrawer,          # Arpeggios
    GraceNoteDrawer,         # Grace notes
    
    SlurDrawer,              # Slurs
    DynamicDrawer,           # Dynamics and hairpins
    PedalDrawer,             # Pedal symbols
    TextDrawer,              # Text annotations
    
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


def _build_stave_layout_data(score: SCORE, page_lines_map: list[dict], notation_color: tuple[float, float, float, float]) -> dict:
    """Build the minimal layout payload required by StaveDrawer."""
    layout = dict((score or {}).get('layout', {}) or {})
    page_lines: list[dict] = []
    key_to_x_for_line: dict[int, object] = {}
    semitone_mm_default = 1.0

    for line_index, line in enumerate(page_lines_map or []):
        x_start = float(line.get('x_start', 0.0) or 0.0)
        x_end = float(line.get('x_end', x_start) or x_start)
        y_top = float(line.get('y_top', 0.0) or 0.0)
        y_bottom = float(line.get('y_bottom', y_top) or y_top)
        width = max(0.0, x_end - x_start)
        semitone_mm = width / float(max(1, PIANO_KEY_AMOUNT - 1))
        semitone_mm_default = max(1e-6, semitone_mm)

        page_lines.append({
            'stave_visible': True,
            'y_top': y_top,
            'y_bottom': y_bottom,
            'line_x_start': x_start,
            'range': [1, PIANO_KEY_AMOUNT],
            'visible_keys': list(range(1, PIANO_KEY_AMOUNT + 1)),
            'natural_bound_left': 1,
            'natural_bound_right': PIANO_KEY_AMOUNT,
            'low_key_left': False,
            'a0_ledger_mode': False,
        })

        key_to_x_for_line[line_index] = (
            lambda key, _x_start=x_start, _semitone_mm=semitone_mm_default:
            _x_start + ((float(key) - 1.0) * _semitone_mm)
        )

    return {
        'page_lines': page_lines,
        'key_to_x_for_line': key_to_x_for_line,
        'key_positions': {k: float(k - 1) * semitone_mm_default for k in range(1, PIANO_KEY_AMOUNT + 1)},
        'line_keys': list(range(1, PIANO_KEY_AMOUNT + 1)),
        'semitone_mm': semitone_mm_default,
        'layout': layout,
        'notation_color': notation_color,
        'scale': 1.0,
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
    notation_color = (
        notation_rgb[0] / 255.0,
        notation_rgb[1] / 255.0,
        notation_rgb[2] / 255.0,
        1.0,
    )
    if pdf_export:
        notation_color = (0.0, 0.0, 0.0, 1.0)

    print_time_map = list(getattr(scratch_du, 'print_time_map', []) or [])
    for page_index, page_lines_map in enumerate(print_time_map):
        du.set_current_page(int(page_index))
        context = EngravingContext(
            score=score or {},
            normalized={},
            layout_data=_build_stave_layout_data(score or {}, list(page_lines_map or []), notation_color),
            pageno=int(page_index),
            pdf_export=bool(pdf_export),
            du=du,
            drawer_caches={},
        )
        StaveDrawer(context).draw()

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
