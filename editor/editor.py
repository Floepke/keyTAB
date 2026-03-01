from __future__ import annotations
from typing import Literal, Optional, Tuple, Dict, Type, TYPE_CHECKING
import math, bisect
from PySide6 import QtCore

from editor.tool.base_tool import BaseTool
from editor.tool_manager import ToolManager
# Import tool templates
from editor.tool.beam_tool import BeamTool
from editor.tool.count_line_tool import CountLineTool
from editor.tool.end_repeat_tool import EndRepeatTool
from editor.tool.grace_note_tool import GraceNoteTool
from editor.tool.line_break_tool import LineBreakTool
from editor.tool.note_tool import NoteTool
from editor.tool.pedal_tool import PedalTool
from editor.tool.slur_tool import SlurTool
from editor.tool.start_repeat_tool import StartRepeatTool
from editor.tool.text_tool import TextTool
from editor.tool.base_grid_tool import BaseGridTool
from editor.tool.time_signature_tool import TimeSignatureTool
from editor.tool.dynamic_tool import DynamicTool
from editor.tool.crescendo_tool import CrescendoTool
from editor.tool.decrescendo_tool import DecrescendoTool
from editor.tool.tempo_tool import TempoTool
from editor.ctlz import CtlZ
from file_model.base_grid import BaseGrid
from settings_manager import get_preferences_manager
from ui.style import Style
from file_model.SCORE import SCORE
from utils.CONSTANT import BE_KEYS, QUARTER_NOTE_UNIT
from editor.drawers.stave_drawer import StaveDrawerMixin
from editor.drawers.snap_drawer import SnapDrawerMixin
from editor.drawers.grid_drawer import GridDrawerMixin
from editor.drawers.note_drawer import NoteDrawerMixin
from editor.drawers.grace_note_drawer import GraceNoteDrawerMixin
from editor.drawers.beam_drawer import BeamDrawerMixin
from editor.drawers.pedal_drawer import PedalDrawerMixin
from editor.drawers.text_drawer import TextDrawerMixin
from editor.drawers.slur_drawer import SlurDrawerMixin
from editor.drawers.start_repeat_drawer import StartRepeatDrawerMixin
from editor.drawers.end_repeat_drawer import EndRepeatDrawerMixin
from editor.drawers.count_line_drawer import CountLineDrawerMixin
from editor.drawers.line_break_drawer import LineBreakDrawerMixin
from editor.drawers.tempo_drawer import TempoDrawerMixin
from editor.drawers.dynamic_drawer import DynamicDrawerMixin
from editor.drawers.crescendo_drawer import CrescendoDrawerMixin
from editor.drawers.decrescendo_drawer import DecrescendoDrawerMixin
from editor.drawers.time_signature_drawer import TimeSignatureDrawerMixin
from utils.CONSTANT import PIANO_KEY_AMOUNT, BLACK_KEYS
from utils.operator import Operator

if TYPE_CHECKING:
    from editor.tool.base_tool import BaseTool
    from ui.widgets.draw_util import DrawUtil


class Editor(QtCore.QObject,
             StaveDrawerMixin,
             SnapDrawerMixin,
             GridDrawerMixin,
             TimeSignatureDrawerMixin,
             NoteDrawerMixin,
             GraceNoteDrawerMixin,
             BeamDrawerMixin,
             SlurDrawerMixin,
             TextDrawerMixin,
             PedalDrawerMixin,
             DynamicDrawerMixin,
             CrescendoDrawerMixin,
             DecrescendoDrawerMixin,
             StartRepeatDrawerMixin,
             EndRepeatDrawerMixin,
             CountLineDrawerMixin,
             LineBreakDrawerMixin,
             TempoDrawerMixin):
    """Main editor class: routes UI events to the current tool.

    Handles click vs drag classification using a 3px threshold.
    """

    DRAG_THRESHOLD: int = 2

    score_changed = QtCore.Signal()

    def __init__(self, tool_manager: ToolManager):
        super().__init__()
        self._tm = tool_manager
        self._tool: BaseTool = BaseGridTool()  # default tool
        self._ctlz: CtlZ | None = None
        self._file_manager = None
        self._score: SCORE = None
        self._tool_classes: Dict[str, Type[BaseTool]] = {
            'beam': BeamTool,
            'count_line': CountLineTool,
            'end_repeat': EndRepeatTool,
            'grace_note': GraceNoteTool,
            'line_break': LineBreakTool,
            'note': NoteTool,
            'pedal': PedalTool,
            'slur': SlurTool,
            'start_repeat': StartRepeatTool,
            'text': TextTool,
            'base_grid': BaseGridTool,
            'time_signature': TimeSignatureTool,
            'dynamic': DynamicTool,
            'crescendo': CrescendoTool,
            'decrescendo': DecrescendoTool,
            'tempo': TempoTool,
        }
        self._tm.set_tool(self._tool)

        # Press/drag state
        self._left_pressed: bool = False
        self._right_pressed: bool = False
        self._press_pos: Tuple[float, float] = (0.0, 0.0)
        self._dragging_left: bool = False
        self._dragging_right: bool = False

        # layout metrics (mm)
        self.margin: float = None
        self.editor_height: float = None
        self.stave_width: float = None
        self.semitone_dist: float = None

        # colors
        notation_rgb = Style.get_named_rgb("notation", fallback=(0, 0, 0))
        self.notation_color = (
            float(notation_rgb[0]) / 255.0,
            float(notation_rgb[1]) / 255.0,
            float(notation_rgb[2]) / 255.0,
            1.0,
        )
        accent_rgb = Style.get_named_rgb("accent", fallback=(51, 153, 255))
        self.accent_color = (
            float(accent_rgb[0]) / 255.0,
            float(accent_rgb[1]) / 255.0,
            float(accent_rgb[2]) / 255.0,
            1.0,
        )
        self.selection_color = (
            float(accent_rgb[0]) / 255.0,
            float(accent_rgb[1]) / 255.0,
            float(accent_rgb[2]) / 255.0,
            0.3,
        )

        # snap size in time units (default matches SnapSizeSelector: base=8, divide=1 -> 128)
        self.snap_size_units: float = (QUARTER_NOTE_UNIT * 4.0) / 8.0

        # Cache for key x-positions (index by piano key number 1..88)
        self._x_positions: Optional[list[float]] = None

        # View metrics for fast pixel↔mm conversions
        self._px_per_mm: float = 1.0            # device px per mm
        self._widget_px_per_mm: float = 1.0     # logical (Qt) px per mm
        self._dpr: float = 1.0                  # device pixel ratio
        # View offset in mm (top of visible clip)
        self._view_y_mm_offset: float = 0.0
        # Viewport height (mm) of the visible clip
        self._viewport_h_mm: float = 0.0
        # Extra time (ticks) to extend viewport cache at the bottom for beam continuity
        self.viewport_bottom_bleed: float = QUARTER_NOTE_UNIT * 2

        # cursor
        self.time_cursor: Optional[float] = None
        self.mm_cursor: Optional[float] = None
        self.pitch_cursor: Optional[int] = None
        self.hand_cursor: Literal['<', '>'] = '<'  # default hand for cursor overlays
        # show/hide guides depending on mouse over editor
        self.guides_active: bool = True
        # Playhead position (app time units). When set, draws a red line overlay.
        self.playhead_time: Optional[float] = None

        # Debounced snapshot for fast transpose bursts
        self._transpose_timer = QtCore.QTimer(self)
        try:
            self._transpose_timer.setSingleShot(True)
            self._transpose_timer.timeout.connect(self._finalize_transpose_snapshot)
        except Exception:
            pass
        self._pending_snapshot_label: str = 'transpose_notes'

        # Per-frame shared render cache (built at draw_all)
        self._draw_cache: dict | None = None
        # One-shot hint to reuse the current draw cache on the next frame
        self._reuse_draw_cache_once: bool = False
        # Per-frame note hit rectangles in absolute mm coordinates
        self._note_hit_rects: list[dict] = []
        # Per-frame text hit rectangles in absolute mm coordinates
        self._text_hit_rects: list[dict] = []
        # Per-frame tempo hit rectangles in absolute mm coordinates
        self._tempo_hit_rects: list[dict] = []

        # Selection window state (time-based, tool-agnostic)
        self._selection_active: bool = False
        self._sel_start_units: float = 0.0
        self._sel_end_units: float = 0.0
        # Anchor time for selection (absolute ticks, unaffected by scroll)
        self._sel_anchor_units: float = 0.0
        # Pitch-constrained selection (1..88)
        self._sel_min_pitch: int = 1
        self._sel_max_pitch: int = 88
        self._sel_anchor_pitch: int = 1
        # Clipboard for cut/copy/paste of detected events
        self.clipboard: dict | None = None
        # Modifier state
        self._shift_down: bool = False
        self._ctrl_down: bool = False
        # Optional player for auditioning
        self.player = None

    # ---- Drawing via mixins ----
    def draw_background_gray(self, du) -> None:
        """Fill the current page with print-view grey (#7a7a7a)."""
        w_mm, h_mm = du.current_page_size_mm()
        grey = (200, 240, 240, 1.0)
        du.add_rectangle(0.0, 0.0, w_mm, h_mm, stroke_color=None, fill_color=grey, id=0, tags=["background"])

    def draw_all(self, du) -> None:
        """Invoke drawer mixin methods; layer order is enforced by DrawUtil tags.

        We simply call all drawer methods; DrawUtil sorts items by tag layering.
        """
        # Reset hit rectangles for this frame; drawers will register rectangles
        self._note_hit_rects = []
        self._text_hit_rects = []
        self._tempo_hit_rects = []
        
        # Build shared render cache for this draw pass (fresh each frame)
        self._build_render_cache()
        
        # Call drawer mixin methods in order
        methods = [
            getattr(self, 'draw_snap', None),
            getattr(self, 'draw_grid', None),
            getattr(self, 'draw_time_signature', None),
            getattr(self, 'draw_stave', None),
            getattr(self, 'draw_note', None),
            getattr(self, 'draw_grace_note', None),
            getattr(self, 'draw_beam', None),
            getattr(self, 'draw_pedal', None),
            getattr(self, 'draw_dynamic', None),
            getattr(self, 'draw_crescendo', None),
            getattr(self, 'draw_decrescendo', None),
            getattr(self, 'draw_text', None),
            getattr(self, 'draw_slur', None),
            getattr(self, 'draw_start_repeat', None),
            getattr(self, 'draw_end_repeat', None),
            getattr(self, 'draw_count_line', None),
            getattr(self, 'draw_tempo', None),
            getattr(self, 'draw_line_break', None),
        ]
        for fn in methods:
            if callable(fn):
                fn(du)

        # Keep render cache available for hit detection until next frame rebuild
        # (cleared at the start of _build_render_cache)

    def draw_frame(self) -> None:
        """Build a full frame immediately (cache + drawer registration) without painting.

        Creates a temporary DrawUtil using current layout page size, calls draw_all.
        Useful for immediate feedback from tools (e.g., updating hit rects/cache) before
        the widget triggers a repaint.
        """
        try:
            from ui.widgets.draw_util import DrawUtil
        except Exception:
            DrawUtil = None  # type: ignore
        du = DrawUtil() if DrawUtil is not None else None
        if du is not None:
            # Derive page size from SCORE layout; fall back to A4
            w_mm = 210.0
            h_mm = 297.0
            sc = self.current_score()
            if sc is not None:
                lay = getattr(sc, 'layout', None)
                if lay is not None:
                    try:
                        w_mm = float(getattr(lay, 'page_width_mm', w_mm) or w_mm)
                        h_mm = float(getattr(lay, 'page_height_mm', h_mm) or h_mm)
                    except Exception:
                        pass
            try:
                du.set_current_page_size_mm(w_mm, h_mm)
            except Exception:
                pass
        # Run the drawer pipeline to rebuild caches and register hit rectangles
        try:
            self.draw_all(du)
        except Exception as exc:
            # As a fallback, still attempt to rebuild render cache
            print("Editor.draw_frame: draw_all failed, attempting cache rebuild")
            try:
                import traceback
                traceback.print_exc()
            except Exception:
                print(f"Editor.draw_frame: draw_all exception: {exc}")
            try:
                self._build_render_cache()
            except Exception as exc2:
                try:
                    import traceback
                    traceback.print_exc()
                except Exception:
                    print(f"Editor.draw_frame: cache rebuild also failed: {exc2}")
        
        # refresh overlay guides if applicable
        self.draw_guides(du)

    def force_redraw_from_model(self) -> None:
        """Rebuild caches from SCORE and request a full widget repaint."""
        try:
            self.draw_frame()
        except Exception:
            pass
        try:
            w = getattr(self, 'widget', None)
            if w is not None and hasattr(w, 'force_full_redraw'):
                w.force_full_redraw()
            elif w is not None and hasattr(w, 'update'):
                w.update()
        except Exception:
            pass

    def _queue_transpose_snapshot(self, delay_ms: int = 200, label: str = 'transpose_notes') -> None:
        """Debounce transpose snapshots to avoid heavy work on every keypress."""
        try:
            if self._transpose_timer.isActive():
                self._transpose_timer.stop()
            self._pending_snapshot_label = str(label or 'transpose_notes')
            self._transpose_timer.start(int(max(1, delay_ms)))
        except Exception:
            # Fallback: snapshot immediately if timer is unavailable
            try:
                self._pending_snapshot_label = str(label or 'transpose_notes')
                self._finalize_transpose_snapshot()
            except Exception:
                pass

    def _finalize_transpose_snapshot(self) -> None:
        try:
            label = getattr(self, '_pending_snapshot_label', 'transpose_notes')
            self._snapshot_if_changed(coalesce=True, label=str(label or 'transpose_notes'))
        except Exception:
            pass

    # ---- Hit rectangles (notes) ----
    def register_note_hit_rect(self, note_id: int, x_left_mm: float, y_top_mm: float, x_right_mm: float, y_bottom_mm: float) -> None:
        """Register a clickable rectangle for a note in absolute mm coordinates.

        Rectangles may overlap; hit test will select the one closest to the rectangle center.
        """
        try:
            cx = (float(x_left_mm) + float(x_right_mm)) * 0.5
            cy = (float(y_top_mm) + float(y_bottom_mm)) * 0.5
            self._note_hit_rects.append({
                '_id': int(note_id),
                'x1': float(x_left_mm),
                'y1': float(y_top_mm),
                'x2': float(x_right_mm),
                'y2': float(y_bottom_mm),
                'cx': cx,
                'cy': cy,
            })
        except Exception:
            pass

    # ---- Hit rectangles (text) ----
    def register_text_hit_rect(self, text_id: int, x_left_mm: float, y_top_mm: float, x_right_mm: float, y_bottom_mm: float, kind: str = 'body') -> None:
        """Register a clickable rectangle for a text element (body or handle).

        kind: 'body' or 'handle' to allow prioritizing handle hits.
        """
        try:
            cx = (float(x_left_mm) + float(x_right_mm)) * 0.5
            cy = (float(y_top_mm) + float(y_bottom_mm)) * 0.5
            self._text_hit_rects.append({
                '_id': int(text_id),
                'x1': float(x_left_mm),
                'y1': float(y_top_mm),
                'x2': float(x_right_mm),
                'y2': float(y_bottom_mm),
                'cx': cx,
                'cy': cy,
                'kind': str(kind or 'body'),
            })
        except Exception:
            pass

    # ---- Hit rectangles (tempo) ----
    def register_tempo_hit_rect(self, tempo_id: int, x_left_mm: float, y_top_mm: float, x_right_mm: float, y_bottom_mm: float) -> None:
        """Register a clickable rectangle for a tempo marker in absolute mm coordinates."""
        try:
            cx = (float(x_left_mm) + float(x_right_mm)) * 0.5
            cy = (float(y_top_mm) + float(y_bottom_mm)) * 0.5
            self._tempo_hit_rects.append({
                '_id': int(tempo_id),
                'x1': float(x_left_mm),
                'y1': float(y_top_mm),
                'x2': float(x_right_mm),
                'y2': float(y_bottom_mm),
                'cx': cx,
                'cy': cy,
            })
        except Exception:
            pass

    def _hit_test_text_internal(self, x_mm: float, y_mm: float):
        candidates = []
        for r in (self._text_hit_rects or []):
            if float(r['x1']) <= x_mm <= float(r['x2']) and float(r['y1']) <= y_mm <= float(r['y2']):
                area = max(0.0, (float(r['x2']) - float(r['x1'])) * (float(r['y2']) - float(r['y1'])))
                kind = str(r.get('kind', 'body'))
                priority = 0 if kind == 'handle' else 1
                candidates.append((priority, area, int(r['_id']), kind, r))
        if not candidates:
            return (None, None, None)
        candidates.sort(key=lambda t: (t[0], t[1]))
        _p, _a, tid, kind, rect = candidates[0]
        return (tid, kind == 'handle', rect)

    def hit_test_text(self, x_px: float, y_px: float):
        """Return (text_id, is_handle, rect) containing the point.

        Expects logical px coordinates; converts to absolute mm before testing.
        Returns (None, None, None) if no hit.
        """
        try:
            w_px_per_mm = float(getattr(self, '_widget_px_per_mm', 1.0) or 1.0)
            if w_px_per_mm <= 0:
                return (None, None, None)
            x_mm = float(x_px) / w_px_per_mm
            y_mm_local = float(y_px) / w_px_per_mm
            y_mm = y_mm_local + float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
            return self._hit_test_text_internal(x_mm, y_mm)
        except Exception:
            return (None, None, None)

    def hit_test_text_mm(self, x_mm: float, y_mm: float):
        """Return (text_id, is_handle, rect) for absolute mm coordinates."""
        try:
            return self._hit_test_text_internal(float(x_mm), float(y_mm))
        except Exception:
            return (None, None, None)

    def hit_test_tempo(self, x_px: float, y_px: float) -> int | None:
        """Return the tempo id whose registered rectangle contains the mouse point."""
        try:
            w_px_per_mm = float(getattr(self, '_widget_px_per_mm', 1.0) or 1.0)
            if w_px_per_mm <= 0:
                return None
            x_mm = float(x_px) / w_px_per_mm
            y_mm_local = float(y_px) / w_px_per_mm
            y_mm = y_mm_local + float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
            matches = []
            for r in (self._tempo_hit_rects or []):
                if float(r['x1']) <= x_mm <= float(r['x2']) and float(r['y1']) <= y_mm <= float(r['y2']):
                    dx = x_mm - float(r['cx'])
                    dy = y_mm - float(r['cy'])
                    dist2 = dx * dx + dy * dy
                    matches.append((dist2, int(r['_id'])))
            if not matches:
                return None
            matches.sort(key=lambda t: t[0])
            return matches[0][1]
        except Exception:
            return None

    def hit_test_note_id(self, x_px: float, y_px: float) -> int | None:
        """Return the note id whose registered rectangle contains the mouse point.

        - Coordinates x_px, y_px are logical (Qt) pixels.
        - Converts to absolute mm using editor metrics and viewport offset.
        - If multiple rectangles contain the point, returns the one with center closest to the point.
        - Returns None if no rectangle contains the point.
        """
        try:
            w_px_per_mm = float(getattr(self, '_widget_px_per_mm', 1.0) or 1.0)
            if w_px_per_mm <= 0:
                return None
            # Convert logical px to local mm, then to absolute mm by adding viewport offset
            x_mm = float(x_px) / w_px_per_mm
            y_mm_local = float(y_px) / w_px_per_mm
            y_mm = y_mm_local + float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
            # Find all rectangles containing the point
            matches = []
            for r in (self._note_hit_rects or []):
                if float(r['x1']) <= x_mm <= float(r['x2']) and float(r['y1']) <= y_mm <= float(r['y2']):
                    dx = x_mm - float(r['cx'])
                    dy = y_mm - float(r['cy'])
                    dist2 = dx * dx + dy * dy
                    matches.append((dist2, int(r['_id'])))
            if not matches:
                return None
            matches.sort(key=lambda t: t[0])
            return matches[0][1]
        except Exception:
            return None

    def _calculate_layout(self, view_width_mm: float) -> None:
        """Compute editor-specific layout based on the current view width.

        - margin: 1/6 of the width
        - stave width: width - 2 * margin
        - semitone spacing: stave width / (PIANO_KEY_AMOUNT - 1)
        """
        # Calculate margin
        self.margin = view_width_mm / 6
        
        # Calculate stave units
        visual_semitone_spaces = 101
        self.stave_width = view_width_mm - (2 * self.margin)
        self.semitone_dist = self.stave_width / visual_semitone_spaces
        
        # Ensure editor_height reflects the current SCORE content height in mm
        self.editor_height = self._calc_editor_height()

        # Rebuild cached x positions
        self._rebuild_x_positions()

    # ---- Note lookup ----
    def get_note_by_id(self, note_id: int):
        """Return the note event for id, preferring current viewport cache.

        Falls back to scanning all notes if cache is unavailable.
        """
        try:
            nid = int(note_id)
        except Exception:
            return None
        # Prefer notes in the current viewport draw cache
        try:
            cache = getattr(self, '_draw_cache', None) or {}
            notes_view = cache.get('notes_view') or []
            if notes_view:
                for n in notes_view:
                    if int(getattr(n, '_id', -1) or -1) == nid:
                        return n
        except Exception:
            pass
        # Fallback: global scan
        try:
            score: SCORE | None = self.current_score()
            if score is None:
                return None
            for n in getattr(score.events, 'note', []) or []:
                if int(getattr(n, '_id', -1) or -1) == nid:
                    return n
        except Exception:
            pass
        return None

    def get_measure_index_for_time(self, ticks: float) -> int:
        """Return 1-based measure index for a given time in ticks.

        Uses barline start positions across the score and finds the last
        barline at or before `ticks`. If no barline is at or before, returns 1.
        """
        try:
            bars = self._get_barline_positions()
            if not bars:
                return 1
            # bisect_right gives insertion point after any equal entries
            i = bisect.bisect_right(bars, float(ticks)) - 1
            return max(1, int(i + 1))
        except Exception:
            return 1

    def _rebuild_x_positions(self) -> None:
        """Precompute x positions for keys 1..PIANO_KEY_AMOUNT with BE gap after B/E."""
        be_set = set(BE_KEYS)
        x_pos = self.margin - self.semitone_dist
        xs = [x_pos]

        for n in range(1, PIANO_KEY_AMOUNT + 1):
            # Apply extra gap AFTER B/E, i.e., when stepping from (n-1) -> n
            if (n - 1) in be_set:
                x_pos += self.semitone_dist
            # Normal semitone step
            x_pos += self.semitone_dist
            xs.append(x_pos)

        self._x_positions = xs

    def set_tool_by_name(self, name: str) -> None:
        cls = self._tool_classes.get(name)
        if cls is None:
            return
        self._tool = cls()
        self._tm.set_tool(self._tool)

    def set_player(self, player) -> None:
        self.player = player

    def set_score(self, score):
        # Set an explicit score model when not using FileManager
        self._score = score

    # Model provider for undo snapshots
    def set_file_manager(self, fm) -> None:
        """Provide FileManager so we can snapshot/restore SCORE for undo/redo."""
        self._file_manager = fm
        # Initialize ctlz with the initial model state
        if self._file_manager is not None:
            try:
                self._ctlz = CtlZ(self._file_manager)
                self._ctlz.reset_ctlz()
            except Exception:
                self._ctlz = None

    def current_score(self) -> SCORE:
        """Return the current SCORE: prefer FileManager; fall back to explicit _score."""
        if self._file_manager is not None:
            return self._file_manager.current()
        return getattr(self, "_score", None)

    def _snapshot_if_changed(self, coalesce: bool = False, label: str = "") -> None:
        if self._file_manager is None:
            return
        # Use dict-based ctlz snapshots
        try:
            if self._ctlz is not None:
                self._ctlz.add_ctlz()
        except Exception:
            pass
        # Notify FileManager so it can handle autosave/session saving/dirty state
        try:
            if hasattr(self._file_manager, 'on_model_changed'):
                self._file_manager.on_model_changed()
            else:
                # Fallback to legacy behavior
                self._file_manager.autosave_current()
                self._file_manager.mark_dirty()
        except Exception:
            pass
        # Ensure any edit is reflected immediately from the model
        try:
            self.force_redraw_from_model()
        except Exception:
            pass
        try:
            self.score_changed.emit()
        except Exception:
            pass

    # Public undo/redo (optional consumers can bind Ctrl+Z / Ctrl+Shift+Z)
    def undo(self) -> None:
        if self._file_manager is None:
            return
        snap = None
        try:
            if self._ctlz is not None:
                snap = self._ctlz.undo()
        except Exception:
            snap = None
        if snap is not None:
            self._file_manager.replace_current(snap)
            try:
                self._file_manager.mark_dirty()
            except Exception:
                pass
            try:
                self.score_changed.emit()
            except Exception:
                pass

    def redo(self) -> None:
        if self._file_manager is None:
            return
        snap = None
        try:
            if self._ctlz is not None:
                snap = self._ctlz.redo()
        except Exception:
            snap = None
        if snap is not None:
            self._file_manager.replace_current(snap)
            try:
                self._file_manager.mark_dirty()
            except Exception:
                pass
            try:
                self.score_changed.emit()
            except Exception:
                pass

    def reset_undo_stack(self) -> None:
        try:
            if self._ctlz is not None:
                self._ctlz.reset_ctlz()
        except Exception:
            pass

    '''
        ---- Mouse event routing ----
    '''
    def mouse_press(self, button: int, x: float, y: float) -> None:
        if button == 1:
            self._left_pressed = True
            self._dragging_left = False
            self._press_pos = (x, y)
            if not self._shift_down:
                self._tool.on_left_press(x, y)
            # If Shift is held, initialize selection anchor on left press
            if self._shift_down:
                try:
                    anchor_t = self.snap_time(self.y_to_time(y))
                    self._sel_anchor_units = float(anchor_t)
                    self._sel_start_units = float(anchor_t)
                    # Initialize end to one snap band ahead to avoid zero-length selection
                    ss = max(1e-6, float(self.snap_size_units))
                    self._sel_end_units = float(anchor_t + ss)
                    # Initialize pitch anchors and range on Shift+Left press
                    anchor_p = int(self.x_to_pitch(x))
                    anchor_p = max(1, min(88, anchor_p))
                    self._sel_anchor_pitch = anchor_p
                    self._sel_min_pitch = anchor_p
                    self._sel_max_pitch = anchor_p
                    self._selection_active = True
                except Exception:
                    pass
        elif button == 2:
            self._right_pressed = True
            self._dragging_right = False
            self._press_pos = (x, y)
            self._tool.on_right_press(x, y)
            # Initialize selection anchor at press to be robust against scrolling
            try:
                anchor_t = self.snap_time(self.y_to_time(y))
                self._sel_anchor_units = float(anchor_t)
                self._sel_start_units = float(anchor_t)
                # Initialize end to one snap band ahead to avoid zero-length selection
                ss = max(1e-6, float(self.snap_size_units))
                self._sel_end_units = float(anchor_t + ss)
                # Initialize pitch anchors and range on Right press (selection)
                anchor_p = int(self.x_to_pitch(x))
                anchor_p = max(1, min(88, anchor_p))
                self._sel_anchor_pitch = anchor_p
                self._sel_min_pitch = anchor_p
                self._sel_max_pitch = anchor_p
                self._selection_active = True
            except Exception:
                pass

    def mouse_move(self, x: float, y: float, dx: float, dy: float) -> None:
        if self._left_pressed:
            if not self._dragging_left and (abs(dx) > self.DRAG_THRESHOLD or abs(dy) > self.DRAG_THRESHOLD):
                self._dragging_left = True
                if not self._shift_down:
                    self._tool.on_left_drag_start(x, y)
            if self._dragging_left:
                if not self._shift_down:
                    self._tool.on_left_drag(x, y, dx, dy)
                # Update selection window when Shift+Left-dragging
                if self._shift_down:
                    try:
                        cur_t = self.snap_time(self.y_to_time(y))
                        anchor_t = float(self._sel_anchor_units)
                        ss = max(1e-6, float(self.snap_size_units))
                        if cur_t >= anchor_t:
                            # Downwards selection: start at anchor, end at current band end
                            self._sel_start_units = float(anchor_t)
                            self._sel_end_units = float(cur_t + ss)
                        else:
                            # Upwards selection: start at current band start, end at anchor
                            self._sel_start_units = float(cur_t)
                            self._sel_end_units = float(anchor_t)
                        # Update pitch range based on drag X
                        cur_p = int(self.x_to_pitch(x))
                        cur_p = max(1, min(88, cur_p))
                        anchor_p = int(self._sel_anchor_pitch)
                        self._sel_min_pitch = int(min(anchor_p, cur_p))
                        self._sel_max_pitch = int(max(anchor_p, cur_p))
                        self._selection_active = True
                    except Exception:
                        pass
                # Do not capture multiple intermediate drag snapshots
        elif self._right_pressed:
            if not self._dragging_right and (abs(dx) > self.DRAG_THRESHOLD or abs(dy) > self.DRAG_THRESHOLD):
                self._dragging_right = True
                self._tool.on_right_drag_start(x, y)
            if self._dragging_right:
                self._tool.on_right_drag(x, y, dx, dy)
                # Update selection window while right-dragging (tool-agnostic)
                try:
                    cur_t = self.snap_time(self.y_to_time(y))
                    anchor_t = float(self._sel_anchor_units)
                    ss = max(1e-6, float(self.snap_size_units))
                    if cur_t >= anchor_t:
                        # Downwards selection: start at anchor, end at current band end
                        self._sel_start_units = float(anchor_t)
                        self._sel_end_units = float(cur_t + ss)
                    else:
                        # Upwards selection: start at current band start, end at anchor
                        self._sel_start_units = float(cur_t)
                        self._sel_end_units = float(anchor_t)
                    # Update pitch range for right-drag selection
                    cur_p = int(self.x_to_pitch(x))
                    cur_p = max(1, min(88, cur_p))
                    anchor_p = int(self._sel_anchor_pitch)
                    self._sel_min_pitch = int(min(anchor_p, cur_p))
                    self._sel_max_pitch = int(max(anchor_p, cur_p))
                    self._selection_active = True
                except Exception:
                    pass
                # Skip intermediate drag snapshots
        else:
            # Update shared cursor state for guide rendering (time + mm), with snapping
            t = self.y_to_time(y)
            t = self.snap_time(t)
            self.time_cursor = t
            # Store cursor mm relative to viewport (local mm)
            abs_mm = self.time_to_mm(float(t))
            self.mm_cursor = abs_mm - float(self._view_y_mm_offset or 0.0)
            # Also track pitch under cursor (logical px → key number)
            self.pitch_cursor = self.x_to_pitch(x)
            self._tool.on_mouse_move(x, y)

    def mouse_release(self, button: int, x: float, y: float) -> None:
        if button == 1:
            if self._dragging_left:
                if not self._shift_down:
                    self._tool.on_left_drag_end(x, y)
                    # Capture a single coalesced snapshot for the whole drag
                    self._snapshot_if_changed(coalesce=True, label="left_drag")
            else:
                # Click if moved <= threshold
                px, py = self._press_pos
                if (abs(x - px) <= self.DRAG_THRESHOLD and abs(y - py) <= self.DRAG_THRESHOLD):
                    if not self._shift_down:
                        self._tool.on_left_click(x, y)
                        # Capture click changes (non-coalesced)
                        self._snapshot_if_changed(coalesce=False, label="left_click")
            # Stop drawing selection on any click
            if not self._dragging_left and not self._shift_down:
                self._selection_active = False
            if not self._shift_down:
                self._tool.on_left_unpress(x, y)
            self._left_pressed = False
            self._dragging_left = False
        elif button == 2:
            if self._dragging_right:
                self._tool.on_right_drag_end(x, y)
                if bool(getattr(self._tool, 'RIGHT_DRAG_EDITS', False)):
                    self._snapshot_if_changed(coalesce=True, label="right_drag")
                # Do not modify clipboard on selection changes
            else:
                px, py = self._press_pos
                if (abs(x - px) <= self.DRAG_THRESHOLD and abs(y - py) <= self.DRAG_THRESHOLD):
                    self._tool.on_right_click(x, y)
                self._snapshot_if_changed(coalesce=False, label="right_click")
            # Stop drawing selection on any click
            if not self._dragging_right:
                self._selection_active = False
            self._tool.on_right_unpress(x, y)
            self._right_pressed = False
            self._dragging_right = False

    def mouse_double_click(self, button: int, x: float, y: float) -> None:
        if button == 1:
            if not self._shift_down:
                self._tool.on_left_double_click(x, y)
        elif button == 2:
            self._tool.on_right_double_click(x, y)

    '''
        ---- Editor drawer mixin helper methods ----
    '''
    def _calc_base_grid_list_total_length(self) -> int:
        """Return the total length of the current SCORE in ticks."""
        score: SCORE = self.current_score()
        
        length_ticks = 0
        for bg in score.base_grid:
            base_grid_length = bg.numerator * (4.0 / bg.denominator) * bg.measure_amount * QUARTER_NOTE_UNIT
            length_ticks += base_grid_length
        return length_ticks
    
    def _calc_editor_height(self) -> float:
        """Calculate the total height of the editor content in mm.

        Height is based on the total score time scaled by the editor zoom, plus
        top/bottom spacing using the editor's margin value. This ensures drawers
        can rely on `self.editor_height` for vertical layout and that DrawUtil
        uses a matching page height for scrolling.
        """
        total_time_ticks = float(self._calc_base_grid_list_total_length())
        score: SCORE | None = self.current_score()
        zpq: float = score.editor.zoom_mm_per_quarter
        stave_length_mm = (total_time_ticks / float(QUARTER_NOTE_UNIT)) * zpq
        top_bottom_mm = float(self.margin or 0.0) * 6.0
        height_mm = max(10.0, stave_length_mm + top_bottom_mm)
        return height_mm

    def update_score_length(self) -> None:
        """Ensure measures cover all music exactly, adding or trimming as needed.

        - Finds the furthest note end (time + duration).
        - If the total measure length is shorter, extend the last segment.
        - If longer, trim trailing measures/segments so total just covers the music.

        Never creates zero-measure segments; always keeps at least one measure.
        """
        # get data
        score: SCORE | None = self.current_score()
        if score is None or not getattr(score, 'base_grid', None):
            return

        # Furthest musical time (notes: end time; grace: start time only)
        furthest_end = 0.0
        grace_times: list[float] = []
        for n in getattr(score.events, 'note', []) or []:
            t = float(getattr(n, 'time', 0.0) or 0.0)
            dur = float(getattr(n, 'duration', 0.0) or 0.0)
            furthest_end = max(furthest_end, t + dur)
        for g in getattr(score.events, 'grace_note', []) or []:
            t = float(getattr(g, 'time', 0.0) or 0.0)
            grace_times.append(t)
            furthest_end = max(furthest_end, t)
        
        # Current score end (sum of segment lengths)
        base_grid_total_length = float(self._calc_base_grid_list_total_length())
        bg_list = score.base_grid
        
        # update only last BaseGrid
        last_bg: BaseGrid = bg_list[-1]
        num = last_bg.numerator
        den = last_bg.denominator
        measure_len = num * (4.0 / den) * float(QUARTER_NOTE_UNIT)
        current_end = base_grid_total_length

        # If any grace starts on/after the current end barline, force one more measure.
        grace_hits_barline = any(gt >= current_end - 1e-6 for gt in grace_times)
        if grace_hits_barline:
            furthest_end = max(furthest_end, current_end + measure_len)

        # extend last segment
        needed_length = furthest_end - current_end
        needed_measures = int((needed_length + measure_len - 1) // measure_len)
        last_bg.measure_amount += needed_measures
        
        # reset to 1 measure if < 1 to prevent zero-measure segments
        if last_bg.measure_amount < 1:
            last_bg.measure_amount = 1
        return

    # ---- Shared render cache ----
    def _build_render_cache(self) -> None:
        """Build per-frame cached, time-sorted viewport data for drawers.

        Cache includes:
        - viewport times (begin/end)
        - Operator(7) comparator
        - notes sorted, starts, ends
        - candidate indices ensuring notes with visible ends or spanning viewport are included
        - notes_view slice and notes grouped by hand
        - beam markers by hand (if available) and grid helpers (for future beam grouping)
        """
        # Optionally reuse the existing cache once (for fast edits like transpose)
        if self._reuse_draw_cache_once and self._draw_cache is not None:
            self._reuse_draw_cache_once = False
            return
        # Clear previous cache at start so callers don't read stale data
        self._draw_cache = None
        score: SCORE | None = self.current_score()
        if score is None:
            self._draw_cache = None
            return

        # Compute viewport times (ticks) from mm offset and height
        try:
            top_mm = float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
            vp_h_mm = float(getattr(self, '_viewport_h_mm', 0.0) or 0.0)
            bottom_mm = top_mm + vp_h_mm
        except Exception:
            top_mm = 0.0
            bottom_mm = 0.0
        # Small bleed similar to prior behavior
        zpq = float(score.editor.zoom_mm_per_quarter)
        bleed_mm = max(2.0, zpq * 0.25)
        time_begin = float(self.mm_to_time(top_mm - bleed_mm))
        bottom_bleed = max(0.0, float(getattr(self, 'viewport_bottom_bleed', 0.0) or 0.0))
        time_end = float(self.mm_to_time(bottom_mm + bleed_mm)) + bottom_bleed

        # Comparator with threshold of 7 ticks
        op = Operator(7)

        # Notes sorted by (time, pitch)
        notes = list(getattr(score.events, 'note', []) or [])
        notes_sorted = sorted(notes, key=lambda n: (float(n.time), int(n.pitch)))
        starts = [float(n.time) for n in notes_sorted]
        ends = [float(n.time + n.duration) for n in notes_sorted]

        # Candidate indices: include
        # 1) notes with start in [time_begin, time_end]
        # 2) notes with end in   [time_begin, time_end] (requires ends sorted by value)
        # 3) notes spanning the viewport entirely: start <= time_begin and end >= time_end
        # 4) a small back expansion to catch near-viewport starters (previous behavior)
        lo_start = bisect.bisect_left(starts, time_begin)
        hi_start = bisect.bisect_right(starts, time_end)

        # Build ends sorted by value with original indices for correct bisecting
        end_pairs = sorted(((ends[i], i) for i in range(len(ends))), key=lambda p: p[0])
        end_values = [p[0] for p in end_pairs]
        lo_end_val = bisect.bisect_left(end_values, time_begin)
        hi_end_val = bisect.bisect_right(end_values, time_end)
        by_end_indices = [end_pairs[j][1] for j in range(lo_end_val, hi_end_val)]

        viewport_len = float(max(0.0, time_end - time_begin))
        slack = float(op.threshold)
        back_lo = bisect.bisect_left(starts, float(time_begin - viewport_len - slack))

        # Spanning notes: those that started before the viewport and end after it
        span_cut = bisect.bisect_right(starts, time_begin)
        span_indices = [i for i in range(max(0, span_cut)) if ends[i] >= time_end]

        candidate_idx_set = (
            set(range(back_lo, hi_start)) | set(by_end_indices) | set(span_indices)
        )
        candidate_indices = sorted(candidate_idx_set)

        # Filtered view: will be further intersection-tested by drawers
        notes_view = [notes_sorted[i] for i in candidate_indices] if candidate_indices else []

        # Group by hand for convenience
        notes_by_hand: dict[str, list] = {}
        for m in notes_view:
            h = str(getattr(m, 'hand', '<'))
            notes_by_hand.setdefault(h, []).append(m)

        # Beam markers (optional; future use)
        beam_markers = list(getattr(score.events, 'beam', []) or [])
        beam_by_hand: dict[str, list] = {}
        for b in beam_markers:
            h = str(getattr(b, 'hand', '<'))
            beam_by_hand.setdefault(h, []).append(b)
        for h in beam_by_hand:
            beam_by_hand[h] = sorted(beam_by_hand[h], key=lambda b: float(getattr(b, 'time', 0.0)))

        # Grid helpers: absolute times (ticks) of drawn grid lines across the score following specific conditions
        grid_den_times: list[float] = []
        barline_times: list[float] = []
        cur_t = 0.0
        for bg in getattr(score, 'base_grid', []) or []:
            # Total measure length in ticks
            measure_len_ticks = float(bg.numerator) * (4.0 / float(bg.denominator)) * float(QUARTER_NOTE_UNIT)
            
            # Beat length inside this measure (ticks)
            beat_len_ticks = measure_len_ticks / max(1, int(bg.numerator))
            
            # For each measure in this segment
            for _ in range(int(bg.measure_amount)):
                barline_times.append(float(cur_t))
                # Append grid line times for configured grid positions
                positions = getattr(bg, 'beat_grouping', None)
                positions_list = list(positions if positions is not None else (getattr(bg, 'beat_grouping', []) or []))
                
                # New scheme: list index represents beat; draw lines where value == 1
                if len(positions_list) == int(bg.numerator):
                    if (positions_list == [v for v in range(1, int(bg.numerator) + 1)]):
                        # we have one group so we draw all beats
                        for idx in range(1, int(bg.numerator) + 1):
                            t_line = cur_t + (idx - 1) * beat_len_ticks
                            grid_den_times.append(float(t_line))
                    else:
                        # we have multiple groups so only draw first beat of each group e.g. 7/8 and 1231234 draws beat 1 and 4
                        for idx, val in enumerate(positions_list, start=1):
                            if int(val) != 1:
                                continue
                            t_line = cur_t + (idx - 1) * beat_len_ticks
                            grid_den_times.append(float(t_line))
                else:
                    # Fallback: if malformed, at least draw the barline
                    grid_den_times.append(float(cur_t))
                
                # Advance to next measure start
                cur_t += measure_len_ticks
        
        # Append final end barline time for completeness
        grid_den_times.append(float(cur_t))
        barline_times.append(float(cur_t))

        self._draw_cache = {
            'time_begin': time_begin,
            'time_end': time_end,
            'op': op,
            'notes_sorted': notes_sorted,
            'starts': starts,
            'ends': ends,
            'candidate_indices': candidate_indices,
            'notes_view': notes_view,
            'notes_by_hand': notes_by_hand,
            'beam_by_hand': beam_by_hand,
            'grid_den_times': grid_den_times,
            'barline_times': barline_times,
        }

    # ---- External controls ----
    def set_snap_size_units(self, units: float) -> None:
        try:
            self.snap_size_units = max(0.0, float(units))
        except Exception:
            pass

    # ---- coordinate calculations ----
    def time_to_mm(self, time: float) -> float:
        """Convert time in ticks to mm position."""
        score: SCORE = self.current_score()
        # Ensure layout metrics are initialized
        if self.margin is None:
            try:
                lay = getattr(score, 'layout', None)
                w_mm = float(getattr(lay, 'page_width_mm', 210.0) or 210.0) if lay is not None else 210.0
            except Exception:
                w_mm = 210.0
            try:
                self._calculate_layout(float(w_mm))
            except Exception:
                pass
        # Layout metrics
        zpq = float(score.editor.zoom_mm_per_quarter)
        return float(self.margin or 0.0) + (float(time) / float(QUARTER_NOTE_UNIT)) * zpq
    
    def pitch_to_x(self, key_number: int) -> float:
        '''Convert piano key number (1-88) to X position using specific Klavarskribo spacing.'''
        # Validate key number
        if key_number < 1 or key_number > PIANO_KEY_AMOUNT:
            return 0.0
        
        # Ensure x-positions cache is built
        if self._x_positions is None:
            self._rebuild_x_positions()
        
        # Return cached x position
        return self._x_positions[key_number]

    def relative_c4pitch_to_x(self, c4_semitone_offset: int) -> float:
        """Convert a semitone offset relative to C4 (key 40) into an X position in mm.

        - Uses the editor's Klavarskribo spacing (`self.semitone_dist`).
        - Positive offsets move to the right; negative to the left.
        - Used for slur handles and text element positions.
        """
        base_x = float(self.pitch_to_x(40))
        dist = float(self.semitone_dist or 0.0)
        offset = int(c4_semitone_offset)
        return base_x + dist * offset

    # ---- Mouse-friendly wrappers (pixels) ----
    def time_to_y(self, ticks: float) -> float:
        """Convert time in ticks to Y position in logical (Qt) pixels."""
        mm = self.time_to_mm(ticks)
        return float(mm) * float(getattr(self, '_widget_px_per_mm', 1.0))

    def y_to_time(self, y_px: float) -> float:
        """Convert Y position in logical (Qt) pixels to time in ticks."""
        return self.px_to_time(y_px)

    def x_to_pitch(self, x_px: float) -> int:
        """Convert X position in logical (Qt) pixels to piano key number (1..88)."""
        return self.x_to_pitch_px(x_px)

    def x_to_pitch_mm(self, x_mm: float) -> int:
        """Inverse of pitch_to_x: map X in mm to nearest piano key number (1..88)."""
        import bisect
        if self._x_positions is None:
            self._rebuild_x_positions()
        xs = self._x_positions
        if x_mm <= xs[1]:
            return 1
        if x_mm >= xs[PIANO_KEY_AMOUNT]:
            return PIANO_KEY_AMOUNT
        i = bisect.bisect_left(xs, x_mm, 1, PIANO_KEY_AMOUNT + 1)
        prev_i = max(1, i - 1)
        if i > PIANO_KEY_AMOUNT:
            return prev_i
        prev_x = xs[prev_i]
        curr_x = xs[i]
        return prev_i if abs(x_mm - prev_x) <= abs(x_mm - curr_x) else i

    def x_to_pitch_px(self, x_px: float) -> int:
        """Map X in logical (Qt) pixels to piano key number using cached widget px/mm."""
        x_mm = float(x_px) / max(1e-6, self._widget_px_per_mm)
        return self.x_to_pitch_mm(x_mm)

    def mm_to_time(self, y_mm: float) -> float:
        """Convert Y in mm to time ticks (inverse of time_to_mm)."""
        score: SCORE = self.current_score()
        # Ensure layout metrics are initialized
        if self.margin is None:
            try:
                lay = getattr(score, 'layout', None)
                w_mm = float(getattr(lay, 'page_width_mm', 210.0) or 210.0) if lay is not None else 210.0
            except Exception:
                w_mm = 210.0
            try:
                self._calculate_layout(float(w_mm))
            except Exception:
                pass
        zpq = float(score.editor.zoom_mm_per_quarter)
        ticks = (float(y_mm) - float(self.margin or 0.0)) / max(1e-6, zpq) * float(QUARTER_NOTE_UNIT)
        return max(0.0, ticks)

    def px_to_time(self, y_px: float) -> float:
        """Convert Y in logical (Qt) pixels to time ticks efficiently using cached px/mm."""
        # Convert local widget px to mm, then add current viewport clip offset
        y_mm_local = float(y_px) / max(1e-6, self._widget_px_per_mm)
        y_mm = y_mm_local + float(self._view_y_mm_offset or 0.0)
        return self.mm_to_time(y_mm)

    def set_view_metrics(self, px_per_mm: float, widget_px_per_mm: float, dpr: float) -> None:
        """Provide current view scale for fast pixel↔mm conversions."""
        self._px_per_mm = float(px_per_mm)
        self._widget_px_per_mm = float(widget_px_per_mm)
        self._dpr = float(dpr)

    def set_view_offset_mm(self, y_mm_offset: float) -> None:
        """Set the current viewport origin offset (top of clip) in mm."""
        self._view_y_mm_offset = float(y_mm_offset)
        # Recompute local mm cursor on scroll so overlays stay aligned
        if self.time_cursor is not None:
            abs_mm = self.time_to_mm(float(self.time_cursor))
            self.mm_cursor = abs_mm - float(self._view_y_mm_offset or 0.0)

    def set_viewport_height_mm(self, h_mm: float) -> None:
        """Provide the current viewport height in mm for drawer culling."""
        self._viewport_h_mm = max(0.0, float(h_mm))

    def snap_time(self, ticks: float) -> float:
        """Snap time ticks to the start of the previous snap band.

        Example: with snap size S, values in [k*S, (k+1)*S) snap to k*S.
        """
        units = max(1e-6, float(self.snap_size_units))
        ratio = float(ticks) / units
        k = math.floor(ratio + 1e-9)  # tiny epsilon to counter floating error
        return k * units

    # ---- Editor guides (tool-agnostic overlays) ----
    def draw_guides(self, du: DrawUtil) -> None:
        """Draw overlays: playhead and mouse cursor guidance.

        Playhead renders regardless of mouse-over; cursor renders only when active.
        """
        margin = float(self.margin or 0.0)
        stave_width = float(self.stave_width or 0.0)

        # --- Playhead overlay (always, if available) ---
        if self.playhead_time is not None:
            y_mm_ph = float(self.time_to_mm(float(self.playhead_time)))
            du.add_line(
                self.pitch_to_x(2),
                y_mm_ph,
                self.pitch_to_x(86),
                y_mm_ph,
                color=(0.0, 0.0, 0.0, 0.75),
                width_mm=1.5,
                id=0,
                tags=['playhead'],
            )

        # --- Mouse cursor guides (hide when mouse leaves) ---
        if self.guides_active and (self.mm_cursor is not None):
            # get cursor mm position: convert local (viewport) mm -> absolute mm
            y_mm = float(self.mm_cursor) + float(self._view_y_mm_offset or 0.0)

            # Left side of stave
            du.add_line(
                2.0,
                y_mm,
                margin,
                y_mm,
                color=(0, 0, 0, 1),
                width_mm=.75,
                dash_pattern=[0, 2],
                id=0,
                tags=['cursor'],
            )

            # Right side of stave
            du.add_line(
                margin + stave_width,
                y_mm,
                margin * 2.0 + stave_width - 2,
                y_mm,
                color=(0, 0, 0, 1),
                width_mm=.75,
                dash_pattern=[0, 2],
                id=0,
                tags=['cursor'],
            )

            if (isinstance(self._tool, NoteTool)) and (self.time_cursor is not None) and (self.pitch_cursor is not None):
                x_mm = float(self.pitch_to_x(int(self.pitch_cursor)))
                w = float(self.semitone_dist or 0.5)
                h = w * 2
                if self.pitch_cursor in BLACK_KEYS:
                    w *= .8
                layout = self.current_score().layout
                l = float(layout.note_stem_length_semitone or 3) * float(self.semitone_dist or 0.5)
                # Draw a translucent preview notehead at cursor
                fill_color = self.accent_color if self.pitch_cursor in BLACK_KEYS else (1,1,1,1)
                
                # draw the notehead and stem
                du.add_oval(
                    x_mm - w,
                    y_mm,
                    x_mm + w,
                    y_mm + h,
                    fill_color=fill_color,
                    stroke_color=self.accent_color,
                    stroke_width_mm=0.5,
                    id=0,
                    tags=['cursor'],
                )
                du.add_line(
                    x_mm,
                    y_mm,
                    x_mm + l if self.hand_cursor == '>' else x_mm - l,
                    y_mm,
                    color=self.accent_color,
                    width_mm=0.75,
                    id=0,
                    tags=['cursor'],
                )
                # draw the left hand dot indicator
                if self.hand_cursor == '<' and self.current_score().layout.note_leftdot_visible:
                    w = float(self.semitone_dist or 0.5) * 2.0
                    dot_d = w * 0.35
                    cy = y_mm + (w / 2.0)
                    fill = (1, 1, 1, 1) if (self.pitch_cursor in BLACK_KEYS) else (0, 0, 0, 1)
                    du.add_oval(
                        x_mm - dot_d / 3.0,
                        cy - dot_d / 3.0,
                        x_mm + dot_d / 3.0,
                        cy + dot_d / 3.0,
                        stroke_color=None,
                        fill_color=fill,
                        id=0,
                        tags=["cursor"],
                    )

            if (isinstance(self._tool, GraceNoteTool)) and (self.time_cursor is not None) and (self.pitch_cursor is not None):
                x_mm = float(self.pitch_to_x(int(self.pitch_cursor)))
                scale = float(getattr(self.current_score().layout, 'grace_note_scale', 0.75) or 0.75)
                outline_w = float(
                    getattr(self.current_score().layout, 'grace_note_outline_width_mm', getattr(self.current_score().layout, 'grace_note_outline_width', 0.3))
                    or 0.3
                )
                w = float(self.semitone_dist or 0.5) * scale
                top = y_mm
                bottom = y_mm + (w * 2.0)
                left = x_mm - w
                right = x_mm + w
                du.add_oval(
                    left,
                    top,
                    right,
                    bottom,
                    stroke_color=self.accent_color,
                    stroke_width_mm=0.0,
                    fill_color=self.accent_color,
                    id=0,
                    tags=['cursor'],
                )

        # --- Selection window overlay (always visible when active) ---
        if self._selection_active:
            # Compute absolute selection bounds in mm
            y1_mm = float(self.time_to_mm(float(self._sel_start_units)))
            y2_mm = float(self.time_to_mm(float(self._sel_end_units)))
            sel_top_mm = min(y1_mm, y2_mm)
            sel_bottom_mm = max(y1_mm, y2_mm)
            # Clamp to current viewport to allow selection beyond the visible area
            vp_top = float(self._view_y_mm_offset or 0.0)
            vp_bottom = vp_top + float(self._viewport_h_mm or 0.0)
            draw_top = max(sel_top_mm, vp_top)
            draw_bottom = min(sel_bottom_mm, vp_bottom)
            if draw_bottom > draw_top:
                # Horizontal extent: span between selected pitch range
                min_p = max(1, min(88, int(self._sel_min_pitch)))
                max_p = max(1, min(88, int(self._sel_max_pitch)))
                x_left = float(self.pitch_to_x(min_p))
                x_right = float(self.pitch_to_x(max_p))
                x2 = min(x_left, x_right)
                x1 = max(x_left, x_right)
                du.add_rectangle(
                    x2,
                    draw_top,
                    x1,
                    draw_bottom,
                    stroke_color=None,
                    fill_color=self.selection_color,
                    id=0,
                    tags=['selection_rect'],
                )

    # ---- Selection detection & clipboard ----
    def set_selection_window(self, start_units: float, end_units: float, active: bool = True) -> None:
        """Programmatically set the selection window in ticks and toggle its visibility."""
        try:
            self._sel_start_units = float(start_units)
            self._sel_end_units = float(end_units)
            self._selection_active = bool(active)
        except Exception:
            pass

    def clear_selection(self) -> None:
        """Clear selection window and clipboard."""
        self._selection_active = False
        self._sel_start_units = 0.0
        self._sel_end_units = 0.0
        self._sel_anchor_units = 0.0
        self._sel_min_pitch = 1
        self._sel_max_pitch = 88
        self._sel_anchor_pitch = 1
        # Persistent clipboard is not cleared here

    def select_all(self) -> None:
        """Select the full score range and all pitches."""
        try:
            total_len = float(self._calc_base_grid_list_total_length())
        except Exception:
            total_len = 0.0
        ss = max(1e-6, float(getattr(self, 'snap_size_units', 1.0) or 1.0))
        end_units = float(total_len) if total_len > ss else float(ss)
        self._sel_anchor_units = 0.0
        self._sel_start_units = 0.0
        self._sel_end_units = end_units
        self._sel_anchor_pitch = 1
        self._sel_min_pitch = 1
        self._sel_max_pitch = 88
        self._selection_active = True

    def transpose_selected_notes(self, delta_semitones: int) -> bool:
        """Move selected notes by semitone steps and shift selection range.

        Returns True if any notes were updated.
        """
        score: SCORE | None = self.current_score()
        if score is None or not self._selection_active:
            return False
        try:
            delta = int(delta_semitones)
        except Exception:
            return False
        if delta == 0:
            return False
        # Prefer cached viewport notes when selection is fully within cache range
        a = float(min(self._sel_start_units, self._sel_end_units))
        b = float(max(self._sel_start_units, self._sel_end_units - 0.1))
        min_p = max(1, min(88, int(getattr(self, '_sel_min_pitch', 1))))
        max_p = max(1, min(88, int(getattr(self, '_sel_max_pitch', 88))))
        notes = []
        used_cache = False
        try:
            cache = getattr(self, '_draw_cache', None) or {}
            t_begin = float(cache.get('time_begin', float('inf')))
            t_end = float(cache.get('time_end', float('-inf')))
            if a >= t_begin and b <= t_end:
                for n in list(cache.get('notes_view') or []):
                    try:
                        t0 = float(getattr(n, 'time', 0.0) or 0.0)
                        p = int(getattr(n, 'pitch', 0) or 0)
                    except Exception:
                        continue
                    if a <= t0 <= b and min_p <= p <= max_p:
                        notes.append(n)
                used_cache = True
        except Exception:
            used_cache = False
        if not notes:
            sel = self.detect_events_from_time_window(self._sel_start_units, self._sel_end_units - 0.1)
            notes = sel.get('note', []) if isinstance(sel, dict) else []
        if not notes:
            return False
        updated = False
        for n in notes:
            try:
                p = int(getattr(n, 'pitch', 0) or 0)
                if p <= 0:
                    continue
                np = max(1, min(88, p + delta))
                if np != p:
                    setattr(n, 'pitch', int(np))
                    updated = True
            except Exception:
                continue
        if not updated:
            return False
        try:
            self._sel_min_pitch = max(1, min(88, int(self._sel_min_pitch) + delta))
            self._sel_max_pitch = max(1, min(88, int(self._sel_max_pitch) + delta))
            self._sel_anchor_pitch = max(1, min(88, int(self._sel_anchor_pitch) + delta))
        except Exception:
            pass
        if used_cache:
            self._reuse_draw_cache_once = True
        # Lightweight redraw now; snapshot is debounced to avoid lag on key repeat
        try:
            w = getattr(self, 'widget', None)
            if w is not None and hasattr(w, 'force_full_redraw'):
                w.force_full_redraw()
        except Exception:
            pass
        self._queue_transpose_snapshot()
        return True

    def shift_selected_notes_time(self, delta_units: float) -> bool:
        """Shift selected notes in time by `delta_units` ticks and move the selection window."""
        score: SCORE | None = self.current_score()
        if score is None or not self._selection_active:
            return False
        try:
            delta = float(delta_units)
        except Exception:
            return False
        if abs(delta) < 1e-9:
            return False
        sel = self.detect_events_from_time_window(self._sel_start_units, self._sel_end_units - 0.1)
        notes = sel.get('note', []) if isinstance(sel, dict) else []
        if not notes:
            return False
        try:
            min_time = min(float(getattr(n, 'time', 0.0) or 0.0) for n in notes)
        except Exception:
            min_time = 0.0
        delta_clamped = delta
        if delta < 0.0:
            # Prevent shifting before time zero
            limit = min_time + delta
            if limit < 0.0:
                delta_clamped = -min_time
        if abs(delta_clamped) < 1e-9:
            return False
        updated = False
        for n in notes:
            try:
                t = float(getattr(n, 'time', 0.0) or 0.0)
                new_t = max(0.0, t + delta_clamped)
                if not math.isclose(new_t, t):
                    setattr(n, 'time', new_t)
                    updated = True
            except Exception:
                continue
        if not updated:
            return False
        try:
            self._sel_start_units = max(0.0, float(self._sel_start_units) + delta_clamped)
            self._sel_end_units = max(0.0, float(self._sel_end_units) + delta_clamped)
            self._sel_anchor_units = max(0.0, float(self._sel_anchor_units) + delta_clamped)
        except Exception:
            pass
        try:
            self.update_score_length()
        except Exception:
            pass
        self._reuse_draw_cache_once = True
        try:
            w = getattr(self, 'widget', None)
            if w is not None and hasattr(w, 'force_full_redraw'):
                w.force_full_redraw()
        except Exception:
            pass
        self._queue_transpose_snapshot(label='shift_selected_notes_time')
        return True

    def set_selected_notes_hand(self, hand: str) -> bool:
        """Assign selected notes to a hand and snapshot the change.

        Returns True if any notes were updated.
        """
        score: SCORE | None = self.current_score()
        if score is None or not self._selection_active:
            return False
        h = str(hand)
        if h not in ('<', '>'):
            return False
        sel = self.detect_events_from_time_window(self._sel_start_units, self._sel_end_units - 0.1)
        notes = sel.get('note', []) if isinstance(sel, dict) else []
        if not notes:
            return False
        updated = False
        for n in notes:
            try:
                note_hand = str(getattr(n, 'hand', '') or '')
                note_color = str(getattr(n, 'color', '') or '')
                if note_hand != h or note_color != h:
                    setattr(n, 'hand', h)
                    setattr(n, 'color', h)
                    updated = True
            except Exception:
                continue
        if updated:
            self._snapshot_if_changed(coalesce=True, label='set_note_hand')
        return updated

    # ---- Modifier updates ----
    def set_shift_down(self, down: bool) -> None:
        self._shift_down = bool(down)

    def set_ctrl_down(self, down: bool) -> None:
        self._ctrl_down = bool(down)

    # Convenience properties for external access
    @property
    def selection_window_start(self) -> float:
        return float(self._sel_start_units)

    @selection_window_start.setter
    def selection_window_start(self, v: float) -> None:
        self._sel_start_units = float(v)
        self._sel_anchor_units = float(v)
    @property
    def selection_window_end(self) -> float:
        return float(self._sel_end_units)

    @selection_window_end.setter
    def selection_window_end(self, v: float) -> None:
        self._sel_end_units = float(v)

    def detect_events_from_time_window(self, start_units: float, end_units: float) -> dict:
        """Scan the SCORE and return a dict of events whose start time falls within [start_units, end_units].

        The returned dict keys are derived dynamically from `score.events` fields,
        so newly added event types in the SCORE model are handled automatically.
        """
        score: SCORE | None = self.current_score()
        if score is None:
            return {}
        a = float(min(start_units, end_units))
        b = float(max(start_units, end_units))
        # Pitch range constraints (inclusive)
        min_p = max(1, min(88, int(getattr(self, '_sel_min_pitch', 1))))
        max_p = max(1, min(88, int(getattr(self, '_sel_max_pitch', 88))))

        import dataclasses

        # Determine event type names from the dataclass fields of score.events
        try:
            event_fields = [f.name for f in dataclasses.fields(type(score.events))]
        except Exception:
            # Fallback: introspect attributes that are lists
            event_fields = [name for name in dir(score.events)
                            if isinstance(getattr(score.events, name, None), list)
                            and not name.startswith('_')]
        # Exclude tempo and line_break from selection rectangle detection
        event_fields = [n for n in event_fields if n not in ('tempo', 'line_break')]

        out: dict[str, list] = {name: [] for name in event_fields}

        def start_time(ev) -> float:
            # Prefer 'time' if present; else use the minimum of any '*_time' fields
            try:
                if hasattr(ev, 'time'):
                    return float(getattr(ev, 'time', 0.0) or 0.0)
            except Exception:
                pass
            try:
                d = dataclasses.asdict(ev)
            except Exception:
                d = getattr(ev, '__dict__', {})
            times = [float(v or 0.0) for k, v in d.items() if k.endswith('_time')]
            if times:
                return float(min(times))
            return 0.0

        def pitch_ok(ev) -> bool:
            # Apply pitch range only for events that have a 'pitch' attribute
            try:
                if hasattr(ev, 'pitch'):
                    p = int(getattr(ev, 'pitch', 0) or 0)
                    return (p and (min_p <= p <= max_p))
            except Exception:
                return False
            return True

        # Prefer cached viewport notes when selection is fully within cache range
        cached_notes_view = None
        try:
            cache = getattr(self, '_draw_cache', None) or {}
            t_begin = float(cache.get('time_begin', float('inf')))
            t_end = float(cache.get('time_end', float('-inf')))
            if a >= t_begin and b <= t_end:
                cached_notes_view = cache.get('notes_view') or []
        except Exception:
            cached_notes_view = None

        # Generic filtering across all event lists
        for name in event_fields:
            if name == 'note' and cached_notes_view is not None:
                lst = list(cached_notes_view)
            else:
                lst = getattr(score.events, name, []) or []
            if name == 'slur':
                # Special-case: slur uses 4 handles with relative pitch from C4 (key 40)
                for ev in lst:
                    try:
                        rpitches = [
                            int(getattr(ev, 'x1_rpitch', 0) or 0),
                            int(getattr(ev, 'x2_rpitch', 0) or 0),
                            int(getattr(ev, 'x3_rpitch', 0) or 0),
                            int(getattr(ev, 'x4_rpitch', 0) or 0),
                        ]
                    except Exception:
                        rpitches = [0, 0, 0, 0]
                    try:
                        times_h = [
                            float(getattr(ev, 'y1_time', 0.0) or 0.0),
                            float(getattr(ev, 'y2_time', 0.0) or 0.0),
                            float(getattr(ev, 'y3_time', 0.0) or 0.0),
                            float(getattr(ev, 'y4_time', 0.0) or 0.0),
                        ]
                    except Exception:
                        times_h = [0.0, 0.0, 0.0, 0.0]
                    # Convert relative pitch to absolute key number around C4 (key 40)
                    keys = [max(1, min(88, 40 + rp)) for rp in rpitches]
                    # If any handle lies within both the time and pitch selection window, include the slur
                    include = False
                    for k, th in zip(keys, times_h):
                        if (min_p <= k <= max_p) and (a <= th <= b):
                            include = True
                            break
                    if include:
                        out[name].append(ev)
            else:
                for ev in lst:
                    t0 = start_time(ev)
                    if a <= t0 <= b and pitch_ok(ev):
                        out[name].append(ev)
        return out

    def copy_selection(self) -> dict | None:
        """Copy current selection window events into the editor clipboard and return it."""
        if not self._selection_active:
            return None
        sel = self.detect_events_from_time_window(self._sel_start_units, self._sel_end_units - 0.1) # slight epsilon to not detect next event at end
        self.clipboard = sel
        return sel

    def cut_selection(self) -> dict | None:
        """Cut current selection window events: copy to clipboard, then remove from SCORE."""
        score: SCORE | None = self.current_score()
        if score is None:
            return None
        sel = self.copy_selection()
        if not sel:
            return None
        # Ensure clipboard holds the cut selection
        self.clipboard = sel
        # Remove selected instances from each list
        for key in sel:
            lst = getattr(score.events, key, None)
            if isinstance(lst, list):
                remain = [ev for ev in lst if ev not in sel[key]]
                setattr(score.events, key, remain)
        # Keep base grid length aligned to remaining notes.
        self.update_score_length()
        # Snapshot change
        self._snapshot_if_changed(coalesce=True, label='cut_selection')
        try:
            self.score_changed.emit()
        except Exception:
            pass
        return sel

    def delete_selection(self) -> bool:
        """Delete current selection window events without copying to clipboard.

        Returns True if deletion occurred, False otherwise. Clears selection overlay.
        """
        score: SCORE | None = self.current_score()
        if score is None or not self._selection_active:
            return False
        sel = self.detect_events_from_time_window(self._sel_start_units, self._sel_end_units - 0.1) # slight epsilon to not detect next event.
        deleted_any = False
        try:
            for key in sel:
                lst = getattr(score.events, key, None)
                if isinstance(lst, list) and sel[key]:
                    remain = [ev for ev in lst if ev not in sel[key]]
                    if len(remain) != len(lst):
                        deleted_any = True
                    setattr(score.events, key, remain)
            if deleted_any:
                # Keep base grid length aligned to remaining notes.
                self.update_score_length()
                self._snapshot_if_changed(coalesce=True, label='delete_selection')
                try:
                    self.score_changed.emit()
                except Exception:
                    pass
        except Exception:
            pass
        # Clear selection window and clipboard after delete
        self.clear_selection()
        return deleted_any

    def paste_selection_at_cursor(self) -> None:
        """Paste events from clipboard so that the earliest selection start aligns to `self.time_cursor`."""
        score: SCORE | None = self.current_score()
        if score is None or self.clipboard is None:
            return
        if self.time_cursor is None:
            return
        import dataclasses

        # Determine alignment offset: selection start -> cursor
        a = float(min(self._sel_start_units, self._sel_end_units))
        target = float(self.time_cursor)
        delta = float(target - a)

        # Track furthest end time to extend timeline if needed
        furthest_end = float(self._calc_base_grid_list_total_length())

        # Iterate types from clipboard dynamically
        for ev_type, items in (self.clipboard.items() if isinstance(self.clipboard, dict) else []):
            if not items:
                continue
            ctor = getattr(score, f"new_{ev_type}", None)
            if ctor is None:
                continue
            for ev in items:
                d = dataclasses.asdict(ev)
                # Remove id; it will be assigned by the constructor
                d.pop('_id', None)
                # Shift all time-related fields
                for k in list(d.keys()):
                    if k == 'time' or k.endswith('_time'):
                        try:
                            d[k] = float(d.get(k, 0.0)) + delta
                        except Exception:
                            pass
                # Create the new event
                ctor(**d)
                # Compute end time generically: max of all time fields, plus duration if applicable
                try:
                    time_fields = [float(v or 0.0) for kk, v in d.items() if kk == 'time' or kk.endswith('_time')]
                    t_end = max(time_fields) if time_fields else float(d.get('time', 0.0) or 0.0)
                    dur = float(d.get('duration', 0.0) or 0.0)
                    if dur > 0.0 and 'time' in d:
                        t_end = float(d.get('time', 0.0) or 0.0) + dur
                    furthest_end = max(furthest_end, float(t_end))
                except Exception:
                    pass
        # Extend timeline if pasted content exceeds current end barline
        cur_end = float(self._calc_base_grid_list_total_length())
        if furthest_end > cur_end:
            bg_list = list(getattr(score, 'base_grid', []) or [])
            if bg_list:
                last_bg = bg_list[-1]
                num = float(getattr(last_bg, 'numerator', 4) or 4)
                den = float(getattr(last_bg, 'denominator', 4) or 4)
                measure_len = num * (4.0 / den) * float(QUARTER_NOTE_UNIT)
                extra_measures = int(max(1, math.ceil((furthest_end - cur_end) / max(1e-6, measure_len))))
                last_bg.measure_amount = int(getattr(last_bg, 'measure_amount', 1) or 1) + extra_measures
        try:
            self.update_score_length()
        except Exception:
            pass
        # Snapshot change
        self._snapshot_if_changed(coalesce=True, label='paste_selection')
        try:
            self.score_changed.emit()
        except Exception:
            pass
        # Stop drawing selection overlay after paste (clipboard stays)
        self._selection_active = False

