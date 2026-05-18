from __future__ import annotations
from typing import Literal, Optional, Tuple, Dict, Type, TYPE_CHECKING
import math, bisect, heapq
import time
from PySide6 import QtCore, QtGui

from editor.selection import SelectionMixin
from editor.tool.base_tool import BaseTool
from editor.tool_manager import ToolManager
# Import tool templates
from editor.tool.beam_tool import BeamTool
from editor.tool.barline_tool import BarlineTool
from editor.tool.count_line_tool import CountLineTool
from editor.tool.grace_note_tool import GraceNoteTool
from editor.tool.line_break_tool import LineBreakTool
from editor.tool.note_tool import NoteTool
from editor.tool.pedal_tool import PedalTool
from editor.tool.slur_tool import SlurTool
from editor.tool.text_tool import TextTool
from editor.tool.base_grid_tool import BaseGridTool
from editor.tool.time_signature_tool import TimeSignatureTool
from editor.tool.dynamic_tool import DynamicTool
from editor.tool.tempo_tool import TempoTool
from editor.tool.grid_band_tool import GridBandTool
from editor.tool.arpeggio_tool import ArpeggioTool
from editor.ctlz import CtlZ
from file_model.base_grid import BaseGrid, resolve_grid_layer_offsets
from settings_manager import get_preferences_manager
from ui.style import Style
from file_model.SCORE import SCORE
from utils.CONSTANT import BE_KEYS, QUARTER_NOTE_UNIT, SHORTEST_DURATION
from editor.drawers.stave_drawer import StaveDrawerMixin
from editor.drawers.snap_drawer import SnapDrawerMixin
from editor.drawers.grid_band_drawer import GridBandDrawerMixin
from editor.drawers.grid_drawer import GridDrawerMixin
from editor.drawers.note_drawer import NoteDrawerMixin
from editor.drawers.accidental_drawer import AccidentalDrawerMixin
from editor.drawers.grace_note_drawer import GraceNoteDrawerMixin
from editor.drawers.beam_drawer import BeamDrawerMixin
from editor.drawers.pedal_drawer import PedalDrawerMixin
from editor.drawers.text_drawer import TextDrawerMixin
from editor.drawers.slur_drawer import SlurDrawerMixin
from editor.drawers.repeat_drawer import RepeatDrawerMixin
from editor.drawers.count_line_drawer import CountLineDrawerMixin
from editor.drawers.line_break_drawer import LineBreakDrawerMixin
from editor.drawers.tempo_drawer import TempoDrawerMixin
from editor.drawers.dynamic_drawer import DynamicDrawerMixin
from editor.drawers.crescendo_drawer import CrescendoDrawerMixin
from editor.drawers.decrescendo_drawer import DecrescendoDrawerMixin
from editor.drawers.time_signature_drawer import TimeSignatureDrawerMixin
from editor.drawers.arpeggio_drawer import ArpeggioDrawerMixin
from utils.CONSTANT import PIANO_KEY_AMOUNT, BLACK_KEYS
from utils.operator import Operator
from editor.hit_testing import HitTestingMixin
from ui.widgets.draw_util import DrawUtil
from symbol_design.noteheads import (
    resolve_notehead_spec,
    sheared_notehead_support_v,
    sheared_notehead_outline_points,
    support_point_from_outline_points,
)
from midi.player import Player

if TYPE_CHECKING:
    from editor.tool.base_tool import BaseTool
    from ui.widgets.draw_util import DrawUtil


class Editor(QtCore.QObject,
             HitTestingMixin,
             SelectionMixin,
             StaveDrawerMixin,
             SnapDrawerMixin,
             GridBandDrawerMixin,
             GridDrawerMixin,
             TimeSignatureDrawerMixin,
             NoteDrawerMixin,
             AccidentalDrawerMixin,
             ArpeggioDrawerMixin,
             GraceNoteDrawerMixin,
             BeamDrawerMixin,
             SlurDrawerMixin,
             TextDrawerMixin,
             PedalDrawerMixin,
             DynamicDrawerMixin,
             CrescendoDrawerMixin,
             DecrescendoDrawerMixin,
             RepeatDrawerMixin,
             CountLineDrawerMixin,
             LineBreakDrawerMixin,
             TempoDrawerMixin):
    """Main editor class: routes UI events to the current tool.

    Handles click vs drag classification using a 3px threshold.
    """

    DRAG_THRESHOLD: int = 1

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
            'barline': BarlineTool,
            'count_line': CountLineTool,
            'grace_note': GraceNoteTool,
            'line_break': LineBreakTool,
            'note': NoteTool,
            'pedal': PedalTool,
            'slur': SlurTool,
            # Backward compatibility: legacy tool names map to the unified barline tool.
            'start_repeat': BarlineTool,
            'end_repeat': BarlineTool,
            'text': TextTool,
            'base_grid': BaseGridTool,
            'time_signature': TimeSignatureTool,
            'dynamic': DynamicTool,
            'tempo': TempoTool,
            'grid_band': GridBandTool,
            'arpeggio': ArpeggioTool,
        }
        self._tm.set_tool(self._tool)

        # Press/drag state
        self._left_pressed: bool = False
        self._right_pressed: bool = False
        self._press_pos: Tuple[float, float] = (0.0, 0.0)
        self._dragging_left: bool = False
        self._dragging_right: bool = False
        self._ignore_next_left_release: bool = False
        self._ignore_next_right_release: bool = False
        self._pending_left_double_release: bool = False
        self._pending_right_double_release: bool = False
        self._pending_left_double_pos: tuple[float, float] | None = None
        self._pending_right_double_pos: tuple[float, float] | None = None

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
        paper_rgb = Style.get_named_rgb("paper", fallback=(255, 255, 255))
        self.paper_color = (
            float(paper_rgb[0]) / 255.0,
            float(paper_rgb[1]) / 255.0,
            float(paper_rgb[2]) / 255.0,
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

        # Global editor-only stroke width override (mm) for selected symbol lines.
        # This does not affect engraving/print output.
        self.editor_line_width_global: float = 0.5

        # Cache for key x-positions (index by piano key number 1..88)
        self._x_positions: Optional[list[float]] = None

        # View metrics for fast pixel <--> mm conversions
        self._px_per_mm: float = 1.0            # device px per mm
        self._widget_px_per_mm: float = 1.0     # logical (Qt) px per mm
        self._dpr: float = 1.0                  # device pixel ratio
        # View offset in mm (top of visible clip)
        self._view_y_mm_offset: float = 0.0
        # Viewport height (mm) of the visible clip
        self._viewport_h_mm: float = 0.0
        # Extra time (half note) to extend viewport cache at the bottom for beam continuity.
        # NOTE if a beam is longer then a half note it disappears partly on the bottom of the viewport.
        self.viewport_bottom_bleed: float = QUARTER_NOTE_UNIT * 2

        # cursor
        self.time_cursor: Optional[float] = None
        self.mm_cursor: Optional[float] = None
        self.pitch_cursor: Optional[int] = None
        self.hand_cursor: Literal['l', 'r'] = 'l'  # default hand for cursor overlays
        # show/hide guides depending on mouse over editor
        self.guides_active: bool = True
        # Playhead position (app time units). When set, draws a red line overlay.
        self.playhead_time: Optional[float] = None

        # Per-frame shared render cache (built at draw_all)
        self._draw_cache: dict | None = None
        # One-shot hint to reuse the current draw cache on the next frame
        self._reuse_draw_cache_once: bool = False
        # Cache for base-grid-derived timeline helpers used by _build_render_cache
        self._grid_time_cache_key: tuple | None = None
        self._grid_time_cache_values: tuple[list[float], list[float]] | None = None
        # Cache for note-time derived arrays used by _build_render_cache
        # to avoid full re-sorts every frame when notes are unchanged.
        self._note_time_cache_key: tuple | None = None
        self._note_time_cache_values: tuple[list, list[float], list[float], list[tuple[float, int]], list[float]] | None = None
        # Per-frame arpeggio Y-override data (populated by _build_render_cache).
        # arpeggio_y_overrides: note_id → projected y_mm for notes in an arpeggio chord.
        self._arpeggio_y_overrides: dict[int, float] = {}
        # Tiny mode: toggled by viewport width (stage 1: simplified drawing,
        # stage 2: skip parts of drawing). tiny_mode_alpha is a continuous fade factor
        # (1.0 = fully opaque, 0.0 = fully transparent) used by the view to
        # Per-frame hit rectangles (all types) in absolute mm coordinates; reset each frame
        self._hit_rects: list[dict] = []

        # Tiny mode: toggled by viewport width (stage 1: simplified drawing,
        self.tiny_mode_stage: int = 0
        self.tiny_mode_alpha: float = 1.0

        self._init_selection_state()
        self._cached_furthest_music_end: float = 0.0
        self._score_length_cache_valid: bool = False
        self._single_note_timing_dirty: dict | None = None
        # Modifier state
        self._shift_down: bool = False
        self._ctrl_down: bool = False
        # Gesture mode (locked on press so mid-drag modifier changes don't misclassify edits)
        self._left_selection_mode: bool = False
        self._right_selection_mode: bool = False
        # player for auditioning
        self.player: Player = None

    # ---- Drawing via mixins ----
    def draw_background_gray(self, du) -> None:
        """Fill the current page with print-view grey."""
        w_mm, h_mm = du.current_page_size_mm()
        du.add_rectangle(0.0, 0.0, w_mm, h_mm, stroke_color=None, fill_color=self.paper_color, id=0, tags=["background"])

    def draw_all(self, du) -> None:
        """Invoke drawer mixin methods; layer order is enforced by DrawUtil tags.

        We simply call all drawer methods; DrawUtil sorts items by tag layering.
        """
        frame_start = time.perf_counter()
        timing_rows: list[tuple[str, float]] = []

        # Reset hit rectangles for this frame; drawers will register rectangles
        self._hit_rects = []

        # In tiny stage 2, skip drawing to keep closing smooth
        if self.is_tiny_mode_ultra():
            total_ms = (time.perf_counter() - frame_start) * 1000.0
            print(f"[draw_all] total={total_ms:.3f} ms (tiny mode ultra skip)")
            return

        # Build shared render cache for this draw pass (fresh each frame)
        cache_start = time.perf_counter()
        self._build_render_cache()
        timing_rows.append(("build_render_cache", (time.perf_counter() - cache_start) * 1000.0))

        # Call drawer mixin methods in order
        methods = [
            getattr(self, 'draw_snap', None),
            getattr(self, 'draw_grid_band', None),
            getattr(self, 'draw_grid', None),
            getattr(self, 'draw_time_signature', None),
            getattr(self, 'draw_stave', None),
            getattr(self, 'draw_note', None),
            getattr(self, 'draw_arpeggio', None),
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
            if fn is None:
                continue
            drawer_start = time.perf_counter()
            fn(du)
            timing_rows.append((fn.__name__, (time.perf_counter() - drawer_start) * 1000.0))

        total_ms = (time.perf_counter() - frame_start) * 1000.0
        print("NEW FRAME TIMINGS:")
        for name, elapsed_ms in timing_rows:
            print(f"{name}={elapsed_ms:.3f} ms")
        print(f"[draw_all] total={total_ms:.3f} ms")

        # Keep render cache available for hit detection until next frame rebuild
        # (cleared at the start of _build_render_cache)

    def refresh_context_toolbar(self) -> None:
        """Ask ToolManager to rebuild contextual toolbar from current tool state."""
        try:
            if self._tm is not None and hasattr(self._tm, 'refresh_context_buttons'):
                self._tm.refresh_context_buttons()
        except Exception:
            pass

    def draw_frame(self) -> None:
        """Build a full frame immediately (cache + drawer registration) without painting.

        Creates a temporary DrawUtil using current layout page size, calls draw_all.
        Useful for immediate feedback from tools (e.g., updating hit rects/cache) before
        the widget triggers a repaint.
        """
        du = DrawUtil()
        if du is not None:
            # Derive page size from SCORE layout; fall back to A4
            w_mm = 210.0
            h_mm = 297.0
            sc = self.current_score()
            if sc is not None:
                lay = getattr(sc, 'layout', None)
                if lay is not None:
                    w_mm = float(getattr(lay, 'page_width_mm', w_mm) or w_mm)
                    h_mm = float(getattr(lay, 'page_height_mm', h_mm) or h_mm)
            du.set_current_page_size_mm(w_mm, h_mm)
        # Run the drawer pipeline to rebuild caches and register hit rectangles
        self.draw_all(du)
        
        # refresh overlay guides if applicable
        self.draw_guides(du)

    def force_redraw_from_model(self) -> None:
        """Request a full widget repaint from SCORE without prebuilding a duplicate frame."""
        from ui.widgets.cairo_views import CairoEditorWidget
        w: CairoEditorWidget = getattr(self, 'widget', None)
        if w is not None and hasattr(w, 'force_full_redraw'):
            w.force_full_redraw()
        elif w is not None and hasattr(w, 'update'):
            w.update()

    def _calculate_layout(self, view_width_mm: float) -> None:
        """Compute editor-specific layout based on the current view dimensions.
        The drawing is programmed in vertical orientation and rotated afterwards
        if the editor orientation == 'horizontal'.

        - margin divisor interpolates from 6 (x_zoom_factor=1.0) to 3 (x_zoom_factor=0.0)
        - stave width: width - 2 * margin
        - semitone spacing: stave width / physical semitone range (101 semitones from A0 to C8 including BE gaps)
        """
        x_zoom_factor = 1.0
        try:
            sc = self.current_score()
            app_state = getattr(sc, 'app_state', None) if sc is not None else None
            x_zoom_factor = float(getattr(app_state, 'x_zoom_factor', 1.0) if app_state is not None else 1.0)
        except Exception:
            x_zoom_factor = 1.0
        x_zoom_factor = max(0.0, min(1.0, x_zoom_factor))
        margin_divisor = 3.0 + (3.0 * x_zoom_factor)
        self.margin = view_width_mm / max(1e-6, float(margin_divisor))
        physical_semitone_spaces = 101
        self.stave_width = view_width_mm - (2 * self.margin)
        self.semitone_dist = self.stave_width / physical_semitone_spaces
        self.editor_height = self._calc_editor_height()
        self._rebuild_x_positions()

    '''
        ---- Note lookup ----
    '''

    def get_note_by_id(self, note_id: int):
        """Return the note event for id, preferring current viewport cache.

        Falls back to scanning all notes if cache is unavailable.
        """
        # Prefer notes in the current viewport draw cache
        cache = getattr(self, '_draw_cache', None) or {}
        notes_view = cache.get('notes_view') or []
        if notes_view:
            for n in notes_view:
                if int(getattr(n, '_id', -1) or -1) == note_id:
                    return n
        # Fallback: global scan
        score: SCORE | None = self.current_score()
        if score is None:
            return None
        for n in getattr(score.events, 'note', []) or []:
            if int(getattr(n, '_id', -1) or -1) == note_id:
                return n
        return None

    def get_measure_index_for_time(self, ticks: float) -> int:
        """Return 1-based measure index for a given time in ticks.

        Uses barline start positions across the score and finds the last
        barline at or before `ticks`. If no barline is at or before, returns 1.
        """
        bars = self._get_barline_positions()
        if not bars:
            return 1
        i = bisect.bisect_right(bars, float(ticks)) - 1
        return max(1, int(i + 1))

    def _rebuild_x_positions(self) -> None:
        """Precompute x positions for keys 1..88 with BE gap after a B or E key."""
        be_set = set(BE_KEYS)
        # to start at the margin for key 1 (A0), we initialize one semitone before the margin and then step forward
        x_pos = self.margin - self.semitone_dist
        x_positions = [x_pos]

        for n in range(1, PIANO_KEY_AMOUNT + 1):
            # Apply extra gap AFTER B/E, i.e., when stepping from (n-1) -> n
            if (n - 1) in be_set:
                x_pos += self.semitone_dist
            # Normal semitone step
            x_pos += self.semitone_dist
            x_positions.append(x_pos)

        self._x_positions = x_positions

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
        self._invalidate_score_length_cache()

    # Model provider for undo snapshots
    def set_file_manager(self, fm) -> None:
        """Provide FileManager so we can snapshot/restore SCORE for undo/redo."""
        self._file_manager = fm
        # Initialize ctlz with the initial model state
        if self._file_manager is not None:
            self._ctlz = CtlZ(self._file_manager)
            self._ctlz.reset_ctlz()

    def current_score(self) -> SCORE:
        """Return the current SCORE: prefer FileManager; fall back to explicit _score."""
        if self._file_manager is not None:
            return self._file_manager.current()
        return getattr(self, "_score", None)

    def _invalidate_score_length_cache(self) -> None:
        self._cached_furthest_music_end = 0.0
        self._score_length_cache_valid = False
        self._single_note_timing_dirty = None

    def mark_single_note_timing_dirty(self, note, previous_time: float, previous_duration: float) -> None:
        note_id = int(getattr(note, '_id', 0) or 0)
        if note_id <= 0:
            self._single_note_timing_dirty = None
            return
        self._single_note_timing_dirty = {
            'note_id': note_id,
            'note': note,
            'previous_time': float(previous_time),
            'previous_duration': float(previous_duration),
        }

    def clear_single_note_timing_dirty(self) -> None:
        self._single_note_timing_dirty = None

    def sync_arpeggios_with_notes(self) -> bool:
        """Keep arpeggio events consistent with current note time/pitch data.

        Rules:
        - Resolve each arpeggio to the best matching (time, hand) note cluster.
        - Update arpeggio.time and arpeggio.note_pitches from that live cluster.
        - Remove arpeggios with fewer than 2 resolved pitches.
        - De-duplicate by (time, note_pitches) to prevent stale duplicates.
        """
        score: SCORE | None = self.current_score()
        if score is None:
            return False

        arps = list(getattr(score.events, 'arpeggio', []) or [])
        if not arps:
            return False

        changed = False
        op = Operator(float(SHORTEST_DURATION))
        notes = list(getattr(score.events, 'note', []) or [])
        notes_by_hand: dict[str, list[object]] = {}
        for note in notes:
            hand = str(getattr(note, 'hand', 'l') or 'l')
            notes_by_hand.setdefault(hand, []).append(note)

        clusters: list[tuple[float, str, set[int]]] = []
        for hand, hand_notes in notes_by_hand.items():
            ordered = sorted(
                hand_notes,
                key=lambda n: (float(getattr(n, 'time', 0.0) or 0.0), int(getattr(n, 'pitch', 0) or 0)),
            )
            active_time: float | None = None
            active_pitches: set[int] = set()
            for note in ordered:
                try:
                    note_time = float(getattr(note, 'time', 0.0) or 0.0)
                    note_pitch = int(getattr(note, 'pitch', 0) or 0)
                except Exception:
                    continue
                if note_pitch <= 0:
                    continue
                if active_time is None or not op.eq(note_time, active_time):
                    if active_time is not None and active_pitches:
                        clusters.append((float(active_time), str(hand), set(active_pitches)))
                    active_time = float(note_time)
                    active_pitches = set()
                active_pitches.add(int(note_pitch))
            if active_time is not None and active_pitches:
                clusters.append((float(active_time), str(hand), set(active_pitches)))

        new_arps = []
        seen_keys: list[tuple[float, tuple[int, ...]]] = []

        for arp in arps:
            try:
                old_time = float(getattr(arp, 'time', 0.0) or 0.0)
                old_pitches = sorted(set(int(p) for p in (getattr(arp, 'note_pitches', []) or []) if int(p) > 0))
            except Exception:
                changed = True
                continue

            if len(old_pitches) < 2:
                changed = True
                continue

            old_pitch_set = set(old_pitches)
            best: tuple[int, float, float, str, set[int]] | None = None
            for t_value, hand, cluster_pitches in clusters:
                overlap = len(cluster_pitches.intersection(old_pitch_set))
                if overlap <= 0:
                    continue
                dist = abs(float(t_value) - float(old_time))
                # Max overlap first, then nearest time.
                cand = (int(overlap), -float(dist), float(t_value), str(hand), cluster_pitches)
                if best is None or cand > best:
                    best = cand

            if best is None:
                changed = True
                continue

            _overlap, _neg_dist, new_time, _hand, cluster_pitches = best
            new_pitches = sorted(int(p) for p in cluster_pitches if int(p) > 0)
            if len(new_pitches) < 2:
                changed = True
                continue

            if not op.eq(float(getattr(arp, 'time', 0.0) or 0.0), new_time):
                arp.time = new_time
                changed = True
            if list(getattr(arp, 'note_pitches', []) or []) != new_pitches:
                arp.note_pitches = list(new_pitches)
                changed = True

            uniq_pitches = tuple(new_pitches)
            duplicate = any(op.eq(float(existing_t), float(new_time)) and existing_p == uniq_pitches for existing_t, existing_p in seen_keys)
            if duplicate:
                changed = True
                continue
            seen_keys.append((float(new_time), uniq_pitches))
            new_arps.append(arp)

        if len(new_arps) != len(arps):
            score.events.arpeggio = new_arps
            changed = True

        return changed

    def _apply_single_note_timing_cache_update(self, notes: list) -> tuple[list, list[float], list[float], list[tuple[float, int]], list[float]] | None:
        dirty = self._single_note_timing_dirty
        cached = self._note_time_cache_values
        if not dirty or cached is None:
            return None
        notes_sorted, starts, ends, end_pairs, end_values = cached
        if len(notes_sorted) != len(notes):
            return None

        note_id = int(dirty.get('note_id', 0) or 0)
        note = dirty.get('note')
        if note is None or int(getattr(note, '_id', 0) or 0) != note_id:
            return None

        old_idx = -1
        for idx, existing in enumerate(notes_sorted):
            if int(getattr(existing, '_id', 0) or 0) == note_id:
                old_idx = idx
                break
        if old_idx < 0:
            return None

        old_notes_sorted = list(notes_sorted)
        old_starts = list(starts)
        old_ends = list(ends)
        old_end_pairs = list(end_pairs)
        old_end_values = list(end_values)

        note_entry = old_notes_sorted.pop(old_idx)
        old_starts.pop(old_idx)
        old_ends.pop(old_idx)

        pair_pos = -1
        for idx, (_end_value, pair_idx) in enumerate(old_end_pairs):
            if int(pair_idx) == old_idx:
                pair_pos = idx
                break
        if pair_pos < 0:
            return None
        old_end_pairs.pop(pair_pos)
        old_end_values.pop(pair_pos)

        adjusted_end_pairs: list[tuple[float, int]] = []
        for end_value, pair_idx in old_end_pairs:
            new_pair_idx = int(pair_idx)
            if new_pair_idx > old_idx:
                new_pair_idx -= 1
            adjusted_end_pairs.append((float(end_value), new_pair_idx))

        new_time = float(getattr(note_entry, 'time', 0.0) or 0.0)
        new_duration = float(getattr(note_entry, 'duration', 0.0) or 0.0)
        new_pitch = int(getattr(note_entry, 'pitch', 0) or 0)
        op_time = Operator(float(SHORTEST_DURATION))
        insert_idx = bisect.bisect_left(old_starts, new_time)
        while insert_idx < len(old_starts):
            if op_time.gt(old_starts[insert_idx], new_time):
                break
            if op_time.eq(old_starts[insert_idx], new_time) and int(getattr(old_notes_sorted[insert_idx], 'pitch', 0) or 0) > new_pitch:
                break
            insert_idx += 1

        old_notes_sorted.insert(insert_idx, note_entry)
        old_starts.insert(insert_idx, new_time)
        new_end = new_time + new_duration
        old_ends.insert(insert_idx, new_end)

        rebased_end_pairs: list[tuple[float, int]] = []
        for end_value, pair_idx in adjusted_end_pairs:
            new_pair_idx = int(pair_idx)
            if new_pair_idx >= insert_idx:
                new_pair_idx += 1
            rebased_end_pairs.append((float(end_value), new_pair_idx))
        end_insert_idx = bisect.bisect_right(old_end_values, new_end)
        old_end_values.insert(end_insert_idx, new_end)
        rebased_end_pairs.insert(end_insert_idx, (new_end, insert_idx))
        return (old_notes_sorted, old_starts, old_ends, rebased_end_pairs, old_end_values)

    def _snapshot_if_changed(self, coalesce: bool = False, label: str = "") -> None:
        if self._file_manager is None:
            return
        has_change = True
        # Use dict-based ctlz snapshots
        if self._ctlz is not None:
            has_change = bool(self._ctlz.add_ctlz())
        if not has_change:
            return
        # Notify FileManager so it can handle autosave/session saving/dirty state
        if hasattr(self._file_manager, 'on_model_changed'):
            self._file_manager.on_model_changed()
        else:
            self._file_manager.autosave_current()
            self._file_manager.mark_dirty()
        # Ensure any edit is reflected immediately from the model
        self.force_redraw_from_model()
        self.score_changed.emit()

    # Public undo/redo (optional consumers can bind Ctrl+Z / Ctrl+Shift+Z)
    def undo(self) -> None:
        if self._file_manager is None:
            return
        snap = None
        if self._ctlz is not None:
            snap = self._ctlz.undo()
        if snap is not None:
            self._file_manager.replace_current(snap)
            self._invalidate_score_length_cache()
            # The SCORE instance was replaced; clear caches that may hold old object references.
            self._draw_cache = None
            self._reuse_draw_cache_once = False
            self._note_time_cache_key = None
            self._note_time_cache_values = None
            self._grid_time_cache_key = None
            self._grid_time_cache_values = None
            self._file_manager.mark_dirty()
            self.score_changed.emit()

    def redo(self) -> None:
        if self._file_manager is None:
            return
        snap = None
        if self._ctlz is not None:
            snap = self._ctlz.redo()
        if snap is not None:
            self._file_manager.replace_current(snap)
            self._invalidate_score_length_cache()
            # The SCORE instance was replaced; clear caches that may hold old object references.
            self._draw_cache = None
            self._reuse_draw_cache_once = False
            self._note_time_cache_key = None
            self._note_time_cache_values = None
            self._grid_time_cache_key = None
            self._grid_time_cache_values = None
            self._file_manager.mark_dirty()
            self.score_changed.emit()

    def reset_undo_stack(self) -> None:
        if self._ctlz is not None:
            self._ctlz.reset_ctlz()

    '''
        ---- Mouse event routing ----
    '''
    def mouse_press(self, button: int, x: float, y: float) -> None:
        if button == 1:
            self._left_pressed = True
            self._dragging_left = False
            self._press_pos = (x, y)
            self._left_selection_mode = bool(self._shift_down)
            if not self._left_selection_mode:
                self._tool.on_left_press(x, y)
            # If Shift is held, initialize selection anchor on left press
            if self._left_selection_mode:
                self._begin_selection_drag(x, y)
        elif button == 2:
            self._right_pressed = True
            self._dragging_right = False
            self._press_pos = (x, y)
            # Right-button gesture is selection unless the active tool explicitly edits on right drag.
            self._right_selection_mode = not bool(getattr(self._tool, 'RIGHT_DRAG_EDITS', False))
            if not self._right_selection_mode:
                self._tool.on_right_press(x, y)
            else:
                # Initialize selection anchor at press to be robust against scrolling
                self._begin_selection_drag(x, y)

    def mouse_move(self, x: float, y: float, dx: float, dy: float) -> None:
        # Recover from stale press state (e.g., modal dialogs interrupt release events).
        try:
            buttons = QtGui.QGuiApplication.mouseButtons()
            left_phys_down = bool(buttons & QtCore.Qt.MouseButton.LeftButton)
            right_phys_down = bool(buttons & QtCore.Qt.MouseButton.RightButton)
        except Exception:
            left_phys_down = self._left_pressed
            right_phys_down = self._right_pressed

        if self._left_pressed and not left_phys_down:
            if not self._left_selection_mode:
                self._tool.on_left_unpress(x, y)
            self._left_pressed = False
            self._dragging_left = False
            self._left_selection_mode = False

        if self._right_pressed and not right_phys_down:
            if self._dragging_right and not self._right_selection_mode:
                self._tool.on_right_drag_end(x, y)
            self._tool.on_right_unpress(x, y)
            if self._dragging_right and self._right_selection_mode:
                # Selection was being built; preserve it and suppress the upcoming release
                self._ignore_next_right_release = True
            else:
                self._selection_active = False
            self._right_pressed = False
            self._dragging_right = False
            self._right_selection_mode = False

        if self._left_pressed:
            if not self._dragging_left and (abs(dx) > self.DRAG_THRESHOLD or abs(dy) > self.DRAG_THRESHOLD):
                self._dragging_left = True
                if not self._left_selection_mode:
                    self._tool.on_left_drag_start(x, y)
            if self._dragging_left:
                if not self._left_selection_mode:
                    self._tool.on_left_drag(x, y, dx, dy)
                # Update selection window when Shift+Left-dragging
                if self._left_selection_mode:
                    self._update_selection_drag(x, y)
                # Do not capture multiple intermediate drag snapshots
        elif self._right_pressed:
            if not self._dragging_right and (abs(dx) > self.DRAG_THRESHOLD or abs(dy) > self.DRAG_THRESHOLD):
                self._dragging_right = True
                if not self._right_selection_mode:
                    self._tool.on_right_drag_start(x, y)
            if self._dragging_right:
                if not self._right_selection_mode:
                    self._tool.on_right_drag(x, y, dx, dy)
                else:
                    # Update selection window while right-dragging (selection mode)
                    self._update_selection_drag(x, y)
                # Skip intermediate drag snapshots
        else:
            # Update shared cursor state for guide rendering (time + mm), with snapping
            t = self.widget_px_to_time(x, y)
            t = self.snap_time(t)
            self.time_cursor = t
            # Store cursor mm relative to viewport (local mm)
            abs_mm = self.time_to_mm(float(t))
            self.mm_cursor = abs_mm - float(self._view_y_mm_offset or 0.0)
            # Track pitch under cursor only when X is within the piano key span.
            # This keeps the preview note hidden while the mouse is out of range.
            x_mm, _page_y_mm = self.widget_px_to_page_mm(x, y)
            if self._x_positions is None:
                self._rebuild_x_positions()
            min_x = float(self._x_positions[1])
            max_x = float(self._x_positions[PIANO_KEY_AMOUNT])
            if min_x <= x_mm <= max_x:
                self.pitch_cursor = self.widget_px_to_pitch(x, y)
            else:
                self.pitch_cursor = None
            self._tool.on_mouse_move(x, y)

    def mouse_release(self, button: int, x: float, y: float) -> None:
        if button == 1:
            is_left_double_release = bool(self._pending_left_double_release)
            if self._ignore_next_left_release:
                self._ignore_next_left_release = False
                if not self._left_selection_mode:
                    self._tool.on_left_unpress(x, y)
                self._selection_active = False
                self._left_pressed = False
                self._dragging_left = False
                self._left_selection_mode = False
                return
            if self._dragging_left:
                if not self._left_selection_mode:
                    self._tool.on_left_drag_end(x, y)
                    # Capture a single coalesced snapshot for the whole drag
                    self._snapshot_if_changed(coalesce=True, label="left_drag")
            else:
                # Click if moved <= threshold
                px, py = self._press_pos
                if (not is_left_double_release) and (abs(x - px) <= self.DRAG_THRESHOLD and abs(y - py) <= self.DRAG_THRESHOLD):
                    if not self._left_selection_mode:
                        self._tool.on_left_click(x, y)
                        # Capture click changes (non-coalesced)
                        self._snapshot_if_changed(coalesce=False, label="left_click")
            # Stop drawing selection on any click
            if not self._dragging_left and not self._left_selection_mode:
                self._selection_active = False
            if not self._left_selection_mode:
                self._tool.on_left_unpress(x, y)
            self._left_pressed = False
            self._dragging_left = False
            self._left_selection_mode = False
        elif button == 2:
            if self._ignore_next_right_release:
                self._ignore_next_right_release = False
                self._right_pressed = False
                self._dragging_right = False
                self._right_selection_mode = False
                return
            is_right_double_release = bool(self._pending_right_double_release)
            if self._dragging_right:
                if not self._right_selection_mode:
                    self._tool.on_right_drag_end(x, y)
                    if bool(getattr(self._tool, 'RIGHT_DRAG_EDITS', False)):
                        self._snapshot_if_changed(coalesce=True, label="right_drag")
                # Do not modify clipboard on selection changes
            else:
                px, py = self._press_pos
                if (not is_right_double_release) and (abs(x - px) <= self.DRAG_THRESHOLD and abs(y - py) <= self.DRAG_THRESHOLD):
                    changed = self._tool.on_right_click(x, y)
                    if changed is not False:
                        self._snapshot_if_changed(coalesce=False, label="right_click")
            # Stop drawing selection on any click
            if not self._dragging_right:
                self._selection_active = False
            self._tool.on_right_unpress(x, y)
            self._right_pressed = False
            self._dragging_right = False
            self._right_selection_mode = False

    def mouse_double_click(self, button: int, x: float, y: float) -> None:
        if button == 1:
            self._pending_left_double_release = True
            self._pending_left_double_pos = (x, y)
            if not self._shift_down:
                self._tool.on_left_double_click(x, y)
            self._schedule_double_unpress_dispatch(button=1)
        elif button == 2:
            self._pending_right_double_release = True
            self._pending_right_double_pos = (x, y)
            self._tool.on_right_double_click(x, y)
            self._schedule_double_unpress_dispatch(button=2)

    def _schedule_double_unpress_dispatch(self, button: int) -> None:
        def _try_dispatch() -> None:
            try:
                buttons = QtGui.QGuiApplication.mouseButtons()
                left_down = bool(buttons & QtCore.Qt.MouseButton.LeftButton)
                right_down = bool(buttons & QtCore.Qt.MouseButton.RightButton)
            except Exception:
                left_down = False
                right_down = False

            if button == 1:
                if not self._pending_left_double_release:
                    return
                if left_down:
                    QtCore.QTimer.singleShot(0, _try_dispatch)
                    return
                pos = self._pending_left_double_pos or self._press_pos
                self._pending_left_double_release = False
                self._pending_left_double_pos = None
                if not self._shift_down:
                    self._tool.on_left_double_unpress(float(pos[0]), float(pos[1]))
                return

            if not self._pending_right_double_release:
                return
            if right_down:
                QtCore.QTimer.singleShot(0, _try_dispatch)
                return
            pos = self._pending_right_double_pos or self._press_pos
            self._pending_right_double_release = False
            self._pending_right_double_pos = None
            self._tool.on_right_double_unpress(float(pos[0]), float(pos[1]))

        QtCore.QTimer.singleShot(0, _try_dispatch)

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
        zpq: float = float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0) or 25.0)
        stave_length_mm = (total_time_ticks / float(QUARTER_NOTE_UNIT)) * zpq
        top_bottom_mm = float(self.margin or 0.0) * 6.0
        height_mm = max(10.0, stave_length_mm + top_bottom_mm)
        self.editor_height = height_mm
        return height_mm

    def update_score_length(self, edited_event=None) -> None:
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

        # Keep arpeggio model data synced while notes are edited in time/pitch.
        self.sync_arpeggios_with_notes()

        current_end = float(self._calc_base_grid_list_total_length())
        bg_list = score.base_grid

        # update only last BaseGrid
        last_bg: BaseGrid = bg_list[-1]
        num = float(getattr(last_bg, 'numerator', 4) or 4)
        den = float(getattr(last_bg, 'denominator', 4) or 4)
        measure_len = num * (4.0 / den) * float(QUARTER_NOTE_UNIT)

        # Keep at least one trailing measure segment and allow both extend/trim.
        last_measures = int(getattr(last_bg, 'measure_amount', 1) or 1)
        prefix_len = float(current_end - (measure_len * float(max(1, last_measures))))
        min_total_end = float(prefix_len + measure_len)

        def _fit_last_segment_to_end(target_end: float, grace_times: list[float] | None = None) -> float:
            target = max(float(target_end), float(min_total_end))
            needed_last_measures = int(max(1, math.ceil((target - prefix_len) / max(1e-6, measure_len))))
            fitted_end = float(prefix_len + (measure_len * float(needed_last_measures)))
            if grace_times:
                if any(float(gt) >= fitted_end - 1e-6 for gt in grace_times):
                    needed_last_measures += 1
                    fitted_end = float(prefix_len + (measure_len * float(needed_last_measures)))
            last_bg.measure_amount = int(max(1, needed_last_measures))
            return fitted_end

        if edited_event is not None:
            event_time = float(getattr(edited_event, 'time', 0.0) or 0.0)
            duration = getattr(edited_event, 'duration', None)
            event_end = event_time + float(duration or 0.0) if duration is not None else event_time
            is_grace_like = duration is None

            prev_end = None
            dirty = getattr(self, '_single_note_timing_dirty', None)
            if isinstance(dirty, dict):
                dirty_id = int(dirty.get('note_id', 0) or 0)
                event_id = int(getattr(edited_event, '_id', 0) or 0)
                if dirty_id > 0 and event_id > 0 and dirty_id == event_id:
                    prev_time = float(dirty.get('previous_time', event_time) or event_time)
                    prev_duration = float(dirty.get('previous_duration', 0.0) or 0.0)
                    prev_end = float(prev_time + prev_duration)

            shrink_may_affect_end = False
            if (self._score_length_cache_valid and prev_end is not None
                and prev_end >= float(self._cached_furthest_music_end or 0.0) - 1e-6
                and event_end < prev_end - 1e-6):
                shrink_may_affect_end = True

            if (not shrink_may_affect_end
                and event_end <= current_end
                and not (is_grace_like and event_end >= current_end - 1e-6)):
                if self._score_length_cache_valid:
                    self._cached_furthest_music_end = max(float(self._cached_furthest_music_end or 0.0), event_end)
                return

            # Fast extend path remains cheap. Potential shrink of the previous max
            # intentionally falls through to the full recompute below.
            if not shrink_may_affect_end:
                furthest_end = float(event_end)
                if is_grace_like and event_end >= current_end - 1e-6:
                    furthest_end = max(furthest_end, current_end + measure_len)
                _fit_last_segment_to_end(furthest_end)
                # Keep this cache as "true furthest musical event end" (not barline-fitted end)
                # so shrink detection on subsequent drag updates remains correct.
                self._cached_furthest_music_end = float(furthest_end)
                self._score_length_cache_valid = True
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
            furthest_end = max(furthest_end, t)
            grace_times.append(t)

        fitted_end = _fit_last_segment_to_end(furthest_end, grace_times)
        self._cached_furthest_music_end = float(furthest_end)
        self._score_length_cache_valid = True
        return

    # ---- Shared render cache ----
    def _compute_note_time_cache_key(self, notes: list) -> tuple[int, int]:
        """Cheap content hash for note timing/pitch fields used by sorting/culling."""
        h = 1469598103934665603  # FNV-1a 64-bit offset basis
        fnv_prime = 1099511628211
        mask = (1 << 64) - 1
        for n in notes:
            try:
                t_i = int(round(float(getattr(n, 'time', 0.0) or 0.0) * 1000.0))
                d_i = int(round(float(getattr(n, 'duration', 0.0) or 0.0) * 1000.0))
                p_i = int(getattr(n, 'pitch', 0) or 0)
                nid = int(getattr(n, '_id', 0) or 0)
            except Exception:
                t_i, d_i, p_i, nid = (0, 0, 0, 0)
            h ^= (t_i & mask)
            h = (h * fnv_prime) & mask
            h ^= (d_i & mask)
            h = (h * fnv_prime) & mask
            h ^= (p_i & mask)
            h = (h * fnv_prime) & mask
            h ^= (nid & mask)
            h = (h * fnv_prime) & mask
        return (len(notes), int(h))

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
        
        # get score data
        score: SCORE | None = self.current_score()
        if score is None:
            return

        # Compute viewport times (ticks) from mm offset and height
        top_mm = self._view_y_mm_offset
        vp_h_mm = self._viewport_h_mm
        bottom_mm = top_mm + vp_h_mm
        # Small bleed similar to prior behavior
        zpq = score.app_state.zoom_mm_per_quarter
        bleed_mm = max(2.0, zpq * 0.25)
        time_begin = float(self.mm_to_time(top_mm - bleed_mm))
        bottom_bleed = self.viewport_bottom_bleed
        time_end = float(self.mm_to_time(bottom_mm + bleed_mm)) + bottom_bleed

        # Comparator with threshold of 7 ticks
        op = Operator(SHORTEST_DURATION)

        # Notes sorted/indexed by timing. Reuse persistent cache when note data is unchanged.
        notes = score.events.note
        patched_values = self._apply_single_note_timing_cache_update(notes)
        if patched_values is not None:
            notes_sorted, starts, ends, end_pairs, end_values = patched_values
            self._note_time_cache_values = patched_values
            self._note_time_cache_key = None
        else:
            note_cache_key = self._compute_note_time_cache_key(notes)
            if self._note_time_cache_key == note_cache_key and self._note_time_cache_values is not None:
                notes_sorted, starts, ends, end_pairs, end_values = self._note_time_cache_values
            else:
                notes_sorted = sorted(notes, key=lambda n: (float(n.time), int(n.pitch)))
                starts = [float(n.time) for n in notes_sorted]
                ends = [float(n.time + n.duration) for n in notes_sorted]
                # End-sorted pairs are reused across frames for end-window queries.
                end_pairs = sorted(((ends[i], i) for i in range(len(ends))), key=lambda p: p[0])
                end_values = [p[0] for p in end_pairs]
                self._note_time_cache_key = note_cache_key
                self._note_time_cache_values = (notes_sorted, starts, ends, end_pairs, end_values)

        # Candidate indices: include
        # 1) notes with start in [time_begin, time_end]
        # 2) notes with end in   [time_begin, time_end] (requires ends sorted by value)
        # 3) notes spanning the viewport entirely: start <= time_begin and end >= time_end
        # 4) a small back expansion to catch near-viewport starters (previous behavior)
        lo_start = bisect.bisect_left(starts, time_begin)
        hi_start = bisect.bisect_right(starts, time_end)

        # Query notes whose end falls inside the viewport from cached end-order arrays.
        lo_end_val = bisect.bisect_left(end_values, time_begin)
        hi_end_val = bisect.bisect_right(end_values, time_end)
        by_end_indices = sorted(end_pairs[j][1] for j in range(lo_end_val, hi_end_val))

        viewport_len = float(max(0.0, time_end - time_begin))
        slack = float(op.threshold)
        back_lo = bisect.bisect_left(starts, float(time_begin - viewport_len - slack))

        # Spanning notes: those that started before the viewport and end after it
        span_cut = bisect.bisect_right(starts, time_begin)
        span_indices = [i for i in range(max(0, span_cut)) if ends[i] >= time_end]

        # Merge sorted candidate streams (start-window, end-window, spanning)
        # and deduplicate in one pass to reduce transient set allocations.
        start_range = range(max(0, back_lo), max(0, hi_start))
        candidate_indices: list[int] = []
        prev_idx: int | None = None
        for idx in heapq.merge(start_range, by_end_indices, span_indices):
            if prev_idx is not None and idx == prev_idx:
                continue
            candidate_indices.append(int(idx))
            prev_idx = int(idx)

        # Filtered view: will be further intersection-tested by drawers
        notes_view = [notes_sorted[i] for i in candidate_indices] if candidate_indices else []

        # Group by hand for convenience
        notes_by_hand: dict[str, list] = {}
        for m in notes_view:
            h = str(getattr(m, 'hand', 'l'))
            notes_by_hand.setdefault(h, []).append(m)

        # Beam markers (optional; future use)
        beam_markers = list(getattr(score.events, 'beam', []) or [])
        beam_by_hand: dict[str, list] = {}
        for b in beam_markers:
            h = str(getattr(b, 'hand', 'l'))
            beam_by_hand.setdefault(h, []).append(b)
        for h in beam_by_hand:
            beam_by_hand[h] = sorted(beam_by_hand[h], key=lambda b: float(getattr(b, 'time', 0.0)))

        # Grid helpers: absolute times (ticks) of drawn grid lines across the score.
        # Cache these by base_grid signature to avoid expensive full recomputation every frame.
        base_grid_list = list(getattr(score, 'base_grid', []) or [])
        grid_key_parts: list[tuple[int, int, int, tuple[int, ...]]] = []
        for bg in base_grid_list:
            positions = getattr(bg, 'beat_grouping', None)
            positions_list = list(positions if positions is not None else (getattr(bg, 'beat_grouping', []) or []))
            grid_key_parts.append(
                (
                    int(getattr(bg, 'numerator', 4) or 4),
                    int(getattr(bg, 'denominator', 4) or 4),
                    int(getattr(bg, 'measure_amount', 1) or 1),
                    tuple(round(float(v), 6) for v in positions_list if isinstance(v, (int, float))),
                )
            )
        grid_cache_key = tuple(grid_key_parts)
        if self._grid_time_cache_key == grid_cache_key and self._grid_time_cache_values is not None:
            grid_den_times = self._grid_time_cache_values[0]
            barline_times = self._grid_time_cache_values[1]
        else:
            grid_den_times = []
            barline_times = []
            cur_t = 0.0
            for bg in base_grid_list:
                numer = int(getattr(bg, 'numerator', 4) or 4)
                denom = int(getattr(bg, 'denominator', 4) or 4)
                measure_len_ticks = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
                positions = getattr(bg, 'beat_grouping', None)
                positions_list = list(positions if positions is not None else (getattr(bg, 'beat_grouping', []) or []))
                bar_offsets, grid_offsets = resolve_grid_layer_offsets(positions_list, numer, denom)
                for _ in range(int(bg.measure_amount)):
                    for off in bar_offsets:
                        barline_times.append(float(cur_t + float(off)))
                    for off in grid_offsets:
                        grid_den_times.append(float(cur_t + float(off)))
                    cur_t += measure_len_ticks
            if barline_times:
                barline_times.append(float(cur_t))
            if grid_den_times:
                grid_den_times.append(float(cur_t))
            self._grid_time_cache_key = grid_cache_key
            self._grid_time_cache_values = (grid_den_times, barline_times)

        # Build per-frame arpeggio Y-override lookup so note_drawer can position
        # arpeggiated noteheads along the arpeggio diagonal without any coupling.
        arpeggio_y_overrides: dict[int, float] = {}
        arps_all = list(getattr(score.events, 'arpeggio', []) or [])
        if arps_all:
            op_arp = Operator(float(SHORTEST_DURATION))
            notes_by_pitch: dict[int, list[object]] = {}
            for n in notes_sorted:
                try:
                    pitch_key = int(getattr(n, 'pitch', 0) or 0)
                except Exception:
                    continue
                if pitch_key <= 0:
                    continue
                notes_by_pitch.setdefault(pitch_key, []).append(n)
            margin_mm = float(self.margin or 0.0)
            zpq_arp = float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0) or 25.0)
            semi_arp = float(self.semitone_dist or 0.5)
            layout_arp = getattr(score, 'layout', None)
            width_scale_arp = max(0.05, float(getattr(layout_arp, 'note_width_scaling', 1.0) or 1.0)) if layout_arp is not None else 1.0
            height_scale_arp = max(0.1, float(getattr(layout_arp, 'notehead_height_scaling', 1.0) or 1.0)) if layout_arp is not None else 1.0
            base_tilt_arp = float(getattr(layout_arp, 'notehead_tilt', 0.0) or 0.0) if layout_arp is not None else 0.0
            base_tilt_arp = max(-1.0, min(1.0, base_tilt_arp))

            def _t2mm(ticks: float) -> float:
                return margin_mm + (float(ticks) / float(QUARTER_NOTE_UNIT)) * zpq_arp

            for arp in arps_all:
                try:
                    base_time = float(getattr(arp, 'time', 0.0) or 0.0)
                    rtime1 = float(getattr(arp, 'rtime1', 0.0) or 0.0)
                    rtime2 = float(getattr(arp, 'rtime2', 0.0) or 0.0)
                except Exception:
                    continue
                # Zero/zero means "normal chord" visual state while editing.
                # Skip arpeggio suppression/overrides so note stems redraw immediately.
                if abs(rtime1) <= 1e-9 and abs(rtime2) <= 1e-9:
                    continue
                pitches = [int(p) for p in (getattr(arp, 'note_pitches', []) or []) if int(p) > 0]
                if len(pitches) < 2:
                    continue
                chord_notes: list[object] = []
                for pitch in pitches:
                    matches = notes_by_pitch.get(int(pitch), [])
                    match_note = None
                    for candidate in matches:
                        candidate_time = float(getattr(candidate, 'time', 0.0) or 0.0)
                        if op_arp.eq(candidate_time, base_time):
                            match_note = candidate
                            break
                    if match_note is not None:
                        chord_notes.append(match_note)
                if len(chord_notes) < 2:
                    continue
                chord_sorted = sorted(chord_notes, key=lambda n: int(getattr(n, 'pitch', 0) or 0))
                y_start = _t2mm(base_time + rtime1)
                y_end = _t2mm(base_time + rtime2)
                stem_xs: list[float] = []
                for note_obj in chord_sorted:
                    pitch = int(getattr(note_obj, 'pitch', 0) or 0)
                    x_note = float(self.pitch_to_x(pitch))
                    stem_xs.append(float(x_note))
                x_line_start = float(stem_xs[0])
                x_line_end = float(stem_xs[-1])
                dx_line = float(x_line_end - x_line_start)
                dy_line = float(y_end - y_start)
                m_line = float(dy_line / dx_line) if abs(dx_line) > 1e-9 else 0.0
                angle = math.atan2(float(y_end - y_start), float(x_line_end - x_line_start))
                hand_chord = str(getattr(chord_sorted[0], 'hand', 'l') or 'l') if chord_sorted else 'l'
                # Compute the same hand-dependent stem-end support point as arpeggio drawer.
                if hand_chord == 'l':
                    anchor_note = chord_sorted[-1]
                    anchor_x = float(stem_xs[-1])
                    anchor_y = float(y_end)
                else:
                    anchor_note = chord_sorted[0]
                    anchor_x = float(stem_xs[0])
                    anchor_y = float(y_start)
                hand_anchor = str(getattr(anchor_note, 'hand', 'l') or 'l')
                try:
                    default_black_above_anchor = bool(self._black_note_above_stem(anchor_note, layout_arp))
                except Exception:
                    default_black_above_anchor = True
                spec_anchor = resolve_notehead_spec(anchor_note, default_black_above=default_black_above_anchor)
                is_up_anchor = bool(getattr(spec_anchor, 'is_up', False))
                outline_anchor = sheared_notehead_outline_points(
                    hand=hand_anchor,
                    is_up=is_up_anchor,
                    semitone_space_mm=semi_arp,
                    width_scale=width_scale_arp,
                    height_scale=height_scale_arp,
                    base_tilt=base_tilt_arp,
                    sample_count=128,
                )
                edge_anchor = support_point_from_outline_points(
                    outline_anchor,
                    m_line=m_line,
                    choose_max=bool(is_up_anchor),
                )
                stem_end_x = float(anchor_x + edge_anchor[0])
                stem_end_y = float(anchor_y + edge_anchor[1])
                b_line = float(stem_end_y - (m_line * stem_end_x))
                support_cache: dict[tuple[str, bool], float] = {}
                for i, (note_obj, x_stem) in enumerate(zip(chord_sorted, stem_xs)):
                    nid = int(getattr(note_obj, '_id', 0) or 0)
                    y_line = float((m_line * float(x_stem)) + b_line)
                    # For left hand: highest notehead never offset. For right hand: lowest notehead never offset.
                    if (hand_chord == 'l' and i == len(chord_sorted) - 1) or (hand_chord == 'r' and i == 0):
                        arpeggio_y_overrides[nid] = float(y_end if hand_chord == 'l' else y_start)
                        continue
                    hand_local = str(getattr(note_obj, 'hand', 'l') or 'l')
                    try:
                        default_black_above_local = bool(self._black_note_above_stem(note_obj, layout_arp))
                    except Exception:
                        default_black_above_local = True
                    spec_local = resolve_notehead_spec(note_obj, default_black_above=default_black_above_local)
                    is_up_local = bool(getattr(spec_local, 'is_up', False))
                    cache_key = (hand_local, is_up_local)
                    if cache_key not in support_cache:
                        support_cache[cache_key] = sheared_notehead_support_v(
                            hand=hand_local,
                            is_up=is_up_local,
                            semitone_space_mm=semi_arp,
                            width_scale=width_scale_arp,
                            height_scale=height_scale_arp,
                            base_tilt=base_tilt_arp,
                            m_line=m_line,
                            sample_count=64,
                        )
                    v_support = float(support_cache[cache_key])
                    arpeggio_y_overrides[nid] = float(y_line - v_support)
        self._arpeggio_y_overrides = arpeggio_y_overrides

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

    # ---- Tiny mode (viewport-based) ----
    def update_tiny_mode_from_width(self, device_px_width: float) -> None:
        """Set tiny_mode_stage based on device-pixel width of the editor widget.

        Uses the rendered semitone gap in device pixels as the metric.
        page_w_mm cancels out of the formula, so only the device pixel
        count matters — no hardware DPI query needed.

        semitone_device_px = device_px_width * 2 / (3 * 101)

        Thresholds (device pixels per semitone gap):
          < 2.0  → stage 2 (skip drawing entirely)
          < 4.5  → stage 1 (simplified drawing)
          >= 4.5 → stage 0 (full rendering)
        """
        w = float(device_px_width)
        semitone_px = w * 2.0 / (3.0 * 101.0) # TODO: understand formula
        semi_start_stage1_px = 5
        semi_start_fade_px = 3
        semi_end_fade_px = 2.0

        # Discrete stage selection (kept for behavior/backwards-compatibility)
        stage = 0
        if semitone_px < semi_end_fade_px:
            stage = 2
        elif semitone_px < semi_start_stage1_px:
            stage = 1
        self.tiny_mode_stage = stage

        # Continuous fade factor between stage 1 and 2.
        # Semitone gap >= 4.5 px  -> alpha = 1.0 (no fade)
        # Semitone gap <= 2.0 px  -> alpha = 0.0 (fully transparent)
        # In between: linear interpolation.
        if semitone_px <= semi_end_fade_px:
            alpha = 0.0
        elif semitone_px >= semi_start_fade_px:
            alpha = 1.0
        else:
            alpha = (semitone_px - semi_end_fade_px) / (semi_start_fade_px - semi_end_fade_px)
        self.tiny_mode_alpha = float(max(0.0, min(1.0, alpha)))

    def is_tiny_mode(self) -> bool:
        return bool(int(getattr(self, 'tiny_mode_stage', 0)) > 0)

    def is_tiny_mode_ultra(self) -> bool:
        return bool(int(getattr(self, 'tiny_mode_stage', 0)) >= 2)

    # ---- External controls ----
    def set_snap_size_units(self, units: float) -> None:
        self.snap_size_units = max(0.0, float(units))

    # ---- coordinate calculations ----
    def time_to_mm(self, time: float) -> float:
        """Convert time in ticks to mm position."""
        score: SCORE = self.current_score()
        # Ensure layout metrics are initialized
        if self.margin is None:
            lay = getattr(score, 'layout', None)
            w_mm = float(getattr(lay, 'page_width_mm', 210.0) or 210.0) if lay is not None else 210.0
            self._calculate_layout(float(w_mm))
        # Layout metrics
        zpq = float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0) or 25.0)
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

    def is_horizontal_editor_orientation(self) -> bool:
        """Return True when the editor viewport is using horizontal orientation."""
        try:
            pm = get_preferences_manager()
            orientation = str(pm.get('editor_orientation', 'vertical') or 'vertical').strip().lower()
            return orientation == 'horizontal'
        except Exception:
            return False

    def widget_px_to_output_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Convert widget-local logical pixels to output-space millimeters.

        Output-space is the post-rotation coordinate system seen on screen.
        In horizontal mode, time scroll applies along output X.
        """
        px_per_mm = max(1e-6, float(getattr(self, '_widget_px_per_mm', 1.0) or 1.0))
        view_offset = float(getattr(self, '_view_y_mm_offset', 0.0) or 0.0)
        if self.is_horizontal_editor_orientation():
            out_x_mm = float(x_px) / px_per_mm + view_offset
            out_y_mm = float(y_px) / px_per_mm
            return (out_x_mm, out_y_mm)
        out_x_mm = float(x_px) / px_per_mm
        out_y_mm = float(y_px) / px_per_mm + view_offset
        return (out_x_mm, out_y_mm)

    def widget_px_to_page_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Convert widget-local logical pixels to page-space millimeters.

        Page-space is the unrotated drawing coordinate system used by drawers and
        registered hit rectangles.
        """
        out_x_mm, out_y_mm = self.widget_px_to_output_mm(x_px, y_px)
        if not self.is_horizontal_editor_orientation():
            return (out_x_mm, out_y_mm)
        score: SCORE | None = self.current_score()
        lay = getattr(score, 'layout', None) if score is not None else None
        page_w_mm = float(getattr(lay, 'page_width_mm', 210.0) or 210.0)
        page_x_mm = page_w_mm - float(out_y_mm)
        page_y_mm = float(out_x_mm)
        return (page_x_mm, page_y_mm)

    def widget_px_to_time(self, x_px: float, y_px: float) -> float:
        """Convert widget-local logical pixels to time ticks for the active orientation."""
        _page_x_mm, page_y_mm = self.widget_px_to_page_mm(float(x_px), float(y_px))
        return self.mm_to_time(float(page_y_mm))

    def widget_px_to_pitch(self, x_px: float, y_px: float) -> int:
        """Convert widget-local logical pixels to piano key number for the active orientation."""
        page_x_mm, _page_y_mm = self.widget_px_to_page_mm(float(x_px), float(y_px))
        return self.x_to_pitch_mm(float(page_x_mm))

    def mm_to_time(self, y_mm: float) -> float:
        """Convert Y in mm to time ticks (inverse of time_to_mm)."""
        score: SCORE = self.current_score()
        # Ensure layout metrics are initialized
        if self.margin is None:
            lay = getattr(score, 'layout', None)
            w_mm = float(getattr(lay, 'page_width_mm', 210.0) or 210.0) if lay is not None else 210.0
            self._calculate_layout(float(w_mm))
        zpq = float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0) or 25.0)
        ticks = (float(y_mm) - float(self.margin or 0.0)) / max(1e-6, zpq) * float(QUARTER_NOTE_UNIT)
        return self.clamp_time_to_visible_range(float(ticks))

    def _visible_time_bounds(self) -> tuple[float, float]:
        """Return the minimum/maximum time ticks visible in editor content Y [0..editor_h_mm]."""
        # get current score and ensure layout metrics are initialized
        score: SCORE | None = self.current_score()
        if score is None:
            t = float(0.0)
            return (t, t)
        if self.margin is None:
            lay = getattr(score, 'layout', None)
            w_mm = float(getattr(lay, 'page_width_mm', 210.0) or 210.0) if lay is not None else 210.0
            self._calculate_layout(float(w_mm))
        editor_h_mm = float(getattr(self, 'editor_height', 0.0) or 0.0)
        if editor_h_mm <= 0.0:
            editor_h_mm = float(self._calc_editor_height())
            self.editor_height = editor_h_mm
        
        # Convert editor Y range to time ticks using current zoom. Clamp to ensure min <= max even if zoom is very small or negative.
        zpq = float(getattr(score.app_state, 'zoom_mm_per_quarter', 25.0) or 25.0)
        margin = float(self.margin or 0.0)
        t_min = ((0.0 - margin) / max(1e-6, zpq)) * float(QUARTER_NOTE_UNIT)
        t_max = ((editor_h_mm - margin) / max(1e-6, zpq)) * float(QUARTER_NOTE_UNIT)
        if t_min > t_max:
            t_min, t_max = t_max, t_min
        return (float(t_min), float(t_max))

    def clamp_time_to_visible_range(self, ticks: float) -> float:
        """Clamp time ticks to values that are visible within page-space vertical bounds."""
        t_min, t_max = self._visible_time_bounds()
        return max(float(t_min), min(float(t_max), float(ticks)))

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
        """Snap time ticks to the nearest snap-band boundary.

        Example: with snap size S, values near k*S snap to k*S, and values
        beyond the midpoint between k*S and (k+1)*S snap to (k+1)*S.
        """
        units = max(1e-6, float(self.snap_size_units))
        ratio = float(self.clamp_time_to_visible_range(float(ticks))) / units
        # Nearest-band snapping (round half up) with a tiny epsilon for float stability.
        k = math.floor(ratio + 0.5 + 1e-9)
        return float(self.clamp_time_to_visible_range(float(k * units)))

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
                color=(self.accent_color[0], self.accent_color[1], self.accent_color[2], 0.75),
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
                color=self.accent_color,
                width_mm=.75,
                dash_pattern=[0, 2],
                id=0,
                tags=['cursor'],
            )

            # Right side of stave
            du.add_line(
                margin * 2 + stave_width - 2.0,
                y_mm,
                margin + stave_width - 2.0,
                y_mm,
                color=self.accent_color,
                width_mm=.75,
                dash_pattern=[0, 2.07],
                id=0,
                tags=['cursor'],
            )

            if (isinstance(self._tool, NoteTool)) and (self.time_cursor is not None) and (self.pitch_cursor is not None):
                x_mm = float(self.pitch_to_x(int(self.pitch_cursor)))
                w = float(self.semitone_dist or 0.5)
                layout = self.current_score().layout
                note_width_scale = float(getattr(layout, 'note_width_scaling', 0.75) or 0.75)
                note_width_scale = max(0.05, note_width_scale)
                head_half_w = w * note_width_scale
                h = w * 2
                l = float(layout.note_stem_length_semitone or 3) * float(self.semitone_dist or 0.5)
                # Draw a translucent preview notehead at cursor
                fill_color = self.accent_color if self.pitch_cursor in BLACK_KEYS else self.paper_color
                
                # draw the notehead and stem
                du.add_oval(
                    x_mm - head_half_w,
                    y_mm,
                    x_mm + head_half_w,
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
                    x_mm + l if self.hand_cursor == 'r' else x_mm - l,
                    y_mm,
                    color=self.accent_color,
                    width_mm=0.75,
                    id=0,
                    tags=['cursor'],
                )
                # draw the left hand dot indicator
                if self.hand_cursor == 'l':
                    w = float(self.semitone_dist or 0.5) * 2.0
                    dot_d = w * 0.35
                    cy = y_mm + (w / 2.0)
                    fill = self.paper_color if (self.pitch_cursor in BLACK_KEYS) else self.accent_color
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

                # Accidental preview toggle (A-key cycle): draw only valid accidentals.
                preview_acc = 0
                try:
                    if hasattr(self._tool, 'preview_accidental_for_pitch'):
                        preview_acc = int(self._tool.preview_accidental_for_pitch(int(self.pitch_cursor)))
                except Exception:
                    preview_acc = 0
                if preview_acc != 0:
                    try:
                        target_pitch = int(self.pitch_cursor) + int(preview_acc)
                        x_target = float(self.pitch_to_x(int(target_pitch)))
                        y_anchor = float(y_mm + h)
                        y_target = float(y_anchor + float(self.semitone_dist or 0.5))
                        du.add_line(
                            x_mm,
                            y_anchor,
                            x_target,
                            y_target,
                            color=self.accent_color,
                            width_mm=0.6,
                            id=0,
                            tags=['cursor'],
                        )
                    except Exception:
                        pass

            if (isinstance(self._tool, GraceNoteTool)) and (self.time_cursor is not None) and (self.pitch_cursor is not None):
                x_mm = float(self.pitch_to_x(int(self.pitch_cursor)))
                scale = float(getattr(self.current_score().layout, 'grace_note_scale', 0.75) or 0.75)
                note_width_scale = float(getattr(self.current_score().layout, 'note_width_scaling', 0.75) or 0.75)
                note_width_scale = max(0.05, note_width_scale)
                outline_w = float(
                    getattr(self.current_score().layout, 'grace_note_outline_width_mm', getattr(self.current_score().layout, 'grace_note_outline_width', 0.3))
                    or 0.3
                )
                w = float(self.semitone_dist or 0.5) * scale
                top = y_mm
                bottom = y_mm + (w * 2.0)
                left = x_mm - (w * note_width_scale)
                right = x_mm + (w * note_width_scale)
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

        # --- Velocity sliders (note tool only) ---
        if isinstance(self._tool, NoteTool) and getattr(self._tool, 'velocity_mode', False):
            self._hit_rects = [r for r in self._hit_rects if r.get('type') != 'velocity']
            score = self.current_score()
            if score is not None:
                # Get custom notehead color and convert from RGB (0-255) to normalized RGBA (0-1)
                color_rgb = Style.get_named_rgb('accent_color2', (128, 0, 0))
                velocity_color = (float(color_rgb[0]) / 255.0, float(color_rgb[1]) / 255.0, float(color_rgb[2]) / 255.0, 1.0)
                selected_note_ids = self.get_selected_note_ids_cached(score)
                top_mm = float(self._view_y_mm_offset or 0.0)
                bottom_mm = top_mm + float(self._viewport_h_mm or 0.0)
                bleed_mm = max(4.0, float(self.semitone_dist or 2.5) * 2.0)
                margin = float(self.margin or 12.0)
                stave_width = float(self.stave_width or 120.0)
                view_left = 0.0
                view_right = max(0.0, margin + stave_width + margin)
                max_len = max(2.0, margin * 0.85)
                handle_r = max(1.5, float(self.semitone_dist or 2.5) * 0.45)
                cache = getattr(self, '_draw_cache', None) or {}
                candidate_notes = list(cache.get('notes_view') or []) if isinstance(cache, dict) else []
                if not candidate_notes:
                    candidate_notes = list(getattr(score.events, 'note', []) or [])
                for n in candidate_notes:
                    y_mm = float(self.time_to_mm(float(getattr(n, 'time', 0.0) or 0.0)))
                    hand = str(getattr(n, 'hand', 'l') or 'l')
                    nid = int(getattr(n, '_id', 0) or 0)
                    vel = int(getattr(n, 'velocity', 64) or 0)
                    if y_mm < (top_mm - bleed_mm) or y_mm > (bottom_mm + bleed_mm):
                        continue
                    slider_color = self.accent_color if nid in selected_note_ids else velocity_color
                    ratio = max(0.0, min(1.0, float(vel) / 127.0))
                    dist_from_inner = max_len * (1.0 - ratio)
                    if hand == 'l':
                        x_inner = margin
                        x_outer = x_inner - dist_from_inner
                        x1, x2 = view_left, x_outer
                    else:
                        x_inner = margin + stave_width
                        x_outer = x_inner + dist_from_inner
                        x1, x2 = x_outer, view_right
                    du.add_line(
                        x1,
                        y_mm,
                        x2,
                        y_mm,
                        color=slider_color,
                        width_mm=1.0,
                        dash_pattern=None,
                        id=0,
                        tags=['velocity_slider'],
                    )
                    du.add_oval(
                        x_outer - handle_r,
                        y_mm - handle_r,
                        x_outer + handle_r,
                        y_mm + handle_r,
                        stroke_color=None,
                        fill_color=slider_color,
                        stroke_width_mm=0.0,
                        id=nid,
                        tags=['velocity_slider_handle'],
                    )
                    self.register_hit_rect(
                        'velocity', nid,
                        x_outer - handle_r * 1.4,
                        y_mm - handle_r * 1.4,
                        x_outer + handle_r * 1.4,
                        y_mm + handle_r * 1.4,
                        hand=hand,
                    )
                    tool = self._tool
                    if isinstance(tool, NoteTool) and getattr(tool, '_velocity_dragging', False):
                        tgt = getattr(tool, '_velocity_target', None)
                        if tgt is not None and int(getattr(tgt, '_id', -1) or -1) == nid:
                            val = getattr(tool, '_velocity_display_value', None)
                            if val is not None:
                                offset = -3.0
                                x_text = x_outer
                                du.add_text(
                                    x_text,
                                    y_mm + offset,
                                    str(int(val)),
                                    size_pt=14,
                                    family='Edwin',
                                    color=slider_color,
                                    anchor='s',
                                    id=0,
                                    tags=['velocity_slider_value'],
                                )

        self._draw_selection_overlay(du)

    # ---- Modifier updates ----
    def set_shift_down(self, down: bool) -> None:
        self._shift_down = bool(down)

    def set_ctrl_down(self, down: bool) -> None:
        self._ctrl_down = bool(down)

