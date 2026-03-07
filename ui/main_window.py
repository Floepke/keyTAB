from PySide6 import QtCore, QtGui, QtWidgets
from typing import Optional
import sys, os, time
from pathlib import Path
from utils.file_associations import is_supported_document
from datetime import datetime
from file_model.appstate import AppState
from file_model.file_manager import FileManager
from file_model.analysis import Analysis
from ui.widgets.toolbar_splitter import ToolbarSplitter
from ui.widgets.cairo_views import CairoEditorWidget
from ui.widgets.editor_scrollbar import EditorScrollBar
from ui.widgets.tool_selector import ToolSelectorDock, LEFT_PANEL_PADDING_PX
from ui.widgets.snap_size_selector import SnapSizeDock
from ui.widgets.draw_util import DrawUtil
from ui.widgets.draw_view import DrawUtilView
from ui.about_dialog import AboutDialog
from settings_manager import open_preferences, get_preferences_manager
from appdata_manager import get_appdata_manager
from utils.CONSTANT import UTILS_SAVE_DIR, QUARTER_NOTE_UNIT
from engraver.engraver import Engraver
from editor.tool_manager import ToolManager
from editor.editor import Editor


class _SlurOptimizerWorker(QtCore.QObject):
    finished = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def __init__(self, score, settings: dict, optimize_fn):
        super().__init__()
        self._score = score
        self._settings = dict(settings or {})
        self._optimize_fn = optimize_fn

    @QtCore.Slot()
    def run(self) -> None:
        try:
            def _int_setting(key: str, default: int) -> int:
                value = self._settings.get(key, default)
                if value is None:
                    value = default
                return int(value)

            def _float_setting(key: str, default: float) -> float:
                value = self._settings.get(key, default)
                if value is None:
                    value = default
                return float(value)

            stats = self._optimize_fn(
                self._score,
                hit_test_finetune=_int_setting("hit_test_finetune", 64),
                anchor_range=_int_setting("anchor_range", 10),
                control_range=_int_setting("control_range", 16),
                max_iterations=_int_setting("max_iterations", 3),
                max_slurs_from_start=_int_setting("max_slurs_from_start", 12),
                include_stave_lines=bool(self._settings.get("include_stave_lines", True)),
                include_beams=bool(self._settings.get("include_beams", True)),
                pointiness_weight=_float_setting("pointiness_weight", 0.5),
                symmetry_weight=_float_setting("symmetry_weight", 0.5),
                pointiness_bonus_points=_float_setting("pointiness_bonus_points", 100.0),
                symmetry_bonus_points=_float_setting("symmetry_bonus_points", 100.0),
                antisymmetry_penalty_points=_float_setting("antisymmetry_penalty_points", 100.0),
                straight_line_penalty_points=_float_setting("straight_line_penalty_points", 100.0),
                neighbor_connection_bonus_points=_float_setting("neighbor_connection_bonus_points", 100.0),
                control_y_pass_penalty_points=_float_setting("control_y_pass_penalty_points", 1000.0),
            )
            self.finished.emit(dict(stats or {}))
        except Exception as exc:
            self.failed.emit(str(exc))


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("keyTAB - new project (unsaved)")
        self.resize(1200, 800)
        self.setAcceptDrops(True)
        # Ensure player attribute always exists
        self.player = None
        self._player_config: tuple[str, str] | None = None
        self._left_panel_width_frozen = False
        self._editor_scroll_step_logical_px: int = 1
        self._slur_opt_thread: QtCore.QThread | None = None
        self._slur_opt_worker: _SlurOptimizerWorker | None = None
        self._slur_opt_progress: QtWidgets.QProgressDialog | None = None
        self._slur_opt_settings: dict | None = None
        self._slur_opt_started_at: float | None = None
        self._slur_opt_dialog = None

        # File management
        self.file_manager = FileManager(self)

        # View options
        try:
            pm = get_preferences_manager()
            self._center_playhead_enabled = bool(
                pm.get("focus_on_playhead_during_playback", pm.get("center_view_on_playhead", True))
            )
        except Exception:
            self._center_playhead_enabled = True
        
        # Install error-backup hook early so any unhandled exception triggers a backup
        self.file_manager.install_error_backup_hook()

        # Periodic autosave (session + project) to reduce per-action latency
        self._autosave_timer = QtCore.QTimer(self)
        self._autosave_timer.timeout.connect(lambda: self.file_manager.autosave_all())
        self._apply_autosave_preferences()

        self._create_menus()

        self.splitter = ToolbarSplitter(QtCore.Qt.Orientation.Horizontal)
        # Hide handle indicator/grip but keep dragging functional
        self.splitter.setStyleSheet(
            "QSplitter::handle { background: transparent; image: none; }\n"
            "QSplitter::handle:hover { background: transparent; }"
        )
        
        # Editor view with external scrollbar for static viewport scrolling
        self.editor_canvas = CairoEditorWidget()
        self.editor_vscroll = EditorScrollBar(QtCore.Qt.Orientation.Vertical)
        self._editor_metric_px_per_mm: float = 1.0
        self._editor_metric_dpr: float = 1.0
        self._editor_metric_viewport_logical_px: int = 0
        self._configure_editor_scrollbar()
        
        # For external code, expose the canvas under the same name
        self.editor_canvas = self.editor_canvas

        self.du = DrawUtil()
        self.du.new_page(width_mm=210, height_mm=297)

        self.print_view = DrawUtilView(self.du)
        self.print_view.set_page_turn_callbacks(self._previous_page, self._next_page)
        
        # Engraver instance (single)
        self.engraver = Engraver(self.du, self)

        # When engraving completes, refresh analysis then re-render the print view
        self.engraver.engraved.connect(self._on_engraver_finished)
        
        # Startup restore: prefer opening the last saved project; else restore unsaved session; else new
        self._session_restore_mode: bool = False
        adm2 = None
        was_saved = False
        saved_path = ""
        try:
            adm2 = get_appdata_manager()
            was_saved = bool(adm2.get("last_session_saved", False))
            saved_path = str(adm2.get("last_session_path", "") or "")
        except Exception:
            pass
        session_path = Path(UTILS_SAVE_DIR) / "session.piano"
        opened = False
        status_msg = ""

        def _try_open_path_with_retries(path_text: str, retries: int = 12, delay_sec: float = 0.25):
            p = str(path_text or "").strip()
            if not p:
                return None
            candidate = Path(p).expanduser()
            for attempt in range(max(1, int(retries))):
                try:
                    if candidate.exists():
                        sc_try = self.file_manager.open_path(str(candidate))
                        if sc_try is not None:
                            return sc_try
                except Exception:
                    pass
                if attempt < (max(1, int(retries)) - 1):
                    try:
                        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)
                    except Exception:
                        pass
                    time.sleep(max(0.0, float(delay_sec)))
            return None

        # Always try real project paths first (handles delayed cloud mounts).
        if saved_path:
            sc = _try_open_path_with_retries(saved_path, retries=16, delay_sec=0.25)
            if sc is not None:
                opened = True
                self._session_restore_mode = False
                status_msg = f"Opened last saved project: {saved_path}"

        if not opened:
            try:
                if adm2 is None:
                    adm2 = get_appdata_manager()
                last_path = str(adm2.get("last_opened_file", "") or "")
            except Exception:
                last_path = ""
            if last_path and str(last_path) != str(saved_path):
                sc = _try_open_path_with_retries(last_path, retries=16, delay_sec=0.25)
                if sc is not None:
                    opened = True
                    self._session_restore_mode = False
                    status_msg = f"Opened last project: {last_path}"

        if not opened:
            # If the last session wasn't saved, try restoring the session snapshot
            restored = False
            try:
                restored = self.file_manager.load_session_if_available()
            except Exception:
                restored = False
            if not restored:
                # Nothing to restore/open; start fresh
                self.file_manager.new()
                self._session_restore_mode = False
                status_msg = "Started new project"
            else:
                self._session_restore_mode = True
                status_msg = "Restored unsaved session (session.piano mode)"

        # Initialize page navigation from persisted app state before first engrave.
        try:
            app_state = self._resolve_app_state_defaults()
            self._page_counter = max(0, int(getattr(app_state, 'print_view_page_index', 0) or 0))
        except Exception:
            self._page_counter = 0

        # Provide initial score to engrave and update titlebar (delay first engrave)
        self._refresh_views_from_score(delay_engrave_ms=1000)
        self._startup_status_message = str(status_msg or "")

        self._update_title()

        # Build a container with the canvas and external vertical scrollbar
        editor_container = QtWidgets.QWidget()
        editor_layout = QtWidgets.QHBoxLayout(editor_container)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(0)
        editor_layout.addWidget(self.editor_canvas, stretch=1)
        editor_layout.addWidget(self.editor_vscroll, stretch=0)
        self.splitter.addWidget(editor_container)
        self.splitter.addWidget(self.print_view)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 2)
        self.setCentralWidget(self.splitter)
        # Status bar for lightweight app messages and default path/dirty info
        self._status_default_text = ""
        try:
            self._statusbar = QtWidgets.QStatusBar(self)
            self.setStatusBar(self._statusbar)
            try:
                self._statusbar.messageChanged.connect(self._on_status_message_changed)
            except Exception:
                pass
            self._show_status_default()
            try:
                if self._startup_status_message:
                    self._status(self._startup_status_message, 7000)
            except Exception:
                pass
        except Exception:
            self._statusbar = None
        # Ensure the editor is the main focus target
        try:
            self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
            self.setFocusProxy(self.editor_canvas)
            self.editor_canvas.setFocus()
        except Exception:
            pass
        # Hide dock sizer handles and prevent resize cursor changes (docks are fixed-size)
        self.setStyleSheet(
            "QMainWindow::separator { width: 0px; height: 0px; background: transparent; }\n"
            "QMainWindow::separator:hover { background: transparent; }"
        )
        # Place Snap Size dock above the Tool Selector dock on the left
        self.snap_dock = SnapSizeDock(self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.snap_dock)
        self.tool_dock = ToolSelectorDock(self)
        self.addDockWidget(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea, self.tool_dock)
        self._tooltip_redirect_source: QtCore.QObject | None = None
        QtWidgets.QApplication.instance().installEventFilter(self)
        # Stack vertically: snap (top) above tool selector (bottom)
        self.splitDockWidget(self.snap_dock, self.tool_dock, QtCore.Qt.Orientation.Vertical)
        # Avoid docks stealing focus from the editor
        try:
            self.snap_dock.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            self.tool_dock.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            self.print_view.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            self.editor_vscroll.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        except Exception:
            pass
        # Wiring
        # Editor + ToolManager
        self.tool_manager = ToolManager(self.splitter)
        self.editor_controller = Editor(self.tool_manager)
        self.editor_canvas.set_editor(self.editor_controller)
        # Provide widget reference to editor for explicit full redraws
        try:
            self.editor_controller.widget = self.editor_canvas
        except Exception:
            pass
        # Provide editor to ToolManager so tools can use editor wrappers
        try:
            self.tool_manager.set_editor(self.editor_controller)
        except Exception:
            pass
        # Provide FileManager to editor (for undo snapshots)
        self.editor_controller.set_file_manager(self.file_manager)
        try:
            self.editor_controller.score_changed.connect(self._on_score_changed)
        except Exception:
            pass

        # Coalesce model-change engrave requests so input handlers return quickly.
        self._score_change_engrave_timer = QtCore.QTimer(self)
        self._score_change_engrave_timer.setSingleShot(True)
        self._score_change_engrave_timer.setInterval(1)
        self._score_change_engrave_timer.timeout.connect(self._flush_score_change_engrave)

        # Wire tool selector to Editor controller
        self.tool_dock.selector.toolSelected.connect(self.editor_controller.set_tool_by_name)
        # Also persist tool selection to appdata
        try:
            self.tool_dock.selector.toolSelected.connect(self._on_tool_selected)
        except Exception:
            pass

        # Persist snap changes and update editor
        self.snap_dock.selector.snapChanged.connect(self._on_snap_changed)
        # Restore tool and snap size from project app state (fallback to appdata defaults)
        try:
            self._restore_app_state_from_score()
        except Exception:
            try:
                self.editor_controller.set_tool_by_name('note')
            except Exception:
                pass
        # 'Fit' button on splitter handle triggers fit action
        self.splitter.fitRequested.connect(self._fit_print_view_to_page)
        self.splitter.fitRequested.connect(self._force_redraw)
        # Default toolbar actions
        self.splitter.nextRequested.connect(self._next_page)
        self.splitter.nextRequested.connect(self._force_redraw)
        self.splitter.previousRequested.connect(self._previous_page)
        self.splitter.previousRequested.connect(self._force_redraw)
        self.splitter.engraveRequested.connect(self._engrave_now)
        self.splitter.engraveRequested.connect(self._force_redraw)
        self.splitter.playRequested.connect(self._play_midi)
        self.splitter.playRequested.connect(self._force_redraw)
        self.splitter.stopRequested.connect(self._stop_midi)
        self.splitter.stopRequested.connect(self._force_redraw)
        self.splitter.styleRequested.connect(self._open_style_dialog)
        self.splitter.infoRequested.connect(self._open_info_dialog)
        self.splitter.lineBreakRequested.connect(self._open_line_break_dialog)
        # Contextual tool buttons should also force redraw
        self.splitter.contextButtonClicked.connect(lambda *_: self._force_redraw())
        # Fit state tracking
        self.is_fit = False
        self.is_startup = True
        # Defer Edwin font prompt until explicitly scheduled by the app (after AppImage install prompt)
        self._edwin_prompt_armed = False

        # Restore splitter sizes from last session if available; else fall back to fit
        adm = get_appdata_manager()
        saved_sizes = adm.get("splitter_sizes", None)
        if isinstance(saved_sizes, list) and len(saved_sizes) == 2 and sum(int(v) for v in saved_sizes) > 0:
            # Apply after layout has settled
            QtCore.QTimer.singleShot(0, lambda: self.splitter.setSizes([int(saved_sizes[0]), int(saved_sizes[1])]))
            # Disable startup fit behavior
            self.is_startup = False
        else:
            # Fit print view to page on startup (schedule to catch late geometry)
            QtCore.QTimer.singleShot(200, self._fit_print_view_to_page)
        # Also request an initial render
        QtCore.QTimer.singleShot(0, self.print_view.request_render)
        # Strip demo timers
        # Center the window on the primary screen shortly after show
        QtCore.QTimer.singleShot(0, self._center_on_primary)

        # After docks are visible, adjust their sizes to fit
        QtCore.QTimer.singleShot(0, self._adjust_docks_to_fit)

        # Page navigation state
        self._page_counter = max(0, int(getattr(self, '_page_counter', 0) or 0))

        # Connect external scrollbar to the editor canvas
        try:
            self.editor_canvas.viewportMetricsChanged.connect(self._on_editor_metrics)
            self.editor_vscroll.valueChanged.connect(self._on_editor_scroll_changed)
            # Keep external scrollbar in sync with wheel-driven scroll from the editor
            self.editor_canvas.scrollLogicalPxChanged.connect(self.editor_vscroll.setValue)
            # Persist app state only on wheel scrolling inside the editor view
            self.editor_canvas.scrollWheelUsed.connect(self._schedule_app_state_save)
        except Exception:
            pass
        # Restore last scroll position once viewport metrics are available
        try:
            app_state = self._resolve_app_state_defaults()
            self._pending_scroll_restore = int(getattr(app_state, "editor_scroll_pos", 0) or 0)
        except Exception:
            self._pending_scroll_restore = 0

        # Initialize player (MIDI or Synth)
        try:
            self._ensure_player()
            try:
                if hasattr(self, 'editor_controller') and self.editor_controller is not None:
                    self.editor_controller.set_player(self.player)
            except Exception:
                pass
        except Exception:
            # Player initialization is optional at startup; keep attribute defined
            self.player = None
        # Playhead overlay timer (60 Hz)
        try:
            self._playhead_timer = QtCore.QTimer(self)
            self._playhead_timer.setTimerType(QtCore.Qt.TimerType.PreciseTimer)
            self._playhead_timer.setInterval(20)
            self._playhead_timer.timeout.connect(self._update_playhead_overlay)
        except Exception:
            self._playhead_timer = None
        # Synth configuration no longer applies; FluidSynth handles playback directly

    def dragEnterEvent(self, event: QtGui.QDragEnterEvent) -> None:
        if self._drop_paths_from_mime(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event: QtGui.QDragMoveEvent) -> None:
        if self._drop_paths_from_mime(event.mimeData()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event: QtGui.QDropEvent) -> None:
        paths = self._drop_paths_from_mime(event.mimeData())
        if not paths:
            event.ignore()
            return
        self.open_documents_from_paths(paths, confirm_dirty=True)
        event.acceptProposedAction()

    def _drop_paths_from_mime(self, mime: QtCore.QMimeData) -> list[str]:
        if not mime.hasUrls():
            return []
        paths: list[str] = []
        for url in mime.urls():
            path = url.toLocalFile() if url.isLocalFile() else ""
            if path and is_supported_document(path):
                paths.append(path)
        return paths

    def keyPressEvent(self, ev: QtGui.QKeyEvent) -> None:
        # Space toggles play/stop from the editor's time cursor (with note chasing)
        try:
            if ev.key() == QtCore.Qt.Key_Space:
                if not hasattr(self, 'player') or self.player is None:
                    from midi.player import Player
                    self.player = Player()
                if hasattr(self.player, 'is_playing') and self.player.is_playing():
                    self.player.stop()
                    # Clear playhead overlay immediately on stop
                    try:
                        self._clear_playhead_overlay()
                    except Exception:
                        pass
                else:
                    # Get start time from editor time cursor; default to 0.0
                    try:
                        t_units = float(getattr(self.editor_controller, 'time_cursor', 0.0) or 0.0)
                    except Exception:
                        t_units = 0.0
                    # Use unified helper to handle port selection prompt and retry
                    self._play_midi_with_prompt(start_units=t_units)
                ev.accept()
                return
        except Exception:
            pass
        # 'S' opens Style dialog when focus is not on a text input
        try:
            if ev.key() == QtCore.Qt.Key_S and ev.modifiers() == QtCore.Qt.KeyboardModifier.NoModifier:
                fw = QtWidgets.QApplication.focusWidget()
                if isinstance(fw, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)):
                    pass
                else:
                    self._open_style_dialog()
                    ev.accept()
                    return
        except Exception:
            pass
        # 'I' opens Info dialog when focus is not on a text input
        try:
            if ev.key() == QtCore.Qt.Key_I and ev.modifiers() == QtCore.Qt.KeyboardModifier.NoModifier:
                fw = QtWidgets.QApplication.focusWidget()
                if isinstance(fw, (QtWidgets.QLineEdit, QtWidgets.QTextEdit, QtWidgets.QPlainTextEdit)):
                    pass
                else:
                    self._open_info_dialog()
                    ev.accept()
                    return
        except Exception:
            pass
        super().keyPressEvent(ev)

    def _create_menus(self) -> None:
        menubar = self.menuBar()
        # Ensure macOS uses the native system menubar
        if sys.platform == "darwin":
            menubar.setNativeMenuBar(True)

        # Create menus in normal left-to-right order (File, Edit, Selection, Document, Tools, View, Playback, About)
        file_menu = menubar.addMenu("&File")
        edit_menu = menubar.addMenu("&Edit")
        view_menu = menubar.addMenu("&View")
        selection_menu = menubar.addMenu("&Selection")
        document_menu = menubar.addMenu("&Document")
        tools_menu = menubar.addMenu("&Tools")
        playback_menu = menubar.addMenu("&Playback")
        help_menu = menubar.addMenu("&About")
        for menu in (file_menu, edit_menu, selection_menu, view_menu, document_menu, tools_menu, playback_menu, help_menu):
            menu.setToolTipsVisible(True)

        # File actions
        new_act = QtGui.QAction("New", self)
        new_act.setToolTip("Create a new project.")
        open_act = QtGui.QAction("Load...", self)
        open_act.setToolTip("Open an existing project file.")
        save_act = QtGui.QAction("Save", self)
        save_act.setToolTip("Save the current project.")
        save_as_act = QtGui.QAction("Save As...", self)
        save_as_act.setToolTip("Save the current project under a new file name.")
        exit_act = QtGui.QAction("Exit", self)
        exit_act.setToolTip("Exit the application.")
        try:
            exit_act.setShortcut(QtGui.QKeySequence("Escape"))
        except Exception:
            pass
        exit_act.triggered.connect(self.close)

        new_act.setShortcut(QtGui.QKeySequence.StandardKey.New)
        open_act.setShortcut(QtGui.QKeySequence.StandardKey.Open)
        save_act.setShortcut(QtGui.QKeySequence.StandardKey.Save)
        save_as_act.setShortcut(QtGui.QKeySequence.StandardKey.SaveAs)

        file_menu.addAction(new_act)
        file_menu.addAction(open_act)
        file_menu.addAction(save_act)
        file_menu.addAction(save_as_act)
        self._recent_menu = file_menu.addMenu("Recent Files")
        self._recent_menu.setToolTipsVisible(True)
        file_menu.addSeparator()

        style_act = QtGui.QAction("Style...", self)
        style_act.setToolTip("Open appearance settings for the score.")
        style_act.setShortcut(QtGui.QKeySequence("S"))
        style_act.triggered.connect(self._open_style_dialog)
        info_act = QtGui.QAction("Info...", self)
        info_act.setToolTip("Open title and metadata settings.")
        info_act.setShortcut(QtGui.QKeySequence("I"))
        info_act.triggered.connect(self._open_info_dialog)
        line_break_act = QtGui.QAction("Line Breaks...", self)
        line_break_act.setToolTip("Open line break and page break settings.")
        line_break_act.setShortcut(QtGui.QKeySequence("L"))
        line_break_act.triggered.connect(self._open_line_break_dialog)
        slur_bow_opt_act = QtGui.QAction("Slur bow optimizer", self)
        slur_bow_opt_act.setToolTip("Optimize slur bow X handles to reduce collisions.")

        document_menu.addAction(style_act)
        document_menu.addAction(info_act)
        document_menu.addAction(line_break_act)
        document_menu.addSeparator()
        tools_menu.addAction(slur_bow_opt_act)

        export_pdf_act = QtGui.QAction("Export PDF...", self)
        export_pdf_act.setToolTip("Export the current score as a PDF document.")
        export_pdf_act.setShortcut(QtGui.QKeySequence("Ctrl+E"))
        export_pdf_act.triggered.connect(self._export_pdf)
        file_menu.addAction(export_pdf_act)

        # Playback menu
        self._playback_menu = playback_menu
        self._playback_mode_group = QtGui.QActionGroup(self)
        self._playback_mode_group.setExclusive(True)

        self._playback_mode_system_action = QtGui.QAction(self._playback_system_label(), self)
        self._playback_mode_system_action.setToolTip("Use the system playback backend.")
        self._playback_mode_system_action.setCheckable(True)
        self._playback_mode_system_action.triggered.connect(lambda checked: self._set_playback_mode('system') if checked else None)
        playback_menu.addAction(self._playback_mode_system_action)
        self._playback_mode_group.addAction(self._playback_mode_system_action)

        self._playback_mode_external_action = QtGui.QAction("Playback using External MIDI port", self)
        self._playback_mode_external_action.setToolTip("Use an external MIDI output port for playback.")
        self._playback_mode_external_action.setCheckable(True)
        self._playback_mode_external_action.triggered.connect(lambda checked: self._set_playback_mode('external') if checked else None)
        playback_menu.addAction(self._playback_mode_external_action)
        self._playback_mode_group.addAction(self._playback_mode_external_action)

        playback_menu.addSeparator()
        self._midi_port_menu = playback_menu.addMenu("MIDI port")
        self._midi_port_menu.setToolTipsVisible(True)
        self._midi_port_menu.aboutToShow.connect(self._rebuild_midi_port_menu)
        self._rebuild_midi_port_menu()
        playback_menu.addSeparator()

        test_tone_act = QtGui.QAction("Play Test Tone", self)
        test_tone_act.setToolTip("Play a short test tone.")
        test_tone_act.triggered.connect(self._play_test_tone)
        playback_menu.addAction(test_tone_act)

        if sys.platform.startswith("linux"):
            playback_menu.addSeparator()
            select_sf_act = QtGui.QAction("Select Custom SoundFont (.sf2/.sf3) for FluidSynth", self)
            select_sf_act.setToolTip("Select a custom SoundFont file for FluidSynth playback.")
            select_sf_act.triggered.connect(lambda: self._prompt_for_soundfont(force_dialog=True))
            playback_menu.addAction(select_sf_act)

            unset_sf_act = QtGui.QAction("Use Default FluidSynth SoundFont", self)
            unset_sf_act.setToolTip("Switch back to the default FluidSynth SoundFont.")
            unset_sf_act.triggered.connect(self._unset_soundfont)
            playback_menu.addAction(unset_sf_act)

        self._set_playback_mode(str(self._get_playback_mode_from_appdata() or 'system'), show_status=False)

        about_act = QtGui.QAction("About keyTAB", self)
        about_act.setToolTip("Show information about keyTAB.")
        about_act.triggered.connect(self._open_about_dialog)
        about_qt_act = QtGui.QAction("About Qt", self)
        about_qt_act.setToolTip("Show information about the Qt framework.")
        about_qt_act.triggered.connect(lambda: QtWidgets.QMessageBox.aboutQt(self))
        help_menu.addAction(about_act)
        help_menu.addSeparator()
        help_menu.addAction(about_qt_act)

        try:
            self._refresh_recent_files_menu()
        except Exception:
            pass

        file_menu.addSeparator()
        file_menu.addAction(exit_act)

        # Edit actions
        undo_act = QtGui.QAction("Undo", self)
        undo_act.setToolTip("Undo the last editing action.")
        undo_act.setShortcut(QtGui.QKeySequence.StandardKey.Undo)
        redo_act = QtGui.QAction("Redo", self)
        redo_act.setToolTip("Redo the last undone editing action.")
        # Use platform-aware Redo shortcut to avoid ambiguity; explicit combos handled in editor
        try:
            redo_act.setShortcut(QtGui.QKeySequence.StandardKey.Redo)
        except Exception:
            pass
        edit_menu.addAction(undo_act)
        edit_menu.addAction(redo_act)
        # Cut/Copy/Paste actions (platform-aware shortcuts)
        cut_act = QtGui.QAction("Cut", self)
        cut_act.setToolTip("Cut the current selection.")
        cut_act.setShortcut(QtGui.QKeySequence.StandardKey.Cut)
        copy_act = QtGui.QAction("Copy", self)
        copy_act.setToolTip("Copy the current selection.")
        copy_act.setShortcut(QtGui.QKeySequence.StandardKey.Copy)
        paste_act = QtGui.QAction("Paste", self)
        paste_act.setToolTip("Paste clipboard content.")
        paste_act.setShortcut(QtGui.QKeySequence.StandardKey.Paste)
        edit_menu.addSeparator()
        edit_menu.addAction(cut_act)
        edit_menu.addAction(copy_act)
        edit_menu.addAction(paste_act)
        # Delete selection action with visible shortcuts (Delete, Backspace)
        delete_act = QtGui.QAction("Delete", self)
        delete_act.setToolTip("Delete the current selection.")
        try:
            delete_act.setShortcuts([
                QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Delete),
                QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Backspace)
            ])
        except Exception:
            # Fallback: set single Delete shortcut
            try:
                delete_act.setShortcut(QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.Delete))
            except Exception:
                pass
        edit_menu.addAction(delete_act)
        edit_menu.addSeparator()

        # Selection menu (discoverability for selection shortcuts/actions)
        select_all_act = QtGui.QAction("Select All", self)
        select_all_act.setToolTip("Select all editable events.")
        select_all_act.setShortcut(QtGui.QKeySequence.StandardKey.SelectAll)

        transpose_left_act = QtGui.QAction("Transpose -1 Semitone", self)
        transpose_left_act.setToolTip("Transpose Selection Down by One Semitone.")
        transpose_left_act.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left))
        transpose_left_act.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetShortcut)

        transpose_right_act = QtGui.QAction("Transpose +1 Semitone", self)
        transpose_right_act.setToolTip("Transpose Selection Up by One Semitone.")
        transpose_right_act.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right))
        transpose_right_act.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetShortcut)

        shift_earlier_act = QtGui.QAction("Move Earlier by Snap Band", self)
        shift_earlier_act.setToolTip("Move Selection Earlier by One Snap Band.")
        shift_earlier_act.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Up))
        shift_earlier_act.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetShortcut)

        shift_later_act = QtGui.QAction("Move Later by Snap Band", self)
        shift_later_act.setToolTip("Move Selection Later by One Snap Band.")
        shift_later_act.setShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Down))
        shift_later_act.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetShortcut)

        quantize_act = QtGui.QAction("Quantize Starts and Ends on Snap Band", self)
        quantize_act.setToolTip("Quantize Selection Starts and Ends to the Current Snap Band.")
        quantize_act.setShortcut(QtGui.QKeySequence("Q"))
        quantize_act.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
        quantize_start_act = QtGui.QAction("Quantize Starts on Snap Band", self)
        quantize_start_act.setToolTip("Quantize Selection Starts to the Current Snap Band.")
        quantize_end_act = QtGui.QAction("Quantize Ends on Snap Band", self)
        quantize_end_act.setToolTip("Quantize Selection Ends to the Current Snap Band.")

        selection_menu.addAction(select_all_act)
        selection_menu.addSeparator()
        selection_menu.addAction(transpose_left_act)
        selection_menu.addAction(transpose_right_act)
        selection_menu.addAction(shift_earlier_act)
        selection_menu.addAction(shift_later_act)
        selection_menu.addSeparator()
        selection_menu.addAction(quantize_act)
        selection_menu.addAction(quantize_start_act)
        selection_menu.addAction(quantize_end_act)
        # Separator between Delete and Preferences
        edit_menu.addSeparator()
        prefs_act = QtGui.QAction("Preferences…", self)
        prefs_act.setToolTip("Open application preferences.")
        prefs_act.triggered.connect(self._open_preferences)
        edit_menu.addAction(prefs_act)
        
        # View actions
        zoom_in_act = QtGui.QAction("Zoom In", self)
        zoom_in_act.setToolTip("Zoom in on the editor view.")
        try:
            zoom_in_act.setShortcuts([
                QtGui.QKeySequence("="),
                QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.ZoomIn)
            ])
        except Exception:
            try:
                zoom_in_act.setShortcut(QtGui.QKeySequence("="))
            except Exception:
                pass
        zoom_out_act = QtGui.QAction("Zoom Out", self)
        zoom_out_act.setToolTip("Zoom out from the editor view.")
        try:
            zoom_out_act.setShortcuts([
                QtGui.QKeySequence("-"),
                QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.ZoomOut)
            ])
        except Exception:
            try:
                zoom_out_act.setShortcut(QtGui.QKeySequence("-"))
            except Exception:
                pass
        view_menu.addSeparator()
        full_screen_act = QtGui.QAction("Full Screen", self)
        full_screen_act.setToolTip("Toggle full screen mode.")
        full_screen_act.setShortcut(QtGui.QKeySequence("F11"))
        full_screen_act.setCheckable(True)
        view_menu.addAction(zoom_in_act)
        view_menu.addAction(zoom_out_act)
        view_menu.addSeparator()
        view_menu.addAction(full_screen_act)

        # Wire up triggers
        new_act.triggered.connect(self._file_new)
        open_act.triggered.connect(self._file_open)
        save_act.triggered.connect(self._file_save)
        save_as_act.triggered.connect(self._file_save_as)
        undo_act.triggered.connect(self._edit_undo)
        redo_act.triggered.connect(self._edit_redo)
        cut_act.triggered.connect(self._edit_cut)
        copy_act.triggered.connect(self._edit_copy)
        paste_act.triggered.connect(self._edit_paste)
        delete_act.triggered.connect(self._edit_delete)
        select_all_act.triggered.connect(self._selection_select_all)
        transpose_left_act.triggered.connect(lambda: self._selection_transpose(-1))
        transpose_right_act.triggered.connect(lambda: self._selection_transpose(1))
        shift_earlier_act.triggered.connect(lambda: self._selection_shift(-1.0))
        shift_later_act.triggered.connect(lambda: self._selection_shift(1.0))
        quantize_act.triggered.connect(lambda: self._selection_quantize('start/end'))
        quantize_start_act.triggered.connect(lambda: self._selection_quantize('start'))
        quantize_end_act.triggered.connect(lambda: self._selection_quantize('end'))
        zoom_in_act.triggered.connect(lambda: self._zoom_editor(1))
        zoom_out_act.triggered.connect(lambda: self._zoom_editor(-1))
        full_screen_act.triggered.connect(self._toggle_full_screen)
        slur_bow_opt_act.triggered.connect(self._run_slur_bow_optimizer)

        # Keep reference for state sync
        self._full_screen_act = full_screen_act

        # ---- Clock label manually positioned at menubar's right edge ----
        self._clock_label = QtWidgets.QLabel(menubar)
        self._clock_label.setObjectName("menuClock")
        # Match menubar font/palette for native look
        try:
            self._clock_label.setFont(menubar.font())
            self._clock_label.setPalette(menubar.palette())
        except Exception:
            pass
        # Vertically center text within the menubar height
        self._clock_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignVCenter)
        # Non-interactive
        self._clock_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._clock_label.setContentsMargins(0, 0, 0, 0)
        self._clock_label.setStyleSheet("")
        self._update_clock()
        # Update every second
        self._clock_timer = QtCore.QTimer(self)
        self._clock_timer.setInterval(1000)
        self._clock_timer.timeout.connect(self._update_clock)
        self._clock_timer.start()
        # Keep position updated on menubar resize
        menubar.installEventFilter(self)
        QtCore.QTimer.singleShot(0, self._position_clock)

    def _configure_editor_scrollbar(self) -> None:
        extent = int(self.style().pixelMetric(QtWidgets.QStyle.PixelMetric.PM_ScrollBarExtent))
        self.editor_vscroll.setFixedWidth(max(12, int(extent * 2)))
        self.editor_vscroll.setToolTip("Editor scrollbar. Drag to scroll. Click outside the scrollbar handle to jump. Hover outside the scrollbar handle to preview the current destination measure.")
        self.editor_vscroll.set_tooltip_provider(self._editor_scrollbar_tooltip_text)
        self.editor_vscroll.set_measure_index_provider(self._editor_scrollbar_measure_index_for_predicted_top)
        self.editor_vscroll.set_jump_target_provider(self._editor_scrollbar_jump_target_for_predicted_top)

    def _score_measure_starts_units(self) -> list[float]:
        ed = self.editor_controller if hasattr(self, 'editor_controller') else None
        if ed is None or not hasattr(ed, '_get_barline_positions'):
            return [0.0]
        starts = list(ed._get_barline_positions() or [])
        if not starts:
            return [0.0]
        return [float(v) for v in starts]

    def _editor_scrollbar_measure_index_for_predicted_top(self, predicted_top_value: int) -> int:
        starts = self._score_measure_starts_units()
        measure_count = max(1, len(starts))
        minimum = int(self.editor_vscroll.minimum())
        maximum = int(self.editor_vscroll.maximum())
        if maximum <= minimum:
            return 0
        ratio = (float(predicted_top_value) - float(minimum)) / float(maximum - minimum)
        ratio = max(0.0, min(1.0, ratio))
        idx = int(round(ratio * float(measure_count - 1)))
        return max(0, min(measure_count - 1, idx))

    def _editor_scrollbar_jump_target_for_predicted_top(self, predicted_top_value: int) -> int:
        ed = self.editor_controller if hasattr(self, 'editor_controller') else None
        if ed is None:
            return int(predicted_top_value)

        starts = self._score_measure_starts_units()
        measure_count = max(1, len(starts))
        if measure_count <= 0:
            return int(predicted_top_value)

        measure_idx = self._editor_scrollbar_measure_index_for_predicted_top(predicted_top_value)
        if measure_idx + 1 < len(starts):
            start_units = float(starts[measure_idx])
            end_units = float(starts[measure_idx + 1])
        elif len(starts) >= 2:
            last_len = float(starts[-1] - starts[-2])
            start_units = float(starts[-1])
            end_units = start_units + max(1.0, last_len)
        else:
            start_units = float(starts[0]) if starts else 0.0
            end_units = start_units + 256.0

        center_units = (start_units + end_units) * 0.5
        center_mm = float(ed.time_to_mm(center_units))
        vp_h_mm = float(getattr(ed, '_viewport_h_mm', 0.0) or 0.0)
        target_top_mm = max(0.0, center_mm - (vp_h_mm * 0.5))

        px_per_mm = float(getattr(self, '_editor_metric_px_per_mm', 0.0) or 0.0)
        dpr = float(getattr(self, '_editor_metric_dpr', 1.0) or 1.0)
        if px_per_mm <= 0.0:
            return int(predicted_top_value)

        target_scroll = int(round(target_top_mm * px_per_mm / max(1.0, dpr)))
        minimum = int(self.editor_vscroll.minimum())
        maximum = int(self.editor_vscroll.maximum())
        return max(minimum, min(maximum, target_scroll))

    def _tooltip_anchor_widget(self) -> QtWidgets.QWidget | None:
        if not hasattr(self, 'tool_dock'):
            return None
        if not hasattr(self.tool_dock, 'tooltip_area'):
            return None
        area = self.tool_dock.tooltip_area
        if area is None or not area.isVisible():
            return None
        return area

    def _is_editor_scrollbar_source(self, watched: QtCore.QObject) -> bool:
        if watched is self.editor_vscroll:
            return True
        if isinstance(watched, QtWidgets.QWidget):
            parent = watched.parentWidget()
            while parent is not None:
                if parent is self.editor_vscroll:
                    return True
                parent = parent.parentWidget()
        return False

    def _extract_tooltip_text(self, watched: QtCore.QObject, event: QtGui.QHelpEvent) -> str:
        if isinstance(watched, QtWidgets.QMenu):
            action = watched.actionAt(event.pos())
            if action is not None:
                return str(action.toolTip() or action.text() or "").strip()
            return str(watched.toolTip() or "").strip()

        if isinstance(watched, QtWidgets.QWidget):
            parent_widget = watched.parentWidget()
            if isinstance(parent_widget, QtWidgets.QListWidget):
                item = parent_widget.itemAt(event.pos())
                if item is not None:
                    item_text = str(item.data(QtCore.Qt.ItemDataRole.ToolTipRole) or item.toolTip() or "").strip()
                    if item_text:
                        return item_text
                return str(parent_widget.toolTip() or "").strip()

        if watched is self.tool_dock.selector.viewport():
            item = self.tool_dock.selector.itemAt(event.pos())
            if item is None:
                return str(self.tool_dock.selector.toolTip() or "").strip()
            return str(item.data(QtCore.Qt.ItemDataRole.ToolTipRole) or item.toolTip() or self.tool_dock.selector.toolTip() or "").strip()

        if isinstance(watched, QtWidgets.QListWidget):
            item = watched.itemAt(event.pos())
            if item is not None:
                return str(item.data(QtCore.Qt.ItemDataRole.ToolTipRole) or item.toolTip() or "").strip()
            return str(watched.toolTip() or "").strip()

        if isinstance(watched, QtWidgets.QWidget):
            return str(watched.toolTip() or "").strip()

        return ""

    def _show_tooltip_in_tool_area(self, text: str, hide_popup: bool = True) -> bool:
        area = self._tooltip_anchor_widget()
        if area is None:
            return False
        self.tool_dock.set_tooltip_text(str(text or ""))
        if hide_popup:
            QtWidgets.QToolTip.hideText()
        return True

    def eventFilter(self, watched: QtCore.QObject, event: QtCore.QEvent) -> bool:
        et = event.type()

        # Ensure scrollbar hover tooltip text is cleared immediately once the
        # cursor is no longer over the custom editor scrollbar.
        try:
            if self._tooltip_redirect_source is not None:
                src_is_scrollbar = self._is_editor_scrollbar_source(self._tooltip_redirect_source)
                if src_is_scrollbar and hasattr(self, 'editor_vscroll') and self.editor_vscroll is not None:
                    if not bool(self.editor_vscroll.underMouse()):
                        self._tooltip_redirect_source = None
                        self.tool_dock.set_tooltip_text("")
                        QtWidgets.QToolTip.hideText()
        except Exception:
            pass

        if et == QtCore.QEvent.Type.ToolTip and isinstance(event, QtGui.QHelpEvent):
            scrollbar_source = self._is_editor_scrollbar_source(watched)
            text = self._extract_tooltip_text(watched, event)
            if text:
                shown = self._show_tooltip_in_tool_area(text, hide_popup=not scrollbar_source)
                if shown:
                    self._tooltip_redirect_source = watched
                    return True
                return False
            if watched is self._tooltip_redirect_source:
                self._tooltip_redirect_source = None
                self.tool_dock.set_tooltip_text("")
                QtWidgets.QToolTip.hideText()
                return True

        if et in (QtCore.QEvent.Type.Leave, QtCore.QEvent.Type.FocusOut, QtCore.QEvent.Type.Hide):
            if watched is self._tooltip_redirect_source or self._is_editor_scrollbar_source(watched):
                self._tooltip_redirect_source = None
                self.tool_dock.set_tooltip_text("")
                QtWidgets.QToolTip.hideText()

        return super().eventFilter(watched, event)

    def _editor_scrollbar_tooltip_text(self, predicted_top_value: int) -> str:
        measure_idx = self._editor_scrollbar_measure_index_for_predicted_top(int(predicted_top_value))
        return f"{max(1, measure_idx + 1)}"

    def _current_app_state(self) -> AppState:
        try:
            sc = self.file_manager.current() if hasattr(self, 'file_manager') else None
        except Exception:
            sc = None
        if sc is None:
            return AppState()
        try:
            if not hasattr(sc, 'app_state') or sc.app_state is None:
                sc.app_state = AppState()
        except Exception:
            sc.app_state = AppState()
        return sc.app_state

    def _resolve_app_state_defaults(self) -> AppState:
        """Return app state; if not present in file, seed from appdata defaults."""
        app_state = self._current_app_state()
        try:
            sc = self.file_manager.current()
            if bool(getattr(sc, '_app_state_from_file', False)):
                return app_state
        except Exception:
            pass
        try:
            adm = get_appdata_manager()
            app_state.zoom_mm_per_quarter = float(adm.get("zoom_mm_per_quarter", app_state.zoom_mm_per_quarter) or app_state.zoom_mm_per_quarter)
            app_state.print_view_page_index = int(adm.get("print_view_page_index", app_state.print_view_page_index) or app_state.print_view_page_index)
            app_state.editor_scroll_pos = int(adm.get("editor_scroll_pos", app_state.editor_scroll_pos) or app_state.editor_scroll_pos)
            app_state.snap_base = int(adm.get("snap_base", app_state.snap_base) or app_state.snap_base)
            app_state.snap_divide = int(adm.get("snap_divide", app_state.snap_divide) or app_state.snap_divide)
            app_state.selected_tool = str(adm.get("selected_tool", app_state.selected_tool) or app_state.selected_tool)
        except Exception:
            pass
        return app_state

    def _restore_app_state_from_score(self) -> None:
        self._is_restoring_app_state = True
        try:
            app_state = self._resolve_app_state_defaults()
            # Tool selection
            try:
                self.tool_dock.selector.set_selected_tool(str(app_state.selected_tool or "note"), emit=True)
            except Exception:
                try:
                    self.editor_controller.set_tool_by_name('note')
                except Exception:
                    pass
            # Snap size
            try:
                sb = int(app_state.snap_base or 8)
                sd = int(app_state.snap_divide or 1)
                self.snap_dock.selector.set_snap(sb, sd, emit=True)
            except Exception:
                pass
            # Scroll restore (used when metrics arrive)
            try:
                self._pending_scroll_restore = int(app_state.editor_scroll_pos or 0)
            except Exception:
                pass
            # Print page restore
            try:
                self._page_counter = max(0, int(getattr(app_state, 'print_view_page_index', 0) or 0))
                self._set_page_index(self._page_counter)
            except Exception:
                pass
        finally:
            self._is_restoring_app_state = False

    def _schedule_app_state_save(self) -> None:
        if self._is_restoring_app_state:
            return
        try:
            if hasattr(self, '_app_state_save_timer') and self._app_state_save_timer is not None:
                self._app_state_save_timer.start(500)
        except Exception:
            pass

    def _read_autosave_preferences(self) -> tuple[bool, int]:
        enabled = True
        interval_minutes = 1
        try:
            pm = get_preferences_manager()
            enabled = bool(pm.get("auto_save", True))
            interval_minutes = int(pm.get("auto_save_interval", 1))
        except Exception:
            pass
        if interval_minutes < 1:
            interval_minutes = 1
        return enabled, interval_minutes

    def _apply_autosave_preferences(self) -> None:
        enabled, interval_minutes = self._read_autosave_preferences()
        interval_ms = int(interval_minutes) * 60_000
        try:
            self._autosave_timer.setInterval(interval_ms)
        except Exception:
            pass
        try:
            if enabled:
                self._autosave_timer.start()
            else:
                self._autosave_timer.stop()
        except Exception:
            pass

    def _toggle_full_screen(self) -> None:
        """Toggle native/fullscreen mode across platforms using F11."""
        try:
            if self.isFullScreen() or (self.windowState() & QtCore.Qt.WindowState.WindowFullScreen):
                self.showNormal()
            else:
                self.showFullScreen()
        except Exception:
            try:
                self.showFullScreen()
            except Exception:
                pass
        try:
            if hasattr(self, '_full_screen_act') and self._full_screen_act is not None:
                self._full_screen_act.setChecked(self.isFullScreen())
        except Exception:
            pass

    def _flush_app_state_save(self) -> None:
        """Persist app state to session and optionally autosave project."""
        auto_save_enabled, _ = self._read_autosave_preferences()
        if not auto_save_enabled:
            return
        try:
            if hasattr(self.file_manager, 'autosave_current'):
                self.file_manager.autosave_current()
        except Exception:
            pass
        try:
            if self.file_manager.path() is not None:
                self.file_manager.save()
        except Exception:
            pass

    def _playback_system_label(self) -> str:
        if sys.platform.startswith('linux'):
            return "Playback using FluidSynth"
        if sys.platform == 'darwin':
            return "Playback using CoreMIDI"
        if sys.platform.startswith('win'):
            return "Playback using WinMM"
        return "Playback using System Synth"

    def _get_playback_mode_from_appdata(self) -> str:
        try:
            adm = get_appdata_manager()
            mode = str(adm.get("playback_mode", "system") or "system").strip().lower()
            if mode in ("system", "external"):
                return mode
        except Exception:
            pass
        return "system"

    def _set_playback_mode_to_appdata(self, mode: str) -> None:
        try:
            adm = get_appdata_manager()
            adm.set("playback_mode", str(mode))
            adm.save()
        except Exception:
            pass

    def _get_midi_out_port_from_appdata(self) -> str:
        try:
            adm = get_appdata_manager()
            return str(adm.get("midi_out_port", "") or "")
        except Exception:
            return ""

    def _set_midi_out_port_to_appdata(self, port_name: str) -> None:
        try:
            adm = get_appdata_manager()
            adm.set("midi_out_port", str(port_name or ""))
            adm.save()
        except Exception:
            pass

    def _rebuild_midi_port_menu(self) -> None:
        try:
            menu = getattr(self, '_midi_port_menu', None)
            if menu is None:
                return
            menu.clear()
            from midi.player import Player
            ports = list(Player.list_midi_output_ports() or [])
            selected = self._get_midi_out_port_from_appdata()
            self._midi_port_group = QtGui.QActionGroup(self)
            self._midi_port_group.setExclusive(True)
            if not ports:
                none_act = QtGui.QAction("(No MIDI output ports found)", self)
                none_act.setEnabled(False)
                menu.addAction(none_act)
                return
            for port_name in ports:
                act = QtGui.QAction(str(port_name), self)
                act.setCheckable(True)
                act.setChecked(bool(selected) and str(selected) == str(port_name))
                act.triggered.connect(
                    lambda checked, p=str(port_name): self._select_external_midi_port(p) if checked else None
                )
                self._midi_port_group.addAction(act)
                menu.addAction(act)
        except Exception:
            pass

    def _send_playback_panic(self) -> None:
        try:
            if hasattr(self, 'player') and self.player is not None:
                if hasattr(self.player, 'panic'):
                    self.player.panic()
                else:
                    self.player.stop()
        except Exception:
            pass

    def _dispose_player(self) -> None:
        try:
            if hasattr(self, 'player') and self.player is not None:
                if hasattr(self.player, 'shutdown'):
                    self.player.shutdown()
                else:
                    self.player.stop()
        except Exception:
            pass
        self.player = None
        self._player_config = None
        try:
            if hasattr(self, 'editor_controller') and self.editor_controller is not None:
                self.editor_controller.set_player(None)
        except Exception:
            pass

    def _select_external_midi_port(self, port_name: str) -> None:
        self._send_playback_panic()
        self._dispose_player()
        self._set_midi_out_port_to_appdata(str(port_name))
        self._status(f"External MIDI port: {port_name}", 2500)
        try:
            self._ensure_player()
        except Exception:
            pass

    def _ensure_player(self) -> None:
        # Always ensure attribute exists and bubble up failures so callers can report
        if not hasattr(self, 'player'):
            self.player = None
        playback_mode = self._get_playback_mode_from_appdata()
        midi_out_port = self._get_midi_out_port_from_appdata()
        cfg = (str(playback_mode), str(midi_out_port))
        try:
            if self.player is None or self._player_config != cfg:
                from midi.player import Player
                self.player = Player(
                    soundfont_path=self._get_soundfont_path_from_appdata(),
                    playback_mode=playback_mode,
                    midi_out_port=(midi_out_port or None),
                )
                self._player_config = cfg
            if hasattr(self.player, 'set_persist_settings'):
                self.player.set_persist_settings(False)
            try:
                if hasattr(self, 'editor_controller') and self.editor_controller is not None:
                    self.editor_controller.set_player(self.player)
            except Exception:
                pass
        except Exception:
            self.player = None
            self._player_config = None
            try:
                if hasattr(self, 'editor_controller') and self.editor_controller is not None:
                    self.editor_controller.set_player(None)
            except Exception:
                pass
            raise

    def _get_soundfont_path_from_appdata(self) -> Optional[str]:
        try:
            adm = get_appdata_manager()
            path = str(adm.get("user_soundfont_path", "") or "")
            if path and Path(path).expanduser().is_file():
                return str(Path(path).expanduser())
        except Exception:
            pass
        return None

    def _set_soundfont_path_to_appdata(self, path: Optional[str]) -> None:
        try:
            adm = get_appdata_manager()
            adm.set("user_soundfont_path", str(path or ""))
            adm.save()
        except Exception:
            pass

    def _unset_soundfont(self) -> None:
        """Clear custom FluidSynth soundfont and revert to default detection."""
        self._set_soundfont_path_to_appdata(None)
        try:
            self._dispose_player()
        except Exception:
            pass
        self._status("Using default FluidSynth soundfont", 2500)

    def _prompt_for_soundfont(self, force_dialog: bool = False) -> Optional[str]:
        """Ensure a soundfont path exists; prompt user if missing or forced."""
        existing = self._get_soundfont_path_from_appdata()
        if existing and not force_dialog:
            return existing
        dlg = QtWidgets.QFileDialog(self)
        dlg.setFileMode(QtWidgets.QFileDialog.FileMode.ExistingFile)
        dlg.setNameFilter("SoundFont (*.sf2 *.sf3)")
        dlg.setViewMode(QtWidgets.QFileDialog.ViewMode.Detail)
        try:
            here = Path(__file__).resolve().parent.parent / "assets" / "soundfonts"
            if here.is_dir():
                dlg.setDirectory(str(here))
        except Exception:
            pass
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            sel = dlg.selectedFiles()[0]
            if sel:
                self._set_soundfont_path_to_appdata(sel)
                try:
                    if hasattr(self, 'player') and self.player is not None:
                        self.player.set_soundfont(sel)
                except Exception:
                    pass
                self._status("Custom FluidSynth soundfont selected", 2500)
                return sel
        return existing if existing else None

    def _ensure_player_with_soundfont(self) -> None:
        mode = self._get_playback_mode_from_appdata()
        if not (sys.platform.startswith('linux') and mode == 'system'):
            self._ensure_player()
            return
        try:
            self._ensure_player()
            return
        except Exception as exc:
            # If missing soundfont, prompt the user and retry once
            msg = str(exc).lower()
            if "soundfont" in msg:
                chosen = self._prompt_for_soundfont(force_dialog=True)
                if chosen:
                    from midi.player import Player
                    self.player = Player(
                        soundfont_path=chosen,
                        playback_mode='system',
                        midi_out_port=None,
                    )
                    self._player_config = ('system', self._get_midi_out_port_from_appdata())
                    try:
                        if hasattr(self.player, 'set_persist_settings'):
                            self.player.set_persist_settings(False)
                    except Exception:
                        pass
                    return
            raise

    def _choose_midi_port(self) -> None:
        self._rebuild_midi_port_menu()
        self._status("Select external MIDI output from Playback > MIDI port", 2500)

    def _update_clock(self) -> None:
        try:
            now = datetime.now()
            timestr = now.strftime("%H:%M:%S")
            if hasattr(self, "_clock_label") and self._clock_label is not None:
                self._clock_label.setText(timestr)
                # Re-position in case width changed
                self._position_clock()
        except Exception:
            pass

    def _position_clock(self) -> None:
        try:
            menubar = self.menuBar()
            if not hasattr(self, "_clock_label") or self._clock_label is None:
                return
            rect = menubar.rect()
            sh = self._clock_label.sizeHint()
            # Height equals menubar height to align vertically; width to hint
            self._clock_label.resize(sh.width(), rect.height())
            x = max(0, rect.width() - self._clock_label.width() - 8)
            self._clock_label.move(x, 0)
            self._clock_label.show()
        except Exception:
            pass

    def _export_pdf(self) -> None:
        dlg = QtWidgets.QFileDialog(self)
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dlg.setNameFilter("PDF Files (*.pdf)")
        dlg.setDefaultSuffix("pdf")
        # Prefill filename with score title when available
        try:
            score_title = ""
            try:
                info = getattr(self.file_manager.current(), 'info', None)
                score_title = str(getattr(info, 'title', "") or "") if info is not None else ""
            except Exception:
                score_title = ""
            safe_title = "".join(ch for ch in score_title if ch not in r'\\/:*?"<>|').strip()
            suggested_name = f"{safe_title or 'Untitled'}.pdf"
        except Exception:
            suggested_name = "Untitled.pdf"
        try:
            adm = get_appdata_manager()
            last_dir = str(adm.get("last_export_pdf_dir", "") or "")
            if last_dir:
                dlg.setDirectory(last_dir)
                dlg.selectFile(os.path.join(last_dir, suggested_name))
            else:
                dlg.selectFile(suggested_name)
        except Exception:
            pass
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            out_path = dlg.selectedFiles()[0]
            try:
                out_dir = os.path.dirname(str(out_path))
                if out_dir:
                    adm = get_appdata_manager()
                    adm.set("last_export_pdf_dir", out_dir)
                    adm.save()
            except Exception:
                pass
            try:
                from utils.CONSTANT import ENGRAVER_LAYERING
                from engraver.engraver import do_engrave
                export_du = DrawUtil()
                do_engrave(self._current_score_dict(), export_du, pdf_export=True)
                total_pages = max(1, export_du.page_count())
                progress = QtWidgets.QProgressDialog("Exporting PDF...", None, 0, total_pages, self)
                progress.setWindowTitle("Export PDF")
                progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(0)
                progress.setValue(0)
                progress.show()

                def _on_progress(done: int, total: int) -> None:
                    if progress.maximum() != int(total):
                        progress.setMaximum(int(total))
                    progress.setValue(int(done))
                    QtWidgets.QApplication.processEvents()

                export_du.save_pdf(out_path, layering=ENGRAVER_LAYERING, progress_cb=_on_progress)
                progress.setValue(total_pages)
                progress.close()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Export PDF failed", str(e))

    def _status(self, message: str, timeout_ms: int = 3000) -> None:
        """Show a transient message on the status bar."""
        try:
            sb = self.statusBar() if hasattr(self, 'statusBar') else None
            if sb is not None:
                sb.showMessage(str(message), int(max(0, timeout_ms)))
        except Exception:
            pass

    def _status_default_message(self) -> str:
        try:
            dirty = bool(self.file_manager.is_dirty())
        except Exception:
            dirty = False
        try:
            p = self.file_manager.path()
        except Exception:
            p = None
        session_mode = bool(getattr(self, '_session_restore_mode', False)) and p is None
        state = "Unsaved changes" if dirty else "Saved"
        path_text = str(p) if p else ("(session.piano restored)" if session_mode else "(unsaved project)")
        prefix = "Session mode • " if session_mode else ""
        return f"{prefix}{state} • {path_text}"

    def _show_status_default(self) -> None:
        try:
            sb = self.statusBar() if hasattr(self, 'statusBar') else None
            if sb is not None and not sb.currentMessage():
                sb.showMessage(self._status_default_message(), 0)
        except Exception:
            pass

    def _on_status_message_changed(self, msg: str) -> None:
        if not msg:
            self._show_status_default()

    def _open_preferences(self) -> None:
        # Ensure preferences file exists and open in system editor
        open_preferences(self)

    def _open_about_dialog(self) -> None:
        """Show licensing and attribution info."""
        try:
            dlg = AboutDialog(self)
            dlg.exec()
        except Exception:
            pass

    def _file_new(self) -> None:
        # If there are unsaved changes, confirm save before starting a new project
        if not self.file_manager.confirm_save_for_action("creating a new project", force_prompt=True):
            return
        self.file_manager.new()
        self._session_restore_mode = False
        self._refresh_views_from_score()
        try:
            QtCore.QTimer.singleShot(1000, lambda: self.engraver.engrave(self._current_score_dict()))
        except Exception:
            pass
        # Provide current score to editor for drawers needing direct access
        try:
            self.editor_controller.set_score(self.file_manager.current())
            # Reset undo stack for new project
            self.editor_controller.reset_undo_stack()
        except Exception:
            pass
        try:
            if hasattr(self.editor_controller, 'force_redraw_from_model'):
                self.editor_controller.force_redraw_from_model()
        except Exception:
            pass
        # Reset editor scroll to top for a fresh project
        try:
            self._pending_scroll_restore = 0
            self.editor_vscroll.setValue(0)
            self.editor_canvas.set_scroll_logical_px(0)
        except Exception:
            pass
        try:
            self._restore_app_state_from_score()
        except Exception:
            pass
        self._update_title()

    def _file_open(self) -> None:
        # If there are unsaved changes, confirm save before opening another project
        if not self.file_manager.confirm_save_for_action("opening another project", force_prompt=True):
            return
        sc = self.file_manager.load()
        if sc:
            self._after_project_loaded()

    def open_documents_from_paths(self, paths: list[str], confirm_dirty: bool = True) -> None:
        candidates = [str(Path(p).expanduser()) for p in paths if str(p).strip()]
        if not candidates:
            return
        was_fullscreen = self.isFullScreen()
        was_minimized = self.isMinimized()
        if confirm_dirty and not self.file_manager.confirm_save_for_action("opening another project", force_prompt=True):
            return
        opened_any = False
        for candidate in candidates:
            sc = self.file_manager.open_path(candidate)
            if sc:
                opened_any = True
                self._after_project_loaded()
        if opened_any:
            try:
                if was_minimized and not was_fullscreen:
                    self.showNormal()
                elif was_fullscreen:
                    self.showFullScreen()
                self.raise_()
                self.activateWindow()
            except Exception:
                pass

    def _after_project_loaded(self) -> None:
        self._session_restore_mode = False
        try:
            self._refresh_recent_files_menu()
        except Exception:
            pass
        try:
            app_state = self._resolve_app_state_defaults()
            self._page_counter = max(0, int(getattr(app_state, 'print_view_page_index', 0) or 0))
        except Exception:
            self._page_counter = 0
        self._refresh_views_from_score()
        try:
            self.editor_controller.set_score(self.file_manager.current())
            self.editor_controller.reset_undo_stack()
        except Exception:
            pass
        try:
            if hasattr(self.editor_controller, 'force_redraw_from_model'):
                self.editor_controller.force_redraw_from_model()
        except Exception:
            pass
        try:
            self._restore_app_state_from_score()
        except Exception:
            pass
        self._update_title()
        self._show_status_default()

    def _file_save(self) -> None:
        if self.file_manager.save():
            if self.file_manager.path() is not None:
                self._session_restore_mode = False
            self._update_title()
            self._show_status_default()

    def _file_save_as(self) -> None:
        if self.file_manager.save_as():
            if self.file_manager.path() is not None:
                self._session_restore_mode = False
            self._update_title()
            self._show_status_default()

    def _refresh_views_from_score(self, delay_engrave_ms: int = 0) -> None:
        try:
            sc_dict = self.file_manager.current().get_dict()
        except Exception:
            sc_dict = {}
        self.print_view.set_score(sc_dict)
        # Request engraving via Engraver; render happens on engraved signal
        if delay_engrave_ms and delay_engrave_ms > 0:
            def _delayed_engrave() -> None:
                try:
                    self.engraver.engrave(self._current_score_dict(), pageno=int(getattr(self, '_page_counter', 0)))
                except Exception:
                    self.print_view.request_render()
            QtCore.QTimer.singleShot(int(delay_engrave_ms), _delayed_engrave)
        else:
            try:
                self.engraver.engrave(sc_dict, pageno=int(getattr(self, '_page_counter', 0)))
            except Exception:
                # Fallback: render current content
                self.print_view.request_render()
        # Also refresh the editor view
        self.editor_canvas.update()

    def _on_score_changed(self) -> None:
        if hasattr(self, '_score_change_engrave_timer') and self._score_change_engrave_timer is not None:
            self._score_change_engrave_timer.start()
        else:
            self.engraver.engrave(self._current_score_dict(), pageno=int(getattr(self, '_page_counter', 0)))
        self._show_status_default()

    def _flush_score_change_engrave(self) -> None:
        self.engraver.engrave(self._current_score_dict(), pageno=int(getattr(self, '_page_counter', 0)))

    def _open_style_dialog(self) -> None:
        from dataclasses import asdict
        from file_model.layout import Layout
        from ui.dialogs.style_dialog import StyleDialog

        sc = self.file_manager.current()
        layout = getattr(sc, 'layout', None)
        original_layout = asdict(layout) if layout is not None else None
        dlg = StyleDialog(parent=self, layout=layout, score=sc)

        app_state = self._current_app_state()
        dlg.set_current_tab(int(getattr(app_state, 'style_dialog_tab_index', 0) or 0))

        preview_timer = QtCore.QTimer(dlg)
        preview_timer.setSingleShot(True)
        preview_timer.setInterval(150)

        def _emit_preview() -> None:
            self.editor_controller.force_redraw_from_model()
            self.editor_controller.score_changed.emit()

        preview_timer.timeout.connect(_emit_preview)

        def _schedule_preview() -> None:
            preview_timer.stop()
            preview_timer.start()

        def _apply_live(commit_snapshot: bool = False) -> None:
            sc.layout = dlg.get_values()
            if commit_snapshot:
                self.editor_controller._snapshot_if_changed(coalesce=False, label='style_edit')
            _schedule_preview()

        def _revert_state() -> None:
            if original_layout is None:
                return
            sc.layout = Layout(**original_layout)
            _schedule_preview()

        dlg.values_changed.connect(lambda: _apply_live(False))
        dlg.accepted.connect(lambda: _apply_live(True))
        dlg.rejected.connect(_revert_state)

        def _persist_tab_index() -> None:
            app_state.style_dialog_tab_index = int(dlg.current_tab_index())
            self._flush_app_state_save()

        dlg.finished.connect(lambda _res: _persist_tab_index())
        dlg.accepted.connect(lambda: self.file_manager.save() if self.file_manager.path() is not None else None)
        dlg.show()

    def _open_info_dialog(self) -> None:
        from ui.dialogs.info_dialog import InfoDialog
        sc = self.file_manager.current()
        dlg = InfoDialog(sc, self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            dlg.apply_to_score()
            self.file_manager.on_model_changed()
            self._refresh_views_from_score()

    def _open_line_break_dialog(self) -> None:
        from ui.dialogs.line_break_dialog import LineBreakDialog

        score = self.file_manager.current()
        if score is None:
            return

        preview_timer = QtCore.QTimer(self)
        preview_timer.setSingleShot(True)
        preview_timer.setInterval(150)

        def _emit_preview() -> None:
            self.editor_controller.force_redraw_from_model()
            self.editor_controller.score_changed.emit()

        preview_timer.timeout.connect(_emit_preview)

        def _schedule_preview() -> None:
            preview_timer.stop()
            preview_timer.start()

        dlg = LineBreakDialog(
            parent=self,
            score=score,
            selected_line_break=None,
            measure_resolver=(lambda t: self.editor_controller.get_measure_index_for_time(t)) if hasattr(self.editor_controller, 'get_measure_index_for_time') else None,
            on_change=_schedule_preview,
        )

        def _on_accept() -> None:
            self.editor_controller._snapshot_if_changed(coalesce=False, label='line_break_edit')
            _schedule_preview()

        def _on_reject() -> None:
            _schedule_preview()

        def _on_finished(_result: int) -> None:
            self.file_manager.on_model_changed()

        dlg.accepted.connect(_on_accept)
        dlg.rejected.connect(_on_reject)
        dlg.finished.connect(_on_finished)
        dlg.show()

    def _run_slur_bow_optimizer(self) -> None:
        score = self.file_manager.current()
        if score is None:
            self._status("Slur bow optimizer: no score loaded", 2500)
            return

        try:
            from ui.dialogs.slur_bow_optimizer import SlurBowOptimizerDialog, optimize_slur_bows_in_score
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "Slur bow optimizer", f"Failed to load optimizer:\n{exc}")
            return

        dlg = self._slur_opt_dialog
        if dlg is None:
            dlg = SlurBowOptimizerDialog(self)
            self._slur_opt_dialog = dlg

            def _on_apply(settings: dict, fn=optimize_slur_bows_in_score) -> None:
                self._start_slur_bow_optimizer_run(dict(settings or {}), fn)

            dlg.applyRequested.connect(_on_apply)
            dlg.finished.connect(self._on_slur_optimizer_dialog_closed)

        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        try:
            dlg.set_info_text("Adjust parameters and click Apply to run optimizer.")
        except Exception:
            pass
        self._status("Adjust parameters and click Apply to run optimizer", 3000)

    @QtCore.Slot()
    def _on_slur_optimizer_dialog_closed(self) -> None:
        self._slur_opt_dialog = None

    def _start_slur_bow_optimizer_run(self, settings: dict, optimize_slur_bows_in_score) -> None:
        if self._slur_opt_thread is not None and self._slur_opt_thread.isRunning():
            self._status("Slur bow optimizer is already running", 2500)
            return

        score = self.file_manager.current()
        if score is None:
            self._status("Slur bow optimizer: no score loaded", 2500)
            return

        self._slur_opt_settings = dict(settings)
        self._slur_opt_started_at = time.perf_counter()

        try:
            if self._slur_opt_dialog is not None:
                self._slur_opt_dialog.set_apply_enabled(False)
        except Exception:
            pass

        progress = QtWidgets.QProgressDialog("Optimizing slur bows...", "", 0, 0, self)
        progress.setWindowTitle("Slur bow optimizer")
        progress.setCancelButton(None)
        progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        self._slur_opt_progress = progress

        try:
            QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.CursorShape.WaitCursor)
        except Exception:
            pass

        thread = QtCore.QThread(self)
        worker = _SlurOptimizerWorker(score, settings, optimize_slur_bows_in_score)
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.finished.connect(self._on_slur_optimizer_finished, QtCore.Qt.ConnectionType.QueuedConnection)
        worker.failed.connect(self._on_slur_optimizer_failed, QtCore.Qt.ConnectionType.QueuedConnection)
        self._slur_opt_worker = worker
        self._slur_opt_thread = thread
        thread.start()

    def _cleanup_slur_optimizer_thread(self) -> None:
        try:
            if self._slur_opt_progress is not None:
                self._slur_opt_progress.close()
                self._slur_opt_progress.deleteLater()
        except Exception:
            pass
        self._slur_opt_progress = None
        try:
            QtWidgets.QApplication.restoreOverrideCursor()
        except Exception:
            pass

        thread = self._slur_opt_thread
        worker = self._slur_opt_worker
        self._slur_opt_thread = None
        self._slur_opt_worker = None

        try:
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait(1000)
        except Exception:
            pass
        try:
            if worker is not None:
                worker.deleteLater()
        except Exception:
            pass
        try:
            if thread is not None:
                thread.deleteLater()
        except Exception:
            pass
        try:
            if self._slur_opt_dialog is not None:
                self._slur_opt_dialog.set_apply_enabled(True)
        except Exception:
            pass

    @QtCore.Slot(dict)
    def _on_slur_optimizer_finished(self, stats: dict) -> None:
        settings = self._slur_opt_settings or {}
        total = int(stats.get("slurs_total", 0) or 0)
        available = int(stats.get("slurs_available", total) or total)
        changed = int(stats.get("slurs_changed", 0) or 0)
        effective_iters = int(stats.get("effective_max_iterations", int(settings.get("max_iterations", 3) or 3)) or 0)
        effective_samples = int(stats.get("effective_hit_test_finetune", int(settings.get("hit_test_finetune", 64) or 64)) or 0)
        requested_limit = int(settings.get("max_slurs_from_start", total) or total)
        openness_before = float(stats.get("openness_before", 0.0) or 0.0)
        openness_after = float(stats.get("openness_after", 0.0) or 0.0)
        openness_delta = float(stats.get("openness_delta", 0.0) or 0.0)
        objective_before = float(stats.get("objective_before_avg", 0.0) or 0.0)
        objective_after = float(stats.get("objective_after_avg", 0.0) or 0.0)
        objective_delta = float(objective_after - objective_before)
        beams_used = bool(settings.get("include_beams", True))
        started_at = self._slur_opt_started_at
        elapsed_s = max(0.0, float(time.perf_counter() - started_at)) if started_at is not None else 0.0

        if changed > 0:
            try:
                self.file_manager.on_model_changed()
            except Exception:
                pass
            try:
                self.editor_controller._snapshot_if_changed(coalesce=False, label='slur_bow_optimizer')
            except Exception:
                pass
            self._refresh_views_from_score()
            try:
                self.editor_controller.set_score(self.file_manager.current())
            except Exception:
                pass
            try:
                self.editor_controller.force_redraw_from_model()
            except Exception:
                pass
            self._status(
                f"Slur bow optimizer: updated {changed}/{total} slurs (iter={effective_iters}, samples={effective_samples})",
                4500,
            )
        else:
            self._status(
                f"Slur bow optimizer: no changes ({total} slurs checked, iter={effective_iters}, samples={effective_samples})",
                4500,
            )

        # Professional completion notification with key run metrics.
        details = [
            f"Processed slurs: {total} (requested first {requested_limit}, available {available})",
            f"Changed slurs: {changed}",
            f"Duration: {elapsed_s:.2f} s",
            f"Effective settings: iterations={effective_iters}, samples={effective_samples}, beams={'on' if beams_used else 'off'}",
            f"Objective avg: {objective_before:.3f} -> {objective_after:.3f} (Δ {objective_delta:+.3f})",
            f"Openness score: {openness_before:.3f} -> {openness_after:.3f} (Δ {openness_delta:+.3f})",
        ]
        prefix = "Optimization completed successfully." if changed > 0 else "Optimization completed with no required changes."
        summary_text = prefix + "\n\n" + "\n".join(details)
        try:
            if self._slur_opt_dialog is not None:
                self._slur_opt_dialog.set_info_text(summary_text)
        except Exception:
            pass

        self._slur_opt_started_at = None
        self._cleanup_slur_optimizer_thread()

    @QtCore.Slot(str)
    def _on_slur_optimizer_failed(self, message: str) -> None:
        self._status("Slur bow optimizer: failed", 4500)
        try:
            if self._slur_opt_dialog is not None:
                self._slur_opt_dialog.set_info_text(f"Optimization failed:\n{message}")
        except Exception:
            pass
        self._slur_opt_started_at = None
        self._cleanup_slur_optimizer_thread()

    def _refresh_recent_files_menu(self) -> None:
        menu = getattr(self, '_recent_menu', None)
        if menu is None:
            return
        menu.clear()
        try:
            adm = get_appdata_manager()
            recent = adm.get("recent_files", []) or []
        except Exception:
            recent = []
        if not isinstance(recent, list):
            recent = []
        recent = [str(p) for p in recent if str(p).strip()]
        if not recent:
            empty_act = QtGui.QAction("No recent files", self)
            empty_act.setEnabled(False)
            menu.addAction(empty_act)
        else:
            for path in recent[:100]:
                act = QtGui.QAction(path, self)
                act.triggered.connect(lambda _c=False, p=path: self._open_recent_file(p))
                menu.addAction(act)
        menu.addSeparator()
        clear_act = QtGui.QAction("Clear Recent Files", self)
        clear_act.triggered.connect(self._clear_recent_files)
        menu.addAction(clear_act)

    def _open_recent_file(self, path: str) -> None:
        if not self.file_manager.confirm_save_for_action("opening another project", force_prompt=True):
            return
        sc = self.file_manager.open_path(str(path))
        if sc:
            try:
                self._refresh_recent_files_menu()
            except Exception:
                pass
            self._refresh_views_from_score()
            try:
                self.editor_controller.set_score(self.file_manager.current())
                self.editor_controller.reset_undo_stack()
            except Exception:
                pass
            try:
                if hasattr(self.editor_controller, 'force_redraw_from_model'):
                    self.editor_controller.force_redraw_from_model()
            except Exception:
                pass
            try:
                self._restore_app_state_from_score()
            except Exception:
                pass
            self._update_title()

    def _clear_recent_files(self) -> None:
        try:
            adm = get_appdata_manager()
            adm.set("recent_files", [])
            adm.save()
        except Exception:
            pass
        self._refresh_recent_files_menu()

    @QtCore.Slot(int, int, float, float)
    def _on_editor_metrics(self, content_px: int, viewport_px: int, px_per_mm: float, dpr: float) -> None:
        # External QScrollBar works in logical pixels
        scale = max(1.0, dpr)
        self._editor_metric_px_per_mm = float(px_per_mm)
        self._editor_metric_dpr = float(dpr)
        self._editor_metric_viewport_logical_px = int(max(0, round(float(viewport_px) / scale)))
        max_scroll = max(0, int(round((content_px - viewport_px) / scale)))
        self.editor_vscroll.setRange(0, max_scroll)
        # Page step ~ 80% of viewport height (logical px)
        self.editor_vscroll.setPageStep(int(max(1, round(0.8 * viewport_px / scale))))
        # Single step follows one snap band in logical pixels.
        logical_px_step = self._editor_scroll_step_from_metrics(px_per_mm, dpr)
        self._editor_scroll_step_logical_px = int(max(1, logical_px_step))
        self.editor_vscroll.setSingleStep(logical_px_step)
        self.editor_canvas.set_scroll_step_logical_px(logical_px_step)
        # Clamp current value within new range to avoid unbounded wheel scroll
        cur = int(self.editor_vscroll.value())
        if cur > max_scroll:
            self.editor_vscroll.setValue(max_scroll)
        # Apply a pending restore once, after we know the range
        pending = int(getattr(self, '_pending_scroll_restore', 0) or 0)
        if pending and max_scroll >= 0:
            target = max(0, min(pending, max_scroll))
            if int(self.editor_vscroll.value()) != target:
                self.editor_vscroll.setValue(target)
            self._pending_scroll_restore = 0

    @QtCore.Slot(int)
    def _on_editor_scroll_changed(self, value: int) -> None:
        value = int(value)
        self.editor_canvas.set_scroll_logical_px(value)
        app_state = self._current_app_state()
        app_state.editor_scroll_pos = int(value)

    def _editor_scroll_step_from_metrics(self, px_per_mm: float, dpr: float) -> int:
        sc = self.file_manager.current()
        app_state = getattr(sc, 'app_state', None) if sc is not None else None
        zoom_mm_per_quarter = float(getattr(app_state, 'zoom_mm_per_quarter', 25.0) or 25.0)

        snap_units = float(getattr(self.editor_controller, 'snap_size_units', 0.0) or 0.0)
        if snap_units <= 0.0 and hasattr(self, 'snap_dock') and hasattr(self.snap_dock, 'selector'):
            snap_units = float(self.snap_dock.selector.get_snap_size() or 0.0)
        if snap_units <= 0.0:
            snap_units = float(QUARTER_NOTE_UNIT) / 2.0

        snap_mm = (float(snap_units) / float(QUARTER_NOTE_UNIT)) * float(zoom_mm_per_quarter)
        scale = max(1.0, float(dpr))
        device_px_step = float(snap_mm) * float(px_per_mm)
        return int(max(1, round(device_px_step / scale)))

    def _quantize_editor_scroll_value(self, value: int) -> int:
        minimum = int(self.editor_vscroll.minimum())
        maximum = int(self.editor_vscroll.maximum())
        clamped = max(minimum, min(maximum, int(value)))
        step = int(max(1, getattr(self, '_editor_scroll_step_logical_px', 1) or 1))
        if step <= 1:
            return clamped
        snapped = int(round(float(clamped) / float(step)) * step)
        return max(minimum, min(maximum, snapped))

    def _zoom_editor(self, steps: int) -> None:
        try:
            if hasattr(self, 'editor') and hasattr(self.editor_canvas, 'apply_zoom_steps'):
                self.editor_canvas.apply_zoom_steps(int(steps))
        except Exception:
            pass

    def _edit_undo(self) -> None:
        self.editor_controller.undo()
        self._refresh_views_from_score()
        try:
            self.editor_controller.set_score(self.file_manager.current())
        except Exception:
            pass
        try:
            self.editor_controller.force_redraw_from_model()
        except Exception:
            pass

    def _edit_redo(self) -> None:
        self.editor_controller.redo()
        self._refresh_views_from_score()
        try:
            self.editor_controller.set_score(self.file_manager.current())
        except Exception:
            pass
        try:
            self.editor_controller.force_redraw_from_model()
        except Exception:
            pass

    def _edit_copy(self) -> None:
        try:
            self.editor_controller.copy_selection()
            self._status("Copied selection", 1200)
        except Exception:
            pass

    def _edit_cut(self) -> None:
        try:
            self.editor_controller.cut_selection()
            self._refresh_views_from_score()
            try:
                self.editor_controller.set_score(self.file_manager.current())
            except Exception:
                pass
            try:
                self.editor_controller.force_redraw_from_model()
            except Exception:
                pass
            self._status("Cut selection", 1200)
        except Exception:
            pass

    def _edit_paste(self) -> None:
        try:
            self.editor_controller.paste_selection_at_cursor()
            self._refresh_views_from_score()
            try:
                self.editor_controller.set_score(self.file_manager.current())
            except Exception:
                pass
            try:
                self.editor_controller.force_redraw_from_model()
            except Exception:
                pass
            self._status("Pasted selection", 1200)
        except Exception:
            pass

    def _edit_delete(self) -> None:
        try:
            deleted = False
            if hasattr(self.editor_controller, 'delete_selection'):
                res = self.editor_controller.delete_selection()
                deleted = bool(res)
            if deleted:
                try:
                    self.editor_controller.set_score(self.file_manager.current())
                except Exception:
                    pass
                try:
                    self.editor_controller.force_redraw_from_model()
                except Exception:
                    pass
                self._status("Deleted selection", 1200)
            else:
                self._status("No selection to delete", 1200)
        except Exception:
            pass

    def _selection_select_all(self) -> None:
        try:
            self.editor_controller.select_all()
            try:
                self.editor_canvas.update()
            except Exception:
                pass
            self._status("Selected all", 1200)
        except Exception:
            pass

    def _selection_transpose(self, semitones: int) -> None:
        try:
            changed = bool(self.editor_controller.transpose_selected_notes(int(semitones)))
            if changed:
                try:
                    self.editor_canvas.update()
                except Exception:
                    pass
                self._status(f"Transposed selection {int(semitones):+d} semitone", 1200)
            else:
                self._status("No selection to transpose", 1200)
        except Exception:
            pass

    def _selection_shift(self, sign: float) -> None:
        try:
            units = float(getattr(self.editor_controller, 'snap_size_units', 0.0) or 0.0)
            if units <= 0.0:
                units = 1.0
            delta = float(sign) * float(units)
            changed = bool(self.editor_controller.shift_selected_notes_time(delta))
            if changed:
                try:
                    self.editor_canvas.update()
                except Exception:
                    pass
                direction = "earlier" if delta < 0 else "later"
                self._status(f"Moved selection {direction} by snap", 1200)
            else:
                self._status("No selection to move", 1200)
        except Exception:
            pass

    def _selection_quantize(self, qtype: str = 'start/end') -> None:
        try:
            changed = bool(getattr(self.editor_controller, 'quantize_selected_notes', lambda *_args, **_kwargs: False)(qtype))
            if changed:
                try:
                    self.editor_canvas.update()
                except Exception:
                    pass
                mode = str(qtype or 'start/end').strip().lower()
                if mode == 'start':
                    self._status("Quantized selection starts to snap", 1200)
                elif mode == 'end':
                    self._status("Quantized selection ends to snap", 1200)
                else:
                    self._status("Quantized selection starts and ends to snap", 1200)
            else:
                self._status("No selection to quantize", 1200)
        except Exception:
            pass

    def _update_title(self) -> None:
        self.setWindowTitle("keyTAB")
        self._show_status_default()

    def _page_dimensions_mm(self) -> tuple[float, float]:
        try:
            sc = self.file_manager.current()
            lay = getattr(sc, 'layout', None)
            if lay:
                return float(lay.page_width_mm), float(lay.page_height_mm)
        except Exception:
            pass
        # Fallback to DrawUtil current page size
        try:
            return self.du.current_page_size_mm()
        except Exception:
            return (210.0, 297.0)

    def _fit_print_view_to_page(self, *_args) -> None:
        """Toggle fit/hidden state and ensure in-between positions snap to fit.

        Behavior:
        - If currently fitted (self.is_fit): hide the print view.
        - Else: run the fit logic.
        - If not hidden and not fitted (in-between): run the fit logic.
        """
        splitter = self.centralWidget()
        if splitter is None:
            return

        # Helper: compute desired fit sizes
        def compute_fit_sizes() -> tuple[int, int]:
            w_mm, h_mm = self._page_dimensions_mm()
            if w_mm <= 0 or h_mm <= 0:
                return (splitter.width(), 0)
            # Exclude handle width to compute available content width
            try:
                handle_w = int(splitter.handleWidth())
            except Exception:
                handle_w = 0
            total_w = max(0, splitter.width() - handle_w)
            # Use splitter height (more stable at startup/maximized) for fit computations
            pv_h = max(1, splitter.height())
            ideal_pv_w = int(round(pv_h * (w_mm / h_mm)))
            # Clamp to available width to avoid oversizing when maximized/startup
            pv_w = min(max(0, ideal_pv_w), total_w)
            editor_w = max(0, total_w - pv_w)
            return (editor_w, pv_w)

        # Current sizes and state
        sizes = splitter.sizes() or [splitter.width(), 0]
        cur_editor_w = int(sizes[0]) if sizes else splitter.width()
        cur_pv_w = int(sizes[1]) if len(sizes) > 1 else 0
        fitted_editor_w, fitted_pv_w = compute_fit_sizes()

        # on startup, we always start fitted
        if self.is_startup:
            self.is_startup = False
            splitter.setSizes([fitted_editor_w, fitted_pv_w])
            return

        # Determine if hidden or fitted (with small tolerance)
        hidden = (cur_pv_w <= 0)
        fit_tolerance = 2
        fitted = (abs(cur_pv_w - fitted_pv_w) <= fit_tolerance and abs(cur_editor_w - fitted_editor_w) <= fit_tolerance)
        self.is_fit = fitted

        if self.is_fit:
            # Hide the print view
            splitter.setSizes([cur_editor_w + cur_pv_w, 0])
            self.is_fit = False
            return

        # If not hidden and not fitted (in-between), or hidden: run fit logic
        if (not hidden and not fitted) or hidden:
            splitter.setSizes([fitted_editor_w, fitted_pv_w])
            self.is_fit = True
            return

    def _current_score_dict(self) -> dict:
        try:
            return self.file_manager.current().get_dict()
        except Exception:
            return {}

    def _on_engraver_finished(self) -> None:
        # Keep print view page selection aligned with restored/app-state page index.
        try:
            page_count = int(self.du.page_count())
        except Exception:
            page_count = 0
        if page_count > 0:
            try:
                desired = int(getattr(self, '_page_counter', 0) or 0)
            except Exception:
                desired = 0
            desired = max(0, min(page_count - 1, desired))
            try:
                self.du.set_current_page(desired)
            except Exception:
                pass
            try:
                self.print_view.set_page(desired, request_render=False)
            except Exception:
                pass
            self._page_counter = desired
        try:
            self._update_analysis_from_engraver()
        except Exception:
            pass
        try:
            self.print_view.request_render()
        except Exception:
            pass

    def _update_analysis_from_engraver(self) -> None:
        analysis_obj = getattr(self.engraver, "analysis", None)
        if analysis_obj is None:
            return
        score = None
        try:
            score = self.file_manager.current()
        except Exception:
            score = None
        if score is None:
            return

        def _value(obj, key: str):
            try:
                return getattr(obj, key)
            except Exception:
                pass
            try:
                return obj.get(key)  # type: ignore[arg-type]
            except Exception:
                return None

        lines_count = _value(analysis_obj, "lines")
        pages_count = _value(analysis_obj, "pages")
        try:
            analysis_snapshot = Analysis.compute(score, lines_count=lines_count, pages_count=pages_count)
        except Exception:
            analysis_snapshot = None
        if analysis_snapshot is None:
            return
        try:
            score.analysis = analysis_snapshot
        except Exception:
            pass

    def _set_page_index(self, index: int) -> None:
        idx = max(0, int(index))
        try:
            page_count = int(self.du.page_count())
        except Exception:
            page_count = 0
        if page_count > 0:
            idx = min(idx, page_count - 1)
        try:
            self.du.set_current_page(idx)
        except Exception:
            pass
        try:
            self.print_view.set_page(idx, request_render=False)
        except Exception:
            pass
        self._page_counter = idx
        try:
            app_state = self._current_app_state()
            app_state.print_view_page_index = idx
        except Exception:
            pass
        try:
            self._schedule_app_state_save()
        except Exception:
            pass

    def _next_page(self) -> None:
        try:
            page_count = int(self.du.page_count())
            if page_count <= 0:
                return
            self._page_counter = (self._page_counter + 1) % page_count
            self._set_page_index(self._page_counter)
            self.engraver.engrave(self._current_score_dict(), pageno=self._page_counter)
        except Exception:
            pass

    def _previous_page(self) -> None:
        try:
            page_count = int(self.du.page_count())
            if page_count <= 0:
                return
            self._page_counter = (self._page_counter - 1) % page_count
            self._set_page_index(self._page_counter)
            self.engraver.engrave(self._current_score_dict(), pageno=self._page_counter)
        except Exception:
            pass

    def _engrave_now(self) -> None:
        try:
            self.engraver.engrave(self._current_score_dict(), pageno=int(getattr(self, '_page_counter', 0)))
        except Exception:
            pass

    def _play_midi(self) -> None:
        # Delegate to unified helper without a time cursor start
        self._play_midi_with_prompt(start_units=None)

    def _stop_midi(self) -> None:
        try:
            if hasattr(self, 'player') and self.player is not None:
                self.player.stop()
            # Clear playhead overlay when stopping
            self._clear_playhead_overlay()
        except Exception:
            pass

    def _scroll_editor_to_start(self) -> None:
        """Ensure the editor viewport is at the start before playback begins."""
        try:
            if hasattr(self, 'editor_vscroll') and self.editor_vscroll is not None:
                self.editor_vscroll.setValue(0)
            elif hasattr(self, 'editor') and self.editor_canvas is not None:
                self.editor_canvas.set_scroll_logical_px(0)
        except Exception:
            pass

    def _play_midi_with_prompt(self, start_units: Optional[float]) -> None:
        """Play the SCORE from start or the editor time cursor using active backend."""
        try:
            self._ensure_player_with_soundfont()
            sc = self.file_manager.current()
            if start_units is None:
                self._scroll_editor_to_start()
            if start_units is None:
                self.player.play_score(sc)
            else:
                self.player.play_from_time_cursor(float(start_units or 0.0), sc)
            self._start_playhead_timer()
            self._show_play_debug_status()
        except Exception as exc:
            try:
                QtWidgets.QMessageBox.critical(self, "Playback", f"Playback failed: {exc}")
            except Exception:
                print(f"Playback failed: {exc}")

    def _start_playhead_timer(self) -> None:
        try:
            if hasattr(self, '_playhead_timer') and self._playhead_timer is not None:
                if not self._playhead_timer.isActive():
                    self._playhead_timer.start()
            # Immediate update for responsiveness
            self._update_playhead_overlay()
        except Exception:
            pass

    def _show_play_debug_status(self) -> None:
        try:
            if hasattr(self, 'player') and self.player is not None and hasattr(self.player, 'get_debug_status'):
                info = self.player.get_debug_status()
                bpm = info.get('bpm', 0)
                ev = info.get('events', 0)
                gain = info.get('gain', 0.0)
                playback_type = str(info.get('playback_type', '') or '')
                if playback_type == 'fluidsynth':
                    sf = info.get('soundfont', '') or 'FluidSynth'
                    backend_info = f"Soundfont: {sf}"
                elif playback_type == 'coreaudio-dls':
                    backend_info = "Output: Apple DLS Synth"
                else:
                    out_name = info.get('output', '') or 'System MIDI'
                    backend_info = f"Output: {out_name}"
                self._status(f"Playing • {ev} notes • {bpm:.0f} BPM • {backend_info} • Gain: {gain:.2f}", 3000)
        except Exception:
            pass

    def _update_playhead_overlay(self) -> None:
        if hasattr(self, 'player') and self.player is not None and hasattr(self.player, 'is_playing') and self.player.is_playing():
            units = None
            sc = None
            sc = self.file_manager.current() if hasattr(self, 'file_manager') else None
            units = self.player.get_playhead_time(sc) if hasattr(self.player, 'get_playhead_time') else None
            # Update editor playhead and trigger overlay refresh
            self.editor_controller.playhead_time = units
            # Center the playhead in the viewport while playing
            if getattr(self, "_center_playhead_enabled", True):
                self._center_playhead_scroll(units)
            if hasattr(self.editor_canvas, 'request_overlay_refresh'):
                self.editor_canvas.request_overlay_refresh()
            else:
                self.editor_canvas.update()
                pass
        else:
            # Not playing: clear and stop timer
            self._clear_playhead_overlay()

    def _center_playhead_scroll(self, units: Optional[float]) -> None:
        if units is None:
            return
        try:
            ed = getattr(self, 'editor_controller', None)
            if ed is None:
                return
            abs_mm = float(ed.time_to_mm(float(units)))
            vp_h_mm = float(getattr(ed, '_viewport_h_mm', 0.0) or 0.0)
            if vp_h_mm <= 0.0:
                return
            target_top_mm = max(0.0, abs_mm - (vp_h_mm * 0.5))
            px_per_mm = float(getattr(ed, '_px_per_mm', 0.0) or 0.0)
            dpr = float(getattr(ed, '_dpr', 1.0) or 1.0)
            if px_per_mm <= 0.0:
                return
            target_scroll = int(round(target_top_mm * px_per_mm / max(1e-6, dpr)))
            if hasattr(self, 'editor_vscroll') and self.editor_vscroll is not None:
                max_scroll = int(self.editor_vscroll.maximum())
                target_scroll = max(0, min(target_scroll, max_scroll))
                if int(self.editor_vscroll.value()) != target_scroll:
                    self.editor_vscroll.setValue(target_scroll)
            elif hasattr(self, 'editor') and self.editor_canvas is not None:
                self.editor_canvas.set_scroll_logical_px(target_scroll)
        except Exception:
            pass


    def _clear_playhead_overlay(self) -> None:
        try:
            if hasattr(self, '_playhead_timer') and self._playhead_timer is not None and self._playhead_timer.isActive():
                self._playhead_timer.stop()
        except Exception:
            pass
        try:
            self.editor_controller.playhead_time = None
            if hasattr(self.editor_canvas, 'request_overlay_refresh'):
                self.editor_canvas.request_overlay_refresh()
            else:
                self.editor_canvas.update()
        except Exception:
            pass

    # FX/editor hooks removed; FluidSynth is the single backend
    def _open_fx_editor(self) -> None:
        self._status("Synth FX editor removed", 2000)

    def _set_playback_mode(self, mode: str, show_status: bool = True) -> None:
        mode_norm = str(mode or 'system').strip().lower()
        if mode_norm not in ('system', 'external'):
            mode_norm = 'system'
        self._send_playback_panic()
        self._dispose_player()
        self._set_playback_mode_to_appdata(mode_norm)
        try:
            if hasattr(self, '_playback_mode_system_action'):
                self._playback_mode_system_action.setChecked(mode_norm == 'system')
            if hasattr(self, '_playback_mode_external_action'):
                self._playback_mode_external_action.setChecked(mode_norm == 'external')
            if hasattr(self, '_midi_port_menu'):
                self._midi_port_menu.setEnabled(mode_norm == 'external')
        except Exception:
            pass
        if mode_norm == 'external':
            self._rebuild_midi_port_menu()
        if show_status:
            if mode_norm == 'external':
                self._status("Playback mode: External MIDI port", 2500)
            else:
                self._status(f"Playback mode: {self._playback_system_label().replace('Playback using ', '')}", 2500)
        try:
            # Recreate backend immediately so audition/test tone keep working after a mode switch.
            self._ensure_player_with_soundfont()
        except Exception:
            pass

    def _set_send_midi_transport(self, enabled: bool) -> None:
        # Legacy stub kept for signal compatibility
        self._status("MIDI transport settings removed", 2000)

    def _play_test_tone(self) -> None:
        try:
            self._ensure_player_with_soundfont()
            self.player.audition_note(pitch=49, velocity=100, duration_sec=1.0)
            self._status("Test tone", 1500)
        except Exception:
            self._status("Test tone unavailable", 2000)

    def _choose_audio_device(self) -> None:
        self._status("Audio output is selected by the active playback backend", 2000)

    def _play_system_test_tone(self) -> None:
        self._play_test_tone()

    def _force_redraw(self, *_args) -> None:
        # Rebuild editor caches and hit-rects for immediate tool feedback
        if hasattr(self, 'editor_controller') and self.editor_controller is not None:
            if hasattr(self.editor_controller, 'force_redraw_from_model'):
                self.editor_controller.force_redraw_from_model()
            else:
                self.editor_controller.draw_frame()
        # Also refresh the canvas overlays so guide stem direction updates instantly
        try:
            if hasattr(self, 'editor') and self.editor_canvas is not None:
                if hasattr(self.editor_canvas, 'request_overlay_refresh'):
                    self.editor_canvas.request_overlay_refresh()
                else:
                    # Fallback: normal repaint
                    self.editor_canvas.update()
        except Exception:
            pass

    def _adjust_docks_to_fit(self) -> None:
        # Ensure both docks are sized and locked to their fit dimensions
        try:
            if hasattr(self.snap_dock, 'selector'):
                self.snap_dock.selector.adjust_to_fit()
        except Exception:
            pass
        try:
            self.tool_dock.adjust_to_fit()
        except Exception:
            pass
        self._freeze_left_panel_width_once()

    def _freeze_left_panel_width_once(self) -> None:
        if self._left_panel_width_frozen:
            return
        if not hasattr(self, 'snap_dock') or not hasattr(self, 'tool_dock'):
            return

        snap_width = int(self.snap_dock.width())
        tool_width = int(self.tool_dock.width())
        if snap_width <= 0:
            snap_width = int(self.snap_dock.sizeHint().width())
        if tool_width <= 0:
            tool_width = int(self.tool_dock.sizeHint().width())

        target_width = max(1, snap_width, tool_width)
        self.snap_dock.setMinimumWidth(target_width)
        self.snap_dock.setMaximumWidth(target_width)
        self.tool_dock.setMinimumWidth(target_width)
        self.tool_dock.setMaximumWidth(target_width)
        self._left_panel_width_frozen = True

    def resizeEvent(self, ev: QtGui.QResizeEvent) -> None:
        super().resizeEvent(ev)
        ...

    def _on_snap_changed(self, base: int, divide: int) -> None:
        # Update editor snap size units and request a redraw
        size_units = self.snap_dock.selector.get_snap_size()
        if hasattr(self, 'editor_controller') and self.editor_controller is not None:
            self.editor_controller.set_snap_size_units(size_units)
            self.editor_controller.draw_frame()

        if hasattr(self, 'editor_canvas') and self.editor_canvas is not None:
            self.editor_canvas.update()

        logical_px_step = self._editor_scroll_step_from_metrics(self._editor_metric_px_per_mm, self._editor_metric_dpr)
        self._editor_scroll_step_logical_px = int(max(1, logical_px_step))
        self.editor_vscroll.setSingleStep(int(self._editor_scroll_step_logical_px))
        self.editor_canvas.set_scroll_step_logical_px(int(self._editor_scroll_step_logical_px))

        # Persist to app state
        app_state = self._current_app_state()
        app_state.snap_base = int(base)
        app_state.snap_divide = int(divide)

    def _on_tool_selected(self, name: str) -> None:
        # Persist selected tool to app state
        try:
            app_state = self._current_app_state()
            app_state.selected_tool = str(name)
        except Exception:
            pass

    def schedule_edwin_prompt(self, delay_ms: int = 150) -> None:
        """Schedule the Edwin font install prompt once, after any other startup dialogs."""
        if self._edwin_prompt_armed:
            return
        self._edwin_prompt_armed = True
        QtCore.QTimer.singleShot(max(0, int(delay_ms)), self._maybe_prompt_edwin_install)

    def _maybe_prompt_edwin_install(self) -> None:
        fonts = [
            {
                "name": "Edwin",
                "installed_key": "edwin_font_installed",
                "dismissed_key": "edwin_install_prompt_dismissed",
                "desc": "Edwin font for headers and engraving (recommended)",
            },
            {
                "name": "Latin Modern Roman Caps",
                "installed_key": "lmromancaps_font_installed",
                "dismissed_key": "lmromancaps_install_prompt_dismissed",
                "desc": "Latin Modern Roman Caps for engraving text and titles (recommended)",
            },
            {
                "name": "Latin Modern Roman",
                "installed_key": "lmroman_font_installed",
                "dismissed_key": "lmroman_install_prompt_dismissed",
                "desc": "Latin Modern Roman for engraving text and titles (recommended)",
            },
        ]
        try:
            adm = get_appdata_manager()
        except Exception:
            return
        try:
            from fonts import has_system_font, install_embedded_font_to_system
        except Exception:
            return
        try:
            # Build list of fonts that are not yet installed
            missing: list[dict] = []
            all_dismissed = True
            for f in fonts:
                name = f["name"]
                installed_key = f["installed_key"]
                dismissed_key = f["dismissed_key"]
                is_dismissed = bool(adm.get(dismissed_key, False))
                all_dismissed = all_dismissed and is_dismissed
                if has_system_font(name):
                    adm.set(installed_key, True)
                    continue
                missing.append(f)
            if not missing:
                adm.save()
                return
            if all_dismissed:
                return
            msg = QtWidgets.QMessageBox(self)
            msg.setIcon(QtWidgets.QMessageBox.Icon.Information)
            msg.setWindowTitle("Install recommended fonts")
            lines = ["keyTAB can install embedded fonts so the preview matches prints/PDFs (recommended):"]
            for f in missing:
                lines.append(f"- {f['name']}: {f['desc']}")
            lines.append("Install these to your user font folder now?")
            msg.setText("\n".join(lines))
            msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
            msg.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Yes)
            result = msg.exec()
            if result != QtWidgets.QMessageBox.StandardButton.Yes:
                for f in fonts:
                    adm.set(f["dismissed_key"], True)
                adm.save()
                return
            successes = []
            failures = []
            for f in missing:
                name = f["name"]
                installed_key = f["installed_key"]
                success, detail = install_embedded_font_to_system(name)
                if success:
                    adm.set(installed_key, True)
                    successes.append(name)
                else:
                    failures.append((name, detail))
            adm.save()
            if successes:
                QtWidgets.QMessageBox.information(
                    self,
                    "Fonts installed",
                    "The following fonts were installed. keyTAB will restart to apply them:\n" + "\n".join(successes),
                )
                QtCore.QTimer.singleShot(100, self._request_app_restart)
            if failures:
                details = "\n".join([f"{n}: {d}" for n, d in failures])
                QtWidgets.QMessageBox.warning(
                    self,
                    "Font installation failed",
                    f"keyTAB could not install some fonts automatically:\n{details}",
                )
        except Exception:
            pass

    def _request_app_restart(self) -> None:
        try:
            exe = sys.executable
            args = list(sys.argv)
            if exe and args:
                QtCore.QProcess.startDetached(exe, args)
        except Exception:
            pass
        try:
            self.prepare_close()
        except Exception:
            pass
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.quit()

    def _center_on_primary(self) -> None:
        # Move window to the center of the primary screen
        try:
            # If the window is maximized or fullscreen, do not attempt to center
            if self.isMaximized() or self.isFullScreen():
                return
            scr = QtGui.QGuiApplication.primaryScreen()
            if not scr:
                return
            avail = scr.availableGeometry()
            if not avail.isValid():
                return
            fg = self.frameGeometry()
            fg.moveCenter(avail.center())
            self.move(fg.topLeft())
        except Exception:
            pass

    # Duplicate keyPressEvent removed; using the earlier implementation for Escape handling

    def prepare_close(self) -> None:
        # Ensure worker threads are stopped before application exits
        # Persist window state to appdata
        try:
            adm = get_appdata_manager()
            adm.set("window_maximized", bool(self.isMaximized()))
            try:
                geom_b64 = bytes(self.saveGeometry().toBase64()).decode("ascii")
                adm.set("window_geometry", geom_b64)
            except Exception:
                pass
            # Save current splitter sizes for next startup
            try:
                sp = self.centralWidget()
                if sp is not None and hasattr(sp, 'sizes'):
                    sizes = list(sp.sizes())
                    adm.set("splitter_sizes", [int(sizes[0]) if sizes else 0, int(sizes[1]) if len(sizes) > 1 else 0])
            except Exception:
                pass
            # Persist whether the session is currently saved to a project file
            try:
                fm = getattr(self, 'file_manager', None)
                if fm is not None:
                    # Session considered saved if we have a project path and it's not dirty
                    was_saved = bool(fm.path() is not None and not fm.is_dirty())
                    adm.set("last_session_saved", was_saved)
                    adm.set("last_session_path", str(fm.path() or ""))
            except Exception:
                pass
            adm.save()
        except Exception:
            pass
        # Stop clock timer gracefully
        try:
            if hasattr(self, "_clock_timer") and self._clock_timer is not None:
                self._clock_timer.stop()
        except Exception:
            pass
        # Stop audio playback gracefully
        try:
            if hasattr(self, 'player') and self.player is not None:
                self.player.stop()
        except Exception:
            pass
        # Stop playhead timer and clear overlay
        try:
            self._clear_playhead_overlay()
        except Exception:
            pass
        # Close FX dialog if open
        try:
            if hasattr(self, '_fx_dialog') and self._fx_dialog is not None:
                self._fx_dialog.close()
                self._fx_dialog = None
        except Exception:
            pass
        if hasattr(self, "print_view") and self.print_view is not None:
            try:
                self.print_view.shutdown()
            except Exception:
                pass
        if hasattr(self, "engraver") and self.engraver is not None:
            try:
                self.engraver.shutdown()
            except Exception:
                pass

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        # Unified close handling: save session and close without prompting.
        try:
            self.file_manager.autosave_current()
        except Exception:
            pass

        save_on_exit = True
        try:
            pm = get_preferences_manager()
            save_on_exit = bool(pm.get("save_on_exit", True))
        except Exception:
            pass
        if save_on_exit:
            try:
                if self.file_manager.path() is not None:
                    self.file_manager.save()
            except Exception:
                pass
        try:
            adm = get_appdata_manager()
            was_saved = bool(self.file_manager.path() is not None and not self.file_manager.is_dirty())
            adm.set("last_session_saved", was_saved)
            adm.set("last_session_path", str(self.file_manager.path() or ""))
            adm.save()
        except Exception:
            pass
        # Persist sizes via prepare_close
        try:
            self.prepare_close()
        except Exception:
            pass
        ev.accept()
