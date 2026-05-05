from PySide6 import QtCore, QtGui, QtWidgets
from typing import Optional
import sys, os, time
import traceback
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
from ui.error_dialog import show_error_dialog
from ui.style import Style
from ui.dialogs.fluidsynth_reverb_config_dialog import FluidSynthReverbConfigDialog
from settings_manager import open_preferences, get_preferences_manager
from appdata_manager import get_appdata_manager
from utils.CONSTANT import UTILS_SAVE_DIR, QUARTER_NOTE_UNIT
from utils.restart import restart_current_process
from engraver.engraver import Engraver
from editor.tool_manager import ToolManager
from editor.editor import Editor
from scripting.engine import ScriptEngine


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(self.tr("keyTAB - new project (unsaved)"))
        self.resize(1200, 800)
        self.setAcceptDrops(True)
        # Ensure player attribute always exists
        self.player = None
        self._fluidsynth_missing_warned = False
        self._player_config: tuple[str, str] | None = None
        self._left_panel_width_frozen = False
        self._prepare_close_done = False
        try:
            adm_width = get_appdata_manager()
            self._left_panel_width_pref_px = int(max(1, adm_width.get("left_panel_width_px", 220)))
        except Exception:
            self._left_panel_width_pref_px = 220
        self._left_panel_width_last_saved_px = int(self._left_panel_width_pref_px)
        self._close_restore_saved_override: bool | None = None
        self._close_restore_path_override: str | None = None
        self._left_panel_width_save_timer = QtCore.QTimer(self)
        self._left_panel_width_save_timer.setSingleShot(True)
        self._left_panel_width_save_timer.setInterval(250)
        self._left_panel_width_save_timer.timeout.connect(self._persist_left_panel_width)
        self._editor_scroll_step_logical_px: int = 1

        # File management
        self.file_manager = FileManager(self)
        self.file_manager.set_before_save_hook(self._collect_app_state_for_save)
        self.file_manager.midi_import_hook = self._handle_midi_import

        # View options
        try:
            pm = get_preferences_manager()
            self._center_playhead_enabled = bool(
                pm.get("focus_on_playhead_during_playback", pm.get("center_view_on_playhead", True))
            )
        except Exception:
            self._center_playhead_enabled = True
        self._playhead_anchor_measure: int | None = None
        self._playhead_last_visible_measure: int | None = None
        self._last_engraver_error_signature: str | None = None
        
        # Install error-backup hook early so any unhandled exception triggers a backup
        self.file_manager.install_error_backup_hook()

        # Periodic autosave (session + project) to reduce per-action latency
        self._autosave_timer = QtCore.QTimer(self)
        self._autosave_timer.timeout.connect(self._on_autosave_timer)
        self._apply_autosave_preferences()

        self._create_menus()

        self.splitter = ToolbarSplitter(QtCore.Qt.Orientation.Horizontal)
        
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
        self.engraver.failed.connect(self._on_engraver_failed)
        
        # Startup restore: prefer opening the last saved project; else restore unsaved session; else new
        self._session_restore_mode: bool = False
        adm2 = None
        was_saved = False
        saved_path = ""
        adm2 = get_appdata_manager()
        was_saved = bool(adm2.get("last_session_saved", False))
        saved_path = str(adm2.get("last_session_path", "") or "")
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

        if not was_saved:
            # In unsaved/new mode, restore the session snapshot first and do not reopen old project paths.
            restored = False
            try:
                restored = self.file_manager.load_session_if_available()
            except Exception:
                restored = False
            if restored:
                opened = True
                self._session_restore_mode = True
                status_msg = "Restored unsaved session (session.piano mode)"

        if not opened:
            # For saved sessions, try real project paths first (handles delayed cloud mounts).
            if saved_path:
                sc = _try_open_path_with_retries(saved_path, retries=16, delay_sec=0.25)
                if sc is not None:
                    opened = True
                    self._session_restore_mode = False
                    status_msg = f"Opened last saved project: {saved_path}"

        if not opened and was_saved:
            if adm2 is None:
                adm2 = get_appdata_manager()
            last_path = str(adm2.get("last_opened_file", "") or "")
            if last_path and str(last_path) != str(saved_path):
                sc = _try_open_path_with_retries(last_path, retries=16, delay_sec=0.25)
                if sc is not None:
                    opened = True
                    self._session_restore_mode = False
                    status_msg = f"Opened last project: {last_path}"

        if not opened:
            # Fallback to session restore, then new score.
            restored = False
            try:
                restored = self.file_manager.load_session_if_available()
            except Exception:
                restored = False
            if not restored:
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
        try:
            self._refresh_recent_files_menu()
        except Exception:
            pass

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
        # Keep separators slim but clickable so left dock width can be adjusted
        self.setStyleSheet(
            "QMainWindow::separator { width: 6px; height: 6px; background: transparent; margin: 0px; }\n"
            "QMainWindow::separator:hover { background: rgb(180,180,180); }"
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

        self.script_engine = ScriptEngine(self.file_manager, self.editor_controller, parent=self)

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
        self.splitter.fitRequested.connect(self.print_view.reset_view_state)
        self.splitter.fitRequested.connect(self._force_redraw)
        # Any manual splitter movement should return print-view Ctrl/Cmd zoom to idle.
        self.splitter.splitterMoved.connect(self._on_splitter_moved)
        # Default toolbar actions
        self.splitter.nextRequested.connect(self._next_page)
        self.splitter.nextRequested.connect(self._force_redraw)
        self.splitter.previousRequested.connect(self._previous_page)
        self.splitter.previousRequested.connect(self._force_redraw)
        self.splitter.undoRequested.connect(self._edit_undo)
        self.splitter.undoRequested.connect(self._force_redraw)
        self.splitter.redoRequested.connect(self._edit_redo)
        self.splitter.redoRequested.connect(self._force_redraw)
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
        # Defer the font install prompt until explicitly scheduled by the app (after AppImage install prompt)
        self._fonts_prompt_armed = False

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
        except Exception as exc:
            # Player initialization is optional at startup; keep attribute defined
            self.player = None
            self._notify_fluidsynth_missing(exc)
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
        # Number keys 1..8 control snap selector/listbox + divider.
        # Applies to both top-row digits and numpad digits.
        try:
            if self._handle_snap_number_shortcut(ev):
                ev.accept()
                return
        except Exception:
            pass

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

    def changeEvent(self, ev: QtCore.QEvent) -> None:
        super().changeEvent(ev)
        if ev.type() == QtCore.QEvent.Type.WindowStateChange:
            self._sync_full_screen_action_state()

    def _is_text_input_focus(self) -> bool:
        fw = QtWidgets.QApplication.focusWidget()
        return isinstance(
            fw,
            (
                QtWidgets.QLineEdit,
                QtWidgets.QTextEdit,
                QtWidgets.QPlainTextEdit,
                QtWidgets.QAbstractSpinBox,
                QtWidgets.QComboBox,
            ),
        )

    def _handle_snap_number_shortcut(self, ev: QtGui.QKeyEvent) -> bool:
        # Do not consume numeric keys while editing text values.
        if self._is_text_input_focus():
            return False

        if not hasattr(self, 'snap_dock') or not hasattr(self.snap_dock, 'selector'):
            return False

        # Accept no modifiers or keypad-only modifier (numpad).
        mods = ev.modifiers()
        if bool(mods & ~QtCore.Qt.KeyboardModifier.KeypadModifier):
            return False

        key = int(ev.key())
        digit_map = {
            int(QtCore.Qt.Key.Key_1): 1,
            int(QtCore.Qt.Key.Key_2): 2,
            int(QtCore.Qt.Key.Key_3): 3,
            int(QtCore.Qt.Key.Key_4): 4,
            int(QtCore.Qt.Key.Key_5): 5,
            int(QtCore.Qt.Key.Key_6): 6,
            int(QtCore.Qt.Key.Key_7): 7,
            int(QtCore.Qt.Key.Key_8): 8,
        }
        digit = digit_map.get(key)
        if digit is None:
            return False

        selector = self.snap_dock.selector
        base = int(selector.get_snap_base() or 8)
        divide = int(selector.get_snap_divide() or 1)

        # Requested mapping:
        # 1 whole, 2 half, 3 divide=3, 4 quarter,
        # 5 divide=5, 6 sixteenth, 7 divide=7, 8 eighth.
        if digit == 1:
            base = 1
            divide = 1
        elif digit == 2:
            base = 2
            divide = 1
        elif digit == 3:
            divide = 3
        elif digit == 4:
            base = 4
            divide = 1
        elif digit == 5:
            divide = 5
        elif digit == 6:
            base = 16
            divide = 1
        elif digit == 7:
            divide = 7
        elif digit == 8:
            base = 8
            divide = 1

        selector.set_snap(base, divide, emit=True)
        return True

    def _create_menus(self) -> None:
        tr = self.tr
        menubar = self.menuBar()
        # Keep menu inside the app window on macOS (also in fullscreen).
        if sys.platform == "darwin":
            menubar.setNativeMenuBar(False)

        # Create menus in normal left-to-right order (File, Edit, Selection, Document, Tools, View, Playback, About)
        file_menu = menubar.addMenu(tr("&File"))
        edit_menu = menubar.addMenu(tr("&Edit"))
        view_menu = menubar.addMenu(tr("&View"))
        selection_menu = menubar.addMenu(tr("&Selection"))
        document_menu = menubar.addMenu(tr("&Document"))
        tools_menu = menubar.addMenu(tr("&Tools"))
        playback_menu = menubar.addMenu(tr("&Playback"))
        help_menu = menubar.addMenu(tr("&About"))
        for menu in (file_menu, edit_menu, selection_menu, view_menu, document_menu, tools_menu, playback_menu, help_menu):
            menu.setToolTipsVisible(True)

        # File actions
        new_act = QtGui.QAction(tr("New"), self)
        new_act.setToolTip(tr("Create a new project."))
        open_act = QtGui.QAction(tr("Load..."), self)
        open_act.setToolTip(tr("Open an existing project file."))
        save_act = QtGui.QAction(tr("Save"), self)
        save_act.setToolTip(tr("Save the current project."))
        save_as_act = QtGui.QAction(tr("Save As..."), self)
        save_as_act.setToolTip(tr("Save the current project under a new file name."))
        exit_act = QtGui.QAction(tr("Exit"), self)
        exit_act.setToolTip(tr("Exit the application."))
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
        self._recent_menu = file_menu.addMenu(tr("Recent Files"))
        self._recent_menu.setToolTipsVisible(True)
        self._rename_file_act = QtGui.QAction(tr("Rename..."), self)
        self._rename_file_act.setToolTip(tr("Rename the currently opened file and update Recent Files."))
        self._rename_file_act.triggered.connect(self._rename_current_file)
        file_menu.addAction(self._rename_file_act)
        self._refresh_rename_file_action()
        file_menu.addSeparator()

        set_default_style_act = QtGui.QAction(tr("Set current style as default"), self)
        set_default_style_act.setToolTip(tr("Save the current style as the default for new projects."))
        set_default_style_act.triggered.connect(self._set_default_style)
        file_menu.addAction(set_default_style_act)

        reset_default_style_act = QtGui.QAction(tr("Reset default style"), self)
        reset_default_style_act.setToolTip(tr("Remove the custom default style and use the built-in defaults."))
        reset_default_style_act.triggered.connect(self._reset_default_style)
        file_menu.addAction(reset_default_style_act)
        file_menu.addSeparator()

        style_act = QtGui.QAction(tr("Style..."), self)
        style_act.setToolTip(tr("Open appearance settings for the score."))
        style_act.setShortcut(QtGui.QKeySequence("S"))
        style_act.triggered.connect(self._open_style_dialog)
        info_act = QtGui.QAction(tr("Info..."), self)
        info_act.setToolTip(tr("Open title and metadata settings."))
        info_act.setShortcut(QtGui.QKeySequence("I"))
        info_act.triggered.connect(self._open_info_dialog)
        line_break_act = QtGui.QAction(tr("Line Breaks..."), self)
        line_break_act.setToolTip(tr("Open line break and page break settings."))
        line_break_act.setShortcut(QtGui.QKeySequence("L"))
        line_break_act.triggered.connect(self._open_line_break_dialog)

        document_menu.addAction(style_act)
        document_menu.addAction(info_act)
        document_menu.addAction(line_break_act)
        document_menu.addSeparator()

        self._tools_menu = tools_menu
        tools_menu.aboutToShow.connect(self._rebuild_tools_menu)
        self._rebuild_tools_menu()

        export_pdf_act = QtGui.QAction(tr("Export PDF..."), self)
        export_pdf_act.setToolTip(tr("Export the current score as a PDF document."))
        export_pdf_act.setShortcut(QtGui.QKeySequence("Ctrl+E"))
        export_pdf_act.triggered.connect(self._export_pdf)
        file_menu.addAction(export_pdf_act)

        export_image_pdf_act = QtGui.QAction(tr("Export Image PDF..."), self)
        export_image_pdf_act.setToolTip(tr("Export the current score as a rasterized PDF document (600 DPI)."))
        export_image_pdf_act.setShortcut(QtGui.QKeySequence("Ctrl+Shift+E"))
        export_image_pdf_act.triggered.connect(self._export_image_pdf)
        file_menu.addAction(export_image_pdf_act)

        # Playback menu
        self._playback_menu = playback_menu
        self._playback_mode_group = QtGui.QActionGroup(self)
        self._playback_mode_group.setExclusive(True)

        self._playback_mode_system_action = QtGui.QAction(self._playback_system_label(), self)
        self._playback_mode_system_action.setToolTip(tr("Use the system playback backend."))
        self._playback_mode_system_action.setCheckable(True)
        self._playback_mode_system_action.triggered.connect(lambda checked: self._set_playback_mode('system') if checked else None)
        playback_menu.addAction(self._playback_mode_system_action)
        self._playback_mode_group.addAction(self._playback_mode_system_action)

        self._playback_mode_external_action = QtGui.QAction(tr("Playback using External MIDI port"), self)
        self._playback_mode_external_action.setToolTip(tr("Use an external MIDI output port for playback."))
        self._playback_mode_external_action.setCheckable(True)
        self._playback_mode_external_action.triggered.connect(lambda checked: self._set_playback_mode('external') if checked else None)
        playback_menu.addAction(self._playback_mode_external_action)
        self._playback_mode_group.addAction(self._playback_mode_external_action)

        playback_menu.addSeparator()
        self._midi_port_menu = playback_menu.addMenu(tr("MIDI port"))
        self._midi_port_menu.setToolTipsVisible(True)
        self._midi_port_menu.aboutToShow.connect(self._rebuild_midi_port_menu)
        self._rebuild_midi_port_menu()
        playback_menu.addSeparator()

        test_tone_act = QtGui.QAction(tr("Play Test Tone"), self)
        test_tone_act.setToolTip(tr("Play a short test tone."))
        test_tone_act.triggered.connect(self._play_test_tone)
        playback_menu.addAction(test_tone_act)

        if sys.platform.startswith("linux"):
            playback_menu.addSeparator()
            select_sf_act = QtGui.QAction(tr("Select Custom SoundFont (.sf2/.sf3) for FluidSynth"), self)
            select_sf_act.setToolTip(tr("Select a custom SoundFont file for FluidSynth playback."))
            select_sf_act.triggered.connect(lambda: self._prompt_for_soundfont(force_dialog=True))
            playback_menu.addAction(select_sf_act)

            unset_sf_act = QtGui.QAction(tr("Use Default FluidSynth SoundFont"), self)
            unset_sf_act.setToolTip(tr("Switch back to the default FluidSynth SoundFont."))
            unset_sf_act.triggered.connect(self._unset_soundfont)
            playback_menu.addAction(unset_sf_act)

            reverb_config_act = QtGui.QAction(tr("FluidSynth Settings"), self)
            reverb_config_act.setToolTip(tr("Configure FluidSynth playback and reverb parameters."))
            reverb_config_act.triggered.connect(self._open_reverb_config_dialog)
            playback_menu.addAction(reverb_config_act)

        self._set_playback_mode(str(self._get_playback_mode_from_appdata() or 'system'), show_status=False)

        about_act = QtGui.QAction(tr("About keyTAB"), self)
        about_act.setToolTip(tr("Show information about keyTAB."))
        about_act.triggered.connect(self._open_about_dialog)
        about_qt_act = QtGui.QAction(tr("About Qt"), self)
        about_qt_act.setToolTip(tr("Show information about the Qt framework."))
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
        undo_act = QtGui.QAction(tr("Undo"), self)
        undo_act.setToolTip(tr("Undo the last editing action."))
        undo_act.setShortcut(QtGui.QKeySequence.StandardKey.Undo)
        redo_act = QtGui.QAction(tr("Redo"), self)
        redo_act.setToolTip(tr("Redo the last undone editing action."))
        # Use platform-aware Redo shortcut to avoid ambiguity; explicit combos handled in editor
        try:
            redo_act.setShortcut(QtGui.QKeySequence.StandardKey.Redo)
        except Exception:
            pass
        edit_menu.addAction(undo_act)
        edit_menu.addAction(redo_act)
        # Cut/Copy/Paste actions (platform-aware shortcuts)
        cut_act = QtGui.QAction(tr("Cut"), self)
        cut_act.setToolTip(tr("Cut the current selection."))
        cut_act.setShortcut(QtGui.QKeySequence.StandardKey.Cut)
        copy_act = QtGui.QAction(tr("Copy"), self)
        copy_act.setToolTip(tr("Copy the current selection."))
        copy_act.setShortcut(QtGui.QKeySequence.StandardKey.Copy)
        paste_act = QtGui.QAction(tr("Paste"), self)
        paste_act.setToolTip(tr("Paste clipboard content."))
        paste_act.setShortcut(QtGui.QKeySequence.StandardKey.Paste)
        edit_menu.addSeparator()
        edit_menu.addAction(cut_act)
        edit_menu.addAction(copy_act)
        edit_menu.addAction(paste_act)
        # Delete selection action with visible shortcuts (Delete, Backspace)
        delete_act = QtGui.QAction(tr("Delete"), self)
        delete_act.setToolTip(tr("Delete the current selection."))
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
        select_all_act = QtGui.QAction(tr("Select All"), self)
        select_all_act.setToolTip(tr("Select all editable events."))
        select_all_act.setShortcut(QtGui.QKeySequence.StandardKey.SelectAll)

        _is_horizontal = str(get_preferences_manager().get('editor_orientation', 'vertical') or 'vertical').strip().lower() == 'horizontal'
        _transpose_key_neg = QtCore.Qt.Key_Down if _is_horizontal else QtCore.Qt.Key_Left
        _transpose_key_pos = QtCore.Qt.Key_Up if _is_horizontal else QtCore.Qt.Key_Right
        _shift_key_earlier = QtCore.Qt.Key_Left if _is_horizontal else QtCore.Qt.Key_Up
        _shift_key_later = QtCore.Qt.Key_Right if _is_horizontal else QtCore.Qt.Key_Down

        transpose_left_act = QtGui.QAction(tr("Transpose -1 Semitone"), self)
        transpose_left_act.setToolTip(tr("Transpose Selection Down by One Semitone."))
        transpose_left_act.setShortcut(QtGui.QKeySequence(_transpose_key_neg))
        transpose_left_act.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetShortcut)

        transpose_right_act = QtGui.QAction(tr("Transpose +1 Semitone"), self)
        transpose_right_act.setToolTip(tr("Transpose Selection Up by One Semitone."))
        transpose_right_act.setShortcut(QtGui.QKeySequence(_transpose_key_pos))
        transpose_right_act.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetShortcut)

        shift_earlier_act = QtGui.QAction(tr("Move Earlier by Snap Band"), self)
        shift_earlier_act.setToolTip(tr("Move Selection Earlier by One Snap Band."))
        shift_earlier_act.setShortcut(QtGui.QKeySequence(_shift_key_earlier))
        shift_earlier_act.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetShortcut)

        shift_later_act = QtGui.QAction(tr("Move Later by Snap Band"), self)
        shift_later_act.setToolTip(tr("Move Selection Later by One Snap Band."))
        shift_later_act.setShortcut(QtGui.QKeySequence(_shift_key_later))
        shift_later_act.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetShortcut)

        quantize_act = QtGui.QAction(tr("Quantize Starts and Ends on Snap Band"), self)
        quantize_act.setToolTip(tr("Quantize Selection Starts and Ends to the Current Snap Band."))
        quantize_act.setShortcut(QtGui.QKeySequence("Q"))
        quantize_act.setShortcutContext(QtCore.Qt.ShortcutContext.WidgetShortcut)
        quantize_start_act = QtGui.QAction(tr("Quantize Starts on Snap Band"), self)
        quantize_start_act.setToolTip(tr("Quantize Selection Starts to the Current Snap Band."))
        quantize_end_act = QtGui.QAction(tr("Quantize Ends on Snap Band"), self)
        quantize_end_act.setToolTip(tr("Quantize Selection Ends to the Current Snap Band."))

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
        prefs_act = QtGui.QAction(tr("Preferences..."), self)
        prefs_act.setToolTip(tr("Open application preferences."))
        prefs_act.triggered.connect(self._open_preferences)
        edit_menu.addAction(prefs_act)
        
        # View actions
        zoom_in_act = QtGui.QAction(tr("Zoom In"), self)
        zoom_in_act.setToolTip(tr("Zoom in on the editor view."))
        try:
            zoom_in_act.setShortcuts([
                QtGui.QKeySequence("="),
                QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.ZoomIn)
            ])
        except Exception:
            zoom_in_act.setShortcut(QtGui.QKeySequence("="))
        zoom_out_act = QtGui.QAction(tr("Zoom Out"), self)
        zoom_out_act.setToolTip(tr("Zoom out from the editor view."))
        try:
            zoom_out_act.setShortcuts([
                QtGui.QKeySequence("-"),
                QtGui.QKeySequence(QtGui.QKeySequence.StandardKey.ZoomOut)
            ])
        except Exception:
            zoom_out_act.setShortcut(QtGui.QKeySequence("-"))
        view_menu.addSeparator()
        full_screen_act = QtGui.QAction(tr("Full Screen"), self)
        full_screen_act.setToolTip(tr("Toggle full screen mode."))
        full_screen_act.setShortcut(QtGui.QKeySequence("F11"))
        full_screen_act.setCheckable(True)
        # Initialize checkbox state to match current window state
        full_screen_act.setChecked(self.isFullScreen())
        view_menu.addAction(zoom_in_act)
        view_menu.addAction(zoom_out_act)
        view_menu.addSeparator()
        view_menu.addAction(full_screen_act)
        view_menu.addSeparator()

        # Language actions
        language_menu = view_menu.addMenu(tr("Language"))
        language_menu.setToolTipsVisible(True)
        self._language_action_group = QtGui.QActionGroup(self)
        self._language_action_group.setExclusive(True)

        self._language_system_action = QtGui.QAction(tr("System"), self)
        self._language_system_action.setToolTip(tr("Use the operating system language for the user interface."))
        self._language_system_action.setCheckable(True)
        self._language_system_action.triggered.connect(
            lambda checked: self._set_ui_language_preference("system") if checked else None
        )
        self._language_action_group.addAction(self._language_system_action)
        language_menu.addAction(self._language_system_action)

        self._language_en_action = QtGui.QAction(tr("English"), self)
        self._language_en_action.setToolTip(tr("Use English for the user interface."))
        self._language_en_action.setCheckable(True)
        self._language_en_action.triggered.connect(
            lambda checked: self._set_ui_language_preference("en") if checked else None
        )
        self._language_action_group.addAction(self._language_en_action)
        language_menu.addAction(self._language_en_action)

        self._language_nl_action = QtGui.QAction(tr("Dutch"), self)
        self._language_nl_action.setToolTip(tr("Use Dutch for the user interface."))
        self._language_nl_action.setCheckable(True)
        self._language_nl_action.triggered.connect(
            lambda checked: self._set_ui_language_preference("nl") if checked else None
        )
        self._language_action_group.addAction(self._language_nl_action)
        language_menu.addAction(self._language_nl_action)

        self._sync_ui_language_actions()

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
        width = max(12, int(extent * 2))
        self.editor_vscroll.setStyleSheet(
            "QScrollBar:vertical {"
            f"width: {width}px;"
            "}"
        )
        self.editor_vscroll.setFixedWidth(int(width))
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

        # Persist left dock width when user drags the dock separator.
        try:
            if et == QtCore.QEvent.Type.Resize:
                if watched is getattr(self, 'snap_dock', None) or watched is getattr(self, 'tool_dock', None):
                    self._schedule_left_panel_width_save()
        except Exception:
            pass

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
            return AppState()
        try:
            if not hasattr(sc, 'app_state') or sc.app_state is None:
                sc.app_state = AppState()
        except Exception:
            sc.app_state = AppState()
        return sc.app_state

    def _collect_app_state_for_save(self, score) -> None:
        """Collect live UI values directly into SCORE.app_state at save time."""
        if score is None:
            return
        try:
            app_state = getattr(score, 'app_state', None)
            if app_state is None:
                app_state = AppState()
                score.app_state = app_state
        except Exception:
            return

        try:
            app_state.print_view_page_index = max(0, int(getattr(self, '_page_counter', 0) or 0))
        except Exception:
            pass
        try:
            if hasattr(self, 'editor_vscroll') and self.editor_vscroll is not None:
                app_state.editor_scroll_pos = int(self.editor_vscroll.value())
        except Exception:
            pass
        try:
            if hasattr(self, 'snap_dock') and hasattr(self.snap_dock, 'selector'):
                app_state.snap_base = int(self.snap_dock.selector.get_snap_base() or app_state.snap_base)
                app_state.snap_divide = int(self.snap_dock.selector.get_snap_divide() or app_state.snap_divide)
        except Exception:
            pass
        try:
            if hasattr(self, 'tool_dock') and hasattr(self.tool_dock, 'selector'):
                items = self.tool_dock.selector.selectedItems()
                if items:
                    selected = items[0].data(QtCore.Qt.ItemDataRole.UserRole)
                    if isinstance(selected, str) and selected:
                        app_state.selected_tool = selected
        except Exception:
            pass

    def _resolve_app_state_defaults(self) -> AppState:
        """Return app state from the currently loaded SCORE only."""
        return self._current_app_state()

    def _on_autosave_timer(self) -> None:
        """Autosave tick: persist via FileManager (SCORE.save collects app_state)."""
        self.file_manager.autosave_all()

    def _restore_app_state_from_score(self) -> None:
        self._is_restoring_app_state = True
        try:
            score = self.file_manager.current()
        except Exception:
            score = None
        try:
            app_state = getattr(score, 'app_state', None)
        except Exception:
            app_state = None
        if app_state is None:
            app_state = AppState()
        # Tool selection
        try:
            self.tool_dock.selector.set_selected_tool(str(app_state.selected_tool or "note"), emit=True)
        except Exception:
            self.editor_controller.set_tool_by_name('note')
        # Snap size
        sb = int(app_state.snap_base or 8)
        sd = int(app_state.snap_divide or 1)
        self.snap_dock.selector.set_snap(sb, sd, emit=True)
        # Scroll restore (used when metrics arrive)
        self._pending_scroll_restore = int(app_state.editor_scroll_pos or 0)
        # Apply immediately on load; keep pending as a fallback if metrics update later.
        try:
            min_v = int(self.editor_vscroll.minimum())
            max_v = int(self.editor_vscroll.maximum())
            target = max(min_v, min(int(self._pending_scroll_restore), max_v))
            self.editor_vscroll.setValue(target)
            self.editor_canvas.set_scroll_logical_px(target)
        except Exception:
            pass
        # Print page restore
        self._page_counter = max(0, int(getattr(app_state, 'print_view_page_index', 0) or 0))
        self._set_page_index(self._page_counter)
        self._is_restoring_app_state = False

    def _read_autosave_preferences(self) -> tuple[bool, int]:
        enabled = True
        interval_minutes = 1
        pm = get_preferences_manager()
        enabled = bool(pm.get("auto_save", True))
        interval_minutes = int(pm.get("auto_save_interval", 1))
        if interval_minutes < 1:
            interval_minutes = 1
        return enabled, interval_minutes

    def _apply_autosave_preferences(self) -> None:
        enabled, interval_minutes = self._read_autosave_preferences()
        interval_ms = int(interval_minutes) * 60_000
        self._autosave_timer.setInterval(interval_ms)
        if enabled:
            self._autosave_timer.start()
        else:
            self._autosave_timer.stop()

    def _toggle_full_screen(self) -> None:
        """Toggle native/fullscreen mode across platforms using F11."""
        if self.isFullScreen() or (self.windowState() & QtCore.Qt.WindowState.WindowFullScreen):
            self.showNormal()
        else:
            self.showFullScreen()
        self._sync_full_screen_action_state()

    def _sync_full_screen_action_state(self) -> None:
        if hasattr(self, '_full_screen_act') and self._full_screen_act is not None:
            self._full_screen_act.setChecked(self.isFullScreen())

    def _playback_system_label(self) -> str:
        if sys.platform.startswith('linux'):
            return self.tr("Playback using FluidSynth")
        if sys.platform == 'darwin':
            return self.tr("Playback using CoreMIDI")
        if sys.platform.startswith('win'):
            return self.tr("Playback using WinMM")
        return self.tr("Playback using System Synth")

    def _get_playback_mode_from_appdata(self) -> str:
        adm = get_appdata_manager()
        mode = str(adm.get("playback_mode", "system") or "system").strip().lower()
        if mode in ("system", "external"):
            return mode
        return "system"

    def _set_playback_mode_to_appdata(self, mode: str) -> None:
        adm = get_appdata_manager()
        adm.set("playback_mode", str(mode))
        adm.save()

    def _get_midi_out_port_from_appdata(self) -> str:
        adm = get_appdata_manager()
        return str(adm.get("midi_out_port", "") or "")

    def _set_midi_out_port_to_appdata(self, port_name: str) -> None:
        adm = get_appdata_manager()
        adm.set("midi_out_port", str(port_name or ""))
        adm.save()

    def _is_fluidsynth_missing_error(self, exc: Exception) -> bool:
        if not sys.platform.startswith('linux'):
            return False
        if self._get_playback_mode_from_appdata() != 'system':
            return False
        text = str(exc or '').lower()
        if 'fluidsynth' not in text:
            return False
        needles = ('not available', 'not installed', 'install', 'pyfluidsynth', 'libfluidsynth')
        return any(n in text for n in needles)

    def _notify_fluidsynth_missing(self, exc: Exception | None = None) -> None:
        if self._fluidsynth_missing_warned:
            return
        if exc is not None and not self._is_fluidsynth_missing_error(exc):
            return
        self._fluidsynth_missing_warned = True
        msg = (
            "FluidSynth is not installed on this system.\n\n"
            "System playback is unavailable. Install it with:\n"
            "sudo apt-get install fluidsynth libfluidsynth3\n\n"
            "You can still use External MIDI playback from the Playback menu."
        )
        QtWidgets.QMessageBox.warning(self, "FluidSynth not installed", msg)
        self._status("FluidSynth missing: install fluidsynth/libfluidsynth3 or use External MIDI", 10000)

    def _show_error_dialog(self, title: str, text: str, details: str = "", informative_text: str = "") -> None:
        show_error_dialog(self, title, text, details=details, informative_text=informative_text)

    def _format_exception_details(self, exc: Exception) -> str:
        try:
            return ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        except Exception:
            return str(exc or '')

    def _rebuild_midi_port_menu(self) -> None:
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

    def _send_playback_panic(self) -> None:
        if hasattr(self, 'player') and self.player is not None:
            if hasattr(self.player, 'panic'):
                self.player.panic()
            else:
                self.player.stop()

    def _dispose_player(self) -> None:
        if hasattr(self, 'player') and self.player is not None:
            if hasattr(self.player, 'shutdown'):
                self.player.shutdown()
            else:
                self.player.stop()
        self.player = None
        self._player_config = None
        if hasattr(self, 'editor_controller') and self.editor_controller is not None:
            self.editor_controller.set_player(None)

    def _select_external_midi_port(self, port_name: str) -> None:
        self._send_playback_panic()
        self._dispose_player()
        self._set_midi_out_port_to_appdata(str(port_name))
        self._status(f"External MIDI port: {port_name}", 2500)
        self._ensure_player()

    def _ensure_player(self) -> None:
        # Always ensure attribute exists and bubble up failures so callers can report
        if not hasattr(self, 'player'):
            self.player = None
        playback_mode = self._get_playback_mode_from_appdata()
        midi_out_port = self._get_midi_out_port_from_appdata()
        cfg = (str(playback_mode), str(midi_out_port))
        if self.player is None or self._player_config != cfg:
            from midi.player import Player
            self.player = Player(
                soundfont_path=self._get_soundfont_path_from_appdata(),
                playback_mode=playback_mode,
                midi_out_port=(midi_out_port or None),
            )
            self._player_config = cfg
        if hasattr(self, 'editor_controller') and self.editor_controller is not None:
            self.editor_controller.set_player(self.player)

    def _get_soundfont_path_from_appdata(self) -> Optional[str]:
        adm = get_appdata_manager()
        path = str(adm.get("user_soundfont_path", "") or "")
        if path and Path(path).expanduser().is_file():
            return str(Path(path).expanduser())
        return None

    def _set_soundfont_path_to_appdata(self, path: Optional[str]) -> None:
        adm = get_appdata_manager()
        adm.set("user_soundfont_path", str(path or ""))
        adm.save()

    def _unset_soundfont(self) -> None:
        """Clear custom FluidSynth soundfont and revert to default detection."""
        self._set_soundfont_path_to_appdata(None)
        self._dispose_player()
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
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            sel = dlg.selectedFiles()[0]
            if sel:
                self._set_soundfont_path_to_appdata(sel)
                if hasattr(self, 'player') and self.player is not None:
                    self.player.set_soundfont(sel)
                self._status("Custom FluidSynth soundfont selected", 2500)
                return sel
        return existing if existing else None

    def _open_reverb_config_dialog(self) -> None:
        """Open the FluidSynth settings dialog (non-blocking)."""
        dlg = FluidSynthReverbConfigDialog(self)
        dlg.reverb_settings_changed.connect(self._apply_reverb_settings)
        dlg.show()

    def _apply_reverb_settings(self, settings: dict) -> None:
        """Apply FluidSynth settings to the current player."""
        try:
            if hasattr(self, 'player') and self.player is not None:
                from midi.player import _FluidsynthBackend
                if isinstance(self.player._backend, _FluidsynthBackend):
                    backend = self.player._backend
                    backend.set_reverb_enabled(settings.get('enabled', True))
                    backend.set_reverb_room_size(settings.get('room_size', 0.6))
                    backend.set_reverb_damp(settings.get('damp', 0.4))
                    backend.set_reverb_width(settings.get('width', 3.0))
                    backend.set_reverb_level(settings.get('level', 0.9))
                    self.player.set_playhead_sync_delay_ms(settings.get('playhead_sync_delay_ms', 0))
                    self._status("FluidSynth settings applied", 2000)
        except Exception as exc:
            self._status("Failed to apply FluidSynth settings", 2000)

    def _ensure_player_with_soundfont(self) -> None:
        mode = self._get_playback_mode_from_appdata()
        if not (sys.platform.startswith('linux') and mode == 'system'):
            self._ensure_player()
            return
        try:
            self._ensure_player()
            return
        except Exception as exc:
            self._notify_fluidsynth_missing(exc)
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
                    if hasattr(self.player, 'set_persist_settings'):
                        self.player.set_persist_settings(False)
                    return
            raise

    def _choose_midi_port(self) -> None:
        self._rebuild_midi_port_menu()
        self._status("Select external MIDI output from Playback > MIDI port", 2500)

    def _update_clock(self) -> None:
        now = datetime.now()
        timestr = now.strftime("%H:%M:%S")
        if hasattr(self, "_clock_label") and self._clock_label is not None:
            self._clock_label.setText(timestr)
            # Re-position in case width changed
            self._position_clock()

    def _position_clock(self) -> None:
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

    def _export_pdf(self) -> None:
        dlg = QtWidgets.QFileDialog(self)
        dlg.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        dlg.setNameFilter("PDF Files (*.pdf)")
        dlg.setDefaultSuffix("pdf")
        # Prefill filename with score title when available
        info = getattr(self.file_manager.current(), 'info', None)
        score_title = str(getattr(info, 'title', "") or "") if info is not None else ""
        safe_title = "".join(ch for ch in score_title if ch not in r'\\/:*?"<>|').strip()
        suggested_name = f"{safe_title or 'Untitled'}.pdf"
        adm = get_appdata_manager()
        last_dir = str(adm.get("last_export_pdf_dir", "") or "")
        if last_dir:
            dlg.setDirectory(last_dir)
            dlg.selectFile(os.path.join(last_dir, suggested_name))
        else:
            dlg.selectFile(suggested_name)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            out_path = dlg.selectedFiles()[0]
            out_dir = os.path.dirname(str(out_path))
            if out_dir:
                adm = get_appdata_manager()
                adm.set("last_export_pdf_dir", out_dir)
                adm.save()
            try:
                from utils.CONSTANT import ENGRAVER_LAYERING
                from engraver.engraver import do_engrave
                export_du = DrawUtil()
                do_engrave(self._current_score_dict(), export_du, pdf_export=True)
                total_pages = max(1, export_du.page_count())
                progress = QtWidgets.QProgressDialog("Exporting PDF...", None, 0, 100, self)
                progress.setWindowTitle("Export PDF")
                progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(0)
                progress.setValue(0)
                progress.show()

                base_percent = 46
                remaining_percent = 100 - base_percent
                self._prime_export_progress(progress, target_percent=base_percent, duration_ms=400)

                def _on_progress(done: int, total: int) -> None:
                    safe_total = max(1, int(total))
                    ratio = max(0.0, min(1.0, float(done) / float(safe_total)))
                    value = int(round(base_percent + (remaining_percent * ratio)))
                    progress.setValue(value)
                    QtWidgets.QApplication.processEvents()

                export_du.save_pdf(out_path, layering=ENGRAVER_LAYERING, progress_cb=_on_progress)
                progress.setValue(100)
                progress.close()
            except Exception as e:
                self._show_error_dialog(
                    "Export PDF failed",
                    str(e),
                    details=self._format_exception_details(e),
                )

    def _export_image_pdf(self) -> None:
        dpi_options = [300, 600, 1200]
        dpi_labels = [str(v) for v in dpi_options]
        dpi_dlg = QtWidgets.QInputDialog(self)
        dpi_dlg.setWindowTitle("Export Image PDF")
        dpi_dlg.setLabelText("Raster DPI:")
        dpi_dlg.setComboBoxItems(dpi_labels)
        dpi_dlg.setComboBoxEditable(False)
        dpi_dlg.setTextValue(dpi_labels[1])
        dpi_dlg.setMinimumWidth(420)
        dpi_dlg.resize(420, dpi_dlg.sizeHint().height())
        ok = dpi_dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted
        dpi_label = dpi_dlg.textValue()
        if not ok:
            return
        selected_dpi = int(str(dpi_label))
        if selected_dpi not in dpi_options:
            selected_dpi = 600

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
                progress = QtWidgets.QProgressDialog(f"Exporting rasterized PDF ({selected_dpi} DPI)...", None, 0, 100, self)
                progress.setWindowTitle("Export Image PDF")
                progress.setWindowModality(QtCore.Qt.WindowModality.WindowModal)
                progress.setMinimumDuration(0)
                progress.setValue(0)
                progress.show()

                base_percent = 46
                remaining_percent = 100 - base_percent
                self._prime_export_progress(progress, target_percent=base_percent, duration_ms=400)

                def _on_progress(done: int, total: int) -> None:
                    safe_total = max(1, int(total))
                    ratio = max(0.0, min(1.0, float(done) / float(safe_total)))
                    value = int(round(base_percent + (remaining_percent * ratio)))
                    progress.setValue(value)
                    QtWidgets.QApplication.processEvents()

                export_du.save_pdf_rasterized(out_path, dpi=selected_dpi, layering=ENGRAVER_LAYERING, progress_cb=_on_progress)
                progress.setValue(100)
                progress.close()
            except Exception as e:
                self._show_error_dialog(
                    "Export Image PDF failed",
                    str(e),
                    details=self._format_exception_details(e),
                )

    def _prime_export_progress(self, progress: QtWidgets.QProgressDialog, target_percent: int = 46, duration_ms: int = 400) -> None:
        """Animate the progress bar to a target percentage over a fixed duration."""
        target = max(0, min(100, int(target_percent)))
        total_ms = max(1, int(duration_ms))
        timer = QtCore.QElapsedTimer()
        timer.start()
        while True:
            elapsed = timer.elapsed()
            ratio = min(1.0, float(elapsed) / float(total_ms))
            value = int(round(target * ratio))
            progress.setValue(value)
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 15)
            if ratio >= 1.0:
                break
            QtCore.QThread.msleep(15)

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
        if p is None:
            state = self.tr("Unsaved changes") if dirty else self.tr("New project")
        else:
            state = self.tr("Unsaved changes") if dirty else self.tr("Saved")
        path_text = str(p) if p else (self.tr("(session.piano restored)") if session_mode else self.tr("(unsaved project)"))
        prefix = self.tr("Session mode") + " • " if session_mode else ""
        return f"{prefix}{state} • {path_text}"

    def _current_file_label_for_status(self) -> str:
        try:
            p = self.file_manager.path()
            if p is not None:
                return str(p)
        except Exception:
            pass
        if bool(getattr(self, '_session_restore_mode', False)):
            return self.tr("session.piano")
        return self.tr("(unsaved project)")

    def _show_file_action_status(self, action: str, timeout_ms: int = 2500) -> None:
        label = self._current_file_label_for_status()
        self._status(f"{action} • {label}", timeout_ms)

    def _show_status_default(self, force: bool = False) -> None:
        try:
            sb = self.statusBar() if hasattr(self, 'statusBar') else None
            if sb is not None:
                new_default = self._status_default_message()
                current = str(sb.currentMessage() or "")
                previous_default = str(getattr(self, '_status_default_text', "") or "")
                if bool(force) or not current or current == previous_default:
                    sb.showMessage(new_default, 0)
                    self._status_default_text = new_default
        except Exception:
            pass

    def _on_status_message_changed(self, msg: str) -> None:
        if not msg:
            self._show_status_default()

    def _open_preferences(self) -> None:
        # Ensure preferences file exists and open in system editor
        open_preferences(self)

    def _normalized_ui_language(self, language: str) -> str:
        lang = str(language or "system").strip().lower()
        if lang not in ("system", "en", "nl"):
            return "system"
        return lang

    def _current_ui_language_preference(self) -> str:
        pm = get_preferences_manager()
        return self._normalized_ui_language(str(pm.get("ui_language", "system") or "system"))

    def _sync_ui_language_actions(self) -> None:
        selected = self._current_ui_language_preference()
        mapping = {
            "system": getattr(self, "_language_system_action", None),
            "en": getattr(self, "_language_en_action", None),
            "nl": getattr(self, "_language_nl_action", None),
        }
        for key, action in mapping.items():
            if action is not None:
                action.setChecked(key == selected)

    def _language_display_name(self, lang: str) -> str:
        normalized = self._normalized_ui_language(lang)
        if normalized == "en":
            return self.tr("English")
        if normalized == "nl":
            return self.tr("Dutch")
        return self.tr("System")

    def _set_ui_language_preference(self, language: str) -> None:
        lang = self._normalized_ui_language(language)
        current = self._current_ui_language_preference()
        self._sync_ui_language_actions()
        if lang == current:
            return

        pm = get_preferences_manager()
        pm.set("ui_language", lang)
        pm.save()
        self._sync_ui_language_actions()

        result = QtWidgets.QMessageBox.question(
            self,
            self.tr("Restart keyTAB"),
            self.tr("Language changed to {language}. Restart now to apply it?").format(
                language=self._language_display_name(lang)
            ),
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.Yes,
        )
        if result == QtWidgets.QMessageBox.StandardButton.Yes:
            self._status(self.tr("Restarting keyTAB to apply language change..."), 2000)
            QtCore.QTimer.singleShot(150, self._request_app_restart)
        else:
            self._status(self.tr("Language preference saved. Restart keyTAB to apply it."), 4000)

    def _open_about_dialog(self) -> None:
        """Show licensing and attribution info."""
        try:
            dlg = AboutDialog(self)
            dlg.show()
        except Exception:
            pass

    def _file_new(self) -> None:
        # If there are unsaved changes, confirm save before starting a new project
        if not self.file_manager.confirm_save_for_action("creating a new project"):
            return
        self.file_manager.new()
        self._session_restore_mode = False
        try:
            self.print_view.reset_view_state()
        except Exception:
            pass
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
        self._show_status_default(force=True)
        self._show_file_action_status("New project")

    def _file_open(self) -> None:
        # If there are unsaved changes, confirm save before opening another project
        if not self.file_manager.confirm_save_for_action("opening another project", force_prompt=True):
            return
        try:
            sc = self.file_manager.load()
        except Exception as e:
            self._show_error_dialog(
                self.tr("MIDI Import failed"),
                str(e),
                details=self._format_exception_details(e),
                informative_text=self.tr("Use 'Copy Error Log' and keep the copied traceback for debugging."),
            )
            return
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
            try:
                sc = self.file_manager.open_path(candidate)
            except Exception as e:
                self._show_error_dialog(
                    self.tr("MIDI Import failed"),
                    str(e),
                    details=self._format_exception_details(e),
                    informative_text=self.tr("Use 'Copy Error Log' and keep the copied traceback for debugging."),
                )
                continue
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
            self.print_view.reset_view_state()
        except Exception:
            pass
        try:
            self._refresh_recent_files_menu()
        except Exception:
            pass
        try:
            self.editor_controller.set_score(self.file_manager.current())
            self.editor_controller.reset_undo_stack()
        except Exception:
            pass
        try:
            self._restore_app_state_from_score()
        except Exception:
            pass
        self._refresh_views_from_score()
        try:
            if hasattr(self.editor_controller, 'force_redraw_from_model'):
                self.editor_controller.force_redraw_from_model()
        except Exception:
            pass
        self._update_title()
        self._show_status_default(force=True)
        self._show_file_action_status(self.tr("Opened"))

    def _file_save(self) -> None:
        if self.file_manager.save():
            if self.file_manager.path() is not None:
                self._session_restore_mode = False
            self._update_title()
            self._show_status_default(force=True)
            self._show_file_action_status(self.tr("Saved"))

    def _file_save_as(self) -> None:
        if self.file_manager.save_as():
            if self.file_manager.path() is not None:
                self._session_restore_mode = False
            self._update_title()
            self._show_status_default(force=True)
            self._show_file_action_status(self.tr("Saved As"))

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

    # ------------------------------------------------------------------
    # MIDI import – track-to-hand assignment dialog
    # ------------------------------------------------------------------

    def _handle_midi_import(self, path: str):
        """MIDI import hook called by FileManager when opening a .mid/.midi file.

        Shows the track-assignment dialog with live preview.  Returns the
        imported SCORE on accept or None if the user cancels.
        """
        from midi.midi_loader import midi_analyze_tracks, midi_load
        from ui.dialogs.midi_import_dialog import MidiImportDialog
        from ui.preview_service import PreviewSession

        # Analyze tracks; fall back to simple load when there is nothing to assign.
        try:
            track_infos = midi_analyze_tracks(path)
        except Exception:
            track_infos = []

        if not track_infos:
            # No assignable tracks – just load without dialog.
            try:
                sc = midi_load(path)
                if hasattr(sc, '_normalize_events_after_load'):
                    sc._normalize_events_after_load()
                return sc
            except Exception as exc:
                raise RuntimeError(f"Failed to load MIDI: {exc}") from exc

        # Snapshot the current score so we can restore it on cancel.
        preview = PreviewSession(self.file_manager, self.editor_controller, parent=self, debounce_ms=80)
        latest_assignments: dict = {}

        def _reload_with_assignments(assignments: dict) -> None:
            """Reload MIDI with the given hand assignments and refresh the view."""
            sc = midi_load(path, track_assignments=assignments)
            if hasattr(sc, '_normalize_events_after_load'):
                sc._normalize_events_after_load()
            self.file_manager.replace_current(sc)

        def _schedule_live_preview(assignments: dict) -> None:
            nonlocal latest_assignments
            latest_assignments = dict(assignments or {})
            preview.schedule_preview(
                mutator=lambda: _reload_with_assignments(latest_assignments),
                restore_first=False,
            )

        # Initial load with default assignments so the editor shows a preview right away.
        initial_assignments = {ti['index']: ti['default_hand'] for ti in track_infos}
        _reload_with_assignments(initial_assignments)
        preview.refresh()

        # Build and show the dialog.
        dlg = MidiImportDialog(track_infos=track_infos, parent=self)
        dlg.assignments_changed.connect(_schedule_live_preview)

        result = dlg.exec()

        if result == QtWidgets.QDialog.DialogCode.Accepted:
            # Flush any queued preview update so the final combo state is applied
            # before committing and returning from the hook.
            try:
                preview._timer.stop()
            except Exception:
                pass
            try:
                _reload_with_assignments(dlg.get_assignments())
                preview.refresh()
            except Exception:
                pass

            # Apply the final selection and hand back the finished SCORE.
            preview.commit(
                mutator=lambda: _reload_with_assignments(dlg.get_assignments()),
                restore_first=False,
            )
            # Extra safety: fully rebind and refresh views so the final imported
            # state is always visible immediately after pressing OK.
            try:
                self.editor_controller.set_score(self.file_manager.current())
            except Exception:
                pass
            try:
                self._refresh_views_from_score()
            except Exception:
                pass
            try:
                if hasattr(self.editor_controller, 'force_redraw_from_model'):
                    self.editor_controller.force_redraw_from_model()
                else:
                    self.editor_controller.draw_frame()
            except Exception:
                pass
            return self.file_manager.current()
        else:
            # User cancelled – restore the original score.
            preview.restore_original()
            return None

    def _open_style_dialog(self) -> None:
        from ui.dialogs.style_dialog import StyleDialog
        from ui.preview_service import PreviewSession

        sc = self.file_manager.current()
        layout = getattr(sc, 'layout', None)
        dlg = StyleDialog(parent=self, layout=layout, score=sc)

        adm = get_appdata_manager()
        dlg_w = int(adm.get('style_dialog_width', 600) or 600)
        dlg_h = int(adm.get('style_dialog_height', 550) or 550)
        dlg.resize(max(280, dlg_w), max(300, dlg_h))
        dlg_x = int(adm.get('style_dialog_x', -1) or -1)
        dlg_y = int(adm.get('style_dialog_y', -1) or -1)
        if dlg_x >= 0 and dlg_y >= 0:
            # Avoid restoring a stale off-screen position after monitor/layout changes.
            target = QtCore.QPoint(dlg_x, dlg_y)
            on_screen = False
            for screen in QtGui.QGuiApplication.screens():
                geo = screen.availableGeometry()
                if geo.contains(target):
                    on_screen = True
                    break
            if on_screen:
                dlg.move(target)

        app_state = self._current_app_state()
        dlg.set_current_tab(int(getattr(app_state, 'style_dialog_tab_index', 0) or 0))

        preview = PreviewSession(self.file_manager, self.editor_controller, parent=dlg, debounce_ms=150)

        def _apply_dialog_values() -> None:
            cur = self.file_manager.current()
            cur.layout = dlg.get_values()

        dlg.values_changed.connect(lambda: preview.schedule_preview(_apply_dialog_values, restore_first=True))
        dlg.accepted.connect(lambda: preview.commit(label='style_edit', mutator=_apply_dialog_values, restore_first=True))
        dlg.rejected.connect(preview.restore_original)

        def _persist_tab_index() -> None:
            # Always resolve app_state from the current SCORE instance.
            # PreviewSession can replace SCORE objects while the dialog is open.
            cur_app_state = self._current_app_state()
            cur_app_state.style_dialog_tab_index = int(dlg.current_tab_index())
            adm = get_appdata_manager()
            adm.set('style_dialog_width', int(dlg.width()))
            adm.set('style_dialog_height', int(dlg.height()))
            pos = dlg.pos()
            adm.set('style_dialog_x', int(pos.x()))
            adm.set('style_dialog_y', int(pos.y()))
            adm.save()

        dlg.finished.connect(lambda _res: _persist_tab_index())
        dlg.accepted.connect(lambda: self.file_manager.save() if self.file_manager.path() is not None else None)
        dlg.show()

    def _open_info_dialog(self) -> None:
        from ui.dialogs.info_dialog import InfoDialog
        sc = self.file_manager.current()
        dlg = InfoDialog(sc, self)
        # Connect accepted signal to apply changes
        dlg.accepted.connect(lambda: (dlg.apply_to_score(), self.file_manager.on_model_changed(), self._refresh_views_from_score()))
        dlg.show()

    def _open_line_break_dialog(self) -> None:
        from ui.dialogs.line_break_dialog import LineBreakDialog
        from ui.preview_service import PreviewSession

        score = self.file_manager.current()
        if score is None:
            return

        preview = PreviewSession(self.file_manager, self.editor_controller, parent=self, debounce_ms=150)

        dlg = LineBreakDialog(
            parent=self,
            score=score,
            selected_line_break=None,
            measure_resolver=(lambda t: self.editor_controller.get_measure_index_for_time(t)) if hasattr(self.editor_controller, 'get_measure_index_for_time') else None,
            on_change=preview.schedule_refresh,
        )

        adm = get_appdata_manager()
        dlg_w = int(adm.get('line_break_dialog_width', 900) or 900)
        dlg_h = int(adm.get('line_break_dialog_height', 700) or 700)
        dlg.resize(max(420, dlg_w), max(320, dlg_h))
        dlg_x = int(adm.get('line_break_dialog_x', -1) or -1)
        dlg_y = int(adm.get('line_break_dialog_y', -1) or -1)
        if dlg_x >= 0 and dlg_y >= 0:
            # Avoid restoring stale off-screen positions after monitor/layout changes.
            target = QtCore.QPoint(dlg_x, dlg_y)
            on_screen = False
            for screen in QtGui.QGuiApplication.screens():
                geo = screen.availableGeometry()
                if geo.contains(target):
                    on_screen = True
                    break
            if on_screen:
                dlg.move(target)

        def _on_accept() -> None:
            preview.commit(label='line_break_edit', restore_first=False)

        def _on_reject() -> None:
            preview.restore_original()

        def _on_finished(_result: int) -> None:
            adm = get_appdata_manager()
            adm.set('line_break_dialog_width', int(dlg.width()))
            adm.set('line_break_dialog_height', int(dlg.height()))
            pos = dlg.pos()
            adm.set('line_break_dialog_x', int(pos.x()))
            adm.set('line_break_dialog_y', int(pos.y()))
            adm.save()
            if int(_result) == int(QtWidgets.QDialog.DialogCode.Accepted):
                self.file_manager.on_model_changed()

        dlg.accepted.connect(_on_accept)
        dlg.rejected.connect(_on_reject)
        dlg.finished.connect(_on_finished)
        dlg.show()

    def _run_script_dialog(self) -> None:
        engine = getattr(self, "script_engine", None)
        if engine is None:
            try:
                self.script_engine = ScriptEngine(self.file_manager, self.editor_controller, parent=self)
                engine = self.script_engine
            except Exception as exc:
                self._show_error_dialog(self.tr("Run Script"), self.tr("Failed to initialize scripting: {error}").format(error=exc), details=self._format_exception_details(exc))
                return
        try:
            engine.choose_and_run()
        except Exception as exc:
            self._show_error_dialog(self.tr("Run Script"), self.tr("Script failed: {error}").format(error=exc), details=self._format_exception_details(exc))

    def _scripts_dir(self) -> Path:
        return Path(__file__).resolve().parent.parent / "scripts"

    def _run_script_path(self, script_path: str) -> None:
        engine = getattr(self, "script_engine", None)
        if engine is None:
            try:
                self.script_engine = ScriptEngine(self.file_manager, self.editor_controller, parent=self)
                engine = self.script_engine
            except Exception as exc:
                self._show_error_dialog(self.tr("Run Script"), self.tr("Failed to initialize scripting: {error}").format(error=exc), details=self._format_exception_details(exc))
                return
        try:
            engine.run_script(Path(script_path))
        except Exception as exc:
            self._show_error_dialog(self.tr("Run Script"), self.tr("Script failed: {error}").format(error=exc), details=self._format_exception_details(exc))

    def _rebuild_tools_menu(self) -> None:
        menu = getattr(self, "_tools_menu", None)
        if menu is None:
            return
        menu.clear()

        run_script_act = QtGui.QAction(self.tr("Run Script..."), self)
        run_script_act.setToolTip(self.tr("Load and run a Python script with preview and cancel support."))
        run_script_act.triggered.connect(self._run_script_dialog)
        menu.addAction(run_script_act)
        menu.addSeparator()

        scripts_dir = self._scripts_dir()
        if not scripts_dir.exists() or not scripts_dir.is_dir():
            empty_act = QtGui.QAction(self.tr("No scripts folder found"), self)
            empty_act.setEnabled(False)
            menu.addAction(empty_act)
            return

        script_files = sorted(
            [p for p in scripts_dir.glob("*.py") if p.is_file()],
            key=lambda p: p.name.lower(),
        )
        if not script_files:
            empty_act = QtGui.QAction(self.tr("No scripts found"), self)
            empty_act.setEnabled(False)
            menu.addAction(empty_act)
            return

        for script_file in script_files:
            act = QtGui.QAction(script_file.stem, self)
            act.setToolTip(str(script_file))
            act.triggered.connect(lambda _checked=False, p=str(script_file): self._run_script_path(p))
            menu.addAction(act)

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
            adm = None
        if not isinstance(recent, list):
            recent = []
        original_recent = [str(p) for p in recent if str(p).strip()]

        # Normalize, remove duplicates, and prune non-existing paths before building the menu.
        seen: set[str] = set()
        cleaned_recent: list[str] = []
        for raw_path in original_recent:
            normalized = str(Path(raw_path).expanduser())
            if normalized in seen:
                continue
            seen.add(normalized)
            if not Path(normalized).is_file():
                continue
            cleaned_recent.append(normalized)

        recent = cleaned_recent
        if recent != original_recent:
            if adm is not None:
                adm.set("recent_files", recent)
                adm.save()

        if not recent:
            empty_act = QtGui.QAction(self.tr("No recent files"), self)
            empty_act.setEnabled(False)
            menu.addAction(empty_act)
        else:
            for path in recent[:100]:
                act = QtGui.QAction(path, self)
                act.triggered.connect(lambda _c=False, p=path: self._open_recent_file(p))
                menu.addAction(act)

        menu.addSeparator()
        clear_act = QtGui.QAction(self.tr("Clear Recent Files"), self)
        clear_act.triggered.connect(self._clear_recent_files)
        menu.addAction(clear_act)
        self._refresh_rename_file_action()

    def _refresh_rename_file_action(self) -> None:
        act = getattr(self, '_rename_file_act', None)
        if act is None:
            return
        try:
            current_path = self.file_manager.path()
            visible = current_path is not None and Path(current_path).is_file()
        except Exception:
            visible = False
        act.setVisible(bool(visible))
        act.setEnabled(bool(visible))

    def _rename_current_file(self) -> None:
        try:
            cur_path = self.file_manager.path()
        except Exception:
            cur_path = None
        if cur_path is None or not Path(cur_path).is_file():
            return

        current = Path(cur_path)
        new_name, ok = QtWidgets.QInputDialog.getText(
            self,
            self.tr("Rename File"),
            self.tr("New file name:"),
            QtWidgets.QLineEdit.EchoMode.Normal,
            current.name,
        )
        if not ok:
            return

        new_name = str(new_name or "").strip()
        if not new_name or new_name == current.name:
            return
        if ("/" in new_name) or ("\\" in new_name):
            QtWidgets.QMessageBox.warning(self, self.tr("Rename File"), self.tr("Please enter only a file name, not a path."))
            return

        target = current.with_name(new_name)
        if target.suffix == "" and current.suffix:
            target = target.with_suffix(current.suffix)

        if self.file_manager.rename_current_file(target):
            self._session_restore_mode = False
            self._refresh_recent_files_menu()
            self._refresh_rename_file_action()
            self._update_title()
            self._show_status_default(force=True)
            self._show_file_action_status(self.tr("Renamed"))

    def _set_default_style(self) -> None:
        """Save the current layout as the default style for new projects."""
        try:
            from ui.dialogs.style_dialog import save_default_style
            current_layout = self.file_manager.current().layout
            save_default_style(current_layout)
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Default Style Saved"),
                self.tr("The current style has been set as the default for new projects.")
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Failed to Save Default Style"),
                self.tr("An error occurred while saving the default style: {error}").format(error=str(e))
            )

    def _reset_default_style(self) -> None:
        """Reset to the built-in default style."""
        try:
            from ui.dialogs.style_dialog import reset_default_style
            reset_default_style()
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Default Style Reset"),
                self.tr("The default style has been reset to the built-in defaults.")
            )
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Failed to Reset Default Style"),
                self.tr("An error occurred while resetting the default style: {error}").format(error=str(e))
            )

    def _open_recent_file(self, path: str) -> None:
        candidate = Path(str(path or "")).expanduser()
        if not candidate.is_file():
            adm = get_appdata_manager()
            recent = adm.get("recent_files", []) or []
            if not isinstance(recent, list):
                recent = []
            candidate_s = str(candidate)
            filtered = [str(p) for p in recent if str(p).strip() and str(p) != candidate_s]
            adm.set("recent_files", filtered)
            adm.save()

            QtWidgets.QMessageBox.information(
                self,
                self.tr("Recent File Missing"),
                self.tr("This file no longer exists and was removed from Recent Files:\n{path}").format(
                    path=str(candidate),
                ),
            )
            self._refresh_recent_files_menu()
            return

        if not self.file_manager.confirm_save_for_action("opening another project", force_prompt=True):
            return
        sc = self.file_manager.open_path(str(candidate))
        if sc:
            self._refresh_recent_files_menu()
            self._after_project_loaded()

    def _clear_recent_files(self) -> None:
        adm = get_appdata_manager()
        adm.set("recent_files", [])
        adm.save()
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
        if pending > 0:
            # During startup, metrics can report max_scroll=0 before the first
            # full layout/engrave is available. Keep pending restore until the
            # range is usable, otherwise non-zero startup positions are lost.
            if max_scroll <= 0:
                return
            target = max(0, min(pending, max_scroll))
            if int(self.editor_vscroll.value()) != target:
                self.editor_vscroll.setValue(target)
            self._pending_scroll_restore = 0
        elif pending == 0:
            self._pending_scroll_restore = 0

    @QtCore.Slot(int)
    def _on_editor_scroll_changed(self, value: int) -> None:
        value = int(value)
        self.editor_canvas.set_scroll_logical_px(value)

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

    def _zoom_editor(self, steps: int) -> None:
        try:
            if hasattr(self, 'editor_canvas') and hasattr(self.editor_canvas, 'apply_zoom_steps'):
                self.editor_canvas.apply_zoom_steps(int(steps))
        except Exception:
            pass

    def _edit_undo(self) -> None:
        self.editor_controller.undo()
        self._refresh_views_from_score()
        self.editor_controller.set_score(self.file_manager.current())
        self.editor_controller.force_redraw_from_model()

    def _edit_redo(self) -> None:
        self.editor_controller.redo()
        self._refresh_views_from_score()
        self.editor_controller.set_score(self.file_manager.current())
        self.editor_controller.force_redraw_from_model()

    def _edit_copy(self) -> None:
        try:
            self.editor_controller.copy_selection()
            self._status(self.tr("Copied selection"), 1200)
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
            self._status(self.tr("Cut selection"), 1200)
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
            self._status(self.tr("Pasted selection"), 1200)
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
                self._status(self.tr("Deleted selection"), 1200)
            else:
                self._status(self.tr("No selection to delete"), 1200)
        except Exception:
            pass

    def _selection_select_all(self) -> None:
        try:
            self.editor_controller.select_all()
            try:
                self.editor_canvas.update()
            except Exception:
                pass
            self._status(self.tr("Selected all"), 1200)
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
                self._status(self.tr("Transposed selection {semitones:+d} semitone").format(semitones=int(semitones)), 1200)
            else:
                self._status(self.tr("No selection to transpose"), 1200)
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
                direction = self.tr("earlier") if delta < 0 else self.tr("later")
                self._status(self.tr("Moved selection {direction} by snap").format(direction=direction), 1200)
            else:
                self._status(self.tr("No selection to move"), 1200)
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
                    self._status(self.tr("Quantized selection starts to snap"), 1200)
                elif mode == 'end':
                    self._status(self.tr("Quantized selection ends to snap"), 1200)
                else:
                    self._status(self.tr("Quantized selection starts and ends to snap"), 1200)
            else:
                self._status(self.tr("No selection to quantize"), 1200)
        except Exception:
            pass

    def _update_title(self) -> None:
        self.setWindowTitle("keyTAB")
        self._show_status_default()

    def _page_dimensions_mm(self) -> tuple[float, float]:
        sc = self.file_manager.current()
        lay = getattr(sc, 'layout', None)
        if not lay:
            return self.du.current_page_size_mm()
        
        w_mm = float(getattr(lay, 'page_width_mm', 210.0) or 210.0)
        h_mm = float(getattr(lay, 'page_height_mm', 297.0) or 297.0)
        page_orientation = str(getattr(lay, 'page_orientation', 'portrait') or 'portrait').strip().lower()
        # Keep compatibility with legacy horizontal/vertical orientation values.
        if page_orientation == 'vertical':
            page_orientation = 'portrait'
        elif page_orientation == 'horizontal':
            page_orientation = 'landscape'
        if page_orientation == 'landscape':
            w_mm, h_mm = h_mm, w_mm
        # The engraver always rotates -90° for horizontal read direction,
        # which swaps the output dimensions reported by DrawUtil.
        read_direction = str(getattr(lay, 'read_direction', 'vertical') or 'vertical').strip().lower()
        if read_direction == 'horizontal':
            w_mm, h_mm = h_mm, w_mm
        return w_mm, h_mm

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
            
            # switch width and height if layout.read_direction is horizontal-
            # to correct for the engraver's -90° rotation after drawing
            score = self.file_manager.current()
            if score.layout.read_direction == 'horizontal':
                w_mm, h_mm = h_mm, w_mm

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

    def _on_splitter_moved(self, _pos: int, _index: int) -> None:
        self.print_view.reset_view_state()

    def _current_score_dict(self) -> dict:
        try:
            return self.file_manager.current().get_dict()
        except Exception:
            return {}

    def _on_engraver_finished(self) -> None:
        self._last_engraver_error_signature = None
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

    @QtCore.Slot(str, str)
    def _on_engraver_failed(self, error_text: str, error_details: str) -> None:
        signature = f"{error_text}\n{error_details}".strip()
        if signature and signature == self._last_engraver_error_signature:
            return
        self._last_engraver_error_signature = signature or None
        self._show_error_dialog(
            self.tr("Engraving failed"),
            str(error_text or self.tr("The engraver failed.")),
            details=str(error_details or ""),
            informative_text=self.tr("Use 'Copy Error Log' and keep the copied traceback for debugging."),
        )
        try:
            self._status(self.tr("Engraving failed. See error dialog for details."), 10000)
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
        def _start_playback() -> None:
            self._ensure_player_with_soundfont()
            sc = self.file_manager.current()
            if start_units is None:
                self._scroll_editor_to_start()
                self.player.play_score(sc)
            else:
                self.player.play_from_time_cursor(float(start_units or 0.0), sc)
            self._start_playhead_timer()
            self._show_play_debug_status()

        def _system_method_name() -> str:
            return str(self._playback_system_label()).replace("Playback using ", "").strip() or "system playback"

        try:
            _start_playback()
        except Exception as exc:
            # If external port playback fails, switch to system playback automatically.
            if self._get_playback_mode_from_appdata() == 'external':
                switched_to = _system_method_name()
                try:
                    self._set_playback_mode('system', show_status=False)
                    _start_playback()
                    try:
                        QtWidgets.QMessageBox.warning(
                            self,
                            self.tr("Playback"),
                            (
                                self.tr("External MIDI playback failed: {error}\n\nSwitched automatically to {backend}.").format(
                                    error=exc,
                                    backend=switched_to,
                                )
                            ),
                        )
                    except Exception:
                        print(
                            "External MIDI playback failed: "
                            f"{exc}. Switched automatically to {switched_to}."
                        )
                    return
                except Exception as fallback_exc:
                    try:
                        QtWidgets.QMessageBox.critical(
                            self,
                            self.tr("Playback"),
                            (
                                self.tr("External MIDI playback failed: {error}\n\nAutomatic fallback to {backend} also failed: {fallback_error}").format(
                                    error=exc,
                                    backend=switched_to,
                                    fallback_error=fallback_exc,
                                )
                            ),
                        )
                    except Exception:
                        print(
                            "External MIDI playback failed: "
                            f"{exc}. Automatic fallback to {switched_to} also failed: {fallback_exc}"
                        )
                    return
            try:
                QtWidgets.QMessageBox.critical(
                    self,
                    self.tr("Playback"),
                    self.tr("Playback failed: {error}\n\nTry '{backend}' from the Playback menu.").format(
                        error=exc,
                        backend=_system_method_name(),
                    ),
                )
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
            # --- Print-view playhead overlay ---
            self._update_print_view_playhead(units)
        else:
            # Not playing: clear and stop timer
            self._clear_playhead_overlay()

    def _update_print_view_playhead(self, units: Optional[float]) -> None:
        """Update the playhead line on the print view and auto-turn pages."""
        try:
            from midi.printview_playhead import time_to_print_position, build_print_time_map
            time_map = build_print_time_map(self.du)
            if not time_map or units is None:
                return
            result = time_to_print_position(float(units), time_map)
            if result is None:
                self.print_view.clear_playhead_overlay()
                return
            target_page, y_mm, x1_mm, x2_mm = result
            # Auto page turn: re-engrave the target page if it differs from the current one
            current_page = int(getattr(self, '_page_counter', 0) or 0)
            if target_page != current_page:
                self._page_counter = int(target_page)
                try:
                    self.du.set_current_page(int(target_page))
                except Exception:
                    pass
                try:
                    self.print_view.set_page(int(target_page), request_render=False)
                except Exception:
                    pass
                # Re-engrave the new page so it is fully rendered
                try:
                    self.engraver.engrave(
                        self._current_score_dict(),
                        pageno=int(target_page),
                    )
                except Exception:
                    pass
            self.print_view.set_playhead_overlay(y_mm, x1_mm, x2_mm)
        except Exception:
            pass

    def _center_playhead_scroll(self, units: Optional[float]) -> None:
        """Scroll so the current playhead measure sits at top+margin and advance only after full measures pass.

        - We anchor the viewport to a barline (measure start) and place it at `top + margin`.
        - While the playhead stays within the fully visible measures, we do not scroll.
        - Once the playhead enters a measure beyond the last fully visible one, we jump so that
          the playhead's measure becomes the new anchor at the top.
        """
        if units is None:
            return
        try:
            ed = getattr(self, 'editor_controller', None)
            if ed is None:
                return

            # Gather basics
            vp_h_mm = float(getattr(ed, '_viewport_h_mm', 0.0) or 0.0)
            px_per_mm = float(getattr(ed, '_px_per_mm', 0.0) or 0.0)
            dpr = float(getattr(ed, '_dpr', 1.0) or 1.0)
            margin_mm = float(getattr(ed, 'margin', 0.0) or 0.0)
            if vp_h_mm <= 0.0 or px_per_mm <= 0.0:
                return

            # Barline positions (include terminal end) to compute measure spans
            bars = self._barlines_with_terminal(ed)
            if not bars:
                return

            # Current playhead measure (1-based)
            measure_idx = max(1, min(len(bars) - 1, int(ed.get_measure_index_for_time(float(units)))))

            # Establish anchor if missing or if we've scrolled beyond visible block
            anchor = self._playhead_anchor_measure or measure_idx
            last_visible = self._playhead_last_visible_measure or anchor

            # Repeat jumps can move playhead backward; reset anchor to follow.
            if measure_idx < anchor:
                anchor = measure_idx

            if measure_idx > last_visible:
                anchor = measure_idx

            # Compute target scroll from anchor
            target_scroll_px, visible_last = self._scroll_plan_for_anchor(ed, bars, anchor, vp_h_mm, px_per_mm, dpr, margin_mm)
            if target_scroll_px is None:
                return

            # Apply scroll only when anchor updates or value differs
            if anchor != self._playhead_anchor_measure or int(self.editor_vscroll.value()) != target_scroll_px:
                max_scroll = int(self.editor_vscroll.maximum()) if hasattr(self, 'editor_vscroll') else target_scroll_px
                target_scroll_px = max(0, min(int(target_scroll_px), int(max_scroll)))
                if hasattr(self, 'editor_vscroll') and self.editor_vscroll is not None:
                    self.editor_vscroll.setValue(int(target_scroll_px))
                elif hasattr(self, 'editor_canvas') and self.editor_canvas is not None:
                    self.editor_canvas.set_scroll_logical_px(int(target_scroll_px))

            # Persist anchor state
            self._playhead_anchor_measure = anchor
            self._playhead_last_visible_measure = visible_last
        except Exception:
            pass

    def _barlines_with_terminal(self, ed) -> list[float]:
        """Return barline starts plus final end position (ticks)."""
        try:
            score = ed.current_score()
            bars: list[float] = []
            cur = 0.0
            for bg in getattr(score, 'base_grid', []) or []:
                numer = float(getattr(bg, 'numerator', 4) or 4)
                denom = float(getattr(bg, 'denominator', 4) or 4)
                measure_len = numer * (4.0 / max(1.0, denom)) * float(QUARTER_NOTE_UNIT)
                for _ in range(int(getattr(bg, 'measure_amount', 1) or 1)):
                    bars.append(cur)
                    cur += measure_len
            bars.append(cur)
            return bars
        except Exception:
            return []

    def _scroll_plan_for_anchor(self, ed, bars: list[float], anchor_measure: int, vp_h_mm: float, px_per_mm: float, dpr: float, margin_mm: float) -> tuple[int | None, int | None]:
        if not bars or anchor_measure < 1 or anchor_measure >= len(bars):
            return None, None

        start_tick = bars[anchor_measure - 1]
        start_mm = float(ed.time_to_mm(start_tick))
        top_mm = max(0.0, start_mm - margin_mm)
        bottom_mm = top_mm + vp_h_mm

        # Determine last fully visible measure
        last_visible = anchor_measure
        for idx in range(anchor_measure - 1, len(bars) - 1):
            m_start_mm = float(ed.time_to_mm(bars[idx]))
            m_end_mm = float(ed.time_to_mm(bars[idx + 1]))
            if m_end_mm <= bottom_mm:
                last_visible = idx + 1
            else:
                break

        target_scroll_px = int(round(top_mm * px_per_mm / max(1e-6, dpr)))
        return target_scroll_px, last_visible


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
        try:
            self.print_view.clear_playhead_overlay()
        except Exception:
            pass
        self._playhead_anchor_measure = None
        self._playhead_last_visible_measure = None

    # FX/editor hooks removed; FluidSynth is the single backend
    def _open_fx_editor(self) -> None:
        self._status(self.tr("Synth FX editor removed"), 2000)

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
                self._status(self.tr("Playback mode: External MIDI port"), 2500)
            else:
                self._status(
                    self.tr("Playback mode: {backend}").format(
                        backend=self._playback_system_label().replace(self.tr("Playback using "), "")
                    ),
                    2500,
                )
        try:
            # Recreate backend immediately so audition/test tone keep working after a mode switch.
            self._ensure_player_with_soundfont()
        except Exception as exc:
            self._notify_fluidsynth_missing(exc)
            pass

    def _set_send_midi_transport(self, enabled: bool) -> None:
        # Legacy stub kept for signal compatibility
        self._status(self.tr("MIDI transport settings removed"), 2000)

    def _play_test_tone(self) -> None:
        try:
            self._ensure_player_with_soundfont()
            self.player.audition_note(pitch=49, velocity=100, duration_sec=1.0)
            self._status(self.tr("Test tone"), 1500)
        except Exception as exc:
            self._notify_fluidsynth_missing(exc)
            self._status(self.tr("Test tone unavailable"), 2000)

    def _choose_audio_device(self) -> None:
        self._status(self.tr("Audio output is selected by the active playback backend"), 2000)

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
        if hasattr(self, 'editor') and self.editor_canvas is not None:
            if hasattr(self.editor_canvas, 'request_overlay_refresh'):
                self.editor_canvas.request_overlay_refresh()
            else:
                # Fallback: normal repaint
                self.editor_canvas.update()

    def _adjust_docks_to_fit(self) -> None:
        # Ensure both docks are sized and locked to their fit dimensions
        if hasattr(self.snap_dock, 'selector'):
            self.snap_dock.selector.adjust_to_fit()
        if hasattr(self.tool_dock, 'adjust_to_fit'):
            self.tool_dock.adjust_to_fit()
        self._freeze_left_panel_width_once()

    def _left_panel_width_px(self) -> int:
        lw = 0
        try:
            if hasattr(self, 'snap_dock'):
                lw = max(lw, int(self.snap_dock.width()))
        except Exception:
            pass
        try:
            if hasattr(self, 'tool_dock'):
                lw = max(lw, int(self.tool_dock.width()))
        except Exception:
            pass
        return int(max(0, lw))

    def _schedule_left_panel_width_save(self) -> None:
        try:
            if hasattr(self, '_left_panel_width_save_timer') and self._left_panel_width_save_timer is not None:
                self._left_panel_width_save_timer.start()
        except Exception:
            pass

    def _persist_left_panel_width(self) -> None:
        lw = int(self._left_panel_width_px())
        if lw <= 0:
            return
        if lw == int(getattr(self, '_left_panel_width_last_saved_px', 0) or 0):
            return
        adm = get_appdata_manager()
        adm.set("left_panel_width_px", int(lw))
        adm.save()
        self._left_panel_width_last_saved_px = int(lw)
        self._left_panel_width_pref_px = int(lw)

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

        pref_width = int(getattr(self, '_left_panel_width_pref_px', 220) or 220)
        if pref_width > 0:
            target_width = max(80, pref_width)
        else:
            target_width = max(80, snap_width, tool_width)
        # Apply an initial width while still allowing user resizing afterward
        self.resizeDocks([self.snap_dock, self.tool_dock], [target_width, target_width], QtCore.Qt.Orientation.Horizontal)
        self._schedule_left_panel_width_save()
        self._left_panel_width_frozen = True

    def resizeEvent(self, ev: QtGui.QResizeEvent) -> None:
        super().resizeEvent(ev)
        self._schedule_left_panel_width_save()

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

    def _on_tool_selected(self, name: str) -> None:
        # Tool state is collected directly at save-time.
        if str(name) != 'note':
            # Leave velocity mode state untouched; it is restored when returning to note tool
            pass

    def schedule_fonts_install_prompt(self, delay_ms: int = 150) -> None:
        """Schedule one startup prompt to install all required fonts."""
        if self._fonts_prompt_armed:
            return
        self._fonts_prompt_armed = True
        QtCore.QTimer.singleShot(max(0, int(delay_ms)), self._maybe_prompt_fonts_install)

    def _maybe_prompt_fonts_install(self) -> None:
        adm = get_appdata_manager()
        fonts_install_ok = bool(adm.get("fonts_install_ok", False))

        fonts = [
            {
                "key": "Edwin",
                "family": "Edwin",
                "check_name": "Edwin",
                "desc": "Edwin font family for headers and engraving.",
            },
            {
                "key": "FiraCode-SemiBold",
                "family": "Fira Code",
                "check_name": "FiraCode-SemiBold",
                "desc": "Fira Code SemiBold for UI consistency.",
            },
            {
                "key": "LelandText",
                "family": "LelandText",
                "check_name": "LelandText",
                "desc": "LelandText for dynamic symbols (f/mp/p etc...).",
            },
        ]
        from fonts import (
            has_system_font,
            has_installed_embedded_font_file,
            install_embedded_font_to_system,
        )
        missing: list[dict] = []
        for f in fonts:
            check_name = str(f.get("check_name", f["family"]))
            if not has_system_font(check_name) and not has_installed_embedded_font_file(str(f.get("key", ""))):
                missing.append(f)
        if not missing:
            if not fonts_install_ok:
                adm.set("fonts_install_ok", True)
                adm.save()
            return
        if fonts_install_ok:
            adm.set("fonts_install_ok", False)
            adm.save()
        msg = QtWidgets.QMessageBox(self)
        msg.setIcon(QtWidgets.QMessageBox.Icon.Information)
        msg.setWindowTitle(self.tr("Install required fonts"))
        lines = [self.tr("keyTAB can install embedded fonts to your user font folder so editing and engraving match:")]
        for f in missing:
            lines.append(f"- {f['family']}: {f['desc']}")
        lines.append(self.tr("Install all missing fonts now?"))
        msg.setText("\n".join(lines))
        msg.setStandardButtons(QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No)
        msg.setDefaultButton(QtWidgets.QMessageBox.StandardButton.Yes)
        result = msg.exec()
        if result != QtWidgets.QMessageBox.StandardButton.Yes:
            adm.save()
            return
        successes = []
        failures = []
        for f in missing:
            key = str(f["key"])
            family = str(f["family"])
            success, detail = install_embedded_font_to_system(key)
            if success:
                successes.append(family)
            else:
                failures.append((family, detail))

        still_missing = [
            f
            for f in fonts
            if not has_system_font(str(f.get("check_name", f["family"])))
            and not has_installed_embedded_font_file(str(f.get("key", "")))
        ]
        adm.set("fonts_install_ok", len(still_missing) == 0)
        adm.save()

        if successes:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Fonts installed"),
                self.tr("The following fonts were installed. keyTAB will restart to apply them:\n") + "\n".join(successes),
            )
            QtCore.QTimer.singleShot(100, self._request_app_restart)
        if failures or still_missing:
            details = "\n".join([f"{n}: {d}" for n, d in failures])
            if still_missing:
                if details:
                    details += "\n"
                details += "Still missing: " + ", ".join(str(f["family"]) for f in still_missing)
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Font installation failed"),
                self.tr("keyTAB could not install some fonts automatically:\n{details}").format(details=details),
            )

    def _request_app_restart(self) -> None:
        try:
            app = QtWidgets.QApplication.instance()
            if app is not None:
                app.setProperty("keytab_restart_in_progress", True)
            restart_current_process()
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

    def restore_window_state_from_appdata(self) -> None:
        try:
            adm = get_appdata_manager()
            start_max = bool(adm.get("window_maximized", True))
            start_fullscreen = bool(adm.get("window_fullscreen", False))
            if start_fullscreen:
                self.showFullScreen()
            elif not start_max:
                geom_b64 = str(adm.get("window_geometry", ""))
                if geom_b64:
                    self.restoreGeometry(QtCore.QByteArray.fromBase64(geom_b64.encode("ascii")))
                self.show()
            else:
                self.showMaximized()
        except Exception:
            self.showMaximized()
        self._sync_full_screen_action_state()

    # Duplicate keyPressEvent removed; using the earlier implementation for Escape handling

    def prepare_close(self) -> None:
        if getattr(self, '_prepare_close_done', False):
            return
        self._prepare_close_done = True
        # Ensure worker threads are stopped before application exits
        # Persist window state to appdata
        adm = get_appdata_manager()
        adm.set("window_maximized", bool(self.isMaximized()))
        adm.set("window_fullscreen", bool(self.isFullScreen()))
        geom_b64 = bytes(self.saveGeometry().toBase64()).decode("ascii")
        adm.set("window_geometry", geom_b64)
        # Save current splitter sizes for next startup
        sp = self.centralWidget()
        if sp is not None and hasattr(sp, 'sizes'):
            sizes = list(sp.sizes())
            adm.set("splitter_sizes", [int(sizes[0]) if sizes else 0, int(sizes[1]) if len(sizes) > 1 else 0])
        # Remember left panel width for next launch
        lw = 0
        if hasattr(self, 'snap_dock'):
            lw = max(lw, int(self.snap_dock.width()))
        if hasattr(self, 'tool_dock'):
            lw = max(lw, int(self.tool_dock.width()))
        if lw > 0:
            adm.set("left_panel_width_px", int(lw))
            self._left_panel_width_last_saved_px = int(lw)
        # Persist whether the session should restore from saved path or session snapshot
        fm = getattr(self, 'file_manager', None)
        if self._close_restore_saved_override is not None:
            adm.set("last_session_saved", bool(self._close_restore_saved_override))
            adm.set("last_session_path", str(self._close_restore_path_override or ""))
        elif fm is not None:
            # Fallback for non-interactive exits: infer from dirty/path state.
            was_saved = bool(fm.path() is not None and not fm.is_dirty())
            adm.set("last_session_saved", was_saved)
            adm.set("last_session_path", str(fm.path() or ""))
        adm.save()
        # Stop clock timer gracefully
        if hasattr(self, "_clock_timer") and self._clock_timer is not None:
            self._clock_timer.stop()
        # Fully dispose audio/MIDI backend so CoreMIDI/AudioToolbox threads are released
        self._dispose_player()
        # Stop playhead timer and clear overlay
        self._clear_playhead_overlay()
        # Close FX dialog if open
        if hasattr(self, '_fx_dialog') and self._fx_dialog is not None:
            self._fx_dialog.close()
            self._fx_dialog = None
        if hasattr(self, "print_view") and self.print_view is not None:
            self.print_view.shutdown()
        if hasattr(self, "engraver") and self.engraver is not None:
            self.engraver.shutdown()

    def _run_close_progress(self, path_text: str, *, did_save_project: bool) -> None:
        """Show a short closing progress animation (~0.4s)."""
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle(self.tr("Exiting keyTAB..."))
        dlg.setModal(True)
        dlg.setFixedWidth(380)
        layout = QtWidgets.QVBoxLayout(dlg)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        if did_save_project:
            label_text = self.tr("Saving...\n\n {path}").format(path=path_text)
        else:
            label_text = self.tr("Exiting in progress...")
        label = QtWidgets.QLabel(label_text)
        label.setWordWrap(True)
        bar = QtWidgets.QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(0)

        layout.addWidget(label)
        layout.addWidget(bar)

        dlg.show()
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, 50)

        steps = 30
        interval_ms = max(1, int(400 / steps))
        for i in range(steps + 1):
            bar.setValue(int((i / steps) * 100))
            QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ProcessEventsFlag.AllEvents, interval_ms)
            QtCore.QThread.msleep(interval_ms)
        dlg.accept()

    def closeEvent(self, ev: QtGui.QCloseEvent) -> None:
        # Unified close handling with optional close confirmation.
        pm = get_preferences_manager()
        save_on_exit = bool(pm.get("save_on_exit", True))

        app = QtWidgets.QApplication.instance()
        restarting = bool(app.property("keytab_restart_in_progress")) if app is not None else False

        restore_saved = False
        restore_path = str(self.file_manager.path() or "")
        did_save_project = False
        saved_progress_path = str(self.file_manager.path() or self.tr("unsaved session"))

        if not restarting and not save_on_exit:
            decision = self.file_manager.confirm_close_decision()
            if decision == "cancel":
                ev.ignore()
                return
            if decision == "saved":
                restore_saved = bool(self.file_manager.path() is not None)
                restore_path = str(self.file_manager.path() or "")
                did_save_project = restore_saved
                saved_progress_path = str(self.file_manager.path() or self.tr("unsaved session"))
            elif decision == "discarded":
                # User chose to discard edits: reopen the current file path on next startup.
                restore_saved = bool(self.file_manager.path() is not None)
                restore_path = str(self.file_manager.path() or "")
            else:  # proceed (e.g. not dirty)
                restore_saved = bool(self.file_manager.path() is not None)
                restore_path = str(self.file_manager.path() or "")
        else:
            # save_on_exit=True or controlled restart: persist via FileManager.
            self.file_manager.autosave_current()
            if self.file_manager.path() is not None:
                did_save_project = bool(self.file_manager.save())
                restore_saved = bool(self.file_manager.path() is not None)
                restore_path = str(self.file_manager.path() or "")
                saved_progress_path = str(self.file_manager.path() or self.tr("unsaved session"))
            else:
                restore_saved = False
                restore_path = ""

        self._close_restore_saved_override = bool(restore_saved)
        self._close_restore_path_override = str(restore_path)

        adm = get_appdata_manager()
        adm.set("last_session_saved", bool(self._close_restore_saved_override))
        adm.set("last_session_path", str(self._close_restore_path_override or ""))
        adm.save()
        # Show short closing progress animation
        self._run_close_progress(saved_progress_path, did_save_project=did_save_project)
        # Persist sizes via prepare_close
        self.prepare_close()
        super().closeEvent(ev)