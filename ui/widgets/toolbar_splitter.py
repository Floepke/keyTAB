from PySide6 import QtCore, QtGui, QtWidgets
from icons.icons import get_qicon
from ui.style import Style
from settings_manager import get_ui_scale


class DraggableToolButton(QtWidgets.QToolButton):
    """A QToolButton that also acts as a splitter handle drag area when dragged.
    Click behaviour is preserved; dragging is forwarded to the parent QSplitterHandle.
    """
    _DRAG_THRESHOLD = 5  # Manhattan-length pixels before a move is treated as a drag

    def __init__(self, handle: QtWidgets.QSplitterHandle, parent=None):
        super().__init__(parent)
        self._splitter_handle = handle
        self._press_pos: QtCore.QPoint | None = None
        self._dragging = False

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            self._press_pos = event.position().toPoint()
            self._dragging = False
            super().mousePressEvent(event)  # Register press visually and internally
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._press_pos is not None and not self._dragging:
            delta = event.position().toPoint() - self._press_pos
            if delta.manhattanLength() > self._DRAG_THRESHOLD:
                self._dragging = True
                # Synthesise a press on the handle at the original press position
                hp = QtCore.QPointF(self.mapToParent(self._press_pos))
                press_ev = QtGui.QMouseEvent(
                    QtCore.QEvent.Type.MouseButtonPress,
                    hp,
                    self._splitter_handle.mapToGlobal(hp.toPoint()),
                    QtCore.Qt.MouseButton.LeftButton,
                    QtCore.Qt.MouseButton.LeftButton,
                    event.modifiers(),
                )
                QtWidgets.QApplication.sendEvent(self._splitter_handle, press_ev)
        if self._dragging:
            hp = QtCore.QPointF(self.mapToParent(event.position().toPoint()))
            move_ev = QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseMove,
                hp,
                self._splitter_handle.mapToGlobal(hp.toPoint()),
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.LeftButton,
                event.modifiers(),
            )
            QtWidgets.QApplication.sendEvent(self._splitter_handle, move_ev)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._dragging:
            hp = QtCore.QPointF(self.mapToParent(event.position().toPoint()))
            release_ev = QtGui.QMouseEvent(
                QtCore.QEvent.Type.MouseButtonRelease,
                hp,
                self._splitter_handle.mapToGlobal(hp.toPoint()),
                QtCore.Qt.MouseButton.LeftButton,
                QtCore.Qt.MouseButton.NoButton,
                event.modifiers(),
            )
            QtWidgets.QApplication.sendEvent(self._splitter_handle, release_ev)
            self._dragging = False
            self._press_pos = None
            # Cancel the button's pressed state without emitting clicked
            self.setDown(False)
            event.accept()
        else:
            self._press_pos = None
            super().mouseReleaseEvent(event)


class StaveSelector(QtWidgets.QWidget):
    """Compact vertical stave selector: [-] [label] [+]."""

    selectedStaveChanged = QtCore.Signal(int)

    def __init__(self, parent=None, max_staves: int = 4):
        super().__init__(parent)
        self._max_staves = max(1, int(max_staves))
        self._stave_count = 1
        self._selected = 0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        self.minus_btn = QtWidgets.QToolButton(self)
        self.minus_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.minus_btn.setAutoRaise(True)
        ic_minus = get_qicon('minus', size=(36, 36))
        if ic_minus:
            self.minus_btn.setIcon(ic_minus)
        else:
            self.minus_btn.setText('-')

        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)

        self.plus_btn = QtWidgets.QToolButton(self)
        self.plus_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.plus_btn.setAutoRaise(True)
        ic_plus = get_qicon('plus', size=(36, 36))
        if ic_plus:
            self.plus_btn.setIcon(ic_plus)
        else:
            self.plus_btn.setText('+')

        _s = get_ui_scale()
        _btn = max(1, int(round(26 * _s)))
        _icon = max(1, int(round(17 * _s)))
        self.minus_btn.setFixedSize(_btn, _btn)
        self.plus_btn.setFixedSize(_btn, _btn)
        self.minus_btn.setIconSize(QtCore.QSize(_icon, _icon))
        self.plus_btn.setIconSize(QtCore.QSize(_icon, _icon))
        self.label.setMinimumHeight(max(1, int(round(20 * _s))))

        layout.addWidget(self.minus_btn, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.label, 0, QtCore.Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.plus_btn, 0, QtCore.Qt.AlignmentFlag.AlignCenter)

        self.minus_btn.clicked.connect(self._decrease)
        self.plus_btn.clicked.connect(self._increase)
        self._update_ui()

    def set_stave_count(self, count: int) -> None:
        self._stave_count = max(1, min(self._max_staves, int(count or 1)))
        if self._selected >= self._stave_count:
            self._selected = self._stave_count - 1
            self.selectedStaveChanged.emit(int(self._selected))
        self._update_ui()

    def set_selected_stave(self, index: int) -> None:
        if self._stave_count <= 0:
            self._selected = 0
        else:
            self._selected = max(0, min(self._stave_count - 1, int(index or 0)))
        self._update_ui()

    def _decrease(self) -> None:
        if self._selected <= 0:
            return
        self._selected -= 1
        self._update_ui()
        self.selectedStaveChanged.emit(int(self._selected))

    def _increase(self) -> None:
        if self._selected >= (self._stave_count - 1):
            return
        self._selected += 1
        self._update_ui()
        self.selectedStaveChanged.emit(int(self._selected))

    def _update_ui(self) -> None:
        self.label.setText(f"{int(self._selected) + 1}")
        self.minus_btn.setEnabled(self._selected > 0)
        self.plus_btn.setEnabled(self._selected < (self._stave_count - 1))
        self.minus_btn.setToolTip(self.tr("Select previous stave."))
        self.plus_btn.setToolTip(self.tr("Select next stave."))
        self.label.setToolTip(self.tr("Selected stave index (1-based)."))


class ToolbarHandle(QtWidgets.QSplitterHandle):
    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self.setObjectName("ToolbarHandle")
        self.setToolTip(self.tr(
            "Drag this splitter to zoom the editor and print-preview. "
            "Double-click to fit the current page entirely in the screen to get an overview of the document."
        ))
        _s = get_ui_scale()
        self._button_size = max(1, int(round(35 * _s)))
        # Keep handle width proportional to button width (base: 35 -> 50).
        parent.setHandleWidth(self._button_size + 15)
        # Prevent resize cursor when hovering the splitter handle
        self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        # Square button size to match ToolSelector list row height; scaled for macOS/Linux
        
        '''this button fits the print view to fit the window.'''
        # Default toolbar (top to bottom): fit, next, previous, engrave, play, stop
        self.fit_btn = DraggableToolButton(self, parent=self)
        self.fit_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.fit_btn.setIcon(QtGui.QIcon())
        self.fit_btn.setText('<-->')
        self.fit_btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        # Keep width; reduce height to half
        self.fit_btn.setFixedWidth(self._button_size)
        self.fit_btn.setFixedHeight(max(1, self._button_size // 2))
        self.fit_btn.setToolTip(self.tr(
            "Click to fit the page to the screen. "
            "If the page doesn't fit; this button fits the page. "
            "If the page already fits; this button hides the page. "
            "If the page is hidden; this button fits the page again. "
            "Drag to move the splitter and resize the editor and print-preview. "
        ))
        layout.addWidget(self.fit_btn)

        self._sep_after_fit_top = QtWidgets.QFrame(self)
        self._sep_after_fit_top.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self._sep_after_fit_top.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(self._sep_after_fit_top)

        self.stave_selector = StaveSelector(self, max_staves=4)
        layout.addWidget(self.stave_selector, 0, QtCore.Qt.AlignmentFlag.AlignHCenter)

        self._sep_after_fit_bottom = QtWidgets.QFrame(self)
        self._sep_after_fit_bottom.setFrameShape(QtWidgets.QFrame.Shape.HLine)
        self._sep_after_fit_bottom.setFrameShadow(QtWidgets.QFrame.Shadow.Sunken)
        layout.addWidget(self._sep_after_fit_bottom)

        '''this button goes to the next page in the print view.'''
        self.next_btn = QtWidgets.QToolButton(self)
        self.next_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        icn = get_qicon('next', size=(64, 64))
        if icn:
            self.next_btn.setIcon(icn)
        self.next_btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
        self.next_btn.setFixedSize(self._button_size, self._button_size)
        self.next_btn.setToolTip(self.tr("Go to the next print page."))
        layout.addWidget(self.next_btn)
        self.next_btn.clicked.connect(parent.nextRequested.emit)

        '''this button goes to the previous page in the print view.'''
        self.prev_btn = QtWidgets.QToolButton(self)
        self.prev_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        icp = get_qicon('previous', size=(64, 64))
        if icp:
            self.prev_btn.setIcon(icp)
        self.prev_btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
        self.prev_btn.setFixedSize(self._button_size, self._button_size)
        self.prev_btn.setToolTip(self.tr("Go to the previous print page."))
        layout.addWidget(self.prev_btn)
        self.prev_btn.clicked.connect(parent.previousRequested.emit)

        '''this button undoes the last editing action.'''
        self.undo_btn = QtWidgets.QToolButton(self)
        self.undo_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        ic_undo = get_qicon('undo', size=(64, 64))
        if ic_undo:
            self.undo_btn.setIcon(ic_undo)
        self.undo_btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
        self.undo_btn.setFixedSize(self._button_size, self._button_size)
        self.undo_btn.setToolTip(self.tr("Undo the last editing action. Shortcut: Ctrl+Z."))
        layout.addWidget(self.undo_btn)
        self.undo_btn.clicked.connect(parent.undoRequested.emit)

        '''this button redoes the last undone editing action.'''
        self.redo_btn = QtWidgets.QToolButton(self)
        self.redo_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        ic_redo = get_qicon('redo', size=(64, 64))
        if ic_redo:
            self.redo_btn.setIcon(ic_redo)
        self.redo_btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
        self.redo_btn.setFixedSize(self._button_size, self._button_size)
        self.redo_btn.setToolTip(self.tr("Redo the last undone editing action. Shortcut: Ctrl+Shift+Z."))
        layout.addWidget(self.redo_btn)
        self.redo_btn.clicked.connect(parent.redoRequested.emit)

        '''this button plays the music.'''
        self.play_btn = QtWidgets.QToolButton(self)
        self.play_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        icplay = get_qicon('play', size=(64, 64))
        if icplay:
            self.play_btn.setIcon(icplay)
        self.play_btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
        self.play_btn.setFixedSize(self._button_size, self._button_size)
        self.play_btn.setToolTip(self.tr(
            "Start playback from the current cursor position. "
            "<space> toggles playback from the current mouse cursor position."
        ))
        layout.addWidget(self.play_btn)
        self.play_btn.clicked.connect(parent.playRequested.emit)

        '''this button stops the music.'''
        self.stop_btn = QtWidgets.QToolButton(self)
        self.stop_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        icstop = get_qicon('stop', size=(64, 64))
        if icstop:
            self.stop_btn.setIcon(icstop)
        self.stop_btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
        self.stop_btn.setFixedSize(self._button_size, self._button_size)
        self.stop_btn.setToolTip(self.tr(
            "Stop playback immediately. "
            "<space> toggles playback from the current mouse cursor position."
        ))
        layout.addWidget(self.stop_btn)
        self.stop_btn.clicked.connect(parent.stopRequested.emit)

        '''this button opens the style dialog.'''
        self.style_btn = QtWidgets.QToolButton(self)
        self.style_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        ic_style = get_qicon('style', size=(64, 64))
        if ic_style:
            self.style_btn.setIcon(ic_style)
            self.style_btn.setText('')
        else:
            self.style_btn.setText('S')
        self.style_btn.setToolTip(self.tr('Appearance. Customize the visual style of the score.'))
        self.style_btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
        self.style_btn.setFixedSize(self._button_size, self._button_size)
        layout.addWidget(self.style_btn)
        self.style_btn.clicked.connect(parent.styleRequested.emit)

        '''this button opens the info dialog.'''
        self.info_btn = QtWidgets.QToolButton(self)
        self.info_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        ic_info = get_qicon('info', size=(64, 64))
        if ic_info:
            self.info_btn.setIcon(ic_info)
            self.info_btn.setText('')
        else:
            self.info_btn.setText('I')
        self.info_btn.setToolTip(self.tr('Title info. Edit title, composer, and copyright. View analysis information.'))
        self.info_btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
        self.info_btn.setFixedSize(self._button_size, self._button_size)
        layout.addWidget(self.info_btn)
        self.info_btn.clicked.connect(parent.infoRequested.emit)

        '''this button opens the line break dialog.'''
        self.line_break_btn = QtWidgets.QToolButton(self)
        self.line_break_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        ic_line_break = get_qicon('line_break', size=(64, 64))
        if ic_line_break:
            self.line_break_btn.setIcon(ic_line_break)
        self.line_break_btn.setToolTip(self.tr('Line breaks. Organize the document into systems and pages.'))
        self.line_break_btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
        self.line_break_btn.setFixedSize(self._button_size, self._button_size)
        layout.addWidget(self.line_break_btn)
        self.line_break_btn.clicked.connect(parent.lineBreakRequested.emit)

        self.selection_left_btn = QtWidgets.QToolButton(self)
        self.selection_left_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        ic_selection_left = get_qicon('selection_left', size=(64, 64))
        if ic_selection_left:
            self.selection_left_btn.setIcon(ic_selection_left)
        self.selection_left_btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
        self.selection_left_btn.setFixedSize(self._button_size, self._button_size)
        self.selection_left_btn.setToolTip(self.tr("Set selected notes to left hand. Shortcut: ["))
        layout.addWidget(self.selection_left_btn)
        self.selection_left_btn.clicked.connect(parent.selectionLeftRequested.emit)

        self.selection_right_btn = QtWidgets.QToolButton(self)
        self.selection_right_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        ic_selection_right = get_qicon('selection_right', size=(64, 64))
        if ic_selection_right:
            self.selection_right_btn.setIcon(ic_selection_right)
        self.selection_right_btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
        self.selection_right_btn.setFixedSize(self._button_size, self._button_size)
        self.selection_right_btn.setToolTip(self.tr("Set selected notes to right hand. Shortcut: ]"))
        layout.addWidget(self.selection_right_btn)
        self.selection_right_btn.clicked.connect(parent.selectionRightRequested.emit)

        layout.addStretch(1)

        self.setStyleSheet(
            "#ToolbarHandle { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #3a3f44, stop:1 #2b2f33); }"
        )

        # Fit button toggles (emit False); double-click will force True
        self.fit_btn.clicked.connect(lambda: parent.fitRequested.emit(False))
        self.stave_selector.selectedStaveChanged.connect(parent.staveSelectionRequested.emit)

    def set_stave_selector_state(self, selected_index: int, stave_count: int) -> None:
        try:
            self.stave_selector.set_stave_count(int(stave_count))
            self.stave_selector.set_selected_stave(int(selected_index))
        except Exception:
            pass

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent) -> None:
        # Forward double-click on the handle to request a fit action
        self.parent().fitRequested.emit(True)
        super().mouseDoubleClickEvent(ev)

    def set_buttons(self, defs: list[dict]) -> None:
        # Contextual buttons moved to a dedicated left-side widget in MainWindow.
        # Keep compatibility so stale callers do not break.
        _ = defs


class ToolbarSplitter(QtWidgets.QSplitter):
    # External trigger to request a fit action (True = force double-fit)
    fitRequested = QtCore.Signal(bool)
    # ToolManager contextual toolbar button clicked
    contextButtonClicked = QtCore.Signal(str)
    # Default toolbar actions
    nextRequested = QtCore.Signal()
    previousRequested = QtCore.Signal()
    undoRequested = QtCore.Signal()
    redoRequested = QtCore.Signal()
    engraveRequested = QtCore.Signal()
    playRequested = QtCore.Signal()
    stopRequested = QtCore.Signal()
    styleRequested = QtCore.Signal()
    infoRequested = QtCore.Signal()
    lineBreakRequested = QtCore.Signal()
    selectionLeftRequested = QtCore.Signal()
    selectionRightRequested = QtCore.Signal()
    staveSelectionRequested = QtCore.Signal(int)

    def __init__(self, orientation: QtCore.Qt.Orientation, parent=None):
        super().__init__(orientation, parent)
        assert orientation == QtCore.Qt.Orientation.Horizontal, \
            "ToolbarSplitter is intended for horizontal orientation"
        self.setObjectName("ToolbarSplitter")
        # Allow dragging the sash to fully collapse either child
        self.setChildrenCollapsible(True)
        self.setHandleWidth(56)
        # Hover cue driven by theme alternate background color
        alt = Style.get_named_qcolor('alternate_background_color', (240, 240, 240))
        self.setStyleSheet(
            "#ToolbarSplitter::handle { background: transparent; image: none; }\n"
            f"#ToolbarSplitter::handle:hover {{ background-color: rgb({alt.red()},{alt.green()},{alt.blue()}); }}"
        )

    def createHandle(self):
        h = ToolbarHandle(self.orientation(), self)
        # Keep a reference for ToolManager to update contextual buttons
        self._handle = h
        return h

    def set_context_buttons(self, defs: list[dict]):
        if hasattr(self, '_handle') and self._handle is not None:
            if hasattr(self._handle, 'set_buttons'):
                self._handle.set_buttons(defs)

    def set_stave_selector_state(self, selected_index: int, stave_count: int) -> None:
        if hasattr(self, '_handle') and self._handle is not None:
            if hasattr(self._handle, 'set_stave_selector_state'):
                self._handle.set_stave_selector_state(int(selected_index), int(stave_count))

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent) -> None:
        # Only trigger fit when double-clicking the splitter handle
        pos = ev.position().toPoint()
        handle_hit = False
        for i in range(1, self.count()):
            h = self.handle(i)
            if h is not None and h.geometry().contains(pos):
                handle_hit = True
                break
        if handle_hit:
            self.fitRequested.emit(True)
        super().mouseDoubleClickEvent(ev)
