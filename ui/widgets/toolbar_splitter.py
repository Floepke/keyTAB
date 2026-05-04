from PySide6 import QtCore, QtGui, QtWidgets
from icons.icons import get_qicon
from ui.style import Style


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


class ToolbarHandle(QtWidgets.QSplitterHandle):
    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self.setObjectName("ToolbarHandle")
        self.setToolTip(self.tr(
            "Drag this splitter to zoom the editor and print-preview. "
            "Double-click to fit the current page entirely in the screen to get an overview of the document."
        ))
        parent.setHandleWidth(50)
        # Prevent resize cursor when hovering the splitter handle
        self.setCursor(QtCore.Qt.CursorShape.ArrowCursor)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        # Square button size to match ToolSelector list row height
        self._button_size = 35
        
        '''this button fits the print view to fit the window.'''
        # Default toolbar (top to bottom): fit, next, previous, engrave, play, stop
        self.fit_btn = DraggableToolButton(self, parent=self)
        self.fit_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        self.fit_btn.setIcon(QtGui.QIcon())
        self.fit_btn.setText('...')
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

        # Visual separator between dialog shortcuts and contextual toolbar
        sep = QtWidgets.QFrame(self)
        sep.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        sep.setFixedHeight(4)
        sep.setStyleSheet(
            "background-color: rgb(0, 0, 0);"
            "border-radius: 2px;"
        )
        layout.addWidget(sep)

        # Contextual tool area managed by ToolManager
        self._toolbar_area = QtWidgets.QWidget(self)
        self._toolbar_layout = QtWidgets.QVBoxLayout(self._toolbar_area)
        # Keep contextual area flush; we'll trim button width by 1px to reveal right border
        self._toolbar_layout.setContentsMargins(0, 0, 0, 0)
        self._toolbar_layout.setSpacing(6)
        layout.addWidget(self._toolbar_area)
        layout.addStretch(1)

        self.setStyleSheet(
            "#ToolbarHandle { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #3a3f44, stop:1 #2b2f33); }"
        )

        # Fit button toggles (emit False); double-click will force True
        self.fit_btn.clicked.connect(lambda: parent.fitRequested.emit(False))

    def mouseDoubleClickEvent(self, ev: QtGui.QMouseEvent) -> None:
        # Forward double-click on the handle to request a fit action
        self.parent().fitRequested.emit(True)
        super().mouseDoubleClickEvent(ev)

    def set_buttons(self, defs: list[dict]):
        # Clear previous buttons
        while self._toolbar_layout.count():
            item = self._toolbar_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        # Add new buttons
        for d in defs or []:
            name = d.get('name', '')
            icon_name = d.get('icon', '')
            text = str(d.get('text', '') or '')
            active = bool(d.get('active', False))
            tooltip = str(d.get('tooltip', name) or '').replace(';', '.')
            tooltip = tooltip.strip()
            if tooltip and not tooltip.endswith('.'):
                tooltip = f"{tooltip}."
            btn = QtWidgets.QToolButton(self._toolbar_area)
            btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            btn.setAutoRaise(False)
            ic = get_qicon(icon_name, size=(64, 64))
            rotation_deg = float(d.get('rotation', 0.0) or 0.0)
            if ic and abs(rotation_deg) > 0.1:
                pm = ic.pixmap(64, 64)
                transform = QtGui.QTransform().rotate(rotation_deg)
                pm = pm.transformed(transform, QtCore.Qt.TransformationMode.SmoothTransformation)
                ic = QtGui.QIcon(pm)
            if ic:
                btn.setIcon(ic)
            if text:
                btn.setText(text)
            btn.setToolTip(tooltip)
            if text and not ic:
                btn.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
            btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
            # Trim width by 1px to ensure the right outline remains visible inside the handle
            btn.setFixedSize(self._button_size - 1, self._button_size)
            if active:
                accent = Style.get_named_qcolor('accent', (0, 120, 215))
                border = accent.darker(120)
                btn.setStyleSheet(
                    "QToolButton {"
                    f"background-color: rgb({accent.red()},{accent.green()},{accent.blue()});"
                    "color: rgb(255,255,255);"
                    f"border: 1px solid rgb({border.red()},{border.green()},{border.blue()});"
                    "border-radius: 4px;"
                    "}"
                )
            # Emit contextButtonClicked(name) from parent splitter
            btn.clicked.connect(lambda _=False, n=name: self.parent().contextButtonClicked.emit(n))
            self._toolbar_layout.addWidget(btn)


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
            self._handle.set_buttons(defs)

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
