from PySide6 import QtCore, QtGui, QtWidgets
from icons.icons import get_qicon


class ToolbarHandle(QtWidgets.QSplitterHandle):
    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self.setObjectName("ToolbarHandle")
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
        self.fit_btn = QtWidgets.QToolButton(self)
        self.fit_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        # Fit button: no icon, no text, half-height
        try:
            self.fit_btn.setText("")
            # Ensure no icon is displayed
            self.fit_btn.setIcon(QtGui.QIcon())
        except Exception:
            pass
        # Keep width; reduce height to half
        self.fit_btn.setFixedWidth(self._button_size)
        self.fit_btn.setFixedHeight(max(1, self._button_size // 2))
        layout.addWidget(self.fit_btn)

        '''this button goes to the next page in the print view.'''
        self.next_btn = QtWidgets.QToolButton(self)
        self.next_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        icn = get_qicon('next', size=(64, 64))
        if icn:
            self.next_btn.setIcon(icn)
        self.next_btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
        self.next_btn.setFixedSize(self._button_size, self._button_size)
        layout.addWidget(self.next_btn)
        try:
            self.next_btn.clicked.connect(parent.nextRequested.emit)
        except Exception:
            pass
        
        '''this button goes to the previous page in the print view.'''
        self.prev_btn = QtWidgets.QToolButton(self)
        self.prev_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        icp = get_qicon('previous', size=(64, 64))
        if icp:
            self.prev_btn.setIcon(icp)
        self.prev_btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
        self.prev_btn.setFixedSize(self._button_size, self._button_size)
        layout.addWidget(self.prev_btn)
        try:
            self.prev_btn.clicked.connect(parent.previousRequested.emit)
        except Exception:
            pass

        '''this button triggers the engrave action in the print view.'''
        self.engrave_btn = QtWidgets.QToolButton(self)
        self.engrave_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        ice = get_qicon('engrave', size=(64, 64))
        if ice:
            self.engrave_btn.setIcon(ice)
        self.engrave_btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
        self.engrave_btn.setFixedSize(self._button_size, self._button_size)
        layout.addWidget(self.engrave_btn)
        try:
            self.engrave_btn.clicked.connect(parent.engraveRequested.emit)
        except Exception:
            pass

        # Visual separator between default toolbar and contextual toolbar
        sep = QtWidgets.QFrame(self)
        # Use a 1px separator that adapts to the current palette
        sep.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        sep.setFixedHeight(1)
        pal = self.palette()
        btn = pal.color(QtGui.QPalette.Button)
        # Slightly darken the button color to get a subtle separator line
        line_r = max(0, min(255, int(btn.red() * 0.75)))
        line_g = max(0, min(255, int(btn.green() * 0.75)))
        line_b = max(0, min(255, int(btn.blue() * 0.75)))
        sep.setStyleSheet(f"background-color: rgb({line_r}, {line_g}, {line_b});")
        layout.addWidget(sep)

        '''this button plays the music.'''
        self.play_btn = QtWidgets.QToolButton(self)
        self.play_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        icplay = get_qicon('play', size=(64, 64))
        if icplay:
            self.play_btn.setIcon(icplay)
        self.play_btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
        self.play_btn.setFixedSize(self._button_size, self._button_size)
        layout.addWidget(self.play_btn)
        try:
            self.play_btn.clicked.connect(parent.playRequested.emit)
        except Exception:
            pass

        '''this button stops the music.'''
        self.stop_btn = QtWidgets.QToolButton(self)
        self.stop_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        icstop = get_qicon('stop', size=(64, 64))
        if icstop:
            self.stop_btn.setIcon(icstop)
        self.stop_btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
        self.stop_btn.setFixedSize(self._button_size, self._button_size)
        layout.addWidget(self.stop_btn)
        try:
            self.stop_btn.clicked.connect(parent.stopRequested.emit)
        except Exception:
            pass

        # Visual separator between playback controls and dialog shortcuts
        sep = QtWidgets.QFrame(self)
        sep.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        sep.setFixedHeight(1)
        pal = self.palette()
        btn = pal.color(QtGui.QPalette.Button)
        line_r = max(0, min(255, int(btn.red() * 0.75)))
        line_g = max(0, min(255, int(btn.green() * 0.75)))
        line_b = max(0, min(255, int(btn.blue() * 0.75)))
        sep.setStyleSheet(f"background-color: rgb({line_r}, {line_g}, {line_b});")
        layout.addWidget(sep)

        # Quick dialogs: Style, Info, Line Breaks
        self.style_btn = QtWidgets.QToolButton(self)
        self.style_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        ic_style = get_qicon('style', size=(64, 64))
        if ic_style and not ic_style.isNull():
            self.style_btn.setIcon(ic_style)
        else:
            self.style_btn.setText('S')
        self.style_btn.setToolTip('Style')
        self.style_btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
        self.style_btn.setFixedSize(self._button_size, self._button_size)
        layout.addWidget(self.style_btn)
        try:
            self.style_btn.clicked.connect(parent.styleRequested.emit)
        except Exception:
            pass

        self.info_btn = QtWidgets.QToolButton(self)
        self.info_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        ic_info = get_qicon('info', size=(64, 64))
        if ic_info and not ic_info.isNull():
            self.info_btn.setIcon(ic_info)
        else:
            self.info_btn.setText('I')
        self.info_btn.setToolTip('Info')
        self.info_btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
        self.info_btn.setFixedSize(self._button_size, self._button_size)
        layout.addWidget(self.info_btn)
        try:
            self.info_btn.clicked.connect(parent.infoRequested.emit)
        except Exception:
            pass

        self.line_break_btn = QtWidgets.QToolButton(self)
        self.line_break_btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        ic_line_break = get_qicon('line_break', size=(64, 64))
        if ic_line_break and not ic_line_break.isNull():
            self.line_break_btn.setIcon(ic_line_break)
        else:
            self.line_break_btn.setText('LB')
        self.line_break_btn.setToolTip('Line Breaks')
        self.line_break_btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
        self.line_break_btn.setFixedSize(self._button_size, self._button_size)
        layout.addWidget(self.line_break_btn)
        try:
            self.line_break_btn.clicked.connect(parent.lineBreakRequested.emit)
        except Exception:
            pass

        # Visual separator between dialog shortcuts and contextual toolbar
        sep = QtWidgets.QFrame(self)
        sep.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        sep.setFixedHeight(1)
        pal = self.palette()
        btn = pal.color(QtGui.QPalette.Button)
        line_r = max(0, min(255, int(btn.red() * 0.75)))
        line_g = max(0, min(255, int(btn.green() * 0.75)))
        line_b = max(0, min(255, int(btn.blue() * 0.75)))
        sep.setStyleSheet(f"background-color: rgb({line_r}, {line_g}, {line_b});")
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
        try:
            self.parent().fitRequested.emit(True)
        except Exception:
            pass
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
            tooltip = d.get('tooltip', name)
            btn = QtWidgets.QToolButton(self._toolbar_area)
            btn.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
            ic = get_qicon(icon_name, size=(64, 64))
            if ic:
                btn.setIcon(ic)
            btn.setToolTip(tooltip)
            btn.setIconSize(QtCore.QSize(self._button_size - 6, self._button_size - 6))
            # Trim width by 1px to ensure the right outline remains visible inside the handle
            btn.setFixedSize(self._button_size - 1, self._button_size)
            # Emit contextButtonClicked(name) from parent splitter
            try:
                btn.clicked.connect(lambda _=False, n=name: self.parent().contextButtonClicked.emit(n))
            except Exception:
                pass
            self._toolbar_layout.addWidget(btn)


class ToolbarSplitter(QtWidgets.QSplitter):
    # External trigger to request a fit action (True = force double-fit)
    fitRequested = QtCore.Signal(bool)
    # ToolManager contextual toolbar button clicked
    contextButtonClicked = QtCore.Signal(str)
    # Default toolbar actions
    nextRequested = QtCore.Signal()
    previousRequested = QtCore.Signal()
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
        # Allow dragging the sash to fully collapse either child
        self.setChildrenCollapsible(True)
        self.setHandleWidth(56)

    def createHandle(self):
        h = ToolbarHandle(self.orientation(), self)
        # Keep a reference for ToolManager to update contextual buttons
        try:
            self._handle = h
        except Exception:
            pass
        return h

    def set_context_buttons(self, defs: list[dict]):
        try:
            if hasattr(self, '_handle') and self._handle is not None:
                self._handle.set_buttons(defs)
        except Exception:
            pass

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
