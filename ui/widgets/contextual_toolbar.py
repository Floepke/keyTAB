from PySide6 import QtCore, QtGui, QtWidgets

from icons.icons import get_qicon
from ui.style import Style


class ContextualToolbar(QtWidgets.QWidget):
    """Vertical contextual toolbar for tool-specific buttons.
    
    Positioned on the right edge of the dock panels and resizes with them.
    """

    contextButtonClicked = QtCore.Signal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("ContextualToolbar")
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Fixed, QtWidgets.QSizePolicy.Policy.Expanding)

        self._button_size = 35
        self._icon_size = self._button_size - 6

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # Contextual tool buttons area (dynamically populated by ToolManager)
        self._toolbar_area = QtWidgets.QWidget(self)
        self._toolbar_layout = QtWidgets.QVBoxLayout(self._toolbar_area)
        self._toolbar_layout.setContentsMargins(0, 0, 0, 0)
        self._toolbar_layout.setSpacing(6)
        
        layout.addWidget(self._toolbar_area)
        layout.addStretch(1)

        self.setStyleSheet(
            "#ContextualToolbar { background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #3a3f44, stop:1 #2b2f33); }"
        )
        
        # Fixed width to contain buttons + padding
        self.setFixedWidth(self._button_size + 16)

    def set_buttons(self, defs: list[dict]) -> None:
        """Update contextual buttons from tool manager."""
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
            btn.setFixedSize(self._button_size, self._button_size)
            btn.setIconSize(QtCore.QSize(self._icon_size, self._icon_size))
            
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
            
            btn.clicked.connect(lambda _=False, n=name: self.contextButtonClicked.emit(n))
            self._toolbar_layout.addWidget(btn)
