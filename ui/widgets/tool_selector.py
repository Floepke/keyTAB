from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets
from icons.icons import get_qicon

# Fixed row height to fit 36px icons comfortably
ITEM_ROW_HEIGHT_PX: int = 42
# Configurable tool items.
# - 'name': internal tool identifier used in code/events
# - 'displayed_name': human-readable label shown in the listbox
# - 'icon': icon key from icons, defaults to 'name' when omitted
# - 'tooltip': hover text description (optional)
TOOL_ITEMS: list[dict] = [
    # basic notation elements
    { 'name': 'note',           'displayed_name': 'Note',           'icon': 'note',           'tooltip': 'Note: enter and edit notes; the basic element of the notation' },
    { 'name': 'grace_note',     'displayed_name': 'Grace Note',     'icon': 'grace_note',     'tooltip': 'Grace Note: a smaller note without stem;\nuse it for decoration notes.\nfor example: to engrave trills.' },
    { 'name': 'count_line',     'displayed_name': 'Count Line',     'icon': 'count_line',     'tooltip': 'Count Line: draw guide lines;\nfor highlighting rhythmic grid subdivisions' },
    { 'name': 'beam',           'displayed_name': 'Beam Grouping',  'icon': 'beam',           'tooltip': 'Beam: group notes with beams' },
    # layout elements
    { 'name': 'line_break',     'displayed_name': 'Line Break/Page Break',     'icon': 'line_break',     'tooltip': 'Line Break: insert line breaks;\nclick on a line break to edit its properties' },
    { 'name': 'time_signature', 'displayed_name': 'Time Signature/Grid Pattern', 'icon': 'time_signature', 'tooltip': 'Base-Grid/Time-Signature: Configure\ntime signature & grid pattern' },
    { 'name': 'tempo',          'displayed_name': 'Tempo',          'icon': 'metronome',           'tooltip': 'Tempo: add tempo regions (units per minute over a duration)'} ,
    { 'name': 'slur',           'displayed_name': 'Slur',           'icon': 'slur',           'tooltip': 'Slur: place phrasing slurs' },
    { 'name': 'text',           'displayed_name': 'Text',           'icon': 'text',           'tooltip': 'Text: place text annotations' },
    { 'name': 'start_repeat',   'displayed_name': 'Start Repeat',   'icon': 'start_repeat',   'tooltip': 'Start Repeat: repeat begin mark' },
    { 'name': 'end_repeat',     'displayed_name': 'End Repeat',     'icon': 'end_repeat',     'tooltip': 'End Repeat: repeat end mark' },
    { 'name': 'dynamic',        'displayed_name': 'Dynamics',       'icon': 'dynamic',        'tooltip': 'Dynamic: place dynamic markings' },
    { 'name': 'crescendo',      'displayed_name': 'Crescendo',      'icon': 'crescendo',      'tooltip': 'Crescendo: place hairpin volume up' },
    { 'name': 'decrescendo',    'displayed_name': 'Decrescendo',    'icon': 'decrescendo',    'tooltip': 'Decrescendo: place hairpin volume down' },
    { 'name': 'pedal',          'displayed_name': 'Pedal',          'icon': 'pedal',          'tooltip': 'Pedal: add pedal markings' },
]


class ToolSelectorWidget(QtWidgets.QListWidget):
    toolSelected = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        # Icon size reduced by a quarter from 48 -> 36
        self.setIconSize(QtCore.QSize(36, 36))
        # Allow per-item size hints; do not enforce uniform sizes
        self.setUniformItemSizes(False)
        self.setSpacing(4)
        # Fill available dock width
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Preferred)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        # Allow vertical scrolling when more tools are added
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        # Remove any inner margins to match Snap Size listbox appearance
        self.setContentsMargins(0, 0, 0, 0)
        self.setViewportMargins(0, 0, 0, 0)
        self.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.itemSelectionChanged.connect(self._emit_selected)
        self._populate()

    def _emit_selected(self) -> None:
        items = self.selectedItems()
        if items:
            name = items[0].data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(name, str):
                self.toolSelected.emit(name)

    def _populate(self) -> None:
        self.clear()
        for conf in TOOL_ITEMS:
            name = str(conf.get('name', ''))
            icon_name = str(conf.get('icon', name))
            label = str(conf.get('displayed_name', name.replace('_', ' ').capitalize()))
            tooltip = str(conf.get('tooltip', label))
            # Request high-DPI crisp icon at 36x36 CSS pixels
            icon = get_qicon(icon_name, size=(36, 36)) or QtGui.QIcon()
            it = QtWidgets.QListWidgetItem(icon, label)
            # 'name' remains the internal identifier used by code; store in UserRole
            it.setData(QtCore.Qt.ItemDataRole.UserRole, name)
            it.setToolTip(tooltip)
            # Make row height comfortably fit the 36px icon + padding
            it.setSizeHint(QtCore.QSize(it.sizeHint().width(), ITEM_ROW_HEIGHT_PX))
            self.addItem(it)
        # Select 'note' tool initially (visually and functionally)
        for i in range(self.count()):
            it = self.item(i)
            if it.data(QtCore.Qt.ItemDataRole.UserRole) == 'note':
                self.setCurrentItem(it)
                # Emit selection to update editor
                self._emit_selected()
                break

    def set_selected_tool(self, name: str, emit: bool = True) -> None:
        """Programmatically select a tool by its internal name and optionally emit."""
        try:
            name = str(name)
        except Exception:
            return
        for i in range(self.count()):
            it = self.item(i)
            if it.data(QtCore.Qt.ItemDataRole.UserRole) == name:
                self.setCurrentItem(it)
                if emit:
                    self._emit_selected()
                return


class ToolSelectorDock(QtWidgets.QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Tools", parent)
        self.setObjectName("ToolSelectorDock")
        # Lock dock: no moving, no floating, no closing
        self.setAllowedAreas(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea | QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        self.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        # Wrap the list in a container with small margins to match Snap Size indent
        container = QtWidgets.QWidget(self)
        lay = QtWidgets.QVBoxLayout(container)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(6)
        self.selector = ToolSelectorWidget(container)
        lay.addWidget(self.selector)
        self.setWidget(container)
        try:
            self.selector.toolSelected.connect(self._on_tool_selected_update_title)
        except Exception:
            pass

    def showEvent(self, ev: QtGui.QShowEvent) -> None:
        super().showEvent(ev)
        try:
            self.adjust_to_fit()
            self._update_title()
        except Exception:
            pass

    def adjust_to_fit(self) -> None:
        """Ensure the list expands to the available width; do not lock dock size.
        Height remains unmanaged and the list scrolls vertically as needed.
        """
        try:
            lst = self.selector
            lst.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                              QtWidgets.QSizePolicy.Policy.Preferred)
            # Do not enforce a fixed width; allow the dock to be resized.
            self.setMinimumWidth(0)
            self.setMaximumWidth(16777215)  # Qt default for "no max"
        except Exception:
            pass

    def _on_tool_selected_update_title(self, name: str) -> None:
        self._update_title()

    def _update_title(self) -> None:
        try:
            # Reflect current selection in the title bar
            items = self.selector.selectedItems()
            if items:
                name = items[0].data(QtCore.Qt.ItemDataRole.UserRole)
                label = str(items[0].text())
                self.setWindowTitle(f"Tool: {label}")
            else:
                self.setWindowTitle("Tool: (none)")
        except Exception:
            pass
