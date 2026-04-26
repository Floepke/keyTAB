from __future__ import annotations
import sys
from PySide6 import QtCore, QtGui, QtWidgets
from icons.icons import get_qicon
from ui.style import Style

# Fixed row height to fit 36px icons comfortably
ITEM_ROW_HEIGHT_PX: int = 42
TOOLTIP_PANEL_HEIGHT_PX: int = 200
LEFT_PANEL_PADDING_PX: int = 6
# Configurable tool items.
# - 'name': internal tool identifier used in code/events
# - 'displayed_name': human-readable label shown in the listbox
# - 'icon': icon key from icons, defaults to 'name' when omitted
# - 'tooltip': hover text description (optional)
TOOL_ITEMS: list[dict] = [
    # basic notation elements
    { 'name': 'note',           'displayed_name': 'Note',           'icon': 'note',           'tooltip': 'Left Click/Drag: edit an existing note (duration on body, pitch+time on notehead). Left Click/Drag in empty space: create a note, then drag to set it. Right Click: delete note. Double Click: set/reset custom notehead.' },
    { 'name': 'grace_note',     'displayed_name': 'Grace Note',     'icon': 'grace_note',     'tooltip': 'Left Click/Drag: move an existing grace note (pitch+time). Left Click/Drag in empty space: create a grace note, then drag to adjust. Right Click: delete grace note.' },
    { 'name': 'count_line',     'displayed_name': 'Count Line',     'icon': 'count_line',     'tooltip': 'Left Click/Drag: edit an existing count line handle (start/end and time). Left Click/Drag in empty space: create a count line, then drag its end handle. Right Click: delete count line by handle.' },
    { 'name': 'dynamic',        'displayed_name': 'Dynamics',       'icon': 'dynamics',       'tooltip': 'Left Click/Drag: edit existing dynamic item (hairpin handle or dynamic symbol). Left Click/Drag in empty space: create new item from current mode (hairpin or symbol), then drag to position/length. Right Click: delete symbol or hairpin.' },
    { 'name': 'text',           'displayed_name': 'Text',           'icon': 'text',           'tooltip': 'Left Click/Drag: move existing text, or drag the red handle to rotate. Left Click/Drag in empty space: create new text + edit afterwards in the dialog. Right Click: delete text item.' },
    { 'name': 'beam',           'displayed_name': 'Note Grouping',  'icon': 'beam',           'tooltip': 'Left Click/Drag: edit an existing beam marker duration. Left Click/Drag in empty space: create a beam marker, then drag to set duration. Right Click: delete marker under cursor (or create one if none exists there).' },
    # layout elements
    { 'name': 'line_break',     'displayed_name': 'Line/Page-Break Marker','icon': 'line_break',     'tooltip': 'Left Click/Drag: edit existing line/page-break marker (toggle L/P on click, move on drag). Left Click/Drag in empty space: create a new line-break marker at cursor time. Right Click: delete marker.' },
    { 'name': 'time_signature', 'displayed_name': 'Time Signature', 'icon': 'time_signature', 'tooltip': 'Left Click/Drag: edit existing meter/grid context (on barline: open time-signature dialog; off barline: add grid line). Left Click/Drag in empty space: create a new subdivision in current measure. Right Click: remove subdivision, or remove a time-signature change at a barline.' },
    { 'name': 'grid_band',      'displayed_name': 'Grid Band',      'icon': 'grid_band',      'tooltip': 'Left Click/Drag: edit an existing grid-band marker duration. Left Click/Drag in empty space: create a grid-band marker, then drag to set duration. Right Click: insert/remove a stop marker (zero duration) at the clicked band start.' },
    { 'name': 'tempo',          'displayed_name': 'Tempo',          'icon': 'metronome',      'tooltip': 'Left Click/Drag: edit existing tempo (click to change BPM, drag to resize duration). Left Click/Drag in empty space: create a tempo marker + drag to set duration. Right Click: delete tempo (first tempo cannot be deleted).'} ,
    { 'name': 'slur',           'displayed_name': 'Slur',           'icon': 'slur',           'tooltip': 'Left Click/Drag: edit existing slur by dragging a control handle. Left Click/Drag in empty space: create a new slur + shape it by dragging. Right Click: delete slur at handle.' },
    { 'name': 'barline',        'displayed_name': 'Barline Symbols','icon': 'repeats',        'tooltip': 'Left Click/Drag: edit existing barline symbol placement by inserting selected symbol at clicked position. Left Click/Drag in empty space: create selected barline symbol (start repeat, end repeat, or double barline). Right Click: delete any barline symbol at nearest event position.' },
    { 'name': 'pedal',          'displayed_name': 'Pedal',          'icon': 'pedal',          'tooltip': 'Left Click/Drag: move existing pedal symbol (paired up/down moves together when linked). Left Click/Drag in empty space: create selected pedal symbol + adjust afterwards. Right Click: delete pedal symbol. Double Click: toggle symbol visibility in engraving.' },
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
        self._apply_platform_scrollbar_width()
        self._populate()

    def _apply_platform_scrollbar_width(self) -> None:
        try:
            vscroll = self.verticalScrollBar()
            if vscroll is None:
                return
            extent = int(self.style().pixelMetric(QtWidgets.QStyle.PixelMetric.PM_ScrollBarExtent))
            width = max(12, int(extent * 2))
            vscroll.setFixedWidth(width)
            vscroll.setStyleSheet(
                "QScrollBar:vertical {"
                f"width: {width}px;"
                "}"
            )
        except Exception:
            pass

    def _emit_selected(self) -> None:
        items = self.selectedItems()
        if items:
            name = items[0].data(QtCore.Qt.ItemDataRole.UserRole)
            if isinstance(name, str):
                self.toolSelected.emit(name)

    def _populate(self) -> None:
        self.clear()
        _names = {
            'note':           self.tr("Note"),
            'grace_note':     self.tr("Grace Note"),
            'count_line':     self.tr("Count Line"),
            'dynamic':        self.tr("Dynamics"),
            'beam':           self.tr("Note Grouping"),
            'line_break':     self.tr("Line/Page Break"),
            'time_signature': self.tr("Time Signature"),
            'grid_band':      self.tr("Grid Band"),
            'tempo':          self.tr("Tempo"),
            'slur':           self.tr("Slur"),
            'text':           self.tr("Text"),
            'barline':        self.tr("Barline Symbols"),
            'pedal':          self.tr("Pedal"),
        }
        _tooltips = {
            'note':           self.tr("Left Click/Drag: edit an existing note (duration on body, pitch+time on notehead). Left Click/Drag in empty space: create a note, then drag to set it. Right Click: delete note. Double Click: set/reset custom notehead."),
            'grace_note':     self.tr("Left Click/Drag: move an existing grace note (pitch+time). Left Click/Drag in empty space: create a grace note, then drag to adjust. Right Click: delete grace note."),
            'count_line':     self.tr("Left Click/Drag: edit an existing count line handle (start/end and time). Left Click/Drag in empty space: create a count line, then drag its end handle. Right Click: delete count line by handle."),
            'dynamic':        self.tr("Left Click/Drag: edit existing dynamic item (hairpin handle or dynamic symbol). Left Click/Drag in empty space: create new item from current mode (hairpin or symbol), then drag to position/length. Right Click: delete symbol or hairpin."),
            'beam':           self.tr("Left Click/Drag: edit an existing beam marker duration. Left Click/Drag in empty space: create a beam marker, then drag to set duration. Right Click: delete marker under cursor (or create one if none exists there)."),
            'line_break':     self.tr("Left Click/Drag: edit existing line/page-break marker (toggle L/P on click, move on drag). Left Click/Drag in empty space: create a new line-break marker at cursor time. Right Click: delete marker."),
            'time_signature': self.tr("Left Click/Drag: edit existing meter/grid context (on barline: open time-signature dialog; off barline: add grid line). Left Click/Drag in empty space: create a new subdivision in current measure. Right Click: remove subdivision, or remove a time-signature change at a barline."),
            'grid_band':      self.tr("Left Click/Drag: edit an existing grid-band marker duration. Left Click/Drag in empty space: create a grid-band marker, then drag to set duration. Right Click: insert/remove a stop marker (zero duration) at the clicked band start."),
            'tempo':          self.tr("Left Click/Drag: edit existing tempo (click to change BPM, drag to resize duration). Left Click/Drag in empty space: create a tempo marker + drag to set duration. Right Click: delete tempo (first tempo cannot be deleted)."),
            'slur':           self.tr("Left Click/Drag: edit existing slur by dragging a control handle. Left Click/Drag in empty space: create a new slur + shape it by dragging. Right Click: delete slur at handle."),
            'text':           self.tr("Left Click/Drag: move existing text, or drag the red handle to rotate. Left Click/Drag in empty space: create new text + edit afterwards in the dialog. Right Click: delete text item."),
            'barline':        self.tr("Left Click/Drag: edit existing barline symbol placement by inserting selected symbol at clicked position. Left Click/Drag in empty space: create selected barline symbol (start repeat, end repeat, or double barline). Right Click: delete any barline symbol at nearest event position."),
            'pedal':          self.tr("Left Click/Drag: move existing pedal symbol (paired up/down moves together when linked). Left Click/Drag in empty space: create selected pedal symbol + adjust afterwards. Right Click: delete pedal symbol. Double Click: toggle symbol visibility in engraving."),
        }
        for conf in TOOL_ITEMS:
            name = str(conf.get('name', ''))
            icon_name = str(conf.get('icon', name))
            label = _names.get(name, str(conf.get('displayed_name', name.replace('_', ' ').capitalize())))
            tooltip = _tooltips.get(name, str(conf.get('tooltip', label)))
            # Request high-DPI crisp icon at 36x36 CSS pixels
            icon = get_qicon(icon_name, size=(36, 36)) or QtGui.QIcon()
            it = QtWidgets.QListWidgetItem(icon, label)
            # 'name' remains the internal identifier used by code; store in UserRole
            it.setData(QtCore.Qt.ItemDataRole.UserRole, name)
            it.setData(QtCore.Qt.ItemDataRole.ToolTipRole, tooltip)
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

    def wheelEvent(self, ev: QtGui.QWheelEvent) -> None:
        """Use wheel to move selection instead of scrolling."""
        delta = ev.angleDelta().y()
        if delta == 0:
            delta = ev.pixelDelta().y()
        if delta == 0:
            ev.accept()
            return
        step = -1 if delta > 0 else 1
        row = max(0, int(self.currentRow()))
        new_row = max(0, min(self.count() - 1, row + step))
        if new_row != row:
            self.setCurrentRow(new_row)
        ev.accept()

    def set_selected_tool(self, name: str, emit: bool = True) -> None:
        """Programmatically select a tool by its internal name and optionally emit."""
        try:
            name = str(name)
        except Exception:
            return
        if name in ('start_repeat', 'end_repeat'):
            name = 'barline'
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
        self.setWindowTitle(self.tr("Tools"))
        self.setObjectName("ToolSelectorDock")
        # Lock dock: no moving, no floating, no closing
        self.setAllowedAreas(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea | QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        self.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        # Wrap the list in a container with small margins to match Snap Size indent
        container = QtWidgets.QWidget(self)
        lay = QtWidgets.QVBoxLayout(container)
        lay.setContentsMargins(LEFT_PANEL_PADDING_PX, LEFT_PANEL_PADDING_PX, LEFT_PANEL_PADDING_PX, LEFT_PANEL_PADDING_PX)
        lay.setSpacing(6)
        self.selector = ToolSelectorWidget(container)
        lay.addWidget(self.selector, stretch=1)

        self.tooltip_area = QtWidgets.QFrame(container)
        self.tooltip_area.setObjectName("toolSelectorTooltipArea")
        self.tooltip_area.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.tooltip_area.setAutoFillBackground(True)
        self.tooltip_area.setBackgroundRole(QtGui.QPalette.ColorRole.Window)
        self.tooltip_area.setFixedHeight(int(TOOLTIP_PANEL_HEIGHT_PX))
        tooltip_layout = QtWidgets.QVBoxLayout(self.tooltip_area)
        tooltip_layout.setContentsMargins(LEFT_PANEL_PADDING_PX, LEFT_PANEL_PADDING_PX, LEFT_PANEL_PADDING_PX, LEFT_PANEL_PADDING_PX)
        tooltip_layout.setSpacing(0)
        self.tooltip_label = QtWidgets.QLabel(self.tooltip_area)
        self.tooltip_label.setWordWrap(True)
        self.tooltip_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignTop)
        self.tooltip_label.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        self.tooltip_label.setText("")
        tooltip_layout.addWidget(self.tooltip_label, stretch=1)
        lay.addWidget(self.tooltip_area, stretch=0)

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
        except Exception:
            pass

    def _on_tool_selected_update_title(self, name: str) -> None:
        self._update_title()

    def _update_title(self) -> None:
        # Reflect current selection in the title bar
        items = self.selector.selectedItems()
        if items:
            name = items[0].data(QtCore.Qt.ItemDataRole.UserRole)
            label = str(items[0].text())
            self.setWindowTitle(f"{self.tr('Tool')}: {label}")
        else:
            self.setWindowTitle(f"{self.tr('Tool')}: (none)")

    def set_tooltip_text(self, text: str) -> None:
        self.tooltip_label.setText(str(text or ""))
