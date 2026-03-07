from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets
from fractions import Fraction
from utils.CONSTANT import QUARTER_NOTE_UNIT, SHORTEST_DURATION
from ui.widgets.tool_selector import LEFT_PANEL_PADDING_PX

BASE_ITEMS: list[tuple[int, str]] = [
    (1, "Whole"),
    (2, "Half"),
    (4, "Quarter"),
    (8, "Eighth"),
    (16, "Sixteenth"),
    (32, "Thirty-second"),
    (64, "Sixty-fourth"),
    (128, "One hundred twenty-eighth"),
]


class SnapSizeSelector(QtWidgets.QWidget):
    snapChanged = QtCore.Signal(int, int)  # (base, divide)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(LEFT_PANEL_PADDING_PX, 6, LEFT_PANEL_PADDING_PX, 6)
        layout.setSpacing(6)

        # Base step list (no icons; compact rows)
        self.list = QtWidgets.QListWidget(self)
        # Make the list span the full dock width
        self.list.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                    QtWidgets.QSizePolicy.Policy.Fixed)
        self.list.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setSelectionMode(QtWidgets.QAbstractItemView.SelectionMode.SingleSelection)
        self.list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.itemSelectionChanged.connect(self._emit_changed)
        layout.addWidget(self.list)
        # State (set before populating to avoid early signal using unset fields)
        self._base = 8
        self._divide = 1
        self._populate_list()

        # Divider control: [-] [label] [+]
        row = QtWidgets.QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        # Create minus button and center it; will use icon when available
        self.minus_btn = QtWidgets.QToolButton(self)
        ic_minus = None
        try:
            from icons.icons import get_qicon
            # Request a smaller source icon for crisper scaling
            ic_minus = get_qicon('minus', size=(36, 36))
        except Exception:
            ic_minus = None
        if ic_minus:
            self.minus_btn.setIcon(ic_minus)
            # 25% smaller than previous 45x45 -> ~34x34
            self.minus_btn.setIconSize(QtCore.QSize(34, 34))
            self.minus_btn.setText("")
        else:
            self.minus_btn.setText("-")
        self.minus_btn.clicked.connect(self._dec_divide)
        # Button visual size square 54x54 (25% smaller)
        self.minus_btn.setFixedSize(54, 54)
        fbtn = self.minus_btn.font()
        try:
            base_sz = fbtn.pointSize()
            target = (base_sz * 2 if base_sz > 0 else 20)
            fbtn.setPointSize(int(round(target * 0.75)))
        except Exception:
            fbtn.setPointSize(15)
        self.minus_btn.setFont(fbtn)
        # Wrapper to preserve original center position (54x54 area)
        self._minus_wrap = QtWidgets.QWidget(self)
        # Wrapper widened and heightened to match button size
        self._minus_wrap.setFixedSize(54, 54)
        wrap_l = QtWidgets.QVBoxLayout(self._minus_wrap)
        wrap_l.setContentsMargins(0, 0, 0, 0)
        wrap_l.setSpacing(0)
        wrap_l.addWidget(self.minus_btn, 0, QtCore.Qt.AlignmentFlag.AlignLeft | QtCore.Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self._minus_wrap)

        self.label = QtWidgets.QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        # Make label font twice as big
        fl = self.label.font()
        try:
            szl = fl.pointSize()
            fl.setPointSize(szl * 2 if szl > 0 else 20)
        except Exception:
            fl.setPointSize(20)
        self.label.setFont(fl)
        self.label.setMinimumHeight(54)
        row.addWidget(self.label, 1)

        self.plus_btn = QtWidgets.QToolButton(self)
        ic_plus = None
        try:
            from icons.icons import get_qicon
            # Request a smaller source icon for crisper scaling
            ic_plus = get_qicon('plus', size=(36, 36))
        except Exception:
            ic_plus = None
        if ic_plus:
            self.plus_btn.setIcon(ic_plus)
            # 25% smaller than previous 45x45 -> ~34x34
            self.plus_btn.setIconSize(QtCore.QSize(34, 34))
            self.plus_btn.setText("")
        else:
            self.plus_btn.setText("+")
        self.plus_btn.clicked.connect(self._inc_divide)
        # Button visual size square 54x54 (25% smaller)
        self.plus_btn.setFixedSize(54, 54)
        fbtn2 = self.plus_btn.font()
        try:
            base_sz2 = fbtn2.pointSize()
            target2 = (base_sz2 * 2 if base_sz2 > 0 else 20)
            fbtn2.setPointSize(int(round(target2 * 0.75)))
        except Exception:
            fbtn2.setPointSize(15)
        self.plus_btn.setFont(fbtn2)
        # Wrapper to preserve original center position (54x54 area)
        self._plus_wrap = QtWidgets.QWidget(self)
        # Wrapper widened and heightened to match button size
        self._plus_wrap.setFixedSize(54, 54)
        wrap_r = QtWidgets.QVBoxLayout(self._plus_wrap)
        wrap_r.setContentsMargins(0, 0, 0, 0)
        wrap_r.setSpacing(0)
        wrap_r.addWidget(self.plus_btn, 0, QtCore.Qt.AlignmentFlag.AlignRight | QtCore.Qt.AlignmentFlag.AlignVCenter)
        row.addWidget(self._plus_wrap)
        layout.addLayout(row)

        # Enable scroll-over behavior and click-to-reset via event filters
        # - Scrolling over the base list changes selection (no scrolling)
        self.list.installEventFilter(self)
        self.minus_btn.installEventFilter(self)
        self.plus_btn.installEventFilter(self)
        self.label.installEventFilter(self)

        # Update UI now that state is initialized
        self._update_ui()
        # Resize to fit all items exactly and keep the dock fixed size
        QtCore.QTimer.singleShot(0, self.adjust_to_fit)

    # --- UI helpers ---
    def _populate_list(self) -> None:
        self.list.clear()
        row_h = 28
        for base, name in BASE_ITEMS:
            text = f"{base} - {name}"
            it = QtWidgets.QListWidgetItem(text)
            # Compact row height
            it.setSizeHint(QtCore.QSize(it.sizeHint().width(), row_h))
            it.setData(QtCore.Qt.ItemDataRole.UserRole, base)
            self.list.addItem(it)
        # Select default base 8 (Eight)
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(QtCore.Qt.ItemDataRole.UserRole) == 8:
                self.list.setCurrentItem(it)
                break

    def _update_ui(self) -> None:
        self.label.setText(f"\u00F7 {self._divide}")  # ÷ symbol
        # Enable/disable minus based on min divide = 1
        self.minus_btn.setEnabled(self._divide > 1)

    def adjust_to_fit(self) -> None:
        """Adjust the widget to fit all list items and controls exactly, without scrollbars."""
        try:
            # Compute list height based on actual per-row size hints
            count = self.list.count()
            total_rows_h = 0
            for i in range(count):
                rh = int(self.list.sizeHintForRow(i))
                if rh <= 0:
                    rh = int(self.list.item(i).sizeHint().height()) or 28
                total_rows_h += rh
            frame = int(self.list.frameWidth()) * 2
            cm = self.list.contentsMargins()
            list_h = total_rows_h + frame + int(cm.top() + cm.bottom())
            self.list.setFixedHeight(list_h)

            # Allow the list to fill the available dock width via size policy
            fm = QtGui.QFontMetrics(self.list.font())
            max_text = ""
            for i in range(count):
                t = self.list.item(i).text()
                if len(t) > len(max_text):
                    max_text = t
            text_w = fm.horizontalAdvance(max_text) + 24  # padding baseline

            # Controls row height (buttons/label)
            ctrl_h = 54

            # Total widget size
            layout_margins = 6 * 2
            layout_spacing = 6
            # Add a small fudge factor to avoid off-by-1 cropping
            fudge = 2
            total_h = list_h + layout_spacing + ctrl_h + layout_margins + fudge
            total_w = max(text_w, 54 + 8 + fm.horizontalAdvance("÷ 64") + 8 + 54) + layout_margins
            # Fix height, allow width to expand with the dock
            self.setFixedHeight(total_h)
            # If hosted in a dock, clamp dock size too (include title bar height)
            p = self.parent()
            if isinstance(p, QtWidgets.QDockWidget):
                try:
                    style = p.style()
                    title_h = style.pixelMetric(QtWidgets.QStyle.PM_TitleBarHeight, None, p)
                except Exception:
                    title_h = 24
                dock_h = total_h + int(title_h)
                # Only clamp height; keep width flexible to match other docks
                p.setMinimumHeight(dock_h)
                p.setMaximumHeight(dock_h)
        except Exception:
            pass

    # --- Event handlers ---
    def _emit_changed(self) -> None:
        sel = self.list.selectedItems()
        if sel:
            base = int(sel[0].data(QtCore.Qt.ItemDataRole.UserRole))
            self._base = base
        self.snapChanged.emit(self._base, self._divide)

    def _dec_divide(self) -> None:
        if self._divide > 1:
            self._divide -= 1
            self._update_ui()
            self.snapChanged.emit(self._base, self._divide)

    def _inc_divide(self) -> None:
        # No explicit max; cap for sanity
        if self._divide < 64:
            self._divide += 1
            self._update_ui()
            self.snapChanged.emit(self._base, self._divide)

    # --- Interaction filters ---
    def eventFilter(self, obj: QtCore.QObject, ev: QtCore.QEvent) -> bool:
        try:
            if ev.type() == QtCore.QEvent.Type.Wheel and obj is self.list:
                # Wheel over the list changes selection up/down instead of scrolling
                if isinstance(ev, QtGui.QWheelEvent):
                    delta = ev.angleDelta().y()
                    if delta == 0:
                        return True
                    step = -1 if delta > 0 else 1
                    row = int(self.list.currentRow())
                    if row < 0:
                        row = 0
                    new_row = max(0, min(self.list.count() - 1, row + step))
                    if new_row != row:
                        self.list.setCurrentRow(new_row)
                return True  # consume to prevent scroll
            if ev.type() == QtCore.QEvent.Type.Wheel and obj in (self.minus_btn, self.plus_btn, self.label):
                # Positive delta -> increase; negative -> decrease
                delta = 0
                if isinstance(ev, QtGui.QWheelEvent):
                    delta = ev.angleDelta().y()
                if delta > 0:
                    self._inc_divide()
                elif delta < 0:
                    self._dec_divide()
                return True  # consume
            if ev.type() == QtCore.QEvent.Type.MouseButtonPress and obj is self.label:
                # Reset divide to 1 on label click
                if self._divide != 1:
                    self._divide = 1
                    self._update_ui()
                    self.snapChanged.emit(self._base, self._divide)
                return True
        except Exception:
            pass
        return super().eventFilter(obj, ev)

    # --- API ---
    def get_snap_base(self) -> int:
        return self._base

    def get_snap_divide(self) -> int:
        return self._divide

    def get_snap_fraction(self) -> Fraction:
        """Return the snap length as a fraction of a whole note.
        Example: base=8, divide=1 -> 1/8; base=4, divide=3 -> 1/12 (quarter triplet).
        """
        return Fraction(1, self._base * self._divide)

    def get_snap_size(self) -> float:
        """Return the snap size in time units based on QUARTER_NOTE_UNIT.

        Mapping: quarter note = QUARTER_NOTE_UNIT (256.0), whole = 4× quarter,
        eighth = quarter/2, sixteenth = quarter/4, etc. Finally divided by
        the current `divide` for tuplets (e.g., quarter triplet: 256 / 3).

        Example: base=8 (eighth), divide=1 -> 128.0
                 base=4 (quarter), divide=3 -> ~85.333...
        """
        # Base denominator (1,2,4,8,...) to time units
        base_den = max(1, int(self._base))
        base_units = (QUARTER_NOTE_UNIT * 4.0) / float(base_den)
        divide = max(1, int(self._divide))
        snap = base_units / float(divide)
        if snap < float(SHORTEST_DURATION):
            snap = float(SHORTEST_DURATION)  # minimum snap size
        return snap

    def set_snap(self, base: int, divide: int, emit: bool = True) -> None:
        """Programmatically set snap base and divide, update UI, and optionally emit."""
        try:
            base = int(base)
            divide = int(divide)
        except Exception:
            return
        # Clamp
        base = base if base in [1, 2, 4, 8, 16, 32, 64, 128] else 8
        divide = max(1, min(64, divide))
        self._base = base
        self._divide = divide
        # Select the matching base row
        try:
            for i in range(self.list.count()):
                it = self.list.item(i)
                if int(it.data(QtCore.Qt.ItemDataRole.UserRole)) == base:
                    # Avoid duplicate selection change emissions; set directly
                    self.list.setCurrentItem(it)
                    break
        except Exception:
            pass
        self._update_ui()
        if emit:
            try:
                self.snapChanged.emit(self._base, self._divide)
            except Exception:
                pass


class SnapSizeDock(QtWidgets.QDockWidget):
    def __init__(self, parent=None):
        super().__init__("Snap Size", parent)
        self.setObjectName("SnapSizeDock")
        # Lock dock: no moving, no floating, no closing
        self.setAllowedAreas(QtCore.Qt.DockWidgetArea.LeftDockWidgetArea | QtCore.Qt.DockWidgetArea.RightDockWidgetArea)
        self.setFeatures(QtWidgets.QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        self.selector = SnapSizeSelector(self)
        self.setWidget(self.selector)
        try:
            self.selector.snapChanged.connect(self._on_snap_changed_update_title)
        except Exception:
            pass

    def showEvent(self, ev: QtGui.QShowEvent) -> None:
        super().showEvent(ev)
        try:
            # Recompute fit once the dock has a real window handle/style metrics
            self.selector.adjust_to_fit()
            self._update_title()
        except Exception:
            pass

    def _on_snap_changed_update_title(self, base: int, divide: int) -> None:
        self._update_title()

    def _update_title(self) -> None:
        frac = self.selector.get_snap_fraction()
        size = self.selector.get_snap_size()
        # Display as numerator/denominator and time units
        text = f"Snap Size: {frac.numerator}/{frac.denominator} | {size:.1f}"
        self.setWindowTitle(text)
