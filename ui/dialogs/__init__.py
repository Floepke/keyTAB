"""ui.dialogs — shared utilities for all QDialog subclasses."""
from __future__ import annotations

from PySide6 import QtCore, QtGui, QtWidgets


class DialogGeometryMixin:
    """Mixin for QDialog subclasses that persists window geometry via appdata.

    Usage
    -----
    1. Inherit *before* QDialog:  ``class MyDialog(DialogGeometryMixin, QtWidgets.QDialog)``
    2. Set the class attribute:   ``DIALOG_KEY = "my_dialog_name"``
       The appdata key used will be ``"my_dialog_name_dialog_geometry"``.
    3. Call ``super().__init__(...)`` normally — no extra setup required.
       On first show the dialog is sized from the screen; on subsequent opens
       the saved geometry (position + size) is restored.
    """

    #: Override in each dialog subclass, e.g. ``DIALOG_KEY = "style"``
    DIALOG_KEY: str = ""

    # Fractional caps relative to available screen geometry.
    _SCREEN_CAP_FRACTION: tuple[float, float] = (0.62, 0.78)
    # Additional width cap derived from the short screen edge to avoid very
    # wide dialogs on ultrawide monitors.
    _SHORT_EDGE_WIDTH_FACTOR: float = 1.35
    # Padding around measured content hint (w, h).
    _CONTENT_PADDING: tuple[int, int] = (24, 20)
    # Extra px added when widening to clear horizontal overflow.
    _H_OVERFLOW_SAFETY_PADDING: int = 20
    # Number of deferred passes to let Qt settle layout/scrollbars.
    _H_OVERFLOW_PASSES: int = 2
    # Hard minimum / maximum in pixels (w, h).
    _MIN_SIZE: tuple[int, int] = (400, 300)
    _MAX_SIZE: tuple[int, int] = (1600, 1200)

    # Internal flag to ensure the geometry is only restored on the first show.
    _dialog_geometry_restored: bool = False

    # ------------------------------------------------------------------ #
    # Qt overrides                                                         #
    # ------------------------------------------------------------------ #

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)  # type: ignore[misc]
        if not self._dialog_geometry_restored:
            self._dialog_geometry_restored = True
            if not self._restore_dialog_geometry():
                w, h = self._compute_initial_size()
                self.resize(w, h)  # type: ignore[attr-defined]
                self._center_on_screen()
        self._schedule_horizontal_overflow_fix(self._H_OVERFLOW_PASSES)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._save_dialog_geometry()
        super().closeEvent(event)  # type: ignore[misc]

    def done(self, result: int) -> None:  # type: ignore[override]
        # QDialog.accept()/reject() call done() and may not emit closeEvent,
        # so persist geometry here as well for immediate in-session restores.
        self._save_dialog_geometry()
        super().done(result)  # type: ignore[misc]

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #

    def _geometry_appdata_key(self) -> str:
        return f"{self.DIALOG_KEY}_dialog_geometry"

    def _compute_initial_size(self) -> tuple[int, int]:
        """Return a balanced (w, h) using content hint with screen-based caps."""
        fallback = (760, 560)
        try:
            screen = QtGui.QGuiApplication.primaryScreen()
            if screen is None:
                return fallback
            avail = screen.availableGeometry()

            # Try content-driven sizing first.
            self.ensurePolished()  # type: ignore[attr-defined]
            layout = self.layout()  # type: ignore[attr-defined]
            if layout is not None:
                hint = layout.totalSizeHint()
                if not hint.isValid():
                    hint = layout.sizeHint()
                content_w = max(0, int(hint.width()))
                content_h = max(0, int(hint.height()))
            else:
                content_w, content_h = fallback

            pad_w, pad_h = self._CONTENT_PADDING
            target_w = content_w + pad_w
            target_h = content_h + pad_h

            # Screen-based caps. Width also uses short-edge protection so very
            # wide monitors do not produce oversized initial dialog widths.
            max_w, max_h = self._screen_caps_from_geometry(avail)

            w = max(self._MIN_SIZE[0], min(target_w, max_w))
            h = max(self._MIN_SIZE[1], min(target_h, max_h))
        except Exception:
            w, h = fallback
        w = max(self._MIN_SIZE[0], min(w, self._MAX_SIZE[0]))
        h = max(self._MIN_SIZE[1], min(h, self._MAX_SIZE[1]))
        return (w, h)

    def _screen_caps_from_geometry(self, avail: QtCore.QRect) -> tuple[int, int]:
        fw, fh = self._SCREEN_CAP_FRACTION
        short_edge = min(avail.width(), avail.height())
        max_w_short_edge = int(short_edge * self._SHORT_EDGE_WIDTH_FACTOR)
        max_w = min(int(avail.width() * fw), max_w_short_edge, self._MAX_SIZE[0])
        max_h = min(int(avail.height() * fh), self._MAX_SIZE[1])
        return (max_w, max_h)

    def _max_width_cap(self) -> int:
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return self._MAX_SIZE[0]
        avail = screen.availableGeometry()
        max_w, _max_h = self._screen_caps_from_geometry(avail)
        return max(self._MIN_SIZE[0], max_w)

    def _estimate_horizontal_overflow_px(self) -> int:
        """Estimate extra width needed to avoid internal horizontal scrolling."""
        extra = 0

        # 1) Existing horizontal overflows in abstract scroll areas.
        for area in self.findChildren(QtWidgets.QAbstractScrollArea):  # type: ignore[attr-defined]
            hbar = area.horizontalScrollBar()
            if hbar is None:
                continue
            overflow = max(0, int(hbar.maximum()))
            if overflow > 0:
                extra = max(extra, overflow + self._H_OVERFLOW_SAFETY_PADDING)

        # 2) Hidden wider pages in tab widgets (e.g., one style tab wider).
        for tabs in self.findChildren(QtWidgets.QTabWidget):  # type: ignore[attr-defined]
            count = int(tabs.count())
            if count <= 1:
                continue
            current_page = tabs.currentWidget()
            current_w = 0
            if current_page is not None:
                current_w = max(
                    int(current_page.sizeHint().width()),
                    int(current_page.minimumSizeHint().width()),
                )
            widest_w = current_w
            for i in range(count):
                page = tabs.widget(i)
                if page is None:
                    continue
                widest_w = max(
                    widest_w,
                    int(page.sizeHint().width()),
                    int(page.minimumSizeHint().width()),
                )
            delta = widest_w - current_w
            if delta > 0:
                extra = max(extra, delta + self._H_OVERFLOW_SAFETY_PADDING)

        return max(0, extra)

    def _schedule_horizontal_overflow_fix(self, passes: int) -> None:
        if passes <= 0:
            return
        QtCore.QTimer.singleShot(0, lambda: self._apply_horizontal_overflow_fix(passes))

    def _apply_horizontal_overflow_fix(self, passes: int) -> None:
        try:
            needed_extra = self._estimate_horizontal_overflow_px()
            if needed_extra > 0:
                cur_w = int(self.width())  # type: ignore[attr-defined]
                cap_w = self._max_width_cap()
                target_w = min(cap_w, cur_w + needed_extra)
                if target_w > cur_w:
                    self.resize(target_w, int(self.height()))  # type: ignore[attr-defined]
        except Exception:
            pass
        if passes > 1:
            self._schedule_horizontal_overflow_fix(passes - 1)

    def _center_on_screen(self) -> None:
        screen = QtGui.QGuiApplication.primaryScreen()
        if screen is None:
            return
        avail = screen.availableGeometry()
        self.move(  # type: ignore[attr-defined]
            avail.center() - self.rect().center()  # type: ignore[attr-defined]
        )

    def _restore_dialog_geometry(self) -> bool:
        """Load saved geometry from appdata.  Returns True when geometry was applied."""
        if not self.DIALOG_KEY:
            return False

        from appdata_manager import get_appdata_manager
        from PySide6 import QtCore
        adm = get_appdata_manager()
        geom_b64 = str(adm.get(self._geometry_appdata_key(), "") or "")
        if not geom_b64:
            return False
        data = QtCore.QByteArray.fromBase64(geom_b64.encode("ascii"))
        if not self.restoreGeometry(data):  # type: ignore[attr-defined]
            return False
        # Make sure the restored position is actually on a screen to avoid
        # invisible dialogs after monitor layout changes.
        pos = self.pos()  # type: ignore[attr-defined]
        on_screen = any(
            screen.availableGeometry().contains(pos)
            for screen in QtGui.QGuiApplication.screens()
        )
        if not on_screen:
            self._center_on_screen()
        return True

    def _save_dialog_geometry(self) -> None:
        if not self.DIALOG_KEY:
            return

        # Save geometry to appdata as base64-encoded QByteArray.
        from appdata_manager import get_appdata_manager
        adm = get_appdata_manager()
        geom_b64 = bytes(self.saveGeometry().toBase64()).decode("ascii")
        adm.set(self._geometry_appdata_key(), geom_b64)
        adm.save()
