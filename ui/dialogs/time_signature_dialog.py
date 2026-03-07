from __future__ import annotations
from PySide6 import QtCore, QtGui, QtWidgets
from typing import Optional
from utils.CONSTANT import QUARTER_NOTE_UNIT
from file_model.base_grid import resolve_grid_layer_offsets, MIN_TIME_GRID_TICKS

VALID_DENOMS = [1, 2, 4, 8, 16, 32, 64, 128]
SPECIAL_NO_BARLINES = "no barlines"
SPECIAL_NO_GRIDLINES = "no grid lines"
SPECIAL_NONE = "no barlines and grid lines"

class TimeSignatureDialog(QtWidgets.QDialog):
    previewChanged = QtCore.Signal(int, int, list, bool)
    def __init__(self, parent=None,
                 initial_numer: int = 4,
                 initial_denom: int = 4,
                 initial_grid_positions: Optional[list[int]] = None,
                 initial_indicator_enabled: Optional[bool] = True,
                 indicator_type: Optional[str] = None,
                 editor_widget: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("Set Time Signature")
        self.setModal(True)
        self.setWindowModality(QtCore.Qt.NonModal)
        #self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self._editor_widget = editor_widget
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(8)

        # Single entry: Time-Signature: "N/D"
        entry_row = QtWidgets.QHBoxLayout()
        entry_row.setContentsMargins(0, 0, 0, 0)
        entry_row.setSpacing(6)
        self.ts_label = QtWidgets.QLabel("Time-Signature:", self)
        self.ts_edit = QtWidgets.QLineEdit(self)
        self.ts_edit.setPlaceholderText("e.g., 4/4")
        self.ts_edit.setClearButtonEnabled(True)
        entry_row.addWidget(self.ts_label)
        entry_row.addWidget(self.ts_edit, 1)
        lay.addLayout(entry_row)
        self.setFocusProxy(self.ts_edit)

        # Validation message
        self.msg_label = QtWidgets.QLabel("", self)
        pal = self.msg_label.palette()
        pal.setColor(QtGui.QPalette.WindowText, QtGui.QColor(200, 0, 0))
        self.msg_label.setPalette(pal)
        lay.addWidget(self.msg_label)

        # Beat grouping presets
        self.info_label = QtWidgets.QLabel(
            "Choose timeline positions (ticks) where lines are engraved."
            f" 0 means barline at measure start; positive values are grid lines (min {int(MIN_TIME_GRID_TICKS)} ticks).",
            self,
        )
        self.info_label.setWordWrap(True)
        lay.addWidget(self.info_label)
        grouping_row = QtWidgets.QHBoxLayout()
        grouping_row.setContentsMargins(0, 0, 0, 0)
        grouping_row.setSpacing(6)
        self.grouping_label = QtWidgets.QLabel("Beat grouping:", self)
        self.grouping_combo = QtWidgets.QComboBox(self)
        self.grouping_combo.setEditable(True)
        self.grouping_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.grouping_combo.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        grouping_row.addWidget(self.grouping_label)
        grouping_row.addWidget(self.grouping_combo, 1)
        lay.addLayout(grouping_row)

        # Indicator enabled toggle (global type comes from Layout; dialog no longer edits it)
        indicator_row = QtWidgets.QHBoxLayout()
        indicator_row.setContentsMargins(0, 0, 0, 0)
        indicator_row.setSpacing(6)
        self.indicator_enabled_cb = QtWidgets.QCheckBox("Indicator enabled", self)
        indicator_row.addWidget(self.indicator_enabled_cb)
        lay.addLayout(indicator_row)

        # Buttons
        self.btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=self)
        self.btns.accepted.connect(self._on_accept_clicked)
        self.btns.rejected.connect(self.reject)
        lay.addWidget(self.btns)
        ok_btn = self.btns.button(QtWidgets.QDialogButtonBox.Ok)
        if ok_btn is not None:
            ok_btn.setDefault(True)
            ok_btn.setAutoDefault(True)

        # State
        self._numer = int(initial_numer)
        self._denom = int(initial_denom) if int(initial_denom) in VALID_DENOMS else 4
        self._max_width_px: Optional[int] = self._compute_max_width_px()
        # Indicator type (affects initial checkbox states)
        self._indicator_type: str = str(indicator_type or 'classical')
        if self._indicator_type not in ('classical', 'klavarskribo', 'both'):
            self._indicator_type = 'classical'
        # Initialize beat grouping sequence (one digit per beat)
        init_gp = list(initial_grid_positions or [])
        if init_gp:
            bar_off, grid_off = resolve_grid_layer_offsets(
                [int(p) for p in init_gp if isinstance(p, (int, float))],
                int(self._numer),
                int(self._denom),
            )
            seq = [int(round(v)) for v in (bar_off + grid_off)]
            self._grid_positions = seq
        else:
            self._grid_positions = []
        if not self._grid_positions:
            # Default: barline + denominator-based grid positions.
            self._grid_positions = self._default_time_positions(include_barline=True)
        # Initialize indicator state
        self._indicator_enabled: bool = bool(initial_indicator_enabled if initial_indicator_enabled is not None else True)
        self.indicator_enabled_cb.setChecked(self._indicator_enabled)

        # Build preset list and select current grouping
        self._rebuild_grouping_options()
        self._update_grouping_text_from_positions()
        # Initialize entry text
        self.ts_edit.setText(f"{self._numer}/{self._denom}")
        # React to changes
        self.ts_edit.textChanged.connect(self._on_text_changed)
        self.grouping_combo.editTextChanged.connect(self._on_grouping_changed)
        self.indicator_enabled_cb.toggled.connect(lambda _: self._emit_preview())

        # Focus and select the main entry when the dialog opens
        QtCore.QTimer.singleShot(0, self._focus_time_signature_entry)
        self._resize_to_content()

    def showEvent(self, event: QtGui.QShowEvent) -> None:
        super().showEvent(event)
        self._resize_to_content()

    def _focus_time_signature_entry(self) -> None:
        try:
            self.ts_edit.setFocus(QtCore.Qt.FocusReason.OtherFocusReason)
            self.ts_edit.selectAll()
        except Exception:
            pass

    def _compute_max_width_px(self) -> Optional[int]:
        try:
            screen = None
            try:
                screen = self.screen() or QtGui.QGuiApplication.primaryScreen()
            except Exception:
                screen = QtGui.QGuiApplication.primaryScreen()
            if screen is None:
                return None
            geo = screen.availableGeometry()
            return int(geo.width() * 0.8)
        except Exception:
            return None

    def _resize_to_content(self) -> None:
        max_w = self._compute_max_width_px()
        if max_w:
            self._max_width_px = max_w
            self.setMaximumWidth(max_w)
        fm = self.fontMetrics()
        texts = [self.grouping_combo.currentText()]
        texts.extend(self.grouping_combo.itemText(i) for i in range(self.grouping_combo.count()))
        max_text_w = max((fm.horizontalAdvance(t) for t in texts if t), default=0)
        label_w = self.grouping_label.sizeHint().width() if hasattr(self, 'grouping_label') and self.grouping_label else 0
        margins = self.layout().contentsMargins() if self.layout() else QtCore.QMargins(0, 0, 0, 0)
        extra = label_w + margins.left() + margins.right() + 140
        target = max(360, max_text_w + extra)
        if self._max_width_px:
            target = min(target, self._max_width_px)
        self.setMinimumWidth(min(target, self._max_width_px) if self._max_width_px else target)
        if target > self.width():
            self.resize(target, self.height())

    def _emit_preview(self) -> None:
        try:
            numer = int(self._numer)
            denom = int(self._denom)
            seq = list(self._grid_positions or [])
            indicator = bool(self.indicator_enabled_cb.isChecked())
            self.previewChanged.emit(numer, denom, seq, indicator)
        except Exception:
            pass

    @staticmethod
    def _sequence_to_text(seq: list[int]) -> str:
        return " ".join(str(int(p)) for p in seq)

    def _default_time_positions(self, include_barline: bool = True) -> list[int]:
        numer = max(1, int(self._numer))
        denom = max(1, int(self._denom))
        beat_len = float(QUARTER_NOTE_UNIT) * (4.0 / float(denom))
        times: list[int] = []
        if include_barline:
            times.append(0)
        for i in range(1, numer):
            times.append(int(round(float(i) * beat_len)))
        return times

    def _special_item_text(self, seq: list[int]) -> Optional[str]:
        values = [int(v) for v in (seq or [])]
        if not values:
            return SPECIAL_NONE
        if 0 in values:
            non_zero = [v for v in values if v != 0]
            if not non_zero:
                return SPECIAL_NO_GRIDLINES
            return None
        return SPECIAL_NO_BARLINES

    @staticmethod
    def _seq_from_groups(groups: list[int]) -> list[int]:
        seq: list[int] = []
        for g in groups:
            seq.extend(list(range(1, int(g) + 1)))
        return seq

    @staticmethod
    def _common_groupings(numer: int, denom: int) -> list[list[int]]:
        n = int(max(1, numer))
        d = int(max(1, denom))
        opts: list[list[int]] = []
        straight = list(range(1, n + 1))
        def add(seq: list[int]):
            if seq not in opts:
                opts.append(seq)
        add(straight)
        # Hand-picked common meters
        if n == 5:
            add(TimeSignatureDialog._seq_from_groups([3, 2]))
            add(TimeSignatureDialog._seq_from_groups([2, 3]))
        elif n == 6:
            add(TimeSignatureDialog._seq_from_groups([3, 3]))
            add(TimeSignatureDialog._seq_from_groups([2, 2, 2]))
        elif n == 7:
            add(TimeSignatureDialog._seq_from_groups([3, 4]))
            add(TimeSignatureDialog._seq_from_groups([4, 3]))
            add(TimeSignatureDialog._seq_from_groups([2, 2, 3]))
            add(TimeSignatureDialog._seq_from_groups([3, 2, 2]))
            add(TimeSignatureDialog._seq_from_groups([2, 3, 2]))
        elif n == 8:
            add(TimeSignatureDialog._seq_from_groups([3, 3, 2]))
            add(TimeSignatureDialog._seq_from_groups([2, 3, 3]))
            add(TimeSignatureDialog._seq_from_groups([3, 2, 3]))
            add(TimeSignatureDialog._seq_from_groups([2, 2, 2, 2]))
            add(TimeSignatureDialog._seq_from_groups([4, 4]))
        elif n == 9:
            add(TimeSignatureDialog._seq_from_groups([3, 3, 3]))
            add(TimeSignatureDialog._seq_from_groups([2, 2, 2, 3]))
            add(TimeSignatureDialog._seq_from_groups([2, 3, 2, 2]))
        elif n == 10:
            add(TimeSignatureDialog._seq_from_groups([3, 3, 4]))
            add(TimeSignatureDialog._seq_from_groups([4, 3, 3]))
            add(TimeSignatureDialog._seq_from_groups([3, 4, 3]))
            add(TimeSignatureDialog._seq_from_groups([2, 3, 2, 3]))
        elif n == 11:
            add(TimeSignatureDialog._seq_from_groups([3, 3, 3, 2]))
            add(TimeSignatureDialog._seq_from_groups([3, 3, 2, 3]))
            add(TimeSignatureDialog._seq_from_groups([3, 2, 3, 3]))
        elif n == 12:
            add(TimeSignatureDialog._seq_from_groups([3, 3, 3, 3]))
            add(TimeSignatureDialog._seq_from_groups([4, 4, 4]))
            add(TimeSignatureDialog._seq_from_groups([2, 2, 2, 2, 2, 2]))
        elif n >= 13:
            # Favor splitting into three near-equal groups for large meters
            g1 = n // 3
            g2 = (n - g1) // 2
            g3 = n - g1 - g2
            add(TimeSignatureDialog._seq_from_groups([g1, g2, g3]))
        # Add a swing-ish 2+3 if 5-based denominators (e.g., fast feels)
        if d == 8 and n in (5, 7, 9):
            add(TimeSignatureDialog._seq_from_groups([2, 3] if n == 5 else [2, 2, 3]))
        return opts

    @staticmethod
    def _generate_groupings(numer: int, limit: int = 25) -> list[list[int]]:
        n = int(max(1, numer))
        results: list[list[int]] = []
        sizes = [3, 2, 4, 1, 5]
        def dfs(rem: int, groups: list[int]):
            if len(results) >= limit:
                return
            if rem == 0:
                results.append(TimeSignatureDialog._seq_from_groups(groups))
                return
            for sz in sizes:
                if sz > rem:
                    continue
                dfs(rem - sz, groups + [sz])
        dfs(n, [])
        # Always include straight count if missing
        straight = list(range(1, n + 1))
        if straight not in results:
            results.insert(0, straight)
        # Dedup preserving order
        seen: set[tuple[int, ...]] = set()
        uniq: list[list[int]] = []
        for seq in results:
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            uniq.append(seq)
            if len(uniq) >= limit:
                break
        return uniq

    def _rebuild_grouping_options(self) -> None:
        numer = int(self._numer)
        denom = int(self._denom)
        preset_seqs = self._common_groupings(numer, denom)
        auto_seqs = self._generate_groupings(numer, limit=25)
        seqs: list[list[int]] = []
        def add(seq: list[int]):
            if len(seq) != numer:
                return
            if seq not in seqs:
                seqs.append(seq)
        for s in preset_seqs:
            add(s)
        for s in auto_seqs:
            add(s)
        current_seq = list(self._grid_positions or [])

        # Convert legacy beat-grouping presets into timeline offsets (ticks).
        measure_len_ticks = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
        beat_len_ticks = measure_len_ticks / float(max(1, numer))
        time_presets: list[list[int]] = []
        for seq in seqs:
            starts = [idx for idx, val in enumerate(seq, start=1) if int(val) == 1]
            if not starts:
                starts = [1]
            times = [int(round(float(s - 1) * beat_len_ticks)) for s in starts]
            if times not in time_presets:
                time_presets.append(times)
        default_times = self._default_time_positions(include_barline=True)
        if default_times not in time_presets:
            time_presets.insert(0, default_times)
        if current_seq and current_seq not in time_presets:
            time_presets.insert(0, current_seq)

        self.grouping_combo.blockSignals(True)
        self.grouping_combo.clear()
        # Special quick-select options (must stay at top).
        self.grouping_combo.addItem(SPECIAL_NO_BARLINES, "__special_no_barlines__")
        self.grouping_combo.addItem(SPECIAL_NO_GRIDLINES, "__special_no_gridlines__")
        self.grouping_combo.addItem(SPECIAL_NONE, "__special_none__")
        for seq in time_presets[:25]:
            txt = self._sequence_to_text(seq)
            self.grouping_combo.addItem(txt, seq)
        # Select current
        special = self._special_item_text(current_seq)
        if special is not None:
            self.grouping_combo.setCurrentText(special)
        else:
            self.grouping_combo.setCurrentText(self._sequence_to_text(current_seq) if current_seq else "")
        self.grouping_combo.blockSignals(False)
        self._resize_to_content()

    def _on_text_changed(self, s: str) -> None:
        numer, denom, err = self._parse_ts(s)
        ok_btn = self.btns.button(QtWidgets.QDialogButtonBox.Ok)
        if err:
            self.msg_label.setText(err)
            if ok_btn is not None:
                ok_btn.setEnabled(False)
            return
        self.msg_label.setText("")
        if ok_btn is not None:
            ok_btn.setEnabled(True)
        if numer is not None and denom is not None:
            changed = False
            if numer != self._numer:
                self._numer = numer
                self._grid_positions = self._default_time_positions(include_barline=True)
                self._update_grouping_text_from_positions()
                changed = True
            if denom != self._denom:
                self._denom = denom
                self._grid_positions = self._default_time_positions(include_barline=True)
                changed = True
            if changed:
                self._rebuild_grouping_options()
                self._emit_preview()
        self._resize_to_content()

        # Keep indicator enabled in sync with widget
        self._indicator_enabled = bool(self.indicator_enabled_cb.isChecked())

    def _on_accept_clicked(self) -> None:
        s = self.ts_edit.text().strip()
        numer, denom, err = self._parse_ts(s)
        if err or numer is None or denom is None:
            return
        if not self._apply_grouping_text():
            return
        self._numer = numer
        self._denom = denom
        # Sync indicator values before accept
        self._indicator_enabled = bool(self.indicator_enabled_cb.isChecked())
        self.accept()

    def _parse_ts(self, s: str) -> tuple[Optional[int], Optional[int], Optional[str]]:
        if not s:
            return None, None, "Enter '<numerator>/<denominator>' with digits and '/'."
        # validator already restricts pattern, but we further validate denominator set
        try:
            parts = s.split('/')
            if len(parts) != 2:
                return None, None, "Format must be N/D (e.g., 4/4)."
            n_str, d_str = parts[0], parts[1]
            if not n_str.isdigit() or not d_str.isdigit():
                return None, None, "Only digits and '/' allowed."
            n = int(n_str)
            d = int(d_str)
            if n <= 0:
                return None, None, "Numerator must be a positive integer."
            if d not in VALID_DENOMS:
                return None, None, f"Denominator must be one of {VALID_DENOMS}."
            return n, d, None
        except Exception:
            return None, None, "Invalid time signature."

    def _update_grouping_text_from_positions(self) -> None:
        # Build grouping string from per-beat sequence (e.g., [1,2,3,1,2,3,4] -> "1 2 3 1 2 3 4")
        if not self._grid_positions:
            txt = SPECIAL_NONE
        else:
            seq = [int(p) for p in self._grid_positions if int(p) >= 0]
            special = self._special_item_text(seq)
            txt = special if special is not None else self._sequence_to_text(seq)
        try:
            self.grouping_combo.blockSignals(True)
            self.grouping_combo.setCurrentText(txt)
        finally:
            self.grouping_combo.blockSignals(False)

    def _on_grouping_changed(self, s: str) -> None:
        # Live-validate grouping string
        if not s:
            self.msg_label.setText("Enter timeline positions in ticks (space-separated).")
            return
        txt = str(s or "").strip().lower()
        if txt not in (SPECIAL_NO_BARLINES, SPECIAL_NO_GRIDLINES, SPECIAL_NONE):
            if any(ch not in "0123456789 " for ch in s):
                self.msg_label.setText("Beat grouping must contain digits and spaces only.")
                return
        if not self._apply_grouping_text(quiet=True):
            return
        # Clear error if parsing is ok
        self.msg_label.setText("")
        self._resize_to_content()
        self._emit_preview()

    def _apply_grouping_text(self, quiet: bool = False) -> bool:
        s = (self.grouping_combo.currentText() or "").strip()
        if not s:
            if not quiet:
                self.msg_label.setText("Enter timeline positions in ticks (space-separated).")
            return False
        lowered = s.lower()
        if lowered == SPECIAL_NONE:
            self._grid_positions = []
            self._resize_to_content()
            self._emit_preview()
            return True
        if lowered == SPECIAL_NO_GRIDLINES:
            self._grid_positions = [0]
            self._resize_to_content()
            self._emit_preview()
            return True
        if lowered == SPECIAL_NO_BARLINES:
            self._grid_positions = self._default_time_positions(include_barline=False)
            self._resize_to_content()
            self._emit_preview()
            return True

        # Parse: time offsets in ticks, relative to measure start.
        parts = [p for p in s.split(" ") if p.strip() != ""]
        try:
            seq: list[int] = [int(p) for p in parts]
        except Exception:
            if not quiet:
                self.msg_label.setText("Grouping must be space-separated integers.")
            return False
        if not seq:
            if not quiet:
                self.msg_label.setText("Provide at least one timeline position.")
            return False

        numer = int(self._numer)
        denom = int(self._denom)
        measure_len_ticks = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
        uniq: list[int] = []
        for pos in seq:
            if pos < 0:
                if not quiet:
                    self.msg_label.setText("Timeline positions must be >= 0 ticks.")
                return False
            if pos > 0 and float(pos) < float(MIN_TIME_GRID_TICKS):
                if not quiet:
                    self.msg_label.setText(f"Positive timeline positions must be >= {int(MIN_TIME_GRID_TICKS)} ticks.")
                return False
            if float(pos) >= measure_len_ticks:
                if not quiet:
                    self.msg_label.setText(f"Timeline positions must be < measure length ({int(round(measure_len_ticks))} ticks).")
                return False
            if pos not in uniq:
                uniq.append(int(pos))
        if not uniq:
            if not quiet:
                self.msg_label.setText("No valid timeline positions were found.")
            return False

        self._grid_positions = sorted(uniq)
        try:
            self.grouping_combo.blockSignals(True)
            special = self._special_item_text(self._grid_positions)
            self.grouping_combo.setCurrentText(special if special is not None else self._sequence_to_text(self._grid_positions))
        finally:
            self.grouping_combo.blockSignals(False)
        self._resize_to_content()
        self._emit_preview()
        return True

    def get_values(self) -> tuple[int, int, list[int], bool]:
        return int(self._numer), int(self._denom), list(self._grid_positions), bool(self._indicator_enabled)

# Test
if __name__ == '__main__':
    # Simple standalone test harness to verify dialog mouse/keyboard interaction
    import sys
    from PySide6 import QtWidgets, QtCore

    app = QtWidgets.QApplication(sys.argv)
    # Optional: use Fusion style for consistent visuals across platforms
    QtWidgets.QApplication.setStyle('Fusion')

    win = QtWidgets.QMainWindow()
    win.setWindowTitle("TimeSignatureDialog Test")
    central = QtWidgets.QWidget(win)
    lay = QtWidgets.QVBoxLayout(central)
    lay.setContentsMargins(12, 12, 12, 12)
    lay.setSpacing(8)

    btn = QtWidgets.QPushButton("Open Time Signature Dialog", central)
    lbl = QtWidgets.QLabel("Result: (none)", central)

    def open_dialog():
        dlg = TimeSignatureDialog(parent=win, initial_numer=4, initial_denom=4, initial_grid_positions=[1, 2, 3, 4], initial_indicator_enabled=True)
        # Ensure dialog is visible and focused
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()
        res = dlg.exec()
        if res == QtWidgets.QDialog.Accepted:
            numer, denom, grid_positions, ind_enabled = dlg.get_values()
            print(f"[accepted] numer={numer}, denom={denom}, grid_positions={grid_positions}, indicator_enabled={ind_enabled}")
            lbl.setText(f"Result: {numer}/{denom} beats={grid_positions} indicator={'enabled' if ind_enabled else 'disabled'}")
        else:
            print("[rejected]")
            lbl.setText("Result: (cancel)")

    btn.clicked.connect(open_dialog)

    lay.addWidget(btn)
    lay.addWidget(lbl)
    win.setCentralWidget(central)
    win.resize(520, 260)
    win.show()
    sys.exit(app.exec())
