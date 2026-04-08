"""FluidSynth Reverb Configuration Dialog."""
from __future__ import annotations

from PySide6 import QtCore, QtWidgets

from appdata_manager import get_appdata_manager


class FluidSynthReverbConfigDialog(QtWidgets.QDialog):
    """Dialog for configuring FluidSynth reverb settings."""

    # Signal emitted when settings are applied
    reverb_settings_changed = QtCore.Signal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        tr = self.tr
        self.setWindowTitle(tr("FluidSynth Reverb Configuration"))
        # Make dialog non-modal (modeless) so it doesn't block the main window
        self.setModal(False)
        self.resize(500, 400)

        self._adm = get_appdata_manager()
        self._load_settings()

        layout = QtWidgets.QVBoxLayout(self)

        # Group box for reverb settings
        reverb_group = QtWidgets.QGroupBox(tr("Reverb Settings"), self)
        reverb_layout = QtWidgets.QFormLayout(reverb_group)
        reverb_layout.setSpacing(12)

        # Reverb enabled checkbox
        self._enabled_cb = QtWidgets.QCheckBox(self)
        self._enabled_cb.setChecked(self._reverb_enabled)
        reverb_layout.addRow(tr("Enable Reverb"), self._enabled_cb)

        # Room size slider
        room_layout = QtWidgets.QHBoxLayout()
        self._room_size_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self._room_size_slider.setRange(0, 100)
        self._room_size_slider.setValue(int(self._reverb_room_size * 100))
        self._room_size_label = QtWidgets.QLabel(f"{self._reverb_room_size:.2f}", self)
        self._room_size_label.setMinimumWidth(50)
        room_layout.addWidget(self._room_size_slider)
        room_layout.addWidget(self._room_size_label)
        reverb_layout.addRow(tr("Room Size (0-1)"), room_layout)
        self._room_size_slider.valueChanged.connect(self._update_room_size_label)

        # Damping slider
        damp_layout = QtWidgets.QHBoxLayout()
        self._damp_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self._damp_slider.setRange(0, 100)
        self._damp_slider.setValue(int(self._reverb_damp * 100))
        self._damp_label = QtWidgets.QLabel(f"{self._reverb_damp:.2f}", self)
        self._damp_label.setMinimumWidth(50)
        damp_layout.addWidget(self._damp_slider)
        damp_layout.addWidget(self._damp_label)
        reverb_layout.addRow(tr("Damping (0-1)"), damp_layout)
        self._damp_slider.valueChanged.connect(self._update_damp_label)

        # Width slider
        width_layout = QtWidgets.QHBoxLayout()
        self._width_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self._width_slider.setRange(0, 100)
        self._width_slider.setValue(int(self._reverb_width))
        self._width_label = QtWidgets.QLabel(f"{self._reverb_width:.2f}", self)
        self._width_label.setMinimumWidth(50)
        width_layout.addWidget(self._width_slider)
        width_layout.addWidget(self._width_label)
        reverb_layout.addRow(tr("Width (0-100)"), width_layout)
        self._width_slider.valueChanged.connect(self._update_width_label)

        # Level slider
        level_layout = QtWidgets.QHBoxLayout()
        self._level_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal, self)
        self._level_slider.setRange(0, 100)
        self._level_slider.setValue(int(self._reverb_level * 100))
        self._level_label = QtWidgets.QLabel(f"{self._reverb_level:.2f}", self)
        self._level_label.setMinimumWidth(50)
        level_layout.addWidget(self._level_slider)
        level_layout.addWidget(self._level_label)
        reverb_layout.addRow(tr("Level (0-1)"), level_layout)
        self._level_slider.valueChanged.connect(self._update_level_label)

        layout.addWidget(reverb_group)

        # Reset button
        reset_button = QtWidgets.QPushButton(tr("Reset to Defaults"), self)
        reset_button.clicked.connect(self._reset_to_defaults)
        layout.addWidget(reset_button)

        layout.addStretch()

        # Dialog buttons
        buttons = QtWidgets.QDialogButtonBox(self)
        apply_button = buttons.addButton(tr("Apply"), QtWidgets.QDialogButtonBox.ButtonRole.ApplyRole)
        close_button = buttons.addButton(tr("Close"), QtWidgets.QDialogButtonBox.ButtonRole.RejectRole)
        apply_button.clicked.connect(self._on_apply)
        close_button.clicked.connect(self.close)
        layout.addWidget(buttons)

    def _update_room_size_label(self, value: int) -> None:
        """Update room size label and value."""
        val = value / 100.0
        self._reverb_room_size = val
        self._room_size_label.setText(f"{val:.2f}")

    def _update_damp_label(self, value: int) -> None:
        """Update damping label and value."""
        val = value / 100.0
        self._reverb_damp = val
        self._damp_label.setText(f"{val:.2f}")

    def _update_width_label(self, value: int) -> None:
        """Update width label and value."""
        val = value / 1.0
        self._reverb_width = val
        self._width_label.setText(f"{val:.2f}")

    def _update_level_label(self, value: int) -> None:
        """Update level label and value."""
        val = value / 100.0
        self._reverb_level = val
        self._level_label.setText(f"{val:.2f}")

    def _reset_to_defaults(self) -> None:
        """Reset all reverb settings to defaults."""
        self._enabled_cb.setChecked(True)
        self._room_size_slider.setValue(60)  # 0.6
        self._damp_slider.setValue(40)  # 0.4
        self._width_slider.setValue(3)  # 3.0
        self._level_slider.setValue(90)  # 0.9

    def _load_settings(self) -> None:
        """Load reverb settings from appdata manager."""
        self._reverb_enabled = bool(self._adm.get("fluidsynth_reverb_enabled", True))
        self._reverb_room_size = float(self._adm.get("fluidsynth_reverb_room_size", 0.6))
        self._reverb_damp = float(self._adm.get("fluidsynth_reverb_damp", 0.4))
        self._reverb_width = float(self._adm.get("fluidsynth_reverb_width", 3.0))
        self._reverb_level = float(self._adm.get("fluidsynth_reverb_level", 0.9))

    def _save_settings(self) -> None:
        """Save reverb settings to appdata manager."""
        self._adm.set("fluidsynth_reverb_enabled", self._reverb_enabled)
        self._adm.set("fluidsynth_reverb_room_size", self._reverb_room_size)
        self._adm.set("fluidsynth_reverb_damp", self._reverb_damp)
        self._adm.set("fluidsynth_reverb_width", self._reverb_width)
        self._adm.set("fluidsynth_reverb_level", self._reverb_level)
        self._adm.save()

    def _on_apply(self) -> None:
        """Apply settings but keep dialog open so user can hear the changes."""
        # Update from UI
        self._reverb_enabled = self._enabled_cb.isChecked()

        # Save to appdata
        self._save_settings()

        # Emit signal with current settings
        settings = {
            'enabled': self._reverb_enabled,
            'room_size': self._reverb_room_size,
            'damp': self._reverb_damp,
            'width': self._reverb_width,
            'level': self._reverb_level,
        }
        self.reverb_settings_changed.emit(settings)

    def get_settings(self) -> dict:
        """Return current reverb settings as a dictionary."""
        return {
            'enabled': self._reverb_enabled,
            'room_size': self._reverb_room_size,
            'damp': self._reverb_damp,
            'width': self._reverb_width,
            'level': self._reverb_level,
        }
