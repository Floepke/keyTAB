from __future__ import annotations

from typing import Any

import mido
from PySide6 import QtCore


class MidiInputManager(QtCore.QObject):
    """Listen to all available MIDI input ports and emit note events on the Qt thread."""

    note_on_received = QtCore.Signal(int, int, str)
    note_off_received = QtCore.Signal(int, int, str)
    listening_changed = QtCore.Signal(bool)

    def __init__(self, parent: QtCore.QObject | None = None, poll_interval_ms: int = 2000) -> None:
        super().__init__(parent)
        self._enabled: bool = False
        self._inputs: dict[str, Any] = {}
        self._backend = self._resolve_backend()

        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(max(500, int(poll_interval_ms)))
        self._poll_timer.timeout.connect(self.refresh_ports)

    def _resolve_backend(self):
        try:
            return mido.Backend("mido.backends.rtmidi")
        except Exception:
            return mido

    def is_enabled(self) -> bool:
        return bool(self._enabled)

    def set_enabled(self, enabled: bool) -> None:
        enabled = bool(enabled)
        if self._enabled == enabled:
            return
        self._enabled = enabled
        if self._enabled:
            self.refresh_ports()
            self._poll_timer.start()
        else:
            self._poll_timer.stop()
            self._close_all_inputs()
        self.listening_changed.emit(bool(self._enabled))

    def shutdown(self) -> None:
        self.set_enabled(False)

    def _available_input_names(self) -> list[str]:
        try:
            names = list(self._backend.get_input_names() or [])
        except Exception:
            names = []
        return [str(n) for n in names if str(n).strip()]

    def refresh_ports(self) -> None:
        if not self._enabled:
            return
        available = set(self._available_input_names())

        # Close disconnected ports.
        for name in list(self._inputs.keys()):
            if name in available:
                continue
            port = self._inputs.pop(name, None)
            if port is not None:
                try:
                    port.close()
                except Exception:
                    pass

        # Open newly connected ports.
        for name in sorted(available):
            if name in self._inputs:
                continue
            try:
                port = self._backend.open_input(name, callback=lambda msg, n=name: self._on_message(msg, n))
                self._inputs[name] = port
            except Exception:
                continue

    def _close_all_inputs(self) -> None:
        for _name, port in list(self._inputs.items()):
            try:
                port.close()
            except Exception:
                pass
        self._inputs.clear()

    def _on_message(self, msg, port_name: str) -> None:
        try:
            msg_type = str(getattr(msg, "type", "") or "")
            midi_note = int(getattr(msg, "note", -1) or -1)
            velocity = int(getattr(msg, "velocity", 0) or 0)
        except Exception:
            return
        if midi_note < 0:
            return

        if msg_type == "note_on":
            if velocity > 0:
                self.note_on_received.emit(midi_note, velocity, str(port_name))
            else:
                self.note_off_received.emit(midi_note, 0, str(port_name))
        elif msg_type == "note_off":
            self.note_off_received.emit(midi_note, velocity, str(port_name))
