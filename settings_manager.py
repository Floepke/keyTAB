from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional

from utils.CONSTANT import UTILS_SAVE_DIR
from utils.external_data_manager import ExternalDataDefinition, ExternalDataManager

PREFERENCES_PATH = Path(UTILS_SAVE_DIR) / "preferences.toml"
_PrefDef = ExternalDataDefinition


class PreferencesManager(ExternalDataManager):
    """Persist user preferences in ``preferences.toml``."""

    def __init__(self, path: Path = PREFERENCES_PATH) -> None:
        super().__init__(
            path,
            (
                "keyTAB preferences (TOML)",
                "You can edit this file to change the application preferences.",
                "Lines starting with '#' are comments. Changes take effect after restarting the app.",
            ),
        )

    def open_in_editor(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.save()
        try:
            if os.name == "nt":
                subprocess.Popen(["notepad", str(self.path)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", "-a", "TextEdit", str(self.path)])
            elif sys.platform.startswith("linux"):
                subprocess.Popen(["xdg-open", str(self.path)])
        except Exception as error:
            print(f"Failed to open preferences editor: {error}", file=sys.stderr)


_prefs_manager: Optional[PreferencesManager] = None
_active_ui_scale = 1.0


def get_ui_scale() -> float:
    """Return the active UI scale factor (1.0 = default)."""
    return _active_ui_scale


def set_ui_scale(scale: float) -> None:
    """Store the active UI scale factor before creating widgets."""
    global _active_ui_scale
    _active_ui_scale = max(0.5, min(3.0, float(scale)))


def get_preferences_manager() -> PreferencesManager:
    global _prefs_manager
    if _prefs_manager is None:
        manager = PreferencesManager()
        manager.register(
            "ui_scale", 1.0,
            "Global UI scale (0.5 .. 3.0)\n(I noticed that choosing other then 1.0 may cause some unwanted  UI artifacts)",
            min=0.5,
            max=3.0,
        )
        manager.register("theme", "light", "UI theme 'light' or 'dark'")
        manager.register("ui_language", "system", "User interface language: 'system', 'en', or 'nl'.")
        manager.register(
            "editor_fps_limit", 25,
            "The maximum frames per second (FPS) for the editor's rendering loop. Higher values may improve visual smoothness but can increase CPU/GPU usage.",
            min=1,
            max=240,
        )
        manager.register("auto_save", True, "Enable periodic automatic saving of session and project files.")
        manager.register("auto_save_interval", 1, "Autosave interval in minutes.", min=1, max=120)
        manager.register(
            "save_on_exit", True,
            "Save the current file if a file is currently open when exiting the app. You will not get the yesnocancel prompt on exit because with this option on you choose 'yes' by default.",
        )
        manager.register("play_note_on_edit", True, "Play a short note when clicking or pitch-editing notes and grace notes.")
        manager.register(
            "focus_on_playhead_during_playback", "measure",
            "Editor playhead focus mode during playback: 'measure' (jump per measure), 'animated' (smoothly keep playhead centered), or 'disabled'.",
        )
        manager.register("editor_orientation", "vertical", "Editor orientation: 'vertical' or 'horizontal'.")
        manager.register(
            "timestamp_format", "%d-%m-%Y",
            "Timestamp format for score creation and modification metadata.\nUses Python datetime.strftime notation:\n\t%d=day, \n\t%m=month, \n\t%Y=year, \n\t%H=hour(24h), \n\t%M=minute, \n\t%S=second.\nExamples: \n\t'%d-%m-%Y' becomes \n\t'%Y-%m-%d %H:%M:%S'\nUse '%%' for a literal percent sign.",
        )
        manager.register("show_tooltips", True, "Show tooltips throughout the application.")
        manager.load()
        _prefs_manager = manager
    return _prefs_manager


def get_preferences() -> Dict:
    return get_preferences_manager()._values


def open_preferences(parent=None) -> None:
    try:
        from ui.dialogs.preferences_dialog import PreferencesDialog
        PreferencesDialog(parent=parent).show()
    except Exception:
        get_preferences_manager().open_in_editor()