from __future__ import annotations

from pathlib import Path
from typing import Optional

from utils.CONSTANT import UTILS_SAVE_DIR
from utils.external_data_manager import ExternalDataDefinition, ExternalDataManager
from version import __version__

APPDATA_PATH = Path(UTILS_SAVE_DIR) / "appdata.toml"
_DataDef = ExternalDataDefinition


class AppDataManager(ExternalDataManager):
    """Persist runtime-managed application data in ``appdata.toml``."""

    def __init__(self, path: Path = APPDATA_PATH) -> None:
        super().__init__(
            path,
            (
                "keyTAB app data (TOML)",
                "Application-managed data. Editing is possible but not generally required.",
            ),
        )


_appdata_manager: Optional[AppDataManager] = None


def get_appdata_manager() -> AppDataManager:
    global _appdata_manager
    if _appdata_manager is None:
        manager = AppDataManager()
        manager.register("recent_files", [], "List of recently opened files (most recent first)")
        manager.register("last_opened_file", "", "Absolute path to the last opened/saved project file")
        manager.register("last_file_dialog_dir", "", "Last directory used in file open/save dialogs")
        manager.register("snap_base", 8, "Last selected snap base (1,2,4,8,...) for editor snapping")
        manager.register("snap_divide", 1, "Last selected snap divide (tuplets factor) for editor snapping")
        manager.register("selected_tool", "note", "Last selected tool name in the tool selector")
        manager.register("editor_scroll_pos", 0, "Last editor scroll position (logical px)")
        manager.register("show_install_question", True, "Ask once to install AppImage desktop integration on Linux")
        manager.register("app_version", __version__, "Version of keyTAB last installed for desktop integration")
        manager.register("playback_mode", "system", "Playback mode: 'system' or 'external'")
        manager.register("midi_out_port", "", "Last selected external MIDI output port name")
        manager.register("last_session_saved", False, "Whether the last session at exit was saved to a project file")
        manager.register("last_session_path", "", "Project file path associated with the last session if it was saved")
        manager.register("window_maximized", True, "Start maximized; updated on exit")
        manager.register("window_geometry", "", "Base64-encoded Qt window geometry for normal state")
        manager.register("left_panel_width_px", 220, "Last width of the left docked panel area in pixels")
        manager.register("style_dialog_geometry", "", "Base64-encoded Qt geometry for the Style dialog")
        manager.register("info_dialog_geometry", "", "Base64-encoded Qt geometry for the Info dialog")
        manager.register("line_break_dialog_geometry", "", "Base64-encoded Qt geometry for the Line Break dialog")
        manager.register("preferences_dialog_geometry", "", "Base64-encoded Qt geometry for the Preferences dialog")
        manager.register("fluidsynth_reverb_config_dialog_geometry", "", "Base64-encoded Qt geometry for the FluidSynth Reverb Config dialog")
        manager.register("dynamic_symbol_dialog_geometry", "", "Base64-encoded Qt geometry for the Dynamic Symbol dialog")
        manager.register("text_dialog_geometry", "", "Base64-encoded Qt geometry for the Text dialog")
        manager.register("time_signature_dialog_geometry", "", "Base64-encoded Qt geometry for the Time Signature dialog")
        manager.register("midi_import_dialog_geometry", "", "Base64-encoded Qt geometry for the MIDI Import dialog")
        manager.register("notehead_dialog_geometry", "", "Base64-encoded Qt geometry for the Notehead dialog")
        manager.register("score_template", {}, "Default score template for new scores (dict of score fields except events)")
        manager.register("fonts_install_ok", False, "True when all required embedded fonts are installed to the user font directory")
        manager.register("user_soundfont_path", "", "Absolute path to last selected user soundfont (.sf2/.sf3)")
        manager.load()
        _appdata_manager = manager
    return _appdata_manager