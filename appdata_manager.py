from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from utils.CONSTANT import UTILS_SAVE_DIR

try:
    import tomllib as _tomlreader  # Python 3.11+
except Exception:  # pragma: no cover
    _tomlreader = None  # type: ignore

try:
    import tomlkit as _tomlkit  # type: ignore
except Exception:  # pragma: no cover
    _tomlkit = None  # type: ignore

# New TOML-based appdata file
APPDATA_PATH: Path = Path(UTILS_SAVE_DIR) / "appdata.toml"
# Legacy Python-based appdata file for migration
LEGACY_APPDATA_PATH: Path = Path(UTILS_SAVE_DIR) / "appdata.py"


def _ensure_dir() -> None:
    os.makedirs(UTILS_SAVE_DIR, exist_ok=True)


@dataclass
class _DataDef:
    default: object
    description: str


class AppDataManager:
    """Register and persist application data (non-preferences) in ~/ .keyTAB/appdata.toml.

    This is for runtime-managed data such as recent files, last session info, etc.
    TOML supports comments and is end-user friendly for occasional edits.
    """

    def __init__(self, path: Path = APPDATA_PATH) -> None:
        self.path = path
        self._schema: Dict[str, _DataDef] = {}
        self._values: Dict[str, object] = {}
        # Parsed TOML document (when tomlkit is available) for round-trip preservation
        self._doc = None

    def register(self, key: str, default: object, description: str) -> None:
        self._schema[key] = _DataDef(default=default, description=description)
        if key not in self._values:
            self._values[key] = default

    def get(self, key: str, default: Optional[object] = None) -> object:
        return self._values.get(key, default)

    def set(self, key: str, value: object) -> None:
        self._values[key] = value

    def remove(self, key: str) -> None:
        if key in self._values:
            self._values.pop(key, None)

    def load(self) -> None:
        _ensure_dir()
        parsed: Dict[str, object] = {}
        changed: bool = False
        if self.path.exists():
            # Load raw text and dict
            try:
                text = self.path.read_text(encoding="utf-8")
            except Exception:
                text = ""
            parsed = self._parse_toml_dict(self.path)
            # Keep tomlkit doc if available
            try:
                if _tomlkit is not None and text:
                    self._doc = _tomlkit.parse(text)
            except Exception:
                self._doc = None
        elif LEGACY_APPDATA_PATH.exists():
            # Migrate from legacy Python appdata file
            legacy_text = LEGACY_APPDATA_PATH.read_text(encoding="utf-8")
            parsed = self._parse_py_dict(legacy_text)
            # Seed values from parsed and schema defaults, then save to TOML
            try:
                self._values = {}
                for k, d in self._schema.items():
                    if k in parsed:
                        self._values[k] = parsed[k]
                    else:
                        self._values.setdefault(k, d.default)
                for k, v in parsed.items():
                    if k not in self._values:
                        self._values[k] = v
                if _tomlkit is not None:
                    try:
                        doc = _tomlkit.document()
                        for k in self._values:
                            doc.add(k, _tomlkit.item(self._values[k]))
                        self._doc = doc
                    except Exception:
                        self._doc = None
                self.save()
            except Exception:
                pass
        else:
            # Initialize defaults and write TOML file
            self.save()
            parsed = {}

        # Merge loaded values into schema defaults
        for k, d in self._schema.items():
            if k in parsed:
                self._values[k] = parsed[k]
            else:
                self._values.setdefault(k, d.default)
                changed = True
        for k, v in parsed.items():
            if k not in self._values:
                self._values[k] = v

        # Persist restored defaults if any schema keys were missing
        if changed:
            self.save()

    def save(self) -> None:
        _ensure_dir()
        # Prefer round-trip preservation when tomlkit is available and we have a document
        if _tomlkit is not None and self._doc is not None:
            try:
                for k, v in self._values.items():
                    try:
                        self._doc[k] = _tomlkit.item(v)
                    except Exception:
                        if k not in self._doc:
                            try:
                                self._doc.add(k, _tomlkit.item(v))
                            except Exception:
                                pass
                content = _tomlkit.dumps(self._doc)
                self.path.write_text(content, encoding="utf-8")
                return
            except Exception:
                pass
        # Fallback to minimal emitter
        content = self._emit_toml_file(self._values)
        self.path.write_text(content, encoding="utf-8")

    # Internals
    def _parse_toml_dict(self, path: Path) -> Dict:
        try:
            if _tomlreader is None:
                try:
                    import tomli as _tomli  # type: ignore
                except Exception:
                    return {}
                with open(path, "rb") as f:
                    return dict(_tomli.load(f) or {})
            with open(path, "rb") as f:
                data = _tomlreader.load(f)  # type: ignore
            return dict(data or {})
        except Exception:
            return {}
    def _parse_py_dict(self, text: str) -> Dict:
        import ast
        tree = ast.parse(text, filename=str(self.path), mode='exec')
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == 'appdata':
                        value = ast.literal_eval(node.value)
                        if isinstance(value, dict):
                            return value
                        return {}
        return {}

    def _emit_toml_file(self, values: Dict[str, object]) -> str:
        lines: list[str] = []
        lines.append("# PianoScript app data (TOML)\n")
        lines.append("# Application-managed data. Editing is possible but not generally required.\n")
        order = list(self._schema.keys()) + [k for k in values.keys() if k not in self._schema]
        seen: set[str] = set()
        for k in order:
            if k in seen:
                continue
            seen.add(k)
            desc = self._schema.get(k).description if k in self._schema else ""
            if desc:
                for dline in desc.splitlines():
                    lines.append(f"# {dline}")
            v = values.get(k)
            lines.append(f"{k} = {self._format_toml_value(v)}\n")
        return "\n".join(lines)

    def _format_toml_value(self, v: object) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, str):
            s = v.replace("\\", "\\\\").replace("\"", "\\\"")
            return f'"{s}"'
        if isinstance(v, list):
            if not v:
                return "[]"
            if len(v) <= 6 and all(isinstance(x, (int, float, bool, str)) for x in v):
                return "[" + ", ".join(self._format_toml_value(x) for x in v) + "]"
            inner = ",\n".join("    " + self._format_toml_value(x) for x in v)
            return "[\n" + inner + "\n]"
        if isinstance(v, dict):
            if not v:
                return "{}"
            items = ", ".join(f"{kk} = {self._format_toml_value(v[kk])}" for kk in v)
            return "{ " + items + " }"
        s = repr(v).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s}"'


# ---- Registration hub (add app data keys here) ----
_appdata_manager: Optional[AppDataManager] = None


def get_appdata_manager() -> AppDataManager:
    global _appdata_manager
    if _appdata_manager is None:
        adm = AppDataManager(APPDATA_PATH)
        # Register known app data here (not user preferences)
        adm.register("recent_files", [], "List of recently opened files (most recent first)")
        adm.register("last_opened_file", "", "Absolute path to the last opened/saved project file")
        adm.register("last_file_dialog_dir", "", "Last directory used in file open/save dialogs")
        adm.register("snap_base", 8, "Last selected snap base (1,2,4,8,...) for editor snapping")
        adm.register("snap_divide", 1, "Last selected snap divide (tuplets factor) for editor snapping")
        adm.register("selected_tool", "note", "Last selected tool name in the tool selector")
        adm.register("editor_scroll_pos", 0, "Last editor scroll position (logical px)")
        adm.register(
            "show_install_question",
            True,
            "Ask once to install AppImage desktop integration on Linux",
        )
        # Playback preferences
        adm.register("playback_mode", "system", "Playback mode: 'system' or 'external'")
        adm.register("midi_out_port", "", "Last selected external MIDI output port name")
        # Session restore preferences
        adm.register("last_session_saved", False, "Whether the last session at exit was saved to a project file")
        adm.register("last_session_path", "", "Project file path associated with the last session if it was saved")
        # Synth-related appdata removed: wavetable and ADSR are no longer used
        # Window state (session-managed)
        adm.register("window_maximized", True, "Start maximized; updated on exit")
        adm.register("window_geometry", "", "Base64-encoded Qt window geometry for normal state")
        adm.register("score_template", {}, "Default score template for new scores (dict of score fields except events)")
        adm.register("edwin_font_installed", False, "True when Edwin font was installed to the user font directory")
        adm.register("edwin_install_prompt_dismissed", False, "User declined the Edwin font installation prompt")
        adm.register("lmromancaps_font_installed", False, "True when Latin Modern Roman Caps font was installed to the user font directory")
        adm.register("lmromancaps_install_prompt_dismissed", False, "User declined the Latin Modern Roman Caps font installation prompt")
        adm.register("lmroman_font_installed", False, "True when Latin Modern Roman font was installed to the user font directory")
        adm.register("lmroman_install_prompt_dismissed", False, "User declined the Latin Modern Roman font installation prompt")
        adm.register("user_soundfont_path", "", "Absolute path to last selected user soundfont (.sf2/.sf3)")
        # Removed window_state persistence to avoid saving/restoring dock/toolbar layout
        adm.load()
        # Strip any legacy keys from stored values
        try:
            adm._values.pop("window_state", None)
        except Exception:
            pass
        try:
            removed = adm._values.pop("layout_template", None)
            if removed is not None:
                adm.save()
        except Exception:
            pass
        try:
            for legacy_key in ("user_styles", "selected_style_name", "user_styles_version"):
                adm._values.pop(legacy_key, None)
        except Exception:
            pass
        _appdata_manager = adm
    return _appdata_manager
