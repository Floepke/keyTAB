from __future__ import annotations
import os
import sys
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from utils.CONSTANT import UTILS_SAVE_DIR

try:
    import tomllib as _tomlreader  # Python 3.11+
except Exception:  # pragma: no cover
    _tomlreader = None  # type: ignore

# Optional round-trip preservation library
try:
    import tomlkit as _tomlkit  # type: ignore
except Exception:  # pragma: no cover
    _tomlkit = None  # type: ignore


# Paths under the user's home folder (~/ .keyTAB)
# New TOML-based preferences file (supports comments with '#').
PREFERENCES_PATH: Path = Path(UTILS_SAVE_DIR) / "preferences.toml"
# Legacy Python-based preferences file retained for one-time migration.
LEGACY_PREFERENCES_PATH: Path = Path(UTILS_SAVE_DIR) / "preferences.py"


def _ensure_dir() -> None:
    os.makedirs(UTILS_SAVE_DIR, exist_ok=True)


@dataclass
class _PrefDef:
    default: object
    description: str
    min: object | None = None
    max: object | None = None


class PreferencesManager:
    """Register and persist application preferences in ~/ .keyTAB/preferences.toml.

    Users can edit this file; changes take effect after restarting the app.
    The TOML format allows comments (lines starting with '#') and is human-friendly.
    """

    def __init__(self, path: Path = PREFERENCES_PATH) -> None:
        self.path = path
        self._schema: Dict[str, _PrefDef] = {}
        self._values: Dict[str, object] = {}
        # Keep parsed TOML document for round-trip preservation when tomlkit is available
        self._doc = None

    def register(
        self,
        key: str,
        default: object,
        description: str,
        min: object | None = None,
        max: object | None = None,
    ) -> None:
        self._schema[key] = _PrefDef(default=default, description=description, min=min, max=max)
        if key not in self._values:
            self._values[key] = default

    def iter_schema(self) -> list[tuple[str, _PrefDef]]:
        return list(self._schema.items())

    def get(self, key: str, default: Optional[object] = None) -> object:
        return self._values.get(key, default)

    def set(self, key: str, value: object) -> None:
        self._values[key] = value

    def load(self) -> None:
        _ensure_dir()
        parsed: Dict[str, object] = {}
        changed: bool = False
        if self.path.exists():
            # Load raw text once
            try:
                text = self.path.read_text(encoding="utf-8")
            except Exception:
                text = ""
            # Parse dict using stdlib/fallback
            parsed = self._parse_toml_dict(self.path)
            # Parse document using tomlkit if available to preserve comments/formatting
            try:
                if _tomlkit is not None and text:
                    self._doc = _tomlkit.parse(text)
            except Exception:
                self._doc = None
        elif LEGACY_PREFERENCES_PATH.exists():
            # One-time migration from legacy Python file
            legacy_text = LEGACY_PREFERENCES_PATH.read_text(encoding="utf-8")
            parsed = self._parse_py_dict(legacy_text)
            # Immediately save to TOML for future runs
            try:
                self._values = {}
                for k, d in self._schema.items():
                    if k in parsed:
                        self._values[k] = self._coerce(parsed[k], d.default)
                    else:
                        self._values.setdefault(k, d.default)
                for k, v in parsed.items():
                    if k not in self._values:
                        self._values[k] = v
                # Build initial tomlkit document if available
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
            # Initialize defaults and write file
            self.save()
            parsed = {}

        # Merge loaded values into schema defaults
        for k, d in self._schema.items():
            if k in parsed:
                self._values[k] = self._coerce(parsed[k], d.default)
            else:
                self._values.setdefault(k, d.default)
                changed = True
        for k, v in parsed.items():
            if k not in self._values:
                self._values[k] = v

        # If any schema key was missing from the file, persist the restored defaults
        if changed:
            self.save()

    def save(self) -> None:
        _ensure_dir()
        # Prefer round-trip preservation when tomlkit is available and we have a document
        if _tomlkit is not None and self._doc is not None:
            try:
                # Update only keys present in _values; preserve unknown keys and comments
                for k, v in self._values.items():
                    try:
                        # Use tomlkit.item to wrap Python value into TOML types
                        self._doc[k] = _tomlkit.item(v)
                    except Exception:
                        # Fallback: add key if missing
                        if k not in self._doc:
                            try:
                                self._doc.add(k, _tomlkit.item(v))
                            except Exception:
                                pass
                content = _tomlkit.dumps(self._doc)
                self.path.write_text(content, encoding="utf-8")
                return
            except Exception:
                # Fall through to emitter if tomlkit fails
                pass
        # Minimal emitter fallback
        content = self._emit_toml_file(self._values)
        self.path.write_text(content, encoding="utf-8")

    def open_in_editor(self) -> None:
        _ensure_dir()
        if not self.path.exists():
            self.save()
        try:
            fpath = str(self.path)
            if os.name == "nt":
                # Always use Notepad on Windows
                subprocess.Popen(["notepad", fpath])
                return
            if sys.platform == "darwin":
                # Always use TextEdit on macOS
                subprocess.Popen(["open", "-a", "TextEdit", fpath])
                return
            if sys.platform.startswith("linux"):
                # Always and only use xdg-open on Linux
                subprocess.Popen(["xdg-open", fpath])
                return
        except Exception as e:
            # Last-resort: log to stderr to avoid crashing UI
            try:
                import sys as _sys
                print(f"Failed to open preferences editor: {e}", file=_sys.stderr)
            except Exception:
                pass

    # Internals
    def _parse_toml_dict(self, path: Path) -> Dict:
        try:
            if _tomlreader is None:
                # Attempt optional fallback to tomli if available
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
                    if isinstance(target, ast.Name) and target.id == 'preferences':
                        value = ast.literal_eval(node.value)
                        if isinstance(value, dict):
                            return value
                        return {}
        return {}

    def _coerce(self, value: object, default: object) -> object:
        if isinstance(default, bool):
            return bool(value)
        if isinstance(default, int):
            try:
                return int(value)
            except Exception:
                return default
        if isinstance(default, float):
            try:
                return float(value)
            except Exception:
                return default
        if isinstance(default, str):
            return str(value)
        return value

    def _emit_toml_file(self, values: Dict[str, object]) -> str:
        lines: list[str] = []
        lines.append("# PianoScript preferences (TOML)\n")
        lines.append("# You can edit this file to change the application preferences.")
        lines.append("# Lines starting with '#' are comments. Changes take effect after restarting the app.\n")
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
            # Bare keys are fine (alnum + underscore). Values are TOML literals.
            lines.append(f"{k} = {self._format_toml_value(v)}\n")
        return "\n".join(lines)

    def _format_toml_value(self, v: object) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return str(v)
        if isinstance(v, str):
            # Escape backslashes and quotes minimally
            s = v.replace("\\", "\\\\").replace("\"", "\\\"")
            return f'"{s}"'
        if isinstance(v, list):
            if not v:
                return "[]"
            # Single-line small arrays
            if len(v) <= 4 and all(isinstance(x, (int, float, bool, str)) for x in v):
                return "[" + ", ".join(self._format_toml_value(x) for x in v) + "]"
            # Multi-line arrays
            inner = ",\n".join("    " + self._format_toml_value(x) for x in v)
            return "[\n" + inner + "\n]"
        if isinstance(v, dict):
            if not v:
                return "{}"  # represent empty inline table
            # Inline table for small dicts; multi-line inline for larger
            items = ", ".join(f"{kk} = {self._format_toml_value(v[kk])}" for kk in v)
            return "{ " + items + " }"
        # Fallback to repr inside a string to keep TOML valid
        s = repr(v).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s}"'


# ---- Registration hub (add new app preferences here) ----
_prefs_manager: Optional[PreferencesManager] = None


def get_preferences_manager() -> PreferencesManager:
    global _prefs_manager
    if _prefs_manager is None:
        pm = PreferencesManager(PREFERENCES_PATH)
        # Register known preferences here
        pm.register(
            key="ui_scale",
            default=1.0,
            description="Global UI scale (0.5 .. 3.0)\n(I noticed that choosing other then 1.0 may cause some unwanted  UI artifacts)",
            min=0.5,
            max=3.0,
        )
        pm.register(
            key="theme",
            default="light",
            description="UI theme 'light' or 'dark'",
        )
        pm.register(
            key="editor_fps_limit",
            default=25,
            description="The maximum frames per second (FPS) for the editor's rendering loop. Higher values may improve visual smoothness but can increase CPU/GPU usage.",
            min=1,
            max=240,
        )
        pm.register(
            key="audition_during_note_input",
            default=True,
            description="Play a short note on input when placing notes.",
        )
        pm.register(
            key="focus_on_playhead_during_playback",
            default=True,
            description="Focus the editor view on the playhead during playback.",
        )
        pm.register(
            key="timestamp_format",
            default="%d-%m-%Y",
            description=(
                "Timestamp format for score creation and modification metadata.\n"
                "Uses Python datetime.strftime notation:\n"
                "\t•%d=day, \n\t•%m=month, \n\t•%Y=year, \n\t•%H=hour(24h), \n\t•%M=minute, \n\t•%S=second.\n"
                "Examples: \n\t'%d-%m-%Y' becomes \n\t'%Y-%m-%d %H:%M:%S'\nUse '%%' for a literal percent sign."
            ),
        )
        pm.load()
        _prefs_manager = pm
    return _prefs_manager


def get_preferences() -> Dict:
    return get_preferences_manager()._values


def open_preferences(parent=None) -> None:
    try:
        from ui.widgets.preferences_dialog import PreferencesDialog
        dlg = PreferencesDialog(parent=parent)
        dlg.exec()
    except Exception:
        get_preferences_manager().open_in_editor()
