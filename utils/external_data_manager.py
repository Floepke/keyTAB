from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

try:
    import tomllib as _tomlreader
except Exception:  # pragma: no cover
    _tomlreader = None  # type: ignore

try:
    import tomlkit as _tomlkit  # type: ignore
except Exception:  # pragma: no cover
    _tomlkit = None  # type: ignore


@dataclass
class ExternalDataDefinition:
    default: object
    description: str
    min: object | None = None
    max: object | None = None


class ExternalDataManager:
    """Register and persist TOML-backed external application data."""

    def __init__(self, path: Path, header: tuple[str, ...]) -> None:
        self.path = path
        self._header = header
        self._schema: Dict[str, ExternalDataDefinition] = {}
        self._values: Dict[str, object] = {}
        self._doc = None

    def register(
        self,
        key: str,
        default: object,
        description: str,
        min: object | None = None,
        max: object | None = None,
    ) -> None:
        self._schema[key] = ExternalDataDefinition(default, description, min, max)
        self._values.setdefault(key, default)

    def iter_schema(self) -> list[tuple[str, ExternalDataDefinition]]:
        return list(self._schema.items())

    def get(self, key: str, default: Optional[object] = None) -> object:
        return self._values.get(key, default)

    def set(self, key: str, value: object) -> None:
        self._values[key] = value

    def remove(self, key: str) -> None:
        self._values.pop(key, None)
        if self._doc is not None and key in self._doc:
            del self._doc[key]

    def load(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parsed: Dict[str, object] = {}
        changed = False

        if self.path.exists():
            try:
                text = self.path.read_text(encoding="utf-8")
            except Exception:
                text = ""
            parsed = self._parse_toml_dict()
            if _tomlkit is not None and text:
                try:
                    self._doc = _tomlkit.parse(text)
                except Exception:
                    self._doc = None
        else:
            self.save()

        for key, definition in self._schema.items():
            if key in parsed:
                self._values[key] = self._coerce(parsed[key], definition.default)
            else:
                self._values.setdefault(key, definition.default)
                changed = True
        for key, value in parsed.items():
            self._values.setdefault(key, value)

        if changed:
            self.save()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if _tomlkit is not None and self._doc is not None:
            try:
                for key, value in self._values.items():
                    self._doc[key] = _tomlkit.item(value)
                self._atomic_write(_tomlkit.dumps(self._doc))
                return
            except Exception:
                pass
        self._atomic_write(self._emit_toml_file())

    def _parse_toml_dict(self) -> Dict[str, object]:
        try:
            if _tomlreader is not None:
                with self.path.open("rb") as file:
                    return dict(_tomlreader.load(file) or {})
            try:
                import tomli as _tomli  # type: ignore
            except Exception:
                return {}
            with self.path.open("rb") as file:
                return dict(_tomli.load(file) or {})
        except Exception:
            return {}

    @staticmethod
    def _coerce(value: object, default: object) -> object:
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

    def _emit_toml_file(self) -> str:
        lines = [f"# {line}\n" for line in self._header]
        order = list(self._schema) + [key for key in self._values if key not in self._schema]
        for key in dict.fromkeys(order):
            definition = self._schema.get(key)
            if definition is not None:
                lines.extend(f"# {line}\n" for line in definition.description.splitlines())
            lines.append(f"{key} = {self._format_toml_value(self._values.get(key))}\n")
        return "\n".join(lines)

    def _atomic_write(self, content: str) -> None:
        temp_path: Optional[str] = None
        try:
            fd, temp_path = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=str(self.path.parent)
            )
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temp_path, self.path)
            temp_path = None
            if os.name != "nt":
                directory_fd = os.open(str(self.path.parent), getattr(os, "O_DIRECTORY", 0))
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            if temp_path is not None:
                try:
                    os.remove(temp_path)
                except FileNotFoundError:
                    pass

    def _format_toml_value(self, value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            escaped = value.replace("\\", "\\\\").replace('"', '\\"')
            return f'"{escaped}"'
        if isinstance(value, list):
            if not value:
                return "[]"
            if len(value) <= 6 and all(isinstance(item, (int, float, bool, str)) for item in value):
                return "[" + ", ".join(self._format_toml_value(item) for item in value) + "]"
            body = ",\n".join("    " + self._format_toml_value(item) for item in value)
            return "[\n" + body + "\n]"
        if isinstance(value, dict):
            if not value:
                return "{}"
            items = ", ".join(f"{key} = {self._format_toml_value(item)}" for key, item in value.items())
            return "{ " + items + " }"
        escaped = repr(value).replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'