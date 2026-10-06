from __future__ import annotations

import base64
import sys
import ctypes
from typing import Optional

try:
    from PySide6.QtCore import QByteArray
    from PySide6.QtGui import QFontDatabase, QFont
    from PySide6.QtWidgets import QApplication
except Exception:
    QByteArray = None
    QFontDatabase = None
    QFont = None
    QApplication = None

# Lazy import of generated base64 mapping
try:
    from .fonts_byte64 import FONTS, FONT_ALIASES, FONT_GROUPS  # type: ignore
except Exception:
    FONTS = {}
    FONT_ALIASES = {}
    FONT_GROUPS = {}


_EMBEDDED_FONT_NAMES: set[str] = set()
_REGISTERED_FONT_CACHE: dict[str, Optional[str]] = {}
_WINDOWS_MEMORY_FONT_HANDLES: dict[str, int] = {}
_WINDOWS_MEMORY_FONT_BUFFERS: dict[str, ctypes.Array] = {}


def _font_members(name: str) -> list[str]:
    group = FONT_GROUPS.get(name)
    if group:
        return [str(member) for member in group]
    alias = FONT_ALIASES.get(name)
    if alias:
        return [str(alias)]
    return [str(name)]


def _normalize_font_name(name: str) -> str:
    return str(name or '').strip().lower()


def _decoded_font_bytes(name: str) -> Optional[bytes]:
    b64 = FONTS.get(name)
    if not b64:
        return None
    try:
        return base64.b64decode(b64)
    except Exception:
        return None


def _register_windows_memory_fonts(name: str) -> bool:
    """Expose embedded fonts to Cairo's Win32 backend without installing files."""
    if not sys.platform.startswith("win"):
        return True
    try:
        add_font = ctypes.windll.gdi32.AddFontMemResourceEx
        add_font.argtypes = (ctypes.c_void_p, ctypes.c_uint32, ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32))
        add_font.restype = ctypes.c_void_p
    except (AttributeError, OSError):
        return False

    registered = True
    for member in _font_members(name):
        if member in _WINDOWS_MEMORY_FONT_HANDLES:
            continue
        raw = _decoded_font_bytes(member)
        if raw is None:
            registered = False
            continue
        buffer = ctypes.create_string_buffer(raw)
        count = ctypes.c_uint32()
        handle = add_font(buffer, len(raw), None, ctypes.byref(count))
        if not handle or count.value == 0:
            registered = False
            continue
        _WINDOWS_MEMORY_FONT_BUFFERS[member] = buffer
        _WINDOWS_MEMORY_FONT_HANDLES[member] = int(handle)
    return registered


def register_embedded_font_with_cairo(name: str) -> bool:
    """Make an embedded font available to Cairo without writing it to disk."""
    return _register_windows_memory_fonts(name)


def register_font_from_bytes(name: str) -> Optional[str]:
    """Register the embedded font by `name` and return the primary family name.

    Returns None if registration fails or PySide6 is unavailable.

    On Linux, LelandText is treated as optional because the native Qt/Cairo font
    backend can crash while resolving it when the font is missing or not
    installed. The editor handles missing text by skipping that symbol instead of
    crashing the application.
    """
    if QFontDatabase is None:
        return None
    cache_key = _normalize_font_name(name)
    if sys.platform.startswith('linux') and cache_key == 'lelandtext':
        return None
    _register_windows_memory_fonts(name)
    if cache_key in _REGISTERED_FONT_CACHE:
        return _REGISTERED_FONT_CACHE[cache_key]
    try:
        resolved: Optional[str] = None
        members = _font_members(name)
        for member in members:
            raw = _decoded_font_bytes(member)
            if raw is None:
                continue
            if QByteArray is not None:
                data = QByteArray(raw)
            else:
                data = raw
            fid = QFontDatabase.addApplicationFontFromData(data)
            if fid < 0:
                continue
            fams = [str(f) for f in QFontDatabase.applicationFontFamilies(fid)]
            _EMBEDDED_FONT_NAMES.add(_normalize_font_name(member))
            for fam in fams:
                _EMBEDDED_FONT_NAMES.add(_normalize_font_name(fam))
            if resolved is None and fams:
                resolved = fams[0]
        if resolved is None:
            _REGISTERED_FONT_CACHE[cache_key] = None
            return None
        _EMBEDDED_FONT_NAMES.add(cache_key)
        _REGISTERED_FONT_CACHE[cache_key] = resolved
        return resolved
    except Exception:
        _REGISTERED_FONT_CACHE[cache_key] = None
        return None


def resolve_font_family(family: str, fallback_family: str = 'Edwin') -> str:
    """Resolve a usable font family name.

    - Prefer the requested system font if available.
    - Otherwise, register the embedded fallback font and use it if available.
    - As a last resort, return the original family string.
    """
    if QFontDatabase is None or QApplication is None or QApplication.instance() is None:
        return family
    try:
        families = set(QFontDatabase.families())
        if family in families:
            return family
    except Exception:
        pass
    try:
        fallback = register_font_from_bytes(fallback_family)
        if fallback:
            return fallback
    except Exception:
        pass
    return family


def install_default_ui_font(app: Optional[QApplication] = None, name: str = 'FiraCode-SemiBold', point_size: int = 11) -> bool:
    """Install the embedded font and set it as the QApplication default.

    - Tries to register the font from embedded base64 (fonts_byte64.py).
    - If embedded font is missing, tries to use system-installed font by name.
    - Returns True if the app font was set; False otherwise.
    """
    if QApplication is None:
        return False
    if app is None:
        app = QApplication.instance()
    if app is None:
        return False

    family = register_font_from_bytes(name)

    # Try a list of likely family names/aliases for Fira Code
    candidates = []
    if family:
        candidates.append(family)
    candidates.extend([
        name,
        'Fira Code',
        'FiraCode',
        'Fira Code SemiBold',
        'FiraCode-SemiBold',
    ])

    try:
        for fam in candidates:
            if not fam:
                continue
            f = QFont(str(fam), point_size)
            # Prefer semi-bold weight when available
            try:
                f.setWeight(QFont.Weight.DemiBold)
            except Exception:
                pass
            if f and f.family():
                app.setFont(f)
                return True
        # Last resort: let Qt pick default font with size
        f = QFont()
        f.setPointSize(point_size)
        app.setFont(f)
    except Exception:
        return False
    return False
