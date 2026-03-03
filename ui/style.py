from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication
from settings_manager import get_preferences


class Style:
    """Simple theme loader: picks light or dark from settings, nothing else."""

    # Base palettes (hex strings) that define the themes.
    _LIGHT = {
        "bg_color": "#E9E6E3",
        "alternate_background_color": "#E9E6E3",
        "text_color": "#000000",
        "accent_color": "#0078d7",
        "paper_color": "#F8F7F6",
        "notation_color": "#030500",
        "left_midi_color": "#99b3cc",
        "right_midi_color": "#ccb399",
    }

    _DARK = {
        "bg_color": "#1e1e28",
        "alternate_background_color": "#14141e",
        "text_color": "#E9E7E4",
        "accent_color": "#056098",
        "paper_color": "#14141e",
        "notation_color": "#A0B3AC",
        "left_midi_color": "#5082a0",
        "right_midi_color": "#a07d5a",
    }

    # Palette role → dict key mapping using the theme colors above.
    _ROLE_MAP = {
        QPalette.Window: "bg_color",
        QPalette.WindowText: "text_color",
        QPalette.Base: "alternate_background_color",
        QPalette.AlternateBase: "alternate_background_color",
        QPalette.ToolTipBase: "bg_color",
        QPalette.ToolTipText: "text_color",
        QPalette.Text: "text_color",
        QPalette.Button: "alternate_background_color",
        QPalette.ButtonText: "text_color",
        QPalette.BrightText: "text_color",
        QPalette.Link: "text_color",
        QPalette.Highlight: "accent_color",
        QPalette.HighlightedText: "text_color",
    }

    def __init__(self):
        theme = self._get_pref_theme()
        self.set_theme(theme)

    # Named custom colors registry
    _NAMED: dict[str, tuple[int, int, int]] = {
        'draw_util': (255, 255, 255),
        'editor': (255, 255, 255),
        'bg': (255, 245, 245),
        'text': (0, 0, 0),
        'alternate_background_color': (255, 245, 245),
        'accent': (0, 120, 215),
        'paper': (255, 255, 255),
        'notation': (3, 5, 0),
        'midi_left': (153, 179, 204),
        'midi_right': (204, 179, 153),
    }
    _active_theme = None

    @staticmethod
    def _hex_to_rgb(val: str) -> tuple[int, int, int]:
        txt = str(val or '').strip()
        if txt.startswith('#'):
            txt = txt[1:]
        if len(txt) == 3:
            txt = ''.join(ch * 2 for ch in txt)
        if len(txt) != 6 or not all(c in '0123456789abcdefABCDEF' for c in txt):
            return (255, 255, 255)
        r = int(txt[0:2], 16)
        g = int(txt[2:4], 16)
        b = int(txt[4:6], 16)
        return (r, g, b)

    @classmethod
    def _as_rgb(cls, val) -> tuple[int, int, int]:
        if isinstance(val, (list, tuple)) and len(val) >= 3:
            return (int(val[0]), int(val[1]), int(val[2]))
        return cls._hex_to_rgb(val)

    @staticmethod
    def _get_pref_theme() -> str:
        try:
            prefs = get_preferences()
            return str(prefs.get('theme', 'light')).lower()
        except Exception:
            return 'light'

    @classmethod
    def _palette_for_theme(cls, theme: str):
        return cls._DARK if str(theme).lower() == 'dark' else cls._LIGHT

    @classmethod
    def _update_named_from_palette(cls, palette) -> None:
        def _set_named(key: str, rgb_key: str) -> None:
            raw = palette.get(rgb_key, (255, 255, 255))
            rgb = cls._as_rgb(raw)
            cls._NAMED[key] = (int(rgb[0]), int(rgb[1]), int(rgb[2]))

        _set_named('bg', 'bg_color')
        _set_named('text', 'text_color')
        _set_named('alternate_background_color', 'alternate_background_color')
        _set_named('accent', 'accent_color')
        _set_named('paper', 'paper_color')
        _set_named('notation', 'notation_color')
        _set_named('midi_left', 'left_midi_color')
        _set_named('midi_right', 'right_midi_color')
        cls._NAMED['draw_util'] = (255, 255, 255)
        cls._NAMED['editor'] = cls._NAMED['paper']

    @classmethod
    def _refresh_named_from_preferences(cls) -> None:
        theme = cls._active_theme or cls._get_pref_theme()
        palette = cls._palette_for_theme(theme)
        cls._update_named_from_palette(palette)

    @classmethod
    def set_named_color(cls, name: str, rgb: tuple[int, int, int]) -> None:
        cls._NAMED[name] = tuple(int(max(0, min(255, c))) for c in rgb)

    @classmethod
    def get_named_qcolor(cls, name: str, fallback: tuple[int, int, int] = (240, 240, 240)) -> QColor:
        cls._refresh_named_from_preferences()
        rgb = cls._NAMED.get(name, fallback)
        return QColor(*rgb)

    @classmethod
    def get_named_rgb(cls, name: str, fallback: tuple[int, int, int] = (240, 240, 240)) -> tuple[int, int, int]:
        cls._refresh_named_from_preferences()
        return cls._NAMED.get(name, fallback)

    def _sync_editor_named_color(self) -> None:
        rgb = Style._NAMED.get('paper', (255, 255, 255))
        self.editor_background_color = rgb
        Style._NAMED['editor'] = tuple(int(c) for c in rgb)

    def set_theme(self, theme: str):
        palette_data = self._palette_for_theme(theme)
        colors = {k: QColor(*self._as_rgb(rgb)) for k, rgb in palette_data.items()}
        Style._active_theme = str(theme).lower()
        self._update_named_from_palette(palette_data)
        self._apply_palette(colors)
        self._sync_editor_named_color()

    def _apply_palette(self, colors_by_key):
        # Apply palette to the current QApplication instance
        app = QApplication.instance()
        if app is None:
            return
        # Use Fusion style across all platforms for consistent look
        try:
            QApplication.setStyle('Fusion')
        except Exception:
            pass
        try:
            QApplication.setEffectEnabled(Qt.UI_AnimateMenu, False)
            QApplication.setEffectEnabled(Qt.UI_FadeMenu, False)
        except Exception:
            pass
        # Clear any previous global stylesheet; we may set a minimal one below
        app.setStyleSheet("")

        pal = QPalette()
        for role, key in self._ROLE_MAP.items():
            pal.setColor(role, colors_by_key[key])

        app.setPalette(pal)

        # Remove menu highlight while keeping text readable
        bg = colors_by_key["bg_color"]
        text = colors_by_key["text_color"]
        app.setStyleSheet(
            "QMenuBar {"
            "padding: 0px;"
            "}"
            "QMenuBar::item {"
            "padding: 4px 8px;"
            "margin: 0px;"
            "border: 0px;"
            "}"
            "QMenu::item:selected {"
            f"background-color: rgb({bg.red()},{bg.green()},{bg.blue()});"
            f"color: rgb({text.red()},{text.green()},{text.blue()});"
            "}"
            "QMenuBar::item:selected {"
            f"background-color: rgb({bg.red()},{bg.green()},{bg.blue()});"
            f"color: rgb({text.red()},{text.green()},{text.blue()});"
            "padding: 4px 8px;"
            "margin: 0px;"
            "border: 0px;"
            "}"
            "QMenuBar::item:pressed {"
            f"background-color: rgb({bg.red()},{bg.green()},{bg.blue()});"
            f"color: rgb({text.red()},{text.green()},{text.blue()});"
            "}"
            "QMenu::item:disabled {"
            "background-color: transparent;"
            "}"
        )

    def set_light_theme(self):
        self.set_theme('light')

    def set_dark_theme(self):
        self.set_theme('dark')

    @classmethod
    def get_notation_color(cls) -> tuple[int, int, int]:
        cls._refresh_named_from_preferences()
        return cls._NAMED.get('notation', (0, 0, 14))

    @classmethod
    def get_editor_background_color(cls) -> tuple[int, int, int]:
        """Get the appropriate editor background color based on current theme."""
        cls._refresh_named_from_preferences()
        return cls._NAMED.get('paper', (255, 255, 255))