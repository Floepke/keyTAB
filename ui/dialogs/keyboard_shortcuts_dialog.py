"""Keyboard Shortcuts Card Dialog - displays all application shortcuts beautifully."""
from __future__ import annotations

import sys
from PySide6 import QtCore, QtGui, QtWidgets
from icons.icons import get_qicon
from settings_manager import get_preferences_manager
from ui.style import Style


class KeyboardShortcutsDialog(QtWidgets.QDialog):
    """Shows a beautifully formatted card with all keyboard shortcuts as a splash screen."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        
        # Configure as frameless splash screen
        self.setWindowFlags(
            QtCore.Qt.WindowType.Dialog 
            | QtCore.Qt.WindowType.FramelessWindowHint
            | QtCore.Qt.WindowType.WindowStaysOnTopHint
        )
        self.setModal(True)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        
        # Set window size
        self.setMinimumWidth(900)
        self.setMinimumHeight(700)
        
        # Determine platform for shortcut display
        self._is_macos = sys.platform == "darwin"
        self._is_windows = sys.platform == "win32"
        
        # Main container
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # Create main card widget with background
        card_widget = QtWidgets.QWidget()
        card_layout = QtWidgets.QVBoxLayout(card_widget)
        card_layout.setContentsMargins(40, 40, 40, 40)
        card_layout.setSpacing(20)
        
        # Title
        title_label = QtWidgets.QLabel(self.tr("Shortcut Reference Card"))
        title_font = title_label.font()
        title_font.setPointSize(28)
        title_font.setBold(True)
        title_label.setFont(title_font)
        text_color = Style.get_named_qcolor('text')
        title_label.setStyleSheet(f"color: {text_color.name()};")
        card_layout.addWidget(title_label)
        
        # Instruction
        instr_label = QtWidgets.QLabel(self.tr("Click or press any key to close.\nScroll down to see all shortcuts."))
        instr_font = instr_label.font()
        instr_font.setPointSize(10)
        instr_label.setFont(instr_font)
        instr_label.setStyleSheet(f"color: {text_color.name()}; margin-bottom: 10px;")
        card_layout.addWidget(instr_label)
        
        # Scrollable shortcuts area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        # Hide scrollbar but keep scrolling enabled
        scroll.setStyleSheet(
            "QScrollArea { border: none; background: transparent; }"
            "QScrollBar:vertical { background: transparent; width: 0px; }"
            "QScrollBar::handle:vertical { background: transparent; }"
        )
        scroll.verticalScrollBar().setFixedWidth(0)
        
        # Scrollable content
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(16)
        
        # Set background color for scroll content
        bg_color = Style.get_named_qcolor('bg')
        scroll_content.setStyleSheet(f"background-color: {bg_color.name()};")
        
        # Add categories
        shortcuts_data = self._get_shortcuts_data()
        for category_name, shortcuts in shortcuts_data:
            category_widget = self._create_category_widget(category_name, shortcuts)
            scroll_layout.addWidget(category_widget)
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        card_layout.addWidget(scroll)
        
        # Set card styling with background
        self._setup_card_background(card_widget)
        
        main_layout.addWidget(card_widget, 1, QtCore.Qt.AlignmentFlag.AlignCenter)
        self.center_on_screen()
    
    def _get_mod_key(self) -> str:
        """Get the modifier key name for the current platform."""
        if self._is_macos:
            return "Cmd"
        return "Ctrl"
    
    def _get_transpose_keys(self) -> tuple[str, str, str, str]:
        """Get transpose and move keys based on editor orientation.
        
        Returns (transpose_down, transpose_up, move_earlier, move_later)
        """
        is_horizontal = str(get_preferences_manager().get('editor_orientation', 'vertical') or 'vertical').strip().lower() == 'horizontal'
        if is_horizontal:
            return ("↓", "↑", "←", "→")
        else:
            return ("←", "→", "↑", "↓")
    
    def _get_shortcuts_data(self) -> list[tuple[str, list[tuple[str, str]]]]:
        """Return organized shortcuts data with single-key shortcuts where applicable."""
        shift = "Shift"
        transpose_down, transpose_up, move_earlier, move_later = self._get_transpose_keys()
        
        return [
            (self.tr("File Operations"), [
                (self.tr("New Project"), "N"),
                (self.tr("Open Project"), "O"),
                (self.tr("Save Project"), "Ctrl+S"),
                (self.tr("Export PDF"), "E"),
                (self.tr("Exit Application"), "Esc"),
            ]),
            (self.tr("Edit Operations"), [
                (self.tr("Undo"), "Z"),
                (self.tr("Redo"), "Y"),
                (self.tr("Cut"), "X"),
                (self.tr("Copy"), "C"),
                (self.tr("Paste"), "V"),
                (self.tr("Delete"), "Delete / Backspace"),
            ]),
            (self.tr("Selection & Editing"), [
                (self.tr("Select All"), "A"),
                (self.tr("Transpose Down"), transpose_down),
                (self.tr("Transpose Up"), transpose_up),
                (self.tr("Move Earlier"), move_earlier),
                (self.tr("Move Later"), move_later),
                (self.tr("Quantize"), "Q"),
                (self.tr("Map Selected Notes to Left Hand"), "["),
                (self.tr("Map Selected Notes to Right Hand"), "]"),
                (self.tr("(in note tool) Switch Cursor to Left Hand"), ","),
                (self.tr("(in note tool) Switch Cursor to Right Hand"), "."),
                (self.tr("Set Snap Size to Whole Note Length"), "1"),
                (self.tr("Set Snap Size to Half Note Length"), "2"),
                (self.tr("Set Snap Size to Quarter Note Length"), "4"),
                (self.tr("Set Snap Size to Eighth Note Length"), "8"),
                (self.tr("Set Snap Size to Sixteenth Note Length"), "6"),
                (self.tr("Set Snap Size Divider to 3"), "3"),
                (self.tr("Set Snap Size Divider to 5"), "5"),
                (self.tr("Set Snap Size Divider to 7"), "7"),
            ]),
            (self.tr("View & Display"), [
                (self.tr("Zoom In"), "="),
                (self.tr("Zoom Out"), "-"),
                (self.tr("Full Screen"), "F11"),
            ]),
            (self.tr("Document Settings"), [
                (self.tr("Style Settings"), "S"),
                (self.tr("Title & Info"), "I"),
                (self.tr("Line Breaks & Pages"), "L"),
            ]),
        ]
    
    def _create_category_widget(self, category_name: str, shortcuts: list[tuple[str, str]]) -> QtWidgets.QWidget:
        """Create a category widget with shortcuts."""
        widget = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(widget)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)
        
        # Get theme colors
        text_color = Style.get_named_qcolor('text')
        alternate_bg = Style.get_named_qcolor('alternate_background_color')
        accent_color = Style.get_named_qcolor('accent')
        
        # Category header
        header = QtWidgets.QLabel(category_name)
        header.setStyleSheet(f"font-weight: bold; font-size: 13px; color: {text_color.name()}; margin-bottom: 8px;")
        layout.addWidget(header)
        
        # Shortcuts grid
        grid = QtWidgets.QGridLayout()
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 0)
        grid.setSpacing(10)
        
        for i, (action_name, shortcut_keys) in enumerate(shortcuts):
            action_label = QtWidgets.QLabel(action_name)
            action_label.setStyleSheet(f"color: {text_color.name()}; font-size: 11px;")
            
            shortcut_label = QtWidgets.QLabel(shortcut_keys)
            is_arrow_symbol = shortcut_keys in {"←", "→", "↑", "↓"}
            shortcut_font_size = "20px" if is_arrow_symbol else "11px"
            shortcut_label.setStyleSheet(
                f"background-color: {alternate_bg.name()}; color: {text_color.name()}; font-family: 'Courier New', monospace; "
                f"font-size: {shortcut_font_size}; padding: 5px 10px; border-radius: 4px; "
                f"border: 1px solid {text_color.name()}; font-weight: bold;"
            )
            shortcut_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
            shortcut_label.setMinimumWidth(80)
            
            grid.addWidget(action_label, i, 0)
            grid.addWidget(shortcut_label, i, 1)
        
        layout.addLayout(grid)
        
        # Style the category box
        bg_color = Style.get_named_qcolor('bg')
        widget.setStyleSheet(
            f"QWidget {{ background-color: {bg_color.name()}; border: 1px solid {text_color.name()}; "
            "border-radius: 6px; padding: 0px; }"
        )
        
        return widget
    
    def _setup_card_background(self, card_widget: QtWidgets.QWidget) -> None:
        """Set up the card with a greyed keyTAB logo background."""
        # Get theme background color
        bg_color = Style.get_named_qcolor('bg')
        
        try:
            icon = get_qicon('keyTAB', size=(256, 256))
            if icon is not None:
                pm = icon.pixmap(256, 256)
                if not pm.isNull():
                    # Create a greyed-out version
                    greyed_pm = QtGui.QPixmap(pm.size())
                    greyed_pm.fill(QtCore.Qt.GlobalColor.transparent)
                    painter = QtGui.QPainter(greyed_pm)
                    painter.setOpacity(0.05)
                    painter.drawPixmap(0, 0, pm)
                    painter.end()
                    
                    # Set as background
                    palette = QtGui.QPalette()
                    brush = QtGui.QBrush(greyed_pm)
                    palette.setBrush(QtGui.QPalette.ColorRole.Window, brush)
                    card_widget.setPalette(palette)
                    card_widget.setAutoFillBackground(True)
        except Exception:
            pass
        
        # Default background using theme color
        card_widget.setStyleSheet(
            f"QWidget {{ background-color: {bg_color.name()}; border-radius: 8px; }}"
        )
    
    def center_on_screen(self) -> None:
        """Center the dialog on the screen."""
        screen_geometry = QtWidgets.QApplication.primaryScreen().availableGeometry()
        dialog_geometry = self.frameGeometry()
        dialog_geometry.moveCenter(screen_geometry.center())
        self.move(dialog_geometry.topLeft())
    
    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        """Close dialog on any key press."""
        event.accept()
        self.close()
    
    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        """Close dialog on any mouse click."""
        event.accept()
        self.close()

