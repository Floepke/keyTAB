from __future__ import annotations
from typing import Optional
from PySide6 import QtCore
from icons.icons import get_qicon
from ui.widgets.toolbar_splitter import ToolbarSplitter


class ToolManager(QtCore.QObject):
    """Manage the active tool and its contextual toolbar in the splitter."""

    toolChanged = QtCore.Signal(str)

    def __init__(self, splitter: ToolbarSplitter):
        super().__init__()
        self._splitter = splitter
        self._tool = None
        self._editor = None
        self._splitter.contextButtonClicked.connect(self._on_context_button_clicked)

    def set_tool(self, tool) -> None:
        # Deactivate previous tool
        if self._tool is not None:
            self._tool.on_deactivate()
        self._tool = tool
        # Build contextual toolbar from tool.toolbar_spec()
        defs = tool.toolbar_spec() or []
        self._splitter.set_context_buttons(defs)
        # Activate new tool
        if self._tool is not None:
            # Provide editor reference for convenience wrappers
            try:
                if self._editor is not None and hasattr(self._tool, 'set_editor'):
                    self._tool.set_editor(self._editor)
            except Exception:
                pass
            self._tool.on_activate()
        name = getattr(tool, 'TOOL_NAME', 'unknown')
        self.toolChanged.emit(str(name))
        if self._editor is not None:
            if hasattr(self._editor, 'force_redraw_from_model'):
                self._editor.force_redraw_from_model()
            elif hasattr(self._editor, 'draw_frame'):
                self._editor.draw_frame()

    def _on_context_button_clicked(self, name: str) -> None:
        if self._tool is not None:
            self._tool.on_toolbar_button(name)
            # Rebuild contextual toolbar to reflect dynamic labels/state.
            try:
                defs = self._tool.toolbar_spec() or []
                self._splitter.set_context_buttons(defs)
            except Exception:
                pass
        # Force immediate visual feedback after any contextual button
        if self._editor is not None:
            if hasattr(self._editor, 'force_redraw_from_model'):
                self._editor.force_redraw_from_model()
            elif hasattr(self._editor, 'draw_frame'):
                self._editor.draw_frame()

    def set_editor(self, editor) -> None:
        """Bind the active Editor so tools can access conversion wrappers."""
        self._editor = editor
