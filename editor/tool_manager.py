from __future__ import annotations
from typing import Optional, TYPE_CHECKING
from PySide6 import QtCore

if TYPE_CHECKING:
    from ui.widgets.contextual_toolbar import ContextualToolbar


class ToolManager(QtCore.QObject):
    """Manage the active tool and its contextual toolbar."""

    toolChanged = QtCore.Signal(str)

    def __init__(self, contextual_toolbar: 'ContextualToolbar'):
        super().__init__()
        self._contextual_toolbar = contextual_toolbar
        self._tool = None
        self._editor = None
        self._contextual_toolbar.contextButtonClicked.connect(self._on_context_button_clicked)

    def set_tool(self, tool) -> None:
        # Deactivate previous tool
        if self._tool is not None:
            self._tool.on_deactivate()
        self._tool = tool
        # Activate new tool
        if self._tool is not None:
            # Provide editor reference for convenience wrappers
            if self._editor is not None and hasattr(self._tool, 'set_editor'):
                self._tool.set_editor(self._editor)
            self._tool.on_activate()
        # Build contextual toolbar after activation so stateful buttons reflect restored tool state.
        defs = tool.toolbar_spec() or []
        self._contextual_toolbar.set_buttons(defs)
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
            defs = self._tool.toolbar_spec() or []
            self._contextual_toolbar.set_buttons(defs)
        # Force immediate visual feedback after any contextual button
        if self._editor is not None:
            if hasattr(self._editor, 'force_redraw_from_model'):
                self._editor.force_redraw_from_model()
            elif hasattr(self._editor, 'draw_frame'):
                self._editor.draw_frame()

    def refresh_context_buttons(self) -> None:
        """Rebuild contextual toolbar from current tool state."""
        if self._tool is None:
            return
        defs = self._tool.toolbar_spec() or []
        self._contextual_toolbar.set_buttons(defs)

    def set_editor(self, editor) -> None:
        """Bind the active Editor so tools can access conversion wrappers."""
        self._editor = editor
