from __future__ import annotations
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Adjust this import to where DrawUtil is defined in your project
    from ui.widgets.draw_util import DrawUtil
    from editor.editor import Editor


class BaseTool:
    """
    Base class for editor tools with convenient event hooks.
    Tools override methods they need; default implementations do nothing.
    """

    TOOL_NAME: str = "base"
    # Set to True in tools that edit the model during right-drag
    RIGHT_DRAG_EDITS: bool = False

    def __init__(self):
        # Optional: shared state across events
        self._active: bool = False
        self._editor: Optional[Editor] = None  # set by ToolManager for convenient access to editor wrappers
        self._du: Optional[DrawUtil] = None    # cached DrawUtil for quick access

    # Lifecycle hooks
    def on_activate(self) -> None:
        """Called when this tool becomes the active tool."""
        self._active = True

    def on_deactivate(self) -> None:
        """Called when this tool is no longer the active tool."""
        self._active = False

    # Toolbar integration
    def toolbar_spec(self) -> list[dict]:
        """Return a list of button definitions: {'name','icon','tooltip'}"""
        return []

    def on_toolbar_button(self, name: str) -> None:
        pass

    # Editor wiring
    def set_editor(self, editor: Editor) -> None:
        """Provide the active Editor instance to tools for convenience wrappers."""
        self._editor = editor
        try:
            self._du = editor.draw_util()
        except Exception:
            self._du = None

    def draw_util(self) -> DrawUtil:
        """Return the active DrawUtil instance."""
        if self._du is not None:
            return self._du
        if self._editor is None:
            raise RuntimeError("Editor not set")
        return self._editor.draw_util()

    # Mouse events
    def on_left_press(self, x: float, y: float) -> None: pass
    def on_left_unpress(self, x: float, y: float) -> None: pass
    def on_left_click(self, x: float, y: float) -> None: pass
    def on_left_double_click(self, x: float, y: float) -> None: pass
    def on_left_double_unpress(self, x: float, y: float) -> None: pass

    def on_left_drag_start(self, x: float, y: float) -> None: pass
    def on_left_drag(self, x: float, y: float, dx: float, dy: float) -> None: pass
    def on_left_drag_end(self, x: float, y: float) -> None: pass

    def on_right_press(self, x: float, y: float) -> None: pass
    def on_right_unpress(self, x: float, y: float) -> None: pass
    def on_right_click(self, x: float, y: float) -> None: pass
    def on_right_double_click(self, x: float, y: float) -> None: pass
    def on_right_double_unpress(self, x: float, y: float) -> None: pass

    def on_right_drag_start(self, x: float, y: float) -> None: pass
    def on_right_drag(self, x: float, y: float, dx: float, dy: float) -> None: pass
    def on_right_drag_end(self, x: float, y: float) -> None: pass

    def on_mouse_move(self, x: float, y: float) -> None:
        '''
            here we draw the cursor position line as that is a shared feature across tools
        '''
        # Provide shared behavior: update the editor's time cursor state.
        # Rendering is handled centrally by Editor.draw_guides().
        if self._editor is None:
            return
        # Convert pointer Y (logical px) → time ticks (includes scroll offset)
        t = self._editor.y_to_time(y)
        # Snap to the current snap size units
        t_snapped = self._editor.snap_time(t)
        self._editor.time_cursor = t_snapped
        # Convert snapped time → local (viewport) millimeters for direct drawing
        abs_mm = self._editor.time_to_mm(t_snapped)
        offset = float(getattr(self._editor, '_view_y_mm_offset', 0.0) or 0.0)
        self._editor.mm_cursor = abs_mm - offset


