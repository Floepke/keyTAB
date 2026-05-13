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

    def on_key_press(self, key: int, modifiers) -> bool:
        """Return True when a tool handled the key press."""
        return False

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

    # Shared rpitch helpers
    def rpitch_bounds(self) -> tuple[int, int]:
        """Return dynamic rpitch bounds for the current visible page width."""
        if self._editor is None:
            return (-68, 73)
        try:
            score = self._editor.current_score()
            layout = getattr(score, 'layout', None) if score is not None else None
            view_width_mm = float(getattr(layout, 'page_width_mm', 210.0) or 210.0)
            self._editor._calculate_layout(view_width_mm)
            base_x = float(self._editor.pitch_to_x(40))
            dist = float(getattr(self._editor, 'semitone_dist', 0.0) or 0.0)
            if dist <= 1e-6:
                return (-68, 73)
            min_rp = int(round((0.0 - base_x) / dist))
            max_rp = int(round((view_width_mm - base_x) / dist))
            if min_rp > max_rp:
                min_rp, max_rp = max_rp, min_rp
            return (min_rp + 5, max_rp - 5)
        except Exception:
            return (-68, 73)

    def clamp_rpitch(self, rpitch: float | int) -> int:
        """Clamp rpitch to current viewport bounds."""
        min_rp, max_rp = self.rpitch_bounds()
        try:
            rp = float(rpitch)
        except Exception:
            rp = 0.0
        return int(max(min_rp, min(max_rp, int(round(rp)))))

    def x_mm_to_rpitch_clamped(self, x_mm: float) -> int:
        """Convert page-space x(mm) to rpitch and clamp to viewport bounds."""
        if self._editor is None:
            return 0
        try:
            base_x = float(self._editor.pitch_to_x(40))
            dist = float(getattr(self._editor, 'semitone_dist', 0.0) or 0.0)
            if dist <= 1e-6:
                return 0
            rp = (float(x_mm) - base_x) / dist
            return self.clamp_rpitch(rp)
        except Exception:
            return 0

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
        # Convert widget coordinates to time ticks for the active orientation.
        t = self._editor.widget_px_to_time(x, y)
        # Snap to the current snap size units
        t_snapped = self._editor.snap_time(t)
        self._editor.time_cursor = t_snapped
        # Convert snapped time → local (viewport) millimeters for direct drawing
        abs_mm = self._editor.time_to_mm(t_snapped)
        offset = float(getattr(self._editor, '_view_y_mm_offset', 0.0) or 0.0)
        self._editor.mm_cursor = abs_mm - offset


