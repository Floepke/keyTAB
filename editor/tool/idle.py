from editor.tool.base_tool import BaseTool


class IdleTool(BaseTool):
    """A mode that intentionally ignores all editor input."""

    TOOL_NAME = "idle"