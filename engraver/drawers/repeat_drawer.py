"""RepeatDrawer: renders repeat symbols and endings."""


class RepeatDrawer:
    """Draw repeat symbols (start/end repeats, endings)."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
    
    def draw(self) -> None:
        """Draw all repeat symbols for the current page and line."""
        # TODO: Extract repeat rendering logic from engraver.py
        # - Start repeat symbols (vertical lines with dots)
        # - End repeat symbols
        # - Repeat endings (1st, 2nd, etc.)
        pass
