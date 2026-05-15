"""MiniPianoDrawer: renders mini piano keyboard visualization at system end."""


class MiniPianoDrawer:
    """Draw mini piano keyboard."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
    
    def draw(self) -> None:
        """Draw mini piano for the current page and line."""
        # TODO: Extract mini piano rendering logic from engraver.py
        # - Keyboard outline
        # - Black key lines (with dash pattern for F/A)
        # - Grey octave bands
        pass
