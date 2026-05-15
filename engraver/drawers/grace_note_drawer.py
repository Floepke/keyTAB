"""GraceNoteDrawer: renders grace notes (small decorative notes)."""


class GraceNoteDrawer:
    """Draw grace notes."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
    
    def draw(self) -> None:
        """Draw all grace notes for the current page and line."""
        # TODO: Extract grace note rendering logic from engraver.py
        # - Grace note noteheads (smaller, untilted)
        # - Grace note stems
        # - Grace note spacing
        pass
