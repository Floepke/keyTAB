"""TextDrawer: renders text annotations, tempo markings, and labels."""


class TextDrawer:
    """Draw text elements."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
    
    def draw(self) -> None:
        """Draw all text elements for the current page and line."""
        # TODO: Extract text rendering logic from engraver.py
        # - User-placed text annotations
        # - Tempo markings
        # - Title, composer, footer
        # - Measure numbers and guides
        pass
