"""BarlineDrawer: renders barlines, gridlines, and beat markers."""


class BarlineDrawer:
    """Draw barlines, grid lines, and constructive beat geometry."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
    
    def draw(self) -> None:
        """Draw all barlines and gridlines for the current page and line."""
        # TODO: Extract barline and gridline logic from engraver.py
        # - Barline positions from base_grid
        # - Constructive geometry (collision detection with notes, arpeggios, etc.)
        # - Grid lines (with dash patterns)
        # - Double barlines
        pass
