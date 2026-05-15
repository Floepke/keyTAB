"""GridBandDrawer: renders alternating grid band backgrounds."""


class GridBandDrawer:
    """Draw grid band (alternating background shading)."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
    
    def draw(self) -> None:
        """Draw grid band bands for the current page and line."""
        # TODO: Extract grid band rendering logic from engraver.py
        # - Alternating background rectangles
        # - Hand-dependent split line (left vs right hand visual separation)
        pass
