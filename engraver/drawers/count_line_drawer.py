"""CountLineDrawer: renders count guide lines."""


class CountLineDrawer:
    """Draw count guide lines."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
    
    def draw(self) -> None:
        """Draw all count lines for the current page and line."""
        # TODO: Extract count line rendering logic from engraver.py
        # - Count line positions and markers
        pass
