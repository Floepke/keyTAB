"""TimeSignatureDrawer: renders time signature indicators."""


class TimeSignatureDrawer:
    """Draw time signature indicators."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
    
    def draw(self) -> None:
        """Draw all time signature indicators for the current page and line."""
        # TODO: Extract time signature rendering logic from engraver.py
        # - Classical time signature (numerator/denominator)
        # - Klavarskribo time signature grid indicators
        pass
