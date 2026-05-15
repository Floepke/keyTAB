"""PedalDrawer: renders pedal symbols."""


class PedalDrawer:
    """Draw pedal symbols."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
    
    def draw(self) -> None:
        """Draw all pedal symbols for the current page and line."""
        # TODO: Extract pedal rendering logic from engraver.py
        # - Pedal symbols (down, up keytab/klavarskribo variants)
        pass
