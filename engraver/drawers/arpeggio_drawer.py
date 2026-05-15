"""ArpeggioDrawer: renders arpeggio stems."""


class ArpeggioDrawer:
    """Draw arpeggio stems."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
    
    def draw(self) -> None:
        """Draw all arpeggios for the current page and line."""
        # TODO: Extract arpeggio rendering logic from engraver.py
        # - Support-line collision geometry
        # - Hand-dependent anchor notes (no offset rule)
        # - Single-line stems
        pass
