"""NoteheadDrawer: renders noteheads, stems, accidental guides, and per-note ledger lines."""


class NoteheadDrawer:
    """Draw noteheads, stems, accidentals, and ledger lines for individual notes."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
    
    def draw(self) -> None:
        """Draw all noteheads and stems for the current page and line."""
        # TODO: Extract notehead rendering logic from engraver.py
        # - Per-note iteration
        # - Notehead geometry and MIDI color
        # - Stem lines (with hand-dependent direction)
        # - Accidental guide lines
        # - Per-note ledger lines (short segments for out-of-range notes)
        # - Continuation dots
        pass
