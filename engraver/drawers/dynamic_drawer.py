"""DynamicDrawer: renders dynamic symbols, hairpins (crescendo/decrescendo)."""


class DynamicDrawer:
    """Draw dynamic symbols and hairpins."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
    
    def draw(self) -> None:
        """Draw all dynamics and hairpins for the current page and line."""
        # TODO: Extract dynamic rendering logic from engraver.py
        # - Dynamic symbols (p, f, mf, etc.)
        # - Hairpin lines (crescendo/decrescendo)
        pass
