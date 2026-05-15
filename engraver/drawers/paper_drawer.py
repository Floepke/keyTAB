"""PaperDrawer: renders the page background rectangle."""


class PaperDrawer:
    """Draw the page-colored background for the active page."""

    def __init__(self, context):
        self.context = context
        self.du = context.du
        self.layout_data = context.layout_data

    def draw(self) -> None:
        """Draw one full-page background rectangle."""
        paper_color = self.layout_data.get('paper_color', (1.0, 1.0, 1.0, 1.0))
        page_w = float(self.layout_data.get('page_width_mm', 210.0) or 210.0)
        page_h = float(self.layout_data.get('page_height_mm', 297.0) or 297.0)
        if page_w <= 0.0 or page_h <= 0.0:
            return

        self.du.add_rectangle(
            0.0,
            0.0,
            page_w,
            page_h,
            stroke_color=None,
            fill_color=paper_color,
            id=0,
            tags=['page_background'],
        )
