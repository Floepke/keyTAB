"""HeaderFooterDrawer: renders title, composer, footer text, and decoration lines."""

from datetime import datetime


class HeaderFooterDrawer:
    """Draw page header and footer."""
    
    def __init__(self, context):
        self.context = context
        self.du = context.du
        self.layout_data = context.layout_data
        self.notation_color = self.layout_data.get('notation_color')
        self.scale = float(self.layout_data.get('scale', 1.0) or 1.0)
    
    def draw(self) -> None:
        """Draw header/footer for the current page."""
        score = self.context.score or {}
        info = dict(score.get('info', {}) or {})
        layout = self.layout_data.get('layout', {})
        page_w = float(self.layout_data.get('page_width_mm', 210.0) or 210.0)
        page_h = float(self.layout_data.get('page_height_mm', 297.0) or 297.0)

        page_top = float(layout.get('page_top_margin_mm', 10.0) or 10.0)
        page_left = float(layout.get('page_left_margin_mm', 10.0) or 10.0)
        page_right = float(layout.get('page_right_margin_mm', 10.0) or 10.0)
        page_bottom = float(layout.get('page_bottom_margin_mm', 10.0) or 10.0)

        def _font(key: str, fallback_family: str, fallback_size: float):
            f = dict(layout.get(key, {}) or {})
            family = str(f.get('family', fallback_family) or fallback_family)
            size_pt = float(f.get('size_pt', fallback_size) or fallback_size) * self.scale
            bold = bool(f.get('bold', False))
            italic = bool(f.get('italic', False))
            underline = bool(f.get('underline', False))
            return family, size_pt, bold, italic, underline

        if self.context.pageno == 0:
            title = str(info.get('title', 'title') or 'title')
            composer = str(info.get('composer', 'composer') or 'composer')
            tf, ts, tb, ti, tu = _font('font_title', 'Edwin', 12.0)
            cf, cs, cb, ci, cu = _font('font_composer', 'Edwin', 10.0)
            tx = float(page_left)
            ty = float(page_top)
            cx = float(page_w - page_right)
            cy = float(page_top)
            self.du.add_text(tx, ty, title, family=tf, size_pt=ts, bold=tb, italic=ti, color=self.notation_color, anchor='nw', id=0, tags=['title'])
            self.du.add_text(cx, cy, composer, family=cf, size_pt=cs, bold=cb, italic=ci, color=self.notation_color, anchor='ne', id=0, tags=['composer'])
            if tu and title:
                xb, yb, w, _h = self.du._get_text_extents_mm(title, tf, ts, ti, tb)
                bx = tx - xb
                by = ty - yb
                self.du.add_line(bx, by + max(0.2, ts * 0.025), bx + w, by + max(0.2, ts * 0.025), color=self.notation_color, width_mm=max(0.2, ts * (0.04 if tb else 0.02)), tags=['title'])
            if cu and composer:
                xb, yb, w, _h = self.du._get_text_extents_mm(composer, cf, cs, ci, cb)
                bx = cx - w - xb
                by = cy - yb
                self.du.add_line(bx, by + max(0.2, cs * 0.025), bx + w, by + max(0.2, cs * 0.025), color=self.notation_color, width_mm=max(0.2, cs * (0.04 if cb else 0.02)), tags=['composer'])

        ff, fs, fb, fi, fu = _font('font_copyright', 'Edwin', 8.0)
        footer_text = str(info.get('copyright', f"© keyTAB {datetime.now().year}") or '')
        if not footer_text:
            return
        text = f"Page {self.context.pageno + 1} • {footer_text}"
        fx = float(page_left)
        fy = float(page_h - page_bottom)
        self.du.add_text(fx, fy, text, family=ff, size_pt=fs, bold=fb, italic=fi, color=self.notation_color, anchor=None, id=0, tags=['copyright'])
        if fu and text:
            xb, yb, w, _h = self.du._get_text_extents_mm(text, ff, fs, fi, fb)
            bx = fx
            by = fy
            self.du.add_line(bx + xb, by + max(0.2, fs * 0.025), bx + xb + w, by + max(0.2, fs * 0.025), color=self.notation_color, width_mm=max(0.2, fs * (0.04 if fb else 0.02)), tags=['copyright'])
