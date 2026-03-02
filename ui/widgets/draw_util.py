from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Optional, Sequence, Tuple
import os, math
import cairo
from PySide6 import QtGui
from utils.CONSTANT import EDITOR_LAYERING

MM_PER_INCH = 25.4
PT_PER_INCH = 72.0
PT_PER_MM = PT_PER_INCH / MM_PER_INCH

Color = Tuple[float, float, float, float]


@dataclass
class Stroke:
    color: Color = (0, 0, 0, 1)
    width_mm: float = 0.3
    dash_pattern_mm: Optional[Sequence[float]] = None
    dash_offset_mm: float = 0.0
    line_cap: str = "round"


@dataclass
class Fill:
    color: Color = (0, 0, 0, 0)


@dataclass
class Line:
    x1_mm: float
    y1_mm: float
    x2_mm: float
    y2_mm: float
    stroke: Stroke = field(default_factory=Stroke)
    # Deprecated: `id` is no longer used for picking; kept for compatibility.
    id: int = 0
    tags: List[str] = field(default_factory=list)
    hit_rect_mm: Optional[Tuple[float, float, float, float]] = None


@dataclass
class Rect:
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    stroke: Optional[Stroke] = field(default_factory=Stroke)
    fill: Optional[Fill] = field(default_factory=Fill)
    # Deprecated: `id` is no longer used for picking; kept for compatibility.
    id: int = 0
    tags: List[str] = field(default_factory=list)
    hit_rect_mm: Optional[Tuple[float, float, float, float]] = None


@dataclass
class Oval:
    x_mm: float
    y_mm: float
    w_mm: float
    h_mm: float
    stroke: Optional[Stroke] = field(default_factory=Stroke)
    fill: Optional[Fill] = field(default_factory=Fill)
    # Deprecated: `id` is no longer used for picking; kept for compatibility.
    id: int = 0
    tags: List[str] = field(default_factory=list)
    hit_rect_mm: Optional[Tuple[float, float, float, float]] = None


@dataclass
class Polyline:
    points_mm: List[Tuple[float, float]]
    closed: bool = False
    stroke: Optional[Stroke] = field(default_factory=Stroke)
    fill: Optional[Fill] = field(default_factory=Fill)
    # Deprecated: `id` is no longer used for picking; kept for compatibility.
    id: int = 0
    tags: List[str] = field(default_factory=list)
    hit_rect_mm: Optional[Tuple[float, float, float, float]] = None


@dataclass
class Page:
    width_mm: float
    height_mm: float
    items: List[object] = field(default_factory=list)


# ---- Text item ----

@dataclass
class Text:
    x_mm: float
    y_mm: float
    text: str
    family: str = "Courier New"
    size_pt: float = 10.0
    italic: bool = False
    bold: bool = False
    color: Color = (0, 0, 0, 1)
    # Optional anchor specifying how (x_mm, y_mm) positions the text bounding box.
    # Supported values: 'center', 'n', 'ne', 'e', 'se', 's', 'sw', 'w', 'nw'.
    # If None, defaults to baseline positioning (legacy behavior).
    anchor: str | None = None
    angle_deg: float = 0.0
    # Deprecated: `id` is no longer used for picking; kept for compatibility.
    id: int = 0
    tags: List[str] = field(default_factory=list)
    hit_rect_mm: Optional[Tuple[float, float, float, float]] = None  # (x,y,w,h)


class DrawUtil:
    def __init__(self) -> None:
        self._pages: List[Page] = []
        self._current_index: int = -1

    @staticmethod
    def _safe_dash(pattern: Optional[Sequence[float]], offset: float) -> tuple[Optional[List[float]], float]:
        """Return (dash_list_mm, offset_mm) with invalid entries stripped.

        Dash lengths are interpreted in millimeters; zeros are coerced to 0.01
        to allow dotted patterns like "0 3". Any negative or non-finite value
        is dropped; if nothing remains we fall back to a solid stroke.
        """
        if not pattern:
            return (None, 0.0)
        cleaned: List[float] = []
        for v in pattern:
            try:
                f = float(v)
            except Exception:
                continue
            if not math.isfinite(f):
                continue
            if f < 0.0:
                continue
            if f == 0.0:
                f = 0.01
            cleaned.append(f)
        if not cleaned:
            return (None, 0.0)
        try:
            off = float(offset)
        except Exception:
            off = 0.0
        return (cleaned, off)

    def new_page(self, width_mm: float, height_mm: float) -> None:
        self._pages.append(Page(width_mm, height_mm))
        self._current_index = len(self._pages) - 1

    def set_current_page(self, index: int) -> None:
        if not (0 <= index < len(self._pages)):
            raise IndexError("Page index out of range")
        self._current_index = index

    def current_page_index(self) -> int:
        return self._current_index

    def page_count(self) -> int:
        return len(self._pages)

    def current_page_size_mm(self) -> Tuple[float, float]:
        if self._current_index < 0:
            return (0.0, 0.0)
        p = self._pages[self._current_index]
        return (p.width_mm, p.height_mm)

    def set_current_page_size_mm(self, width_mm: float, height_mm: float) -> None:
        """Update the current page dimensions (mm) without altering items.

        If there is no current page, creates one.
        """
        width_mm = float(max(0.0, width_mm))
        height_mm = float(max(0.0, height_mm))
        if self._current_index < 0:
            self.new_page(width_mm or 210.0, height_mm or 297.0)
            return
        p = self._pages[self._current_index]
        p.width_mm = width_mm or p.width_mm
        p.height_mm = height_mm or p.height_mm

    def add_line(self, x1_mm: float, y1_mm: float, x2_mm: float, y2_mm: float,
                 color: Color = (0, 0, 0, 1), width_mm: float = 0.3,
                 dash_pattern: Optional[Sequence[float]] = None,
                 dash_offset_mm: float = 0.0,
                 line_cap: str = "round",
                 id: int = 0, tags: Optional[List[str]] = None,
                 hit_rect_mm: Optional[Tuple[float, float, float, float]] = None) -> None:
        self._ensure_page()
        stroke = Stroke(color=color, width_mm=width_mm,
                   dash_pattern_mm=dash_pattern, dash_offset_mm=dash_offset_mm, line_cap=line_cap)
        if tags is None:
            tags = []
        # Compute a default hit rect for picking if not provided.
        if hit_rect_mm is None:
            x = min(x1_mm, x2_mm)
            y = min(y1_mm, y2_mm)
            w = abs(x2_mm - x1_mm)
            h = abs(y2_mm - y1_mm)
            hit_rect_mm = (x, y, w, h)
        self._pages[self._current_index].items.append(Line(x1_mm, y1_mm, x2_mm, y2_mm, stroke, id, tags, hit_rect_mm))

    def add_rectangle(self,
                      x1_mm: float,
                      y1_mm: float,
                      x2_mm: float,
                      y2_mm: float,
                      stroke_color: Optional[Color] = (0, 0, 0, 1),
                      stroke_width_mm: float = 0.3,
                      fill_color: Optional[Color] = None,
                      dash_pattern: Optional[Sequence[float]] = None,
                      dash_offset_mm: float = 0.0,
                      id: int = 0,
                      tags: Optional[List[str]] = None,
                      hit_rect_mm: Optional[Tuple[float, float, float, float]] = None) -> None:
        """Add a rectangle by specifying two opposite corner points (x1,y1) and (x2,y2)."""
        self._ensure_page()
        stroke = None
        fill = None
        if stroke_color is not None:
            stroke = Stroke(stroke_color, stroke_width_mm, dash_pattern, dash_offset_mm)
        if fill_color is not None:
            fill = Fill(fill_color)
        if tags is None:
            tags = []
        # Interpret inputs as two opposite corners
        x_a = float(x1_mm)
        y_a = float(y1_mm)
        x_b = float(x2_mm)
        y_b = float(y2_mm)
        rx = min(x_a, x_b)
        ry = min(y_a, y_b)
        rw = abs(x_b - x_a)
        rh = abs(y_b - y_a)

        if hit_rect_mm is None:
            hit_rect_mm = (rx, ry, rw, rh)
        self._pages[self._current_index].items.append(Rect(rx, ry, rw, rh, stroke, fill, id, tags, hit_rect_mm))

    def add_oval(self,
                 x1_mm: float,
                 y1_mm: float,
                 x2_mm: float,
                 y2_mm: float,
                 stroke_color: Optional[Color] = (0, 0, 0, 1),
                 stroke_width_mm: float = 0.3,
                 fill_color: Optional[Color] = None,
                 dash_pattern: Optional[Sequence[float]] = None,
                 dash_offset_mm: float = 0.0,
                 id: int = 0,
                 tags: Optional[List[str]] = None,
                 hit_rect_mm: Optional[Tuple[float, float, float, float]] = None) -> None:
        """Add an oval defined by two opposite corners (x1_mm,y1_mm) and (x2_mm,y2_mm).

        The oval is inscribed in the axis-aligned rectangle given by these corners.
        """
        self._ensure_page()
        stroke = None
        fill = None
        if stroke_color is not None:
            stroke = Stroke(stroke_color, stroke_width_mm, dash_pattern, dash_offset_mm)
        if fill_color is not None:
            fill = Fill(fill_color)
        if tags is None:
            tags = []
        # Normalize to rectangle origin + size
        xa = float(x1_mm)
        ya = float(y1_mm)
        xb = float(x2_mm)
        yb = float(y2_mm)
        rx = min(xa, xb)
        ry = min(ya, yb)
        rw = abs(xb - xa)
        rh = abs(yb - ya)
        
        # # possible to inset the oval to account for stroke width
        # if stroke is not None:
        #     inset = float(stroke.width_mm) / 4.0 #TODO: check if this is needed.
        #     rx += inset
        #     ry += inset
        #     rw = max(0.0, rw - (2.0 * inset))
        #     rh = max(0.0, rh - (2.0 * inset))
        
        if hit_rect_mm is None:
            hit_rect_mm = (rx, ry, rw, rh)
        self._pages[self._current_index].items.append(Oval(rx, ry, rw, rh, stroke, fill, id, tags, hit_rect_mm))

    def add_polygon(self, points_mm: Sequence[Tuple[float, float]],
                    stroke_color: Optional[Color] = (0, 0, 0, 1),
                    stroke_width_mm: float = 0.3,
                    fill_color: Optional[Color] = None,
                    dash_pattern: Optional[Sequence[float]] = None,
                    dash_offset_mm: float = 0.0,
                    id: int = 0, tags: Optional[List[str]] = None,
                    hit_rect_mm: Optional[Tuple[float, float, float, float]] = None) -> None:
        self._ensure_page()
        stroke = None
        fill = None
        if stroke_color is not None:
            stroke = Stroke(stroke_color, stroke_width_mm, dash_pattern, dash_offset_mm)
        if fill_color is not None:
            fill = Fill(fill_color)
        if tags is None:
            tags = []
        if hit_rect_mm is None:
            xs = [p[0] for p in points_mm]
            ys = [p[1] for p in points_mm]
            x = min(xs) if xs else 0.0
            y = min(ys) if ys else 0.0
            w = (max(xs) - x) if xs else 0.0
            h = (max(ys) - y) if ys else 0.0
            hit_rect_mm = (x, y, w, h)
        self._pages[self._current_index].items.append(Polyline(list(points_mm), True, stroke, fill, id, tags, hit_rect_mm))

    def add_polyline(self, points_mm: Sequence[Tuple[float, float]],
                     stroke_color: Optional[Color] = (0, 0, 0, 1),
                     stroke_width_mm: float = 0.3,
                     dash_pattern: Optional[Sequence[float]] = None,
                     dash_offset_mm: float = 0.0,
                     id: int = 0, tags: Optional[List[str]] = None,
                     hit_rect_mm: Optional[Tuple[float, float, float, float]] = None) -> None:
        self._ensure_page()
        stroke = Stroke(stroke_color, stroke_width_mm, dash_pattern, dash_offset_mm)
        if tags is None:
            tags = []
        if hit_rect_mm is None:
            xs = [p[0] for p in points_mm]
            ys = [p[1] for p in points_mm]
            x = min(xs) if xs else 0.0
            y = min(ys) if ys else 0.0
            w = (max(xs) - x) if xs else 0.0
            h = (max(ys) - y) if ys else 0.0
            hit_rect_mm = (x, y, w, h)
        self._pages[self._current_index].items.append(Polyline(list(points_mm), False, stroke, None, id, tags, hit_rect_mm))

    def add_text(self, x_mm: float, y_mm: float, text: str,
                 family: str = "Sans", size_pt: float = 10.0,
                 italic: bool = False, bold: bool = False,
                 color: Color = (0, 0, 0, 1),
                 anchor: str | None = None,
                 id: int = 0, tags: Optional[List[str]] = None,
                 hit_rect_mm: Optional[Tuple[float, float, float, float]] = None,
                 angle_deg: float = 0.0) -> None:
        """Add a text item. y_mm is the baseline.

        If hit_rect_mm is not provided, it will be computed from toy-text extents
        (approximate; upgrade to PangoCairo later for robust metrics).
        """
        self._ensure_page()
        if tags is None:
            tags = []
        # Compute hit rect and baseline origin based on anchor
        if hit_rect_mm is None:
            hit_rect_mm = self._compute_text_hit_rect_mm(x_mm, y_mm, text, family, size_pt, italic, bold, anchor)
        bx = x_mm
        by = y_mm
        if anchor:
            xb_mm, yb_mm, w_mm, h_mm = self._get_text_extents_mm(text, family, size_pt, italic, bold)
            ax = x_mm
            ay = y_mm
            if anchor == 'center':
                rx = ax - w_mm / 2.0
                ry = ay - h_mm / 2.0
            elif anchor == 'n':
                rx = ax - w_mm / 2.0
                ry = ay
            elif anchor == 's':
                rx = ax - w_mm / 2.0
                ry = ay - h_mm
            elif anchor == 'w':
                rx = ax
                ry = ay - h_mm / 2.0
            elif anchor == 'e':
                rx = ax - w_mm
                ry = ay - h_mm / 2.0
            elif anchor == 'nw':
                rx = ax
                ry = ay
            elif anchor == 'ne':
                rx = ax - w_mm
                ry = ay
            elif anchor == 'sw':
                rx = ax
                ry = ay - h_mm
            elif anchor == 'se':
                rx = ax - w_mm
                ry = ay - h_mm
            else:
                rx = None
                ry = None
            if rx is not None and ry is not None:
                bx = rx - xb_mm
                by = ry - yb_mm
        self._pages[self._current_index].items.append(
            Text(bx, by, text, family, size_pt, italic, bold, color, anchor, angle_deg, id, tags, hit_rect_mm)
        )

    def _ensure_page(self) -> None:
        if self._current_index < 0:
            raise RuntimeError("No page: call new_page(width_mm, height_mm) first")

    def render_to_cairo(self, ctx: cairo.Context, page_index: int, px_per_mm: float,
                        clip_rect_mm: Optional[Tuple[float, float, float, float]] = None,
                        overscan_mm: float = 0.0,
                        layering: Optional[Sequence[str]] = None) -> None:
        if page_index < 0 or page_index >= len(self._pages):
            return
        page = self._pages[page_index]
        ctx.save()
        # Prefer highest quality to keep text and thin lines smooth across scales
        ctx.set_antialias(cairo.ANTIALIAS_BEST)
        ctx.scale(px_per_mm, px_per_mm)
        # Static viewport: translate to the clip origin only; do not apply Cairo clipping.
        # Determine viewport origin and size in mm and translate to anchor at (0,0)
        x_o = 0.0
        y_o = 0.0
        vp_w_mm = page.width_mm
        vp_h_mm = page.height_mm
        if clip_rect_mm is not None:
            clip_x, clip_y, w, h = clip_rect_mm
            # Translate to the viewport origin; shapes are drawn as-is.
            ctx.translate(-clip_x, -clip_y)
            # Logical viewport dimensions
            x_o = clip_x
            y_o = clip_y
            vp_w_mm = w
            vp_h_mm = h
        # Do not auto-fill a white background; honor caller-supplied background
        # (e.g., explicit rectangle item or widget painter).

        layering_list = list(layering) if layering is not None else list(EDITOR_LAYERING)
        for item in self._iter_items_in_editor_order(page, clip_rect_mm, layering_list):
            if isinstance(item, Line):
                # Draw lines without trimming; rely on culling by hit-rect only.
                self._draw_line(ctx, item)
            elif isinstance(item, Rect):
                self._draw_rect(ctx, item)
            elif isinstance(item, Oval):
                self._draw_oval(ctx, item)
            elif isinstance(item, Polyline):
                self._draw_polyline(ctx, item)
            elif isinstance(item, Text):
                self._draw_text(ctx, item)
        ctx.restore()

    # ---- Tag system (tkinter-style) ----

    def find_with_tag(self, tag: str, page_index: Optional[int] = None) -> List[object]:
        """Return all items on the page that have the given tag."""
        if page_index is None:
            page_index = self._current_index
        if page_index < 0 or page_index >= len(self._pages):
            return []
        page = self._pages[page_index]
        return [it for it in page.items if tag in getattr(it, "tags", [])]

    def find_all(self, tags: Iterable[str], match_all: bool = False, page_index: Optional[int] = None) -> List[object]:
        """Find items that match one or all of the provided tags.

        - match_all=False: item has any of the tags
        - match_all=True:  item has all of the tags
        """
        tag_set = set(tags)
        if not tag_set:
            return []
        if page_index is None:
            page_index = self._current_index
        if page_index < 0 or page_index >= len(self._pages):
            return []
        page = self._pages[page_index]
        if match_all:
            return [it for it in page.items if tag_set.issubset(set(getattr(it, "tags", [])))]
        else:
            return [it for it in page.items if set(getattr(it, "tags", [])).intersection(tag_set)]

    def delete_with_tag(self, tag: str, page_index: Optional[int] = None) -> int:
        """Delete all items with the given tag. Returns count removed."""
        return self.delete_with_tags([tag], match_all=False, page_index=page_index)

    def delete_with_tags(self, tags: Iterable[str], match_all: bool = False, page_index: Optional[int] = None) -> int:
        """Delete all items that match one/all of the tags. Returns count removed."""
        tag_set = set(tags)
        if page_index is None:
            page_index = self._current_index
        if page_index < 0 or page_index >= len(self._pages):
            return 0
        page = self._pages[page_index]
        before = len(page.items)
        if match_all:
            page.items = [it for it in page.items if not tag_set.issubset(set(getattr(it, "tags", [])))]
        else:
            page.items = [it for it in page.items if not set(getattr(it, "tags", [])).intersection(tag_set)]
        return before - len(page.items)

    def add_tag(self, item: object, tag: str) -> None:
        """Add a tag to an item if not present."""
        tags = getattr(item, "tags", None)
        if tags is not None and tag not in tags:
            tags.append(tag)

    def remove_tag(self, item: object, tag: str) -> None:
        """Remove a tag from an item if present."""
        tags = getattr(item, "tags", None)
        if tags is not None and tag in tags:
            tags.remove(tag)

    # ---- Drawing order based on tags ----

    def _item_layer_index(self, item: object, layering: Sequence[str]) -> int:
        """Return the index in provided layering sequence for the item's tags.

        If multiple tags match layers, the earliest layer index is used.
        Items with no matching tags are placed after all known layers.
        """
        item_tags = getattr(item, "tags", [])
        best = len(layering)
        for t in item_tags:
            try:
                idx = layering.index(t)
                if idx < best:
                    best = idx
            except ValueError:
                continue
        return best

    def _rect_intersects_rect(self, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> bool:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        return not (ax + aw < bx or bx + bw < ax or ay + ah < by or by + bh < ay)

    def _iter_items_in_editor_order(self, page: Page, clip_rect_mm: Optional[Tuple[float, float, float, float]] = None,
                                    layering: Sequence[str] = tuple(EDITOR_LAYERING)):
        # Stable sort: by layer index, then by insertion order
        with_index = list(enumerate(page.items))
        # Optional culling by clip rect
        if clip_rect_mm is not None:
            culled = []
            for idx, it in with_index:
                rect = getattr(it, "hit_rect_mm", None)
                if rect is None:
                    # If no rect, keep (e.g., text without metrics) — rely on Cairo clip
                    culled.append((idx, it))
                else:
                    if self._rect_intersects_rect(rect, clip_rect_mm):
                        culled.append((idx, it))
            with_index = culled
        with_index.sort(key=lambda pair: (self._item_layer_index(pair[1], layering), pair[0]))
        for _i, item in with_index:
            yield item

    # ---- Hit detection ----

    def _rect_contains_point(self, rect_mm: Tuple[float, float, float, float], x_mm: float, y_mm: float) -> bool:
        rx, ry, rw, rh = rect_mm
        return (x_mm >= rx) and (y_mm >= ry) and (x_mm <= rx + rw) and (y_mm <= ry + rh)

    def hit_test_point_mm(self, x_mm: float, y_mm: float, page_index: Optional[int] = None):
        """Return the smallest-area clickable item at (x_mm, y_mm) or None.

        Only items with a `hit_rect_mm` are considered.
        """
        if page_index is None:
            page_index = self._current_index
        if page_index < 0 or page_index >= len(self._pages):
            return None
        page = self._pages[page_index]
        candidates = []
        for item in page.items:
            rect = getattr(item, "hit_rect_mm", None)
            if rect is None:
                continue
            if self._rect_contains_point(rect, x_mm, y_mm):
                rx, ry, rw, rh = rect
                area = max(0.0, rw * rh)
                candidates.append((area, item))
        if not candidates:
            return None
        candidates.sort(key=lambda t: t[0])
        return candidates[0][1]

    def hit_test_all_point_mm(self, x_mm: float, y_mm: float, page_index: Optional[int] = None):
        """Return list of all clickable items at (x_mm, y_mm) sorted by area ascending."""
        if page_index is None:
            page_index = self._current_index
        if page_index < 0 or page_index >= len(self._pages):
            return []
        page = self._pages[page_index]
        out = []
        for item in page.items:
            rect = getattr(item, "hit_rect_mm", None)
            if rect is None:
                continue
            if self._rect_contains_point(rect, x_mm, y_mm):
                rx, ry, rw, rh = rect
                area = max(0.0, rw * rh)
                out.append((area, item))
        out.sort(key=lambda t: t[0])
        return [i for (_a, i) in out]

    def save_pdf(
        self,
        path: str,
        layering: Optional[Sequence[str]] = None,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        if not self._pages:
            return
        surface: Optional[cairo.PDFSurface] = None
        ctx: Optional[cairo.Context] = None
        total_pages = len(self._pages)
        for i, page in enumerate(self._pages):
            width_pt = page.width_mm * PT_PER_MM
            height_pt = page.height_mm * PT_PER_MM
            if i == 0:
                surface = cairo.PDFSurface(path, width_pt, height_pt)
            else:
                surface.set_size(width_pt, height_pt)
            ctx = cairo.Context(surface)
            ctx.save()
            ctx.scale(PT_PER_MM, PT_PER_MM)
            ctx.set_source_rgb(1, 1, 1)
            ctx.rectangle(0, 0, page.width_mm, page.height_mm)
            ctx.fill()
            layering_list = list(layering) if layering is not None else list(EDITOR_LAYERING)
            for item in self._iter_items_in_editor_order(page, None, layering_list):
                if isinstance(item, Line):
                    self._draw_line(ctx, item)
                elif isinstance(item, Rect):
                    self._draw_rect(ctx, item)
                elif isinstance(item, Oval):
                    self._draw_oval(ctx, item)
                elif isinstance(item, Polyline):
                    self._draw_polyline(ctx, item)
                elif isinstance(item, Text):
                    self._draw_text(ctx, item)
            ctx.restore()
            if progress_cb is not None:
                try:
                    progress_cb(i + 1, total_pages)
                except Exception:
                    pass
            if i < (len(self._pages) - 1):
                surface.show_page()
        if surface is not None:
            surface.finish()

    def _apply_stroke(self, ctx: cairo.Context, stroke: Stroke):
        ctx.set_source_rgba(*stroke.color)
        ctx.set_line_width(stroke.width_mm)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        if stroke.line_cap == "butt":
            ctx.set_line_cap(cairo.LINE_CAP_BUTT)
        elif stroke.line_cap == "square":
            ctx.set_line_cap(cairo.LINE_CAP_SQUARE)
        else:
            ctx.set_line_cap(cairo.LINE_CAP_ROUND)
        dash, offset = self._safe_dash(stroke.dash_pattern_mm, stroke.dash_offset_mm)
        try:
            if dash:
                ctx.set_dash(dash, offset)
            else:
                ctx.set_dash([])
        except Exception:
            ctx.set_dash([])

    def _apply_fill(self, ctx: cairo.Context, fill: Optional[Fill]):
        if fill and fill.color[3] > 0:
            ctx.set_source_rgba(*fill.color)
            ctx.fill_preserve()

    def _clip_line_to_viewport(self, x1: float, y1: float, x2: float, y2: float, vp_w: float, vp_h: float):
        """Liang-Barsky line clipping. Returns None if fully outside, else endpoints in viewport space."""
        dx = x2 - x1
        dy = y2 - y1
        p = [-dx, dx, -dy, dy]
        q = [x1, vp_w - x1, y1, vp_h - y1]
        u1 = 0.0
        u2 = 1.0
        for pi, qi in zip(p, q):
            if pi == 0.0:
                if qi < 0.0:
                    return None
            else:
                t = qi / pi
                if pi < 0.0:
                    if t > u2:
                        return None
                    if t > u1:
                        u1 = t
                else:  # pi > 0
                    if t < u1:
                        return None
                    if t < u2:
                        u2 = t
        cx1 = x1 + u1 * dx
        cy1 = y1 + u1 * dy
        cx2 = x1 + u2 * dx
        cy2 = y1 + u2 * dy
        return (cx1, cy1, cx2, cy2)

    def _draw_line_clipped(self, ctx: cairo.Context, line: Line, x_o: float, y_o: float, vp_w_mm: float, vp_h_mm: float):
        # Transform line endpoints into viewport space (after translation by -x_o, -y_o)
        x1 = line.x1_mm - x_o
        y1 = line.y1_mm - y_o
        x2 = line.x2_mm - x_o
        y2 = line.y2_mm - y_o
        clipped = self._clip_line_to_viewport(x1, y1, x2, y2, vp_w_mm, vp_h_mm)
        if clipped is None:
            return
        cx1, cy1, cx2, cy2 = clipped
        self._apply_stroke(ctx, line.stroke)
        ctx.move_to(cx1, cy1)
        ctx.line_to(cx2, cy2)
        ctx.stroke()

    def _draw_line(self, ctx: cairo.Context, line: Line):
        self._apply_stroke(ctx, line.stroke)
        ctx.move_to(line.x1_mm, line.y1_mm)
        ctx.line_to(line.x2_mm, line.y2_mm)
        ctx.stroke()

    def _draw_rect(self, ctx: cairo.Context, r: Rect):
        ctx.new_path()
        ctx.rectangle(r.x_mm, r.y_mm, r.w_mm, r.h_mm)
        if r.fill:
            self._apply_fill(ctx, r.fill)
        if r.stroke:
            self._apply_stroke(ctx, r.stroke)
            ctx.stroke()
        else:
            ctx.new_path()

    def _draw_oval(self, ctx: cairo.Context, o: Oval):
        cx = o.x_mm + o.w_mm / 2.0
        cy = o.y_mm + o.h_mm / 2.0
        rx = max(0.0, o.w_mm / 2.0)
        ry = max(0.0, o.h_mm / 2.0)
        ctx.save()
        ctx.translate(cx, cy)
        if rx > 0 and ry > 0:
            ctx.scale(rx, ry)
            ctx.new_path()
            ctx.arc(0, 0, 1.0, 0, 2*3.1415926535)
            if o.fill:
                self._apply_fill(ctx, o.fill)
            if o.stroke:
                scale = max(rx, ry)
                adj = Stroke(
                    color=o.stroke.color,
                    width_mm=o.stroke.width_mm / scale if scale > 0 else o.stroke.width_mm,
                    dash_pattern_mm=o.stroke.dash_pattern_mm,
                    dash_offset_mm=o.stroke.dash_offset_mm,
                    line_cap=o.stroke.line_cap,
                )
                self._apply_stroke(ctx, adj)
                ctx.stroke()
            else:
                ctx.new_path()
            ctx.restore()
        else:
            ctx.restore()

    def _draw_polyline(self, ctx: cairo.Context, pl: Polyline):
        pts = pl.points_mm
        if not pts:
            return
        ctx.new_path()
        ctx.move_to(pts[0][0], pts[0][1])
        for (x, y) in pts[1:]:
            ctx.line_to(x, y)
        if pl.closed:
            ctx.close_path()
        if pl.fill and pl.closed:
            self._apply_fill(ctx, pl.fill)
        if pl.stroke:
            self._apply_stroke(ctx, pl.stroke)
            ctx.stroke()
        else:
            ctx.new_path()

    def _draw_text(self, ctx: cairo.Context, t: Text):
        # Cairo toy text: render via text_path + fill to avoid any implicit stroke
        # and ensure a single-color raster without edge bleed.
        slant = cairo.FONT_SLANT_ITALIC if t.italic else cairo.FONT_SLANT_NORMAL
        weight = cairo.FONT_WEIGHT_BOLD if t.bold else cairo.FONT_WEIGHT_NORMAL
        ctx.save()
        angle = float(getattr(t, 'angle_deg', 0.0) or 0.0)
        xb_mm, yb_mm, w_mm, h_mm = self._get_text_extents_mm(t.text, t.family, t.size_pt, t.italic, t.bold)
        anchor = getattr(t, 'anchor', None)
        # Compute rotation pivot based on anchor
        ax = t.x_mm + xb_mm + w_mm * 0.5
        ay = t.y_mm + yb_mm + h_mm * 0.5
        if anchor == 'n':
            ay = t.y_mm + yb_mm
        elif anchor == 's':
            ay = t.y_mm + yb_mm + h_mm
        elif anchor == 'w':
            ax = t.x_mm + xb_mm
        elif anchor == 'e':
            ax = t.x_mm + xb_mm + w_mm
        elif anchor == 'nw':
            ax = t.x_mm + xb_mm
            ay = t.y_mm + yb_mm
        elif anchor == 'ne':
            ax = t.x_mm + xb_mm + w_mm
            ay = t.y_mm + yb_mm
        elif anchor == 'sw':
            ax = t.x_mm + xb_mm
            ay = t.y_mm + yb_mm + h_mm
        elif anchor == 'se':
            ax = t.x_mm + xb_mm + w_mm
            ay = t.y_mm + yb_mm + h_mm

        # Rotate around pivot, then draw at stored position
        ctx.translate(ax, ay)
        if angle:
            ctx.rotate(angle * math.pi / 180.0)
        ctx.translate(-ax, -ay)
        ctx.select_font_face(t.family, slant, weight)
        ctx.set_font_size(t.size_pt / PT_PER_MM)
        ctx.move_to(t.x_mm, t.y_mm)
        ctx.text_path(t.text)
        ctx.set_source_rgba(*t.color)
        ctx.fill()
        ctx.restore()
        # Optional debug: draw text bounds and anchor point
        if os.getenv('PIANOSCRIPT_DEBUG_TEXT_BOUNDS', '0') in ('1', 'true', 'True'):
            rect = getattr(t, 'hit_rect_mm', None)
            if rect is not None:
                rx, ry, rw, rh = rect
                ctx.save()
                ctx.set_source_rgba(1.0, 0.2, 0.2, 0.8)
                ctx.set_line_width(0.2)
                ctx.rectangle(rx, ry, rw, rh)
                ctx.stroke()
                # Anchor marker
                ax = rx + rw / 2.0
                ay = ry + rh / 2.0
                if t.anchor == 'w':
                    ax = rx
                elif t.anchor == 'e':
                    ax = rx + rw
                elif t.anchor == 'n':
                    ay = ry
                elif t.anchor == 's':
                    ay = ry + rh
                elif t.anchor == 'nw':
                    ax = rx; ay = ry
                elif t.anchor == 'ne':
                    ax = rx + rw; ay = ry
                elif t.anchor == 'sw':
                    ax = rx; ay = ry + rh
                elif t.anchor == 'se':
                    ax = rx + rw; ay = ry + rh
                # draw cross
                ctx.set_source_rgba(0.2, 0.8, 0.2, 0.9)
                ctx.move_to(ax - 1.5, ay)
                ctx.line_to(ax + 1.5, ay)
                ctx.move_to(ax, ay - 1.5)
                ctx.line_to(ax, ay + 1.5)
                ctx.stroke()
                ctx.restore()

    def _compute_text_hit_rect_mm(self, x_mm: float, y_mm: float, text: str,
                                  family: str, size_pt: float,
                                  italic: bool, bold: bool,
                                  anchor: str | None) -> Tuple[float, float, float, float]:
        # Compute extents in mm using a scratch context (points → mm conversion).
        xb_mm, yb_mm, width_mm, height_mm = self._get_text_extents_mm(text, family, size_pt, italic, bold)
        if anchor:
            ax = x_mm
            ay = y_mm
            if anchor == 'center':
                rect_x = ax - width_mm / 2.0
                rect_y = ay - height_mm / 2.0
            elif anchor == 'n':
                rect_x = ax - width_mm / 2.0
                rect_y = ay
            elif anchor == 's':
                rect_x = ax - width_mm / 2.0
                rect_y = ay - height_mm
            elif anchor == 'w':
                rect_x = ax
                rect_y = ay - height_mm / 2.0
            elif anchor == 'e':
                rect_x = ax - width_mm
                rect_y = ay - height_mm / 2.0
            elif anchor == 'nw':
                rect_x = ax
                rect_y = ay
            elif anchor == 'ne':
                rect_x = ax - width_mm
                rect_y = ay
            elif anchor == 'sw':
                rect_x = ax
                rect_y = ay - height_mm
            elif anchor == 'se':
                rect_x = ax - width_mm
                rect_y = ay - height_mm
            else:
                rect_x = x_mm + xb_mm
                rect_y = y_mm + yb_mm
        else:
            # Baseline semantics
            rect_x = x_mm + xb_mm
            rect_y = y_mm + yb_mm
        return (rect_x, rect_y, width_mm, height_mm)

    def _get_text_extents_mm(self, text: str,
                              family: str, size_pt: float,
                              italic: bool, bold: bool) -> Tuple[float, float, float, float]:
        """Return (x_bearing_mm, y_bearing_mm, width_mm, height_mm) for given text settings."""
        surf = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1)
        ctx = cairo.Context(surf)
        slant = cairo.FONT_SLANT_ITALIC if italic else cairo.FONT_SLANT_NORMAL
        weight = cairo.FONT_WEIGHT_BOLD if bold else cairo.FONT_WEIGHT_NORMAL
        ctx.select_font_face(family, slant, weight)
        ctx.set_font_size(size_pt)  # extents in points
        te = ctx.text_extents(text)
        x_bearing_mm = te.x_bearing / PT_PER_MM
        y_bearing_mm = te.y_bearing / PT_PER_MM
        width_mm = te.width / PT_PER_MM
        height_mm = te.height / PT_PER_MM
        return (x_bearing_mm, y_bearing_mm, width_mm, height_mm)


def make_image_surface(width_px: int, height_px: int):
    """Create a QImage + cairo surface pair for rasterizing DrawUtil content."""
    width = max(1, int(width_px))
    height = max(1, int(height_px))
    stride = width * 4
    buf = bytearray(height * stride)
    surface = cairo.ImageSurface.create_for_data(buf, cairo.FORMAT_ARGB32, width, height, stride)
    image = QtGui.QImage(buf, width, height, stride, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
    return image, surface, buf


def finalize_image_surface(image: QtGui.QImage, device_pixel_ratio: float = 1.0) -> QtGui.QImage:
    """Detach a rasterized QImage from its temporary buffer and free the buffer."""
    if image is None:
        raise ValueError("finalize_image_surface() requires a valid QImage")
    final = image.copy()
    final.setDevicePixelRatio(float(device_pixel_ratio))
    image.swap(QtGui.QImage())
    return final
