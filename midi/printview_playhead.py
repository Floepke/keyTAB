"""Print-view playhead helpers.

Converts a playback time (in score units) to the page, x-span and y position
that should be highlighted as the playhead in the print (engraved) view.

The `print_time_map` used by these helpers is produced by the engraver and
stored as an attribute on `DrawUtil` after each successful engrave.  Its
structure is::

    [
        # one entry per page (indexed by page_index)
        [
            {
                'time_start': float,   # score units where this stave line starts
                'time_end':   float,   # score units where this stave line ends
                'y_top':      float,   # top of the printable area on this page (mm)
                'y_bottom':   float,   # bottom of the printable area (mm)
                'x_start':    float,   # left edge of the stave line (mm)
                'x_end':      float,   # right edge of the stave line (mm)
            },
            ...
        ],
        ...
    ]

Because the engraver maps time linearly along the y-axis within each stave
line (exactly like the editor does along its own y-axis), the playhead y
position is a simple linear interpolation between y_top and y_bottom.
"""

from __future__ import annotations
from typing import Optional


# ---------------------------------------------------------------------------
# Core query
# ---------------------------------------------------------------------------

def time_to_print_position(
    time_units: float,
    print_time_map: list,
) -> Optional[tuple[int, float, float, float]]:
    """Map *time_units* to its engraved location.

    Returns ``(page_index, y_mm, x_start_mm, x_end_mm)`` when a matching
    stave line is found, or ``None`` when the time is out of range.

    The search is performed in page order so the first match (the page where
    the time first appears) is returned.  Repeats are handled transparently
    because the caller supplies the *source* time already resolved by
    :meth:`Player.get_playhead_time`.
    """
    if not print_time_map:
        return None
    t = float(time_units)
    for page_index, lines in enumerate(print_time_map):
        for line in lines:
            t0 = float(line.get('time_start', 0.0))
            t1 = float(line.get('time_end', 0.0))
            if t1 <= t0:
                continue
            if t0 <= t <= t1:
                rel = max(0.0, min(1.0, (t - t0) / (t1 - t0)))
                y_top = float(line.get('y_top', 0.0))
                y_bot = float(line.get('y_bottom', 0.0))
                y_mm = y_top + (y_bot - y_top) * rel
                x1 = float(line.get('x_start', 0.0))
                x2 = float(line.get('x_end', 0.0))
                return (page_index, y_mm, x1, x2)
    return None


def build_print_time_map(du) -> list:
    """Return the ``print_time_map`` stored on *du*, or an empty list."""
    return list(getattr(du, 'print_time_map', None) or [])


def page_count_from_time_map(print_time_map: list) -> int:
    """Return the number of pages encoded in *print_time_map*."""
    return int(len(print_time_map or []))
