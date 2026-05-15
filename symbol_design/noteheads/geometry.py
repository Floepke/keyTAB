from __future__ import annotations

import math
from functools import lru_cache


def _norm_round(v: float, ndigits: int = 6) -> float:
    return float(round(float(v), ndigits))


@lru_cache(maxsize=4096)
def _cached_sheared_outline_points(
    hand: str,
    is_up: bool,
    semitone_space_mm: float,
    width_scale: float,
    height_scale: float,
    base_tilt: float,
    sample_count: int,
) -> tuple[tuple[float, float], ...]:
    """Return sheared notehead outline points in local coordinates.

    Local frame:
    - x origin at note center
    - y origin at note's y1 anchor used by drawers
    """
    semitone = max(1e-6, _norm_round(semitone_space_mm))
    width = max(0.05, _norm_round(width_scale))
    height = max(0.1, _norm_round(height_scale))
    tilt = max(-1.0, min(1.0, _norm_round(base_tilt)))
    n = max(16, int(sample_count))

    rx = semitone * width
    ry = semitone * height
    tilt_local = -tilt if str(hand or "l") == "r" else tilt

    full_h = float(ry * 2.0)
    top = -full_h if bool(is_up) else 0.0
    cy = top + (full_h * 0.5)

    pts: list[tuple[float, float]] = []
    for i in range(n):
        t = (2.0 * math.pi * float(i)) / float(n)
        c = math.cos(t)
        s = math.sin(t)
        x_l = rx * c
        y_l = cy + (ry * s) + (tilt_local * rx * c)
        pts.append((float(x_l), float(y_l)))
    return tuple(pts)


def sheared_notehead_outline_points(
    *,
    hand: str,
    is_up: bool,
    semitone_space_mm: float,
    width_scale: float,
    height_scale: float,
    base_tilt: float,
    sample_count: int = 64,
) -> tuple[tuple[float, float], ...]:
    """Public wrapper for cached sheared notehead outline points."""
    return _cached_sheared_outline_points(
        str(hand or "l"),
        bool(is_up),
        _norm_round(semitone_space_mm),
        _norm_round(width_scale),
        _norm_round(height_scale),
        _norm_round(base_tilt),
        int(sample_count),
    )


def support_v_from_outline_points(
    points: tuple[tuple[float, float], ...],
    *,
    m_line: float,
    is_up: bool,
) -> float:
    """Return support value v=y-mx on an outline point set.

    For up noteheads the top support (max v) is used.
    For down noteheads the bottom support (min v) is used.
    """
    m = float(m_line)
    v_min = float("inf")
    v_max = float("-inf")
    for x_l, y_l in points:
        v = float(y_l - (m * x_l))
        if v < v_min:
            v_min = v
        if v > v_max:
            v_max = v
    return float(v_max if bool(is_up) else v_min)


def support_point_from_outline_points(
    points: tuple[tuple[float, float], ...],
    *,
    m_line: float,
    choose_max: bool,
) -> tuple[float, float]:
    """Return the outline point on the extreme parallel support line.

    For the line family y = m*x + b, this returns the point whose
    support value v = y - m*x is maximal or minimal.
    """
    m = float(m_line)
    best_point = (0.0, 0.0)
    best_v = float("-inf") if bool(choose_max) else float("inf")
    for x_l, y_l in points:
        v = float(y_l - (m * x_l))
        if bool(choose_max):
            if v > best_v:
                best_v = v
                best_point = (float(x_l), float(y_l))
        else:
            if v < best_v:
                best_v = v
                best_point = (float(x_l), float(y_l))
    return best_point


def sheared_notehead_support_v(
    *,
    hand: str,
    is_up: bool,
    semitone_space_mm: float,
    width_scale: float,
    height_scale: float,
    base_tilt: float,
    m_line: float,
    sample_count: int = 64,
) -> float:
    """Convenience helper: cached outline + support query."""
    points = sheared_notehead_outline_points(
        hand=str(hand or "l"),
        is_up=bool(is_up),
        semitone_space_mm=float(semitone_space_mm),
        width_scale=float(width_scale),
        height_scale=float(height_scale),
        base_tilt=float(base_tilt),
        sample_count=int(sample_count),
    )
    return support_v_from_outline_points(points, m_line=float(m_line), is_up=bool(is_up))
