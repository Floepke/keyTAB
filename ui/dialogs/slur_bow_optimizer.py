from __future__ import annotations

import argparse
from dataclasses import fields
import math
from pathlib import Path
import random
from typing import Iterable

from PySide6 import QtCore, QtWidgets

from appdata_manager import get_appdata_manager
from file_model.SCORE import SCORE
from file_model.base_grid import resolve_grid_layer_offsets
from utils.CONSTANT import BE_KEYS, PIANO_KEY_AMOUNT, QUARTER_NOTE_UNIT


MAX_ITERATIONS_SAFE = 12
MAX_HIT_TEST_SAMPLES = 192


def _build_x_positions(page_width_mm: float) -> list[float]:
    margin = float(page_width_mm) / 6.0
    stave_width = float(page_width_mm) - (2.0 * margin)
    semitone_dist = stave_width / 101.0
    be_set = set(BE_KEYS)
    x_pos = margin - semitone_dist
    xs = [x_pos]
    for key_number in range(1, PIANO_KEY_AMOUNT + 1):
        if (key_number - 1) in be_set:
            x_pos += semitone_dist
        x_pos += semitone_dist
        xs.append(x_pos)
    return xs


class _Geometry:
    def __init__(self, score: SCORE):
        self.score = score
        self.layout = getattr(score, "layout", None)
        self.app_state = getattr(score, "app_state", None)
        self.page_width_mm = float(getattr(self.layout, "page_width_mm", 210.0) or 210.0)
        self.margin_mm = self.page_width_mm / 6.0
        self.stave_width_mm = self.page_width_mm - (2.0 * self.margin_mm)
        self.semitone_mm = self.stave_width_mm / 101.0
        self.zoom_mm_per_quarter = float(getattr(self.app_state, "zoom_mm_per_quarter", 25.0) or 25.0)
        self.x_positions = _build_x_positions(self.page_width_mm)
        self.base_x_c4 = float(self.x_positions[40])
        self._rpitch_bounds_cache: tuple[int, int] | None = None

    def time_to_mm(self, ticks: float) -> float:
        return self.margin_mm + (float(ticks) / float(QUARTER_NOTE_UNIT)) * self.zoom_mm_per_quarter

    def pitch_to_x(self, key_number: int) -> float:
        idx = max(1, min(PIANO_KEY_AMOUNT, int(key_number)))
        return float(self.x_positions[idx])

    def rpitch_to_x(self, rpitch: float) -> float:
        return float(self.base_x_c4) + float(rpitch) * float(self.semitone_mm)

    def rpitch_bounds_for_editor_width(self) -> tuple[int, int]:
        if self._rpitch_bounds_cache is not None:
            return self._rpitch_bounds_cache
        if abs(float(self.semitone_mm)) < 1e-9:
            self._rpitch_bounds_cache = (-128, 128)
            return self._rpitch_bounds_cache
        left_mm = 0.0
        right_mm = float(max(1.0, self.page_width_mm))
        min_rp = int(math.floor((left_mm - float(self.base_x_c4)) / float(self.semitone_mm)))
        max_rp = int(math.ceil((right_mm - float(self.base_x_c4)) / float(self.semitone_mm)))
        if min_rp > max_rp:
            min_rp, max_rp = max_rp, min_rp
        self._rpitch_bounds_cache = (min_rp, max_rp)
        return self._rpitch_bounds_cache


def _time_resolution_ticks(score: SCORE) -> float:
    app_state = getattr(score, "app_state", None)
    try:
        snap_base = int(getattr(app_state, "snap_base", 8) or 8)
    except Exception:
        snap_base = 8
    try:
        snap_divide = int(getattr(app_state, "snap_divide", 1) or 1)
    except Exception:
        snap_divide = 1
    snap_base = max(1, snap_base)
    snap_divide = max(1, snap_divide)
    # Matches SnapSizeSelector.get_snap_size(): (whole_note_ticks / base) / divide
    whole_note_ticks = float(QUARTER_NOTE_UNIT) * 4.0
    return max(1.0, whole_note_ticks / float(snap_base) / float(snap_divide))


def _estimate_total_time_ticks(score: SCORE) -> float:
    max_t = 0.0
    ev = getattr(score, "events", None)
    if ev is None:
        return float(QUARTER_NOTE_UNIT) * 8.0
    for event_name in [f.name for f in fields(type(ev)) if not f.name.startswith("_")]:
        items = list(getattr(ev, event_name, []) or [])
        for item in items:
            t0 = float(getattr(item, "time", 0.0) or 0.0)
            dur = float(getattr(item, "duration", 0.0) or 0.0)
            max_t = max(max_t, t0 + max(0.0, dur))
            for handle_name in ("y1_time", "y2_time", "y3_time", "y4_time"):
                if hasattr(item, handle_name):
                    max_t = max(max_t, float(getattr(item, handle_name, 0.0) or 0.0))
    if max_t <= 0.0:
        max_t = float(QUARTER_NOTE_UNIT) * 8.0
    return max_t


def _rect_intersection_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return (ix2 - ix1) * (iy2 - iy1)


def _rects_intersect(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def _slur_points_mm(geom: _Geometry, slur, sample_count: int) -> list[tuple[float, float, float]]:
    side_w = float(getattr(geom.layout, "slur_width_sides_mm", 0.5) or 0.5)
    mid_w = float(getattr(geom.layout, "slur_width_middle_mm", 2.5) or 2.5)

    def width_at(t: float) -> float:
        tri = max(0.0, 1.0 - abs(2.0 * t - 1.0))
        return side_w + (mid_w - side_w) * tri

    x1 = geom.rpitch_to_x(float(getattr(slur, "x1_rpitch", 0) or 0))
    x2 = geom.rpitch_to_x(float(getattr(slur, "x2_rpitch", 0) or 0))
    x3 = geom.rpitch_to_x(float(getattr(slur, "x3_rpitch", 0) or 0))
    x4 = geom.rpitch_to_x(float(getattr(slur, "x4_rpitch", 0) or 0))

    y1 = geom.time_to_mm(float(getattr(slur, "y1_time", 0.0) or 0.0))
    y2 = geom.time_to_mm(float(getattr(slur, "y2_time", 0.0) or 0.0))
    y3 = geom.time_to_mm(float(getattr(slur, "y3_time", 0.0) or 0.0))
    y4 = geom.time_to_mm(float(getattr(slur, "y4_time", 0.0) or 0.0))

    n = max(8, int(sample_count))
    points: list[tuple[float, float, float]] = []
    for i in range(n):
        t = float(i) / float(max(1, n - 1))
        omt = 1.0 - t
        bx = (
            omt * omt * omt * x1
            + 3.0 * omt * omt * t * x2
            + 3.0 * omt * t * t * x3
            + t * t * t * x4
        )
        by = (
            omt * omt * omt * y1
            + 3.0 * omt * omt * t * y2
            + 3.0 * omt * t * t * y3
            + t * t * t * y4
        )
        points.append((bx, by, width_at(t)))
    return points


def _slur_segment_boxes(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for i in range(len(points) - 1):
        xa, ya, wa = points[i]
        xb, yb, wb = points[i + 1]
        half_w = max(0.05, 0.5 * max(wa, wb))
        x1 = min(xa, xb) - half_w
        y1 = min(ya, yb) - half_w
        x2 = max(xa, xb) + half_w
        y2 = max(ya, yb) + half_w
        boxes.append((x1, y1, x2, y2))
    return boxes


def _build_obstacle_boxes(score: SCORE, geom: _Geometry, include_stave_lines: bool = True) -> list[tuple[float, float, float, float, float]]:
    obstacles: list[tuple[float, float, float, float, float]] = []
    total_time = _estimate_total_time_ticks(score)
    y_min = geom.margin_mm
    y_max = geom.time_to_mm(total_time) + geom.margin_mm

    if include_stave_lines:
        stave_w = max(0.08, float(getattr(geom.layout, "stave_two_line_thickness_mm", 0.5) or 0.5))
        half = 0.5 * stave_w
        for key_number in range(1, PIANO_KEY_AMOUNT + 1):
            x = geom.pitch_to_x(key_number)
            obstacles.append((x - half, y_min, x + half, y_max, 0.03))

    grace_notes = list(getattr(getattr(score, "events", None), "grace_note", []) or [])
    for note in grace_notes:
        x = geom.pitch_to_x(int(getattr(note, "pitch", 40) or 40))
        y = geom.time_to_mm(float(getattr(note, "time", 0.0) or 0.0))
        w = max(0.2, geom.semitone_mm * 0.7)
        obstacles.append((x - w, y - w, x + w, y + w, 0.8))

    texts = list(getattr(getattr(score, "events", None), "text", []) or [])
    default_size = float(getattr(getattr(geom.layout, "font_text", None), "size_pt", 12.0) or 12.0)
    padding = float(getattr(geom.layout, "text_background_padding_mm", 1.0) or 1.0)
    for tx in texts:
        text = str(getattr(tx, "text", "") or "")
        size_pt = default_size
        if bool(getattr(tx, "use_custom_font", False)) and getattr(tx, "font", None) is not None:
            try:
                size_pt = float(getattr(tx.font, "size_pt", default_size) or default_size)
            except Exception:
                size_pt = default_size
        width_mm = max(2.0, len(text) * size_pt * 0.18 + (padding * 2.0))
        height_mm = max(2.0, size_pt * 0.35 + (padding * 2.0))
        x = geom.rpitch_to_x(float(getattr(tx, "x_rpitch", 0.0) or 0.0)) + float(getattr(tx, "x_offset_mm", 0.0) or 0.0)
        y = geom.time_to_mm(float(getattr(tx, "time", 0.0) or 0.0)) + float(getattr(tx, "y_offset_mm", 0.0) or 0.0)
        obstacles.append((x - width_mm * 0.5, y - height_mm * 0.5, x + width_mm * 0.5, y + height_mm * 0.5, 0.9))

    tempos = list(getattr(getattr(score, "events", None), "tempo", []) or [])
    for tp in tempos:
        t0 = float(getattr(tp, "time", 0.0) or 0.0)
        y = geom.time_to_mm(t0)
        x = geom.pitch_to_x(40)
        obstacles.append((x - 3.0, y - 1.5, x + 8.0, y + 1.5, 0.5))

    count_lines = list(getattr(getattr(score, "events", None), "count_line", []) or [])
    for cl in count_lines:
        y = geom.time_to_mm(float(getattr(cl, "time", 0.0) or 0.0))
        x1 = geom.rpitch_to_x(float(getattr(cl, "rpitch1", 0) or 0))
        x2 = geom.rpitch_to_x(float(getattr(cl, "rpitch2", 4) or 4))
        obstacles.append((min(x1, x2) - 0.5, y - 0.5, max(x1, x2) + 0.5, y + 0.5, 0.4))

    return obstacles


def _assign_groups_by_windows(notes_sorted: list, windows: list[tuple[float, float]]) -> list[list]:
    if not notes_sorted or not windows:
        return []
    starts = [float(getattr(n, "time", 0.0) or 0.0) for n in notes_sorted]
    ends = [float(getattr(n, "time", 0.0) or 0.0) + float(getattr(n, "duration", 0.0) or 0.0) for n in notes_sorted]
    groups: list[list] = []
    for w0, w1 in windows:
        grp: list = []
        for idx, st in enumerate(starts):
            en = ends[idx]
            if st < float(w1) and en > float(w0):
                grp.append(notes_sorted[idx])
        if grp:
            try:
                grp = sorted({int(getattr(n, "_id", id(n))): n for n in grp}.values(), key=lambda n: float(getattr(n, "time", 0.0) or 0.0))
            except Exception:
                grp = sorted(grp, key=lambda n: float(getattr(n, "time", 0.0) or 0.0))
        groups.append(grp)
    return groups


def _build_grid_windows_for_beams(score: SCORE) -> list[tuple[float, float]]:
    base_grid = list(getattr(score, "base_grid", []) or [])
    windows: list[tuple[float, float]] = []
    cursor = 0.0
    for bg in base_grid:
        numer = int(getattr(bg, "numerator", 4) or 4)
        denom = int(getattr(bg, "denominator", 4) or 4)
        measures = int(getattr(bg, "measure_amount", 1) or 1)
        seq = list(getattr(bg, "beat_grouping", []) or [])
        measure_len = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
        _bar_offsets, grid_offsets = resolve_grid_layer_offsets(seq, numer, denom)
        boundaries_local = [0.0] + [float(v) for v in grid_offsets if 0.0 < float(v) < measure_len] + [float(measure_len)]
        boundaries_local = sorted(dict.fromkeys(round(v, 6) for v in boundaries_local))
        if len(boundaries_local) < 2:
            boundaries_local = [0.0, float(measure_len)]
        for _ in range(max(0, measures)):
            for i in range(len(boundaries_local) - 1):
                w0 = float(cursor) + float(boundaries_local[i])
                w1 = float(cursor) + float(boundaries_local[i + 1])
                if w1 > w0:
                    windows.append((w0, w1))
            cursor += measure_len
    return windows


def _line_bbox(x1: float, y1: float, x2: float, y2: float, half_w: float) -> tuple[float, float, float, float]:
    return (
        min(float(x1), float(x2)) - float(half_w),
        min(float(y1), float(y2)) - float(half_w),
        max(float(x1), float(x2)) + float(half_w),
        max(float(y1), float(y2)) + float(half_w),
    )


def _build_beam_obstacle_boxes(score: SCORE, geom: _Geometry) -> list[tuple[float, float, float, float, float]]:
    layout = getattr(score, "layout", None)
    if not bool(getattr(layout, "beam_visible", True)):
        return []

    notes = list(getattr(getattr(score, "events", None), "note", []) or [])
    if not notes:
        return []

    notes_by_hand: dict[str, list] = {"l": [], "r": []}
    for n in notes:
        hand = str(getattr(n, "hand", "<") or "<")
        key = "l" if hand in ("<", "l") else "r"
        notes_by_hand[key].append(n)

    for hand in notes_by_hand:
        notes_by_hand[hand] = sorted(notes_by_hand[hand], key=lambda n: float(getattr(n, "time", 0.0) or 0.0))

    windows = _build_grid_windows_for_beams(score)
    if not windows:
        return []

    groups_l = _assign_groups_by_windows(notes_by_hand.get("l") or [], windows)
    groups_r = _assign_groups_by_windows(notes_by_hand.get("r") or [], windows)

    stem_len = float(getattr(layout, "note_stem_length_semitone", 5) or 5) * float(geom.semitone_mm)
    beam_thick = max(0.3, float(getattr(layout, "beam_thickness_mm", 2.5) or 2.5))
    half_w = 0.5 * beam_thick
    dx = float(geom.semitone_mm)

    obstacles: list[tuple[float, float, float, float, float]] = []

    def _add_for_group(grp: list, hand_key: str) -> None:
        if not grp:
            return
        starts = sorted({float(getattr(n, "time", 0.0) or 0.0) for n in grp})
        if len(starts) < 2:
            return
        t_first = starts[0]
        t_last = starts[-1]
        y1 = geom.time_to_mm(t_first)
        y2 = geom.time_to_mm(t_last)

        if hand_key == "r":
            pitch = max(int(getattr(n, "pitch", 40) or 40) for n in grp)
            x1 = geom.pitch_to_x(pitch) + stem_len
            x2 = x1 + dx
        else:
            pitch = min(int(getattr(n, "pitch", 40) or 40) for n in grp)
            x1 = geom.pitch_to_x(pitch) - stem_len
            x2 = x1 - dx

        bx1, by1, bx2, by2 = _line_bbox(x1, y1, x2, y2, half_w=half_w)
        obstacles.append((bx1, by1, bx2, by2, 1.25))

    for grp in groups_r:
        _add_for_group(grp, "r")
    for grp in groups_l:
        _add_for_group(grp, "l")

    return obstacles


def _slur_collision_score(slur,
                         geom: _Geometry,
                         obstacles: Iterable[tuple[float, float, float, float, float]],
                         sample_count: int,
                         score_ceiling: float | None = None) -> float:
    points = _slur_points_mm(geom, slur, sample_count=sample_count)
    seg_boxes = _slur_segment_boxes(points)
    if not seg_boxes:
        return 0.0

    sx1 = min(b[0] for b in seg_boxes)
    sy1 = min(b[1] for b in seg_boxes)
    sx2 = max(b[2] for b in seg_boxes)
    sy2 = max(b[3] for b in seg_boxes)
    slur_bbox = (sx1, sy1, sx2, sy2)

    score = 0.0
    slur_id = int(getattr(slur, "_id", 0) or 0)
    for obstacle in obstacles:
        ox1, oy1, ox2, oy2, weight = obstacle
        ob = (ox1, oy1, ox2, oy2)
        if not _rects_intersect(slur_bbox, ob):
            continue
        for sb in seg_boxes:
            area = _rect_intersection_area(sb, ob)
            if area > 0.0:
                score += float(weight) * area
                if score_ceiling is not None and score >= float(score_ceiling):
                    return score

    # Light regularizer to avoid over-stretching very wide slurs.
    x_values = [
        float(getattr(slur, "x1_rpitch", 0) or 0),
        float(getattr(slur, "x2_rpitch", 0) or 0),
        float(getattr(slur, "x3_rpitch", 0) or 0),
        float(getattr(slur, "x4_rpitch", 0) or 0),
    ]
    span = max(x_values) - min(x_values)
    score += 0.02 * span
    _ = slur_id
    return score


def _openness_from_collision(collision_value: float) -> float:
    val = max(0.0, float(collision_value))
    return 100.0 / (1.0 + val)


def _slur_style_adjustment(
    slur,
    time_resolution_ticks: float,
    *,
    pointiness_weight: float,
    symmetry_weight: float,
    pointiness_bonus_points: float,
    symmetry_bonus_points: float,
    antisymmetry_penalty_points: float,
    straight_line_penalty_points: float,
    neighbor_connection_bonus_points: float,
    control_y_pass_penalty_points: float,
    neighbor_slurs: list,
) -> float:
    """Return additive objective adjustment (positive=penalty, negative=bonus)."""
    adj = 0.0

    x1 = float(getattr(slur, "x1_rpitch", 0) or 0)
    x2 = float(getattr(slur, "x2_rpitch", 0) or 0)
    x3 = float(getattr(slur, "x3_rpitch", 0) or 0)
    x4 = float(getattr(slur, "x4_rpitch", 0) or 0)
    y1 = float(getattr(slur, "y1_time", 0.0) or 0.0)
    y2 = float(getattr(slur, "y2_time", 0.0) or 0.0)
    y3 = float(getattr(slur, "y3_time", 0.0) or 0.0)
    y4 = float(getattr(slur, "y4_time", 0.0) or 0.0)

    # Pointiness: max when control-Y matches its anchor-Y.
    tr = max(1.0, float(time_resolution_ticks))
    p1 = max(0.0, 1.0 - (abs(y2 - y1) / tr))
    p2 = max(0.0, 1.0 - (abs(y3 - y4) / tr))
    pointiness_raw = 0.5 * (p1 + p2)
    adj -= float(pointiness_weight) * float(pointiness_bonus_points) * float(pointiness_raw)

    # Symmetry from relative control offsets around red anchors.
    d_left = float(x2 - x1)
    d_right = float(x4 - x3)
    denom = max(1e-6, abs(d_left) + abs(d_right))
    symmetry_raw = max(0.0, 1.0 - (abs(d_left - d_right) / denom))
    adj -= float(symmetry_weight) * float(symmetry_bonus_points) * float(symmetry_raw)

    # Anti-symmetry: opposite offset direction around anchors.
    if (d_left * d_right) < 0.0:
        adj += float(antisymmetry_penalty_points)

    # Too-straight/flat near zero control offsets.
    straight_threshold = 2.0
    straight_measure = abs(d_left) + abs(d_right)
    if straight_measure < straight_threshold:
        frac = 1.0 - (straight_measure / straight_threshold)
        adj += float(straight_line_penalty_points) * max(0.0, min(1.0, frac))

    # Hard constraint: controls cannot pass their red neighbors in Y direction.
    # For increasing-time slurs y1->y4: y2>=y1 and y3<=y4. Reverse when y4<y1.
    if y4 >= y1:
        if y2 < y1 or y3 > y4:
            adj += float(control_y_pass_penalty_points)
    else:
        if y2 > y1 or y3 < y4:
            adj += float(control_y_pass_penalty_points)

    # Neighbor connection bonus on red handles (time-near endpoint continuity).
    conn_count = 0
    for nb in neighbor_slurs or []:
        if nb is slur:
            continue
        ny1 = float(getattr(nb, "y1_time", 0.0) or 0.0)
        ny4 = float(getattr(nb, "y4_time", 0.0) or 0.0)
        if abs(y4 - ny1) <= tr:
            conn_count += 1
        elif abs(y1 - ny4) <= tr:
            conn_count += 1
    if conn_count > 0:
        adj -= float(neighbor_connection_bonus_points) * float(conn_count)

    return float(adj)


def _slur_objective_score(
    slur,
    geom: _Geometry,
    obstacles: Iterable[tuple[float, float, float, float, float]],
    sample_count: int,
    *,
    time_resolution_ticks: float,
    pointiness_weight: float,
    symmetry_weight: float,
    pointiness_bonus_points: float,
    symmetry_bonus_points: float,
    antisymmetry_penalty_points: float,
    straight_line_penalty_points: float,
    neighbor_connection_bonus_points: float,
    control_y_pass_penalty_points: float,
    neighbor_slurs: list,
    score_ceiling: float | None = None,
) -> float:
    collision = _slur_collision_score(
        slur,
        geom,
        obstacles,
        sample_count,
        score_ceiling=score_ceiling,
    )
    style_adj = _slur_style_adjustment(
        slur,
        time_resolution_ticks,
        pointiness_weight=pointiness_weight,
        symmetry_weight=symmetry_weight,
        pointiness_bonus_points=pointiness_bonus_points,
        symmetry_bonus_points=symmetry_bonus_points,
        antisymmetry_penalty_points=antisymmetry_penalty_points,
        straight_line_penalty_points=straight_line_penalty_points,
        neighbor_connection_bonus_points=neighbor_connection_bonus_points,
        control_y_pass_penalty_points=control_y_pass_penalty_points,
        neighbor_slurs=neighbor_slurs,
    )
    return float(collision + style_adj)


def _optimize_one_slur(slur, geom: _Geometry, obstacles: list[tuple[float, float, float, float, float]],
                       sample_count: int,
                       anchor_range: int,
                       control_range: int,
                       max_iterations: int,
                       *,
                       time_resolution_ticks: float,
                       pointiness_weight: float,
                       symmetry_weight: float,
                       pointiness_bonus_points: float,
                       symmetry_bonus_points: float,
                       antisymmetry_penalty_points: float,
                       straight_line_penalty_points: float,
                       neighbor_connection_bonus_points: float,
                       control_y_pass_penalty_points: float,
                       neighbor_slurs: list) -> bool:
    original = {
        "x1": int(getattr(slur, "x1_rpitch", 0) or 0),
        "x2": int(getattr(slur, "x2_rpitch", 0) or 0),
        "x3": int(getattr(slur, "x3_rpitch", 0) or 0),
        "x4": int(getattr(slur, "x4_rpitch", 0) or 0),
    }

    def get_xs() -> dict[str, int]:
        return {
            "x1": int(getattr(slur, "x1_rpitch", 0) or 0),
            "x2": int(getattr(slur, "x2_rpitch", 0) or 0),
            "x3": int(getattr(slur, "x3_rpitch", 0) or 0),
            "x4": int(getattr(slur, "x4_rpitch", 0) or 0),
        }

    def set_handle(name: str, value: int) -> None:
        if name == "x1":
            slur.x1_rpitch = int(value)
        elif name == "x2":
            slur.x2_rpitch = int(value)
        elif name == "x3":
            slur.x3_rpitch = int(value)
        elif name == "x4":
            slur.x4_rpitch = int(value)

    def restore(xs: dict[str, int]) -> None:
        slur.x1_rpitch = int(xs["x1"])
        slur.x2_rpitch = int(xs["x2"])
        slur.x3_rpitch = int(xs["x3"])
        slur.x4_rpitch = int(xs["x4"])

    # Safety clamps to avoid runaway optimization settings.
    sample_count = max(8, min(int(sample_count), int(MAX_HIT_TEST_SAMPLES)))
    anchor_range = max(0, int(anchor_range))
    control_range = max(0, int(control_range))
    max_iterations = max(1, min(int(max_iterations), int(MAX_ITERATIONS_SAFE)))

    # Respect editor X bounds for handle search.
    rp_min_bound, rp_max_bound = geom.rpitch_bounds_for_editor_width()

    # Adapt sample density to snap/time resolution and slur duration.
    span_ticks = max(
        0.0,
        float(max(
            float(getattr(slur, "y1_time", 0.0) or 0.0),
            float(getattr(slur, "y2_time", 0.0) or 0.0),
            float(getattr(slur, "y3_time", 0.0) or 0.0),
            float(getattr(slur, "y4_time", 0.0) or 0.0),
        )) - float(min(
            float(getattr(slur, "y1_time", 0.0) or 0.0),
            float(getattr(slur, "y2_time", 0.0) or 0.0),
            float(getattr(slur, "y3_time", 0.0) or 0.0),
            float(getattr(slur, "y4_time", 0.0) or 0.0),
        )),
    )
    adaptive_samples = max(8, int(math.ceil(span_ticks / max(1.0, float(time_resolution_ticks)))) + 2)
    sample_count = max(8, min(sample_count, adaptive_samples, int(MAX_HIT_TEST_SAMPLES)))

    # Pre-filter obstacles to likely collision neighborhood for this slur.
    y_vals = [
        geom.time_to_mm(float(getattr(slur, "y1_time", 0.0) or 0.0)),
        geom.time_to_mm(float(getattr(slur, "y2_time", 0.0) or 0.0)),
        geom.time_to_mm(float(getattr(slur, "y3_time", 0.0) or 0.0)),
        geom.time_to_mm(float(getattr(slur, "y4_time", 0.0) or 0.0)),
    ]
    x_ranges = [
        (float(getattr(slur, "x1_rpitch", 0) or 0), anchor_range),
        (float(getattr(slur, "x2_rpitch", 0) or 0), control_range),
        (float(getattr(slur, "x3_rpitch", 0) or 0), control_range),
        (float(getattr(slur, "x4_rpitch", 0) or 0), anchor_range),
    ]
    x_vals_min = [geom.rpitch_to_x(center - radius) for center, radius in x_ranges]
    x_vals_max = [geom.rpitch_to_x(center + radius) for center, radius in x_ranges]
    x_min = min(x_vals_min) - 2.0
    x_max = max(x_vals_max) + 2.0
    y_min = min(y_vals) - 3.0
    y_max = max(y_vals) + 3.0
    search_bbox = (x_min, y_min, x_max, y_max)
    local_obstacles = [
        ob for ob in obstacles
        if _rects_intersect((float(ob[0]), float(ob[1]), float(ob[2]), float(ob[3])), search_bbox)
    ]
    if not local_obstacles:
        local_obstacles = list(obstacles)

    current = get_xs()
    best_score = _slur_objective_score(
        slur,
        geom,
        local_obstacles,
        sample_count,
        time_resolution_ticks=float(time_resolution_ticks),
        pointiness_weight=float(pointiness_weight),
        symmetry_weight=float(symmetry_weight),
        pointiness_bonus_points=float(pointiness_bonus_points),
        symmetry_bonus_points=float(symmetry_bonus_points),
        antisymmetry_penalty_points=float(antisymmetry_penalty_points),
        straight_line_penalty_points=float(straight_line_penalty_points),
        neighbor_connection_bonus_points=float(neighbor_connection_bonus_points),
        control_y_pass_penalty_points=float(control_y_pass_penalty_points),
        neighbor_slurs=neighbor_slurs,
    )
    changed = False

    handle_order = ["x1", "x2", "x3", "x4"]
    for _ in range(max(1, int(max_iterations))):
        improved = False
        for name in handle_order:
            center = int(current[name])
            radius = int(control_range if name in ("x2", "x3") else anchor_range)
            min_candidate = max(int(center - radius), int(rp_min_bound))
            max_candidate = min(int(center + radius), int(rp_max_bound))
            if min_candidate > max_candidate:
                continue
            local_best_val = center
            local_best_score = best_score

            for candidate in range(min_candidate, max_candidate + 1):
                if candidate == center:
                    continue
                set_handle(name, candidate)
                c_score = _slur_objective_score(
                    slur,
                    geom,
                    local_obstacles,
                    sample_count,
                    time_resolution_ticks=float(time_resolution_ticks),
                    pointiness_weight=float(pointiness_weight),
                    symmetry_weight=float(symmetry_weight),
                    pointiness_bonus_points=float(pointiness_bonus_points),
                    symmetry_bonus_points=float(symmetry_bonus_points),
                    antisymmetry_penalty_points=float(antisymmetry_penalty_points),
                    straight_line_penalty_points=float(straight_line_penalty_points),
                    neighbor_connection_bonus_points=float(neighbor_connection_bonus_points),
                    control_y_pass_penalty_points=float(control_y_pass_penalty_points),
                    neighbor_slurs=neighbor_slurs,
                    score_ceiling=local_best_score,
                )
                if c_score < local_best_score:
                    local_best_score = c_score
                    local_best_val = candidate

            set_handle(name, local_best_val)
            current[name] = int(local_best_val)
            if local_best_score < best_score:
                best_score = local_best_score
                improved = True

        if not improved:
            break

    if current != original:
        changed = True
    if not changed:
        restore(original)
    return changed


def optimize_slur_bows_in_score(score: SCORE, *,
                                hit_test_finetune: int = 64,
                                anchor_range: int = 10,
                                control_range: int = 16,
                                max_iterations: int = 3,
                                include_stave_lines: bool = True,
                                include_beams: bool = True,
                                max_slurs_from_start: int | None = None,
                                pointiness_weight: float = 0.5,
                                symmetry_weight: float = 0.5,
                                pointiness_bonus_points: float = 100.0,
                                symmetry_bonus_points: float = 100.0,
                                antisymmetry_penalty_points: float = 100.0,
                                straight_line_penalty_points: float = 100.0,
                                neighbor_connection_bonus_points: float = 100.0,
                                control_y_pass_penalty_points: float = 1000.0) -> dict[str, int | float]:
    """Optimize slur bow handle X positions in-place on a SCORE object.

    Time-Locked Endpoints rule:
    - y1_time and y4_time are not modified.

    Only x1/x2/x3/x4 relative-pitch positions may be remapped.
    """
    if score is None:
        return {"slurs_total": 0, "slurs_changed": 0}

    effective_hit_test = max(8, min(int(hit_test_finetune), int(MAX_HIT_TEST_SAMPLES)))
    effective_iterations = max(1, min(int(max_iterations), int(MAX_ITERATIONS_SAFE)))

    geom = _Geometry(score)
    obstacles = _build_obstacle_boxes(score, geom, include_stave_lines=include_stave_lines)
    if include_beams:
        try:
            obstacles.extend(_build_beam_obstacle_boxes(score, geom))
        except Exception:
            pass
    slurs_all = list(getattr(getattr(score, "events", None), "slur", []) or [])
    try:
        slurs_all = sorted(slurs_all, key=lambda sl: float(getattr(sl, "y1_time", 0.0) or 0.0))
    except Exception:
        pass

    if max_slurs_from_start is None:
        slurs = slurs_all
    else:
        limit = max(0, int(max_slurs_from_start))
        slurs = slurs_all[:limit]

    time_resolution_ticks = _time_resolution_ticks(score)

    changed_count = 0
    collision_before = 0.0
    collision_after = 0.0
    objective_before = 0.0
    objective_after = 0.0
    for idx, sl in enumerate(slurs):
        lo = max(0, idx - 4)
        hi = min(len(slurs), idx + 5)
        neighbors = [s for j, s in enumerate(slurs[lo:hi], start=lo) if j != idx]
        before = _slur_collision_score(sl, geom, obstacles, sample_count=effective_hit_test)
        before_obj = _slur_objective_score(
            sl,
            geom,
            obstacles,
            sample_count=effective_hit_test,
            time_resolution_ticks=float(time_resolution_ticks),
            pointiness_weight=float(pointiness_weight),
            symmetry_weight=float(symmetry_weight),
            pointiness_bonus_points=float(pointiness_bonus_points),
            symmetry_bonus_points=float(symmetry_bonus_points),
            antisymmetry_penalty_points=float(antisymmetry_penalty_points),
            straight_line_penalty_points=float(straight_line_penalty_points),
            neighbor_connection_bonus_points=float(neighbor_connection_bonus_points),
            control_y_pass_penalty_points=float(control_y_pass_penalty_points),
            neighbor_slurs=neighbors,
        )
        changed = _optimize_one_slur(
            sl,
            geom,
            obstacles,
            sample_count=effective_hit_test,
            anchor_range=max(0, int(anchor_range)),
            control_range=max(0, int(control_range)),
            max_iterations=effective_iterations,
            time_resolution_ticks=float(time_resolution_ticks),
            pointiness_weight=float(pointiness_weight),
            symmetry_weight=float(symmetry_weight),
            pointiness_bonus_points=float(pointiness_bonus_points),
            symmetry_bonus_points=float(symmetry_bonus_points),
            antisymmetry_penalty_points=float(antisymmetry_penalty_points),
            straight_line_penalty_points=float(straight_line_penalty_points),
            neighbor_connection_bonus_points=float(neighbor_connection_bonus_points),
            control_y_pass_penalty_points=float(control_y_pass_penalty_points),
            neighbor_slurs=neighbors,
        )
        after = _slur_collision_score(sl, geom, obstacles, sample_count=effective_hit_test)
        after_obj = _slur_objective_score(
            sl,
            geom,
            obstacles,
            sample_count=effective_hit_test,
            time_resolution_ticks=float(time_resolution_ticks),
            pointiness_weight=float(pointiness_weight),
            symmetry_weight=float(symmetry_weight),
            pointiness_bonus_points=float(pointiness_bonus_points),
            symmetry_bonus_points=float(symmetry_bonus_points),
            antisymmetry_penalty_points=float(antisymmetry_penalty_points),
            straight_line_penalty_points=float(straight_line_penalty_points),
            neighbor_connection_bonus_points=float(neighbor_connection_bonus_points),
            control_y_pass_penalty_points=float(control_y_pass_penalty_points),
            neighbor_slurs=neighbors,
        )
        collision_before += float(before)
        collision_after += float(after)
        objective_before += float(before_obj)
        objective_after += float(after_obj)
        if changed:
            changed_count += 1

    avg_before = (collision_before / float(len(slurs))) if slurs else 0.0
    avg_after = (collision_after / float(len(slurs))) if slurs else 0.0
    avg_objective_before = (objective_before / float(len(slurs))) if slurs else 0.0
    avg_objective_after = (objective_after / float(len(slurs))) if slurs else 0.0
    openness_before = _openness_from_collision(avg_before)
    openness_after = _openness_from_collision(avg_after)

    return {
        "slurs_total": len(slurs),
        "slurs_available": len(slurs_all),
        "slurs_changed": int(changed_count),
        "effective_hit_test_finetune": int(effective_hit_test),
        "effective_max_iterations": int(effective_iterations),
        "collision_before_total": float(collision_before),
        "collision_after_total": float(collision_after),
        "collision_before_avg": float(avg_before),
        "collision_after_avg": float(avg_after),
        "objective_before_avg": float(avg_objective_before),
        "objective_after_avg": float(avg_objective_after),
        "openness_before": float(openness_before),
        "openness_after": float(openness_after),
        "openness_delta": float(openness_after - openness_before),
    }


def optimize_file(input_path: str, output_path: str | None = None, *,
                  hit_test_finetune: int = 64,
                  anchor_range: int = 10,
                  control_range: int = 16,
                  max_iterations: int = 3,
                  include_stave_lines: bool = True,
                  include_beams: bool = True,
                  max_slurs_from_start: int | None = None,
                  pointiness_weight: float = 0.5,
                  symmetry_weight: float = 0.5,
                  pointiness_bonus_points: float = 100.0,
                  symmetry_bonus_points: float = 100.0,
                  antisymmetry_penalty_points: float = 100.0,
                  straight_line_penalty_points: float = 100.0,
                  neighbor_connection_bonus_points: float = 100.0,
                  control_y_pass_penalty_points: float = 1000.0,
                  in_place: bool = False) -> dict[str, int | float | str]:
    source = Path(str(input_path)).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Input file does not exist: {source}")

    score = SCORE().load(str(source))
    stats = optimize_slur_bows_in_score(
        score,
        hit_test_finetune=hit_test_finetune,
        anchor_range=anchor_range,
        control_range=control_range,
        max_iterations=max_iterations,
        include_stave_lines=include_stave_lines,
        include_beams=include_beams,
        max_slurs_from_start=max_slurs_from_start,
        pointiness_weight=pointiness_weight,
        symmetry_weight=symmetry_weight,
        pointiness_bonus_points=pointiness_bonus_points,
        symmetry_bonus_points=symmetry_bonus_points,
        antisymmetry_penalty_points=antisymmetry_penalty_points,
        straight_line_penalty_points=straight_line_penalty_points,
        neighbor_connection_bonus_points=neighbor_connection_bonus_points,
        control_y_pass_penalty_points=control_y_pass_penalty_points,
    )

    if in_place:
        target = source
    elif output_path:
        target = Path(str(output_path)).expanduser().resolve()
    else:
        target = source.with_name(f"{source.stem}.slur_optimized{source.suffix}")

    score.save(str(target))
    return {
        "input": str(source),
        "output": str(target),
        "slurs_total": int(stats["slurs_total"]),
        "slurs_changed": int(stats["slurs_changed"]),
    }


class SlurBowOptimizerDialog(QtWidgets.QDialog):
    applyRequested = QtCore.Signal(dict)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Slur bow optimizer")
        self.setModal(False)
        self._appdata_key = "slur_bow_optimizer_settings"

        root = QtWidgets.QVBoxLayout(self)

        self.info_label = QtWidgets.QLabel("Adjust parameters and click Apply to run optimizer.", self)
        self.info_label.setWordWrap(True)
        self.info_label.setTextInteractionFlags(QtCore.Qt.TextInteractionFlag.TextSelectableByMouse)
        root.addWidget(self.info_label)

        search_group = QtWidgets.QGroupBox("Search")
        search_form = QtWidgets.QFormLayout(search_group)

        self.hit_test_spin = QtWidgets.QSpinBox(self)
        self.hit_test_spin.setRange(0, 1000)
        self.hit_test_spin.setSingleStep(8)
        self.hit_test_spin.setValue(64)
        search_form.addRow("Hit-test finetune", self.hit_test_spin)

        self.iter_spin = QtWidgets.QSpinBox(self)
        self.iter_spin.setRange(1, int(MAX_ITERATIONS_SAFE))
        self.iter_spin.setValue(3)
        search_form.addRow("Max iterations", self.iter_spin)

        self.max_slurs_spin = QtWidgets.QSpinBox(self)
        self.max_slurs_spin.setRange(1, 100000)
        self.max_slurs_spin.setValue(12)
        search_form.addRow("Slurs from start", self.max_slurs_spin)

        root.addWidget(search_group)

        handle_group = QtWidgets.QGroupBox("Handle ranges")
        handle_form = QtWidgets.QFormLayout(handle_group)

        self.anchor_spin = QtWidgets.QSpinBox(self)
        self.anchor_spin.setRange(0, 64)
        self.anchor_spin.setValue(10)
        handle_form.addRow("Anchor range (x1/x4)", self.anchor_spin)

        self.control_spin = QtWidgets.QSpinBox(self)
        self.control_spin.setRange(0, 64)
        self.control_spin.setValue(16)
        handle_form.addRow("Control range (x2/x3)", self.control_spin)

        root.addWidget(handle_group)

        obstacle_group = QtWidgets.QGroupBox("Obstacles")
        obstacle_layout = QtWidgets.QVBoxLayout(obstacle_group)
        self.include_stave_lines_cb = QtWidgets.QCheckBox("Include stave lines in collision testing", self)
        self.include_stave_lines_cb.setChecked(True)
        obstacle_layout.addWidget(self.include_stave_lines_cb)
        self.include_beams_cb = QtWidgets.QCheckBox("Include beam lines in collision testing", self)
        self.include_beams_cb.setChecked(True)
        obstacle_layout.addWidget(self.include_beams_cb)
        root.addWidget(obstacle_group)

        aesthetics_group = QtWidgets.QGroupBox("Aesthetics scoring")
        aesthetics_form = QtWidgets.QFormLayout(aesthetics_group)

        self.pointiness_weight_spin = QtWidgets.QDoubleSpinBox(self)
        self.pointiness_weight_spin.setRange(0.0, 1.0)
        self.pointiness_weight_spin.setSingleStep(0.05)
        self.pointiness_weight_spin.setDecimals(2)
        self.pointiness_weight_spin.setValue(0.50)
        aesthetics_form.addRow("Pointiness weight", self.pointiness_weight_spin)

        self.symmetry_weight_spin = QtWidgets.QDoubleSpinBox(self)
        self.symmetry_weight_spin.setRange(0.0, 1.0)
        self.symmetry_weight_spin.setSingleStep(0.05)
        self.symmetry_weight_spin.setDecimals(2)
        self.symmetry_weight_spin.setValue(0.50)
        aesthetics_form.addRow("Symmetry weight", self.symmetry_weight_spin)

        self.pointiness_points_spin = QtWidgets.QDoubleSpinBox(self)
        self.pointiness_points_spin.setRange(0.0, 5000.0)
        self.pointiness_points_spin.setSingleStep(10.0)
        self.pointiness_points_spin.setDecimals(1)
        self.pointiness_points_spin.setValue(100.0)
        aesthetics_form.addRow("Pointiness bonus pts", self.pointiness_points_spin)

        self.symmetry_points_spin = QtWidgets.QDoubleSpinBox(self)
        self.symmetry_points_spin.setRange(0.0, 5000.0)
        self.symmetry_points_spin.setSingleStep(10.0)
        self.symmetry_points_spin.setDecimals(1)
        self.symmetry_points_spin.setValue(100.0)
        aesthetics_form.addRow("Symmetry bonus pts", self.symmetry_points_spin)

        self.antisym_penalty_spin = QtWidgets.QDoubleSpinBox(self)
        self.antisym_penalty_spin.setRange(0.0, 5000.0)
        self.antisym_penalty_spin.setSingleStep(10.0)
        self.antisym_penalty_spin.setDecimals(1)
        self.antisym_penalty_spin.setValue(100.0)
        aesthetics_form.addRow("Anti-symmetry penalty", self.antisym_penalty_spin)

        self.straight_penalty_spin = QtWidgets.QDoubleSpinBox(self)
        self.straight_penalty_spin.setRange(0.0, 5000.0)
        self.straight_penalty_spin.setSingleStep(10.0)
        self.straight_penalty_spin.setDecimals(1)
        self.straight_penalty_spin.setValue(100.0)
        aesthetics_form.addRow("Too-straight penalty", self.straight_penalty_spin)

        self.neighbor_bonus_spin = QtWidgets.QDoubleSpinBox(self)
        self.neighbor_bonus_spin.setRange(0.0, 5000.0)
        self.neighbor_bonus_spin.setSingleStep(10.0)
        self.neighbor_bonus_spin.setDecimals(1)
        self.neighbor_bonus_spin.setValue(100.0)
        aesthetics_form.addRow("Neighbor connection bonus", self.neighbor_bonus_spin)

        self.control_y_penalty_spin = QtWidgets.QDoubleSpinBox(self)
        self.control_y_penalty_spin.setRange(0.0, 20000.0)
        self.control_y_penalty_spin.setSingleStep(50.0)
        self.control_y_penalty_spin.setDecimals(1)
        self.control_y_penalty_spin.setValue(1000.0)
        aesthetics_form.addRow("Control-Y pass penalty", self.control_y_penalty_spin)

        root.addWidget(aesthetics_group)

        buttons = QtWidgets.QDialogButtonBox(parent=self)
        self.apply_button = buttons.addButton(
            QtWidgets.QDialogButtonBox.StandardButton.Apply
        )
        self.random_apply_button = buttons.addButton("Random Apply", QtWidgets.QDialogButtonBox.ButtonRole.ActionRole)
        self.close_button = buttons.addButton(
            QtWidgets.QDialogButtonBox.StandardButton.Close
        )
        self.apply_button.clicked.connect(self._on_apply_clicked)
        self.random_apply_button.clicked.connect(self._on_random_apply_clicked)
        self.close_button.clicked.connect(self.close)
        root.addWidget(buttons)

        self._load_from_appdata()

    def settings(self) -> dict[str, int | float | bool]:
        return {
            "hit_test_finetune": int(self.hit_test_spin.value()),
            "anchor_range": int(self.anchor_spin.value()),
            "control_range": int(self.control_spin.value()),
            "max_iterations": int(self.iter_spin.value()),
            "max_slurs_from_start": int(self.max_slurs_spin.value()),
            "include_stave_lines": bool(self.include_stave_lines_cb.isChecked()),
            "include_beams": bool(self.include_beams_cb.isChecked()),
            "pointiness_weight": float(self.pointiness_weight_spin.value()),
            "symmetry_weight": float(self.symmetry_weight_spin.value()),
            "pointiness_bonus_points": float(self.pointiness_points_spin.value()),
            "symmetry_bonus_points": float(self.symmetry_points_spin.value()),
            "antisymmetry_penalty_points": float(self.antisym_penalty_spin.value()),
            "straight_line_penalty_points": float(self.straight_penalty_spin.value()),
            "neighbor_connection_bonus_points": float(self.neighbor_bonus_spin.value()),
            "control_y_pass_penalty_points": float(self.control_y_penalty_spin.value()),
        }

    def set_apply_enabled(self, enabled: bool) -> None:
        try:
            self.apply_button.setEnabled(bool(enabled))
        except Exception:
            pass
        try:
            self.random_apply_button.setEnabled(bool(enabled))
        except Exception:
            pass

    def set_info_text(self, text: str) -> None:
        try:
            self.info_label.setText(str(text or ""))
        except Exception:
            pass

    def _load_from_appdata(self) -> None:
        try:
            adm = get_appdata_manager()
            raw = adm.get(self._appdata_key, {})
        except Exception:
            raw = {}
        data = raw if isinstance(raw, dict) else {}

        def _set_int(widget: QtWidgets.QSpinBox, key: str) -> None:
            if key not in data:
                return
            try:
                widget.setValue(int(data.get(key)))
            except Exception:
                pass

        def _set_float(widget: QtWidgets.QDoubleSpinBox, key: str) -> None:
            if key not in data:
                return
            try:
                widget.setValue(float(data.get(key)))
            except Exception:
                pass

        def _set_bool(widget: QtWidgets.QCheckBox, key: str) -> None:
            if key not in data:
                return
            try:
                widget.setChecked(bool(data.get(key)))
            except Exception:
                pass

        _set_int(self.hit_test_spin, "hit_test_finetune")
        _set_int(self.anchor_spin, "anchor_range")
        _set_int(self.control_spin, "control_range")
        _set_int(self.iter_spin, "max_iterations")
        _set_int(self.max_slurs_spin, "max_slurs_from_start")
        _set_bool(self.include_stave_lines_cb, "include_stave_lines")
        _set_bool(self.include_beams_cb, "include_beams")
        _set_float(self.pointiness_weight_spin, "pointiness_weight")
        _set_float(self.symmetry_weight_spin, "symmetry_weight")
        _set_float(self.pointiness_points_spin, "pointiness_bonus_points")
        _set_float(self.symmetry_points_spin, "symmetry_bonus_points")
        _set_float(self.antisym_penalty_spin, "antisymmetry_penalty_points")
        _set_float(self.straight_penalty_spin, "straight_line_penalty_points")
        _set_float(self.neighbor_bonus_spin, "neighbor_connection_bonus_points")
        _set_float(self.control_y_penalty_spin, "control_y_pass_penalty_points")

    def _save_to_appdata(self) -> None:
        try:
            adm = get_appdata_manager()
            adm.set(self._appdata_key, dict(self.settings()))
            adm.save()
        except Exception:
            pass

    @QtCore.Slot()
    def _on_apply_clicked(self) -> None:
        payload = dict(self.settings())
        self._save_to_appdata()
        self.applyRequested.emit(payload)

    @QtCore.Slot()
    def _on_random_apply_clicked(self) -> None:
        self.hit_test_spin.setValue(int(random.randint(0, 1000)))
        self._on_apply_clicked()

    def closeEvent(self, event) -> None:
        self._save_to_appdata()
        super().closeEvent(event)

    @staticmethod
    def get_settings(parent=None) -> dict[str, int | float | bool] | None:
        dlg = SlurBowOptimizerDialog(parent)
        dlg.setModal(True)
        captured: dict[str, int | float | bool] = {}

        def _capture_and_accept(settings: dict) -> None:
            captured.clear()
            captured.update(dict(settings or {}))
            dlg.accept()

        dlg.applyRequested.connect(_capture_and_accept)
        if dlg.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            return dict(captured or dlg.settings())
        return None


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Optimize slur bow X-handle positions to reduce collisions.")
    p.add_argument("input", help="Input .piano file path")
    p.add_argument("--output", help="Output .piano file path (default: <input>.slur_optimized.piano)")
    p.add_argument("--in-place", action="store_true", help="Write back into the input file")
    p.add_argument("--hit-test-finetune", type=int, default=64, help="Collision sampling density along each slur")
    p.add_argument("--anchor-range", type=int, default=10, help="Search radius in semitone steps for x1/x4")
    p.add_argument("--control-range", type=int, default=16, help="Search radius in semitone steps for x2/x3")
    p.add_argument("--max-iterations", type=int, default=3, help="Coordinate-descent iteration count")
    p.add_argument("--max-slurs-from-start", type=int, default=None, help="Limit processing to the first N slurs by start time")
    p.add_argument("--no-stave-lines", action="store_true", help="Exclude stave lines from obstacle hit testing")
    p.add_argument("--no-beams", action="store_true", help="Exclude beam lines from obstacle hit testing")
    p.add_argument("--pointiness-weight", type=float, default=0.5, help="Pointiness weight (0..1)")
    p.add_argument("--symmetry-weight", type=float, default=0.5, help="Symmetry weight (0..1)")
    p.add_argument("--pointiness-bonus-points", type=float, default=100.0, help="Pointiness bonus points")
    p.add_argument("--symmetry-bonus-points", type=float, default=100.0, help="Symmetry bonus points")
    p.add_argument("--antisymmetry-penalty-points", type=float, default=100.0, help="Anti-symmetry penalty points")
    p.add_argument("--straight-line-penalty-points", type=float, default=100.0, help="Too-straight penalty points")
    p.add_argument("--neighbor-connection-bonus-points", type=float, default=100.0, help="Neighbor red-handle connection bonus points")
    p.add_argument("--control-y-pass-penalty-points", type=float, default=1000.0, help="Penalty when control Y passes red-anchor Y")
    return p


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()
    result = optimize_file(
        args.input,
        output_path=args.output,
        hit_test_finetune=int(args.hit_test_finetune),
        anchor_range=int(args.anchor_range),
        control_range=int(args.control_range),
        max_iterations=int(args.max_iterations),
        max_slurs_from_start=(None if args.max_slurs_from_start is None else int(args.max_slurs_from_start)),
        include_stave_lines=not bool(args.no_stave_lines),
        include_beams=not bool(args.no_beams),
        pointiness_weight=float(args.pointiness_weight),
        symmetry_weight=float(args.symmetry_weight),
        pointiness_bonus_points=float(args.pointiness_bonus_points),
        symmetry_bonus_points=float(args.symmetry_bonus_points),
        antisymmetry_penalty_points=float(args.antisymmetry_penalty_points),
        straight_line_penalty_points=float(args.straight_line_penalty_points),
        neighbor_connection_bonus_points=float(args.neighbor_connection_bonus_points),
        control_y_pass_penalty_points=float(args.control_y_pass_penalty_points),
        in_place=bool(args.in_place),
    )
    print(
        f"Slur bow optimization finished. "
        f"changed={result['slurs_changed']}/{result['slurs_total']} "
        f"output={result['output']}"
    )


if __name__ == "__main__":
    main()
