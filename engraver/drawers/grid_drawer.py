from __future__ import annotations

from file_model.base_grid import resolve_grid_layer_offsets
from utils.CONSTANT import QUARTER_NOTE_UNIT, SHORTEST_DURATION
from utils.operator import Operator


def _item_get(item, key: str, default=None):
    if isinstance(item, dict):
        return item.get(key, default)
    return getattr(item, key, default)


def _time_to_y(t: float, t0: float, t1: float, y0: float, y1: float) -> float:
    if t1 <= t0:
        return y0
    u = max(0.0, min(1.0, (float(t) - float(t0)) / (float(t1) - float(t0))))
    return float(y0 + (u * (y1 - y0)))


def _build_grid_times(base_grid: list) -> tuple[list[float], list[float], dict[float, int]]:
    """Return (barline_times, grid_times, measure_start_number_by_time)."""
    barline_times: list[float] = []
    grid_times: list[float] = []
    measure_numbers: dict[float, int] = {}
    cur_t = 0.0
    measure_no = 0
    for bg in list(base_grid or []):
        numer = int(_item_get(bg, 'numerator', 4) or 4)
        denom = int(_item_get(bg, 'denominator', 4) or 4)
        mcount = int(_item_get(bg, 'measure_amount', 1) or 1)
        measure_len_ticks = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
        positions = list(_item_get(bg, 'beat_grouping', []) or [])
        bar_offsets, grid_offsets = resolve_grid_layer_offsets(positions, numer, denom)
        for _ in range(max(0, mcount)):
            measure_no += 1
            t_measure_start = float(cur_t)
            measure_numbers[round(t_measure_start, 6)] = int(measure_no)
            for off in list(bar_offsets or []):
                barline_times.append(float(cur_t + float(off)))
            for off in list(grid_offsets or []):
                grid_times.append(float(cur_t + float(off)))
            cur_t += measure_len_ticks
    barline_times.append(float(cur_t))
    return barline_times, grid_times, measure_numbers


def grid_drawer(du, pre_calc: dict) -> None:
    system = dict(pre_calc.get('system', {}) or {})
    layout = dict(pre_calc.get('layout', {}) or {})
    base_grid = list(pre_calc.get('base_grid', []) or [])
    notation_color = tuple(pre_calc.get('notation_color', (0.0, 0.0, 0.0, 1.0)) or (0.0, 0.0, 0.0, 1.0))
    y0 = float(pre_calc.get('y0', 0.0) or 0.0)
    y1 = float(pre_calc.get('y1', 0.0) or 0.0)
    t0 = float(system.get('time_start', 0.0) or 0.0)
    t1 = float(system.get('time_end', t0) or t0)

    x_left = float(pre_calc.get('system_content_left_mm', 0.0) or 0.0)
    x_right = float(x_left + float(pre_calc.get('system_content_width_mm', 0.0) or 0.0))

    op = Operator(SHORTEST_DURATION)

    def _in_system(t: float) -> bool:
        return op.ge(float(t), float(t0)) and op.le(float(t), float(t1))

    staves = list(system.get('staves', []) or [])
    system_composite_scale = max([float(st.get('composite_scale', 1.0) or 1.0) for st in staves] or [1.0])
    barline_w = max(0.01, float(layout.get('grid_barline_thickness_mm', 0.1) or 0.1) * system_composite_scale)
    grid_w = max(0.01, float(layout.get('grid_gridline_thickness_mm', 0.15) or 0.15) * system_composite_scale)

    barline_times, grid_times, measure_numbers = _build_grid_times(base_grid)
    barline_keys = {round(float(t), 6) for t in barline_times}

    # Sub-grid lines, excluding primary barline layer times.
    for t in grid_times:
        if round(float(t), 6) in barline_keys:
            continue
        if not _in_system(float(t)):
            continue
        y = _time_to_y(float(t), t0, t1, y0, y1)
        du.add_line(
            x_left,
            y,
            x_right,
            y,
            color=(notation_color[0], notation_color[1], notation_color[2], 0.35),
            width_mm=float(grid_w),
            dash_pattern=[0.8, 0.8],
            tags=['grid_line'],
        )

    # Barlines connect the full green content rectangle width.
    for i, t in enumerate(barline_times):
        if not _in_system(float(t)):
            continue
        y = _time_to_y(float(t), t0, t1, y0, y1)
        is_last = i == (len(barline_times) - 1)
        du.add_line(
            x_left,
            y,
            x_right,
            y,
            color=(notation_color[0], notation_color[1], notation_color[2], 0.85),
            width_mm=(float(barline_w) * 2.0) if is_last else float(barline_w),
            tags=['end_barline' if is_last else 'barline'],
        )

    # Measure numbering at right side of the system for visible starts.
    for t in barline_times[:-1]:
        if not _in_system(float(t)):
            continue
        key = round(float(t), 6)
        if key not in measure_numbers:
            continue
        y = _time_to_y(float(t), t0, t1, y0, y1)
        du.add_text(
            x_right + 0.8,
            y + 0.1,
            str(int(measure_numbers[key])),
            family='Edwin',
            size_pt=8.5,
            color=notation_color,
            anchor='w',
            tags=['measure_number'],
        )
