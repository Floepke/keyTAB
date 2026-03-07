from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Literal
from utils.CONSTANT import QUARTER_NOTE_UNIT

LEGACY_MODE_MAX_VALUE: int = 8
MIN_TIME_GRID_TICKS: int = int(round(float(QUARTER_NOTE_UNIT) * (4.0 / 32.0)))

@dataclass
class BaseGrid:
    """
    Defines the musical grid across a sequence of measures.

    - numerator: time signature numerator (e.g., 4 in 4/4)
    - denominator: time signature denominator (e.g., 4 in 4/4)

    Denominator defines the smallest possible time step for the base grid in
    this context. A denominator of 1 enforces drawing the barline (beat 1) for
    each measure. Higher denominators subdivide the measure into smaller units
    and `beat_grouping` selects which beats are drawn/enabled.

    For example, in 4/4, `beat_grouping=[1,2,3,4]` means a single group of 4.
    In 7/8 with 3+4 grouping, `beat_grouping=[1,2,3,1,2,3,4]`.

    - beat_grouping: per-beat sequence describing grouping. The sequence length
      must equal the numerator. It must start at 1 and can only count up or
      reset to 1 (e.g., 1231234).
    - measure_amount: number of measures to generate with these settings.
    """
    numerator: int = 4
    denominator: int = 4
    beat_grouping: List[int] = field(default_factory=lambda: [1, 2, 3, 4])
    measure_amount: int = 1
    indicator_enabled: bool = True


def _is_legacy_group_sequence(seq: List[int], numerator: int) -> bool:
    numer = max(1, int(numerator))
    if len(seq) != numer or not seq or int(seq[0]) != 1:
        return False
    for prev, cur in zip(seq, seq[1:]):
        p = int(prev)
        c = int(cur)
        if c != 1 and c != p + 1:
            return False
    return True


def resolve_grid_layer_offsets(beat_grouping: List[int], numerator: int, denominator: int) -> tuple[List[float], List[float]]:
    """Resolve per-measure barline/grid offsets (in ticks) from beat_grouping.

    Returns `(bar_offsets, grid_offsets)` where each offset is measured from the
    start of a measure.

        Detection strategy:
    - Legacy per-beat grouping sequence (len == numerator, 1/reset semantics)
        - Compact enabled-beat list (e.g. [1,2]) when all values are <= 8

    New timing mode:
        - Any value > 8 switches to timeline mode.
        - Values are interpreted directly as timeline tick offsets in the measure.
      `0` is a barline layer position; positive values are grid-line positions.
        - Positive time positions smaller than 32 ticks (1/32 with QUARTER_NOTE_UNIT=256)
            are ignored.
    """
    numer = max(1, int(numerator))
    denom = max(1, int(denominator))
    measure_len_ticks = float(numer) * (4.0 / float(denom)) * float(QUARTER_NOTE_UNIT)
    beat_len_ticks = measure_len_ticks / float(numer)

    seq = [int(v) for v in (beat_grouping or []) if isinstance(v, (int, float))]
    if not seq:
        return [], []

    # Legacy full sequence mode
    if _is_legacy_group_sequence(seq, numer):
        bar_offsets = [0.0]
        if seq == list(range(1, numer + 1)):
            grid_offsets = [float(i * beat_len_ticks) for i in range(1, numer)]
            return bar_offsets, grid_offsets
        starts = [i for i, v in enumerate(seq, start=1) if int(v) == 1 and i > 1]
        grid_offsets = [float((i - 1) * beat_len_ticks) for i in starts]
        return bar_offsets, grid_offsets

    has_time_hint = any(int(v) > int(LEGACY_MODE_MAX_VALUE) for v in seq)

    # Legacy compact enabled-beat mode (e.g. [1,2])
    if (not has_time_hint) and all(1 <= int(v) <= numer for v in seq):
        uniq_beats = sorted(set(int(v) for v in seq))
        has_barline = 1 in uniq_beats
        bar_offsets = [0.0] if has_barline else []
        grid_offsets = [float((b - 1) * beat_len_ticks) for b in uniq_beats if b > 1]
        return bar_offsets, grid_offsets

    # New time-position mode: values are absolute offsets inside the measure.
    # 0 -> barline layer, >0 -> gridline layer.
    uniq_times: list[float] = []
    for val in seq:
        t = float(val)
        if t > 0.0 and t < float(MIN_TIME_GRID_TICKS):
            continue
        if t < 0.0 or t >= measure_len_ticks:
            continue
        if any(abs(t - existing) < 1e-6 for existing in uniq_times):
            continue
        uniq_times.append(t)
    uniq_times = sorted(uniq_times)
    bar_offsets = [0.0] if any(abs(t) < 1e-6 for t in uniq_times) else []
    grid_offsets = [float(t) for t in uniq_times if abs(t) >= 1e-6]
    return bar_offsets, grid_offsets