# Engraver Pre-Calculate Layout Model

This document describes the `pre_calculated` dictionary produced by
`do_engrave(...)._pre_calculate()` in `engraver/engraver.py`.

## Top-Level Structure

```text
pre_calculated
- page_width_mm: float
- page_height_mm: float
- page_left_margin_mm: float
- page_right_margin_mm: float
- page_top_margin_mm: float
- page_bottom_margin_mm: float
- layout: dict
- pages: list[page]
```

### Field Details

- `page_width_mm`, `page_height_mm`
  Effective page size after orientation handling.

- `page_left_margin_mm`, `page_right_margin_mm`, `page_top_margin_mm`, `page_bottom_margin_mm`
  Physical page margins in millimeters.

- `layout`
  Snapshot copy of score layout settings used by drawers.

- `pages`
  Ordered list of rendered pages. Each page contains systems.

## Page Object

```text
page
- page_index: int
- systems: list[system]
- used_width_mm: float
- rest_space_mm: float
- over_space_mm: float
- rest_space_per_slot_mm: float
```

### Field Details

- `page_index`
  Zero-based page number in the pre-calc result.

- `systems`
  List of systems assigned to this page.

- `used_width_mm`
  Sum of all `system_reserved_width_mm` values on this page.

- `rest_space_mm`
  Remaining horizontal space after system packing.

- `over_space_mm`
  Overflow amount if packed systems exceed available width.

- `rest_space_per_slot_mm`
  Spacing slot value used for page-level centering between systems.

## System Object

```text
system
- system_index: int
- time_start: float
- time_end: float
- page_break: bool
- system_reserved_width_mm: float
- staves: list[stave]
- y_start_mm: float
- y_end_mm: float

# Computed during placement pass:
- system_outer_left_mm: float
- system_outer_width_mm: float
- system_stave_left_mm: float
- system_stave_width_mm: float
- system_content_left_mm: float
- system_content_width_mm: float
```

### Field Details

- `system_index`
  Zero-based system index in the full score sequence.

- `time_start`, `time_end`
  Time range (ticks) covered by this system.

- `page_break`
  Indicates a forced new-page boundary at this system start.

- `system_reserved_width_mm`
  Horizontal width reserved for this system during packing.
  It includes per-stave margins and overhang reservations.

- `staves`
  Per-stave pre-calculated geometry and event slices.

- `y_start_mm`, `y_end_mm`
  Vertical system extents on page.

- `system_outer_left_mm`, `system_outer_width_mm`
  Outer system box used for layout/debug footprint.

- `system_stave_left_mm`, `system_stave_width_mm`
  Min/max bounds from stave line geometry only.

- `system_content_left_mm`, `system_content_width_mm`
  Min/max bounds from stave content spans (including reservations).

## Stave Object

```text
stave
- stave_index: int
- mode: str  # auto | manual
- manual_range: list[int] | None
- left_margin_mm: float
- right_margin_mm: float
- note_pitch_low: int
- note_pitch_high: int
- stave_low_key: int
- stave_high_key: int
- note_span_low_key: int
- note_span_high_key: int
- stave_width_mm: float
- stave_content_span_width_mm: float
- reserve_left_overhang_mm: float
- reserve_right_overhang_mm: float
- group_segments: list[dict]
- stave_scale: float
- composite_scale: float
- semitone_mm: float
- key_offsets: dict[int, float]
- events_in_line: dict[str, list]

# Computed during placement pass:
- stave_content_span_left_mm: float
- stave_left_mm: float
- black_lines: list[black_line]
```

### Field Details

- `stave_index`
  Index in score staves list.

- `mode`
  `auto` when range is auto expanded, `manual` when line-break range is forced.

- `manual_range`
  Forced key range for manual mode, otherwise `None`.

- `left_margin_mm`, `right_margin_mm`
  Per-stave margins from line-break settings (already scale-adjusted).

- `note_pitch_low`, `note_pitch_high`
  Actual note pitch span in this system time range.

- `stave_low_key`, `stave_high_key`
  Expanded stave line range after group-boundary snapping.

- `note_span_low_key`, `note_span_high_key`
  Content span key range used for horizontal reservation.
  In manual mode this can extend beyond `stave_low_key` / `stave_high_key`
  to reserve ledger-line space equivalent to auto-mode content spacing.

- `stave_width_mm`
  Width of stave line geometry span.

- `stave_content_span_width_mm`
  Base content span width before overhang reservations.

- `reserve_left_overhang_mm`, `reserve_right_overhang_mm`
  Reserved extra content space on each side.
  Current phase default is `0.0` for both.

- `group_segments`
  Klavarskribo line-group segments intersecting this stave range.

- `stave_scale`
  Per-stave scale from score data.

- `composite_scale`
  Product of `layout.scale * stave_scale`.

- `semitone_mm`
  Effective semitone width in mm for this stave.

- `key_offsets`
  Mapping from piano key index to x-offset in mm for this stave scale.

- `events_in_line`
  Event slices filtered to the system time window.

- `stave_content_span_left_mm`
  Placed left x of content span (includes left overhang reservation).

- `stave_left_mm`
  Placed left x of stave line span.

- `black_lines`
  Vertical stave line draw commands, each containing:
  - `key`: int
  - `x_mm`: float
  - `kind`: str (`two`, `three`, `clef`)
  - `width_mm`: float
  - `dash`: list[float] | None

## Placement Relationships

For each stave during placement:

```text
content_span_width = stave_content_span_width_mm
                   + reserve_left_overhang_mm
                   + reserve_right_overhang_mm

stave_content_span_left_mm = local_x + left_margin_mm - reserve_left_overhang_mm
```

Then system metrics are aggregated from placed stave metrics:

- Outer metrics from packed system extents
- Stave metrics from `stave_left_mm + stave_width_mm`
- Content metrics from `stave_content_span_left_mm + content_span_width`

## Notes For Next Phase

- Overhang fields are wired but intentionally set to zero.
- Next phase can compute non-zero reservation values from noteheads/beams/slurs.
- Once non-zero, packing and content bounds will automatically include them.
