from __future__ import annotations

from typing import TYPE_CHECKING, cast

from editor.editor_defaults import SCALE
from file_model.SCORE import SCORE
from file_model.events.note import Note
from symbol_design.noteheads import (
    resolve_notehead_spec,
    sheared_notehead_support_v,
    sheared_notehead_outline_points,
    support_point_from_outline_points,
)
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import SHORTEST_DURATION
from utils.operator import Operator

if TYPE_CHECKING:
    from editor.editor import Editor
    from file_model.events.arpeggio import Arpeggio


class ArpeggioDrawerMixin:
    _arp_time_op: Operator = Operator(float(SHORTEST_DURATION))

    def _resolve_arpeggio_notes(self, arp: "Arpeggio", notes_all: list[object]) -> list[object]:
        """Resolve arpeggio member notes using float times with Operator tolerance."""
        base_time = arp.time
        pitches = [int(p) for p in (arp.note_pitches or []) if int(p) > 0]
        if len(pitches) < 2:
            return []
        notes_by_pitch: dict[int, list[Note]] = {}
        for note_obj in notes_all:
            note_obj: Note
            pitch = note_obj.pitch
            notes_by_pitch.setdefault(note_obj.pitch, []).append(note_obj)
        resolved = []
        for pitch in pitches:
            note_obj = None
            for candidate in notes_by_pitch.get(int(pitch), []):
                candidate_time = candidate.time
                if self._arp_time_op.eq(candidate_time, base_time):
                    note_obj = candidate
                    break
            if note_obj is not None:
                resolved.append(note_obj)
        return resolved

    def _arpeggio_time_window(self, base_time: float, rtime1: float, rtime2: float) -> tuple[float, float]:
        return (float(base_time) + float(rtime1), float(base_time) + float(rtime2))

    def draw_arpeggio(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        score: SCORE = self.current_score()
        if score is None:
            return

        # get data
        arps = score.events.arpeggio
        notes_all = score.events.note
        layout = score.layout
        stem_w = layout.note_stem_thickness_mm * SCALE
        stem_len = layout.note_stem_length_semitone * self.semitone_dist
        width_scale = layout.note_width_scaling
        height_scale = layout.notehead_height_scaling
        base_tilt = layout.notehead_tilt

        active_tool_name = self._tool.TOOL_NAME
        show_handles = active_tool_name == "arpeggio"
        arp_keys: list[tuple[float, tuple[int, ...]]] = []

        for arp in arps:
            base_time = arp.time
            rtime1 = arp.rtime1
            rtime2 = arp.rtime2
            arp_id = arp._id

            arp_pitches = tuple(sorted(int(p) for p in (arp.note_pitches or []) if int(p) > 0))
            if len(arp_pitches) >= 2:
                arp_keys.append((float(base_time), arp_pitches))

            chord_notes = self._resolve_arpeggio_notes(arp, notes_all)
            if len(chord_notes) < 2:
                continue

            start_t, end_t = self._arpeggio_time_window(base_time, rtime1, rtime2)
            chord_sorted = sorted(chord_notes, key=lambda n: n.pitch)
            if len(chord_sorted) < 2:
                continue

            # The arpeggio diagonal line endpoints and projected notehead positions
            # are already pre-computed by _build_render_cache into _arpeggio_y_overrides.
            # Read them back here so the arpeggio drawer uses exactly the same geometry.
            arp_y_overrides: dict[int, float] = getattr(self, "_arpeggio_y_overrides", {}) or {}
            span = max(1, len(chord_sorted) - 1)
            y0 = float(self.time_to_mm(start_t))
            y1 = float(self.time_to_mm(end_t))

            stem_x_positions: list[float] = []
            for note_obj in chord_sorted:
                pitch = int(getattr(note_obj, "pitch", 0) or 0)
                x_note = float(self.pitch_to_x(pitch))
                stem_x_positions.append(float(x_note))

            if len(stem_x_positions) < 2:
                continue

            x_line_start = float(stem_x_positions[0])
            x_line_end = float(stem_x_positions[-1])
            dx_line = float(x_line_end - x_line_start)
            dy_line = float(y1 - y0)
            m_line = float(dy_line / dx_line) if abs(dx_line) > 1e-9 else 0.0
            support_cache: dict[tuple[str, bool], float] = {}

            # Projected endpoint positions used for hit-rect and handles.
            # These mirror the logic in _build_render_cache exactly.
            projected: list[tuple[float, float]] = []
            hand_first = str(getattr(chord_sorted[0], "hand", "l") or "l") if chord_sorted else "l"
            for i, (note_obj, x_stem) in enumerate(zip(chord_sorted, stem_x_positions)):
                nid = int(getattr(note_obj, "_id", 0) or 0)
                y_proj = arp_y_overrides.get(nid)
                if y_proj is None:
                    # Fallback: recompute inline if cache miss (e.g. first frame)
                    if abs(x_line_end - x_line_start) <= 1e-6:
                        ratio = float(i) / float(span)
                    else:
                        ratio = (float(x_stem) - x_line_start) / float(x_line_end - x_line_start)
                    ratio = max(0.0, min(1.0, ratio))
                    y_line = float(y0 + (y1 - y0) * ratio)
                    # For left hand: highest notehead never offset. For right hand: lowest notehead never offset.
                    if (hand_first == "l" and i == len(chord_sorted) - 1):
                        # Highest note in left arpeggio: NO OFFSET
                        y_proj = y_line
                    elif (hand_first == "r" and i == 0):
                        # Lowest note in right arpeggio: NO OFFSET
                        y_proj = y_line
                    else:
                        hand_local = str(getattr(note_obj, "hand", "l") or "l")
                        try:
                            default_black_above_local = bool(self._black_note_above_stem(note_obj, layout))
                        except Exception:
                            default_black_above_local = True
                        spec_local = resolve_notehead_spec(note_obj, default_black_above=default_black_above_local)
                        is_up_local = bool(getattr(spec_local, "is_up", False))
                        cache_key = (hand_local, is_up_local)
                        if cache_key not in support_cache:
                            support_cache[cache_key] = sheared_notehead_support_v(
                                hand=hand_local,
                                is_up=is_up_local,
                                semitone_space_mm=self.semitone_dist,
                                width_scale=width_scale,
                                height_scale=height_scale,
                                base_tilt=base_tilt,
                                m_line=m_line,
                                sample_count=64,
                            )
                        v_support = float(support_cache[cache_key])
                        y_proj = float(y_line - v_support)
                projected.append((float(x_stem), float(y_proj)))

            if len(projected) < 2:
                continue

            # Skip drawing the diagonal line if both handles are at 0 (phantom arpeggio)
            is_phantom = (rtime1 == 0.0 and rtime2 == 0.0)

            # Draw the arpeggio annotation line (the diagonal) only for non-phantom arpeggios.
            if not is_phantom:

                import math
                # --- Step 1: Find the stem end point on the hand-dependent anchor notehead ---
                if hand_first == "l":
                    anchor_note = chord_sorted[-1]
                    anchor_x = float(stem_x_positions[-1])
                    anchor_y = float(y1)  # highest note never offset on left hand
                else:
                    anchor_note = chord_sorted[0]
                    anchor_x = float(stem_x_positions[0])
                    anchor_y = float(y0)  # lowest note never offset on right hand

                hand_anchor = str(getattr(anchor_note, "hand", "l") or "l")
                layout_anchor = layout
                default_black_above_anchor = bool(self._black_note_above_stem(anchor_note, layout_anchor))
                spec_anchor = resolve_notehead_spec(anchor_note, default_black_above=default_black_above_anchor)
                is_up_anchor = bool(getattr(spec_anchor, "is_up", False))
                outline_anchor = sheared_notehead_outline_points(
                    hand=hand_anchor,
                    is_up=is_up_anchor,
                    semitone_space_mm=self.semitone_dist,
                    width_scale=width_scale,
                    height_scale=height_scale,
                    base_tilt=base_tilt,
                    sample_count=128,
                )
                # Use the raw arpeggio endpoints for the stem angle.
                x_high = float(stem_x_positions[-1])
                x_low = float(stem_x_positions[0])
                angle = math.atan2(float(y1 - y0), float(x_high - x_low))
                m_line_local = float(math.tan(angle)) if abs(math.cos(angle)) > 1e-9 else 0.0
                # Symmetric noteheads make centroid-based side detection ambiguous.
                # Use the notehead/stem side directly: up noteheads are hit from above,
                # down noteheads are hit from below.
                edge_anchor = support_point_from_outline_points(
                    outline_anchor,
                    m_line=m_line_local,
                    choose_max=bool(is_up_anchor),
                )
                # The end xy point on the anchor notehead
                stem_end_x = anchor_x + edge_anchor[0]
                stem_end_y = anchor_y + edge_anchor[1]

                # --- Step 2: Find the matching outline contact point on the opposite notehead ---
                if hand_first == "l":
                    side_idx = 0
                else:
                    side_idx = -1

                side_note = chord_sorted[side_idx]
                side_center_x = float(stem_x_positions[side_idx])
                side_center_y = float(projected[side_idx][1])

                hand_side = str(getattr(side_note, "hand", "l") or "l")
                default_black_above_side = bool(self._black_note_above_stem(side_note, layout))
                spec_side = resolve_notehead_spec(side_note, default_black_above=default_black_above_side)
                is_up_side = bool(getattr(spec_side, "is_up", False))
                outline_side = sheared_notehead_outline_points(
                    hand=hand_side,
                    is_up=is_up_side,
                    semitone_space_mm=self.semitone_dist,
                    width_scale=width_scale,
                    height_scale=height_scale,
                    base_tilt=base_tilt,
                    sample_count=128,
                )

                # Use the current stem-end to opposite-center direction to query the true support point.
                side_angle = math.atan2(side_center_y - stem_end_y, side_center_x - stem_end_x)
                side_m_line = float(math.tan(side_angle)) if abs(math.cos(side_angle)) > 1e-9 else 0.0
                edge_side = support_point_from_outline_points(
                    outline_side,
                    m_line=side_m_line,
                    choose_max=bool(is_up_side),
                )
                base_x = side_center_x + edge_side[0]
                base_y = side_center_y + edge_side[1]

                # Recompute direction from anchor stem-end to opposite outline contact point.
                angle = math.atan2(base_y - stem_end_y, base_x - stem_end_x)

                # --- Step 3: Extend outward from that handle point by one stem length ---
                ux = math.cos(angle)
                uy = math.sin(angle)
                tip_x = base_x + (ux * stem_len)
                tip_y = base_y + (uy * stem_len)

                # --- Step 4: Draw the full stem as a single line ---
                du.add_line(
                    stem_end_x, stem_end_y, tip_x, tip_y,
                    color=self.notation_color,
                    width_mm=stem_w,
                    id=arp_id,
                    tags=["stem"],
                )

            x_values = [p[0] for p in projected]
            y_values = [p[1] for p in projected]
            self.register_hit_rect(
                "arpeggio",
                arp_id,
                min(x_values) - (self.semitone_dist * 0.5),
                min(y_values) - (self.semitone_dist * 0.5),
                max(x_values) + (self.semitone_dist * 0.5),
                max(y_values) + (self.semitone_dist * 0.5),
                handle="body",
            )

            if show_handles:
                handle_r = self.semitone_dist
                hand_first = str(getattr(chord_sorted[0], "hand", "l") or "l") if chord_sorted else "l"
                # Always use rtime1/y0 for low, rtime2/y1 for high
                x_low = float(stem_x_positions[0])
                x_high = float(stem_x_positions[-1])
                # y0/y1 are already computed above
                handle_positions = []
                if hand_first == "l":
                    # Left arpeggio: low = projected[0], high = (x_high, y1)
                    handle_positions = [("low", projected[0]), ("high", (x_high, y1))]
                else:
                    # Right arpeggio: low = (x_low, y0), high = projected[-1]
                    handle_positions = [("low", (x_low, y0)), ("high", projected[-1])]
                for handle_name, (hx, hy) in handle_positions:
                    du.add_oval(
                        hx - handle_r,
                        hy - handle_r,
                        hx + handle_r,
                        hy + handle_r,
                        stroke_color=None,
                        fill_color=(0.5, 0.0, 0.0, 0.75),
                        id=arp_id,
                        tags=["arpeggio_handle"],
                    )
                    self.register_hit_rect(
                        "arpeggio",
                        arp_id,
                        hx - handle_r * 1.2,
                        hy - handle_r * 1.2,
                        hx + handle_r * 1.2,
                        hy + handle_r * 1.2,
                        handle=handle_name,
                    )

        if not show_handles:
            return

        # draw handles for phantom arpeggios (rtime1 == rtime2 == 0) that have no diagonal line
        chord_groups: list[tuple[float, str, list[object]]] = []
        notes_by_hand: dict[str, list[object]] = {}
        for note_obj in notes_all:
            hand = str(getattr(note_obj, "hand", "l") or "l")
            notes_by_hand.setdefault(hand, []).append(note_obj)

        for hand, hand_notes in notes_by_hand.items():
            ordered = sorted(
                hand_notes,
                key=lambda n: (float(getattr(n, "time", 0.0) or 0.0), int(getattr(n, "pitch", 0) or 0)),
            )
            active_time: float | None = None
            active_notes: list[object] = []
            for note_obj in ordered:
                note_time = float(getattr(note_obj, "time", 0.0) or 0.0)
                if active_time is None or not self._arp_time_op.eq(note_time, active_time):
                    if active_time is not None and active_notes:
                        chord_groups.append((float(active_time), str(hand), list(active_notes)))
                    active_time = float(note_time)
                    active_notes = []
                active_notes.append(note_obj)
            if active_time is not None and active_notes:
                chord_groups.append((float(active_time), str(hand), list(active_notes)))

        for base_time, _hand, chord_notes in chord_groups:
            if len(chord_notes) < 2:
                continue

            pitches = tuple(sorted(set(int(getattr(n, "pitch", 0) or 0) for n in chord_notes if int(getattr(n, "pitch", 0) or 0) > 0)))
            if len(pitches) < 2:
                continue

            if any(self._arp_time_op.eq(base_time, arp_time) and arp_p == pitches for arp_time, arp_p in arp_keys):
                continue

            chord_sorted = sorted(chord_notes, key=lambda n: int(getattr(n, "pitch", 0) or 0))
            stem_x_positions: list[float] = []
            for note_obj in chord_sorted:
                pitch = int(getattr(note_obj, "pitch", 0) or 0)
                x_note = float(self.pitch_to_x(pitch))
                stem_x_positions.append(float(x_note))

            if len(stem_x_positions) < 2:
                continue

            y_flat = float(self.time_to_mm(base_time))
            handle_r = self.semitone_dist
            for handle_name, hx in (("low", stem_x_positions[0]), ("high", stem_x_positions[-1])):
                du.add_oval(
                    hx - handle_r,
                    y_flat - handle_r,
                    hx + handle_r,
                    y_flat + handle_r,
                    stroke_color=None,
                    fill_color=(0.5, 0.0, 0.0, 0.75),
                    id=0,
                    tags=["arpeggio_handle"],
                )
                self.register_hit_rect(
                    "arpeggio",
                    0,
                    hx - handle_r * 1.2,
                    y_flat - handle_r * 1.2,
                    hx + handle_r * 1.2,
                    y_flat + handle_r * 1.2,
                    handle=handle_name,
                    base_time=base_time,
                    note_pitches=list(pitches),
                )
