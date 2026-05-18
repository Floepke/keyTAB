from __future__ import annotations

import bisect
import heapq
import math
from typing import TYPE_CHECKING

from file_model.base_grid import resolve_grid_layer_offsets
from symbol_design.noteheads import (
    resolve_notehead_spec,
    sheared_notehead_outline_points,
    sheared_notehead_support_v,
    support_point_from_outline_points,
)
from utils.CONSTANT import QUARTER_NOTE_UNIT, SHORTEST_DURATION
from utils.operator import Operator

if TYPE_CHECKING:
    from file_model.SCORE import SCORE


class CachingMixin:
    """Shared render-cache lifecycle and cache builders for Editor."""

    def _init_caching_state(self) -> None:
        # Per-frame shared render cache (built at draw_all)
        self._draw_cache: dict | None = None
        # One-shot hint to reuse the current draw cache on the next frame
        self._reuse_draw_cache_once: bool = False
        # Cache for base-grid-derived timeline helpers used by _build_render_cache
        self._grid_time_cache_key: tuple | None = None
        self._grid_time_cache_values: tuple[list[float], list[float]] | None = None
        # Cache for note-time derived arrays used by _build_render_cache
        self._note_time_cache_key: tuple | None = None
        self._note_time_cache_values: tuple[list, list[float], list[float], list[tuple[float, int]], list[float]] | None = None
        # Per-frame arpeggio Y-override data (populated by _build_render_cache)
        self._arpeggio_y_overrides: dict[int, float] = {}

    def _apply_single_note_timing_cache_update(
        self, notes: list
    ) -> tuple[list, list[float], list[float], list[tuple[float, int]], list[float]] | None:
        dirty = self._single_note_timing_dirty
        cached = self._note_time_cache_values
        if not dirty or cached is None:
            return None
        notes_sorted, starts, ends, end_pairs, end_values = cached
        if len(notes_sorted) != len(notes):
            return None

        note_id = int(dirty.get("note_id", 0) or 0)
        note = dirty.get("note")
        if note is None or int(getattr(note, "_id", 0) or 0) != note_id:
            return None

        old_idx = -1
        for idx, existing in enumerate(notes_sorted):
            if int(getattr(existing, "_id", 0) or 0) == note_id:
                old_idx = idx
                break
        if old_idx < 0:
            return None

        old_notes_sorted = list(notes_sorted)
        old_starts = list(starts)
        old_ends = list(ends)
        old_end_pairs = list(end_pairs)
        old_end_values = list(end_values)

        note_entry = old_notes_sorted.pop(old_idx)
        old_starts.pop(old_idx)
        old_ends.pop(old_idx)

        pair_pos = -1
        for idx, (_end_value, pair_idx) in enumerate(old_end_pairs):
            if int(pair_idx) == old_idx:
                pair_pos = idx
                break
        if pair_pos < 0:
            return None
        old_end_pairs.pop(pair_pos)
        old_end_values.pop(pair_pos)

        adjusted_end_pairs: list[tuple[float, int]] = []
        for end_value, pair_idx in old_end_pairs:
            new_pair_idx = int(pair_idx)
            if new_pair_idx > old_idx:
                new_pair_idx -= 1
            adjusted_end_pairs.append((float(end_value), new_pair_idx))

        new_time = float(getattr(note_entry, "time", 0.0) or 0.0)
        new_duration = float(getattr(note_entry, "duration", 0.0) or 0.0)
        new_pitch = int(getattr(note_entry, "pitch", 0) or 0)
        op_time = Operator(float(SHORTEST_DURATION))
        insert_idx = bisect.bisect_left(old_starts, new_time)
        while insert_idx < len(old_starts):
            if op_time.gt(old_starts[insert_idx], new_time):
                break
            if op_time.eq(old_starts[insert_idx], new_time) and int(
                getattr(old_notes_sorted[insert_idx], "pitch", 0) or 0
            ) > new_pitch:
                break
            insert_idx += 1

        old_notes_sorted.insert(insert_idx, note_entry)
        old_starts.insert(insert_idx, new_time)
        new_end = new_time + new_duration
        old_ends.insert(insert_idx, new_end)

        rebased_end_pairs: list[tuple[float, int]] = []
        for end_value, pair_idx in adjusted_end_pairs:
            new_pair_idx = int(pair_idx)
            if new_pair_idx >= insert_idx:
                new_pair_idx += 1
            rebased_end_pairs.append((float(end_value), new_pair_idx))
        end_insert_idx = bisect.bisect_right(old_end_values, new_end)
        old_end_values.insert(end_insert_idx, new_end)
        rebased_end_pairs.insert(end_insert_idx, (new_end, insert_idx))
        return (old_notes_sorted, old_starts, old_ends, rebased_end_pairs, old_end_values)

    def _compute_note_time_cache_key(self, notes: list) -> tuple[int, int]:
        """Cheap content hash for note timing/pitch fields used by sorting/culling."""
        h = 1469598103934665603
        fnv_prime = 1099511628211
        mask = (1 << 64) - 1
        for n in notes:
            try:
                t_i = int(round(float(getattr(n, "time", 0.0) or 0.0) * 1000.0))
                d_i = int(round(float(getattr(n, "duration", 0.0) or 0.0) * 1000.0))
                p_i = int(getattr(n, "pitch", 0) or 0)
                nid = int(getattr(n, "_id", 0) or 0)
            except Exception:
                t_i, d_i, p_i, nid = (0, 0, 0, 0)
            h ^= t_i & mask
            h = (h * fnv_prime) & mask
            h ^= d_i & mask
            h = (h * fnv_prime) & mask
            h ^= p_i & mask
            h = (h * fnv_prime) & mask
            h ^= nid & mask
            h = (h * fnv_prime) & mask
        return (len(notes), int(h))

    def _build_beam_groups_cache(
        self,
        score: "SCORE",
        op: Operator,
        notes_by_hand: dict[str, list],
        beam_by_hand: dict[str, list],
        grid_den_times: list[float],
    ) -> tuple[dict[str, list[list]], dict[str, list[tuple[float, float]]], dict[str, list[tuple[float, float]]]]:
        def build_grid_windows(times: list[float], a: float, b: float) -> list[tuple[float, float]]:
            windows: list[tuple[float, float]] = []
            cur = 0.0
            for bg in getattr(score, "base_grid", []) or []:
                numer = int(getattr(bg, "numerator", 4) or 4)
                denom = int(getattr(bg, "denominator", 4) or 4)
                measure_len_ticks = float(numer) * (4.0 / float(denom)) * float(QUARTER_NOTE_UNIT)
                seq = list(getattr(bg, "beat_grouping", []) or [])
                _bar_offsets, grid_offsets = resolve_grid_layer_offsets(seq, numer, denom)
                for _ in range(int(getattr(bg, "measure_amount", 1) or 1)):
                    m_start = float(cur)
                    m_end = float(cur + measure_len_ticks)
                    if op.lt(m_end, float(a)):
                        cur = m_end
                        continue
                    if op.gt(m_start, float(b)):
                        cur = m_end
                        continue
                    boundaries = [0.0] + [float(v) for v in grid_offsets if 0.0 < float(v) < measure_len_ticks] + [
                        float(measure_len_ticks)
                    ]
                    boundaries = sorted(dict.fromkeys(round(v, 6) for v in boundaries))
                    if len(boundaries) < 2:
                        boundaries = [0.0, float(measure_len_ticks)]
                    for idx in range(len(boundaries) - 1):
                        w0 = m_start + float(boundaries[idx])
                        w1 = m_start + float(boundaries[idx + 1])
                        w0 = max(float(a), w0)
                        w1 = min(float(b), w1)
                        if op.lt(w0, w1):
                            windows.append((w0, w1))
                    cur = m_end
            return windows

        def process_beam_marker_override(default_windows: list[tuple[float, float]], markers: list) -> list[tuple[float, float]]:
            if not default_windows:
                return []
            if not markers:
                return default_windows
            windows = sorted(default_windows, key=lambda w: float(w[0]))
            for mk in sorted(markers, key=lambda m: float(getattr(m, "time", 0.0))):
                mt = float(getattr(mk, "time", 0.0) or 0.0)
                dur = float(getattr(mk, "duration", 0.0) or 0.0)
                end = mt + max(0.0, dur)
                filtered: list[tuple[float, float]] = []
                for w0, w1 in windows:
                    if op.ge(w0, end) or op.le(w1, mt):
                        filtered.append((w0, w1))
                if dur > 0.0:
                    filtered.append((mt, end))
                windows = sorted(filtered, key=lambda w: float(w[0]))
            return windows

        def marker_windows_exact(markers: list) -> list[tuple[float, float]]:
            if not markers:
                return []
            eps = max(1e-3, float(op.threshold))
            windows: list[tuple[float, float]] = []
            for mk in sorted(markers, key=lambda m: float(getattr(m, "time", 0.0))):
                mt = float(getattr(mk, "time", 0.0) or 0.0)
                dur = float(getattr(mk, "duration", 0.0) or 0.0)
                end = mt + (dur if dur > 0 else eps)
                windows.append((mt, end))
            return windows

        def assign_groups(notes_sorted: list, starts: list[float], windows: list[tuple[float, float]]) -> list[list]:
            if not notes_sorted or not windows:
                return []
            ends = [float(getattr(n, "time", 0.0) or 0.0) + float(getattr(n, "duration", 0.0) or 0.0) for n in notes_sorted]
            result: list[list] = []
            j = 0
            for t0, t1 in windows:
                j = bisect.bisect_left(starts, float(t0) - float(op.threshold), j)
                group: list = []
                k = j
                while k < len(starts):
                    s = starts[k]
                    if op.ge(s, float(t1) + float(op.threshold)):
                        break
                    e = ends[k]
                    if op.gt(e, float(t0)) and op.lt(s, float(t1)):
                        group.append(notes_sorted[k])
                    k += 1
                b = j - 1
                while b >= 0:
                    s = starts[b]
                    e = ends[b]
                    if op.gt(e, float(t0)) and op.lt(s, float(t1)):
                        group.append(notes_sorted[b])
                    b -= 1
                if group:
                    group = sorted({int(getattr(m, "_id", -1) or -1): m for m in group}.values(), key=lambda n: float(getattr(n, "time", 0.0) or 0.0))
                result.append(group)
            return result

        groups_all: dict[str, list[list]] = {"l": [], "r": []}
        windows_all: dict[str, list[tuple[float, float]]] = {"l": [], "r": []}
        marker_windows_all: dict[str, list[tuple[float, float]]] = {"l": [], "r": []}

        for hand_key in ("l", "r"):
            # notes_by_hand is built from time-sorted notes_view and preserves order,
            # so this avoids an extra per-frame sort per hand.
            notes = list(notes_by_hand.get(hand_key, []) or [])
            starts = [float(getattr(n, "time", 0.0) or 0.0) for n in notes]
            markers = list(beam_by_hand.get(hand_key, []) or [])

            start_candidates = [0.0]
            if starts:
                start_candidates.append(float(min(starts)))
            if markers:
                start_candidates.append(float(min(float(getattr(m, "time", 0.0) or 0.0) for m in markers)))
            score_start = float(min(start_candidates))

            end_candidates = [float(self._calc_base_grid_list_total_length())]
            if grid_den_times:
                end_candidates.append(float(grid_den_times[-1]))
            if starts:
                end_candidates.append(float(max(starts)))
            if markers:
                end_candidates.append(
                    float(
                        max(
                            float(getattr(m, "time", 0.0) or 0.0)
                            + max(0.0, float(getattr(m, "duration", 0.0) or 0.0))
                            for m in markers
                        )
                    )
                )
            score_end = float(max(end_candidates))

            default_windows = build_grid_windows(grid_den_times, score_start, score_end)
            windows = process_beam_marker_override(default_windows, markers)
            groups = assign_groups(notes, starts, windows) if notes else []

            groups_all[hand_key] = groups
            windows_all[hand_key] = windows
            marker_windows_all[hand_key] = marker_windows_exact(markers)

        return groups_all, windows_all, marker_windows_all

    def _build_render_cache(self) -> None:
        """Build per-frame cached, time-sorted viewport data for drawers."""
        if self._reuse_draw_cache_once and self._draw_cache is not None:
            self._reuse_draw_cache_once = False
            return
        self._draw_cache = None

        score = self.current_score()
        if score is None:
            return

        top_mm = self._view_y_mm_offset
        vp_h_mm = self._viewport_h_mm
        bottom_mm = top_mm + vp_h_mm
        zpq = score.app_state.zoom_mm_per_quarter
        bleed_mm = max(2.0, zpq * 0.25)
        time_begin = float(self.mm_to_time(top_mm - bleed_mm))
        bottom_bleed = self.viewport_bottom_bleed
        time_end = float(self.mm_to_time(bottom_mm + bleed_mm)) + bottom_bleed

        op = Operator(SHORTEST_DURATION)

        notes = score.events.note
        patched_values = self._apply_single_note_timing_cache_update(notes)
        if patched_values is not None:
            notes_sorted, starts, ends, end_pairs, end_values = patched_values
            self._note_time_cache_values = patched_values
            self._note_time_cache_key = None
        else:
            note_cache_key = self._compute_note_time_cache_key(notes)
            if self._note_time_cache_key == note_cache_key and self._note_time_cache_values is not None:
                notes_sorted, starts, ends, end_pairs, end_values = self._note_time_cache_values
            else:
                notes_sorted = sorted(notes, key=lambda n: (float(n.time), int(n.pitch)))
                starts = [float(n.time) for n in notes_sorted]
                ends = [float(n.time + n.duration) for n in notes_sorted]
                end_pairs = sorted(((ends[i], i) for i in range(len(ends))), key=lambda p: p[0])
                end_values = [p[0] for p in end_pairs]
                self._note_time_cache_key = note_cache_key
                self._note_time_cache_values = (notes_sorted, starts, ends, end_pairs, end_values)

        lo_start = bisect.bisect_left(starts, time_begin)
        hi_start = bisect.bisect_right(starts, time_end)

        lo_end_val = bisect.bisect_left(end_values, time_begin)
        hi_end_val = bisect.bisect_right(end_values, time_end)
        by_end_indices = sorted(end_pairs[j][1] for j in range(lo_end_val, hi_end_val))

        viewport_len = float(max(0.0, time_end - time_begin))
        slack = float(op.threshold)
        back_lo = bisect.bisect_left(starts, float(time_begin - viewport_len - slack))

        span_cut = bisect.bisect_right(starts, time_begin)
        # Long-span notes: started before viewport and end at/after viewport end.
        # Build from end_pairs slice instead of scanning all indices < span_cut.
        span_pairs_lo = bisect.bisect_left(end_values, time_end)
        span_indices = sorted(
            idx for _end, idx in end_pairs[span_pairs_lo:] if int(idx) < int(span_cut)
        )

        start_range = range(max(0, back_lo), max(0, hi_start))
        candidate_indices: list[int] = []
        prev_idx: int | None = None
        for idx in heapq.merge(start_range, by_end_indices, span_indices):
            if prev_idx is not None and idx == prev_idx:
                continue
            candidate_indices.append(int(idx))
            prev_idx = int(idx)

        notes_view = [notes_sorted[i] for i in candidate_indices] if candidate_indices else []

        notes_by_hand: dict[str, list] = {"l": [], "r": []}
        for m in notes_view:
            h = "l" if str(getattr(m, "hand", "l") or "l") == "l" else "r"
            notes_by_hand[h].append(m)

        beam_markers = list(getattr(score.events, "beam", []) or [])
        beam_by_hand: dict[str, list] = {"l": [], "r": []}
        for b in beam_markers:
            h = "l" if str(getattr(b, "hand", "l") or "l") == "l" else "r"
            beam_by_hand[h].append(b)
        for h in beam_by_hand:
            beam_by_hand[h] = sorted(beam_by_hand[h], key=lambda b: float(getattr(b, "time", 0.0)))

        base_grid_list = list(getattr(score, "base_grid", []) or [])
        grid_key_parts: list[tuple[int, int, int, tuple[int, ...]]] = []
        for bg in base_grid_list:
            positions = getattr(bg, "beat_grouping", None)
            positions_list = list(positions if positions is not None else (getattr(bg, "beat_grouping", []) or []))
            grid_key_parts.append(
                (
                    int(getattr(bg, "numerator", 4) or 4),
                    int(getattr(bg, "denominator", 4) or 4),
                    int(getattr(bg, "measure_amount", 1) or 1),
                    tuple(round(float(v), 6) for v in positions_list if isinstance(v, (int, float))),
                )
            )
        grid_cache_key = tuple(grid_key_parts)
        if self._grid_time_cache_key == grid_cache_key and self._grid_time_cache_values is not None:
            grid_den_times = self._grid_time_cache_values[0]
            barline_times = self._grid_time_cache_values[1]
        else:
            grid_den_times: list[float] = []
            barline_times: list[float] = []
            cur_t = 0.0
            for bg in base_grid_list:
                numer = int(getattr(bg, "numerator", 4) or 4)
                denom = int(getattr(bg, "denominator", 4) or 4)
                measure_len_ticks = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
                positions = getattr(bg, "beat_grouping", None)
                positions_list = list(positions if positions is not None else (getattr(bg, "beat_grouping", []) or []))
                bar_offsets, grid_offsets = resolve_grid_layer_offsets(positions_list, numer, denom)
                for _ in range(int(bg.measure_amount)):
                    for off in bar_offsets:
                        barline_times.append(float(cur_t + float(off)))
                    for off in grid_offsets:
                        grid_den_times.append(float(cur_t + float(off)))
                    cur_t += measure_len_ticks
            if barline_times:
                barline_times.append(float(cur_t))
            if grid_den_times:
                grid_den_times.append(float(cur_t))
            self._grid_time_cache_key = grid_cache_key
            self._grid_time_cache_values = (grid_den_times, barline_times)

        arpeggio_y_overrides: dict[int, float] = {}
        arps_all = list(getattr(score.events, "arpeggio", []) or [])
        if arps_all:
            op_arp = Operator(float(SHORTEST_DURATION))
            notes_by_pitch: dict[int, list[object]] = {}
            for n in notes_sorted:
                try:
                    pitch_key = int(getattr(n, "pitch", 0) or 0)
                except Exception:
                    continue
                if pitch_key <= 0:
                    continue
                notes_by_pitch.setdefault(pitch_key, []).append(n)
            margin_mm = float(self.margin or 0.0)
            zpq_arp = float(getattr(score.app_state, "zoom_mm_per_quarter", 25.0) or 25.0)
            semi_arp = float(self.semitone_dist or 0.5)
            layout_arp = getattr(score, "layout", None)
            width_scale_arp = max(0.05, float(getattr(layout_arp, "note_width_scaling", 1.0) or 1.0)) if layout_arp is not None else 1.0
            height_scale_arp = max(0.1, float(getattr(layout_arp, "notehead_height_scaling", 1.0) or 1.0)) if layout_arp is not None else 1.0
            base_tilt_arp = float(getattr(layout_arp, "notehead_tilt", 0.0) or 0.0) if layout_arp is not None else 0.0
            base_tilt_arp = max(-1.0, min(1.0, base_tilt_arp))

            def _t2mm(ticks: float) -> float:
                return margin_mm + (float(ticks) / float(QUARTER_NOTE_UNIT)) * zpq_arp

            for arp in arps_all:
                try:
                    base_time = float(getattr(arp, "time", 0.0) or 0.0)
                    rtime1 = float(getattr(arp, "rtime1", 0.0) or 0.0)
                    rtime2 = float(getattr(arp, "rtime2", 0.0) or 0.0)
                except Exception:
                    continue
                if abs(rtime1) <= 1e-9 and abs(rtime2) <= 1e-9:
                    continue
                pitches = [int(p) for p in (getattr(arp, "note_pitches", []) or []) if int(p) > 0]
                if len(pitches) < 2:
                    continue
                chord_notes: list[object] = []
                for pitch in pitches:
                    matches = notes_by_pitch.get(int(pitch), [])
                    match_note = None
                    for candidate in matches:
                        candidate_time = float(getattr(candidate, "time", 0.0) or 0.0)
                        if op_arp.eq(candidate_time, base_time):
                            match_note = candidate
                            break
                    if match_note is not None:
                        chord_notes.append(match_note)
                if len(chord_notes) < 2:
                    continue
                chord_sorted = sorted(chord_notes, key=lambda n: int(getattr(n, "pitch", 0) or 0))
                y_start = _t2mm(base_time + rtime1)
                y_end = _t2mm(base_time + rtime2)
                stem_xs: list[float] = []
                for note_obj in chord_sorted:
                    pitch = int(getattr(note_obj, "pitch", 0) or 0)
                    x_note = float(self.pitch_to_x(pitch))
                    stem_xs.append(float(x_note))
                x_line_start = float(stem_xs[0])
                x_line_end = float(stem_xs[-1])
                dx_line = float(x_line_end - x_line_start)
                dy_line = float(y_end - y_start)
                m_line = float(dy_line / dx_line) if abs(dx_line) > 1e-9 else 0.0
                hand_chord = str(getattr(chord_sorted[0], "hand", "l") or "l") if chord_sorted else "l"
                if hand_chord == "l":
                    anchor_note = chord_sorted[-1]
                    anchor_x = float(stem_xs[-1])
                    anchor_y = float(y_end)
                else:
                    anchor_note = chord_sorted[0]
                    anchor_x = float(stem_xs[0])
                    anchor_y = float(y_start)
                hand_anchor = str(getattr(anchor_note, "hand", "l") or "l")
                try:
                    default_black_above_anchor = bool(self._black_note_above_stem(anchor_note, layout_arp))
                except Exception:
                    default_black_above_anchor = True
                spec_anchor = resolve_notehead_spec(anchor_note, default_black_above=default_black_above_anchor)
                is_up_anchor = bool(getattr(spec_anchor, "is_up", False))
                outline_anchor = sheared_notehead_outline_points(
                    hand=hand_anchor,
                    is_up=is_up_anchor,
                    semitone_space_mm=semi_arp,
                    width_scale=width_scale_arp,
                    height_scale=height_scale_arp,
                    base_tilt=base_tilt_arp,
                    sample_count=128,
                )
                edge_anchor = support_point_from_outline_points(
                    outline_anchor,
                    m_line=m_line,
                    choose_max=bool(is_up_anchor),
                )
                stem_end_x = float(anchor_x + edge_anchor[0])
                stem_end_y = float(anchor_y + edge_anchor[1])
                b_line = float(stem_end_y - (m_line * stem_end_x))
                support_cache: dict[tuple[str, bool], float] = {}
                for i, (note_obj, x_stem) in enumerate(zip(chord_sorted, stem_xs)):
                    nid = int(getattr(note_obj, "_id", 0) or 0)
                    y_line = float((m_line * float(x_stem)) + b_line)
                    if (hand_chord == "l" and i == len(chord_sorted) - 1) or (hand_chord == "r" and i == 0):
                        arpeggio_y_overrides[nid] = float(y_end if hand_chord == "l" else y_start)
                        continue
                    hand_local = str(getattr(note_obj, "hand", "l") or "l")
                    try:
                        default_black_above_local = bool(self._black_note_above_stem(note_obj, layout_arp))
                    except Exception:
                        default_black_above_local = True
                    spec_local = resolve_notehead_spec(note_obj, default_black_above=default_black_above_local)
                    is_up_local = bool(getattr(spec_local, "is_up", False))
                    cache_key = (hand_local, is_up_local)
                    if cache_key not in support_cache:
                        support_cache[cache_key] = sheared_notehead_support_v(
                            hand=hand_local,
                            is_up=is_up_local,
                            semitone_space_mm=semi_arp,
                            width_scale=width_scale_arp,
                            height_scale=height_scale_arp,
                            base_tilt=base_tilt_arp,
                            m_line=m_line,
                            sample_count=64,
                        )
                    v_support = float(support_cache[cache_key])
                    arpeggio_y_overrides[nid] = float(y_line - v_support)
        self._arpeggio_y_overrides = arpeggio_y_overrides

        beam_groups_by_hand, beam_windows_by_hand, beam_marker_windows_by_hand = self._build_beam_groups_cache(
            score=score,
            op=op,
            notes_by_hand=notes_by_hand,
            beam_by_hand=beam_by_hand,
            grid_den_times=grid_den_times,
        )

        self._draw_cache = {
            "time_begin": time_begin,
            "time_end": time_end,
            "op": op,
            "notes_sorted": notes_sorted,
            "starts": starts,
            "ends": ends,
            "candidate_indices": candidate_indices,
            "notes_view": notes_view,
            "notes_by_hand": notes_by_hand,
            "beam_by_hand": beam_by_hand,
            "beam_groups_by_hand": beam_groups_by_hand,
            "beam_windows_by_hand": beam_windows_by_hand,
            "beam_marker_windows_by_hand": beam_marker_windows_by_hand,
            "grid_den_times": grid_den_times,
            "barline_times": barline_times,
        }
