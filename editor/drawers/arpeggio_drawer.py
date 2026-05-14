from __future__ import annotations

from typing import TYPE_CHECKING, cast

from symbol_design.noteheads import resolve_notehead_spec, sheared_notehead_support_v
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import SHORTEST_DURATION
from utils.operator import Operator

if TYPE_CHECKING:
    from editor.editor import Editor
    from file_model.events.arpeggio import Arpeggio


class ArpeggioDrawerMixin:
    _arp_time_op: Operator = Operator(float(SHORTEST_DURATION))

    def _resolve_arpeggio_notes(self, arp: "Arpeggio", notes_by_time_pitch: dict[tuple[int, int], object]) -> list[object]:
        """Resolve arpeggio member notes by stable (time, pitch) keys."""
        base_time_key = int(round(float(getattr(arp, "time", 0.0) or 0.0)))
        pitches = [int(p) for p in (getattr(arp, "note_pitches", []) or []) if int(p) > 0]
        if len(pitches) < 2:
            return []
        resolved = []
        for pitch in pitches:
            note_obj = notes_by_time_pitch.get((base_time_key, pitch))
            if note_obj is not None:
                resolved.append(note_obj)
        return resolved

    def _arpeggio_time_window(self, base_time: float, rtime1: float, rtime2: float) -> tuple[float, float]:
        return (float(base_time) + float(rtime1), float(base_time) + float(rtime2))

    def draw_arpeggio(self, du: DrawUtil) -> None:
        self = cast("Editor", self)
        score = self.current_score()
        if score is None:
            return

        arps = list(getattr(score.events, "arpeggio", []) or [])

        notes_all = list(getattr(score.events, "note", []) or [])
        notes_by_time_pitch: dict[tuple[int, int], object] = {
            (int(round(float(getattr(n, "time", 0.0) or 0.0))), int(getattr(n, "pitch", 0) or 0)): n
            for n in notes_all
        }
        semi = float(self.semitone_dist or 0.5)
        layout = getattr(score, "layout", None)
        scale = float(getattr(layout, "scale", 1.0) or 1.0) if layout is not None else 1.0
        stem_w = float(getattr(layout, "note_stem_thickness_mm", 0.8) or 0.8) * scale if layout is not None else 0.8
        stem_len = float(getattr(layout, "note_stem_length_semitone", 3.0) or 3.0) * semi
        width_scale = max(0.05, float(getattr(layout, "note_width_scaling", 1.0) or 1.0)) if layout is not None else 1.0
        height_scale = max(0.1, float(getattr(layout, "notehead_height_scaling", 1.0) or 1.0)) if layout is not None else 1.0
        base_tilt = float(getattr(layout, "notehead_tilt", 0.0) or 0.0) if layout is not None else 0.0
        base_tilt = max(-1.0, min(1.0, base_tilt))

        active_tool_name = str(getattr(getattr(self, "_tool", None), "TOOL_NAME", "") or "")
        show_handles = active_tool_name == "arpeggio"
        arp_keys: set[tuple[int, tuple[int, ...]]] = set()

        for arp in arps:
            try:
                base_time = float(getattr(arp, "time", 0.0) or 0.0)
                rtime1 = float(getattr(arp, "rtime1", 0.0) or 0.0)
                rtime2 = float(getattr(arp, "rtime2", 0.0) or 0.0)
                arp_id = int(getattr(arp, "_id", 0) or 0)
            except Exception:
                continue

            arp_pitches = tuple(sorted(int(p) for p in (getattr(arp, "note_pitches", []) or []) if int(p) > 0))
            if len(arp_pitches) >= 2:
                arp_keys.add((int(round(base_time)), arp_pitches))

            chord_notes = self._resolve_arpeggio_notes(arp, notes_by_time_pitch)
            if len(chord_notes) < 2:
                continue

            start_t, end_t = self._arpeggio_time_window(base_time, rtime1, rtime2)
            chord_sorted = sorted(chord_notes, key=lambda n: int(getattr(n, "pitch", 0) or 0))
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
                            semitone_space_mm=semi,
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
                hand = str(getattr(chord_sorted[0], "hand", "l") or "l") if chord_sorted else "l"
                dx = float(x_line_end - x_line_start)
                dy = float(y1 - y0)
                seg_len = (dx * dx + dy * dy) ** 0.5
                if seg_len <= 1e-6:
                    ux, uy = (1.0, 0.0)
                else:
                    ux, uy = (dx / seg_len, dy / seg_len)

                # Keep handles/anchors at note positions, and extend only the outer stem tip
                # in hand direction as a rotated stem extension.
                if hand == "l":
                    tip_x = float(x_line_start - (ux * stem_len))
                    tip_y = float(y0 - (uy * stem_len))
                    stem_points = [
                        (tip_x, tip_y),
                        (x_line_start, y0),
                        (x_line_end, y1),
                    ]
                else:
                    tip_x = float(x_line_end + (ux * stem_len))
                    tip_y = float(y1 + (uy * stem_len))
                    stem_points = [
                        (x_line_start, y0),
                        (x_line_end, y1),
                        (tip_x, tip_y),
                    ]

                du.add_polyline(
                    stem_points,
                    stroke_color=self.notation_color,
                    stroke_width_mm=stem_w,
                    id=arp_id,
                    tags=["chord_connect"],
                )

            x_values = [p[0] for p in projected]
            y_values = [p[1] for p in projected]
            self.register_hit_rect(
                "arpeggio",
                arp_id,
                min(x_values) - (semi * 0.5),
                min(y_values) - (semi * 0.5),
                max(x_values) + (semi * 0.5),
                max(y_values) + (semi * 0.5),
                handle="body",
            )

            if show_handles:
                handle_r = semi
                for handle_name, (hx, hy) in (("low", projected[0]), ("high", projected[-1])):
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

        # Synthetic handles for plain chords with no arpeggio object yet.
        # These are draggable; tool code creates model arpeggio lazily on first drag.
        chord_groups: dict[tuple[int, str], list[object]] = {}
        for note_obj in notes_all:
            try:
                t_key = int(round(float(getattr(note_obj, "time", 0.0) or 0.0)))
            except Exception:
                t_key = 0
            hand = str(getattr(note_obj, "hand", "l") or "l")
            chord_groups.setdefault((t_key, hand), []).append(note_obj)

        for (_t_key, _hand), chord_notes in chord_groups.items():
            if len(chord_notes) < 2:
                continue

            pitches = tuple(sorted(set(int(getattr(n, "pitch", 0) or 0) for n in chord_notes if int(getattr(n, "pitch", 0) or 0) > 0)))
            if len(pitches) < 2:
                continue

            base_time = float(getattr(chord_notes[0], "time", 0.0) or 0.0)
            key = (int(round(base_time)), pitches)
            if key in arp_keys:
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
            handle_r = semi
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
