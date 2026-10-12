from __future__ import annotations

import math

class HitTestingMixin:
    # ---- Hit rectangles ----
    def register_hit_rect(self, type: str, _id: int, x1: float, y1: float, x2: float, y2: float, **extra) -> None:
        """Register a clickable rectangle for hit detection."""
        cx = (float(x1) + float(x2)) * 0.5
        cy = (float(y1) + float(y2)) * 0.5
        record: dict = {
            "type": str(type),
            "_id": int(_id),
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "cx": cx,
            "cy": cy,
        }
        record.update(extra)
        self._hit_rects.append(record)

    def hit_test_hit_rect(self, x_mm: float, y_mm: float, type: str) -> dict | None:
        """Return the best-matching hit rect dict of the given type at (x_mm, y_mm)."""
        x_mm = float(x_mm)
        y_mm = float(y_mm)
        if type == "text":
            candidates = []
            for r in self._hit_rects:
                if r.get("type") != "text":
                    continue
                if float(r["x1"]) <= x_mm <= float(r["x2"]) and float(r["y1"]) <= y_mm <= float(r["y2"]):
                    area = max(0.0, (float(r["x2"]) - float(r["x1"])) * (float(r["y2"]) - float(r["y1"])))
                    priority = 0 if r.get("kind") == "handle" else 1
                    candidates.append((priority, area, r))
            if not candidates:
                return None
            candidates.sort(key=lambda t: (t[0], t[1]))
            return candidates[0][2]

        matches = []
        for r in self._hit_rects:
            if r.get("type") != type:
                continue
            if float(r["x1"]) <= x_mm <= float(r["x2"]) and float(r["y1"]) <= y_mm <= float(r["y2"]):
                dx = x_mm - float(r["cx"])
                dy = y_mm - float(r["cy"])
                matches.append((dx * dx + dy * dy, r))
        if not matches:
            return None
        matches.sort(key=lambda t: t[0])
        return matches[0][1]

    def _px_to_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        """Convert logical (Qt) pixel coordinates to absolute drawing-space mm.

        Page-space mm is the unrotated drawing coordinate system used by the
        drawers and registered hit rectangles.
        """
        try:
            return self.widget_px_to_page_mm(float(x_px), float(y_px))
        except Exception:
            pass
        w_px_per_mm = float(getattr(self, "_widget_px_per_mm", 1.0) or 1.0)
        if w_px_per_mm <= 0:
            return 0.0, 0.0
        scroll = float(getattr(self, "_view_y_mm_offset", 0.0) or 0.0)
        x_mm = float(x_px) / w_px_per_mm
        y_mm = float(y_px) / w_px_per_mm + scroll
        return x_mm, y_mm

    # ---- Hit rect backward-compatible wrappers ----
    def hit_test_note_id(self, x_px: float, y_px: float) -> int | None:
        x_mm, y_mm = self._px_to_mm(x_px, y_px)
        r = self.hit_test_hit_rect(x_mm, y_mm, "note")
        return int(r["_id"]) if r is not None else None

    def hit_test_tempo(self, x_px: float, y_px: float) -> int | None:
        x_mm, y_mm = self._px_to_mm(x_px, y_px)
        r = self.hit_test_hit_rect(x_mm, y_mm, "tempo")
        return int(r["_id"]) if r is not None else None

    def hit_test_grace_note_id(self, x_px: float, y_px: float) -> int | None:
        x_mm, y_mm = self._px_to_mm(x_px, y_px)
        r = self.hit_test_hit_rect(x_mm, y_mm, "grace_note")
        return int(r["_id"]) if r is not None else None

    def hit_test_arpeggio_handle(self, x_px: float, y_px: float) -> int | None:
        x_mm, y_mm = self._px_to_mm(x_px, y_px)
        r = self.hit_test_hit_rect(x_mm, y_mm, "arpeggio")
        return int(r["_id"]) if r is not None else None

    def hit_test_text(self, x_px: float, y_px: float):
        x_mm, y_mm = self._px_to_mm(x_px, y_px)
        return self.hit_test_text_mm(x_mm, y_mm)

    def hit_test_text_mm(self, x_mm: float, y_mm: float):
        r = self.hit_test_hit_rect(float(x_mm), float(y_mm), "text")
        if r is None:
            return (None, None, None)
        return (int(r["_id"]), r.get("kind") == "handle", r)

    def hit_test_hairpin_mm(self, x_mm: float, y_mm: float):
        r = self.hit_test_hit_rect(float(x_mm), float(y_mm), "hairpin")
        if r is None:
            return (None, None, None)
        hp_id = int(r["_id"])
        hp_type = str(r.get("htype", ""))
        handle = str(r.get("handle", ""))
        score = self.current_score()
        if score is None:
            return (None, None, None)
        score_events = self.current_events(score)
        for ev in (getattr(score_events, hp_type, []) or []):
            if int(getattr(ev, "_id", -1) or -1) == hp_id:
                return (ev, hp_type, handle)
        return (None, None, None)

    def hit_test_dynamic_symbol_mm(self, x_mm: float, y_mm: float):
        r = self.hit_test_hit_rect(float(x_mm), float(y_mm), "dynamic_symbol")
        if r is None:
            return (None, None, None)
        symbol_id = int(r["_id"])
        score = self.current_score()
        if score is None:
            return (None, None, None)
        score_events = self.current_events(score)
        for ev in (getattr(score_events, "dynamic_symbol", []) or []):
            if int(getattr(ev, "_id", -1) or -1) == symbol_id:
                return (ev, "dynamic_symbol", "")
        return (None, None, None)

    def hit_test_count_line_mm(self, x_mm: float, y_mm: float):
        r = self.hit_test_hit_rect(float(x_mm), float(y_mm), "count_line")
        if r is None:
            return (None, None)
        count_line_id = int(r["_id"])
        part = str(r.get("part", "line") or "line")
        score = self.current_score()
        if score is None:
            return (None, None)
        score_events = self.current_events(score)
        for ev in (getattr(score_events, "count_line", []) or []):
            if int(getattr(ev, "_id", -1) or -1) == count_line_id:
                return (ev, part)
        return (None, None)

    def hit_test_line_mm(self, x_mm: float, y_mm: float):
        score = self.current_score()
        if score is None:
            return (None, None)
        score_events = self.current_events(score)
        if score_events is None:
            return (None, None)

        best_handle = None
        best_handle_distance = float('inf')
        best_line = None
        best_line_distance = float('inf')
        handle_half_size = max(2.0, float(getattr(self, "semitone_dist", 2.5) or 2.5)) * 0.7
        for line in getattr(score_events, "line", []) or []:
            try:
                x1 = float(self.relative_c4pitch_to_x(int(getattr(line, "rpitch1", 0) or 0)))
                y1 = float(self.time_to_mm(float(getattr(line, "time1", 0.0) or 0.0)))
                x2 = float(self.relative_c4pitch_to_x(int(getattr(line, "rpitch2", 0) or 0)))
                y2 = float(self.time_to_mm(float(getattr(line, "time2", 0.0) or 0.0)))
            except Exception:
                continue

            for part, px, py in (("start", x1, y1), ("end", x2, y2)):
                distance = math.hypot(float(x_mm) - px, float(y_mm) - py)
                if (
                    abs(float(x_mm) - px) <= handle_half_size + 1e-9
                    and abs(float(y_mm) - py) <= handle_half_size + 1e-9
                    and distance < best_handle_distance
                ):
                    best_handle = (line, part)
                    best_handle_distance = distance

            dx = x2 - x1
            dy = y2 - y1
            length_squared = dx * dx + dy * dy
            if length_squared <= 1e-9:
                continue
            ratio = max(0.0, min(1.0, ((float(x_mm) - x1) * dx + (float(y_mm) - y1) * dy) / length_squared))
            nearest_x = x1 + ratio * dx
            nearest_y = y1 + ratio * dy
            distance = math.hypot(float(x_mm) - nearest_x, float(y_mm) - nearest_y)
            if distance <= 3.0 and distance < best_line_distance:
                best_line = (line, "line")
                best_line_distance = distance

        if best_handle is not None:
            return best_handle
        return best_line if best_line is not None else (None, None)
