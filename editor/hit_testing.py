from __future__ import annotations


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
        for ev in (getattr(score.events, hp_type, []) or []):
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
        for ev in (getattr(score.events, "dynamic_symbol", []) or []):
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
        for ev in (getattr(score.events, "count_line", []) or []):
            if int(getattr(ev, "_id", -1) or -1) == count_line_id:
                return (ev, part)
        return (None, None)
