from __future__ import annotations

import dataclasses

from PySide6 import QtCore

from scripting import ActionButtonField, DialogSpec, LabelField


ACTION_REVERSE = "reverse"
_CTX_CLICK_COUNT_ATTR = "_reverse_click_count"


def _has_selection(ctx) -> bool:
    editor = getattr(ctx, "_editor", None)
    if editor is None:
        return False
    return bool(getattr(editor, "_selection_active", False))


def build_dialog(ctx):
    setattr(ctx, _CTX_CLICK_COUNT_ATTR, 0)
    return DialogSpec(
        title=QtCore.QCoreApplication.translate("ReverseSelectionAction", "Reverse"),
        fields=[
            LabelField(
                text=QtCore.QCoreApplication.translate(
                    "ReverseSelectionAction",
                    "Reverses events in time so the passage plays back as if going backwards. Right-click and drag to select a region; without a selection the whole file is edited.",
                )
            ),
            ActionButtonField(
                name=ACTION_REVERSE,
                label=QtCore.QCoreApplication.translate("ReverseSelectionAction", "Reverse"),
            ),
        ],
    )


def _is_reverse_action(values) -> bool:
    if not isinstance(values, dict):
        return False
    return str(values.get("_action", "") or "") == ACTION_REVERSE


def _numeric(v) -> bool:
    return isinstance(v, (int, float))


def _field_names(ev) -> list[str]:
    if dataclasses.is_dataclass(ev):
        return [f.name for f in dataclasses.fields(type(ev))]
    return list(getattr(ev, "__dict__", {}).keys())


def _event_time_span(ev) -> tuple[float, float]:
    """Return (start, start+dur) for an event, or (0, 0) if undetectable."""
    names = _field_names(ev)
    if "time" in names:
        try:
            t = float(getattr(ev, "time", 0.0) or 0.0)
        except Exception:
            return (0.0, 0.0)
        dur = 0.0
        if "duration" in names:
            try:
                d = getattr(ev, "duration", 0.0)
                if _numeric(d):
                    dur = max(0.0, float(d))
            except Exception:
                pass
        return (t, t + dur)
    # Multi-anchor: use all *_time values
    times = []
    for name in names:
        if not str(name).endswith("_time"):
            continue
        try:
            v = getattr(ev, name)
            if _numeric(v):
                times.append(float(v))
        except Exception:
            pass
    if times:
        return (min(times), max(times))
    return (0.0, 0.0)


def _reverse_event_time(ev, start: float, end: float) -> None:
    names = _field_names(ev)
    if "time" in names:
        try:
            t = float(getattr(ev, "time", 0.0) or 0.0)
        except Exception:
            return
        dur = 0.0
        if "duration" in names:
            try:
                d = getattr(ev, "duration", 0.0)
                if _numeric(d):
                    dur = max(0.0, float(d))
            except Exception:
                pass
        setattr(ev, "time", float(start + end - (t + dur)))
        return
    for name in names:
        if not str(name).endswith("_time"):
            continue
        try:
            v = getattr(ev, name)
        except Exception:
            continue
        if not _numeric(v):
            continue
        setattr(ev, name, float(start + end - float(v)))


def _all_score_events(ctx) -> list:
    """Collect all events from the score (any list on score.events)."""
    score = getattr(ctx, "score", None)
    if score is None:
        return []
    events_obj = getattr(score, "events", None)
    if events_obj is None:
        return []
    out: list = []
    seen: set[int] = set()
    try:
        if dataclasses.is_dataclass(events_obj):
            field_names = [f.name for f in dataclasses.fields(type(events_obj))]
        else:
            field_names = list(getattr(events_obj, "__dict__", {}).keys())
        for name in field_names:
            lst = getattr(events_obj, name, None)
            if not isinstance(lst, list):
                continue
            for ev in lst:
                k = id(ev)
                if k not in seen:
                    seen.add(k)
                    out.append(ev)
    except Exception:
        pass
    return out


def _get_window_and_events(ctx) -> tuple[float, float, list] | None:
    editor = getattr(ctx, "_editor", None)
    if _has_selection(ctx) and editor is not None:
        try:
            start = float(getattr(editor, "_sel_start_units", 0.0))
            end = float(getattr(editor, "_sel_end_units", 0.0)) - 0.1
            a, b = min(start, end), max(start, end)
            detected = editor.detect_events_from_time_window(a, b)
            out: list = []
            seen: set[int] = set()
            for ev_list in detected.values():
                if not isinstance(ev_list, list):
                    continue
                for ev in ev_list:
                    k = id(ev)
                    if k not in seen:
                        seen.add(k)
                        out.append(ev)
            return (a, b, out)
        except Exception:
            pass

    # Full-file fallback: compute window from actual event times.
    all_evs = _all_score_events(ctx)
    if not all_evs:
        return None
    min_t = float("inf")
    max_t = float("-inf")
    for ev in all_evs:
        t0, t1 = _event_time_span(ev)
        if t0 < min_t:
            min_t = t0
        if t1 > max_t:
            max_t = t1
    if min_t == float("inf") or max_t == float("-inf") or max_t <= min_t:
        return None
    return (min_t, max_t, all_evs)


def _do_reverse(ctx) -> None:
    result = _get_window_and_events(ctx)
    if result is None:
        return
    start, end, events = result
    for ev in events:
        _reverse_event_time(ev, start, end)


def _get_click_count(ctx) -> int:
    try:
        return max(0, int(getattr(ctx, _CTX_CLICK_COUNT_ATTR, 0) or 0))
    except Exception:
        return 0


def _increment_click_count(ctx) -> int:
    value = _get_click_count(ctx) + 1
    setattr(ctx, _CTX_CLICK_COUNT_ATTR, value)
    return value


def preview(ctx, values):
    if not _is_reverse_action(values):
        return
    click_count = _increment_click_count(ctx)
    if (click_count % 2) == 1:
        _do_reverse(ctx)
    ctx.refresh()


def apply(ctx, values):
    click_count = _get_click_count(ctx)
    if click_count == 0 and _is_reverse_action(values):
        click_count = 1
    if (click_count % 2) == 1:
        _do_reverse(ctx)
    ctx.refresh()
