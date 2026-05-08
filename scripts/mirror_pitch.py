from __future__ import annotations

from PySide6 import QtCore

from scripting.spec import ActionButtonField, DialogSpec, LabelField


ACTION_MIRROR = "mirror_pitch"
_CTX_CLICK_COUNT_ATTR = "_mirror_pitch_click_count"


def _has_selection(ctx) -> bool:
    editor = getattr(ctx, "_editor", None)
    if editor is None:
        return False
    return bool(getattr(editor, "_selection_active", False))


def build_dialog(ctx):
    setattr(ctx, _CTX_CLICK_COUNT_ATTR, 0)
    return DialogSpec(
        title=QtCore.QCoreApplication.translate("MirrorPitchSelectionAction", "Mirror Pitch"),
        fields=[
            LabelField(
                text=QtCore.QCoreApplication.translate(
                    "MirrorPitchSelectionAction",
                    "Mirrors pitches around their lowest and highest value. Right-click and drag to select a region; without a selection the whole file is edited.",
                )
            ),
            ActionButtonField(
                name=ACTION_MIRROR,
                label=QtCore.QCoreApplication.translate("MirrorPitchSelectionAction", "Mirror pitch"),
            ),
        ],
    )


def _pitch_events(ctx) -> list:
    """Return selected notes+grace_notes, or all of them if no selection is active."""
    editor = getattr(ctx, "_editor", None)
    score = getattr(ctx, "score", None)
    if editor is None or score is None:
        return []
    try:
        if _has_selection(ctx):
            start = float(getattr(editor, "_sel_start_units", 0.0))
            end = float(getattr(editor, "_sel_end_units", 0.0)) - 0.1
            detected = editor.detect_events_from_time_window(min(start, end), max(start, end))
            return list(detected.get("note", []) or []) + list(detected.get("grace_note", []) or [])
        else:
            events = getattr(score, "events", None)
            notes = list(getattr(events, "note", []) or [])
            grace = list(getattr(events, "grace_note", []) or [])
            return notes + grace
    except Exception:
        return []


def _is_mirror_action(values) -> bool:
    if not isinstance(values, dict):
        return False
    return str(values.get("_action", "") or "") == ACTION_MIRROR


def _get_click_count(ctx) -> int:
    try:
        return max(0, int(getattr(ctx, _CTX_CLICK_COUNT_ATTR, 0) or 0))
    except Exception:
        return 0


def _increment_click_count(ctx) -> int:
    value = _get_click_count(ctx) + 1
    setattr(ctx, _CTX_CLICK_COUNT_ATTR, value)
    return value


def _do_mirror(ctx) -> None:
    events = _pitch_events(ctx)
    if len(events) <= 1:
        return

    pitches: list[int] = [int(getattr(ev, "pitch", 0) or 0) for ev in events]
    pitches = [p for p in pitches if p > 0]
    if len(pitches) <= 1:
        return

    low = min(pitches)
    high = max(pitches)

    for ev in events:
        p = int(getattr(ev, "pitch", 0) or 0)
        if p <= 0:
            continue
        mirrored = max(1, min(88, int(low + high - p)))
        setattr(ev, "pitch", mirrored)


def preview(ctx, values):
    if not _is_mirror_action(values):
        return
    click_count = _increment_click_count(ctx)
    if (click_count % 2) == 1:
        _do_mirror(ctx)
    ctx.refresh()


def apply(ctx, values):
    click_count = _get_click_count(ctx)
    if click_count == 0 and _is_mirror_action(values):
        click_count = 1
    if (click_count % 2) == 1:
        _do_mirror(ctx)
    ctx.refresh()
