from __future__ import annotations

from PySide6 import QtCore

from scripting.spec import ActionButtonField, DialogSpec, LabelField


ACTION_SWITCH = "switch_hands"
_CTX_CLICK_COUNT_ATTR = "_switch_hands_click_count"


def _bust_editor_note_cache(ctx) -> None:
    """Force editor to rebuild its note-time cache on next repaint.

    _compute_note_time_cache_key hashes only time/duration/pitch/_id, not hand.
    After a snapshot restore the new note objects hash identically and the cache
    reuses stale references with original hand values.  Nulling the key forces a
    rebuild so the editor canvas reflects hand changes.
    """
    editor = getattr(ctx, "_editor", None)
    if editor is None:
        return
    editor._note_time_cache_key = None
    editor._note_time_cache_values = None


def _has_selection(ctx) -> bool:
    editor = getattr(ctx, "_editor", None)
    if editor is None:
        return False
    return bool(getattr(editor, "_selection_active", False))


def build_dialog(ctx):
    setattr(ctx, _CTX_CLICK_COUNT_ATTR, 0)
    return DialogSpec(
        title=QtCore.QCoreApplication.translate("SwitchHandsSelectionAction", "Switch Hands"),
        fields=[
            LabelField(
                text=QtCore.QCoreApplication.translate(
                    "SwitchHandsSelectionAction",
                    "Switches the hand (left ↔ right) of note events. Right-click and drag to select a region; without a selection the whole file is edited.",
                )
            ),
            ActionButtonField(
                name=ACTION_SWITCH,
                label=QtCore.QCoreApplication.translate("SwitchHandsSelectionAction", "Switch hands"),
            ),
        ],
    )


def _is_switch_action(values) -> bool:
    if not isinstance(values, dict):
        return False
    return str(values.get("_action", "") or "") == ACTION_SWITCH


def _hand_events(ctx) -> list:
    """Return selected notes+beams, or all notes+beams if no selection is active."""
    editor = getattr(ctx, "_editor", None)
    score = getattr(ctx, "score", None)
    if editor is None or score is None:
        return []
    try:
        if _has_selection(ctx):
            start = float(getattr(editor, "_sel_start_units", 0.0))
            end = float(getattr(editor, "_sel_end_units", 0.0)) - 0.1
            detected = editor.detect_events_from_time_window(min(start, end), max(start, end))
            out: list = []
            seen: set[int] = set()
            for ev_list in (detected.get("note", []), detected.get("beam", [])):
                if not isinstance(ev_list, list):
                    continue
                for ev in ev_list:
                    k = id(ev)
                    if k not in seen:
                        seen.add(k)
                        out.append(ev)
            return out
        else:
            events = getattr(score, "events", None)
            notes = list(getattr(events, "note", []) or [])
            beams = list(getattr(events, "beam", []) or [])
            return notes + beams
    except Exception:
        return []


def _get_click_count(ctx) -> int:
    try:
        return max(0, int(getattr(ctx, _CTX_CLICK_COUNT_ATTR, 0) or 0))
    except Exception:
        return 0


def _increment_click_count(ctx) -> int:
    value = _get_click_count(ctx) + 1
    setattr(ctx, _CTX_CLICK_COUNT_ATTR, value)
    return value


def _do_switch(ctx) -> None:
    for ev in _hand_events(ctx):
        h = str(getattr(ev, "hand", "l") or "l")
        setattr(ev, "hand", "r" if h == "l" else "l")


def preview(ctx, values):
    if not _is_switch_action(values):
        return
    click_count = _increment_click_count(ctx)
    if (click_count % 2) == 1:
        _do_switch(ctx)
    _bust_editor_note_cache(ctx)
    ctx.refresh()


def apply(ctx, values):
    click_count = _get_click_count(ctx)
    if click_count == 0 and _is_switch_action(values):
        click_count = 1
    if (click_count % 2) == 1:
        _do_switch(ctx)
    _bust_editor_note_cache(ctx)
    ctx.refresh()
