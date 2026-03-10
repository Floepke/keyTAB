from __future__ import annotations

from scripting import DialogSpec, BoolField, IntField


def build_dialog(ctx):
    return DialogSpec(
        title="Transpose Notes",
        fields=[
            IntField(
                name="semitones",
                label="Semitones",
                minimum=-24,
                maximum=24,
                step=1,
                default=0,
            ),
            BoolField(
                name="selected_only",
                label="Only selected notes",
                default=True,
            ),
        ],
    )


def _iter_target_notes(ctx, selected_only: bool):
    score = ctx.score
    editor = getattr(ctx, "_editor", None)

    if bool(selected_only) and editor is not None:
        try:
            if bool(getattr(editor, "_selection_active", False)):
                start = float(getattr(editor, "_sel_start_units", 0.0))
                end = float(getattr(editor, "_sel_end_units", 0.0))
                detected = editor.detect_events_from_time_window(start, end - 0.1)
                notes = list(detected.get("note", []) or [])
                return notes
        except Exception:
            pass

    return list(getattr(score.events, "note", []) or [])


def _transpose_notes(ctx, values):
    semitones = int(values.get("semitones", 0) or 0)
    selected_only = bool(values.get("selected_only", True))
    if semitones == 0:
        return

    notes = _iter_target_notes(ctx, selected_only)
    for note in notes:
        pitch = int(getattr(note, "pitch", 0) or 0)
        if pitch <= 0:
            continue
        new_pitch = max(1, min(88, pitch + semitones))
        setattr(note, "pitch", int(new_pitch))


# Preview is called when user presses "Preview"
def preview(ctx, values):
    _transpose_notes(ctx, values)
    ctx.refresh()


# Apply is called when user presses "OK"
def apply(ctx, values):
    _transpose_notes(ctx, values)
    ctx.refresh()
