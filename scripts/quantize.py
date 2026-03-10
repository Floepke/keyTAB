from __future__ import annotations

from scripting import DialogSpec, BoolField, LabelField
from utils.CONSTANT import SHORTEST_DURATION
from utils.operator import Operator


def build_dialog(ctx):
    return DialogSpec(
        title="Quantize",
        fields=[
            LabelField(
                text="Snap size is set using the Snap Size selector.",
            ),
            BoolField(
                name="quantize_starts",
                label="Quantize Starts",
                default=True,
            ),
            BoolField(
                name="quantize_ends",
                label="Quantize Ends",
                default=True,
            ),
        ],
    )


def _quantize(ctx, values):
    editor = getattr(ctx, "_editor", None)
    if editor is None:
        return

    if not bool(getattr(editor, "_selection_active", False)):
        return

    q_start = bool(values.get("quantize_starts", True))
    q_end = bool(values.get("quantize_ends", True))
    if not q_start and not q_end:
        return

    units = float(max(1e-6, getattr(editor, "snap_size_units", 0.0) or 0.0))
    op = Operator(float(SHORTEST_DURATION))
    sel = editor.detect_events_from_time_window(editor._sel_start_units, editor._sel_end_units - 0.1)
    notes = sel.get("note", []) if isinstance(sel, dict) else []
    if not notes:
        return

    def _q(value: float) -> float:
        return float(round(float(value) / units) * units)

    changed = False
    for note in notes:
        t0 = float(getattr(note, "time", 0.0) or 0.0)
        dur = float(getattr(note, "duration", 0.0) or 0.0)
        t1 = t0 + max(0.0, dur)

        qt0 = max(0.0, _q(t0)) if q_start else t0
        qt1 = max(0.0, _q(t1)) if q_end else t1
        if qt1 <= qt0:
            qt1 = qt0 + units
        qdur = max(units, qt1 - qt0)

        if op.ne(float(qt0), float(t0)) or op.ne(float(qdur), float(dur)):
            setattr(note, "time", float(qt0))
            setattr(note, "duration", float(qdur))
            changed = True

    if changed:
        try:
            editor.update_score_length()
        except Exception:
            pass


# Called when the Preview button is pressed
def preview(ctx, values):
    _quantize(ctx, values)
    ctx.refresh()


# Called when OK is pressed
def apply(ctx, values):
    _quantize(ctx, values)
    ctx.refresh()
