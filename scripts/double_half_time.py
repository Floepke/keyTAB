from __future__ import annotations

from dataclasses import fields, is_dataclass

from scripting import ActionButtonField, DialogSpec, LabelField


ACTION_HALF = "half"
ACTION_DOUBLE = "double"


def build_dialog(ctx):
    return DialogSpec(
        title="Half Time / Double Time",
        fields=[
            LabelField(
                text="Preview directly by clicking one of the actions below.",
            ),
            ActionButtonField(
                name=ACTION_HALF,
                label="half time (÷2)",
            ),
            ActionButtonField(
                name=ACTION_DOUBLE,
                label="double time (×2)",
            ),
        ],
    )


def _is_time_like_field(name: str) -> bool:
    if name in ("time", "duration"):
        return True
    return name.endswith("_time") or name.endswith("_duration")


def _scale_event_object_time_fields(ev, factor: float) -> None:
    if is_dataclass(ev):
        names = [f.name for f in fields(ev)]
    else:
        names = list(getattr(ev, "__dict__", {}).keys())

    for name in names:
        if not _is_time_like_field(str(name)):
            continue
        try:
            val = getattr(ev, name)
        except Exception:
            continue
        if not isinstance(val, (int, float)):
            continue
        try:
            setattr(ev, name, float(val) * float(factor))
        except Exception:
            continue


def _apply_scale(ctx, factor: float) -> None:
    score = ctx.score
    events = getattr(score, "events", None)
    if events is None:
        return

    for _event_name, ev_list in getattr(events, "__dict__", {}).items():
        if not isinstance(ev_list, list):
            continue
        for ev in ev_list:
            _scale_event_object_time_fields(ev, factor)

    editor = getattr(ctx, "_editor", None)
    if editor is not None:
        try:
            editor.update_score_length()
        except Exception:
            pass


def _factor_from_values(values: dict | None) -> float | None:
    if not isinstance(values, dict):
        return None
    action = str(values.get("_action", "") or "")
    if action == ACTION_HALF:
        return 0.5
    if action == ACTION_DOUBLE:
        return 2.0
    return None


def preview(ctx, values):
    factor = _factor_from_values(values)
    if factor is None:
        return
    _apply_scale(ctx, factor)
    ctx.refresh()


def apply(ctx, values):
    factor = _factor_from_values(values)
    if factor is None:
        return
    _apply_scale(ctx, factor)
    ctx.refresh()
