from __future__ import annotations

from dataclasses import fields, is_dataclass
from PySide6 import QtCore

from scripting import ActionButtonField, DialogSpec, LabelField


ACTION_HALF = "half"
ACTION_DOUBLE = "double"
_CTX_MULTIPLIER_ATTR = "_double_half_time_multiplier"


def build_dialog(ctx):
    # Keep per-dialog cumulative scale state on the shared context.
    setattr(ctx, _CTX_MULTIPLIER_ATTR, 1.0)
    return DialogSpec(
        title=QtCore.QCoreApplication.translate("DoubleHalfTimeAction", "Half Time / Double Time"),
        fields=[
            LabelField(
                text=QtCore.QCoreApplication.translate(
                    "DoubleHalfTimeAction", "Preview by clicking below. You can click multiple times to stack changes."
                ),
            ),
            ActionButtonField(
                name=ACTION_HALF,
                label=QtCore.QCoreApplication.translate("DoubleHalfTimeAction", "half time (÷2)"),
            ),
            ActionButtonField(
                name=ACTION_DOUBLE,
                label=QtCore.QCoreApplication.translate("DoubleHalfTimeAction", "double time (×2)"),
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


def _get_cumulative_multiplier(ctx) -> float:
    try:
        value = float(getattr(ctx, _CTX_MULTIPLIER_ATTR, 1.0) or 1.0)
    except Exception:
        value = 1.0
    if value <= 0.0:
        return 1.0
    return value


def _set_cumulative_multiplier(ctx, value: float) -> None:
    setattr(ctx, _CTX_MULTIPLIER_ATTR, float(value))


def _update_cumulative_multiplier(ctx, values: dict | None) -> float | None:
    step_factor = _factor_from_values(values)
    if step_factor is None:
        return None
    cumulative = _get_cumulative_multiplier(ctx) * float(step_factor)
    _set_cumulative_multiplier(ctx, cumulative)
    return cumulative


def preview(ctx, values):
    cumulative_factor = _update_cumulative_multiplier(ctx, values)
    if cumulative_factor is None:
        return
    _apply_scale(ctx, cumulative_factor)
    ctx.refresh()


def apply(ctx, values):
    cumulative_factor = _get_cumulative_multiplier(ctx)
    if abs(float(cumulative_factor) - 1.0) <= 1e-12:
        return
    _apply_scale(ctx, cumulative_factor)
    ctx.refresh()
