from __future__ import annotations

import copy
import inspect
import logging
from types import ModuleType
from typing import Any, Callable

_log = logging.getLogger(__name__)

from PySide6 import QtCore

import scripts.double_half_time as double_half_time_script
import scripts.mirror_pitch as mirror_pitch_script
import scripts.reverse as reverse_script
import scripts.switch_hands as switch_hands_script
from file_model.SCORE import SCORE
from scripting.dialog import ScriptDialog
from scripting.spec import DialogSpec
from ui.preview_service import PreviewSession


class ScriptContext:
    def __init__(self, file_manager, editor, parent=None) -> None:
        self._file_manager = file_manager
        self._editor = editor
        self.parent = parent

    @property
    def score(self):
        return self._file_manager.current()

    def score_dict(self) -> dict:
        try:
            return copy.deepcopy(self._file_manager.current().get_dict())
        except Exception:
            return {}

    def replace_score_from_dict(self, data: dict) -> None:
        sc = SCORE.from_dict(copy.deepcopy(data))
        self._file_manager.replace_current(sc)
        self.refresh()

    def refresh(self) -> None:
        try:
            self._editor.force_redraw_from_model()
            self._editor.score_changed.emit()
        except Exception:
            pass


class ScriptEngine:
    def __init__(self, file_manager, editor, parent=None) -> None:
        self._file_manager = file_manager
        self._editor = editor
        self._parent = parent
        self._open_dialogs: list[ScriptDialog] = []
        self._builtins = [
            {
                "id": "double_half_time",
                "tr_context": "DoubleHalfTimeAction",
                "label": "Half Time / Double Time",
                "tooltip": "Scale the full score timing with preview and cancel recovery.",
            },
            {
                "id": "mirror_pitch_selection",
                "tr_context": "MirrorPitchSelectionAction",
                "label": "Mirror Pitch",
                "tooltip": "Mirror note pitches between lowest and highest. Applies to selection, or entire file if no selection.",
            },
            {
                "id": "reverse_selection",
                "tr_context": "ReverseSelectionAction",
                "label": "Reverse",
                "tooltip": "Reverse events in time as if playback were going backwards. Applies to selection, or entire file if no selection.",
            },
            {
                "id": "switch_hands_selection",
                "tr_context": "SwitchHandsSelectionAction",
                "label": "Switch Hands",
                "tooltip": "Switch hand assignment (left ↔ right) for notes and beams. Applies to selection, or entire file if no selection.",
            },
        ]
        self._builtin_modules: dict[str, ModuleType] = {
            "double_half_time": double_half_time_script,
            "mirror_pitch_selection": mirror_pitch_script,
            "reverse_selection": reverse_script,
            "switch_hands_selection": switch_hands_script,
        }
        self._validate_builtin_modules()

    def _validate_builtin_modules(self) -> None:
        for action in self._builtins:
            action_id = str(action.get("id", "") or "")
            module = self._builtin_modules.get(action_id)
            if module is None:
                _log.error("[ScriptEngine] action '%s' has no module in registry", action_id)
                continue
            has_entry = (
                callable(getattr(module, "build_dialog", None))
                or isinstance(getattr(module, "DIALOG_SPEC", None), DialogSpec)
                or callable(getattr(module, "apply", None))
                or callable(getattr(module, "preview", None))
                or callable(getattr(module, "run", None))
                or callable(getattr(module, "main", None))
            )
            if not has_entry:
                _log.error(
                    "[ScriptEngine] action '%s' module '%s' exposes no entry point "
                    "(expected build_dialog, DIALOG_SPEC, apply, preview, run, or main)",
                    action_id,
                    module.__name__,
                )

    def list_actions(self) -> list[dict[str, str]]:
        return [dict(item) for item in self._builtins]

    def run_action(self, action_id: str) -> None:
        action_key = str(action_id or "").strip().lower()
        action = next((a for a in self._builtins if str(a.get("id", "")).lower() == action_key), None)
        if action is None:
            raise ValueError(f"Unknown tool action: {action_id}")
        module = self._builtin_modules.get(action_key)
        if module is None:
            raise ValueError(f"No module configured for action: {action_id}")
        self._run_module(module, action_key)

    def _run_module(self, module, action_key: str) -> None:
        ctx = ScriptContext(self._file_manager, self._editor, parent=self._parent)
        dialog_factory = getattr(module, "build_dialog", None)
        dialog_spec = getattr(module, "DIALOG_SPEC", None)
        apply_fn = getattr(module, "apply", None)
        preview_fn = getattr(module, "preview", None)
        run_fn = getattr(module, "run", None) or getattr(module, "main", None)
        label = f"tool_action:{action_key}"

        if callable(dialog_factory) or isinstance(dialog_spec, DialogSpec):
            spec = dialog_spec if isinstance(dialog_spec, DialogSpec) else dialog_factory(ctx)
            if not isinstance(spec, DialogSpec):
                raise ValueError("Dialog spec must be a DialogSpec instance")
            self._run_with_dialog(spec, ctx, apply_fn, preview_fn, label)
            return

        if callable(run_fn):
            session = PreviewSession(self._file_manager, self._editor, parent=self._parent, debounce_ms=0)
            session.commit(label=label, mutator=lambda: self._invoke(run_fn, ctx, None), restore_first=True)
            return

        if callable(apply_fn):
            session = PreviewSession(self._file_manager, self._editor, parent=self._parent, debounce_ms=0)
            session.commit(label=label, mutator=lambda: self._invoke(apply_fn, ctx, None), restore_first=True)
            return

        raise ValueError("Tool action module must define build_dialog(), apply(), preview(), run(), or main().")

    def _run_with_dialog(
        self,
        spec: DialogSpec,
        ctx: ScriptContext,
        apply_fn: Callable | None,
        preview_fn: Callable | None,
        label: str,
    ) -> None:
        session = PreviewSession(self._file_manager, self._editor, parent=self._parent, debounce_ms=0)

        def _preview(values: dict[str, Any]) -> None:
            if callable(preview_fn):
                session.preview(mutator=lambda: self._invoke(preview_fn, ctx, values), restore_first=True)

        def _apply(values: dict[str, Any]) -> None:
            if callable(apply_fn):
                session.commit(
                    label=label,
                    mutator=lambda: self._invoke(apply_fn, ctx, values),
                    restore_first=True,
                )
            elif callable(preview_fn):
                session.commit(
                    label=label,
                    mutator=lambda: self._invoke(preview_fn, ctx, values),
                    restore_first=True,
                )

        def _cancel() -> None:
            session.restore_original()

        dlg = ScriptDialog(spec=spec, on_preview=_preview, on_apply=_apply, on_cancel=_cancel, parent=self._parent)
        dlg.setModal(False)
        dlg.setWindowModality(QtCore.Qt.WindowModality.NonModal)
        dlg.setAttribute(QtCore.Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self._open_dialogs.append(dlg)

        def _cleanup_open_dialog(_result: int, d=dlg) -> None:
            try:
                if d in self._open_dialogs:
                    self._open_dialogs.remove(d)
            except Exception:
                pass

        dlg.finished.connect(_cleanup_open_dialog)
        dlg.show()
        dlg.raise_()
        dlg.activateWindow()

    def _invoke(self, fn: Callable, ctx: ScriptContext, values: dict | None) -> None:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
        if len(params) == 0:
            fn()
            return
        if len(params) == 1:
            fn(ctx)
            return
        fn(ctx, values)

ToolActionEngine = ScriptEngine
ToolActionContext = ScriptContext

__all__ = ["ScriptEngine", "ScriptContext", "ToolActionEngine", "ToolActionContext"]