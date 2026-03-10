from __future__ import annotations

import copy
import importlib.util
import inspect
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from PySide6 import QtWidgets

from appdata_manager import get_appdata_manager
from file_model.SCORE import SCORE
from scripting.dialog import ScriptDialog
from scripting.spec import DialogSpec


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
        self._last_dir = Path.home()
        try:
            adm = get_appdata_manager()
            last = str(adm.get("last_script_dir", "") or "")
            if last:
                self._last_dir = Path(last)
        except Exception:
            pass

    def choose_and_run(self) -> None:
        start = str(self._last_dir)
        fname, _ = QtWidgets.QFileDialog.getOpenFileName(
            self._parent,
            "Run Script",
            start,
            "Python Scripts (*.py);;All Files (*)",
        )
        if not fname:
            return
        path = Path(fname)
        self._last_dir = path.parent
        try:
            adm = get_appdata_manager()
            adm.set("last_script_dir", str(self._last_dir))
            adm.save()
        except Exception:
            pass
        self.run_script(path)

    def run_script(self, path: Path) -> None:
        try:
            if path is None:
                raise ValueError("No script path provided")
            module = self._load_module(Path(path))
            self._run_module(module, path)
        except Exception as exc:
            self._show_error("Script error", exc)

    def _run_module(self, module, path: Path) -> None:
        snapshot = copy.deepcopy(self._file_manager.current().get_dict())
        try:
            dirty_before = bool(self._file_manager.is_dirty())
        except Exception:
            dirty_before = True
        ctx = ScriptContext(self._file_manager, self._editor, parent=self._parent)
        dialog_factory = getattr(module, "build_dialog", None)
        dialog_spec = getattr(module, "DIALOG_SPEC", None)
        apply_fn = getattr(module, "apply", None)
        preview_fn = getattr(module, "preview", None)
        run_fn = getattr(module, "run", None) or getattr(module, "main", None)
        label = f"script:{path.stem}" if path else "script:custom"

        if callable(dialog_factory) or isinstance(dialog_spec, DialogSpec):
            spec = dialog_spec if isinstance(dialog_spec, DialogSpec) else dialog_factory(ctx)
            if not isinstance(spec, DialogSpec):
                raise ValueError("Dialog spec must be a DialogSpec instance")
            self._run_with_dialog(spec, ctx, snapshot, dirty_before, apply_fn, preview_fn, label)
            return

        if callable(run_fn):
            self._restore_snapshot(snapshot)
            self._invoke(run_fn, ctx, None)
            self._finalize_apply(label)
            return

        if callable(apply_fn):
            self._restore_snapshot(snapshot)
            self._invoke(apply_fn, ctx, None)
            self._finalize_apply(label)
            return

        raise ValueError("Script must define build_dialog(), apply(), preview(), run(), or main().")

    def _run_with_dialog(
        self,
        spec: DialogSpec,
        ctx: ScriptContext,
        snapshot: dict,
        dirty_before: bool,
        apply_fn: Callable | None,
        preview_fn: Callable | None,
        label: str,
    ) -> None:
        def _preview(values: dict[str, Any]) -> None:
            self._restore_snapshot(snapshot, dirty_state=dirty_before)
            if callable(preview_fn):
                self._invoke(preview_fn, ctx, values)
            ctx.refresh()

        def _apply(values: dict[str, Any]) -> None:
            self._restore_snapshot(snapshot)
            if callable(apply_fn):
                self._invoke(apply_fn, ctx, values)
            elif callable(preview_fn):
                self._invoke(preview_fn, ctx, values)
            self._finalize_apply(label)

        def _cancel() -> None:
            self._restore_snapshot(snapshot, dirty_state=dirty_before)
            ctx.refresh()

        dlg = ScriptDialog(spec=spec, on_preview=_preview, on_apply=_apply, on_cancel=_cancel, parent=self._parent)
        dlg.exec()

    def _restore_snapshot(self, snapshot: dict, dirty_state: bool | None = None) -> None:
        try:
            sc = SCORE.from_dict(copy.deepcopy(snapshot))
            self._file_manager.replace_current(sc)
            if dirty_state is not None:
                if dirty_state:
                    self._file_manager.mark_dirty()
                else:
                    self._file_manager.clear_dirty()
        except Exception:
            pass

    def _finalize_apply(self, label: str) -> None:
        try:
            self._editor._snapshot_if_changed(coalesce=False, label=label)
        except Exception:
            pass
        try:
            self._editor.force_redraw_from_model()
            self._editor.score_changed.emit()
        except Exception:
            pass

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

    def _load_module(self, path: Path):
        if not path.exists():
            raise FileNotFoundError(f"Script not found: {path}")
        name = f"user_script_{path.stem}_{int(time.time() * 1000)}"
        spec = importlib.util.spec_from_file_location(name, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load script: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _show_error(self, title: str, exc: Exception) -> None:
        msg = QtWidgets.QMessageBox(self._parent)
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.setWindowTitle(title)
        detail = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        msg.setText(str(exc))
        msg.setDetailedText(detail)
        msg.exec()


__all__ = ["ScriptEngine", "ScriptContext"]