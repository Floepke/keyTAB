from __future__ import annotations

import copy
import importlib.util
import inspect
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from PySide6 import QtCore, QtWidgets

from appdata_manager import get_appdata_manager
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

        raise ValueError("Script must define build_dialog(), apply(), preview(), run(), or main().")

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