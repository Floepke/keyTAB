from __future__ import annotations

import copy
from typing import Callable

from PySide6 import QtCore

from file_model.SCORE import SCORE


class PreviewSession:
    """Shared preview transaction for dialogs and scripts.

    - Captures a baseline snapshot of the current score and dirty state.
    - Can preview from baseline (restore -> mutate -> refresh).
    - Can preview current-state changes with refresh only.
    - Can commit a final state and create an undo snapshot label.
    - Can restore baseline on cancel.
    """

    def __init__(self, file_manager, editor, parent=None, debounce_ms: int = 150) -> None:
        self._file_manager = file_manager
        self._editor = editor
        self._snapshot = copy.deepcopy(self._file_manager.current().get_dict())
        try:
            self._dirty_before = bool(self._file_manager.is_dirty())
        except Exception:
            self._dirty_before = True

        self._timer = QtCore.QTimer(parent)
        self._timer.setSingleShot(True)
        self._timer.setInterval(max(0, int(debounce_ms)))
        self._pending_mutator: Callable[[], None] | None = None
        self._pending_restore_first: bool = True
        self._timer.timeout.connect(self._on_timer)

    def _on_timer(self) -> None:
        self.preview(mutator=self._pending_mutator, restore_first=self._pending_restore_first)

    def _restore_snapshot(self, dirty_state: bool | None = None) -> None:
        sc = SCORE.from_dict(copy.deepcopy(self._snapshot))
        self._file_manager.replace_current(sc)
        # Bust the editor's note-time render cache so stale Python object
        # references from the replaced SCORE instance are not reused.  The
        # cache key only hashes time/duration/pitch/_id, so fields like 'hand'
        # that changed during preview would survive a snapshot restore undetected.
        # Also clear _draw_cache: detect_events_from_time_window reads
        # cache['notes_view'] which holds references to the old score's note
        # objects; if not cleared, selection-based mutations hit those stale
        # objects instead of the freshly restored score's notes.
        try:
            self._editor._note_time_cache_key = None
            self._editor._note_time_cache_values = None
            self._editor._draw_cache = None
        except Exception:
            pass
        if dirty_state is not None:
            if dirty_state:
                self._file_manager.mark_dirty()
            else:
                self._file_manager.clear_dirty()

    def refresh(self) -> None:
        try:
            self._editor.force_redraw_from_model()
            self._editor.score_changed.emit()
        except Exception:
            pass

    def schedule_preview(self, mutator: Callable[[], None] | None = None, *, restore_first: bool = True) -> None:
        self._pending_mutator = mutator
        self._pending_restore_first = bool(restore_first)
        self._timer.stop()
        self._timer.start()

    def schedule_refresh(self) -> None:
        self.schedule_preview(mutator=None, restore_first=False)

    def preview(self, mutator: Callable[[], None] | None = None, *, restore_first: bool = True) -> None:
        if restore_first:
            self._restore_snapshot(dirty_state=self._dirty_before)
        if callable(mutator):
            mutator()
        self.refresh()

    def commit(
        self,
        *,
        label: str | None = None,
        mutator: Callable[[], None] | None = None,
        restore_first: bool = False,
    ) -> None:
        self._timer.stop()
        if restore_first:
            self._restore_snapshot(dirty_state=self._dirty_before)
        if callable(mutator):
            mutator()
        if label:
            try:
                self._editor._snapshot_if_changed(coalesce=False, label=str(label))
            except Exception:
                pass
        self.refresh()

    def restore_original(self) -> None:
        self._timer.stop()
        self._restore_snapshot(dirty_state=self._dirty_before)
        self.refresh()
