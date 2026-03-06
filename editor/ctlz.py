from __future__ import annotations
import copy
from typing import Optional

class CtlZ:
    """
    Dict-based undo/redo manager modeled after provided design.

    - Stores deep-copied SCORE dict snapshots
    - On add_ctlz(), truncates forward history if not at tail
    - undo()/redo() return a SCORE instance reconstructed from dict
    """
    def __init__(self, file_manager, max_ctlz_num: int = 200) -> None:
        self._fm = file_manager
        self.buffer: list[dict] = []
        self.index: int = -1
        self.max_ctlz_num: int = max(10, int(max_ctlz_num))

    def _current_dict(self) -> dict:
        score = self._fm.current()
        try:
            return score.get_dict()
        except Exception:
            return {}

    def reset_ctlz(self) -> None:
        d = copy.deepcopy(self._current_dict())
        self.buffer = [d]
        self.index = 0

    def add_ctlz(self) -> bool:
        cur = self._current_dict()
        if self.index >= 0 and self.index < len(self.buffer):
            if cur == self.buffer[self.index]:
                # no change, do nothing
                return False
        # if we are in the past (undo/redo), drop future branch
        if self.index != (len(self.buffer) - 1):
            self.buffer = self.buffer[: self.index + 1]
        # append new snapshot
        self.buffer.append(copy.deepcopy(cur))
        # enforce limit
        if len(self.buffer) > self.max_ctlz_num:
            self.buffer.pop(0)
        self.index = len(self.buffer) - 1
        return True

    def undo(self):
        if not self.buffer:
            return None
        self.index -= 1
        if self.index < 0:
            self.index = 0
        d = self.buffer[self.index]
        # reconstruct SCORE from dict
        from file_model.SCORE import SCORE
        try:
            score = SCORE.from_dict(d)
        except Exception:
            score = self._fm.current()
        return score

    def redo(self):
        if not self.buffer:
            return None
        self.index += 1
        if self.index > len(self.buffer) - 1:
            self.index = len(self.buffer) - 1
        d = self.buffer[self.index]
        from file_model.SCORE import SCORE
        try:
            score = SCORE.from_dict(d)
        except Exception:
            score = self._fm.current()
        return score
