from __future__ import annotations
from typing import TYPE_CHECKING, cast

from ui.widgets.draw_util import DrawUtil

if TYPE_CHECKING:
    from editor.editor import Editor


class DoubleBarDrawerMixin:
    def draw_double_bar(self, du: DrawUtil) -> None:
        # Double barlines are now rendered as constructive stave barlines in GridDrawerMixin.
        # Keep this mixin as a compatibility no-op for existing Editor composition.
        _ = du
        _ = cast("Editor", self)
        return
