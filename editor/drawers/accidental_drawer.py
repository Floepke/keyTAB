from __future__ import annotations

from typing import TYPE_CHECKING, cast

from file_model.events.note import Note
from symbol_design.noteheads import resolve_notehead_spec

if TYPE_CHECKING:
    from editor.editor import Editor
    from ui.widgets.draw_util import DrawUtil


class AccidentalDrawerMixin:
    """Draw accidental guide lines for notes with a valid non-zero `acc` value."""

    def _draw_note_accidental(self, du: "DrawUtil", note: Note, x: float, y_start: float) -> None:
        self = cast("Editor", self)
        acc = int(getattr(note, 'acc', 0) or 0)
        
        if acc == 0:
            return
        
        if not Note.is_valid_accidental(note):
            return

        score = self.current_score()
        if score is None:
            return
        layout = score.layout

        default_black_above = self._black_note_above_stem(note, layout)
        spec = resolve_notehead_spec(note, default_black_above=default_black_above)
        is_above_stem = bool(getattr(spec, 'is_up', False))

        semitone = float(getattr(self, 'semitone_dist', 0.5) or 0.5)
        note_h = 2.0 * semitone

        # Anchor from top (above-stem heads) or bottom (under-stem heads).
        y_anchor = float(y_start - note_h) if is_above_stem else float(y_start + note_h)
        derived_pitch = int(getattr(note, 'pitch', 0) or 0) + int(acc)
        x_target = float(self.pitch_to_x(derived_pitch))

        # Short diagonal hint: upward for above-stem heads, downward for under-stem heads.
        y_target = float(y_anchor - semitone) if is_above_stem else float(y_anchor + semitone)

        line_w = max(0.01, float(getattr(self, 'editor_line_width_global', 0.1) or 0.1))

        du.add_line(
            float(x),
            float(y_anchor),
            float(x_target),
            float(y_target),
            color=self.notation_color,
            width_mm=line_w,
            id=0,
            tags=['accidental_line'],
        )
