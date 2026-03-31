from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from file_model.events.note import Note
from symbol_design.noteheads.circle import draw_circle_left_dot, draw_circle_notehead
from symbol_design.noteheads.triangle import draw_triangle_left_dot, draw_triangle_notehead
from symbol_design.symbol_util import SymbolUtil
from ui.widgets.draw_util import DrawUtil
from utils.CONSTANT import BLACK_KEYS

NoteheadForm = Literal["circle", "triangle", "bullet", "x"]
NoteheadDirection = Literal["up", "down"]
NoteheadLiteral = Literal[
    "auto",
    "circle_white_up",
    "circle_white_down",
    "circle_black_up",
    "circle_black_down",
    "bullet_white_up",
    "bullet_white_down",
    "bullet_black_up",
    "bullet_black_down",
    "triangle_white_up",
    "triangle_white_down",
    "triangle_black_up",
    "triangle_black_down",
    "cross_up",
    "cross_down",
]


@dataclass(frozen=True)
class NoteheadSpec:
    literal: NoteheadLiteral
    form: NoteheadForm
    direction: NoteheadDirection
    filled: bool

    @property
    def is_up(self) -> bool:
        return self.direction == "up"


_ALLOWED: set[str] = {
    "auto",
    "circle_white_up",
    "circle_white_down",
    "circle_black_up",
    "circle_black_down",
    "bullet_white_up",
    "bullet_white_down",
    "bullet_black_up",
    "bullet_black_down",
    "triangle_white_up",
    "triangle_white_down",
    "triangle_black_up",
    "triangle_black_down",
    "cross_up",
    "cross_down",
}


def normalize_notehead_literal(value: object) -> NoteheadLiteral:
    txt = str(value or "auto").strip()
    if txt not in _ALLOWED:
        return "auto"
    return txt  # type: ignore[return-value]


def resolve_notehead_spec(note: Note | dict, default_black_above: bool) -> NoteheadSpec:
    raw = normalize_notehead_literal(_read_note_field(note, "notehead", "auto"))
    pitch = int(_read_note_field(note, "pitch", 0) or 0)

    if raw == "auto":
        if pitch in BLACK_KEYS:
            return NoteheadSpec(
                literal="circle_black_up" if bool(default_black_above) else "circle_black_down",
                form="circle",
                direction="up" if bool(default_black_above) else "down",
                filled=True,
            )
        return NoteheadSpec(
            literal="circle_white_down",
            form="circle",
            direction="down",
            filled=False,
        )

    if raw.startswith("circle_"):
        is_black = "_black_" in raw
        return NoteheadSpec(
            literal=raw,
            form="circle",
            direction="up" if raw.endswith("_up") else "down",
            filled=is_black,
        )

    if raw.startswith("bullet_"):
        is_black = "_black_" in raw
        return NoteheadSpec(
            literal=raw,
            form="bullet",
            direction="up" if raw.endswith("_up") else "down",
            filled=is_black,
        )

    if raw.startswith("triangle_"):
        is_black = "_black_" in raw
        return NoteheadSpec(
            literal=raw,
            form="triangle",
            direction="up" if raw.endswith("_up") else "down",
            filled=is_black,
        )

    if raw.startswith("cross_"):
        return NoteheadSpec(
            literal=raw,
            form="x",
            direction="up" if raw.endswith("_up") else "down",
            filled=False,
        )

    return NoteheadSpec(
        literal="circle_white_down",
        form="circle",
        direction="down",
        filled=False,
    )


def _read_note_field(note: Note | dict, name: str, default):
    if isinstance(note, dict):
        return note.get(name, default)
    return getattr(note, name, default)


@dataclass
class Notehead(SymbolUtil):
    x_mm: float
    y_mm: float
    note: Note | dict
    form: NoteheadForm
    direction: NoteheadDirection
    filled: bool = False
    outline_width_mm_override: float | None = None

    @property
    def pitch(self) -> int:
        return int(_read_note_field(self.note, "pitch", 0) or 0)

    @property
    def hand(self) -> str:
        return str(_read_note_field(self.note, "hand", "") or "")

    def draw_notehead(self, du: DrawUtil, item_id: int = 0, tags: list[str] | None = None, use_custom_color: bool = False) -> None:
        draw_tags = list(tags or [])
        
        # Check if this notehead has a custom override (not auto) and custom coloring is enabled
        notehead_literal = str(_read_note_field(self.note, "notehead", "auto") or "auto").strip()
        is_custom = notehead_literal != "auto" and bool(use_custom_color)
        
        # If custom, use the custom color (accent_color2: (128, 0, 0) → (0.5, 0, 0, 1) normalized)
        stroke_override = (128 / 255.0, 0.0, 0.0, 1.0) if is_custom else None
        # For filled custom noteheads, also override the fill color
        fill_override = (128 / 255.0, 0.0, 0.0, 1.0) if (is_custom and bool(self.filled)) else None
        
        if self.form == "circle":
            draw_circle_notehead(
                du,
                self,
                x_mm=float(self.x_mm),
                y_mm=float(self.y_mm),
                direction=self.direction,
                filled=bool(self.filled),
                item_id=int(item_id),
                tags=draw_tags,
                stroke_color_override=stroke_override,
                fill_color_override=fill_override,
            )
            draw_circle_left_dot(du, self)
            return

        if self.form == "triangle":
            draw_triangle_notehead(
                du,
                self,
                x_mm=float(self.x_mm),
                y_mm=float(self.y_mm),
                direction=self.direction,
                filled=bool(self.filled),
                item_id=int(item_id),
                tags=draw_tags,
                stroke_color_override=stroke_override,
                fill_color_override=fill_override,
            )
            draw_triangle_left_dot(du, self)
            return

        draw_circle_notehead(
            du,
            self,
            x_mm=float(self.x_mm),
            y_mm=float(self.y_mm),
            direction=self.direction,
            filled=bool(self.filled),
            item_id=int(item_id),
            tags=draw_tags,
            stroke_color_override=stroke_override,
            fill_color_override=fill_override,
        )
        draw_circle_left_dot(du, self)

    @classmethod
    def from_note(
        cls,
        *,
        x_mm: float,
        y_mm: float,
        note: Note | dict,
        layout,
        semitone_space_mm: float,
        notation_color: tuple[float, float, float, float],
        paper_color: tuple[float, float, float, float],
        default_black_above: bool,
        outline_width_mm_override: float | None = None,
    ) -> "Notehead":
        spec = resolve_notehead_spec(note, default_black_above=bool(default_black_above))
        return cls(
            layout=layout,
            semitone_space_mm=float(semitone_space_mm),
            notation_color=notation_color,
            paper_color=paper_color,
            x_mm=float(x_mm),
            y_mm=float(y_mm),
            note=note,
            form=spec.form,
            direction=spec.direction,
            filled=spec.filled,
            outline_width_mm_override=(None if outline_width_mm_override is None else float(outline_width_mm_override)),
        )