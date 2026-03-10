from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Sequence


@dataclass
class Field:
    name: str
    label: str
    default: Any = None


@dataclass
class BoolField(Field):
    default: bool = False


@dataclass
class IntField(Field):
    minimum: int = 0
    maximum: int = 100
    step: int = 1
    default: int = 0


@dataclass
class FloatField(Field):
    minimum: float = 0.0
    maximum: float = 1.0
    step: float = 0.1
    decimals: int = 3
    default: float = 0.0


@dataclass
class StringField(Field):
    placeholder: str = ""
    default: str = ""


@dataclass
class ArrayField(Field):
    element_type: Literal["int", "float", "string"] = "int"
    separator: str = ","
    default: str = ""


@dataclass
class LabelField(Field):
    name: str = ""
    label: str = ""
    text: str = ""


@dataclass
class DialogSpec:
    title: str
    fields: Sequence[Field]


__all__ = [
    "Field",
    "BoolField",
    "IntField",
    "FloatField",
    "StringField",
    "ArrayField",
    "LabelField",
    "DialogSpec",
]