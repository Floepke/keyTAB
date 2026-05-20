from __future__ import annotations

from pathlib import Path
from typing import Iterable

DOCUMENT_EXTENSIONS = {".keytab", ".piano", ".mid", ".midi", ".musicxml", ".mxl", ".xml"}


def is_supported_document(path: str | Path) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in DOCUMENT_EXTENSIONS


def extract_document_paths(args: Iterable[str]) -> list[str]:
    documents: list[str] = []
    for raw in args:
        candidate = str(raw or "").strip()
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        suffix = path.suffix.lower()
        if suffix in DOCUMENT_EXTENSIONS:
            documents.append(str(path))
            continue
        if path.exists() and is_supported_document(path):
            documents.append(str(path))
    return documents
