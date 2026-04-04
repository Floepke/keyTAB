#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass


PLACEHOLDER_RE = re.compile(r"\{[^{}]+\}|%\d+|&[a-zA-Z]+;|&[#a-zA-Z0-9]+;|\\n")


@dataclass
class MaskedText:
    text: str
    tokens: dict[str, str]


def mask_placeholders(text: str) -> MaskedText:
    tokens: dict[str, str] = {}

    def repl(match: re.Match[str]) -> str:
        key = f"__PH_{len(tokens)}__"
        tokens[key] = match.group(0)
        return key

    return MaskedText(text=PLACEHOLDER_RE.sub(repl, text), tokens=tokens)


def unmask_placeholders(text: str, tokens: dict[str, str]) -> str:
    out = text
    for key, value in tokens.items():
        out = out.replace(key, value)
    return out


def translate_text(
    source_text: str,
    *,
    source_lang: str,
    target_lang: str,
    url: str,
    api_key: str,
    timeout_s: float,
) -> str:
    payload = {
        "q": source_text,
        "source": source_lang,
        "target": target_lang,
        "format": "text",
    }
    if api_key:
        payload["api_key"] = api_key

    data = urllib.parse.urlencode(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Connection error: {exc}") from exc

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response: {raw[:200]}") from exc

    translated = obj.get("translatedText")
    if not isinstance(translated, str):
        raise RuntimeError(f"Unexpected response: {obj}")
    return translated


def is_unfinished(message: ET.Element) -> bool:
    tr = message.find("translation")
    if tr is None:
        return True
    tr_type = (tr.get("type") or "").strip().lower()
    if tr_type == "unfinished":
        return True
    return (tr.text or "").strip() == ""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Auto-translate Qt .ts files using a LibreTranslate-compatible API",
    )
    parser.add_argument("ts_file", help="Path to Qt .ts file")
    parser.add_argument("--source", default="en", help="Source language code (default: en)")
    parser.add_argument("--target", default="nl", help="Target language code (default: nl)")
    parser.add_argument(
        "--url",
        default="https://libretranslate.com/translate",
        help="Translate endpoint URL (default: https://libretranslate.com/translate)",
    )
    parser.add_argument("--api-key", default="", help="Optional API key for your translation service")
    parser.add_argument("--timeout", type=float, default=20.0, help="HTTP timeout in seconds")
    parser.add_argument(
        "--only-context",
        action="append",
        default=[],
        help="Only translate these context names (repeatable)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing non-empty translations as well",
    )
    args = parser.parse_args()

    tree = ET.parse(args.ts_file)
    root = tree.getroot()

    allowed_contexts = set(args.only_context or [])
    translated_count = 0
    skipped_count = 0
    failed_count = 0

    for context in root.findall("context"):
        name_el = context.find("name")
        context_name = (name_el.text or "").strip() if name_el is not None else ""
        if allowed_contexts and context_name not in allowed_contexts:
            continue

        for message in context.findall("message"):
            source_el = message.find("source")
            tr_el = message.find("translation")

            if source_el is None:
                continue
            source = (source_el.text or "").strip()
            if not source:
                continue

            if tr_el is None:
                tr_el = ET.SubElement(message, "translation")

            current = (tr_el.text or "").strip()
            unfinished = is_unfinished(message)
            if (not args.overwrite) and (not unfinished) and current:
                skipped_count += 1
                continue

            masked = mask_placeholders(source)
            try:
                translated = translate_text(
                    masked.text,
                    source_lang=args.source,
                    target_lang=args.target,
                    url=args.url,
                    api_key=args.api_key,
                    timeout_s=args.timeout,
                )
                translated = unmask_placeholders(translated, masked.tokens)
            except Exception as exc:
                failed_count += 1
                print(f"[warn] {context_name}: '{source}' -> {exc}", file=sys.stderr)
                continue

            tr_el.text = translated
            if "type" in tr_el.attrib:
                del tr_el.attrib["type"]
            translated_count += 1

    tree.write(args.ts_file, encoding="utf-8", xml_declaration=True)
    print(
        f"Updated {args.ts_file}: translated={translated_count}, skipped={skipped_count}, failed={failed_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
