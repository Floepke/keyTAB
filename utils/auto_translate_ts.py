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
DEFAULT_ENDPOINTS = [
    "https://translate.argosopentech.com/translate",
    "https://libretranslate.com/translate",
]
DEFAULT_MYMEMORY_URL = "https://api.mymemory.translated.net/get"


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


def _parse_translated_text(raw: str) -> str:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response: {raw[:200]}") from exc

    translated = obj.get("translatedText")
    if isinstance(translated, str):
        return translated

    # Google-style wrapper used by some hosted services.
    data_obj = obj.get("data")
    if isinstance(data_obj, dict):
        translations = data_obj.get("translations")
        if isinstance(translations, list) and translations:
            first = translations[0]
            if isinstance(first, dict):
                candidate = first.get("translatedText")
                if isinstance(candidate, str):
                    return candidate

    raise RuntimeError(f"Unexpected response: {obj}")


def _request_translate(
    url: str,
    payload: dict[str, str],
    *,
    timeout_s: float,
    use_json: bool,
    bearer_key: str,
) -> str:
    if use_json:
        data = json.dumps(payload).encode("utf-8")
        content_type = "application/json"
    else:
        data = urllib.parse.urlencode(payload).encode("utf-8")
        content_type = "application/x-www-form-urlencoded"

    headers = {"Content-Type": content_type}
    if bearer_key:
        headers["Authorization"] = f"Bearer {bearer_key}"

    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return _parse_translated_text(raw)


def _request_mymemory(
    source_text: str,
    *,
    source_lang: str,
    target_lang: str,
    mymemory_url: str,
    timeout_s: float,
) -> str:
    query = urllib.parse.urlencode(
        {
            "q": source_text,
            "langpair": f"{source_lang}|{target_lang}",
        }
    )
    url = f"{mymemory_url}?{query}"
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read().decode("utf-8", errors="replace")

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response: {raw[:200]}") from exc

    response_data = obj.get("responseData")
    if isinstance(response_data, dict):
        translated = response_data.get("translatedText")
        if isinstance(translated, str):
            return translated

    raise RuntimeError(f"Unexpected response: {obj}")


def translate_text(
    source_text: str,
    *,
    source_lang: str,
    target_lang: str,
    urls: list[str],
    api_key: str,
    timeout_s: float,
    fallback_mymemory_url: str,
) -> str:
    payload = {
        "q": source_text,
        "source": source_lang,
        "target": target_lang,
        "format": "text",
    }
    if api_key:
        payload["api_key"] = api_key

    attempts: list[str] = []
    for url in urls:
        for use_json in (False, True):
            try:
                return _request_translate(
                    url,
                    payload,
                    timeout_s=timeout_s,
                    use_json=use_json,
                    bearer_key=api_key,
                )
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                mode = "json" if use_json else "form"
                attempts.append(f"{url} [{mode}] HTTP {exc.code}: {body[:160]}")
            except urllib.error.URLError as exc:
                mode = "json" if use_json else "form"
                attempts.append(f"{url} [{mode}] Connection error: {exc}")
            except Exception as exc:
                mode = "json" if use_json else "form"
                attempts.append(f"{url} [{mode}] {exc}")

    try:
        return _request_mymemory(
            source_text,
            source_lang=source_lang,
            target_lang=target_lang,
            mymemory_url=fallback_mymemory_url,
            timeout_s=timeout_s,
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        attempts.append(f"{fallback_mymemory_url} [mymemory] HTTP {exc.code}: {body[:160]}")
    except urllib.error.URLError as exc:
        attempts.append(f"{fallback_mymemory_url} [mymemory] Connection error: {exc}")
    except Exception as exc:
        attempts.append(f"{fallback_mymemory_url} [mymemory] {exc}")

    raise RuntimeError("All translation endpoints failed: " + " | ".join(attempts))


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
        action="append",
        default=[],
        help="Translate endpoint URL (repeatable). If omitted, built-in fallback endpoints are used.",
    )
    parser.add_argument("--api-key", default="", help="Optional API key for your translation service")
    parser.add_argument(
        "--mymemory-url",
        default=DEFAULT_MYMEMORY_URL,
        help="MyMemory fallback endpoint URL (default: https://api.mymemory.translated.net/get)",
    )
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
    urls = args.url or DEFAULT_ENDPOINTS

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
                    urls=urls,
                    api_key=args.api_key,
                    timeout_s=args.timeout,
                    fallback_mymemory_url=args.mymemory_url,
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
