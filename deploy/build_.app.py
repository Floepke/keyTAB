#!/usr/bin/env python3
"""Build a macOS .app bundle (plus DMG) for keyTAB using PyInstaller.

When executed directly, the script performs these steps in order:
1) Parse CLI arguments for entry point, app name, output directory, and icon path.
    - Default output directory is ~/Desktop when no args are provided.
2) Create a dedicated build workspace at <output>/keyTAB_build/.
3) Scan the project for imported PySide6 modules and exclude unused ones from the build.
4) Convert the provided icon to ICNS (via sips/iconutil) if needed.
5) Run PyInstaller in that workspace to produce a windowed .app bundle.
6) Copy Qt/PySide6 license files into the bundle for compliance.
7) Ensure a document icon exists in Resources and update Info.plist with bundle IDs,
    category, author, and UTI/file-association declarations for .piano and MIDI files.
8) Optionally build a drag-and-drop DMG containing the .app and an Applications symlink,
    applying a volume icon when possible.
9) Move the resulting .app (and DMG if built) to <output> and remove the build workspace.
    If the build fails, leave <output>/keyTAB_build/ intact for inspection.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENTRY = PROJECT_ROOT / "keyTAB.py"
DEFAULT_ICON = PROJECT_ROOT / "icons" / "keyTAB.png"

PREFERRED_APP_CATEGORY = "public.app-category.music"
PIANO_UTI = "org.philipbergwerf.keytab.piano"
BUNDLE_IDENTIFIER = "org.philipbergwerf.keytab"
AUTHOR_NAME = "Philip Bergwerf"
DOCUMENT_ICON_FILENAME = "keyTABDocument.icns"

EXCLUDABLE_QT_MODULES = [
    "PySide6.QtWebEngine",
    "PySide6.QtWebEngineCore",
    "PySide6.QtWebEngineWidgets",
    "PySide6.QtWebEngineQuick",
    "PySide6.QtWebView",
    "PySide6.QtPdf",
    "PySide6.QtPdfWidgets",
    "PySide6.QtMultimedia",
    "PySide6.QtMultimediaWidgets",
    "PySide6.Qt3DCore",
    "PySide6.Qt3DRender",
    "PySide6.QtQuick",
    "PySide6.QtQuickWidgets",
    "PySide6.QtNetworkAuth",
    "PySide6.QtBluetooth",
]

SKIPPED_SCAN_DIRS = {
    "build",
    "dist",
    "__pycache__",
    ".git",
    "venv",
    ".mypy_cache",
    ".pytest_cache",
}


def copy_qt_licenses(app_path: Path) -> None:
    """Bundle Qt/PySide6 license files into the .app for compliance."""
    try:
        import PySide6  # type: ignore
    except Exception:
        return

    qt_root = Path(getattr(PySide6, "__file__", "")).resolve().parent
    search_roots = [qt_root, qt_root / "Qt", qt_root / "Qt" / "LICENSES"]
    dest_root = app_path / "Contents" / "Resources" / "licenses" / "qt"
    copied_any = False
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.glob("**/LICENSE*"):
            if not path.is_file():
                continue
            rel = path.relative_to(root)
            dest = dest_root / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(path, dest)
                copied_any = True
            except Exception:
                pass
    if copied_any:
        print(f"Bundled Qt/PySide6 license files into {dest_root}")


def _iter_python_files(root: Path) -> Iterator[Path]:
    for path in root.rglob("*.py"):
        if any(part in SKIPPED_SCAN_DIRS for part in path.parts):
            continue
        yield path


def detect_used_qt_modules(project_root: Path) -> set[str]:
    modules: set[str] = set()
    for py_file in _iter_python_files(project_root):
        try:
            source = py_file.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(py_file))
        except (UnicodeDecodeError, SyntaxError):
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.name
                    if name == "PySide6" or name.startswith("PySide6."):
                        modules.add(name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module == "PySide6":
                    modules.add("PySide6")
                    modules.update(f"PySide6.{alias.name}" for alias in node.names)
                elif module.startswith("PySide6."):
                    modules.add(module)
    return modules


def determine_unused_qt_modules(project_root: Path) -> list[str]:
    used_modules = detect_used_qt_modules(project_root)
    return [module for module in EXCLUDABLE_QT_MODULES if module not in used_modules]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a macOS .app bundle with PyInstaller.")
    parser.add_argument(
        "--entry",
        type=Path,
        default=DEFAULT_ENTRY,
        help="Path to the Python entry point (default: keyTAB.py).",
    )
    parser.add_argument(
        "--name",
        default="keyTAB",
        help="Name for the resulting .app bundle (default: keyTAB).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.home() / "Desktop",
        help="Directory where PyInstaller runs and where the final .app is stored.",
    )
    parser.add_argument(
        "--icon",
        type=Path,
        default=DEFAULT_ICON,
        help="Path to an icon image (PNG or ICNS) to embed into the app bundle.",
    )
    return parser.parse_args()


def ensure_command(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise SystemExit(f"Required command '{name}' was not found in PATH.")
    return path


def ensure_pyinstaller_available(python_cmd: str) -> None:
    """Verify PyInstaller is importable in the interpreter used for the build."""
    if importlib.util.find_spec("PyInstaller") is not None:
        return
    raise SystemExit(
        "PyInstaller is not installed in the active Python environment. "
        f"Install it with: {python_cmd} -m pip install pyinstaller"
    )


def prepare_icon(icon_path: Path, work_dir: Path) -> tuple[Path, bool]:
    icon_path = icon_path.resolve()
    if not icon_path.exists():
        raise SystemExit(f"Icon not found: {icon_path}")
    if icon_path.suffix.lower() == ".icns":
        return icon_path, False

    sips = ensure_command("sips")
    iconutil = ensure_command("iconutil")

    iconset_dir = work_dir / "keytab.iconset"
    if iconset_dir.exists():
        shutil.rmtree(iconset_dir)
    iconset_dir.mkdir(parents=True, exist_ok=True)

    sizes = [16, 32, 64, 128, 256, 512]
    for size in sizes:
        for scale in (1, 2):
            px = size * scale
            suffix = "@2x" if scale == 2 else ""
            out_file = iconset_dir / f"icon_{size}x{size}{suffix}.png"
            subprocess.run(
                [sips, "-z", str(px), str(px), str(icon_path), "--out", str(out_file)],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
    icns_path = work_dir / "keytab.icns"
    subprocess.run(
        [iconutil, "-c", "icns", str(iconset_dir), "-o", str(icns_path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    shutil.rmtree(iconset_dir, ignore_errors=True)
    return icns_path, True


def find_bundle_icns(app_path: Path) -> Path | None:
    resources_dir = app_path / "Contents" / "Resources"
    if not resources_dir.is_dir():
        return None
    icns_files = sorted(resources_dir.glob("*.icns"))
    return icns_files[0] if icns_files else None


def apply_dmg_volume_icon(staging_dir: Path, icon_path: Path) -> None:
    volume_icon = staging_dir / ".VolumeIcon.icns"
    shutil.copy(icon_path, volume_icon)
    setfile = shutil.which("SetFile")
    if not setfile:
        print("Warning: 'SetFile' not found; DMG will lack a custom icon.", file=sys.stderr)
        return
    subprocess.run([setfile, "-c", "icnC", str(volume_icon)], check=True)
    subprocess.run([setfile, "-a", "C", str(staging_dir)], check=True)


def ensure_document_icon(app_path: Path, work_dir: Path, icon_hint: Path | None = None) -> str | None:
    resources_dir = app_path / "Contents" / "Resources"
    resources_dir.mkdir(parents=True, exist_ok=True)
    target = resources_dir / DOCUMENT_ICON_FILENAME
    source = find_bundle_icns(app_path)
    temp_icon: Path | None = None
    if source is None and icon_hint is not None:
        try:
            source, generated = prepare_icon(icon_hint, work_dir)
            temp_icon = source if generated else None
        except Exception:
            source = None
            temp_icon = None
    if source is None:
        return None
    try:
        if source.resolve() != target.resolve():
            shutil.copy(source, target)
    except FileNotFoundError:
        return None
    finally:
        if temp_icon is not None and temp_icon.exists():
            temp_icon.unlink()
    return target.name


def run_pyinstaller(
    entry: Path,
    name: str,
    work_dir: Path,
    icon: Path,
    exclude_modules: list[str],
) -> Path:
    entry_path = entry.resolve()
    if not entry_path.exists():
        raise SystemExit(f"Entry file not found: {entry_path}")

    work_dir.mkdir(parents=True, exist_ok=True)
    icon_to_use, generated = prepare_icon(icon, work_dir)

    python_cmd = sys.executable or ensure_command("python3")
    ensure_pyinstaller_available(python_cmd)
    cmd = [
        python_cmd,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--strip",
        "--windowed",
        "--name",
        name,
        "--icon",
        str(icon_to_use),
    ]

    # Ensure MIDI backend (python-rtmidi) is bundled even when imported dynamically.
    cmd.extend([
        "--hidden-import=rtmidi",
        "--hidden-import=rtmidi._rtmidi",
        "--collect-all=rtmidi",
        "--hidden-import=mido.backends.rtmidi",
        "--collect-all=mido",
    ])

    for module in exclude_modules:
        cmd.extend(["--exclude-module", module])

    cmd.append(str(entry_path))

    print(f"Running: {' '.join(cmd)} (cwd={work_dir})")
    try:
        subprocess.run(cmd, cwd=work_dir, check=True)
    finally:
        if generated and icon_to_use.exists():
            icon_to_use.unlink()

    produced = work_dir / "dist" / f"{name}.app"
    if not produced.exists():
        raise SystemExit("PyInstaller finished but the .app bundle was not found.")

    final_target = work_dir / f"{name}.app"
    if final_target.exists():
        shutil.rmtree(final_target, ignore_errors=True)
    shutil.move(str(produced), final_target)
    return final_target


def clean_pyinstaller_artifacts(work_dir: Path, name: str) -> None:
    for folder in (work_dir / "build", work_dir / "dist"):
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
    spec_file = work_dir / f"{name}.spec"
    if spec_file.exists():
        spec_file.unlink()


def update_info_plist(app_path: Path, name: str, doc_icon_file: str | None = None) -> None:
    plist_path = app_path / "Contents" / "Info.plist"
    if not plist_path.exists():
        raise SystemExit(f"Info.plist not found: {plist_path}")

    with plist_path.open("rb") as handle:
        info = plistlib.load(handle)

    doc_icon_stem = Path(doc_icon_file).stem if doc_icon_file else None

    info.setdefault("CFBundleName", name)
    info.setdefault("CFBundleDisplayName", name)
    info.setdefault("CFBundleIdentifier", BUNDLE_IDENTIFIER)
    info["LSApplicationCategoryType"] = PREFERRED_APP_CATEGORY
    info.setdefault("NSHumanReadableCopyright", AUTHOR_NAME)

    info["UTExportedTypeDeclarations"] = [
        {
            "UTTypeIdentifier": PIANO_UTI,
            "UTTypeDescription": "keyTAB score",
            "UTTypeConformsTo": ["public.data"],
            "UTTypeTagSpecification": {
                "public.filename-extension": ["piano"],
                "public.mime-type": ["application/x-keytab"],
            },
        }
    ]

    if doc_icon_stem:
        info["UTExportedTypeDeclarations"][0]["UTTypeIconFiles"] = [doc_icon_stem]

    piano_doc = {
        "CFBundleTypeName": "keyTAB score",
        "CFBundleTypeRole": "Editor",
        "LSItemContentTypes": [PIANO_UTI],
        "CFBundleTypeExtensions": ["piano"],
        "LSHandlerRank": "Owner",
    }
    midi_doc = {
        "CFBundleTypeName": "MIDI audio",
        "CFBundleTypeRole": "Editor",
        "LSItemContentTypes": ["public.midi"],
        "CFBundleTypeExtensions": ["mid", "midi"],
        "LSHandlerRank": "Alternate",
    }
    if doc_icon_stem:
        piano_doc["CFBundleTypeIconFile"] = doc_icon_stem
        midi_doc["CFBundleTypeIconFile"] = doc_icon_stem
    info["CFBundleDocumentTypes"] = [piano_doc, midi_doc]

    with plist_path.open("wb") as handle:
        plistlib.dump(info, handle)


def build_installer_dmg(app_path: Path, work_dir: Path, icon_hint: Path | None = None) -> Path:
    """Create a drag-and-drop DMG containing the .app and Applications link."""
    hdiutil = ensure_command("hdiutil")

    staging_dir = work_dir / f"{app_path.stem}_dmg"
    if staging_dir.exists():
        shutil.rmtree(staging_dir, ignore_errors=True)
    staging_dir.mkdir(parents=True, exist_ok=True)

    app_copy = staging_dir / app_path.name
    shutil.copytree(app_path, app_copy)

    applications_link = staging_dir / "Applications"
    if applications_link.exists() or applications_link.is_symlink():
        applications_link.unlink()
    applications_link.symlink_to("/Applications")

    icon_source = find_bundle_icns(app_path)
    temp_icon: Path | None = None
    if icon_source is None and icon_hint is not None:
        try:
            converted_icon, generated = prepare_icon(icon_hint, work_dir)
            icon_source = converted_icon
            temp_icon = converted_icon if generated else None
        except Exception as exc:  # pragma: no cover - best-effort customization
            print(f"Warning: Failed to prepare DMG icon ({exc}); continuing without it.", file=sys.stderr)

    if icon_source is not None:
        try:
            apply_dmg_volume_icon(staging_dir, icon_source)
        except Exception as exc:  # pragma: no cover - cosmetic feature only
            print(f"Warning: Unable to apply DMG icon ({exc}).", file=sys.stderr)
    if temp_icon is not None and temp_icon.exists():
        temp_icon.unlink()

    dmg_path = work_dir / f"{app_path.stem}.dmg"
    if dmg_path.exists():
        dmg_path.unlink()

    cmd = [
        hdiutil,
        "create",
        "-volname",
        app_path.stem,
        "-srcfolder",
        str(staging_dir),
        "-fs",
        "HFS+",
        "-format",
        "UDZO",
        "-ov",
        str(dmg_path),
    ]
    print(f"Creating installer DMG: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    shutil.rmtree(staging_dir, ignore_errors=True)
    return dmg_path


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    build_dir = output_dir / "keyTAB_build"

    if build_dir.exists():
        shutil.rmtree(build_dir, ignore_errors=True)
    build_dir.mkdir(parents=True, exist_ok=True)

    unused_modules = determine_unused_qt_modules(PROJECT_ROOT)
    if unused_modules:
        print("Excluding unused PySide6 modules:")
        for mod in sorted(unused_modules):
            print(f"  - {mod}")
    else:
        print("No excludable PySide6 modules detected; keeping defaults.")

    result_path: Path | None = None
    dmg_path: Path | None = None
    success = False
    try:
        result_path = run_pyinstaller(
            args.entry,
            args.name,
            build_dir,
            args.icon,
            exclude_modules=unused_modules,
        )
        copy_qt_licenses(result_path)
        doc_icon_file = ensure_document_icon(result_path, build_dir, args.icon)
        update_info_plist(result_path, args.name, doc_icon_file)
        print(f"App bundle created at: {result_path}")
        try:
            dmg_path = build_installer_dmg(result_path, build_dir, args.icon)
            print(f"Installer DMG created at: {dmg_path}")
        except Exception as exc:
            print(f"Warning: Failed to produce DMG: {exc}", file=sys.stderr)

        final_app_path = output_dir / f"{args.name}.app"
        if final_app_path.exists():
            shutil.rmtree(final_app_path, ignore_errors=True)
        shutil.move(str(result_path), final_app_path)
        print(f"Copied .app to: {final_app_path}")

        if dmg_path and dmg_path.exists():
            final_dmg_path = output_dir / dmg_path.name
            if final_dmg_path.exists():
                final_dmg_path.unlink()
            shutil.move(str(dmg_path), final_dmg_path)
            print(f"Copied DMG to: {final_dmg_path}")
        success = True
    finally:
        if success:
            shutil.rmtree(build_dir, ignore_errors=True)
            print(f"Cleaned build directory {build_dir}.")
        else:
            print(f"Build failed or incomplete; leaving artifacts in {build_dir} for inspection.")

    if result_path is None or not (output_dir / f"{args.name}.app").exists():
        raise SystemExit("Build failed: .app bundle missing after cleanup.")
    if dmg_path is None:
        print("Installer DMG unavailable; .app bundle still produced.", file=sys.stderr)


if __name__ == "__main__":
    if sys.platform != "darwin":
        print("Warning: This script is intended to run on macOS (darwin).", file=sys.stderr)
    main()
