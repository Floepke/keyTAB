#!/usr/bin/env python3
"""Build a Windows keyTAB-setup.exe installer using PyInstaller + Inno Setup.

When run directly, the script performs these steps:
1) Parse CLI args (entry, executable name, app version, output dir, icon path, extra args).
   - Default output dir is ~/Desktop.
2) Ensure the script is executed on Windows and that Inno Setup is installed.
3) Create an isolated build workspace at <output>/keyTAB_build/.
4) Prepare an .ico icon (convert from PNG when needed).
5) Run PyInstaller in onedir mode — produces a folder with .exe + all libs (no runtime extraction).
6) Generate a keyTAB.iss Inno Setup script targeting the onedir output.
7) Compile the .iss with ISCC.exe to produce keyTAB-setup.exe.
8) Copy keyTAB-setup.exe to <output>.
9) Remove the build workspace on success.
   - If any step fails, keep <output>/keyTAB_build/ for inspection.

Requirements:
  pip install pyinstaller pillow
  Install Inno Setup from https://jrsoftware.org/isdl.php
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENTRY = PROJECT_ROOT / "keyTAB.py"
DEFAULT_ICON = PROJECT_ROOT / "icons" / "keyTAB.png"
DEFAULT_OUTPUT = Path.home() / "Desktop"
DEFAULT_NAME = "keyTAB"
DEFAULT_INSTALLER_NAME = "keyTAB-setup"
DEFAULT_ENGRAVING_FONT = PROJECT_ROOT / "fonts" / "Edwin.otf"
DEFAULT_DYNAMIC_SYMBOL_FONT = PROJECT_ROOT / "fonts" / "LelandText.otf"

sys.path.insert(0, str(PROJECT_ROOT))
from version import __version__ as _app_version  # noqa: E402
DEFAULT_APP_VERSION = _app_version


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build keyTAB.exe with PyInstaller on Windows.")
    parser.add_argument(
        "--entry",
        type=Path,
        default=DEFAULT_ENTRY,
        help="Path to Python entry point (default: keyTAB.py).",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help="Executable base name (default: keyTAB).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for final .exe (default: ~/Desktop).",
    )
    parser.add_argument(
        "--icon",
        type=Path,
        default=DEFAULT_ICON,
        help="Path to icon file (.ico preferred; PNG accepted).",
    )
    parser.add_argument(
        "--app-version",
        default=DEFAULT_APP_VERSION,
        help=f"Application version string used in the installer (default: {DEFAULT_APP_VERSION}).",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help="Extra arguments forwarded to PyInstaller (quoted string).",
    )
    return parser.parse_args()


def ensure_windows() -> None:
    if not sys.platform.startswith("win"):
        raise SystemExit("This build script is for Windows only.")


def ensure_pyinstaller_available() -> None:
    if importlib.util.find_spec("PyInstaller") is not None:
        return
    raise SystemExit(
        "PyInstaller is not installed in the active Python environment. "
        f"Install it with: {sys.executable} -m pip install pyinstaller"
    )


def ensure_requirements_installed(project_root: Path) -> None:
    req = project_root / "requirements.txt"
    if not req.exists():
        return
    cmd = [sys.executable, "-m", "pip", "install", "-r", str(req)]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit("Failed to install requirements.txt")


_ISCC_CANDIDATES: list[str] = [
    r"C:\Program Files (x86)\Inno Setup 6\ISCC.exe",
    r"C:\Program Files\Inno Setup 6\ISCC.exe",
    r"C:\Program Files (x86)\Inno Setup 5\ISCC.exe",
    r"C:\Program Files\Inno Setup 5\ISCC.exe",
]


def find_iscc() -> Path | None:
    for candidate in _ISCC_CANDIDATES:
        p = Path(candidate)
        if p.exists():
            return p
    found = shutil.which("ISCC")
    if found:
        return Path(found)
    return None


def ensure_inno_setup_available() -> Path:
    iscc = find_iscc()
    if iscc is None:
        raise SystemExit(
            "Inno Setup is not installed (ISCC.exe not found).\n"
            "Download and install it from: https://jrsoftware.org/isdl.php\n"
            "Then re-run this script."
        )
    return iscc


def normalize_windows_version(version: str) -> str:
    """Convert a semantic-like version to a 4-part numeric Windows version."""
    numeric_parts = [p for p in re.split(r"[^0-9]+", version) if p]
    if not numeric_parts:
        return "0.0.0.0"
    parts = [str(int(p)) for p in numeric_parts[:4]]
    while len(parts) < 4:
        parts.append("0")
    return ".".join(parts)


def generate_iss_script(
    app_name: str,
    app_version: str,
    app_dir: Path,
    engraving_font_path: Path,
    dynamic_symbol_font_path: Path,
    ico_path: Path,
    installer_output_dir: Path,
    installer_name: str,
    iss_path: Path,
) -> None:
    """Write an Inno Setup script that packages the onedir build into an installer."""
    installer_output_dir.mkdir(parents=True, exist_ok=True)
    windows_version = normalize_windows_version(app_version)
    script = f"""[Setup]
AppId={app_name}
AppName={app_name}
AppVersion={app_version}
AppVerName={app_name} {app_version}
AppPublisher=Philip Bergwerf
DefaultDirName={{autopf}}\\{app_name}
DefaultGroupName={app_name}
OutputDir={installer_output_dir}
OutputBaseFilename={installer_name}
SetupIconFile={ico_path}
VersionInfoVersion={windows_version}
VersionInfoTextVersion={app_version}
Compression=lzma
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{{cm:CreateDesktopIcon}}"; GroupDescription: "{{cm:AdditionalIcons}}"; Flags: unchecked

[Files]
Source: "{app_dir}\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{engraving_font_path}"; DestDir: "{{autofonts}}"; FontInstall: "Edwin"; Flags: onlyifdoesntexist uninsneveruninstall
Source: "{dynamic_symbol_font_path}"; DestDir: "{{autofonts}}"; FontInstall: "LelandText"; Flags: onlyifdoesntexist uninsneveruninstall

[Icons]
Name: "{{group}}\\{app_name}"; Filename: "{{app}}\\{app_name}.exe"
Name: "{{group}}\\{{cm:UninstallProgram,{app_name}}}"; Filename: "{{uninstallexe}}"
Name: "{{autodesktop}}\\{app_name}"; Filename: "{{app}}\\{app_name}.exe"; Tasks: desktopicon

[Run]
Filename: "{{app}}\\{app_name}.exe"; Description: "{{cm:LaunchProgram,{app_name}}}"; Flags: nowait postinstall skipifsilent
"""
    iss_path.write_text(script, encoding="utf-8")


def run_iscc(iscc: Path, iss_path: Path) -> None:
    cmd = [str(iscc), str(iss_path)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"ISCC.exe exited with code {result.returncode}")


def _png_to_ico(png_path: Path, ico_path: Path) -> None:
    try:
        from PIL import Image  # type: ignore
    except Exception as exc:
        raise SystemExit(
            "Pillow is required to convert PNG icons to ICO. "
            "Install it with requirements.txt or pip install pillow."
        ) from exc

    img = Image.open(png_path).convert("RGBA")
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    ico_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(ico_path, format="ICO", sizes=sizes)


def prepare_icon(icon_path: Path, work_dir: Path) -> tuple[Path, bool]:
    resolved = icon_path.resolve()
    if not resolved.exists():
        raise SystemExit(f"Icon not found: {resolved}")
    suffix = resolved.suffix.lower()
    if suffix == ".ico":
        return resolved, False
    if suffix == ".png":
        ico_out = work_dir / "keyTAB.ico"
        _png_to_ico(resolved, ico_out)
        return ico_out, True
    raise SystemExit("Unsupported icon format. Use .ico or .png.")


def cleanup_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        try:
            path.unlink()
        except Exception:
            pass


def run_pyinstaller(
    entry: Path,
    name: str,
    work_dir: Path,
    icon: Path,
    extra_args: str,
) -> Path:
    """Run PyInstaller in onedir mode. Returns the onedir output folder."""
    entry_path = entry.resolve()
    if not entry_path.exists():
        raise SystemExit(f"Entry file not found: {entry_path}")

    work_dir.mkdir(parents=True, exist_ok=True)
    dist_dir = work_dir / "dist"
    build_dir = work_dir / "build"
    spec_dir = work_dir / "spec"
    for d in (dist_dir, build_dir, spec_dir):
        d.mkdir(parents=True, exist_ok=True)

    icon_path, generated = prepare_icon(icon, work_dir)

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--onedir",
        "--windowed",
        "--name",
        name,
        "--icon",
        str(icon_path),
        "--distpath",
        str(dist_dir),
        "--workpath",
        str(build_dir),
        "--specpath",
        str(spec_dir),
        "--contents-directory",
        "_internal",
        "--hidden-import=rtmidi",
        "--hidden-import=rtmidi._rtmidi",
        "--collect-all=rtmidi",
        "--hidden-import=mido.backends.rtmidi",
        "--collect-all=mido",
        "--add-data",
        f"{str(PROJECT_ROOT / 'i18n')};i18n",
        str(entry_path),
    ]

    if extra_args.strip():
        cmd.extend(shlex.split(extra_args, posix=False))

    print(f"Running: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    finally:
        if generated:
            cleanup_path(icon_path)

    produced_dir = dist_dir / name
    if not produced_dir.is_dir():
        raise SystemExit("PyInstaller finished but onedir output folder was not produced.")
    return produced_dir


def main() -> None:
    ensure_windows()
    args = parse_args()

    name = str(args.name or DEFAULT_NAME).strip() or DEFAULT_NAME
    app_version = str(args.app_version or DEFAULT_APP_VERSION).strip() or DEFAULT_APP_VERSION
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    workspace_dir = output_dir / f"{name}_build"
    installer_output_dir = workspace_dir / "installer_out"
    installer_name = f"{DEFAULT_INSTALLER_NAME}-{app_version}"
    final_installer = output_dir / f"{installer_name}.exe"

    cleanup_path(workspace_dir)

    iscc = ensure_inno_setup_available()
    ensure_pyinstaller_available()
    ensure_requirements_installed(PROJECT_ROOT)
    engraving_font_path = DEFAULT_ENGRAVING_FONT.resolve()
    if not engraving_font_path.exists():
        raise SystemExit(f"Required engraving font not found: {engraving_font_path}")
    dynamic_symbol_font_path = DEFAULT_DYNAMIC_SYMBOL_FONT.resolve()
    if not dynamic_symbol_font_path.exists():
        raise SystemExit(f"Required dynamic symbol font not found: {dynamic_symbol_font_path}")

    try:
        # Step 1: Build app with PyInstaller (onedir — no runtime extraction).
        onedir_folder = run_pyinstaller(
            entry=args.entry,
            name=name,
            work_dir=workspace_dir,
            icon=args.icon,
            extra_args=str(args.extra_args or ""),
        )

        # Step 2: Prepare .ico for the installer wizard icon.
        ico_for_installer = workspace_dir / f"{name}.ico"
        icon_resolved = args.icon.resolve()
        if icon_resolved.suffix.lower() == ".ico":
            shutil.copy2(icon_resolved, ico_for_installer)
        else:
            _png_to_ico(icon_resolved, ico_for_installer)

        # Step 3: Generate and compile the Inno Setup script.
        iss_path = workspace_dir / f"{name}.iss"
        generate_iss_script(
            app_name=name,
            app_version=app_version,
            app_dir=onedir_folder,
            engraving_font_path=engraving_font_path,
            dynamic_symbol_font_path=dynamic_symbol_font_path,
            ico_path=ico_for_installer,
            installer_output_dir=installer_output_dir,
            installer_name=installer_name,
            iss_path=iss_path,
        )
        run_iscc(iscc, iss_path)

        # Step 4: Copy installer to output directory.
        produced_installer = installer_output_dir / f"{installer_name}.exe"
        if not produced_installer.exists():
            raise RuntimeError(f"ISCC finished but installer not found at: {produced_installer}")
        if final_installer.exists():
            cleanup_path(final_installer)
        shutil.copy2(produced_installer, final_installer)

    except Exception as exc:
        print(f"Build failed: {exc}", file=sys.stderr)
        print(f"Build workspace retained for inspection: {workspace_dir}", file=sys.stderr)
        raise SystemExit(1)

    cleanup_path(workspace_dir)
    print(f"Build complete: {final_installer}")


if __name__ == "__main__":
    main()
