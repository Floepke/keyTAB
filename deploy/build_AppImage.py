#!/usr/bin/env python3
"""Build a Linux AppImage for keyTAB using PyInstaller + linuxdeploy.

When run directly, the script does the following in order:
1) Parse CLI arguments (output directory, executable name, entry script, icon, extra PyInstaller args).
    - Default output directory is ~/Desktop when no args are provided.
2) Verify Linux x86_64 environment and ensure required system packages are present.
3) Install Python dependencies from requirements.txt and ensure PyInstaller is available.
4) Create a dedicated build workspace at <output>/keyTAB_build/ and invoke PyInstaller with
    its dist/work/spec paths inside that workspace.
5) Stage an AppDir: copy the PyInstaller bundle into usr/lib/<name>, create usr/bin launcher symlink,
   and strip portable-unfriendly Qt plugins.
6) Bundle project licenses/notices, Qt licenses, and selected audio-related shared libraries.
7) Embed a 512x512 PNG icon, write desktop entry, MIME package, AppStream metadata, and AppRun launcher.
8) Download/prepare linuxdeploy.AppImage, set env (including APPIMAGE_EXTRACT_AND_RUN=1), and build the AppImage.
9) Rename/make-executable the produced AppImage, move it to <output>, and remove the build workspace.
    If the build fails, leave <output>/keyTAB_build/ intact for inspection.
Usage:
    python3 deploy/build_AppImage.py --output /path/to/out
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import shlex
import shutil
import subprocess
import sys
from glob import glob
from pathlib import Path
from urllib.request import urlretrieve

LINUXDEPLOY_URL = (
    "https://github.com/linuxdeploy/linuxdeploy/releases/download/continuous/"
    "linuxdeploy-x86_64.AppImage"
)

MIME_TYPE_KEYTAB = "application/x-keytab"
MIME_TYPES_MIDI = "audio/midi;audio/x-midi"
MIME_TYPES_MUSICXML = "application/vnd.recordare.musicxml+xml;application/vnd.recordare.musicxml"
APPSTREAM_ID = "org.philipbergwerf.keytab"
AUTHOR_NAME = "Philip Bergwerf"
APP_SUMMARY = (
    "A music engraver for clear and readable MIDI-file music notation."
)
APP_DESCRIPTION = (
    "keyTAB is a professional engraver for engraving MIDI files as music notation. "
    "It is based on the Klavarskribo notation system and revised by "
    "Philip Bergwerf to be more flexible.\n"
    "\n"
    "Import a .mid file and keyTAB converts it instantly into a readable MIDI notation sheet. "
    "Use the mouse-centric editor to refine layout, design, and musical details."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build keyTAB AppImage.")
    parser.add_argument(
        "--output",
        default=str(Path.home() / "Desktop"),
        help="Output directory for build files and final AppImage.",
    )
    parser.add_argument(
        "--name",
        default="keyTAB",
        help="Executable name (default: keyTAB).",
    )
    parser.add_argument(
        "--entry",
        default="keyTAB.py",
        help="Entry script (default: keyTAB.py).",
    )
    parser.add_argument(
        "--icon-path",
        default="",
        help="Path to PNG icon to embed in the AppImage.",
    )
    parser.add_argument(
        "--extra-args",
        default="",
        help="Extra PyInstaller args (quoted string).",
    )
    return parser.parse_args()


def ensure_clean_output_dir(out_dir: Path) -> None:
    if out_dir.exists():
        if not out_dir.is_dir():
            raise SystemExit(f"Output path is not a directory: {out_dir}")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)


def cleanup_build_artifacts(*paths: Path) -> None:
    for p in paths:
        if p.exists():
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                try:
                    p.unlink()
                except Exception:
                    pass


def ensure_appimage_tools(tools_dir: Path) -> Path:
    tools_dir.mkdir(parents=True, exist_ok=True)
    linuxdeploy = tools_dir / "linuxdeploy.AppImage"
    if not linuxdeploy.exists():
        urlretrieve(LINUXDEPLOY_URL, linuxdeploy)
    for tool in (linuxdeploy,):
        try:
            mode = tool.stat().st_mode
            tool.chmod(mode | 0o111)
        except Exception:
            pass
    return linuxdeploy


def ensure_pip_dependency(package: str) -> None:
    try:
        __import__(package)
        return
    except Exception:
        pass

    pip3 = shutil.which("pip3")
    if not pip3:
        raise SystemExit("pip3 not found in PATH. Please install pip3.")
    cmd = [pip3, "install", package, "--break-system-packages"]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"Failed to install {package} with pip3")


def ensure_pip_requirements(project_root: Path) -> None:
    req_path = project_root / "requirements.txt"
    if not req_path.exists():
        return
    pip3 = shutil.which("pip3")
    if not pip3:
        raise SystemExit("pip3 not found in PATH. Please install pip3.")
    cmd = [pip3, "install", "-r", str(req_path), "--break-system-packages"]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit("Failed to install requirements.txt with pip3")


def resolve_fluidsynth_module_file() -> Path | None:
    """Return the installed fluidsynth.py module path, if available."""
    try:
        spec = importlib.util.find_spec("fluidsynth")
    except Exception:
        return None
    if spec is None:
        return None
    origin = getattr(spec, "origin", None)
    if not origin:
        return None
    try:
        p = Path(str(origin)).resolve()
    except Exception:
        return None
    if p.exists() and p.is_file():
        return p
    return None


def _is_debian_pkg_installed(pkg: str) -> bool:
    result = subprocess.run(["dpkg", "-s", pkg], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def ensure_system_packages(packages: list[str]) -> None:
    missing = [p for p in packages if not _is_debian_pkg_installed(p)]
    if not missing:
        return
    apt_get = shutil.which("apt-get")
    if not apt_get:
        raise SystemExit("apt-get not found. Please install system packages manually.")
    subprocess.run(["sudo", apt_get, "update"], check=False)
    cmd = ["sudo", apt_get, "install", "-y", *missing]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"Failed to install system packages: {', '.join(missing)}")


def ensure_system_package_alternatives(options: list[str]) -> None:
    for pkg in options:
        if _is_debian_pkg_installed(pkg):
            return
    apt_get = shutil.which("apt-get")
    if not apt_get:
        raise SystemExit("apt-get not found. Please install system packages manually.")
    subprocess.run(["sudo", apt_get, "update"], check=False)
    for pkg in options:
        result = subprocess.run(["sudo", apt_get, "install", "-y", pkg])
        if result.returncode == 0:
            return
    raise SystemExit(f"Failed to install any of: {', '.join(options)}")


def write_desktop_file(appdir: Path, name: str) -> Path:
    desktop_dir = appdir / "usr" / "share" / "applications"
    desktop_dir.mkdir(parents=True, exist_ok=True)
    desktop_path = desktop_dir / f"{name}.desktop"
    desktop_path.write_text(
        "[Desktop Entry]\n"
        f"Name={name}\n"
        f"Comment={APP_SUMMARY}\n"
        f"Exec={name}\n"
        f"Icon={name}\n"
        "Type=Application\n"
        "Categories=AudioVideo;Audio;Music;\n"
        f"MimeType={MIME_TYPE_KEYTAB};{MIME_TYPES_MIDI};{MIME_TYPES_MUSICXML};\n"
        "Terminal=false\n",
        encoding="utf-8",
    )
    return desktop_path


def write_appstream_metadata(appdir: Path, name: str) -> Path:
    metainfo_dir = appdir / "usr" / "share" / "metainfo"
    metainfo_dir.mkdir(parents=True, exist_ok=True)
    metainfo_path = metainfo_dir / f"{APPSTREAM_ID}.metainfo.xml"
    metainfo_path.write_text(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<component type=\"desktop\">\n"
        f"  <id>{APPSTREAM_ID}</id>\n"
        f"  <name>{name}</name>\n"
        f"  <summary>{APP_SUMMARY}</summary>\n"
        f"  <developer_name>{AUTHOR_NAME}</developer_name>\n"
        "  <metadata_license>CC0-1.0</metadata_license>\n"
        "  <project_license>MIT</project_license>\n"
        "  <description>\n"
        "    <p>keyTAB is a professional MIDI engraver for music notation. "
        "It is based on Klavarskribo notation and revised by Philip Bergwerf to be "
        "clearer and more flexible.</p>\n"
        "    <p>Import a .mid file and keyTAB converts it instantly into readable "
        "notation. Use the mouse-centric editor to refine layout, design, and "
        "musical details.</p>\n"
        "  </description>\n"
        "</component>\n",
        encoding="utf-8",
    )
    return metainfo_path


def write_mime_package(appdir: Path, name: str) -> Path:
    mime_dir = appdir / "usr" / "share" / "mime" / "packages"
    mime_dir.mkdir(parents=True, exist_ok=True)
    mime_path = mime_dir / f"{name}.xml"
    mime_path.write_text(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
        "<mime-info xmlns=\"http://www.freedesktop.org/standards/shared-mime-info\">\n"
        f"  <mime-type type=\"{MIME_TYPE_KEYTAB}\">\n"
        "    <comment>keyTAB score</comment>\n"
        "    <glob pattern=\"*.piano\"/>\n"
        "  </mime-type>\n"
        "  <mime-type type=\"application/vnd.recordare.musicxml+xml\">\n"
        "    <comment>MusicXML score</comment>\n"
        "    <glob pattern=\"*.musicxml\"/>\n"
        "  </mime-type>\n"
        "  <mime-type type=\"application/vnd.recordare.musicxml\">\n"
        "    <comment>Compressed MusicXML score</comment>\n"
        "    <glob pattern=\"*.mxl\"/>\n"
        "  </mime-type>\n"
        "</mime-info>\n",
        encoding="utf-8",
    )
    return mime_path


def copy_png_icon(icon_path: Path, target_path: Path) -> None:
    if not icon_path.exists():
        raise SystemExit(f"Icon not found: {icon_path}")
    if icon_path.suffix.lower() != ".png":
        raise SystemExit("Icon must be a PNG file.")
    try:
        from PIL import Image
    except Exception:
        raise SystemExit("Pillow is required to resize icon. Install via requirements.txt.")
    img = Image.open(icon_path).convert("RGBA")
    if img.size != (512, 512):
        img = img.resize((512, 512), resample=Image.LANCZOS)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(target_path, format="PNG")


def write_apprun(appdir: Path, name: str) -> Path:
    apprun = appdir / "AppRun"
    apprun.write_text(
        "#!/bin/sh\n"
        "HERE=\"$(dirname \"$(readlink -f \"$0\")\")\"\n"
        # Expose shared libs, PyInstaller _internal, and ALSA plugins/config
        f"export LD_LIBRARY_PATH=\"$HERE/usr/lib:$HERE/usr/lib/alsa-lib:$HERE/usr/lib/{name}/_internal:$LD_LIBRARY_PATH\"\n"
        f"export ALSA_CONFIG_PATH=\"$HERE/usr/share/alsa/alsa.conf\"\n"
        f"export ALSA_CONFIG_DIR=\"$HERE/usr/share/alsa\"\n"
        f"export ALSA_PLUGIN_DIR=\"$HERE/usr/lib/alsa-lib\"\n"
        # Hint pyfluidsynth to the bundled FluidSynth shared library first.
        # Try both versioned and plain soname variants.
        "for lib in \"$HERE/usr/lib/libfluidsynth.so\"*; do\n"
        "  if [ -f \"$lib\" ]; then\n"
        "    export PYFLUIDSYNTH_LIB=\"$lib\"\n"
        "    break\n"
        "  fi\n"
        "done\n"
        # Ensure bundled Python modules (including rtmidi) are discoverable
        f"export PYTHONPATH=\"$HERE/usr/lib/{name}/_internal:$PYTHONPATH\"\n"
        # Force mido to use bundled RtMidi backend
        "export MIDO_BACKEND=mido.backends.rtmidi\n"
        f"exec \"$HERE/usr/bin/{name}\" \"$@\"\n",
        encoding="utf-8",
    )
    try:
        mode = apprun.stat().st_mode
        apprun.chmod(mode | 0o111)
    except Exception:
        pass
    return apprun


def remove_qt_plugins_for_portability(app_bundle: Path) -> None:
    qtiff = app_bundle / "_internal" / "PySide6" / "Qt" / "plugins" / "imageformats" / "libqtiff.so"
    if qtiff.exists():
        try:
            qtiff.unlink()
        except Exception:
            pass


def _copy_file_if_exists(src: Path, dest: Path) -> None:
    if not src.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest)


def _copy_tree_if_exists(src: Path, dest: Path) -> None:
    if not src.exists():
        return
    if dest.exists():
        shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(src, dest)


def copy_qt_licenses(appdir: Path, exe_name: str) -> None:
    try:
        import PySide6  # type: ignore
    except Exception:
        return

    qt_root = Path(getattr(PySide6, "__file__", "")).resolve().parent
    search_roots = [qt_root, qt_root / "Qt", qt_root / "Qt" / "LICENSES"]
    dest_root = appdir / "usr" / "share" / "licenses" / exe_name / "qt"
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


def copy_bundled_assets(project_root: Path, lib_app_dir: Path, appdir: Path, exe_name: str) -> None:
    # Licenses and notices (ship both alongside the binary and under /usr/share for compliance).
    licenses_src = project_root / "licenses"
    licenses_dst_bin = lib_app_dir / "licenses"
    _copy_tree_if_exists(licenses_src, licenses_dst_bin)

    _copy_file_if_exists(project_root / "LICENSE", lib_app_dir / "LICENSE")
    _copy_file_if_exists(project_root / "THIRD_PARTY_NOTICES.md", lib_app_dir / "THIRD_PARTY_NOTICES.md")
    _copy_file_if_exists(project_root / "docs" / "LICENSING_AND_PACKAGING.md", lib_app_dir / "docs" / "LICENSING_AND_PACKAGING.md")

    share_root = appdir / "usr" / "share"
    share_license_dir = share_root / "licenses" / exe_name
    _copy_tree_if_exists(licenses_src, share_license_dir)
    _copy_file_if_exists(project_root / "LICENSE", share_license_dir / "LICENSE")

    doc_dir = share_root / "doc" / exe_name
    _copy_file_if_exists(project_root / "THIRD_PARTY_NOTICES.md", doc_dir / "THIRD_PARTY_NOTICES.md")
    _copy_file_if_exists(project_root / "docs" / "LICENSING_AND_PACKAGING.md", doc_dir / "LICENSING_AND_PACKAGING.md")

    # Bundle ALSA config/plugins to keep RtMidi device discovery working in AppImage
    try:
        alsa_conf_src = Path("/usr/share/alsa")
        alsa_conf_dst = appdir / "usr" / "share" / "alsa"
        _copy_tree_if_exists(alsa_conf_src, alsa_conf_dst)
    except Exception:
        pass
    try:
        alsa_plugins_src = Path("/usr/lib/x86_64-linux-gnu/alsa-lib")
        alsa_plugins_dst = appdir / "usr" / "lib" / "alsa-lib"
        _copy_tree_if_exists(alsa_plugins_src, alsa_plugins_dst)
    except Exception:
        pass


def copy_shared_libs(appdir: Path, lib_names: list[str], extra_targets: list[Path] | None = None) -> None:
    lib_dst = appdir / "usr" / "lib"
    lib_dst.mkdir(parents=True, exist_ok=True)
    extra_targets = extra_targets or []
    for target in extra_targets:
        target.mkdir(parents=True, exist_ok=True)
    search_dirs = [
        "/usr/lib",
        "/usr/local/lib",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/lib/x86_64-linux-gnu",
        "/lib",
        "/lib/x86_64-linux-gnu",
    ]
    for name in lib_names:
        found = None
        for d in search_dirs:
            for candidate in glob(str(Path(d) / f"{name}*.so*")):
                p = Path(candidate)
                if p.is_file():
                    found = p
                    break
            if found:
                break
        if not found:
            print(f"Warning: could not find {name} to bundle; ensure it is available at runtime.")
            continue
        try:
            # Copy the exact soname and add a plain .so symlink so ctypes.find_library finds it.
            copied_paths = []
            dest_main = lib_dst / found.name
            shutil.copy2(found, dest_main)
            copied_paths.append(dest_main)
            for target in extra_targets:
                dest_extra = target / found.name
                shutil.copy2(found, dest_extra)
                copied_paths.append(dest_extra)

            for dest in copied_paths:
                # Only add a compatibility symlink when the filename has a version (e.g. libfoo.so.3).
                if ".so." not in dest.name:
                    continue
                plain_so = dest.parent / (dest.name.split(".so.", 1)[0] + ".so")
                if plain_so.name == dest.name:
                    continue
                try:
                    if plain_so.exists() or plain_so.is_symlink():
                        plain_so.unlink()
                    plain_so.symlink_to(dest.name)
                except Exception:
                    pass
            print(f"Bundled shared library: {found}")
        except Exception as exc:
            print(f"Warning: failed to copy {found}: {exc}")


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    entry_script = project_root / args.entry
    if not entry_script.exists():
        raise SystemExit(f"Entry script not found: {entry_script}")

    out_dir = Path(args.output).expanduser().resolve()
    ensure_clean_output_dir(out_dir)

    build_root = out_dir / "keyTAB_build"
    if build_root.exists():
        shutil.rmtree(build_root, ignore_errors=True)
    build_root.mkdir(parents=True, exist_ok=True)

    if platform.system().lower() != "linux":
        raise SystemExit("AppImage builds are supported on Linux only.")
    if platform.machine().lower() not in ("x86_64", "amd64"):
        raise SystemExit("AppImage build currently supports x86_64 only.")

    ensure_system_packages(["patchelf", "desktop-file-utils", "file", "libfuse2"])
    ensure_system_package_alternatives(["libtiff5", "libtiff6"])
    ensure_system_package_alternatives(["libfluidsynth3", "libfluidsynth2", "libfluidsynth1"])
    ensure_pip_requirements(project_root)
    ensure_pip_dependency("PyInstaller")

    work_dir = build_root / "_build"
    spec_dir = build_root / "_spec"
    dist_dir = build_root / "_dist"
    appdir = build_root / "_appdir"
    tools_dir = build_root / "_tools"

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        f"--name={args.name}",
        f"--distpath={str(dist_dir)}",
        f"--workpath={str(work_dir)}",
        f"--specpath={str(spec_dir)}",
        str(entry_script),
    ]
    # Ensure MIDI backends are bundled even if imported dynamically.
    # pyfluidsynth is imported as module name "fluidsynth".
    cmd.extend([
        "--hidden-import=rtmidi",
        "--hidden-import=rtmidi._rtmidi",
        "--collect-all=rtmidi",
        "--hidden-import=mido.backends.rtmidi",
        "--collect-all=mido",
        "--hidden-import=fluidsynth",
        "--collect-all=fluidsynth",
    ])
    # Bundle translation files so non-English UI works inside the AppImage.
    cmd.append(f"--add-data={str(project_root / 'i18n')}:i18n")
    fluidsynth_module = resolve_fluidsynth_module_file()
    if fluidsynth_module is not None:
        # Ensure the binding file is present at runtime even when module graph
        # analysis misses it due dynamic/conditional import patterns.
        cmd.append(f"--add-data={str(fluidsynth_module)}:_internal")
    else:
        print("Warning: fluidsynth module file not found; pyfluidsynth may be missing from bundle.")
    if args.extra_args.strip():
        cmd.extend(shlex.split(args.extra_args))

    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(project_root))
    if result.returncode != 0:
        print(f"Build failed; leaving artifacts in {build_root} for inspection.")
        return result.returncode

    exe_name = args.name
    app_bundle = dist_dir / exe_name
    exe_path = app_bundle / exe_name
    if not exe_path.exists():
        raise SystemExit(f"Build succeeded but executable not found: {exe_path}")

    if appdir.exists():
        shutil.rmtree(appdir, ignore_errors=True)
    bin_dir = appdir / "usr" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    lib_app_dir = appdir / "usr" / "lib" / exe_name
    if lib_app_dir.exists():
        shutil.rmtree(lib_app_dir, ignore_errors=True)
    shutil.copytree(app_bundle, lib_app_dir)
    remove_qt_plugins_for_portability(lib_app_dir)
    launcher = bin_dir / exe_name
    if launcher.exists():
        launcher.unlink()
    launcher.symlink_to(Path("../lib") / exe_name / exe_name)

    # Bundle licenses and notices for compliance; user supplies their own soundfont.
    copy_bundled_assets(project_root, lib_app_dir, appdir, exe_name)
    copy_qt_licenses(appdir, exe_name)

    # Bundle FluidSynth and all of its transitive shared-library dependencies that
    # are typically absent on a minimal/fresh Linux install.  The list covers every
    # library reachable via `ldd libfluidsynth.so.3` that is NOT part of glibc/libm/
    # libpthread/libdl (those are always present and must NOT be bundled).
    copy_shared_libs(
        appdir,
        [
            # FluidSynth itself
            "libfluidsynth",
            # FluidSynth direct deps
            "libinstpatch-1.0",   # instrument-patch support – most common missing dep
            "libsoxr",            # sample-rate conversion
            "libopus",            # Opus codec
            "libvorbis",          # Vorbis codec (libvorbis + libvorbisenc + libvorbisfile)
            "libvorbisenc",
            "libvorbisfile",
            "libogg",             # Ogg container (dep of Vorbis)
            "libFLAC",            # FLAC codec
            "libsndfile",         # unified audio I/O
            "libglib-2.0",        # GLib (FluidSynth threading / event loop)
            "libgobject-2.0",     # GObject (dep of GLib / instpatch)
            "libgmodule-2.0",
            "libgio-2.0",
            # Audio backends
            "libpulse",           # PulseAudio client
            "libpulse-simple",
            "libasound",          # ALSA
            # instpatch deps
            "libxml2",            # XML parsing used by instpatch
            "liblzma",            # dep of libxml2
            "libz",               # zlib – dep of libxml2 / libsndfile
        ],
        extra_targets=[],
    )

    icon_dir = appdir / "usr" / "share" / "icons" / "hicolor" / "256x256" / "apps"
    icon_dir.mkdir(parents=True, exist_ok=True)
    icon_path = icon_dir / f"{exe_name}.png"
    if args.icon_path:
        icon_src = Path(args.icon_path).expanduser().resolve()
    else:
        icon_src = project_root / "icons" / "keyTAB.png"
    copy_png_icon(icon_src, icon_path)

    desktop_file = write_desktop_file(appdir, exe_name)
    write_mime_package(appdir, exe_name)
    write_appstream_metadata(appdir, exe_name)
    write_apprun(appdir, exe_name)

    linuxdeploy = ensure_appimage_tools(tools_dir)
    env = os.environ.copy()
    env["PATH"] = f"{tools_dir}:{env.get('PATH', '')}"
    env.setdefault("LINUXDEPLOY_OUTPUT_VERSION", "dev")
    # Avoid FUSE mount requirement when running AppImage tools.
    env.setdefault("APPIMAGE_EXTRACT_AND_RUN", "1")
    # We bundle project/Qt licenses explicitly; disable linuxdeploy's dpkg-query
    # copyright probing to avoid warnings for private/bundled .so files.
    env.setdefault("DISABLE_COPYRIGHT_FILES_DEPLOYMENT", "1")

    appimage_cmd = [
        str(linuxdeploy),
        "--appdir",
        str(appdir),
        "--executable",
        str(bin_dir / exe_name),
        "--desktop-file",
        str(desktop_file),
        "--icon-file",
        str(icon_path),
        "--output",
        "appimage",
    ]
    result = subprocess.run(appimage_cmd, cwd=str(build_root), env=env)
    if result.returncode != 0:
        print(f"AppImage packaging failed; leaving artifacts in {build_root} for inspection.")
        return result.returncode

    produced = sorted(build_root.glob("*.AppImage"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not produced:
        raise SystemExit("AppImage build finished but no AppImage was produced.")
    final_appimage = out_dir / f"{exe_name}.AppImage"
    if final_appimage.exists():
        try:
            final_appimage.unlink()
        except Exception:
            pass
    shutil.move(str(produced[0]), str(final_appimage))
    try:
        mode = final_appimage.stat().st_mode
        final_appimage.chmod(mode | 0o111)
    except Exception:
        pass

    try:
        if exe_path.exists():
            exe_path.unlink()
    except Exception:
        pass

    cleanup_build_artifacts(work_dir, spec_dir, dist_dir, appdir, tools_dir)
    shutil.rmtree(build_root, ignore_errors=True)
    print(f"Build complete: {final_appimage}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
