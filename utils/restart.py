from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def _build_restart_command() -> list[str]:
    argv = list(sys.argv)
    if not argv:
        return []

    exe = str(sys.executable or "").strip()
    if getattr(sys, "frozen", False):
        # AppImage builds must restart via the outer AppImage launcher.
        # Restarting the inner mounted executable can fail once the old
        # runtime unmounts the squashfs during shutdown.
        appimage = str(os.environ.get("APPIMAGE", "") or "").strip()
        if os.name == "posix" and appimage:
            appimage_path = Path(appimage).expanduser()
            if appimage_path.exists() and os.access(str(appimage_path), os.X_OK):
                return [str(appimage_path), *argv[1:]]
        if exe:
            return [exe, *argv[1:]]
        return argv

    raw0 = str(argv[0] or "").strip()
    if not raw0:
        return [exe, *argv[1:]] if exe else []

    path0 = Path(raw0).expanduser()
    suffix = path0.suffix.lower()
    if suffix in {".py", ".pyw"} and exe:
        if not path0.is_absolute():
            path0 = (Path.cwd() / path0).resolve()
        return [exe, str(path0), *argv[1:]]

    if not path0.is_absolute():
        path0 = (Path.cwd() / path0).resolve()
    return [str(path0), *argv[1:]]


def restart_current_process() -> bool:
    command = _build_restart_command()
    if not command:
        return False

    kwargs: dict[str, object] = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
        "cwd": str(Path.cwd()),
        "env": dict(os.environ),
    }
    if os.name == "posix":
        kwargs["start_new_session"] = True
    elif os.name == "nt":
        kwargs["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        subprocess.Popen(command, **kwargs)
        return True
    except Exception:
        return False
