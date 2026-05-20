from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple
from dataclasses import fields
import os
import sys
from datetime import datetime
from PySide6.QtCore import QCoreApplication

from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget
from ui.error_dialog import show_error_dialog

from file_model.SCORE import SCORE, MetaData, _merge_with_defaults
from file_model.info import Info
from file_model.analysis import Analysis
from file_model.base_grid import BaseGrid
from file_model.appstate import AppState
from file_model.layout import Layout
from midi.midi_exporter import export_score_to_midi
from utils.piano2musicxml import export_score_to_musicxml
from utils.CONSTANT import UTILS_SAVE_DIR
from appdata_manager import get_appdata_manager


class FileManager:
    """
    Manages creating, opening, and saving SCORE files with native dialogs.

    - Holds the current SCORE instance and its filesystem path
    - Uses SCORE.new(), SCORE.load(path), and SCORE.save(path)
    - Provides new(), open(), save(), and save_as() methods
    """

    # Open dialog: .keytab is native; .piano is import-only legacy format.
    OPEN_FILE_FILTER = (
        "Supported Files (*.keytab *.piano *.mid *.midi *.musicxml *.mxl *.xml);;"
        "keyTAB Score (*.keytab *.piano);;"
        "MIDI File (*.mid *.midi);;"
        "MusicXML File (*.musicxml *.mxl *.xml);;"
    )
    # Save dialog: native format is .keytab.
    SAVE_FILE_FILTER = (
        "keyTAB Score (*.keytab);;"
        "MIDI File (*.mid *.midi);;"
        "MusicXML File [unusable in its current state] (*.musicxml *.xml);;"
    )

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        self._parent: Optional[QWidget] = parent
        self._current: SCORE = SCORE().new()
        self._path: Optional[Path] = None
        # Initialize last_dir from appdata if available, else home directory
        adm = get_appdata_manager()
        last_dir_str = str(adm.get("last_file_dialog_dir", "") or "")
        self._last_dir: Path = Path(last_dir_str) if last_dir_str else Path.home()
        # Track whether the current SCORE has unsaved changes since the last save/load
        self._dirty: bool = False
        self._last_autosave_ts: datetime | None = None
        self._before_save_hook: Callable[[SCORE], None] | None = None
        # Ensure the autosave directory exists on initialization
        os.makedirs(UTILS_SAVE_DIR, exist_ok=True)

    # Accessors
    def current(self) -> SCORE:
        return self._current

    def path(self) -> Optional[Path]:
        return self._path

    def set_parent(self, parent: Optional[QWidget]) -> None:
        self._parent = parent

    def set_before_save_hook(self, hook: Callable[[SCORE], None] | None) -> None:
        self._before_save_hook = hook

    def _apply_before_save_hook(self) -> None:
        if hasattr(self._current, 'set_before_save_hook'):
            self._current.set_before_save_hook(self._before_save_hook)

    # Core operations
    def new(self) -> SCORE:
        """Create a new SCORE and clear the current path."""
        self._current = SCORE().new()
        # Load default style if it exists
        self._apply_default_style()
        self._path = None
        self._dirty = False
        # Snapshot a fresh session immediately so restore works even before edits.
        self.autosave_current()
        return self._current

    def _apply_score_template(self, template: dict) -> None:
        if not isinstance(template, dict):
            return
        score = self._current
        data = dict(template or {})
        data.pop('events', None)

        def _build_info(entry: dict) -> Info:
            base = Info()
            if not isinstance(entry, dict):
                return base
            info_block = data.get('info', {}) if isinstance(data.get('info', {}), dict) else {}
            title = str(info_block.get('title', base.title) or base.title)
            composer = str(info_block.get('composer', base.composer) or base.composer)
            copyright_text = str(info_block.get('copyright', base.copyright) or base.copyright)
            arranger = str(info_block.get('arranger', base.arranger) or base.arranger)
            lyricist = str(info_block.get('lyricist', base.lyricist) or base.lyricist)
            comment = str(info_block.get('comment', base.comment) or base.comment)
            return Info(
                title=title,
                composer=composer,
                copyright=copyright_text,
                arranger=arranger,
                lyricist=lyricist,
                comment=comment,
            )

        score.info = _build_info(data.get('info', {}) if isinstance(data.get('info', {}), dict) else {})
        editor_data = data.get('editor')
        if isinstance(editor_data, dict) and 'zoom_mm_per_quarter' in editor_data:
            score.app_state.zoom_mm_per_quarter = float(editor_data.get('zoom_mm_per_quarter'))
        app_state_data = data.get('app_state')
        if isinstance(app_state_data, dict):
            try:
                allowed = {f.name for f in fields(AppState)}
                filtered = {k: v for k, v in app_state_data.items() if k in allowed}
                score.app_state = AppState(**filtered)
            except Exception:
                score.app_state = AppState()
        meta_data = data.get('meta_data')
        if isinstance(meta_data, dict):
            score.meta_data = MetaData(**meta_data)
        base_grid = data.get('base_grid')
        if isinstance(base_grid, list):
            score.base_grid = [BaseGrid(**bg) if isinstance(bg, dict) else BaseGrid() for bg in base_grid]

    def _apply_default_style(self) -> None:
        """Load and apply the default style to the current score if it exists."""
        from ui.dialogs.style_dialog import load_default_style
        default_layout = load_default_style()
        if default_layout is not None:
            self._current.layout = default_layout

    def replace_current(self, new_score: SCORE) -> None:
        """Replace the current SCORE instance (used by undo/redo)."""
        self._current = new_score
        # Autosave on any model replacement (e.g., undo/redo application)
        self.autosave_current()
        # Consider undo/redo a model change relative to last explicit save
        self._dirty = True

    def load(self) -> Optional[SCORE]:
        """Load a keyTAB project file via a native file dialog."""
        start_dir = str(self._path.parent if self._path else self._last_dir)
        fname, _ = QFileDialog.getOpenFileName(
            self._parent,
            "Load Score",
            start_dir,
            self.OPEN_FILE_FILTER,
        )
        if not fname:
            return None
        
        # Run load
        suffix = Path(fname).suffix.lower()
        if suffix in (".mid", ".midi"):
            # Allow an external hook (e.g. the MIDI import dialog) to load the file.
            # Hook signature: (path: str) -> SCORE | None.
            # Return None to cancel the import; return a SCORE to use instead of
            # the default midi_load().
            hook = getattr(self, 'midi_import_hook', None)
            if callable(hook):
                try:
                    hook_result = hook(str(fname))
                except Exception as exc:
                    raise RuntimeError(f"Failed to load MIDI: {exc}") from exc
                if hook_result is None:
                    return None  # user cancelled the import dialog
                self._current = hook_result
            else:
                # Default MIDI load (no dialog)
                try:
                    from midi.midi_loader import midi_load
                    self._current = midi_load(fname)
                    if hasattr(self._current, '_normalize_events_after_load'):
                        self._current._normalize_events_after_load()
                except Exception as exc:
                    raise RuntimeError(f"Failed to load MIDI: {exc}") from exc
            self._path = None
            self._last_dir = Path(fname).parent
            adm = get_appdata_manager()
            adm.set("last_file_dialog_dir", str(self._last_dir))
            adm.save()
            # Imported from external format; mark dirty until explicitly saved
            self._dirty = True
            self.autosave_current()
            adm = get_appdata_manager()
            adm.set("last_opened_file", str(fname))
            adm.save()
            self._push_recent_file(str(fname))
            self._show_load_checks_info(self._current)
            return self._current
        elif suffix in (".musicxml", ".mxl", ".xml"):
            # Load MusicXML via utils.musicxml2piano parser; keep project path unset
            try:
                from utils.musicxml2piano import parse_musicxml
                self._current, _stats = parse_musicxml(Path(fname))
                if hasattr(self._current, '_normalize_events_after_load'):
                    self._current._normalize_events_after_load()
            except Exception as exc:
                raise RuntimeError(f"Failed to load MusicXML: {exc}")
            self._path = None
            self._last_dir = Path(fname).parent
            adm = get_appdata_manager()
            adm.set("last_file_dialog_dir", str(self._last_dir))
            adm.save()
            # Imported from external format; mark dirty until explicitly saved
            self._dirty = True
            self.autosave_current()
            adm = get_appdata_manager()
            adm.set("last_opened_file", str(fname))
            adm.save()
            self._push_recent_file(str(fname))
            self._show_load_checks_info(self._current)
            return self._current
        else:
            # Native keyTAB JSON file (.keytab) or legacy .piano import.
            self._current = SCORE().load(fname)
            source_path = Path(fname)
            self._path = source_path.with_suffix('.keytab') if source_path.suffix.lower() == '.piano' else source_path
            self._last_dir = self._path.parent
            adm = get_appdata_manager()
            adm.set("last_file_dialog_dir", str(self._last_dir))
            adm.save()
            # Opening legacy .piano keeps project dirty until user saves .keytab.
            self._dirty = source_path.suffix.lower() == '.piano'
            self.autosave_current()
            # Track last opened file in appdata
            adm = get_appdata_manager()
            adm.set("last_opened_file", str(self._path))
            adm.save()
            self._push_recent_file(str(self._path))
            self._show_load_checks_info(self._current)
            return self._current

    def open_path(self, path: str) -> Optional[SCORE]:
        """Programmatically open a keyTAB project path.

        Returns the SCORE on success, None on failure.
        """
        suffix = Path(path).suffix.lower()
        if suffix in (".mid", ".midi"):
            hook = getattr(self, 'midi_import_hook', None)
            if callable(hook):
                try:
                    hook_result = hook(str(path))
                except Exception as exc:
                    raise RuntimeError(f"Failed to load MIDI: {exc}") from exc
                if hook_result is None:
                    return None  # user cancelled the import dialog
                self._current = hook_result
            else:
                try:
                    from midi.midi_loader import midi_load
                    self._current = midi_load(path)
                    if hasattr(self._current, '_normalize_events_after_load'):
                        self._current._normalize_events_after_load()
                except Exception as exc:
                    raise RuntimeError(f"Failed to load MIDI: {exc}") from exc
            self._path = None
            self._last_dir = Path(path).parent
            self._dirty = True
            self.autosave_current(apply_hook=False)
            adm = get_appdata_manager()
            adm.set("last_opened_file", str(path))
            adm.save()
            self._push_recent_file(str(path))
            self._show_load_checks_info(self._current)
            return self._current
        elif suffix in (".musicxml", ".mxl", ".xml"):
            from utils.musicxml2piano import parse_musicxml
            self._current, _stats = parse_musicxml(Path(path))
            if hasattr(self._current, '_normalize_events_after_load'):
                self._current._normalize_events_after_load()
            self._path = None
            self._last_dir = Path(path).parent
            self._dirty = True
            self.autosave_current(apply_hook=False)
            adm = get_appdata_manager()
            adm.set("last_opened_file", str(path))
            adm.save()
            self._push_recent_file(str(path))
            self._show_load_checks_info(self._current)
            return self._current
        else:
            self._current = SCORE().load(path)
            source_path = Path(path)
            self._path = source_path.with_suffix('.keytab') if source_path.suffix.lower() == '.piano' else source_path
            self._last_dir = self._path.parent
            self._dirty = source_path.suffix.lower() == '.piano'
            # Do NOT apply hook here: the UI hasn't restored yet so current
            # scroll/page values would overwrite the freshly-loaded app_state.
            self.autosave_current(apply_hook=False)
            adm = get_appdata_manager()
            adm.set("last_opened_file", str(self._path))
            adm.save()
            self._push_recent_file(str(self._path))
            self._show_load_checks_info(self._current)
            return self._current

    def save(self) -> bool:
        """Save to the current path, or prompt Save As if none."""
        if self._path is None:
            return self.save_as(allow_export=False)
        try:
            self._apply_before_save_hook()
            self._refresh_analysis()
            suffix = str(self._path.suffix or '').lower()
            if suffix in ('.mid', '.midi'):
                return self.save_as(allow_export=False)
            target = self._path.with_suffix('.keytab') if suffix == '.piano' else self._path
            old_path = str(self._path)
            self._current.save(str(target))
            self._path = target
            self._dirty = False
            adm = get_appdata_manager()
            adm.set("last_opened_file", str(self._path))
            adm.save()
            if old_path != str(self._path):
                self._replace_recent_file_path(old_path, str(self._path))
            return True
        except Exception as exc:
            self._show_error("Failed to save score", f"{exc}")
            return False

    def save_as(self, allow_export: bool = True) -> bool:
        """Prompt for a path and save the current SCORE there."""
        start_dir = Path(self._path.parent if self._path else self._last_dir)

        def _default_name() -> str:
            try:
                title = str(getattr(getattr(self._current, 'info', None), 'title', '') or '').strip()
            except Exception:
                title = ''
            if not title:
                title = 'untitled'
            # Basic sanitization: remove path separators
            safe = title.replace('/', ' ').replace('\\', ' ')
            return f"{safe}.keytab"

        suggested = start_dir / _default_name()
        file_filter = self.SAVE_FILE_FILTER if allow_export else "keyTAB Score (*.keytab);;All Files (*)"
        fname, selected_filter = QFileDialog.getSaveFileName(
            self._parent,
            "Save Score As",
            str(suggested),
            file_filter,
        )
        if not fname:
            return False
        target = self._ensure_save_suffix(Path(fname), str(selected_filter or ''), allow_export=allow_export)
        self._refresh_analysis()
        suffix = str(target.suffix or '').lower()
        if suffix in ('.mid', '.midi'):
            export_score_to_midi(self._current, target)
            self._last_dir = target.parent
            adm = get_appdata_manager()
            adm.set("last_file_dialog_dir", str(self._last_dir))
            adm.save()
            return True
        elif suffix in ('.musicxml', '.xml'):
            self._apply_before_save_hook()
            export_score_to_musicxml(self._current, target)
            self._last_dir = target.parent
            adm = get_appdata_manager()
            adm.set("last_file_dialog_dir", str(self._last_dir))
            adm.save()
            return True
        else:
            self._apply_before_save_hook()
            self._current.save(str(target))
        self._path = target
        self._last_dir = target.parent
        adm = get_appdata_manager()
        adm.set("last_file_dialog_dir", str(self._last_dir))
        self._dirty = False
        adm.set("last_opened_file", str(self._path))
        adm.save()
        return True

    # Confirmation helpers for destructive actions
    def confirm_save_for_action(self, action_description: str, force_prompt: bool = False) -> bool:
        """If dirty, ask to save before proceeding with an action.

        Returns True to proceed, False to cancel the action.

        - Yes: save (Save As if no path). Proceed only if save succeeds.
        - No: proceed without saving.
        - Cancel: abort the action.
        """
        # Always snapshot the session so state can be restored even if action discards changes
        self.autosave_current()

        if not self.is_dirty():
            return True
        if self._parent is None:
            # In non-GUI context, default to proceed
            return True
        msg = QMessageBox(self._parent)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle(QCoreApplication.translate("FileManager", "Save changes?"))
        msg.setText(QCoreApplication.translate("FileManager", f"Do you want to save changes before {action_description}?"))
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Yes)
        result = msg.exec()

        if result == QMessageBox.Yes:
            success = self.save() if (self._path is not None) else self.save_as()
            return bool(success)
        elif result == QMessageBox.No:
            return True
        else:
            return False

    # Helpers
    def _ensure_keytab_suffix(self, p: Path) -> Path:
        if p.suffix.lower() != ".keytab":
            p = p.with_suffix(".keytab")
        return p

    def _ensure_save_suffix(self, p: Path, selected_filter: str, allow_export: bool = True) -> Path:
        suffix = str(p.suffix or '').lower()
        selected = str(selected_filter or '').lower()
        if allow_export and 'midi' in selected:
            if suffix not in ('.mid', '.midi'):
                return p.with_suffix('.mid')
            return p
        if allow_export and 'musicxml' in selected:
            if suffix not in ('.musicxml', '.xml'):
                return p.with_suffix('.musicxml')
            return p
        if suffix == '.keytab':
            return p
        if suffix == '.piano':
            return p.with_suffix('.keytab')
        return p.with_suffix('.keytab')

    def _show_error(self, title: str, text: str) -> None:
        if self._parent is None:
            # If no parent (non-GUI context), silently ignore GUI messagebox
            return
        show_error_dialog(self._parent, title, text)

    def _show_info(self, title: str, text: str) -> None:
        if self._parent is None:
            return
        msg = QMessageBox(self._parent)
        msg.setIcon(QMessageBox.Information)
        msg.setWindowTitle(title)
        msg.setText(text)
        msg.exec()

    def _show_load_checks_info(self, score: Optional[SCORE]) -> None:
        if score is None:
            return
        getter = getattr(score, 'get_load_checks_report', None)
        if not callable(getter):
            return
        report = getter() or {}
        deduped_removed = int(report.get('deduped_removed', 0) or 0)
        shortened_overlaps = int(report.get('shortened_overlaps', 0) or 0)
        converted_to_grace = int(report.get('converted_to_grace', 0) or 0)
        if deduped_removed <= 0 and shortened_overlaps <= 0 and converted_to_grace <= 0:
            return
        lines = []
        if deduped_removed > 0:
            lines.append(f"Removed duplicate-start notes: {deduped_removed}")
        if shortened_overlaps > 0:
            lines.append(f"Shortened overlapping notes: {shortened_overlaps}")
        if converted_to_grace > 0:
            lines.append(f"Converted short notes to grace notes: {converted_to_grace}")
        self._show_info("Score checks applied", "\n".join(lines))

    # Autosave and error-backup utilities
    def autosave_current(self, apply_hook: bool = True) -> None:
        """Save the current SCORE to the session file in session.keytab (JSON)."""
        target = Path(UTILS_SAVE_DIR) / "session.keytab"
        if apply_hook:
            self._apply_before_save_hook()
        self._refresh_analysis()
        self._current.save(str(target))

    def autosave_all(self, force: bool = False) -> None:
        """Persist session snapshot and project file (if available) throttled by dirty flag."""
        self.autosave_current()

        if self._path is None:
            # No project path yet; keep dirty so user is warned until they save explicitly
            return

        if not self._dirty and not force:
            return

        ok = self.save()
        if ok:
            self._dirty = False
            self._last_autosave_ts = datetime.now()
        else:
            self._dirty = True

    def _push_recent_file(self, path: str) -> None:
        p = str(path or "").strip()
        if not p:
            return
        adm = get_appdata_manager()
        recent = adm.get("recent_files", []) or []
        if not isinstance(recent, list):
            recent = []
        recent = [str(x) for x in recent if str(x).strip()]
        recent = [x for x in recent if x != p]
        recent.insert(0, p)
        recent = recent[:100]
        adm.set("recent_files", recent)
        adm.save()

    def _replace_recent_file_path(self, old_path: str, new_path: str) -> None:
        old_p = str(old_path or "").strip()
        new_p = str(new_path or "").strip()
        if not new_p:
            return
        adm = get_appdata_manager()
        recent = adm.get("recent_files", []) or []
        if not isinstance(recent, list):
            recent = []
        normalized = [str(x) for x in recent if str(x).strip()]
        replaced = [new_p if x == old_p else x for x in normalized]
        # Keep unique order, new path first.
        deduped: list[str] = []
        seen: set[str] = set()
        for x in [new_p] + replaced:
            if x in seen:
                continue
            seen.add(x)
            deduped.append(x)
        adm.set("recent_files", deduped[:100])
        adm.save()

    def rename_current_file(self, new_path: Path | str) -> bool:
        """Rename the current on-disk project file and update tracked path metadata."""
        if self._path is None:
            self._show_error("Rename file", "Current project has no saved file path.")
            return False
        old_path = Path(self._path)
        if not old_path.exists() or not old_path.is_file():
            self._show_error("Rename file", "Current file does not exist on disk.")
            return False

        target = Path(new_path).expanduser()
        if target == old_path:
            return True
        if target.exists():
            self._show_error("Rename file", f"Target already exists:\n{target}")
            return False
        if target.parent != old_path.parent and not target.parent.exists():
            self._show_error("Rename file", f"Target directory does not exist:\n{target.parent}")
            return False

        try:
            old_path.rename(target)
        except Exception as exc:
            self._show_error("Rename file", f"Failed to rename file:\n{exc}")
            return False

        self._path = target
        self._last_dir = target.parent
        adm = get_appdata_manager()
        adm.set("last_file_dialog_dir", str(self._last_dir))
        adm.set("last_opened_file", str(self._path))
        adm.save()
        self._replace_recent_file_path(str(old_path), str(target))
        return True

    def _refresh_analysis(self) -> None:
        """Recompute analysis so it persists in saved files.

        Retains existing engraved page counts when available.
        """
        sc = self._current
        if sc is None:
            return
        existing = getattr(sc, "analysis", None)
        lines_hint = None  # always derive lines from events to stay fresh
        pages_hint = getattr(existing, "pages", None)
        sc.analysis = Analysis.compute(sc, lines_count=lines_hint, pages_count=pages_hint)

    def on_model_changed(self) -> None:
        """Handle model change: mark dirty; autosave now happens on a timer/on close."""
        self._dirty = True

    def install_error_backup_hook(self) -> None:
        """Install a global excepthook to save a timestamped backup on errors."""
        # Preserve the original hook
        original_hook = sys.excepthook

        def _hook(exctype, value, tb):
            # Save timestamped error backup; format: dd-mm-YYYY-HH.MM.SS
            ts = datetime.now().strftime("%d-%m-%Y-%H.%M.%S")
            try:
                info = getattr(self._current, 'info', None)
                raw_title = str(getattr(info, 'title', '') or '').strip()
            except Exception:
                raw_title = ''
            safe_title = ''.join(ch for ch in raw_title if ch not in r'\\/:*?"<>|').strip()
            if not safe_title:
                safe_title = 'Untitled'
            safe_title = safe_title[:80]
            fname = f"keyTAB_error_backup_{safe_title}_{ts}.keytab"
            target = Path(UTILS_SAVE_DIR) / fname
            self._current.save(str(target))
            # Delegate to original hook to print traceback to terminal
            original_hook(exctype, value, tb)

        sys.excepthook = _hook

    def load_session_if_available(self) -> bool:
        """Load session snapshot from ~/.keyTAB folder into current score; keep path unset.

        Returns True if a session was restored.
        """
        session_path = Path(UTILS_SAVE_DIR) / "session.keytab"
        if not session_path.exists():
            legacy_session_path = Path(UTILS_SAVE_DIR) / "session.piano"
            if legacy_session_path.exists():
                session_path = legacy_session_path
            else:
                return False
        sc = SCORE().load(str(session_path))
        # Do not treat the session file as the project path
        self._current = sc
        self._path = None
        self._dirty = True
        return True

    # ---- Close confirmation ----
    def confirm_close_decision(self) -> str:
        """Ask user to save before quitting with Yes/No/Cancel.

        Returns one of:
        - "saved": user chose Yes and save succeeded
        - "discarded": user chose No
        - "proceed": no prompt needed (e.g. not dirty or no GUI parent)
        - "cancel": user canceled close or save failed
        """
        if not self.is_dirty():
            return "proceed"
        if self._parent is None:
            return "proceed"

        msg = QMessageBox(self._parent)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle(QCoreApplication.translate("FileManager", "Save before exiting?"))
        msg.setText(QCoreApplication.translate("FileManager", "Do you want to save changes before quitting?"))
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Yes)
        result = msg.exec()

        if result == QMessageBox.Yes:
            # Snapshot session before attempting save to preserve UI state
            self.autosave_current()
            success = self.save() if (self._path is not None) else self.save_as()
            return "saved" if bool(success) else "cancel"
        if result == QMessageBox.No:
            return "discarded"
        return "cancel"

    def confirm_close(self) -> bool:
        """Backward-compatible bool API for close confirmation."""
        return self.confirm_close_decision() != "cancel"

    # ---- Dirty tracking helpers ----
    def mark_dirty(self) -> None:
        self._dirty = True

    def clear_dirty(self) -> None:
        self._dirty = False

    def is_dirty(self) -> bool:
        return bool(self._dirty)
