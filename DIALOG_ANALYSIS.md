# Dialog Modal/Modeless Analysis

## Summary
This document identifies all dialog classes in the keyTAB project and their modal/modeless status.

---

## Dialogs with setModal(True) - NEED CONVERSION

### 1. AboutDialog
- **File:** [ui/about_dialog.py](ui/about_dialog.py)
- **Class Name:** `AboutDialog`
- **setModal(True):** ✅ Yes (line 13)
- **Called In:** [ui/main_window.py](ui/main_window.py#L1733-L1734)
- **Usage Pattern:** `.exec()`
- **Status:** Modal - NEEDS CONVERSION

---

### 2. InfoDialog
- **File:** [ui/dialogs/info_dialog.py](ui/dialogs/info_dialog.py)
- **Class Name:** `InfoDialog`
- **setModal(True):** ✅ Yes (line 17)
- **Called In:** [ui/main_window.py](ui/main_window.py#L1955-L1958)
- **Usage Pattern:** `.exec()`
- **Status:** Modal - NEEDS CONVERSION

---

### 3. PreferencesDialog
- **File:** [ui/dialogs/preferences_dialog.py](ui/dialogs/preferences_dialog.py)
- **Class Name:** `PreferencesDialog`
- **setModal(True):** ✅ Yes (line 18)
- **Called In:** [settings_manager.py](settings_manager.py#L380-L382)
- **Usage Pattern:** `.exec()`
- **Status:** Modal - NEEDS CONVERSION

---

### 4. StyleDialog
- **File:** [ui/dialogs/style_dialog.py](ui/dialogs/style_dialog.py)
- **Class Name:** `StyleDialog`
- **setModal(True):** ✅ Yes (line 640)
- **Called In:** [ui/main_window.py](ui/main_window.py#L1891-L1952)
- **Usage Pattern:** `.show()` (already modeless in usage, but setModal(True) in init!)
- **Status:** Mismatched - has `setModal(True)` but called with `.show()`
- **Note:** This dialog is already using `.show()` instead of `.exec()`, but still has `setModal(True)` in __init__

---

### 5. TimeSignatureDialog
- **File:** [ui/dialogs/time_signature_dialog.py](ui/dialogs/time_signature_dialog.py)
- **Class Name:** `TimeSignatureDialog`
- **setModal(True):** ✅ Yes (line 26)
- **setWindowModality():** ✅ Also has `QtCore.Qt.WindowModality.WindowModal` (line 27)
- **Called In:** [editor/tool/time_signature_tool.py](editor/tool/time_signature_tool.py#L219)
- **Usage Pattern:** `.exec()` (line 227)
- **Status:** Modal - NEEDS CONVERSION

---

### 6. NoteheadDialog
- **File:** [ui/dialogs/notehead_dialog.py](ui/dialogs/notehead_dialog.py)
- **Class Name:** `NoteheadDialog`
- **setModal(True):** ✅ Yes (line 68)
- **setWindowModality():** ✅ Also has `QtCore.Qt.WindowModality.WindowModal` (line 69)
- **Called In:** 
  - [editor/tool/note_tool.py](editor/tool/note_tool.py#L662)
  - [editor/tool/grace_note_tool.py](editor/tool/grace_note_tool.py#L251)
- **Usage Pattern:** `.exec()` (line 401 in get_notehead classmethod)
- **Status:** Modal - NEEDS CONVERSION

---

### 7. BulkKeyRangeDialog
- **File:** [ui/dialogs/line_break_dialog.py](ui/dialogs/line_break_dialog.py)
- **Class Name:** `BulkKeyRangeDialog`
- **setModal(True):** ✅ Yes (line 23)
- **setWindowModality():** ✅ Also has `QtCore.Qt.ApplicationModal` (line 24)
- **Called In:** [ui/dialogs/line_break_dialog.py](ui/dialogs/line_break_dialog.py) (within LineBreakDialog)
- **Usage Pattern:** Likely `.exec()` in LineBreakDialog usage
- **Status:** Modal - NEEDS CONVERSION
- **Note:** This is a helper dialog used by LineBreakDialog

---

### 8. LineBreakDialog
- **File:** [ui/dialogs/line_break_dialog.py](ui/dialogs/line_break_dialog.py)
- **Class Name:** `LineBreakDialog`
- **setModal(True):** ✅ Yes (line 139)
- **setWindowModality():** ✅ Also has `QtCore.Qt.NonModal` (line 140)
- **Called In:** [ui/main_window.py](ui/main_window.py#L1964-2010)
- **Usage Pattern:** `.show()` (line 2007)
- **Status:** Mismatched - has both `setModal(True)` and `setWindowModality(QtCore.Qt.NonModal)`
- **Note:** Already using `.show()` but has conflicting parameters in __init__

---

## Dialogs Already Modeless - NO CHANGES NEEDED

### 1. FluidSynthReverbConfigDialog
- **File:** [ui/dialogs/fluidsynth_reverb_config_dialog.py](ui/dialogs/fluidsynth_reverb_config_dialog.py)
- **Class Name:** `FluidSynthReverbConfigDialog`
- **setModal(True):** ❌ No - has `setModal(False)` (line 22)
- **Called In:** [ui/main_window.py](ui/main_window.py#L1377-L1379)
- **Usage Pattern:** `.show()`
- **Status:** ✅ Already Modeless - NO CHANGES NEEDED

---

## Dialogs Without Custom Class (QMessageBox)

### 1. ErrorDialog (show_error_dialog function)
- **File:** [ui/error_dialog.py](ui/error_dialog.py)
- **Type:** Function-based using `QMessageBox`
- **Class Name:** N/A (uses `QtWidgets.QMessageBox`)
- **setModal():** Not explicitly set - uses QMessageBox default (modal)
- **Called By:** Error handling code throughout the project
- **Usage Pattern:** `.exec()`
- **Status:** Modal - doesn't use custom QDialog class

---

## Other Dialogs Found

### 1. TextDialog
- **File:** [ui/dialogs/text_dialog.py](ui/dialogs/text_dialog.py)
- **Class Name:** `TextDialog`
- **setModal():** ❌ No setModal() call found
- **Called In:** [editor/tool/text_tool.py](editor/tool/text_tool.py#L337)
- **Usage Pattern:** `.show()` (line 356 in text_tool.py)
- **Status:** Already modeless - NO CHANGES NEEDED

---

### 2. ScriptDialog
- **File:** [scripting/dialog.py](scripting/dialog.py)
- **Class Name:** `ScriptDialog`
- **setModal():** ❌ No setModal() call found
- **Status:** Not explicitly set - likely default (modal)
- **Note:** Used by scripting engine, would need to verify actual usage pattern

---

## Conversion Priority

**High Priority (Used Frequently):**
1. ✅ AboutDialog - Simple, just used for display
2. ✅ InfoDialog - Used from main_window, but already has good structure
3. ✅ PreferencesDialog - Important settings dialog
4. ✅ TimeSignatureDialog - Editor tool dialog
5. ✅ NoteheadDialog - Editor tool dialog used in note/grace-note tools

**Medium Priority (Utility Dialogs):**
6. StyleDialog - Already using `.show()` but has conflicting setModal settings
7. LineBreakDialog - Already using `.show()` but has conflicting setModal settings
8. BulkKeyRangeDialog - Helper dialog for LineBreakDialog

**Low Priority:**
9. TextDialog - Already modeless via `.show()`
10. FluidSynthReverbConfigDialog - Already modeless ✅
11. ScriptDialog - No explicit setModal
12. ErrorDialog - Uses QMessageBox, not a custom QDialog

---

## Key Findings

1. **Modeless Already Implemented:** StyleDialog and LineBreakDialog are already using `.show()` but have redundant/conflicting `setModal(True)` in their __init__ methods.

2. **Modal via .exec():** Most dialogs like AboutDialog, InfoDialog, TimeSignatureDialog, and NoteheadDialog use `.exec()` which blocks until dismissed.

3. **Consistent Patterns Needed:** The LineBreakDialog has `setModal(True)` AND `setWindowModality(QtCore.Qt.NonModal)` which is contradictory.

4. **TextDialog Special:** TextDialog doesn't call setModal() at all and uses `.show()`, making it already modeless.

5. **FluidSynth Dialog:** Already correctly set to `setModal(False)` and using `.show()`.
