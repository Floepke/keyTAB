from __future__ import annotations


def _normalize_hand_value(raw_hand: object) -> str:
    hand = str(raw_hand or '<').strip()
    if hand.lower() == 'l':
        return '<'
    if hand.lower() == 'r':
        return '>'
    if hand not in ('<', '>'):
        return '<'
    return hand


def _normalize_color_value(raw_color: object) -> str:
    if raw_color is None:
        return 'auto'
    if not isinstance(raw_color, str):
        return 'auto'
    color = raw_color.strip()
    if not color:
        return 'auto'

    lowered = color.lower()
    if lowered in ('default', 'auto'):
        return 'auto'
    if color in ('<', '>'):
        return 'auto'
    if lowered in ('l', 'r'):
        return 'auto'

    return color


def _normalize_notehead_value(raw_notehead: object) -> str:
    if raw_notehead is None:
        return 'auto'
    if not isinstance(raw_notehead, str):
        return 'auto'
    notehead = raw_notehead.strip()
    if not notehead:
        return 'auto'
    if notehead.lower() in ('default', 'auto'):
        return 'auto'
    return notehead


def convert_legacy_piano_data(data: dict) -> dict:
    """Convert legacy .piano conventions in-place and return the same dict.

    Current migration:
    - events.note[].color: '<'/'>'/empty/default -> 'auto'
    - events.note[].notehead: empty/default -> 'auto'
    - events.note[].hand: legacy 'l'/'r' -> '<'/'>'

    This function is intentionally idempotent.
    """
    if not isinstance(data, dict):
        return data

    # Legacy schema migration: move editor zoom into app_state.
    try:
        editor = data.get('editor', None)
        if isinstance(editor, dict) and 'zoom_mm_per_quarter' in editor:
            app_state = data.get('app_state', None)
            if not isinstance(app_state, dict):
                app_state = {}
                data['app_state'] = app_state
            if 'zoom_mm_per_quarter' not in app_state:
                app_state['zoom_mm_per_quarter'] = editor.get('zoom_mm_per_quarter')
    except Exception:
        pass

    events = data.get('events', None)
    if not isinstance(events, dict):
        return data

    notes = events.get('note', None)
    if not isinstance(notes, list):
        return data

    for note in notes:
        if not isinstance(note, dict):
            continue
        note['hand'] = _normalize_hand_value(note.get('hand', '<'))
        note['color'] = _normalize_color_value(note.get('color', None))
        note['notehead'] = _normalize_notehead_value(note.get('notehead', None))

    return data
