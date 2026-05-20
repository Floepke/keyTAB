from __future__ import annotations

from copy import deepcopy


DEFAULT_STAVE_COUNT = 4


def _normalize_hand_value(raw_hand: object) -> str:
    hand = str(raw_hand or 'l').strip()
    if hand == '<':
        return 'l'
    if hand == '>':
        return 'r'
    if hand.lower() == 'l':
        return 'l'
    if hand.lower() == 'r':
        return 'r'
    return 'l'


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
    - events.note[].hand: legacy '<'/'>' -> 'l'/'r'
    - events.beam[].hand: legacy '<'/'>' -> 'l'/'r'
    - layout.notehead_tilt: negative values -> positive values

    This function is intentionally idempotent.
    """
    if not isinstance(data, dict):
        return data

    # Normalize notehead_tilt values
    layout = data.get('layout', {})
    if isinstance(layout, dict):
        notehead_tilt = layout.get('notehead_tilt', 0)
        if isinstance(notehead_tilt, (int, float)) and notehead_tilt < 0:
            layout['notehead_tilt'] = abs(notehead_tilt)

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
        # Forward compatibility: allow files that only store staves[0].events.
        staves = data.get('staves', None)
        if isinstance(staves, list) and staves:
            first = staves[0] if isinstance(staves[0], dict) else {}
            first_events = first.get('events', None) if isinstance(first, dict) else None
            if isinstance(first_events, dict):
                data['events'] = deepcopy(first_events)
                events = data.get('events', None)
        if not isinstance(events, dict):
            return data

    notes = events.get('note', None)
    if not isinstance(notes, list):
        return data

    for note in notes:
        if not isinstance(note, dict):
            continue
        note['hand'] = _normalize_hand_value(note.get('hand', 'l'))
        note['color'] = _normalize_color_value(note.get('color', None))
        note['notehead'] = _normalize_notehead_value(note.get('notehead', None))

    beams = events.get('beam', None)
    if isinstance(beams, list):
        for beam in beams:
            if not isinstance(beam, dict):
                continue
            beam['hand'] = _normalize_hand_value(beam.get('hand', 'l'))

    # Legacy -> new structure bridge: ensure 4 default staves exist.
    staves = data.get('staves', None)
    if not isinstance(staves, list) or not staves:
        staves = []
        data['staves'] = staves
        staves.append(
            {
                'name': 'Stave 1',
                'scale': 1.0,
                'enabled': True,
                'events': deepcopy(events),
            }
        )
    else:
        first = staves[0] if isinstance(staves[0], dict) else {}
        if not isinstance(first, dict):
            first = {}
            staves[0] = first
        if 'name' not in first:
            first['name'] = 'Stave 1'
        if 'scale' not in first:
            first['scale'] = 1.0
        if 'enabled' not in first:
            first['enabled'] = True
        first['events'] = deepcopy(events)

    # Ensure fixed-size stave list for editor (4 staves).
    normalized = []
    for idx, raw in enumerate(list(staves)[:DEFAULT_STAVE_COUNT]):
        item = raw if isinstance(raw, dict) else {}
        if 'name' not in item or not str(item.get('name', '') or '').strip():
            item['name'] = f'Stave {idx + 1}'
        if 'scale' not in item:
            item['scale'] = 1.0
        if 'enabled' not in item:
            item['enabled'] = True
        if 'events' not in item or not isinstance(item.get('events', None), dict):
            item['events'] = deepcopy(events) if idx == 0 else {}
        normalized.append(item)
    for idx in range(len(normalized), DEFAULT_STAVE_COUNT):
        normalized.append(
            {
                'name': f'Stave {idx + 1}',
                'scale': 1.0,
                'enabled': True,
                'events': {} if idx > 0 else deepcopy(events),
            }
        )
    data['staves'] = normalized

    return data
