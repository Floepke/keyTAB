# my json structure design for *.keytab files.
from __future__ import annotations
from dataclasses import dataclass, field, fields, MISSING, is_dataclass
from typing import Callable, List, Optional, get_args, get_origin, get_type_hints, Literal
import json
from datetime import datetime
from pathlib import Path

from file_model.events.note import Note
from file_model.events.grace_note import GraceNote
from file_model.events.pedal import Pedal
from file_model.events.text import Text
from copy import deepcopy
from file_model.events.slur import Slur
from file_model.events.beam import Beam
from file_model.events.start_repeat import StartRepeat
from file_model.events.end_repeat import EndRepeat
from file_model.events.double_bar import DoubleBar
from file_model.events.count_line import CountLine
from file_model.events.line_break import LineBreak
from file_model.events.tempo import Tempo
from file_model.events.grid_band import GridBand
from file_model.events.arpeggio import Arpeggio
from file_model.events.crescendo import Crescendo
from file_model.events.decrescendo import Decrescendo
from file_model.events.dynamic import DynamicSymbol
from file_model.layout import Layout
from file_model.font import Font
from file_model.info import Info
from file_model.analysis import Analysis
from utils.CONSTANT import GRACENOTE_THRESHOLD, QUARTER_NOTE_UNIT, SHORTEST_DURATION
from file_model.base_grid import BaseGrid
from file_model.appstate import AppState
from file_model.old_file_conversion import convert_legacy_piano_data
from utils.operator import Operator


DEFAULT_STAVE_COUNT: int = 4


def _timestamp_now() -> str:
	"""Return current timestamp formatted from preferences with a safe fallback."""
	default_fmt = "%d-%m-%Y_%H:%M:%S"
	fmt = default_fmt
	from settings_manager import get_preferences_manager
	pm = get_preferences_manager()
	raw_fmt = pm.get('timestamp_format', default_fmt)
	fmt = str(raw_fmt).strip() if raw_fmt is not None else default_fmt
	if not fmt:
		fmt = default_fmt
	return datetime.now().strftime(fmt)



@dataclass
class MetaData:
	description: str = 'This is a .keytab score file created with keyTAB.'
	extension: str = '.keytab'
	format: str = 'json'
	creation_timestamp: str = ''
	modification_timestamp: str = ''


@dataclass
class Events:
	note: List[Note] = field(default_factory=list)
	grace_note: List[GraceNote] = field(default_factory=list)
	pedal: List[Pedal] = field(default_factory=list)
	text: List[Text] = field(default_factory=list)
	slur: List[Slur] = field(default_factory=list)
	beam: List[Beam] = field(default_factory=list)
	grid_band: List[GridBand] = field(default_factory=list)
	start_repeat: List[StartRepeat] = field(default_factory=list)
	end_repeat: List[EndRepeat] = field(default_factory=list)
	double_bar: List[DoubleBar] = field(default_factory=list)
	count_line: List[CountLine] = field(default_factory=list)
	line_break: List[LineBreak] = field(default_factory=list)
	tempo: List[Tempo] = field(default_factory=list)
	arpeggio: List[Arpeggio] = field(default_factory=list)
	crescendo: List[Crescendo] = field(default_factory=list)
	decrescendo: List[Decrescendo] = field(default_factory=list)
	dynamic_symbol: List[DynamicSymbol] = field(default_factory=list)


@dataclass
class Stave:
	name: str = 'Stave 1'
	scale: float = 1.0
	enabled: bool = True
	events: Events = field(default_factory=Events)


def _defaults_for(dc_type):
	defaults = {}
	for f in fields(dc_type):
		if f.name.startswith('_'):
			continue
		if f.default is not MISSING:
			defaults[f.name] = f.default
		elif f.default_factory is not MISSING:  # type: ignore[attr-defined]
			defaults[f.name] = f.default_factory()
		else:
			defaults[f.name] = None
	return defaults


def _apply_legacy_conversion(data: dict) -> dict:
	"""Apply legacy file conversions (fail-open)."""
	data = convert_legacy_piano_data(data)

	# Layout key migration: dynamic symbol background padding rename.
	if isinstance(data, dict):
		layout = data.get('layout', None)
		if isinstance(layout, dict):
			if layout.get('time_signature_indicator_type') == 'both':
				layout['time_signature_indicator_type'] = 'classical & klavarskribo'
			if 'hairpin_font_size_pt' not in layout and 'hairpin_text_size_pt' in layout:
				layout['hairpin_font_size_pt'] = layout.get('hairpin_text_size_pt')
			if 'dynamic_symbol_background_padding_mm' not in layout:
				if 'dynamic_symbol_background_padding' in layout:
					layout['dynamic_symbol_background_padding_mm'] = layout.get('dynamic_symbol_background_padding')
				elif 'dynamic_background_padding' in layout:
					layout['dynamic_symbol_background_padding_mm'] = layout.get('dynamic_background_padding')
	return data


def _merge_with_defaults(dc_type, incoming: dict, context: str, skip_keys: set = {'id', '_id'}) -> dict:
	incoming = incoming or {}
	if not isinstance(incoming, dict):
		incoming = {}
	if dc_type is Note:
		incoming = dict(incoming)
		h = str(incoming.get('hand', 'l') or 'l').strip()
		if h not in ('l', 'r'):
			h = 'l'
		incoming['hand'] = h
		try:
			acc = int(incoming.get('acc', 0) or 0)
		except Exception:
			acc = 0
		incoming['acc'] = int(max(-2, min(2, acc)))
		raw_color = incoming.get('color', 'auto')
		if isinstance(raw_color, str):
			color = raw_color.strip()
		else:
			color = ''
		incoming['color'] = color if color else 'auto'
	defaults = _defaults_for(dc_type)
	try:
		type_hints = get_type_hints(dc_type, globals(), locals())
	except Exception:
		type_hints = {}
	merged = {}
	for f in fields(dc_type):
		name = f.name
		if name.startswith('_') or name in skip_keys:
			continue
		field_type = type_hints.get(name, f.type)
		default_value = defaults.get(name)
		raw_value = incoming.get(name, MISSING)
		if raw_value is MISSING:
			merged[name] = default_value
			continue
		if is_dataclass(field_type):
			if isinstance(raw_value, str):
				raw_value = {'text': raw_value}
			if isinstance(raw_value, field_type):
				merged[name] = raw_value
				continue
			if isinstance(raw_value, dict):
				child = _merge_with_defaults(field_type, raw_value, f"{context}.{name}")
				merged[name] = field_type(**child)
			else:
				merged[name] = default_value
			continue
		merged[name] = raw_value
	return merged


@dataclass
class SCORE:
	meta_data: MetaData = field(default_factory=MetaData)
	info: Info = field(default_factory=Info)
	analysis: Analysis = field(default_factory=Analysis)
	base_grid: List[BaseGrid] = field(default_factory=list)
	layout: Layout = field(default_factory=Layout)
	app_state: AppState = field(default_factory=AppState)
	tempo: List[Tempo] = field(default_factory=list)
	events: Events = field(default_factory=Events)
	staves: List[Stave] = field(default_factory=lambda: [Stave(name=f'Stave {i + 1}', enabled=True) for i in range(DEFAULT_STAVE_COUNT)])
	_next_id: int = 1
	_app_state_from_file: bool = False
	_last_load_checks_report: dict = field(default_factory=dict)
	_before_save_hook: Optional[Callable[["SCORE"], None]] = None

	# ---- Builders (ensure unique _id) ----
	def _gen_id(self) -> int:
		i = self._next_id
		self._next_id += 1
		return i

	def new_note(self, **kwargs) -> Note:
		base = {'pitch': 40, 'time': 0.0, 'duration': 100.0, 'hand': 'l', 'color': 'auto', 'acc': 0}
		base.update(kwargs)
		h = str(base.get('hand', 'l') or 'l').strip()
		if h not in ('l', 'r'):
			h = 'l'
		base['hand'] = h
		try:
			acc = int(base.get('acc', 0) or 0)
		except Exception:
			acc = 0
		base['acc'] = int(max(-2, min(2, acc)))
		raw_color = base.get('color', 'auto')
		if isinstance(raw_color, str):
			color = raw_color.strip()
		else:
			color = ''
		base['color'] = color if color else 'auto'
		obj = Note(**base, _id=self._gen_id())
		self.events.note.append(obj)
		return obj

	def new_grace_note(self, **kwargs) -> GraceNote:
		base = {'pitch': 41, 'time': 50.0, 'notehead': 'auto'}
		base.update(kwargs)
		obj = GraceNote(**base, _id=self._gen_id())
		self.events.grace_note.append(obj)
		return obj

	def new_pedal(self, **kwargs) -> Pedal:
		base = {'time': 0.0, 'rpitch': 0, 'symbol': 'down_keytab'}
		base.update(kwargs)
		obj = Pedal(**base, _id=self._gen_id())
		self.events.pedal.append(obj)
		return obj

	def new_text(self, **kwargs) -> Text:
		# Text anchor is center; store x as semitone offset and rotation in degrees.
		# Default font clones the score's layout font_text to avoid shared mutation.
		default_font = deepcopy(getattr(self.layout, 'font_text', Font()))
		base = {
			'text': 'Text',
			'time': 0.0,
			'x_rpitch': 0,
			'rotation': 0.0,
			'x_offset_mm': 0.0,
			'y_offset_mm': 0.0,
			'font': default_font,
			'use_custom_font': False,
			'text_background_width_offset_mm': 0.0,
		}
		base.update(kwargs)
		obj = Text(**base, _id=self._gen_id())
		self.events.text.append(obj)
		return obj

	def new_slur(self, **kwargs) -> Slur:
		# Default slur: straight line at c4 (0 semitone offset) over a short time window
		base = {
			'x1_rpitch': 0, 'y1_time': 0.0,
			'x2_rpitch': 0, 'y2_time': 25.0,
			'x3_rpitch': 0, 'y3_time': 75.0,
			'x4_rpitch': 0, 'y4_time': 100.0,
		}
		base.update(kwargs)
		obj = Slur(**base, _id=self._gen_id())
		self.events.slur.append(obj)
		return obj

	def new_beam(self, **kwargs) -> Beam:
		base = {'time': 0.0, 'duration': 100.0, 'hand': 'l'}
		base.update(kwargs)
		h = str(base.get('hand', 'l') or 'l').strip()
		if h not in ('l', 'r'):
			h = 'l'
		base['hand'] = h
		obj = Beam(**base, _id=self._gen_id())
		self.events.beam.append(obj)
		return obj

	def new_start_repeat(self, **kwargs) -> StartRepeat:
		base = {'time': 0.0}
		base.update(kwargs)
		obj = StartRepeat(**base, _id=self._gen_id())
		self.events.start_repeat.append(obj)
		return obj

	def new_end_repeat(self, **kwargs) -> EndRepeat:
		base = {'time': 0.0}
		base.update(kwargs)
		obj = EndRepeat(**base, _id=self._gen_id())
		self.events.end_repeat.append(obj)
		return obj

	def new_double_bar(self, **kwargs) -> DoubleBar:
		base = {'time': 0.0}
		base.update(kwargs)
		obj = DoubleBar(**base, _id=self._gen_id())
		self.events.double_bar.append(obj)
		return obj

	def new_count_line(self, **kwargs) -> CountLine:
		# Count lines now store horizontal position as semitone offsets from C4 (key 40).
		base = {'time': 0.0, 'rpitch1': 0, 'rpitch2': 4}
		base.update(kwargs)
		obj = CountLine(**base, _id=self._gen_id())
		self.events.count_line.append(obj)
		return obj


	def new_line_break(self, **kwargs) -> LineBreak:
		defaults = LineBreak()
		default_range = 'auto' if defaults.stave_range == 'auto' else list(defaults.stave_range or [0, 0])
		base = {
			'time': 0.0,
			'margin_mm': list(defaults.margin_mm),
			'stave_range': default_range
		}
		base.update(kwargs)
		obj = LineBreak(**base, _id=self._gen_id())
		events = self.stave_events_at()
		if events is None:
			if not isinstance(getattr(self, 'staves', None), list) or not self.staves:
				self.staves = [Stave(name=f'Stave {i + 1}', enabled=True, events=Events()) for i in range(DEFAULT_STAVE_COUNT)]
			events = self.stave_events_at(0)
		if events is not None:
			events.line_break.append(obj)
		return obj

	def new_tempo(self, **kwargs) -> Tempo:
		base = {'time': 0.0, 'duration': 0.0, 'tempo': Tempo().tempo}
		base.update(kwargs)
		obj = Tempo(**base, _id=self._gen_id())
		self.tempo.append(obj)
		return obj

	def new_arpeggio(self, **kwargs) -> Arpeggio:
		base = {'time': 0.0, 'rtime1': 0.0, 'rtime2': 32.0, 'note_pitches': []}
		base.update(kwargs)
		obj = Arpeggio(**base, _id=self._gen_id())
		self.events.arpeggio.append(obj)
		return obj

	def new_crescendo(self, **kwargs) -> Crescendo:
		base = {'time': 0.0, 'duration': 256.0, 'x_rpitch': 0}
		base.update(kwargs)
		obj = Crescendo(**base, _id=self._gen_id())
		self.events.crescendo.append(obj)
		return obj

	def new_decrescendo(self, **kwargs) -> Decrescendo:
		base = {'time': 0.0, 'duration': 256.0, 'x_rpitch': 0}
		base.update(kwargs)
		obj = Decrescendo(**base, _id=self._gen_id())
		self.events.decrescendo.append(obj)
		return obj

	def new_dynamic_symbol(self, **kwargs) -> DynamicSymbol:
		base = {'time': 0.0, 'x_rpitch': 0, 'symbol': ''}
		base.update(kwargs)
		obj = DynamicSymbol(**base, _id=self._gen_id())
		self.events.dynamic_symbol.append(obj)
		return obj

	# ---- Dict conversion ----
	def set_before_save_hook(self, hook: Optional[Callable[["SCORE"], None]]) -> None:
		self._before_save_hook = hook

	def get_dict(self) -> dict:
		def to_dict(obj):
			if isinstance(obj, list):
				return [to_dict(x) for x in obj]
			if hasattr(obj, "__dataclass_fields__"):
				out = {}
				for k in obj.__dataclass_fields__.keys():
					# Skip private/internal fields like _next_id
					if k.startswith('_'):
						continue
					out[k] = to_dict(getattr(obj, k))
				return out
			return obj
		return to_dict(self)

	def _load_events_container(self, ev_data: object, context_prefix: str) -> Events:
		"""Parse an events dict into an Events dataclass with fresh sequential ids."""
		ev = ev_data if isinstance(ev_data, dict) else {}
		out = Events()
		# Resolve postponed annotations (from __future__ import annotations)
		try:
			_ev_hints = get_type_hints(Events, globals(), locals())
		except Exception:
			_ev_hints = {}
		for f_ev in fields(Events):
			ann = _ev_hints.get(f_ev.name, f_ev.type)
			origin = get_origin(ann)
			args = get_args(ann)
			elem_type = args[0] if origin is list or origin is List else None
			if elem_type is None:
				continue
			name = f_ev.name
			items = ev.get(name, []) or []
			if not isinstance(items, list):
				continue
			lst = getattr(out, name)
			for idx, item in enumerate(items):
				incoming = item if isinstance(item, dict) else {}
				obj = elem_type(**_merge_with_defaults(elem_type, incoming, f'{context_prefix}.{name}[{idx}]'))
				try:
					setattr(obj, '_id', self._gen_id())
				except Exception:
					pass
				lst.append(obj)
		return out

	def _load_top_level_tempo(self, tempo_data: object) -> List[Tempo]:
		"""Parse top-level tempo list into Tempo dataclasses with fresh ids."""
		out: List[Tempo] = []
		items = tempo_data if isinstance(tempo_data, list) else []
		for idx, item in enumerate(items):
			incoming = item if isinstance(item, dict) else {}
			obj = Tempo(**_merge_with_defaults(Tempo, incoming, f'tempo[{idx}]'))
			try:
				setattr(obj, '_id', self._gen_id())
			except Exception:
				pass
			out.append(obj)
		return out

	def _load_staves(self, staves_data: object, fallback_events: Events) -> List[Stave]:
		"""Load staves from dict data and always provide DEFAULT_STAVE_COUNT staves."""
		staves: List[Stave] = []
		if isinstance(staves_data, list) and staves_data:
			for i, raw in enumerate(staves_data):
				incoming = raw if isinstance(raw, dict) else {}
				name = str(incoming.get('name', f'Stave {i + 1}') or f'Stave {i + 1}')
				try:
					scale = float(incoming.get('scale', 1.0) or 1.0)
				except Exception:
					scale = 1.0
				enabled = bool(incoming.get('enabled', True))
				ev_raw = incoming.get('events', None)
				if isinstance(ev_raw, dict):
					ev_obj = self._load_events_container(ev_raw, f'staves[{i}].events')
				elif i == 0:
					ev_obj = deepcopy(fallback_events)
				else:
					ev_obj = Events()
				staves.append(Stave(name=name, scale=scale, enabled=enabled, events=ev_obj))
		if not staves:
			staves = [Stave(name='Stave 1', enabled=True, events=deepcopy(fallback_events))]
		# Always expose 4 editable staves in the model for editor/UI consistency.
		staves = list(staves[:DEFAULT_STAVE_COUNT])
		for i in range(len(staves), DEFAULT_STAVE_COUNT):
			staves.append(Stave(name=f'Stave {i + 1}', enabled=True, events=Events()))
		for i, st in enumerate(staves):
			if not str(getattr(st, 'name', '') or '').strip():
				st.name = f'Stave {i + 1}'
		return staves

	def _sync_events_staves_legacy_bridge(self, commit_to_selected_stave: bool = False) -> None:
		"""Bridge legacy self.events and stave events without destructive load-time overwrite.

		- commit_to_selected_stave=True: write self.events into selected stave (save path)
		- commit_to_selected_stave=False: read selected stave into self.events (load/new path)
		"""
		if not isinstance(getattr(self, 'staves', None), list) or not self.staves:
			self.staves = [Stave(name=f'Stave {i + 1}', enabled=True) for i in range(DEFAULT_STAVE_COUNT)]
		for i, st in enumerate(list(self.staves)):
			if not isinstance(st, Stave):
				self.staves[i] = Stave(name=f'Stave {i + 1}', enabled=True)
		idx = 0
		try:
			app_state = getattr(self, 'app_state', None)
			raw_idx = int(getattr(app_state, 'selected_stave_index', 0) or 0)
			idx = int(raw_idx % max(1, len(self.staves)))
		except Exception:
			idx = 0

		if commit_to_selected_stave:
			# Save path: commit working legacy container into selected stave.
			if isinstance(getattr(self, 'events', None), Events):
				# Tempo is system/global and no longer persisted per-stave.
				self.events.tempo = []
				op_load = Operator(float(SHORTEST_DURATION))
				lb_list = list(getattr(self.events, 'line_break', []) or [])
				if not lb_list:
					self.events.line_break = [LineBreak(time=0.0)]
				elif not any(op_load.eq(float(getattr(lb, 'time', 0.0) or 0.0), 0.0) for lb in lb_list):
					lb_list.insert(0, LineBreak(time=0.0))
					lb_list.sort(key=lambda lb: float(getattr(lb, 'time', 0.0) or 0.0))
					self.events.line_break = lb_list
			self.staves[idx].events = deepcopy(self.events)
		else:
			# Load/new path: never overwrite stave data from legacy container.
			st_events = getattr(self.staves[idx], 'events', None)
			if isinstance(st_events, Events):
				self.events = deepcopy(st_events)
				self.events.tempo = []
			elif not isinstance(getattr(self, 'events', None), Events):
				self.events = Events()
			else:
				self.events.tempo = []
		# Keep a non-empty legacy fallback stave at index 0.
		if len(self.staves) > 1 and not isinstance(getattr(self.staves[0], 'events', None), Events):
			self.staves[0].events = Events()

	def selected_stave_index(self) -> int:
		"""Return the normalized selected stave index from app_state."""
		staves = list(getattr(self, 'staves', []) or [])
		if not staves:
			return 0
		try:
			raw_idx = int(getattr(getattr(self, 'app_state', None), 'selected_stave_index', 0) or 0)
		except Exception:
			raw_idx = 0
		return int(raw_idx % len(staves))

	def stave_at(self, stave_index: int | None = None) -> Stave | None:
		staves = list(getattr(self, 'staves', []) or [])
		if not staves:
			return None
		idx = self.selected_stave_index() if stave_index is None else int(stave_index)
		return staves[int(idx % len(staves))]

	def stave_events_at(self, stave_index: int | None = None) -> Events | None:
		stave = self.stave_at(stave_index)
		if stave is None:
			return None
		return getattr(stave, 'events', None)

	def line_breaks_at(self, stave_index: int | None = None) -> list[LineBreak]:
		events = self.stave_events_at(stave_index)
		if events is None:
			return []
		return list(getattr(events, 'line_break', []) or [])

	def sync_linked_line_breaks(self, source_stave_index: int | None = None) -> None:
		"""Propagate line-break time/page_break across all staves while keeping per-stave margins/ranges."""
		def _clone_range_value(value):
			if value == 'auto' or value is None:
				return 'auto'
			try:
				return list(value)
			except Exception:
				return 'auto'

		staves = list(getattr(self, 'staves', []) or [])
		if not staves:
			return
		source_idx = self.selected_stave_index() if source_stave_index is None else int(source_stave_index) % len(staves)
		source_events = self.stave_events_at(source_idx)
		if source_events is None:
			return
		source_breaks = list(getattr(source_events, 'line_break', []) or [])
		source_breaks.sort(key=lambda lb: float(getattr(lb, 'time', 0.0) or 0.0))
		if not source_breaks:
			source_breaks = [LineBreak(time=0.0)]
			setattr(source_events, 'line_break', source_breaks)
		for stave in staves:
			events = getattr(stave, 'events', None)
			if events is None:
				continue
			current = list(getattr(events, 'line_break', []) or [])
			current.sort(key=lambda lb: float(getattr(lb, 'time', 0.0) or 0.0))
			while len(current) < len(source_breaks):
				idx = len(current)
				src = source_breaks[idx]
				current.append(LineBreak(
					time=float(getattr(src, 'time', 0.0) or 0.0),
					margin_mm=list(getattr(src, 'margin_mm', [5.0, 5.0]) or [5.0, 5.0]),
					stave_range=_clone_range_value(getattr(src, 'stave_range', 'auto')),
					page_break=bool(getattr(src, 'page_break', False)),
				))
			if len(current) > len(source_breaks):
				current = current[:len(source_breaks)]
			for idx, src in enumerate(source_breaks):
				tgt = current[idx]
				tgt.time = float(getattr(src, 'time', 0.0) or 0.0)
				tgt.page_break = bool(getattr(src, 'page_break', False))
			if current and not any(self._time_eq_lb(lb, 0.0) for lb in current):
				current.insert(0, LineBreak(time=0.0))
			events.line_break = current

	def _time_eq_lb(self, lb: LineBreak, t: float) -> bool:
		try:
			return Operator(float(SHORTEST_DURATION)).eq(float(getattr(lb, 'time', 0.0) or 0.0), float(t))
		except Exception:
			return float(getattr(lb, 'time', 0.0) or 0.0) == float(t)

	def _convert_imported_piano_to_keytab(self) -> None:
		"""Normalize imported legacy .piano data into keyTAB stave-first schema."""
		staves = list(getattr(self, 'staves', []) or [])
		if not staves:
			staves = [Stave(name='Piano', scale=1.0, enabled=True, events=deepcopy(self.events))]
		staves = list(staves[:DEFAULT_STAVE_COUNT])
		for i in range(len(staves), DEFAULT_STAVE_COUNT):
			staves.append(Stave(name=f'Stave {i + 1}', scale=1.0, enabled=False, events=Events()))

		first = staves[0]
		first.name = 'Piano'
		first.scale = 1.0
		first.enabled = True
		if not isinstance(getattr(first, 'events', None), Events):
			first.events = deepcopy(self.events)

		for i in range(1, len(staves)):
			st = staves[i]
			st.scale = 1.0
			st.enabled = False
			if not isinstance(getattr(st, 'events', None), Events):
				st.events = Events()

		self.staves = staves
		self.events = deepcopy(first.events)
		self.meta_data.extension = '.keytab'
		self.meta_data.description = 'This is a .keytab score file created with keyTAB.'
		try:
			self.app_state.selected_stave_index = 0
		except Exception:
			pass

	# ---- Persistence ----
	def save(self, path: str) -> None:
		try:
			if callable(self._before_save_hook):
				self._before_save_hook(self)
		except Exception:
			pass
		# Update modification timestamp before writing
		self.meta_data.modification_timestamp = _timestamp_now()
		self.meta_data.extension = '.keytab'
		self._sync_events_staves_legacy_bridge(commit_to_selected_stave=True)
		payload = self.get_dict()
		if isinstance(payload, dict):
			payload.pop('editor', None)
			# New schema: only per-stave event containers are persisted.
			payload.pop('events', None)
		target_path = str(Path(path).with_suffix('.keytab')) if str(path or '').lower().endswith('.piano') else str(path)
		if str(Path(target_path).suffix or '').lower() != '.keytab':
			target_path = str(Path(target_path).with_suffix('.keytab'))
		with open(target_path, 'w', encoding='utf-8') as f:
			# Store non-ASCII symbols (e.g., dynamic glyphs) as \uXXXX escapes for stable/plain-text readability.
			json.dump(payload, f, indent=4, ensure_ascii=True, separators=(',', ':'))

	def load(self, path: str) -> "SCORE":
		with open(path, 'r', encoding='utf-8') as f:
			data = json.load(f)
		is_piano_import = str(path or '').lower().endswith('.piano')

		# Migrate legacy file conventions (fail-open).
		data = _apply_legacy_conversion(data)

		# Meta/Info
		md = data.get('meta_data', {})
		self.meta_data = MetaData(**_merge_with_defaults(MetaData, md, 'meta_data'))
		info_data = data.get('info', {})
		self.info = Info(**_merge_with_defaults(Info, info_data, 'info'))
		analysis_data = data.get('analysis', {}) or {}
		self.analysis = Analysis(**_merge_with_defaults(Analysis, analysis_data, 'analysis'))
		# Base grid: at least one
		bg_list = data.get('base_grid', [])
		if isinstance(bg_list, list) and bg_list:
			self.base_grid = [
				BaseGrid(**_merge_with_defaults(BaseGrid, item if isinstance(item, dict) else {}, f'base_grid[{i}]'))
				for i, item in enumerate(bg_list)
			]
		else:
			self.base_grid = [BaseGrid(**_merge_with_defaults(BaseGrid, {}, 'base_grid[0]'))]
		# Layout: simple dataclass-merge with defaults, no legacy migration
		lay = data.get('layout', {}) or {}
		self.layout = Layout(**_merge_with_defaults(Layout, lay, 'layout'))

		# App state (optional)
		app = data.get('app_state', None)
		if isinstance(app, dict):
			self.app_state = AppState(**_merge_with_defaults(AppState, app, 'app_state'))
			self._app_state_from_file = True
		else:
			self.app_state = AppState()
			self._app_state_from_file = False

		# Events lists
		ev = data.get('events', {}) or {}
		self._next_id = 1
		self.tempo = self._load_top_level_tempo(data.get('tempo', []))
		self.events = self._load_events_container(ev, 'events')
		self.events.tempo = []
		self.staves = self._load_staves(data.get('staves', None), self.events)
		if not isinstance(data.get('events', None), dict):
			self.events = deepcopy(self.staves[0].events)
			self.events.tempo = []

		# Normalize hand values and convert short notes to grace notes.
		try:
			self._normalize_events_after_load()
		except Exception:
			pass


		# Ensure an initial tempo marker exists at time 0
		try:
			if not getattr(self, 'tempo', None):
				self.tempo = []
			# Determine a reasonable default duration: one beat of the first base grid
			numer = int(getattr(self.base_grid[0], 'numerator', 4) or 4) if self.base_grid else 4
			denom = int(getattr(self.base_grid[0], 'denominator', 4) or 4) if self.base_grid else 4
			measure_len = float(numer) * (4.0 / float(denom)) * float(QUARTER_NOTE_UNIT)
			beat_len = measure_len / max(1, int(numer))
			# Check if any tempo at time 0 exists
			op_load = Operator(float(SHORTEST_DURATION))
			at_zero = any(op_load.eq(float(getattr(tp, 'time', 0.0) or 0.0), 0.0) for tp in self.tempo)
			if not at_zero:
				self.new_tempo(time=0.0, duration=float(beat_len))
		except Exception:
			pass

		# Ensure a line break exists at time 0
		self._ensure_line_break_zero()
		if is_piano_import:
			self._convert_imported_piano_to_keytab()
		self._sync_events_staves_legacy_bridge()
		return self

	@classmethod
	def from_dict(cls, data: dict) -> "SCORE":
		"""Construct a SCORE from its dict representation (like load, but in-memory)."""
		# Keep in-memory construction aligned with file load conversion.
		data = _apply_legacy_conversion(data)

		self = cls()

		# Meta/Info
		md = (data or {}).get('meta_data', {})
		self.meta_data = MetaData(**_merge_with_defaults(MetaData, md, 'meta_data'))
		info_data = (data or {}).get('info', {})
		self.info = Info(**_merge_with_defaults(Info, info_data, 'info'))
		analysis_data = (data or {}).get('analysis', {}) or {}
		self.analysis = Analysis(**_merge_with_defaults(Analysis, analysis_data, 'analysis'))

		# Base grid
		bg_list = (data or {}).get('base_grid', [])
		if isinstance(bg_list, list) and bg_list:
			self.base_grid = [
				BaseGrid(**_merge_with_defaults(BaseGrid, item if isinstance(item, dict) else {}, f'base_grid[{i}]'))
				for i, item in enumerate(bg_list)
			]
		else:
			self.base_grid = [BaseGrid(**_merge_with_defaults(BaseGrid, {}, 'base_grid[0]'))]

		# Layout
		lay = (data or {}).get('layout', {}) or {}
		self.layout = Layout(**_merge_with_defaults(Layout, lay, 'layout'))

		# App state
		app = (data or {}).get('app_state', None)
		if isinstance(app, dict):
			self.app_state = AppState(**_merge_with_defaults(AppState, app, 'app_state'))
			self._app_state_from_file = True
		else:
			self.app_state = AppState()
			self._app_state_from_file = False

		# Events
		ev = (data or {}).get('events', {}) or {}
		self._next_id = 1
		self.tempo = self._load_top_level_tempo((data or {}).get('tempo', []))
		self.events = self._load_events_container(ev, 'events')
		self.events.tempo = []
		self.staves = self._load_staves((data or {}).get('staves', None), self.events)
		if not isinstance((data or {}).get('events', None), dict):
			self.events = deepcopy(self.staves[0].events)
			self.events.tempo = []

		# Normalize hand values and convert short notes to grace notes.
		try:
			self._normalize_events_after_load()
		except Exception:
			pass

		# Ensure a line break exists at time 0
		try:
			self._ensure_line_break_zero()
		except Exception:
			pass

		self._sync_events_staves_legacy_bridge()

		return self

	# ---- New minimal template ----
	def new(self) -> "SCORE":
		self.meta_data = MetaData()
		# Set creation timestamp using user-configurable timestamp format.
		self.meta_data.creation_timestamp = _timestamp_now()
		self.info = Info()
		self.analysis = Analysis()
		self.info.copyright = f"© keyTAB {datetime.now().year}"
		self.base_grid = [BaseGrid()]
		self.tempo = []
		self.events = Events()
		self.events.tempo = []
		self.staves = [Stave(name=f'Stave {i + 1}', enabled=True if i == 0 else False, events=Events()) for i in range(DEFAULT_STAVE_COUNT)]
		self.staves[0].events = deepcopy(self.events)
		self.layout = Layout()
		self.app_state = AppState()
		self._next_id = 1
		self._app_state_from_file = False
		self._last_load_checks_report = {}
		self._ensure_line_break_zero()
		# Add an initial tempo at time 0 for a default 4/4 beat length
		numer = int(getattr(self.base_grid[0], 'numerator', 4) or 4) if self.base_grid else 4
		denom = int(getattr(self.base_grid[0], 'denominator', 4) or 4) if self.base_grid else 4
		measure_len = float(numer) * (4.0 / float(denom)) * float(QUARTER_NOTE_UNIT)
		beat_len = measure_len / max(1, int(numer))
		self.new_tempo(time=0.0, duration=float(beat_len))
		self._sync_events_staves_legacy_bridge()
		return self

	def _ensure_line_break_zero(self) -> None:
		"""Ensure every stave has a line break at time 0 and keep linked breaks aligned."""
		staves = list(getattr(self, 'staves', []) or [])
		op_load = Operator(float(SHORTEST_DURATION))

		if not staves:
			if getattr(self, 'events', None) is not None:
				lb_list = list(getattr(self.events, 'line_break', []) or [])
				if not lb_list:
					self.events.line_break = [LineBreak(time=0.0)]
				elif not any(op_load.eq(float(getattr(lb, 'time', 0.0) or 0.0), 0.0) for lb in lb_list):
					lb_list.insert(0, LineBreak(time=0.0))
					lb_list.sort(key=lambda lb: float(getattr(lb, 'time', 0.0) or 0.0))
					self.events.line_break = lb_list
			return

		for stave in staves:
			events = getattr(stave, 'events', None)
			if events is None:
				continue
			lb_list = list(getattr(events, 'line_break', []) or [])
			if not lb_list:
				events.line_break = [LineBreak(time=0.0)]
				continue
			if not any(op_load.eq(float(getattr(lb, 'time', 0.0) or 0.0), 0.0) for lb in lb_list):
				lb_list.insert(0, LineBreak(time=0.0))
			lb_list.sort(key=lambda lb: float(getattr(lb, 'time', 0.0) or 0.0))
			events.line_break = lb_list

		# Use stave 0 as the canonical source to avoid skipping it when selected stave differs.
		self.sync_linked_line_breaks(source_stave_index=0)

	def _normalize_events_after_load(self) -> None:
		"""Normalize event fields after parsing and convert short notes to grace notes."""
		converted_grace: List[GraceNote] = []
		remaining_notes: List[Note] = []
		op_load = Operator(float(SHORTEST_DURATION))
		for n in getattr(self.events, 'note', []) or []:
			# hand
			h = str(getattr(n, 'hand', 'l') or 'l').strip()
			if h not in ('l', 'r'):
				h = 'l'
			setattr(n, 'hand', h)
			
			# accidental
			acc = int(getattr(n, 'acc', 0) or 0)
			acc = int(max(-2, min(2, acc)))
			setattr(n, 'acc', acc)
			
			# color
			c = str(getattr(n, 'color', '') or '').strip()
			if not c:
				setattr(n, 'color', 'auto')
			
			# duration -> grace note conversion
			dur = float(getattr(n, 'duration', 0.0) or 0.0)
			if dur < float(GRACENOTE_THRESHOLD):
				converted_grace.append(
					GraceNote(
						pitch=int(getattr(n, 'pitch', 40) or 40),
						time=float(getattr(n, 'time', 0.0) or 0.0),
						notehead=str(getattr(n, 'notehead', 'auto') or 'auto'),
					)
				)
			else:
				remaining_notes.append(n)

		# Replace original notes with remaining valid notes and add converted grace notes.
		# De-duplicate notes that start at effectively the same time (with a small load-time threshold),
		# share the same pitch, and are in the same hand. Keep the shortest duration note among duplicates.
		remaining_notes.sort(
			key=lambda n: (
				int(getattr(n, 'pitch', 0) or 0),
				float(getattr(n, 'time', 0.0) or 0.0),
				str(getattr(n, 'hand', 'l') or 'l'),
				float(getattr(n, 'duration', 0.0) or 0.0),
			)
		)
		deduped_notes: List[Note] = []
		for n in remaining_notes:
			if not deduped_notes:
				deduped_notes.append(n)
				continue
			prev = deduped_notes[-1]
			prev_pitch = int(getattr(prev, 'pitch', 0) or 0)
			prev_time = float(getattr(prev, 'time', 0.0) or 0.0)
			prev_hand = str(getattr(prev, 'hand', 'l') or 'l')
			cur_pitch = int(getattr(n, 'pitch', 0) or 0)
			cur_time = float(getattr(n, 'time', 0.0) or 0.0)
			cur_hand = str(getattr(n, 'hand', 'l') or 'l')
			if cur_pitch == prev_pitch and op_load.eq(cur_time, prev_time) and cur_hand == prev_hand:
				prev_dur = float(getattr(prev, 'duration', 0.0) or 0.0)
				cur_dur = float(getattr(n, 'duration', 0.0) or 0.0)
				if cur_dur < prev_dur:
					deduped_notes[-1] = n
				continue
			deduped_notes.append(n)
		deduped_removed = max(0, len(remaining_notes) - len(deduped_notes))

		# Prevent overlapping same-pitch notes by shortening each note to the
		# first later same-pitch note start that falls inside its duration window.
		by_pitch: dict[int, List[Note]] = {}
		for n in deduped_notes:
			pitch = int(getattr(n, 'pitch', 0) or 0)
			by_pitch.setdefault(pitch, []).append(n)
		shortened_overlaps = 0
		for pitch_notes in by_pitch.values():
			pitch_notes.sort(key=lambda n: float(getattr(n, 'time', 0.0) or 0.0))
			for i, n in enumerate(pitch_notes):
				start_t = float(getattr(n, 'time', 0.0) or 0.0)
				duration = float(getattr(n, 'duration', 0.0) or 0.0)
				if duration <= 0.0:
					continue
				end_t = float(start_t + duration)
				overlap_start = None
				for other in pitch_notes[i + 1:]:
					other_start = float(getattr(other, 'time', 0.0) or 0.0)
					if other_start <= start_t:
						continue
					if other_start >= end_t:
						break
					overlap_start = other_start
					break
				if overlap_start is not None:
					new_duration = float(max(0.0, overlap_start - start_t))
					if new_duration < duration:
						shortened_overlaps += 1
					setattr(n, 'duration', new_duration)

		self.events.note = deduped_notes
		for g in converted_grace:
			self.new_grace_note(pitch=int(g.pitch), time=float(g.time), notehead=str(getattr(g, 'notehead', 'auto') or 'auto'))

		# De-duplicate grace notes that share effectively the same time and exact pitch.
		grace_items = list(getattr(self.events, 'grace_note', []) or [])
		grace_items.sort(
			key=lambda g: (
				int(getattr(g, 'pitch', 0) or 0),
				float(getattr(g, 'time', 0.0) or 0.0),
			)
		)
		deduped_grace: List[GraceNote] = []
		for g in grace_items:
			if not deduped_grace:
				deduped_grace.append(g)
				continue
			prev = deduped_grace[-1]
			prev_pitch = int(getattr(prev, 'pitch', 0) or 0)
			prev_time = float(getattr(prev, 'time', 0.0) or 0.0)
			cur_pitch = int(getattr(g, 'pitch', 0) or 0)
			cur_time = float(getattr(g, 'time', 0.0) or 0.0)
			if cur_pitch == prev_pitch and op_load.eq(cur_time, prev_time):
				continue
			deduped_grace.append(g)
		grace_deduped_removed = max(0, len(grace_items) - len(deduped_grace))
		self.events.grace_note = deduped_grace

		self._last_load_checks_report = {
			'deduped_removed': int(deduped_removed),
			'shortened_overlaps': int(shortened_overlaps),
			'converted_to_grace': int(len(converted_grace)),
			'grace_deduped_removed': int(grace_deduped_removed),
		}
	
		# Normalize beam hand values as well.
		for b in getattr(self.events, 'beam', []) or []:
			h = str(getattr(b, 'hand', 'l') or 'l').strip()
			if h not in ('l', 'r'):
				h = 'l'
			setattr(b, 'hand', h)

	def get_load_checks_report(self) -> dict:
		report = getattr(self, '_last_load_checks_report', None)
		if isinstance(report, dict):
			return dict(report)
		return {}

	def apply_quick_line_breaks(self, groups: List[int]) -> bool:
		"""Distribute line breaks in repeating measure groups across the score.

		- Uses existing line break margin/stave_range as a template when available.
		- Always inserts a line break at time 0, then repeats the provided group sizes.
		- Carries forward page/line type, margins, and ranges from existing line breaks
		  in order; if there are fewer existing breaks than needed, the last known
		  value is reused.
		"""
		try:
			group_list = [int(g) for g in groups if int(g) > 0]
		except Exception:
			group_list = []
		if not group_list:
			return False

		# Build absolute measure start times from the base grid
		starts: List[float] = [0.0]
		cursor = 0.0
		for bg in list(getattr(self, 'base_grid', []) or []):
			numer = int(getattr(bg, 'numerator', 4) or 4)
			denom = int(getattr(bg, 'denominator', 4) or 4)
			measures = int(getattr(bg, 'measure_amount', 1) or 1)
			if measures <= 0:
				continue
			measure_len = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
			for _ in range(measures):
				cursor += measure_len
				starts.append(float(cursor))
		if len(starts) < 2:
			return False

		events = self.stave_events_at()
		if events is None:
			return False

		# Preserve styling from active-stave line breaks in order; reuse last when exhausted
		existing = sorted(list(getattr(events, 'line_break', []) or []), key=lambda lb: float(getattr(lb, 'time', 0.0) or 0.0))
		defaults = LineBreak()

		def _template_for(idx: int) -> tuple[list[float], list[int] | Literal['auto'] | bool, bool]:
			tmpl = existing[idx] if idx < len(existing) else (existing[-1] if existing else None)
			margin_mm = list(getattr(tmpl, 'margin_mm', defaults.margin_mm) or defaults.margin_mm) if tmpl else list(defaults.margin_mm)
			tmpl_range = getattr(tmpl, 'stave_range', defaults.stave_range) if tmpl else defaults.stave_range
			if tmpl_range == 'auto' or tmpl_range is True:
				stave_range: list[int] | Literal['auto'] | bool = 'auto'
			else:
				fallback = 'auto' if defaults.stave_range == 'auto' else list(defaults.stave_range or [0, 0])
				stave_range = list(tmpl_range or fallback)
			page_break = bool(getattr(tmpl, 'page_break', False)) if tmpl else False
			return (margin_mm, stave_range, page_break)

		# Clear and rebuild active-stave line breaks following the requested grouping
		events.line_break = []
		total_measures = len(starts) - 1
		index = 0
		group_idx = 0
		tmpl_idx = 0
		last_group = int(group_list[-1])
		while index < total_measures:
			margin_mm, stave_range, page_break = _template_for(tmpl_idx)
			if index == 0:
				self.new_line_break(time=0.0, margin_mm=margin_mm, stave_range=stave_range, page_break=page_break)
			else:
				self.new_line_break(time=float(starts[index]), margin_mm=margin_mm, stave_range=stave_range, page_break=page_break)
			if tmpl_idx < len(existing) - 1:
				tmpl_idx += 1
			if group_idx < len(group_list):
				group_len = int(group_list[group_idx])
				group_idx += 1
			else:
				group_len = last_group
			if group_len <= 0:
				break
			index += group_len
		events.line_break.sort(key=lambda lb: float(getattr(lb, 'time', 0.0) or 0.0))
		self.sync_linked_line_breaks()
		return True

