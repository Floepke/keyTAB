

# my json structure design for *.piano files.
from __future__ import annotations
from dataclasses import dataclass, field, fields, MISSING
from typing import List, get_args, get_origin, get_type_hints, Literal
import json
from datetime import datetime

from file_model.events.note import Note
from file_model.events.grace_note import GraceNote
from file_model.events.pedal import Pedal
from file_model.events.text import Text
from copy import deepcopy
from file_model.events.slur import Slur
from file_model.events.beam import Beam
from file_model.events.start_repeat import StartRepeat
from file_model.events.end_repeat import EndRepeat
from file_model.events.count_line import CountLine
from file_model.events.line_break import LineBreak
from file_model.events.tempo import Tempo
from file_model.layout import Layout, LayoutFont
from file_model.info import Info
from file_model.analysis import Analysis
from utils.CONSTANT import GRACENOTE_THRESHOLD, QUARTER_NOTE_UNIT
from file_model.base_grid import BaseGrid
from file_model.appstate import AppState


@dataclass
class EditorSettings:
	"""Editor-specific settings for the piano-roll style editor.

	- zoom_mm_per_quarter: how many millimeters a quarter note occupies vertically.
	"""
	zoom_mm_per_quarter: float = 25.0



@dataclass
class MetaData:
	description: str = 'This is a .piano score file created with keyTAB.'
	version: int = 1
	extension: str = '.piano'
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
	start_repeat: List[StartRepeat] = field(default_factory=list)
	end_repeat: List[EndRepeat] = field(default_factory=list)
	count_line: List[CountLine] = field(default_factory=list)
	line_break: List[LineBreak] = field(default_factory=list)
	tempo: List[Tempo] = field(default_factory=list)


@dataclass
class SCORE:
	meta_data: MetaData = field(default_factory=MetaData)
	info: Info = field(default_factory=Info)
	analysis: Analysis = field(default_factory=Analysis)
	base_grid: List[BaseGrid] = field(default_factory=list)
	events: Events = field(default_factory=Events)
	layout: Layout = field(default_factory=Layout)
	editor: EditorSettings = field(default_factory=EditorSettings)
	app_state: AppState = field(default_factory=AppState)
	_next_id: int = 1
	_app_state_from_file: bool = False

	# ---- Builders (ensure unique _id) ----
	def _gen_id(self) -> int:
		i = self._next_id
		self._next_id += 1
		return i

	def new_note(self, **kwargs) -> Note:
		base = {'pitch': 40, 'time': 0.0, 'duration': 100.0, 'hand': '<'}
		base.update(kwargs)
		h = str(base.get('hand', '<') or '<').strip()
		if h.lower() == 'l':
			h = '<'
		elif h.lower() == 'r':
			h = '>'
		elif h not in ('<', '>'):
			h = '<'
		base['hand'] = h
		base['color'] = h
		obj = Note(**base, _id=self._gen_id())
		self.events.note.append(obj)
		return obj

	def new_grace_note(self, **kwargs) -> GraceNote:
		base = {'pitch': 41, 'time': 50.0}
		base.update(kwargs)
		obj = GraceNote(**base, _id=self._gen_id())
		self.events.grace_note.append(obj)
		return obj

	def new_pedal(self, **kwargs) -> Pedal:
		base = {'type': 'v', 'time': 0.0}
		base.update(kwargs)
		obj = Pedal(**base, _id=self._gen_id())
		self.events.pedal.append(obj)
		return obj

	def new_text(self, **kwargs) -> Text:
		# Text anchor is center; store x as semitone offset and rotation in degrees.
		# Default font clones the score's layout font_text to avoid shared mutation.
		default_font = deepcopy(getattr(self.layout, 'font_text', LayoutFont()))
		base = {
			'text': 'Text',
			'time': 0.0,
			'x_rpitch': 0,
			'rotation': 0.0,
			'x_offset_mm': 0.0,
			'y_offset_mm': 0.0,
			'font': default_font,
			'use_custom_font': False,
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
		base = {'time': 0.0, 'duration': 100.0, 'hand': '<'}
		base.update(kwargs)
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
		self.events.line_break.append(obj)
		return obj

	def new_tempo(self, **kwargs) -> Tempo:
		base = {'time': 0.0, 'duration': 0.0, 'tempo': 60}
		base.update(kwargs)
		obj = Tempo(**base, _id=self._gen_id())
		self.events.tempo.append(obj)
		return obj

	# ---- Dict conversion ----
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

	# ---- Persistence ----
	def save(self, path: str) -> None:
		# Update modification timestamp before writing
		self.meta_data.modification_timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
		with open(path, 'w', encoding='utf-8') as f:
			json.dump(self.get_dict(), f, indent=None, ensure_ascii=False, separators=(',', ':'))

	def load(self, path: str) -> "SCORE":
		with open(path, 'r', encoding='utf-8') as f:
			data = json.load(f)

		# Helper: compute dataclass defaults (respecting default_factory)
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

		# Helper: merge incoming dict with dataclass defaults and report repairs
		def _merge_with_defaults(dc_type, incoming: dict, context: str, skip_keys: set = {'id', '_id'}) -> dict:
			from dataclasses import is_dataclass

			incoming = incoming or {}
			if not isinstance(incoming, dict):
				incoming = {}
			if dc_type is Note:
				incoming = dict(incoming)
				h = str(incoming.get('hand', '<') or '<').strip()
				if h.lower() == 'l':
					h = '<'
				elif h.lower() == 'r':
					h = '>'
				elif h not in ('<', '>'):
					h = '<'
				incoming['hand'] = h
				incoming['color'] = h
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

		# Editor settings (optional)
		ed = data.get('editor', {}) or {}
		self.editor = EditorSettings(**_merge_with_defaults(EditorSettings, ed, 'editor'))

		# App state (optional)
		app = data.get('app_state', None)
		if isinstance(app, dict):
			self.app_state = AppState(**_merge_with_defaults(AppState, app, 'app_state'))
			self._app_state_from_file = True
		else:
			self.app_state = AppState()
			self._app_state_from_file = False

		# Events lists: generic loader based on Events dataclass field types
		ev = data.get('events', {}) or {}
		self.events = Events()
		self._next_id = 1
		# Resolve postponed annotations (from __future__ import annotations)
		_ev_hints = {}
		try:
			_ev_hints = get_type_hints(Events, globals(), locals())
		except Exception:
			_ev_hints = {}
		for f_ev in fields(Events):
			# Expect typing like List[Note]; resolve element type from hints
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
			lst = getattr(self.events, name)
			for idx, item in enumerate(items):
				incoming = item if isinstance(item, dict) else {}
				obj = elem_type(**_merge_with_defaults(elem_type, incoming, f'events.{name}[{idx}]'))
				# Assign sequential _id regardless of incoming value
				try:
					setattr(obj, '_id', self._gen_id())
				except Exception:
					pass
				lst.append(obj)

		# Normalize hand values across events to '<' or '>' (repair legacy 'l'/'r')
		try:
			# Convert very short notes to grace notes and normalize hand values
			converted_grace: List[GraceNote] = []
			remaining_notes: List[Note] = []
			for n in getattr(self.events, 'note', []) or []:
				h = str(getattr(n, 'hand', '<') or '<').strip()
				if h.lower() == 'l':
					setattr(n, 'hand', '<')
				elif h.lower() == 'r':
					setattr(n, 'hand', '>')
				elif h not in ('<', '>'):
					setattr(n, 'hand', '<')
				setattr(n, 'color', str(getattr(n, 'hand', '<') or '<'))
				# Convert to grace note if shorter than threshold
				try:
					du = float(getattr(n, 'duration', 0.0) or 0.0)
					if du < float(GRACENOTE_THRESHOLD):
						converted_grace.append(GraceNote(pitch=int(getattr(n, 'pitch', 40) or 40), time=float(getattr(n, 'time', 0.0) or 0.0)))
					else:
						remaining_notes.append(n)
				except Exception:
					remaining_notes.append(n)
			# Replace lists: keep remaining notes, append converted grace notes via builder to assign ids
			self.events.note = remaining_notes
			for g in converted_grace:
				self.new_grace_note(pitch=int(g.pitch), time=float(g.time))
			for b in getattr(self.events, 'beam', []) or []:
				h = str(getattr(b, 'hand', '<') or '<').strip()
				if h.lower() == 'l':
					setattr(b, 'hand', '<')
				elif h.lower() == 'r':
					setattr(b, 'hand', '>')
				elif h not in ('<', '>'):
					setattr(b, 'hand', '<')
		except Exception:
			pass


		# Ensure an initial tempo marker exists at time 0
		try:
			if not getattr(self.events, 'tempo', None):
				self.events.tempo = []
			# Determine a reasonable default duration: one beat of the first base grid
			numer = int(getattr(self.base_grid[0], 'numerator', 4) or 4) if self.base_grid else 4
			denom = int(getattr(self.base_grid[0], 'denominator', 4) or 4) if self.base_grid else 4
			measure_len = float(numer) * (4.0 / float(denom)) * float(QUARTER_NOTE_UNIT)
			beat_len = measure_len / max(1, int(numer))
			# Check if any tempo at time 0 exists
			at_zero = any(float(getattr(tp, 'time', 0.0) or 0.0) == 0.0 for tp in self.events.tempo)
			if not at_zero:
				self.new_tempo(time=0.0, duration=float(beat_len), tempo=60)
		except Exception:
			pass

		# Ensure a line break exists at time 0
		try:
			self._ensure_line_break_zero()
		except Exception:
			pass
		return self

	@classmethod
	def from_dict(cls, data: dict) -> "SCORE":
		"""Construct a SCORE from its dict representation (like load, but in-memory)."""
		self = cls()

		from dataclasses import fields, MISSING
		from typing import List, get_args, get_origin, get_type_hints

		def _defaults_for(dc_type):
			defaults = {}
			for f in fields(dc_type):
				if f.name.startswith('_'):
					continue
				if f.default is not MISSING:
					defaults[f.name] = f.default
				elif getattr(f, 'default_factory', MISSING) is not MISSING:  # type: ignore[attr-defined]
					defaults[f.name] = f.default_factory()
				else:
					defaults[f.name] = None
			return defaults

		def _merge_with_defaults(dc_type, incoming: dict, context: str, skip_keys: set = {'id', '_id'}) -> dict:
			from dataclasses import is_dataclass

			incoming = incoming or {}
			if not isinstance(incoming, dict):
				incoming = {}
			if dc_type is Note:
				incoming = dict(incoming)
				h = str(incoming.get('hand', '<') or '<').strip()
				if h.lower() == 'l':
					h = '<'
				elif h.lower() == 'r':
					h = '>'
				elif h not in ('<', '>'):
					h = '<'
				incoming['hand'] = h
				incoming['color'] = h
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

		# Editor settings
		ed = (data or {}).get('editor', {}) or {}
		self.editor = EditorSettings(**_merge_with_defaults(EditorSettings, ed, 'editor'))

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
		self.events = Events()
		self._next_id = 1
		# Resolve postponed annotations
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
			lst = getattr(self.events, name)
			for idx, item in enumerate(items):
				incoming = item if isinstance(item, dict) else {}
				obj = elem_type(**_merge_with_defaults(elem_type, incoming, f'events.{name}[{idx}]'))
				try:
					setattr(obj, '_id', self._gen_id())
				except Exception:
					pass
				lst.append(obj)

		# Normalize hand values across events to '<' or '>' (repair legacy 'l'/'r')
		try:
			# Convert very short notes to grace notes and normalize hand values
			converted_grace: List[GraceNote] = []
			remaining_notes: List[Note] = []
			for n in getattr(self.events, 'note', []) or []:
				h = str(getattr(n, 'hand', '<') or '<').strip()
				if h.lower() == 'l':
					setattr(n, 'hand', '<')
				elif h.lower() == 'r':
					setattr(n, 'hand', '>')
				elif h not in ('<', '>'):
					setattr(n, 'hand', '<')
				setattr(n, 'color', str(getattr(n, 'hand', '<') or '<'))
				# Convert to grace note if shorter than threshold
				try:
					du = float(getattr(n, 'duration', 0.0) or 0.0)
					if du < float(GRACENOTE_THRESHOLD):
						converted_grace.append(GraceNote(pitch=int(getattr(n, 'pitch', 40) or 40), time=float(getattr(n, 'time', 0.0) or 0.0)))
					else:
						remaining_notes.append(n)
				except Exception:
					remaining_notes.append(n)
			self.events.note = remaining_notes
			for g in converted_grace:
				self.new_grace_note(pitch=int(g.pitch), time=float(g.time))
			for b in getattr(self.events, 'beam', []) or []:
				h = str(getattr(b, 'hand', '<') or '<').strip()
				if h.lower() == 'l':
					setattr(b, 'hand', '<')
				elif h.lower() == 'r':
					setattr(b, 'hand', '>')
				elif h not in ('<', '>'):
					setattr(b, 'hand', '<')
		except Exception:
			pass

		# Ensure a line break exists at time 0
		try:
			self._ensure_line_break_zero()
		except Exception:
			pass

		return self

	# ---- New minimal template ----
	def new(self) -> "SCORE":
		self.meta_data = MetaData()
		# Set creation timestamp in format dd-mm-YYYY_HH:MM:SS
		self.meta_data.creation_timestamp = datetime.now().strftime("%d-%m-%Y_%H:%M:%S")
		self.info = Info()
		self.analysis = Analysis()
		try:
			year = datetime.now().year
			self.info.copyright = f"keyTAB all copyrights reserved {year}"
		except Exception:
			pass
		self.base_grid = [BaseGrid()]
		self.events = Events()
		self.layout = Layout()
		self.editor = EditorSettings()
		self.app_state = AppState()
		self._next_id = 1
		self._app_state_from_file = False
		try:
			self._ensure_line_break_zero()
		except Exception:
			pass
		# Add an initial tempo at time 0 for a default 4/4 beat length
		try:
			numer = int(getattr(self.base_grid[0], 'numerator', 4) or 4) if self.base_grid else 4
			denom = int(getattr(self.base_grid[0], 'denominator', 4) or 4) if self.base_grid else 4
			measure_len = float(numer) * (4.0 / float(denom)) * float(QUARTER_NOTE_UNIT)
			beat_len = measure_len / max(1, int(numer))
			self.new_tempo(time=0.0, duration=float(beat_len), tempo=60)
		except Exception:
			pass
		return self

	def _ensure_line_break_zero(self) -> None:
		"""Ensure there is always a line break at time 0."""
		try:
			lb_list = list(getattr(self.events, 'line_break', []) or [])
		except Exception:
			lb_list = []
		if not lb_list:
			self.new_line_break(time=0.0)
			return
		tol = 1e-6
		if not any(abs(float(getattr(lb, 'time', 0.0) or 0.0)) <= tol for lb in lb_list):
			self.new_line_break(time=0.0)
		try:
			self.events.line_break.sort(key=lambda lb: float(getattr(lb, 'time', 0.0) or 0.0))
		except Exception:
			pass

	def apply_quick_line_breaks(self, groups: List[int]) -> bool:
		"""Distribute line breaks in repeating measure groups across the score.

		- Uses existing line break margin/stave_range as a template when available.
		- Always inserts a line break at time 0, then repeats the provided group sizes.
		- Carries forward page/line type, margins, and ranges from existing line breaks
		  in order; if there are fewer existing breaks than needed, the last known
		  values are reused.
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
			try:
				numer = int(getattr(bg, 'numerator', 4) or 4)
				denom = int(getattr(bg, 'denominator', 4) or 4)
				measures = int(getattr(bg, 'measure_amount', 1) or 1)
			except Exception:
				continue
			if measures <= 0:
				continue
			measure_len = float(numer) * (4.0 / float(max(1, denom))) * float(QUARTER_NOTE_UNIT)
			for _ in range(measures):
				cursor += measure_len
				starts.append(float(cursor))
		if len(starts) < 2:
			return False

		# Preserve styling from existing line breaks in order; reuse last when exhausted
		try:
			existing = sorted(list(getattr(self.events, 'line_break', []) or []), key=lambda lb: float(getattr(lb, 'time', 0.0) or 0.0))
		except Exception:
			existing = []
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

		# Clear and rebuild line breaks following the requested grouping
		self.events.line_break = []
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

		try:
			self.events.line_break.sort(key=lambda lb: float(getattr(lb, 'time', 0.0) or 0.0))
		except Exception:
			pass
		return True
	
	# ---- Convenience methods ----
