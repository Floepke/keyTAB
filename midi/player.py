from __future__ import annotations
import os
import sys
import threading
import time
import ctypes
import ctypes.util
from pathlib import Path
from typing import List, Optional, Tuple

import mido
import traceback

fluidsynth = None
_FLUIDSYNTH_AVAILABLE = True
_FLUIDSYNTH_IMPORT_ERROR = ""


def _mido_io_backend():
    """Prefer explicit RtMidi backend; fall back to mido default backend."""
    try:
        return mido.Backend("mido.backends.rtmidi")
    except Exception:
        return mido


def _is_fluidsynth_port(name: str) -> bool:
    """Return True when the port name refers to a FluidSynth endpoint."""
    lowered = str(name or "").lower()
    needles = (
        "fluidsynth",
        "fluid synth",
        "qsynth",
        "synth input port (qsynth",
    )
    return any(n in lowered for n in needles)


def list_midi_output_ports() -> List[str]:
    """Return available MIDI output names from mido/RtMidi."""
    try:
        backend = _mido_io_backend()
        names = list(backend.get_output_names() or [])
    except Exception:
        names = []
        traceback.print_exc()
    filtered: list[str] = []
    for n in names:
        if not str(n).strip():
            continue
        if _is_fluidsynth_port(n):
            continue
        filtered.append(str(n))
    if not filtered:
        try:
            sys.stderr.write(
                "[midi] No MIDI outputs discovered. backend=%s rtmidi=%s\n" % (
                    getattr(mido, "backend", ""),
                    getattr(sys.modules.get("rtmidi"), "__version__", "unknown"),
                )
            )
        except Exception:
            pass
    return filtered


def _try_load_cdll(path: str) -> ctypes.CDLL:
    """Load a shared library via ctypes, raising OSError with path info on failure."""
    try:
        return ctypes.CDLL(path)
    except OSError as exc:
        # PyInstaller wraps the real dlopen error with a generic message.
        # Extract the original cause for better diagnostics.
        real = exc.__cause__ if exc.__cause__ is not None else exc
        raise OSError(f"ctypes.CDLL({path!r}) failed: {real}") from exc


def _ensure_fluidsynth_lib() -> None:
    load_errors: list[str] = []

    env_lib = str(os.environ.get("PYFLUIDSYNTH_LIB", "") or "").strip()
    if env_lib:
        try:
            _try_load_cdll(env_lib)
            sys.stderr.write(f"[midi] Loaded libfluidsynth from PYFLUIDSYNTH_LIB={env_lib!r}\n")
            return
        except OSError as exc:
            load_errors.append(str(exc))
            sys.stderr.write(f"[midi] PYFLUIDSYNTH_LIB set but load failed: {exc}\n")

    appdir = str(os.environ.get("APPDIR", "") or "").strip()
    app_candidates: list[Path] = []
    if appdir:
        app_candidates.extend([
            Path(appdir) / "usr" / "lib" / "libfluidsynth.so",
            Path(appdir) / "usr" / "lib" / "libfluidsynth.so.3",
        ])

    candidates = [
        *app_candidates,
        Path("/usr/lib/x86_64-linux-gnu/libfluidsynth.so.3"),
        Path("/usr/lib/libfluidsynth.so.3"),
        Path("/lib/x86_64-linux-gnu/libfluidsynth.so.3"),
        Path("/usr/local/lib/libfluidsynth.so.3"),
        Path("/usr/lib/libfluidsynth.so"),
        Path("/usr/local/lib/libfluidsynth.so"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            _try_load_cdll(str(path))
            sys.stderr.write(f"[midi] Loaded libfluidsynth from candidate: {path}\n")
            os.environ.setdefault("PYFLUIDSYNTH_LIB", str(path))
            return
        except OSError as exc:
            load_errors.append(str(exc))
            sys.stderr.write(f"[midi] Candidate {path} exists but load failed: {exc}\n")

    found = ctypes.util.find_library("fluidsynth")
    if found:
        sys.stderr.write(f"[midi] ctypes.util.find_library('fluidsynth') returned: {found!r}\n")
        try:
            _try_load_cdll(str(found))
            os.environ.setdefault("PYFLUIDSYNTH_LIB", str(found))
            return
        except OSError as exc:
            load_errors.append(str(exc))
            sys.stderr.write(f"[midi] find_library result {found!r} failed to load: {exc}\n")
            # find_library may return just the soname; search common dirs for the actual file
            for search_dir in ["/usr/lib/x86_64-linux-gnu", "/lib/x86_64-linux-gnu", "/usr/lib", "/lib", "/usr/local/lib"]:
                full_path = Path(search_dir) / found
                if full_path.exists():
                    try:
                        _try_load_cdll(str(full_path))
                        sys.stderr.write(f"[midi] Loaded libfluidsynth from full path: {full_path}\n")
                        os.environ.setdefault("PYFLUIDSYNTH_LIB", str(full_path))
                        return
                    except OSError as exc2:
                        load_errors.append(str(exc2))
                        sys.stderr.write(f"[midi] Full path {full_path} failed to load: {exc2}\n")
    else:
        sys.stderr.write("[midi] ctypes.util.find_library('fluidsynth') returned nothing\n")

    error_detail = "\n  ".join(load_errors) if load_errors else "no candidates found"
    raise ImportError(
        "FluidSynth native library could not be loaded. "
        "Install it with 'sudo apt-get install fluidsynth libfluidsynth3' (or equivalent).\n"
        f"Load attempts:\n  {error_detail}"
    )


if sys.platform.startswith("linux"):
    _fluidsynth_init_error: Exception | None = None
    try:
        _ensure_fluidsynth_lib()
    except Exception as exc:
        _fluidsynth_init_error = exc
        sys.stderr.write(f"[midi] _ensure_fluidsynth_lib failed: {exc}\n")

    if _fluidsynth_init_error is None:
        try:
            import fluidsynth as _fluidsynth  # type: ignore
            fluidsynth = _fluidsynth
            _FLUIDSYNTH_AVAILABLE = True
            _FLUIDSYNTH_IMPORT_ERROR = ""
        except Exception as exc:
            _FLUIDSYNTH_AVAILABLE = False
            _FLUIDSYNTH_IMPORT_ERROR = f"pyfluidsynth import failed: {exc}"
            sys.stderr.write(f"[midi] {_FLUIDSYNTH_IMPORT_ERROR}\n")
    else:
        _FLUIDSYNTH_AVAILABLE = False
        _FLUIDSYNTH_IMPORT_ERROR = str(_fluidsynth_init_error)


def fluidsynth_available() -> bool:
    return bool(_FLUIDSYNTH_AVAILABLE)


def fluidsynth_unavailable_reason() -> str:
    return str(_FLUIDSYNTH_IMPORT_ERROR or "FluidSynth not available.")

from utils.CONSTANT import QUARTER_NOTE_UNIT, SHORTEST_DURATION
from utils.operator import Operator

# Threshold-aware time comparisons: treats values within one shortest-duration
# unit as equal. Prevents spurious note triggers from triplet floating-point drift.
_time_op = Operator(SHORTEST_DURATION)


class _Backend:
    def program_select(self) -> None:  # pragma: no cover - runtime side effects
        raise NotImplementedError

    def set_gain(self, gain: float) -> None:  # pragma: no cover - runtime side effects
        raise NotImplementedError

    def note_on(self, midi_note: int, velocity: int) -> None:  # pragma: no cover - runtime side effects
        raise NotImplementedError

    def note_off(self, midi_note: int) -> None:  # pragma: no cover - runtime side effects
        raise NotImplementedError

    def control_change(self, control: int, value: int) -> None:  # pragma: no cover - runtime side effects
        raise NotImplementedError

    def all_notes_off(self) -> None:  # pragma: no cover - runtime side effects
        raise NotImplementedError

    def shutdown(self) -> None:  # pragma: no cover - runtime side effects
        pass


class _FluidsynthBackend(_Backend):
    def __init__(self, soundfont_path: Optional[str]) -> None:
        if not fluidsynth_available():
            raise ImportError(fluidsynth_unavailable_reason())
        if not hasattr(fluidsynth, "Synth"):
            raise ImportError(
                "FluidSynth not available. Install it with 'sudo apt-get install fluidsynth libfluidsynth3' and ensure pyfluidsynth is installed."
            )
        self._fs: Optional[_fluidsynth.Synth] = None
        self._sfid: Optional[int] = None
        self._channel: int = 0
        self._gain: float = 0.35
        self._startup_mute_delay_sec: float = 0.15
        # Reverb settings (defaults match FluidSynth defaults)
        self._reverb_enabled: bool = True
        self._reverb_room_size: float = 0.6
        self._reverb_damp: float = 0.4
        self._reverb_width: float = 3.0
        self._reverb_level: float = 0.9
        self._soundfont_path = soundfont_path or self._autodetect_soundfont()
        if self._soundfont_path is None:
            raise RuntimeError(
                "No soundfont found. Install a GM soundfont (e.g. FluidR3_GM.sf2) or set KEYTAB_SOUNDFONT."
            )
        self._init_synth()

    def _autodetect_soundfont(self) -> Optional[str]:
        candidates: list[Path] = []
        env_sf = os.environ.get("KEYTAB_SOUNDFONT")
        if env_sf:
            candidates.append(Path(env_sf))
        candidates.append(Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"))
        for p in candidates:
            if p.expanduser().is_file():
                return str(p.expanduser())
        return None

    def _init_synth(self) -> None:
        if self._fs is not None:
            try:
                self._fs.delete()
            except Exception:
                pass
        target_gain = self._gain
        self._fs = fluidsynth.Synth()
        try:
            self._fs.setting('synth.gain', 0.0)
        except Exception:
            pass
        started = False
        for drv in ("pulseaudio", None):
            try:
                self._fs.start(driver=drv)
                started = True
                break
            except Exception:
                continue
        if not started:
            try:
                self._fs.start(driver=None)
            except Exception:
                pass
        self._sfid = self._fs.sfload(self._soundfont_path)
        self._fs.program_select(self._channel, self._sfid, 0, 0)
        self._apply_reverb_settings()
        self._fade_in_gain_after_startup(self._fs, target_gain)

    def _fade_in_gain_after_startup(self, synth: _fluidsynth.Synth, target_gain: float) -> None:
        def _worker() -> None:
            try:
                time.sleep(max(0.0, float(self._startup_mute_delay_sec)))
                steps = 6
                for step in range(1, steps + 1):
                    if self._fs is not synth:
                        return
                    synth.setting('synth.gain', float(target_gain) * (step / steps))
                    time.sleep(0.02)
            except Exception:
                try:
                    if self._fs is synth:
                        synth.setting('synth.gain', float(target_gain))
                except Exception:
                    pass

        threading.Thread(target=_worker, name="fluidsynth-startup-unmute", daemon=True).start()

    def program_select(self) -> None:
        if self._fs is not None and self._sfid is not None:
            self._fs.program_select(self._channel, self._sfid, 0, 0)

    def set_gain(self, gain: float) -> None:
        self._gain = max(0.0, float(gain))
        if self._fs is not None:
            self._fs.setting('synth.gain', self._gain)

    def _apply_reverb_settings(self) -> None:
        """Apply all reverb settings to the FluidSynth synthesizer."""
        if self._fs is None:
            return
        try:
            self._fs.setting('synth.reverb.active', 1 if self._reverb_enabled else 0)
            self._fs.setting('synth.reverb.room-size', float(self._reverb_room_size))
            self._fs.setting('synth.reverb.damp', float(self._reverb_damp))
            self._fs.setting('synth.reverb.width', float(self._reverb_width))
            self._fs.setting('synth.reverb.level', float(self._reverb_level))
        except Exception:
            # Silently ignore if reverb settings are not supported
            pass

    def set_reverb_enabled(self, enabled: bool) -> None:
        self._reverb_enabled = bool(enabled)
        self._apply_reverb_settings()

    def set_reverb_room_size(self, value: float) -> None:
        self._reverb_room_size = max(0.0, min(1.0, float(value)))
        self._apply_reverb_settings()

    def set_reverb_damp(self, value: float) -> None:
        self._reverb_damp = max(0.0, min(1.0, float(value)))
        self._apply_reverb_settings()

    def set_reverb_width(self, value: float) -> None:
        self._reverb_width = max(0.0, min(100.0, float(value)))
        self._apply_reverb_settings()

    def set_reverb_level(self, value: float) -> None:
        self._reverb_level = max(0.0, min(1.0, float(value)))
        self._apply_reverb_settings()

    def get_reverb_settings(self) -> dict:
        """Return current reverb settings as a dictionary."""
        return {
            'enabled': self._reverb_enabled,
            'room_size': self._reverb_room_size,
            'damp': self._reverb_damp,
            'width': self._reverb_width,
            'level': self._reverb_level,
        }

    def note_on(self, midi_note: int, velocity: int) -> None:
        if self._fs is not None:
            self._fs.noteon(self._channel, midi_note, velocity)

    def note_off(self, midi_note: int) -> None:
        if self._fs is not None:
            self._fs.noteoff(self._channel, midi_note)

    def control_change(self, control: int, value: int) -> None:
        if self._fs is not None:
            self._fs.cc(self._channel, int(max(0, min(127, control))), int(max(0, min(127, value))))

    def all_notes_off(self) -> None:
        if self._fs is not None:
            self._fs.all_notes_off(self._channel)
        if self._fs is not None:
            self._fs.system_reset()

    def shutdown(self) -> None:
        self.all_notes_off()
        if self._fs is not None:
            self._fs.delete()
        self._fs = None


class _MidiOutBackend(_Backend):
    """Use OS-provided wavetable synth via RtMidi (CoreMIDI/WinMM)."""

    def __init__(
        self,
        port_name: Optional[str] = None,
        *,
        require_named_port: bool = False,
        prefer_system_synth: bool = False,
    ) -> None:
        self._backend = _mido_io_backend()
        self._port = None
        self._port_name: str = ""
        names = self._list_output_names()
        open_errors: list[str] = []

        env_target = str(port_name or "").strip()
        if env_target and _is_fluidsynth_port(env_target):
            # Reject FluidSynth virtual ports to avoid crashes when switching playback modes.
            env_target = ""
        if not env_target:
            env_target = str(os.environ.get("KEYTAB_MIDI_OUT", "") or "").strip()

        if env_target:
            exact = [candidate for candidate in names if candidate == env_target]
            ci = [candidate for candidate in names if candidate.lower() == env_target.lower()]
            for candidate in (exact + ci):
                if candidate in exact and candidate in ci and exact.index(candidate) != ci.index(candidate):
                    continue
                try:
                    self._port = self._backend.open_output(candidate)
                    self._port_name = str(candidate)
                    break
                except Exception as exc:
                    open_errors.append(f"{candidate}: {exc}")
            if self._port is None and require_named_port:
                raise RuntimeError(
                    f"Requested MIDI output port not found or unavailable: {env_target}"
                )

        preferred_names = self._preferred_output_names(names)
        search_names: list[str] = list(preferred_names)

        for candidate in search_names:
            if self._port is not None:
                break
            try:
                self._port = self._backend.open_output(candidate)
                self._port_name = str(candidate)
                break
            except Exception as exc:
                open_errors.append(f"{candidate}: {exc}")

        if self._port is None:
            fallback_names = self._non_virtual_output_names(names)
            for candidate in fallback_names:
                if prefer_system_synth and preferred_names and candidate not in preferred_names:
                    continue
                try:
                    self._port = self._backend.open_output(candidate)
                    self._port_name = str(candidate)
                    break
                except Exception as exc:
                    open_errors.append(f"{candidate}: {exc}")

        if self._port is None:
            detail = ""
            if open_errors:
                detail = "\nTried outputs:\n- " + "\n- ".join(open_errors[:8])
            elif names:
                detail = "\nAvailable outputs:\n- " + "\n- ".join(str(n) for n in names[:8])
            else:
                detail = "\nNo MIDI outputs reported by RtMidi." \
                    + f" backend={getattr(mido, 'backend', '')} rtmidi={getattr(sys.modules.get('rtmidi'), '__version__', 'unknown')}" \
                    + f" PATH={os.environ.get('PATH','')}" \
                    + f" LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH','')}" \
                    + f" ALSA_CONFIG_PATH={os.environ.get('ALSA_CONFIG_PATH','')}" \
                    + f" ALSA_PLUGIN_DIR={os.environ.get('ALSA_PLUGIN_DIR','')}" \
                    + f" PYTHONPATH={os.environ.get('PYTHONPATH','')}"
            raise RuntimeError(
                "No usable MIDI output synth found. On macOS, open Audio MIDI Setup and enable an output synth endpoint (e.g. Apple DLS Synth)."
                + detail
            )

    def _list_output_names(self) -> List[str]:
        try:
            names = list(self._backend.get_output_names() or [])
        except Exception:
            names = []
        filtered: list[str] = []
        for n in names:
            if not str(n).strip():
                continue
            if _is_fluidsynth_port(n):
                continue
            filtered.append(str(n))
        return filtered

    def _preferred_output_names(self, names: List[str]) -> List[str]:
        if not names:
            return []
        lower_map = {str(n): str(n).lower() for n in names}
        priorities: list[str]
        if sys.platform == "darwin":
            priorities = [
                "apple dls synth",
                "dls synth",
                "synth",
                "software instrument",
            ]
        elif sys.platform.startswith("win"):
            priorities = [
                "microsoft gs wavetable synth",
                "gs wavetable",
                "synth",
            ]
        else:
            priorities = ["synth"]

        preferred: list[str] = []
        for needle in priorities:
            for original, lowered in lower_map.items():
                if needle in lowered and original not in preferred:
                    preferred.append(original)
        return preferred

    def _non_virtual_output_names(self, names: List[str]) -> List[str]:
        excluded_keywords = [
            "iac",
            "through",
            "network",
            "session",
            "loop",
            "bridge",
            "bus",
        ]
        filtered: list[str] = []
        for n in names:
            lowered = str(n).lower()
            if any(k in lowered for k in excluded_keywords):
                continue
            filtered.append(str(n))
        return filtered

    def output_name(self) -> str:
        return str(self._port_name)

    def program_select(self) -> None:
        self._port.send(mido.Message("program_change", program=0))

    def set_gain(self, gain: float) -> None:
        # Not supported on OS synth; ignore.
        pass

    def note_on(self, midi_note: int, velocity: int) -> None:
        self._port.send(mido.Message("note_on", note=int(midi_note), velocity=int(velocity)))

    def note_off(self, midi_note: int) -> None:
        self._port.send(mido.Message("note_off", note=int(midi_note), velocity=0))

    def control_change(self, control: int, value: int) -> None:
        self._port.send(
            mido.Message(
                "control_change",
                control=int(max(0, min(127, control))),
                value=int(max(0, min(127, value))),
            )
        )

    def all_notes_off(self) -> None:
        for n in range(128):
            self._port.send(mido.Message("note_off", note=n, velocity=0))
        self._port.send(mido.Message("control_change", control=123, value=0))

    def shutdown(self) -> None:
        self.all_notes_off()
        self._port.close()


class _MacDLSSynthBackend(_Backend):
    """Use macOS built-in Apple DLS synth via AudioToolbox/AUGraph."""

    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise RuntimeError("_MacDLSSynthBackend is only available on macOS.")

        self._graph = ctypes.c_void_p()
        self._synth_unit = ctypes.c_void_p()
        self._output_unit = ctypes.c_void_p()
        self._channel: int = 0
        self._name: str = "Apple DLS Synth"

        self._audio_toolbox = ctypes.cdll.LoadLibrary(
            "/System/Library/Frameworks/AudioToolbox.framework/AudioToolbox"
        )

        class _AudioComponentDescription(ctypes.Structure):
            _fields_ = [
                ("componentType", ctypes.c_uint32),
                ("componentSubType", ctypes.c_uint32),
                ("componentManufacturer", ctypes.c_uint32),
                ("componentFlags", ctypes.c_uint32),
                ("componentFlagsMask", ctypes.c_uint32),
            ]

        self._acd_cls = _AudioComponentDescription
        self._init_graph()

    def output_name(self) -> str:
        return str(self._name)

    def _fourcc(self, txt: str) -> int:
        b = txt.encode("ascii")
        if len(b) != 4:
            raise ValueError(f"Invalid fourcc: {txt}")
        return int.from_bytes(b, byteorder="big", signed=False)

    def _check(self, status: int, where: str) -> None:
        if int(status) != 0:
            raise RuntimeError(f"{where} failed with OSStatus {int(status)}")

    def _init_graph(self) -> None:
        at = self._audio_toolbox
        acd = self._acd_cls

        kAudioUnitType_MusicDevice = self._fourcc("aumu")
        kAudioUnitSubType_DLSSynth = self._fourcc("dls ")
        kAudioUnitType_Output = self._fourcc("auou")
        kAudioUnitSubType_DefaultOutput = self._fourcc("def ")
        kAudioUnitManufacturer_Apple = self._fourcc("appl")

        graph = ctypes.c_void_p()
        self._check(at.NewAUGraph(ctypes.byref(graph)), "NewAUGraph")

        synth_desc = acd(
            componentType=kAudioUnitType_MusicDevice,
            componentSubType=kAudioUnitSubType_DLSSynth,
            componentManufacturer=kAudioUnitManufacturer_Apple,
            componentFlags=0,
            componentFlagsMask=0,
        )
        out_desc = acd(
            componentType=kAudioUnitType_Output,
            componentSubType=kAudioUnitSubType_DefaultOutput,
            componentManufacturer=kAudioUnitManufacturer_Apple,
            componentFlags=0,
            componentFlagsMask=0,
        )

        synth_node = ctypes.c_int32(0)
        out_node = ctypes.c_int32(0)
        self._check(at.AUGraphAddNode(graph, ctypes.byref(synth_desc), ctypes.byref(synth_node)), "AUGraphAddNode(synth)")
        self._check(at.AUGraphAddNode(graph, ctypes.byref(out_desc), ctypes.byref(out_node)), "AUGraphAddNode(output)")
        self._check(at.AUGraphOpen(graph), "AUGraphOpen")

        synth_unit = ctypes.c_void_p()
        out_unit = ctypes.c_void_p()
        self._check(at.AUGraphNodeInfo(graph, synth_node, None, ctypes.byref(synth_unit)), "AUGraphNodeInfo(synth)")
        self._check(at.AUGraphNodeInfo(graph, out_node, None, ctypes.byref(out_unit)), "AUGraphNodeInfo(output)")

        self._check(at.AUGraphConnectNodeInput(graph, synth_node, 0, out_node, 0), "AUGraphConnectNodeInput")
        self._check(at.AUGraphInitialize(graph), "AUGraphInitialize")
        self._check(at.AUGraphStart(graph), "AUGraphStart")

        self._graph = graph
        self._synth_unit = synth_unit
        self._output_unit = out_unit

    def _midi(self, status: int, data1: int, data2: int) -> None:
        try:
            self._audio_toolbox.MusicDeviceMIDIEvent(
                self._synth_unit,
                ctypes.c_uint32(status),
                ctypes.c_uint32(max(0, min(127, int(data1)))),
                ctypes.c_uint32(max(0, min(127, int(data2)))),
                ctypes.c_uint32(0),
            )
        except Exception:
            pass

    def program_select(self) -> None:
        self._midi(0xC0 | self._channel, 0, 0)

    def set_gain(self, gain: float) -> None:
        pass

    def note_on(self, midi_note: int, velocity: int) -> None:
        self._midi(0x90 | self._channel, int(midi_note), int(velocity))

    def note_off(self, midi_note: int) -> None:
        self._midi(0x80 | self._channel, int(midi_note), 0)

    def control_change(self, control: int, value: int) -> None:
        self._midi(0xB0 | self._channel, int(control), int(value))

    def all_notes_off(self) -> None:
        self._midi(0xB0 | self._channel, 123, 0)

    def shutdown(self) -> None:
        self.all_notes_off()
        if bool(self._graph):
            self._audio_toolbox.AUGraphStop(self._graph)
        if bool(self._graph):
            self._audio_toolbox.DisposeAUGraph(self._graph)
        self._graph = ctypes.c_void_p()
        self._synth_unit = ctypes.c_void_p()
        self._output_unit = ctypes.c_void_p()


class _WinMMSynthBackend(_Backend):
    """Use Windows built-in Microsoft GS Wavetable synth via WinMM output."""

    def __init__(self) -> None:
        self._impl = _MidiOutBackend(prefer_system_synth=True)
        self._name = "Microsoft GS Wavetable Synth"

    def output_name(self) -> str:
        try:
            return str(self._impl.output_name())
        except Exception:
            return str(self._name)

    def program_select(self) -> None:
        self._impl.program_select()

    def set_gain(self, gain: float) -> None:
        self._impl.set_gain(gain)

    def note_on(self, midi_note: int, velocity: int) -> None:
        self._impl.note_on(midi_note, velocity)

    def note_off(self, midi_note: int) -> None:
        self._impl.note_off(midi_note)

    def control_change(self, control: int, value: int) -> None:
        self._impl.control_change(control, value)

    def all_notes_off(self) -> None:
        self._impl.all_notes_off()

    def shutdown(self) -> None:
        self._impl.shutdown()


class Player:
    """Playback of `SCORE` using system synth or external MIDI output."""

    def __init__(
        self,
        soundfont_path: Optional[str] = None,
        playback_mode: str = "system",
        midi_out_port: Optional[str] = None,
    ) -> None:
        self._backend: Optional[_Backend] = None
        self._backend_kind: str = "rtmidi"
        self._output_name: str = ""

        mode = str(playback_mode or "system").strip().lower()
        if mode not in ("system", "external"):
            mode = "system"

        if mode == "external":
            self._backend = _MidiOutBackend(
                port_name=midi_out_port,
                require_named_port=bool(str(midi_out_port or "").strip()),
            )
            self._backend_kind = "external-midi"
            try:
                self._output_name = str(self._backend.output_name())
            except Exception:
                self._output_name = ""
        elif sys.platform.startswith("linux"):
            self._backend = _FluidsynthBackend(soundfont_path)
            self._backend_kind = "fluidsynth"
        elif sys.platform == "darwin":
            self._backend = _MacDLSSynthBackend()
            self._backend_kind = "coreaudio-dls"
            self._output_name = "Apple DLS Synth"
        elif sys.platform.startswith("win"):
            self._backend = _WinMMSynthBackend()
            self._backend_kind = "winmm"
            try:
                self._output_name = str(self._backend.output_name())
            except Exception:
                self._output_name = "Microsoft GS Wavetable Synth"
        else:
            self._backend = _MidiOutBackend()
            self._backend_kind = "rtmidi"
            try:
                self._output_name = str(self._backend.output_name())
            except Exception:
                self._output_name = ""

        self._pitch_offset: int = 20  # App pitch 49 == MIDI 69 (A4); MIDI = app + 20
        self._channel: int = 0
        self._gain: float = 0.35
        self._thread: Optional[threading.Thread] = None
        self._running: bool = False
        self._bpm: float = 120.0
        self._t0: float = 0.0
        self._start_units: float = 0.0
        self._last_event_count: int = 0
        self._playhead_timeline: Optional[List[Tuple[float, float, float, float]]] = None
        self._playhead_sync_delay_ms: int = 0
        self._off_epsilon_sec: float = 0.003  # ~3 ms safety gap before offs
        self._repedal_gap_sec: float = 0.012  # ~12 ms gap for quick up->down re-pedal
        self._min_duration_units: float = 4.0
        self._grace_duration_units: float = 32.0  # Default grace note length (32nd note)

        if self._backend is not None:
            self._backend.program_select()

        # Load and apply reverb settings for FluidSynth backend
        if self._backend_kind == "fluidsynth" and isinstance(self._backend, _FluidsynthBackend):
            self._load_reverb_settings_from_appdata()

    def set_playhead_sync_delay_ms(self, delay_ms: int) -> None:
        """Set a playhead delay offset so visuals align with audible output latency."""
        try:
            self._playhead_sync_delay_ms = int(delay_ms)
        except Exception:
            self._playhead_sync_delay_ms = 0

    def set_soundfont(self, path: str) -> None:
        if self._backend_kind != "fluidsynth":
            return
        backend = self._backend
        if not isinstance(backend, _FluidsynthBackend):
            return
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"Soundfont not found: {p}")
        backend._soundfont_path = str(p)
        backend._init_synth()

    def set_gain(self, gain: float) -> None:
        g = float(max(0.0, gain))
        self._gain = g
        if self._backend is not None:
            try:
                self._backend.set_gain(g)
            except Exception:
                pass

    def _load_reverb_settings_from_appdata(self) -> None:
        """Load and apply reverb settings from appdata to the FluidSynth backend."""
        try:
            from appdata_manager import get_appdata_manager
            adm = get_appdata_manager()
            if adm is not None and isinstance(self._backend, _FluidsynthBackend):
                enabled = bool(adm.get("fluidsynth_reverb_enabled", True))
                room_size = float(adm.get("fluidsynth_reverb_room_size", 0.6))
                damp = float(adm.get("fluidsynth_reverb_damp", 0.4))
                width = float(adm.get("fluidsynth_reverb_width", 3.0))
                level = float(adm.get("fluidsynth_reverb_level", 0.9))
                sync_delay_ms = int(adm.get("fluidsynth_playhead_sync_delay_ms", 0))

                self._backend.set_reverb_enabled(enabled)
                self._backend.set_reverb_room_size(room_size)
                self._backend.set_reverb_damp(damp)
                self._backend.set_reverb_width(width)
                self._backend.set_reverb_level(level)
                self.set_playhead_sync_delay_ms(sync_delay_ms)
        except Exception:
            # Silently ignore if unable to load settings
            pass

    # ------------------------------------------------------------------
    # Playback control
    # ------------------------------------------------------------------
    def _stop_for_restart(self) -> None:
        self._running = False
        try:
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=0.2)
        except Exception:
            pass
        self._thread = None
        try:
            self._all_notes_off()
        except Exception:
            pass
        try:
            time.sleep(0.02)
        except Exception:
            pass

    def play_score(self, score) -> None:
        if self.is_playing():
            self._stop_for_restart()
        events = self._build_events_full(score)
        self._run_events(events)

    def play_from_time_cursor(self, start_units: float, score) -> None:
        if self.is_playing():
            self._stop_for_restart()
        events = self._build_events_from_time(start_units, score)
        self._run_events(events)

    def stop(self) -> None:
        self._running = False
        try:
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=0.5)
        except Exception:
            pass
        self._thread = None
        try:
            self._all_notes_off()
        except Exception:
            pass

    def panic(self) -> None:
        self.stop()

    def shutdown(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
        try:
            if self._backend is not None:
                self._backend.shutdown()
        except Exception:
            pass
        self._backend = None

    def audition_note(self, pitch: int = 40, velocity: int = 80, duration_sec: float = 0.2) -> None:
        if self.is_playing():
            return
        if self._backend is None:
            raise RuntimeError("Audio backend not initialized.")
        midi_pitch = max(0, min(127, int(pitch) + int(self._pitch_offset)))
        vel = int(max(1, min(127, velocity)))
        dur = float(max(0.02, duration_sec))

        def _run():
            try:
                self._backend.note_on(midi_pitch, vel)
                time.sleep(dur)
                self._backend.note_off(midi_pitch)
            except Exception:
                pass

        th = threading.Thread(target=_run, daemon=True)
        th.start()

    def is_playing(self) -> bool:
        return bool(self._running)

    # ------------------------------------------------------------------
    # Event scheduling
    # ------------------------------------------------------------------
    def _run_events(self, events: List[Tuple[str, float, int, int]]) -> None:
        if self._backend is None:
            raise RuntimeError("Audio backend not initialized.")
        self._running = True

        def _runner() -> None:
            t0 = time.time()
            try:
                self._t0 = float(t0)
            except Exception:
                pass
            for kind, t_rel, midi_note, vel in events:
                if not self._running:
                    break
                now = time.time()
                delay = max(0.0, t0 + float(t_rel) - now)
                if delay > 0:
                    time.sleep(delay)
                if not self._running:
                    break
                try:
                    if kind == 'on':
                        self._backend.note_on(int(midi_note), int(vel))
                    elif kind == 'cc':
                        self._backend.control_change(int(midi_note), int(vel))
                    else:
                        self._backend.note_off(int(midi_note))
                except Exception:
                    pass
            self._running = False
            try:
                self._all_notes_off()
            except Exception:
                pass

        th = threading.Thread(target=_runner, daemon=True)
        th.start()
        self._thread = th

    def _all_notes_off(self) -> None:
        if self._backend is None:
            return
        try:
            self._backend.all_notes_off()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Event building
    # ------------------------------------------------------------------
    def _build_events_full(self, score) -> List[Tuple[str, float, int, int]]:
        events: List[Tuple[str, float, int, int]] = []
        segs = self._get_tempo_segments(score)
        try:
            self._bpm = float(segs[0][2]) if segs else 120.0
            self._start_units = 0.0
        except Exception:
            pass

        playable: List[Tuple[float, float, int, int]] = []
        score_end_units = 0.0
        for n, dur_units in self._iter_playable_events(score):
            if dur_units < float(self._min_duration_units):
                continue
            start_units = float(getattr(n, 'time', 0.0) or 0.0)
            end_units = float(start_units + dur_units)
            vel = int(getattr(n, 'velocity', 64) or 64)
            app_pitch = int(getattr(n, 'pitch', 0))
            midi_pitch = max(0, min(127, app_pitch + self._pitch_offset))
            playable.append((start_units, end_units, midi_pitch, vel))
            if end_units > score_end_units:
                score_end_units = end_units

        if not playable:
            self._playhead_timeline = None
            self._last_event_count = 0
            return events

        score_end_units = self._resolve_playback_end_units(score, score_end_units)

        # Repeat flow is applied only to full playback (play button from beginning).
        # Inline playback from the current cursor intentionally ignores repeat jumps.
        play_segments = self._build_repeat_play_segments(score, score_end_units)
        segment_sec_cursor = 0.0
        timed_segments: List[Tuple[float, float, float]] = []
        playhead_timeline: List[Tuple[float, float, float, float]] = []
        for seg_start, seg_end in play_segments:
            timed_segments.append((seg_start, seg_end, segment_sec_cursor))
            seg_sec = self._seconds_between(seg_start, seg_end, segs)
            playhead_timeline.append((seg_start, seg_end, segment_sec_cursor, segment_sec_cursor + seg_sec))
            segment_sec_cursor += seg_sec

        self._playhead_timeline = playhead_timeline

        # Sustain pedal symbols are mapped to CC64 (0 = up, 127 = down).
        pedal_points = self._collect_sustain_pedal_points(score)
        if pedal_points:
            prev_seg_end: Optional[float] = None
            for seg_start, seg_end, seg_sec_start in timed_segments:
                seg_start = float(seg_start)
                seg_end = float(seg_end)
                seg_sec_start = float(seg_sec_start)
                if prev_seg_end is None or abs(seg_start - float(prev_seg_end)) > 1e-9:
                    state_at_start = self._sustain_state_at(seg_start, pedal_points)
                    events.append(('cc', seg_sec_start, 64, int(state_at_start)))

                prev_src_time: Optional[float] = None
                prev_src_val: Optional[int] = None
                for p_time, p_val in pedal_points:
                    if p_time < seg_start or p_time >= seg_end:
                        continue
                    cc_t = seg_sec_start + self._seconds_between(seg_start, float(p_time), segs)
                    if (
                        prev_src_time is not None
                        and abs(float(p_time) - float(prev_src_time)) <= 1e-9
                        and int(prev_src_val or 0) == 0
                        and int(p_val) == 127
                    ):
                        cc_t = float(cc_t) + float(self._repedal_gap_sec)
                    events.append(('cc', float(cc_t), 64, int(p_val)))
                    prev_src_time = float(p_time)
                    prev_src_val = int(p_val)

                prev_seg_end = seg_end

        for note_start, note_end, midi_pitch, vel in playable:
            for seg_start, seg_end, seg_sec_start in timed_segments:
                ov_start = max(note_start, seg_start)
                ov_end = min(note_end, seg_end)
                if not _time_op.greater(ov_end, ov_start):
                    continue
                on_t = seg_sec_start + self._seconds_between(seg_start, ov_start, segs)
                off_t = seg_sec_start + self._seconds_between(seg_start, ov_end, segs)
                events.append(('on', float(on_t), int(midi_pitch), int(vel)))
                events.append(('off', max(0.0, float(off_t) - float(self._off_epsilon_sec)), int(midi_pitch), 0))

        events.sort(key=lambda e: (e[1], 0 if e[0] == 'off' else 1))
        try:
            self._last_event_count = int(len([1 for e in events if e[0] == 'on']))
        except Exception:
            pass
        return events

    def _build_events_from_time(self, start_units: float, score) -> List[Tuple[str, float, int, int]]:
        segs = self._get_tempo_segments(score)
        su = float(max(0.0, start_units))
        self._playhead_timeline = None
        try:
            self._bpm = float(segs[0][2]) if segs else 120.0
            self._start_units = float(su)
        except Exception:
            pass
        events: List[Tuple[str, float, int, int]] = []

        pedal_points = self._collect_sustain_pedal_points(score)
        if pedal_points:
            state_at_start = self._sustain_state_at(su, pedal_points)
            events.append(('cc', 0.0, 64, int(state_at_start)))
            prev_src_time: Optional[float] = None
            prev_src_val: Optional[int] = None
            for p_time, p_val in pedal_points:
                if p_time < su:
                    continue
                cc_t = self._seconds_between(su, float(p_time), segs)
                if (
                    prev_src_time is not None
                    and abs(float(p_time) - float(prev_src_time)) <= 1e-9
                    and int(prev_src_val or 0) == 0
                    and int(p_val) == 127
                ):
                    cc_t = float(cc_t) + float(self._repedal_gap_sec)
                events.append(('cc', float(cc_t), 64, int(p_val)))
                prev_src_time = float(p_time)
                prev_src_val = int(p_val)
        for n, dur_units in self._iter_playable_events(score):
            start = float(getattr(n, 'time', 0.0) or 0.0)
            end = float(start + dur_units)
            if dur_units < float(self._min_duration_units):
                continue
            app_pitch = int(getattr(n, 'pitch', 0))
            midi_pitch = max(0, min(127, app_pitch + self._pitch_offset))
            vel = int(getattr(n, 'velocity', 64) or 64)
            if not _time_op.greater(end, su):
                continue
            if _time_op.less(start, su) and _time_op.greater(end, su):
                events.append(('on', 0.0, midi_pitch, vel))
                off_t = self._seconds_between(su, end, segs)
                off_t = max(0.0, float(off_t) - float(self._off_epsilon_sec))
                events.append(('off', float(off_t), midi_pitch, 0))
            elif not _time_op.less(start, su):
                on_t = self._seconds_between(su, start, segs)
                dur_t = self._seconds_between(start, end, segs)
                events.append(('on', float(on_t), midi_pitch, vel))
                off_t = float(on_t + max(0.0, dur_t) - float(self._off_epsilon_sec))
                events.append(('off', max(0.0, off_t), midi_pitch, 0))
        events.sort(key=lambda e: (e[1], 0 if e[0] == 'off' else 1))
        try:
            self._last_event_count = int(len([1 for e in events if e[0] == 'on']))
        except Exception:
            pass
        return events

    def _iter_playable_events(self, score):
        """Yield (event, duration_units) for normal and grace notes."""
        notes = getattr(getattr(score, 'events', None), 'note', []) or []
        note_spans: List[Tuple[float, float, int]] = []
        # Normal notes carry their own duration; skip malformed entries.
        for n in notes:
            dur_units = float(getattr(n, 'duration', 0.0) or 0.0)
            start_units = float(getattr(n, 'time', 0.0) or 0.0)
            pitch = int(getattr(n, 'pitch', 0))
            note_spans.append((start_units, float(start_units + dur_units), pitch))
            yield n, dur_units

        # Grace notes have no stored duration; fall back to the default 32.0 units.
        for g in getattr(getattr(score, 'events', None), 'grace_note', []) or []:
            # Default grace duration is fixed at 32 units unless overlapped with a sustaining note.
            dur_units = float(self._grace_duration_units)
            start_units = float(getattr(g, 'time', 0.0) or 0.0)
            # If a grace starts during a note, extend its end to that note's end to avoid cutting the note off.
            overlap_end: Optional[float] = None
            g_pitch = int(getattr(g, 'pitch', 0))
            for s, e, p in note_spans:
                if p == g_pitch and s <= start_units < e:
                    overlap_end = e if overlap_end is None else max(overlap_end, e)
            if overlap_end is not None:
                dur_units = max(dur_units, float(overlap_end - start_units))
            yield g, dur_units

    def _collect_sustain_pedal_points(self, score) -> List[Tuple[float, int]]:
        """Collect sustain pedal CC64 points from pedal symbols.

        Returns sorted list of (time_units, cc_value) where cc_value is 127 (down)
        or 0 (up). Only *_keytab and *_klavarskribo up/down symbols are considered.

        Two consecutive down symbols are interpreted as a quick re-pedal:
        release and immediately press again at the second symbol time.
        """
        out: List[Tuple[float, int]] = []
        events_obj = getattr(score, 'events', None)
        pedal_events = list(getattr(events_obj, 'pedal', []) or [])
        pedal_events = sorted(pedal_events, key=lambda ev: float(getattr(ev, 'time', 0.0) or 0.0))
        last_symbol: Optional[str] = None
        for ev in pedal_events:
            p_time = float(getattr(ev, 'time', 0.0) or 0.0)
            symbol = str(getattr(ev, 'symbol', '') or '').strip().lower()
            
            if symbol in ('down_keytab', 'down_klavarskribo'):
                if last_symbol == 'down':
                    out.append((p_time, 0))
                out.append((p_time, 127))
                last_symbol = 'down'
            elif symbol in ('up_keytab', 'up_klavarskribo'):
                out.append((p_time, 0))
                last_symbol = 'up'

        out.sort(key=lambda m: float(m[0]))
        return out

    def _sustain_state_at(self, t_units: float, pedal_points: List[Tuple[float, int]]) -> int:
        """Return sustain state (CC64 value) at source time, defaulting to off."""
        state = 0
        t = float(t_units)
        for p_time, p_val in pedal_points:
            if float(p_time) > t:
                break
            state = int(p_val)
        return int(state)

    def _build_repeat_play_segments(self, score, score_end_units: float) -> List[Tuple[float, float]]:
        """Build source-time segments in playback order using start/end repeat symbols.

        Each end-repeat triggers one jump back to the nearest preceding start-repeat
        (or to 0.0 when none exists), which matches common repeat playback behavior.
        """
        score_end = float(max(0.0, score_end_units))
        if score_end <= 0.0:
            return []

        events_obj = getattr(score, 'events', None)
        start_repeat_events = list(getattr(events_obj, 'start_repeat', []) or [])
        end_repeat_events = list(getattr(events_obj, 'end_repeat', []) or [])

        start_times = sorted(
            float(getattr(ev, 'time', 0.0) or 0.0)
            for ev in start_repeat_events
            if float(getattr(ev, 'time', 0.0) or 0.0) >= 0.0
        )
        end_times = sorted(
            float(getattr(ev, 'time', 0.0) or 0.0)
            for ev in end_repeat_events
            if 0.0 <= float(getattr(ev, 'time', 0.0) or 0.0) <= score_end
        )

        if not end_times:
            return [(0.0, score_end)]

        eps = 1e-9

        def _nearest_repeat_start(end_t: float) -> float:
            nearest: Optional[float] = None
            for st in start_times:
                # Start+end at the same barline is valid (:||:). For an end-repeat
                # jump target, only consider starts strictly before the end marker.
                if st < (end_t - eps):
                    nearest = st
                else:
                    break
            if nearest is None:
                return 0.0
            return float(nearest)

        cursor = 0.0
        consumed_end_idx: set[int] = set()
        segments: List[Tuple[float, float]] = []
        # Safety bound prevents malformed repeat data from causing an infinite loop.
        max_steps = max(16, len(end_times) * 4 + 16)
        steps = 0

        while cursor < (score_end - eps) and steps < max_steps:
            steps += 1
            next_idx: Optional[int] = None
            for i, t_end in enumerate(end_times):
                if t_end > (cursor + eps):
                    next_idx = i
                    break

            if next_idx is None:
                segments.append((cursor, score_end))
                break

            end_t = float(end_times[next_idx])
            if end_t > score_end:
                segments.append((cursor, score_end))
                break

            if end_t > (cursor + eps):
                segments.append((cursor, end_t))

            if next_idx not in consumed_end_idx:
                consumed_end_idx.add(next_idx)
                jump_t = _nearest_repeat_start(end_t)
                if jump_t < (end_t - eps):
                    cursor = jump_t
                else:
                    cursor = end_t
            else:
                cursor = end_t

        if steps >= max_steps and cursor < (score_end - eps):
            segments.append((cursor, score_end))

        return [(a, b) for (a, b) in segments if b > (a + eps)]

    def _resolve_playback_end_units(self, score, note_end_units: float) -> float:
        """Playback end should include explicit repeat end markers, not only note ends."""
        end_u = float(max(0.0, note_end_units))
        try:
            events_obj = getattr(score, 'events', None)
            for ev in list(getattr(events_obj, 'end_repeat', []) or []):
                t = float(getattr(ev, 'time', 0.0) or 0.0)
                if t > end_u:
                    end_u = t
        except Exception:
            pass
        return end_u

    # ------------------------------------------------------------------
    # Tempo helpers
    # ------------------------------------------------------------------
    def _get_tempo_segments(self, score) -> List[Tuple[float, float, float]]:
        segs: List[Tuple[float, float, float]] = []
        try:
            lst = sorted(list(getattr(score.events, 'tempo', []) or []), key=lambda e: float(getattr(e, 'time', 0.0) or 0.0))
        except Exception:
            lst = []
        if not lst:
            return [(0.0, float('inf'), 60.0 / (120.0 * float(QUARTER_NOTE_UNIT)))]
        for i, ev in enumerate(lst):
            start = float(getattr(ev, 'time', 0.0) or 0.0)
            s_per_unit = self._calculate_tempo(ev)
            if i + 1 < len(lst):
                next_start = float(getattr(lst[i + 1], 'time', 0.0) or 0.0)
                end = max(start, next_start)
            else:
                end = float('inf')
            segs.append((start, end, float(s_per_unit)))
        return segs

    def _seconds_between(self, a_units: float, b_units: float, segs: List[Tuple[float, float, float]]) -> float:
        if b_units <= a_units:
            return 0.0
        total = 0.0
        for s, e, s_per_unit in segs:
            if e <= a_units:
                continue
            if s >= b_units:
                break
            lo = max(a_units, s)
            hi = min(b_units, e)
            if hi > lo:
                total += (hi - lo) * float(s_per_unit)
        if segs and b_units > segs[-1][1]:
            _s_last, e_last, s_per_unit_last = segs[-1]
            lo = max(a_units, e_last)
            hi = b_units
            if hi > lo:
                total += (hi - lo) * float(s_per_unit_last)
        return total

    def _units_from_elapsed(self, elapsed_sec: float, start_units: float, segs: List[Tuple[float, float, float]]) -> float:
        u = float(start_units)
        rem = float(elapsed_sec)
        idx = 0
        for i, (s, e, _s_per_unit) in enumerate(segs):
            if s <= start_units < e:
                idx = i
                break
            if start_units >= e:
                idx = i + 1
        while rem > 0.0 and idx < len(segs):
            s, e, s_per_unit = segs[idx]
            seg_lo = max(s, u)
            seg_hi = e
            if seg_hi > seg_lo:
                seg_sec = (seg_hi - seg_lo) * float(s_per_unit)
                if rem >= seg_sec:
                    u = seg_hi
                    rem -= seg_sec
                    idx += 1
                    continue
                u += rem / float(s_per_unit)
                rem = 0.0
                return u
            idx += 1
        if segs and rem > 0.0:
            _s, _e, s_per_unit_last = segs[-1]
            u += rem / float(s_per_unit_last)
        return u

    def _calculate_tempo(self, tempo_event) -> float:
        try:
            tpm = float(getattr(tempo_event, 'tempo', 60.0) or 60.0)
        except Exception:
            tpm = 60.0
        try:
            dur_units = float(getattr(tempo_event, 'duration', 0.0) or 0.0)
        except Exception:
            dur_units = 0.0
        if dur_units <= 0.0:
            return 60.0 / (120.0 * float(QUARTER_NOTE_UNIT))
        return 60.0 / (float(tpm) * float(dur_units))

    # ------------------------------------------------------------------
    # Status and playhead
    # ------------------------------------------------------------------
    def get_playhead_time(self, score=None) -> Optional[float]:
        if not bool(self._running):
            return None
        try:
            delay_sec = max(0.0, float(self._playhead_sync_delay_ms) / 1000.0)
            elapsed = max(0.0, time.time() - float(self._t0) - delay_sec)
            if score is None:
                s_per_unit = float(self._bpm) if self._bpm > 0 else (60.0 / (120.0 * float(QUARTER_NOTE_UNIT)))
                units = float(self._start_units) + float(elapsed) / float(s_per_unit)
                return units
            segs = self._get_tempo_segments(score)
            if self._playhead_timeline:
                eps = 1e-9
                for u0, u1, s0, s1 in self._playhead_timeline:
                    if elapsed < (s1 - eps):
                        local_elapsed = max(0.0, float(elapsed - s0))
                        u_local = self._units_from_elapsed(local_elapsed, float(u0), segs)
                        return min(float(u1), float(u_local))
                if self._playhead_timeline:
                    return float(self._playhead_timeline[-1][1])
            u = self._units_from_elapsed(float(elapsed), float(self._start_units), segs)
            return u
        except Exception:
            return None

    def get_debug_status(self) -> dict:
        sf = None
        if isinstance(self._backend, _FluidsynthBackend):
            sf = getattr(self._backend, "_soundfont_path", None)
        return {
            'playback_type': self._backend_kind,
            'bpm': float(self._bpm),
            'events': int(self._last_event_count),
            'soundfont': str(sf or ''),
            'output': str(self._output_name),
            'gain': float(self._gain),
            'playhead_sync_delay_ms': int(self._playhead_sync_delay_ms),
        }

    @staticmethod
    def list_midi_output_ports() -> List[str]:
        return list_midi_output_ports()