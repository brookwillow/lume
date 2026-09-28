"""Voice input: global hotkey + audio recording + FunASR local STT."""

import re
import tempfile
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import sounddevice as sd

from .asr_correction import normalize_asr_text_with_trace, normalize_hotword_aliases
from .asr_lexicon import ASRLexiconEngine, ASRLexiconSettings
from .config import load_config


@dataclass
class ASRResult:
    """ASR result with metadata tags from SenseVoiceSmall."""
    text: str = ""           # Clean transcribed text
    language: str = ""       # e.g. "zh", "en", "ja", "ko"
    emotion: str = ""        # e.g. "NEUTRAL", "HAPPY", "SAD", "ANGRY"
    event: str = ""          # e.g. "Speech", "Music", "Noise"
    raw: str = ""            # Original output with tags
    hotwords_requested: list[dict] = field(default_factory=list)
    hotword_matches: list[dict] = field(default_factory=list)
    hotword_param_accepted: bool = False

# Recording state
_recording = False
_audio_frames: list[np.ndarray] = []
_sample_rate = 16000
_input_device = None  # None = auto-detect

# Lazy-loaded ASR model
_asr_model = None
_asr_settings = load_config().asr
_hotword_engine = ASRLexiconEngine(ASRLexiconSettings.from_config(_asr_settings))

class AudioInputError(RuntimeError):
    """Raised when CoreAudio cannot open any usable microphone stream."""


def _input_device_candidates() -> list[int]:
    """Return usable input devices, preferring the selected/default microphone."""
    devices = sd.query_devices()
    candidates = []
    default = sd.default.device[0]
    if isinstance(default, int) and 0 <= default < len(devices):
        candidates.append(default)
    if isinstance(_input_device, int) and 0 <= _input_device < len(devices):
        candidates.append(_input_device)

    available = [i for i, device in enumerate(devices) if device["max_input_channels"] >= 1]
    # macOS built-in input is generally the safest fallback after the user's
    # selected/default device; then try all remaining input devices.
    available.sort(key=lambda i: ("macbook" not in devices[i]["name"].lower() and "built-in" not in devices[i]["name"].lower(), i))
    candidates.extend(available)
    return list(dict.fromkeys(candidates))


def _candidate_rates(device_info: dict) -> list[int]:
    default_rate = int(device_info.get("default_samplerate") or 0)
    return list(dict.fromkeys(rate for rate in (default_rate, 48000, 44100, 16000) if rate > 0))


def _open_input_stream(callback) -> tuple[sd.InputStream, int, int]:
    """Open a one-channel input stream with device/rate fallback.

    CoreAudio can expose a stale default device (including ``-1``) or reject a
    format after a headset/aggregate-device change. Try each valid input device
    and a small set of standard rates before reporting a recoverable error.
    """
    global _input_device
    devices = sd.query_devices()
    failures = []
    for device_idx in _input_device_candidates():
        device_info = devices[device_idx]
        for rate in _candidate_rates(device_info):
            stream = None
            try:
                sd.check_input_settings(device=device_idx, channels=1, dtype="float32", samplerate=rate)
                stream = sd.InputStream(samplerate=rate, channels=1, dtype="float32", callback=callback, device=device_idx)
                stream.start()
                _input_device = device_idx
                print(f"  🎙️  Using: {device_info['name']} ({rate} Hz)")
                return stream, device_idx, rate
            except Exception as exc:
                failures.append(f"{device_info['name']}@{rate}: {exc}")
                if stream is not None:
                    try:
                        stream.close()
                    except Exception:
                        pass
    detail = failures[-1] if failures else "no input device is available"
    raise AudioInputError(f"Unable to open a microphone input stream ({detail})")


def _get_asr_model():
    """Lazy-load FunASR SenseVoiceSmall model."""
    global _asr_model
    if _asr_model is None:
        import logging
        logging.getLogger("modelscope").setLevel(logging.WARNING)
        from funasr import AutoModel
        _asr_model = AutoModel(
            model="iic/SenseVoiceSmall",
            vad_model="fsmn-vad",
            device="cpu",
            disable_update=True,
        )
    return _asr_model


def preload_model():
    """Preload ASR model so first recognition is fast. Call from background thread."""
    print("  ⏳ Preloading ASR model...")
    _get_asr_model()
    print("  ✓ ASR model ready")


def start_recording(on_partial: Callable[[str], None] | None = None,
                    on_level: Callable[[float], None] | None = None):
    """Start capturing audio from default microphone.
    
    on_partial is accepted for API compatibility but not used (non-streaming).
    on_level: callback with audio RMS level (0.0-1.0) for waveform display.
    """
    global _recording, _audio_frames

    # Clean up any leftover stream
    old_stream = getattr(start_recording, "_stream", None)
    if old_stream:
        try:
            old_stream.stop()
            old_stream.close()
        except Exception:
            pass
        start_recording._stream = None

    _audio_frames = []
    _recording = True

    def callback(indata, frames, time, status):
        if _recording:
            _audio_frames.append(indata.copy())
            if on_level:
                # Calculate RMS and scale to 0-1 (peak at ~0.1 RMS is loud speech)
                rms = float(np.sqrt(np.mean(indata**2)))
                level = min(1.0, rms / 0.08)
                on_level(level)

    try:
        stream, _, rate = _open_input_stream(callback)
    except Exception:
        _recording = False
        raise
    start_recording._stream = stream
    start_recording._rate = rate


def stop_recording(scene: str | None = None) -> ASRResult:
    """Stop recording, run ASR, return transcribed text."""
    global _recording
    _recording = False

    stream = getattr(start_recording, "_stream", None)
    if stream:
        stream.stop()
        stream.close()
        start_recording._stream = None

    if not _audio_frames:
        return ASRResult()

    # Concatenate audio
    audio = np.concatenate(_audio_frames, axis=0)
    rec_rate = getattr(start_recording, "_rate", _sample_rate)
    duration = len(audio) / rec_rate
    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))
    print(f"  📊 Audio: {duration:.1f}s, RMS={rms:.4f}, peak={peak:.4f}")

    if duration < 0.3:
        return ASRResult()

    # Resample to 16kHz if needed (ASR model expects 16kHz)
    if rec_rate != _sample_rate:
        import scipy.signal
        num_samples = int(len(audio) * _sample_rate / rec_rate)
        audio = scipy.signal.resample(audio, num_samples)

    wav_path = Path(tempfile.mktemp(suffix=".wav"))

    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_sample_rate)
        audio_int16 = (audio.flatten() * 32767).astype(np.int16)
        wf.writeframes(audio_int16.tobytes())

    # Run ASR
    try:
        model = _get_asr_model()
        hotwords = _hotword_engine.recall_hotwords(scene=scene)
        options = {"input": str(wav_path), "disable_pbar": True}
        if hotwords:
            options["hotword"] = " ".join(item["text"] for item in hotwords)
        hotword_param_accepted = bool(hotwords)
        try:
            result = model.generate(**options)
        except TypeError:
            # Older SenseVoice builds may not expose hotword biasing; their
            # final text still benefits from persisted variant normalization.
            options.pop("hotword", None)
            hotword_param_accepted = False
            result = model.generate(**options)
        if result and result[0].get("text"):
            raw = result[0]["text"]
            return _parse_asr_output(raw, hotwords, hotword_param_accepted)
    finally:
        import os
        try:
            os.unlink(wav_path)
        except OSError:
            pass

    return ASRResult()


def _parse_asr_output(raw: str, hotwords_requested: list[dict] | None = None, hotword_param_accepted: bool = False) -> ASRResult:
    """Parse SenseVoiceSmall output tags into structured result."""
    tags = re.findall(r'<\|([^|]*)\|>', raw)
    raw_text = re.sub(r'<\|[^|]*\|>', '', raw).strip()
    text, replacements = normalize_asr_text_with_trace(raw_text, _asr_settings)
    hotwords_requested = hotwords_requested or []
    text, hotword_replacements = normalize_hotword_aliases(text, hotwords_requested)
    replacements.extend(hotword_replacements)
    requested_terms = {str(item.get("text", "")) for item in hotwords_requested}
    hotword_matches = [
        {"canonical": term, "kind": "direct_text_match"}
        for term in requested_terms if term and term.lower() in raw_text.lower()
    ]
    hotword_matches.extend(
        {**item, "kind": "normalization_applied"}
        for item in replacements if str(item["canonical"]) in requested_terms
    )

    language = ""
    emotion = ""
    event = ""

    # Tag order is typically: language, emotion, event, itn_mode
    for tag in tags:
        if tag in ("zh", "en", "ja", "ko", "yue"):
            language = tag
        elif tag in ("NEUTRAL", "HAPPY", "SAD", "ANGRY", "FEARFUL", "DISGUSTED", "SURPRISED"):
            emotion = tag
        elif tag in ("Speech", "Music", "Noise", "Applause", "Laughter"):
            event = tag

    return ASRResult(text=text, language=language, emotion=emotion, event=event, raw=raw,
                     hotwords_requested=hotwords_requested, hotword_matches=hotword_matches,
                     hotword_param_accepted=hotword_param_accepted)
