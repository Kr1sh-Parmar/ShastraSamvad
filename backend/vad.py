"""Server-side voice activity detection — Silero VAD (arch doc 4.1).

The browser build never needs this: page.jsx already owns the mic and runs
its own RMS-loudness barge-in loop (threshold 0.08, sustained >12 frames) for
free, in JS, with no model to load. That path is untouched by this module.

A kiosk build with a mic wired directly into the device has no browser in
front of it to do that job, so config.VAD_USE_SILERO routes audio here
instead: same two jobs — "is the user talking right now" (barge-in) and "has
the user stopped talking" (endpointing) — done with Silero VAD because an RMS
threshold with no browser DSP behind it is unreliable on raw device mic input
(background hum, room noise).

Silero VAD expects 16kHz mono float32 PCM, one chunk at a time, in the exact
sizes it was trained on (512 samples per chunk at 16kHz). audio_chunk here is
raw PCM16LE bytes at 16kHz mono — the format a kiosk's own mic capture loop
hands over turn is expected to already be in, there is no browser-container
(webm/ogg) to decode as there is for the /stt upload path in stt.py.
"""
import numpy as np

from . import config

_model = None
_CHUNK_SAMPLES = 512            # Silero's trained window at 16kHz
_SAMPLE_RATE = 16000
_MS_PER_CHUNK = _CHUNK_SAMPLES / _SAMPLE_RATE * 1000  # 32ms

# Endpointing state, reset() between turns — how long the current run of
# speech/silence has lasted, and whether enough speech has already been seen
# to make trailing silence mean "end of turn" rather than "hasn't started yet".
_speech_run_ms = 0.0
_silence_run_ms = 0.0
_seen_enough_speech = False


def _load_model():
    global _model
    if _model is None:
        import torch
        _model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad")
    return _model


def _speech_prob(audio_chunk: bytes) -> float:
    """One chunk -> Silero's speech probability in [0, 1].

    Chunks that don't fill a full window (the tail of a turn) are zero-padded
    rather than dropped, so the last sliver of speech before a hang-up still
    counts.
    """
    import torch

    pcm = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
    if pcm.size < _CHUNK_SAMPLES:
        pcm = np.pad(pcm, (0, _CHUNK_SAMPLES - pcm.size))
    elif pcm.size > _CHUNK_SAMPLES:
        pcm = pcm[:_CHUNK_SAMPLES]
    tensor = torch.from_numpy(pcm)
    with torch.no_grad():
        return float(_load_model()(tensor, _SAMPLE_RATE).item())


def is_speech(audio_chunk: bytes) -> bool:
    """Does this one chunk contain speech, right now — for barge-in.

    A single-chunk decision, not endpointing: the caller decides what a
    sustained "yes" across chunks means, the same way page.jsx's RMS loop
    requires 12 consecutive loud frames rather than acting on one.
    """
    return _speech_prob(audio_chunk) >= config.VAD_THRESHOLD


def detect_endpoint(audio_chunk: bytes) -> bool:
    """Feed one more chunk of the current turn; True once the user has
    stopped talking — a real pause, not just one quiet chunk.

    Requires VAD_MIN_SPEECH_MS of speech to have already accumulated (so
    silence before the user starts talking doesn't count as them finishing),
    then VAD_MIN_SILENCE_MS of trailing silence to call the turn over.
    """
    global _speech_run_ms, _silence_run_ms, _seen_enough_speech

    if is_speech(audio_chunk):
        _speech_run_ms += _MS_PER_CHUNK
        _silence_run_ms = 0.0
        if _speech_run_ms >= config.VAD_MIN_SPEECH_MS:
            _seen_enough_speech = True
        return False

    _silence_run_ms += _MS_PER_CHUNK
    _speech_run_ms = 0.0
    return _seen_enough_speech and _silence_run_ms >= config.VAD_MIN_SILENCE_MS


def reset() -> None:
    """Clear endpointing state between turns — a fresh turn starts having
    heard neither speech nor silence yet.
    """
    global _speech_run_ms, _silence_run_ms, _seen_enough_speech
    _speech_run_ms = 0.0
    _silence_run_ms = 0.0
    _seen_enough_speech = False
