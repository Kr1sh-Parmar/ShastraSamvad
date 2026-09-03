"""Speech in. faster-whisper on the laptop, whisper.cpp base.en on the device."""
import io

from . import config

_model = None
_whisper_cpp_model = None


def model():
    """Lazily load and return the resident model for whichever backend is
    configured. main.py calls this once at startup (arch doc 6) so the first
    mic press doesn't pay the cold load; it doesn't care which backend that
    turns out to be.
    """
    if config.STT_USE_WHISPER_CPP:
        return _whisper_cpp()
    return _faster_whisper()


def _faster_whisper():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(config.WHISPER_MODEL, device="cpu",
                              compute_type=config.WHISPER_COMPUTE)
    return _model


def _whisper_cpp():
    global _whisper_cpp_model
    if _whisper_cpp_model is None:
        from pywhispercpp.model import Model
        _whisper_cpp_model = Model(str(config.WHISPER_CPP_MODEL_PATH))
    return _whisper_cpp_model


def available() -> bool:
    """Is the model loaded and ready to transcribe without a cold load?

    Reports readiness rather than mere installability, so /health cannot claim
    STT is fine while the first mic press still has to download the weights.
    """
    if config.STT_USE_WHISPER_CPP:
        return _whisper_cpp_model is not None
    return _model is not None


def _decode_to_pcm16k(audio: bytes):
    """webm/ogg/wav bytes -> mono float32 PCM at 16kHz, as a numpy array.

    pywhispercpp, unlike faster-whisper, does not decode a compressed
    container itself — it wants raw samples. PyAV (already a faster-whisper
    dependency, so no new install for this) does the same resampling job
    faster-whisper does internally, just made explicit here.
    """
    import av
    import numpy as np

    out = []
    with av.open(io.BytesIO(audio)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        for frame in container.decode(stream):
            for rframe in resampler.resample(frame):
                out.append(rframe.to_ndarray())
    if not out:
        return np.zeros(0, dtype=np.float32)
    pcm = np.concatenate(out, axis=1).reshape(-1).astype(np.float32) / 32768.0
    return pcm


def _transcribe_whisper_cpp(audio: bytes) -> dict:
    pcm = _decode_to_pcm16k(audio)
    segments = _whisper_cpp().transcribe(pcm)
    # whisper.cpp reports timestamps in centiseconds, not seconds.
    segs = [{"start": round(s.t0 / 100.0, 2), "end": round(s.t1 / 100.0, 2),
             "text": s.text.strip()} for s in segments]
    return {"text": " ".join(s["text"] for s in segs).strip(), "segments": segs}


def _transcribe_faster_whisper(audio: bytes) -> dict:
    segments, _ = model().transcribe(
        io.BytesIO(audio),
        language="en",
        vad_filter=True,          # drops the silence around a push-to-talk press
        beam_size=1,              # greedy: latency budget is 1-3s (arch doc 6)
    )
    segs = [{"start": round(s.start, 2), "end": round(s.end, 2),
             "text": s.text.strip()} for s in segments]
    return {"text": " ".join(s["text"] for s in segs).strip(), "segments": segs}


def transcribe(audio: bytes) -> dict:
    """-> {text, segments}  (contract: arch doc 8)

    Takes whatever the browser's MediaRecorder produced (webm/ogg/wav);
    faster-whisper decodes it via PyAV, so no ffmpeg install is needed. Same
    is true of the whisper.cpp path, which decodes through PyAV explicitly
    (see _decode_to_pcm16k) since pywhispercpp itself takes raw PCM only.

    Backend picked by config.STT_USE_WHISPER_CPP; the {text, segments} shape
    returned is identical either way, so main.py's /stt endpoint needs no
    changes.
    """
    if config.STT_USE_WHISPER_CPP:
        return _transcribe_whisper_cpp(audio)
    return _transcribe_faster_whisper(audio)
