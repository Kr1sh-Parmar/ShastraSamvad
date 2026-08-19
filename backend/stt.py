"""Speech in. faster-whisper on the laptop, whisper.cpp base.en on the device."""
import io

from . import config

_model = None


def model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel(config.WHISPER_MODEL, device="cpu",
                              compute_type=config.WHISPER_COMPUTE)
    return _model


def available() -> bool:
    """Is the model loaded and ready to transcribe without a cold load?

    Reports readiness rather than mere installability, so /health cannot claim
    STT is fine while the first mic press still has to download the weights.
    """
    return _model is not None


def transcribe(audio: bytes) -> dict:
    """-> {text, segments}  (contract: arch doc 8)

    Takes whatever the browser's MediaRecorder produced (webm/ogg/wav);
    faster-whisper decodes it via PyAV, so no ffmpeg install is needed.
    """
    segments, _ = model().transcribe(
        io.BytesIO(audio),
        language="en",
        vad_filter=True,          # drops the silence around a push-to-talk press
        beam_size=1,              # greedy: latency budget is 1-3s (arch doc 6)
    )
    segs = [{"start": round(s.start, 2), "end": round(s.end, 2),
             "text": s.text.strip()} for s in segments]
    return {"text": " ".join(s["text"] for s in segs).strip(), "segments": segs}
