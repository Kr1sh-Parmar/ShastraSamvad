"""Speech out. Piper when it's installed, browser speechSynthesis until then.

Same endpoint either way, so dropping Piper in later changes nothing upstream.
"""
import subprocess

from . import config


def available() -> bool:
    return (not config.USE_BROWSER_TTS
            and config.PIPER_BIN.exists() and config.PIPER_VOICE.exists())


def synth(text: str) -> bytes | None:
    """-> WAV bytes, or None meaning 'client, speak this yourself'."""
    if not available():
        return None
    proc = subprocess.run(
        [str(config.PIPER_BIN), "--model", str(config.PIPER_VOICE),
         "--output_file", "-"],
        input=text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return proc.stdout or None
