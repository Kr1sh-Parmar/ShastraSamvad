"""Download Piper + an English voice into piper/ (arch doc 4.9).

    python scripts/05_fetch_piper.py

Until this runs, backend/tts.py returns None and the browser speaks instead —
so this is what moves the prototype off the quick-start shortcut and onto the
same TTS the device uses. Re-run it on the board (with --platform linux_x86_64
or the aarch64 build) to get the same voice there.

Reads  nothing
Writes piper/piper.exe, piper/en_US-lessac-medium.onnx{,.json}
"""
import io
import sys
import zipfile
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from backend import config

PIPER_RELEASE = "2023.11.14-2"
PIPER_ZIP = (f"https://github.com/rhasspy/piper/releases/download/"
             f"{PIPER_RELEASE}/piper_windows_amd64.zip")

VOICE = "en_US-lessac-medium"
VOICE_BASE = ("https://huggingface.co/rhasspy/piper-voices/resolve/main/"
              "en/en_US/lessac/medium/")


def fetch(url: str) -> bytes:
    with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        buf = bytearray()
        for chunk in r.iter_bytes(1 << 18):
            buf += chunk
            if total:
                print(f"\r  {len(buf) / 1e6:6.1f} / {total / 1e6:.1f} MB", end="")
        print()
        return bytes(buf)


def main() -> None:
    dest = config.PIPER_BIN.parent
    dest.mkdir(parents=True, exist_ok=True)

    if config.PIPER_BIN.exists():
        print(f"{config.PIPER_BIN.name} already present, skipping the binary")
    else:
        print(f"piper {PIPER_RELEASE} ...")
        zf = zipfile.ZipFile(io.BytesIO(fetch(PIPER_ZIP)))
        # The archive's top level is already `piper/`, so extract to the parent.
        zf.extractall(dest.parent)

    # Both files are required — Piper reads the .json for the voice's sample
    # rate and phoneme map, and will not start on the .onnx alone.
    for suffix in (".onnx", ".onnx.json"):
        out = dest / f"{VOICE}{suffix}"
        if out.exists():
            print(f"{out.name} already present, skipping")
            continue
        print(f"{out.name} ...")
        out.write_bytes(fetch(f"{VOICE_BASE}{VOICE}{suffix}"))

    ok = config.PIPER_BIN.exists() and config.PIPER_VOICE.exists()
    print(f"\n{'OK' if ok else 'INCOMPLETE'} -> {dest}")
    if ok and config.USE_BROWSER_TTS:
        print("Note: config.USE_BROWSER_TTS is still True — flip it to False "
              "to actually use Piper.")
    if not ok:
        raise SystemExit("piper.exe or the voice is missing — see above.")


if __name__ == "__main__":
    main()
