from __future__ import annotations

from pathlib import Path
import wave


def write_silent_wav(path: Path, *, sample_rate: int = 16000, duration_ms: int = 50) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = int(sample_rate * (duration_ms / 1000.0))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * num_frames)
