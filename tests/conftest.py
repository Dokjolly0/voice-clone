import importlib
import math
import struct
import wave

import pytest


@pytest.fixture
def voices_dir(tmp_path, monkeypatch):
    """Point every module at an empty voices folder and cache."""
    monkeypatch.setenv("VOICE_CLONE_VOICES", str(tmp_path / "voices"))
    monkeypatch.setenv("VOICE_CLONE_CACHE", str(tmp_path / "cache"))
    import voice_clone.paths
    import voice_clone.workspace

    importlib.reload(voice_clone.paths)
    importlib.reload(voice_clone.workspace)
    yield tmp_path / "voices"
    monkeypatch.undo()
    importlib.reload(voice_clone.paths)
    importlib.reload(voice_clone.workspace)


@pytest.fixture
def make_wav(tmp_path):
    """Write a mono 16-bit sine-wave WAV and return its path."""

    def make(name="tone.wav", seconds=1.0, rate=22050, freq=440.0):
        path = tmp_path / name
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            frames = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * freq * i / rate)))
                              for i in range(int(seconds * rate)))
            w.writeframes(frames)
        return path

    return make
