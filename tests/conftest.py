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


@pytest.fixture
def studio(tmp_path, monkeypatch, voices_dir):
    """An empty studio folder (models, dictionaries, history) and download cache."""
    from voice_clone import history, lexicon, models

    root = tmp_path / "studio"
    for module in (models, lexicon, history):
        monkeypatch.setattr(module, "STUDIO_DIR", root)
    monkeypatch.setattr(models, "CACHE_DIR", tmp_path / "cache")
    return root


@pytest.fixture
def fake_model_files(tmp_path):
    """A .onnx + .onnx.json pair that passes validation (the .onnx is not a real network)."""
    import json

    def make(name="it_IT-anna-medium", speakers=None):
        folder = tmp_path / "files"
        folder.mkdir(exist_ok=True)
        onnx = folder / f"{name}.onnx"
        onnx.write_bytes(b"not a real model")
        config = {"audio": {"sample_rate": 22050}, "espeak": {"voice": "it"}, "num_speakers": len(speakers or []) or 1,
                  "speaker_id_map": {s: i for i, s in enumerate(speakers or [])}, "phoneme_id_map": {"a": [1]},
                  "inference": {"noise_scale": 0.667, "length_scale": 1.0, "noise_w": 0.8}}
        (folder / f"{name}.onnx.json").write_text(json.dumps(config))
        return onnx

    return make
