"""Prosody measurements, to pick the most typical of several takes of a short sentence.

Drama 3 has no context for a one- or two-word sentence, so each take guesses its intonation: one comes out much
higher, the next flat or much lower. For short sentences the dataset stage generates several takes and keeps the one
whose pitch level, pitch movement and speaking rate are closest to the voice's own, measured on its long sentences.

Pitch is measured with librosa's pYIN. librosa lives in Piper's environment (installed by `setup`), so this module
also runs as a script there: `python prosody.py FILE...` prints one JSON object per file.
"""

import json
import math
import subprocess
import sys
from pathlib import Path


def _measure_files(paths: list[str]) -> list[dict]:
    """Runs inside Piper's environment."""
    import numpy as np
    import librosa

    out = []
    for path in paths:
        y, sr = librosa.load(path, sr=22050)
        trimmed, _ = librosa.effects.trim(y, top_db=35)
        seconds = len(trimmed) / sr
        f0, _, _ = librosa.pyin(trimmed, fmin=60, fmax=400, sr=sr, frame_length=1024)
        f0 = f0[~np.isnan(f0)]
        voiced = len(f0) >= 5
        out.append({"file": path, "seconds": round(seconds, 3),
                    "f0": round(float(np.median(f0)), 1) if voiced else None,
                    "f0_range": round(float(np.percentile(f0, 90) - np.percentile(f0, 10)), 1) if voiced else None})
    return out


def measure(paths: list[Path]) -> dict[str, dict]:
    """Measurements keyed by path, computed in Piper's environment."""
    from voice_clone.installer import piper_ready
    from voice_clone.paths import PIPER_PYTHON
    from voice_clone.util import UserError

    if not paths:
        return {}
    if not piper_ready():
        raise UserError("measuring prosody needs Piper's environment: run `voice-clone setup`")
    result = subprocess.run([str(PIPER_PYTHON), __file__, *map(str, paths)], capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise UserError(f"prosody measurement failed: {result.stderr.strip()[-1500:]}")
    rows = [json.loads(l) for l in result.stdout.splitlines() if l.startswith("{")]
    return {r["file"]: r for r in rows}


def letters(text: str) -> int:
    return sum(ch.isalpha() for ch in text)


def rate(m: dict, text: str) -> float | None:
    return letters(text) / m["seconds"] if m.get("seconds") else None


def profile(measurements: list[tuple[dict, str]]) -> dict:
    """The voice's typical pitch level, pitch movement and speaking rate (medians over long sentences)."""
    def median(values):
        values = sorted(v for v in values if v)
        return values[len(values) // 2] if values else None

    return {"f0": median(m["f0"] for m, _ in measurements),
            "f0_range": median(m["f0_range"] for m, _ in measurements),
            "rate": median(rate(m, t) for m, t in measurements),
            "clips": len(measurements)}


def distance(m: dict, text: str, ref: dict) -> float:
    """How far a take is from the voice's typical prosody; lower is better.

    Pitch level counts most (a semitone is ~6%), then a flat delivery (pitch movement under half the usual), then
    speaking rate. Short sentences are naturally a little slower, so only a large rate gap weighs much.
    """
    if not m.get("f0") or not ref.get("f0"):
        return 10.0
    score = abs(math.log(m["f0"] / ref["f0"])) / 0.12
    if ref.get("f0_range") and m.get("f0_range") is not None:
        score += max(0.0, 0.5 - m["f0_range"] / ref["f0_range"]) * 4
    r = rate(m, text)
    if r and ref.get("rate"):
        score += abs(math.log(r / ref["rate"])) / 0.35
    return round(score, 3)


if __name__ == "__main__":
    for row in _measure_files(sys.argv[1:]):
        print(json.dumps(row))
