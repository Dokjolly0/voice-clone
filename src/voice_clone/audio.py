"""Audio helpers built on ffmpeg/ffprobe."""

import io
import re
import subprocess
import wave
from pathlib import Path

from voice_clone.util import UserError

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".wma", ".webm", ".mp4", ".mkv"}


_DURATIONS: dict[tuple[str, int, int], float] = {}


def duration(path: Path) -> float:
    """Duration in seconds of any file ffprobe can read (cached per file version: status pages ask often)."""
    st = Path(path).stat()
    key = (str(path), st.st_size, st.st_mtime_ns)
    if key not in _DURATIONS:
        _DURATIONS[key] = _probe_duration(path)
    return _DURATIONS[key]


def _probe_duration(path: Path) -> float:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
        return float(out)
    except (subprocess.CalledProcessError, ValueError) as exc:
        raise UserError(f"not a readable audio file: {path}") from exc


def wav_bytes_duration(data: bytes) -> float:
    with wave.open(io.BytesIO(data)) as w:
        return w.getnframes() / w.getframerate()


def wav_duration(path: Path) -> float:
    with wave.open(str(path)) as w:
        return w.getnframes() / w.getframerate()


def ffmpeg(args: list, *, capture_stderr: bool = False) -> str:
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "info" if capture_stderr else "error", "-y",
           *[str(a) for a in args]]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise UserError(f"ffmpeg failed: {result.stderr.strip()[-1500:]}")
    return result.stderr


def convert(src: Path, dst: Path, *, rate: int, filters: str | None = None) -> None:
    """Decode any audio file to mono 16-bit WAV at the given rate."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    args = ["-i", src]
    if filters:
        args += ["-af", filters]
    ffmpeg([*args, "-ac", 1, "-ar", rate, "-sample_fmt", "s16", dst])


# Trim silence at both ends: remove from the start, reverse, remove again, reverse back.
TRIM_SILENCE = ("silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.1,"
                "areverse,silenceremove=start_periods=1:start_threshold=-45dB:start_silence=0.1,areverse")


def concatenate(parts: list[Path], dst: Path, *, rate: int, gap: float = 0.4) -> None:
    """Join WAV files (same rate, mono) with `gap` seconds of silence between them."""
    if not parts:
        raise UserError("nothing to concatenate")
    inputs, labels = [], []
    for i, part in enumerate(parts):
        inputs += ["-i", part]
        labels.append(f"[{i}:a]")
    n = len(parts)
    if n == 1:
        graph = "[0:a]anull[out]"
    else:
        pads = "".join(f"[{i}:a]apad=pad_dur={gap}[p{i}];" for i in range(n - 1))
        joined = "".join(f"[p{i}]" for i in range(n - 1)) + f"[{n - 1}:a]"
        graph = f"{pads}{joined}concat=n={n}:v=0:a=1[out]"
    ffmpeg([*inputs, "-filter_complex", graph, "-map", "[out]", "-ac", 1, "-ar", rate, "-sample_fmt", "s16", dst])


def silences(path: Path, *, noise_db: float = -35, min_duration: float = 0.25) -> list[tuple[float, float]]:
    """(start, end) of each silence ffmpeg's silencedetect finds."""
    log = ffmpeg(["-i", path, "-af", f"silencedetect=noise={noise_db}dB:d={min_duration}", "-f", "null", "-"],
                 capture_stderr=True)
    starts = [float(m) for m in re.findall(r"silence_start: ([\d.]+)", log)]
    ends = [float(m) for m in re.findall(r"silence_end: ([\d.]+)", log)]
    return list(zip(starts, ends))


def cut_point(silence_list: list[tuple[float, float]], limit: float) -> float | None:
    """The middle of the last silence that starts before `limit`, so a cut there never splits a word."""
    best = None
    for start, end in silence_list:
        middle = (start + end) / 2
        if middle <= limit:
            best = middle
    return best


def cut(src: Path, dst: Path, seconds: float) -> None:
    ffmpeg(["-i", src, "-t", f"{seconds:.3f}", "-c", "copy", dst])
