"""The studio's speech engine: text in, a finished audio file out.

Runs with `piper-tts` (onnxruntime) in the app's own environment, on the CPU: no GPU, PyTorch or training setup
needed, and still about 50x faster than real time on a desktop CPU.

Text layout drives the pauses:
- a blank line separates paragraphs  → `paragraph_pause`;
- every other line break, and every sentence Piper finds inside a line → `sentence_pause`;
- `[pausa 1.5s]`, `[pause 500ms]` or `[break 2s]` anywhere → exactly that much silence;
- Piper's raw phonemes `[[ ... ]]` pass through untouched.
After synthesis, one ffmpeg pass applies pitch (rubberband, duration unchanged), volume, loudness normalization,
the output sample rate and the codec.
"""

import re
import subprocess
import tempfile
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from voice_clone.models import Model
from voice_clone.util import UserError

FORMATS = {
    #        extension, ffmpeg codec, default bitrate (None: lossless or quality-based), allowed sample rates
    "mp3": ("mp3", "libmp3lame", "192k", None),
    "wav": ("wav", "pcm_s16le", None, None),
    "ogg": ("ogg", "libvorbis", None, None),  # Vorbis refuses high bitrates for mono 22 kHz: quality scale instead
    "opus": ("opus", "libopus", "64k", (8000, 12000, 16000, 24000, 48000)),
    "flac": ("flac", "flac", None, None),
    "m4a": ("m4a", "aac", "128k", None),
}
SAMPLE_RATES = (16000, 22050, 24000, 44100, 48000)
PAUSE_RE = re.compile(r"\[\s*(?:pausa|pause|break)\s+(\d+(?:[.,]\d+)?)\s*(ms|s)?\s*\]", re.IGNORECASE)
MAX_PAUSE = 30.0


@dataclass
class Options:
    speed: float = 1.0              # 0.5-2.0
    volume_db: float = 0.0          # -20..+20
    pitch: float = 0.0              # semitones, -12..+12
    expressiveness: float = 1.0     # 0-1.5 x the model's default variation
    speaker: str | None = None      # name (or index) for multi-speaker models
    sentence_pause: float = 0.3     # seconds
    paragraph_pause: float = 0.8    # seconds
    normalize: bool = False         # loudness normalization to -16 LUFS
    format: str = "mp3"
    bitrate: str | None = None      # e.g. "192k"; default per format
    sample_rate: int | None = None  # None: the model's own
    use_lexicon: bool = True        # apply the pronunciation dictionary of the model's language

    def validate(self) -> "Options":
        if self.format not in FORMATS:
            raise UserError(f"unknown format {self.format!r}; choose one of {', '.join(FORMATS)}")
        checks = [("speed", self.speed, 0.5, 2.0), ("volume", self.volume_db, -20, 20), ("pitch", self.pitch, -12, 12),
                  ("expressiveness", self.expressiveness, 0, 1.5),
                  ("sentence pause", self.sentence_pause, 0, MAX_PAUSE),
                  ("paragraph pause", self.paragraph_pause, 0, MAX_PAUSE)]
        for name, value, low, high in checks:
            if not low <= value <= high:
                raise UserError(f"{name} must be between {low} and {high} (got {value})")
        if self.sample_rate and self.sample_rate not in SAMPLE_RATES:
            raise UserError(f"sample rate must be one of {', '.join(map(str, SAMPLE_RATES))}")
        return self

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Result:
    path: Path
    seconds: float
    sample_rate: int
    format: str


# --- text --------------------------------------------------------------------------------------------------------

def parse(text: str) -> list[tuple[str, object]]:
    """The text as ("speak", str) and ("pause", seconds) items, with sentence/paragraph pauses not yet sized:
    ("line", None) between lines and ("paragraph", None) between paragraphs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    items: list[tuple[str, object]] = []
    paragraphs = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    for pi, paragraph in enumerate(paragraphs):
        if pi:
            items.append(("paragraph", None))
        lines = [l for l in paragraph.split("\n") if l.strip()]
        for li, line in enumerate(lines):
            if li:
                items.append(("line", None))
            pos = 0
            for m in PAUSE_RE.finditer(line):
                before = line[pos:m.start()].strip()
                if before:
                    items.append(("speak", before))
                seconds = float(m.group(1).replace(",", "."))
                if (m.group(2) or "s").lower() == "ms":
                    seconds /= 1000
                items.append(("pause", min(seconds, MAX_PAUSE)))
                pos = m.end()
            rest = line[pos:].strip()
            if rest:
                items.append(("speak", rest))
    return items


def units(text: str) -> int:
    """How many pieces will be spoken, for progress reporting."""
    return sum(1 for kind, _ in parse(text) if kind == "speak")


# --- synthesis ---------------------------------------------------------------------------------------------------

_VOICES: dict[tuple[str, int], object] = {}


def load_voice(model: Model):
    """A loaded PiperVoice, kept in memory (loading takes ~1 s)."""
    from piper import PiperVoice

    key = (str(model.onnx), model.onnx.stat().st_mtime_ns)
    if key not in _VOICES:
        if len(_VOICES) >= 4:
            _VOICES.pop(next(iter(_VOICES)))
        _VOICES[key] = PiperVoice.load(model.onnx, config_path=model.config_path)
    return _VOICES[key]


def speaker_id(model: Model, speaker: str | None) -> int | None:
    if not model.speakers:
        if speaker not in (None, "", "0", 0):
            raise UserError(f"{model.name} has a single speaker; leave the speaker empty")
        return None
    if speaker is None or speaker == "":
        return 0
    if speaker in model.speakers:
        return model.speakers.index(speaker)
    if str(speaker).isdigit() and int(speaker) < len(model.speakers):
        return int(speaker)
    raise UserError(f"{model.name} has no speaker {speaker!r} ({len(model.speakers)} speakers: "
                    f"{', '.join(model.speakers[:10])}{'…' if len(model.speakers) > 10 else ''})")


def synthesis_config(model: Model, options: Options):
    from piper import SynthesisConfig

    inference = model.inference
    return SynthesisConfig(
        speaker_id=speaker_id(model, options.speaker),
        length_scale=inference.get("length_scale", 1.0) / options.speed,
        noise_scale=inference.get("noise_scale", 0.667) * options.expressiveness,
        noise_w_scale=inference.get("noise_w", 0.8) * options.expressiveness,
    )


def render(model: Model, text: str, options: Options,
           progress: Callable[[int, int], None] | None = None) -> np.ndarray:
    """Speak the text into one int16 array at the model's sample rate, pauses included."""
    if options.use_lexicon:
        from voice_clone import lexicon

        text = lexicon.apply(text, model.language_family)
    items = parse(text)
    total = sum(1 for kind, _ in items if kind == "speak")
    if not total:
        raise UserError("there is nothing to say: the text is empty")
    voice = load_voice(model)
    config = synthesis_config(model, options)
    rate = model.sample_rate

    def silence(seconds: float) -> np.ndarray:
        return np.zeros(int(round(seconds * rate)), dtype=np.int16)

    parts: list[np.ndarray] = []
    done = 0
    for kind, value in items:
        if kind == "speak":
            chunks = [c.audio_int16_array for c in voice.synthesize(value, config)]
            for i, chunk in enumerate(chunks):
                if i:
                    parts.append(silence(options.sentence_pause))
                parts.append(chunk.astype(np.int16).reshape(-1))
            done += 1
            if progress:
                progress(done, total)
        elif kind == "pause":
            parts.append(silence(value))
        elif kind == "line":
            parts.append(silence(options.sentence_pause))
        elif kind == "paragraph":
            parts.append(silence(options.paragraph_pause))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)


def output_rate(options: Options, model_rate: int) -> int:
    """The requested (or the model's) sample rate, moved to the nearest one the codec accepts (Opus: up to 48 kHz)."""
    allowed_rates = FORMATS[options.format][3]
    rate = options.sample_rate or model_rate
    if allowed_rates and rate not in allowed_rates:
        rate = min(allowed_rates, key=lambda r: (r < rate, abs(r - rate)))  # nearest allowed, preferring higher
    return rate


def ffmpeg_args(src: Path, dst: Path, options: Options, model_rate: int) -> list[str]:
    """The single ffmpeg pass that turns the raw WAV into the requested file."""
    _, codec, default_bitrate, _ = FORMATS[options.format]
    filters = []
    if options.pitch:
        filters.append(f"rubberband=pitch={2 ** (options.pitch / 12):.6f}")
    if options.volume_db:
        filters.append(f"volume={options.volume_db:+.1f}dB")
    if options.normalize:
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")
    rate = output_rate(options, model_rate)
    args = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src)]
    if filters:
        args += ["-af", ",".join(filters)]
    args += ["-ac", "1", "-ar", str(rate), "-c:a", codec]
    if options.bitrate and options.format not in ("wav", "flac"):
        args += ["-b:a", options.bitrate]
    elif default_bitrate:
        args += ["-b:a", default_bitrate]
    elif options.format == "ogg":
        args += ["-q:a", "6"]
    return args + [str(dst)]


def encode(samples: np.ndarray, model_rate: int, options: Options, out: Path) -> Result:
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        raw = Path(tmp) / "raw.wav"
        with wave.open(str(raw), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(model_rate)
            w.writeframes(samples.tobytes())
        result = subprocess.run(ffmpeg_args(raw, out, options, model_rate), capture_output=True, text=True,
                                encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise UserError(f"ffmpeg failed: {result.stderr.strip()[-1000:]}")
    from voice_clone import audio

    return Result(path=out, seconds=audio.duration(out), sample_rate=output_rate(options, model_rate),
                  format=options.format)


def generate(model: Model, text: str, out: Path, options: Options | None = None,
             progress: Callable[[int, int], None] | None = None) -> Result:
    """Speak `text` with `model` into `out` (its extension is set from the format)."""
    options = (options or Options()).validate()
    out = Path(out).with_suffix("." + FORMATS[options.format][0])
    samples = render(model, text, options, progress)
    return encode(samples, model.sample_rate, options, out)


def preview(model: Model, text: str, out: Path, options: Options | None = None) -> Result:
    """Only the first spoken piece (line or sentence group), to try settings quickly."""
    first = next((value for kind, value in parse(text) if kind == "speak"), None)
    if first is None:
        raise UserError("there is nothing to say: the text is empty")
    return generate(model, first, out, options)


def _slug(text: str, limit: int = 40) -> str:
    import unicodedata

    ascii_text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")
    return slug[:limit].rstrip("-") or "line"


def generate_per_line(model: Model, text: str, out_dir: Path, options: Options | None = None,
                      progress: Callable[[int, int], None] | None = None) -> list[Result]:
    """One file per non-empty line (e.g. a kiosk's messages): 001-first-words.mp3, 002-..., in out_dir."""
    options = (options or Options()).validate()
    lines = [l.strip() for l in text.replace("\r\n", "\n").split("\n") if l.strip()]
    if not lines:
        raise UserError("there is nothing to say: the text is empty")
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for n, line in enumerate(lines, 1):
        out = out_dir / f"{n:03d}-{_slug(line)}"
        results.append(generate(model, line, out, options))
        if progress:
            progress(n, len(lines))
    return results


def zip_files(files: list[Path], zip_path: Path) -> Path:
    import zipfile

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as z:  # audio is already compressed
        for f in files:
            z.write(f, arcname=f.name)
    return zip_path
