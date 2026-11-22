"""Stage 5: export a checkpoint to ONNX and speak test sentences with it.

The result is two files in export/: <locale>-<name>-medium.onnx and its .onnx.json, which must stay side by side
with the same name. They run in real time on a CPU with Piper, without a GPU and without Drama 3.

To choose between checkpoints, `compare` exports each into export/variants/<label>/ (model + the same test
sentences), and `promote` makes one of them the voice's model.
"""

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from voice_clone import train
from voice_clone.installer import piper_ready
from voice_clone.paths import PIPER_PYTHON
from voice_clone.util import UserError, info, read_json, run, step, write_json
from voice_clone.workspace import Voice


@dataclass
class Checkpoint:
    label: str            # "last" or "epoch-1816"
    path: Path
    epoch: int | None     # 0-based, as in the file names
    val_mel: float | None


def checkpoints(voice: Voice) -> list[Checkpoint]:
    """last.ckpt and the best-by-val_mel checkpoints of every run, newest epoch first."""
    found = {}
    for path in voice.training_dir.glob("lightning_logs/version_*/checkpoints/epoch=*.ckpt"):
        m = re.match(r"epoch=(\d+)-val_mel=(\d+\.\d+)\.ckpt$", path.name)
        if m:
            epoch = int(m.group(1))
            found[epoch] = Checkpoint(f"epoch-{epoch}", path, epoch, float(m.group(2)))
    result = sorted(found.values(), key=lambda c: -c.epoch)
    last = train.last_checkpoint(voice)
    if last:
        state = read_json(voice.training_dir / "progress.json") or {}
        epoch = state["epoch"] - 1 if "epoch" in state else None
        result.insert(0, Checkpoint("last", last, epoch, state.get("val_mel")))
    return result


def resolve(voice: Voice, which: str) -> Checkpoint:
    """'last', 'best', an epoch number ('1816' or 'epoch-1816'), or a .ckpt path."""
    available = checkpoints(voice)
    if not available:
        raise UserError(f"no checkpoint yet: run `voice-clone train {voice.name}` first")
    if which == "last":
        return available[0]  # last.ckpt, or the newest epoch when there is none
    if which == "best":
        scored = [c for c in available if c.label != "last" and c.val_mel is not None]
        if not scored:
            raise UserError("no scored checkpoint yet")
        return min(scored, key=lambda c: c.val_mel)
    number = which.removeprefix("epoch-")
    if number.isdigit():
        for c in available:
            if c.label != "last" and c.epoch == int(number):
                return c
        kept = ", ".join(str(c.epoch) for c in available if c.label != "last")
        raise UserError(f"no checkpoint of epoch {number}; kept epochs: {kept}")
    path = Path(which)
    if not path.is_file():
        raise UserError(f"checkpoint not found: {which}")
    return Checkpoint(path.stem, path, None, None)


def _export_onnx(voice: Voice, ckpt: Path, onnx: Path) -> None:
    if not piper_ready():
        raise UserError("Piper is not installed: run `voice-clone setup`")
    if not voice.onnx_config_path.exists():
        raise UserError(f"{voice.onnx_config_path.name} is missing; Piper writes it when training starts")
    onnx.parent.mkdir(parents=True, exist_ok=True)
    tmp = onnx.with_suffix(".onnx.part")
    # Exporting while training runs is fine: it reads the checkpoint file, which Lightning replaces atomically.
    run([PIPER_PYTHON, "-m", "piper.train.export_onnx", "--checkpoint", ckpt, "--output-file", tmp], capture=True)
    tmp.replace(onnx)
    config = onnx.with_name(onnx.name + ".json")
    if config != voice.onnx_config_path:
        shutil.copy(voice.onnx_config_path, config)


def speak(onnx: Path, text: str, out: Path) -> Path:
    if not onnx.exists():
        raise UserError(f"model not found: {onnx}")
    out.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([str(PIPER_PYTHON), "-m", "piper", "-m", str(onnx), "-f", str(out), "--", text],
                            capture_output=True, text=True)
    if result.returncode != 0 or not out.exists():
        raise UserError(f"piper failed: {result.stderr.strip()[-1500:]}")
    return out


def _samples(voice: Voice, onnx: Path, out_dir: Path, extra: list[str] | None) -> list[Path]:
    sentences = [*voice.language.samples, *(extra or [])]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    step(f"Speaking {len(sentences)} test sentences")
    outs = []
    for n, text in enumerate(sentences, 1):
        out = speak(onnx, text, out_dir / f"sample-{n}.wav")
        info(f"samples/{out.name}: {text}")
        outs.append(out)
    (out_dir / "sentences.txt").write_text(
        "\n".join(f"sample-{n}.wav|{t}" for n, t in enumerate(sentences, 1)) + "\n", encoding="utf-8")
    return outs


def read_samples(sample_dir: Path) -> list[tuple[Path, str]]:
    """(wav, sentence) pairs of a samples folder, in order."""
    listing = sample_dir / "sentences.txt"
    if not listing.exists():
        return []
    pairs = []
    for line in listing.read_text(encoding="utf-8").splitlines():
        if "|" in line:
            name, text = line.split("|", 1)
            if (sample_dir / name).exists():
                pairs.append((sample_dir / name, text))
    return pairs


def export(voice: Voice, checkpoint: str = "last") -> Path:
    ckpt = resolve(voice, checkpoint)
    step(f"Exporting {ckpt.path.relative_to(voice.dir) if voice.dir in ckpt.path.parents else ckpt.path} to ONNX")
    _export_onnx(voice, ckpt.path, voice.onnx_path)
    write_json(voice.export_dir / "source.json", {"checkpoint": ckpt.label, "epoch": ckpt.epoch,
                                                  "val_mel": ckpt.val_mel})
    info(f"model: {voice.onnx_path} ({voice.onnx_path.stat().st_size / 1e6:.0f} MB)")
    info(f"config: {voice.onnx_config_path}")
    return voice.onnx_path


def say(voice: Voice, text: str, out: Path) -> Path:
    """Speak with the voice's exported model through the studio's engine (no training environment needed)."""
    if not voice.onnx_path.exists():
        raise UserError(f"no exported model yet: run `voice-clone export {voice.name}`")
    from voice_clone import models, synthesis

    fmt = out.suffix.lstrip(".").lower()
    options = synthesis.Options(format=fmt if fmt in synthesis.FORMATS else "wav")
    return synthesis.generate(models.get(f"voice:{voice.name}"), text, out, options).path


def export_and_test(voice: Voice, checkpoint: str = "last", extra: list[str] | None = None) -> None:
    export(voice, checkpoint)
    _samples(voice, voice.onnx_path, voice.samples_dir, extra)
    info(f"\nListen to {voice.samples_dir}. Use the voice anywhere Piper runs:\n"
         f"  voice-clone say {voice.name} \"Some text.\" -o out.wav")


# --- comparing checkpoints -------------------------------------------------------------------------------------------

def variants_dir(voice: Voice) -> Path:
    return voice.export_dir / "variants"


def variant_onnx(voice: Voice, label: str) -> Path:
    return variants_dir(voice) / label / f"{voice.model_name}.onnx"


def export_variant(voice: Voice, ckpt: Checkpoint) -> Path:
    """export/variants/<label>/: the checkpoint as ONNX and the test sentences spoken by it."""
    vdir = variants_dir(voice) / ckpt.label
    onnx = variant_onnx(voice, ckpt.label)
    meta = read_json(vdir / "source.json", {})
    fresh = (onnx.exists() and meta.get("checkpoint_mtime") == ckpt.path.stat().st_mtime_ns
             and (vdir / "samples" / "sentences.txt").exists())
    if fresh:
        info(f"{ckpt.label}: already exported")
        return vdir
    step(f"Exporting {ckpt.label} for comparison")
    _export_onnx(voice, ckpt.path, onnx)
    _samples(voice, onnx, vdir / "samples", None)
    write_json(vdir / "source.json", {"checkpoint": ckpt.label, "epoch": ckpt.epoch, "val_mel": ckpt.val_mel,
                                      "checkpoint_mtime": ckpt.path.stat().st_mtime_ns})
    return vdir


def compare(voice: Voice, first: str, second: str) -> tuple[Path, Path]:
    a, b = resolve(voice, first), resolve(voice, second)
    if a.path == b.path:
        raise UserError("choose two different checkpoints")
    dirs = export_variant(voice, a), export_variant(voice, b)
    write_json(variants_dir(voice) / "comparison.json", {"a": a.label, "b": b.label})
    info(f"\nCompare {dirs[0] / 'samples'} with {dirs[1] / 'samples'}, then keep one with:\n"
         f"  voice-clone promote {voice.name} {a.label}   (or {b.label})")
    return dirs


def promote(voice: Voice, label: str) -> None:
    """Make an exported variant the voice's model (export/ and samples/)."""
    vdir = variants_dir(voice) / label
    onnx = variant_onnx(voice, label)
    if not onnx.exists():
        raise UserError(f"no exported variant {label!r}; run `voice-clone compare` first")
    voice.export_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(onnx, voice.onnx_path)
    if voice.samples_dir.exists():
        shutil.rmtree(voice.samples_dir)
    shutil.copytree(vdir / "samples", voice.samples_dir)
    meta = read_json(vdir / "source.json", {})
    meta.pop("checkpoint_mtime", None)
    write_json(voice.export_dir / "source.json", meta)
    info(f"{label} is now the model of {voice.name}: {voice.onnx_path}")
