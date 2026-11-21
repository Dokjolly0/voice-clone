"""Stage 5: export a checkpoint to ONNX and speak test sentences with it.

The result is two files in export/: <locale>-<name>-medium.onnx and its .onnx.json, which must stay side by side
with the same name. They run in real time on a CPU with Piper, without a GPU and without Drama 3.
"""

import shutil
import subprocess
from pathlib import Path

from voice_clone import train
from voice_clone.installer import piper_ready
from voice_clone.paths import PIPER_PYTHON
from voice_clone.util import UserError, info, run, step
from voice_clone.workspace import Voice


def choose_checkpoint(voice: Voice, which: str) -> Path:
    if which == "last":
        ckpt = train.last_checkpoint(voice)
    elif which == "best":
        ckpt = train.best_checkpoint(voice)
    else:
        ckpt = Path(which)
        if not ckpt.is_file():
            raise UserError(f"checkpoint not found: {ckpt}")
    if not ckpt:
        raise UserError(f"no checkpoint yet: run `voice-clone train {voice.name}` first")
    return ckpt


def export(voice: Voice, checkpoint: str = "last") -> Path:
    if not piper_ready():
        raise UserError("Piper is not installed: run `voice-clone setup`")
    ckpt = choose_checkpoint(voice, checkpoint)
    if not voice.onnx_config_path.exists():
        raise UserError(f"{voice.onnx_config_path.name} is missing; Piper writes it when training starts")
    step(f"Exporting {ckpt.relative_to(voice.dir)} to ONNX")
    voice.export_dir.mkdir(parents=True, exist_ok=True)
    tmp = voice.onnx_path.with_suffix(".onnx.part")
    # Exporting while training runs is fine: it reads the checkpoint file, which Lightning replaces atomically.
    run([PIPER_PYTHON, "-m", "piper.train.export_onnx", "--checkpoint", ckpt, "--output-file", tmp], capture=True)
    tmp.replace(voice.onnx_path)
    info(f"model: {voice.onnx_path} ({voice.onnx_path.stat().st_size / 1e6:.0f} MB)")
    info(f"config: {voice.onnx_config_path}")
    return voice.onnx_path


def say(voice: Voice, text: str, out: Path) -> Path:
    if not voice.onnx_path.exists():
        raise UserError(f"no exported model yet: run `voice-clone export {voice.name}`")
    out.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([str(PIPER_PYTHON), "-m", "piper", "-m", str(voice.onnx_path), "-f", str(out), "--", text],
                            capture_output=True, text=True)
    if result.returncode != 0 or not out.exists():
        raise UserError(f"piper failed: {result.stderr.strip()[-1500:]}")
    return out


def samples(voice: Voice, extra: list[str] | None = None) -> list[Path]:
    """Speak the language's test sentences (plus any given) into samples/."""
    sentences = [*voice.language.samples, *(extra or [])]
    if voice.samples_dir.exists():
        shutil.rmtree(voice.samples_dir)
    step(f"Speaking {len(sentences)} test sentences")
    outs = []
    for n, text in enumerate(sentences, 1):
        out = say(voice, text, voice.samples_dir / f"sample-{n}.wav")
        info(f"{out.relative_to(voice.dir)}: {text}")
        outs.append(out)
    (voice.samples_dir / "sentences.txt").write_text(
        "\n".join(f"sample-{n}.wav|{t}" for n, t in enumerate(sentences, 1)) + "\n", encoding="utf-8")
    return outs


def export_and_test(voice: Voice, checkpoint: str = "last", extra: list[str] | None = None) -> None:
    export(voice, checkpoint)
    samples(voice, extra)
    info(f"\nListen to {voice.samples_dir}. Use the voice anywhere Piper runs:\n"
         f"  voice-clone say {voice.name} \"Some text.\" -o out.wav")
