"""Stage 4: fine-tune a Piper voice on the dataset, starting from a trained checkpoint in the same language.

Starting from an existing voice (warm start) instead of from scratch takes hours instead of days and brings the
language's pronunciation along; the model mostly has to learn the new timbre.
"""

import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from voice_clone import settings as settings_module
from voice_clone import system
from voice_clone.installer import piper_ready
from voice_clone.paths import CACHE_DIR, PIPER_PYTHON
from voice_clone.util import UserError, download, info, read_json, step, warn, write_json
from voice_clone.workspace import Voice

HF_TREE = "https://huggingface.co/api/datasets/rhasspy/piper-checkpoints/tree/main/"
HF_FILE = "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/"
ENTRY = Path(__file__).with_name("piper_train.py")


@dataclass
class Settings:
    epochs: int = 2000
    batch_size: int = 0   # 0: chosen from the GPU's memory
    num_workers: int = 4
    resume: bool = False
    init_from: Path | None = None  # start from this checkpoint instead of the language's base voice


def batch_for_vram(memory_mib: int) -> int:
    """Batch size that fits Piper medium training in the GPU's memory, with headroom for the desktop."""
    gib = memory_mib / 1024
    if gib >= 20:
        return 32
    if gib >= 14:
        return 24
    if gib >= 10:
        return 16
    if gib >= 7:
        return 12
    return 8


def batch_for_cpu(ram_gib: float) -> int:
    """On the CPU the batch costs RAM, not VRAM; 32 trains fastest per epoch when it fits."""
    return 32 if ram_gib >= 24 else 16


def base_checkpoint(voice: Voice) -> tuple[Path, dict]:
    """The language's base checkpoint and its config.json, downloaded once into the cache."""
    folder = voice.language.checkpoint
    cache = CACHE_DIR / "checkpoints" / folder
    config_path = cache / "config.json"
    ckpts = sorted(cache.glob("*.ckpt"))
    if not (ckpts and config_path.exists()):
        step(f"Downloading the base voice {folder}")
        with urllib.request.urlopen(HF_TREE + urllib.parse.quote(folder), timeout=60) as r:
            names = [e["path"].rsplit("/", 1)[1] for e in json.load(r)]
        ckpt_names = [n for n in names if n.endswith(".ckpt")]
        if not ckpt_names or "config.json" not in names:
            raise UserError(f"no checkpoint found in rhasspy/piper-checkpoints/{folder}")
        download(HF_FILE + urllib.parse.quote(f"{folder}/config.json"), config_path)
        download(HF_FILE + urllib.parse.quote(f"{folder}/{ckpt_names[0]}"), cache / ckpt_names[0])
        ckpts = [cache / ckpt_names[0]]
    return ckpts[0], json.loads(config_path.read_text())


def last_checkpoint(voice: Voice) -> Path | None:
    found = sorted(voice.training_dir.glob("lightning_logs/version_*/checkpoints/last.ckpt"),
                   key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def best_checkpoint(voice: Voice) -> Path | None:
    def val(p: Path) -> float:
        m = re.search(r"val_mel=(\d+\.\d+)\.ckpt$", p.name)
        return float(m.group(1)) if m else float("inf")

    found = sorted(voice.training_dir.glob("lightning_logs/version_*/checkpoints/epoch=*.ckpt"), key=val)
    return found[0] if found else None


def prune_checkpoints(voice: Voice, keep: int = 5) -> int:
    """Keep the `keep` best checkpoints by val_mel across all runs, plus the newest last.ckpt; delete the rest.

    Every resumed run gets its own lightning_logs/version_N with up to 5 best checkpoints and a last.ckpt of
    ~850 MB each, so without pruning a few resumes fill tens of GB. Call only while no training runs.
    Returns the number of files deleted.
    """
    def val(p: Path) -> float:
        m = re.search(r"val_mel=(\d+\.\d+)\.ckpt$", p.name)
        return float(m.group(1)) if m else float("inf")

    scored = sorted(voice.training_dir.glob("lightning_logs/version_*/checkpoints/epoch=*.ckpt"), key=val)
    lasts = sorted(voice.training_dir.glob("lightning_logs/version_*/checkpoints/last.ckpt"),
                   key=lambda p: p.stat().st_mtime)
    doomed = scored[keep:] + lasts[:-1]
    for path in doomed:
        path.unlink(missing_ok=True)
    if doomed:
        info(f"removed {len(doomed)} old checkpoints (kept the {keep} best and the latest)")
    return len(doomed)


def command(voice: Voice, settings: Settings) -> list[str]:
    metadata = voice.dataset_dir / "metadata.csv"
    if not metadata.exists():
        raise UserError(f"no dataset yet: run `voice-clone dataset {voice.name}`")
    base, config = base_checkpoint(voice)
    espeak_voice = config.get("espeak", {}).get("voice")
    if not espeak_voice:
        raise UserError(f"the base checkpoint's config.json names no espeak voice: {base.parent / 'config.json'}")
    sample_rate = config.get("audio", {}).get("sample_rate", 22050)

    batch = settings.batch_size
    on_cpu = settings_module.is_cpu()
    if not batch:
        if on_cpu:
            batch = batch_for_cpu(system.ram_gib())
            info(f"batch size {batch} (CPU mode, {system.ram_gib():.0f} GB of RAM)")
        else:
            gpu = system.query_gpu()
            batch = batch_for_vram(gpu.memory_mib) if gpu else 16
            info(f"batch size {batch}" + (f" (for {gpu.memory_mib / 1024:.0f} GB of VRAM)" if gpu else ""))

    voice.export_dir.mkdir(parents=True, exist_ok=True)
    cmd = [str(PIPER_PYTHON), str(ENTRY), "fit",
           "--data.voice_name", voice.name,
           "--data.csv_path", str(metadata),
           "--data.audio_dir", str(voice.dataset_dir / "wav"),
           "--data.espeak_voice", espeak_voice,
           "--data.cache_dir", str(voice.training_dir / "cache"),
           "--data.config_path", str(voice.onnx_config_path),
           "--data.batch_size", str(batch),
           "--data.num_workers", str(settings.num_workers),
           "--model.sample_rate", str(sample_rate),
           "--trainer.max_epochs", str(settings.epochs),
           "--trainer.default_root_dir", str(voice.training_dir),
           "--trainer.accelerator", "cpu" if on_cpu else "gpu"]
    phoneme_type = config.get("phoneme_type")
    if phoneme_type and phoneme_type != "espeak":
        cmd += ["--data.phoneme_type", phoneme_type]

    if settings.resume:
        last = last_checkpoint(voice)
        if not last:
            raise UserError("nothing to resume: no last.ckpt in training/ yet (drop --resume to start)")
        info(f"resuming from {last}")
        cmd += ["--ckpt_path", str(last)]
    else:
        if last_checkpoint(voice):
            warn("starting a new training run from the base voice; the previous run's checkpoints stay in "
                 "training/lightning_logs (use --resume to continue it instead)")
        start = base
        if settings.init_from:
            if not Path(settings.init_from).is_file():
                raise UserError(f"checkpoint not found: {settings.init_from}")
            start = Path(settings.init_from)
            info(f"starting from {start} instead of the base voice")
        cmd += ["--model.warmstart_ckpt", str(start)]
    return cmd


def run(voice: Voice, settings: Settings | None = None) -> None:
    settings = settings or Settings()
    if not piper_ready():
        raise UserError("Piper is not installed: run `voice-clone setup`")
    cmd = command(voice, settings)
    prune_checkpoints(voice)
    step(f"Training {settings.epochs} epochs (follow it with `voice-clone status {voice.name}`)")
    voice.training_dir.mkdir(parents=True, exist_ok=True)
    write_json(voice.training_dir / "run.json", {"epochs": settings.epochs, "device": settings_module.device()})
    log_path = voice.training_dir / "train.log"
    info(f"log: {log_path}")
    with open(log_path, "a", encoding="utf-8") as log:
        log.write("\n$ " + " ".join(cmd) + "\n")
        log.flush()
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""} if settings_module.is_cpu() else None
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=voice.training_dir,
                                env=env)
        try:
            for raw in iter(lambda: proc.stdout.read(4096), b""):
                text = raw.decode(errors="replace")
                log.write(text)
                log.flush()
                print(text, end="", flush=True)
        finally:
            code = proc.wait()
    prune_checkpoints(voice)
    if code != 0:
        oom = "out of memory" in log_path.read_text(encoding="utf-8", errors="replace")[-20000:]
        hint = " The GPU ran out of memory: try a smaller --batch-size." if oom else ""
        raise UserError(f"training stopped with exit code {code}; see {log_path}.{hint} "
                        f"Continue with `voice-clone train {voice.name} --resume`.")
    info(f"Training finished. Next: voice-clone export {voice.name}")


def metrics(voice: Voice) -> list[dict]:
    """Per-epoch metrics recorded during training (metrics.csv), oldest first. When a run was resumed, the epochs
    it repeats replace the earlier rows."""
    path = voice.training_dir / "metrics.csv"
    if not path.exists():
        return []
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines()[1:]:
        parts = line.split(",")
        if len(parts) < 4 or not parts[0].isdigit():
            continue
        rows[int(parts[0])] = {"epoch": int(parts[0]), "val_mel": float(parts[1]) if parts[1] else None,
                               "seconds": float(parts[2]), "time": float(parts[3])}
    return [rows[e] for e in sorted(rows)]


def eta_seconds(voice: Voice, window: int = 20) -> float | None:
    """Time left in the current run: remaining epochs × the average duration of the last `window` epochs."""
    state = read_json(voice.training_dir / "progress.json")
    rows = metrics(voice)[-(window + 1):]
    if not state or not rows:
        return None
    # Wall time between consecutive epochs also counts checkpoint saving; fall back to the epoch's own duration.
    # Pairs spanning a pause (a resumed run) are skipped: their gap is far longer than the epoch itself.
    durations = [b["time"] - a["time"] for a, b in zip(rows, rows[1:])
                 if b["epoch"] == a["epoch"] + 1 and b["time"] - a["time"] <= 2 * b["seconds"] + 30]
    durations = durations or [r["seconds"] for r in rows]
    remaining = max(0, state["max_epochs"] - state["epoch"])
    return remaining * sum(durations) / len(durations)


def progress(voice: Voice) -> dict:
    """Current epoch (from the log), and the best checkpoint so far."""
    log_path = voice.training_dir / "train.log"
    epoch, finished = None, False
    if log_path.exists():
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 200_000))
            current_run = f.read().decode(errors="replace").rsplit("\n$ ", 1)[-1]
        found = re.findall(r"Epoch (\d+):", current_run)
        epoch = int(found[-1]) if found else None
        finished = "`Trainer.fit` stopped" in current_run
    best = best_checkpoint(voice)
    m = re.search(r"epoch=(\d+)-val_mel=(\d+\.\d+)", best.name) if best else None
    return {"epoch": epoch, "finished": finished, "best_epoch": int(m.group(1)) if m else None,
            "best_val_mel": float(m.group(2)) if m else None, "last": last_checkpoint(voice)}
