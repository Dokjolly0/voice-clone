"""Piper's training entry point with this project's checkpoint callbacks.

Runs inside Piper's own environment (third_party/piper1-gpl/.venv), so it imports nothing from voice_clone.
Arguments are Piper's: `python piper_train.py fit --data.csv_path ... --model.warmstart_ckpt ...`.

Why not `python -m piper.train`: its default callbacks also keep the best checkpoints by `val_mos`, a quality
score computed by UTMOS, which needs torchaudio. torchaudio has no build for the PyTorch Piper installs, so the
score is never logged and Lightning stops with an error at the end of the first epoch. These callbacks keep the
5 best checkpoints by validation mel loss plus last.ckpt, rewritten every epoch, and record per-epoch metrics.
"""

import json
import logging
import os
import time
from pathlib import Path

import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from piper.train.__main__ import VitsLightningCLI
from piper.train.vits.dataset import VitsDataModule
from piper.train.vits.lightning import VitsModel


class EpochMetrics(Callback):
    """After every epoch, append `epoch,val_mel,seconds,time` to metrics.csv and rewrite progress.json in the
    trainer's root dir, so voice-clone can show progress, a time estimate and a quality chart."""

    def __init__(self):
        self.epoch_start = None

    def on_train_epoch_start(self, trainer, pl_module):
        self.epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.global_rank != 0 or self.epoch_start is None:
            return
        root = Path(trainer.default_root_dir)
        now = time.time()
        seconds = now - self.epoch_start
        val = trainer.callback_metrics.get("val_mel")
        val_mel = float(val) if val is not None else None
        metrics = root / "metrics.csv"
        new = not metrics.exists()
        with open(metrics, "a", encoding="utf-8") as f:
            if new:
                f.write("epoch,val_mel,seconds,time\n")
            f.write(f"{trainer.current_epoch},{'' if val_mel is None else f'{val_mel:.5f}'},{seconds:.2f},{now:.0f}\n")
        progress = {"epoch": trainer.current_epoch + 1, "max_epochs": trainer.max_epochs, "epoch_seconds": seconds,
                    "val_mel": val_mel, "updated": now}
        tmp = root / "progress.json.tmp"
        tmp.write_text(json.dumps(progress))
        os.replace(tmp, root / "progress.json")


def callbacks() -> list:
    return [
        ModelCheckpoint(monitor="val_mel", mode="min", save_top_k=5, save_last=False,
                        filename="epoch={epoch}-val_mel={val_mel:.4f}", auto_insert_metric_name=False),
        ModelCheckpoint(monitor=None, save_top_k=1, every_n_epochs=1, filename="last",
                        enable_version_counter=False),
        EpochMetrics(),
    ]


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    VitsLightningCLI(VitsModel, VitsDataModule,
                     trainer_defaults={"max_epochs": -1, "callbacks": callbacks()})


if __name__ == "__main__":
    main()
