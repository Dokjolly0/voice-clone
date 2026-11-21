"""Piper's training entry point with this project's checkpoint callbacks.

Runs inside Piper's own environment (third_party/piper1-gpl/.venv), so it imports nothing from voice_clone.
Arguments are Piper's: `python piper_train.py fit --data.csv_path ... --model.warmstart_ckpt ...`.

Why not `python -m piper.train`: its default callbacks also keep the best checkpoints by `val_mos`, a quality
score computed by UTMOS, which needs torchaudio. torchaudio has no build for the PyTorch Piper installs, so the
score is never logged and Lightning stops with an error at the end of the first epoch. These callbacks keep the
5 best checkpoints by validation mel loss plus last.ckpt, rewritten every epoch.
"""

import logging

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from piper.train.__main__ import VitsLightningCLI
from piper.train.vits.dataset import VitsDataModule
from piper.train.vits.lightning import VitsModel


def callbacks() -> list:
    return [
        ModelCheckpoint(monitor="val_mel", mode="min", save_top_k=5, save_last=False,
                        filename="epoch={epoch}-val_mel={val_mel:.4f}", auto_insert_metric_name=False),
        ModelCheckpoint(monitor=None, save_top_k=1, every_n_epochs=1, filename="last",
                        enable_version_counter=False),
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
