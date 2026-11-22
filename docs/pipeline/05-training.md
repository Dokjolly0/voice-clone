# Stage 4: training

Modules: `train.py` (runs Piper), `piper_train.py` (Piper's entry point with this project's callbacks). Command:
`train NAME [--epochs N] [--batch-size N] [--num-workers N] [--resume] [--init-from CKPT]`. Keeps the computer awake.

Piper is a VITS model. Training from scratch takes days; instead voice-clone **fine-tunes** an existing Piper voice
in the same language (warm start), which brings the language's pronunciation along, so the model mostly has to
learn the new timbre.

## What `train.run` does

1. Checks that Piper's environment is ready (`installer.piper_ready()`).
2. Builds the command (`train.command`):
   - requires `dataset/metadata.csv`;
   - **base checkpoint** (`base_checkpoint`): the language's checkpoint folder in
     `rhasspy/piper-checkpoints` (e.g. `it/it_IT/paola/medium`), downloaded once into
     `<cache>/checkpoints/<folder>/` (`config.json` + the `.ckpt`); its `config.json` gives the espeak voice, the
     sample rate and the phoneme type;
   - **batch size**: `--batch-size`, else from the GPU's VRAM (`batch_for_vram`: ≥ 20 GB → 32, ≥ 14 → 24,
     ≥ 10 → 16, ≥ 7 → 12, else 8) or, in CPU mode, the RAM (`batch_for_cpu`: ≥ 24 GB → 32, else 16);
   - the starting point: `--resume` → `--ckpt_path <newest last.ckpt>` (fails if there is none); otherwise
     `--model.warmstart_ckpt <base>` or `<--init-from>` (made absolute, because Piper runs in the training folder).
     Starting a new run while an old one exists warns that the old checkpoints stay.
3. Deletes old checkpoints (`prune_checkpoints`, see below).
4. Writes `training/run.json` (`{"epochs": N, "device": "gpu"}`).
5. Runs Piper with its output streamed to the terminal (or job log) and appended to `training/train.log` (each run
   starts with a `$ <command>` line).
6. Prunes again and reports; a non-zero exit becomes an error that suggests a smaller `--batch-size` when the log
   says "out of memory", and `--resume`.

The command (abridged):

```
<piper python> piper_train.py fit
  --data.voice_name anna --data.csv_path dataset/metadata.csv --data.audio_dir dataset/wav
  --data.espeak_voice it --data.cache_dir training/cache --data.config_path export/it_IT-anna-medium.onnx.json
  --data.batch_size 32 --data.num_workers 4 --model.sample_rate 22050
  --trainer.max_epochs 2000 --trainer.default_root_dir training --trainer.accelerator gpu
  --model.warmstart_ckpt <base.ckpt>            (or --ckpt_path <last.ckpt> to resume)
```

Piper writes the model's `.onnx.json` configuration into `export/` when training starts; export needs it.

In CPU mode the accelerator is `cpu` and `CUDA_VISIBLE_DEVICES` is empty.

## `piper_train.py`

Piper's own `python -m piper.train` also keeps the best checkpoints by `val_mos`, a quality score from UTMOS, which
needs torchaudio. torchaudio has no build for the PyTorch version Piper installs, so the score is never logged and
Lightning stops at the end of the first epoch. `piper_train.py` runs the same `VitsLightningCLI` with its own
callbacks:

| Callback | Keeps |
|---|---|
| `ModelCheckpoint(monitor="val_mel", save_top_k=5)` | The 5 best checkpoints of the run by validation mel loss, named `epoch=<n>-val_mel=<x>.ckpt` |
| `ModelCheckpoint(monitor=None, save_top_k=1, every_n_epochs=1, filename="last")` | `last.ckpt`, rewritten every epoch |
| `EpochMetrics` | After every epoch: appends `epoch,val_mel,seconds,time` to `training/metrics.csv` and rewrites `training/progress.json` |

`progress.json`: `{"epoch": 447, "max_epochs": 500, "epoch_seconds": 9.4, "val_mel": 0.2997, "updated": 1795317223.6}`
(`epoch` is 1-based here; file names are 0-based). The file is written to a temporary name and renamed; on Windows
the rename is retried while a reader has it open.

It also enables TF32 matmul and cuDNN for speed.

## Checkpoints and pruning

Every run (including every resume) gets its own `training/lightning_logs/version_N/checkpoints/` with up to 5 best
checkpoints and a `last.ckpt` of ~850 MB each. `prune_checkpoints(keep=5)` keeps the 5 best by val_mel **across all
runs** plus the newest `last.ckpt`, and deletes the rest. It runs before and after training, never during.

## Progress and time left

- `train.progress` (used by `status`): the current epoch from the current run's part of `train.log`, whether the
  run finished (`Trainer.fit` stopped), and the best checkpoint (epoch and val_mel).
- `train.metrics`: the rows of `metrics.csv`; when a run is resumed, the epochs it repeats replace the earlier rows.
- `train.eta_seconds`: remaining epochs × the average wall time of the last 20 epochs; pairs of epochs spanning a
  pause (a resume) are skipped, falling back to the epochs' own durations.

## val_mel

The validation mel-spectrogram loss: how far the model's output is from the real clips of the validation split.
Lower is better. It drops fast at first and then flattens; past that point the voice mostly improves in cleanness.
It is only comparable between runs on the **same** dataset: a different dataset has a different validation split.
Judge by ear: export the best and the last checkpoint and compare them ([stage 5](06-export.md)).

## Resuming and continuing

- Interrupted (Ctrl-C, `stop`, reboot, crash): `train NAME --resume` continues from the newest `last.ckpt`.
  `run NAME` does this by itself.
- Finished, but you want more: `train NAME --resume --epochs 3000` continues to 3000.
- Starting over from another voice's checkpoint: `train NAME --init-from path/to/epoch=…ckpt`.

On Windows, a job's `stop` interrupts voice-clone, which then terminates Piper's process; the epoch in progress is
lost and the previous `last.ckpt` is kept. On Linux Piper gets the same Ctrl-C and stops by itself.

## Timing (RTX 4090)

About 9–11 s per epoch for an hour of audio at batch 32, so 2000 epochs take about 5–6 hours and 500 epochs from an
already trained voice about 1.5 hours. On a 16-core CPU an epoch takes ~18× longer.
