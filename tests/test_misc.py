import re

import pytest

from voice_clone import audio, jobs, languages, system, train


@pytest.mark.parametrize("mib, batch", [(24564, 32), (16384, 24), (12288, 16), (8192, 12), (4096, 8)])
def test_batch_for_vram(mib, batch):
    assert train.batch_for_vram(mib) == batch


def test_cut_point_picks_last_pause_before_limit():
    silences = [(2.0, 2.4), (9.0, 9.6), (14.0, 15.0)]
    assert audio.cut_point(silences, 12) == pytest.approx(9.3)
    assert audio.cut_point(silences, 1) is None


def test_languages_are_consistent():
    for code, lang in languages.LANGUAGES.items():
        assert code == lang.code
        assert lang.checkpoint.endswith("/medium")
        assert re.fullmatch(r"[a-z]{2,3}_[A-Z]{2}", lang.locale), lang.locale
        assert lang.samples, f"{code} has no test sentences"


def test_install_command_per_package_manager():
    report = system.SystemReport(distro="x", package_manager="dnf", missing=["g++", "ninja", "ffmpeg", "ffprobe"])
    assert report.install_command() == "sudo dnf install -y ffmpeg-free gcc-c++ ninja-build"
    report.package_manager = "apt"
    assert report.install_command() == "sudo apt update && sudo apt install -y build-essential ffmpeg ninja-build"
    assert system.SystemReport(distro="x", package_manager="dnf").install_command() is None


def test_log_tail_collapses_progress_bars(voices_dir):
    from voice_clone import workspace

    voice = workspace.create("anna", "it")
    voice.logs_dir.mkdir()
    (voice.logs_dir / "job.log").write_bytes(
        b"start\nEpoch 1:  10%\rEpoch 1:  50%\rEpoch 1: 100%\n\x1b[Avalidation\ndone\n")
    assert jobs.log_tail(voice) == "start\nEpoch 1: 100%\nvalidation\ndone"


def test_best_checkpoint_has_lowest_val_mel(voices_dir):
    from voice_clone import workspace

    voice = workspace.create("anna", "it")
    ckpts = voice.training_dir / "lightning_logs" / "version_0" / "checkpoints"
    ckpts.mkdir(parents=True)
    for name in ("epoch=10-val_mel=0.4100.ckpt", "epoch=20-val_mel=0.3900.ckpt", "epoch=30-val_mel=0.3950.ckpt",
                 "last.ckpt"):
        (ckpts / name).write_bytes(b"")
    assert train.best_checkpoint(voice).name == "epoch=20-val_mel=0.3900.ckpt"
    assert train.last_checkpoint(voice).name == "last.ckpt"
    assert train.progress(voice)["best_epoch"] == 20


def test_prune_keeps_best_across_runs_and_newest_last(voices_dir):
    import os

    from voice_clone import workspace

    voice = workspace.create("anna", "it")
    names = {0: ["epoch=1-val_mel=0.5000.ckpt", "epoch=2-val_mel=0.4000.ckpt", "last.ckpt"],
             1: ["epoch=5-val_mel=0.3000.ckpt", "epoch=6-val_mel=0.4500.ckpt", "last.ckpt"]}
    for version, files in names.items():
        d = voice.training_dir / "lightning_logs" / f"version_{version}" / "checkpoints"
        d.mkdir(parents=True)
        for n, name in enumerate(files):
            (d / name).write_bytes(b"")
            os.utime(d / name, (1000 + version * 10 + n, 1000 + version * 10 + n))
    assert train.prune_checkpoints(voice, keep=2) == 3
    left = sorted(p.relative_to(voice.training_dir / "lightning_logs").as_posix()
                  for p in voice.training_dir.rglob("*.ckpt"))
    assert left == ["version_0/checkpoints/epoch=2-val_mel=0.4000.ckpt",
                    "version_1/checkpoints/epoch=5-val_mel=0.3000.ckpt", "version_1/checkpoints/last.ckpt"]
