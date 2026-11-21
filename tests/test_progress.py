import json

import pytest


@pytest.fixture
def voice(voices_dir):
    from voice_clone import workspace

    v = workspace.create("anna", "it")
    v.logs_dir.mkdir(parents=True)
    return v


def write_job(voice, args, log):
    (voice.logs_dir / "job.json").write_text(json.dumps({"pid": 999999999, "args": args,
                                                          "started": "2026-01-01T00:00:00"}))
    (voice.logs_dir / "job.log").write_text(log)


def test_progress_reads_generation_counts_and_eta(voice):
    from voice_clone import jobs

    write_job(voice, ["dataset", "anna"], "==> Generating 20 clips with Drama 3 (loading the model takes a minute)\n"
              "[1/20] clip 00000 seed 1: 0.1 min of audio, RTFx 1.60, ~5 min left\n"
              "[7/20] clip 00006 seed 601: 0.4 min of audio, RTFx 1.70, ~3 min left\n")
    p = jobs.progress(voice)
    assert (p["phase"], p["stage"], p["done"], p["total"], p["eta"]) == ("generating", 3, 7, 20, 180)


def test_progress_before_first_clip_is_loading_the_model(voice):
    from voice_clone import jobs

    write_job(voice, ["dataset", "anna"], "==> Generating 20 clips with Drama 3 (loading the model takes a minute)\n")
    assert jobs.progress(voice)["phase"] == "loading_model"


def test_progress_reports_errors_and_interruptions(voice):
    from voice_clone import jobs

    write_job(voice, ["train", "anna"], "==> Training 2000 epochs\nerror: training stopped with exit code 1\n")
    p = jobs.progress(voice)
    assert not p["running"] and p["error"] == "training stopped with exit code 1"

    write_job(voice, ["train", "anna"], "==> Training 2000 epochs\ninterrupted; run the same command again\n")
    assert jobs.progress(voice)["error"] == "interrupted"

    write_job(voice, ["export", "anna"], "==> Speaking 2 test sentences\nsamples/sample-1.wav: a\n"
              "samples/sample-2.wav: b\n")
    p = jobs.progress(voice)
    assert p["finished"] and (p["phase"], p["done"], p["total"]) == ("speaking", 2, 2)


def test_steps_lock_until_the_previous_stage_is_done(voice):
    from voice_clone import pipeline

    states = [s["state"] for s in pipeline.steps(voice)]
    assert states == ["todo", "locked", "locked", "locked", "locked"]
    assert pipeline.next_step(pipeline.steps(voice)) == 1


def test_eta_skips_the_gap_before_a_resume(voice):
    from voice_clone import train

    voice.training_dir.mkdir(parents=True)
    rows = ["epoch,val_mel,seconds,time", "0,0.5,8,1000", "1,0.5,8,1010", "2,0.5,8,1020",
            "3,0.5,8,9000", "4,0.5,8,9010"]  # resumed hours later at epoch 3
    (voice.training_dir / "metrics.csv").write_text("\n".join(rows) + "\n")
    (voice.training_dir / "progress.json").write_text(json.dumps({"epoch": 5, "max_epochs": 15}))
    assert train.eta_seconds(voice) == pytest.approx(100)


def test_resolve_checkpoints(voice):
    from voice_clone import export
    from voice_clone.util import UserError

    d = voice.training_dir / "lightning_logs" / "version_0" / "checkpoints"
    d.mkdir(parents=True)
    for name in ("epoch=10-val_mel=0.4100.ckpt", "epoch=20-val_mel=0.3900.ckpt", "last.ckpt"):
        (d / name).write_bytes(b"")
    assert export.resolve(voice, "last").label == "last"
    assert export.resolve(voice, "best").label == "epoch-20"
    assert export.resolve(voice, "10").path.name == "epoch=10-val_mel=0.4100.ckpt"
    assert export.resolve(voice, "epoch-20").val_mel == 0.39
    with pytest.raises(UserError):
        export.resolve(voice, "15")
