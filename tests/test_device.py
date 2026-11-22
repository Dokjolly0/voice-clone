import pytest

from voice_clone import system, train
from voice_clone.util import UserError


@pytest.fixture
def isolated_settings(tmp_path, monkeypatch):
    from voice_clone import settings

    monkeypatch.setattr(settings, "SETTINGS_FILE", tmp_path / "settings.json")
    monkeypatch.delenv("VOICE_CLONE_DEVICE", raising=False)
    settings._detected.cache_clear()
    yield settings
    settings._detected.cache_clear()


def test_device_is_saved_and_env_overrides(isolated_settings, monkeypatch):
    settings = isolated_settings
    monkeypatch.setattr(system, "query_gpu", lambda: None)
    assert settings.device() == "cpu"  # detected: no GPU
    settings.set_device("gpu")
    assert settings.device() == "gpu"
    monkeypatch.setenv("VOICE_CLONE_DEVICE", "cpu")
    assert settings.is_cpu()
    with pytest.raises(ValueError):
        settings.set_device("tpu")


def test_choose_device(monkeypatch):
    from voice_clone import installer

    gpu = system.GpuInfo(name="NVIDIA GeForce RTX 4090", memory_mib=24564, driver="1")
    monkeypatch.setattr(system, "query_gpu", lambda: gpu)
    assert installer.choose_device("auto") == "gpu"
    assert installer.choose_device("cpu") == "cpu"
    monkeypatch.setattr(system, "query_gpu", lambda: None)
    assert installer.choose_device("auto") == "cpu"
    with pytest.raises(UserError):
        installer.choose_device("gpu")


def test_cpu_mode_does_not_need_vulkan(monkeypatch):
    monkeypatch.setattr(system.ctypes.util, "find_library", lambda name: None)
    assert "lib:vulkan" in system.check("gpu").missing
    assert "lib:vulkan" not in system.check("cpu").missing


@pytest.mark.parametrize("ram, batch", [(62, 32), (24, 32), (16, 16)])
def test_batch_for_cpu(ram, batch):
    assert train.batch_for_cpu(ram) == batch


def test_speech_gets_cpu_device_only_in_cpu_mode(monkeypatch):
    from voice_clone import installer

    monkeypatch.setenv("VOICE_CLONE_DEVICE", "cpu")
    assert installer.device_args() == ["--device", "cpu"]
    monkeypatch.setenv("VOICE_CLONE_DEVICE", "gpu")
    assert installer.device_args() == []


def test_inference_only_checks_just_ffmpeg(monkeypatch):
    monkeypatch.setattr(system, "_present", lambda req: False)
    full = system.check("gpu").missing
    assert {"git", "lib:vulkan"} <= set(full)
    missing = system.check("gpu", inference_only=True).missing
    assert set(missing) == {"ffmpeg", "ffprobe"}


def test_rubberband_is_found_in_ffmpeg_filters(monkeypatch):
    from voice_clone import installer

    rubberband = " ... rubberband        A->A       Apply time-stretching and pitch-shifting.\n"
    volume = " ... volume            A->A       Change input volume.\n"
    for out, expected in [(rubberband + volume, True), (volume, False)]:
        monkeypatch.setattr(installer.subprocess, "run", lambda *a, out=out, **k: type("R", (), {"stdout": out})())
        assert installer.ffmpeg_has_rubberband() is expected


def test_setup_inference_only_rejects_training_options(monkeypatch):
    from voice_clone import cli, installer

    called = []
    monkeypatch.setattr(installer, "setup_inference", lambda: called.append("inference"))
    monkeypatch.setattr(installer, "setup", lambda **k: called.append("full"))
    cli.build_parser().parse_args(["setup", "--inference-only"]).func(
        cli.build_parser().parse_args(["setup", "--inference-only"]))
    assert called == ["inference"]
    args = cli.build_parser().parse_args(["setup", "--inference-only", "--device", "cpu"])
    with pytest.raises(UserError):
        args.func(args)
