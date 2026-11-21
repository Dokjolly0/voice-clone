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
