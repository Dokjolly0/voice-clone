"""Platform differences, with Windows simulated on Linux (and the other way round in the Windows CI job)."""

import sys
from pathlib import Path

import pytest

from voice_clone import host, installer, system, util


@pytest.fixture
def windows(monkeypatch, tmp_path):
    monkeypatch.setattr(host, "IS_WINDOWS", True)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))
    return tmp_path / "AppData" / "Local"


@pytest.fixture
def linux(monkeypatch):
    monkeypatch.setattr(host, "IS_WINDOWS", False)


def test_windows_paths(windows):
    venv = Path("piper") / ".venv"
    assert host.venv_python(venv) == venv / "Scripts" / "python.exe"
    assert host.venv_script(venv, "cythonize") == venv / "Scripts" / "cythonize.exe"
    assert host.ext_suffixes() == (".pyd",)
    assert host.speech_dir() == windows / "Programs" / "speech.cpp" / "current"
    assert host.speech_dir() in host.extra_bin_dirs()
    assert host.cache_dir() == windows / "voice-clone" / "cache"


def test_linux_paths(linux):
    venv = Path("piper") / ".venv"
    assert host.venv_python(venv) == venv / "bin" / "python"
    assert host.venv_script(venv, "cythonize") == venv / "bin" / "cythonize"
    assert host.ext_suffixes() == (".so",)
    assert host.speech_dir() is None
    assert host.extra_bin_dirs() == [Path.home() / ".local" / "bin"]
    assert host.cache_dir() == Path.home() / ".cache" / "voice-clone"


def test_speech_installer(windows):
    cmd = installer.speech_install_command()
    assert cmd[0] == "powershell" and cmd[-1].endswith("install.ps1 | iex")


def test_speech_installer_linux(linux):
    assert installer.speech_install_command() == ["sh", "-c", f"curl -fsSL {installer.SPEECH_INSTALLER} | sh"]


def test_torch_backend(monkeypatch):
    monkeypatch.setattr(host, "IS_WINDOWS", True)
    assert installer.torch_backend("gpu") == ["--torch-backend", "auto"]
    assert installer.torch_backend("cpu") == ["--torch-backend", "cpu"]
    monkeypatch.setattr(host, "IS_WINDOWS", False)
    assert installer.torch_backend("gpu") == []  # PyPI's Linux build already has CUDA
    assert installer.torch_backend("cpu") == ["--torch-backend", "cpu"]


def test_winget_install_commands(windows):
    assert system.detect_package_manager() == "winget"
    report = system.SystemReport(distro="Windows 11", package_manager="winget",
                                 missing=["ffmpeg", "ffprobe", "git", "msvc"])
    assert report.missing_packages == ["Git.Git", "Gyan.FFmpeg", "Microsoft.VisualStudio.2022.BuildTools"]
    lines = [l.strip() for l in report.install_command().splitlines()]
    assert lines[1] == "winget install -e --id Gyan.FFmpeg"
    assert "Microsoft.VisualStudio.Workload.VCTools" in lines[2]
    assert "sudo" not in report.install_command()


def test_requirements_per_system(monkeypatch):
    monkeypatch.setattr(host, "IS_WINDOWS", True)
    on_windows = {r for r, p in system.REQUIREMENTS.items() if system.needed_here(p)}
    monkeypatch.setattr(host, "IS_WINDOWS", False)
    on_linux = {r for r, p in system.REQUIREMENTS.items() if system.needed_here(p)}
    assert on_windows == {"git", "ffmpeg", "ffprobe", "lib:vulkan", "msvc"}
    assert "msvc" not in on_linux and {"gcc", "cmake", "ffmpeg"} <= on_linux


def test_linux_install_command_unchanged():
    report = system.SystemReport(distro="Fedora", package_manager="dnf", missing=["ffmpeg", "g++"])
    assert report.install_command() == "sudo dnf install -y ffmpeg-free gcc-c++"


def test_which_looks_in_install_folders(monkeypatch, tmp_path):
    tool = tmp_path / ("speech.exe" if sys.platform == "win32" else "speech-test-tool")
    tool.write_text("#!/bin/sh\n")
    tool.chmod(0o755)
    monkeypatch.setattr(host, "extra_bin_dirs", lambda: [tmp_path])
    assert Path(util.which(tool.stem if sys.platform == "win32" else tool.name)) == tool


def test_run_decodes_utf8_output():
    out = util.run([sys.executable, "-c", "import sys; sys.stdout.buffer.write('perché è così'.encode())"],
                   capture=True).stdout
    assert out == "perché è così"


def test_ram_is_known():
    assert system.ram_gib() > 0.5
