"""Differences between Linux and Windows, in one place: virtual environment layout, compiled extension suffixes,
where tools install themselves, and the cache folder."""

import os
import sys
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"


def _local_appdata() -> Path:
    return Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")


def venv_python(venv: Path) -> Path:
    """The Python interpreter of a virtual environment."""
    return venv / "Scripts" / "python.exe" if IS_WINDOWS else venv / "bin" / "python"


def venv_script(venv: Path, name: str) -> Path:
    """A console script installed in a virtual environment (cythonize, voice-clone...)."""
    return venv / "Scripts" / f"{name}.exe" if IS_WINDOWS else venv / "bin" / name


def ext_suffixes() -> tuple[str, ...]:
    """File suffixes of compiled Python extensions."""
    return (".pyd",) if IS_WINDOWS else (".so",)


def local_bin() -> Path:
    """Where uv installs itself (and speech.cpp links its binary on Linux)."""
    return Path.home() / ".local" / "bin"


def speech_dir() -> Path | None:
    """The folder of speech.exe installed by speech.cpp's install.ps1 (Windows only)."""
    return _local_appdata() / "Programs" / "speech.cpp" / "current" if IS_WINDOWS else None


def extra_bin_dirs() -> list[Path]:
    """Folders searched for tools besides PATH: they are added to PATH only for new shells after installing."""
    dirs = [local_bin()]
    if IS_WINDOWS:
        dirs.append(speech_dir())
    return dirs


def cache_dir() -> Path:
    if IS_WINDOWS:
        return _local_appdata() / "voice-clone" / "cache"
    return Path.home() / ".cache" / "voice-clone"
