"""Machine settings chosen by `voice-clone setup`, kept in settings.json at the project root (not versioned).

device: "gpu" (NVIDIA + CUDA, the fast path) or "cpu" (no GPU needed; dataset ~5x and training ~18x slower on a
16-core desktop CPU, far slower on smaller ones). VOICE_CLONE_DEVICE overrides it for one command.
"""

import functools
import os

from voice_clone.paths import PROJECT_ROOT
from voice_clone.util import read_json, write_json

SETTINGS_FILE = PROJECT_ROOT / "settings.json"
DEVICES = ("gpu", "cpu")


@functools.lru_cache(maxsize=1)
def _detected() -> str:
    from voice_clone import system

    return "gpu" if system.query_gpu() else "cpu"


def device() -> str:
    env = os.environ.get("VOICE_CLONE_DEVICE", "").lower()
    if env in DEVICES:
        return env
    saved = read_json(SETTINGS_FILE, {}).get("device")
    return saved if saved in DEVICES else _detected()


def is_cpu() -> bool:
    return device() == "cpu"


def set_device(value: str) -> None:
    if value not in DEVICES:
        raise ValueError(value)
    data = read_json(SETTINGS_FILE, {})
    data["device"] = value
    write_json(SETTINGS_FILE, data)
