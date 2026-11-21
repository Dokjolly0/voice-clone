"""Where everything lives. Override the voices folder with VOICE_CLONE_VOICES and the cache with VOICE_CLONE_CACHE."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VOICES_DIR = Path(os.environ.get("VOICE_CLONE_VOICES", PROJECT_ROOT / "voices"))
CACHE_DIR = Path(os.environ.get("VOICE_CLONE_CACHE", Path.home() / ".cache" / "voice-clone"))

THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
PIPER_DIR = THIRD_PARTY_DIR / "piper1-gpl"
PIPER_PYTHON = PIPER_DIR / ".venv" / "bin" / "python"

# Piper training code, pinned to a commit that is known to train correctly with this project.
PIPER_REPO = "https://github.com/OHF-voice/piper1-gpl.git"
PIPER_REV = "39ac0a04ff01120b1cb7867d21723fa320305b03"

LOCAL_BIN = Path.home() / ".local" / "bin"
