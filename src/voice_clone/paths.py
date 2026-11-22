"""Where everything lives. Override with VOICE_CLONE_VOICES (voices), VOICE_CLONE_STUDIO (studio data: models,
dictionaries, history) and VOICE_CLONE_CACHE (downloads)."""

import os
from pathlib import Path

from voice_clone import host

PROJECT_ROOT = Path(__file__).resolve().parents[2]
VOICES_DIR = Path(os.environ.get("VOICE_CLONE_VOICES", PROJECT_ROOT / "voices"))
CACHE_DIR = Path(os.environ.get("VOICE_CLONE_CACHE", host.cache_dir()))
STUDIO_DIR = Path(os.environ.get("VOICE_CLONE_STUDIO", PROJECT_ROOT / "studio"))

THIRD_PARTY_DIR = PROJECT_ROOT / "third_party"
PIPER_DIR = THIRD_PARTY_DIR / "piper1-gpl"
PIPER_PYTHON = host.venv_python(PIPER_DIR / ".venv")

# Piper training code, pinned to a commit that is known to train correctly with this project.
PIPER_REPO = "https://github.com/OHF-voice/piper1-gpl.git"
PIPER_REV = "39ac0a04ff01120b1cb7867d21723fa320305b03"

LOCAL_BIN = host.local_bin()
