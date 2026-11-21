"""Keep the computer awake while a long stage runs.

The process re-executes itself under `systemd-inhibit --what=idle:sleep`, which blocks automatic (idle) and
requested suspend until it exits, then lifts the block by itself. Closing a laptop lid can still suspend it.
"""

import os
import sys

from voice_clone.util import which

ENV_FLAG = "VOICE_CLONE_INHIBITED"


def keep_awake(reason: str) -> None:
    """Re-exec under systemd-inhibit, once. Returns only when already inhibited or when inhibiting is impossible."""
    if os.environ.get(ENV_FLAG) or os.environ.get("VOICE_CLONE_NO_INHIBIT"):
        return
    tool = which("systemd-inhibit")
    if not tool:
        return
    os.environ[ENV_FLAG] = "1"
    os.execv(tool, [tool, "--what=idle:sleep", "--who=voice-clone", f"--why={reason}",
                    sys.executable, "-m", "voice_clone", *sys.argv[1:]])
