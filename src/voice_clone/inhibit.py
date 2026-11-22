"""Keep the computer awake while a long stage runs.

Linux: the process re-executes itself under `systemd-inhibit --what=idle:sleep`, which blocks automatic (idle) and
requested suspend until it exits, then lifts the block by itself. Closing a laptop lid can still suspend it.

Windows: SetThreadExecutionState tells the system this process needs it awake, until the process exits. Sleep the
user asks for (Start menu, lid, power button) still happens.
"""

import os
import sys

from voice_clone import host
from voice_clone.util import which

ENV_FLAG = "VOICE_CLONE_INHIBITED"

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def keep_awake(reason: str) -> None:
    """Block automatic sleep for the rest of this process. On Linux this re-execs under systemd-inhibit, once, and
    returns only when already inhibited or when inhibiting is impossible."""
    if os.environ.get("VOICE_CLONE_NO_INHIBIT"):
        return
    if host.IS_WINDOWS:
        import ctypes

        # Per process and idempotent: no flag to pass on, a child job asks again for itself.
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
        return
    if os.environ.get(ENV_FLAG):
        return
    tool = which("systemd-inhibit")
    if not tool:
        return
    os.environ[ENV_FLAG] = "1"
    os.execv(tool, [tool, "--what=idle:sleep", "--who=voice-clone", f"--why={reason}",
                    sys.executable, "-m", "voice_clone", *sys.argv[1:]])
