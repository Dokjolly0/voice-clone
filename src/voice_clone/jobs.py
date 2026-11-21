"""Background jobs: one voice-clone command per voice, detached from the terminal (and from the web UI).

The job runs in its own session, so closing the terminal or the browser does not stop it. Its output goes to
voices/<name>/logs/job.log and its pid to logs/job.json.
"""

import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime

from voice_clone.util import UserError, read_json, write_json
from voice_clone.workspace import Voice


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # A zombie still answers kill(0); check its state.
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().split(")")[-1].split()[0] != "Z"
    except OSError:
        return False


def current(voice: Voice) -> dict | None:
    """The running job, or None."""
    job = read_json(voice.logs_dir / "job.json")
    if job and job.get("pid") and _alive(job["pid"]):
        return job
    return None


def start(voice: Voice, args: list[str]) -> dict:
    running = current(voice)
    if running:
        raise UserError(f"{voice.name} is already running `{' '.join(running['args'])}` (pid {running['pid']}); "
                        f"stop it with `voice-clone stop {voice.name}`")
    voice.logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = voice.logs_dir / "job.log"
    if log_path.exists():
        log_path.replace(voice.logs_dir / "job.previous.log")
    log = open(log_path, "wb")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen([sys.executable, "-m", "voice_clone", *args], stdin=subprocess.DEVNULL, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True, env=env)
    job = {"pid": proc.pid, "args": args, "started": datetime.now().isoformat(timespec="seconds"),
           "log": str(log_path)}
    write_json(voice.logs_dir / "job.json", job)
    return job


def stop(voice: Voice, timeout: float = 60) -> bool:
    """Interrupt the job (like Ctrl-C, so it stops cleanly), then terminate it if it does not exit."""
    job = current(voice)
    if not job:
        return False
    pgid = os.getpgid(job["pid"])
    for sig, wait in ((signal.SIGINT, timeout), (signal.SIGTERM, 15), (signal.SIGKILL, 5)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return True
        deadline = time.time() + wait
        while time.time() < deadline:
            if not _alive(job["pid"]):
                return True
            time.sleep(0.5)
    return not _alive(job["pid"])


def log_tail(voice: Voice, max_bytes: int = 20000) -> str:
    """The end of the job log, with progress-bar redraws (carriage returns) collapsed to their last state."""
    path = voice.logs_dir / "job.log"
    if not path.exists():
        return ""
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - max_bytes))
        text = f.read().decode(errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", text)  # terminal control sequences of progress bars
    lines = []
    for line in text.split("\n"):
        redraws = [seg for seg in line.split("\r") if seg.strip()]
        if redraws:
            lines.append(redraws[-1])
    return "\n".join(lines)


def background(voice: Voice, argv: list[str]) -> None:
    """Start the current command again as a background job (without --background) and report how to follow it."""
    args = [a for a in argv if a != "--background"]
    job = start(voice, args)
    print(f"Started in the background (pid {job['pid']}): voice-clone {' '.join(args)}")
    print(f"Follow it with: voice-clone log {voice.name} -f   ·   status: voice-clone status {voice.name}"
          f"   ·   stop: voice-clone stop {voice.name}")
