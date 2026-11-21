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


# --- progress -------------------------------------------------------------------------------------------------------

# Step headers the stages print ("==> ..."), mapped to a phase name and the stage (1-5) it belongs to.
PHASES = [
    ("Merging", "merging", 2),
    ("Transcribing the reference", "transcribing", 2),
    ("Building the Drama 3 voice", "building_voice", 2),
    ("Choosing sentences", "choosing_sentences", 3),
    ("Generating", "generating", 3),
    ("Verifying", "verifying", 3),
    ("Exporting the dataset", "exporting_dataset", 3),
    ("Downloading the base voice", "downloading_base", 4),
    ("Training", "training", 4),
    ("Exporting", "exporting", 5),
    ("Speaking", "speaking", 5),
]
COMMAND_STAGE = {"reference": 2, "try-voice": 2, "sentences": 3, "dataset": 3, "train": 4, "export": 5}


def _phase_of(header: str) -> tuple[str, int]:
    for prefix, phase, stage in PHASES:
        if header.startswith(prefix):
            return phase, stage
    return "working", 0


def progress(voice: Voice) -> dict:
    """What the current (or last) job is doing: phase, stage 1-5, done/total, seconds left, and its error if it
    failed. Read from the job log, and for training from training/progress.json."""
    job = current(voice)
    last = read_json(voice.logs_dir / "job.json") or {}
    text = log_tail(voice, 200_000)
    lines = text.splitlines()
    result = {"running": job is not None, "command": last.get("args", []), "phase": "idle", "stage": 0,
              "done": None, "total": None, "eta": None, "error": None, "finished": False}
    if not lines:
        return result
    if last.get("args"):
        result["stage"] = COMMAND_STAGE.get(last["args"][0], 0)

    header_at = max((i for i, l in enumerate(lines) if l.startswith("==> ")), default=None)
    if header_at is not None:
        header = lines[header_at][4:]
        result["phase"], stage = _phase_of(header)
        result["stage"] = stage or result["stage"]
        after = lines[header_at + 1:]
        if result["phase"] == "generating":
            m = re.search(r"Generating (\d+) clips", header)
            result["total"] = int(m.group(1)) if m else None
            result["done"] = 0
            for line in reversed(after):
                m = re.match(r"\[(\d+)/(\d+)\].*~(\d+) min left", line)
                if m:
                    result["done"], result["total"] = int(m.group(1)), int(m.group(2))
                    result["eta"] = int(m.group(3)) * 60
                    break
            else:
                result["phase"] = "loading_model"
        elif result["phase"] == "verifying":
            m = re.search(r"Verifying (\d+) clips", header)
            result["total"], result["done"] = (int(m.group(1)) if m else None), 0
            for line in reversed(after):
                m = re.match(r"verified (\d+)/(\d+)", line)
                if m:
                    result["done"], result["total"] = int(m.group(1)), int(m.group(2))
                    break
        elif result["phase"] == "training":
            from voice_clone import train

            state = read_json(voice.training_dir / "progress.json") or {}
            started = datetime.fromisoformat(last["started"]).timestamp() if last.get("started") else 0
            m = re.search(r"Training (\d+) epochs", header)
            result["total"] = int(m.group(1)) if m else state.get("max_epochs")
            if state and state.get("updated", 0) >= started:
                result["done"], result["total"] = state["epoch"], state["max_epochs"]
                result["eta"] = train.eta_seconds(voice)
            else:
                result["done"] = None  # preparing: phonemizing the dataset, loading the base voice
        elif result["phase"] == "speaking":
            m = re.search(r"Speaking (\d+) test sentences", header)
            result["total"] = int(m.group(1)) if m else None
            result["done"] = sum(1 for l in after if l.startswith("samples/sample-"))

    if not job:
        errors = [l for l in lines[-30:] if l.startswith("error: ")]
        if errors:
            result["error"] = errors[-1][len("error: "):]
        elif any(l.startswith("interrupted") for l in lines[-5:]):
            result["error"] = "interrupted"
        else:
            result["finished"] = True
    return result
