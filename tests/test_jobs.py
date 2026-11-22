import sys
import time

import psutil
import pytest

from voice_clone import jobs


@pytest.fixture
def voice(voices_dir):
    from voice_clone import workspace

    return workspace.create("anna", "it")

# A stand-in for a long stage: prints, then waits until interrupted, which it reports like voice-clone does.
JOB = """
import os, sys, time
from voice_clone import cli, host
if os.environ.get("SIMULATE_WINDOWS"):
    host.IS_WINDOWS = True
cli.interruptible()
print("working", flush=True)
try:
    time.sleep(120)
except KeyboardInterrupt:
    print("interrupted; run the same command again to resume.", flush=True)
    sys.exit(130)
"""


def wait_for(predicate, seconds=20):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.2)
    return False


def test_background_job_runs_detached_and_stops_cleanly(voice):
    job = jobs.start(voice, ["dataset", voice.name], command=[sys.executable, "-c", JOB])
    assert job["created"]
    assert jobs.current(voice)["pid"] == job["pid"]
    assert wait_for(lambda: "working" in jobs.log_tail(voice))
    started = time.time()
    assert jobs.stop(voice, timeout=20)
    assert time.time() - started < 10  # interrupted, not killed at the timeout
    assert jobs.current(voice) is None
    assert jobs.progress(voice)["error"] == "interrupted"
    assert not (voice.logs_dir / jobs.STOP_REQUEST).exists()


def test_windows_stop_request_interrupts_the_job(voice, monkeypatch):
    """The Windows way of stopping (a stop.request file the job watches), simulated on Linux too."""
    monkeypatch.setenv("SIMULATE_WINDOWS", "1")
    if sys.platform != "win32":
        monkeypatch.setattr(jobs, "_popen_options", lambda: {"start_new_session": True})
    monkeypatch.setattr(jobs.host, "IS_WINDOWS", True)
    jobs.start(voice, ["dataset", voice.name], command=[sys.executable, "-c", JOB])
    assert wait_for(lambda: "working" in jobs.log_tail(voice))
    started = time.time()
    assert jobs.stop(voice, timeout=20)
    assert time.time() - started < 10  # interrupted, not killed at the timeout
    assert jobs.progress(voice)["error"] == "interrupted"
    assert not (voice.logs_dir / jobs.STOP_REQUEST).exists()


def test_reused_pid_is_not_the_job(voice):
    me = psutil.Process()
    jobs.write_json(voice.logs_dir / "job.json", {"pid": me.pid, "created": me.create_time() - 1000, "args": []})
    assert jobs.current(voice) is None
    jobs.write_json(voice.logs_dir / "job.json", {"pid": me.pid, "created": me.create_time(), "args": []})
    assert jobs.current(voice)["pid"] == me.pid
