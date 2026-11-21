"""speech.cpp wrapper: speech recognition, Drama 3 voice files and the `speech serve` HTTP server."""

import json
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

from voice_clone.installer import ASR_MODEL, SYNTH_MODEL, speech_bin
from voice_clone.util import UserError, run

VOICE_NAME = "voice"


def transcribe(files: list[Path], lang: str, batch: int = 100) -> dict[str, str]:
    """Text of each file, keyed by file path as given. Files are recognized in batches by one process each."""
    texts = {}
    for k in range(0, len(files), batch):
        chunk = [str(f) for f in files[k:k + batch]]
        result = subprocess.run([speech_bin(), "asr", ASR_MODEL, "--language", lang, "--format", "json", *chunk],
                                capture_output=True, text=True, encoding="utf-8")
        if result.returncode != 0:
            raise UserError(f"speech asr failed: {result.stderr.strip()[-1500:]}")
        for line in result.stdout.splitlines():
            if line.strip():
                r = json.loads(line)
                texts[r["file"]] = r["text"].strip()
    return texts


def make_voice(recordings: list[tuple[Path, str]], out: Path) -> None:
    """Build a Drama 3 voice file from recordings and their exact transcripts."""
    args = []
    for wav, text in recordings:
        args += [wav, "--transcript", text]
    run([speech_bin(), "voice", SYNTH_MODEL, *args, out])


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class SpeechServer:
    """`speech serve` with Drama 3 and one voice, kept loaded across requests. Use as a context manager."""

    def __init__(self, voice_file: Path, log_file: Path):
        self.voice_file = voice_file
        self.log_file = log_file
        self.port = free_port()
        self.proc = None

    def __enter__(self):
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_file, "ab")
        self.proc = subprocess.Popen(
            [speech_bin(), "serve", SYNTH_MODEL, "--port", str(self.port),
             "--add-voice", f"{VOICE_NAME}={self.voice_file}"],
            stdout=log, stderr=log)
        self._wait_healthy()
        return self

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def _wait_healthy(self, timeout: float = 1800) -> None:
        start = time.time()
        while time.time() - start < timeout:
            if self.proc.poll() is not None:
                raise UserError(f"speech serve exited with code {self.proc.returncode}; see {self.log_file}")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=5) as r:
                    if r.status == 200:
                        return
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(3)
        raise UserError(f"speech serve did not start in time; see {self.log_file}")

    def synthesize(self, text: str, seed: int, max_seconds: float = 30) -> tuple[bytes, str]:
        """WAV bytes and the stop reason: 'complete', 'max_seconds' or 'model_limit'."""
        body = json.dumps({"input": text, "voice": VOICE_NAME, "seed": seed, "max_seconds": max_seconds}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/audio/speech", data=body,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=900) as r:
                return r.read(), r.headers.get("X-Speech-Stop", "complete")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:500]
            raise UserError(f"speech serve refused the request ({exc.code}): {detail}") from exc
