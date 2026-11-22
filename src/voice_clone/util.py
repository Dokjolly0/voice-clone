"""Small helpers shared by every stage: messages, subprocesses, atomic JSON."""

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path


class UserError(Exception):
    """An error the user can fix; printed without a traceback."""


def info(msg: str) -> None:
    print(msg, flush=True)


def step(msg: str) -> None:
    print(f"\n==> {msg}", flush=True)


def warn(msg: str) -> None:
    print(f"warning: {msg}", file=sys.stderr, flush=True)


def which(name: str) -> str | None:
    """Like shutil.which, also looking in ~/.local/bin, where uv and speech.cpp install themselves."""
    from voice_clone.paths import LOCAL_BIN

    return shutil.which(name) or shutil.which(name, path=str(LOCAL_BIN))


def run(cmd: list, *, check: bool = True, capture: bool = False, cwd: Path | None = None,
        env: dict | None = None) -> subprocess.CompletedProcess:
    """Run a command, echoing it first. With capture=True, stdout and stderr are returned as text."""
    cmd = [str(c) for c in cmd]
    info(f"$ {shlex.join(cmd)}")
    full_env = {**os.environ, **(env or {})}
    try:
        return subprocess.run(cmd, check=check, cwd=cwd, env=full_env, text=True,
                              capture_output=capture)
    except FileNotFoundError as exc:
        raise UserError(f"command not found: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        detail = f"\n{exc.stderr.strip()}" if capture and exc.stderr else ""
        raise UserError(f"command failed with exit code {exc.returncode}: {shlex.join(cmd)}{detail}") from exc


def read_json(path: Path, default=None):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def write_json(path: Path, data) -> None:
    """Write JSON atomically, so an interrupted run never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def download(url: str, dst: Path, md5: str | None = None) -> Path:
    """Download to dst through a .part file (an interrupted download never looks complete), checking md5 if given."""
    import urllib.request

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    info(f"downloading {url}")
    digest = hashlib.md5()
    req = urllib.request.Request(url, headers={"User-Agent": "voice-clone"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
            digest.update(chunk)
    if md5 and digest.hexdigest() != md5:
        tmp.unlink()
        raise UserError(f"download of {dst.name} is corrupted (checksum mismatch); try again")
    tmp.replace(dst)
    return dst
