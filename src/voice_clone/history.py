"""Generation history of the studio: studio/history/<id>/ holds the audio file(s) and meta.json.

Each entry keeps the text, the model, the options and the result, so a generation can be replayed, downloaded or
done again with other settings. The newest 200 entries are kept.
"""

import secrets
import shutil
import time
from pathlib import Path
from typing import Callable

from voice_clone import synthesis
from voice_clone.models import Model
from voice_clone.paths import STUDIO_DIR
from voice_clone.util import UserError, read_json, write_json

MAX_ENTRIES = 200


def history_dir() -> Path:
    return STUDIO_DIR / "history"


def new_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def create(model: Model, text: str, options: synthesis.Options, per_line: bool = False,
           progress: Callable[[int, int], None] | None = None) -> dict:
    """Generate into a new history entry and return its metadata."""
    entry_id = new_id()
    folder = history_dir() / entry_id
    folder.mkdir(parents=True)
    try:
        if per_line:
            results = synthesis.generate_per_line(model, text, folder / "files", options, progress)
            download = synthesis.zip_files([r.path for r in results], folder / f"{entry_id}.zip")
            files = [str(r.path.relative_to(folder)) for r in results]
        else:
            result = synthesis.generate(model, text, folder / "audio", options, progress)
            results, download, files = [result], result.path, [result.path.name]
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    meta = {"id": entry_id, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "model_id": model.id,
            "model_label": model.label, "text": text, "options": options.to_dict(), "per_line": per_line,
            "files": files, "download": download.name, "seconds": round(sum(r.seconds for r in results), 2)}
    write_json(folder / "meta.json", meta)
    prune()
    return meta


def entries(limit: int = 50) -> list[dict]:
    """Newest first."""
    if not history_dir().exists():
        return []
    found = []
    for folder in sorted(history_dir().iterdir(), reverse=True):
        meta = read_json(folder / "meta.json")
        if meta:
            found.append(meta)
            if len(found) >= limit:
                break
    return found


def get(entry_id: str) -> dict:
    meta = read_json(history_dir() / entry_id / "meta.json")
    if not meta:
        raise UserError(f"no generation {entry_id!r} in the history")
    return meta


def file_path(meta: dict, name: str | None = None) -> Path:
    """The entry's download (audio or zip), or one of its files."""
    return history_dir() / meta["id"] / (name or meta["download"])


def first_audio(meta: dict) -> Path:
    return history_dir() / meta["id"] / meta["files"][0]


def delete(entry_id: str) -> None:
    get(entry_id)
    shutil.rmtree(history_dir() / entry_id)


def prune(keep: int = MAX_ENTRIES) -> None:
    if not history_dir().exists():
        return
    folders = sorted((p for p in history_dir().iterdir() if p.is_dir()), reverse=True)
    for old in folders[keep:]:
        shutil.rmtree(old, ignore_errors=True)
