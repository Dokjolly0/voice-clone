"""The model library: every Piper voice the studio can speak with.

Three origins, one list:
- trained:  voices made here, read in place from voices/<name>/export/*.onnx (never copied);
- catalog:  official Piper voices downloaded from rhasspy/piper-voices (183 voices, 53 languages);
- imported: any .onnx + .onnx.json added by hand (e.g. a voice trained on another machine).
Downloaded and imported models live in studio/models/<id>/ with a model.json describing them.
"""

import json
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from voice_clone import workspace
from voice_clone.paths import CACHE_DIR, STUDIO_DIR
from voice_clone.util import UserError, download, info, read_json, write_json

CATALOG_URL = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
CATALOG_MAX_AGE = 24 * 3600
ORIGIN_LABELS = {"trained": "tua voce", "catalog": "catalogo Piper", "imported": "importata"}


@dataclass
class Model:
    id: str                 # "voice:anna", "catalog:it_IT-paola-medium", "imported:pos"
    name: str
    language: str           # locale such as it_IT, or the espeak voice when unknown
    quality: str
    origin: str             # trained | catalog | imported
    onnx: Path
    config_path: Path
    sample_rate: int
    speakers: list[str] = field(default_factory=list)  # names of a multi-speaker model, else empty
    inference: dict = field(default_factory=dict)       # the model's default noise_scale, noise_w, length_scale

    @property
    def label(self) -> str:
        return f"{self.name} · {self.language} · {self.quality} ({ORIGIN_LABELS[self.origin]})"

    @property
    def language_family(self) -> str:
        return re.split(r"[_-]", self.language)[0].lower()

    @property
    def config(self) -> dict:
        return json.loads(self.config_path.read_text(encoding="utf-8"))


def models_dir() -> Path:
    return STUDIO_DIR / "models"


def validate_config(path: Path) -> dict:
    """The parts of a Piper .onnx.json the studio needs; raises UserError when it is not one."""
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise UserError(f"{path.name} is not a readable JSON file") from exc
    missing = [k for k in ("audio", "num_speakers", "phoneme_id_map") if k not in config]
    if missing or "sample_rate" not in config.get("audio", {}):
        raise UserError(f"{path.name} is not a Piper voice configuration (missing: {', '.join(missing) or 'sample_rate'})")
    if config.get("phoneme_type", "espeak") == "espeak" and not config.get("espeak", {}).get("voice"):
        raise UserError(f"{path.name} names no espeak voice")
    return config


def _from_files(model_id: str, origin: str, onnx: Path, config_path: Path, meta: dict | None = None) -> Model:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    meta = meta or {}
    stem = onnx.name.removesuffix(".onnx")
    m = re.match(r"^([a-z]{2,3}_[A-Z]{2})-(.+?)-(x_low|low|medium|high)$", stem)
    language = (config.get("language", {}).get("code") or meta.get("language")
                or (m.group(1) if m else None) or config.get("espeak", {}).get("voice", "?"))
    quality = config.get("audio", {}).get("quality") or meta.get("quality") or (m.group(3) if m else "?")
    name = meta.get("name") or (m.group(2) if m else stem)
    speaker_map = config.get("speaker_id_map") or {}
    speakers = [s for s, _ in sorted(speaker_map.items(), key=lambda kv: kv[1])] if config.get("num_speakers", 1) > 1 \
        else []
    return Model(id=model_id, name=name, language=language, quality=quality, origin=origin, onnx=onnx,
                 config_path=config_path, sample_rate=config["audio"]["sample_rate"], speakers=speakers,
                 inference=config.get("inference", {}))


def list_models() -> list[Model]:
    """Trained voices first, then downloaded and imported models, each group by name."""
    result = []
    for name in workspace.list_names():
        voice = workspace.load(name)
        if voice.onnx_path.exists() and voice.onnx_config_path.exists():
            result.append(_from_files(f"voice:{name}", "trained", voice.onnx_path, voice.onnx_config_path,
                                      {"name": name}))
    library = []
    if models_dir().exists():
        for d in sorted(models_dir().iterdir()):
            meta = read_json(d / "model.json")
            if not meta:
                continue
            onnx = d / meta["onnx"]
            config = d / (meta["onnx"] + ".json")
            if onnx.exists() and config.exists():
                library.append(_from_files(f"{meta['origin']}:{d.name}", meta["origin"], onnx, config, meta))
    order = {"catalog": 0, "imported": 1}
    return result + sorted(library, key=lambda m: (order[m.origin], m.name))


def get(ref: str) -> Model:
    """A model by id ("catalog:it_IT-paola-medium"), by plain name ("anna", "paola"), or by .onnx path."""
    path = Path(ref)
    if ref.endswith(".onnx") and path.exists():
        config = path.with_name(path.name + ".json")
        if not config.exists():
            raise UserError(f"{config.name} must sit next to {path.name}")
        validate_config(config)
        return _from_files(f"file:{path}", "imported", path, config)
    available = list_models()
    for m in available:
        if ref in (m.id, m.id.split(":", 1)[1]):
            return m
    by_name = [m for m in available if m.name == ref]
    if len(by_name) == 1:
        return by_name[0]
    if by_name:
        raise UserError(f"several models are called {ref!r}: {', '.join(m.id for m in by_name)}; use the id")
    names = ", ".join(m.id for m in available) or "none yet"
    raise UserError(f"no model {ref!r} (available: {names}); see `voice-clone models list`")


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9_-]+", "-", text.lower()).strip("-") or "model"


def import_files(onnx: Path, config: Path | None = None, name: str | None = None) -> Model:
    """Copy a Piper model (.onnx and its .onnx.json) into the library."""
    onnx = Path(onnx)
    config = Path(config) if config else onnx.with_name(onnx.name + ".json")
    if onnx.suffix != ".onnx" or not onnx.is_file():
        raise UserError(f"not an .onnx file: {onnx}")
    if not config.is_file():
        raise UserError(f"the model's configuration is missing: {config.name} (it comes with the .onnx)")
    validate_config(config)
    model_id = _slug(name or onnx.name.removesuffix(".onnx"))
    target = models_dir() / model_id
    if target.exists():
        raise UserError(f"a model {model_id!r} already exists; remove it first or choose another name")
    target.mkdir(parents=True)
    shutil.copy(onnx, target / onnx.name)
    shutil.copy(config, target / (onnx.name + ".json"))
    write_json(target / "model.json", {"origin": "imported", "onnx": onnx.name, "name": name or None,
                                       "added": time.strftime("%Y-%m-%dT%H:%M:%S")})
    info(f"imported {onnx.name} as imported:{model_id}")
    return get(f"imported:{model_id}")


def remove(model_id: str) -> None:
    m = get(model_id)
    if m.origin == "trained":
        raise UserError("trained voices are not removed here; they live in voices/<name>/")
    shutil.rmtree(m.onnx.parent)
    info(f"removed {m.id}")


# --- Piper's catalog -------------------------------------------------------------------------------------------------

def catalog(refresh: bool = False) -> dict:
    """voices.json of rhasspy/piper-voices, cached for a day (and used stale when offline)."""
    path = CACHE_DIR / "piper-voices.json"
    fresh = path.exists() and time.time() - path.stat().st_mtime < CATALOG_MAX_AGE
    if refresh or not fresh:
        try:
            download(CATALOG_URL + "voices.json", path)
        except OSError as exc:
            if not path.exists():
                raise UserError(f"cannot download Piper's voice catalog: {exc}") from exc
    return json.loads(path.read_text(encoding="utf-8"))


def search(lang: str | None = None, quality: str | None = None) -> list[dict]:
    """Catalog entries filtered by language (family 'it' or locale 'it_IT') and quality, with download state."""
    installed = {m.id.split(":", 1)[1] for m in list_models() if m.origin == "catalog"}
    result = []
    for key, v in sorted(catalog().items()):
        language = v["language"]
        if lang and lang.lower() not in (language["family"].lower(), language["code"].lower()):
            continue
        if quality and v["quality"] != quality:
            continue
        size = sum(f["size_bytes"] for name, f in v["files"].items() if name.endswith(".onnx"))
        result.append({"key": key, "name": v["name"], "language": language["code"],
                       "language_name": language["name_english"], "quality": v["quality"],
                       "speakers": v["num_speakers"], "size_mb": round(size / 1e6), "installed": key in installed})
    return result


def download_catalog(key: str) -> Model:
    entries = catalog()
    if key not in entries:
        raise UserError(f"no voice {key!r} in Piper's catalog; see `voice-clone models catalog --lang <code>`")
    entry = entries[key]
    target = models_dir() / key
    onnx_name = None
    for remote, f in entry["files"].items():
        local = target / Path(remote).name
        if local.exists() and local.stat().st_size == f["size_bytes"]:
            continue
        download(CATALOG_URL + remote, local, md5=f["md5_digest"])
    for remote in entry["files"]:
        if remote.endswith(".onnx"):
            onnx_name = Path(remote).name
    write_json(target / "model.json", {"origin": "catalog", "onnx": onnx_name, "name": entry["name"],
                                       "language": entry["language"]["code"], "quality": entry["quality"],
                                       "added": time.strftime("%Y-%m-%dT%H:%M:%S")})
    info(f"downloaded {key}")
    return get(f"catalog:{key}")
