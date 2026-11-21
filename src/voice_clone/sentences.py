"""Stage 3a: choose the sentences of the dataset.

Sentences come from Common Voice (server/data/<locale>/*.txt in github.com/common-voice/common-voice, public domain
CC0), filtered to plain sentences that a TTS model reads one way only: no digits, abbreviations or symbols. Optional
domain sentences (product names, the phrases the voice will really say) go first.

sentences.txt is append-only: line N is clip N, so growing the dataset later keeps every clip already made.
"""

import json
import random
import re
import unicodedata
import urllib.request
from pathlib import Path

from voice_clone.languages import Language
from voice_clone.paths import CACHE_DIR
from voice_clone.util import UserError, info, step, warn
from voice_clone.workspace import Voice

CV_API = "https://api.github.com/repos/common-voice/common-voice/contents/server/data/"
POOL_FACTOR = 20  # download until there are this many raw lines per wanted sentence

# Characters that make a reading ambiguous or are not speech.
BAD_CHARS = re.compile(r"[\d#$%&*+/<=>@\\^_`{|}~\[\]()§°€£¥]")
ALL_CAPS_WORD = re.compile(r"\b[^\W\d_]{2,}\b")


def _fetch_json(url: str):
    req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "voice-clone"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def _download(url: str, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(".part")
    req = urllib.request.Request(url, headers={"User-Agent": "voice-clone"})
    with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)
    tmp.replace(dst)


def corpus_files(lang: Language, wanted: int) -> list[Path]:
    """Common Voice sentence files for the language, downloaded (once) into the cache, smallest first."""
    cache = CACHE_DIR / "common-voice" / lang.common_voice
    index_file = cache / "index.json"
    if index_file.exists():
        index = json.loads(index_file.read_text())
    else:
        info(f"Listing Common Voice sentences for {lang.name}")
        try:
            listing = _fetch_json(CV_API + lang.common_voice)
        except OSError as exc:
            raise UserError(f"cannot reach GitHub to download Common Voice sentences: {exc}") from exc
        index = sorted(({"name": e["name"], "size": e["size"], "url": e["download_url"]}
                        for e in listing if e["name"].endswith(".txt")), key=lambda e: e["size"])
        cache.mkdir(parents=True, exist_ok=True)
        index_file.write_text(json.dumps(index, indent=1))

    files, lines = [], 0
    for entry in index:
        path = cache / entry["name"]
        if not path.exists():
            info(f"downloading {entry['name']} ({entry['size'] / 1e6:.1f} MB)")
            _download(entry["url"], path)
        files.append(path)
        lines += entry["size"] // 60  # rough: 60 bytes per line
        if lines >= wanted * POOL_FACTOR:
            break
    return files


def clean(sentence: str) -> str:
    s = unicodedata.normalize("NFC", sentence.strip())
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", s)


def acceptable(sentence: str, lang: Language) -> bool:
    if not (lang.min_chars <= len(sentence) <= lang.max_chars):
        return False
    if BAD_CHARS.search(sentence):
        return False
    if sentence[-1] not in ".!?。！？":
        return False
    if any(w.isupper() for w in ALL_CAPS_WORD.findall(sentence)):  # abbreviations: read as letters or as a word?
        return False
    first = sentence[0]
    if first.isalpha() and first.lower() != first.upper() and not first.isupper():  # cased scripts start uppercase
        return False
    return True


def pool(lang: Language, wanted: int) -> list[str]:
    seen, result = set(), []
    for path in corpus_files(lang, wanted):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            s = clean(line)
            if s and s not in seen and acceptable(s, lang):
                seen.add(s)
                result.append(s)
    random.Random(42).shuffle(result)
    return result


def read_lines(path: Path) -> list[str]:
    return [clean(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def choose(voice: Voice, count: int, extra: Path | None = None) -> list[str]:
    """Make sentences.txt hold `count` sentences (domain sentences included), keeping the existing ones in order."""
    lang = voice.language
    existing = voice.sentences()
    if len(existing) >= count and not extra:
        if len(existing) > count:
            warn(f"sentences.txt already has {len(existing)} sentences; keeping them all (it only grows)")
        info(f"{len(existing)} sentences, nothing to add")
        return existing

    step(f"Choosing sentences ({lang.name})")
    chosen = list(existing)
    seen = set(chosen)
    if extra:
        if not extra.is_file():
            raise UserError(f"file not found: {extra}")
        domain = [s for s in read_lines(extra) if s not in seen]
        chosen += domain
        seen.update(domain)
        info(f"{len(domain)} domain sentences from {extra.name}")
    need = count - len(chosen)
    if need > 0:
        for s in pool(lang, need):
            if s not in seen:
                chosen.append(s)
                seen.add(s)
                need -= 1
                if need == 0:
                    break
        if need > 0:
            warn(f"Common Voice has only {count - need} usable {lang.name} sentences; the dataset will be smaller")

    voice.sentences_file.write_text("\n".join(chosen) + "\n", encoding="utf-8")
    added = len(chosen) - len(existing)
    info(f"sentences.txt: {len(chosen)} sentences ({added} new), about {estimate_minutes(chosen, lang):.0f} minutes "
         "of speech")
    return chosen


def estimate_minutes(sentences: list[str], lang: Language) -> float:
    return sum(len(s) for s in sentences) / lang.chars_per_second / 60
