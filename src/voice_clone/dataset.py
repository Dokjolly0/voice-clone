"""Stage 3b: speak every sentence with Drama 3, check each clip with Qwen3-ASR, export a Piper dataset.

Rounds of generate + verify run until every sentence has a good clip or has used all its attempts. A clip whose
transcript differs from its sentence by more than `max_cer` of the characters (skipped, repeated or invented words)
is deleted and generated again with another seed. Seeds are fixed per sentence and attempt, so a rerun reproduces
the same audio. Everything is resumable: progress lives in state.json, one entry per clip.
"""

import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from voice_clone import audio, reference, speech
from voice_clone.util import UserError, info, step, warn
from voice_clone.workspace import Voice

SAMPLE_RATE = 22050  # Piper "medium" models


@dataclass
class Settings:
    max_attempts: int = 4
    max_cer: float = 0.05
    max_seconds: float = 30.0  # a sentence that speaks longer than this is a runaway generation
    limit: int = 0             # generate at most this many clips per round (0: all), for quick tests


def clip_id(i: int) -> str:
    return f"{i:05d}"


def seed_for(i: int, attempt: int) -> int:
    return 1 + i * 100 + attempt


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text.lower().replace("’", "'"))
    text = re.sub(r"[^\w' ]+", " ", text)
    return " ".join(text.split())


def cer(ref: str, hyp: str) -> float:
    """Character error rate of hyp against ref, after normalizing case, punctuation and spaces."""
    ref, hyp = normalize(ref), normalize(hyp)
    prev = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        cur = [i]
        for j, b in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a != b)))
        prev = cur
    return prev[-1] / max(len(ref), 1)


def sync_state(voice: Voice) -> dict:
    """state.json's clips, matched to sentences.txt: an entry whose sentence changed starts over."""
    sentences = voice.sentences()
    if not sentences:
        raise UserError(f"no sentences yet: run `voice-clone sentences {voice.name}`")
    state = voice.load_state()
    clips = state.setdefault("clips", {})
    for i, text in enumerate(sentences):
        cid = clip_id(i)
        entry = clips.get(cid)
        if entry is None or entry["text"] != text:
            (voice.clips_dir / f"{cid}.wav").unlink(missing_ok=True)
            clips[cid] = {"text": text, "attempts": 0}
    for cid in [c for c in clips if int(c) >= len(sentences)]:
        del clips[cid]
    voice.save_state(state)
    return state


def to_generate(voice: Voice, state: dict, settings: Settings) -> list[str]:
    return [cid for cid, e in sorted(state["clips"].items())
            if not e.get("verified") and not (voice.clips_dir / f"{cid}.wav").exists()
            and e["attempts"] < settings.max_attempts]


def to_verify(voice: Voice, state: dict) -> list[str]:
    return [cid for cid, e in sorted(state["clips"].items())
            if not e.get("verified") and (voice.clips_dir / f"{cid}.wav").exists()]


def generate(voice: Voice, state: dict, todo: list[str], settings: Settings) -> None:
    voice_file = reference.paths(voice)["voice"]
    if not voice_file.exists():
        raise UserError(f"no Drama 3 voice yet: run `voice-clone reference {voice.name}`")
    voice.clips_dir.mkdir(parents=True, exist_ok=True)
    step(f"Generating {len(todo)} clips with Drama 3 (loading the model takes a minute)")
    with speech.SpeechServer(voice_file, voice.logs_dir / "serve.log") as server:
        t0, spoken = time.time(), 0.0
        for n, cid in enumerate(todo, 1):
            entry = state["clips"][cid]
            seed = seed_for(int(cid), entry["attempts"])
            entry["attempts"] += 1
            entry["seed"] = seed
            wav, stop = server.synthesize(entry["text"], seed, max_seconds=settings.max_seconds)
            seconds = audio.wav_bytes_duration(wav) if wav else 0.0
            if stop != "complete" or seconds == 0:
                entry["rejected"] = f"stopped: {stop}" if stop != "complete" else "empty audio"
            else:
                (voice.clips_dir / f"{cid}.wav").write_bytes(wav)
                entry.pop("rejected", None)
                spoken += seconds
            voice.save_state(state)
            elapsed = time.time() - t0
            eta = elapsed / n * (len(todo) - n)
            info(f"[{n}/{len(todo)}] clip {cid} seed {seed}: {spoken / 60:.1f} min of audio, "
                 f"RTFx {spoken / max(elapsed, 1e-6):.2f}, ~{eta / 60:.0f} min left")


def verify(voice: Voice, state: dict, todo: list[str], settings: Settings) -> tuple[int, int]:
    step(f"Verifying {len(todo)} clips with Qwen3-ASR")
    good = bad = 0
    for k in range(0, len(todo), 100):
        batch = todo[k:k + 100]
        files = [voice.clips_dir / f"{cid}.wav" for cid in batch]
        texts = speech.transcribe(files, voice.language.code)
        for cid, path in zip(batch, files):
            entry = state["clips"][cid]
            heard = texts.get(str(path), "")
            entry["heard"] = heard
            entry["cer"] = round(cer(entry["text"], heard), 3)
            if entry["cer"] <= settings.max_cer:
                entry["verified"] = True
                entry.pop("rejected", None)
                good += 1
            else:
                path.unlink()
                entry["rejected"] = f"CER {entry['cer']}"
                bad += 1
        voice.save_state(state)
        info(f"verified {min(k + 100, len(todo))}/{len(todo)}")
    info(f"good {good}, rejected {bad} (rejected clips are generated again with another seed)")
    return good, bad


def counts(state: dict, settings: Settings) -> dict:
    clips = state.get("clips", {})
    verified = sum(1 for e in clips.values() if e.get("verified"))
    failed = sum(1 for e in clips.values() if not e.get("verified") and e["attempts"] >= settings.max_attempts)
    return {"total": len(clips), "verified": verified, "failed": failed, "pending": len(clips) - verified - failed}


def export(voice: Voice, state: dict) -> tuple[int, float]:
    """Verified clips at 22050 Hz mono in dataset/wav, and dataset/metadata.csv (id|text)."""
    step("Exporting the dataset for Piper")
    wav_dir = voice.dataset_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    rows, seconds, keep = [], 0.0, set()
    for cid, entry in sorted(state["clips"].items()):
        if not entry.get("verified"):
            continue
        src, dst = voice.clips_dir / f"{cid}.wav", wav_dir / f"{cid}.wav"
        if not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime:
            audio.convert(src, dst, rate=SAMPLE_RATE)
        seconds += audio.wav_duration(dst)
        keep.add(dst.name)
        rows.append(f"{cid}|{entry['text']}")
    for stale in wav_dir.glob("*.wav"):
        if stale.name not in keep:
            stale.unlink()
    (voice.dataset_dir / "metadata.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    info(f"{len(rows)} clips, {seconds / 60:.1f} minutes of audio in {voice.dataset_dir}")
    if seconds < 30 * 60:
        warn("under 30 minutes of audio: the voice will be rougher; about an hour (1000 sentences) is recommended")
    return len(rows), seconds


def build(voice: Voice, settings: Settings | None = None) -> None:
    settings = settings or Settings()
    state = sync_state(voice)
    for round_no in range(1, settings.max_attempts + 1):
        todo = to_generate(voice, state, settings)
        if settings.limit:
            todo = todo[: settings.limit]
        if todo:
            info(f"\nRound {round_no}")
            generate(voice, state, todo, settings)
        pending = to_verify(voice, state)
        if pending:
            verify(voice, state, pending, settings)
        if not to_generate(voice, state, settings) or settings.limit:
            break
    c = counts(state, settings)
    info(f"\n{c['verified']}/{c['total']} sentences have a good clip; {c['failed']} used all "
         f"{settings.max_attempts} attempts and stay out")
    if c["verified"] == 0:
        raise UserError("no good clips: listen to reference/try.wav (voice-clone try-voice) and check the transcript")
    export(voice, state)
    info(f"Listen to a few files in {voice.dataset_dir / 'wav'}: Piper learns exactly what it hears there.")


def status(voice: Voice) -> dict:
    state = voice.load_state()
    result = counts(state, Settings())
    meta = voice.dataset_dir / "metadata.csv"
    result["exported"] = sum(1 for l in meta.read_text(encoding="utf-8").splitlines() if l.strip()) \
        if meta.exists() else 0
    return result
