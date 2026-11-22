"""Stage 3b: speak every sentence with Drama 3, check each clip with Qwen3-ASR, export a Piper dataset.

Rounds of generate + verify run until every sentence has a good clip or has used all its attempts. A clip whose
transcript differs from its sentence by more than `max_cer` of the characters (skipped, repeated or invented words)
is deleted and generated again with another seed. Seeds are fixed per sentence and attempt, so a rerun reproduces
the same audio. Everything is resumable: progress lives in state.json, one entry per clip.

Short sentences (under the language's min_chars) get `short_best_of` takes per attempt, because Drama 3 guesses a
different intonation for each one: every take is checked by ASR, and of those that pass, the one closest to the
voice's typical pitch and rate (prosody.py, measured on its long sentences) is kept.
"""

import re
import shutil
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from voice_clone import audio, prosody, reference, speech
from voice_clone.util import UserError, info, step, warn
from voice_clone.workspace import Voice

SAMPLE_RATE = 22050  # Piper "medium" models


@dataclass
class Settings:
    max_attempts: int = 4
    max_cer: float = 0.05
    max_seconds: float = 30.0  # a sentence that speaks longer than this is a runaway generation
    limit: int = 0             # generate at most this many clips per round (0: all), for quick tests
    short_best_of: int = 5     # takes per short sentence, the most typical one is kept (1: off)
    max_distance: float = 3.0  # a best take farther than this from the voice's prosody is tried again
    redo_short: bool = False   # regenerate verified short sentences that were made with a single take


def clip_id(i: int) -> str:
    return f"{i:05d}"


def seed_for(i: int, attempt: int, take: int = 0) -> int:
    return 1 + i * 100 + attempt + 7919 * take


def is_short(voice: Voice, text: str) -> bool:
    return len(text) < voice.language.min_chars


def takes_dir(voice: Voice) -> Path:
    return voice.clips_dir / "takes"


def takes_of(voice: Voice, cid: str) -> list[Path]:
    return sorted(takes_dir(voice).glob(f"{cid}-*.wav")) if takes_dir(voice).exists() else []


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
            if not e.get("verified") and not (voice.clips_dir / f"{cid}.wav").exists() and not takes_of(voice, cid)
            and e["attempts"] < settings.max_attempts]


def to_verify(voice: Voice, state: dict) -> list[str]:
    return [cid for cid, e in sorted(state["clips"].items())
            if not e.get("verified") and ((voice.clips_dir / f"{cid}.wav").exists() or takes_of(voice, cid))]


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
            attempt = entry["attempts"]
            entry["attempts"] += 1
            seed = entry["seed"] = seed_for(int(cid), attempt)
            several = settings.short_best_of > 1 and is_short(voice, entry["text"])
            kept = 0
            for take in range(settings.short_best_of if several else 1):
                take_seed = seed_for(int(cid), attempt, take)
                wav, stop = server.synthesize(entry["text"], take_seed, max_seconds=settings.max_seconds)
                seconds = audio.wav_bytes_duration(wav) if wav else 0.0
                if stop != "complete" or seconds == 0:
                    entry["rejected"] = f"stopped: {stop}" if stop != "complete" else "empty audio"
                    continue
                if several:
                    takes_dir(voice).mkdir(parents=True, exist_ok=True)
                    (takes_dir(voice) / f"{cid}-{take_seed}.wav").write_bytes(wav)
                else:
                    (voice.clips_dir / f"{cid}.wav").write_bytes(wav)
                kept += 1
                spoken += seconds
            if kept:
                entry.pop("rejected", None)
            voice.save_state(state)
            elapsed = time.time() - t0
            eta = elapsed / n * (len(todo) - n)
            info(f"[{n}/{len(todo)}] clip {cid} seed {seed}: {spoken / 60:.1f} min of audio, "
                 f"RTFx {spoken / max(elapsed, 1e-6):.2f}, ~{eta / 60:.0f} min left")


def voice_profile(voice: Voice, state: dict, sample: int = 60) -> dict:
    """Typical pitch and rate of the voice, from up to `sample` verified long sentences (cached in state.json)."""
    cached = state.get("prosody_profile")
    if cached and cached.get("clips", 0) >= min(sample, 20):
        return cached
    long_ids = [cid for cid, e in sorted(state["clips"].items())
                if e.get("verified") and not is_short(voice, e["text"]) and (voice.clips_dir / f"{cid}.wav").exists()]
    if len(long_ids) < 10:
        return {}  # too early to know the voice: the first take that passes ASR is kept
    picked = long_ids[:: max(1, len(long_ids) // sample)][:sample]
    info(f"measuring the voice's typical pitch and rate on {len(picked)} long sentences")
    found = prosody.measure([voice.clips_dir / f"{cid}.wav" for cid in picked])
    pairs = [(found[str(voice.clips_dir / f"{cid}.wav")], state["clips"][cid]["text"]) for cid in picked]
    state["prosody_profile"] = prosody.profile(pairs)
    voice.save_state(state)
    p = state["prosody_profile"]
    info(f"typical pitch {p['f0']:.0f} Hz, pitch movement {p['f0_range']:.0f} Hz, {p['rate']:.1f} letters/s")
    return state["prosody_profile"]


def choose_takes(voice: Voice, state: dict, cids: list[str], settings: Settings) -> tuple[int, int]:
    """Short sentences with several takes: ASR-check every take, keep the most typical one that passes."""
    step(f"Choosing the best take of {len(cids)} short sentences")
    files = [t for cid in cids for t in takes_of(voice, cid)]
    texts = speech.transcribe(files, voice.language.code)
    passing = {cid: [t for t in takes_of(voice, cid)
                     if cer(state["clips"][cid]["text"], texts.get(str(t), "")) <= settings.max_cer] for cid in cids}
    profile = voice_profile(voice, state)
    measured = prosody.measure([t for ts in passing.values() for t in ts]) if profile else {}
    good = bad = 0
    for cid in cids:
        entry = state["clips"][cid]
        candidates = passing[cid]
        def far(t):
            return prosody.distance(measured.get(str(t), {}), entry["text"], profile) if profile else 0

        best = min(candidates, key=far) if candidates else None
        if best is not None and far(best) > settings.max_distance and entry["attempts"] < settings.max_attempts:
            entry["rejected"] = f"best take too far from the voice's prosody (distance {far(best):.1f})"
            for t in takes_of(voice, cid):
                t.unlink()
            bad += 1
            continue
        if candidates:
            shutil.move(best, voice.clips_dir / f"{cid}.wav")
            entry.update(verified=True, best_of=settings.short_best_of, heard=texts.get(str(best), ""),
                         cer=round(cer(entry["text"], texts.get(str(best), "")), 3), seed=int(best.stem.split("-")[1]))
            if profile:
                entry["prosody"] = {**measured.get(str(best), {}), "distance": prosody.distance(
                    measured.get(str(best), {}), entry["text"], profile)}
                entry["prosody"].pop("file", None)
            entry.pop("rejected", None)
            good += 1
        else:
            entry["rejected"] = "no take passed ASR"
            bad += 1
        for t in takes_of(voice, cid):
            t.unlink()
    voice.save_state(state)
    info(f"short sentences: {good} kept their best take, {bad} had no good or typical enough take "
         "(generated again, until the last attempt keeps the best)")
    return good, bad


def verify(voice: Voice, state: dict, todo: list[str], settings: Settings) -> tuple[int, int]:
    with_takes = [cid for cid in todo if takes_of(voice, cid)]
    good = bad = 0
    if with_takes:
        good, bad = choose_takes(voice, state, with_takes, settings)
    todo = [cid for cid in todo if cid not in with_takes]
    if not todo:
        return good, bad
    step(f"Verifying {len(todo)} clips with Qwen3-ASR")
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


def redo_short(voice: Voice, state: dict, settings: Settings) -> int:
    """Send verified short sentences made with a single take back to generation, to get their best take."""
    single = [cid for cid, e in state["clips"].items()
              if e.get("verified") and is_short(voice, e["text"]) and e.get("best_of", 1) < settings.short_best_of]
    # Best takes that stayed far from the voice and still have attempts left: new seeds (the attempt goes on).
    far = [cid for cid, e in state["clips"].items()
           if e.get("verified") and cid not in single and e.get("prosody", {}).get("distance", 0) > settings.max_distance
           and e["attempts"] < settings.max_attempts]
    for cid in single + far:
        entry = state["clips"][cid]
        (voice.clips_dir / f"{cid}.wav").unlink(missing_ok=True)
        entry.pop("verified", None)
        if cid in single:
            entry["attempts"] = 0  # take 0 of attempt 0 reproduces the original clip, so it competes too
    voice.save_state(state)
    info(f"{len(single) + len(far)} short sentences will be generated again ({len(single)} single takes, "
         f"{len(far)} far from the voice), keeping the best of {settings.short_best_of} takes")
    return len(single) + len(far)


def build(voice: Voice, settings: Settings | None = None) -> None:
    settings = settings or Settings()
    state = sync_state(voice)
    if settings.redo_short and settings.short_best_of > 1:
        redo_short(voice, state, settings)
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
    result = counts(voice.load_state(), Settings())
    # Sentences without a state entry yet (dataset never started on them) are still to do.
    result["total"] = max(result["total"], len(voice.sentences()))
    result["pending"] = result["total"] - result["verified"] - result["failed"]
    meta = voice.dataset_dir / "metadata.csv"
    result["exported"] = sum(1 for l in meta.read_text(encoding="utf-8").splitlines() if l.strip()) \
        if meta.exists() else 0
    return result
