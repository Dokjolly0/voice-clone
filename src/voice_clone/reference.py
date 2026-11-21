"""Stage 2: merge the input recordings into one reference and turn it into a Drama 3 voice.

1. Every input file is decoded, trimmed of leading/trailing silence and joined with short pauses: merged.wav.
2. merged.wav is cut at a pause to at most `max_seconds`: reference.wav. Drama 3 prompts every request with the
   whole reference, so a longer one costs speed and leaves the text less room; 20-30 s is plenty.
3. Qwen3-ASR writes transcript.txt. Drama 3 needs the exact words, so check it and fix it if needed: an edited
   transcript is kept, and the next run rebuilds the voice from it.
4. `speech voice drama-3` builds voice.gguf.

Each step is redone only when what it depends on changed.
"""

import hashlib
from pathlib import Path

from voice_clone import audio, speech
from voice_clone.util import UserError, info, read_json, step, write_json
from voice_clone.workspace import Voice

RATE = 44100  # Drama 3 resamples to 44.1 kHz anyway
DEFAULT_MAX_SECONDS = 30.0


def _inputs_signature(voice: Voice, max_seconds: float) -> str:
    h = hashlib.sha256(f"max={max_seconds}".encode())
    for p in voice.inputs():
        st = p.stat()
        h.update(f"{p.name}:{st.st_size}:{st.st_mtime_ns}".encode())
    return h.hexdigest()[:16]


def paths(voice: Voice) -> dict[str, Path]:
    d = voice.reference_dir
    return {"merged": d / "merged.wav", "reference": d / "reference.wav", "transcript": d / "transcript.txt",
            "voice": d / "voice.gguf", "meta": d / "reference.json"}


def merge(voice: Voice, max_seconds: float) -> bool:
    """Build merged.wav and reference.wav if the inputs changed. True when they were rebuilt."""
    p = paths(voice)
    inputs = voice.inputs()
    if not inputs:
        raise UserError(f"no input audio: add some with `voice-clone add-audio {voice.name} <files...>`")
    signature = _inputs_signature(voice, max_seconds)
    meta = read_json(p["meta"], {})
    if meta.get("signature") == signature and p["reference"].exists():
        info(f"reference.wav is up to date ({meta.get('reference_seconds', 0):.1f} s)")
        return False

    step(f"Merging {len(inputs)} input files")
    tmp = voice.reference_dir / "parts"
    tmp.mkdir(parents=True, exist_ok=True)
    parts = []
    for i, src in enumerate(inputs):
        dst = tmp / f"{i:03d}.wav"
        audio.convert(src, dst, rate=RATE, filters=audio.TRIM_SILENCE)
        if audio.wav_duration(dst) < 0.3:
            info(f"skipping {src.name}: silent after trimming")
            dst.unlink()
            continue
        parts.append(dst)
    audio.concatenate(parts, p["merged"], rate=RATE)
    for part in parts:
        part.unlink()
    tmp.rmdir()
    merged_seconds = audio.wav_duration(p["merged"])
    info(f"merged.wav: {merged_seconds:.1f} s")

    if merged_seconds <= max_seconds:
        p["reference"].write_bytes(p["merged"].read_bytes())
    else:
        point = audio.cut_point(audio.silences(p["merged"]), max_seconds)
        if point is None or point < min(5.0, max_seconds / 2):
            point = max_seconds  # no usable pause: hard cut
            info("no pause found near the limit; cutting at exactly the limit (check the transcript's last word)")
        audio.cut(p["merged"], p["reference"], point)
    ref_seconds = audio.wav_duration(p["reference"])
    info(f"reference.wav: {ref_seconds:.1f} s (limit {max_seconds:.0f} s)")
    if ref_seconds < 6:
        info("note: under 6 s of reference audio; 10-30 s gives a more faithful voice")

    # A new reference invalidates the transcript and the voice.
    for key in ("transcript", "voice"):
        p[key].unlink(missing_ok=True)
    write_json(p["meta"], {"signature": signature, "merged_seconds": merged_seconds, "reference_seconds": ref_seconds,
                           "inputs": [i.name for i in inputs]})
    return True


def transcribe(voice: Voice) -> str:
    p = paths(voice)
    if p["transcript"].exists() and p["transcript"].read_text(encoding="utf-8").strip():
        text = p["transcript"].read_text(encoding="utf-8").strip()
        info(f"transcript.txt (kept): {text}")
        return text
    step("Transcribing the reference with Qwen3-ASR")
    texts = speech.transcribe([p["reference"]], voice.language.code)
    text = " ".join(texts.values()).strip()
    if not text:
        raise UserError("the recognizer heard no words in reference.wav")
    p["transcript"].write_text(text + "\n", encoding="utf-8")
    info(f"transcript.txt: {text}")
    info(f"Check it against reference.wav: if any word is wrong, edit {p['transcript']} and run this again.")
    return text


def build_voice(voice: Voice, text: str) -> None:
    p = paths(voice)
    if p["voice"].exists() and p["voice"].stat().st_mtime >= max(p["transcript"].stat().st_mtime,
                                                                p["reference"].stat().st_mtime):
        info("voice.gguf is up to date")
        return
    step("Building the Drama 3 voice")
    p["voice"].unlink(missing_ok=True)
    speech.make_voice([(p["reference"], text)], p["voice"])
    if not p["voice"].exists():
        raise UserError("speech voice did not write voice.gguf")
    info(f"voice: {p['voice']}")


def build(voice: Voice, max_seconds: float = DEFAULT_MAX_SECONDS, retranscribe: bool = False) -> None:
    merge(voice, max_seconds)
    if retranscribe:
        paths(voice)["transcript"].unlink(missing_ok=True)
    text = transcribe(voice)
    build_voice(voice, text)


def try_voice(voice: Voice, text: str, out: Path, seed: int = 1) -> Path:
    """Speak one sentence with Drama 3 and the voice, to hear it before generating the dataset."""
    p = paths(voice)
    if not p["voice"].exists():
        raise UserError(f"no voice yet: run `voice-clone reference {voice.name}`")
    with speech.SpeechServer(p["voice"], voice.logs_dir / "serve.log") as server:
        wav, stop = server.synthesize(text, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(wav)
    info(f"{out} ({audio.wav_bytes_duration(wav):.1f} s, stop: {stop})")
    return out


def status(voice: Voice) -> dict:
    p = paths(voice)
    meta = read_json(p["meta"], {})
    transcript = p["transcript"].read_text(encoding="utf-8").strip() if p["transcript"].exists() else None
    return {"reference_seconds": meta.get("reference_seconds") if p["reference"].exists() else None,
            "transcript": transcript, "voice": p["voice"].exists()}

