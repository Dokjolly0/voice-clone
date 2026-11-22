# Files and state

Every file voice-clone reads or writes, where it lives and what it holds. Nothing outside these places is touched,
except what the installers of uv and speech.cpp manage themselves.

## The repository

```
voice-clone/
  voice-clone              Linux launcher (bash)
  voice-clone.cmd          Windows launcher (runs voice-clone.ps1)
  voice-clone.ps1          Windows launcher (PowerShell)
  pyproject.toml, uv.lock  the app's dependencies, pinned
  src/voice_clone/         the code (see the module map)
  tests/                   unit tests (pytest)
  docs/                    this documentation
  .github/workflows/       CI (ci.yml) and releases (release.yml)
  .gitattributes           line endings per file type; what release archives leave out
  .gitignore               the generated folders below are not versioned
```

Generated, not versioned: `.venv/`, `third_party/`, `voices/`, `studio/`, `settings.json`.

## `voices/<name>/` (one voice)

```
voices/<name>/
  voice.json             {"name", "lang", "created"}
  input/                 the recordings, as added (names sanitized)
  reference/
    merged.wav           all inputs trimmed and joined (44.1 kHz mono)
    reference.wav        what Drama 3 hears: merged.wav cut at a pause to ≤ max_seconds
    transcript.txt       its exact words (edit to fix; kept across runs)
    voice.gguf           the Drama 3 voice file
    reference.json       {"signature", "merged_seconds", "reference_seconds", "inputs"} or {"imported_voice"}
    try.wav              the last try-voice
  sentences.txt          dataset sentences, one per line; line N is clip N (append-only)
  domain-sentences.txt   the web page's uploaded domain sentences (when used)
  state.json             per-clip progress of the dataset and the voice's prosody profile
  clips/
    <id>.wav             Drama 3's output (44.1 kHz), kept while verified
    takes/<id>-<seed>.wav  takes of short sentences waiting to be chosen
  dataset/
    wav/<id>.wav         verified clips at 22050 Hz mono: Piper's input
    metadata.csv         <id>|<text> per line
  training/
    run.json             {"epochs", "device"} of the current run
    train.log            Piper's output; every run starts with "$ <command>"
    metrics.csv          epoch,val_mel,seconds,time per epoch
    progress.json        {"epoch", "max_epochs", "epoch_seconds", "val_mel", "updated"}
    cache/               Piper's phonemized dataset
    lightning_logs/version_N/
      checkpoints/       epoch=<n>-val_mel=<x>.ckpt (best) and last.ckpt, ~850 MB each, pruned
      config.yaml, hparams.yaml
  export/
    <locale>-<name>-medium.onnx       the model
    <locale>-<name>-medium.onnx.json  its configuration (written by Piper when training starts)
    source.json          {"checkpoint", "epoch", "val_mel"}: where the model came from
    say.wav              the web page's last "Leggi"
    variants/
      comparison.json    {"a", "b"}: the last comparison
      <label>/           <model>.onnx, samples/, source.json (with the checkpoint's mtime)
  samples/
    sample-<n>.wav       test sentences spoken by the exported model
    sentences.txt        sample-<n>.wav|<text>
  logs/
    job.json             the background job: pid, start time, args, log path
    job.log              its output (the previous one in job.previous.log)
    stop.request         Windows only, briefly: a stop asked for
    serve.log            speech serve's output
```

Clip ids are the 0-based line number, zero-padded to five digits (`00000`).

### `state.json`

```json
{
 "clips": {
  "<id>": {
   "text": "the sentence (line <id> of sentences.txt)",
   "attempts": 1,                 // generations so far (max_attempts = 4)
   "seed": 1,                     // seed of the kept clip (or of the last attempt)
   "verified": true,              // passed ASR (and, for short sentences, the prosody check)
   "heard": "what Qwen3-ASR heard",
   "cer": 0.0,                    // character error rate
   "rejected": "CER 0.12",        // why the last attempt failed (absent once verified)
   "best_of": 5,                  // short sentences: takes compared
   "prosody": {"seconds": 0.83, "f0": 158.2, "f0_range": 71.4, "distance": 0.92}
  }
 },
 "prosody_profile": {"f0": 154.0, "f0_range": 102.0, "rate": 15.1, "clips": 60}
}
```

Rejection reasons: `CER <x>`, `stopped: max_seconds`, `stopped: model_limit`, `empty audio`, `no take passed ASR`,
`best take too far from the voice's prosody (distance <x>)`.

### `metrics.csv`

```
epoch,val_mel,seconds,time
0,0.32570,9.41,1795310412
1,0.32900,9.38,1795310422
```

`epoch` is 0-based, `seconds` the epoch's own duration, `time` the Unix time at its end.

## `studio/` (the audio studio)

```
studio/
  models/<id>/           <name>.onnx, <name>.onnx.json, model.json (and MODEL_CARD for catalog voices)
  lexicon/<lang>.json    {"rules": [{"written", "spoken", "whole_word", "match_case"}]}
  history/<id>/          audio.<ext> or files/ + <id>.zip, and meta.json
  preferences.json       the web page's last settings
  tmp/                   previews and dictionary tests
```

`model.json`: `{"origin": "catalog"|"imported", "onnx": "<file>.onnx", "name", "language", "quality", "added"}`.

## Machine-wide

| Path | Content |
|---|---|
| `settings.json` (project root) | `{"device": "gpu"}` or `"cpu"` |
| `third_party/piper1-gpl/` | Piper's training code at the pinned commit, its `.venv`, compiled `espeakbridge` and `monotonic_align` |
| `third_party/.piper-ready` | `<commit> <device>`: Piper's environment is installed and checked |
| `<cache>/common-voice/<locale>/` | `index.json` and the downloaded sentence files |
| `<cache>/checkpoints/<lang>/<locale>/<voice>/medium/` | Base checkpoints (`config.json` + `.ckpt`) |
| `<cache>/piper-voices.json` | Piper's voice catalog, refreshed daily |
| `~/.cache/speech.cpp/` | Drama 3 and Qwen3-ASR (managed by speech.cpp) |
| `~/.local/bin/uv`, `~/.local/bin/speech` (Linux) | uv; the speech.cpp link |
| `~\.local\bin\uv.exe`, `%LOCALAPPDATA%\Programs\speech.cpp\` (Windows) | uv; speech.cpp |

`<cache>` is `~/.cache/voice-clone` on Linux and `%LOCALAPPDATA%\voice-clone\cache` on Windows, or
`VOICE_CLONE_CACHE`.

## Disk usage, roughly

| What | Size |
|---|---|
| Drama 3 + Qwen3-ASR | 21.6 GB |
| Piper's environment (GPU) | 5–7 GB |
| One voice: clips + dataset for 1000 sentences | ~0.5 GB |
| One voice: checkpoints (5 best + last) | ~5 GB |
| One exported model | ~60 MB |
| A catalog voice | 28–115 MB |
