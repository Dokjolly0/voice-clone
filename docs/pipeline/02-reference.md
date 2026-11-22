# Stage 2: the reference and the Drama 3 voice

Modules: `reference.py`, `speech.py` (the speech.cpp wrapper).

Drama 3 clones a voice from a **voice file** (`voice.gguf`), which speech.cpp builds from a short recording and its
**exact** transcript. This stage produces that file from the inputs.

```
input/*  ──trim+merge──►  merged.wav  ──cut at a pause──►  reference.wav  ──Qwen3-ASR──►  transcript.txt
                                                                  │                            │
                                                                  └────── speech voice ────────┴──► voice.gguf
```

## Steps (`reference.build`)

1. **Merge** (`merge`): every input is decoded to mono 44.1 kHz WAV with leading and trailing silence trimmed
   (`audio.TRIM_SILENCE`); inputs shorter than 0.3 s after trimming are skipped as silent. The parts are joined with
   0.4 s pauses into `merged.wav`.
2. **Cut** to at most `max_seconds` (default 30): if `merged.wav` is longer, it is cut in the middle of the last
   pause (silencedetect at −35 dB, at least 0.25 s) that starts before the limit. If there is no usable pause (none,
   or one before `min(5 s, limit/2)`), it is cut at exactly the limit and a note asks to check the last word.
   Result: `reference.wav`. Under 6 s, a note says 10–30 s gives a more faithful voice.
3. **Transcribe** (`transcribe`): `speech asr qwen3-asr-0.6b --language <code> --format json reference.wav` writes
   `transcript.txt`. An existing, non-empty transcript is **kept**: edit it to fix names and acronyms, and the next
   run uses it. `--retranscribe` deletes it first.
4. **Build the voice** (`build_voice`): `speech voice drama-3 reference.wav --transcript "<text>" voice.gguf`.

Each step is redone only when what it depends on changed:

- `reference.json` stores a **signature** of the inputs (name, size, mtime of each) and `max_seconds`; if it matches
  and `reference.wav` exists, merging is skipped. A new reference deletes the transcript and the voice.
- `voice.gguf` is rebuilt when it is older than `transcript.txt` or `reference.wav`, so saving an edited transcript
  is enough to rebuild it.

`reference.json`:

```json
{"signature": "3f9c0a1b2d4e5f60", "merged_seconds": 41.2, "reference_seconds": 28.7,
 "inputs": ["rec1.wav", "rec2.m4a"]}
```

## Importing an existing voice file

`reference NAME --voice-file x.voice.gguf` copies a Drama 3 voice made elsewhere (for example with `speech voice`
from several recordings) to `reference/voice.gguf` and writes `{"imported_voice": "x.voice.gguf"}`. From then on
`reference` keeps it and does not merge the inputs; `--from-inputs` drops it and builds from the inputs again.
`status` shows "imported voice x.voice.gguf".

## Trying the voice

`try-voice NAME "text" [-o out.wav] [--seed N]` starts `speech serve` with the voice, speaks one sentence and writes
`reference/try.wav`. Listen to it before generating the dataset: if this sounds wrong, the dataset will too. Fix the
transcript or the inputs first.

## The speech.cpp wrapper (`speech.py`)

| Function | Command | Notes |
|---|---|---|
| `transcribe(files, lang, batch=100)` | `speech asr qwen3-asr-0.6b [--device cpu] --language L --format json FILE...` | One process per batch of 100 files. Output is one JSON object per line (`{"file": …, "text": …}`). speech asr has been seen to exit with an error after printing every result: complete output is accepted with a warning; otherwise the batch is retried once |
| `make_voice(recordings, out)` | `speech voice drama-3 WAV --transcript TEXT [WAV --transcript TEXT …] OUT` | |
| `SpeechServer(voice_file, log_file)` | `speech serve drama-3 [--device cpu] --port P --add-voice voice=<file>` | A context manager: picks a free port, waits up to 30 min for `/health` (loading Drama 3 takes about a minute), logs to `logs/serve.log`, terminates the server on exit |
| `SpeechServer.synthesize(text, seed, max_seconds=30)` | `POST /v1/audio/speech` with `{"input", "voice", "seed", "max_seconds"}` | Returns the WAV bytes and the stop reason from the `X-Speech-Stop` header: `complete`, `max_seconds` or `model_limit` |

The model names are `SYNTH_MODEL = "drama-3"` and `ASR_MODEL = "qwen3-asr-0.6b"` in `installer.py`. In CPU mode
every call gets `--device cpu`.

## Tips

- The transcript must match the audio word for word, including fillers and false starts if they are audible. A
  wrong word makes Drama 3 mispronounce or garble parts of every sentence.
- A reference that ends mid-word (hard cut) gives a worse voice: add a pause near the limit or change
  `--max-seconds`.
- 20–30 s is plenty. Drama 3 prompts every request with the whole reference, so a longer one costs speed and leaves
  less room for the text.
