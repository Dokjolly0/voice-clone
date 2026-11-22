# Stage 1: voices and audio

Modules: `workspace.py` (the voice workspace), `audio.py` (ffmpeg helpers), `languages.py` (language definitions).

## The voice workspace

A voice is a folder `voices/<name>/` (or under `VOICE_CLONE_VOICES`) with a `voice.json`:

```json
{"name": "anna", "lang": "it", "created": "2026-11-21T18:02:11"}
```

`workspace.Voice` is a small dataclass (`name`, `dir`) whose properties give every path of the voice
(`input_dir`, `reference_dir`, `sentences_file`, `clips_dir`, `dataset_dir`, `training_dir`, `export_dir`,
`samples_dir`, `logs_dir`, `state_file`) and a few derived values:

| Property | Value |
|---|---|
| `language` | The `Language` of `voice.json`'s `lang` |
| `model_name` | `<locale>-<name>-medium`, e.g. `it_IT-anna-medium` (locale from the base checkpoint path) |
| `onnx_path`, `onnx_config_path` | `export/<model_name>.onnx` and `.onnx.json` |
| `inputs()` | The audio files in `input/`, sorted |
| `sentences()` | The non-empty lines of `sentences.txt` |
| `load_state()`, `save_state()` | `state.json` (atomic write) |

Functions: `create(name, lang)` (validates the name with `^[a-z0-9][a-z0-9_-]{0,39}$` and the language, refuses an
existing voice), `load(name)` (lists the existing ones in the error), `list_names()`.

## Adding audio

`add_audio(voice, files)` copies each file into `input/`:

1. the file must exist and have a known extension (`audio.AUDIO_EXTENSIONS`: `.wav .mp3 .flac .ogg .opus .m4a .aac
   .wma .webm .mp4 .mkv`);
2. `audio.duration()` runs ffprobe on it: a file ffprobe cannot read is refused ("not a readable audio file");
3. the name is sanitized (anything but letters, digits, `.`, `-` becomes `_`; the extension is lowercased) and
   de-duplicated (`take.wav`, `take-2.wav`, …);
4. the copy keeps the file's timestamps (`shutil.copy2`).

`remove_audio(voice, names)` deletes input files by name. Changing the inputs invalidates the reference
(see [stage 2](02-reference.md)): its signature includes every input's name, size and modification time.

### What makes good input

- One speaker, clean speech, no music or background noise, no reverb.
- 10–60 seconds in total is enough. The reference Drama 3 hears is cut to at most 30 s (configurable); more input
  only helps by giving a better choice of material.
- The style of the recordings carries over: a calm, neutral reading gives a calm, neutral voice.

## Audio helpers (`audio.py`)

| Function | What it does |
|---|---|
| `duration(path)` | Duration via ffprobe, cached per (path, size, mtime) because status pages ask often |
| `wav_duration(path)`, `wav_bytes_duration(data)` | Duration of a WAV file or WAV bytes, with Python's `wave` |
| `ffmpeg(args, capture_stderr=False)` | Runs `ffmpeg -nostdin -hide_banner -loglevel error -y …`; raises `UserError` with the end of stderr on failure; returns stderr (used by silencedetect) |
| `convert(src, dst, rate, filters=None)` | Decodes anything to mono 16-bit WAV at `rate`, optionally through an `-af` filter |
| `TRIM_SILENCE` | Filter that removes silence below −45 dB at both ends (silenceremove, reverse, silenceremove, reverse) |
| `concatenate(parts, dst, rate, gap=0.4)` | Joins WAV files with `gap` seconds of silence between them (apad + concat) |
| `silences(path, noise_db=-35, min_duration=0.25)` | (start, end) of each silence found by `silencedetect` |
| `cut_point(silences, limit)` | The middle of the last silence that starts before `limit`, so a cut never splits a word |
| `cut(src, dst, seconds)` | Copies the first `seconds` of a file |

## Languages (`languages.py`)

A language works end to end when Drama 3 speaks it, Qwen3-ASR recognizes it, Common Voice has CC0 sentences for it
and a Piper checkpoint exists to start from. Each `Language` has:

| Field | Meaning | Default |
|---|---|---|
| `code` | What users type (`--lang it`) | |
| `name` | Display name | |
| `common_voice` | Folder under Common Voice's `server/data/` | |
| `checkpoint` | Folder under `rhasspy/piper-checkpoints` (quality medium), e.g. `it/it_IT/paola/medium` | |
| `min_chars`, `max_chars` | Length of normal dataset sentences | 30, 150 |
| `short_min_chars` | Shortest "short" sentence (short = `short_min_chars`..`min_chars − 1`) | 12 |
| `short_share` | Share of short sentences among those added | 0.15 |
| `chars_per_second` | Speaking rate, only for estimates | 15 |
| `samples` | Test sentences spoken after export | |

`locale` is derived from the checkpoint path (`it_IT`). Supported: it, en (US), de, es, fr, pt (Brazil), nl, pl, ru,
ro, tr, cs, hu, fi, el, vi, zh (Mandarin; shorter sentence bounds 8–50, short from 4, 4.5 characters per second).
The espeak-ng voice and phoneme type come from the base checkpoint's `config.json` at training time.
