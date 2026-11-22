# Overview

voice-clone turns a few recordings of a voice into a **Piper TTS model**: a ~60 MB ONNX file that speaks that voice
in real time on an ordinary CPU, with no GPU and no large model at run time. It also includes an **audio studio**
that turns text into audio files with any Piper model: yours, one from Piper's catalog, or one imported by hand.

## The idea: distillation

Two very different models are used:

| | Drama 3 (teacher) | Piper (student) |
|---|---|---|
| Size | 20.8 GB | ~60 MB |
| Needs | A GPU (or a lot of RAM and time) | A CPU |
| Voice cloning | From a few seconds of audio (zero-shot) | Learns one voice by training |
| Speed | ~1.7× real time on an RTX 4090 | ~50× real time on a desktop CPU |

Drama 3 can imitate a voice from a short reference, but it is too big to run everywhere. Piper is tiny and fast,
but it has to be trained on about an hour of speech in the target voice. voice-clone uses Drama 3 to **produce that
hour of speech** (the dataset) and then fine-tunes Piper on it. Once the model is exported, Drama 3 is no longer
needed.

## The five stages

```
 your recordings ──► 1. audio ──► 2. reference ──► 3. dataset ──► 4. training ──► 5. export
   (any format)      input/       Drama 3 voice    ~1 h of speech   Piper fine-tune   .onnx + samples
```

1. **Audio** ([details](pipeline/01-audio.md)): recordings of the voice are copied into the voice's workspace.
2. **Reference** ([details](pipeline/02-reference.md)): the recordings are trimmed and merged, cut at a pause to at
   most 30 s, transcribed by Qwen3-ASR (the transcript can be corrected), and turned into a Drama 3 voice file
   (`voice.gguf`).
3. **Dataset** ([sentences](pipeline/03-sentences.md), [generation](pipeline/04-dataset.md)): about 1000 sentences
   are chosen from Common Voice (public domain). Drama 3 reads each one in the cloned voice; Qwen3-ASR listens to
   every clip and rejects those that skip, repeat or invent words, which are generated again with another seed.
   Short sentences get several takes and the one closest to the voice's usual pitch and rate is kept.
4. **Training** ([details](pipeline/05-training.md)): Piper is fine-tuned on the dataset, starting from an existing
   Piper voice in the same language (warm start), so it only has to learn the new timbre.
5. **Export** ([details](pipeline/06-export.md)): a checkpoint becomes `<locale>-<name>-medium.onnx` plus its
   `.onnx.json`, which then speaks a set of test sentences. Checkpoints can be compared by ear.

Every stage is **resumable**: stop it (Ctrl-C, `voice-clone stop`, a reboot) and run the same command again to
continue where it stopped. Every stage can also run as a **background job**, detached from the terminal and the
browser ([details](internals/jobs-and-status.md)).

## Architecture

```
                    ┌──────────────────────────────────────────────────────────┐
   ./voice-clone ──►│ app environment (.venv, Python 3.12)                     │
   voice-clone.cmd  │   cli.py ── web.py / web_studio.py (Gradio)              │
                    │     │                                                    │
                    │     ├─ workspace, audio, reference, sentences, dataset   │
                    │     ├─ train, export, pipeline, jobs, inhibit            │
                    │     ├─ studio: models, synthesis (piper-tts), lexicon,   │
                    │     │          history                                   │
                    │     └─ host, system, installer, settings, paths, util    │
                    └───────┬───────────────────────┬──────────────────────────┘
                            │ subprocess            │ subprocess
              ┌─────────────▼────────────┐ ┌────────▼──────────────────────────────┐
              │ speech.cpp (`speech`)    │ │ Piper environment                     │
              │  serve: Drama 3 (HTTP)   │ │ third_party/piper1-gpl/.venv          │
              │  asr:   Qwen3-ASR        │ │  PyTorch + Lightning (CUDA or CPU)    │
              │  voice: builds .gguf     │ │  piper_train.py, export_onnx, librosa │
              └──────────────────────────┘ └───────────────────────────────────────┘
                            │                                │
                       ffmpeg / ffprobe (decode, trim, merge, encode)
```

There are **two Python environments** on purpose:

- **The app environment** (`.venv`, managed by `uv` from `pyproject.toml` and `uv.lock`) holds the CLI, the web
  interface (Gradio, pandas) and the studio's synthesis (`piper-tts`, which brings onnxruntime). It is small and
  installs in seconds. The studio needs nothing else.
- **Piper's training environment** (`third_party/piper1-gpl/.venv`) holds PyTorch, Lightning and their CUDA
  libraries (several GB), plus Piper's training code pinned to a known-good commit. The app runs it only through
  subprocesses: training (`piper_train.py`), ONNX export, and pitch measurement (`prosody.py` uses librosa from
  there).

Keeping them apart means the CUDA stack never touches the app, the app starts fast, and the studio works on
machines that never ran `setup`.

## External tools

| Tool | Used for | Installed by |
|---|---|---|
| [uv](https://docs.astral.sh/uv/) | Both Python environments | The launcher (`./voice-clone`, `voice-clone.cmd`) |
| [speech.cpp](https://github.com/nyosegawa/speech.cpp) | Drama 3 (`speech serve`), Qwen3-ASR (`speech asr`), voice files (`speech voice`) | `voice-clone setup` |
| Drama 3 | Voice cloning, dataset generation | `setup` (`speech pull drama-3`, 20.8 GB) |
| Qwen3-ASR 0.6B | Transcribing the reference, verifying clips | `setup` (`speech pull qwen3-asr-0.6b`, 0.8 GB) |
| [Piper (piper1-gpl)](https://github.com/OHF-voice/piper1-gpl) | Training and export | `setup` (git clone at a pinned commit) |
| piper-tts 1.8.0 (PyPI) | The studio's synthesis | `uv sync` (app dependency) |
| ffmpeg / ffprobe | Decoding, trimming, merging, encoding, measuring | You (system package; `doctor` gives the command) |
| Common Voice sentences | Dataset text | Downloaded on demand into the cache |
| Piper checkpoints | Base voices for warm start | Downloaded on demand from `rhasspy/piper-checkpoints` |
| Piper voice catalog | Ready-made voices for the studio | Downloaded on demand from `rhasspy/piper-voices` |

## Where things live

| Path | Content |
|---|---|
| `voices/<name>/` | Everything about one voice: inputs, reference, sentences, clips, dataset, training, export |
| `studio/` | Studio data: downloaded and imported models, dictionaries, history, preferences |
| `third_party/piper1-gpl/` | Piper's training code and its environment |
| `settings.json` | The machine's mode (`gpu` or `cpu`) |
| `~/.cache/voice-clone/` (Linux), `%LOCALAPPDATA%\voice-clone\cache` (Windows) | Common Voice files, base checkpoints, the Piper catalog |
| `~/.cache/speech.cpp/` | Drama 3 and Qwen3-ASR models |

The full layout, with every file and its format, is in [Files and state](internals/files-and-state.md).
