# voice-clone

Clone a voice from a few recordings into a **lightweight [Piper](https://github.com/OHF-voice/piper1-gpl) TTS
model**: a ~60 MB ONNX file that speaks in real time on a CPU, without a GPU and without the large model that
created it.

```
 your recordings ──► 1. audio ──► 2. reference ──► 3. dataset ──► 4. training ──► 5. export
   (any format)      input/       Drama 3 voice    ~1 h of speech   Piper fine-tune   .onnx + samples
```

1. **Audio**: add one or more recordings of the voice (any format ffmpeg reads).
2. **Reference**: they are trimmed and merged into one file, cut at a pause to ≤ 30 s, transcribed with Qwen3-ASR
   (you can correct the transcript), and turned into a voice for **Drama 3**, a large model that imitates a voice
   from a few seconds of audio.
3. **Dataset**: Drama 3 reads *n* sentences (default **1000**, about an hour of speech) in the cloned voice. Every
   clip is checked by speech recognition; clips that skip, repeat or invent words are generated again with another
   seed.
4. **Training**: Piper is fine-tuned on that dataset, starting from an existing Piper voice in the same language.
5. **Export**: the checkpoint becomes `<locale>-<name>-medium.onnx`, which then speaks test sentences.

Everything is resumable: stop any stage (Ctrl-C, `voice-clone stop`, a reboot) and run it again to continue where
it was.

## Requirements

- Linux with `dnf` (Fedora) or `apt` (Ubuntu/Debian). Tested on Fedora 45.
- An **NVIDIA GPU** with the proprietary driver: 8 GB of VRAM at the very least, 16–24 GB recommended (Drama 3
  alone uses up to 15 GB of RAM and 7+ GB of VRAM). Tested on an RTX 4090.
- About 30 GB of free disk space (Drama 3 is 20.8 GB) and an internet connection for the first setup.

You do **not** need to install Python, PyTorch or CUDA yourself: the project brings its own.

## Quick start

```sh
cd voice-clone
./voice-clone setup
```

`./voice-clone` is the entry point: on first use it installs [uv](https://docs.astral.sh/uv/) and a private
Python 3.12 environment. `setup` then checks the system and installs everything else, which takes 20–40 minutes,
mostly downloads:

- **speech.cpp** (Vulkan build) in `~/.local/bin`, with **Drama 3** and **Qwen3-ASR 0.6B** in `~/.cache/speech.cpp`;
- **Piper's training code**, pinned to a tested commit, in `third_party/piper1-gpl` with its own PyTorch + CUDA
  environment, and its compiled extensions built.

If system packages are missing, `setup` stops and prints the one `sudo` command to run, for example
`sudo dnf install -y gcc-c++ cmake ninja-build git`; run it and run `setup` again. Without a working NVIDIA driver
it prints the driver install steps for your distribution. `./voice-clone doctor` shows the same checks without
changing anything.

Then, either from the **web interface**:

```sh
./voice-clone web        # open http://127.0.0.1:7860
```

or from the **command line**:

```sh
./voice-clone new anna --lang it
./voice-clone add-audio anna ~/recordings/*.wav
./voice-clone reference anna                  # check reference/transcript.txt, fix it, run again if needed
./voice-clone try-voice anna "Buongiorno, come posso aiutarti?"   # listen to reference/try.wav
./voice-clone run anna --background           # dataset + training + export: several hours
./voice-clone status anna                     # anytime; `voice-clone log anna -f` to follow
```

The result is in `voices/anna/export/`, and test sentences are in `voices/anna/samples/`.

## Getting a good voice

- **Recordings**: one speaker, clean speech, no music or background noise, 10–60 s in total. More is not better
  for the reference: only 30 s (`--max-seconds`) reach Drama 3.
- **Transcript**: Drama 3 needs the *exact* words of the reference. Recognition errors are common on names and
  acronyms (in testing it heard "Socio AC" for "Socio ACI"). Edit `reference/transcript.txt`, or edit it in the web
  page, and run `reference` again: the voice is rebuilt from your text.
- **Listen before training**: `try-voice` for one sentence, then a few files of `dataset/wav/` (or "Listen to random
  clips" in the web page). Piper learns exactly what it hears there; if the dataset voice is not right, fix the
  reference first.
- **Domain sentences**: if the voice will say specific phrases (product names, a kiosk's messages), put them in a
  text file, one per line, and pass `--extra phrases.txt`. They are always included, ahead of the generic sentences.
- **Data size**: 1000 sentences (~1 h) is a good default. If the final voice sounds metallic after full training,
  grow the dataset (`dataset anna -n 1500`: only new sentences are generated) and train again.

## Commands

| Command | What it does |
|---|---|
| `setup [--skip-models]` | install and verify everything (no root needed; prints the sudo command if packages are missing) |
| `doctor` | report what is installed, change nothing |
| `languages` | list supported languages and their base Piper voice |
| `new NAME --lang L` | create the voice workspace `voices/NAME/` |
| `list` | list voices |
| `add-audio NAME FILES...` / `remove-audio NAME FILES...` | manage input recordings |
| `reference NAME [--max-seconds 30] [--retranscribe]` | merge inputs, transcribe, build the Drama 3 voice |
| `try-voice NAME "text"` | speak one sentence with Drama 3 into `reference/try.wav` |
| `sentences NAME [-n 1000] [--extra FILE]` | choose the dataset sentences only |
| `dataset NAME [-n 1000] [--extra FILE] [--max-cer 0.05] [--max-attempts 4]` | generate, verify and export the dataset |
| `train NAME [--epochs 2000] [--batch-size N] [--resume]` | fine-tune Piper |
| `export NAME [--checkpoint last\|best\|PATH] [--sample "text"]` | ONNX export and test sentences (works mid-training) |
| `say NAME "text" [-o out.wav]` | speak with the exported model |
| `run NAME [options of the stages above]` | every stage in order, skipping what is done, resuming training |
| `status [NAME] [--json]` | where each stage stands |
| `log NAME [-f]` / `stop NAME` | follow or cleanly stop the background job |
| `web [--host] [--port] [--share]` | the web interface |

`reference`, `dataset`, `train`, `export` and `run` accept `--background`: the job keeps running after you close
the terminal or the browser, and only one job per voice runs at a time. `dataset`, `train` and `run` keep the
computer from suspending while they run (`systemd-inhibit`; closing a laptop lid can still suspend it).

## Languages

| Code | Language | Base voice |
|---|---|---|
| it | Italian | it_IT paola |
| en | English (US) | en_US lessac |
| de | German | de_DE thorsten |
| es | Spanish | es_ES davefx |
| fr | French | fr_FR siwis |
| pt | Portuguese (Brazil) | pt_BR faber |
| nl | Dutch | nl_NL pim |
| pl | Polish | pl_PL gosia |
| ru | Russian | ru_RU denis |
| ro, tr, cs, hu, fi, el, vi, zh | Romanian, Turkish, Czech, Hungarian, Finnish, Greek, Vietnamese, Mandarin | see `voice-clone languages` |

A language needs all four tools: Drama 3 speaks it, Qwen3-ASR recognizes it, Common Voice has public-domain
sentences in it, and a Piper checkpoint exists in it. **Italian has been tested end to end**; for the others the
sentence sources and checkpoints exist, but no full run has been done. To add a language, add an entry to
`src/voice_clone/languages.py`.

## Time and resources (RTX 4090)

| Stage | Time |
|---|---|
| setup | 20–40 min, mostly the 21 GB download |
| reference | 1–2 min |
| dataset, 1000 sentences | roughly 1–2 h (estimate: Drama 3 ran at 1.7× real time in a short run, plus checks and retries) |
| training, 2000 epochs on ~1 h of audio | about 5 h (measured: ≈ 9 s per epoch at batch size 32) |
| export | under a minute |

On a GPU with less memory, training uses a smaller batch size, chosen automatically, and takes longer.

## Files

```
voices/<name>/
  voice.json            name and language
  input/                your recordings
  reference/            merged.wav, reference.wav, transcript.txt, voice.gguf, try.wav
  sentences.txt         dataset sentences; line N is clip N (the list only grows)
  state.json            per-clip progress: seed, attempts, what the recognizer heard, error rate
  clips/                Drama 3 output
  dataset/              wav/ at 22050 Hz + metadata.csv (Piper's input)
  training/             train.log, lightning_logs/version_*/checkpoints/ (5 best by val_mel + last.ckpt)
  export/               <locale>-<name>-medium.onnx + .onnx.json
  samples/              test sentences spoken by the exported model
  logs/                 background job log and pid, speech serve log
```

Shared downloads are cached in `~/.cache/voice-clone` (Common Voice sentences, base checkpoints) and
`~/.cache/speech.cpp` (models). Set `VOICE_CLONE_VOICES` or `VOICE_CLONE_CACHE` to move them.

## Using the voice

The two files in `export/` are all you need, and the `.onnx.json` must sit next to the `.onnx` with the same name.
They work with anything that runs Piper voices: the [`piper-tts`](https://pypi.org/project/piper-tts/) package
(`python3 -m piper -m it_IT-anna-medium.onnx -f out.wav -- "Text."`), Piper's C/C++ library, or ONNX Runtime
directly with espeak-ng for phonemes.

## Troubleshooting

- **`speech.cpp does not see the GPU`**: the NVIDIA driver or the Vulkan loader is missing. Install them, delete
  `~/.local/bin/speech` and run `setup` again: the installer picks the Vulkan build only when it finds Vulkan.
- **Many clips rejected**: look at `state.json`; entries show the sentence (`text`) and what was heard (`heard`).
  If only spellings differ (the recognizer wrote "di un" for "d'un"), raise `--max-cer` a little (e.g. 0.08). If the
  audio really skips or repeats words, a few per hundred is normal and retries fix them.
- **`CUDA out of memory`** during training: `train NAME --resume --batch-size 16` (or 12, 8).
- **Training stopped**: `train NAME --resume`, or `run NAME`, which resumes by itself.

## Development

```sh
uv run pytest        # unit tests
```

Design notes:

- The app (CLI, web UI) and Piper's training each have their own environment: PyTorch, Lightning and CUDA stay out
  of the app's, and the Piper training code is pinned in `src/voice_clone/paths.py`.
- `src/voice_clone/piper_train.py` replaces `python -m piper.train`, whose default callbacks also track `val_mos`,
  a score that needs `torchaudio`. `torchaudio` has no build for the PyTorch version Piper installs, so upstream
  training stops after the first epoch.
- Seeds are fixed per sentence and attempt (`1 + index × 100 + attempt`), so the dataset is reproducible.

## Licenses and rights

- **The voice**: make sure you have the right to clone the voice in your recordings and to use the result.
- **Drama 3** is under the [Fish Audio Community License](https://huggingface.co/fishaudio/drama-3/blob/main/LICENSE):
  check its terms for commercial use and for **training another model on its output**, which is what this project
  does.
- **Piper** (installed separately into `third_party/`, not part of this repository) is GPL-3.0; base checkpoints come
  from [rhasspy/piper-checkpoints](https://huggingface.co/datasets/rhasspy/piper-checkpoints), each with its own
  model card.
- **Common Voice** sentences are public domain (CC0).
