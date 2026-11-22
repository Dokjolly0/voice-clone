# voice-clone

[![CI](https://github.com/Dokjolly0/voice-clone/actions/workflows/ci.yml/badge.svg)](https://github.com/Dokjolly0/voice-clone/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/Dokjolly0/voice-clone?include_prereleases)](https://github.com/Dokjolly0/voice-clone/releases)

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

Once you have a voice (yours, one from Piper's catalog, or any Piper model), the **audio studio** turns text into
audio files: MP3, WAV, OGG, Opus, FLAC or M4A, with speed, volume, pitch, expressiveness, pauses, a pronunciation
dictionary, one file per line and a history. See [Audio studio](#audio-studio).

Everything is resumable: stop any stage (Ctrl-C, `voice-clone stop`, a reboot) and run it again to continue where
it was.

## Requirements

- Linux with `dnf` (Fedora) or `apt` (Ubuntu/Debian), tested on Fedora 45; or **Windows 10/11** (64-bit), see
  [Windows](#windows).
- An **NVIDIA GPU** with the proprietary driver: 8 GB of VRAM at the very least, 16–24 GB recommended (Drama 3
  alone uses up to 15 GB of RAM and 7+ GB of VRAM). Tested on an RTX 4090.
  Without one, a **CPU-only mode** works too, much more slowly: see [CPU-only mode](#cpu-only-mode).
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

`setup` picks GPU mode when a working NVIDIA driver is found and CPU mode otherwise; force one with
`--device gpu` or `--device cpu` (the choice is saved in `settings.json`).

If system packages are missing, `setup` stops and prints the one `sudo` command to run, for example
`sudo dnf install -y gcc-c++ cmake ninja-build git`; run it and run `setup` again. Without a working NVIDIA driver
it prints the driver install steps for your distribution. `./voice-clone doctor` shows the same checks without
changing anything.

Then, either from the **web interface**:

```sh
./voice-clone web        # open http://127.0.0.1:7860
```

The page (in Italian) is a guided wizard: the sidebar lists your voices and the five steps, each marked done ✓,
running ▶, to do ○ or locked 🔒 until the step before it is done. The main area shows one step at a time with a
single main action; rarely needed settings sit under "Opzioni avanzate". At the bottom, the job panel shows what is
running, a progress bar with counts and time left, errors, a Stop button and the log. Step 4 charts the training
quality (val_mel per epoch); step 5 plays the test sentences, lets you type any sentence, offers the model files for
download, and compares two checkpoints side by side.

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
| `setup [--skip-models] [--device auto\|gpu\|cpu]` | install and verify everything (no root needed; prints the sudo command if packages are missing) |
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
| `speak MODEL "text" \| --file F [-o out.mp3] [options]` | generate audio with any studio model (see [Audio studio](#audio-studio)) |
| `models list \| import F.onnx [F.onnx.json] \| catalog [--lang it] \| download KEY \| remove ID` | the studio's model library and Piper's catalog |
| `lexicon list \| add \| remove \| test --lang L` | pronunciation dictionary per language |
| `checkpoints NAME` | list the kept checkpoints and their val_mel |
| `compare NAME A B` | export two checkpoints (`last`, `best`, an epoch number) with the same test sentences into `export/variants/` |
| `promote NAME LABEL` | make a compared checkpoint (e.g. `epoch-1816`) the voice's model |
| `run NAME [options of the stages above]` | every stage in order, skipping what is done, resuming training |
| `status [NAME] [--json]` | where each stage stands |
| `log NAME [-f]` / `stop NAME` | follow or cleanly stop the background job |
| `web [--host] [--port] [--share]` | the web interface |

`reference`, `dataset`, `train`, `export`, `compare` and `run` accept `--background`: the job keeps running after you close
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
| dataset, 1000 sentences | about 1 h (estimate: Drama 3 ran at 1.7× real time in a short run, plus checks and retries) |
| training, 2000 epochs on ~1 h of audio | about 5 h (measured: ≈ 9 s per epoch at batch size 32) |
| export | under a minute |

On a GPU with less memory, training uses a smaller batch size, chosen automatically, and takes longer.

Disk: each checkpoint is about 850 MB. Training keeps the five best by val_mel (across resumed runs) and the latest
`last.ckpt`, about 5 GB per voice, and deletes older ones before and after every run.

## Audio studio

The web page's **Genera audio** section (switch at the top of the sidebar), and `voice-clone speak` on the command
line, turn text into audio with any Piper model:

- **Models**:
  - your trained voices appear by themselves;
  - **Piper's catalog** (183 official voices in 53 languages, filtered by language and quality) downloads with checksum
    verification: `voice-clone models catalog --lang it`, then `voice-clone models download it_IT-paola-medium`;
  - any Piper model (`name.onnx` + `name.onnx.json`, e.g. trained on another machine) can be **imported**.
- **Settings**:
  - format MP3 / WAV / OGG / Opus / FLAC / M4A, with bitrate and sample rate;
  - speed 0.5–2× (approximate: punctuation pauses change less);
  - volume ±20 dB, loudness normalization to −16 LUFS;
  - pitch ±12 semitones at the same duration (rubberband; past ±4 it sounds processed);
  - expressiveness 0–1.5 (Piper's natural variation in intonation and rhythm, not emotions);
  - speaker, for multi-speaker models.
- **Pauses**: a blank line separates paragraphs (paragraph pause); line breaks and sentences get the sentence pause;
  `[pausa 1.5s]`, `[pause 500ms]` or `[break 2s]` add an exact pause anywhere.
- **Pronunciation dictionary** per language: "written → spoken" rules for acronyms, names and brands, as text
  (`ACI → A C I`) or Piper phonemes (`Coop → [[ kˈɔp ]]`). Whole words and any case by default; never applied inside
  `[[ ]]` or pause markers. `voice-clone lexicon add --lang it ACI "A C I"`.
- **One file per line**: each non-empty line becomes its own file (`001-first-words.mp3`, …), downloaded as a .zip
  (`--per-line FOLDER` on the command line). Handy for a kiosk's set of messages.
- **History**: every generation from the web page is kept (the newest 200) with its text, model and settings, to
  replay, download, reuse or delete.

The studio runs on the CPU with `piper-tts` (onnxruntime) in the app's own environment: no GPU, PyTorch or `setup`
needed. Measured on a Ryzen 9 7950X3D: 226 s of audio in 4.3 s (~50× real time); a 4-core CPU still runs several
times faster than real time. Its data lives in `studio/` (`VOICE_CLONE_STUDIO` moves it).

```sh
./voice-clone speak paola "Benvenuti. [pausa 1s] Il negozio chiude alle diciannove." \
    --format mp3 --speed 1.1 --pitch -1 --paragraph-pause 1 -o annuncio.mp3
./voice-clone speak imported:pos --file messaggi.txt --per-line messaggi/ --format wav --sample-rate 44100
```

## Windows

Everything works natively on Windows 10/11, training included. Use `voice-clone.cmd` (from cmd or PowerShell)
wherever this README says `./voice-clone`:

```bat
git clone https://github.com/Dokjolly0/voice-clone.git
cd voice-clone
voice-clone.cmd doctor          rem lists what is missing, with the winget command for it
voice-clone.cmd setup
voice-clone.cmd web
```

Or download `voice-clone-<version>-windows.zip` from [Releases](https://github.com/Dokjolly0/voice-clone/releases)
and run `voice-clone.cmd` from the unpacked folder.

What you install yourself (`doctor` and `setup` print these commands when something is missing):

| | Command | Needed for |
|---|---|---|
| NVIDIA driver | from [nvidia.com/drivers](https://www.nvidia.com/drivers) or the NVIDIA app | GPU mode (it also brings Vulkan and `nvidia-smi`) |
| Git | `winget install -e --id Git.Git` | setup (Piper's training code) |
| FFmpeg | `winget install -e --id Gyan.FFmpeg` | everything that reads or writes audio |
| C++ Build Tools | `winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"` | setup: Piper's training code has compiled parts |

Open a new terminal after installing them, so PATH includes the new tools. The rest comes by itself, as on Linux:
`uv` (in `%USERPROFILE%\.local\bin`), speech.cpp's Vulkan build (in `%LOCALAPPDATA%\Programs\speech.cpp`, through
its `install.ps1`), and Piper's environment with the CUDA build of PyTorch that matches your driver. The cache is in
`%LOCALAPPDATA%\voice-clone\cache`.

Differences from Linux:

- Background jobs run without a console window; `voice-clone.cmd stop NAME` (or the web page) stops them cleanly.
- During the dataset and training stages Windows does not go to sleep by itself (sleep you ask for still happens).
- The audio studio only needs the app: `voice-clone.cmd speak ...` and the "Genera audio" page work without
  `setup`, without a GPU and without the Build Tools.
- WSL2 with Ubuntu is an alternative: there the Linux instructions apply as they are.

## CPU-only mode

Every stage also runs without a GPU: `./voice-clone setup --device cpu` (chosen automatically when no NVIDIA driver
works). speech.cpp then runs Drama 3 and Qwen3-ASR on the CPU, Piper gets a CPU-only PyTorch (no CUDA libraries),
and training runs on the CPU. The result is the same kind of model, but it takes much longer:

| | RTX 4090 | Ryzen 9 7950X3D (16 cores), CPU only |
|---|---|---|
| Drama 3 speed (measured) | 1.7× real time | 0.27× real time |
| Training speed (measured, batch 32) | ~0.3 s per batch | ~5.6 s per batch |
| Dataset of 1000 sentences (estimate) | ~1 h | ~4 h |
| 2000 training epochs on ~1 h of audio (estimate) | ~5 h | ~3 days |

- **RAM**: Drama 3 holds about 21 GB on the CPU; 32 GB of RAM is the practical minimum.
- **Smaller CPUs** are much slower: a 4-core desktop CPU (e.g. a 6th-generation Core i7) would need weeks of
  training, which is not practical. On such machines, train elsewhere (a GPU, or a rented cloud GPU for a few hours)
  and only *use* the exported model there: it runs in real time on modest CPUs.
- **Fewer epochs**: starting from a trained voice, 500–1000 epochs may already sound good; export mid-training
  (step 5, or `voice-clone export`) and listen to decide when to stop.
- The web page shows the mode in the sidebar and estimates times for it, and once training runs, from the voice's
  own measured epochs.
- `VOICE_CLONE_DEVICE=cpu` (or `gpu`) overrides the saved mode for one command.

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
  training/             train.log, metrics.csv (val_mel per epoch), progress.json,
                        lightning_logs/version_*/checkpoints/ (5 best by val_mel + latest last.ckpt)
  export/               <locale>-<name>-medium.onnx + .onnx.json, source.json (which checkpoint),
                        variants/<label>/ (compared checkpoints with their samples)
  samples/              test sentences spoken by the exported model
  logs/                 background job log and pid, speech serve log
```

The audio studio keeps its data in `studio/`: `models/` (downloaded and imported models), `lexicon/<lang>.json`,
`history/` and `preferences.json`.

Shared downloads are cached in `~/.cache/voice-clone` (Common Voice sentences, base checkpoints; on Windows
`%LOCALAPPDATA%\voice-clone\cache`) and
`~/.cache/speech.cpp` (models). Set `VOICE_CLONE_VOICES` or `VOICE_CLONE_CACHE` to move them.

## Using the voice

The two files in `export/` are all you need, and the `.onnx.json` must sit next to the `.onnx` with the same name.
They work with anything that runs Piper voices: the [`piper-tts`](https://pypi.org/project/piper-tts/) package
(`python3 -m piper -m it_IT-anna-medium.onnx -f out.wav -- "Text."`), Piper's C/C++ library, or ONNX Runtime
directly with espeak-ng for phonemes.

## Troubleshooting

- **`speech.cpp does not see the GPU`**: the NVIDIA driver or the Vulkan loader is missing. Install them, delete
  `~/.local/bin/speech` and run `setup` again: the installer picks the Vulkan build only when it finds Vulkan.
  (On Windows the build is always Vulkan: update the NVIDIA driver.)
- **Windows: Piper fails to compile** during `setup`: install the C++ Build Tools (see [Windows](#windows)), open a
  new terminal and run `setup` again.
- **Many clips rejected**: look at `state.json`; entries show the sentence (`text`) and what was heard (`heard`).
  If only spellings differ (the recognizer wrote "di un" for "d'un"), raise `--max-cer` a little (e.g. 0.08). If the
  audio really skips or repeats words, a few per hundred is normal and retries fix them.
- **`CUDA out of memory`** during training: `train NAME --resume --batch-size 16` (or 12, 8).
- **Training stopped**: `train NAME --resume`, or `run NAME`, which resumes by itself.

## Development

```sh
uv run pytest        # unit tests
```

CI (`.github/workflows/ci.yml`) runs the tests, the launchers, `doctor` and a real synthesis on Ubuntu and Windows
for every push and pull request. When it passes on `main`, `release.yml` publishes a release
`v<version>-build.<n>` with the wheel, the sdist and archives for Windows and Linux. Platform differences live in
`src/voice_clone/host.py`.

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
- **Piper catalog voices** each come with their own license, in the MODEL_CARD downloaded next to the model.
