# Validation checklist

Manual end-to-end checks of what the automatic tests and CI cannot cover: the GPU, Drama 3, training, background
jobs on a real desktop, sleep prevention and the web page. Run them after changes to the platform code, the
installer or a stage, and before telling someone a platform works.

How to use it: tick each box, write the result next to it (version, time, numbers), and paste any error into the
"Notes" of that section. Commands are written for Linux (`./voice-clone`) and Windows (`voice-clone.cmd`).

**Status**

| Platform | Machine | Version / commit | Date | Result |
|---|---|---|---|---|
| Linux | Fedora 45, RTX 4090 (driver 620.70), Ryzen 9 7950X3D, 62 GB | `70fde91` (merge of the port, with the `--init-from` fix `f6fe5ba`), then CI on `fc59ead` | 2026-11-22 | ✅ passed |
| Windows | | | | ⬜ to do |

---

## Automatic checks (CI)

Every push runs `.github/workflows/ci.yml` on Ubuntu and Windows ([details](../development.md#ci-and-releases)).

- [x] Ubuntu: lock file, unit tests, launcher, `doctor`, `languages`, studio synthesis (MP3, WAV, OGG per line)
  — green on `fc59ead`
- [x] Windows: the same with `voice-clone.cmd` — green on `fc59ead` (the first run on `70fde91` found two Windows
  bugs, fixed in `11319a4` and `fc59ead`)
- [x] Release on `main`: `v0.1.0-build.2` with the Windows zip, the Linux tar.gz, wheel, sdist and `SHA256SUMS`

---

## Linux (done)

Fedora 45, GPU mode. All checks below passed on 2026-11-22.

### Installation

- [x] `./voice-clone setup --skip-models` on an existing install: every part reported as already installed, PyTorch
  not reinstalled ("Piper training code (39ac0a0, PyTorch for CUDA): already installed and working")
- [x] `./voice-clone doctor`: every line `[ok]`, "Everything is ready."
- [x] `uv run pytest`: 104 passed
- [x] Fresh install (earlier, 2026-11-21): `setup` installed speech.cpp (Vulkan build), Piper's environment with
  CUDA and both models

### Pipeline on a test voice (`provawin`, deleted afterwards)

- [x] `new provawin --lang it`, `add-audio` with 8 files (32.5 s)
- [x] `reference provawin --voice-file voices/alfeo/reference/voice.gguf` (imported voice)
- [x] `sentences provawin -n 20`
- [x] `dataset provawin --limit 6 --short-best-of 2 --background`: Drama 3 via `speech serve` on the GPU
  (RTFx 1.3–1.5), best take of 3 short sentences, 6/6 verified by Qwen3-ASR, dataset exported
- [x] Sleep blocked while the job ran: `systemd-inhibit --list` showed
  `voice-clone … sleep:idle … generating the dataset of provawin`
- [x] `job.json` has `pid`, `created`, `args`, `started`
- [x] `train provawin --epochs 6 --batch-size 4 --background --init-from <relative path>`: found and fixed a bug
  (relative `--init-from` was looked up in the training folder; commit `f6fe5ba`)
- [x] `stop provawin` during training: "stopped; run the same command again to resume", log ends with
  "interrupted", `last.ckpt` kept
- [x] `train provawin --epochs 6 --batch-size 4 --resume`: resumed from `last.ckpt`, "Training finished"
- [x] `status provawin`: "epoch 6/6 (finished)"
- [x] `export provawin --checkpoint last`: model exported, 5 test sentences spoken
- [x] `say provawin "…" -o say.wav`

### Studio

- [x] `speak voice:alfeo "Buongiorno, è tutto pronto." --format ogg --pitch 2 --speed 1.1` (rubberband pitch)
- [x] `speak imported:alfeo "…" --format mp3`
- [x] `models download it_IT-paola-medium` (md5 checked), `models list`, `speak it_IT-paola-medium … --format mp3`
- [x] Prosody measurement in Piper's environment (`prosody.measure` on a generated file: f0 162 Hz)

### Web page

- [x] `./voice-clone web --port 7871` serves the page; both sections ("Crea una voce", "Genera audio") present

### Long run

- [x] Alfeo: 500 epochs overnight in the background (`--init-from` voce-pos epoch 1816), sleep blocked during the
  run, export of the best checkpoint, then the PC suspended by the follow-up script as planned

**Notes (Linux):** Windows-specific code paths (stop request, `SetThreadExecutionState`, winget hints, `.pyd`,
`install.ps1`) were also exercised on Linux by simulation in the unit tests.

---

## Windows (to do)

Fill in the machine first:

| | |
|---|---|
| Windows version (`winver`) | |
| GPU and driver (`nvidia-smi`) | |
| CPU / RAM | |
| voice-clone version (release zip or commit) | |
| Path of the project folder (does it contain spaces?) | |

Open **cmd** or **PowerShell** in the project folder for every command.

### 1. Prerequisites

- [ ] NVIDIA driver installed: `nvidia-smi` shows the GPU and the driver version
- [ ] `winget install -e --id Git.Git` (or Git already present: `git --version`)
- [ ] `winget install -e --id Gyan.FFmpeg`, then in a **new** terminal `ffmpeg -version` works
- [ ] The rubberband filter exists: `ffmpeg -hide_banner -filters | findstr rubberband` prints a line
- [ ] Visual Studio Build Tools with C++:
  `winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"`

### 2. First start and diagnostics

- [ ] Unzip `voice-clone-<version>-windows.zip` (or `git clone`); `voice-clone.cmd --help` installs uv by itself
  (first time only), creates `.venv` and prints the help
- [ ] `voice-clone.cmd --version`
- [ ] Before installing the prerequisites (optional, on a clean machine): `voice-clone.cmd doctor` lists what is
  missing and prints `fix:` with `winget install …` lines (no `sudo`)
- [ ] `voice-clone.cmd doctor` with the prerequisites: the system line says "Windows …", the RAM is right,
  `[ok] NVIDIA GPU`, `[ok] system packages`, `[ok] uv`; speech.cpp and Piper not installed yet
- [ ] `voice-clone.cmd languages` prints accented names correctly (UTF-8)

### 3. Studio only (works before `setup`)

- [ ] `voice-clone.cmd models catalog --lang it` lists 4 Italian voices
- [ ] `voice-clone.cmd models download it_IT-paola-medium` (downloads 3 files, md5 checked)
- [ ] `voice-clone.cmd speak it_IT-paola-medium "Ciao, è una prova. Perché no?" -o prova` → `prova.mp3`; it plays
  and the accents are pronounced
- [ ] Every format: `--format wav`, `ogg`, `opus`, `flac`, `m4a` each produce a playable file
- [ ] `--pitch 3` and `--speed 0.8` change the voice as expected
- [ ] `[pausa 2s]` in the text adds a two-second pause
- [ ] `--per-line out` with a text file of 3 lines makes 3 files in `out\`
- [ ] `voice-clone.cmd lexicon add --lang it ACI "A C I"`, then `lexicon test --lang it "Socio ACI"` prints
  "Socio A C I"

### 4. Setup

- [ ] `voice-clone.cmd setup --skip-models`:
  - [ ] speech.cpp installed with `install.ps1`; "speech.cpp sees: Vulkan0 (NVIDIA …)"
  - [ ] Piper cloned and its environment created
  - [ ] PyTorch with CUDA installed (the check prints `torch …, CUDA …, GPU <your GPU>`)
  - [ ] `monotonic_align` and `espeakbridge` compiled with MSVC (no compiler errors)
  - Time taken: ____
- [ ] `voice-clone.cmd setup` downloads Drama 3 (20.8 GB) and Qwen3-ASR (0.8 GB). Time taken: ____
- [ ] `voice-clone.cmd doctor`: every line `[ok]`, "Everything is ready."
- [ ] `voice-clone.cmd setup` again: everything "already installed", nothing reinstalled
- [ ] A new terminal: `speech --help` works (the installer added it to `PATH`)

### 5. Pipeline

Use 10–60 s of a clean recording of one speaker.

- [ ] `voice-clone.cmd new test --lang it`
- [ ] `voice-clone.cmd add-audio test C:\path\to\recording.wav` (try a path with spaces, in quotes)
- [ ] `voice-clone.cmd reference test`: merged, cut, transcribed (the transcript is correct, accents included in
  `voices\test\reference\transcript.txt`), voice built
- [ ] `voice-clone.cmd try-voice test "Buongiorno, come posso aiutarti?"` → `voices\test\reference\try.wav`
  sounds like the recording
- [ ] `voice-clone.cmd dataset test -n 20 --background`:
  - [ ] prints the pid; **no console window** opens
  - [ ] `voice-clone.cmd log test -f` follows the progress (`[n/N] clip …`, then verification)
  - [ ] **closing the terminal** that started it does not stop the job (check `status` from a new one)
  - [ ] sleep is blocked: in an **administrator** terminal `powercfg /requests` lists `python.exe` under SYSTEM
  - [ ] `voice-clone.cmd status test` shows "running: voice-clone dataset test …"
- [ ] `voice-clone.cmd stop test` while the dataset runs: returns within a few seconds with "stopped"; the log ends
  with "interrupted; run the same command again to resume."; no `speech.exe` left running (Task Manager)
- [ ] The same `dataset` command again resumes and finishes; `status` shows the dataset exported
- [ ] `voice-clone.cmd train test --epochs 5 --background`:
  - [ ] the GPU works (`nvidia-smi` shows python.exe and memory in use)
  - [ ] `status test` shows the epoch going up
  - Seconds per epoch: ____ (batch size: ____)
- [ ] `voice-clone.cmd stop test` during training: stops within a few seconds; no `python.exe` of Piper left
  (Task Manager); `last.ckpt` exists in `voices\test\training\lightning_logs\version_*\checkpoints\`
- [ ] `voice-clone.cmd train test --epochs 5 --resume` resumes and finishes: `status` shows "epoch 5/5 (finished)"
- [ ] `voice-clone.cmd checkpoints test` lists `last` and the best epochs
- [ ] `voice-clone.cmd export test` exports the `.onnx` and speaks the test sentences into `voices\test\samples\`
- [ ] `voice-clone.cmd say test "Prova finale." -o finale.wav`
- [ ] `voice-clone.cmd speak test "Prova con lo studio." --format mp3 -o studio`
- [ ] `voice-clone.cmd compare test best last`, then `promote test last`
- [ ] Optional: `voice-clone.cmd run test2 --sentences 20 --epochs 5` on a second voice, from start to end

### 6. Web page

- [ ] `voice-clone.cmd web`, open http://127.0.0.1:7860 in the browser
- [ ] "Crea una voce": the voice `test` shows its steps as done (✓); the players in steps 2, 3 and 5 play
- [ ] Start a job from the page (for example "Esporta e prova"): the job panel shows the phase and progress, the
  log fills, "■ Ferma" appears
- [ ] "■ Ferma" stops a running job (start "Genera il dataset" with a few more sentences, then stop it)
- [ ] Closing the browser tab does not stop a running job
- [ ] "Genera audio" → Genera: a generation with `test`, one with `it_IT-paola-medium`; the result plays and
  downloads
- [ ] Modelli: the catalog lists voices; download one; import an `.onnx` + `.onnx.json`
- [ ] Pronuncia: add a rule, "Prova e ascolta"
- [ ] Cronologia: the generations are there; "Riusa testo e impostazioni" and "Elimina" work

### 7. Optional: CPU mode

- [ ] `voice-clone.cmd setup --device cpu` reinstalls PyTorch as the CPU build; `doctor` shows "Mode: CPU"
- [ ] `voice-clone.cmd train test --epochs 6 --resume` runs one more epoch on the CPU
- [ ] `voice-clone.cmd setup --device gpu` switches back

**Notes (Windows):**

```
(paste errors and observations here)
```
