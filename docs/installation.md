# Installation

Everything is installed by two commands, `setup` and the launcher itself. Nothing needs root (Linux) or
administrator rights (Windows), except the system packages, for which `doctor` and `setup` print the exact command.

## The launcher

| System | Launcher | What it does |
|---|---|---|
| Linux | `./voice-clone` (bash) | Adds `~/.local/bin` to `PATH`, sets `PYTHONUTF8=1`, installs `uv` if missing (`curl … astral.sh/uv/install.sh`), runs `uv sync --project <root> --quiet`, then `exec .venv/bin/voice-clone "$@"` |
| Windows | `voice-clone.cmd` → `voice-clone.ps1` | The `.cmd` runs the PowerShell script with `-ExecutionPolicy Bypass` (so no policy change is needed); the script does the same as the bash launcher with `irm https://astral.sh/uv/install.ps1 \| iex` and `.venv\Scripts\voice-clone.exe`, and returns its exit code |

`uv sync` creates `.venv` with Python 3.12 (downloaded by uv if the system has none) and the exact versions in
`uv.lock`. It runs on every start and takes a fraction of a second when nothing changed.

## System requirements

| | Linux | Windows |
|---|---|---|
| OS | Fedora (`dnf`) or Ubuntu/Debian (`apt`); tested on Fedora 45 | Windows 10/11, 64-bit |
| GPU mode | NVIDIA GPU with the proprietary driver, Vulkan loader | NVIDIA GPU with the standard driver (brings Vulkan and `nvidia-smi`) |
| VRAM | 8 GB at the very least, 16–24 GB recommended | same |
| RAM | 16 GB (GPU mode), 30+ GB (CPU mode, Drama 3 holds ~21 GB) | same |
| Disk | ~30 GB free (Drama 3 is 20.8 GB, Piper's environment several GB, checkpoints ~850 MB each) | same |

### System packages `doctor` checks

The table lives in `system.py` (`REQUIREMENTS`). A requirement applies to a system when it has a package name for
that system's package manager.

| Requirement | dnf | apt | winget | Why |
|---|---|---|---|---|
| `git` | git | git | Git.Git | Cloning Piper's training code |
| `curl` | curl | curl | – | speech.cpp's and uv's installers on Linux |
| `gcc`, `g++`, `make` | gcc, gcc-c++, make | build-essential | – | Compiling Piper's parts on Linux |
| `cmake`, `ninja` | cmake, ninja-build | cmake, ninja-build | – (pip provides them) | Building espeak-ng for Piper |
| `ffmpeg`, `ffprobe` | ffmpeg-free | ffmpeg | Gyan.FFmpeg | All audio work; the studio's pitch uses the `rubberband` filter |
| `lib:vulkan` (GPU mode only) | vulkan-loader | libvulkan1 | NVIDIA driver | speech.cpp's Vulkan build |
| `msvc` | – | – | Microsoft.VisualStudio.2022.BuildTools with the VCTools workload | Compiling Piper's parts on Windows |

On Linux the missing packages become one command (`sudo dnf install -y …` or
`sudo apt update && sudo apt install -y …`). On Windows each becomes a `winget install -e --id …` line; the Build
Tools line includes `--override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"`.
After installing on Windows, open a new terminal so `PATH` includes the new tools.

MSVC is found with `vswhere.exe` (`%ProgramFiles(x86)%\Microsoft Visual Studio\Installer`), asking for an
installation that has `Microsoft.VisualStudio.Component.VC.Tools.x86.x64`. The Vulkan loader is found with
`ctypes.util.find_library("vulkan")` on Linux and `("vulkan-1")` on Windows.

## `voice-clone setup`

```
voice-clone setup [--device auto|gpu|cpu] [--skip-models]
```

Steps, in order (`installer.setup`):

1. **Choose the mode** (`choose_device`): `auto` picks `gpu` when `nvidia-smi` reports a GPU, else `cpu`. Asking for
   `gpu` without a working driver fails with the driver install instructions for the system.
2. **Check the system** (`check_system`): OS name, RAM, mode, missing system packages (fails with the install
   command if any), and a warning when the RAM is below the mode's minimum (16 GB GPU, 30 GB CPU).
3. **Save the mode** in `settings.json`.
4. **Install speech.cpp** (`install_speech`), if `speech` is not found:
   - Linux: `curl -fsSL https://raw.githubusercontent.com/nyosegawa/speech.cpp/main/install.sh | sh`. It installs
     into `~/.local/share/speech.cpp/<version>/` and links `~/.local/bin/speech`; it picks the Vulkan build when the
     Vulkan loader is present, else the CPU build.
   - Windows: `powershell -NoProfile -ExecutionPolicy Bypass -Command "irm …/install.ps1 | iex"`. It installs the
     Vulkan build into `%LOCALAPPDATA%\Programs\speech.cpp\<version>\`, points `…\speech.cpp\current` at it and
     adds that folder to the user `PATH`. voice-clone also looks in `current` directly, because the new `PATH` only
     applies to new shells.
   - In GPU mode, `speech devices --json` must list a GPU, otherwise setup stops with a hint (CPU build installed?).
5. **Install Piper's training environment** (`install_piper`), skipped when it is already installed and working:
   - `git clone` of `https://github.com/OHF-voice/piper1-gpl.git` into `third_party/piper1-gpl`, checkout of the
     pinned commit (`PIPER_REV` in `paths.py`);
   - `uv venv --seed --python 3.12 third_party/piper1-gpl/.venv`;
   - `uv pip install -e third_party/piper1-gpl[train] scikit-build "cmake>=3.26,<4" ninja cython`, with the PyTorch
     build chosen by `torch_backend(device)`:

     | | GPU mode | CPU mode |
     |---|---|---|
     | Linux | PyPI's default build (comes with CUDA) | `--torch-backend cpu` (no ~3 GB of CUDA libraries) |
     | Windows | `--torch-backend auto` (uv picks the CUDA build matching the driver; PyPI only has CPU builds for Windows) | `--torch-backend cpu` |

     When the mode changed since the last install, `--reinstall-package torch` switches PyTorch in place.
   - Two compiled parts Piper's editable install does not build: `cythonize -i core.pyx` for
     `monotonic_align` (the resulting `core*.so` / `core*.pyd` is moved into `monotonic_align/monotonic_align/`),
     and `python setup.py build_ext --inplace` for `espeakbridge` (espeak-ng built with CMake). On Windows this
     needs MSVC; without it setup stops with the winget command for the Build Tools.
   - A check script imports torch, `piper.espeakbridge` and `monotonic_align`, and in GPU mode asserts
     `torch.cuda.is_available()`. It prints e.g. `torch 2.x, CUDA 13.x, GPU NVIDIA GeForce RTX 4090`.
   - `third_party/.piper-ready` records `<commit> <device>`; `piper_ready()` checks it and reruns the check script.
6. **Download the models** (`pull_models`), unless `--skip-models`: `speech pull drama-3` and
   `speech pull qwen3-asr-0.6b`, each retried up to 5 times (speech.cpp resumes interrupted downloads).

Running `setup` again is cheap: every part already in place is skipped.

## `voice-clone doctor`

Reports without changing anything:

```
System: Fedora Linux 45 (KDE Plasma Desktop Edition), 62 GB RAM
Mode: GPU
  [ok] NVIDIA GPU: NVIDIA GeForce RTX 4090, driver 620.70.06
  [ok] system packages
  [ok] uv
  [ok] speech.cpp
  [ok] speech.cpp sees the GPU
  [ok] Piper training environment (GPU)
Everything is ready.
```

When packages are missing it prints `fix: <command>`. "Mode: … (detected; not set up yet)" means `settings.json`
does not exist yet and the mode shown is the detected one.

## GPU and CPU modes

The mode is stored in `settings.json` (`{"device": "gpu"}`); `VOICE_CLONE_DEVICE=cpu` or `=gpu` overrides it for one
command. Without either, the mode is detected (`gpu` if `nvidia-smi` works). In CPU mode:

- speech.cpp gets `--device cpu` on every call (`installer.device_args()`), so it never touches a GPU it may find;
- Piper's environment has the CPU-only PyTorch, training runs with `--trainer.accelerator cpu` and
  `CUDA_VISIBLE_DEVICES=""`, and the batch size follows the RAM (32 with 24 GB or more, else 16);
- everything is much slower; see the table in the README's "CPU-only mode".

The studio ignores the mode: synthesis always runs on the CPU with onnxruntime.

## Windows specifics

See [Platforms](internals/platforms.md) for the details. In short: use `voice-clone.cmd`; install the NVIDIA driver,
Git, FFmpeg and the C++ Build Tools (the last only for training); the rest comes with `setup`. The cache is
`%LOCALAPPDATA%\voice-clone\cache`. WSL2 with Ubuntu is an alternative where the Linux instructions apply as they
are.

## Moving data

| Variable | Default | Moves |
|---|---|---|
| `VOICE_CLONE_VOICES` | `<project>/voices` | The voices |
| `VOICE_CLONE_STUDIO` | `<project>/studio` | Studio models, dictionaries, history |
| `VOICE_CLONE_CACHE` | `~/.cache/voice-clone` or `%LOCALAPPDATA%\voice-clone\cache` | Shared downloads |
| `VOICE_CLONE_DEVICE` | from `settings.json` | The mode, for one command |
| `VOICE_CLONE_NO_INHIBIT` | unset | Set to anything to let the computer sleep during long stages |
