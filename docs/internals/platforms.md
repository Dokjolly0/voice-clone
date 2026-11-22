# Platforms: Linux and Windows

voice-clone runs natively on Linux (Fedora, Ubuntu/Debian) and Windows 10/11, training included. Platform
differences are kept in a few places:

| Concern | Where | Linux | Windows |
|---|---|---|---|
| Virtual environment layout | `host.venv_python`, `host.venv_script` | `.venv/bin/python`, `.venv/bin/cythonize` | `.venv\Scripts\python.exe`, `.venv\Scripts\cythonize.exe` |
| Compiled extensions | `host.ext_suffixes` | `.so` | `.pyd` |
| Where tools install themselves | `host.extra_bin_dirs` (searched by `util.which` besides `PATH`) | `~/.local/bin` (uv, speech) | `~\.local\bin` (uv), `%LOCALAPPDATA%\Programs\speech.cpp\current` (speech.exe) |
| Cache | `host.cache_dir` | `~/.cache/voice-clone` | `%LOCALAPPDATA%\voice-clone\cache` |
| Launcher | repository root | `voice-clone` (bash) | `voice-clone.cmd` → `voice-clone.ps1` |
| Package manager and hints | `system.py` | dnf / apt, one `sudo` command | winget, one command per package |
| C/C++ compilers | `system.REQUIREMENTS` | gcc, g++, make, cmake, ninja | Visual Studio Build Tools (VCTools), found with vswhere; cmake and ninja from pip |
| speech.cpp | `installer.speech_install_command` | `install.sh` (Vulkan or CPU build) | `install.ps1` (Vulkan build) |
| PyTorch with CUDA | `installer.torch_backend` | PyPI's default build | `--torch-backend auto` (CUDA build matching the driver) |
| Background job process | `jobs._popen_options` | new session (`start_new_session`) | new process group, no window |
| Stopping a job | `jobs.stop` | SIGINT → SIGTERM → SIGKILL to the group | `stop.request` → in-process SIGINT → terminate/kill the tree |
| Keeping awake | `inhibit.keep_awake` | `systemd-inhibit` re-exec | `SetThreadExecutionState` |
| Replacing a file another process reads | `util.replace`, `piper_train.py` | always works | retried briefly (Windows refuses while the file is open) |

`host.IS_WINDOWS` is read at call time, so tests can simulate Windows on Linux by monkeypatching it
(`tests/test_host.py`, `tests/test_jobs.py`).

## UTF-8 everywhere

Windows defaults to a legacy code page (cp1252 in Western Europe) for pipes, files and the console, which mangles
accented sentences and transcripts. So:

- the launchers set `PYTHONUTF8=1`; `cli.main` also sets it in `os.environ` (inherited by jobs and Piper's
  processes) and reconfigures stdout/stderr to UTF-8 when they are not;
- every `subprocess.run(..., text=True)` passes `encoding="utf-8", errors="replace"`;
- every text file is read and written with `encoding="utf-8"`.

## The launchers

`voice-clone` (bash):

```bash
export PATH="$HOME/.local/bin:$PATH"
export PYTHONUTF8=1
command -v uv || curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --project "$ROOT" --quiet
exec "$ROOT/.venv/bin/voice-clone" "$@"
```

`voice-clone.cmd`:

```bat
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0voice-clone.ps1" %*
exit /b %ERRORLEVEL%
```

`voice-clone.ps1` does what the bash launcher does (uv from `https://astral.sh/uv/install.ps1`, then
`.venv\Scripts\voice-clone.exe @args`) and exits with its exit code. `-ExecutionPolicy Bypass` applies to that one
process, so the machine's policy is not changed.

`.gitattributes` keeps LF endings for `voice-clone` and `*.sh` and CRLF for `*.ps1` and `*.cmd`, in checkouts and in
the release archives (`git archive` applies it).

## Windows: what to install

| | Command | Needed for |
|---|---|---|
| NVIDIA driver | https://www.nvidia.com/drivers or the NVIDIA app | GPU mode (it brings Vulkan and `nvidia-smi`) |
| Git | `winget install -e --id Git.Git` | `setup` (Piper's training code) |
| FFmpeg | `winget install -e --id Gyan.FFmpeg` | Everything that reads or writes audio; the studio's pitch needs its `rubberband` filter (check with `ffmpeg -hide_banner -filters \| findstr rubberband`) |
| C++ Build Tools | `winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"` | `setup`: Piper's training code has compiled parts |

Open a new terminal afterwards. The studio alone (`speak`, "Genera audio") needs only FFmpeg: no `setup`, no GPU, no
Build Tools.

## Windows: known risks

- **Compiling Piper with MSVC** (espeak-ng with CMake, `monotonic_align` with Cython) is the step most likely to
  need fixes; Piper's own CI builds Windows wheels, and its CMakeLists has Windows and MSVC handling.
- **DataLoader workers** use spawn on Windows: starting them is slower and uses more RAM. Piper keeps them
  persistent, so the cost is paid once per run; `--num-workers 2` reduces RAM if needed.
- **Stopping training** loses the epoch in progress (Piper is terminated, not interrupted); the previous
  `last.ckpt` is kept.
- **Paths with spaces** (e.g. a user folder named "Mario Rossi") are passed as separate arguments everywhere, never
  through a shell, except the speech.cpp installer's one-line PowerShell command, which has no paths.

## WSL2

WSL2 with Ubuntu is an alternative on Windows: the Linux instructions apply as they are (apt packages, NVIDIA's
CUDA works in WSL2). speech.cpp's Vulkan build inside WSL2 depends on the driver's Vulkan support there; if it does
not see the GPU, run the dataset stage with `VOICE_CLONE_DEVICE=cpu` (speech.cpp on the CPU) and train in GPU
mode as usual.
