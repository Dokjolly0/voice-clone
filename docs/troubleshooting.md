# Troubleshooting

Start with `voice-clone doctor`: it lists what is missing with the command to fix it. For a running or failed job,
`voice-clone log NAME` (or the log in the web page) shows the details; `voice-clone status NAME` shows where each
stage stands.

## Installation

| Problem | Cause and fix |
|---|---|
| `missing system tools: …` | Run the command `setup` printed (`sudo dnf …`, `sudo apt …`, or `winget install …`), open a new terminal on Windows, and run `setup` again |
| `GPU mode needs a working NVIDIA driver` | `nvidia-smi` is missing or fails. Install the driver (the message has the commands for your system) and reboot, or use `setup --device cpu` |
| `speech.cpp does not see the GPU` | Linux: the NVIDIA driver or the Vulkan loader (`vulkan-loader` / `libvulkan1`) is missing; install it, delete `~/.local/bin/speech` and run `setup` again (the installer picks the Vulkan build only when it finds Vulkan). Windows: update the NVIDIA driver |
| `speech.cpp's installer finished but the speech program was not found` | Windows: open a new terminal (the installer added it to `PATH` for new shells) and run `setup` again |
| `Piper was installed but does not work` | The check script's error follows. "PyTorch does not see a CUDA GPU": the driver is not working, or (Windows) the CPU build of PyTorch was installed: run `setup` again, it reinstalls with the CUDA build |
| Windows: `Piper's training code is compiled with Visual Studio's C++ compilers, which were not found` | Install the Build Tools with the printed winget command, open a new terminal, run `setup` again |
| Windows: Piper's build fails with a compiler error | Send the output of `voice-clone setup` (the end of the error): the MSVC build is the step most likely to need a fix |
| `could not download drama-3` | The download was interrupted 5 times; run `setup` again, it resumes |
| `uv could not be installed` | Install uv by hand (https://docs.astral.sh/uv/) and run the launcher again |

## Reference

| Problem | Cause and fix |
|---|---|
| `no input audio` | `add-audio NAME FILE...` first |
| `not a readable audio file` | ffprobe cannot read it: convert it, or check it is really audio |
| `the recognizer heard no words in reference.wav` | The recordings are silent or too quiet; check `reference/merged.wav` |
| The cloned voice sounds wrong in `try-voice` | Check `transcript.txt` against `reference.wav` word for word and fix it; a reference cut mid-word (hard cut) also hurts: change `--max-seconds` |

## Dataset

| Problem | Cause and fix |
|---|---|
| Many clips rejected | Look at `state.json`: `text` vs `heard`. Spelling-only differences ("di un" for "d'un"): raise `--max-cer` (0.08). Real skips or repetitions: a few per hundred is normal; retries fix them |
| `speech serve did not start in time` | See `logs/serve.log`. Often out of memory: close other programs (Drama 3 needs ~15 GB of RAM with a GPU, ~21 GB on the CPU) |
| `speech serve exited with code …` | See `logs/serve.log` |
| `speech asr failed` | See the message; rerun the command, it resumes |
| `no good clips` | The voice or the transcript is wrong: listen to `try-voice` and check the transcript |
| `Common Voice has only N more usable sentences` | The language has few sentences that pass the filters; add your own with `--extra` |
| Short sentences sound rushed or odd | `dataset NAME --redo-short` (best of 5 takes, retried when far from the voice) |

## Training

| Problem | Cause and fix |
|---|---|
| `CUDA out of memory` | `train NAME --resume --batch-size 16` (or 12, 8) |
| `training stopped with exit code …` | See `training/train.log`; continue with `train NAME --resume` |
| `nothing to resume` | There is no `last.ckpt` yet: drop `--resume` |
| `no dataset yet` | Run `dataset NAME` first |
| `checkpoint not found` (with `--init-from`) | The path is wrong; relative paths are relative to where you run the command |
| The disk fills up | Checkpoints are ~850 MB each; old ones are pruned at every start and end of training (5 best + the newest last) |
| val_mel higher than a previous run | Not comparable across datasets (different validation split); compare by ear |
| The PC went to sleep during training | Linux: check `systemd-inhibit --list` during the run; a closed laptop lid still suspends. Windows: `powercfg /requests` (as administrator) should list Python under SYSTEM; sleep you ask for still happens |

## Export and studio

| Problem | Cause and fix |
|---|---|
| `….onnx.json is missing; Piper writes it when training starts` | Train at least once (it is written at the start) |
| `no checkpoint of epoch N; kept epochs: …` | Only the best ones are kept; choose one from the list (`checkpoints NAME`) |
| `no model 'x'` | See `models list`; use the full id when several models share a name |
| `ffmpeg failed: … rubberband` | Your FFmpeg build has no rubberband filter: install one that has it (Fedora `ffmpeg-free` and Gyan's Windows builds do), or leave the pitch at 0 |
| A word is read wrong | Add a rule in the pronunciation dictionary (`lexicon add`, or the "Pronuncia" tab) |
| `<model> has a single speaker` | Leave the speaker empty for single-speaker models |

## Background jobs

| Problem | Cause and fix |
|---|---|
| `… is already running …` | One job per voice: wait, or `stop NAME` |
| `stop` takes a minute and kills the job | The stage did not react to the interrupt in 60 s (for example a long ASR batch): it was terminated; run the same command again to resume |
| The web page shows an old job | It reads `logs/job.json` and `job.log`: the last job's outcome is shown until a new one starts |
| Accented text looks garbled (Windows) | Use the launcher (`voice-clone.cmd`), which sets UTF-8; in other terminals set `PYTHONUTF8=1` |
