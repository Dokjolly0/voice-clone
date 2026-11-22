# Module map

All code is in `src/voice_clone/`. Modules import lazily inside functions where a dependency is heavy (Gradio,
piper, numpy) or would create an import cycle, so `voice-clone --help` and simple commands start fast.

## Layers

```
 entry points     cli.py   __main__.py   web.py ── web_studio.py
                     │                      │
 orchestration    pipeline.py   jobs.py   inhibit.py
                     │
 stages           workspace.py → reference.py → sentences.py → dataset.py → train.py → export.py
                                     │                              │           │
 tools            audio.py   speech.py   prosody.py   piper_train.py (runs in Piper's environment)
 studio           models.py → synthesis.py → history.py      lexicon.py
 machine          host.py   system.py   installer.py   settings.py   paths.py   util.py   languages.py
```

## Modules

| Module | Owns | Main functions | Depends on |
|---|---|---|---|
| `__init__.py` | `__version__` | | |
| `__main__.py` | `python -m voice_clone` | calls `cli.main()` | cli |
| `cli.py` | Argument parsing, the commands, UTF-8 console setup, interrupt handling, `--background` | `main`, `build_parser`, `cmd_*`, `interruptible`, `BACKGROUND_COMMANDS` | everything, lazily |
| `paths.py` | Where things live: `PROJECT_ROOT`, `VOICES_DIR`, `STUDIO_DIR`, `CACHE_DIR`, `PIPER_DIR`, `PIPER_PYTHON`, `PIPER_REPO`, `PIPER_REV`, `LOCAL_BIN` | | host |
| `host.py` | Linux/Windows differences | `IS_WINDOWS`, `venv_python`, `venv_script`, `ext_suffixes`, `local_bin`, `speech_dir`, `extra_bin_dirs`, `cache_dir` | |
| `util.py` | Shared helpers | `UserError`, `info`, `step` (`==> …` headers), `warn`, `which`, `run`, `read_json`, `write_json` (atomic), `replace` (retried on Windows), `download` (`.part` + md5) | host |
| `settings.py` | `settings.json` (the mode) | `device`, `is_cpu`, `set_device` | paths, system |
| `system.py` | The machine: OS, packages, GPU, RAM | `check`, `query_gpu`, `ram_gib`, `find_msvc`, `detect_package_manager`, `SystemReport.install_command`, `REQUIREMENTS`, `DRIVER_HELP` | host, util, psutil |
| `installer.py` | `setup` and `doctor` | `setup`, `doctor`, `install_speech`, `install_piper`, `pull_models`, `piper_ready`, `speech_bin`, `device_args`, `speech_install_command`, `torch_backend` | host, settings, system, paths |
| `languages.py` | Supported languages | `Language`, `LANGUAGES`, `get` | |
| `workspace.py` | A voice's folder and metadata | `Voice`, `create`, `load`, `list_names`, `add_audio`, `remove_audio` | audio, languages, paths |
| `audio.py` | ffmpeg/ffprobe helpers | `duration`, `convert`, `concatenate`, `silences`, `cut_point`, `cut`, `TRIM_SILENCE` | |
| `speech.py` | speech.cpp: ASR, voice files, `speech serve` | `transcribe`, `make_voice`, `SpeechServer` | installer |
| `reference.py` | Stage 2 | `build`, `merge`, `transcribe`, `build_voice`, `import_voice`, `try_voice`, `status` | audio, speech |
| `sentences.py` | Stage 3a | `choose`, `pool`, `acceptable`, `corpus_files`, `estimate_minutes` | languages, paths |
| `prosody.py` | Pitch and rate measurement (also a script for Piper's environment) | `measure`, `profile`, `distance` | installer, paths |
| `dataset.py` | Stage 3b | `build`, `generate`, `verify`, `choose_takes`, `voice_profile`, `redo_short`, `export`, `status`, `cer`, `seed_for` | audio, prosody, reference, speech |
| `train.py` | Stage 4 | `run`, `command`, `base_checkpoint`, `prune_checkpoints`, `progress`, `metrics`, `eta_seconds`, `batch_for_vram`, `batch_for_cpu` | host, system, settings, installer, paths |
| `piper_train.py` | Piper's entry point with this project's callbacks; runs in Piper's environment and imports nothing from voice_clone | `EpochMetrics`, `callbacks`, `main` | torch, lightning, piper |
| `export.py` | Stage 5 | `export`, `export_and_test`, `say`, `checkpoints`, `resolve`, `compare`, `promote`, `read_samples` | train, installer, models, synthesis |
| `pipeline.py` | `status` and `run` over all stages; step states for the web page | `status`, `format_status`, `run`, `steps`, `next_step` | every stage, jobs |
| `jobs.py` | Background jobs and their progress | `start`, `stop`, `current`, `log_tail`, `progress`, `background`, `watch_stop_request` | host, util, psutil |
| `inhibit.py` | Keeping the computer awake | `keep_awake` | host, util |
| `models.py` | The studio's model library and Piper's catalog | `Model`, `list_models`, `get`, `import_files`, `remove`, `catalog`, `search`, `download_catalog` | workspace, paths, util |
| `synthesis.py` | The studio's engine | `Options`, `parse`, `render`, `generate`, `preview`, `generate_per_line`, `zip_files`, `ffmpeg_args`, `output_rate` | models, lexicon, audio, piper, numpy |
| `lexicon.py` | Pronunciation dictionaries | `rules`, `add`, `remove`, `apply`, `apply_rules` | paths, util |
| `history.py` | The studio's generation history | `create`, `entries`, `get`, `delete`, `prune` | synthesis, models |
| `web.py` | The web page and the "Crea una voce" wizard | `build`, `serve`, `poll`, `show`, view functions per step | gradio, pandas, stages, jobs, web_studio |
| `web_studio.py` | The "Genera audio" section | `build`, `generate`, `preview`, tab handlers | gradio, models, synthesis, lexicon, history |

## Conventions in the code

- **Errors**: anything the user can fix raises `UserError` with a message that says what to do; `cli.main` prints it
  as `error: …`. The web page turns it into `gr.Error`.
- **Output**: `util.step("…")` prints `==> …` headers. `jobs.progress` reads these headers from the job log to know
  the phase, so their wording matters (see `jobs.PHASES`).
- **State files** are written atomically (`write_json`: temporary file, then rename).
- **Downloads** go through a `.part` file, so an interrupted download never looks complete; catalog files are
  checked against their md5.
- **Subprocess text** is decoded as UTF-8 with `errors="replace"`, and text files are read and written as UTF-8,
  on every platform.
- **Idempotence**: every stage checks what is already done and skips it, so running a command again resumes it.
