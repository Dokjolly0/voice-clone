# voice-clone documentation

This folder explains how voice-clone works, module by module. The [project README](../README.md) is the quick
start; these pages are the reference.

## Reading order

| Page | What it covers |
|---|---|
| [Overview](overview.md) | What the project does, the five stages, the architecture and the external tools |
| [Installation](installation.md) | `setup` and `doctor`, GPU and CPU modes, what gets installed where, Linux and Windows |
| [Command line](cli.md) | Every command and option |
| [Web interface](web-interface.md) | The guided "Crea una voce" wizard and the "Genera audio" studio |

### The voice pipeline

| Stage | Page | Module |
|---|---|---|
| 1. Audio | [pipeline/01-audio.md](pipeline/01-audio.md) | `workspace.py`, `audio.py` |
| 2. Reference | [pipeline/02-reference.md](pipeline/02-reference.md) | `reference.py`, `speech.py` |
| 3a. Sentences | [pipeline/03-sentences.md](pipeline/03-sentences.md) | `sentences.py`, `languages.py` |
| 3b. Dataset | [pipeline/04-dataset.md](pipeline/04-dataset.md) | `dataset.py`, `prosody.py` |
| 4. Training | [pipeline/05-training.md](pipeline/05-training.md) | `train.py`, `piper_train.py` |
| 5. Export and test | [pipeline/06-export.md](pipeline/06-export.md) | `export.py` |

### The audio studio

| Page | Module |
|---|---|
| [Model library](studio/models.md) | `models.py` |
| [Synthesis engine](studio/synthesis.md) | `synthesis.py` |
| [Pronunciation dictionary and history](studio/lexicon-and-history.md) | `lexicon.py`, `history.py` |

### Internals

| Page | What it covers |
|---|---|
| [Module map](internals/modules.md) | Every module, what it owns and what it depends on |
| [Background jobs, progress and status](internals/jobs-and-status.md) | `jobs.py`, `pipeline.py`, `inhibit.py` |
| [Platforms: Linux and Windows](internals/platforms.md) | `host.py`, `system.py`, `installer.py`, launchers, UTF-8 |
| [Files and state](internals/files-and-state.md) | Every file the project writes, with the JSON formats |

### Working on the project

| Page | What it covers |
|---|---|
| [Development](development.md) | Tests, CI, releases, conventions, adding a language |
| [Troubleshooting](troubleshooting.md) | Common problems and their fixes |
| [Validation checklist](checklist/validation.md) | Manual end-to-end checks on Linux (done) and Windows (to do) |
