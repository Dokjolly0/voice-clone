# Development

## Running the tests

```sh
uv run pytest            # all tests (about 100, a few seconds)
uv run pytest -k jobs -v # some of them
```

The tests never touch your voices, studio or cache: the fixtures in `tests/conftest.py` point `VOICE_CLONE_VOICES`,
`VOICE_CLONE_CACHE` and `VOICE_CLONE_STUDIO` at a temporary folder and reload the modules that read them.

| File | Covers |
|---|---|
| `test_workspace.py` | Creating, loading and listing voices; name validation; adding and refusing audio |
| `test_sentences.py` | Sentence filters (plain, ambiguous, uncased scripts, short), cleaning, append-only choice with domain sentences first, the share of short sentences |
| `test_dataset.py` | Normalization and CER, seeds per sentence/attempt/take, state sync, what to generate, redoing short sentences |
| `test_prosody.py` | The prosody profile and distance |
| `test_progress.py` | Job logs into phases, counts, time left, errors and interruptions; step states; ETA across a resume; checkpoint resolution |
| `test_misc.py` | Cut points, the language table, install commands per package manager, log tails, best checkpoint, pruning |
| `test_device.py` | GPU/CPU mode: settings, batch sizes, speech.cpp arguments |
| `test_studio.py` | Model library, catalog search, import validation, lexicon, history |
| `test_synthesis.py` | Parsing pauses and paragraphs, phonemes, option validation, ffmpeg arguments per format, speed and expressiveness mapping, speakers, lengths |
| `test_host.py` | Platform paths, winget and dnf commands, the speech.cpp installer, the PyTorch backend, `which`, UTF-8 output; Windows simulated on Linux |
| `test_jobs.py` | A real detached job: start, current, clean stop within seconds, the Windows stop request (simulated on Linux, real on the Windows runner), reused pids |

Tests that need the GPU, Drama 3 or a training run are not part of the suite; the
[validation checklist](checklist/validation.md) covers them by hand.

## CI and releases

`.github/workflows/ci.yml` runs on every push, pull request and by hand, on `ubuntu-latest` and `windows-latest`:

1. `astral-sh/setup-uv`; FFmpeg (`apt-get` / `choco`);
2. `uv lock --check` (the lock file matches `pyproject.toml`);
3. `uv run pytest -v`;
4. the launchers: `./voice-clone --help`, `doctor`, `languages` on Linux; `voice-clone.cmd` with the same on Windows;
5. a real synthesis: download `it_IT-paola-medium` from Piper's catalog (md5 checked), speak MP3, WAV at 0.8× speed
   and two OGG files with `--per-line`, check them with ffprobe.

Runners have no GPU and no room for Drama 3, so the dataset and training stages are not run in CI.

`.github/workflows/release.yml` runs when CI completes on `main` after a push, and only if it succeeded:

1. checks out the tested commit; reads the version from `pyproject.toml`;
2. `uv build` (wheel and sdist) and `git archive` into `voice-clone-<version>-windows.zip` and
   `voice-clone-<version>-linux.tar.gz` (sources and launchers; `.github` and `.gitattributes` are left out); a
   `SHA256SUMS` file;
3. `gh release create v<version>-build.<n>` with notes generated from the commits, using the workflow's own
   `GITHUB_TOKEN` (`permissions: contents: write`).

`<n>` is the Release workflow's run number, so it also counts skipped runs. Pushing changes to `.github/workflows/`
needs a token with the `workflow` scope (or SSH).

## Conventions

- Code, CLI messages, README, docs and commit messages in English; the web page's text in Italian.
- One commit per task, with a descriptive English message (`feat:`, `fix:`, `test:`, `docs:`, `ci:`).
- Run `uv run pytest` before committing.
- Lines up to 120 characters.
- User-facing errors are `UserError` with a message that says what to do.
- Platform-specific code goes through `host.py` (or checks `host.IS_WINDOWS` at call time), so tests can simulate
  the other platform.
- Text I/O is always UTF-8.

## Adding a language

A language needs all four: Drama 3 speaks it, Qwen3-ASR recognizes it, Common Voice has sentences for it, and
`rhasspy/piper-checkpoints` has a medium checkpoint in it. Then add a `Language` to `languages.py`:

```python
Language("sv", "Swedish", "sv-SE", "sv/sv_SE/nst/medium", samples=(
    "God morgon, hur kan jag hjälpa dig i dag?",
)),
```

- `common_voice`: the folder name under `server/data/` in the Common Voice repository;
- `checkpoint`: the folder in `rhasspy/piper-checkpoints` (its `config.json` gives the espeak voice and phoneme
  type);
- adjust `min_chars`, `max_chars`, `short_min_chars` and `chars_per_second` for scripts without spaces (see `zh`).

Then try it end to end with a short dataset (`dataset NAME -n 50 --limit 50`, `train --epochs 5`).

## Updating Piper

`PIPER_REV` in `paths.py` pins the training code. To move it: change the commit, run `setup` (it checks out the new
commit and reinstalls), train a few epochs and export, and check that `piper_train.py`'s imports
(`piper.train.__main__.VitsLightningCLI`, `piper.train.vits.dataset.VitsDataModule`,
`piper.train.vits.lightning.VitsModel`) still exist. The studio's `piper-tts` version in `pyproject.toml` should
match the training code's version, so exported models load the same way.
