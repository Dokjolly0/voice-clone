# Background jobs, progress and status

Modules: `jobs.py`, `pipeline.py`, `inhibit.py`.

## Background jobs

Long stages can run **detached** from the terminal and the browser:

```sh
voice-clone dataset anna --background       # also: reference, train, export, compare, run
voice-clone log anna -f                     # follow
voice-clone status anna                     # stages + "running: voice-clone dataset anna (pid …)"
voice-clone stop anna                       # clean stop; run the same command again to resume
```

`--background` makes `cli.main` call `jobs.background(voice, argv)`, which starts the same command again **without**
`--background` as a job and prints how to follow it. The web page starts every long action the same way
(`jobs.start`), so the CLI and the page always see the same job. There is **one job per voice** at a time: starting
a second one fails with the command already running.

### Starting (`jobs.start`)

1. Refuses if a job is running for the voice.
2. Deletes a leftover `logs/stop.request`; rotates `logs/job.log` to `logs/job.previous.log`.
3. Starts `python -m voice_clone <args>` with stdin from the null device, stdout and stderr into `job.log`, and the
   environment plus `PYTHONUNBUFFERED=1`, `PYTHONUTF8=1` and `VOICE_CLONE_JOB_LOGS=<logs folder>`:
   - Linux: `start_new_session=True` (its own session and process group, so closing the terminal does not stop it);
   - Windows: `CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW` (its own group, no console window).
4. Writes `logs/job.json`:

```json
{"pid": 306012, "created": 1795340312.08, "args": ["dataset", "anna", "--limit", "6"],
 "started": "2026-11-22T10:38:32", "log": "/projects/voice-clone/voices/anna/logs/job.log"}
```

### Is it running? (`jobs.current`)

The job runs while its pid exists, is not a zombie, and has the **same start time** as recorded (`created`, from
psutil). The start time protects against a pid reused by an unrelated process after a reboot.

### Stopping (`jobs.stop`)

Stopping is meant to be like Ctrl-C, so the stage cleans up (closes `speech serve`, saves state) and can be resumed.

| | Linux | Windows |
|---|---|---|
| Clean stop | `SIGINT` to the job's process group: voice-clone and its children (Piper's training, `speech serve`) all get Ctrl-C | `stop` writes `logs/stop.request`. The job watches for it (a thread started by `cli.interruptible()` in jobs that have `VOICE_CLONE_JOB_LOGS`) and raises `SIGINT` **in itself**, which Python turns into `KeyboardInterrupt` and which also wakes the main thread from sleeps and waits |
| Children | Got the same SIGINT | Stopped by the code that started them as the interruption unwinds: `SpeechServer.__exit__` terminates `speech serve`; training terminates Piper's process (the previous `last.ckpt` stays for `--resume`) |
| After the timeout (60 s; the CLI and the page use the default) | `SIGTERM` (15 s), then `SIGKILL` (5 s) to the group | Terminate (15 s), then kill (5 s) the job and every child process found with psutil |

Why the file on Windows: Windows cannot send Ctrl-C to a process on another console, and a job has no console
window at all; a console Ctrl-Break also would not wake Python's main thread from a sleep. A `SIGBREAK` handler is
still installed, so Ctrl-Break in a console works like Ctrl-C.

The job's log ends with `interrupted; run the same command again to resume.`, which the progress parser reports as
"interrupted".

### Logs (`jobs.log_tail`)

The end of `job.log` with terminal control sequences removed and progress-bar redraws (carriage returns) collapsed
to their last state. `log NAME` shows 8 KB (200 KB with `--all`); `log NAME -f` follows the file until the job ends.

## Progress (`jobs.progress`)

The web page's job panel and the CLI read the same structure:

```json
{"running": true, "command": ["train", "anna", "--epochs", "500"], "phase": "training", "stage": 4,
 "done": 120, "total": 500, "eta": 3560, "error": null, "finished": false}
```

It is derived from the log, so it works for any job, including one started from another terminal:

- **phase**: the last `==> …` header, matched against `jobs.PHASES` (Merging → merging, Transcribing the reference →
  transcribing, Building the Drama 3 voice → building_voice, Choosing sentences, Generating, Verifying, Exporting
  the dataset, Downloading the base voice, Training, Exporting, Speaking). The stage (1–5) comes from the phase or
  the command.
- **generating**: total from `Generating N clips`, done and minutes left from the last `[n/N] … ~M min left` line;
  before the first line the phase is `loading_model`.
- **verifying**: done/total from `verified n/N`.
- **training**: epoch and total from `training/progress.json` (only if updated after the job started; before that
  the phase shows as preparing), time left from `train.eta_seconds`.
- **speaking**: total from `Speaking N test sentences`, done = sample lines printed so far.
- When the job is not running: `error` is the last `error: …` line, or `interrupted`; otherwise `finished`.

## Status (`pipeline.status`, `format_status`)

`voice-clone status [NAME] [--json]` collects every stage's status:

```
alfeo (Italian)
  [x] 1. audio      8 files, 32.5 s
  [x] 2. reference  imported voice pos.voice.gguf
  [x] 3. dataset    1132 sentences: 1103 good clips, 0 to do, 29 failed; 1103 exported
  [x] 4. training   epoch 500/500, best val_mel 0.2997 at epoch 447 (finished)
  [x] 5. export     2026-11-22T04:14
```

| Stage | Done when |
|---|---|
| 1. audio | There is at least one input file |
| 2. reference | `reference/voice.gguf` exists |
| 3. dataset | `dataset/metadata.csv` has at least one line |
| 4. training | The current run reached its end (`Trainer.fit` stopped) |
| 5. export | The `.onnx` exists (the date is its modification time) |

`--json` prints the full structure (inputs, reference, dataset counts, training progress, export, job).

For the web page, `pipeline.steps` gives each stage a state: `running` (the job's stage), `done`, `todo` (what it
needs exists) or `locked`. A stage is ready when the previous one produced what it needs: inputs → reference, voice
file → dataset, exported dataset → training, a `last.ckpt` → export. `next_step` picks the running one, else the
first not done.

## `run`: every stage in order (`pipeline.run`)

1. `reference.build` (skips what is up to date);
2. `sentences.choose` and `dataset.build` (resume where they were);
3. training: skipped if the last run finished with the same number of epochs; otherwise it runs, with `--resume`
   when a `last.ckpt` exists (an interrupted run, or a finished run asked for more epochs);
4. `export.export_and_test` with `--checkpoint`.

## Keeping the computer awake (`inhibit.keep_awake`)

`dataset`, `train` and `run` call it at the start:

- **Linux**: the process re-executes itself under
  `systemd-inhibit --what=idle:sleep --who=voice-clone --why="voice-clone: training anna"`, which blocks automatic
  (idle) and requested suspend until it exits. `VOICE_CLONE_INHIBITED=1` marks the re-executed process so it
  happens once. Without `systemd-inhibit` nothing happens. Check it with `systemd-inhibit --list`. Closing a laptop
  lid can still suspend it.
- **Windows**: `SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)`, in the process itself, until it exits.
  Sleep the user asks for (Start menu, lid, power button) still happens. Check it with `powercfg /requests` (in an
  administrator terminal): the Python process appears under SYSTEM.
- `VOICE_CLONE_NO_INHIBIT=1` disables both.
