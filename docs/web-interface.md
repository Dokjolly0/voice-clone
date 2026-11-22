# Web interface

```
voice-clone web [--host 127.0.0.1] [--port 7860] [--share]
```

A [Gradio](https://www.gradio.app/) page (`web.py`, `web_studio.py`), with its text in Italian. A switch at the top of
the sidebar chooses between two sections:

- **🛠️ Crea una voce**: a guided wizard over the five stages;
- **🔊 Genera audio**: the audio studio.

The sidebar also shows the mode ("Modalità: GPU (RTX 4090)" or "solo CPU").

## Crea una voce

### Layout

- **Sidebar**: the voices (a radio list), "＋ Nuova voce", and the five steps as buttons with a state icon:
  `✓` done, `▶` running, `○` to do, `🔒` locked (a step before it is missing). Locked steps cannot be opened.
- **Main area**: one step at a time. Selecting a voice opens the step that needs attention: the running one, else
  the first one not done (`pipeline.next_step`). Each step ends with "Avanti →".
- **Job panel** at the bottom: the running job with its phase, a progress bar (`done/total · %`), the time left,
  a "■ Ferma" button, and the log in an accordion. When no job runs it shows the last outcome: completed,
  interrupted (resume by starting the step again) or the error.

With no voices, a welcome page explains the five steps.

### The steps

| Step | What you see | What the buttons start |
|---|---|---|
| **1 · Audio** | Upload area (multiple files), the list of recordings with durations and the total ("Bastano 10–60 secondi in tutto"), "Rimuovi una registrazione" | Files are added at once (`workspace.add_audio`) |
| **2 · Riferimento** | Advanced option: maximum reference length (6–60 s, default 30). The reference player, the transcript (editable), "Prova la voce clonata" with a text box and a player | "Prepara il riferimento" → job `reference`; "Salva la trascrizione e ricrea la voce" writes `transcript.txt` and starts `reference`; "Ascolta" → job `try-voice` |
| **3 · Dataset** | Number of sentences (default 1000) with an estimate of speech minutes and work time for the mode; optional `.txt` of your sentences; advanced: max error per clip (0.05) and attempts (4); the clip counts; "🔊 Ascolta 4 clip a caso" | "Genera il dataset" → job `dataset --sentences N --max-cer X --max-attempts N [--extra …]` (the uploaded file is copied to `voices/<name>/domain-sentences.txt`) |
| **4 · Addestramento** | Epochs (default 2000), the time estimate, advanced: batch size (0 = automatic); the chart of `val_mel` per epoch; the best checkpoint so far | The button reads "Avvia l'addestramento", "Riprendi l'addestramento" or "Continua fino a N epoche"; it starts `train --epochs N --batch-size B`, with `--resume` when a `last.ckpt` exists |
| **5 · Esporta e prova** | Checkpoint to export (Ultimo / Migliore), the test sentences' players, a free text box ("Leggi"), the model files to download (`.onnx` + `.onnx.json`), "Confronta due checkpoint" (A and B dropdowns, two columns of players, "Usa A/B come modello") | "Esporta e prova" → job `export --checkpoint last\|best`; "Leggi" speaks at once (`export.say`, into `export/say.wav`); "Confronta" → job `compare A B`; "Usa A/B" → `export.promote` |

**Estimates.** The dataset estimate uses the language's speaking rate and a measured Drama 3 speed (1.7× real time
on an RTX 4090, 0.3× on a 16-core CPU) plus 30% for checks and retries. The training estimate uses the voice's own
last 20 epochs when they were measured on the same kind of device (from `training/run.json`), else a measured rate
(10.5 s per epoch per hour of audio on the 4090, 170 s on the CPU).

### How it stays in sync

- Every long action starts a **background job** (`jobs.start`), exactly as `voice-clone <command> --background`
  would. Closing the page does not stop it, and the CLI sees the same job (`status`, `log`, `stop`).
- A `gr.Timer` polls every 3 seconds (`poll`): job panel, log, stop button, step states, and the action buttons
  (disabled while a job runs).
- Panels with audio players and the chart reload **only when their files change**: `signatures()` builds a string
  of file names and modification times (inputs, reference, transcript, voice, try.wav, metadata.csv, the model,
  samples, comparison files; and separately metrics.csv and checkpoints). A changed signature triggers the refresh,
  so a player is never reset while you listen.
- Navigation is separate from loading: clicking a step shows it at once; the content loads afterwards and never
  moves the highlighted step.
- Polling never changes the highlighted step either (`step_buttons(..., current=None)`).

## Genera audio

Four tabs (`web_studio.py`). Synthesis runs in the page's process on the CPU, fast enough that no background job
is needed.

| Tab | Content |
|---|---|
| **Genera** | Model dropdown grouped by origin (★ your voices, ⬇ downloaded, 📁 imported) with "↻" to refresh, an info line (language, quality, sample rate, origin, speakers) and a speaker dropdown for multi-speaker models. The text box with stats (characters, lines, estimated duration) and "Inserisci pausa" (inserts `[pausa Xs]`). Settings: format, bitrate (hidden for WAV/FLAC), sample rate, speed, volume, pitch, expressiveness, sentence pause, paragraph pause, normalize, use the dictionary, one file per line. "Ripristina le impostazioni predefinite". "▶ Anteprima" (first sentence only, WAV, not saved) and "Genera" (with a progress bar), then a player and the download |
| **Modelli** | The library table with "Rimuovi"; Piper's catalog filtered by language and quality, with "Scarica"; "Importa" for an `.onnx` + `.onnx.json` pair with an optional name |
| **Pronuncia** | The language (defaults to the chosen model's), the rules table, add a rule (written, spoken, whole words only, exact case), remove a rule, and "Prova e ascolta" (shows how a sentence will be read and speaks it with the model chosen in Genera) |
| **Cronologia** | The last 100 generations (date, model, duration, format, text), a player and the download for the chosen one, "Riusa testo e impostazioni" (back to Genera with everything filled in) and "Elimina" |

The settings used for the last generation are saved in `studio/preferences.json` and restored when the page opens.
The sidebar of this section shows tips: blank line = new paragraph, `[pausa 1.5s]`, `[[ phonemes ]]`, and the
Pronuncia tab for acronyms.

## Network and files

- By default the page listens on `127.0.0.1` only. `--host 0.0.0.0` makes it reachable from the local network;
  there is no authentication, so use it only on a trusted network.
- `--share` creates a temporary public link through Gradio's relay.
- Gradio may serve files from `voices/` (`allowed_paths`); studio files are served from `studio/`, which is inside
  the project.
