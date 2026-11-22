# Command line

```
voice-clone <command> [options]          (Linux: ./voice-clone, Windows: voice-clone.cmd)
voice-clone --help | <command> --help | --version
```

Errors the user can fix are printed as `error: …` with exit code 1 (`UserError`, no traceback). Ctrl-C prints
`interrupted; run the same command again to resume.` and exits with 130. Commands marked **bg** accept
`--background`: the same command is started again as a detached job ([details](internals/jobs-and-status.md)).

## Setup and information

| Command | What it does |
|---|---|
| `setup [--device auto\|gpu\|cpu] [--skip-models]` | Installs speech.cpp, Piper's training environment and the models; checks the system ([details](installation.md#voice-clone-setup)) |
| `setup --inference-only` | Only for speaking with ready models: checks FFmpeg and the speech engine, installs nothing for training ([details](installation.md#inference-only-setup)) |
| `doctor` | Reports what is installed, changes nothing |
| `languages` | Lists the supported languages with their base voice |

## Voices

| Command | What it does |
|---|---|
| `new NAME --lang CODE` | Creates `voices/NAME/` with `voice.json`. Names: lowercase letters, digits, `-` and `_`, at most 40, starting with a letter or digit |
| `list` | Lists the voices with their language and number of input files |
| `add-audio NAME FILE...` | Copies recordings into `input/` (any format ffmpeg reads: wav, mp3, flac, ogg, opus, m4a, aac, wma, webm, mp4, mkv); each is checked with ffprobe; names are sanitized and de-duplicated (`a.wav`, `a-2.wav`) |
| `remove-audio NAME FILE...` | Removes input files by file name |
| `status [NAME] [--json]` | Where each stage of one voice (or all) stands, and the running job |

## Stages

### `reference NAME` (bg)

Merges and trims the inputs, cuts the reference at a pause, transcribes it, builds the Drama 3 voice
([details](pipeline/02-reference.md)).

| Option | Default | Meaning |
|---|---|---|
| `--max-seconds S` | 30 | Longest reference Drama 3 hears, cut at a pause |
| `--retranscribe` | off | Discard `transcript.txt` and recognize it again |
| `--voice-file X.gguf` | – | Use an existing Drama 3 voice file instead of the inputs |
| `--from-inputs` | off | Drop an imported voice file and build from the inputs |

### `try-voice NAME TEXT [-o OUT] [--seed N]`

Speaks one sentence with Drama 3 and the voice, to check it before generating the dataset. Default output:
`reference/try.wav`, seed 1.

### `sentences NAME [-n N] [--extra FILE]`

Makes `sentences.txt` hold N sentences (default 1000), keeping the existing ones; `--extra` adds domain sentences
(one per line) first ([details](pipeline/03-sentences.md)).

### `dataset NAME` (bg)

Runs `sentences` with the same `-n`/`--extra`, then generates, verifies and exports the dataset
([details](pipeline/04-dataset.md)). Keeps the computer awake.

| Option | Default | Meaning |
|---|---|---|
| `-n`, `--sentences N` | 1000 | Sentences in the dataset |
| `--extra FILE` | – | Domain sentences, always included first |
| `--max-attempts N` | 4 | Generations per sentence before giving up |
| `--max-cer X` | 0.05 | Largest character error rate a clip may have |
| `--limit N` | 0 (all) | Generate at most N clips (one round), for quick tests |
| `--short-best-of N` | 5 | Takes per short sentence (1 = off) |
| `--max-distance X` | 3.0 | A short sentence whose best take is farther than this from the voice's prosody is tried again |
| `--redo-short` | off | Regenerate short sentences made with a single take, or whose take is far from the voice |

### `train NAME` (bg)

Fine-tunes Piper on the dataset ([details](pipeline/05-training.md)). Keeps the computer awake.

| Option | Default | Meaning |
|---|---|---|
| `--epochs N` | 2000 | Total epochs (the run stops there; with `--resume` it continues to N) |
| `--batch-size N` | 0 (auto) | 0: from the VRAM (32 with 20+ GB, 24, 16, 12, 8) or, in CPU mode, the RAM |
| `--num-workers N` | 4 | Data loader processes |
| `--resume` | off | Continue from the newest `last.ckpt` |
| `--init-from CKPT` | – | Start from this checkpoint instead of the language's base voice |

### `export NAME` (bg)

Exports a checkpoint to ONNX and speaks the test sentences into `samples/` ([details](pipeline/06-export.md)).

| Option | Default | Meaning |
|---|---|---|
| `--checkpoint X` | last | `last`, `best` (lowest val_mel), an epoch number (`1816` or `epoch-1816`) or a `.ckpt` path |
| `--sample TEXT` | – | Extra test sentence, repeatable |

### Other export commands

| Command | What it does |
|---|---|
| `say NAME TEXT [-o OUT]` | Speaks text with the exported model (default `say.wav`; the format follows the extension) |
| `checkpoints NAME` | Lists the checkpoints kept: `last` (with its epoch) and the best ones with their val_mel |
| `compare NAME A B` (bg) | Exports two checkpoints into `export/variants/<label>/` with the same test sentences |
| `promote NAME LABEL` | Makes a compared checkpoint (e.g. `epoch-1816`) the voice's model |

### `run NAME` (bg)

Every stage in order (reference, sentences + dataset, training, export), skipping the finished ones and resuming
an interrupted training. Accepts the options of `reference` (`--max-seconds`), `dataset`, `train` (without
`--resume`/`--init-from`) and `export`.

## Background jobs

| Command | What it does |
|---|---|
| `status NAME` | Shows the running job (command, pid, start time) |
| `log NAME [-f] [--all]` | The job log (the last 8 KB, or 200 KB with `--all`); `-f` follows until the job ends |
| `stop NAME` | Stops the job cleanly (like Ctrl-C), then terminates it if it does not exit in time |

## Audio studio

### `speak MODEL [TEXT] [options]`

Generates an audio file from text with any model in the library ([details](studio/synthesis.md)). MODEL is a model
id (`voice:anna`, `catalog:it_IT-paola-medium`, `imported:pos`), the part after the colon, a plain name when it is
unique, or a path to a `.onnx` with its `.onnx.json` next to it.

| Option | Default | Meaning |
|---|---|---|
| `TEXT` or `--file FILE` | – | The text (exactly one of the two) |
| `-o`, `--output OUT` | `speech.<ext>` | Output file; the extension follows the format |
| `--per-line FOLDER` | – | One file per non-empty line into FOLDER (`001-first-words.mp3`, …) |
| `--format F` | mp3 | mp3, wav, ogg, opus, flac, m4a |
| `--bitrate B` | per format | For mp3/ogg/opus/m4a, e.g. `128k` (defaults 192k, quality 6, 64k, 128k) |
| `--sample-rate R` | the model's | 16000, 22050, 24000, 44100, 48000 |
| `--speed X` | 1.0 | 0.5–2.0 |
| `--volume DB` | 0 | −20..+20 dB |
| `--pitch ST` | 0 | −12..+12 semitones; beyond ±4 it sounds processed |
| `--expressiveness X` | 1.0 | 0–1.5 × the model's natural variation |
| `--speaker S` | first | Name or number, for multi-speaker models |
| `--sentence-pause S` | 0.3 | Seconds between sentences and lines |
| `--paragraph-pause S` | 0.8 | Seconds between paragraphs |
| `--normalize` | off | Loudness normalization to −16 LUFS |
| `--no-lexicon` | – | Ignore the pronunciation dictionary |

Progress (`n/total`) and the result (`path (2.0 s, mp3, 22050 Hz)`) go to stderr.

### `models <action>`

| Action | What it does |
|---|---|
| `models list` | Every model the studio can use: id, language, quality, origin, speakers |
| `models import FILE.onnx [FILE.onnx.json] [--name N]` | Copies a Piper model into the library (`imported:<slug>`) |
| `models catalog [--lang it\|it_IT] [--quality x_low\|low\|medium\|high]` | Browses Piper's official voices |
| `models download KEY...` | Downloads catalog voices, e.g. `it_IT-paola-medium` (md5 verified) |
| `models remove ID...` | Removes downloaded or imported models (trained voices are not removed here) |

### `lexicon <action> --lang CODE`

| Action | What it does |
|---|---|
| `lexicon list --lang it` | Shows the rules |
| `lexicon add --lang it WRITTEN SPOKEN [--anywhere] [--match-case]` | Adds or replaces a rule; SPOKEN may be Piper phonemes `[[ … ]]` |
| `lexicon remove --lang it WRITTEN` | Removes a rule |
| `lexicon test --lang it TEXT` | Shows how a text will be read |

## Web interface

`web [--host H] [--port P] [--share]`: serves the web interface (default `http://127.0.0.1:7860`). `--host 0.0.0.0`
opens it to the local network; `--share` also publishes a temporary public `gradio.live` link
([details](web-interface.md)).

## Typical sessions

```sh
voice-clone new anna --lang it
voice-clone add-audio anna rec1.wav rec2.m4a
voice-clone run anna --background          # everything, detached
voice-clone log anna -f                    # follow it
voice-clone status anna

voice-clone compare anna best 1500         # two checkpoints, same sentences
voice-clone promote anna epoch-1500

voice-clone speak anna "Buongiorno!" --format mp3 -o saluto
voice-clone speak anna --file messaggi.txt --per-line out/ --format wav
```
