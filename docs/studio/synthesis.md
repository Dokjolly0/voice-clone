# Studio: the synthesis engine

Module: `synthesis.py`. Command: `speak`. Web: tab "Genera".

Text in, a finished audio file out. It runs with `piper-tts` 1.8.0 (onnxruntime) in the app's own environment, on
the CPU: no GPU, no PyTorch, no `setup` needed. On a desktop CPU it is about 50× faster than real time (226 s of
audio in 4.3 s, model loading included), so an hour of audio takes a little over a minute.

```
text ─► lexicon ─► parse (paragraphs, lines, [pausa]) ─► Piper per piece ─► int16 + silences ─► ffmpeg ─► file
```

## Text layout drives the pauses

`parse(text)` turns the text into items:

| In the text | Item | Becomes |
|---|---|---|
| A blank line | `("paragraph", None)` | `paragraph_pause` seconds of silence (default 0.8) |
| Any other line break | `("line", None)` | `sentence_pause` (default 0.3) |
| `[pausa 1.5s]`, `[pause 500ms]`, `[break 2s]`, `[pausa 1,5]` | `("pause", seconds)` | Exactly that much silence (at most 30 s; the unit defaults to seconds) |
| Text | `("speak", text)` | Piper speaks it; between the sentences Piper finds inside it, `sentence_pause` |
| `[[ fonemi ]]` | part of the text | Piper's raw phonemes, passed through untouched |

## Options

| Option | Range / values | Default | How it is applied |
|---|---|---|---|
| `speed` | 0.5–2.0 | 1.0 | Piper's `length_scale = model default / speed` |
| `volume_db` | −20..+20 | 0 | ffmpeg `volume=+X dB` |
| `pitch` | −12..+12 semitones | 0 | ffmpeg `rubberband=pitch=2^(st/12)`: duration unchanged; beyond ±4–5 it sounds processed |
| `expressiveness` | 0–1.5 | 1.0 | × the model's `noise_scale` (0.667) and `noise_w` (0.8): variation in intonation and rhythm, not emotions |
| `speaker` | name or index | first | For multi-speaker models (`speaker_id`); refused for single-speaker models |
| `sentence_pause`, `paragraph_pause` | 0–30 s | 0.3, 0.8 | Silence inserted between pieces |
| `normalize` | on/off | off | ffmpeg `loudnorm=I=-16:TP=-1.5:LRA=11` (−16 LUFS) |
| `format` | mp3, wav, ogg, opus, flac, m4a | mp3 | Codec below |
| `bitrate` | e.g. `128k` | per format | `-b:a` for lossy formats |
| `sample_rate` | 16000, 22050, 24000, 44100, 48000 | the model's | `-ar`; Opus only accepts 8/12/16/24/48 kHz, so the nearest one (preferring higher) is used |
| `use_lexicon` | on/off | on | Apply the dictionary of the model's language family first |

`Options.validate()` checks every range and raises `UserError` with the allowed values.

### Formats

| Format | Extension | Codec | Default quality |
|---|---|---|---|
| mp3 | `.mp3` | libmp3lame | 192k |
| wav | `.wav` | pcm_s16le | lossless |
| ogg | `.ogg` | libvorbis | `-q:a 6` (Vorbis refuses high bitrates for mono 22 kHz, so a quality scale is used) |
| opus | `.opus` | libopus | 64k |
| flac | `.flac` | flac | lossless |
| m4a | `.m4a` | aac | 128k |

## How a file is made

1. **`render`**: applies the dictionary, parses the text, loads the model (`load_voice`, cached: up to 4 models
   stay in memory, keyed by path and modification time), builds the `SynthesisConfig`, and for each item appends
   Piper's audio chunks or silence to one int16 array at the model's sample rate. A progress callback receives
   `(done, total)` spoken pieces.
2. **`encode`**: writes the array as a temporary WAV and runs **one** ffmpeg pass (`ffmpeg_args`) with the filters
   (pitch, volume, loudnorm), `-ac 1 -ar <rate> -c:a <codec>` and the bitrate. The result's duration is measured
   with ffprobe.
3. **`generate(model, text, out, options)`** returns `Result(path, seconds, sample_rate, format)`; the output's
   extension is set from the format.

Other entry points:

- `preview(model, text, out, options)`: only the first spoken piece, to try settings quickly (the web page's
  "▶ Anteprima", WAV, not saved in the history);
- `generate_per_line(model, text, out_dir, options)`: one file per non-empty line, named
  `001-first-words.mp3` (ASCII slug of the line, at most 40 characters), for example a kiosk's messages;
- `zip_files(files, zip_path)`: an uncompressed zip (audio is already compressed), used by the web page's "un file
  per ogni riga".

## Examples

```sh
voice-clone speak anna "Buongiorno! [pausa 1s] Come posso aiutarti?" --format mp3 -o saluto
voice-clone speak catalog:it_IT-paola-medium --file capitolo1.txt --format m4a --normalize -o capitolo1
voice-clone speak anna --file messaggi.txt --per-line messaggi/ --format wav --sample-rate 44100
voice-clone speak anna "Un po' più lento e più basso." --speed 0.85 --pitch -2
```

## Limits

- Pitch is post-processing: beyond ±4–5 semitones the voice sounds unnatural.
- Expressiveness is Piper's random variation, not emotions or styles.
- `x_low` and `low` models are 16 kHz; asking for 44.1 kHz resamples but cannot add detail.
- Numbers, dates and symbols are read by espeak-ng's rules for the language; when they come out wrong, write them
  as words or add a dictionary rule.
