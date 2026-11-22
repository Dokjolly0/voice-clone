# Studio: pronunciation dictionary and history

Modules: `lexicon.py`, `history.py`. Commands: `lexicon list|add|remove|test`. Web: tabs "Pronuncia" and
"Cronologia".

## Pronunciation dictionary

"Written → spoken" rules per language, applied to the text before synthesis. They fix what espeak-ng reads wrong:
acronyms, brand and product names, foreign words, place names.

```sh
voice-clone lexicon add --lang it ACI "A C I"
voice-clone lexicon add --lang it PinPad "pin pad"
voice-clone lexicon add --lang it Coop "[[ kˈɔp ]]"          # Piper phonemes
voice-clone lexicon add --lang it bancomat "bancomàt" --anywhere
voice-clone lexicon test --lang it "Socio ACI con PinPad Coop"
```

### Rules

Stored in `studio/lexicon/<lang>.json` (language codes of 2–3 lowercase letters):

```json
{"rules": [
  {"written": "ACI", "spoken": "A C I", "whole_word": true, "match_case": false},
  {"written": "Coop", "spoken": "[[ kˈɔp ]]", "whole_word": true, "match_case": true}
]}
```

| Field | Default | Meaning |
|---|---|---|
| `written` | | The text as written; adding a rule with the same `written` replaces it |
| `spoken` | | What to say instead: plain text, or Piper's raw phonemes in `[[ … ]]` |
| `whole_word` | true | Match whole words only (`--anywhere` to match inside words too) |
| `match_case` | false | Match this exact capitalization only (`--match-case`) |

### How they apply (`apply_rules`)

- Rules are applied **longest first**, so "Socio ACI" wins over "ACI".
- Each rule only looks at text that earlier (longer) rules left alone: a replacement is never replaced again.
- Text inside `[[ phonemes ]]` and pause markers (`[pausa …]`, `[pause …]`, `[break …]`) is never touched.
- Whole-word matching uses "not preceded or followed by a word character", so it works with accented letters and
  punctuation.

The synthesis engine applies the dictionary of the model's **language family** (`it` for `it_IT`) unless
`use_lexicon` is off (`--no-lexicon`). The web tab defaults to the language of the model chosen in "Genera", and
"Prova e ascolta" shows the rewritten text and speaks it with that model.

## History

Every generation made from the web page is saved in `studio/history/<id>/`, so it can be replayed, downloaded,
reused or deleted. (`speak` on the command line writes where `-o` says and is not recorded.)

- `<id>` is `YYYYMMDD-HHMMSS-<4 hex>`, so folders sort by time.
- The folder holds the audio file (`audio.<ext>`), or for "un file per ogni riga" the files in `files/` and a
  `<id>.zip`, plus `meta.json`:

```json
{"id": "20261122-011502-a3f9", "created": "2026-11-22 01:15:02", "model_id": "voice:alfeo",
 "model_label": "alfeo · it_IT · medium (tua voce)", "text": "Buongiorno!",
 "options": {"speed": 1.0, "volume_db": 0.0, "pitch": 0.0, "expressiveness": 1.0, "speaker": null,
             "sentence_pause": 0.3, "paragraph_pause": 0.8, "normalize": false, "format": "mp3",
             "bitrate": "192k", "sample_rate": null, "use_lexicon": true},
 "per_line": false, "files": ["audio.mp3"], "download": "audio.mp3", "seconds": 1.02}
```

| Function | What it does |
|---|---|
| `create(model, text, options, per_line, progress)` | Generates into a new entry; on any failure the folder is removed |
| `entries(limit)` | Newest first |
| `get(id)`, `file_path(meta)`, `first_audio(meta)` | An entry, its download (audio or zip), its first audio file |
| `delete(id)` | Removes an entry |
| `prune(keep=200)` | Keeps the newest 200 entries (run after every generation) |

"Riusa testo e impostazioni" in the web page puts an entry's model, text and every option back into "Genera".

## Preferences

`studio/preferences.json` keeps the settings of the last generation from the web page (format, bitrate, sample
rate, speed, volume, pitch, expressiveness, pauses, normalize, dictionary, one file per line) and restores them when
the page opens. "Ripristina le impostazioni predefinite" resets them.
