# Studio: the model library

Module: `models.py`. Commands: `models list|import|catalog|download|remove`. Web: tab "Modelli".

Every Piper voice the studio can speak with, from three origins, in one list:

| Origin | Id | Where it lives | Removable |
|---|---|---|---|
| `trained` | `voice:<name>` | Read in place from `voices/<name>/export/*.onnx` (never copied) | No (it belongs to the voice) |
| `catalog` | `catalog:<key>` | `studio/models/<key>/` | Yes |
| `imported` | `imported:<slug>` | `studio/models/<slug>/` | Yes |

In the web page the icons are ★ (trained), ⬇ (catalog) and 📁 (imported).

## `Model`

| Field | Meaning |
|---|---|
| `id`, `name`, `origin` | See above |
| `language` | Locale (`it_IT`), from the config's `language.code`, the library metadata, the file name, or the espeak voice |
| `quality` | `x_low`, `low`, `medium`, `high` (config, metadata or file name) |
| `onnx`, `config_path` | The two files |
| `sample_rate` | From the config (`audio.sample_rate`): 16000 for `x_low`/`low`, 22050 for `medium`/`high` |
| `speakers` | Speaker names in id order, for multi-speaker models (else empty) |
| `inference` | The model's defaults: `noise_scale`, `noise_w`, `length_scale` |
| `language_family` | `it` from `it_IT`: the pronunciation dictionary used |

File names of the form `<ll_CC>-<name>-<quality>.onnx` give the language, name and quality when the config does not.

## Finding a model (`get`)

`get(ref)` accepts, in this order:

1. a path to a `.onnx` file with its `.onnx.json` next to it (validated, not copied; id `file:<path>`);
2. an id (`catalog:it_IT-paola-medium`) or the part after the colon (`it_IT-paola-medium`, `anna`);
3. a plain name, when exactly one model has it (otherwise the error lists the ids).

`list_models()` returns trained voices first, then catalog models, then imported ones, each group by name.

## Importing

`models import FILE.onnx [FILE.onnx.json] [--name N]` (web: upload both files, optional name):

1. the config defaults to `FILE.onnx.json` next to the model;
2. `validate_config` requires `audio.sample_rate`, `num_speakers`, `phoneme_id_map`, and an espeak voice when the
   phoneme type is espeak;
3. the id is a slug of the name or file name; an existing id is refused;
4. both files are copied into `studio/models/<slug>/` with a `model.json`:
   `{"origin": "imported", "onnx": "x.onnx", "name": null, "added": "2026-11-22T01:10:00"}`.

The web upload renames the config to `<model>.onnx.json` first, because uploads keep their original names only by
chance.

## Piper's catalog

`catalog()` downloads `voices.json` from `huggingface.co/rhasspy/piper-voices` into `<cache>/piper-voices.json`,
refreshed once a day and used stale when offline. It lists 183 voices in 53 languages.

`models catalog [--lang it|it_IT] [--quality medium]` (`search`) shows key, language, quality, speakers, size and
whether it is installed:

```
it_IT-paola-medium                   Italian      medium     1 spk   63 MB  installed
it_IT-riccardo-x_low                 Italian      x_low      1 spk   28 MB
it_IT-serena-high                    Italian      high       1 spk  114 MB
it_IT-serena-medium                  Italian      medium     1 spk   64 MB
4 voices
```

`models download KEY` (`download_catalog`) downloads every file of the entry (`.onnx`, `.onnx.json`, `MODEL_CARD`)
into `studio/models/<key>/`, each through a `.part` file and **verified against the catalog's md5**; files already
present with the right size are skipped. It writes `model.json` with the name, language and quality.

Catalog voices have their own licenses, in the `MODEL_CARD` downloaded next to the model.

## Removing

`models remove ID` deletes the model's folder in `studio/models/`. Trained voices are refused: they live in
`voices/<name>/`.
