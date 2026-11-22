# Stage 5: export, test and compare

Module: `export.py`. Commands: `export`, `say`, `checkpoints`, `compare`, `promote`.

The result of the whole pipeline is two files in `export/`:

- `<locale>-<name>-medium.onnx`: the model (~60 MB);
- `<locale>-<name>-medium.onnx.json`: its configuration (phonemes, sample rate, inference defaults).

They must stay side by side with the same name. They run anywhere Piper voices run: the `piper-tts` package
(`python -m piper -m it_IT-anna-medium.onnx -f out.wav -- "Testo."`), Piper's C/C++ library, or onnxruntime with
espeak-ng. No GPU, no Drama 3, no training environment.

## Checkpoints

`export.checkpoints(voice)` lists what can be exported:

- `last`: the newest `last.ckpt`, with the epoch and val_mel from `progress.json`;
- `epoch-<n>`: every `epoch=<n>-val_mel=<x>.ckpt` kept (the best ones of every run), newest epoch first.

`resolve(voice, which)` accepts `last` (or the newest epoch if there is no `last.ckpt`), `best` (lowest val_mel),
an epoch number (`1816` or `epoch-1816`; the error lists the epochs kept) or a path to a `.ckpt`.

`voice-clone checkpoints NAME` prints them:

```
last         (epoch-499)  val_mel 0.3061
epoch-467                 val_mel 0.3018
epoch-447                 val_mel 0.2997
epoch-440                 val_mel 0.3021
```

## `export NAME [--checkpoint X] [--sample TEXT]...`

1. `_export_onnx`: `python -m piper.train.export_onnx --checkpoint <ckpt> --output-file <model>.onnx.part` in Piper's
   environment, then the `.part` file is renamed (an interrupted export never leaves a broken model). Exporting while
   training runs is fine: Lightning replaces checkpoint files atomically.
2. `export/source.json` records where it came from: `{"checkpoint": "epoch-447", "epoch": 447, "val_mel": 0.2997}`.
3. **Test sentences**: the language's `samples` (5 for Italian and English, fewer for others) plus every `--sample`
   are spoken with `python -m piper` into `samples/sample-<n>.wav`, listed in `samples/sentences.txt`
   (`sample-1.wav|Buongiorno, come posso aiutarti oggi?`). The folder is recreated each time.

## `say NAME TEXT [-o OUT]`

Speaks with the exported model through the studio's engine (`synthesis.generate` with the model `voice:<name>`), so
it needs no training environment. The output format follows the extension (`.mp3`, `.ogg`, …; WAV otherwise).

## Comparing checkpoints

The lowest val_mel is not always the best-sounding checkpoint. To choose by ear:

```
voice-clone compare NAME best 1500
voice-clone promote NAME epoch-1500
```

- `compare A B` resolves both (they must differ), exports each into `export/variants/<label>/` (the model, the
  test sentences in `samples/`, and `source.json` with the checkpoint's modification time), and writes
  `export/variants/comparison.json` (`{"a": "epoch-447", "b": "last"}`). A variant already exported from the same
  checkpoint file is not exported again.
- `promote LABEL` copies the variant's model over `export/<model>.onnx`, replaces `samples/` with the variant's,
  and writes `source.json`.

In the web page, step 5's "Confronta due checkpoint" does the same with two columns of players and "Usa A/B come
modello".

## Using the model in the studio

An exported voice appears in the studio automatically as `voice:<name>` (★ in the web page), read in place from
`export/`. See [Model library](../studio/models.md).
