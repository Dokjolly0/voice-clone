# Stage 3a: choosing the sentences

Module: `sentences.py`. Command: `sentences NAME [-n N] [--extra FILE]` (also run by `dataset` and `run`).

The dataset is only as good as its text. Piper learns exactly what it hears, so the sentences must be readable one
way only, varied, and include the kinds of phrases the voice will really say.

## Source: Common Voice

Sentences come from Mozilla Common Voice's sentence collection
(`github.com/common-voice/common-voice`, `server/data/<locale>/*.txt`), which is public domain (CC0).

`corpus_files(lang, wanted)`:

1. lists the language's folder through the GitHub API (once; cached as `index.json`), sorted by size, smallest
   first;
2. downloads files into `<cache>/common-voice/<locale>/` until there are roughly 20 raw lines per wanted sentence
   (`POOL_FACTOR`, estimating 60 bytes per line);
3. reuses the cache on later runs.

## Filters (`acceptable`)

After `clean()` (Unicode NFC, typographic quotes to plain, spaces collapsed), a sentence is kept only if:

| Rule | Why |
|---|---|
| Length within the language's bounds: normal `min_chars`..`max_chars` (30–150), short `short_min_chars`..`min_chars − 1` (12–29) | Very long sentences are hard to generate cleanly; very short ones are fragments |
| Quotes balanced (`"` even, `«` = `»`) | Unbalanced quotes are cut sentences |
| Does not end with an abbreviation (`…Vol.`, `…l'art.`) | Truncated text |
| Short sentences have at least two words (except Chinese) | One word is a title, not a sentence |
| No digits or symbols (`BAD_CHARS`: # $ % & * + / < = > @ \ ^ _ ~ §  ° € £ ¥, brackets, parentheses, braces, the vertical bar and the backtick) | Numbers and symbols can be read several ways |
| Ends with `. ! ?` (or the CJK forms) | Complete sentences |
| No all-caps words of 2+ letters | Acronyms: read as letters or as a word? |
| Starts with an uppercase letter (in cased scripts) | Fragments start lowercase |

## Short sentences

A model that never heard a short sentence rushes greetings and replies with an odd intonation. So 15% of the
sentences added (`short_share`) are short. Short ones come from the **curated** files first and from Wikipedia
(`wiki*` files) only if needed, because Wikipedia's short lines are mostly titles, fragments and foreign names
(`pool(lang, wanted, short=True)`).

Each pool is shuffled with a fixed seed (42), so the same command chooses the same sentences.

## Domain sentences

`--extra FILE` (one sentence per line) adds sentences that are always included, first: product names, the phrases
a kiosk or an assistant will really say, the greetings it needs. They skip the filters, so write numbers and
acronyms as they should be spoken ("ventiquattro ore", "A C I").

## `sentences.txt` only grows

`choose(voice, count, extra)` keeps every existing line in order and appends new ones; **line N is clip N**. Growing
the dataset later (`-n 1500`) therefore keeps every clip already made. Asking for fewer sentences than the file has
keeps them all (with a warning). Changing a line by hand makes that clip start over (the dataset stage notices the
text changed).

The command reports the total and an estimate of speech minutes (`characters / chars_per_second`).
