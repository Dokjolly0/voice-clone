# Stage 3b: generating and verifying the dataset

Modules: `dataset.py`, `prosody.py`. Command: `dataset NAME [options]` (it first runs `sentences` with the same
`-n`/`--extra`). Keeps the computer awake.

Drama 3 reads every sentence of `sentences.txt` in the cloned voice; Qwen3-ASR listens to each clip; clips that do
not say what they should are generated again. The result is a Piper dataset: `dataset/wav/*.wav` at 22050 Hz and
`dataset/metadata.csv`.

## Rounds

`dataset.build` repeats rounds until every sentence has a good clip or has used all its attempts:

```
sync_state ─► [redo_short] ─► round 1..max_attempts:
                                 generate (sentences without a clip, attempts left)
                                 verify   (clips not verified yet)
                                 stop when nothing is left to generate (or with --limit)
           ─► counts ─► export
```

1. **`sync_state`** matches `state.json` to `sentences.txt`: a new line gets an entry
   `{"text": …, "attempts": 0}`; an entry whose text changed starts over (its clip is deleted); entries beyond the
   end of the file are dropped.
2. **`generate`** starts one `SpeechServer` (Drama 3 loaded once) and, for each sentence to do, increments
   `attempts` and synthesizes with a fixed seed. A generation that stops for another reason than `complete`
   (`max_seconds`: longer than 30 s, a runaway; `model_limit`) or returns no audio is recorded as
   `rejected` and retried in the next round. Progress lines:
   `[12/1000] clip 00011 seed 1101: 0.8 min of audio, RTFx 1.62, ~41 min left`.
3. **`verify`** transcribes the clips in batches of 100 and computes the **character error rate** (CER) between
   the sentence and what was heard, after normalizing case, punctuation and spaces (Levenshtein distance / length).
   `CER ≤ max_cer` (default 0.05) → `verified`; otherwise the clip is deleted and marked `rejected: "CER 0.12"`.
4. **`export`** converts the verified clips to 22050 Hz mono WAV in `dataset/wav/<id>.wav` (only when the source is
   newer), deletes stale files, and writes `metadata.csv` as `id|text` lines. Under 30 minutes of audio a warning
   says the voice will be rougher.

Sentences that used every attempt (`max_attempts`, default 4) without a good clip **stay out** of the dataset; that
is normal for a few per hundred.

### Seeds

`seed_for(i, attempt, take=0) = 1 + i × 100 + attempt + 7919 × take`. A seed is fixed per sentence, attempt and take,
so a rerun reproduces the same audio and an interrupted run resumes exactly. The seed of the kept clip is stored in
`state.json`.

## Short sentences: the best of several takes

Drama 3 has no context for a one- or two-word sentence, so each take guesses an intonation: one much higher, the
next flat or much lower. For **short sentences** (shorter than the language's `min_chars`) each attempt generates
`short_best_of` takes (default 5) into `clips/takes/<id>-<seed>.wav`, and `choose_takes`:

1. transcribes every take and keeps those with `CER ≤ max_cer`;
2. measures their pitch and rate (`prosody.measure`) and computes their **distance** from the voice's typical
   prosody (`prosody.distance`);
3. keeps the closest take as `clips/<id>.wav`, recording `best_of`, `heard`, `cer`, `seed` and
   `prosody: {seconds, f0, f0_range, distance}`;
4. if even the closest take is farther than `max_distance` (default 3) and attempts are left, rejects all takes
   ("best take too far from the voice's prosody") so the next round tries new seeds; on the last attempt the best
   one is kept anyway.

### The voice's typical prosody (`voice_profile`)

Measured once on up to 60 verified **long** sentences (evenly spread), cached in `state.json` as
`prosody_profile: {f0, f0_range, rate, clips}`. Until at least 10 long sentences are verified there is no profile and
the first take that passes ASR is kept.

### Measurements (`prosody.py`)

`prosody.py` runs **as a script in Piper's environment**, where librosa is installed: `python prosody.py FILE...`
prints one JSON object per file. For each file:

- load at 22050 Hz, trim silence (`top_db=35`), `seconds` = trimmed duration;
- pYIN pitch tracking between 60 and 400 Hz; with at least 5 voiced frames: `f0` = median pitch,
  `f0_range` = 90th − 10th percentile.

`profile()` takes the medians of `f0`, `f0_range` and the rate (letters per second of the sentence's text over
`seconds`).

`distance(m, text, ref)` (lower is better):

```
|ln(f0 / ref.f0)| / 0.12                         pitch level (0.12 ≈ two semitones per point)
+ max(0, 0.5 − f0_range / ref.f0_range) × 4      a flat delivery (movement under half the usual)
+ |ln(rate / ref.rate)| / 0.35                   speaking rate (short sentences are naturally a bit slower)
```

A take with no measurable pitch scores 10.

### Redoing short sentences: `--redo-short`

For datasets made before this feature, or with a smaller `--short-best-of`, `redo_short` sends back to generation:

- verified short sentences made with fewer takes than `short_best_of` (their attempts restart at 0; take 0 of
  attempt 0 reproduces the original clip, so it competes too);
- verified short sentences whose kept take is farther than `max_distance` and still have attempts left (the
  attempt goes on, so the seeds are new).

## `state.json`

```json
{
 "clips": {
  "00000": {"text": "Il treno per Milano partirà alle otto.", "attempts": 1, "seed": 1,
            "heard": "Il treno per Milano partirà alle otto.", "cer": 0.0, "verified": true},
  "00957": {"text": "Grazie mille.", "attempts": 2, "seed": 103594, "best_of": 5, "verified": true,
            "heard": "Grazie mille.", "cer": 0.0,
            "prosody": {"seconds": 0.83, "f0": 158.2, "f0_range": 71.4, "distance": 0.92}},
  "00412": {"text": "…", "attempts": 4, "seed": 41204, "rejected": "CER 0.111"}
 },
 "prosody_profile": {"f0": 154.0, "f0_range": 102.0, "rate": 15.1, "clips": 60}
}
```

## Status and counts

`dataset.status` (shown by `status`): total sentences, `verified` (good clips), `failed` (all attempts used),
`pending` (the rest, including sentences never started) and `exported` (lines of `metadata.csv`).

## Tuning

| Symptom | What to change |
|---|---|
| Many clips rejected for spelling only (the recognizer wrote "di un" for "d'un") | Raise `--max-cer` a little (0.08) |
| Real skips or repetitions | Normal at a few per hundred; retries fix them |
| A quick end-to-end test | `--limit 20` (one round of at most 20 clips), then train a few epochs |
| Short sentences still odd | `--redo-short`, or a larger `--short-best-of` / smaller `--max-distance` |
| More data | `-n 1500`: existing clips are kept, only the new lines are generated |

Look at `state.json` (`text` vs `heard`) and listen to a few files in `dataset/wav/`: Piper learns exactly what it
hears there. The web page's "Ascolta 4 clip a caso" helps.
