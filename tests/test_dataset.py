import pytest

from voice_clone import dataset


def test_normalize_ignores_case_punctuation_and_spaces():
    assert dataset.normalize("  Ciao,   MONDO!  ") == "ciao mondo"
    assert dataset.normalize("l’altro") == "l'altro"


def test_cer_identical_after_normalization_is_zero():
    assert dataset.cer("Socio ACI rilevato.", "socio aci rilevato") == 0.0


def test_cer_counts_edits_over_reference_length():
    assert dataset.cer("abcd", "abed") == pytest.approx(0.25)
    assert dataset.cer("abcd", "") == 1.0
    assert dataset.cer("", "") == 0.0


def test_seeds_differ_per_sentence_and_attempt():
    seeds = {dataset.seed_for(i, a) for i in range(50) for a in range(4)}
    assert len(seeds) == 200


def test_sync_state_resets_clips_whose_sentence_changed(voices_dir):
    from voice_clone import workspace

    voice = workspace.create("v", "it")
    voice.sentences_file.write_text("Prima frase.\nSeconda frase.\n", encoding="utf-8")
    state = dataset.sync_state(voice)
    state["clips"]["00000"].update(verified=True, attempts=1)
    voice.clips_dir.mkdir()
    (voice.clips_dir / "00000.wav").write_bytes(b"x")
    voice.save_state(state)

    voice.sentences_file.write_text("Frase cambiata.\nSeconda frase.\n", encoding="utf-8")
    state = dataset.sync_state(voice)
    assert state["clips"]["00000"] == {"text": "Frase cambiata.", "attempts": 0}
    assert not (voice.clips_dir / "00000.wav").exists()


def test_to_generate_skips_verified_existing_and_exhausted(voices_dir):
    from voice_clone import workspace

    voice = workspace.create("v", "it")
    voice.clips_dir.mkdir(parents=True)
    (voice.clips_dir / "00001.wav").write_bytes(b"x")
    state = {"clips": {
        "00000": {"text": "a", "attempts": 1, "verified": True},
        "00001": {"text": "b", "attempts": 1},
        "00002": {"text": "c", "attempts": 4},
        "00003": {"text": "d", "attempts": 2},
    }}
    settings = dataset.Settings(max_attempts=4)
    assert dataset.to_generate(voice, state, settings) == ["00003"]
    assert dataset.to_verify(voice, state) == ["00001"]
    assert dataset.counts(state, settings) == {"total": 4, "verified": 1, "failed": 1, "pending": 2}
