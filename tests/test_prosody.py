import pytest

from voice_clone import dataset, prosody

REF = {"f0": 160.0, "f0_range": 110.0, "rate": 15.0}


def take(f0, f0_range, seconds):
    return {"f0": f0, "f0_range": f0_range, "seconds": seconds}


def test_distance_prefers_typical_pitch_movement_and_rate():
    typical = prosody.distance(take(158, 105, 0.66), "Buongiorno!", REF)   # 10 letters in 0.66 s
    high = prosody.distance(take(210, 105, 0.66), "Buongiorno!", REF)
    flat = prosody.distance(take(158, 10, 0.66), "Buongiorno!", REF)
    rushed = prosody.distance(take(158, 105, 0.30), "Buongiorno!", REF)
    assert typical < min(high, flat, rushed)
    assert prosody.distance({"f0": None, "seconds": 0.5}, "Ciao!", REF) == 10.0


def test_profile_takes_medians():
    measured = [(take(150, 100, 2.0), "a" * 30), (take(160, 110, 2.0), "a" * 30), (take(170, 140, 2.0), "a" * 32)]
    p = prosody.profile(measured)
    assert (p["f0"], p["f0_range"], p["rate"], p["clips"]) == (160, 110, 15.0, 3)


def test_take_seeds_are_distinct_and_take_zero_is_the_plain_seed():
    assert dataset.seed_for(12, 0, 0) == dataset.seed_for(12, 0)
    assert len({dataset.seed_for(12, a, t) for a in range(4) for t in range(5)}) == 20


def test_redo_short_resets_only_single_take_short_sentences(voices_dir):
    from voice_clone import workspace

    voice = workspace.create("v", "it")
    voice.clips_dir.mkdir(parents=True)
    state = {"clips": {
        "00000": {"text": "Buongiorno!", "attempts": 1, "verified": True},
        "00001": {"text": "Ciao!", "attempts": 1, "verified": True, "best_of": 5},
        "00002": {"text": "Una frase abbastanza lunga da non essere breve.", "attempts": 1, "verified": True},
    }}
    for cid in state["clips"]:
        (voice.clips_dir / f"{cid}.wav").write_bytes(b"x")
    assert dataset.redo_short(voice, state, dataset.Settings(short_best_of=5)) == 1
    assert not state["clips"]["00000"].get("verified") and state["clips"]["00000"]["attempts"] == 0
    assert not (voice.clips_dir / "00000.wav").exists()
    assert state["clips"]["00001"]["verified"] and state["clips"]["00002"]["verified"]
