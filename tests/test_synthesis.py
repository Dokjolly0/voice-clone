import shutil
import types

import numpy as np
import pytest

from voice_clone import synthesis
from voice_clone.synthesis import Options
from voice_clone.util import UserError

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg not installed")


def test_parse_paragraphs_lines_and_pauses():
    text = "Prima riga. [pausa 1,5s] Ancora.\nSeconda riga [pause 250ms]\n\n\nNuovo paragrafo [break 99s]"
    assert synthesis.parse(text) == [
        ("speak", "Prima riga."), ("pause", 1.5), ("speak", "Ancora."), ("line", None),
        ("speak", "Seconda riga"), ("pause", 0.25), ("paragraph", None),
        ("speak", "Nuovo paragrafo"), ("pause", synthesis.MAX_PAUSE),
    ]
    assert synthesis.units(text) == 4


def test_phonemes_pass_through():
    assert synthesis.parse("Il [[ kˈɔp ]] apre.") == [("speak", "Il [[ kˈɔp ]] apre.")]


@pytest.mark.parametrize("field, value", [("speed", 2.5), ("pitch", -13), ("volume_db", 25), ("expressiveness", 2),
                                          ("sentence_pause", -1), ("format", "aiff"), ("sample_rate", 11025)])
def test_options_are_validated(field, value):
    with pytest.raises(UserError):
        Options(**{field: value}).validate()


@pytest.mark.parametrize("fmt, codec, quality", [("mp3", "libmp3lame", ["-b:a", "192k"]), ("wav", "pcm_s16le", []),
                                                 ("ogg", "libvorbis", ["-q:a", "6"]), ("flac", "flac", []),
                                                 ("opus", "libopus", ["-b:a", "64k"]), ("m4a", "aac", ["-b:a", "128k"])])
def test_ffmpeg_args_per_format(tmp_path, fmt, codec, quality):
    args = synthesis.ffmpeg_args(tmp_path / "in.wav", tmp_path / f"out.{fmt}", Options(format=fmt), 22050)
    assert args[args.index("-c:a") + 1] == codec
    tail = args[args.index("-c:a") + 2:-1]
    assert tail == quality
    rate = args[args.index("-ar") + 1]
    assert rate == ("24000" if fmt == "opus" else "22050")  # Opus only takes 8/12/16/24/48 kHz


def test_ffmpeg_filters_for_pitch_volume_and_loudness(tmp_path):
    options = Options(pitch=12, volume_db=-6, normalize=True, bitrate="320k", sample_rate=48000)
    args = synthesis.ffmpeg_args(tmp_path / "in.wav", tmp_path / "out.mp3", options, 22050)
    assert args[args.index("-af") + 1] == "rubberband=pitch=2.000000,volume=-6.0dB,loudnorm=I=-16:TP=-1.5:LRA=11"
    assert args[args.index("-b:a") + 1] == "320k" and args[args.index("-ar") + 1] == "48000"


def fake_model(speakers=None):
    return types.SimpleNamespace(name="anna", speakers=speakers or [], sample_rate=1000, language_family="it",
                                 inference={"noise_scale": 0.6, "length_scale": 1.2, "noise_w": 0.8})


def test_synthesis_config_maps_speed_and_expressiveness():
    config = synthesis.synthesis_config(fake_model(), Options(speed=2.0, expressiveness=0.5))
    assert config.length_scale == pytest.approx(0.6)
    assert config.noise_scale == pytest.approx(0.3) and config.noise_w_scale == pytest.approx(0.4)


def test_speakers():
    assert synthesis.speaker_id(fake_model(), None) is None
    with pytest.raises(UserError):
        synthesis.speaker_id(fake_model(), "3")
    multi = fake_model(["alice", "bob"])
    assert synthesis.speaker_id(multi, None) == 0
    assert synthesis.speaker_id(multi, "bob") == 1 and synthesis.speaker_id(multi, "1") == 1
    with pytest.raises(UserError):
        synthesis.speaker_id(multi, "carol")


@needs_ffmpeg
def test_generated_length_counts_speech_and_every_pause(tmp_path, monkeypatch, studio):
    # A fake voice: every sentence (split on ".") is 0.5 s of sound at 1000 Hz.
    class FakeVoice:
        def synthesize(self, text, config):
            for _ in [s for s in text.split(".") if s.strip()]:
                yield types.SimpleNamespace(audio_int16_array=np.full(500, 1000, dtype=np.int16))

    monkeypatch.setattr(synthesis, "load_voice", lambda model: FakeVoice())
    text = "Uno. Due.\nTre. [pausa 2s] Quattro.\n\nCinque."
    options = Options(format="wav", sentence_pause=0.25, paragraph_pause=1.0)
    result = synthesis.generate(fake_model(), text, tmp_path / "out", options)
    # 5 sentences x 0.5 s + "Uno."→"Due." 0.25 + line 0.25 + 2 s + paragraph 1 s
    assert result.path.suffix == ".wav"
    assert result.seconds == pytest.approx(2.5 + 0.25 + 0.25 + 2 + 1, abs=0.01)


def test_slug():
    assert synthesis._slug("Seleziona l’erogatore, è facile!") == "seleziona-lerogatore-e-facile"
    assert synthesis._slug("!!!") == "line"
