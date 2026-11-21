import json
import shutil

import pytest

from voice_clone.util import UserError

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffprobe"), reason="ffmpeg not installed")


def test_create_load_and_list(voices_dir):
    from voice_clone import workspace

    voice = workspace.create("anna", "it")
    assert json.loads((voices_dir / "anna" / "voice.json").read_text())["lang"] == "it"
    assert workspace.load("anna").language.code == "it"
    assert workspace.list_names() == ["anna"]
    assert voice.model_name == "it_IT-anna-medium"


@pytest.mark.parametrize("name", ["Anna", "an na", "-anna", "a" * 41, ""])
def test_invalid_names_are_refused(voices_dir, name):
    from voice_clone import workspace

    with pytest.raises(UserError):
        workspace.create(name, "it")


def test_unknown_language_and_duplicates_are_refused(voices_dir):
    from voice_clone import workspace

    with pytest.raises(UserError):
        workspace.create("anna", "xx")
    workspace.create("anna", "it")
    with pytest.raises(UserError):
        workspace.create("anna", "it")
    with pytest.raises(UserError):
        workspace.load("nobody")


@needs_ffmpeg
def test_add_audio_copies_and_deduplicates_names(voices_dir, make_wav):
    from voice_clone import workspace

    voice = workspace.create("anna", "it")
    wav = make_wav("my clip.wav", seconds=0.5)
    workspace.add_audio(voice, [wav, wav])
    assert [p.name for p in voice.inputs()] == ["my_clip-2.wav", "my_clip.wav"]
    workspace.remove_audio(voice, ["my_clip-2.wav"])
    assert [p.name for p in voice.inputs()] == ["my_clip.wav"]


def test_add_audio_refuses_non_audio(voices_dir, tmp_path):
    from voice_clone import workspace

    voice = workspace.create("anna", "it")
    text = tmp_path / "notes.txt"
    text.write_text("hello")
    with pytest.raises(UserError):
        workspace.add_audio(voice, [text])
