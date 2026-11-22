import json

import pytest

from voice_clone.util import UserError


def test_lexicon_rules(studio):
    from voice_clone import lexicon

    lexicon.add("it", "ACI", "A C I")
    lexicon.add("it", "Socio ACI", "socio a-ci-ì")
    lexicon.add("it", "Coop", "[[ kˈɔp ]]")
    lexicon.add("it", "pp", "P", whole_word=False, match_case=True)
    text = "Socio ACI? aci, coop, cooperativa, aPPle, happy. [[ ACI ]] [pausa 1s ACI]"
    assert lexicon.apply(text, "it") == ("socio a-ci-ì? A C I, [[ kˈɔp ]], cooperativa, aPPle, haPy. "
                                         "[[ ACI ]] [pausa 1s ACI]")
    lexicon.add("it", "ACI", "aci")  # replaces
    assert [r["spoken"] for r in lexicon.rules("it") if r["written"] == "ACI"] == ["aci"]
    lexicon.remove("it", "ACI")
    with pytest.raises(UserError):
        lexicon.remove("it", "ACI")
    assert lexicon.apply("ACI", "en") == "ACI"  # no rules for that language
    with pytest.raises(UserError):
        lexicon.rules("../x")


def test_import_list_get_and_remove(studio, fake_model_files):
    from voice_clone import models

    onnx = fake_model_files("it_IT-anna-medium")
    m = models.import_files(onnx, name="Anna Rossi")
    assert m.id == "imported:anna-rossi" and m.language == "it_IT" and m.quality == "medium"
    assert models.get("anna-rossi").id == m.id
    assert [x.id for x in models.list_models()] == [m.id]
    with pytest.raises(UserError):
        models.import_files(onnx, name="Anna Rossi")  # same name twice
    models.remove(m.id)
    assert models.list_models() == []


def test_invalid_configs_are_refused(studio, fake_model_files, tmp_path):
    from voice_clone import models

    onnx = fake_model_files("broken")
    onnx.with_name("broken.onnx.json").write_text(json.dumps({"audio": {}}))
    with pytest.raises(UserError):
        models.import_files(onnx)
    lonely = tmp_path / "lonely.onnx"
    lonely.write_bytes(b"x")
    with pytest.raises(UserError):
        models.import_files(lonely)


def test_multi_speaker_names_in_id_order(studio, fake_model_files):
    from voice_clone import models

    m = models.import_files(fake_model_files("en_US-many-high", speakers=["zed", "amy", "bob"]))
    assert m.speakers == ["zed", "amy", "bob"] and m.language_family == "en"


def test_catalog_search(studio, monkeypatch):
    from voice_clone import models

    entry = lambda code, q, n: {"name": n, "quality": q, "num_speakers": 1,
                                "language": {"code": code, "family": code[:2], "name_english": code},
                                "files": {f"x/{n}.onnx": {"size_bytes": 5_000_000, "md5_digest": ""}}}
    monkeypatch.setattr(models, "catalog", lambda refresh=False: {
        "it_IT-paola-medium": entry("it_IT", "medium", "paola"), "it_IT-riccardo-x_low": entry("it_IT", "x_low", "r"),
        "en_US-amy-medium": entry("en_US", "medium", "amy")})
    assert [r["key"] for r in models.search("it")] == ["it_IT-paola-medium", "it_IT-riccardo-x_low"]
    assert [r["key"] for r in models.search("it_IT", "medium")] == ["it_IT-paola-medium"]
    assert models.search("en")[0]["size_mb"] == 5
    with pytest.raises(UserError):
        models.download_catalog("xx_XX-nobody-low")


def test_history_records_and_prunes(studio, monkeypatch):
    from voice_clone import history, synthesis

    def fake_generate(model, text, out, options, progress=None):
        path = out.with_suffix(".mp3")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"mp3")
        return synthesis.Result(path=path, seconds=1.5, sample_rate=22050, format="mp3")

    monkeypatch.setattr(synthesis, "generate", fake_generate)
    model = type("M", (), {"id": "imported:anna", "label": "anna · it_IT · medium (importata)"})()
    ids = []
    for n in range(3):
        monkeypatch.setattr(history, "new_id", lambda n=n: f"20260101-00000{n}-abcd")
        ids.append(history.create(model, f"testo {n}", synthesis.Options())["id"])
    assert [e["id"] for e in history.entries()] == ids[::-1]
    assert history.file_path(history.get(ids[0])).exists()
    history.prune(keep=2)
    assert [e["id"] for e in history.entries()] == ids[:0:-1]
    history.delete(ids[2])
    with pytest.raises(UserError):
        history.get(ids[2])
