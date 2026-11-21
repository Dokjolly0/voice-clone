import pytest

from voice_clone import languages, sentences

IT = languages.get("it")
ZH = languages.get("zh")


@pytest.mark.parametrize("s", [
    "Il gatto dorme tranquillo sul divano di casa.",
    "È arrivato il momento di partire per le vacanze!",
    "Dove hai messo le chiavi della macchina?",
])
def test_accepts_plain_sentences(s):
    assert sentences.acceptable(s, IT)


@pytest.mark.parametrize("s", [
    "Il treno parte alle 10 dal binario sei.",          # digits
    "Scrivi a info@esempio.it per saperne di più.",      # symbols
    "La NATO ha annunciato nuove misure oggi stesso.",   # abbreviation
    "il gatto dorme tranquillo sul divano di casa.",     # lowercase start
    "Il gatto dorme tranquillo sul divano di casa",      # no final punctuation
    "Corta.",                                            # too short
    "Parola " * 30 + "fine.",                            # too long
])
def test_rejects_ambiguous_sentences(s):
    assert not sentences.acceptable(s, IT)


def test_uncased_scripts_are_not_rejected_for_case():
    assert sentences.acceptable("今天的天气非常好，我们去公园散步吧。", ZH)


def test_clean_normalizes_quotes_and_spaces():
    assert sentences.clean("  L’uomo   disse “ciao”.  ") == 'L\'uomo disse "ciao".'


def test_choose_is_append_only_and_puts_domain_first(voices_dir, tmp_path, monkeypatch):
    from voice_clone import workspace

    pool = [f"Frase numero {w} per la prova." for w in ("uno", "due", "tre", "quattro", "cinque", "sei")]
    monkeypatch.setattr(sentences, "pool", lambda lang, wanted: list(pool))
    voice = workspace.create("v", "it")
    domain = tmp_path / "domain.txt"
    domain.write_text("Benvenuto nel nostro negozio.\n\nGrazie e arrivederci.\n", encoding="utf-8")

    first = sentences.choose(voice, 3, extra=domain)
    assert first == ["Benvenuto nel nostro negozio.", "Grazie e arrivederci.", pool[0]]

    grown = sentences.choose(voice, 5)
    assert grown[:3] == first
    assert len(grown) == 5 and len(set(grown)) == 5

    assert sentences.choose(voice, 2) == grown  # never shrinks
