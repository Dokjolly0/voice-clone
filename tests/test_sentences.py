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
    monkeypatch.setattr(sentences, "pool", lambda lang, wanted, short=False: [] if short else list(pool))
    voice = workspace.create("v", "it")
    domain = tmp_path / "domain.txt"
    domain.write_text("Benvenuto nel nostro negozio.\n\nGrazie e arrivederci.\n", encoding="utf-8")

    first = sentences.choose(voice, 3, extra=domain)
    assert first == ["Benvenuto nel nostro negozio.", "Grazie e arrivederci.", pool[0]]

    grown = sentences.choose(voice, 5)
    assert grown[:3] == first
    assert len(grown) == 5 and len(set(grown)) == 5

    assert sentences.choose(voice, 2) == grown  # never shrinks


@pytest.mark.parametrize("s, ok", [
    ("Il mare era mosso.", True),
    ("E tu non avevi paura?", True),
    ("Bibliografia.", False),                 # one word
    ("Members Only, Vol.", False),            # cut at an abbreviation
    ('Dopo "Fermo con le mani!', False),      # unbalanced quotes
    ("Ciao.", False),                         # too short
])
def test_short_sentences(s, ok):
    assert sentences.acceptable(s, IT, short=True) is ok


def test_choose_mixes_in_a_share_of_short_sentences(voices_dir, monkeypatch):
    from voice_clone import workspace

    long_pool = [f"Questa è una frase lunga numero {w} della prova." for w in "abcdefghijklmnopqrst"]
    short_pool = [f"Frase breve {w}." for w in "abcdefghij"]
    monkeypatch.setattr(sentences, "pool", lambda lang, wanted, short=False: list(short_pool if short else long_pool))
    chosen = sentences.choose(workspace.create("v", "it"), 20)
    assert sum(s in short_pool for s in chosen) == 3 and len(chosen) == 20  # 15% of 20
