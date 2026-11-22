"""Supported languages.

A language works end to end when all four tools cover it: Drama 3 speaks it (83 languages), Qwen3-ASR can check
the clips, Common Voice has CC0 sentences for it, and a Piper checkpoint in it exists to start training from
(rhasspy/piper-checkpoints). The espeak-ng voice and phoneme type are read from the checkpoint's config.json.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Language:
    code: str            # what users type: --lang it
    name: str
    common_voice: str    # folder under common-voice/server/data/
    checkpoint: str      # folder under rhasspy/piper-checkpoints, quality "medium"
    min_chars: int = 30  # sentence length bounds for the dataset
    max_chars: int = 150
    short_min_chars: int = 12   # short sentences (short_min_chars..min_chars-1) teach greetings, replies...
    short_share: float = 0.15   # ...and make up this share of the sentences added
    chars_per_second: float = 15.0  # speaking rate, only for duration estimates
    samples: tuple[str, ...] = ()   # test sentences for `voice-clone export`

    @property
    def locale(self) -> str:
        """it_IT, en_US...: the part of the checkpoint path Piper uses in model names."""
        return self.checkpoint.split("/")[1]


LANGUAGES = {lang.code: lang for lang in [
    Language("it", "Italian", "it", "it/it_IT/paola/medium", samples=(
        "Buongiorno, come posso aiutarti oggi?",
        "Il treno per Milano partirà dal binario tre con dieci minuti di ritardo.",
        "Grazie per la pazienza, l'operazione è stata completata.",
        "Domani il cielo sarà sereno al nord e nuvoloso al sud.",
        "Ricordati di spegnere le luci prima di uscire.",
    )),
    Language("en", "English (US)", "en", "en/en_US/lessac/medium", samples=(
        "Good morning, how can I help you today?",
        "The train to Boston will leave from platform three, ten minutes late.",
        "Thank you for your patience, the operation is complete.",
        "Tomorrow will be sunny in the north and cloudy in the south.",
        "Remember to switch off the lights before you leave.",
    )),
    Language("de", "German", "de", "de/de_DE/thorsten/medium", samples=(
        "Guten Morgen, wie kann ich Ihnen heute helfen?",
        "Vielen Dank für Ihre Geduld, der Vorgang ist abgeschlossen.",
    )),
    Language("es", "Spanish", "es", "es/es_ES/davefx/medium", samples=(
        "Buenos días, ¿en qué puedo ayudarte hoy?",
        "Gracias por tu paciencia, la operación se ha completado.",
    )),
    Language("fr", "French", "fr", "fr/fr_FR/siwis/medium", samples=(
        "Bonjour, comment puis-je vous aider aujourd'hui ?",
        "Merci de votre patience, l'opération est terminée.",
    )),
    Language("pt", "Portuguese (Brazil)", "pt", "pt/pt_BR/faber/medium", samples=(
        "Bom dia, como posso ajudar você hoje?",
        "Obrigado pela paciência, a operação foi concluída.",
    )),
    Language("nl", "Dutch", "nl", "nl/nl_NL/pim/medium", samples=(
        "Goedemorgen, waarmee kan ik u vandaag helpen?",
        "Bedankt voor uw geduld, de bewerking is voltooid.",
    )),
    Language("pl", "Polish", "pl", "pl/pl_PL/gosia/medium", samples=(
        "Dzień dobry, w czym mogę dziś pomóc?",
        "Dziękujemy za cierpliwość, operacja została zakończona.",
    )),
    Language("ru", "Russian", "ru", "ru/ru_RU/denis/medium", samples=(
        "Доброе утро, чем я могу вам помочь сегодня?",
        "Спасибо за терпение, операция завершена.",
    )),
    Language("ro", "Romanian", "ro", "ro/ro_RO/mihai/medium", samples=(
        "Bună dimineața, cu ce vă pot ajuta astăzi?",
    )),
    Language("tr", "Turkish", "tr", "tr/tr_TR/dfki/medium", samples=(
        "Günaydın, bugün size nasıl yardımcı olabilirim?",
    )),
    Language("cs", "Czech", "cs", "cs/cs_CZ/jirka/medium", samples=(
        "Dobré ráno, jak vám mohu dnes pomoci?",
    )),
    Language("hu", "Hungarian", "hu", "hu/hu_HU/anna/medium", samples=(
        "Jó reggelt, miben segíthetek ma?",
    )),
    Language("fi", "Finnish", "fi", "fi/fi_FI/harri/medium", samples=(
        "Hyvää huomenta, miten voin auttaa tänään?",
    )),
    Language("el", "Greek", "el", "el/el_GR/rapunzelina/medium", samples=(
        "Καλημέρα, πώς μπορώ να σας βοηθήσω σήμερα;",
    )),
    Language("vi", "Vietnamese", "vi", "vi/vi_VN/vais1000/medium", samples=(
        "Xin chào, hôm nay tôi có thể giúp gì cho bạn?",
    )),
    Language("zh", "Chinese (Mandarin)", "zh-CN", "zh/zh_CN/huayan/medium",
             min_chars=8, max_chars=50, short_min_chars=4, chars_per_second=4.5, samples=(
        "早上好，今天我能为您做些什么？",
    )),
]}


def get(code: str) -> Language:
    from voice_clone.util import UserError

    try:
        return LANGUAGES[code]
    except KeyError:
        raise UserError(f"unsupported language {code!r}; choose one of: {', '.join(sorted(LANGUAGES))}") from None
