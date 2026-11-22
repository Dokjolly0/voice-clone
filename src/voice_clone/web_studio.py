"""The "Genera audio" section of the web page: speak any text with any model in the library (page text in Italian).

Four tabs: Genera (text, model and every synthesis setting), Modelli (library, import, Piper's catalog), Pronuncia
(dictionary per language) and Cronologia (past generations). Synthesis runs in the page's own process on the CPU,
fast enough (~50x real time) that no background job is needed.
"""

import shutil
from pathlib import Path

import gradio as gr

from voice_clone import history, lexicon, models, synthesis
from voice_clone.paths import STUDIO_DIR
from voice_clone.util import UserError, read_json, write_json

BITRATES = {
    "mp3": (["64k", "96k", "128k", "192k", "256k", "320k"], "192k"),
    "ogg": (["auto", "64k", "96k", "128k"], "auto"),
    "opus": (["24k", "32k", "48k", "64k", "96k", "128k"], "64k"),
    "m4a": (["64k", "96k", "128k", "192k", "256k"], "128k"),
}
FORMAT_LABELS = [("MP3", "mp3"), ("WAV (senza perdita)", "wav"), ("OGG Vorbis", "ogg"), ("Opus", "opus"),
                 ("FLAC (senza perdita)", "flac"), ("M4A / AAC", "m4a")]
RATE_LABELS = [("Come il modello", 0), ("22050 Hz", 22050), ("24000 Hz", 24000), ("44100 Hz", 44100),
               ("48000 Hz", 48000)]
QUALITIES = [("Tutte", ""), ("Molto bassa (x_low)", "x_low"), ("Bassa (low)", "low"), ("Media (medium)", "medium"),
             ("Alta (high)", "high")]
DEFAULTS = {"format": "mp3", "bitrate": "192k", "sample_rate": 0, "speed": 1.0, "volume_db": 0.0, "pitch": 0.0,
            "expressiveness": 1.0, "sentence_pause": 0.3, "paragraph_pause": 0.8, "normalize": False,
            "use_lexicon": True, "per_line": False}
TEXT_PLACEHOLDER = ("Scrivi o incolla qui il testo.\n\n"
                    "Una riga vuota separa i paragrafi. Per una pausa precisa scrivi [pausa 1.5s] o [pausa 500ms].")


def _err(exc: UserError):
    return gr.Error(str(exc))


def prefs_path() -> Path:
    return STUDIO_DIR / "preferences.json"


def load_prefs() -> dict:
    return {**DEFAULTS, **read_json(prefs_path(), {})}


# --- models --------------------------------------------------------------------------------------------------------

def model_choices() -> list[tuple[str, str]]:
    icons = {"trained": "★", "catalog": "⬇", "imported": "📁"}
    return [(f"{icons[m.origin]} {m.name} · {m.language} · {m.quality}", m.id) for m in models.list_models()]


def model_info(model_id: str | None):
    """Info line and the speaker dropdown (shown only for multi-speaker models)."""
    if not model_id:
        return "", gr.Dropdown(visible=False, choices=[], value=None)
    try:
        m = models.get(model_id)
    except UserError as exc:
        return str(exc), gr.Dropdown(visible=False, choices=[], value=None)
    origin = models.ORIGIN_LABELS[m.origin]
    text = f"{m.language} · qualità {m.quality} · {m.sample_rate} Hz · {origin}"
    if m.speakers:
        text += f" · {len(m.speakers)} parlanti"
        return text, gr.Dropdown(visible=True, choices=m.speakers, value=m.speakers[0])
    return text, gr.Dropdown(visible=False, choices=[], value=None)


def bitrate_dropdown(fmt: str, current: str | None = None, **kwargs):
    """Bitrate choices for the format; hidden for the lossless ones (WAV, FLAC)."""
    if fmt not in BITRATES:
        return gr.Dropdown(visible=False, choices=[], value=None, **kwargs)
    choices, default = BITRATES[fmt]
    return gr.Dropdown(visible=True, choices=choices, value=current if current in choices else default, **kwargs)


def text_stats(text: str, speed: float) -> str:
    text = text or ""
    if not text.strip():
        return ""
    clean = synthesis.PAUSE_RE.sub("", text)
    pauses = sum(v for k, v in synthesis.parse(text) if k == "pause")
    seconds = len(clean.strip()) / 14 / max(speed or 1, 0.1) + pauses
    lines = sum(1 for l in text.splitlines() if l.strip())
    minutes = f"{seconds / 60:.1f} min" if seconds >= 90 else f"{seconds:.0f} s"
    return f"{len(text)} caratteri · {lines} righe · durata stimata ≈ {minutes}"


def build_options(fmt, bitrate, rate, speed, volume, pitch, expressiveness, speaker, sentence_pause, paragraph_pause,
                  normalize, use_lexicon) -> synthesis.Options:
    return synthesis.Options(
        format=fmt, bitrate=None if bitrate in (None, "", "auto") else bitrate, sample_rate=int(rate) or None,
        speed=float(speed), volume_db=float(volume), pitch=float(pitch), expressiveness=float(expressiveness),
        speaker=speaker or None, sentence_pause=float(sentence_pause), paragraph_pause=float(paragraph_pause),
        normalize=bool(normalize), use_lexicon=bool(use_lexicon))


SETTINGS_ORDER = ["format", "bitrate", "sample_rate", "speed", "volume_db", "pitch", "expressiveness",
                  "sentence_pause", "paragraph_pause", "normalize", "use_lexicon", "per_line"]


def save_prefs(values: dict) -> None:
    write_json(prefs_path(), values)


def generate(model_id, text, speaker, fmt, bitrate, rate, speed, volume, pitch, expressiveness, sentence_pause,
             paragraph_pause, normalize, use_lexicon, per_line, progress=gr.Progress()):
    if not model_id:
        raise gr.Error("Scegli un modello (scheda Modelli per scaricarne o importarne uno).")
    if not (text or "").strip():
        raise gr.Error("Scrivi il testo da leggere.")
    try:
        options = build_options(fmt, bitrate, rate, speed, volume, pitch, expressiveness, speaker, sentence_pause,
                                paragraph_pause, normalize, use_lexicon).validate()
        model = models.get(model_id)
        meta = history.create(model, text, options, per_line=bool(per_line),
                              progress=lambda d, t: progress(d / t, desc=f"{d}/{t}"))
    except UserError as exc:
        raise _err(exc) from exc
    save_prefs(dict(zip(SETTINGS_ORDER, [fmt, bitrate, int(rate), speed, volume, pitch, expressiveness,
                                         sentence_pause, paragraph_pause, normalize, use_lexicon, per_line])))
    download = history.file_path(meta)
    size = download.stat().st_size / 1e6
    what = f"{len(meta['files'])} file in uno .zip" if meta["per_line"] else f"{fmt.upper()}"
    summary = f"✓ {what} · {meta['seconds']:.1f} s di audio · {size:.1f} MB"
    return str(history.first_audio(meta)), str(download), summary


def preview(model_id, text, speaker, fmt, bitrate, rate, speed, volume, pitch, expressiveness, sentence_pause,
            paragraph_pause, normalize, use_lexicon):
    if not model_id:
        raise gr.Error("Scegli un modello.")
    try:
        options = build_options("wav", None, rate, speed, volume, pitch, expressiveness, speaker, sentence_pause,
                                paragraph_pause, normalize, use_lexicon)
        result = synthesis.preview(models.get(model_id), text or "", STUDIO_DIR / "tmp" / "preview.wav", options)
    except UserError as exc:
        raise _err(exc) from exc
    return str(result.path), "Anteprima della prima frase (non salvata nella cronologia)."


# --- models tab ----------------------------------------------------------------------------------------------------

def library_table():
    rows = [[m.id, m.language, m.quality, models.ORIGIN_LABELS[m.origin], len(m.speakers) or 1]
            for m in models.list_models()]
    removable = [m.id for m in models.list_models() if m.origin != "trained"]
    return rows, gr.Dropdown(choices=removable, value=None)


def import_model(files, name):
    if not files:
        raise gr.Error("Carica il file .onnx e il suo .onnx.json.")
    paths = [Path(f) for f in files]
    onnx = [p for p in paths if p.name.endswith(".onnx")]
    config = [p for p in paths if p.name.endswith(".onnx.json") or p.suffix == ".json"]
    if len(onnx) != 1 or len(config) != 1:
        raise gr.Error("Servono esattamente due file: il modello .onnx e la sua configurazione .onnx.json.")
    tmp = STUDIO_DIR / "tmp" / "import"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    # The upload keeps the original names only by chance; give the config the name the .onnx expects.
    target = tmp / onnx[0].name
    shutil.copy(onnx[0], target)
    shutil.copy(config[0], tmp / (onnx[0].name + ".json"))
    try:
        m = models.import_files(target, name=(name or "").strip() or None)
    except UserError as exc:
        raise _err(exc) from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    gr.Info(f"Importato: {m.name}")
    return (*library_table(), gr.Dropdown(choices=model_choices(), value=m.id), None, "")


def remove_model(model_id):
    if not model_id:
        raise gr.Error("Scegli il modello da rimuovere.")
    try:
        models.remove(model_id)
    except UserError as exc:
        raise _err(exc) from exc
    gr.Info("Modello rimosso.")
    return (*library_table(), gr.Dropdown(choices=model_choices(), value=None))


def catalog_languages():
    try:
        entries = models.catalog()
    except UserError:
        return [("(catalogo non raggiungibile)", "")]
    langs = {}
    for v in entries.values():
        lang = v["language"]
        langs[lang["family"]] = f"{lang['name_native']} ({lang['family']})"
    return sorted(((label, code) for code, label in langs.items()), key=lambda x: x[0].lower())


def catalog_table(lang, quality):
    if not lang:
        return [], gr.Dropdown(choices=[], value=None)
    try:
        rows = models.search(lang, quality or None)
    except UserError as exc:
        raise _err(exc) from exc
    table = [[r["key"], r["language"], r["quality"], r["speakers"], r["size_mb"], "✓" if r["installed"] else ""]
             for r in rows]
    choices = [r["key"] for r in rows if not r["installed"]]
    return table, gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def download_model(key, lang, quality, progress=gr.Progress()):
    if not key:
        raise gr.Error("Scegli una voce del catalogo.")
    progress(0.1, desc=f"Download di {key}…")
    try:
        m = models.download_catalog(key)
    except UserError as exc:
        raise _err(exc) from exc
    gr.Info(f"Scaricato: {m.name}")
    return (*catalog_table(lang, quality), *library_table(), gr.Dropdown(choices=model_choices(), value=m.id))


# --- pronunciation tab ---------------------------------------------------------------------------------------------

def language_of(model_id: str | None) -> str:
    try:
        return models.get(model_id).language_family if model_id else "it"
    except UserError:
        return "it"


def rules_table(lang):
    try:
        rules = lexicon.rules(lang)
    except UserError:
        rules = []
    rows = [[r["written"], r["spoken"], "sì" if r.get("whole_word", True) else "no",
             "sì" if r.get("match_case") else "no"] for r in sorted(rules, key=lambda r: r["written"].lower())]
    return rows, gr.Dropdown(choices=[r[0] for r in rows], value=None)


def add_rule(lang, written, spoken, whole_word, match_case):
    try:
        lexicon.add(lang, written or "", spoken or "", whole_word=whole_word, match_case=match_case)
    except UserError as exc:
        raise _err(exc) from exc
    return (*rules_table(lang), "", "")


def remove_rule(lang, written):
    if not written:
        raise gr.Error("Scegli la regola da rimuovere.")
    try:
        lexicon.remove(lang, written)
    except UserError as exc:
        raise _err(exc) from exc
    return rules_table(lang)


def test_rules(lang, text, model_id):
    if not (text or "").strip():
        raise gr.Error("Scrivi una frase di prova.")
    spoken = lexicon.apply(text, lang)
    audio = None
    if model_id:
        try:
            options = synthesis.Options(format="wav", use_lexicon=False)
            audio = str(synthesis.generate(models.get(model_id), spoken, STUDIO_DIR / "tmp" / "lexicon-test.wav",
                                           options).path)
        except UserError as exc:
            raise _err(exc) from exc
    return f"Verrà letto: **{spoken}**", audio


# --- history tab ---------------------------------------------------------------------------------------------------

def history_view():
    entries = history.entries(100)
    rows = []
    choices = []
    for e in entries:
        flat = " ".join(e["text"].split())
        snippet = flat[:50] + ("…" if len(flat) > 50 else "")
        fmt = e["options"]["format"].upper() + (f" ×{len(e['files'])}" if e["per_line"] else "")
        rows.append([e["created"], e["model_label"].split(" · ")[0], f"{e['seconds']:.1f} s", fmt, snippet])
        choices.append((f"{e['created']} — {snippet[:40]}", e["id"]))
    return rows, gr.Dropdown(choices=choices, value=choices[0][1] if choices else None)


def history_entry(entry_id):
    if not entry_id:
        return None, None, ""
    try:
        meta = history.get(entry_id)
    except UserError:
        return None, None, ""
    o = meta["options"]
    details = (f"**{meta['model_label']}** · {meta['seconds']:.1f} s · {o['format'].upper()} · velocità {o['speed']} · "
               f"tono {o['pitch']:+g} · volume {o['volume_db']:+g} dB\n\n> " + meta["text"][:1500].replace("\n", "\n> "))
    return str(history.first_audio(meta)), str(history.file_path(meta)), details


def delete_entry(entry_id):
    if entry_id:
        history.delete(entry_id)
    rows, dd = history_view()
    remaining = history.entries(1)
    return (rows, dd, *history_entry(remaining[0]["id"] if remaining else None))


def reuse_entry(entry_id):
    """Text, model and settings of a past generation, back into the Genera tab."""
    if not entry_id:
        raise gr.Error("Scegli una generazione.")
    meta = history.get(entry_id)
    o = {**DEFAULTS, **meta["options"], "per_line": meta["per_line"]}
    o["bitrate"] = o.get("bitrate") or ("auto" if o["format"] == "ogg" else BITRATES.get(o["format"], ([], None))[1])
    return (gr.Tabs(selected="genera"), meta["model_id"], meta["text"], *settings_values(o))


def settings_values(p: dict) -> list:
    return [p["format"], bitrate_dropdown(p["format"], p["bitrate"]), int(p.get("sample_rate") or 0), p["speed"],
            p["volume_db"], p["pitch"], p["expressiveness"], p["sentence_pause"], p["paragraph_pause"],
            p["normalize"], p["use_lexicon"], p["per_line"]]


# --- section -------------------------------------------------------------------------------------------------------

def build():
    """The studio section. Returns its root column (hidden until the mode switch shows it) and the
    (function, inputs, outputs) to run when it is shown, which refreshes models, catalog, rules and history."""
    prefs = load_prefs()
    choices = model_choices()

    with gr.Column(visible=False) as root:
        gr.Markdown("# Genera audio\nTrasforma un testo in un file audio con una delle tue voci, una voce Piper "
                    "scaricata dal catalogo o un modello importato.")
        with gr.Tabs() as tabs:
            # --- Genera ------------------------------------------------------------------------------------------
            with gr.Tab("Genera", id="genera"):
                with gr.Row(equal_height=True):
                    model_dd = gr.Dropdown(choices=choices, value=choices[0][1] if choices else None,
                                           label="Modello", scale=5,
                                           info="★ le tue voci · ⬇ scaricate dal catalogo · 📁 importate")
                    refresh_btn = gr.Button("↻", scale=0, min_width=50)
                model_md = gr.Markdown()
                speaker_dd = gr.Dropdown(label="Parlante", visible=False, filterable=True)
                text = gr.Textbox(lines=10, max_lines=30, label="Testo", placeholder=TEXT_PLACEHOLDER)
                with gr.Row(equal_height=True):
                    stats = gr.Markdown()
                    pause_len = gr.Number(value=1.0, minimum=0.1, maximum=30, step=0.1, label="Pausa (s)", scale=0,
                                          min_width=100)
                    pause_btn = gr.Button("Inserisci pausa", scale=0, min_width=150)
                    upload_txt = gr.UploadButton("Carica .txt", file_types=[".txt"], scale=0, min_width=130)

                with gr.Accordion("Impostazioni", open=True):
                    with gr.Row():
                        fmt = gr.Dropdown(FORMAT_LABELS, value=prefs["format"], label="Formato")
                        bitrate = bitrate_dropdown(prefs["format"], prefs["bitrate"], label="Qualità (bitrate)")
                        rate = gr.Dropdown(RATE_LABELS, value=int(prefs["sample_rate"] or 0), label="Frequenza")
                    with gr.Row():
                        speed = gr.Slider(0.5, 2.0, value=prefs["speed"], step=0.05, label="Velocità",
                                          info="1 = normale; è approssimativa: le pause della punteggiatura "
                                               "cambiano meno")
                        volume = gr.Slider(-20, 20, value=prefs["volume_db"], step=1, label="Volume (dB)")
                    with gr.Row():
                        pitch = gr.Slider(-12, 12, value=prefs["pitch"], step=0.5, label="Tono (semitoni)",
                                          info="Oltre ±4 la voce suona artificiale")
                        expressiveness = gr.Slider(0, 1.5, value=prefs["expressiveness"], step=0.05,
                                                   label="Espressività",
                                                   info="Variazione di intonazione e ritmo: 0 = piatta, 1 = naturale")
                    with gr.Row():
                        sentence_pause = gr.Slider(0, 3, value=prefs["sentence_pause"], step=0.05,
                                                   label="Pausa tra le frasi (s)")
                        paragraph_pause = gr.Slider(0, 5, value=prefs["paragraph_pause"], step=0.1,
                                                    label="Pausa tra i paragrafi (s)")
                    with gr.Row():
                        normalize = gr.Checkbox(value=prefs["normalize"], label="Normalizza il volume (−16 LUFS)")
                        use_lexicon = gr.Checkbox(value=prefs["use_lexicon"], label="Usa il dizionario di pronuncia")
                        per_line = gr.Checkbox(value=prefs["per_line"],
                                               label="Un file per ogni riga (scaricati in uno .zip)")
                    reset_btn = gr.Button("Ripristina le impostazioni predefinite", size="sm")
                with gr.Row():
                    preview_btn = gr.Button("▶ Anteprima")
                    generate_btn = gr.Button("Genera", variant="primary", scale=2)
                result_md = gr.Markdown()
                result_audio = gr.Audio(label="Risultato", interactive=False)
                result_file = gr.File(label="Scarica", interactive=False)

            # --- Modelli -----------------------------------------------------------------------------------------
            with gr.Tab("Modelli", id="modelli"):
                gr.Markdown("### La tua libreria")
                library_df = gr.Dataframe(headers=["id", "lingua", "qualità", "origine", "parlanti"],
                                          interactive=False, wrap=True)
                with gr.Row():
                    remove_dd = gr.Dropdown(label="Rimuovi un modello scaricato o importato", scale=4)
                    remove_btn = gr.Button("Rimuovi", variant="stop", scale=1)
                gr.Markdown("### Scarica dal catalogo di Piper\nVoci ufficiali, gratuite, in decine di lingue. "
                            "Ognuna ha la sua licenza, scaricata insieme al modello (MODEL_CARD).")
                with gr.Row():
                    cat_lang = gr.Dropdown(label="Lingua", scale=2)
                    cat_quality = gr.Dropdown(QUALITIES, value="", label="Qualità", scale=1)
                catalog_df = gr.Dataframe(headers=["voce", "lingua", "qualità", "parlanti", "MB", "scaricata"],
                                          interactive=False, wrap=True)
                with gr.Row():
                    cat_key = gr.Dropdown(label="Voce da scaricare", scale=4)
                    download_btn = gr.Button("Scarica", variant="primary", scale=1)
                gr.Markdown("### Importa un modello\nUn modello Piper è fatto di due file: `nome.onnx` e "
                            "`nome.onnx.json`. Caricali entrambi.")
                import_files = gr.File(file_count="multiple", file_types=[".onnx", ".json"], type="filepath",
                                       label=".onnx + .onnx.json")
                with gr.Row():
                    import_name = gr.Textbox(label="Nome (facoltativo)", scale=4)
                    import_btn = gr.Button("Importa", variant="primary", scale=1)

            # --- Pronuncia ---------------------------------------------------------------------------------------
            with gr.Tab("Pronuncia", id="pronuncia"):
                gr.Markdown("Regole \"scritto → detto\" per lingua, applicate prima della lettura: sigle, nomi, "
                            "marchi. Il \"detto\" può essere testo (`A C I`) o fonemi di Piper (`[[ kˈɔp ]]`).")
                lex_lang = gr.Textbox(value="it", label="Lingua (codice, es. it, en)", max_lines=1)
                rules_df = gr.Dataframe(headers=["scritto", "detto", "solo parola intera", "maiuscole esatte"],
                                        interactive=False, wrap=True)
                with gr.Row():
                    written = gr.Textbox(label="Scritto", placeholder="es. ACI")
                    spoken = gr.Textbox(label="Detto", placeholder="es. A C I")
                with gr.Row():
                    whole_word = gr.Checkbox(value=True, label="Solo parole intere")
                    match_case = gr.Checkbox(value=False, label="Maiuscole esatte")
                    add_rule_btn = gr.Button("Aggiungi regola", variant="primary")
                with gr.Row():
                    rule_dd = gr.Dropdown(label="Rimuovi la regola", scale=4)
                    remove_rule_btn = gr.Button("Rimuovi", variant="stop", scale=1)
                gr.Markdown("### Prova")
                with gr.Row():
                    lex_test = gr.Textbox(label="Frase di prova", scale=4)
                    lex_test_btn = gr.Button("Prova e ascolta", scale=1)
                lex_result = gr.Markdown()
                lex_audio = gr.Audio(label="Con il modello scelto in Genera", interactive=False)

            # --- Cronologia --------------------------------------------------------------------------------------
            with gr.Tab("Cronologia", id="cronologia"):
                history_refresh = gr.Button("↻ Aggiorna", size="sm")
                history_df = gr.Dataframe(headers=["data", "modello", "durata", "formato", "testo"],
                                          interactive=False, wrap=True)
                entry_dd = gr.Dropdown(label="Generazione")
                entry_md = gr.Markdown()
                entry_audio = gr.Audio(label="Ascolta", interactive=False)
                entry_file = gr.File(label="Scarica", interactive=False)
                with gr.Row():
                    reuse_btn = gr.Button("Riusa testo e impostazioni")
                    delete_btn = gr.Button("Elimina", variant="stop")

        # --- wiring ------------------------------------------------------------------------------------------------
        settings = [fmt, bitrate, rate, speed, volume, pitch, expressiveness, sentence_pause, paragraph_pause,
                    normalize, use_lexicon, per_line]
        synth_inputs = [model_dd, text, speaker_dd, fmt, bitrate, rate, speed, volume, pitch, expressiveness,
                        sentence_pause, paragraph_pause, normalize, use_lexicon]

        model_dd.change(model_info, model_dd, [model_md, speaker_dd])
        model_dd.change(language_of, model_dd, lex_lang)
        refresh_btn.click(lambda mid: gr.Dropdown(choices=model_choices(), value=mid), model_dd, model_dd)
        fmt.change(lambda f, b: bitrate_dropdown(f, b), [fmt, bitrate], bitrate)
        for comp in (text, speed):
            comp.change(text_stats, [text, speed], stats, show_progress="hidden")
        pause_btn.click(lambda t, s: (t or "").rstrip() + f" [pausa {float(s):g}s] ", [text, pause_len], text)
        upload_txt.upload(lambda f: Path(f).read_text(encoding="utf-8", errors="replace"), upload_txt, text)
        reset_btn.click(lambda: settings_values(DEFAULTS), None, settings)

        generate_btn.click(generate, [*synth_inputs, per_line], [result_audio, result_file, result_md]).then(
            history_view, None, [history_df, entry_dd])
        preview_btn.click(preview, synth_inputs, [result_audio, result_md])

        remove_btn.click(remove_model, remove_dd, [library_df, remove_dd, model_dd])
        cat_lang.change(catalog_table, [cat_lang, cat_quality], [catalog_df, cat_key])
        cat_quality.change(catalog_table, [cat_lang, cat_quality], [catalog_df, cat_key])
        download_btn.click(download_model, [cat_key, cat_lang, cat_quality],
                           [catalog_df, cat_key, library_df, remove_dd, model_dd])
        import_btn.click(import_model, [import_files, import_name],
                         [library_df, remove_dd, model_dd, import_files, import_name])

        lex_lang.change(rules_table, lex_lang, [rules_df, rule_dd])
        add_rule_btn.click(add_rule, [lex_lang, written, spoken, whole_word, match_case],
                           [rules_df, rule_dd, written, spoken])
        remove_rule_btn.click(remove_rule, [lex_lang, rule_dd], [rules_df, rule_dd])
        lex_test_btn.click(test_rules, [lex_lang, lex_test, model_dd], [lex_result, lex_audio])

        history_refresh.click(history_view, None, [history_df, entry_dd])
        entry_dd.change(history_entry, entry_dd, [entry_audio, entry_file, entry_md])
        delete_btn.click(delete_entry, entry_dd, [history_df, entry_dd, entry_audio, entry_file, entry_md])
        reuse_btn.click(reuse_entry, entry_dd, [tabs, model_dd, text, *settings])

        def on_show(model_id):
            current = model_choices()
            ids = [v for _, v in current]
            model_id = model_id if model_id in ids else (ids[0] if ids else None)
            lang = language_of(model_id)
            langs = catalog_languages()
            default_lang = lang if any(code == lang for _, code in langs) else None
            return (gr.Dropdown(choices=current, value=model_id), *model_info(model_id), *library_table(),
                    gr.Dropdown(choices=langs, value=default_lang), lang, *rules_table(lang), *history_view(),
                    text_stats("", 1.0))

        show = (on_show, [model_dd], [model_dd, model_md, speaker_dd, library_df, remove_dd, cat_lang, lex_lang,
                                      rules_df, rule_dd, history_df, entry_dd, stats])
    return root, show
