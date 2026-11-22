"""`voice-clone web`: a guided web interface for the whole workflow (page text in Italian).

Layout: a sidebar with the voices and the five steps (done / running / to do / locked), one step at a time in the
main area, and a job panel at the bottom with a progress bar, time left and the log.

Long stages run as background jobs (see jobs.py), exactly as `voice-clone <command> --background` would, so they
keep going when the page is closed and the CLI sees the same state. The page polls every few seconds; audio
players and the chart reload only when their files change, so listening is never interrupted.
"""

import html
import random
import shutil
import wave
from pathlib import Path

import gradio as gr
import pandas as pd

from voice_clone import (audio, dataset, export, jobs, languages, pipeline, reference, settings, train, web_studio,
                         workspace)
from voice_clone.util import UserError, read_json

SAMPLE_SLOTS = 5
CLIP_SLOTS = 4
POLL_SECONDS = 3
# Measured speeds, used for estimates until the voice's own training metrics exist.
SPEEDS = {
    # Drama 3 real-time factor, seconds per training epoch per hour of audio, and where it was measured.
    "gpu": {"drama_rtf": 1.7, "epoch_s_per_hour": 10.5, "where": "su una RTX 4090"},
    "cpu": {"drama_rtf": 0.3, "epoch_s_per_hour": 170.0,
            "where": "con la sola CPU (misurato su un Ryzen 9 16 core; su CPU più piccole molto di più)"},
}


def speed() -> dict:
    return SPEEDS[settings.device()]

STEP_TITLES = ["Audio", "Riferimento", "Dataset", "Addestramento", "Esporta e prova"]
STATE_ICONS = {"done": "✓", "running": "▶", "todo": "○", "locked": "🔒"}

PHASE_LABELS = {
    "merging": "Unione delle registrazioni",
    "transcribing": "Trascrizione del riferimento",
    "building_voice": "Creazione della voce per Drama 3",
    "choosing_sentences": "Scelta delle frasi",
    "loading_model": "Caricamento di Drama 3 (circa un minuto)",
    "generating": "Generazione delle clip",
    "verifying": "Verifica delle clip",
    "exporting_dataset": "Esportazione del dataset",
    "downloading_base": "Download della voce di partenza",
    "training": "Addestramento",
    "exporting": "Esportazione del modello",
    "speaking": "Lettura delle frasi di prova",
}
COMMAND_LABELS = {"reference": "Riferimento", "try-voice": "Prova della voce", "dataset": "Dataset",
                  "train": "Addestramento", "export": "Esportazione", "compare": "Confronto", "run": "Tutti i passi"}

CSS = """
.vc-progress { padding: 10px 14px; border-radius: 10px; border: 1px solid var(--border-color-primary);
               background: var(--background-fill-secondary); }
.vc-progress .vc-row { display: flex; justify-content: space-between; gap: 12px; flex-wrap: wrap;
                       margin-bottom: 6px; }
.vc-progress .vc-track { height: 10px; border-radius: 5px; background: var(--background-fill-primary);
                         overflow: hidden; border: 1px solid var(--border-color-primary); }
.vc-progress .vc-fill { height: 100%; background: var(--color-accent); transition: width 0.6s; }
.vc-progress .vc-indeterminate { width: 30%; animation: vc-slide 1.4s ease-in-out infinite; }
@keyframes vc-slide { 0% { margin-left: -30%; } 100% { margin-left: 100%; } }
.vc-progress.vc-error { border-color: #e53e3e; }
.vc-muted { opacity: 0.7; }
.vc-step-btn { justify-content: flex-start !important; text-align: left !important; }
"""


# --- helpers -----------------------------------------------------------------------------------------------------

def _voice(name: str | None) -> workspace.Voice:
    if not name:
        raise gr.Error("Crea o seleziona prima una voce.")
    try:
        return workspace.load(name)
    except UserError as exc:
        raise gr.Error(str(exc)) from exc


def _start(name: str, args: list[str]) -> None:
    voice = _voice(name)
    try:
        jobs.start(voice, args)
    except UserError as exc:
        raise gr.Error(str(exc)) from exc
    gr.Info(f"Avviato: {COMMAND_LABELS.get(args[0], args[0])}. Segui l'avanzamento in basso.")


def _file(path: Path) -> str | None:
    return str(path) if path.exists() else None


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    minutes = int(round(seconds / 60))
    if minutes < 1:
        return "meno di un minuto"
    if minutes < 60:
        return f"{minutes} min"
    return f"{minutes // 60} h {minutes % 60:02d} min"


def dataset_minutes(voice: workspace.Voice) -> float:
    wav_dir = voice.dataset_dir / "wav"
    total = 0.0
    for path in wav_dir.glob("*.wav") if wav_dir.exists() else []:
        try:
            with wave.open(str(path)) as w:
                total += w.getnframes() / w.getframerate()
        except (wave.Error, EOFError):
            pass
    return total / 60


def device_label() -> str:
    if settings.is_cpu():
        return "Modalità: **solo CPU** (lenta)"
    from voice_clone import system

    gpu = system.query_gpu()
    return f"Modalità: **GPU** ({gpu.name.replace('NVIDIA GeForce ', '')})" if gpu else "Modalità: **GPU**"


# --- job panel -----------------------------------------------------------------------------------------------------

def progress_html(name: str | None) -> tuple[str, bool]:
    """The job panel's HTML, and whether a job is running."""
    if not name:
        return "", False
    p = jobs.progress(workspace.load(name))
    command = COMMAND_LABELS.get(p["command"][0], p["command"][0]) if p["command"] else ""
    if not p["running"]:
        if not p["command"]:
            return '<div class="vc-progress vc-muted">Nessun lavoro in corso.</div>', False
        if p["error"] == "interrupted":
            return (f'<div class="vc-progress">⏸ {command}: interrotto. Avvia di nuovo il passo per riprendere '
                    "da dove era arrivato.</div>"), False
        if p["error"]:
            return (f'<div class="vc-progress vc-error">✗ {command}: errore — {html.escape(p["error"])}'
                    "<br><span class='vc-muted'>Dettagli nel log qui sotto.</span></div>"), False
        return f'<div class="vc-progress">✓ {command}: completato.</div>', False

    label = PHASE_LABELS.get(p["phase"], "In corso")
    if p["phase"] == "training" and p["done"] is None:
        label = "Preparazione dell'addestramento"
    title = f"<b>{command}</b> — {label}" if label != command else f"<b>{command}</b>"
    count, bar = "", '<div class="vc-fill vc-indeterminate"></div>'
    if p["done"] is not None and p["total"]:
        pct = max(0.0, min(100.0, 100 * p["done"] / p["total"]))
        unit = " epoche" if p["phase"] == "training" else ""
        count = f"{p['done']}/{p['total']}{unit} · {pct:.0f}%"
        bar = f'<div class="vc-fill" style="width:{pct:.1f}%"></div>'
    eta = f"mancano ~{fmt_duration(p['eta'])}" if p["eta"] else ""
    right = " · ".join(x for x in (count, eta) if x)
    page = (f'<div class="vc-progress"><div class="vc-row"><span>▶ {title}</span>'
            f"<span>{right}</span></div><div class='vc-track'>{bar}</div></div>")
    return page, True


# --- steps and navigation ------------------------------------------------------------------------------------------

def step_states(name: str | None) -> list[dict]:
    if not name:
        return [{"number": i + 1, "state": "locked", "ready": False} for i in range(5)]
    return pipeline.steps(workspace.load(name))


def step_buttons(states: list[dict], current: int | None) -> list:
    """Sidebar step buttons. With current=None (polling) the highlight set by navigation is left alone, so a slow
    poll can never move it back to an old step."""
    updates = []
    for st in states:
        kwargs = {"value": f"{STATE_ICONS[st['state']]}  {st['number']} · {STEP_TITLES[st['number'] - 1]}",
                  "interactive": st["state"] != "locked"}
        if current is not None:
            kwargs["variant"] = "primary" if st["number"] == current else "secondary"
        updates.append(gr.Button(**kwargs))
    return updates


def show(name: str | None, step: int) -> list:
    """Navigate: current step, the five panels' visibility, the step buttons, welcome and new-voice panels."""
    states = step_states(name)
    if name and states[step - 1]["state"] == "locked":
        step = pipeline.next_step(states)
    panels = [gr.Column(visible=bool(name) and i + 1 == step) for i in range(5)]
    return [step, *panels, *step_buttons(states, step), gr.Column(visible=not name), gr.Column(visible=False)]


# --- 1. audio ------------------------------------------------------------------------------------------------------

def audio_view(name: str | None):
    if not name:
        return [], "", gr.Dropdown(choices=[])
    voice = workspace.load(name)
    durations = [(p.name, audio.duration(p)) for p in voice.inputs()]
    total = sum(d for _, d in durations)
    if not durations:
        summary = "Nessuna registrazione ancora."
    else:
        hint = "Bastano 10–60 secondi in tutto." if total >= 10 else "Meglio arrivare ad almeno 10 secondi."
        summary = f"**{len(durations)} file, {total:.0f} secondi in tutto.** {hint}"
    rows = [[n, f"{d:.1f} s"] for n, d in durations]
    return rows, summary, gr.Dropdown(choices=[n for n, _ in durations], value=None)


def add_files(name: str, files: list[str] | None):
    voice = _voice(name)
    if files:
        try:
            workspace.add_audio(voice, [Path(f) for f in files])
        except UserError as exc:
            raise gr.Error(str(exc)) from exc
        gr.Info(f"Aggiunti {len(files)} file.")
    return (*audio_view(name), None)


def remove_file(name: str, filename: str | None):
    voice = _voice(name)
    if not filename:
        raise gr.Error("Scegli il file da rimuovere.")
    workspace.remove_audio(voice, [filename])
    return audio_view(name)


# --- 2. reference --------------------------------------------------------------------------------------------------

def reference_view(name: str | None):
    if not name:
        return None, "", None, ""
    voice = workspace.load(name)
    p = reference.paths(voice)
    transcript = p["transcript"].read_text(encoding="utf-8").strip() if p["transcript"].exists() else ""
    st = reference.status(voice)
    if st["voice"]:
        summary = f"✓ Voce pronta, da {st['reference_seconds']:.0f} secondi di riferimento."
    elif st["reference_seconds"]:
        summary = "Riferimento pronto, manca ancora la voce per Drama 3."
    else:
        summary = ""
    return _file(p["reference"]), transcript, _file(voice.reference_dir / "try.wav"), summary


def save_transcript(name: str, text: str, max_seconds: float):
    voice = _voice(name)
    p = reference.paths(voice)
    if not p["reference"].exists():
        raise gr.Error("Prepara prima il riferimento.")
    if not text.strip():
        raise gr.Error("La trascrizione è vuota.")
    p["transcript"].write_text(text.strip() + "\n", encoding="utf-8")
    _start(name, ["reference", name, "--max-seconds", str(max_seconds)])


def try_voice(name: str, text: str):
    if not text.strip():
        raise gr.Error("Scrivi una frase da far leggere.")
    _start(name, ["try-voice", name, text.strip()])


# --- 3. dataset ----------------------------------------------------------------------------------------------------

def dataset_estimate(name: str | None, count: float | None) -> str:
    if not name or not count:
        return ""
    lang = workspace.load(name).language
    avg_chars = (lang.min_chars + lang.max_chars) / 2 * 0.6  # Common Voice sentences sit in the lower half
    minutes = int(count) * avg_chars / lang.chars_per_second / 60
    work = minutes * 60 / speed()["drama_rtf"] * 1.3  # plus checks and retries
    return f"≈ {minutes:.0f} min di parlato · circa {fmt_duration(work)} di lavoro {speed()['where']}"


def dataset_view(name: str | None):
    if not name:
        return ""
    voice = workspace.load(name)
    d = dataset.status(voice)
    if not d["total"]:
        return ""
    text = (f"**{d['verified']}** clip buone su {d['total']} frasi · {d['pending']} da fare · "
            f"{d['failed']} scartate dopo tutti i tentativi")
    if d["exported"]:
        text += f"\n\n✓ Dataset esportato: {d['exported']} clip, {dataset_minutes(voice):.0f} min di audio."
    return text


def start_dataset(name: str, count: float, extra: str | None, max_cer: float, attempts: float):
    voice = _voice(name)
    args = ["dataset", name, "--sentences", str(int(count)), "--max-cer", str(max_cer),
            "--max-attempts", str(int(attempts))]
    if extra:
        domain = voice.dir / "domain-sentences.txt"
        shutil.copy(extra, domain)
        args += ["--extra", str(domain)]
    _start(name, args)


def _players(pairs: list[tuple[Path, str]], slots: int) -> list:
    updates = [gr.Audio(value=str(p), label=t[:100], visible=True) for p, t in pairs[:slots]]
    return updates + [gr.Audio(value=None, label="", visible=False) for _ in range(slots - len(updates))]


def random_clips(name: str | None):
    if not name:
        return _players([], CLIP_SLOTS)
    voice = workspace.load(name)
    wav_dir = voice.dataset_dir / "wav"
    clips = sorted(wav_dir.glob("*.wav")) if wav_dir.exists() else []
    if not clips:
        gr.Info("Nessuna clip ancora: genera prima il dataset.")
    sentences = voice.sentences()
    pairs = []
    for path in random.sample(clips, min(CLIP_SLOTS, len(clips))):
        idx = int(path.stem) if path.stem.isdigit() else -1
        pairs.append((path, sentences[idx] if 0 <= idx < len(sentences) else path.stem))
    return _players(pairs, CLIP_SLOTS)


# --- 4. training ---------------------------------------------------------------------------------------------------

def training_view(name: str | None, epochs: float | None):
    """Chart data, summary, and the label of the train button."""
    empty = pd.DataFrame({"epoca": pd.Series(dtype=int), "val_mel": pd.Series(dtype=float)})
    if not name:
        return empty, "", gr.Button(value="Avvia l'addestramento")
    voice = workspace.load(name)
    rows = [r for r in train.metrics(voice) if r["val_mel"] is not None]
    if rows:
        df = pd.DataFrame({"epoca": [r["epoch"] for r in rows], "val_mel": [r["val_mel"] for r in rows]})
    else:  # runs recorded before metrics.csv existed: the kept checkpoints only
        pts = sorted((c.epoch, c.val_mel) for c in export.checkpoints(voice)
                     if c.label != "last" and c.val_mel is not None)
        df = pd.DataFrame({"epoca": [e for e, _ in pts], "val_mel": [v for _, v in pts]}) if pts else empty

    prog = train.progress(voice)
    target = int(epochs or 2000)
    parts = []
    minutes = dataset_minutes(voice)
    run_device = read_json(voice.training_dir / "run.json", {}).get("device", "gpu")
    measured = train.metrics(voice)[-20:] if run_device == settings.device() else []
    if measured:  # this voice's own recent epochs, on the same kind of device
        per_epoch = sum(r["seconds"] for r in measured) / len(measured)
        parts.append(f"{target} epoche: circa {fmt_duration(target * per_epoch)} in tutto, misurato sulle ultime "
                     f"epoche ({per_epoch:.0f} s l'una).")
    elif minutes:
        est = target * speed()["epoch_s_per_hour"] * max(minutes, 10) / 60
        parts.append(f"{target} epoche su {minutes:.0f} min di audio: circa {fmt_duration(est)} {speed()['where']}.")
    if settings.is_cpu():
        parts.append("Con la sola CPU l'addestramento è lento: valuta 500–1000 epoche, e ascolta le prove a metà "
                     "strada (passo 5) per decidere quando fermarti.")
    if prog["best_val_mel"] is not None:
        parts.append(f"Migliore finora: **val_mel {prog['best_val_mel']:.4f}** all'epoca {prog['best_epoch']}. "
                     "Più basso è meglio; quando la curva si appiattisce, la voce migliora soprattutto in pulizia.")
    if prog["last"] is None:
        label = "Avvia l'addestramento"
    elif prog["finished"]:
        label = f"Continua fino a {target} epoche"
    else:
        label = "Riprendi l'addestramento"
    return df, "\n\n".join(parts), gr.Button(value=label)


def start_train(name: str, epochs: float, batch: float):
    voice = _voice(name)
    args = ["train", name, "--epochs", str(int(epochs)), "--batch-size", str(int(batch or 0))]
    if train.last_checkpoint(voice):
        args.append("--resume")
    _start(name, args)


# --- 5. export -----------------------------------------------------------------------------------------------------

def export_view(name: str | None):
    """Summary, sample players, model files, checkpoint choices, comparison text and players."""
    if not name:
        return ("", *_players([], SAMPLE_SLOTS), None, gr.Dropdown(choices=[]), gr.Dropdown(choices=[]), "",
                *_players([], SAMPLE_SLOTS), *_players([], SAMPLE_SLOTS))
    voice = workspace.load(name)
    source = read_json(voice.export_dir / "source.json", {})
    if voice.onnx_path.exists():
        origin = f" (da {source['checkpoint']})" if source.get("checkpoint") else ""
        summary = f"✓ Modello esportato{origin}: `{voice.onnx_path.name}`"
    else:
        summary = ""
    files = [str(p) for p in (voice.onnx_path, voice.onnx_config_path) if p.exists()] or None

    ckpts = export.checkpoints(voice)
    choices = []
    for c in ckpts:
        if c.label == "last":
            choices.append((f"Ultimo" + (f" — epoca {c.epoch}" if c.epoch is not None else ""), "last"))
        else:
            choices.append((f"Epoca {c.epoch} — val_mel {c.val_mel:.4f}", c.label))
    values = [v for _, v in choices]
    best = min((c for c in ckpts if c.label != "last" and c.val_mel is not None), key=lambda c: c.val_mel,
               default=None)
    comp = read_json(export.variants_dir(voice) / "comparison.json", {})
    a = comp.get("a") or (values[0] if values else None)
    b = comp.get("b") or (best.label if best else None)
    dd_a = gr.Dropdown(choices=choices, value=a if a in values else None)
    dd_b = gr.Dropdown(choices=choices, value=b if b in values else None)

    pairs_a = export.read_samples(export.variants_dir(voice) / comp["a"] / "samples") if comp else []
    pairs_b = export.read_samples(export.variants_dir(voice) / comp["b"] / "samples") if comp else []
    comp_text = (f"**A** = {comp['a']} · **B** = {comp['b']}: stesse frasi, ascoltale a coppie."
                 if pairs_a and pairs_b else "")
    return (summary, *_players(export.read_samples(voice.samples_dir), SAMPLE_SLOTS), files, dd_a, dd_b,
            comp_text, *_players(pairs_a, SAMPLE_SLOTS), *_players(pairs_b, SAMPLE_SLOTS))


def start_compare(name: str, a: str | None, b: str | None):
    if not a or not b:
        raise gr.Error("Scegli due checkpoint.")
    if a == b:
        raise gr.Error("Scegli due checkpoint diversi.")
    _start(name, ["compare", name, a, b])


def promote(name: str, side: str):
    voice = _voice(name)
    comp = read_json(export.variants_dir(voice) / "comparison.json", {})
    if not comp:
        raise gr.Error("Confronta prima due checkpoint.")
    try:
        export.promote(voice, comp[side])
    except UserError as exc:
        raise gr.Error(str(exc)) from exc
    gr.Info(f"{comp[side]} è ora il modello della voce.")


def say_text(name: str, text: str):
    voice = _voice(name)
    if not text.strip():
        raise gr.Error("Scrivi qualcosa da far dire.")
    try:
        return str(export.say(voice, text.strip(), voice.export_dir / "say.wav"))
    except UserError as exc:
        raise gr.Error(str(exc)) from exc


# --- change detection ----------------------------------------------------------------------------------------------

def _mtimes(paths) -> str:
    return "|".join(f"{p.name}:{p.stat().st_mtime_ns}" for p in paths if p.exists())


def signatures(voice: workspace.Voice) -> tuple[str, str]:
    """(files the panels show, training metrics): each changes only when its files do."""
    p = reference.paths(voice)
    files = [*voice.inputs(), p["reference"], p["transcript"], p["voice"], voice.reference_dir / "try.wav",
             voice.dataset_dir / "metadata.csv", voice.onnx_path, voice.export_dir / "source.json",
             export.variants_dir(voice) / "comparison.json"]
    if voice.samples_dir.exists():
        files += sorted(voice.samples_dir.glob("*.wav"))
    if export.variants_dir(voice).exists():
        files += sorted(export.variants_dir(voice).glob("*/source.json"))
    train_files = [voice.training_dir / "metrics.csv", *voice.training_dir.glob("lightning_logs/*/checkpoints/*.ckpt")]
    return _mtimes(files), _mtimes(train_files)


ACTION_COUNT = 7  # buttons that start a job; disabled while one runs


def poll(name: str | None):
    """Every few seconds: job panel, log, stop button, step buttons, action buttons, change signatures."""
    page, running = progress_html(name)
    voice = workspace.load(name) if name else None
    log = jobs.log_tail(voice, 15000) if voice else ""
    sig_files, sig_train = signatures(voice) if voice else ("", "")
    actions = [gr.Button(interactive=not running) for _ in range(ACTION_COUNT)]
    return [page, log, gr.Button(visible=running), *step_buttons(step_states(name), None), *actions,
            sig_files, sig_train]


# --- page ----------------------------------------------------------------------------------------------------------

def build() -> gr.Blocks:
    lang_choices = [(f"{l.name} ({l.code})", l.code) for l in languages.LANGUAGES.values()]
    names = workspace.list_names()

    with gr.Blocks(title="voice-clone") as demo:
        current = gr.State(1)
        sig_files = gr.Textbox(visible=False)
        sig_train = gr.Textbox(visible=False)

        # --- sidebar ----------------------------------------------------------------------------------------
        with gr.Sidebar(open=True):
            gr.Markdown("## 🎙️ voice-clone")
            gr.Markdown(device_label(), elem_classes="vc-muted")
            mode = gr.Radio([("🛠️  Crea una voce", "create"), ("🔊  Genera audio", "studio")], value="create",
                            show_label=False, container=False)
            with gr.Column() as create_side:
                voice_radio = gr.Radio(choices=names, value=names[0] if names else None, label="Voci")
                new_btn = gr.Button("＋ Nuova voce", size="sm")
                gr.Markdown("**Passi**")
                step_btns = [gr.Button(f"🔒  {i + 1} · {t}", elem_classes="vc-step-btn", size="sm", interactive=False)
                             for i, t in enumerate(STEP_TITLES)]
            with gr.Column(visible=False) as studio_side:
                gr.Markdown("**Suggerimenti**\n\n- Riga vuota = nuovo paragrafo\n- `[pausa 1.5s]` = pausa precisa\n"
                            "- `[[ fonemi ]]` = pronuncia esatta\n- Le sigle si sistemano nella scheda **Pronuncia**")

        with gr.Column() as create_main:
            # --- welcome / new voice -----------------------------------------------------------------------------
            with gr.Column(visible=not names) as welcome:
                gr.Markdown("# Benvenuto\nCon **voice-clone** trasformi poche registrazioni di una voce in un modello "
                            "Piper leggero, che parla in tempo reale anche senza GPU.\n\n"
                            "1. **Audio**: carichi le registrazioni\n2. **Riferimento**: controlli la trascrizione\n"
                            "3. **Dataset**: Drama 3 legge un migliaio di frasi con quella voce\n"
                            "4. **Addestramento**: Piper impara la voce (qualche ora)\n"
                            "5. **Esporta e prova**: ascolti il risultato e scarichi il modello\n\n"
                            "Per iniziare: **＋ Nuova voce** nella barra a sinistra.")
            with gr.Column(visible=False) as new_panel:
                gr.Markdown("# Nuova voce")
                new_name = gr.Textbox(label="Nome", placeholder="es. anna",
                                      info="Lettere minuscole, numeri, - e _. Diventa il nome della cartella e del "
                                           "modello.")
                new_lang = gr.Dropdown(choices=lang_choices, value="it", label="Lingua",
                                       info="La lingua in cui la voce parlerà.")
                with gr.Row():
                    create_btn = gr.Button("Crea la voce", variant="primary")
                    cancel_btn = gr.Button("Annulla")

            # --- step 1: audio -------------------------------------------------------------------------------------
            with gr.Column(visible=False) as panel1:
                gr.Markdown("# Passo 1 di 5 · Audio\nCarica una o più registrazioni della voce da clonare, in "
                            "qualsiasi formato. Meglio se: **un solo parlante, parlato pulito, niente musica né "
                            "rumore**.")
                upload = gr.File(file_count="multiple", type="filepath", label="Trascina qui i file audio")
                audio_summary = gr.Markdown()
                inputs_df = gr.Dataframe(headers=["file", "durata"], interactive=False, label="Registrazioni",
                                         wrap=True)
                with gr.Accordion("Rimuovi una registrazione", open=False):
                    with gr.Row():
                        remove_dd = gr.Dropdown(choices=[], label="File", scale=3)
                        remove_btn = gr.Button("Rimuovi", variant="stop", scale=1)
                next1 = gr.Button("Avanti →", variant="primary")

            # --- step 2: reference ---------------------------------------------------------------------------------
            with gr.Column(visible=False) as panel2:
                gr.Markdown("# Passo 2 di 5 · Riferimento\nLe registrazioni vengono unite, tagliate in una pausa e "
                            "trascritte: da qui nasce la voce che Drama 3 imiterà.")
                with gr.Accordion("Opzioni avanzate", open=False):
                    max_seconds = gr.Slider(6, 60, value=30, step=1, label="Durata massima del riferimento (secondi)",
                                            info="Drama 3 ascolta tutto il riferimento a ogni frase: 20–30 s bastano.")
                ref_btn = gr.Button("Prepara il riferimento", variant="primary")
                ref_summary = gr.Markdown()
                ref_audio = gr.Audio(label="Riferimento", interactive=False)
                transcript = gr.Textbox(label="Trascrizione", lines=4,
                                        info="Controllala ascoltando il riferimento: Drama 3 ha bisogno delle parole "
                                             "esatte. Correggi nomi e sigle, poi salva.")
                save_btn = gr.Button("Salva la trascrizione e ricrea la voce")
                gr.Markdown("### Prova la voce clonata")
                with gr.Row():
                    try_text = gr.Textbox(show_label=False, placeholder="Scrivi una frase da far leggere a Drama 3",
                                          scale=4)
                    try_btn = gr.Button("Ascolta", scale=1)
                try_audio = gr.Audio(label="Prova", interactive=False)
                next2 = gr.Button("Avanti →", variant="primary")

            # --- step 3: dataset -----------------------------------------------------------------------------------
            with gr.Column(visible=False) as panel3:
                gr.Markdown("# Passo 3 di 5 · Dataset\nDrama 3 legge molte frasi con la voce clonata; ogni clip viene "
                            "controllata con il riconoscimento vocale e quelle sbagliate vengono rifatte.")
                count = gr.Number(value=1000, precision=0, minimum=10, label="Numero di frasi",
                                  info="1000 frasi ≈ un'ora di parlato: un buon compromesso.")
                estimate = gr.Markdown()
                extra = gr.File(type="filepath", file_types=[".txt"],
                                label="Frasi tue (facoltativo): un file .txt con una frase per riga, incluse sempre")
                with gr.Accordion("Opzioni avanzate", open=False):
                    max_cer = gr.Number(value=0.05, label="Errore massimo per clip (0.05 = 5% dei caratteri)")
                    attempts = gr.Number(value=4, precision=0, label="Tentativi per frase")
                ds_btn = gr.Button("Genera il dataset", variant="primary")
                ds_summary = gr.Markdown()
                clips_btn = gr.Button("🔊 Ascolta 4 clip a caso")
                clip_players = [gr.Audio(interactive=False, visible=False) for _ in range(CLIP_SLOTS)]
                next3 = gr.Button("Avanti →", variant="primary")

            # --- step 4: training ----------------------------------------------------------------------------------
            with gr.Column(visible=False) as panel4:
                gr.Markdown("# Passo 4 di 5 · Addestramento\nPiper impara la voce partendo da una voce già addestrata "
                            "nella stessa lingua. Il computer resta acceso finché lavora, e puoi chiudere questa pagina.")
                epochs = gr.Number(value=2000, precision=0, minimum=1, label="Epoche")
                train_info = gr.Markdown()
                with gr.Accordion("Opzioni avanzate", open=False):
                    batch = gr.Number(value=0, precision=0, label="Batch size (0 = automatico in base alla GPU)")
                train_btn = gr.Button("Avvia l'addestramento", variant="primary")
                chart = gr.LinePlot(x="epoca", y="val_mel", height=280,
                                    title="Qualità durante l'addestramento (val_mel: più basso è meglio)")
                next4 = gr.Button("Avanti →", variant="primary")

            # --- step 5: export and test ---------------------------------------------------------------------------
            with gr.Column(visible=False) as panel5:
                gr.Markdown("# Passo 5 di 5 · Esporta e prova\nIl modello finale è un file `.onnx` (con il suo "
                            "`.onnx.json`) che gira con Piper anche senza GPU. Si può esportare anche a metà "
                            "addestramento, per sentire a che punto è.")
                with gr.Row(equal_height=True):
                    which = gr.Radio([("Ultimo", "last"), ("Migliore (val_mel più basso)", "best")], value="last",
                                     label="Checkpoint da esportare", scale=3)
                    export_btn = gr.Button("Esporta e prova", variant="primary", scale=1)
                export_summary = gr.Markdown()
                sample_players = [gr.Audio(interactive=False, visible=False) for _ in range(SAMPLE_SLOTS)]
                with gr.Row():
                    say_box = gr.Textbox(show_label=False, placeholder="Scrivi una frase qualsiasi da far dire al modello",
                                         scale=4)
                    say_btn = gr.Button("Leggi", scale=1)
                say_audio = gr.Audio(label="Frase", interactive=False)
                downloads = gr.File(label="Scarica il modello (.onnx + .onnx.json)", file_count="multiple",
                                    interactive=False)
                with gr.Accordion("Confronta due checkpoint", open=False):
                    gr.Markdown("Esporta due checkpoint, ascolta le stesse frasi lette da entrambi e tieni quello che "
                                "ti piace di più.")
                    with gr.Row():
                        cmp_a = gr.Dropdown(choices=[], label="A")
                        cmp_b = gr.Dropdown(choices=[], label="B")
                    compare_btn = gr.Button("Confronta", variant="primary")
                    compare_text = gr.Markdown()
                    with gr.Row():
                        with gr.Column():
                            players_a = [gr.Audio(interactive=False, visible=False) for _ in range(SAMPLE_SLOTS)]
                            use_a = gr.Button("Usa A come modello")
                        with gr.Column():
                            players_b = [gr.Audio(interactive=False, visible=False) for _ in range(SAMPLE_SLOTS)]
                            use_b = gr.Button("Usa B come modello")

            # --- job panel -----------------------------------------------------------------------------------------
            gr.Markdown("---")
            with gr.Row(equal_height=False):
                job_html = gr.HTML(scale=6)
                stop_btn = gr.Button("■ Ferma", variant="stop", visible=False, scale=0, min_width=120, size="sm")
            with gr.Accordion("Mostra il log", open=False):
                log_box = gr.Textbox(show_label=False, lines=14, max_lines=14, autoscroll=True, interactive=False)

        studio_main, (studio_show, studio_inputs, studio_outputs) = web_studio.build()

        # --- wiring --------------------------------------------------------------------------------------------
        panels = [panel1, panel2, panel3, panel4, panel5]
        nav_outputs = [current, *panels, *step_btns, welcome, new_panel]
        actions = [ref_btn, save_btn, try_btn, ds_btn, train_btn, export_btn, compare_btn]
        assert len(actions) == ACTION_COUNT

        audio_outputs = [inputs_df, audio_summary, remove_dd]
        ref_outputs = [ref_audio, transcript, try_audio, ref_summary]
        export_outputs = [export_summary, *sample_players, downloads, cmp_a, cmp_b, compare_text, *players_a,
                          *players_b]
        panel_outputs = [*audio_outputs, *ref_outputs, ds_summary, *export_outputs]
        train_outputs = [chart, train_info, train_btn]

        def refresh_panels(name):
            return (*audio_view(name), *reference_view(name), dataset_view(name), *export_view(name))

        # Selecting a voice: navigate at once, then load the panels' content (which never changes the step, so a
        # step clicked meanwhile stays shown).
        def open_voice(name):
            return show(name, pipeline.next_step(step_states(name)) if name else 1)

        def load_content(name, epochs_value, count_value):
            return (*refresh_panels(name), *training_view(name, epochs_value), dataset_estimate(name, count_value))

        content_outputs = [*panel_outputs, *train_outputs, estimate]
        for event in (voice_radio.change, demo.load):
            event(open_voice, voice_radio, nav_outputs, show_progress="hidden").then(
                load_content, [voice_radio, epochs, count], content_outputs, show_progress="hidden")

        for i, btn in enumerate(step_btns):
            btn.click(lambda name, n=i + 1: show(name, n), voice_radio, nav_outputs, show_progress="hidden")
        for btn in (next1, next2, next3, next4):
            btn.click(lambda name, step: show(name, min(5, step + 1)), [voice_radio, current], nav_outputs,
                      show_progress="hidden")

        new_btn.click(lambda: [gr.Column(visible=False)] * 6 + [gr.Column(visible=True)], None,
                      [*panels, welcome, new_panel])
        cancel_btn.click(show, [voice_radio, current], nav_outputs)

        def create(name, lang):
            try:
                voice = workspace.create((name or "").strip(), lang)
            except UserError as exc:
                raise gr.Error(str(exc)) from exc
            gr.Info(f"Voce {voice.name} creata.")
            return gr.Radio(choices=workspace.list_names(), value=voice.name), ""

        create_btn.click(create, [new_name, new_lang], [voice_radio, new_name])

        timer = gr.Timer(POLL_SECONDS)
        poll_outputs = [job_html, log_box, stop_btn, *step_btns, *actions, sig_files, sig_train]
        timer.tick(poll, voice_radio, poll_outputs, show_progress="hidden")

        def stop(name):
            if jobs.stop(_voice(name)):
                gr.Info("Fermato: avvia di nuovo il passo per riprendere da dove era arrivato.")

        stop_btn.click(stop, voice_radio, None)

        sig_files.change(refresh_panels, voice_radio, panel_outputs, show_progress="hidden")
        sig_train.change(training_view, [voice_radio, epochs], train_outputs, show_progress="hidden")
        epochs.change(training_view, [voice_radio, epochs], train_outputs)
        count.change(dataset_estimate, [voice_radio, count], estimate)

        upload.upload(add_files, [voice_radio, upload], [*audio_outputs, upload])
        remove_btn.click(remove_file, [voice_radio, remove_dd], audio_outputs)

        ref_btn.click(lambda n, s: _start(n, ["reference", n, "--max-seconds", str(s)]), [voice_radio, max_seconds])
        save_btn.click(save_transcript, [voice_radio, transcript, max_seconds])
        try_btn.click(try_voice, [voice_radio, try_text])

        ds_btn.click(start_dataset, [voice_radio, count, extra, max_cer, attempts])
        clips_btn.click(random_clips, voice_radio, clip_players)

        train_btn.click(start_train, [voice_radio, epochs, batch])

        export_btn.click(lambda n, w: _start(n, ["export", n, "--checkpoint", w]), [voice_radio, which])
        say_btn.click(say_text, [voice_radio, say_box], say_audio)
        compare_btn.click(start_compare, [voice_radio, cmp_a, cmp_b])
        use_a.click(lambda n: promote(n, "a"), voice_radio).then(refresh_panels, voice_radio, panel_outputs)
        use_b.click(lambda n: promote(n, "b"), voice_radio).then(refresh_panels, voice_radio, panel_outputs)
        def switch(which):
            studio = which == "studio"
            return (gr.Column(visible=not studio), gr.Column(visible=studio), gr.Column(visible=not studio),
                    gr.Column(visible=studio))

        mode.change(switch, mode, [create_side, studio_side, create_main, studio_main], show_progress="hidden").then(
            studio_show, studio_inputs, studio_outputs, show_progress="hidden")
    return demo


def serve(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    from voice_clone.paths import VOICES_DIR

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    build().queue().launch(server_name=host, server_port=port, share=share, allowed_paths=[str(VOICES_DIR)],
                           inbrowser=False, css=CSS)
