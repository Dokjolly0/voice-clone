"""`voice-clone web`: a local web page for the whole workflow.

Long stages run as background jobs (see jobs.py), exactly as `voice-clone <command> --background` would, so they
keep going when the page is closed and the CLI sees the same state. The page polls status and log every few seconds.
"""

import random
import shutil
from pathlib import Path

import gradio as gr

from voice_clone import audio, jobs, languages, pipeline, reference, workspace
from voice_clone.util import UserError

SAMPLE_SLOTS = 5
CLIP_SLOTS = 4


def _voice(name: str | None) -> workspace.Voice:
    if not name:
        raise gr.Error("Create or select a voice first.")
    try:
        return workspace.load(name)
    except UserError as exc:
        raise gr.Error(str(exc)) from exc


def _start(name: str, args: list[str]) -> str:
    voice = _voice(name)
    try:
        jobs.start(voice, args)
    except UserError as exc:
        raise gr.Error(str(exc)) from exc
    return f"Started: voice-clone {' '.join(args)}"


def _file(path: Path) -> str | None:
    return str(path) if path.exists() else None


# --- voice selection ---------------------------------------------------------------------------------------------

def voice_choices():
    return workspace.list_names()


def create_voice(name: str, lang_label: str):
    code = lang_label.split(" ", 1)[0]
    try:
        voice = workspace.create(name.strip(), code)
    except UserError as exc:
        raise gr.Error(str(exc)) from exc
    return gr.Dropdown(choices=voice_choices(), value=voice.name), f"Created {voice.name}."


# --- polling -------------------------------------------------------------------------------------------------------

def files_signature(voice: workspace.Voice) -> str:
    """Changes when an audio file the page shows changes, so players reload only then (not on every epoch)."""
    p = reference.paths(voice)
    watched = [p["reference"], p["transcript"], voice.reference_dir / "try.wav", voice.onnx_path]
    if voice.samples_dir.exists():
        watched += sorted(voice.samples_dir.glob("sample-*.wav"))
    return "|".join(f"{f.name}:{f.stat().st_mtime_ns}" for f in watched if f.exists())


def refresh(name: str | None):
    if not name:
        return "No voice selected.", "", ""
    try:
        voice = workspace.load(name)
        status = pipeline.format_status(pipeline.status(voice))
    except UserError as exc:
        return str(exc), "", ""
    return f"```\n{status}\n```", jobs.log_tail(voice, 12000), files_signature(voice)


def stop_job(name: str):
    voice = _voice(name)
    if not jobs.current(voice):
        return "Nothing is running."
    return "Stopped (run the stage again to resume)." if jobs.stop(voice) else "Could not stop the job."


# --- 1. audio --------------------------------------------------------------------------------------------------------

def inputs_table(name: str | None):
    if not name:
        return []
    voice = workspace.load(name)
    return [[p.name, round(audio.duration(p), 1)] for p in voice.inputs()]


def add_files(name: str, files: list[str] | None):
    voice = _voice(name)
    if not files:
        raise gr.Error("Choose one or more audio files first.")
    try:
        workspace.add_audio(voice, [Path(f) for f in files])
    except UserError as exc:
        raise gr.Error(str(exc)) from exc
    names = [p.name for p in voice.inputs()]
    return inputs_table(name), gr.Dropdown(choices=names, value=None), None


def remove_file(name: str, filename: str | None):
    voice = _voice(name)
    if filename:
        workspace.remove_audio(voice, [filename])
    names = [p.name for p in voice.inputs()]
    return inputs_table(name), gr.Dropdown(choices=names, value=None)


# --- 2. reference ----------------------------------------------------------------------------------------------------

def reference_view(name: str | None):
    if not name:
        return None, "", None
    voice = workspace.load(name)
    p = reference.paths(voice)
    transcript = p["transcript"].read_text(encoding="utf-8").strip() if p["transcript"].exists() else ""
    return _file(p["reference"]), transcript, _file(voice.reference_dir / "try.wav")


def save_transcript(name: str, text: str, max_seconds: float):
    voice = _voice(name)
    p = reference.paths(voice)
    if not p["reference"].exists():
        raise gr.Error("Build the reference first.")
    if not text.strip():
        raise gr.Error("The transcript is empty.")
    p["transcript"].write_text(text.strip() + "\n", encoding="utf-8")
    return _start(name, ["reference", name, "--max-seconds", str(max_seconds)])


# --- 3. dataset ------------------------------------------------------------------------------------------------------

def start_dataset(name: str, count: float, extra: str | None, max_cer: float, attempts: float):
    voice = _voice(name)
    args = ["dataset", name, "--sentences", str(int(count)), "--max-cer", str(max_cer),
            "--max-attempts", str(int(attempts))]
    if extra:
        domain = voice.dir / "domain-sentences.txt"
        shutil.copy(extra, domain)
        args += ["--extra", str(domain)]
    return _start(name, args)


def random_clips(name: str | None):
    if not name:
        return [None] * CLIP_SLOTS
    voice = workspace.load(name)
    clips = sorted((voice.dataset_dir / "wav").glob("*.wav")) if (voice.dataset_dir / "wav").exists() else []
    picked = random.sample(clips, min(CLIP_SLOTS, len(clips)))
    sentences = voice.sentences()
    updates = []
    for path in picked:
        idx = int(path.stem) if path.stem.isdigit() else -1
        label = sentences[idx] if 0 <= idx < len(sentences) else path.stem
        updates.append(gr.Audio(value=str(path), label=label[:90]))
    return updates + [gr.Audio(value=None, label="")] * (CLIP_SLOTS - len(updates))


# --- 5. export -------------------------------------------------------------------------------------------------------

def samples_view(name: str | None):
    if not name:
        return [None] * SAMPLE_SLOTS
    voice = workspace.load(name)
    labels = {}
    listing = voice.samples_dir / "sentences.txt"
    if listing.exists():
        for line in listing.read_text(encoding="utf-8").splitlines():
            if "|" in line:
                fname, text = line.split("|", 1)
                labels[fname] = text
    files = sorted(voice.samples_dir.glob("sample-*.wav"))[:SAMPLE_SLOTS] if voice.samples_dir.exists() else []
    updates = [gr.Audio(value=str(f), label=labels.get(f.name, f.name)[:90]) for f in files]
    return updates + [gr.Audio(value=None, label="")] * (SAMPLE_SLOTS - len(updates))


def say_text(name: str, text: str):
    from voice_clone import export

    voice = _voice(name)
    if not text.strip():
        raise gr.Error("Type something to say.")
    try:
        return str(export.say(voice, text.strip(), voice.samples_dir / "say.wav"))
    except UserError as exc:
        raise gr.Error(str(exc)) from exc


def model_files(name: str | None):
    if not name:
        return None
    voice = workspace.load(name)
    files = [str(p) for p in (voice.onnx_path, voice.onnx_config_path) if p.exists()]
    return files or None


# --- page ------------------------------------------------------------------------------------------------------------

def build() -> gr.Blocks:
    lang_labels = [f"{l.code} — {l.name}" for l in languages.LANGUAGES.values()]
    names = voice_choices()

    with gr.Blocks(title="voice-clone") as demo:
        gr.Markdown("# voice-clone\nClone a voice into a lightweight Piper model: "
                    "**audio → reference → dataset → training → export**.")
        with gr.Row():
            voice_dd = gr.Dropdown(choices=names, value=names[0] if names else None, label="Voice", scale=3)
            with gr.Accordion("New voice", open=not names):
                new_name = gr.Textbox(label="Name", placeholder="e.g. anna")
                new_lang = gr.Dropdown(choices=lang_labels, value=lang_labels[0], label="Language")
                new_btn = gr.Button("Create")
                new_msg = gr.Markdown()

        with gr.Row():
            with gr.Column(scale=2):
                status_md = gr.Markdown()
            with gr.Column(scale=3):
                log_box = gr.Textbox(label="Job log", lines=12, max_lines=12, autoscroll=True, interactive=False)
                with gr.Row():
                    stop_btn = gr.Button("Stop job", variant="stop")
                    job_msg = gr.Markdown()

        with gr.Tabs():
            with gr.Tab("1 · Audio"):
                gr.Markdown("Recordings of the voice to clone, in any format. Clean speech, one speaker, "
                            "no music; 10–60 s in total is plenty.")
                upload = gr.File(file_count="multiple", type="filepath", label="Audio files")
                add_btn = gr.Button("Add files", variant="primary")
                inputs_df = gr.Dataframe(headers=["file", "seconds"], interactive=False, label="Input files")
                with gr.Row():
                    remove_dd = gr.Dropdown(choices=[], label="Remove a file")
                    remove_btn = gr.Button("Remove")

            with gr.Tab("2 · Reference"):
                gr.Markdown("All input files are joined into one recording, cut at a pause to the length below, "
                            "transcribed, and turned into a Drama 3 voice. **Check the transcript**: Drama 3 "
                            "needs the exact words.")
                max_seconds = gr.Slider(6, 60, value=30, step=1, label="Reference length (seconds)")
                ref_btn = gr.Button("Build reference", variant="primary")
                ref_audio = gr.Audio(label="reference.wav", interactive=False)
                transcript = gr.Textbox(label="Transcript", lines=4)
                save_btn = gr.Button("Save transcript and rebuild voice")
                with gr.Row():
                    try_text = gr.Textbox(label="Try the cloned voice", scale=3,
                                          placeholder="A sentence to speak with Drama 3")
                    try_btn = gr.Button("Speak")
                try_audio = gr.Audio(label="try.wav", interactive=False)

            with gr.Tab("3 · Dataset"):
                gr.Markdown("Drama 3 speaks every sentence in the cloned voice; Qwen3-ASR checks each clip and "
                            "bad ones are generated again. 1000 sentences ≈ 1 hour of audio ≈ 2–3 hours of work.")
                with gr.Row():
                    count = gr.Number(value=1000, precision=0, label="Sentences")
                    max_cer = gr.Number(value=0.05, label="Max character error rate")
                    attempts = gr.Number(value=4, precision=0, label="Attempts per sentence")
                extra = gr.File(type="filepath", file_types=[".txt"],
                                label="Domain sentences (optional .txt, one per line, always included)")
                ds_btn = gr.Button("Generate dataset", variant="primary")
                clips_btn = gr.Button("Listen to random clips")
                clip_players = [gr.Audio(interactive=False, label="") for _ in range(CLIP_SLOTS)]

            with gr.Tab("4 · Training"):
                gr.Markdown("Fine-tunes Piper from a trained voice in the same language. 2000 epochs take about "
                            "5 hours on an RTX 4090. The computer is kept awake while it runs.")
                with gr.Row():
                    epochs = gr.Number(value=2000, precision=0, label="Epochs")
                    batch = gr.Number(value=0, precision=0, label="Batch size (0 = from GPU memory)")
                    resume = gr.Checkbox(value=True, label="Resume from the last checkpoint if there is one")
                train_btn = gr.Button("Train", variant="primary")

            with gr.Tab("5 · Export & test"):
                gr.Markdown("Exports a checkpoint to ONNX (works mid-training too) and speaks test sentences.")
                ckpt = gr.Radio(["last", "best"], value="last", label="Checkpoint")
                export_btn = gr.Button("Export and test", variant="primary")
                sample_players = [gr.Audio(interactive=False, label="") for _ in range(SAMPLE_SLOTS)]
                with gr.Row():
                    say_box = gr.Textbox(label="Say anything with the exported model", scale=3)
                    say_btn = gr.Button("Say")
                say_audio = gr.Audio(label="say.wav", interactive=False)
                downloads = gr.File(label="Model files (.onnx + .onnx.json)", file_count="multiple",
                                    interactive=False)

        # --- wiring --------------------------------------------------------------------------------------------
        def load_voice(name):
            names = [p.name for p in workspace.load(name).inputs()] if name else []
            return (inputs_table(name), gr.Dropdown(choices=names, value=None), *reference_view(name),
                    *samples_view(name), model_files(name))

        voice_outputs = [inputs_df, remove_dd, ref_audio, transcript, try_audio, *sample_players, downloads]
        voice_dd.change(load_voice, voice_dd, voice_outputs)
        demo.load(load_voice, voice_dd, voice_outputs)
        new_btn.click(create_voice, [new_name, new_lang], [voice_dd, new_msg])

        signature = gr.Textbox(visible=False)
        timer = gr.Timer(4)
        timer.tick(refresh, voice_dd, [status_md, log_box, signature])
        demo.load(refresh, voice_dd, [status_md, log_box, signature])
        stop_btn.click(stop_job, voice_dd, job_msg)

        add_btn.click(add_files, [voice_dd, upload], [inputs_df, remove_dd, upload])
        remove_btn.click(remove_file, [voice_dd, remove_dd], [inputs_df, remove_dd])

        ref_btn.click(lambda n, s: _start(n, ["reference", n, "--max-seconds", str(s)]), [voice_dd, max_seconds],
                      job_msg)
        save_btn.click(save_transcript, [voice_dd, transcript, max_seconds], job_msg)
        try_btn.click(lambda n, t: _start(n, ["try-voice", n, t]) if t.strip() else "Type a sentence first.",
                      [voice_dd, try_text], job_msg)
        signature.change(reference_view, voice_dd, [ref_audio, transcript, try_audio])

        ds_btn.click(start_dataset, [voice_dd, count, extra, max_cer, attempts], job_msg)
        clips_btn.click(random_clips, voice_dd, clip_players)

        def start_train(n, e, b, r):
            args = ["train", n, "--epochs", str(int(e)), "--batch-size", str(int(b))]
            voice = _voice(n)
            from voice_clone import train

            if r and train.last_checkpoint(voice):
                args.append("--resume")
            return _start(n, args)

        train_btn.click(start_train, [voice_dd, epochs, batch, resume], job_msg)

        export_btn.click(lambda n, c: _start(n, ["export", n, "--checkpoint", c]), [voice_dd, ckpt], job_msg)
        signature.change(lambda n: (*samples_view(n), model_files(n)), voice_dd, [*sample_players, downloads])
        say_btn.click(say_text, [voice_dd, say_box], say_audio)
    return demo


def serve(host: str = "127.0.0.1", port: int = 7860, share: bool = False) -> None:
    from voice_clone.paths import VOICES_DIR

    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    build().queue().launch(server_name=host, server_port=port, share=share, allowed_paths=[str(VOICES_DIR)],
                           inbrowser=False)
