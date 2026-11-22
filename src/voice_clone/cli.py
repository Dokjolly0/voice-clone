"""Command line interface: `voice-clone <command>`."""

import argparse
import sys
from pathlib import Path

from voice_clone import __version__
from voice_clone.util import UserError


def cmd_setup(args):
    from voice_clone import installer

    installer.setup(skip_models=args.skip_models, device=args.device)


def cmd_doctor(args):
    from voice_clone import installer

    installer.doctor()


def cmd_new(args):
    from voice_clone import workspace

    voice = workspace.create(args.name, args.lang)
    print(f"Created voice {voice.name!r} ({voice.language.name}) in {voice.dir}")
    print(f"Next: voice-clone add-audio {voice.name} <recordings...>")


def cmd_list(args):
    from voice_clone import workspace

    names = workspace.list_names()
    if not names:
        print("No voices yet: voice-clone new <name> --lang <language>")
    for name in names:
        voice = workspace.load(name)
        print(f"{name}  ({voice.language.name}, {len(voice.inputs())} input files)")


def cmd_languages(args):
    from voice_clone.languages import LANGUAGES

    for lang in LANGUAGES.values():
        print(f"{lang.code:4} {lang.name:22} base voice: {lang.checkpoint}")


def cmd_add_audio(args):
    from voice_clone import audio, workspace

    voice = workspace.load(args.name)
    workspace.add_audio(voice, args.files)
    total = sum(audio.duration(p) for p in voice.inputs())
    print(f"{len(voice.inputs())} input files, {total:.1f} s in total")


def cmd_remove_audio(args):
    from voice_clone import workspace

    workspace.remove_audio(workspace.load(args.name), args.files)


def cmd_reference(args):
    from voice_clone import reference, workspace

    reference.build(workspace.load(args.name), max_seconds=args.max_seconds, retranscribe=args.retranscribe)


def cmd_try_voice(args):
    from voice_clone import reference, workspace

    voice = workspace.load(args.name)
    out = args.output or voice.reference_dir / "try.wav"
    reference.try_voice(voice, args.text, out, seed=args.seed)


def cmd_sentences(args):
    from voice_clone import sentences, workspace

    sentences.choose(workspace.load(args.name), args.sentences, extra=args.extra)


def dataset_settings(args):
    from voice_clone.dataset import Settings

    return Settings(max_attempts=args.max_attempts, max_cer=args.max_cer, limit=args.limit)


def cmd_dataset(args):
    from voice_clone import dataset, inhibit, sentences, workspace

    voice = workspace.load(args.name)
    inhibit.keep_awake(f"voice-clone: generating the dataset of {voice.name}")
    sentences.choose(voice, args.sentences, extra=args.extra)
    dataset.build(voice, dataset_settings(args))


def train_settings(args):
    from voice_clone.train import Settings

    return Settings(epochs=args.epochs, batch_size=args.batch_size, num_workers=args.num_workers,
                    resume=getattr(args, "resume", False))


def cmd_train(args):
    from voice_clone import inhibit, train, workspace

    voice = workspace.load(args.name)
    inhibit.keep_awake(f"voice-clone: training {voice.name}")
    train.run(voice, train_settings(args))


def cmd_export(args):
    from voice_clone import export, workspace

    export.export_and_test(workspace.load(args.name), checkpoint=args.checkpoint, extra=args.sample)


def cmd_say(args):
    from voice_clone import export, workspace

    out = export.say(workspace.load(args.name), args.text, args.output)
    print(out)


def cmd_compare(args):
    from voice_clone import export, workspace

    export.compare(workspace.load(args.name), args.first, args.second)


def cmd_promote(args):
    from voice_clone import export, workspace

    export.promote(workspace.load(args.name), args.label)


def cmd_checkpoints(args):
    from voice_clone import export, workspace

    for c in export.checkpoints(workspace.load(args.name)):
        epoch = f"(epoch-{c.epoch})" if c.label == "last" and c.epoch is not None else ""
        score = f"val_mel {c.val_mel:.4f}" if c.val_mel is not None else ""
        print(f"{c.label:12} {epoch:12} {score}")


def cmd_status(args):
    import json

    from voice_clone import pipeline, workspace

    names = [args.name] if args.name else workspace.list_names()
    if not names:
        print("No voices yet: voice-clone new <name> --lang <language>")
    for name in names:
        s = pipeline.status(workspace.load(name))
        print(json.dumps(s, indent=1, default=str) if args.json else pipeline.format_status(s))


def cmd_run(args):
    from voice_clone import inhibit, pipeline, workspace

    voice = workspace.load(args.name)
    inhibit.keep_awake(f"voice-clone: building {voice.name}")
    pipeline.run(voice, max_seconds=args.max_seconds, sentence_count=args.sentences, extra=args.extra,
                 dataset_settings=dataset_settings(args), train_settings=train_settings(args),
                 checkpoint=args.checkpoint)


def cmd_stop(args):
    from voice_clone import jobs, workspace

    voice = workspace.load(args.name)
    print("stopping..." if jobs.current(voice) else "nothing is running")
    if jobs.stop(voice):
        print("stopped; run the same command again to resume")


def cmd_log(args):
    import time

    from voice_clone import jobs, workspace

    voice = workspace.load(args.name)
    path = voice.logs_dir / "job.log"
    if not path.exists():
        print("no background job has run yet")
        return
    if not args.follow:
        print(jobs.log_tail(voice, 200_000 if args.all else 8000))
        return
    with open(path, "rb") as f:
        f.seek(max(0, path.stat().st_size - 8000))
        while True:
            chunk = f.read()
            if chunk:
                sys.stdout.write(chunk.decode(errors="replace"))
                sys.stdout.flush()
            elif not jobs.current(voice):
                break
            else:
                time.sleep(1)


def cmd_web(args):
    from voice_clone import web

    web.serve(host=args.host, port=args.port, share=args.share)


def cmd_models(args):
    from voice_clone import models

    if args.action == "list":
        found = models.list_models()
        if not found:
            print("No models yet: train a voice, `voice-clone models catalog --lang it`, or `models import`.")
        for m in found:
            speakers = f", {len(m.speakers)} speakers" if m.speakers else ""
            print(f"{m.id:40} {m.language:6} {m.quality:7} {models.ORIGIN_LABELS[m.origin]}{speakers}")
    elif args.action == "import":
        models.import_files(args.files[0], args.files[1] if len(args.files) > 1 else None, name=args.name)
    elif args.action == "catalog":
        rows = models.search(args.lang, args.quality)
        for r in rows:
            mark = "installed" if r["installed"] else ""
            print(f"{r['key']:36} {r['language_name']:12} {r['quality']:7} {r['speakers']:>4} spk "
                  f"{r['size_mb']:>4} MB  {mark}")
        print(f"{len(rows)} voices" + ("" if args.lang else "; filter with --lang it"))
    elif args.action == "download":
        for key in args.keys:
            models.download_catalog(key)
    elif args.action == "remove":
        for model_id in args.ids:
            models.remove(model_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-clone",
        description="Clone a voice into a lightweight Piper TTS model.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("setup", help="install speech.cpp, the models and Piper; check the system")
    p.add_argument("--skip-models", action="store_true", help="do not download Drama 3 and Qwen3-ASR now")
    p.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto",
                   help="gpu (NVIDIA, fast), cpu (no GPU, much slower) or auto (gpu if an NVIDIA driver works)")
    p.set_defaults(func=cmd_setup)

    p = sub.add_parser("doctor", help="report what is installed, without changing anything")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("languages", help="list supported languages")
    p.set_defaults(func=cmd_languages)

    p = sub.add_parser("new", help="create a voice workspace")
    p.add_argument("name", help="voice name, e.g. anna (lowercase, digits, - and _)")
    p.add_argument("--lang", required=True, help="language code, see `voice-clone languages`")
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("list", help="list voices")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("add-audio", help="add recordings of the voice to clone (any format ffmpeg reads)")
    p.add_argument("name")
    p.add_argument("files", nargs="+", type=Path)
    p.set_defaults(func=cmd_add_audio)

    p = sub.add_parser("remove-audio", help="remove input recordings by file name")
    p.add_argument("name")
    p.add_argument("files", nargs="+")
    p.set_defaults(func=cmd_remove_audio)

    p = sub.add_parser("reference", help="merge the inputs, transcribe them and build the Drama 3 voice")
    p.add_argument("name")
    p.add_argument("--max-seconds", type=float, default=30.0,
                   help="longest reference Drama 3 hears, cut at a pause (default 30)")
    p.add_argument("--retranscribe", action="store_true", help="discard transcript.txt and recognize it again")
    p.set_defaults(func=cmd_reference)

    p = sub.add_parser("try-voice", help="speak one sentence with Drama 3, to check the cloned voice")
    p.add_argument("name")
    p.add_argument("text")
    p.add_argument("-o", "--output", type=Path, help="output WAV (default reference/try.wav)")
    p.add_argument("--seed", type=int, default=1)
    p.set_defaults(func=cmd_try_voice)

    p = sub.add_parser("sentences", help="choose the dataset sentences (Common Voice + optional domain file)")
    p.add_argument("name")
    add_sentence_args(p)
    p.set_defaults(func=cmd_sentences)

    p = sub.add_parser("dataset", help="generate the dataset with Drama 3, verify it with Qwen3-ASR, export it")
    p.add_argument("name")
    add_sentence_args(p)
    add_dataset_args(p)
    p.set_defaults(func=cmd_dataset)

    p = sub.add_parser("train", help="fine-tune Piper on the dataset (keeps the computer awake)")
    p.add_argument("name")
    add_train_args(p)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("export", help="export a checkpoint to ONNX and speak test sentences into samples/")
    p.add_argument("name")
    add_export_args(p)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("say", help="speak text with the exported model")
    p.add_argument("name")
    p.add_argument("text")
    p.add_argument("-o", "--output", type=Path, default=Path("say.wav"), help="output WAV (default say.wav)")
    p.set_defaults(func=cmd_say)

    p = sub.add_parser("checkpoints", help="list the checkpoints kept so far")
    p.add_argument("name")
    p.set_defaults(func=cmd_checkpoints)

    p = sub.add_parser("compare", help="export two checkpoints with the same test sentences, to choose by ear")
    p.add_argument("name")
    p.add_argument("first", help="'last', 'best', an epoch number or a .ckpt path")
    p.add_argument("second")
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser("promote", help="make a compared checkpoint (e.g. epoch-1816) the voice's model")
    p.add_argument("name")
    p.add_argument("label")
    p.set_defaults(func=cmd_promote)

    p = sub.add_parser("models", help="the studio's models: list, import, browse and download Piper's catalog")
    msub = p.add_subparsers(dest="action", required=True, metavar="<action>")
    msub.add_parser("list", help="every model the studio can use")
    q = msub.add_parser("import", help="add a Piper model: FILE.onnx [FILE.onnx.json]")
    q.add_argument("files", nargs="+", type=Path)
    q.add_argument("--name", help="name in the library (default: the file name)")
    q = msub.add_parser("catalog", help="browse Piper's official voices")
    q.add_argument("--lang", help="language, e.g. it or it_IT")
    q.add_argument("--quality", choices=["x_low", "low", "medium", "high"])
    q = msub.add_parser("download", help="download voices from the catalog by key, e.g. it_IT-paola-medium")
    q.add_argument("keys", nargs="+")
    q = msub.add_parser("remove", help="remove downloaded or imported models by id")
    q.add_argument("ids", nargs="+")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("run", help="every stage in order (reference, dataset, train, export), skipping done ones")
    p.add_argument("name")
    p.add_argument("--max-seconds", type=float, default=30.0, help="longest reference (default 30)")
    add_sentence_args(p)
    add_dataset_args(p)
    add_train_args(p, resume=False)
    add_export_args(p)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("status", help="where each stage of a voice (or of every voice) stands")
    p.add_argument("name", nargs="?")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("log", help="show the log of the background job")
    p.add_argument("name")
    p.add_argument("-f", "--follow", action="store_true", help="keep printing until the job ends")
    p.add_argument("--all", action="store_true", help="show more of the log")
    p.set_defaults(func=cmd_log)

    p = sub.add_parser("stop", help="stop the background job cleanly (it can be resumed)")
    p.add_argument("name")
    p.set_defaults(func=cmd_stop)

    p = sub.add_parser("web", help="open the web interface (http://127.0.0.1:7860)")
    p.add_argument("--host", default="127.0.0.1", help="address to listen on (0.0.0.0 for the local network)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="also publish a temporary public gradio.live link")
    p.set_defaults(func=cmd_web)

    for name in BACKGROUND_COMMANDS:
        sub.choices[name].add_argument("--background", action="store_true",
                                       help="run detached; follow with `voice-clone log <name> -f`")
    return parser


BACKGROUND_COMMANDS = ("reference", "dataset", "train", "export", "compare", "run")


def add_export_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--checkpoint", default="last",
                   help="'last' (default), 'best' (lowest validation loss), an epoch number or a .ckpt path")
    p.add_argument("--sample", action="append", metavar="TEXT", help="extra test sentence (repeatable)")


def add_train_args(p: argparse.ArgumentParser, resume: bool = True) -> None:
    p.add_argument("--epochs", type=int, default=2000, help="total epochs (default 2000)")
    p.add_argument("--batch-size", type=int, default=0, help="default: chosen from the GPU memory (32 on 24 GB) or, in CPU mode, the RAM")
    p.add_argument("--num-workers", type=int, default=4, help="data loader processes (default 4)")
    if resume:
        p.add_argument("--resume", action="store_true", help="continue from the last checkpoint")


def add_dataset_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--max-attempts", type=int, default=4, help="generations per sentence before giving up (4)")
    p.add_argument("--max-cer", type=float, default=0.05,
                   help="largest character error rate a clip may have to be kept (0.05)")
    p.add_argument("--limit", type=int, default=0, help="generate at most N clips, for a quick test")


def add_sentence_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("-n", "--sentences", type=int, default=1000,
                   help="number of sentences in the dataset (default 1000, about 1 hour of speech)")
    p.add_argument("--extra", type=Path,
                   help="text file of domain sentences, one per line, always included first")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    argv = sys.argv[1:] if argv is None else argv
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(1)
    try:
        if getattr(args, "background", False):
            from voice_clone import jobs, workspace

            jobs.background(workspace.load(args.name), argv)
            return
        args.func(args)
    except UserError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\ninterrupted; run the same command again to resume.", file=sys.stderr)
        sys.exit(130)
