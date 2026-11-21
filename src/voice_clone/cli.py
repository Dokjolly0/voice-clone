"""Command line interface: `voice-clone <command>`."""

import argparse
import sys
from pathlib import Path

from voice_clone import __version__
from voice_clone.util import UserError


def cmd_setup(args):
    from voice_clone import installer

    installer.setup(skip_models=args.skip_models)


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
    from voice_clone import dataset, sentences, workspace

    voice = workspace.load(args.name)
    sentences.choose(voice, args.sentences, extra=args.extra)
    dataset.build(voice, dataset_settings(args))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-clone",
        description="Clone a voice into a lightweight Piper TTS model.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("setup", help="install speech.cpp, the models and Piper; check the system")
    p.add_argument("--skip-models", action="store_true", help="do not download Drama 3 and Qwen3-ASR now")
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
    return parser


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
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        sys.exit(1)
    try:
        args.func(args)
    except UserError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\ninterrupted; run the same command again to resume.", file=sys.stderr)
        sys.exit(130)
