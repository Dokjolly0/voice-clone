"""Command line interface: `voice-clone <command>`."""

import argparse
import sys

from voice_clone import __version__
from voice_clone.util import UserError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="voice-clone",
        description="Clone a voice into a lightweight Piper TTS model.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.add_subparsers(dest="command", metavar="<command>")
    return parser


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
