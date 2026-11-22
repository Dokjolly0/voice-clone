"""Pronunciation dictionary: "written → spoken" rules per language (studio/lexicon/<lang>.json).

A rule replaces text before synthesis: `ACI → A C I`, `PinPad → pin pad`, or Piper's raw phonemes
`Coop → [[ kˈɔp ]]`. Rules apply to whole words unless told otherwise, ignore case unless told otherwise, the longest
first (so "Socio ACI" wins over "ACI"), and never inside [[ phonemes ]] or [pausa ...] markers.
"""

import re

from voice_clone.paths import STUDIO_DIR
from voice_clone.util import UserError, read_json, write_json

# Pieces of text rules must not touch: raw phonemes and pause markers.
PROTECTED = re.compile(r"\[\[.*?\]\]|\[\s*(?:pausa|pause|break)\b[^\]]*\]", re.IGNORECASE | re.DOTALL)


def path(lang: str):
    if not re.fullmatch(r"[a-z]{2,3}", lang or ""):
        raise UserError(f"language code like 'it' or 'en' expected, got {lang!r}")
    return STUDIO_DIR / "lexicon" / f"{lang}.json"


def rules(lang: str) -> list[dict]:
    return read_json(path(lang), {"rules": []})["rules"]


def save(lang: str, rule_list: list[dict]) -> None:
    write_json(path(lang), {"rules": rule_list})


def add(lang: str, written: str, spoken: str, whole_word: bool = True, match_case: bool = False) -> None:
    written, spoken = written.strip(), spoken.strip()
    if not written or not spoken:
        raise UserError("both the written form and how to say it are needed")
    kept = [r for r in rules(lang) if r["written"] != written]
    kept.append({"written": written, "spoken": spoken, "whole_word": whole_word, "match_case": match_case})
    save(lang, kept)


def remove(lang: str, written: str) -> None:
    current = rules(lang)
    kept = [r for r in current if r["written"] != written]
    if len(kept) == len(current):
        raise UserError(f"no rule for {written!r} in the {lang} dictionary")
    save(lang, kept)


def _pattern(rule: dict) -> re.Pattern:
    body = re.escape(rule["written"])
    if rule.get("whole_word", True):
        body = rf"(?<!\w){body}(?!\w)"
    return re.compile(body, 0 if rule.get("match_case") else re.IGNORECASE)


def apply_rules(text: str, rule_list: list[dict]) -> str:
    ordered = sorted(rule_list, key=lambda r: -len(r["written"]))
    patterns = [(_pattern(r), r["spoken"]) for r in ordered]

    def plain(segment: str) -> str:
        # One pass per rule, each over text the earlier (longer) rules left alone.
        pieces = [(segment, False)]
        for pattern, spoken in patterns:
            next_pieces = []
            for piece, done in pieces:
                if done:
                    next_pieces.append((piece, True))
                    continue
                pos = 0
                for m in pattern.finditer(piece):
                    next_pieces.append((piece[pos:m.start()], False))
                    next_pieces.append((spoken, True))
                    pos = m.end()
                next_pieces.append((piece[pos:], False))
            pieces = next_pieces
        return "".join(p for p, _ in pieces)

    out, pos = [], 0
    for m in PROTECTED.finditer(text):
        out.append(plain(text[pos:m.start()]))
        out.append(m.group(0))
        pos = m.end()
    out.append(plain(text[pos:]))
    return "".join(out)


def apply(text: str, lang: str) -> str:
    """The text with the language's rules applied (unchanged when there are none)."""
    try:
        found = rules(lang)
    except UserError:
        return text
    return apply_rules(text, found) if found else text
