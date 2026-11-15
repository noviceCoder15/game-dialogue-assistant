"""Command line interface for the Game Dialogue Assistant.

    python -m app.cli generate "The stars"
    python -m app.cli generate "Halt" --speaker guard --count 3 --length 20
    python -m app.cli suggest "The stars are"
    python -m app.cli speakers
    python -m app.cli evaluate
    python -m app.cli chat --speaker wizard
"""
from __future__ import annotations

import argparse
import os
import random
import sys

from . import core, llm, storage

ORDER_NAMES = {1: "unigram", 2: "bigram", 3: "trigram"}


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str):
    value = os.environ.get(name, "")
    return int(value) if value else None


def format_table(headers, rows) -> str:
    widths = [
        max([len(str(h))] + [len(str(row[i])) for row in rows])
        for i, h in enumerate(headers)
    ]
    line = "  ".join(str(h).ljust(w) for h, w in zip(headers, widths))
    rule = "  ".join("-" * w for w in widths)
    body = ["  ".join(str(c).ljust(w) for c, w in zip(row, widths)) for row in rows]
    return "\n".join([line, rule] + body)


def load_entries(args):
    return core.parse_dialogue(storage.read_corpus(args.corpus))


def load_model(args):
    """Build the model the arguments ask for. Returns (model, speaker or None)."""
    entries = load_entries(args)
    speaker = llm.resolve_speaker(entries, args.speaker) if args.speaker else None
    return llm.build_model(entries, args.order, args.k, speaker), speaker


def _snippet(model, phrase, args, rng, min_tokens=None):
    return llm.generate_snippet(
        model, phrase,
        max_tokens=args.length,
        min_tokens=args.min_length if min_tokens is None else min_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        rng=rng,
    )


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_generate(args) -> int:
    if args.count < 1:
        raise ValueError("count must be at least 1")
    if args.min_length > args.length:
        raise ValueError("min-length cannot be larger than length")
    model, speaker = load_model(args)
    rng = random.Random(args.seed)
    label = f"{speaker}: " if speaker else ""
    lines = []
    for number in range(1, args.count + 1):
        line = label + _snippet(model, args.phrase, args, rng)
        lines.append(line)
        print(f"{number}. {line}" if args.count > 1 else line)
    if args.save:
        storage.save_snippets(args.save, lines)
        print(f"saved {len(lines)} snippet(s) to {args.save}")
    return 0


def cmd_suggest(args) -> int:
    model, speaker = load_model(args)
    options = llm.suggest(model, args.phrase, args.top)
    where = f'after "{args.phrase}"' if args.phrase.strip() else "at the start of a line"
    who = f" ({speaker})" if speaker else ""
    print(f"most likely next words {where}{who}:")
    rows = [
        ["[end of line]" if word == core.END else word, f"{p:.3f}", f"{share:.0%}"]
        for word, p, share in options
    ]
    print(format_table(["word", "probability", "share"], rows))
    return 0


def cmd_speakers(args) -> int:
    counts = llm.speaker_counts(load_entries(args))
    if not counts:
        raise ValueError("no 'Speaker: line' labels found in the corpus")
    rows = [[name, count] for name, count in sorted(counts.items())]
    print(format_table(["speaker", "lines"], rows))
    return 0


def cmd_evaluate(args) -> int:
    entries = load_entries(args)
    speaker = llm.resolve_speaker(entries, args.speaker) if args.speaker else None
    results = llm.compare_orders(
        entries, args.k, args.max_order, speaker, args.test_fraction, seed=args.seed or 0
    )
    rows = [
        [order, ORDER_NAMES.get(order, f"{order}-gram"), f"{ppl:.1f}"]
        for order, ppl in results
    ]
    print("Perplexity on held-out lines (lower is better)")
    print(format_table(["order", "model", "perplexity"], rows))
    return 0


def run_chat(model, speaker, args, input_fn=input, out=print) -> int:
    """Interactive loop: type a phrase, then /more, /retry or /quit."""
    rng = random.Random(args.seed)
    label = f"{speaker}: " if speaker else ""
    out("Game Dialogue Assistant - type a starting phrase, or /more  /retry  /quit")
    last_phrase, last_text = None, None
    while True:
        try:
            raw = input_fn("> ").strip()
        except (EOFError, KeyboardInterrupt):
            out("")
            break
        if not raw:
            continue
        if raw in ("/quit", "/exit", "/q"):
            break
        if raw == "/retry":
            if last_phrase is None:
                out("nothing to retry yet - type a starting phrase first")
                continue
            last_text = _snippet(model, last_phrase, args, rng)
        elif raw == "/more":
            if last_text is None:
                out("nothing to continue yet - type a starting phrase first")
                continue
            last_text = _snippet(model, last_text, args, rng, min_tokens=max(args.min_length, 5))
        else:
            last_phrase = raw
            last_text = _snippet(model, raw, args, rng)
        out(label + last_text)
    return 0


def cmd_chat(args) -> int:
    if args.min_length > args.length:
        raise ValueError("min-length cannot be larger than length")
    model, speaker = load_model(args)
    return run_chat(model, speaker, args)


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--corpus", default=_env("GAME_CORPUS_PATH", "data/dialogue.txt"),
                        help="dialogue file with one 'Speaker: line' per line")
    common.add_argument("--speaker", default=None,
                        help="train on one character's lines only (keeps their voice)")
    common.add_argument("--order", type=int, default=int(_env("GAME_ORDER", "3")),
                        help="n-gram order: 2 = bigram, 3 = trigram (default 3)")
    common.add_argument("--k", type=float, default=float(_env("GAME_SMOOTHING_K", "0.1")),
                        help="add-k smoothing constant (default 0.1)")
    common.add_argument("--seed", type=int, default=_env_int("GAME_SEED"),
                        help="fix the random seed for repeatable output")

    sampling = argparse.ArgumentParser(add_help=False)
    sampling.add_argument("--length", type=int, default=25, help="maximum words to add (default 25)")
    sampling.add_argument("--min-length", type=int, default=0, help="do not end the line before this many words")
    sampling.add_argument("--temperature", type=float, default=1.0,
                          help="below 1 = safer, above 1 = more surprising (default 1.0)")
    sampling.add_argument("--top-k", type=int, default=None, help="only sample from the k most likely words")

    parser = argparse.ArgumentParser(
        prog="python -m app.cli",
        description="Generate NPC dialogue from a corpus of game scripts with an n-gram model.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", parents=[common, sampling], help="generate dialogue snippets")
    gen.add_argument("phrase", nargs="?", default="", help="starting phrase (optional)")
    gen.add_argument("--count", type=int, default=1, help="how many alternative snippets (default 1)")
    gen.add_argument("--save", metavar="FILE", help="append the snippets to this file")
    gen.set_defaults(func=cmd_generate)

    sug = sub.add_parser("suggest", parents=[common], help="show the most likely next words")
    sug.add_argument("phrase", nargs="?", default="", help="phrase to continue (optional)")
    sug.add_argument("--top", type=int, default=5, help="how many suggestions (default 5)")
    sug.set_defaults(func=cmd_suggest)

    spk = sub.add_parser("speakers", parents=[common], help="list the characters in the corpus")
    spk.set_defaults(func=cmd_speakers)

    ev = sub.add_parser("evaluate", parents=[common], help="compare n-gram orders by perplexity")
    ev.add_argument("--max-order", type=int, default=4, help="highest order to try (default 4)")
    ev.add_argument("--test-fraction", type=float, default=0.2, help="share of lines held out (default 0.2)")
    ev.set_defaults(func=cmd_evaluate)

    chat = sub.add_parser("chat", parents=[common, sampling], help="interactive mode with /more and /retry")
    chat.set_defaults(func=cmd_chat)

    return parser


def main(argv=None) -> int:
    storage.load_env()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
