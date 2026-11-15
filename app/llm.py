"""Dialogue generation and evaluation on top of the n-gram model."""
from __future__ import annotations

import math
import random
from collections import Counter

from .core import END, NGramModel, clean_text, detokenize, tokenize


# --------------------------------------------------------------------------
# Building models (optionally for one character only)
# --------------------------------------------------------------------------

def speaker_counts(entries) -> Counter:
    """How many lines each speaker has."""
    return Counter(who for who, _ in entries if who)


def resolve_speaker(entries, name: str) -> str:
    """Match a speaker name case-insensitively, or explain what is available."""
    counts = speaker_counts(entries)
    for known in counts:
        if known.lower() == name.strip().lower():
            return known
    available = ", ".join(sorted(counts)) or "none found"
    raise ValueError(f"unknown speaker '{name}' - choose from: {available}")


def build_model(entries, order: int = 3, k: float = 0.1, speaker: str | None = None) -> NGramModel:
    """Train on every line, or only on the lines of one speaker."""
    lines = [
        tokenize(text)
        for who, text in entries
        if speaker is None or who == speaker
    ]
    return NGramModel(order, k).fit(lines)


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------

def prompt_tokens(model: NGramModel, phrase: str) -> list[str]:
    """Tokenize the user's starting phrase and match the corpus spelling."""
    return model.match_case(tokenize(clean_text(phrase)))


def generate_tokens(
    model: NGramModel,
    prompt,
    max_tokens: int = 25,
    min_tokens: int = 0,
    temperature: float = 1.0,
    top_k: int | None = None,
    rng: random.Random | None = None,
) -> list[str]:
    """Chain next-word samples after `prompt`; returns only the new tokens.

    Stops when the model emits the end-of-line token or after `max_tokens`.
    The end token is not allowed before `min_tokens` words have been generated.
    """
    if max_tokens < 0:
        raise ValueError("length cannot be negative")
    if temperature <= 0:
        raise ValueError("temperature must be greater than 0")
    if top_k is not None and top_k < 1:
        raise ValueError("top-k must be at least 1")

    rng = rng or random.Random()
    history = list(prompt)
    new: list[str] = []
    while len(new) < max_tokens:
        exclude = (END,) if len(new) < min_tokens else ()
        options = model.continuations(history, exclude)
        if not options:
            break
        if top_k:
            options = options[:top_k]
        best = options[0][1]
        words = [word for word, _ in options]
        # Dividing by the best probability keeps weights in a safe range for any temperature.
        weights = [(p / best) ** (1.0 / temperature) for _, p in options]
        word = rng.choices(words, weights=weights, k=1)[0]
        if word == END:
            break
        new.append(word)
        history.append(word)
    return new


def generate_snippet(model: NGramModel, phrase: str = "", **options) -> str:
    """Starting phrase in, finished dialogue snippet out."""
    prompt = prompt_tokens(model, phrase)
    text = detokenize(prompt + generate_tokens(model, prompt, **options))
    return text[:1].upper() + text[1:]


def suggest(model: NGramModel, phrase: str, top_n: int = 5) -> list[tuple[str, float, float]]:
    """The most likely next words after `phrase` as (word, probability, share).

    `probability` is the smoothed model probability. Smoothing keeps some of it
    back for unseen words, so `share` shows each word's part of the choices the
    model has actually seen here (the shares add up to 1).
    """
    options = model.continuations(prompt_tokens(model, phrase))
    total = sum(p for _, p in options)
    return [(word, p, p / total) for word, p in options[:top_n]]


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def train_test_split(entries, test_fraction: float = 0.2, seed: int = 0):
    """Shuffle (reproducibly) and hold out a share of lines for testing."""
    if len(entries) < 2:
        raise ValueError("need at least 2 lines to hold some out for testing")
    order = list(range(len(entries)))
    random.Random(seed).shuffle(order)
    n_test = min(len(entries) - 1, max(1, round(len(entries) * test_fraction)))
    test_ids = set(order[:n_test])
    train = [e for i, e in enumerate(entries) if i not in test_ids]
    test = [e for i, e in enumerate(entries) if i in test_ids]
    return train, test


def perplexity(model: NGramModel, token_lines) -> float:
    """exp(average negative log-probability per token). Lower is better."""
    total_log, total_tokens = 0.0, 0
    for tokens in token_lines:
        if not tokens:
            continue
        log_prob, count = model.sentence_log_prob(tokens)
        total_log += log_prob
        total_tokens += count
    if total_tokens == 0:
        raise ValueError("no test text to score")
    return math.exp(-total_log / total_tokens)


def compare_orders(entries, k: float = 0.1, max_order: int = 4,
                   speaker: str | None = None, test_fraction: float = 0.2, seed: int = 0):
    """Held-out perplexity for n-gram orders 1 .. max_order."""
    if speaker is not None:
        entries = [e for e in entries if e[0] == speaker]
    train, test = train_test_split(entries, test_fraction, seed)
    test_lines = [tokenize(text) for _, text in test]
    results = []
    for order in range(1, max_order + 1):
        model = build_model(train, order, k)
        results.append((order, perplexity(model, test_lines)))
    return results
