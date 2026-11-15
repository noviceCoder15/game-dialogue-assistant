"""Text preprocessing and the n-gram language model.

Standard library only. Nothing in this module reads or writes files.

How the model works
-------------------
* Every dialogue line is padded with START tokens in front and an END token at
  the back, so the model also learns how lines begin and end.
* For an n-gram model of order n, we count how often each word follows each
  context of 0 .. n-1 previous words.
* Smoothing: add-k.  P(word | context) = (count + k) / (total + k * V).
  Words never seen after a context still get a small non-zero probability.
* Backoff: if the full context was never seen, use the longest shorter
  context that was (down to no context at all).
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

START = "<s>"   # padding before the first word of a line
END = "</s>"    # marks the end of a line

# --------------------------------------------------------------------------
# Text preprocessing and tokenization
# --------------------------------------------------------------------------

_REPLACEMENTS = {
    "\u2018": "'", "\u2019": "'",      # curly single quotes
    "\u201c": '"', "\u201d": '"',      # curly double quotes
    "\u2013": "-", "\u2014": "-",      # en / em dash
    "\u2026": "...",                   # ellipsis
    "\u00a0": " ",                     # non-breaking space
}


def clean_text(text: str) -> str:
    """Normalise punctuation, collapse whitespace and drop blank lines."""
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    lines = [" ".join(line.split()) for line in text.splitlines()]
    body = "\n".join(line for line in lines if line)
    return body + "\n" if body else ""


_SPEAKER_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 _'-]{0,29}):\s*(.+)$")


def parse_dialogue(text: str) -> list[tuple[str | None, str]]:
    """Split a script into (speaker, line) pairs.

    Expected format is one line per utterance: ``Speaker: what they say``.
    A line without a ``Speaker:`` label gets speaker None.
    """
    entries = []
    for line in clean_text(text).splitlines():
        match = _SPEAKER_RE.match(line)
        entries.append((match.group(1), match.group(2)) if match else (None, line))
    return entries


_TOKEN_RE = re.compile(r"\w+(?:'\w+)?|[^\w\s]")


def tokenize(text: str) -> list[str]:
    """Split text into words and single punctuation marks (case is kept)."""
    return _TOKEN_RE.findall(text)


_NO_SPACE_BEFORE = set(".,!?;:)")


def detokenize(tokens) -> str:
    """Join tokens back into readable text (no space before . , ! ? ; : and ))."""
    out = ""
    for token in tokens:
        if not out:
            out = token
        elif token in _NO_SPACE_BEFORE or out.endswith("("):
            out += token
        else:
            out += " " + token
    return out


# --------------------------------------------------------------------------
# The n-gram language model
# --------------------------------------------------------------------------

class NGramModel:
    """Order-n language model with add-k smoothing and backoff."""

    def __init__(self, order: int = 3, k: float = 0.1):
        if order < 1:
            raise ValueError("order must be at least 1")
        if k <= 0:
            raise ValueError("smoothing constant k must be greater than 0")
        self.order = order
        self.k = k
        self.counts: dict[tuple, Counter] = defaultdict(Counter)
        self.totals: dict[tuple, int] = defaultdict(int)
        self.vocab: set[str] = set()
        self.lower_map: dict[str, str] = {}

    # ---- training ----------------------------------------------------------

    def fit(self, token_lines) -> "NGramModel":
        """Count n-grams from a list of token lists (one list per dialogue line)."""
        n = self.order
        for tokens in token_lines:
            if not tokens:
                continue
            seq = [START] * (n - 1) + list(tokens) + [END]
            for i in range(n - 1, len(seq)):
                word = seq[i]
                self.vocab.add(word)
                for size in range(n):                    # context length 0 .. n-1
                    context = tuple(seq[i - size : i])
                    self.counts[context][word] += 1
                    self.totals[context] += 1
        if not self.vocab:
            raise ValueError("no training text: the corpus (or speaker) has no lines")
        self._build_case_map()
        return self

    def _build_case_map(self) -> None:
        """Map lower-case words to their most frequent spelling in the corpus."""
        unigrams = self.counts[()]
        best: dict[str, str] = {}
        for word, count in unigrams.items():
            low = word.lower()
            if low not in best or count > unigrams[best[low]]:
                best[low] = word
        self.lower_map = best

    # ---- helpers -----------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Known words plus one slot of probability for any unseen word."""
        return len(self.vocab) + 1

    def match_case(self, tokens) -> list[str]:
        """Map user-typed tokens onto the corpus spelling, e.g. 'the' -> 'The'."""
        return [
            t if t in self.vocab else self.lower_map.get(t.lower(), t) for t in tokens
        ]

    def _context_key(self, tokens) -> tuple:
        size = self.order - 1
        if size == 0:
            return ()
        tail = list(tokens[-size:])
        return tuple([START] * (size - len(tail)) + tail)

    def _backoff_key(self, tokens, exclude=()):
        """Longest context (from the end of `tokens`) that has usable continuations."""
        full = self._context_key(tokens)
        for size in range(len(full), -1, -1):
            key = full[len(full) - size :]
            counter = self.counts.get(key)
            if counter and any(word not in exclude for word in counter):
                return key
        return None

    # ---- probabilities -----------------------------------------------------

    def prob(self, word: str, tokens) -> float:
        """Smoothed P(word | the words in `tokens`), backing off if needed."""
        key = self._backoff_key(tokens)
        if key is None:
            raise ValueError("model has no training data")
        counter = self.counts[key]
        return (counter.get(word, 0) + self.k) / (self.totals[key] + self.k * self.vocab_size)

    def continuations(self, tokens, exclude=()) -> list[tuple[str, float]]:
        """Words seen after the context, as (word, probability), most likely first."""
        key = self._backoff_key(tokens, exclude)
        if key is None:
            return []
        denominator = self.totals[key] + self.k * self.vocab_size
        items = [
            (word, (count + self.k) / denominator)
            for word, count in self.counts[key].items()
            if word not in exclude
        ]
        items.sort(key=lambda item: (-item[1], item[0]))
        return items

    def sentence_log_prob(self, tokens) -> tuple[float, int]:
        """Natural-log probability of one line (including its END) and its token count."""
        history: list[str] = []
        total = 0.0
        sequence = list(tokens) + [END]
        for word in sequence:
            total += math.log(self.prob(word, history))
            history.append(word)
        return total, len(sequence)
