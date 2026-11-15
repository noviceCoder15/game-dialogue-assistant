"""File I/O: .env loading, the dialogue corpus and saved snippets."""
from __future__ import annotations

import os
from pathlib import Path


def load_env(path: str = ".env") -> None:
    """Read KEY=VALUE lines from a .env file into os.environ (existing vars win)."""
    env_file = Path(path)
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def read_corpus(path: str) -> str:
    """Return the raw text of the dialogue corpus."""
    corpus_file = Path(path)
    if not corpus_file.exists():
        raise FileNotFoundError(f"corpus not found: {path}")
    return corpus_file.read_text(encoding="utf-8")


def save_snippets(path: str, snippets) -> None:
    """Append generated snippets to a text file, one per line."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        for snippet in snippets:
            handle.write(snippet + "\n")
