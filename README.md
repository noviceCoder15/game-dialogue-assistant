# Game Dialogue Assistant
[![Built with askThili](https://msrit.ask.thili.ai/api/v1/public/projects/tzdup4gat7/badge.svg)](https://msrit.ask.thili.ai/b/tzdup4gat7)

A command-line tool that helps game designers prototype NPC dialogue. Give it a starting
phrase and it continues the line, word by word, using an n-gram language model trained on a
corpus of game scripts. Train on one character's lines to keep their voice consistent.

Plain Python, standard library only. It runs on a laptop with no GPU and no API keys.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # installs pytest, used for the tests only
pytest                                  # check everything works

python -m app.cli generate "The stars"
python -m app.cli generate "Halt" --speaker guard --count 3
python -m app.cli suggest "The stars are"
python -m app.cli speakers
python -m app.cli evaluate
python -m app.cli chat --speaker wizard
```

## Commands

| Command | What it does |
|---|---|
| `generate [phrase]` | Continue a starting phrase. Leave the phrase out to start a fresh line. |
| `suggest [phrase]` | Show the most likely next words after a phrase. |
| `speakers` | List the characters in the corpus and how many lines each has. |
| `evaluate` | Compare n-gram orders by perplexity on held-out lines. |
| `chat` | Interactive mode: type a phrase, then `/more`, `/retry` or `/quit`. |

Options for `generate` and `chat`:

| Option | Meaning |
|---|---|
| `--speaker NAME` | Train on that character's lines only (case-insensitive). |
| `--length N` | Maximum number of words to add (default 25). |
| `--min-length N` | Do not let the line end before N words. |
| `--count N` | Generate N alternative snippets (`generate` only). |
| `--temperature T` | Below 1 is safer, above 1 is more surprising (default 1.0). |
| `--top-k K` | Only sample from the K most likely words. |
| `--order N` | n-gram order: 2 is bigram, 3 is trigram (default 3). |
| `--k X` | Add-k smoothing constant (default 0.1). |
| `--seed N` | Fix the random seed to repeat the same output. |
| `--save FILE` | Append the snippets to a file (`generate` only). |

Defaults can also be set in a `.env` file (copy `.env.example`).

## Example output

Your output will differ unless you pass the same `--seed`.

```
$ python -m app.cli generate "The stars" --speaker wizard --seed 7
Wizard: The stars are restless tonight. Something stirs beneath the old tower.

$ python -m app.cli generate "Halt" --speaker guard --count 3 --seed 3
1. Guard: Halt! Who goes there? Speak quickly, traveler. Those are the rules.
2. Guard: Halt, traveler. State your business in Ironvale.
3. Guard: Halt, traveler. Take the ford and keep your boots dry.

$ python -m app.cli suggest "I will" --top 4
most likely next words after "I will":
word   probability  share
-----  -----------  -----
make   0.032        18%
find   0.017        9%
forge  0.017        9%
give   0.017        9%

$ python -m app.cli evaluate --seed 1
Perplexity on held-out lines (lower is better)
order  model    perplexity
-----  -------  ----------
1      unigram  315.6
2      bigram   157.1
3      trigram  183.4
4      4-gram   188.6
```

In `chat`, `/more` keeps writing after the last snippet and `/retry` makes a new version
from the same starting phrase.

## Milestones

| Milestone | Where |
|---|---|
| Collect and preprocess a corpus | `data/dialogue.txt`; `clean_text`, `parse_dialogue`, `tokenize` in `app/core.py` |
| N-gram model with smoothing | `NGramModel` in `app/core.py` (add-k smoothing with backoff) |
| Dialogue generation logic | `generate_tokens` and `generate_snippet` in `app/llm.py` |
| Command-line interface | `app/cli.py` (`argparse`) |
| Stretch: set the length | `--length` and `--min-length` |
| Stretch: alternative snippets | `--count` |
| Stretch: generate more / retry | `chat` with `/more` and `/retry` |

## How it works

1. **Preprocess.** Punctuation is normalised, whitespace collapsed, and each `Speaker: line`
   is split into speaker and text. Text is tokenized into words and punctuation marks.
2. **Count.** Every line is padded with start markers and an end marker. For each word we
   count what followed each context of 0 to n-1 previous words.
3. **Smooth.** `P(word | context) = (count + k) / (total + k * V)`. A word never seen after a
   context still gets a small non-zero probability.
4. **Back off.** If the full context was never seen, use the longest shorter context that was.
5. **Generate.** Look up the words seen after the current context, weight them by
   probability (adjusted by `--temperature` and `--top-k`), sample one, add it to the context,
   and repeat until the end marker or the length limit.

## Things to know

- **A small corpus means familiar lines.** With about 100 lines, the model often reproduces
  corpus lines or long stretches of them. Add more scripts to `data/dialogue.txt` for more
  variety. A lower `--order` or a higher `--temperature` gives more novelty, at the cost of
  coherence. Order 2 with `--temperature 1.2` can produce lines like "Halt, and one very
  angry goat."
- **Higher order is not always better.** In `evaluate`, the trigram scores worse than the
  bigram on this small corpus. Most trigrams are seen only once or never, so the counts are
  sparse. Interpolating the orders (mixing their probabilities) instead of backing off is a
  natural improvement.
- **`probability` in `suggest` looks small.** Smoothing keeps part of the probability for
  words the model has not seen, so the seen words share the rest. The `share` column shows
  each word's part of the choices the model has actually seen.
- **`--length` is a cap.** A snippet that hits it can stop mid-sentence.
- **`--speaker` shrinks the training data.** One character has only 16 to 27 lines here, so
  their lines repeat more.
- **Corpus format.** One utterance per line, written as `Speaker: text`.

## Tests

```bash
pytest
```

The tests cover preprocessing, probabilities (they sum to 1 and unseen words stay above
zero), backoff, reproducible sampling, per-character models and every CLI command.
`.github/workflows/ci.yml` runs them on GitHub after each push.

## Layout

```
app/core.py      cleaning, tokenizer, n-gram model with smoothing (no file I/O)
app/llm.py       generation, suggestions, perplexity evaluation
app/storage.py   .env, corpus file, saved snippets
app/cli.py       generate / suggest / speakers / evaluate / chat
data/            dialogue corpus
tests/           unit tests and CLI smoke tests
```
