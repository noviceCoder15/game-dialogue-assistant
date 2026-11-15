from app import cli

SCRIPT = """Guard: Halt, traveler. State your business.
Guard: Move along, traveler. The gate closes at dusk.
Guard: Welcome back, hero. The captain asked about you.
Wizard: The stars are restless tonight, young one.
Wizard: The stars are silent tonight, young one.
Wizard: Magic is subtle, young one. Listen to the wind.
Hero: I seek the blacksmith. I need a new sword.
Hero: Tell me everything you know about the old tower.
Blacksmith: Steel does not lie. Bring me good iron.
Blacksmith: The forge never sleeps, and neither do I.
Merchant: Fine wares at fair prices, my friend.
Merchant: Everything in this cart has a story, my friend.
"""


def corpus(tmp_path):
    path = tmp_path / "dialogue.txt"
    path.write_text(SCRIPT, encoding="utf-8")
    return str(path)


def run(capsys, *argv):
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_generate_continues_the_starting_phrase(tmp_path, capsys):
    code, out, _ = run(capsys, "generate", "Halt", "--corpus", corpus(tmp_path),
                       "--seed", "1", "--length", "6")
    assert code == 0
    assert out.strip().startswith("Halt")


def test_generate_with_speaker_adds_their_name(tmp_path, capsys):
    code, out, _ = run(capsys, "generate", "The stars", "--speaker", "wizard",
                       "--corpus", corpus(tmp_path), "--seed", "1")
    assert code == 0
    assert out.startswith("Wizard: The stars")


def test_count_prints_numbered_alternatives(tmp_path, capsys):
    code, out, _ = run(capsys, "generate", "Halt", "--count", "3",
                       "--corpus", corpus(tmp_path), "--seed", "1")
    lines = out.strip().splitlines()
    assert code == 0 and len(lines) == 3
    assert [line[:2] for line in lines] == ["1.", "2.", "3."]


def test_same_seed_gives_the_same_output(tmp_path, capsys):
    args = ("generate", "The", "--count", "3", "--corpus", corpus(tmp_path), "--seed", "9")
    assert run(capsys, *args)[1] == run(capsys, *args)[1]


def test_save_appends_snippets_to_a_file(tmp_path, capsys):
    target = tmp_path / "out" / "favourites.txt"
    run(capsys, "generate", "Halt", "--corpus", corpus(tmp_path), "--seed", "1",
        "--save", str(target))
    run(capsys, "generate", "Halt", "--corpus", corpus(tmp_path), "--seed", "2",
        "--save", str(target))
    assert len(target.read_text().splitlines()) == 2


def test_suggest_lists_likely_next_words(tmp_path, capsys):
    code, out, _ = run(capsys, "suggest", "The stars are", "--corpus", corpus(tmp_path))
    assert code == 0
    assert "restless" in out and "probability" in out


def test_speakers_lists_every_character(tmp_path, capsys):
    code, out, _ = run(capsys, "speakers", "--corpus", corpus(tmp_path))
    assert code == 0
    for name in ("Guard", "Wizard", "Hero", "Blacksmith", "Merchant"):
        assert name in out


def test_evaluate_prints_a_perplexity_table(tmp_path, capsys):
    code, out, _ = run(capsys, "evaluate", "--corpus", corpus(tmp_path), "--seed", "1")
    assert code == 0
    assert "perplexity" in out and "trigram" in out


def test_chat_supports_more_and_retry(tmp_path):
    args = cli.build_parser().parse_args(
        ["chat", "--corpus", corpus(tmp_path), "--seed", "1", "--length", "6"]
    )
    model, speaker = cli.load_model(args)
    typed = iter(["/retry", "Halt", "/more", "/retry", "", "/quit"])
    lines = []
    cli.run_chat(model, speaker, args, input_fn=lambda prompt="": next(typed), out=lines.append)
    assert "nothing to retry" in lines[1]          # /retry before any phrase
    assert len(lines) == 5                         # banner, message, three snippets
    assert lines[2].startswith("Halt")
    assert lines[3].startswith("Halt")             # /more keeps the earlier text


def test_missing_corpus_is_reported(tmp_path, capsys):
    code, _, err = run(capsys, "generate", "--corpus", str(tmp_path / "nope.txt"))
    assert code == 1 and "corpus not found" in err


def test_unknown_speaker_is_reported(tmp_path, capsys):
    code, _, err = run(capsys, "generate", "--speaker", "nobody", "--corpus", corpus(tmp_path))
    assert code == 1 and "unknown speaker" in err


def test_min_length_larger_than_length_is_reported(tmp_path, capsys):
    code, _, err = run(capsys, "generate", "--length", "3", "--min-length", "9",
                       "--corpus", corpus(tmp_path))
    assert code == 1 and "min-length" in err
