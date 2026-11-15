import random

import pytest

from app import core, llm

CORPUS = """Guard: Halt, traveler. State your business.
Guard: Move along, traveler.
Wizard: The stars are restless tonight.
Wizard: The stars are silent tonight.
Hero: I seek the blacksmith."""


def model_for(text=CORPUS, order=3, k=0.1, speaker=None):
    return llm.build_model(core.parse_dialogue(text), order, k, speaker)


# ---- milestone 1: preprocessing and tokenization ---------------------------

def test_clean_text_normalises_and_drops_blank_lines():
    raw = "Hero:  \u201cHi\u201d \u2014 there\u2019s a  gap\n\n\n  Guard: ok  \n"
    assert core.clean_text(raw) == 'Hero: "Hi" - there\'s a gap\nGuard: ok\n'


def test_clean_text_of_nothing_is_empty():
    assert core.clean_text("  \n\n ") == ""


def test_parse_dialogue_splits_speaker_and_line():
    entries = core.parse_dialogue("Guard: Halt, traveler.\nJust narration here")
    assert entries == [("Guard", "Halt, traveler."), (None, "Just narration here")]


def test_tokenize_separates_words_and_punctuation():
    assert core.tokenize("Halt, traveler! Don't go.") == [
        "Halt", ",", "traveler", "!", "Don't", "go", ".",
    ]


def test_detokenize_round_trips_tokenize():
    text = "Halt, traveler. State your business!"
    assert core.detokenize(core.tokenize(text)) == text


# ---- milestone 2: n-gram model and smoothing -------------------------------

def test_probabilities_sum_to_one_for_any_context():
    model = model_for()
    for context in ([], ["Halt"], ["The", "stars"], ["never", "seen"]):
        total = sum(model.prob(word, context) for word in model.vocab)
        total += model.prob("zzz-never-seen", context)    # the 'unseen word' slot
        assert total == pytest.approx(1.0)


def test_unseen_word_still_gets_a_non_zero_probability():
    assert model_for().prob("dragon", ["The", "stars"]) > 0


def test_seen_continuation_is_more_likely_than_unseen():
    model = model_for()
    context = ["Halt", ","]
    assert model.prob("traveler", context) > model.prob("blacksmith", context)


def test_unseen_context_backs_off_to_a_shorter_one():
    model = model_for(order=3)
    assert model.continuations(["unknown", "stars"]) == model.continuations(["stars"])


def test_continuations_are_ranked_by_probability():
    options = model_for().continuations(["The", "stars"])
    assert options[0][0] == "are"
    probabilities = [p for _, p in options]
    assert probabilities == sorted(probabilities, reverse=True)


def test_suggest_shares_add_up_to_one():
    suggestions = llm.suggest(model_for(), "", top_n=100)
    assert sum(share for _, _, share in suggestions) == pytest.approx(1.0)
    assert llm.suggest(model_for(), "The stars", top_n=3)[0][0] == "are"


def test_speaker_model_only_learns_that_speakers_lines():
    model = model_for("A: red red red\nB: blue blue blue", speaker="A")
    assert model.vocab == {"red", core.END}


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        core.NGramModel(order=0)
    with pytest.raises(ValueError):
        core.NGramModel(k=0)
    with pytest.raises(ValueError):
        core.NGramModel().fit([])


# ---- milestone 3: generation ----------------------------------------------

def test_prompt_spelling_is_matched_to_the_corpus():
    # "halt" only appears as "Halt" in the corpus; "the" already appears in lower case.
    assert llm.prompt_tokens(model_for(), "halt the STARS") == ["Halt", "the", "stars"]


def test_generation_is_reproducible_with_a_seed():
    model = model_for()
    first = llm.generate_tokens(model, ["The"], 12, rng=random.Random(4))
    second = llm.generate_tokens(model, ["The"], 12, rng=random.Random(4))
    assert first == second


def test_generation_respects_the_length_limit():
    model = model_for("Guard: " + "go on " * 40)
    assert len(llm.generate_tokens(model, [], max_tokens=5, rng=random.Random(1))) <= 5


def test_min_length_stops_the_line_ending_early():
    model = model_for("Guard: Hi.\nGuard: Hi.\nGuard: Hi.")
    words = llm.generate_tokens(model, [], max_tokens=8, min_tokens=5, rng=random.Random(2))
    assert len(words) >= 5


def test_generation_only_uses_words_from_the_speakers_corpus():
    model = model_for("A: red red red\nB: blue blue blue", speaker="A")
    words = llm.generate_tokens(model, [], max_tokens=6, rng=random.Random(3))
    assert set(words) <= {"red"}


def test_generate_snippet_starts_with_a_capital_letter():
    text = llm.generate_snippet(model_for(), "the stars", max_tokens=4, rng=random.Random(5))
    assert text.startswith("The stars")


def test_bad_sampling_settings_are_rejected():
    model = model_for()
    with pytest.raises(ValueError):
        llm.generate_tokens(model, [], temperature=0)
    with pytest.raises(ValueError):
        llm.generate_tokens(model, [], top_k=0)
    with pytest.raises(ValueError):
        llm.generate_tokens(model, [], max_tokens=-1)


# ---- evaluation -------------------------------------------------------------

def test_text_from_the_corpus_has_lower_perplexity_than_scrambled_text():
    model = model_for()
    real = [core.tokenize("The stars are restless tonight.")]
    scrambled = [core.tokenize("tonight restless stars are The .")]
    assert llm.perplexity(model, real) < llm.perplexity(model, scrambled)


def test_train_test_split_holds_lines_out():
    entries = [("A", f"line {i}") for i in range(10)]
    train, test = llm.train_test_split(entries, 0.2, seed=1)
    assert len(test) == 2 and len(train) == 8
    assert not set(train) & set(test)


def test_compare_orders_returns_one_result_per_order():
    entries = core.parse_dialogue((CORPUS + "\n") * 3)
    results = llm.compare_orders(entries, max_order=3)
    assert [order for order, _ in results] == [1, 2, 3]
    assert all(ppl > 1 for _, ppl in results)
