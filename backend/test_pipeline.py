"""The smallest checks that fail if the branching logic breaks.

    python -m pytest backend/test_pipeline.py -q
"""
import importlib.util
from pathlib import Path

import pytest

from . import citation_verifier as cv
from . import config
from . import dialogue_manager as dm


def script(name: str):
    """Import scripts/NN_name.py, whose leading digits make it unimportable."""
    path = Path(__file__).resolve().parent.parent / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

VERSES = [
    {"verse_id": "gita_2_47",
     "text": "You have a right to perform your prescribed duty, but you are not "
             "entitled to the fruits of action. Never consider yourself the cause "
             "of the results of your activities, and never be attached to not "
             "doing your duty."},
    {"verse_id": "gita_2_13",
     "text": "As the embodied soul continuously passes, in this body, from boyhood "
             "to youth to old age, the soul similarly passes into another body at "
             "death. A sober person is not bewildered by such a change."},
]


# --- citation verifier: the "no verse, no answer" guardrail -------------------

def test_supported_sentence_passes():
    r = cv.check("Act from duty, but do not claim the fruits of your action "
                 "[gita_2_47].", VERSES)
    assert r["ok"] and r["cited_ids"] == ["gita_2_47"]


def test_invented_anchor_is_rejected():
    r = cv.check("The Gita says to abandon all duty [gita_99_99].", VERSES)
    assert not r["ok"] and "invented" in r["reason"]


def test_unsupported_claim_is_rejected():
    r = cv.check("The scriptures set out a detailed schedule of quarterly tax "
                 "filings for merchants in every province.", VERSES)
    assert not r["ok"]


def test_plausible_sounding_fabrication_is_rejected():
    # Shares battlefield vocabulary with the verses, so it scores high on topic
    # while asserting something the texts never say. This is the case a single
    # loose threshold waves through.
    r = cv.check("One should file a legal petition before entering the "
                 "battlefield.", VERSES)
    assert not r["ok"]


def test_prose_citation_to_an_unretrieved_verse_is_rejected():
    # Caught live: the model wrote "(Gita 2.37)" in prose when 2.38 was retrieved.
    # Bracket-only checking never saw it, and speakable() renders real anchors in
    # the same shape — so the student hears a fabricated reference as a real one.
    r = cv.check("Fighting without attachment is noble, as Gita 2.37 teaches.",
                 VERSES)
    assert not r["ok"] and "invented" in r["reason"]


def test_prose_citation_to_a_retrieved_verse_passes():
    r = cv.check("Gita 2.47 says to act from duty without claiming the fruits.",
                 VERSES)
    assert r["ok"] and r["cited_ids"] == ["gita_2_47"]


def test_prose_citation_is_satisfied_by_the_purport_chunk():
    verses = [{"verse_id": "gita_18_8_p1",
               "text": "One who is in Krishna consciousness should not give up "
                       "earning money out of fear of fruitive activities."}]
    r = cv.check("Gita 18.8 says not to abandon earning out of fear.", verses)
    assert r["ok"] and r["cited_ids"] == ["gita_18_8_p1"]


def test_sentence_that_drifts_from_its_own_citation_is_rejected():
    # Cites 2.47 (duty and fruits) while talking about the soul and death.
    r = cv.check("The soul is destroyed at the moment the body dies "
                 "[gita_2_47].", VERSES)
    assert not r["ok"]


def test_verse_negation_is_rejected():
    # Was a strict xfail: cosine scored the negation 0.777 against gita_2_13 and
    # the faithful version 0.840, so no threshold could separate them. The NLI
    # stage reads the pair instead and calls this contradiction at 1.00.
    r = cv.check("The soul is destroyed at death and does not pass into another "
                 "body [gita_2_13].", VERSES)
    assert not r["ok"] and "contradicts" in r["reason"]


def test_the_plainest_reading_of_2_47_is_not_called_a_contradiction():
    # The trap that disqualified nli-deberta-v3-xsmall: it reads "let the
    # results go" as abandoning the duty — which 2.47 does forbid — and scored
    # this 0.970, against a weakest true contradiction of 0.997. If a future
    # swap to a smaller NLI model regresses, this is the test that says so.
    r = cv.check("Do your duty and let the results go [gita_2_47].", VERSES)
    assert r["ok"], r["reason"]


def test_faithful_paraphrase_of_the_same_verse_still_passes():
    # The other half of the test above: an NLI stage that rejects everything
    # would satisfy it just as well, and silence the Guru. Faithful teaching
    # frequently reads as `neutral` rather than `entailment`, which is why the
    # gate rejects only on contradiction.
    r = cv.check("The soul passes into another body at death, and the sober are "
                 "not bewildered by it [gita_2_13].", VERSES)
    assert r["ok"] and r["cited_ids"] == ["gita_2_13"]


def test_conversational_sentence_needs_no_citation():
    assert cv.check("Shall we begin?", VERSES)["ok"]


def test_no_verses_means_no_answer():
    assert not cv.check("Duty is its own reward.", [])["ok"]


def test_anchors_are_stripped_before_speaking():
    assert "[" not in cv.strip_anchors("Do your duty [gita_2_47].")


# --- sentence streaming ------------------------------------------------------

def test_split_stream_holds_back_partial_sentence():
    done, rest = cv.split_stream("First one. Second one. And a par")
    assert done == ["First one.", "Second one."] and rest == "And a par"


def test_split_stream_waits_when_nothing_is_complete():
    assert cv.split_stream("no end yet") == ([], "no end yet")


# --- dialogue manager --------------------------------------------------------

def test_rule_hits():
    s = dm.new_state()
    assert dm.classify("teach me the second chapter", s) == "TEACH"
    assert dm.classify("I disagree with that entirely", s) == "DEBATE"
    assert dm.classify("what does that mean", s) == "DOUBT"


def test_lesson_advances_by_a_whole_window_not_one_verse():
    # Stepping by 1 would re-teach TEACH_WINDOW-1 of the verses just heard,
    # since the retriever slices verses[pos : pos + TEACH_WINDOW].
    s = dm.new_state()
    s["position"] = 3
    assert dm.route("continue", s)["state"]["position"] == 3 + config.TEACH_WINDOW


def test_forced_mode_overrides_classification():
    s = dm.new_state()
    s["forced_mode"] = "COUNSEL"
    assert dm.route("teach me chapter two", s)["mode"] == "COUNSEL"


def test_changing_chapter_restarts_the_lesson():
    # Caught live: teach chapter 2, say "continue" three times, then switch the
    # picker to chapter 12 — the lesson opened at verse 5 of 12 because position
    # carried over, and the prompt announced it as "verse 5 of chapter 12".
    s = dm.new_state()
    s["forced_mode"], s["source"], s["chapter"] = "TEACH", "Gita", 2
    for q in ("teach me chapter 2", "continue", "continue", "continue"):
        dm.route(q, s)
    assert s["position"] == 3 * config.TEACH_WINDOW

    s["chapter"] = 12
    assert dm.route("teach me chapter 12", s)["state"]["position"] == 0


def test_continuing_the_same_chapter_does_not_restart_it():
    s = dm.new_state()
    s["forced_mode"], s["source"], s["chapter"] = "TEACH", "Gita", 2
    dm.route("teach me chapter 2", s)
    dm.route("continue", s)
    assert s["position"] == config.TEACH_WINDOW


# --- retrieval strategy ------------------------------------------------------

def test_teach_walks_the_chapter_from_the_position():
    from . import retriever
    s = dm.new_state() | {"source": "Gita", "chapter": 2, "position": 0}
    first = retriever.search("teach me chapter 2", "TEACH", s)
    assert first[0]["verse_id"] == "gita_2_1"
    # The verse travels with its purport: a third-person paraphrase of the verse
    # alone scores as drift, and the whole turn is withheld.
    assert any(v["verse_id"].startswith("gita_2_1_p") for v in first)

    s["position"] = config.TEACH_WINDOW
    assert retriever.search("continue", "TEACH", s)[0]["verse_id"] != "gita_2_1"


def test_debate_retrieves_both_sides_and_more_than_counsel():
    # Debate used to fetch TOP_K // 2 per side and, after dedup, argue from 3
    # verses where COUNSEL had 6 — so the model filled the gap from memory and
    # every sentence was withheld as unsupported. It is the one mode holding two
    # positions at once; it gets the most material, not the least.
    from . import retriever
    q = "I disagree that fighting can ever be right"
    hits = retriever.search(q, "DEBATE", {"source": "Gita"})
    assert len(hits) > config.TOP_K // 2
    assert len(hits) <= config.DEBATE_TOP_K
    assert len({h["verse_id"] for h in hits}) == len(hits)      # no duplicates

    # The counter-query must actually reach verses the plain query does not,
    # or "both sides" is just a reordering of one side.
    plain = {h["verse_id"] for h in retriever._semantic(q, config.DEBATE_TOP_K, "Gita")}
    assert {h["verse_id"] for h in hits} - plain


def test_teach_walks_a_mahabharata_parva_in_section_order():
    from . import retriever
    s = dm.new_state() | {"source": "Mahabharata", "chapter": "Adi", "position": 0}
    first = retriever.search("teach me the Adi Parva", "TEACH", s)
    assert first and first[0]["verse_id"] == "mbh_adi_1"
    assert all(h["source"] == "Mahabharata" for h in first)

    s["position"] = config.TEACH_WINDOW
    assert retriever.search("continue", "TEACH", s)[0]["verse_id"] != "mbh_adi_1"


def test_a_parva_is_ordered_by_section_not_lexically():
    # Sorting ids as strings would teach section 1, 10, 100, 1000, 101 — and a
    # long section's split chunks must follow their own section, not interleave.
    from . import retriever
    ids = [r["id"] for r in retriever._lesson_chunks("Mahabharata", "Adi")]
    order = [retriever._lesson_order(r)
             for r in retriever._lesson_chunks("Mahabharata", "Adi")]
    assert order == sorted(order)
    assert ids[0] == "mbh_adi_1"
    assert order[0] == (1, 1)


def test_the_parva_name_is_matched_regardless_of_case():
    from . import retriever
    assert (retriever._lesson_chunks("Mahabharata", "adi")
            == retriever._lesson_chunks("Mahabharata", "Adi"))


def test_mbh_span_names_the_section_and_parva():
    from . import prompt_builder as pb
    assert pb._span([{"verse_id": "mbh_udyoga_33"}]) == \
        "section 33 of the Udyoga Parva"
    # A section split across chunks is still one section to the student.
    assert pb._span([{"verse_id": "mbh_karna_1"}, {"verse_id": "mbh_karna_1_2"}]) == \
        "section 1 of the Karna Parva"


def test_a_numeric_chapter_from_the_client_stays_a_number():
    # The picker sends its raw value now, because Number("Udyoga") is NaN. A
    # Gita chapter arriving as "2" must not compare unequal to the stored 2, or
    # dialogue_manager restarts the lesson on every turn.
    from .main import _unit
    assert _unit("2") == 2 and _unit(2) == 2
    assert _unit("Udyoga") == "Udyoga"
    assert _unit("") is None and _unit(None) is None


def test_switching_from_a_chapter_to_a_parva_restarts_the_lesson():
    s = dm.new_state()
    s["forced_mode"], s["source"], s["chapter"] = "TEACH", "Gita", 2
    dm.route("teach me chapter 2", s)
    dm.route("continue", s)
    assert s["position"] == config.TEACH_WINDOW

    s["source"], s["chapter"] = "Mahabharata", "Adi"
    assert dm.route("teach me the Adi Parva", s)["state"]["position"] == 0


def test_the_source_filter_does_not_starve_now_the_gita_is_a_minority():
    # The Gita is 11% of 17k chunks. The old flat 8x over-fetch was set when the
    # index held only the Gita and the filter discarded nothing; against the full
    # corpus it returned 6 of the 8 asked for on the DEBATE counter-query, which
    # is the starvation that had the model arguing from memory.
    from . import retriever
    for q in ("I disagree that fighting can ever be right",
              "the scriptural answer refuting the claim that fighting is wrong"):
        assert len(retriever._semantic(q, config.DEBATE_TOP_K, "Gita")) == \
            config.DEBATE_TOP_K, q


def test_teach_falls_back_to_search_when_the_chapter_is_unknown():
    from . import retriever
    s = dm.new_state() | {"source": "Gita", "chapter": None}
    hits = retriever.search("how do I act without craving the result", "TEACH", s)
    assert hits and all(h["score"] < 1.0 for h in hits)   # scored, not sequential


# --- prompt builder ----------------------------------------------------------

def test_span_names_a_single_verse():
    from . import prompt_builder as pb
    assert pb._span([{"verse_id": "gita_2_47"}]) == "verse 47 of chapter 2"


def test_span_names_a_range():
    from . import prompt_builder as pb
    span = pb._span([{"verse_id": "gita_2_1"}, {"verse_id": "gita_2_2"},
                     {"verse_id": "gita_2_3"}])
    assert span == "verses 1 to 3 of chapter 2"


def test_span_ignores_the_purport_chunks_travelling_with_a_verse():
    # gita_2_5_p1 is commentary on verse 5, not a verse of its own — counting it
    # would announce a range the student was never taught.
    from . import prompt_builder as pb
    assert pb._span([{"verse_id": "gita_2_5"},
                     {"verse_id": "gita_2_5_p1"}]) == "verse 5 of chapter 2"


# --- Mahabharata anchors ------------------------------------------------------

MBH = [{"verse_id": "mbh_udyoga_33",
        "text": "Krishna said, 'Give the Pandavas five villages only, and let "
                "there be peace between you and them, O Bharata.' Duryodhana "
                "answered that he would not yield land enough to cover the "
                "point of a needle."}]


def test_a_mahabharata_anchor_is_spoken_as_a_reference_not_read_out_literally():
    # Was "Mbh 33": speakable() kept only the digits, so the parva name was
    # dropped and the prefix was read aloud as a word.
    said = cv.speakable("Krishna sought peace first [mbh_udyoga_33].")
    assert "Mahabharata" in said and "Udyoga" in said
    assert "Mbh" not in said and "mbh_udyoga_33" not in said


def test_gita_anchors_are_still_spoken_the_old_way():
    assert cv.speakable("Do your duty [gita_12_11].") == "Do your duty (Gita 12.11)."


def test_mahabharata_prose_citation_to_an_unretrieved_section_is_rejected():
    # The Gita side of this door was closed by PROSE_REF; the Mahabharata side
    # was still open, and speakable() now renders real anchors in exactly this
    # shape — so the student cannot tell the invented one from the real one.
    r = cv.check("Krishna asked for five villages, as the Udyoga Parva, "
                 "Section 34 recounts.", MBH)
    assert not r["ok"] and "invented" in r["reason"]


def test_mahabharata_prose_citation_to_a_retrieved_section_passes():
    r = cv.check("In the Udyoga Parva, Section 33, Krishna sought peace and "
                 "Duryodhana refused him any land at all.", MBH)
    assert r["ok"] and r["cited_ids"] == ["mbh_udyoga_33"]


def test_an_anchor_written_in_capitals_is_still_the_verse_it_names():
    # Measured over 52 sentences: the model writes [MBH_SANTI_176_5] often
    # enough to matter. ANCHOR_RE matches case-insensitively but ids are
    # lowercase by construction, so the lookup missed and a verse the model had
    # actually been handed was rejected as an invented anchor.
    r = cv.check("Krishna sought peace before the war [MBH_UDYOGA_33].", MBH)
    assert r["ok"] and r["cited_ids"] == ["mbh_udyoga_33"], r["reason"]
    r = cv.check("Act from duty, but do not claim the fruits [GITA_2_47].", VERSES)
    assert r["ok"] and r["cited_ids"] == ["gita_2_47"], r["reason"]


def test_a_capitalised_anchor_that_was_not_retrieved_is_still_rejected():
    # Lowercasing loosens parsing, not the guardrail.
    assert not cv.check("The Gita says to abandon duty [GITA_99_99].", VERSES)["ok"]


def test_naming_a_parva_without_a_section_is_not_read_as_a_citation():
    # "the Udyoga Parva, 5 kings gathered" is prose. Treating the stray number
    # as a section would withhold an otherwise sound sentence.
    assert cv.anchors("In the Udyoga Parva, 5 kings gathered.") == []


# --- corpus: the anchors everything downstream is built on --------------------

MBH_FIXTURE = """BOOK 1

ADI PARVA

Translated into English Prose by Kisari Mohan Ganguli.

Scanned and Proofed at sacred-texts.com by J. B. Hare, October 2003.

SECTION I

Om! Having bowed down unto Narayana, and unto that most exalted of male
beings, must the word Jaya be uttered before all else.

SECTION II

Sauti said that he had heard the whole of that history at the great
sacrifice of the wise king Janamejaya, son of Parikshit.

BOOK 8

Karna-parva

Translated into English Prose by Kisari Mohan Ganguli.

1

After Drona had been slain, the royal warriors of the Kaurava army
repaired to Drona's son with hearts filled with great anxiety.

2

Then commenced a fierce battle between the Kurus and the Pandavas, each
of them desirous of vanquishing the other in that dreadful encounter.
"""


def test_each_book_gets_its_own_parva_and_both_section_styles_parse():
    # The bug this replaces: one parva name was read off the top of the file and
    # applied to all 3-6 parvas under it, while section numbers restart at 1 in
    # every parva — so Karna section 1 was emitted as mbh_adi_1, colliding with
    # Adi's own section 1 and mislabelling the text of five parvas per volume.
    recs = script("02_chunk_and_tag.py").chunk_mahabharata(MBH_FIXTURE)
    by_id = {r["id"]: r for r in recs}
    assert set(by_id) == {"mbh_adi_1", "mbh_adi_2", "mbh_karna_1", "mbh_karna_2"}
    assert "Narayana" in by_id["mbh_adi_1"]["text"]          # SECTION I, roman
    assert "Drona" in by_id["mbh_karna_1"]["text"]           # bare "1", book 8
    assert by_id["mbh_karna_1"]["ref"] == "Mahabharata, Karna Parva, Section 1"


def test_front_matter_does_not_become_citable_scripture():
    recs = script("02_chunk_and_tag.py").chunk_mahabharata(MBH_FIXTURE)
    assert not any("sacred-texts" in r["text"] or "Ganguli" in r["text"]
                   for r in recs)


def test_the_gutenberg_licence_is_stripped_and_never_indexed():
    clean = script("01_clean_texts.py")
    text = ("The Project Gutenberg eBook of something. You may copy it under "
            "the terms of the Project Gutenberg License.\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK THE MAHABHARATA ***\n"
            "BOOK 1\n\nADI PARVA\n\nOm! Having bowed down unto Narayana.\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK THE MAHABHARATA ***\n"
            "Please read the full licence terms and redistribution conditions.")
    body = clean.strip_gutenberg(text, "fixture.txt")
    assert "Narayana" in body
    assert "Project Gutenberg" not in body and "licence" not in body


def test_a_gutenberg_file_with_no_markers_is_refused_not_guessed():
    clean = script("01_clean_texts.py")
    with pytest.raises(SystemExit):
        clean.strip_gutenberg("The Project Gutenberg License applies.", "x.txt")


def test_an_over_long_paragraph_is_split_instead_of_shipped_whole():
    # Ganguli runs a whole section of dialogue as one paragraph — the longest
    # measured 60,770 chars. pack() never cut inside a paragraph, so bge-small
    # (512 tokens, ~2,000 chars) indexed such a chunk on its opening 3% and the
    # rest could never be retrieved at all.
    chunker = script("02_chunk_and_tag.py")
    huge = " ".join(["Then the king spoke unto his son in great sorrow."] * 200)
    chunks = chunker.pack([huge])
    assert len(chunks) > 1
    assert all(len(c) <= chunker.MAX_CHARS for c in chunks)
    assert "".join(chunks).replace(" ", "") == huge.replace(" ", "")  # nothing lost


# --- cross-reference (arch doc 4.4) -------------------------------------------

def test_crossref_illustrates_gita_verses_with_mahabharata_passages():
    # Reads the verse out of the map rather than naming one: which verses carry
    # an illustration is a property of CROSSREF_MIN_SCORE, and hardcoding one
    # would make this fail on a threshold change rather than on a broken expand.
    from . import crossref
    linked = next(iter(crossref._load()))
    links = crossref.expand([linked])
    assert links, f"{linked} is in the map but expand returned nothing"
    assert all(l["source"] == "Mahabharata" for l in links)
    assert all(l["illustrates"] == linked for l in links)
    assert all(l["text"] and l["score"] >= config.CROSSREF_MIN_SCORE for l in links)


def test_crossref_is_a_curated_map_not_everything_above_the_noise_floor():
    # At the old 0.55 every one of the 1,979 Gita chunks kept a link, so every
    # turn carried an "illustration" whether or not one existed. A verse with no
    # strong parallel must get none.
    from . import crossref
    assert 0 < len(crossref._load()) < 1500


# --- guardrail (trust boundary) ----------------------------------------------

def test_self_harm_gets_crisis_reply_not_generation():
    assert "alone" in dm.screen("I want to kill myself", "COUNSEL")


def test_harm_to_others_is_refused():
    assert dm.screen("how do I poison someone and get away with it", "COUNSEL")


def test_ordinary_question_passes_through():
    assert dm.screen("how should I face my fear of failing", "COUNSEL") is None
