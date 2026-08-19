# Shastra Samvad — laptop prototype

Offline, scripture-grounded Guru. Speak or type a question; get a streamed answer
that can only assert what the retrieved verses support, with the citations shown.

Same logical modules as the RK3588 device build (`software_architecture.md`) —
only the shells differ (Ollama→llama.cpp, faster-whisper→whisper.cpp, browser→kiosk).

## Run it

```bash
# 1. one-time: dependencies
pip install -r requirements.txt

# 2. one-time: LLM
#    install Ollama from https://ollama.com/download, then:
ollama pull phi4-mini
#    NOT qwen3:4b (webapp_context.md 12's first pick). It is a reasoning model:
#    on these rule-heavy prompts it burned 1800 tokens deliberating and produced
#    no answer, and with think=false it wrote the deliberation into the reply.

# 3. one-time: fetch the Mahabharata (~12 MB) and build the index
python scripts/00_fetch_mahabharata.py
python scripts/01_clean_texts.py
python scripts/02_chunk_and_tag.py
python scripts/03_build_embeddings.py   # ~17k chunks, several minutes on CPU
python scripts/04_build_crossref.py

# 4. one-time: the voice (~85 MB). Skip it and the browser speaks instead.
python scripts/05_fetch_piper.py

# 5. run
python -m uvicorn backend.main:app --port 8000      # backend
cd frontend && npm install && npm run dev           # UI on :3001
```

The embedder and the NLI cross-encoder download themselves on first use.

`GET /health` reports whether the index, Ollama, Whisper, Piper, and the
cross-reference map are each ready. Every model is loaded at startup, not on
first use — Whisper included, or the first press of the mic pays the cold load.

## Checks

```bash
python -m pytest backend/test_pipeline.py -q   # verifier, modes, guardrail
python -m backend.retriever                    # eyeball retrieval quality

# these two need the backend running
python scripts/smoke_chat.py --llm             # four real turns, end to end
python scripts/06_measure_turns.py             # how much of a turn is spoken
```

`06_measure_turns.py` is the instrument for anything measured across whole turns
rather than single sentences — spoken vs withheld, why, and whether the Guru
recited a verse instead of teaching it. Run it before and after a prompt or
threshold change. Read its warning about variance first: the spoken rate moves
between 48% and 60% run to run with no code change at all, so trust categorical
shifts (a whole rejection reason appearing or vanishing) over rate deltas.

## Layout

| Path | What |
|---|---|
| `scripts/00..04` | fetch → raw text → verse-anchored chunks → FAISS index → crossref |
| `scripts/05..06` | fetch the Piper voice; measure how much of a turn is spoken |
| `backend/retriever.py` | per-mode retrieval (TEACH sequential, DOUBT focused, DEBATE for+against, COUNSEL broad) |
| `backend/dialogue_manager.py` | intent classification, lesson state, harm guardrail |
| `backend/citation_verifier.py` | "no verse, no answer" — gates every sentence before it is spoken |
| `backend/main.py` | WS `/chat` orchestration, `/stt`, `/tts`, `/texts`, `/verse/{id}` |
| `frontend/app/page.jsx` | chat, pickers, citation panel, mic, barge-in |

## Corpus

16,551 chunks: 657 Gita verse translations + 1,322 purport chunks, and 14,572
Mahabharata passages across 2,108 sections in all 18 parvas.

**Gita** — `data/raw/bhagavad-gita-as-it-is.pdf`. Anchors (`gita_2_47`) come from
the PDF's own table of contents, and `01_clean_texts.py` refuses to emit anchors
if the TOC and the in-text `TEXT n` headers ever disagree.

**Mahabharata** — Ganguli translation, Project Gutenberg 15474–15477, fetched by
`00_fetch_mahabharata.py`. Anchors are `mbh_udyoga_33`.

sacred-texts.com, the source the spec names, is behind Cloudflare now (403).
Two things about the Gutenberg files are worth knowing:

- The parva names are inconsistent — `ADI PARVA` in volumes 1–2, `Karna-parva`
  in 3–4, and volume 3 uses both. But **every volume delimits its parvas with
  `BOOK n`, uniformly, across all 18**, so that is what the chunker splits on and
  the naming never has to be reconciled. Section headers still come in two
  styles (`SECTION I` and a bare integer on its own line); the chunker picks per
  book. Section counts come out at Ganguli's canonical numbers — Adi 236, Udyoga
  199, Karna 96, Santi 365.
- Each volume holds three to six parvas and section numbers restart at 1 in each,
  so a parva name read once per file would misattribute the rest and collide
  their anchors. Anchors are unique by construction and `02_chunk_and_tag.py`
  refuses to emit a duplicate.

The Project Gutenberg licence is cut out in `01_clean_texts.py`, as is the
translator front matter — left in, both are embedded and retrieved exactly like
scripture, and the Guru would cite the terms of use as though they were the
Mahabharata. A Gutenberg file whose `*** START/END OF ***` markers are missing is
refused rather than guessed at.

Chunks are capped at 1,200 characters and `pack()` now cuts inside a paragraph
when it has to. Ganguli runs a whole section of dialogue as one paragraph — the
longest is 60,770 characters — and bge-small truncates at 512 tokens (~2,000),
so such a chunk was indexed on its opening 3% with the rest unreachable. This
was quietly true of the Gita's longer purports too: they went from 977 chunks
to 1,322 once the cap was actually enforced.

Ganguli's footnotes are dropped. They are collected as numbered paragraphs at
the end of a section — "109. The Bengal reading of the second line is vicious."
— which is scholarly apparatus carrying the section's anchor, so left in they
were retrieved and cited as though Vyasa had written them. That turned out to be
5.65% of the Mahabharata text, not the 0.16% a chunk-level count suggested: the
footnotes are interleaved with real prose rather than isolated, so almost no
chunk was purely footnote but 297 were part-footnote.

## Cross-reference

`CROSSREF_MIN_SCORE` is 0.80, measured against the real corpus — the old 0.55
predated having a Mahabharata at all. Both texts are the same register, so
cosine runs high throughout and the whole distribution sits between 0.58 and
0.89; at 0.55 every one of the 1,979 Gita chunks kept a link, which is no filter.
Sampled by hand: 0.70 is noise ("there is no truth superior to Me" linked to
Bhima boasting he is the superior brother for carrying Arjuna), 0.78 is mixed,
0.80 and up is the thing the index exists for — Arjuna seeing his relatives
arrayed (1.26) links to Drona Parva 28, where he kills those same uncles.

710 of 1,979 chunks now carry an illustration and the rest correctly carry none.

Note that Bhishma Parva 25–42 *is* the Gita inside the epic, so some links are
Ganguli's translation of the same verse rather than a narrative illustration.
That reads as corroboration across two translations, which is no bad thing.

The BBT diacritic font stores Sanskrit letters in Latin-1 slots; those are folded
to plain ASCII (`Kṛṣṇa`→`Krishna`) so the embedder indexes and the TTS pronounces
what people actually say.

Note: this edition is BBT-copyrighted, unlike the public-domain Telang translation
named in the spec. Fine for prototyping; worth revisiting before anything ships.

## The verifier's two stages

Cosine similarity catches invented anchors, off-topic drift, and unsupported
assertions — but it scores *topic*, not entailment, so it waved through a
sentence that negated the verse it cited (measured: faithful paraphrase 0.840,
its exact negation 0.777, and no threshold separates them). A cross-encoder now
runs behind it on whatever cosine lets through.

Anchor parsing is lenient about form and strict about existence. Brackets or
parentheses, upper or lower case, prose references in either text — all are read
as citations, and every one of them must name a verse that was actually
retrieved. The leniency matters because a real citation misread as no citation
puts a faithful sentence under the stricter uncited floor and silences it:
measured over 52 sentences, the model writes `[MBH_SANTI_176_5]` in capitals
often enough that the case mismatch alone was rejecting verses it had been
handed, as though they were invented.

It rejects on **contradiction only** — never requires entailment. Faithful
teaching routinely reads as `neutral` to an NLI model (measured 0.81 and 0.99
on two correct paraphrases), so demanding entailment would have silenced the
honest answers along with the false ones.

Model choice was decided by measurement, not size: `nli-deberta-v3-xsmall` —
the obvious pick for the device — reads "let the results go" as abandoning the
duty, and so scored *"Do your duty and let the results go"*, the plainest true
reading of 2.47 there is, a contradiction at 0.970 against a weakest true
contradiction of 0.997. `nli-deberta-v3-small` separates the same 16 sentences
by 0.661. That trap is pinned in
`test_pipeline.py::test_the_plainest_reading_of_2_47_is_not_called_a_contradiction`
— if anyone swaps back to a smaller model, it fails there rather than in a
lesson.

## Sentence audio

Sentences are announced over the WebSocket with an `audio_url`, and the client
fetches the WAV from `GET /tts/{id}`. Synthesis starts but is not awaited, so the
sentence is on screen while Piper is still working.

It used to be base64'd inline — ~0.9 MB for a 283-char sentence, with nothing
sent at all until Piper had finished, which gave away the sentence-by-sentence
streaming the latency budget assumes (arch doc 4.9) for a shortcut. Non-browser
clients no longer need `max_size` raised either.

The clips live in a 32-entry dict in the backend process; one backend serves one
student, and a clip is worthless once played. A `GET` that arrives before
synthesis finishes waits for it rather than 404-ing, since the client is told the
URL first on purpose.

The browser plays them through a queue. Piper synthesizes faster than real time,
so sentence 2's audio lands while sentence 1 is still playing — `speechSynthesis`
queues for us but an `<audio>` element does not, and the two used to talk over
each other.

The refusal and the no-scriptural-basis message go through the same path. They
used to be handed straight to the browser voice, so the one reply that matters
most — the crisis reply — came out in a different voice from the rest of the Guru.

The interrupt button is enabled while audio is playing, not only while the turn
is generating. Moving synthesis off the turn meant `done` now arrives long
before the queued audio has finished, and keying the button off `busy` alone
left it dead for the whole tail of the answer — precisely when a student wants
to cut in. Verified in the browser: three sentences played strictly in sequence
(never two at once, ~50 ms apart), and interrupt stopped playback dead after the
turn had already reported `done`.

## Repeated sentences

`phi4-mini` can lock into a loop and re-emit a sentence until it hits
MAX_TOKENS. Measured on a DEBATE turn: one good sentence, then six identical
copies of the next — and the verifier passed every copy, correctly, because each
one on its own is supported by the verse it cites. Only the orchestrator can see
the repetition, so `main.py` drops a sentence it has already spoken this turn,
and `smoke_chat.py` asserts no turn repeats itself.

## Lessons

TEACH walks a Gita chapter verse by verse, or a Mahabharata parva section by
section — `retriever._lesson_chunks` picks the units, `_lesson_order` puts them
in order, and a section long enough to have been split across chunks is taught
in that order rather than interleaved. The picker sends its raw value, because
`Number("Udyoga")` is `NaN`, which `JSON.stringify` writes as `null`; the
backend turns a numeric string back into a number so a Gita chapter arriving as
`"2"` still matches the stored `2` and does not restart the lesson every turn.

A parva runs to 365 prose sections against a chapter's 47 verses, so a lesson
through one is long. That is the text's shape, not a defect.

## Still to do

- **The Guru sometimes recites a verse instead of teaching it.** The model
  reproduces a retrieved passage verbatim with an anchor bolted on — "(Gita
  1.36) Sin will overcome us if we slay such aggressors ... (Gita 1.36)". It
  passes the verifier legitimately, since a verse quoted exactly *is* supported
  by that verse, so this is style rather than grounding.

  Measured over seven runs of twelve turns: **0, 0, 2, 2, 7, 14 and 20%** of
  sentences. There is no stable pattern to it — the single turn that accounted
  for most of one run's echoes produced none at all across the next three, and
  it happens on short Gita verses as readily as on long Mahabharata narrative.
  The metric is noise-dominated at twelve turns, which is the actual finding:
  judging any prompt change against it needs many runs, not a before and after.

  This entry has been wrong twice — first claiming the problem was frequent on
  the strength of one instance seen in a browser, then that it was negligible on
  the strength of two quiet runs. Both were undersampled.

- **About half of each turn is withheld** — 48-60% of sentences reach the
  student, across four runs, with no turn ever falling through to the
  no-scriptural-basis message. Nearly all of the remainder is `unsupported`:
  uncited general life-advice that does not land within `CITATION_MIN_SIM` of
  any retrieved verse. That is the guardrail doing its job rather than a defect,
  and the lever worth pulling is getting the model to cite more, not lowering
  the floor. Run-to-run variance at temperature 0.4 is wide enough that any
  change here needs several runs to judge, not one.
