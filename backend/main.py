"""FastAPI orchestrator: the async loop tying the stages together (arch doc 4.10)."""
import asyncio
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from . import (citation_verifier, config, crossref, dialogue_manager, embed, llm,
               nli, prompt_builder, retriever, stt, tts)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load every model before the first question, not during it.

    The first cold encode takes ~30s. Inside a turn that blocks the event loop
    long enough that the client's WebSocket keepalive times out and kills the
    connection — the turn never gets to answer. The LLM is the same story
    without the disconnect: seconds of cold load on the first thing the student
    asks, which is the one answer a demo is judged on.
    """
    await asyncio.to_thread(retriever.all_records)
    await asyncio.to_thread(embed.encode, "warm")
    await asyncio.to_thread(nli.contradicts, "warm", "warm")
    # Whisper too, for the same reason: it was the one model left loading inside
    # a turn, so the first press of the mic paid the cold load — and the mic is
    # how the device is meant to be used at all.
    await asyncio.to_thread(stt.model)
    await llm.warm()
    yield


app = FastAPI(title="Shastra Samvad", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {
        "ok": True,
        "index": config.FAISS_INDEX.exists(),
        "ollama": await llm.health(),
        "model": config.LLM_MODEL,
        "whisper": stt.available(),
        "piper": tts.available(),
        # False until a Mahabharata corpus exists — crossref.expand then returns
        # [] on every turn, which is correct but otherwise invisible.
        "crossref": config.CROSSREF_MAP.exists(),
    }


@app.get("/texts")
async def texts():
    """Chapter / parva list for the pickers."""
    out: dict[str, dict] = {}
    for r in retriever.all_records():
        src = out.setdefault(r["source"], {"source": r["source"], "chapters": {}})
        if r["source"] == "Gita":
            key, order = r.get("chapter"), r.get("chapter")
        else:
            # By book number, not alphabetically: a reader picks the Adi Parva
            # to start at the beginning, and sorting by name opens with Adi,
            # Anusasana, Asramavasika — books 1, 13 and 15.
            key, order = r.get("parva"), r.get("book")
        if key is not None:
            src["chapters"].setdefault(key, order)
    return [
        {"source": s["source"],
         "chapters": [c for c, _ in sorted(s["chapters"].items(),
                                           key=lambda kv: (kv[1] is None, kv[1]))]}
        for s in out.values()
    ]


@app.get("/verse/{verse_id}")
async def verse(verse_id: str):
    rec = retriever.get(verse_id)
    if not rec:
        raise HTTPException(404, f"no verse {verse_id}")
    return rec


@app.post("/stt")
async def transcribe(audio: UploadFile):
    # Read bounded, not all-at-once: this is the network handing us bytes to
    # hold in memory, so the cap has to apply before the whole body exists.
    parts, total = [], 0
    while chunk := await audio.read(1 << 20):
        total += len(chunk)
        if total > config.MAX_AUDIO_BYTES:
            raise HTTPException(413, "audio too large")
        parts.append(chunk)
    if not total:
        raise HTTPException(400, "empty audio")
    return await asyncio.to_thread(stt.transcribe, b"".join(parts))


@app.post("/tts")
async def speak(payload: dict):
    wav = await asyncio.to_thread(tts.synth, payload.get("text", ""))
    if wav is None:
        return {"browser_tts": True}      # client speaks it with speechSynthesis
    return Response(wav, media_type="audio/wav")


# Sentence audio waiting to be collected over GET /tts/{id}, rather than
# base64'd into the WebSocket frame that announces the sentence. Inline, a
# 283-char sentence is ~0.9 MB on the wire and nothing reaches the student until
# Piper has finished synthesizing — which is the sentence-by-sentence streaming
# the latency budget assumes (arch doc 6) given away for a shortcut.
#
# ponytail: a dict in this process, not Redis or a temp dir. One backend serves
# one student; the audio is worthless the moment it has been played once.
_AUDIO_CACHE_SIZE = 32
_audio: OrderedDict[str, bytes | None] = OrderedDict()
_audio_ready: dict[str, asyncio.Event] = {}
# asyncio holds only a weak reference to a running task, so one that nothing
# else refers to can be collected mid-synthesis and the audio just never arrives.
_synth_tasks: set[asyncio.Task] = set()


async def _synth_into_cache(clip_id: str, text: str, ready: asyncio.Event) -> None:
    """Synthesize in the background so the sentence frame need not wait on it."""
    try:
        _audio[clip_id] = await asyncio.to_thread(tts.synth, text)
    except Exception:
        _audio[clip_id] = None            # the GET 404s; the turn carries on
    finally:
        # The event is passed in rather than looked up: eviction below may
        # already have taken this id out of the map.
        ready.set()
        while len(_audio) > _AUDIO_CACHE_SIZE:
            old, _ = _audio.popitem(last=False)
            evicted = _audio_ready.pop(old, None)
            if evicted:
                evicted.set()     # or a GET still waiting on it never returns


@app.get("/tts/{clip_id}")
async def clip(clip_id: str):
    ready = _audio_ready.get(clip_id)
    if ready is None:
        raise HTTPException(404, "no such clip")
    # The client can ask before Piper has finished — it is told the URL first on
    # purpose. Wait for the synthesis rather than 404 on the race.
    await ready.wait()
    wav = _audio.get(clip_id)
    if not wav:
        raise HTTPException(404, "no audio for that clip")
    return Response(wav, media_type="audio/wav")


def _unit(value):
    """The lesson unit: a Gita chapter number, or a Mahabharata parva name.

    Normalized here, once, because dialogue_manager compares (source, unit)
    against the lesson in progress to decide whether to restart it — and "2"
    arriving where 2 was stored would silently reopen chapter 2 at verse 1.
    """
    if isinstance(value, str) and value.strip().isdigit():
        return int(value)
    return value or None


async def _send_sentence(ws: WebSocket, text: str, cited_ids: list[str]) -> None:
    """Announce a sentence now; let Piper catch up behind it.

    Synthesis starts but is not awaited, so the sentence is on screen while
    Piper is still working and the client collects the audio from the URL.
    audio_url is None when Piper is absent — the client's signal to speak it
    with speechSynthesis instead.

    Every spoken line goes through here, including the refusal and the
    no-scriptural-basis message. Those used to be handed straight to the browser
    voice, so the one reply that matters most — the crisis reply — came out in a
    different voice from the rest of the Guru.
    """
    audio_url = None
    if tts.available():
        clip_id = uuid.uuid4().hex
        ready = _audio_ready[clip_id] = asyncio.Event()
        _audio[clip_id] = None
        task = asyncio.create_task(_synth_into_cache(clip_id, text, ready))
        _synth_tasks.add(task)
        task.add_done_callback(_synth_tasks.discard)
        audio_url = f"/tts/{clip_id}"
    await ws.send_json({"type": "sentence", "text": text,
                        "cited_ids": cited_ids, "audio_url": audio_url})


async def _run_turn(ws: WebSocket, msg: dict, state: dict) -> None:
    """One full turn: route -> retrieve -> prompt -> generate -> verify -> speak."""
    query = (msg.get("text") or "").strip()
    if not query:
        return

    state["source"] = msg.get("source") or state.get("source")
    state["chapter"] = _unit(msg.get("chapter")) or state.get("chapter")
    state["forced_mode"] = msg.get("mode") if msg.get("mode") != "AUTO" else None

    routed = dialogue_manager.route(query, state)
    mode, state = routed["mode"], routed["state"]
    await ws.send_json({"type": "mode", "mode": mode})

    refusal = dialogue_manager.screen(query, mode)
    if refusal:
        await ws.send_json({"type": "verses", "items": []})
        await _send_sentence(ws, refusal, [])
        await ws.send_json({"type": "citations", "ids": []})
        await ws.send_json({"type": "done", "refused": True})
        return

    verses = await asyncio.to_thread(retriever.search, query, mode, state)
    verses += await asyncio.to_thread(
        crossref.expand, [v["verse_id"] for v in verses]
    )
    await ws.send_json({"type": "verses", "items": verses})

    if mode == "TEACH" and verses:
        state["current_verse_text"] = verses[0]["text"]

    if not verses:
        await _send_sentence(ws, config.NO_BASIS_MESSAGE, [])
        await ws.send_json({"type": "citations", "ids": []})
        await ws.send_json({"type": "done"})
        return

    prompt = prompt_builder.build(mode, verses, state["history"], query, state)

    # ponytail: sentences are emitted, not raw tokens. The verifier gates each
    # sentence, so streaming tokens would put unverified claims on screen — the
    # one thing "no verse, no answer" exists to prevent. First sentence lands at
    # ~2s either way (arch doc 6), so nothing is lost but the typewriter effect.
    buffer, spoken, cited = "", [], []
    async for token in llm.generate(prompt):
        buffer += token
        done, buffer = citation_verifier.split_stream(buffer)
        for sentence in done:
            await _emit_sentence(ws, sentence, verses, spoken, cited)

    # Whatever is left is only a sentence if the model finished it. Hitting
    # MAX_TOKENS mid-clause used to speak the stump — "Arjuna's attachment was
    # rooted in his concern both for" — which the verifier happily passed,
    # because a truncated true statement is still true.
    if buffer.strip().endswith((".", "!", "?", '."', ".'", '!"', '?"')):
        await _emit_sentence(ws, buffer, verses, spoken, cited)

    if not spoken:
        await _send_sentence(ws, config.NO_BASIS_MESSAGE, [])

    answer = " ".join(spoken) or config.NO_BASIS_MESSAGE
    state["history"] += [{"role": "student", "text": query},
                         {"role": "guru", "text": answer}]
    await ws.send_json({"type": "citations", "ids": cited})
    await ws.send_json({"type": "done"})


async def _emit_sentence(ws: WebSocket, sentence: str, verses: list[dict],
                         spoken: list[str], cited: list[str]) -> None:
    result = await asyncio.to_thread(citation_verifier.check, sentence, verses)
    if not result["ok"]:
        await ws.send_json({"type": "withheld", "text": sentence.strip(),
                            "reason": result["reason"]})
        return

    clean = citation_verifier.speakable(sentence)
    # A small model can lock into a loop and re-emit the same sentence until it
    # hits MAX_TOKENS. Measured on a DEBATE turn: phi4-mini produced one good
    # sentence and then six identical copies of the next, and the verifier
    # passed every copy — each one is supported, it is only the repetition that
    # is wrong, and nothing downstream of here can see that. Caught once, where
    # every sentence already passes through.
    if clean in spoken:
        return
    spoken.append(clean)
    for cid in result["cited_ids"]:
        if cid not in cited:
            cited.append(cid)
    await _send_sentence(ws, clean, result["cited_ids"])


@app.websocket("/chat")
async def chat(ws: WebSocket):
    await ws.accept()
    state = dialogue_manager.new_state()
    turn: asyncio.Task | None = None
    try:
        while True:
            msg = await ws.receive_json()
            # Barge-in: the user talking over the Guru cancels generation mid-flight.
            if msg.get("type") == "interrupt":
                if turn and not turn.done():
                    turn.cancel()
                await ws.send_json({"type": "interrupted"})
                continue
            if msg.get("type") == "ask":
                if turn and not turn.done():
                    turn.cancel()
                turn = asyncio.create_task(_run_turn(ws, msg, state))
    except WebSocketDisconnect:
        pass
    finally:
        if turn and not turn.done():
            turn.cancel()
