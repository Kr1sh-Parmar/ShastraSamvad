"""Streaming client for the LLM. Two backends, gated by config.LLM_USE_OLLAMA.

Ollama (JSON-lines HTTP stream, no framework) is the dev/demo path — it needs
a daemon, which is fine on a laptop and not available on embedded hardware.
llama-cpp-python loads a GGUF file in-process instead, for boards with no
Ollama install: Jetson Nano 4GB (CPU, or CUDA via LLM_GPU_LAYERS) and any
generic Linux board.

RK3588 does NOT use this module's llama.cpp path at all — its NPU is driven
by rkllm's own Python bindings, a different runtime with a different model
format (.rkllm, converted from the GGUF/safetensors source). A board on
RK3588 would replace this module entirely with an rkllm-backed one behind
the same generate()/warm()/health() contract, not add a third branch here.
"""
import asyncio
import json
import queue
from typing import AsyncIterator

import httpx

from . import config

# --- llama.cpp backend: lazy singleton, loaded once and kept resident ---
_llama = None


def _load_llama():
    global _llama
    if _llama is None:
        import llama_cpp
        _llama = llama_cpp.Llama(
            model_path=str(config.LLM_MODEL_PATH),
            n_ctx=config.LLM_CONTEXT_SIZE,
            n_gpu_layers=config.LLM_GPU_LAYERS,
            verbose=False,
        )
    return _llama


def _messages(prompt: dict) -> list[dict]:
    """Same message shape both backends send: system, history, then user."""
    messages = [{"role": "system", "content": prompt["system"]}]
    messages += prompt.get("history", [])
    messages.append({"role": "user", "content": prompt["user"]})
    return messages


def _llama_stream_worker(messages: list[dict], q: "queue.Queue") -> None:
    """Runs on a worker thread — llama-cpp-python's create_chat_completion is
    a blocking generator, and blocking it inside the event loop would stall
    every other WebSocket turn in the process. Tokens cross to the async side
    over a plain Queue; None is the end-of-stream sentinel, an exception
    instance signals failure so the async side can raise it in-context.
    """
    try:
        llm = _load_llama()
        stream = llm.create_chat_completion(
            messages=messages,
            stream=True,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )
        for chunk in stream:
            delta = chunk["choices"][0].get("delta", {})
            token = delta.get("content")
            if token:
                q.put(token)
    except Exception as e:  # pragma: no cover - surfaced to the async caller
        q.put(e)
    finally:
        q.put(None)


async def _generate_llama(prompt: dict) -> AsyncIterator[str]:
    messages = _messages(prompt)
    q: "queue.Queue" = queue.Queue()
    worker = asyncio.get_event_loop().run_in_executor(
        None, _llama_stream_worker, messages, q)
    try:
        while True:
            item = await asyncio.to_thread(q.get)
            if item is None:
                return
            if isinstance(item, Exception):
                raise item
            yield item
    finally:
        # Cancel (barge-in) closes this generator; the worker thread still
        # runs create_chat_completion to completion in the background since
        # llama.cpp offers no mid-stream cancel, but its tokens are simply
        # never read once nothing is left consuming the queue.
        worker.cancel()


async def _generate_ollama(prompt: dict, think: bool) -> AsyncIterator[str]:
    """Uses /api/chat, not /api/generate: with roles, the model answers instead
    of continuing text — given one blob ending in "GURU:", a small model
    narrates its own instructions before answering, and that reaches the
    speaker.

    Only `message.content` is yielded. If LLM_MODEL is ever swapped back to a
    reasoning model, set think=True so Ollama routes its deliberation to the
    separate `thinking` field instead of into the reply.
    """
    payload = {
        "model": config.LLM_MODEL,
        "messages": _messages(prompt),
        "stream": True,
        "think": think,          # keeps reasoning out of `content` — see docstring
        "keep_alive": config.LLM_KEEP_ALIVE,
        "options": {
            "temperature": config.TEMPERATURE,
            "num_predict": config.MAX_TOKENS,
        },
    }
    timeout = httpx.Timeout(10.0, read=300.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("POST", f"{config.OLLAMA_URL}/api/chat",
                                 json=payload) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                token = (chunk.get("message") or {}).get("content")
                if token:
                    yield token
                if chunk.get("done"):
                    return


async def generate(prompt: dict, think: bool = False) -> AsyncIterator[str]:
    """Yields answer tokens as they arrive (contract: arch doc 8).

    Backend picked by config.LLM_USE_OLLAMA; the interface — an async
    generator of tokens, same input shape — is identical either way, so
    main.py's `async for token in llm.generate(prompt)` needs no changes.

    Cancel by closing the generator (the barge-in path does exactly that).
    """
    if config.LLM_USE_OLLAMA:
        async for token in _generate_ollama(prompt, think):
            yield token
    else:
        async for token in _generate_llama(prompt):
            yield token


async def warm() -> None:
    """Pull the model into memory now rather than during the first question.

    A cold phi4-mini load is seconds — more than the whole rest of the pipeline
    (retrieval 13ms, verifier 0.2s/turn). Ollama's generation call sends
    keep_alive so an idle gap doesn't undo this; the llama.cpp backend simply
    keeps its singleton resident once loaded, for as long as the process runs.

    Failure is fine and silent: Ollama may not be running (or the GGUF file
    may not exist yet), and /health says so.
    """
    if config.LLM_USE_OLLAMA:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, read=180.0)) as c:
                await c.post(f"{config.OLLAMA_URL}/api/chat", json={
                    "model": config.LLM_MODEL,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "keep_alive": config.LLM_KEEP_ALIVE,
                    "options": {"num_predict": 1},
                })
        except Exception:
            pass
    else:
        try:
            await asyncio.to_thread(_load_llama)
        except Exception:
            pass


async def health() -> bool:
    if config.LLM_USE_OLLAMA:
        try:
            async with httpx.AsyncClient(timeout=2.0) as c:
                return (await c.get(f"{config.OLLAMA_URL}/api/tags")).status_code == 200
        except Exception:
            return False
    else:
        try:
            await asyncio.to_thread(_load_llama)
            return True
        except Exception:
            return False
