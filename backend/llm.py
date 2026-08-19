"""Streaming client for Ollama. It's a JSON-lines HTTP stream — no framework."""
import json
from typing import AsyncIterator

import httpx

from . import config


async def generate(prompt: dict, think: bool = False) -> AsyncIterator[str]:
    """Yields answer tokens as they arrive (contract: arch doc 8).

    Uses /api/chat, not /api/generate: with roles, the model answers instead of
    continuing text — given one blob ending in "GURU:", a small model narrates
    its own instructions before answering, and that reaches the speaker.

    Only `message.content` is yielded. If LLM_MODEL is ever swapped back to a
    reasoning model, set think=True so Ollama routes its deliberation to the
    separate `thinking` field instead of into the reply.

    Cancel by closing the generator (the barge-in path does exactly that).
    """
    messages = [{"role": "system", "content": prompt["system"]}]
    messages += prompt.get("history", [])
    messages.append({"role": "user", "content": prompt["user"]})

    payload = {
        "model": config.LLM_MODEL,
        "messages": messages,
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


async def warm() -> None:
    """Pull the model into memory now rather than during the first question.

    A cold phi4-mini load is seconds — more than the whole rest of the pipeline
    (retrieval 13ms, verifier 0.2s/turn). Generation sends keep_alive so an idle
    gap doesn't undo this.

    Failure is fine and silent: Ollama may not be running, and /health says so.
    """
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


async def health() -> bool:
    try:
        async with httpx.AsyncClient(timeout=2.0) as c:
            return (await c.get(f"{config.OLLAMA_URL}/api/tags")).status_code == 200
    except Exception:
        return False
