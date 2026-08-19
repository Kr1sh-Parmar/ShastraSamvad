"""How much of the Guru's answer reaches the student, and why the rest does not.

    python -m uvicorn backend.main:app --port 8000     # in another shell
    python scripts/06_measure_turns.py                 # 12 turns, ~3 minutes
    python scripts/06_measure_turns.py before.json     # ...and save the detail

Every threshold and prompt in this project is supposed to be set by measurement.
This is the instrument for the ones that are about whole turns rather than single
sentences: change a prompt or a floor, run this before and after, and compare.

It reports, per turn and in total:

  spoken       sentences that passed the verifier and were spoken
  withheld     and why, by category — `unsupported` means an uncited sentence
               that landed near no retrieved verse, which is the guardrail
               working; `invented anchor` on a verse that WAS retrieved is a bug
  echoed       sentences reproducing a retrieved passage verbatim rather than
               teaching from it. Legitimate to the verifier (a verse quoted
               exactly is supported by that verse) but poor teaching.
  empty turns  turns that fell through to the no-scriptural-basis message

READ THE VARIANCE BEFORE BELIEVING A DELTA. At TEMPERATURE 0.4, with no change
to the code at all, twelve turns give a spoken rate anywhere in 48-62% and an
echo rate anywhere in 0-20%. Echo especially has no stable shape: the one turn
responsible for most of a run's echoes produced none across the next three. A
before-and-after pair of runs therefore tells you nothing about a prompt edit.

Categorical changes are the trustworthy ones. `invented anchor` going 3 -> 0 and
staying there over three runs is what caught the anchor-case bug — a whole
rejection reason appearing or vanishing means something; a rate moving does not.
"""
import asyncio
import json
import sys
from collections import Counter

import websockets

WS = "ws://localhost:8000/chat"
# Two turns per mode, plus a lesson continued, so the set exercises the state
# machine and not just one-shot questions. Both texts, since a Mahabharata
# passage is long narrative prose where a Gita verse is a couplet.
CASES = [
    ("TEACH", "teach me chapter 2", {"source": "Gita", "chapter": 2}),
    ("TEACH", "continue", {"source": "Gita", "chapter": 2}),
    ("TEACH", "teach me the Adi Parva", {"source": "Mahabharata", "chapter": "Adi"}),
    ("TEACH", "continue", {"source": "Mahabharata", "chapter": "Adi"}),
    ("DOUBT", "what does detachment actually mean here", {"source": "Gita"}),
    ("DOUBT", "why does Krishna tell him to fight", {"source": "Gita"}),
    ("DEBATE", "I disagree that fighting can ever be right", {"source": "Gita"}),
    ("DEBATE", "surely duty cannot justify killing kinsmen", {"source": "Gita"}),
    ("COUNSEL", "I am afraid of failing at my work", {}),
    ("COUNSEL", "my family wants me to take a job I hate", {}),
    ("COUNSEL", "I cannot forgive someone who wronged me", {}),
    ("COUNSEL", "what happened when Krishna sought peace with the Kauravas", {}),
]
RUN = 12                 # a shared 12-word run is a quotation, not a paraphrase


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def echoes(sentence: str, verses: list[str]) -> bool:
    """Is this sentence lifted from a retrieved passage rather than written?"""
    words = _norm(sentence).split()
    if len(words) < RUN:
        return False
    blob = " ".join(_norm(v) for v in verses)
    return any(" ".join(words[i:i + RUN]) in blob
               for i in range(len(words) - RUN + 1))


async def run_case(mode: str, text: str, kw: dict) -> tuple[list, list, list]:
    # One connection per case: a fresh dialogue state, so a turn is not judged
    # on history left behind by the case before it.
    async with websockets.connect(WS, max_size=None) as ws:
        await ws.send(json.dumps({"type": "ask", "text": text, "mode": mode, **kw}))
        verses, spoken, withheld = [], [], []
        while True:
            msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=300))
            if msg["type"] == "verses":
                verses = [v["text"] for v in msg["items"]]
            elif msg["type"] == "sentence":
                spoken.append(msg["text"])
            elif msg["type"] == "withheld":
                withheld.append((msg["reason"].split(" (")[0], msg["text"]))
            elif msg["type"] == "done":
                return verses, spoken, withheld


async def main() -> None:
    cases, totals = [], Counter()
    for mode, text, kw in CASES:
        verses, spoken, withheld = await run_case(mode, text, kw)
        n = len(spoken) + len(withheld)
        echoed = (sum(1 for s in spoken if echoes(s, verses))
                  + sum(1 for _, s in withheld if echoes(s, verses)))
        totals.update({"sentences": n, "spoken": len(spoken),
                       "withheld": len(withheld), "echoed": echoed})
        for reason, _ in withheld:
            totals[f"why:{reason}"] += 1
        cases.append({"mode": mode, "text": text, "spoken": spoken,
                      "withheld": withheld, "echoed": echoed})
        print(f"{mode:8s} {len(spoken):2d}/{n:2d} spoken  echo={echoed}  {text[:44]}")

    n = totals["sentences"] or 1
    empty = sum(1 for c in cases if not c["spoken"])
    print(f"\n  spoken      {totals['spoken']:3d}/{n} ({totals['spoken'] / n:.0%})"
          f"   [48-60% is the observed range with no code change]")
    print(f"  echoed      {totals['echoed']:3d}/{n} ({totals['echoed'] / n:.0%})")
    print(f"  empty turns {empty:3d}/{len(cases)}")
    for key in sorted(k for k in totals if k.startswith("why:")):
        print(f"    {key[4:]:34s} {totals[key]}")
    if any(k.startswith("why:invented") for k in totals):
        print("\n  NOTE: `invented anchor` naming a verse that WAS retrieved is a "
              "parsing bug, not the model fabricating. Check the case and "
              "brackets against citation_verifier.anchors().")

    if len(sys.argv) > 1:
        with open(sys.argv[1], "w", encoding="utf-8") as f:
            json.dump({"totals": dict(totals), "cases": cases}, f, indent=1)
        print(f"\ndetail -> {sys.argv[1]}")


asyncio.run(main())
