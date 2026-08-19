"use client";

import { useCallback, useEffect, useRef, useState } from "react";

const API = process.env.NEXT_PUBLIC_API ?? "http://localhost:8000";
const WS = API.replace(/^http/, "ws") + "/chat";
const MODES = ["AUTO", "TEACH", "DOUBT", "DEBATE", "COUNSEL"];

export default function Page() {
  const [texts, setTexts] = useState([]);
  const [source, setSource] = useState("Gita");
  const [chapter, setChapter] = useState("");
  const [mode, setMode] = useState("AUTO");
  const [turns, setTurns] = useState([]);
  const [verses, setVerses] = useState([]);
  const [cited, setCited] = useState([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState("connecting…");
  const [recording, setRecording] = useState(false);

  const ws = useRef(null);
  const logRef = useRef(null);
  const speaking = useRef(false);

  useEffect(() => {
    fetch(`${API}/texts`).then((r) => r.json()).then(setTexts).catch(() => {});
  }, []);

  // Reconnects on its own: restarting the backend used to leave the page dead
  // until a manual reload.
  useEffect(() => {
    let unmounted = false;
    let retry;
    let attempt = 0;

    const connect = () => {
      const sock = new WebSocket(WS);
      ws.current = sock;
      sock.onopen = () => {
        attempt = 0;
        setStatus("ready");
      };
      sock.onmessage = (e) => handle(JSON.parse(e.data));
      sock.onclose = () => {
        if (unmounted) return;
        setBusy(false); // a turn in flight died with the socket
        const wait = Math.min(1000 * 2 ** attempt++, 10000);
        setStatus(`disconnected — reconnecting in ${Math.round(wait / 1000)}s…`);
        retry = setTimeout(connect, wait);
      };
      // onerror always precedes onclose; let onclose own the retry.
      sock.onerror = () => sock.close();
    };
    connect();

    return () => {
      unmounted = true;
      clearTimeout(retry);
      ws.current?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    logRef.current?.scrollTo(0, logRef.current.scrollHeight);
  }, [turns]);

  function handle(msg) {
    switch (msg.type) {
      case "mode":
        setStatus(`${msg.mode.toLowerCase()} — consulting the texts…`);
        setTurns((t) => [...t, { who: "guru", mode: msg.mode, text: "", withheld: [] }]);
        break;
      case "verses":
        setVerses(msg.items);
        setCited([]);
        break;
      case "sentence":
        setStatus("speaking…");
        setTurns((t) => patchLast(t, (g) => ({ ...g, text: `${g.text} ${msg.text}`.trim() })));
        if (msg.cited_ids?.length) setCited((c) => [...new Set([...c, ...msg.cited_ids])]);
        speak(msg.text, msg.audio_url);
        break;
      case "withheld":
        // The verifier blocked this sentence. Showing it (greyed, not spoken)
        // is what makes the guardrail visible in a demo.
        setTurns((t) => patchLast(t, (g) => ({ ...g, withheld: [...g.withheld, msg] })));
        break;
      case "citations":
        setCited(msg.ids);
        break;
      case "done":
        setBusy(false);
        setStatus("ready");
        break;
      case "interrupted":
        setBusy(false);
        setStatus("interrupted — go ahead");
        break;
    }
  }

  const patchLast = (turns, fn) => {
    const i = turns.map((t) => t.who).lastIndexOf("guru");
    return i < 0 ? turns : turns.map((t, j) => (j === i ? fn(t) : t));
  };

  // --- speech out -----------------------------------------------------------
  // Sentences must be played one after another, not the moment each arrives.
  // Piper synthesizes faster than real time, so sentence 2's audio lands while
  // sentence 1 is still playing and the two talk over each other. speechSynthesis
  // queues for us; an <audio> element does not, so the queue lives here.
  const queue = useRef([]);
  const current = useRef(null);
  // `speaking` is a ref because the barge-in loop reads it every animation
  // frame; this mirrors it as state, purely so the interrupt button can be
  // enabled while the Guru is still talking. The turn reports `done` when
  // generation ends, but the queued audio plays on well past that — so keying
  // the button off `busy` alone left it dead for the whole tail of the answer,
  // which is exactly when a student wants to cut in.
  const [audible, setAudible] = useState(false);
  const voice = useCallback((on) => {
    speaking.current = on;
    setAudible(on);
  }, []);

  const playNext = useCallback(() => {
    const url = queue.current.shift();
    if (!url) {
      current.current = null;
      voice(false);
      return;
    }
    const el = new Audio(url);
    current.current = el;
    // Only advance if this clip is still the one playing. stopSpeaking can clear
    // the queue while play() is still pending; its rejection would otherwise
    // pull the next turn's first sentence forward, on top of itself.
    const advance = () => current.current === el && playNext();
    el.onended = advance;
    el.onerror = advance; // a clip that failed to synthesize must not stall the rest
    el.play().catch(advance);
  }, [voice]);

  const speak = useCallback(
    (text, audioUrl) => {
      voice(true);
      if (audioUrl) {
        queue.current.push(`${API}${audioUrl}`);
        if (!current.current) playNext();
        return;
      }
      if (typeof speechSynthesis === "undefined") return;
      const u = new SpeechSynthesisUtterance(text);
      u.rate = 0.95;
      u.onend = () => speechSynthesis.pending || voice(false);
      speechSynthesis.speak(u); // the browser queues these for us
    },
    [playNext, voice]
  );

  const stopSpeaking = useCallback(() => {
    if (typeof speechSynthesis !== "undefined") speechSynthesis.cancel();
    queue.current = [];
    current.current?.pause();
    current.current = null; // the identity guard in playNext reads this
    voice(false);
  }, [voice]);

  function interrupt() {
    stopSpeaking();
    ws.current?.send(JSON.stringify({ type: "interrupt" }));
  }

  // --- barge-in: the mic is already open, so watch its level while we speak ---
  // ponytail: an RMS threshold, not a VAD model. Move to silero server-side if
  // the real room proves noisy.
  useEffect(() => {
    let ctx, stream, raf;
    let loud = 0;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      } catch {
        return; // no mic permission: typing still works, barge-in just won't
      }
      ctx = new AudioContext();
      const analyser = ctx.createAnalyser();
      analyser.fftSize = 512;
      ctx.createMediaStreamSource(stream).connect(analyser);
      const buf = new Uint8Array(analyser.fftSize);

      const tick = () => {
        analyser.getByteTimeDomainData(buf);
        let sum = 0;
        for (const v of buf) sum += (v - 128) ** 2;
        const rms = Math.sqrt(sum / buf.length) / 128;
        loud = rms > 0.08 ? loud + 1 : 0;
        if (loud > 12 && speaking.current) {
          loud = 0;
          interrupt();
        }
        raf = requestAnimationFrame(tick);
      };
      tick();
    })();
    return () => {
      cancelAnimationFrame(raf);
      stream?.getTracks().forEach((t) => t.stop());
      ctx?.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // --- speech in ------------------------------------------------------------
  const recorder = useRef(null);
  async function toggleRecord() {
    if (recording) {
      recorder.current?.stop();
      setRecording(false);
      return;
    }
    stopSpeaking();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const rec = new MediaRecorder(stream);
    const parts = [];
    rec.ondataavailable = (e) => parts.push(e.data);
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      setStatus("transcribing…");
      const form = new FormData();
      form.append("audio", new Blob(parts, { type: "audio/webm" }), "turn.webm");
      const r = await fetch(`${API}/stt`, { method: "POST", body: form });
      const { text } = await r.json();
      setStatus("ready");
      if (text?.trim()) ask(text.trim());
    };
    recorder.current = rec;
    rec.start();
    setRecording(true);
  }

  // --- send -----------------------------------------------------------------
  function ask(text) {
    if (!text.trim() || ws.current?.readyState !== WebSocket.OPEN) return;
    stopSpeaking();
    setTurns((t) => [...t, { who: "student", text }]);
    setBusy(true);
    setStatus("thinking…");
    ws.current.send(
      JSON.stringify({
        // Raw, not Number(): a Gita chapter is "2" but a Mahabharata parva is
        // "Udyoga", and Number("Udyoga") is NaN, which JSON.stringify writes as
        // null — so the parva never arrived and TEACH fell back to search.
        // The backend normalizes a numeric string back to a number.
        type: "ask",
        text,
        source,
        chapter: chapter || null,
        mode,
      })
    );
    setInput("");
  }

  const chapters = texts.find((t) => t.source === source)?.chapters ?? [];

  return (
    <div className="app">
      <main>
        <header>
          <h1>
            Shastra Samvad
            <small>OFFLINE · GURU–SHISHYA</small>
          </h1>
          <select
            value={source}
            onChange={(e) => {
              setSource(e.target.value);
              setChapter(""); // chapter 2 is not a parva, and vice versa
            }}
          >
            {(texts.length ? texts.map((t) => t.source) : ["Gita"]).map((s) => (
              <option key={s}>{s}</option>
            ))}
          </select>
          <select value={chapter} onChange={(e) => setChapter(e.target.value)}>
            <option value="">{source === "Gita" ? "any chapter" : "any parva"}</option>
            {chapters.map((c) => (
              <option key={c} value={c}>
                {source === "Gita" ? `Chapter ${c}` : `${c} Parva`}
              </option>
            ))}
          </select>
          <select value={mode} onChange={(e) => setMode(e.target.value)}>
            {MODES.map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        </header>

        <div className="log" ref={logRef}>
          {turns.map((t, i) =>
            t.who === "student" ? (
              <div key={i} className="turn student">
                <div className="who">You</div>
                {t.text}
              </div>
            ) : (
              <div key={i} className="turn guru">
                <div className="who">
                  Guru
                  {t.mode && <span className="badge">{t.mode}</span>}
                </div>
                {t.text || <span style={{ color: "var(--dim)" }}>…</span>}
                {t.withheld.map((w, j) => (
                  <div key={j} className="withheld">
                    withheld — {w.reason}: “{w.text}”
                  </div>
                ))}
              </div>
            )
          )}
        </div>

        <div className="status">{status}</div>

        <div className="composer">
          <button className={recording ? "rec" : ""} onClick={toggleRecord}>
            {recording ? "◼ stop" : "🎙 speak"}
          </button>
          <input
            value={input}
            placeholder="Ask the Guru…"
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && ask(input)}
          />
          <button onClick={() => ask(input)} disabled={busy || !input.trim()}>
            ask
          </button>
          <button onClick={interrupt} disabled={!busy && !audible}>
            interrupt
          </button>
        </div>
      </main>

      <aside>
        <h2>Verses consulted</h2>
        {verses.length === 0 && <p style={{ color: "var(--dim)", fontSize: 14 }}>none yet</p>}
        {verses.map((v) => (
          <div key={v.verse_id} className={`verse ${cited.includes(v.verse_id) ? "cited" : ""}`}>
            <span className="ref">
              {v.verse_id}
              {v.illustrates ? " · illustration" : ""}
            </span>
            <p>{v.text.slice(0, 230)}{v.text.length > 230 ? "…" : ""}</p>
          </div>
        ))}
      </aside>
    </div>
  );
}
