"use client";

import { Fragment, useEffect, useRef, useState } from "react";
import { API, chatStream } from "@/lib/api";

interface Msg {
  role: "user" | "assistant";
  text: string;
  tools?: string[];
  meta?: string;
}

interface Job {
  id: number;
  title: string;
  company: string | null;
  location: string | null;
  url: string | null;
  description: string;
}

const newThread = () => Math.random().toString(36).slice(2, 10);

/** Bold (**x**) and [job N] citations; everything else stays plain text (white-space: pre-wrap). */
function Rich({ text, onCite }: { text: string; onCite: (id: number) => void }) {
  const parts = text.split(/(\*\*[^*]+\*\*|\[job\s*#?\d+\])/gi);
  return (
    <>
      {parts.map((p, i) => {
        const cite = p.match(/^\[job\s*#?(\d+)\]$/i);
        if (cite) {
          return (
            <button key={i} className="cite" onClick={() => onCite(Number(cite[1]))}>
              job {cite[1]}
            </button>
          );
        }
        if (p.startsWith("**") && p.endsWith("**")) return <strong key={i}>{p.slice(2, -2)}</strong>;
        return <Fragment key={i}>{p}</Fragment>;
      })}
    </>
  );
}

export default function Chat({ user }: { user: string }) {
  const [thread, setThread] = useState<string>("");
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [draft, setDraft] = useState<Msg | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const bottom = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let t = "";
    try {
      t = localStorage.getItem(`sjr-thread-${user}`) ?? "";
    } catch {}
    if (!t) t = newThread();
    setThread(t);
  }, [user]);

  useEffect(() => {
    if (!thread) return;
    try {
      localStorage.setItem(`sjr-thread-${user}`, thread);
    } catch {}
    fetch(`${API}/chat/${encodeURIComponent(user)}/${thread}`)
      .then((r) => (r.ok ? r.json() : { messages: [] }))
      .then((d) => setMsgs(d.messages))
      .catch(() => setMsgs([]));
  }, [user, thread]);

  useEffect(() => bottom.current?.scrollIntoView({ behavior: "smooth" }), [msgs, draft]);

  async function openJob(id: number) {
    const r = await fetch(`${API}/jobs/${id}`);
    if (r.ok) setJob(await r.json());
  }

  async function send(e: React.FormEvent) {
    e.preventDefault();
    const text = input.trim();
    if (!text || busy) return;
    setInput("");
    setBusy(true);
    setMsgs((m) => [...m, { role: "user", text }]);
    let cur: Msg = { role: "assistant", text: "", tools: [] };
    setDraft(cur);
    for await (const ev of chatStream(user, thread, text)) {
      if (ev.type === "token") cur = { ...cur, text: cur.text + ev.text };
      else if (ev.type === "tool_call") {
        const args = Object.entries(ev.args).map(([k, v]) => `${k}=${JSON.stringify(v)}`).join(", ");
        cur = { ...cur, text: "", tools: [...(cur.tools ?? []), `${ev.name}(${args})`] };
      } else if (ev.type === "retract") cur = { ...cur, text: "", tools: [...(cur.tools ?? []), "↺ draft cited an unknown job; regenerating"] };
      else if (ev.type === "final") {
        cur = { ...cur, text: ev.text, meta: `${(ev.latency_ms / 1000).toFixed(1)}s · ${ev.usage.llm_calls} LLM calls` };
      } else if (ev.type === "error") cur = { ...cur, text: `⚠ ${ev.message}` };
      setDraft(cur);
    }
    setMsgs((m) => [...m, cur]);
    setDraft(null);
    setBusy(false);
  }

  const shown = draft ? [...msgs, draft] : msgs;
  return (
    <section>
      <div className="chat">
        {shown.length === 0 && (
          <p className="note">
            Try: “What jobs fit my background?”, “Find remote data engineer jobs”, “What am I missing for the first one?”,
            “Remember I only want remote roles”.
          </p>
        )}
        {shown.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            {m.tools?.map((t, j) => <div key={j} className="tool">→ {t}</div>)}
            {m.role === "assistant" ? <Rich text={m.text} onCite={openJob} /> : m.text}
            {m.meta && <div className="tool" style={{ marginTop: 6 }}>{m.meta}</div>}
          </div>
        ))}
        <div ref={bottom} />
      </div>
      {job && (
        <article className="card" style={{ marginBottom: 12 }}>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <h3>{job.url ? <a href={job.url} target="_blank" rel="noreferrer">{job.title}</a> : job.title}</h3>
            <button className="secondary" onClick={() => setJob(null)} aria-label="Close job details">✕</button>
          </div>
          <p className="meta">{job.company ?? "Company not listed"} · {job.location ?? "Location not listed"} · job {job.id}</p>
          <p className="snippet">{job.description.slice(0, 600)}…</p>
        </article>
      )}
      <form onSubmit={send} className="row">
        <input type="text" value={input} onChange={(e) => setInput(e.target.value)} placeholder="Ask the career agent…"
          aria-label="Message" disabled={busy} />
        <button className="primary" disabled={busy || !input.trim()}>{busy ? "Thinking…" : "Send"}</button>
        <button type="button" className="secondary" onClick={() => { setThread(newThread()); setMsgs([]); setJob(null); }} disabled={busy}>
          New conversation
        </button>
      </form>
    </section>
  );
}
