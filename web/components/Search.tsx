"use client";

import { useState } from "react";
import { api, type JobHit, type SearchMode } from "@/lib/api";

const MODES: { value: SearchMode; label: string }[] = [
  { value: "hybrid_rerank", label: "Hybrid + rerank" },
  { value: "hybrid", label: "Hybrid (RRF)" },
  { value: "vector", label: "Vector" },
  { value: "keyword", label: "Keyword" },
];

export default function Search({ user }: { user: string }) {
  const [q, setQ] = useState("");
  const [mode, setMode] = useState<SearchMode>("hybrid_rerank");
  const [remote, setRemote] = useState(false);
  const [hits, setHits] = useState<JobHit[] | null>(null);
  const [timings, setTimings] = useState<Record<string, number>>({});
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [saved, setSaved] = useState<Set<number>>(new Set());

  async function run(e: React.FormEvent) {
    e.preventDefault();
    if (!q.trim()) return;
    setBusy(true);
    setErr(null);
    try {
      const r = await api.search(q, mode, remote);
      setHits(r.results);
      setTimings(r.timings_ms);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function save(id: number) {
    try {
      await api.feedback(user, id, "save");
      setSaved((s) => new Set(s).add(id));
    } catch (e) {
      setErr(String(e));
    }
  }

  return (
    <section>
      <form onSubmit={run} className="row">
        <input type="text" placeholder="e.g. remote kafka data engineer, or 'nursing jobs in Chicago'" value={q}
          onChange={(e) => setQ(e.target.value)} aria-label="Search query" />
        <select value={mode} onChange={(e) => setMode(e.target.value as SearchMode)} aria-label="Search mode">
          {MODES.map((m) => <option key={m.value} value={m.value}>{m.label}</option>)}
        </select>
        <label className="check"><input type="checkbox" checked={remote} onChange={(e) => setRemote(e.target.checked)} /> Remote</label>
        <button className="primary" disabled={busy}>{busy ? "Searching…" : "Search"}</button>
      </form>
      {err && <p className="error">{err}</p>}
      {hits && (
        <p className="note">
          {hits.length} results · {Object.entries(timings).map(([k, v]) => `${k} ${Math.round(v)}ms`).join(" · ")}
        </p>
      )}
      {hits?.map((h) => (
        <article key={h.job_id} className="card">
          <h3>{h.url ? <a href={h.url} target="_blank" rel="noreferrer">{h.title}</a> : h.title}</h3>
          <p className="meta">
            {h.company ?? "Company not listed"} · {h.location ?? "Location not listed"}
            {h.is_remote ? " · remote" : ""}{h.seniority ? ` · ${h.seniority}` : ""} · job {h.job_id}
          </p>
          <p className="snippet">{h.snippet}</p>
          <div className="row" style={{ justifyContent: "space-between" }}>
            <span className="signals">
              {h.keyword_rank ? `keyword #${h.keyword_rank}` : "keyword –"} · {h.vector_rank ? `vector #${h.vector_rank}` : "vector –"}
              {h.rerank_score != null ? ` · rerank ${h.rerank_score.toFixed(2)}` : ""}
            </span>
            <button className="secondary" onClick={() => save(h.job_id)} disabled={saved.has(h.job_id)}>
              {saved.has(h.job_id) ? "Saved" : "Save"}
            </button>
          </div>
        </article>
      ))}
    </section>
  );
}
