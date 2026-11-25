"use client";

import { useCallback, useEffect, useState } from "react";
import { api, type Explanation, type Profile, type RecommendResponse } from "@/lib/api";

export default function ForYou({ user }: { user: string }) {
  const [profile, setProfile] = useState<Profile | null>(null);
  const [recs, setRecs] = useState<RecommendResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [explanations, setExplanations] = useState<Record<number, Explanation | "loading" | string>>({});

  const load = useCallback(async () => {
    setErr(null);
    try {
      setProfile(await api.profile(user));
      setRecs(await api.recommend(user));
    } catch (e) {
      setProfile(null);
      setRecs(null);
      if (!String(e).startsWith("404")) setErr(String(e));
    }
  }, [user]);

  useEffect(() => {
    load();
  }, [load]);

  async function upload(file: File) {
    setBusy(true);
    setErr(null);
    try {
      await api.uploadResume(user, file);
      await load();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  async function feedback(id: number, action: "save" | "dismiss") {
    try {
      await api.feedback(user, id, action);
      setRecs(await api.recommend(user)); // re-rank with the new signal
    } catch (e) {
      setErr(String(e));
    }
  }

  async function why(id: number) {
    setExplanations((x) => ({ ...x, [id]: "loading" }));
    try {
      const e = await api.explain(user, id);
      setExplanations((x) => ({ ...x, [id]: e }));
    } catch (e) {
      setExplanations((x) => ({ ...x, [id]: String(e) }));
    }
  }

  return (
    <section>
      <div className="row">
        <label className="secondary" style={{ padding: "9px 14px", border: "1px solid var(--border)", borderRadius: 10, cursor: "pointer" }}>
          {busy ? "Reading resume…" : profile ? "Replace resume" : "Upload resume (PDF, MD, TXT)"}
          <input type="file" accept=".pdf,.md,.txt" hidden disabled={busy}
            onChange={(e) => e.target.files?.[0] && upload(e.target.files[0])} />
        </label>
      </div>
      {err && <p className="error">{err}</p>}
      {!profile && !busy && <p className="note">No resume on file for “{user}”. Upload one to get recommendations.</p>}
      {profile && (
        <div className="card">
          <h3>{profile.profile.headline}</h3>
          <p className="meta">
            {profile.profile.seniority} · {profile.profile.years_experience} years · targets {profile.profile.target_roles.join(", ")}
          </p>
          <div className="chips">{profile.canonical_skills.map((s) => <span key={s} className="chip">{s}</span>)}</div>
        </div>
      )}
      {recs && (
        <p className="note">
          Feedback used: {recs.feedback_used.save ?? 0} saved, {recs.feedback_used.dismiss ?? 0} dismissed · query moved{" "}
          {(recs.query_shift * 100).toFixed(1)}% · {Math.round(recs.timings_ms.total)} ms
        </p>
      )}
      {recs?.results.map((r) => {
        const ex = explanations[r.job_id];
        return (
          <article key={r.job_id} className="card">
            <h3>{r.url ? <a href={r.url} target="_blank" rel="noreferrer">{r.title}</a> : r.title}</h3>
            <p className="meta">
              {r.company ?? "Company not listed"} · {r.location ?? "Location not listed"}{r.is_remote ? " · remote" : ""} · job {r.job_id}
            </p>
            <div className="chips">
              {r.matched_skills.map((s) => <span key={s} className="chip good">✓ {s}</span>)}
              {r.missing_skills.map((s) => <span key={s} className="chip bad">✗ {s}</span>)}
            </div>
            <div className="row" style={{ justifyContent: "space-between", marginTop: 8 }}>
              <span className="signals">
                score {r.score.toFixed(3)} · similarity {r.similarity.toFixed(3)}
                {r.skill_coverage != null ? ` · skills ${(r.skill_coverage * 100).toFixed(0)}%` : ""}
              </span>
              <span className="row">
                <button className="secondary" onClick={() => why(r.job_id)} disabled={ex === "loading"}>Why?</button>
                <button className="secondary" onClick={() => feedback(r.job_id, "dismiss")}>Not for me</button>
                <button className="primary" onClick={() => feedback(r.job_id, "save")}>Save</button>
              </span>
            </div>
            {ex === "loading" && <p className="note">Checking the posting against your resume…</p>}
            {typeof ex === "string" && ex !== "loading" && <p className="error">{ex}</p>}
            {ex && typeof ex === "object" && (
              <div className="explain">
                <p><strong>{ex.fit} fit.</strong> {ex.summary}</p>
                {ex.reasons.map((x, i) => (
                  <p key={i}>✓ {x.point} <q>{x.job_evidence}</q></p>
                ))}
                {ex.gaps.map((g, i) => (
                  <p key={i}>✗ {g.requirement} <q>{g.job_evidence}</q></p>
                ))}
                {ex.dropped.length > 0 && <p className="note">{ex.dropped.length} unsupported claim(s) removed by the grounding check.</p>}
              </div>
            )}
          </article>
        );
      })}
    </section>
  );
}
