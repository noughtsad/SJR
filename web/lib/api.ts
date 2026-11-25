// Thin client for the FastAPI backend. Types mirror the Pydantic response models.

export const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type SearchMode = "keyword" | "vector" | "hybrid" | "hybrid_rerank";

export interface JobHit {
  job_id: number;
  title: string;
  company: string | null;
  location: string | null;
  is_remote: boolean;
  seniority: string | null;
  skills: string[];
  url: string | null;
  snippet: string;
  keyword_rank: number | null;
  vector_rank: number | null;
  rerank_score: number | null;
}

export interface SearchResponse {
  results: JobHit[];
  timings_ms: Record<string, number>;
}

export interface Recommendation {
  job_id: number;
  title: string;
  company: string | null;
  location: string | null;
  is_remote: boolean;
  seniority: string | null;
  url: string | null;
  snippet: string;
  score: number;
  similarity: number;
  skill_coverage: number | null;
  matched_skills: string[];
  missing_skills: string[];
}

export interface RecommendResponse {
  profile_version: number;
  feedback_used: { save?: number; dismiss?: number };
  query_shift: number;
  results: Recommendation[];
  timings_ms: Record<string, number>;
}

export interface Profile {
  profile: { headline: string; seniority: string; years_experience: number; skills: string[]; target_roles: string[] };
  canonical_skills: string[];
  version: number;
}

export interface Explanation {
  fit: "strong" | "partial" | "weak";
  summary: string;
  reasons: { point: string; job_evidence: string; profile_evidence: string }[];
  gaps: { requirement: string; job_evidence: string }[];
  dropped: string[];
}

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API}${path}`, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {}
    throw new Error(`${res.status}: ${detail}`);
  }
  return res.status === 204 ? (undefined as T) : res.json();
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  search: (query: string, mode: SearchMode, remote: boolean) =>
    call<SearchResponse>("/search", json({ query, mode, k: 10, filters: remote ? { remote: true } : {} })),
  profile: (user: string) => call<Profile>(`/users/${encodeURIComponent(user)}/profile`),
  uploadResume: (user: string, file: File) => {
    const fd = new FormData();
    fd.append("file", file);
    return call<Profile>(`/users/${encodeURIComponent(user)}/resume`, { method: "POST", body: fd });
  },
  recommend: (user: string) => call<RecommendResponse>("/recommend", json({ user_id: user, k: 10 })),
  feedback: (user: string, job_id: number, action: "save" | "dismiss" | null) =>
    call<void>("/feedback", json({ user_id: user, job_id, action })),
  explain: (user: string, job_id: number) =>
    call<Explanation>(`/users/${encodeURIComponent(user)}/jobs/${job_id}/explain`),
};

// --- /chat server-sent events ---------------------------------------------

export type ChatEvent =
  | { type: "token"; text: string }
  | { type: "tool_call"; name: string; args: Record<string, unknown> }
  | { type: "tool_result"; name: string; preview: string }
  | { type: "retract"; reason: string }
  | { type: "final"; text: string; cited_job_ids: number[]; usage: Record<string, number>; latency_ms: number }
  | { type: "error"; kind: string; message: string };

/** POST /chat and yield parsed SSE events (EventSource can't POST, so parse the stream by hand). */
export async function* chatStream(user: string, thread: string, message: string): AsyncGenerator<ChatEvent> {
  const res = await fetch(`${API}/chat`, json({ user_id: user, thread_id: thread, message }));
  if (!res.ok || !res.body) {
    yield { type: "error", kind: res.status === 429 ? "rate_limited" : "failed", message: `HTTP ${res.status}` };
    return;
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const frame = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      const data = frame.split("\n").find((l) => l.startsWith("data: "));
      if (data) yield JSON.parse(data.slice(6)) as ChatEvent;
    }
  }
}
