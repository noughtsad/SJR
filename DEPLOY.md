# Deploying: Supabase (DB) + Hugging Face Spaces (API) + Vercel (UI)

```
 Vercel (Next.js UI) ──HTTPS──► HF Space (FastAPI + models, Docker) ──► Supabase Postgres + pgvector
                                         ▲                                      ▲
               GitHub Actions: deploy-hf.yml (on push)          refresh-index.yml (3x/day: ingest + embed)
```

All three have free tiers that fit: the index is ~270 MB (Supabase free: 500 MB) and the API needs
~1.5 GB RAM (free CPU Spaces have 16 GB). A free Space sleeps after ~48 h without traffic; the next
request wakes it in a minute or two.

## 1. Supabase: the database

1. Create a project at [supabase.com](https://supabase.com) and save the database password.
2. **Database → Extensions**: enable `vector`. The app also runs `CREATE EXTENSION IF NOT EXISTS vector`, so this is a safeguard.
3. **Connect** (top bar) → **Session pooler** → copy the URI. It looks like
   `postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres`.
   Append `?sslmode=require`.
   - Use the **session** pooler (port 5432), not the transaction pooler (6543): the search code uses
     `SET LOCAL` and prepared statements. The direct connection is IPv6-only, which Spaces and GitHub
     runners can't always reach.
4. Load the index from your machine (no re-embedding; vectors are copied):
   ```powershell
   python -m app.evals.fixture build --all --out data/full_index.jsonl.gz   # already done once
   $env:DATABASE_URL = "<session pooler URI>?sslmode=require"
   python -m app.evals.fixture load --path data/full_index.jsonl.gz
   Remove-Item Env:DATABASE_URL
   ```
   This applies the schema, inserts ~19k jobs with embeddings, builds the HNSW index and refreshes
   the keyword statistics. It takes a few minutes, depending on your upload speed.

## 2. Hugging Face Space: the API

1. Create a Space at [huggingface.co/new-space](https://huggingface.co/new-space): SDK **Docker**,
   template **Blank**, hardware **CPU basic (free)**. Example name: `sjr-api`.
2. **Space → Settings → Variables and secrets**, add these **secrets**:
   | Name | Value |
   |---|---|
   | `DATABASE_URL` | the Supabase session pooler URI with `?sslmode=require` |
   | `GEMINI_API_KEY` | your Gemini key |
   | `LLM_RPM` | `14` (free-tier Gemini limit is 15/min) |
   | `DB_POOL_MAX` | `5` (stays under the pooler's connection limit) |
   | `CORS_ORIGINS` | `["https://<your-vercel-app>.vercel.app"]`; set it after step 3, then restart the Space |
   | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` | optional, enables tracing |
3. Create a **write** token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
4. In **GitHub → SJR → Settings → Secrets and variables → Actions**, add:
   - `HF_TOKEN`: the write token
   - `HF_SPACE`: `<your-hf-username>/sjr-api`
   - `DATABASE_URL`: the same Supabase URI (used by the 3×/day index refresh)
5. **GitHub → Actions → deploy-hf → Run workflow**. Later pushes to `main` that touch the API
   deploy automatically. The first image build on HF takes ~5–10 minutes; follow the Space's **Logs** tab.
6. Check `https://<hf-username>-sjr-api.hf.space/health`. It should report ~19k active jobs.
   `/docs` has the interactive API.

## 3. Vercel: the UI

1. [vercel.com/new](https://vercel.com/new) → import `noughtsad/SJR`.
2. **Root Directory**: `web` (Framework preset: Next.js is detected).
3. **Environment variable**: `NEXT_PUBLIC_API_URL` = `https://<hf-username>-sjr-api.hf.space`
4. Deploy, then put the Vercel URL into the Space's `CORS_ORIGINS` secret (step 2) and restart the Space.

## 4. Keeping the index fresh

`refresh-index.yml` runs 3× a day: it pulls HN "Who is hiring" and Remotive, upserts changes,
expires delisted posts and embeds only new or changed jobs. Once the `DATABASE_URL` secret exists it
just runs; trigger it once by hand from the Actions tab to check.

## Troubleshooting

- **Space build fails on memory or timeouts**: re-run it from the Space's Settings → Factory rebuild.
- **`connection refused` / `timeout` to Supabase**: you are probably using the direct (IPv6) host. Use the session pooler.
- **Browser console shows a CORS error**: `CORS_ORIGINS` must exactly match the Vercel origin
  (scheme, no trailing slash), as a JSON list.
- **429 from `/chat`**: per-IP limit of 10 LLM requests/min (`RATE_LIMIT_LLM_PER_MIN`), or the
  Gemini free-tier quota.
