# Deploying: Supabase (DB) + Google Cloud Run (API) + Vercel (UI)

```
 Vercel (Next.js UI) ──HTTPS──► Cloud Run (FastAPI + models)  ──► Supabase Postgres + pgvector
                                   ▲  deploy-cloudrun.yml (on push)        ▲  refresh-index.yml (3x/day)
```

**Cost.** A demo fits the free tiers:
- *Cloud Run*: 2M requests and 360k GiB-seconds a month. The service scales to zero when idle.
- *Supabase*: 500 MB; the index is ~270 MB.
- *Vercel*: hobby plan.

Expect a few cents a month for storing the container image (Artifact Registry gives 0.5 GB free; the image is ~1 GB compressed). Cloud Run still needs a billing account with a card, but you aren't charged within the free tier.

A cold start, the first request after the service has scaled to zero, takes ~20–40 s while the models load.

## 1. Supabase: the database

1. Create a project at [supabase.com](https://supabase.com) and save the database password.
2. **Database → Extensions**: enable `vector`.
3. **Connect** (top bar) → **Session pooler** → copy the URI and append `?sslmode=require`:
   `postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:5432/postgres?sslmode=require`.
   - Use the **session** pooler (port 5432), not the transaction pooler (6543), because the search code uses `SET LOCAL`. The direct host is IPv6-only.
4. Load the index from your machine. Nothing is re-embedded; the vectors are copied:
   ```powershell
   $env:DATABASE_URL = "<session pooler URI>"
   .venv\Scripts\python -m app.evals.fixture load --path data/full_index.jsonl.gz
   Remove-Item Env:DATABASE_URL
   ```
   It uploads ~160 MB in one transaction, so `select count(*) from jobs` shows 0 until it finishes.
   Expect 10–30 minutes depending on upload speed.

## 2. Google Cloud Run: the API

**Prerequisite:** an open billing account at [console.cloud.google.com/billing](https://console.cloud.google.com/billing). New accounts get a free trial credit. The commands below are what gets run (Claude ran them for this repo). Replace `<...>` placeholders.

```bash
PROJECT=sjr-api-<suffix>          # globally unique
REGION=us-central1                # a free-tier region
gcloud projects create $PROJECT
gcloud billing projects link $PROJECT --billing-account=<BILLING_ACCOUNT_ID>
gcloud config set project $PROJECT
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com

# Secrets live in Secret Manager, not in plain env vars
printf '%s' '<supabase session pooler URI>' | gcloud secrets create database-url --data-file=-
printf '%s' '<gemini key>'                  | gcloud secrets create gemini-api-key --data-file=-
RUNTIME_SA=$(gcloud projects describe $PROJECT --format='value(projectNumber)')-compute@developer.gserviceaccount.com
for s in database-url gemini-api-key; do
  gcloud secrets add-iam-policy-binding $s --member=serviceAccount:$RUNTIME_SA --role=roles/secretmanager.secretAccessor
done

# First deploy: Cloud Build builds the Dockerfile (~10 min), then the service is created
gcloud run deploy sjr-api --source . --region $REGION --allow-unauthenticated \
  --memory 2Gi --cpu 1 --cpu-boost --min-instances 0 --max-instances 1 --timeout 300 \
  --set-secrets DATABASE_URL=database-url:latest,GEMINI_API_KEY=gemini-api-key:latest \
  --set-env-vars "^@^LLM_MODEL=gemini-3.5-flash-lite@LLM_RPM=14@DB_POOL_MAX=5@TRUST_PROXY=true@CORS_ORIGINS=[\"http://localhost:3000\"]"
```

- `--max-instances 1` keeps the in-process rate limiter exact and puts a hard ceiling on cost.
- `--timeout 300` lets streamed `/chat` responses finish.
- `^@^` switches the env-var separator so the JSON list in `CORS_ORIGINS` can contain commas.

Check `https://<service-url>/health` (about 19k active jobs) and `/docs`.

**Keep the image store small** (stays near the free 0.5 GB):
```bash
gcloud artifacts repositories set-cleanup-policies cloud-run-source-deploy --location=$REGION \
  --policy=<(echo '[{"name":"keep-2","action":{"type":"Keep"},"mostRecentVersions":{"keepCount":2}}]')
```

**Continuous deploys from GitHub** (`deploy-cloudrun.yml`). Create a deployer service account and add its key as a GitHub secret:
```bash
gcloud iam service-accounts create github-deployer
SA=github-deployer@$PROJECT.iam.gserviceaccount.com
for r in run.admin cloudbuild.builds.editor artifactregistry.writer storage.admin iam.serviceAccountUser serviceusage.serviceUsageConsumer; do
  gcloud projects add-iam-policy-binding $PROJECT --member=serviceAccount:$SA --role=roles/$r
done
gcloud iam service-accounts keys create key.json --iam-account=$SA
gh secret set GCP_SA_KEY < key.json && gh secret set GCP_PROJECT --body $PROJECT && rm key.json
```
After that, every push to `main` that touches the API rebuilds and redeploys it. Memory, secrets and env vars set at the first deploy carry over.

## 3. Vercel: the UI

1. [vercel.com/new](https://vercel.com/new) → import `noughtsad/SJR`. **Root Directory**: `web`.
2. Environment variable `NEXT_PUBLIC_API_URL` = the Cloud Run service URL.
3. Deploy, then allow the UI's origin on the API:
   ```bash
   gcloud run services update sjr-api --region us-central1 \
     --update-env-vars "^@^CORS_ORIGINS=[\"https://<your-app>.vercel.app\",\"http://localhost:3000\"]"
   ```

## 4. Keeping the index fresh

Add a GitHub secret `DATABASE_URL` with the Supabase URI. `refresh-index.yml` then runs 3× a day: it pulls HN "Who is hiring" and Remotive, upserts changes, expires delisted posts and embeds only new or changed jobs. Until the secret exists, it skips with a notice.

## Troubleshooting

- **`connection refused` / timeouts to Supabase**: use the session pooler host, not the direct (IPv6) one.
- **CORS error in the browser**: `CORS_ORIGINS` must exactly match the Vercel origin (scheme, no trailing slash).
- **Build fails with a permission error on a new project**: grant the Cloud Build role to the default compute service account:
  `gcloud projects add-iam-policy-binding $PROJECT --member=serviceAccount:$RUNTIME_SA --role=roles/cloudbuild.builds.builder`.
- **429 from `/chat`**: the per-IP LLM limit (10/min) or the Gemini free-tier quota.
