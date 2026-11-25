# Production API image (docker compose and Google Cloud Run).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf DEVICE=cpu LOG_FORMAT=json
WORKDIR /srv

RUN useradd -m -u 1000 app && mkdir -p /opt/hf && chown app /opt/hf

# CPU-only torch first, so sentence-transformers doesn't pull multi-GB CUDA wheels
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model and reranker into the image: no downloads on cold start
USER app
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
ENV HF_HUB_OFFLINE=1

COPY --chown=app app ./app
COPY --chown=app db ./db

# Cloud Run sets PORT (8080); docker compose uses the 8000 default
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "python -m app.cli init-db && exec uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
