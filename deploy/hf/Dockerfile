# Hugging Face Space (Docker SDK) image for the API. The deploy-hf workflow copies
# this file to the Space root as `Dockerfile`, next to app/, db/ and requirements.txt.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf DEVICE=cpu LOG_FORMAT=json TRUST_PROXY=true
WORKDIR /srv

# Spaces run containers as uid 1000
RUN useradd -m -u 1000 user && mkdir -p /opt/hf && chown user /opt/hf

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model and reranker into the image: no downloads on cold start
USER user
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
ENV HF_HUB_OFFLINE=1

COPY --chown=user app ./app
COPY --chown=user db ./db

EXPOSE 7860
CMD ["sh", "-c", "python -m app.cli init-db && uvicorn app.api.main:app --host 0.0.0.0 --port 7860 --proxy-headers --forwarded-allow-ips='*'"]
