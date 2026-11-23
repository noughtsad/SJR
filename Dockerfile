FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /srv

# CPU-only torch first, so sentence-transformers doesn't pull multi-GB CUDA wheels
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app
COPY db ./db

EXPOSE 8000
CMD ["sh", "-c", "python -m app.cli init-db && uvicorn app.api.main:app --host 0.0.0.0 --port 8000"]
