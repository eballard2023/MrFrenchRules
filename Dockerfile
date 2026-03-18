# AI Coach Interview System (MrFrenchRules)
# Image is large due to PyTorch + sentence-transformers.
FROM python:3.11-bookworm

WORKDIR /app

# Build deps for psycopg2; keep libpq5 for runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc libpq5 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

# Pre-download HuggingFace models at build time so cold starts don't timeout
RUN python -c "\
from transformers import pipeline; \
pipeline('text-classification', model='SamLowe/roberta-base-go_emotions', top_k=None); \
pipeline('text-classification', model='Minej/bert-base-personality', top_k=None); \
print('Models cached successfully')"

COPY . .

# APP_PORT = web server port (avoids conflict with DB "port" in .env)
ENV APP_PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${APP_PORT:-8000} --log-level info"]
