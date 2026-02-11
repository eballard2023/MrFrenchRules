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
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# APP_PORT = web server port (avoids conflict with DB "port" in .env)
ENV APP_PORT=8000
EXPOSE 8000

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${APP_PORT:-8000} --log-level info"]
