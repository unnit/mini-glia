# Mini-Glia — single container serving the FastAPI backend + static frontend.
# The server (backend/server.py) serves frontend/index.html at "/", the
# captured replay at /api/replay, and a live SSE agent run at /api/run.

FROM python:3.11-slim

# small, no build tools needed for these pure-python deps
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# install deps first (layer cache)
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# copy the app (backend code + captured replay, and the frontend it serves)
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# the server resolves the frontend as HERE.parent / "frontend", and the
# replay as HERE / "runs" / "replay.json"; running from backend/ satisfies both
WORKDIR /app/backend

EXPOSE 8000

# GEMINI_API_KEY is injected at runtime (compose / Lightsail env) for the live
# button; the replay path needs no key.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]

