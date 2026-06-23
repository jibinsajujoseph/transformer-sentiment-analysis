"""
FastAPI application for Sentiment Model Arena.

Loads both models at startup via the lifespan handler and
exposes a single POST /api/analyze endpoint.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.services.distilroberta_service import DistilRoBERTaService
from app.services.scratch_service import ScratchTransformerService

logger = logging.getLogger(__name__)

# ── Model services (initialized at startup) ──────────────────────────────────
scratch_service: ScratchTransformerService | None = None
distilroberta_service: DistilRoBERTaService | None = None

# ── Model paths ──────────────────────────────────────────────────────────────
# Resolve relative to the project root (one level above backend/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRATCH_MODEL_DIR = PROJECT_ROOT / "models" / "scratch-transformer"
DISTILROBERTA_MODEL_DIR = PROJECT_ROOT / "models" / "distilroberta-imdb"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load both models into memory at startup, clean up on shutdown."""
    global scratch_service, distilroberta_service

    logger.info("Loading scratch Transformer from %s ...", SCRATCH_MODEL_DIR)
    scratch_service = ScratchTransformerService(SCRATCH_MODEL_DIR)
    logger.info("Scratch Transformer loaded successfully.")

    logger.info("Loading DistilRoBERTa from %s ...", DISTILROBERTA_MODEL_DIR)
    distilroberta_service = DistilRoBERTaService(DISTILROBERTA_MODEL_DIR)
    logger.info("DistilRoBERTa loaded successfully.")

    yield  # App is running

    # Cleanup
    scratch_service = None
    distilroberta_service = None
    logger.info("Models unloaded.")


# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sentiment Model Arena",
    description=(
        "Compare a custom scratch Transformer vs fine-tuned DistilRoBERTa "
        "for sentiment analysis on movie reviews."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow Vite dev server and common local origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API routes
app.include_router(router, prefix="/api")


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Health check endpoint."""
    return {
        "status": "ok",
        "scratch_model": "loaded" if scratch_service else "not loaded",
        "distilroberta_model": "loaded" if distilroberta_service else "not loaded",
    }
