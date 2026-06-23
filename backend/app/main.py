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
from app.services.model_manager import ModelManager

logger = logging.getLogger(__name__)

# ── Model manager (initialized at startup) ──────────────────────────────────
model_manager: ModelManager | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Load both models into memory at startup, clean up on shutdown."""
    global model_manager

    model_manager = ModelManager()
    model_manager.initialize_models()

    yield  # App is running

    # Cleanup
    model_manager = None
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
        "models_loaded": model_manager is not None,
    }
