# Sentiment Model Arena — Backend

FastAPI backend that serves two sentiment analysis models for side-by-side comparison.

## Models

| Model | Architecture | Parameters | Expected Accuracy |
|---|---|---|---|
| **Scratch Transformer** | Custom encoder (3 blocks, 4 heads, 256d) | ~6M | ~81–82% |
| **DistilRoBERTa** | Fine-tuned `distilroberta-base` | ~82M | ~92–94% |

## Setup

```bash
cd backend

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running

```bash
# Start the development server
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

- **Health check:** `GET /health`
- **Analyze:** `POST /api/analyze`
- **API docs:** `GET /docs` (Swagger UI)

## API

### POST /api/analyze

**Request:**

```json
{
  "text": "This movie was absolutely fantastic!"
}
```

**Response:**

```json
{
  "scratch": {
    "label": "positive",
    "confidence": 0.81,
    "latency_ms": 12.34,
    "tokens": ["this", "movie", "was", "absolutely", "fantastic"],
    "attention": [0.15, 0.42, 0.08, 0.72, 0.95]
  },
  "distilroberta": {
    "label": "positive",
    "confidence": 0.96,
    "latency_ms": 28.56
  }
}
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run individual test suites
pytest tests/test_scratch_model.py -v
pytest tests/test_distilroberta.py -v
```

## Project Structure

```
backend/
├── app/
│   ├── main.py                     # FastAPI app, CORS, lifespan
│   ├── config.py                   # Central configuration & HF Repo IDs
│   ├── schemas.py                  # Pydantic request/response models
│   ├── api/
│   │   └── routes.py               # POST /analyze endpoint
│   ├── models/
│   │   ├── scratch_transformer.py  # Custom PyTorch architecture
│   │   └── distilroberta.py        # HuggingFace model loader
│   └── services/
│       ├── model_manager.py        # HF Hub downloads & initialization
│       ├── scratch_service.py      # Scratch model inference + attention
│       └── distilroberta_service.py # DistilRoBERTa inference
├── tests/
│   ├── test_scratch_model.py
│   └── test_distilroberta.py
├── requirements.txt
└── README.md
```
