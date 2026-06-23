# Sentiment Model Arena

A project that compares two sentiment analysis models side-by-side: a custom Transformer built from scratch and a fine-tuned DistilRoBERTa model.

## Features

- **Side-by-Side Comparison**: Analyze the same movie review with both models simultaneously.
- **Attention Visualization**: Interactive heatmap showing exactly which words the custom Transformer focused on when making its prediction.
- **Latency Tracking**: Compare the inference speed of a small ~6M parameter custom model vs an ~82M parameter production model.
- **Modern UI**: Built with React, Vite, and a custom design system featuring glassmorphism, fluid animations, and a responsive layout.

## Project Structure

This repository is split into three main components:

1. **`notebooks/`**: Contains the Jupyter notebooks used to train both models from scratch. Includes detailed walkthroughs of the math and implementation.
2. **`backend/`**: FastAPI server that loads the PyTorch checkpoints, handles tokenization/preprocessing, and runs inference. Also extracts self-attention weights using forward hooks.
3. **`frontend/`**: React SPA (Single Page Application) built with Vite and TypeScript.

## Quick Start

### 1. Start the Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

_Note: The first startup will take longer as it automatically downloads the models from Hugging Face Hub._

### 2. Start the Frontend

In a new terminal:

```bash
cd frontend
yarn install
yarn dev
```

Then open your browser to `http://localhost:5173`.

## Model Storage

Model weights are hosted on Hugging Face Hub:

- Custom Transformer: [jibinsajujoseph/scratch-transformer](https://huggingface.co/jibinsajujoseph/scratch-transformer)
- DistilRoBERTa: [jibinsajujoseph/distilroberta-imdb](https://huggingface.co/jibinsajujoseph/distilroberta-imdb)

The application automatically downloads and caches model artifacts on first launch. No manual model download is required.

First startup may take longer while artifacts are downloaded. Subsequent launches use the local Hugging Face cache.

## Architecture Details

### Custom Scratch Transformer

- **Parameters**: ~6.1 Million
- **Accuracy**: ~81-82% (IMDb test set)
- **Architecture**: 3 Encoder Blocks, 4 Attention Heads, 256 embedding dimension.
- **Tokenization**: Word-level vocabulary (20,000 tokens) with basic preprocessing (lowercase, alphanumeric filtering).
- **Implementation**: 100% PyTorch primitives (`nn.Linear`, `nn.Embedding`, `nn.LayerNorm`). No pre-built `nn.Transformer` modules.

### Fine-tuned DistilRoBERTa

- **Parameters**: ~82 Million
- **Accuracy**: ~92-94% (IMDb test set)
- **Architecture**: 6 Encoder Blocks, 12 Attention Heads, 768 embedding dimension.
- **Tokenization**: BPE (Byte-Pair Encoding) with 50,265 token vocabulary.
- **Implementation**: HuggingFace `AutoModelForSequenceClassification` fine-tuned via the `Trainer` API.

## Requirements

- Python 3.11+
- Node.js 18+ (with `yarn` or `npm`)
