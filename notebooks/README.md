# Notebooks

This folder contains two Jupyter notebooks that together form the core of this sentiment analysis project. They cover the full spectrum from building a Transformer architecture from scratch to fine-tuning a state-of-the-art pretrained model — both applied to binary sentiment classification on the [IMDb movie review dataset](https://huggingface.co/datasets/stanfordnlp/imdb).

---

## `sentiment-transformer-from-scratch.ipynb`

**Goal:** Build and train a Transformer encoder for sentiment classification without using any pretrained weights or high-level abstractions like `nn.Transformer` or HuggingFace models.

Every component is implemented from first principles using standard PyTorch primitives (`nn.Linear`, `nn.LayerNorm`, `nn.Embedding`):

- Custom tokenizer and vocabulary builder (word-level, 20k tokens)
- Sinusoidal positional encoding
- Scaled dot-product self-attention
- Multi-head attention
- Position-wise feed-forward network
- Transformer block with residual connections and LayerNorm
- Stacked Transformer encoder with padding masks
- Classification head using masked mean pooling → linear projection → softmax

**Architecture at a glance**

| Hyperparameter         | Value      |
| ---------------------- | ---------- |
| Embedding dimension    | 256        |
| Attention heads        | 4          |
| Feed-forward dimension | 512        |
| Transformer blocks     | 3          |
| Max sequence length    | 200 tokens |
| Vocabulary size        | 20,000     |
| Dropout                | 0.3        |

**Training setup:** Adam optimizer with learning rate `1e-4`, weight decay `1e-4`, `ReduceLROnPlateau` scheduler, and early stopping (patience = 5). Gradient clipping is applied at norm 1.0.

**Performance:** ~81–82% test accuracy on the full IMDb dataset.

**Outputs saved:**

- `sentiment_transformer.pt` — model weights
- `vocab.pkl` — word-to-index vocabulary
- `model_config.json` — hyperparameters for architecture reconstruction

The notebook also includes a bonus section that visualizes per-head attention weights using forward hooks, letting you inspect what the model attends to in a given review.

---

## `distilroberta_finetuning_imdb.ipynb`

**Goal:** Fine-tune `distilroberta-base` on IMDb using the HuggingFace `transformers` and `Trainer` APIs, and compare results against the custom Transformer trained in the first notebook.

Rather than building from scratch, this notebook adapts a model pretrained on 160GB+ of text, which provides a strong accuracy baseline with only a few epochs of fine-tuning.

**Key differences from the scratch notebook**

| Aspect             | Custom Transformer     | DistilRoBERTa         |
| ------------------ | ---------------------- | --------------------- |
| Tokenizer          | Word-level (20k vocab) | BPE (50,265 tokens)   |
| Pretrained weights | None                   | Yes (160GB+ corpus)   |
| Parameters         | ~few million           | ~82M                  |
| Training loop      | Manual PyTorch         | HuggingFace `Trainer` |
| Expected accuracy  | ~81–82%                | ~92–94%               |

**Training setup:** The `Trainer` API handles the full loop — LR warmup (10% of steps), linear decay, mixed-precision training (`fp16` on GPU), per-epoch evaluation, best-checkpoint restoration, and early stopping (patience = 2). Best checkpoint is selected by F1 score.

**Outputs saved** (HuggingFace format, reloadable in 4 lines):

- `distilroberta-imdb-final/` — model weights and config
- `distilroberta-imdb-final/tokenizer/` — tokenizer files
- `distilroberta-imdb.zip` — zipped archive for download

The notebook closes with a comparison cell that prints the accuracy and F1 delta between the two approaches and explains the sources of the performance gap.

---

## Running the Notebooks

Both notebooks are designed for **Google Colab** with a GPU runtime (`Runtime → Change runtime type → T4 GPU`). They will run on CPU but training will be significantly slower.

Each notebook has a `QUICK_RUN` flag at the top:

```python
QUICK_RUN = True   # ~1–2k samples, 2 epochs — good for a fast sanity check
QUICK_RUN = False  # Full 25k IMDb training set — recommended for real results
```

Dependencies are installed in the first cell of each notebook via `pip`.

---

## Relationship Between the Two Notebooks

These notebooks are intended to be run in order. The custom Transformer notebook (`sentiment-transformer-from-scratch.ipynb`) establishes a conceptual baseline by implementing every component by hand, making the architecture transparent and fully understandable. The fine-tuning notebook (`distilroberta_finetuning_imdb.ipynb`) then shows how transfer learning from a large pretrained model closes the gap to production-grade accuracy. The comparison cell in the second notebook expects you to paste in your results from the first.
