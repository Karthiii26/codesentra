"""
perplexity.py
=============
Code-LM perplexity feature extraction using microsoft/codebert-base.

For a given Python code snippet, this module computes:
  - mean_perplexity     : mean token-level perplexity from the masked LM loss
                          (AI-generated code has LOWER perplexity — it's more
                          "predictable" to CodeBERT, which was trained on similar
                          code)
  - perplexity_variance : variance of per-position losses (AI code is more uniform)

The loaded model is cached as a module-level singleton so repeated calls
within the same process do not reload weights.

merge_features() combines stylometric and perplexity dicts into one flat
numpy array for classifier input.

Usage:
    python src/features/perplexity.py path/to/file.py
    python src/features/perplexity.py  # runs built-in smoke test

Requirements: transformers, torch (see requirements.txt)
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Lazy model loading — singleton cache
# ---------------------------------------------------------------------------

_MODEL = None
_TOKENIZER = None
_DEVICE = None

# Model to use — microsoft/codebert-base is a RoBERTa-based masked LM.
# It was pretrained on code so its perplexity is informative about
# how "natural" the code looks to the model.
MODEL_NAME = "microsoft/codebert-base"


def _load_model():
    """Load CodeBERT and tokenizer once; cache as module-level globals."""
    global _MODEL, _TOKENIZER, _DEVICE  # noqa: PLW0603

    if _MODEL is not None:
        return _MODEL, _TOKENIZER, _DEVICE

    try:
        import os
        os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"
        sys.modules["torchvision"] = None
        import sklearn
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer  # type: ignore
    except ImportError:
        sys.exit(
            "[ERROR] torch or transformers not installed.\n"
            "        Run: pip install torch transformers"
        )

    import torch

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[perplexity] Loading {MODEL_NAME} on {_DEVICE} (first call only)...",
          file=sys.stderr, flush=True)

    _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_NAME)
    _MODEL = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    _MODEL.eval()
    _MODEL.to(_DEVICE)

    print(f"[perplexity] Model loaded.", file=sys.stderr, flush=True)
    return _MODEL, _TOKENIZER, _DEVICE


# ---------------------------------------------------------------------------
# Perplexity computation
# ---------------------------------------------------------------------------

def compute_perplexity_features(source: str, max_length: int = 128) -> dict[str, float]:
    """Compute CodeBERT perplexity features for a Python source string.

    Strategy: Masked Language Model (MLM) pseudo-perplexity.
    We iterate over token positions, mask them in a batched pass,
    and measure the model's cross-entropy loss.
    """
    import torch
    import torch.nn.functional as F  # noqa: N812

    zero = {"mean_perplexity": 0.0, "perplexity_variance": 0.0}

    if not source or not source.strip():
        return zero

    model, tokenizer, device = _load_model()

    # Tokenize — truncate to max_length (128 tokens is plenty for code stylometry & naturalness)
    encoded = tokenizer(
        source,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    input_ids: "torch.Tensor" = encoded["input_ids"].to(device)
    attention_mask: "torch.Tensor" = encoded["attention_mask"].to(device)

    seq_len = input_ids.shape[1]
    if seq_len < 3:
        # Too short to compute meaningful perplexity
        return zero

    # We skip [CLS] (idx 0) and [SEP] (last idx) — only mask "real" tokens
    valid_positions = list(range(1, seq_len - 1))
    max_eval_positions = 10
    if len(valid_positions) > max_eval_positions:
        step = len(valid_positions) / max_eval_positions
        eval_positions = [valid_positions[int(i * step)] for i in range(max_eval_positions)]
    else:
        eval_positions = valid_positions

    # Vectorized batched evaluation: 1 single model forward pass for all positions
    n_pos = len(eval_positions)
    masked_batch = input_ids.repeat(n_pos, 1)
    true_labels: list[int] = []
    for idx, pos in enumerate(eval_positions):
        true_labels.append(input_ids[0, pos].item())
        masked_batch[idx, pos] = tokenizer.mask_token_id

    batch_attention = attention_mask.repeat(n_pos, 1)

    token_losses: list[float] = []
    with torch.no_grad():
        outputs = model(input_ids=masked_batch, attention_mask=batch_attention)
        logits = outputs.logits  # shape: (n_pos, seq_len, vocab_size)

        for idx, pos in enumerate(eval_positions):
            pos_logits = logits[idx, pos, :]
            target = torch.tensor([true_labels[idx]], device=device)
            loss = F.cross_entropy(pos_logits.unsqueeze(0), target).item()
            token_losses.append(loss)

    if not token_losses:
        return zero

    losses_arr = np.array(token_losses, dtype=np.float64)
    # Perplexity = exp(mean cross-entropy loss)
    mean_perplexity = float(np.exp(np.mean(losses_arr)))
    perplexity_variance = float(np.var(losses_arr))

    return {
        "mean_perplexity": mean_perplexity,
        "perplexity_variance": perplexity_variance,
    }


def perplexity_feature_names() -> list[str]:
    """Return ordered perplexity feature names."""
    return ["mean_perplexity", "perplexity_variance"]


# ---------------------------------------------------------------------------
# Feature merger
# ---------------------------------------------------------------------------

def merge_features(
    stylometric: dict[str, float],
    perplexity: dict[str, float],
) -> np.ndarray:
    """Merge stylometric and perplexity feature dicts into one flat numpy array.

    The ordering follows:
      [stylometric features in their canonical order] + [perplexity features]

    This is the function called by all classifier scripts to build the
    feature vector for a single code sample.

    Parameters
    ----------
    stylometric : dict[str, float]
        Output of stylometry.extract_stylometric_features()
    perplexity : dict[str, float]
        Output of compute_perplexity_features()

    Returns
    -------
    np.ndarray of shape (n_features,), dtype float32
    """
    # Import here to avoid circular dependency if scripts import from both
    from src.features.stylometry import feature_names as stylo_names  # type: ignore

    stylo_values = [stylometric.get(k, 0.0) for k in stylo_names()]
    perp_values = [perplexity.get(k, 0.0) for k in perplexity_feature_names()]
    combined = stylo_values + perp_values
    return np.array(combined, dtype=np.float32)


def all_feature_names() -> list[str]:
    """Return the full ordered feature name list (stylometric + perplexity)."""
    from src.features.stylometry import feature_names as stylo_names  # type: ignore
    return stylo_names() + perplexity_feature_names()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ("-h", "--help"):
            print("Usage: python perplexity.py [path/to/file.py]")
            print("Extracts CodeBERT perplexity features for a Python source file.")
            sys.exit(0)
        path = Path(sys.argv[1])
        if not path.exists():
            sys.exit(f"[ERROR] File not found: {path}")
        source = path.read_text(encoding="utf-8")
    else:
        # Built-in smoke test — use a simple snippet so it's fast
        source = """\
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
"""
        print("Built-in smoke test (microsoft/codebert-base on a short snippet):\n")

    feats = compute_perplexity_features(source)
    print(f"  mean_perplexity     : {feats['mean_perplexity']:.4f}")
    print(f"  perplexity_variance : {feats['perplexity_variance']:.6f}")
