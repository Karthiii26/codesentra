"""
dnn_classifier.py
=================
PyTorch MLP classifier for AI-generated code detection.

Architecture: 3 hidden layers with dropout and ReLU activations, trained on
the combined stylometric + perplexity feature vector.

Split: 60% train / 15% validation / 25% test (supports calibration eval on ≥50 test samples)
Training: early stopping on validation loss, best model saved to models/dnn_model.pt

Metrics reported: Accuracy, Precision, Recall, F1, PR-AUC
  (PR-AUC is preferred over accuracy for potentially imbalanced deployment data)

Usage:
    python src/models/dnn_classifier.py
    python src/models/dnn_classifier.py --human-dir data/human --ai-dir data/ai
    python src/models/dnn_classifier.py --epochs 100 --batch-size 32 --lr 1e-3
    python src/models/dnn_classifier.py --stylometric-only  # train without perplexity features

Saved artifacts:
    models/dnn_model.pt          — trained model weights + architecture config
    models/dnn_scaler.pkl        — fitted StandardScaler for inference
    models/dnn_test_results.npz  — y_test, y_pred, y_prob for eval scripts
"""

import argparse
import pickle
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(
    human_dir: Path,
    ai_dir: Path,
    use_perplexity: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all .py files, extract features, return (X, y, sources).

    Labels: 0 = human, 1 = AI
    If use_perplexity=False, loads stylometric features only (faster, used
    for ablation in adversarial_robustness.py).
    """
    from src.features.stylometry import extract_stylometric_features, feature_names

    if use_perplexity:
        from src.features.perplexity import (
            compute_perplexity_features,
            merge_features,
        )

    records: list[tuple[np.ndarray, int, str]] = []
    human_source_counter: Counter = Counter()
    ai_source_counter: Counter = Counter()

    # ── Human files ──────────────────────────────────────────────────────────
    for fpath in sorted(human_dir.glob("*.py")):
        source = fpath.read_text(encoding="utf-8", errors="replace")
        stylo = extract_stylometric_features(source)
        if use_perplexity:
            perp = compute_perplexity_features(source)
            vec = merge_features(stylo, perp)
        else:
            vec = np.array([stylo[k] for k in feature_names()], dtype=np.float32)
        records.append((vec, 0, fpath.stem))
        author = fpath.stem.split("__")[-1] if "__" in fpath.stem else "unknown"
        human_source_counter[author] += 1

    # ── AI files ─────────────────────────────────────────────────────────────
    for fpath in sorted(ai_dir.glob("*.py")):
        source = fpath.read_text(encoding="utf-8", errors="replace")
        stylo = extract_stylometric_features(source)
        if use_perplexity:
            perp = compute_perplexity_features(source)
            vec = merge_features(stylo, perp)
        else:
            vec = np.array([stylo[k] for k in feature_names()], dtype=np.float32)
        records.append((vec, 1, fpath.stem))
        model = fpath.stem.split("__")[-1] if "__" in fpath.stem else "unknown"
        ai_source_counter[model] += 1

    if not records:
        sys.exit("[ERROR] No data found. Run data generation scripts first.")

    n_human = sum(1 for _, label, _ in records if label == 0)
    n_ai = sum(1 for _, label, _ in records if label == 1)

    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total samples     : {len(records)}")
    print(f"  Human (0)         : {n_human}")
    print(f"  AI    (1)         : {n_ai}")
    print(f"  Feature type      : {'stylometric + perplexity' if use_perplexity else 'stylometric only'}")
    print(f"  Feature dimension : {records[0][0].shape[0]}")
    print(f"\n  Human samples by author:")
    for author, cnt in sorted(human_source_counter.items()):
        print(f"    {author:<30} {cnt}")
    print(f"\n  AI samples by model:")
    for model, cnt in sorted(ai_source_counter.items()):
        print(f"    {model:<30} {cnt}")
    print("=" * 60 + "\n")

    if n_human < 20 or n_ai < 20:
        print(
            "[WARN] Very few samples — metrics will be unreliable. "
            "Generate more data before trusting results.",
            file=sys.stderr,
        )

    X = np.stack([r[0] for r in records])
    y = np.array([r[1] for r in records], dtype=int)
    sources = [r[2] for r in records]
    return X, y, sources


# ---------------------------------------------------------------------------
# PyTorch MLP
# ---------------------------------------------------------------------------

def build_mlp(input_dim: int, hidden_dims: list[int], dropout: float):
    """Build a PyTorch MLP with ReLU activations and dropout."""
    import torch.nn as nn

    layers = []
    in_dim = input_dim
    for out_dim in hidden_dims:
        layers += [
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]
        in_dim = out_dim
    layers.append(nn.Linear(in_dim, 1))  # binary output (logit)
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    import torch
    import torch.nn as nn
    from sklearn.metrics import (
        average_precision_score,
        classification_report,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DNN] Using device: {device}")

    human_dir = Path(args.human_dir)
    ai_dir = Path(args.ai_dir)

    if not human_dir.exists():
        sys.exit(f"[ERROR] Human dir not found: {human_dir}")
    if not ai_dir.exists():
        sys.exit(f"[ERROR] AI dir not found: {ai_dir}")

    use_perplexity = not args.stylometric_only
    X, y, sources = load_dataset(human_dir, ai_dir, use_perplexity=use_perplexity)

    # ── 60 / 15 / 25 split ───────────────────────────────────────────────────
    # First split off 25% test, then split remaining 75% into 80/20 (= 60/15 overall)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=args.random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.20, stratify=y_trainval,
        random_state=args.random_state
    )

    print(f"  Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}\n")

    # ── Scale features ───────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    # ── Build model ───────────────────────────────────────────────────────────
    input_dim = X_train.shape[1]
    hidden_dims = [args.hidden1, args.hidden2, args.hidden3]
    model = build_mlp(input_dim, hidden_dims, args.dropout).to(device)
    print(f"  Model architecture:\n    {model}\n")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # ── DataLoader ────────────────────────────────────────────────────────────
    from torch.utils.data import DataLoader, TensorDataset

    train_ds = TensorDataset(
        torch.from_numpy(X_train), torch.from_numpy(y_train.astype(np.float32))
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val), torch.from_numpy(y_val.astype(np.float32))
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # ── Training loop with early stopping ────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    print("-" * 60)
    print("  TRAINING")
    print("-" * 60)

    for epoch in range(1, args.epochs + 1):
        # Train
        model.train()
        train_loss_sum = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch).squeeze(1)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * len(X_batch)

        # Validate
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch).squeeze(1)
                val_loss_sum += criterion(logits, y_batch).item() * len(X_batch)

        train_loss = train_loss_sum / len(X_train)
        val_loss = val_loss_sum / len(X_val)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>4}/{args.epochs}  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

        # Early stopping
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n  [Early stopping] No improvement for {args.patience} epochs. Stopping at epoch {epoch}.")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # ── Test evaluation ───────────────────────────────────────────────────────
    print("\n" + "-" * 60)
    print("  TEST EVALUATION")
    print("-" * 60)

    model.eval()
    X_test_t = torch.from_numpy(X_test).to(device)
    with torch.no_grad():
        logits_test = model(X_test_t).squeeze(1).cpu().numpy()

    y_prob = 1.0 / (1.0 + np.exp(-logits_test))  # sigmoid
    y_pred = (y_prob >= 0.5).astype(int)

    print(classification_report(y_test, y_pred, target_names=["human", "AI"], digits=4))
    pr_auc = average_precision_score(y_test, y_prob)
    print(f"  PR-AUC  : {pr_auc:.4f}")
    print()

    # ── Save artifacts ────────────────────────────────────────────────────────
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save model (weights + config for reconstruction)
    model_path = models_dir / "dnn_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "input_dim": input_dim,
            "hidden_dims": hidden_dims,
            "dropout": args.dropout,
            "use_perplexity": use_perplexity,
        },
    }, model_path)
    print(f"  Model saved to: {model_path}")

    # Save scaler
    scaler_path = models_dir / "dnn_scaler.pkl"
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"  Scaler saved to: {scaler_path}")

    # Save test results for eval scripts (calibration, etc.)
    results_path = models_dir / "dnn_test_results.npz"
    np.savez(results_path, y_test=y_test, y_pred=y_pred, y_prob=y_prob)
    print(f"  Test results saved to: {results_path}")


# ---------------------------------------------------------------------------
# Inference helper (used by eval scripts and the live demo)
# ---------------------------------------------------------------------------

def load_trained_model(models_dir: Path = Path("models")):
    """Load the trained DNN and scaler from disk.

    Returns
    -------
    model     : nn.Module (eval mode)
    scaler    : fitted StandardScaler
    config    : dict with model config
    """
    import torch

    model_path = models_dir / "dnn_model.pt"
    scaler_path = models_dir / "dnn_scaler.pkl"

    if not model_path.exists():
        sys.exit(f"[ERROR] Model not found at {model_path}. Run dnn_classifier.py first.")
    if not scaler_path.exists():
        sys.exit(f"[ERROR] Scaler not found at {scaler_path}. Run dnn_classifier.py first.")

    checkpoint = torch.load(model_path, map_location="cpu")
    config = checkpoint["config"]
    model = build_mlp(config["input_dim"], config["hidden_dims"], config["dropout"])
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler, config


def predict_proba(source: str, model, scaler, config: dict) -> float:
    """Return the AI probability (0–1) for a single code snippet.

    Used by the live demo and eval scripts.
    Includes safeguards for micro-snippets, competitive programming scripts,
    and LLM conversational/pedagogical comment and docstring patterns.
    """
    import ast
    import re
    import torch

    from src.features.stylometry import extract_stylometric_features, feature_names
    from src.features.perplexity import compute_perplexity_features, merge_features

    stylo = extract_stylometric_features(source)
    if config.get("use_perplexity", True):
        perp = compute_perplexity_features(source)
        vec = merge_features(stylo, perp)
    else:
        vec = np.array([stylo[k] for k in feature_names()], dtype=np.float32)

    vec_scaled = scaler.transform(vec.reshape(1, -1)).astype(np.float32)
    x_t = torch.from_numpy(vec_scaled)

    with torch.no_grad():
        logit = model(x_t).squeeze().item()
    raw_prob = 1.0 / (1.0 + np.exp(-logit))

    # --- LLM Pedagogical & Stylometric Pattern Analysis ---
    lines = source.splitlines()
    comments = [l.strip() for l in lines if "#" in l]

    # 1. Numbered steps: '# 1. ', '# 2. ', 'Step 1:'
    step_comments = [c for c in comments if re.search(r"#\s*(?:\d+[\.:\)]|Step\s*\d+)\s+[A-Z]", c)]

    # 2. Section divider comments: '# --- ... ---', '# Driver Code', '# Example Usage'
    divider_comments = [c for c in comments if re.search(r"#\s*[-=]{2,}.*[-=]{2,}|#\s*(?:driver code|example usage|main driver|test case|driver example)", c, re.IGNORECASE)]

    # 3. Inline expected output annotations: '# Output: Woof!', '# Expected: True'
    output_comments = [
        c for c in comments
        if re.search(r"#\s*(?:output|expected(?:\s*output)?|result):\s*[\w!\"'<\[\{]", c, re.IGNORECASE)
    ]

    # 4. Tautological / line-by-line pedagogical comments (explaining obvious operations)
    tautological_patterns = [
        r"#\s*(?:pointer to|next node|start from|loop until|move the pointer|print the|create individual|link the|traverse and print|initialize|return the)",
        r"#\s*(?:check if|base case|recursive call|increment|update|iterate through|assign|store the|instantiate|call the function)",
        # LLM pedagogical class/function descriptor comments
        r"#\s*(?:base\s+(?:method|class)|base\s+implementation|to\s+be\s+overridden|override\s+(?:this|in))",
        r"#\s*creat(?:e|ing)\s+(?:an?\s+)?(?:instance|object|node|variable)",
        r"#\s*a?\s*(?:single|standalone|helper|utility)?\s*(?:function|method|class)\s+(?:that|which|to)",
        r"#\s*(?:the\s+)?(?:same|identical)\s+(?:call|function|method)\s+(?:call\s+)?(?:yields?|returns?|produces?|gives?)\s+different",
        r"#\s*(?:demonstrat|illustrat|showcas|show)\w*\s+(?:polymorphism|inheritance|encapsulation|abstraction|overriding|usage)",
    ]
    tautological_comments = [
        c for c in comments
        if any(re.search(p, c, re.IGNORECASE) for p in tautological_patterns)
    ]

    # 5. Grammatical docstrings
    docstrings = []
    try:
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                ds = ast.get_docstring(node)
                if ds:
                    docstrings.append(ds.strip())
    except Exception:
        pass

    grammatical_docstrings = [
        ds for ds in docstrings
        if len(ds) > 10 and ds[0].isupper() and ds.endswith(".") and not any(k in ds for k in [":type", ":rtype", "@param"])
    ]

    # Calculate AI Evidence Score
    ai_evidence = 0.0
    if len(step_comments) >= 2:
        ai_evidence += 0.45
    elif len(step_comments) == 1:
        ai_evidence += 0.20

    if len(divider_comments) >= 1:
        ai_evidence += 0.35

    if len(output_comments) >= 2:
        ai_evidence += 0.45
    elif len(output_comments) == 1:
        ai_evidence += 0.30

    if len(tautological_comments) >= 4:
        ai_evidence += 0.55
    elif len(tautological_comments) >= 2:
        ai_evidence += 0.40
    elif len(tautological_comments) == 1:
        ai_evidence += 0.20

    if (len(grammatical_docstrings) >= 2 and
            stylo.get("operator_spacing_score", 0) > 0.85 and
            stylo.get("naming_style_uniformity", 0) > 0.70):
        ai_evidence += 0.35
    elif (len(grammatical_docstrings) >= 1 and
          stylo.get("docstring_ratio", 0) >= 0.50 and
          stylo.get("operator_spacing_score", 0) > 0.85):
        ai_evidence += 0.20

    # Human LeetCode / competitive programming signatures
    has_complexity_header = any(re.search(r"#\s*(?:time|space)\s*:\s*o\(", c, re.IGNORECASE) for c in comments)
    has_type_docstrings = any(re.search(r"#\s*@(param|return)|:(type|rtype):", l) for l in lines)
    has_python2_object = bool(re.search(r"class\s+\w+\(object\):", source))
    has_leetcode_class = bool(re.search(r"\bclass\s+Solution\b", source))

    is_human_competitive = (
        has_complexity_header or
        has_type_docstrings or
        has_python2_object or
        (has_leetcode_class and ai_evidence == 0 and (
            stylo.get("operator_spacing_score", 1.0) < 0.75 or
            (stylo.get("docstring_ratio", 0) == 0 and stylo.get("comment_density", 0) <= 0.05 and stylo.get("avg_identifier_length", 10.0) < 6.0)
        )) or
        (stylo.get("operator_spacing_score", 1.0) < 0.35 and stylo.get("docstring_ratio", 0) == 0 and ai_evidence == 0)
    )

    if is_human_competitive:
        return float(min(raw_prob * 0.04, 0.045))

    # Domain heuristic for competitive programming/casual quick scripts
    try:
        tree = ast.parse(source)
        has_functions = any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) for n in ast.walk(tree))
        has_classes = any(isinstance(n, ast.ClassDef) for n in ast.walk(tree))
        uses_interactive_input = any(
            (isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "input")
            or (isinstance(n, ast.Attribute) and n.attr in ("readline", "stdin"))
            for n in ast.walk(tree)
        )
        non_blank_lines = [l for l in source.splitlines() if l.strip()]

        if not has_classes and not has_functions and uses_interactive_input and stylo.get("type_annotation_ratio", 0) == 0:
            return float(min(raw_prob * 0.15, 0.12))
        elif len(non_blank_lines) <= 6 and stylo.get("type_annotation_ratio", 0) == 0 and stylo.get("docstring_ratio", 0) == 0 and ai_evidence == 0:
            return float(min(raw_prob, 0.25))
    except Exception:
        pass

    # If AI evidence is strong, correct the dataset shortcut artifact (where 0 types + comments wrongly dragged down logit)
    if ai_evidence >= 0.50:
        adjusted_prob = max(raw_prob, 0.55 + ai_evidence * 0.42)
        return float(min(adjusted_prob, 0.985))
    elif ai_evidence >= 0.25:
        adjusted_prob = max(raw_prob, 0.35 + ai_evidence * 0.40)
        return float(adjusted_prob)

    return float(raw_prob)



# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a PyTorch MLP on stylometric + perplexity features to detect\n"
            "AI-generated Python code. Saves model to models/dnn_model.pt."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--human-dir", default="data/human")
    parser.add_argument("--ai-dir", default="data/ai")
    parser.add_argument("--models-dir", default="models",
                        help="Directory to save model artifacts (default: models)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden1", type=int, default=128)
    parser.add_argument("--hidden2", type=int, default=64)
    parser.add_argument("--hidden3", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (epochs without val improvement)")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--stylometric-only",
        action="store_true",
        help="Use stylometric features only (no perplexity). Faster training for ablation.",
    )
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
