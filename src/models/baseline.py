"""
baseline.py
===========
Logistic Regression + Random Forest baseline classifiers using stylometric
features only. This establishes the performance floor that the DNN +
perplexity model must exceed.

Usage:
    python src/models/baseline.py
    python src/models/baseline.py --human-dir data/human --ai-dir data/ai
    python src/models/baseline.py --test-size 0.25 --random-state 42

Outputs:
  - Classification report (precision / recall / F1) for both classifiers
  - Random Forest feature importances (sorted descending)
  - Per-class and per-source sample counts at startup (catches too-small datasets)

Labels: 0 = human, 1 = AI
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Add project root to path so src.features imports work
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.features.stylometry import extract_stylometric_features, feature_names  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(
    human_dir: Path,
    ai_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load all .py files from human_dir and ai_dir, extract stylometric features.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
    y : np.ndarray of shape (n_samples,), dtype int   (0=human, 1=AI)
    sources : list[str]   the stem of each file (e.g. 'two_sum__gpt')
    """
    records: list[tuple[np.ndarray, int, str]] = []

    # ── Human files ──────────────────────────────────────────────────────────
    human_files = sorted(human_dir.glob("*.py"))
    if not human_files:
        print(f"[WARN] No .py files found in {human_dir}", file=sys.stderr)

    human_source_counter: Counter = Counter()
    for fpath in human_files:
        source = fpath.read_text(encoding="utf-8", errors="replace")
        feats = extract_stylometric_features(source)
        vec = np.array([feats[k] for k in feature_names()], dtype=np.float32)
        records.append((vec, 0, fpath.stem))
        # Extract author from stem: slug__author
        parts = fpath.stem.split("__")
        author = parts[-1] if len(parts) > 1 else "unknown"
        human_source_counter[author] += 1

    # ── AI files ─────────────────────────────────────────────────────────────
    ai_files = sorted(ai_dir.glob("*.py"))
    if not ai_files:
        print(f"[WARN] No .py files found in {ai_dir}", file=sys.stderr)

    ai_source_counter: Counter = Counter()
    for fpath in ai_files:
        source = fpath.read_text(encoding="utf-8", errors="replace")
        feats = extract_stylometric_features(source)
        vec = np.array([feats[k] for k in feature_names()], dtype=np.float32)
        records.append((vec, 1, fpath.stem))
        parts = fpath.stem.split("__")
        model = parts[-1] if len(parts) > 1 else "unknown"
        ai_source_counter[model] += 1

    if not records:
        sys.exit("[ERROR] No data found. Generate AI samples and collect human samples first.")

    # ── Print per-class / per-source summary ─────────────────────────────────
    n_human = sum(1 for _, label, _ in records if label == 0)
    n_ai = sum(1 for _, label, _ in records if label == 1)
    print("\n" + "=" * 60)
    print("  DATASET SUMMARY")
    print("=" * 60)
    print(f"  Total samples : {len(records)}")
    print(f"  Human (0)     : {n_human}")
    print(f"  AI    (1)     : {n_ai}")
    print(f"\n  Human samples by author:")
    for author, cnt in sorted(human_source_counter.items()):
        print(f"    {author:<30} {cnt}")
    print(f"\n  AI samples by model:")
    for model, cnt in sorted(ai_source_counter.items()):
        print(f"    {model:<30} {cnt}")
    print("=" * 60 + "\n")

    if n_human < 10 or n_ai < 10:
        print(
            "[WARN] Very few samples — metrics will be unreliable. "
            "Generate/collect more data before trusting these numbers.",
            file=sys.stderr,
        )

    X = np.stack([r[0] for r in records])
    y = np.array([r[1] for r in records], dtype=int)
    sources = [r[2] for r in records]
    return X, y, sources


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def run_baseline(args: argparse.Namespace) -> None:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    human_dir = Path(args.human_dir)
    ai_dir = Path(args.ai_dir)

    if not human_dir.exists():
        sys.exit(f"[ERROR] Human directory not found: {human_dir}")
    if not ai_dir.exists():
        sys.exit(f"[ERROR] AI directory not found: {ai_dir}")

    X, y, _ = load_dataset(human_dir, ai_dir)

    # ── Train / test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )
    print(f"  Train: {len(X_train)} samples  |  Test: {len(X_test)} samples\n")

    # ── Standardise features ─────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # ── Logistic Regression ──────────────────────────────────────────────────
    print("-" * 60)
    print("  LOGISTIC REGRESSION")
    print("-" * 60)
    lr = LogisticRegression(max_iter=1000, random_state=args.random_state)
    lr.fit(X_train_sc, y_train)
    y_pred_lr = lr.predict(X_test_sc)
    print(classification_report(y_test, y_pred_lr, target_names=["human", "AI"], digits=4))

    # ── Random Forest ────────────────────────────────────────────────────────
    print("-" * 60)
    print("  RANDOM FOREST (n_estimators=200)")
    print("-" * 60)
    rf = RandomForestClassifier(
        n_estimators=200,
        random_state=args.random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)   # RF doesn't need scaled features
    y_pred_rf = rf.predict(X_test)
    print(classification_report(y_test, y_pred_rf, target_names=["human", "AI"], digits=4))

    # ── Feature importances ──────────────────────────────────────────────────
    print("-" * 60)
    print("  RANDOM FOREST FEATURE IMPORTANCES (sorted)")
    print("-" * 60)
    names = feature_names()
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    for rank, idx in enumerate(sorted_idx, 1):
        print(f"  {rank:>2}. {names[idx]:<35} {importances[idx]:.4f}")
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Baseline classifiers (Logistic Regression + Random Forest) using\n"
            "stylometric features only. Establishes the accuracy floor for the\n"
            "DNN + perplexity model to beat."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--human-dir", default="data/human",
                        help="Directory of human .py files (default: data/human)")
    parser.add_argument("--ai-dir", default="data/ai",
                        help="Directory of AI .py files (default: data/ai)")
    parser.add_argument("--test-size", type=float, default=0.25,
                        help="Fraction of data for test set (default: 0.25)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    run_baseline(args)
