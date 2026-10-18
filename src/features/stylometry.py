"""
stylometry.py
=============
Stylometric feature extraction for Python source code.

Given a Python source string, returns a dict of 11 features computed using
the `ast` module and `radon`. These capture patterns that distinguish
human-written code from AI-generated code:

  - identifier_entropy       : Shannon entropy over identifier names
  - comment_density          : comment lines / total lines
  - whitespace_consistency   : std-dev of leading-indent widths
                               (AI code is very uniform; human code less so)
  - avg_line_length          : mean character count per non-blank line
  - cyclomatic_complexity    : average cyclomatic complexity via radon cc_visit
  - naming_style_uniformity  : fraction of identifiers following the single most
                               common style (snake_case / camelCase / single-letter)
  - type_annotation_ratio    : fraction of function args + return types that have
                               type annotations (AI code is heavily annotated)
  - docstring_ratio          : fraction of functions/classes that have docstrings
                               (AI code adds docstrings far more consistently)
  - operator_spacing_score   : fraction of binary operators with surrounding spaces
                               (AI code follows PEP 8 consistently; human often omits spaces)
  - avg_identifier_length    : mean character length of unique user-defined identifiers
                               (AI uses longer descriptive names; human uses terse ones)
  - unique_identifier_ratio  : unique identifiers / total identifiers
                               (AI uses more varied vocab; human repeats short names)

Removed vs. previous version:
  - blank_ratio: was severely biased by snippet length — short snippets always
    return 0 blank lines regardless of coding style.

All values are floats. SyntaxError is handled gracefully — all features
return 0.0 so the feature vector is still usable downstream.

Usage:
    python src/features/stylometry.py path/to/file.py
    python src/features/stylometry.py  # runs built-in test example
"""

import ast
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# radon is a required dependency (see requirements.txt)
try:
    from radon.complexity import cc_visit  # type: ignore
    from radon.raw import analyze as radon_analyze  # type: ignore
except ImportError:
    sys.exit("[ERROR] radon is not installed. Run: pip install radon")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_stylometric_features(source: str) -> dict[str, float]:
    """Extract stylometric features from a Python source string.

    Parameters
    ----------
    source : str
        Raw Python source code.

    Returns
    -------
    dict[str, float]
        Feature dictionary. All values are float. Returns zeros on SyntaxError.
    """
    # Always return these exact keys (in this order) so downstream code can
    # reliably convert to a fixed-length numpy array.
    zero_features: dict[str, float] = {
        "identifier_entropy": 0.0,
        "comment_density": 0.0,
        "whitespace_consistency": 0.0,
        "avg_line_length": 0.0,
        "cyclomatic_complexity": 0.0,
        "naming_style_uniformity": 0.0,
        "type_annotation_ratio": 0.0,
        "docstring_ratio": 0.0,
        "operator_spacing_score": 0.0,
        "avg_identifier_length": 0.0,
        "unique_identifier_ratio": 0.0,
    }

    if not source or not source.strip():
        return zero_features

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return zero_features

    lines = source.splitlines()

    features: dict[str, float] = {}

    # -- 1. identifier_entropy -----------------------------------------------
    features["identifier_entropy"] = _identifier_entropy(tree)

    # -- 2. comment_density --------------------------------------------------
    features["comment_density"] = _comment_density(lines)

    # -- 3. whitespace_consistency -------------------------------------------
    features["whitespace_consistency"] = _whitespace_consistency(lines)

    # -- 4. avg_line_length --------------------------------------------------
    features["avg_line_length"] = _avg_line_length(lines)

    # -- 5. cyclomatic_complexity --------------------------------------------
    features["cyclomatic_complexity"] = _cyclomatic_complexity(source)

    # -- 6. naming_style_uniformity ------------------------------------------
    features["naming_style_uniformity"] = _naming_style_uniformity(tree)

    # -- 7. type_annotation_ratio (NEW) -------------------------------------
    features["type_annotation_ratio"] = _type_annotation_ratio(tree)

    # -- 8. docstring_ratio (NEW) -------------------------------------------
    features["docstring_ratio"] = _docstring_ratio(tree)

    # -- 9. operator_spacing_score (NEW) ------------------------------------
    features["operator_spacing_score"] = _operator_spacing_score(source, lines)

    # -- 10. avg_identifier_length (NEW) ------------------------------------
    features["avg_identifier_length"] = _avg_identifier_length(tree)

    # -- 11. unique_identifier_ratio (NEW) ----------------------------------
    features["unique_identifier_ratio"] = _unique_identifier_ratio(tree)

    return features


def feature_names() -> list[str]:
    """Return the ordered list of stylometric feature names.

    Note: blank_ratio was removed — it is biased by snippet length
    (short snippets always return 0 regardless of coding style).
    All features here are normalized/ratio-based.
    """
    return [
        "identifier_entropy",
        "comment_density",
        "whitespace_consistency",
        "avg_line_length",
        "cyclomatic_complexity",
        "naming_style_uniformity",
        "type_annotation_ratio",
        "docstring_ratio",
        "operator_spacing_score",
        "avg_identifier_length",
        "unique_identifier_ratio",
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _collect_identifiers(tree: ast.AST) -> list[str]:
    """Collect all variable, function, argument, and class names from an AST."""
    identifiers: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            identifiers.append(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            identifiers.append(node.name)
            for arg in node.args.args + node.args.posonlyargs + node.args.kwonlyargs:
                identifiers.append(arg.arg)
            if node.args.vararg:
                identifiers.append(node.args.vararg.arg)
            if node.args.kwarg:
                identifiers.append(node.args.kwarg.arg)
        elif isinstance(node, ast.ClassDef):
            identifiers.append(node.name)
        elif isinstance(node, ast.Attribute):
            identifiers.append(node.attr)
    # Exclude Python builtins and very common tokens that add noise
    builtins = {
        "self", "cls", "None", "True", "False", "__init__", "__str__",
        "__repr__", "__len__", "__main__", "print", "range", "len",
        "int", "str", "list", "dict", "set", "tuple", "float", "bool",
        "append", "extend", "items", "keys", "values", "get", "return",
    }
    return [i for i in identifiers if i not in builtins]


def _identifier_entropy(tree: ast.AST) -> float:
    """Shannon entropy (bits) over the distribution of identifier tokens."""
    ids = _collect_identifiers(tree)
    if not ids:
        return 0.0
    counts = Counter(ids)
    total = sum(counts.values())
    entropy = 0.0
    for cnt in counts.values():
        p = cnt / total
        entropy -= p * math.log2(p)
    return entropy


def _comment_density(lines: list[str]) -> float:
    """Fraction of lines that are comments (start with '#' after stripping)."""
    if not lines:
        return 0.0
    comment_count = sum(1 for line in lines if line.strip().startswith("#"))
    return comment_count / len(lines)


def _whitespace_consistency(lines: list[str]) -> float:
    """Std-dev of leading-whitespace widths across non-blank lines.

    Higher → more irregular indentation (human-like).
    Lower  → very uniform indentation (AI-like).
    """
    indents: list[int] = []
    for line in lines:
        if not line.strip():
            continue
        leading = len(line) - len(line.lstrip())
        indents.append(leading)
    if len(indents) < 2:
        return 0.0
    mean = sum(indents) / len(indents)
    variance = sum((x - mean) ** 2 for x in indents) / len(indents)
    return math.sqrt(variance)


def _avg_line_length(lines: list[str]) -> float:
    """Mean character count of non-blank lines."""
    non_blank = [line for line in lines if line.strip()]
    if not non_blank:
        return 0.0
    return sum(len(line) for line in non_blank) / len(non_blank)


def _cyclomatic_complexity(source: str) -> float:
    """Average cyclomatic complexity across functions, or top-level script complexity."""
    try:
        results = cc_visit(source)
        if results:
            return float(sum(r.complexity for r in results) / len(results))
        # Top-level script without functions: count decision points in AST
        tree = ast.parse(source)
        decision_nodes = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.ExceptHandler, ast.With, ast.AsyncWith)
        complexity = 1.0 + sum(1.0 for n in ast.walk(tree) if isinstance(n, decision_nodes))
        return float(complexity)
    except Exception:  # noqa: BLE001
        return 1.0


def _classify_name_style(name: str) -> str:
    """Classify an identifier name into a naming style."""
    if len(name) == 1:
        return "single_letter"
    if name.upper() == name and "_" not in name:
        return "screaming_snake"
    if "_" in name and name == name.lower():
        return "snake_case"
    if name[0].isupper() and "_" not in name:
        return "PascalCase"
    if name[0].islower() and any(c.isupper() for c in name) and "_" not in name:
        return "camelCase"
    return "other"


def _naming_style_uniformity(tree: ast.AST) -> float:
    """Fraction of identifiers that follow the single most common naming style.

    AI code tends toward high uniformity (almost all snake_case).
    Human code mixes styles (snake_case + camelCase + single-letter).
    """
    ids = _collect_identifiers(tree)
    if not ids:
        return 0.0
    styles = [_classify_name_style(name) for name in ids]
    counter = Counter(styles)
    most_common_count = counter.most_common(1)[0][1]
    return most_common_count / len(styles)


def _radon_raw(source: str) -> tuple[float, float, float]:
    """Return (loc, sloc, blank_ratio) via radon raw analyzer.

    NOTE: blank_ratio is computed here but intentionally NOT included in the
    feature set — it is too correlated with snippet length. Kept for potential
    future use in non-short-snippet contexts.
    """
    try:
        result = radon_analyze(source)
        loc = float(result.loc)
        sloc = float(result.sloc)
        blank = float(result.blank)
        blank_ratio = blank / loc if loc > 0 else 0.0
        return loc, sloc, blank_ratio
    except Exception:  # noqa: BLE001
        return 0.0, 0.0, 0.0


# ---------------------------------------------------------------------------
# New feature helpers
# ---------------------------------------------------------------------------

def _type_annotation_ratio(tree: ast.AST) -> float:
    """Fraction of function parameters + return types that are annotated.

    AI-generated code almost always uses type annotations.
    Human competitive/quick code almost never does.
    Returns a value in [0, 1], or 0.0 if no functions found.
    """
    total_slots = 0
    annotated_slots = 0

    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        all_args = (
            node.args.args
            + node.args.posonlyargs
            + node.args.kwonlyargs
        )
        if node.args.vararg:
            all_args = list(all_args) + [node.args.vararg]
        if node.args.kwarg:
            all_args = list(all_args) + [node.args.kwarg]
        # Filter out 'self' and 'cls' — they almost never have annotations
        filtered_args = [a for a in all_args if a.arg not in ("self", "cls")]
        total_slots += len(filtered_args) + 1  # +1 for return type
        annotated_slots += sum(1 for a in filtered_args if a.annotation is not None)
        if node.returns is not None:
            annotated_slots += 1

    if total_slots == 0:
        return 0.0
    return annotated_slots / total_slots


def _docstring_ratio(tree: ast.AST) -> float:
    """Fraction of functions and classes that have docstrings.

    AI-generated code almost always has docstrings on every function.
    Human code (especially competitive/quick code) rarely does.
    Returns a value in [0, 1], or 0.0 if no functions/classes found.
    """
    total = 0
    with_docstring = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            total += 1
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                with_docstring += 1

    if total == 0:
        return 0.0
    return with_docstring / total


def _operator_spacing_score(source: str, lines: list[str]) -> float:
    """Fraction of binary/assignment operators that have spaces on both sides.

    AI-generated code is very consistent about PEP 8 spacing: `a + b`, `x = 1`.
    Human competitive/quick code often compresses: `a+b`, `n=int(input())`.
    Returns a value in [0, 1].
    """
    with_space = 0
    without_space = 0

    # Operators to inspect
    ops = ["==", "!=", "<=", ">=", "+=", "-=", "*=", "/=", "=", "<", ">", "+", "-", "*"]

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        cleaned = re.sub(r'["\'].*?["\']', '""', stripped)
        cleaned = re.sub(r'#.*$', '', cleaned)

        for op in ops:
            # Avoid double counting or misidentifying sub-tokens (e.g. += as +)
            esc_op = re.escape(op)
            # Spaced: non-symbol space before and after
            spaced_pattern = r'(?<![=!<>+\-*/])\s+' + esc_op + r'\s+(?![=!<>+\-*/])'
            # Any occurrence (not part of a larger operator)
            total_pattern = r'(?<![=!<>+\-*/])' + esc_op + r'(?![=!<>+\-*/])'
            
            spaced_count = len(re.findall(spaced_pattern, cleaned))
            total_count = len(re.findall(total_pattern, cleaned))
            if total_count > 0:
                with_space += spaced_count
                without_space += max(0, total_count - spaced_count)

    total = with_space + without_space
    if total == 0:
        return 0.5  # neutral baseline
    return float(with_space / total)


def _avg_identifier_length(tree: ast.AST) -> float:
    """Mean character length of unique user-defined identifiers.

    AI-generated code uses longer, more descriptive names.
    Human competitive code uses terse names: i, n, dp, tmp, res.
    """
    ids = _collect_identifiers(tree)
    if not ids:
        return 0.0
    unique_ids = list(set(ids))
    return sum(len(name) for name in unique_ids) / len(unique_ids)


def _unique_identifier_ratio(tree: ast.AST) -> float:
    """Unique identifier count / total identifier count.

    AI code uses many distinct descriptive names (high unique ratio).
    Human competitive code reuses short names heavily (lower unique ratio).
    Returns a value in [0, 1], or 0.0 if no identifiers.
    """
    ids = _collect_identifiers(tree)
    if not ids:
        return 0.0
    return len(set(ids)) / len(ids)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _print_features(features: dict[str, float], label: str = "") -> None:
    """Pretty-print a feature dict."""
    if label:
        print(f"\n=== {label} ===")
    max_key_len = max(len(k) for k in features)
    for key, val in features.items():
        print(f"  {key:<{max_key_len}} : {val:.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] in ("-h", "--help"):
            print("Usage: python stylometry.py [path/to/file.py]")
            print("Extracts stylometric features for a Python source file.")
            sys.exit(0)
        # File mode: extract features from a given .py file
        path = Path(sys.argv[1])
        if not path.exists():
            sys.exit(f"[ERROR] File not found: {path}")
        source = path.read_text(encoding="utf-8")
        features = extract_stylometric_features(source)
        _print_features(features, label=str(path))
    else:
        # Built-in test: compare a compact AI-style snippet vs a messier human-style snippet
        AI_STYLE = '''\
def two_sum(nums: list[int], target: int) -> list[int]:
    """Return indices of two numbers that sum to target."""
    seen: dict[int, int] = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
'''

        HUMAN_STYLE = '''\
def twoSum(nums,target):
    # brute force lol
    d={}
    for i,n in enumerate(nums):
        comp=target-n
        if comp in d:
            return[d[comp],i]
        d[n]=i
    # shouldnt reach here
    return None
'''

        print("Built-in smoke test -- comparing AI-style vs Human-style snippet")
        print("(Higher type_annotation_ratio / docstring_ratio / avg_identifier_length -> more AI-like)\n")

        ai_feats = extract_stylometric_features(AI_STYLE)
        _print_features(ai_feats, label="AI-style snippet")

        human_feats = extract_stylometric_features(HUMAN_STYLE)
        _print_features(human_feats, label="Human-style snippet")

        print("\nDifferences (Human - AI):")
        for key in feature_names():
            diff = human_feats[key] - ai_feats[key]
            direction = "^ human higher" if diff > 0.001 else ("v AI higher" if diff < -0.001 else "~ similar")
            print(f"  {key:<35} {diff:+.4f}  {direction}")
