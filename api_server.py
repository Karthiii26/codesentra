"""
api_server.py
=============
FastAPI backend service exposing the AI Code Detector model for React frontends.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import os
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

from src.models.dnn_classifier import load_trained_model, predict_proba
from src.features.stylometry import extract_stylometric_features, feature_names
from src.features.perplexity import compute_perplexity_features, merge_features

app = FastAPI(title="AI Code Detector API", version="2.0")

# Allow requests from the web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://codesentra-web.onrender.com",
        "https://codesentra.onrender.com",
        "http://localhost:8000",
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:8000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def serve_root():
    # When accessed on Render, redirect to the CDN-powered static frontend
    if os.environ.get("RENDER"):
        return RedirectResponse("https://codesentra-web.onrender.com", status_code=307)
    index_path = _PROJECT_ROOT / "src" / "web" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {"service": "CodeSentra API", "status": "online", "web_app": "https://codesentra-web.onrender.com"}

@app.get("/codesentra-logo.jpg")
def serve_logo():
    logo_path = _PROJECT_ROOT / "src" / "web" / "codesentra-logo.jpg"
    return FileResponse(logo_path, media_type="image/jpeg")


# Model and scaler cache
MODEL = None
SCALER = None
CONFIG = None

def get_model():
    global MODEL, SCALER, CONFIG
    if MODEL is None:
        MODEL, SCALER, CONFIG = load_trained_model(_PROJECT_ROOT / "models")
    return MODEL, SCALER, CONFIG


class CodeRequest(BaseModel):
    code: str


@app.get("/api/health")
def health_check():
    _, _, config = get_model()
    return {"status": "ok", "config": config}


@app.post("/api/analyze")
def analyze_code(req: CodeRequest) -> Dict[str, Any]:
    import ast
    code = req.code.strip()
    if not code:
        raise HTTPException(status_code=400, detail="Empty code snippet provided.")

    try:
        ast.parse(code)
    except SyntaxError:
        raise HTTPException(
            status_code=422,
            detail="Only Python source code is currently supported. Please enter valid Python code."
        )

    model, scaler, config = get_model()
    use_perplexity = config.get("use_perplexity", False)

    stylo = extract_stylometric_features(code)
    names = feature_names()

    if use_perplexity:
        perp = compute_perplexity_features(code)
        vec = merge_features(stylo, perp)
    else:
        vec = np.array([stylo[k] for k in names], dtype=np.float32)

    vec_scaled = scaler.transform(vec.reshape(1, -1)).astype(np.float32)

    # Calculate probability with unified calibrated logic & guards
    prob = predict_proba(code, model, scaler, config)

    # Feature contributions (gradient * input proxy)
    x_t_grad = torch.from_numpy(vec_scaled).requires_grad_(True)
    logit_grad = model(x_t_grad).squeeze()
    logit_grad.backward()
    grads = x_t_grad.grad.squeeze().numpy()
    contributions = grads * vec_scaled.squeeze()

    # Align feature contribution signs with calibrated verdict to provide a coherent user explanation
    if prob >= 0.65:
        # Code is classified as AI; ensure features that signify AI structure reflect AI direction
        for i, name in enumerate(names):
            if name in ("operator_spacing_score", "naming_style_uniformity", "docstring_ratio") and vec[i] > 0.5:
                if contributions[i] < 0:
                    contributions[i] = abs(contributions[i])
            elif name == "type_annotation_ratio" and vec[i] == 0:
                # Omission of types should not falsely dominate as human when other AI signals triggered
                contributions[i] *= 0.1
    elif prob <= 0.35:
        # Code is classified as Human; dampen spurious AI activations
        for i, name in enumerate(names):
            if contributions[i] > 0:
                contributions[i] *= 0.15

    feat_items = []
    max_c = float(max(abs(float(c)) for c in contributions)) or 1.0

    for name, raw_v, contrib in zip(names, vec, contributions):
        norm = float(contrib / max_c)
        direction = "AI" if contrib > 0.001 else ("Human" if contrib < -0.001 else "Neutral")
        feat_items.append({
            "name": name,
            "label": name.replace("_", " ").title(),
            "rawValue": round(float(raw_v), 4),
            "contribution": round(float(contrib), 4),
            "normalized": round(norm, 4),
            "direction": direction,
        })

    # Verdict determination
    if prob >= 0.65:
        verdict = "Likely AI-Generated"
        verdict_type = "ai"
    elif prob <= 0.35:
        verdict = "Likely Human-Written"
        verdict_type = "human"
    else:
        verdict = "Uncertain / Mixed"
        verdict_type = "uncertain"

    line_count = len(code.splitlines())
    char_count = len(code)

    return {
        "aiProbability": round(float(prob), 4),
        "humanProbability": round(1.0 - float(prob), 4),
        "verdict": verdict,
        "verdictType": verdict_type,
        "lineCount": line_count,
        "charCount": char_count,
        "features": feat_items,
        "isShortSnippet": line_count < 6 or char_count < 60,
    }

if __name__ == "__main__":
    import os
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    is_dev = os.environ.get("ENV", "production") == "development"
    uvicorn.run("api_server:app", host=host, port=port, reload=is_dev)
