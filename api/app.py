from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.inference import InferenceConfig, get_inference_service


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Student feedback text.")


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="List of student feedback texts.")


app = FastAPI(
    title="Student Voice Intelligence API",
    description="Inference API for Vietnamese student feedback analysis.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "message": "Student Voice Intelligence API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    config = InferenceConfig.from_project()
    return {
        "status": "ok",
        "sentiment_model_exists": config.sentiment_model_dir.exists(),
        "topic_model_exists": config.topic_model_dir.exists(),
        "toxic_baseline_exists": config.toxic_baseline_path.exists(),
        "urgency_baseline_exists": config.urgency_baseline_path.exists(),
        "project_dir": str(config.project_dir),
    }


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    try:
        service = get_inference_service()
        return service.health()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, Any]:
    try:
        service = get_inference_service()
        return service.analyze(request.text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/predict-batch")
def predict_batch(request: BatchPredictRequest) -> dict[str, Any]:
    try:
        service = get_inference_service()
        return {"results": service.analyze_many(request.texts)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

