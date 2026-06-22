from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.inference import InferenceConfig, get_inference_service

MAX_CSV_ROWS = 5_000


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


@app.post("/predict-csv")
async def predict_csv(file: UploadFile = File(...)) -> StreamingResponse:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file.")

    try:
        payload = await file.read()
        if not payload:
            raise HTTPException(status_code=400, detail="CSV file is empty.")
        frame = pd.read_csv(BytesIO(payload), encoding="utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=400,
            detail="CSV must use UTF-8 encoding.",
        ) from exc
    except pd.errors.EmptyDataError as exc:
        raise HTTPException(status_code=400, detail="CSV file is empty.") from exc
    finally:
        await file.close()

    if "text" not in frame.columns:
        raise HTTPException(status_code=400, detail="CSV must contain a 'text' column.")
    if frame.empty:
        raise HTTPException(status_code=400, detail="CSV must contain at least one row.")
    if len(frame) > MAX_CSV_ROWS:
        raise HTTPException(
            status_code=400,
            detail=f"CSV has {len(frame)} rows; the limit is {MAX_CSV_ROWS}.",
        )

    texts = frame["text"].fillna("").astype(str).tolist()
    if any(not text.strip() for text in texts):
        raise HTTPException(status_code=400, detail="The 'text' column must not contain empty rows.")

    try:
        predictions = get_inference_service().analyze_many(texts)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    prediction_frame = pd.DataFrame(predictions)[
        [
            "sentiment",
            "sentiment_confidence",
            "topic",
            "topic_confidence",
            "toxic",
            "urgency",
        ]
    ]
    result = pd.concat([frame.reset_index(drop=True), prediction_frame], axis=1)
    content = result.to_csv(index=False).encode("utf-8-sig")

    return StreamingResponse(
        BytesIO(content),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=student_voice_predictions.csv"},
    )
