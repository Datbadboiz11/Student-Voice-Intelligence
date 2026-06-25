from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.inference import InferenceConfig, get_inference_service
from src.analytics import get_analytics_service
from src.chat import get_chat_service
from src.feedbacks import get_feedback_service
from src.rag import RAGConfigurationError, RAGGenerationError, get_rag_service
from src.reporting import get_report_service
from src.retrieval import get_retrieval_service
from src.reviews import get_review_service
from src.topic_discovery import get_topic_discovery_service

MAX_CSV_ROWS = 5_000


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Student feedback text.")


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(..., min_length=1, description="List of student feedback texts.")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Semantic search query.")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of reranked feedbacks.")
    topic: str | None = Field(default=None, description="Optional topic filter.")
    sentiment: str | None = Field(default=None, description="Optional sentiment filter.")
    urgency: str | None = Field(default=None, description="Optional urgency filter.")
    toxic: int | None = Field(default=None, ge=0, le=1, description="Optional toxic filter.")


class SearchComparisonRequest(SearchRequest):
    candidate_k: int = Field(default=20, ge=5, le=50, description="Candidates to compare before and after reranking.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="Grounded question about student feedback.")
    top_k: int = Field(default=6, ge=1, le=10, description="Number of feedbacks used as evidence.")
    topic: str | None = Field(default=None, description="Optional topic filter.")
    sentiment: str | None = Field(default=None, description="Optional sentiment filter.")
    urgency: str | None = Field(default=None, description="Optional urgency filter.")
    toxic: int | None = Field(default=None, ge=0, le=1, description="Optional toxic filter.")
    history: list[dict[str, str]] = Field(default_factory=list, max_length=16)


class ChatSessionRequest(BaseModel):
    title: str | None = Field(default=None, max_length=120)


class ChatAskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2_000)
    top_k: int = Field(default=6, ge=1, le=10)
    topic: str | None = None
    sentiment: str | None = None
    urgency: str | None = None
    toxic: int | None = Field(default=None, ge=0, le=1)


class FeedbackCreateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4_000)
    location: str = Field(default="", max_length=160)


class FeedbackStatusRequest(BaseModel):
    status: str = Field(..., pattern="^(new|in_progress|resolved)$")


class ReviewUpdateRequest(BaseModel):
    urgency_final: str = Field(..., pattern="^(low|medium|high)$")
    reviewer: str = Field(default="admin", max_length=80)
    note: str = Field(default="", max_length=500)


class ReportRequest(BaseModel):
    title: str | None = Field(default=None, max_length=160)
    dataset: str | None = None
    topic: str | None = None
    sentiment: str | None = None
    urgency: str | None = None
    toxic: int | None = Field(default=None, ge=0, le=1)


class DiscoveryRequest(BaseModel):
    topic: str | None = Field(default="others")
    max_items: int = Field(default=1200, ge=50, le=3000)
    min_cluster_size: int = Field(default=6, ge=3, le=50)


class ClusterApprovalRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)


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


@app.get("/analytics")
def analytics(
    dataset: str | None = None,
    topic: str | None = None,
    sentiment: str | None = None,
    urgency: str | None = None,
    toxic: int | None = Query(default=None, ge=0, le=1),
) -> dict[str, Any]:
    try:
        return get_analytics_service().get_analytics(
            dataset=dataset,
            topic=topic,
            sentiment=sentiment,
            urgency=urgency,
            toxic=toxic,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


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


@app.post("/feedbacks")
def create_feedback(request: FeedbackCreateRequest) -> dict[str, Any]:
    try:
        item, created = get_feedback_service().create(request.text, request.location)
        return {"item": item, "created": created}
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/feedbacks/import-csv")
async def import_feedback_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a .csv file.")
    try:
        frame = pd.read_csv(BytesIO(await file.read()), encoding="utf-8-sig")
    except (UnicodeDecodeError, pd.errors.EmptyDataError) as exc:
        raise HTTPException(status_code=400, detail="CSV must be a non-empty UTF-8 file.") from exc
    finally:
        await file.close()
    if "text" not in frame.columns or frame.empty:
        raise HTTPException(status_code=400, detail="CSV must contain at least one row and a 'text' column.")
    if len(frame) > MAX_CSV_ROWS:
        raise HTTPException(status_code=400, detail=f"CSV has {len(frame)} rows; the limit is {MAX_CSV_ROWS}.")
    rows = frame.fillna("").to_dict(orient="records")
    return get_feedback_service().import_rows(rows)


@app.get("/feedbacks")
def list_feedbacks(status: str | None = Query(default=None, pattern="^(new|in_progress|resolved)$"), topic: str | None = None, limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
    items = get_feedback_service().list(status, topic, limit)
    return {"items": items, "count": len(items)}


@app.patch("/feedbacks/{feedback_id}/status")
def update_feedback_status(feedback_id: str, request: FeedbackStatusRequest) -> dict[str, Any]:
    item = get_feedback_service().update_status(feedback_id, request.status)
    if item is None:
        raise HTTPException(status_code=404, detail="Feedback not found.")
    return item


@app.delete("/feedbacks/{feedback_id}")
def delete_feedback(feedback_id: str) -> dict[str, bool]:
    if not get_feedback_service().delete(feedback_id):
        raise HTTPException(status_code=404, detail="Feedback not found.")
    return {"deleted": True}


@app.get("/search-health")
def search_health() -> dict[str, Any]:
    return get_retrieval_service().health()


@app.post("/search")
def search(request: SearchRequest) -> dict[str, Any]:
    try:
        results = get_retrieval_service().search(
            query=request.query,
            top_k=request.top_k,
            topic=request.topic,
            sentiment=request.sentiment,
            urgency=request.urgency,
            toxic=request.toxic,
        )
        return {
            "query": request.query,
            "top_k": request.top_k,
            "results": results,
        }
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (FileNotFoundError, ConnectionError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/search/compare")
def compare_search_rankings(request: SearchComparisonRequest) -> dict[str, Any]:
    try:
        rankings = get_retrieval_service().search_rankings(
            query=request.query,
            candidate_k=request.candidate_k,
            topic=request.topic,
            sentiment=request.sentiment,
            urgency=request.urgency,
            toxic=request.toxic,
        )
        return {"query": request.query, "candidate_k": request.candidate_k, **rankings}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (FileNotFoundError, ConnectionError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask")
def ask(request: AskRequest) -> dict[str, Any]:
    try:
        return get_rag_service().ask(
            question=request.question,
            top_k=request.top_k,
            topic=request.topic,
            sentiment=request.sentiment,
            urgency=request.urgency,
            toxic=request.toxic,
            history=request.history,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RAGConfigurationError, RAGGenerationError, FileNotFoundError, ConnectionError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/chat-sessions")
def list_chat_sessions(limit: int = Query(default=50, ge=1, le=100)) -> dict[str, Any]:
    return {"items": get_chat_service().list_sessions(limit)}


@app.post("/chat-sessions")
def create_chat_session(request: ChatSessionRequest) -> dict[str, Any]:
    return get_chat_service().create_session(request.title)


@app.get("/chat-sessions/{session_id}")
def get_chat_session(session_id: int) -> dict[str, Any]:
    session = get_chat_service().get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return session


@app.delete("/chat-sessions/{session_id}")
def delete_chat_session(session_id: int) -> dict[str, bool]:
    if not get_chat_service().delete_session(session_id):
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return {"deleted": True}


@app.post("/chat-sessions/{session_id}/ask")
def ask_in_chat_session(session_id: int, request: ChatAskRequest) -> dict[str, Any]:
    try:
        response = get_chat_service().ask(
            session_id=session_id,
            question=request.question,
            top_k=request.top_k,
            topic=request.topic,
            sentiment=request.sentiment,
            urgency=request.urgency,
            toxic=request.toxic,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except (RAGConfigurationError, RAGGenerationError, FileNotFoundError, ConnectionError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if response is None:
        raise HTTPException(status_code=404, detail="Chat session not found.")
    return response


@app.get("/reviews")
def list_reviews(
    state: str = Query(default="pending", pattern="^(pending|reviewed|all)$"),
    urgency: str | None = Query(default=None, pattern="^(low|medium|high)$"),
    limit: int = Query(default=30, ge=1, le=100),
) -> dict[str, Any]:
    try:
        return get_review_service().list_feedback(state=state, urgency=urgency, limit=limit)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/reviews/{feedback_id}")
def update_review(feedback_id: str, request: ReviewUpdateRequest) -> dict[str, Any]:
    try:
        return get_review_service().save_review(
            feedback_id=feedback_id,
            urgency_final=request.urgency_final,
            reviewer=request.reviewer,
            note=request.note,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/reports/generate")
def generate_report(request: ReportRequest) -> dict[str, Any]:
    try:
        return get_report_service().generate(
            title=request.title,
            dataset=request.dataset,
            topic=request.topic,
            sentiment=request.sentiment,
            urgency=request.urgency,
            toxic=request.toxic,
        )
    except (FileNotFoundError, ConnectionError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/reports")
def list_reports(limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    return {"items": get_report_service().list_reports(limit)}


@app.get("/reports/{report_id}")
def get_report(report_id: int) -> dict[str, Any]:
    report = get_report_service().get_report(report_id)
    if report is None:
        raise HTTPException(status_code=404, detail="Report not found.")
    return report


@app.post("/topic-discovery/run")
def run_topic_discovery(request: DiscoveryRequest) -> dict[str, Any]:
    try:
        return get_topic_discovery_service().run(
            topic=request.topic,
            max_items=request.max_items,
            min_cluster_size=request.min_cluster_size,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/topic-discovery/clusters")
def list_topic_clusters(
    status: str | None = Query(default=None, pattern="^(pending|approved|rejected)$"),
    limit: int = Query(default=50, ge=1, le=100),
) -> dict[str, Any]:
    return {"items": get_topic_discovery_service().list_clusters(status=status, limit=limit)}


@app.post("/topic-discovery/clusters/{cluster_id}/approve")
def approve_topic_cluster(cluster_id: int, request: ClusterApprovalRequest) -> dict[str, Any]:
    try:
        cluster = get_topic_discovery_service().approve_cluster(cluster_id, request.name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if cluster is None:
        raise HTTPException(status_code=404, detail="Topic cluster not found.")
    return cluster


@app.post("/topic-discovery/clusters/{cluster_id}/reject")
def reject_topic_cluster(cluster_id: int) -> dict[str, Any]:
    cluster = get_topic_discovery_service().reject_cluster(cluster_id)
    if cluster is None:
        raise HTTPException(status_code=404, detail="Topic cluster not found.")
    return cluster
