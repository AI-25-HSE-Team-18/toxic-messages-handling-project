"""
Monitoring endpoints (unauthenticated) for Grafana polling.
Provides recent predictions data with source info.
"""
from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from datetime import datetime

from database import get_db
from domain.models import User, UserRequests

router = APIRouter(
    prefix="/monitoring",
    tags=["Monitoring"]
)


class RecentPredictionResponse(BaseModel):
    id: int
    timestamp: datetime
    text_raw: str
    prediction: int
    prediction_label: str
    model_id: Optional[str]
    processing_time_ms: Optional[float]
    text_length: Optional[int]
    source: str


@router.get("/recent", response_model=List[RecentPredictionResponse])
async def get_recent_predictions(
    limit: int = Query(default=50, ge=1, le=500),
    db: AsyncSession = Depends(get_db)
):
    """
    Return the N most recent predictions with source info.
    Source is derived from the user name who made the request.
    No auth required — intended for Grafana polling.
    """
    result = await db.execute(
        select(UserRequests, User.name)
        .join(User, UserRequests.user_id == User.id)
        .order_by(UserRequests.timestamp.desc())
        .limit(limit)
    )
    rows = result.all()

    return [
        RecentPredictionResponse(
            id=req.id,
            timestamp=req.timestamp,
            text_raw=req.text_raw[:200],
            prediction=req.prediction,
            prediction_label=req.prediction_label or (
                "toxic" if req.prediction == 1 else "non_toxic"
            ),
            model_id=req.model_id,
            processing_time_ms=req.processing_time_ms,
            text_length=req.text_length,
            source=user_name,
        )
        for req, user_name in rows
    ]
