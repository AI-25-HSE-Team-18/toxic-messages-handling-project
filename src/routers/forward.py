from typing import List

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domain.models import User, UserRequests
from schemas.schemas import RequestsBase, RequestResponse
from auth.dependencies import get_current_user

router = APIRouter(
    prefix="/forward",
    tags=["(RU) Predicting message toxicity"]
)


@router.post("", response_model=List[RequestResponse])
async def forward(
    request: Request,
    body: RequestsBase,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Run all configured models in parallel for the submitted text.
    Parallel execution and Prometheus metrics are handled inside the service layer.
    """
    if not body.text_raw:
        raise HTTPException(status_code=400, detail="bad request")

    registry = request.app.state.registry
    results = await registry.run_all(body.text_raw)

    if not results:
        raise HTTPException(status_code=503, detail="no models available")

    db_rows: List[UserRequests] = []
    for model_id, pred_int, pred_label, processing_time_ms in results:
        db_rows.append(UserRequests(
            user_id=current_user.id,
            text_raw=body.text_raw,
            prediction=pred_int,
            prediction_label=pred_label,
            model_id=model_id,
            processing_time_ms=processing_time_ms,
            text_length=len(body.text_raw),
        ))

    db.add_all(db_rows)
    await db.commit()
    for row in db_rows:
        await db.refresh(row)

    return db_rows
