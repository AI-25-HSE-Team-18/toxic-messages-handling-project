import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domain.models import User, UserRequests
from schemas.schemas import RequestsBase, RequestResponse
from auth.dependencies import get_current_user
from services.model import PickleModel

model = PickleModel()

router = APIRouter(
    prefix="/forward",
    tags=["(RU) Predicting message toxicity"]
)


@router.post("", response_model=RequestResponse)
async def forward(
    request: RequestsBase,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    start_time = time.perf_counter()
    
    if not request.text_raw:
        raise HTTPException(status_code=400, detail="bad request")

    text = model.preprocess(request.text_raw)
    prediction = model.predict(text)

    if prediction is None:
        raise HTTPException(
            status_code=403,
            detail="модель не смогла обработать данные"
        )
    
    end_time = time.perf_counter()
    processing_time_ms = (end_time - start_time) * 1000

    db_request = UserRequests(
        user_id=current_user.id,
        text_raw=request.text_raw,
        prediction=prediction,
        processing_time_ms=processing_time_ms,
        text_length=len(request.text_raw)
    )

    db.add(db_request)
    await db.commit()
    await db.refresh(db_request)

    return db_request
