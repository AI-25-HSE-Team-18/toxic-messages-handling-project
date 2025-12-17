from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from schemas.schemas import RequestsBase, RequestResponse
from auth.dependencies import get_current_user
from database import get_db
from domain.models import User, UserRequests

from services.model import PickleModel


# init model class and object: 
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
    if not request.text_raw:
        raise HTTPException(status_code=400, detail="bad request")

    # our magic is here:
    # print('request.text_raw', request.text_raw)
    text = model.preprocess(request.text_raw)
    # print('preprocessed text', text)
    prediction = model.predict(text)
    # print('prediction', prediction)
    # pred_str = "Toxic" if prediction==1 else 'Non-toxic'

    if prediction is None:
        raise HTTPException(
            status_code=403,
            detail="модель не смогла обработать данные"
        )
    
    db_request = UserRequests(
        user_id=current_user.id,
        text_raw=request.text_raw,
        prediction=prediction
    )

    db.add(db_request)
    await db.commit()
    await db.refresh(db_request)

    return db_request