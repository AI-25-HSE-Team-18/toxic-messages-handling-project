from fastapi import APIRouter, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from database import get_db
from domain.models import User
from core.security import create_access_token
from fastapi import Depends

router = APIRouter(
    prefix="/login", 
    tags=["Authentication"]
    )

@router.post("")
async def login(email: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(User).where(User.email == email)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    token = create_access_token(user.id)

    return {
        "access_token": token, 
        "token_type": "bearer"
        }