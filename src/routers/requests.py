
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domain.models import User, UserRequests
# from models.request import UserRequests
from schemas.schemas import RequestResponse



router = APIRouter(
    prefix="/history",
    tags=["Requests History"]
)
@router.get(
    "",
    response_model=List[RequestResponse],
    summary="Get all requests from all users",
    description="Retrieve all requests from the database"
)
async def get_requests(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
) -> List[User]:
    """
    Get user requests, written as analog of thee function to get users
    """
    result = await db.execute(
        select(UserRequests)
        .offset(skip)
        .limit(limit)
    )

    # .scalars() returns scalar values (User objects)
    # .all() converts to list: 
    users = result.scalars().all()

    return list(users)