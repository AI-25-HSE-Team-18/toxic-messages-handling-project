
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from domain.models import User, UserRequests
# from models.request import UserRequests
from schemas.schemas import UserCreate, UserResponse


router = APIRouter(
    prefix="/users",
    tags=["Users"]
)

@router.post(
    "",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user",
    description="Create a new user in the database"
)
async def create_user(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    CREATE operation / Операция СОЗДАНИЯ

    Steps / Шаги:
    1. Check if email already exists / Проверить, существует ли email
    2. Create SQLAlchemy model instance / Создать экземпляр модели SQLAlchemy
    3. Add to session / Добавить в сессию
    4. Commit transaction / Зафиксировать транзакцию
    5. Refresh to get generated ID / Обновить для получения сгенерированного ID
    6. Return user / Вернуть пользователя
    """
    # Check if email exists / Проверка существования email
    # SQLAlchemy 2.0 style: use execute() with select()
    # Стиль 2.0: использовать execute() с select()
    result = await db.execute(
        select(User).where(User.email == user_data.email)
    )
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"User with email {user_data.email} already exists"
        )

    # Create new user model / Создание новой модели пользователя
    # Convert Pydantic schema to SQLAlchemy model
    # Преобразование Pydantic схемы в SQLAlchemy модель
    db_user = User(
        name=user_data.name,
        email=user_data.email,
        age=user_data.age
    )

    # Add to session / Добавить в сессию
    db.add(db_user)

    # Commit transaction / Зафиксировать транзакцию
    await db.commit()

    # Refresh to get auto-generated fields / Обновить для получения auto-generated полей
    await db.refresh(db_user)

    return db_user


@router.get(
    "",
    response_model=List[UserResponse],
    summary="Get all users",
    description="Retrieve all users from the database"
)
async def get_users(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db)
) -> List[User]:
    """
    READ operation (list) / Операция ЧТЕНИЯ (список)

    SQLAlchemy 2.0 pattern:
    Паттерн SQLAlchemy 2.0:
    - OLD: db.query(User).offset(skip).limit(limit).all()
    - NEW: db.execute(select(User).offset(skip).limit(limit))
    """
    result = await db.execute(
        select(User)
        .offset(skip)
        .limit(limit)
    )

    # .scalars() returns scalar values (User objects)
    # .all() converts to list
    users = result.scalars().all()

    return list(users)


# @router.get(
#     "/{user_id}",
#     response_model=UserResponse,
#     summary="Get user and their requests by user ID",
#     description="Retrieve a specific user by their ID"
# )
# async def get_user(
#     user_id: int,
#     db: AsyncSession = Depends(get_db)
# ) -> User:
#     """
#     READ operation (single item) / Операция ЧТЕНИЯ (один элемент)

#     SQLAlchemy 2.0 pattern:
#     - OLD: db.query(User).filter(User.id == user_id).first()
#     - NEW: db.execute(select(User).where(User.id == user_id))
#     """
#     result = await db.execute(
#         select(User).where(User.id == user_id)
#     )

#     # .scalar_one_or_none() returns single object or None
#     # .scalar_one_or_none() возвращает один объект или None
#     user = result.scalar_one_or_none()

#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail=f"User with id {user_id} not found"
#         )

#     return user


# @router.delete(
#     "/{user_id}",
#     status_code=status.HTTP_204_NO_CONTENT,
#     summary="Delete user and user's requests by their ID",
#     description="Delete a user from the database"
# )
# async def delete_user(
#     user_id: int,
#     db: AsyncSession = Depends(get_db)
# ) -> None:
#     """
#     DELETE operation / Операция УДАЛЕНИЯ

#     SQLAlchemy 2.0 pattern:
#     - OLD: db.query(User).filter(User.id == user_id).delete()
#     - NEW: db.execute(delete(User).where(User.id == user_id))
#     """
#     # Check if user exists / Проверить существование пользователя
#     result = await db.execute(
#         select(User).where(User.id == user_id)
#     )
#     user = result.scalar_one_or_none()

#     if not user:
#         raise HTTPException(
#             status_code=status.HTTP_404_NOT_FOUND,
#             detail=f"User with id {user_id} not found"
#         )

#     # Delete user from user's DB: 
#     await db.execute(
#         delete(User).where(User.id == user_id)
#     )
#     # delete appropriate requests: 
#     await db.execute(
#         delete(UserRequests).where(UserRequests.user_id == user_id)
#     )

#     await db.commit()
