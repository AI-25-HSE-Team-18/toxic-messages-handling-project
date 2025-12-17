from pydantic import BaseModel, ConfigDict, Field
from typing import AsyncGenerator, List, Optional
from datetime import datetime, timedelta

# ==============================================================================
# PYDANTIC SCHEMAS (API Validation) / PYDANTIC СХЕМЫ (Валидация API)
# ==============================================================================

class UserBase(BaseModel):
    """
    Base Pydantic Schema - shared fields
    Базовая Pydantic схема - общие поля

    This is a Pydantic model for API validation, NOT a database model!
    Это Pydantic модель для валидации API, НЕ модель базы данных!
    """
    name: str = Field(..., min_length=1, max_length=100, description="User name")
    email: str = Field(..., description="User email address")
    age: Optional[int] = Field(None, ge=0, le=150, description="User age")

    
class RequestsBase(BaseModel):
    """
    Base Pydantic schema for user requests (ML inference input)
    Used for validation and serialization in API layer.
    """

    text_raw: str = Field(
        ...,
        min_length=1,
        description="Raw text input sent to the ML model"
    )

class RequestCreate(BaseModel):
    text_raw: str

class RequestResponse(RequestsBase):
    """
    Schema for returning logged request data
    """

    id: int
    user_id: int
    timestamp: datetime
    prediction: int
    model_config = ConfigDict(from_attributes=True)

class UserCreate(UserBase):
    """
    Schema for creating a user (input from API request)
    Схема для создания пользователя (входные данные из API запроса)

    Used in POST /users endpoint
    Используется в POST /users эндпоинте
    """
    pass


class UserUpdate(BaseModel):
    """
    Schema for updating a user (all fields optional)
    Схема для обновления пользователя (все поля необязательные)

    Used in PUT /users/{user_id} endpoint
    Используется в PUT /users/{user_id} эндпоинте
    """
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    email: Optional[str] = None
    age: Optional[int] = Field(None, ge=0, le=150)


class UserResponse(UserBase):
    """
    Schema for returning user data (output to API response)
    Схема для возврата данных пользователя (вывод в API ответ)

    Includes database-generated fields like 'id'
    Включает сгенерированные БД поля, такие как 'id'

    Used in GET /users and GET /users/{user_id} endpoints
    Используется в GET /users и GET /users/{user_id} эндпоинтах
    """
    id: int

    # Configure Pydantic to work with SQLAlchemy models
    # Настроить Pydantic для работы с моделями SQLAlchemy
    model_config = ConfigDict(from_attributes=True)


