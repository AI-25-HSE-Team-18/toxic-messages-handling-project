from pydantic import BaseModel, ConfigDict, Field
from typing import Optional
from datetime import datetime
from domain.models import UserRole


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
        max_length=5000,
        description="Raw text input sent to the ML model"
    )


class RequestResponse(RequestsBase):
    """
    Schema for returning logged request data
    """
    id: int
    user_id: int
    timestamp: datetime
    prediction: int
    processing_time_ms: Optional[float] = None
    text_length: Optional[int] = None
    model_config = ConfigDict(from_attributes=True)


class StatsResponse(BaseModel):
    """
    Schema for statistics response
    """
    total_requests: int
    avg_processing_time_ms: float
    processing_time_quantiles: dict  # mean/50%/95%/99%/min/max/std
    text_characteristics: dict       # avg_length/min_length/max_length/std_length
    prediction_distribution: dict    # toxic/non_toxic/toxic_percentage


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
    role: UserRole
    model_config = ConfigDict(from_attributes=True)


class UserRegistrationResponse(BaseModel):
    """
    Schema for user registration response with token
    """
    user: UserResponse
    access_token: str
    token_type: str = "bearer"
