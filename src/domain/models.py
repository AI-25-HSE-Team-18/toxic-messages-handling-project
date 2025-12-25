from sqlalchemy.sql import func
from datetime import datetime, timedelta
from typing import AsyncGenerator, List, Optional
from sqlalchemy import String, Integer, select, delete, TIMESTAMP, ForeignKey, Enum
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
import enum


class Base(DeclarativeBase):
    """
    Base class for all SQLAlchemy models
    Базовый класс для всех моделей SQLAlchemy
    """
    pass


class UserRole(enum.Enum):
    """
    User roles enum
    Возможные роли пользователей
    """
    USER = "user"
    ADMIN = "admin"


class User(Base):
    """
    SQLAlchemy Model - represents database table
    Модель SQLAlchemy - представляет таблицу в базе данных

    This is NOT a Pydantic model! It's a database model.
    Это НЕ Pydantic модель! Это модель базы данных.

    Key differences from Pydantic Schema:
    Ключевые отличия от Pydantic Schema:
    - Inherits from Base (DeclarativeBase) / Наследуется от Base
    - Uses mapped_column() for fields / Использует mapped_column для полей
    - Maps to actual database table / Соответствует реальной таблице БД
    - Used with db.execute(select(User)) / Используется с db.execute
    """
    __tablename__ = "users"

    # SQLAlchemy 2.0 style with Mapped and mapped_column
    # Стиль SQLAlchemy 2.0 с Mapped и mapped_column

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
        autoincrement=True
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        index=True
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )

    age: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True  # Optional field / Необязательное поле
    )

    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole),
        nullable=False,
        default=UserRole.USER,
        index=True
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, name={self.name}, email={self.email}, role={self.role})>"


class UserRequests(Base):
    """SQLAlchemy Model to log current single user requests"""

    __tablename__ = "user_requests"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
        autoincrement=True
    )

    user_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False
    )

    timestamp: Mapped[datetime] = mapped_column(
        TIMESTAMP(timezone=True),
        server_default=func.now(),
        index=True,
        nullable=False
    )

    text_raw: Mapped[str] = mapped_column(
        String,
        nullable=False
    )

    prediction: Mapped[int] = mapped_column(
        nullable=False,
        index=True
    )

    processing_time_ms: Mapped[Optional[float]] = mapped_column(
        nullable=True,
        index=True,
    )

    text_length: Mapped[Optional[int]] = mapped_column(
        nullable=True,
        index=True,
    )

    def __repr__(self) -> str:
        return f"<UserRequests(id={self.id}, user_id={self.user_id}, timestamp={self.timestamp})>"
