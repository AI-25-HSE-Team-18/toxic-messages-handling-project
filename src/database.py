from pathlib import Path
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    AsyncSession,
    async_sessionmaker,
    AsyncEngine
)
from typing import AsyncGenerator, List, Optional
# from sqlalchemy.orm import DeclarativeBase
from domain.models import Base


async def init_db() -> None:
    # Таблицы теперь создаются через миграции Alembic
    # alembic upgrade head
    pass


async def close_db() -> None:
    await engine.dispose()

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency that provides database session
    Зависимость, предоставляющая сессию базы данных

    This is a FastAPI dependency that:
    Это FastAPI зависимость, которая:
    1. Creates a new async session / Создает новую асинхронную сессию
    2. Yields it to the endpoint / Передает её в эндпоинт
    3. Closes it after request completes / Закрывает после завершения запроса

    Usage in endpoints:
    Использование в эндпоинтах:
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


# ==============================================================================
# DATABASE SETUP / НАСТРОЙКА БАЗЫ ДАННЫХ
# ==============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = f"sqlite+aiosqlite:///{BASE_DIR}/local_requests.db"
# Create async engine / Создание асинхронного движка
engine: AsyncEngine = create_async_engine(
    DATABASE_URL,
    echo=True,  # Log all SQL queries / Логировать все SQL запросы
    future=True  # Use SQLAlchemy 2.0 style / Использовать стиль 2.0
)

# Create async session factory / Фабрика асинхронных сессий
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Don't expire objects after commit / Не истекать объекты после commit
    autocommit=False,
    autoflush=False
) 
