"""
This app uses the FastAPITutorials repository as the main sample.
It implements FastAPI service to predict whether input str messages are toxic or not.

"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from contextlib import asynccontextmanager
from database import init_db, close_db
from routers import users, forward, requests
from auth import authenticate


# ==============================================================================
# LIFESPAN CONTEXT MANAGER / КОНТЕКСТ ЖИЗНЕННОГО ЦИКЛА
# ==============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    yield

    await close_db()

# ==============================================================================
# FASTAPI APPLICATION / ПРИЛОЖЕНИЕ FASTAPI
# ==============================================================================

app = FastAPI(
    title="FastAPI SQLAlchemy Async Example",
    description="Example with async database operations using SQLAlchemy 2.0",
    version="1.0.0",
    lifespan=lifespan
)
# connect routers: 
app.include_router(users.router)     
app.include_router(authenticate.router)   
app.include_router(forward.router)   
app.include_router(requests.router)   


@app.get("/")
async def root():
    """
    Root endpoint with API information
    Корневой эндпоинт с информацией об API
    """
    return {
        "message": "FastAPI SQLAlchemy Async Toxicity handling app",
        "endpoints": {
            "create_user": "POST /users",
            "get_users": "GET /users",
            "login": "GET /login",
            "forward": "POST /forward",
            "history": "GET /history"

            # "delete_user": "DELETE /users/{user_id}",
            # "get_user": "GET /users/{user_id}",
            # "update_user": "PUT /users/{user_id}",
        },
        "docs": {
            "swagger": "/docs",
            # "redoc": "/redoc"
        },
        # "key_concepts": {
        #     "schema_vs_model": {
        #         "schema": "Pydantic models for API validation (UserCreate, UserResponse)",
        #         "model": "SQLAlchemy models for database tables (User)"
        #     },
        #     "sqlalchemy_2.0": {
        #         "old_query": "db.query(User).filter(...).all()",
        #         "new_execute": "db.execute(select(User).where(...)).scalars().all()"
        #     },
        #     "async_patterns": {
        #         "session": "AsyncSession from async_sessionmaker",
        #         "engine": "create_async_engine with aiosqlite/asyncpg",
        #         "dependency": "Depends(get_db) for session injection"
        #     }
        # }
    }