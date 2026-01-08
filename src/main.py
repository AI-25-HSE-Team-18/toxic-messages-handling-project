from contextlib import asynccontextmanager
from fastapi import FastAPI

from database import init_db, close_db
from routers import users, forward, requests


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
    title="Toxicity Detection API",
    description="API for text toxicity detection with user management and statistics",
    version="1.0.0",
    lifespan=lifespan
)

app.include_router(users.router)     
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
        "version": "1.0.0",
        "endpoints": {
            "register": "POST /register",
            "forward": "POST /forward",
            "history": "GET /history",
            "stats": "GET /history/stats",
            "users": "GET /users"
        },
        "docs": "/docs"
    }
