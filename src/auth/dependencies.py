from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.security import decode_access_token
from database import get_db
from domain.models import User, UserRole

security = HTTPBearer(auto_error=False)  # auto_error=False allows optional auth


def is_localhost(request: Request) -> bool:
    """
    Check if request is coming from localhost.
    
    Args:
        request: FastAPI request object
        
    Returns:
        True if request is from localhost, False otherwise
    """
    client_host = request.client.host
    return client_host in ("127.0.0.1", "localhost")


async def get_localhost_user(db: AsyncSession) -> User:
    """
    Get or create a default localhost user for development/testing.
    
    Args:
        db: Database session
        
    Returns:
        Localhost user for unauthenticated access
    """
    # Try to find existing localhost user
    result = await db.execute(
        select(User).where(User.email == "localhost@localhost.local")
    )
    user = result.scalar_one_or_none()
    
    # Create localhost user if doesn't exist
    if not user:
        user = User(
            name="Localhost User",
            email="localhost@localhost.local",
            age=None,
            role=UserRole.USER
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    
    return user


async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current authenticated user.
    
    - For localhost requests: Token optional, returns localhost user if not provided
    - For remote requests: Token required
    
    Args:
        request: FastAPI request object
        credentials: Optional HTTP Bearer token
        db: Database session
        
    Returns:
        Authenticated User object
        
    Raises:
        HTTPException: If token is invalid or user not authenticated
    """
    # Allow localhost to proceed without token
    if is_localhost(request):
        if not credentials:
            return await get_localhost_user(db)
    else:
        # Remote clients must provide token
        if not credentials:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No authentication token provided"
            )
    
    # If credentials provided, validate them
    if credentials:
        token = credentials.credentials

        try:
            payload = decode_access_token(token)
            user_id = int(payload["sub"])
            user_role = UserRole(payload["role"])
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=str(e)
            )
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()

        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )

        return user


async def get_admin_user(
    current_user: User = Depends(get_current_user)
) -> User:
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user
