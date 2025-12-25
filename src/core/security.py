from datetime import datetime, timedelta
import jwt
from core.config import SECRET_KEY, ALGORITHM
from domain.models import UserRole

def create_access_token(user_id: int, user_role: UserRole) -> str:
    payload = {
        "sub": str(user_id),
        "role": user_role.value,
        "exp": datetime.utcnow() + timedelta(minutes=60)
    }

    if not SECRET_KEY:
      raise ValueError("SECRET_KEY environment variable is required")

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_access_token(token: str) -> dict:
    payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    return payload
