#!/usr/bin/env python3

import asyncio
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from database import AsyncSessionLocal, init_db, engine
from domain.models import User, UserRole
from core.security import create_access_token
from sqlalchemy import select


async def create_admin(name: str, email: str, age: int = None):
    """
    Create new admin user
    """
    try:
        await init_db()
        
        async with AsyncSessionLocal() as db:
            result = await db.execute(select(User).where(User.email == email))
            
            if result.scalar_one_or_none():
                print(f"User with email {email} already exists")
                return False

            admin_user = User(name=name, email=email, age=age, role=UserRole.ADMIN)
            
            db.add(admin_user)
            await db.commit()
            await db.refresh(admin_user)

            token = create_access_token(admin_user.id, admin_user.role)
            
            print(f"Admin user created:")
            print(f"Name: {name}")
            print(f"Email: {email}")
            print(f"Age: {age or 'Not specified'}")
            print(f"Role: {UserRole.ADMIN.value}")
            print(f"Admin token: {token}")
            
            return True
            
    except Exception as e:
        print(f"Error: {e}")
        return False
    finally:
        await engine.dispose()


def main():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python3 create_admin.py <name> <email> <optional age>")
        sys.exit(1)
    
    name = sys.argv[1]
    email = sys.argv[2]
    age = int(sys.argv[3]) if len(sys.argv) == 4 else None
    
    success = asyncio.run(create_admin(name, email, age))
    
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
