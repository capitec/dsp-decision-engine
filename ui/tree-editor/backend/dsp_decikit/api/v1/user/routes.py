from fastapi import APIRouter

TAGS = [
    {"name": "User Profile", "description": "User profile and team management"},
    {"name": "Users", "description": "User search and management"},
]

user_router = APIRouter(prefix="/user", tags=["User Profile"])
users_router = APIRouter(prefix="/users", tags=["Users"])

__all__ = ["user_router", "users_router", "TAGS"]
