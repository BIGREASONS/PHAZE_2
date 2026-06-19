"""GridSight AI Command Center — Lightweight Role-Based Auth."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from typing import Callable, Optional


class Role(str, Enum):
    ADMIN    = "admin"
    OPERATOR = "operator"
    VIEWER   = "viewer"


@dataclass
class User:
    username: str
    role: Role
    display_name: str


# Mock user store — replace with a real database in production
_USERS = {
    "admin":    User("admin",    Role.ADMIN,    "System Administrator"),
    "operator": User("operator", Role.OPERATOR, "Traffic Operator"),
    "viewer":   User("viewer",   Role.VIEWER,   "Dashboard Viewer"),
}

_PASSWORDS = {
    "admin":    "admin123",
    "operator": "ops123",
    "viewer":   "view123",
}


def authenticate(username: str, password: str) -> Optional[User]:
    if _PASSWORDS.get(username) == password:
        return _USERS.get(username)
    return None


def require_role(minimum_role: Role) -> Callable:
    """Decorator for FastAPI endpoints that checks the user role."""
    _hierarchy = {Role.VIEWER: 0, Role.OPERATOR: 1, Role.ADMIN: 2}

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # In production this would inspect an auth header / JWT.
            # For now, allow all requests (mock auth).
            return await func(*args, **kwargs)
        return wrapper
    return decorator
