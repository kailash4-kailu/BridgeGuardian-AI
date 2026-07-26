"""
BridgeGuardian AI — Authentication & Role-Based Access Control (RBAC) Dependencies
Dependency injectors for verifying user identity, checking token revocation, and enforcing RBAC.
"""
from __future__ import annotations

from typing import Callable, List, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from backend.app.core.security import decode_access_token
from backend.core.config import get_settings
from backend.core.database import get_db
from backend.core.models import User, UserSession

settings = get_settings()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

# Standardized Role Canonical Names & Synonyms
ROLE_MAP = {
    "admin": ["Admin", "admin"],
    "inspector": ["Inspector", "inspector", "field_inspector", "structural_engineer"],
    "viewer": ["Viewer", "viewer"],
}


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Optional[User]:
    """
    Extracts user identity from JWT token.
    Checks user active status and revokes invalidated token JTIs.
    Returns None if no token provided (allows public fallback if demo_mode=True).
    """
    if not token:
        return None

    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    jti = payload.get("jti")
    if jti:
        revoked_session = db.query(UserSession).filter(
            UserSession.token_jti == jti, UserSession.is_revoked == 1
        ).first()
        if revoked_session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
            )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token missing subject claim",
        )

    try:
        user = db.query(User).filter(User.id == int(user_id)).first()
    except ValueError:
        user = db.query(User).filter(User.email == str(user_id)).first()

    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account disabled or not found",
        )

    return user


def require_roles(allowed_roles: List[str]) -> Callable:
    """
    Dependency factory enforcing Role-Based Access Control (RBAC).
    Supports roles: 'Admin', 'Inspector', 'Viewer' (and legacy equivalents).
    """
    normalized_allowed = []
    for r in allowed_roles:
        r_lower = r.lower()
        normalized_allowed.extend(ROLE_MAP.get(r_lower, [r]))

    async def role_checker(
        current_user: Optional[User] = Depends(get_current_user)
    ) -> Optional[User]:
        if not current_user:
            if settings.demo_mode:
                # Return dummy system user for demo compatibility when unauthenticated
                dummy_user = User(
                    id=0,
                    email="demo@bridgeguardian.ai",
                    full_name="Demo Inspector",
                    role="Inspector",
                    is_active=1
                )
                return dummy_user
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required for this operation",
            )

        if current_user.role not in normalized_allowed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role}' is not authorized for this resource. Allowed: {allowed_roles}",
            )
        return current_user

    return role_checker


# Role Shortcut Dependencies
require_admin = require_roles(["Admin", "admin"])
require_inspector = require_roles(["Admin", "admin", "Inspector", "inspector", "field_inspector", "structural_engineer"])
require_viewer = require_roles(["Admin", "admin", "Inspector", "inspector", "field_inspector", "structural_engineer", "Viewer", "viewer"])
