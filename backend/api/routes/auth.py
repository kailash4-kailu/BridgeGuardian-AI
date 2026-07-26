"""
BridgeGuardian AI — Authentication API Endpoints (/auth)
Provides user registration, JWT login, refresh token issuance, logout token revocation, and profile fetching.
"""
from __future__ import annotations

from typing import Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Header, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from backend.api.deps import get_current_user
from backend.app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from backend.core.database import get_db
from backend.core.models import User, UserSession

router = APIRouter()


class RegisterRequest(BaseModel):
    email: str
    password: str
    full_name: str
    role: str = "Inspector"  # "Admin", "Inspector", "Viewer"


class RefreshTokenRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    role: str
    user_id: int


class UserProfileResponse(BaseModel):
    id: int
    email: str
    full_name: str
    role: str
    is_active: bool


@router.post("/auth/register", response_model=UserProfileResponse, status_code=status.HTTP_201_CREATED)
async def register(payload: RegisterRequest, db: Session = Depends(get_db)) -> Any:
    """Register a new user account."""
    existing = db.query(User).filter(User.email == payload.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email address already registered",
        )

    allowed_roles = ["Admin", "admin", "Inspector", "inspector", "field_inspector", "structural_engineer", "Viewer", "viewer"]
    if payload.role not in allowed_roles:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role '{payload.role}'. Allowed roles: Admin, Inspector, Viewer.",
        )

    user = User(
        email=payload.email,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        role=payload.role,
        is_active=1,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return UserProfileResponse(
        id=user.id,
        email=user.email,
        full_name=user.full_name or "",
        role=user.role,
        is_active=bool(user.is_active),
    )


@router.post("/auth/login", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
) -> Any:
    """Authenticate user credentials and issue signed JWT access and refresh tokens."""
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is deactivated",
        )

    access_token = create_access_token(subject=user.id, role=user.role)
    refresh_token = create_refresh_token(subject=user.id, role=user.role)

    # Track active session JTI
    payload = decode_access_token(access_token)
    if payload and "jti" in payload:
        session_rec = UserSession(user_id=user.id, token_jti=payload["jti"], is_revoked=0)
        db.add(session_rec)
        db.commit()

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        role=user.role,
        user_id=user.id,
    )


@router.post("/auth/refresh", response_model=TokenResponse)
async def refresh_token(
    payload: RefreshTokenRequest,
    db: Session = Depends(get_db),
) -> Any:
    """Exchange a valid refresh token for a new access and refresh token pair."""
    token_data = decode_access_token(payload.refresh_token)
    if not token_data or token_data.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    user_id = token_data.get("sub")
    user = db.query(User).filter(User.id == int(user_id)).first()
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated",
        )

    new_access_token = create_access_token(subject=user.id, role=user.role)
    new_refresh_token = create_refresh_token(subject=user.id, role=user.role)

    return TokenResponse(
        access_token=new_access_token,
        refresh_token=new_refresh_token,
        token_type="bearer",
        role=user.role,
        user_id=user.id,
    )


@router.post("/auth/logout", status_code=status.HTTP_200_OK)
async def logout(
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
) -> Any:
    """Revoke active JWT token session on logout."""
    if authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ")[1]
        payload = decode_access_token(token)
        if payload and "jti" in payload:
            jti = payload["jti"]
            session_rec = db.query(UserSession).filter(UserSession.token_jti == jti).first()
            if session_rec:
                session_rec.is_revoked = 1
            else:
                user_id = int(payload.get("sub", 0))
                db.add(UserSession(user_id=user_id, token_jti=jti, is_revoked=1))
            db.commit()
    return {"message": "Successfully logged out and token revoked"}


@router.get("/auth/me", response_model=UserProfileResponse)
async def get_profile(current_user: Optional[User] = Depends(get_current_user)) -> Any:
    """Fetch profile details of the authenticated user."""
    if not current_user:
        return UserProfileResponse(
            id=0,
            email="guest@bridgeguardian.ai",
            full_name="Guest Viewer",
            role="Viewer",
            is_active=True
        )
    return UserProfileResponse(
        id=current_user.id,
        email=current_user.email,
        full_name=current_user.full_name or "",
        role=current_user.role,
        is_active=bool(current_user.is_active),
    )
