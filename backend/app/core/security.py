"""
BridgeGuardian AI — Security Core Module
Handles password hashing, salted verification, and OAuth2 JWT Token generation/validation.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

SECRET_KEY = os.getenv("SECRET_KEY", "bridgeguardian-super-secret-production-key-change-me")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 24 hours


# ─────────────────────────── Password Hashing ───────────────────────────── #

def hash_password(password: str) -> str:
    """Hash password securely using PBKDF2 HMAC-SHA256 with random salt."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)
    return base64.b64encode(salt + key).decode("ascii")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain text password against stored hash."""
    try:
        data = base64.b64decode(hashed_password.encode("ascii"))
        salt = data[:16]
        stored_key = data[16:]
        key = hashlib.pbkdf2_hmac("sha256", plain_password.encode("utf-8"), salt, 100000)
        return hmac.compare_digest(stored_key, key)
    except Exception:
        return False


# ─────────────────────────── JWT Token Management ───────────────────────── #

def create_access_token(
    subject: Union[str, int],
    role: str = "field_inspector",
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Create a signed JWT access token containing subject and role claims."""
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    payload = {
        "sub": str(subject),
        "role": role,
        "exp": int(expire.timestamp()),
        "iat": int(datetime.utcnow().timestamp()),
    }

    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode("utf-8")).decode("utf-8").rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode("utf-8")).decode("utf-8").rstrip("=")

    signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
    signature = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
    signature_b64 = base64.urlsafe_b64encode(signature).decode("utf-8").rstrip("=")

    return f"{header_b64}.{payload_b64}.{signature_b64}"


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decode and verify signature and expiration of a JWT access token."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None

        header_b64, payload_b64, signature_b64 = parts
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")

        # Verify signature
        expected_sig = hmac.new(SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
        
        # Add padding back for b64decode
        rem = len(signature_b64) % 4
        if rem > 0:
            signature_b64 += "=" * (4 - rem)
        actual_sig = base64.urlsafe_b64decode(signature_b64)

        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        # Decode payload
        rem_p = len(payload_b64) % 4
        if rem_p > 0:
            payload_b64 += "=" * (4 - rem_p)
        payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode("utf-8"))

        # Expiration check
        exp = payload.get("exp")
        if exp and datetime.utcnow().timestamp() > exp:
            return None

        return payload
    except Exception:
        return None
