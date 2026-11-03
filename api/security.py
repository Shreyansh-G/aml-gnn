"""
security.py — API-key authentication.

Clients send the key in the `X-API-Key` header; the server reads the expected
key from the AML_API_KEY environment variable. Fails closed: if no key is
configured, protected routes return 503 instead of being left open.
"""

import hmac
import os
from typing import Optional

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY_ENV = "AML_API_KEY"
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: Optional[str] = Security(_api_key_header)) -> None:
    expected = os.environ.get(API_KEY_ENV)
    if not expected:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            f"Server misconfigured: {API_KEY_ENV} is not set")
    # Constant-time comparison avoids leaking the key through response timing
    if not key or not hmac.compare_digest(key.encode(), expected.encode()):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or missing API key",
                            headers={"WWW-Authenticate": "API-Key"})
