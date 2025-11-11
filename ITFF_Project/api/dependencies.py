from fastapi import Header, HTTPException, status

from api.config import get_settings


def enforce_api_token(x_api_token: str | None = Header(default=None)) -> None:
    settings = get_settings()
    required_token = settings.api_token
    if required_token and x_api_token != required_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or missing API token")

