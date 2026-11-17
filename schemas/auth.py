from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserRegisterRequest(BaseModel):
    email: EmailStr = Field(..., description="Valid user email address", examples=["engineer@example.com"])
    password: str = Field(..., min_length=8, max_length=128, description="Secure user password (min 8 characters)")


class UserLoginRequest(BaseModel):
    email: EmailStr = Field(..., description="Registered user email", examples=["engineer@example.com"])
    password: str = Field(..., description="User password")


class TokenResponse(BaseModel):
    access_token: str = Field(..., description="JWT Bearer access token")
    token_type: str = Field(default="bearer", description="Token type")


class TokenPayload(BaseModel):
    sub: Optional[str] = None
    exp: Optional[int] = None
