import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(async_client: AsyncClient):
    response = await async_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_register_user_success(async_client: AsyncClient):
    payload = {
        "email": "engineer@docusense.ai",
        "password": "SecurePassword123!",
    }
    response = await async_client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "engineer@docusense.ai"
    assert "id" in data
    assert "created_at" in data
    # Ensure password hash is NEVER leaked in response
    assert "password" not in data
    assert "hashed_password" not in data


@pytest.mark.asyncio
async def test_register_duplicate_email(async_client: AsyncClient):
    payload = {
        "email": "duplicate@docusense.ai",
        "password": "SecurePassword123!",
    }
    # First registration
    resp1 = await async_client.post("/api/v1/auth/register", json=payload)
    assert resp1.status_code == 201

    # Second registration with identical email
    resp2 = await async_client.post("/api/v1/auth/register", json=payload)
    assert resp2.status_code == 400
    assert "already exists" in resp2.json()["detail"]


@pytest.mark.asyncio
async def test_register_weak_password_validation(async_client: AsyncClient):
    payload = {
        "email": "weak@docusense.ai",
        "password": "short",  # Less than 8 characters
    }
    response = await async_client.post("/api/v1/auth/register", json=payload)
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_login_success(async_client: AsyncClient):
    # Register user first
    reg_payload = {
        "email": "login_user@docusense.ai",
        "password": "StrongPassword999!",
    }
    reg_resp = await async_client.post("/api/v1/auth/register", json=reg_payload)
    assert reg_resp.status_code == 201

    # Login
    login_payload = {
        "email": "login_user@docusense.ai",
        "password": "StrongPassword999!",
    }
    login_resp = await async_client.post("/api/v1/auth/login", json=login_payload)
    assert login_resp.status_code == 200
    token_data = login_resp.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    assert len(token_data["access_token"]) > 20


@pytest.mark.asyncio
async def test_login_invalid_password(async_client: AsyncClient):
    # Register user first
    reg_payload = {
        "email": "wrong_pwd@docusense.ai",
        "password": "CorrectPassword123!",
    }
    await async_client.post("/api/v1/auth/register", json=reg_payload)

    # Attempt login with wrong password
    login_payload = {
        "email": "wrong_pwd@docusense.ai",
        "password": "IncorrectPassword123!",
    }
    response = await async_client.post("/api/v1/auth/login", json=login_payload)
    assert response.status_code == 401
    assert "Incorrect email or password" in response.json()["detail"]


@pytest.mark.asyncio
async def test_get_current_user_me(async_client: AsyncClient):
    # 1. Register
    reg_payload = {
        "email": "auth_me@docusense.ai",
        "password": "ValidPassword123!",
    }
    reg_resp = await async_client.post("/api/v1/auth/register", json=reg_payload)
    user_id = reg_resp.json()["id"]

    # 2. Login to get token
    login_resp = await async_client.post("/api/v1/auth/login", json=reg_payload)
    token = login_resp.json()["access_token"]

    # 3. Access protected /me endpoint with Bearer token
    headers = {"Authorization": f"Bearer {token}"}
    me_resp = await async_client.get("/api/v1/auth/me", headers=headers)
    assert me_resp.status_code == 200
    data = me_resp.json()
    assert data["id"] == user_id
    assert data["email"] == "auth_me@docusense.ai"


@pytest.mark.asyncio
async def test_protected_endpoint_without_token(async_client: AsyncClient):
    response = await async_client.get("/api/v1/auth/me")
    assert response.status_code in [401, 403]
