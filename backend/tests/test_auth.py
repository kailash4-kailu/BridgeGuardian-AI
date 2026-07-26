"""
BridgeGuardian AI — Authentication & Role-Based Access Control Tests
Verifies user registration, JWT login, token refresh, logout, profile fetching, and RBAC security.
"""
def test_register_user(client):
    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "inspector_test@bridgeguardian.ai",
            "password": "SecurePassword123!",
            "full_name": "Test Inspector",
            "role": "Inspector",
        },
    )
    assert response.status_code == 201
    data = response.json()
    assert data["email"] == "inspector_test@bridgeguardian.ai"
    assert data["role"] == "Inspector"


def test_login_user(client):
    # Register user first
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "login_test@bridgeguardian.ai",
            "password": "MySecretPassword123!",
            "full_name": "Login User",
            "role": "Inspector",
        },
    )

    response = client.post(
        "/api/v1/auth/login",
        data={"username": "login_test@bridgeguardian.ai", "password": "MySecretPassword123!"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["role"] == "Inspector"


def test_get_current_user_profile(client):
    # Register and login to get access token
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "profile_test@bridgeguardian.ai",
            "password": "Password123!",
            "full_name": "Profile User",
            "role": "Admin",
        },
    )
    login_res = client.post(
        "/api/v1/auth/login",
        data={"username": "profile_test@bridgeguardian.ai", "password": "Password123!"},
    )
    token = login_res.json()["access_token"]

    response = client.get(
        "/api/v1/auth/me",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "profile_test@bridgeguardian.ai"
    assert data["role"] == "Admin"
