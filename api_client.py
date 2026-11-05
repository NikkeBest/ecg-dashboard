"""Small HTTP client for the FastAPI authentication endpoints."""
import os
import requests

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
TIMEOUT_SECONDS = 15


def register_user(full_name: str, email: str, password: str, role: str) -> dict:
    response = requests.post(
        f"{API_URL}/auth/register",
        json={
            "full_name": full_name,
            "email": email,
            "password": password,
            "role": role,
        },
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json() if response.content else {}


def login_user(email: str, password: str) -> dict:
    response = requests.post(
        f"{API_URL}/auth/login",
        json={"email": email, "password": password},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()
