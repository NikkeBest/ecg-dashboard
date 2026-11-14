"""HTTP client for the ECG Monitoring FastAPI backend."""
import os
from typing import Any, Optional

import requests
import streamlit as st

try:
    API_URL = str(st.secrets["API_URL"]).rstrip("/")
except Exception:
    API_URL = os.getenv(
        "API_URL",
        "http://127.0.0.1:8000"
    ).rstrip("/")
TIMEOUT_SECONDS = 30


def _headers(token: str) -> dict:
    if not token:
        raise ValueError("You must log in before accessing ECG recordings.")
    return {"Authorization": f"Bearer {token}"}


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


def save_recording(
    token: str,
    file_name: str,
    sampling_rate: int,
    predictions: list[dict],
    started_at: Optional[str] = None,
) -> dict:
    """Save recording metadata and beat predictions for the logged-in patient."""
    payload: dict[str, Any] = {
        "file_name": file_name,
        "sampling_rate": int(sampling_rate),
        "predictions": predictions,
    }
    if started_at:
        payload["started_at"] = started_at

    response = requests.post(
        f"{API_URL}/recordings",
        json=payload,
        headers=_headers(token),
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_recordings(token: str) -> list[dict]:
    response = requests.get(
        f"{API_URL}/recordings",
        headers=_headers(token),
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_recording(token: str, recording_id: int) -> dict:
    response = requests.get(
        f"{API_URL}/recordings/{int(recording_id)}",
        headers=_headers(token),
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_recording_predictions(token: str, recording_id: int) -> list[dict]:
    response = requests.get(
        f"{API_URL}/recordings/{int(recording_id)}/predictions",
        headers=_headers(token),
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_assigned_patients(token: str) -> list[dict]:
    response = requests.get(
        f"{API_URL}/doctor/patients",
        headers=_headers(token),
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def share_recording(token: str, recording_id: int) -> dict:
    response = requests.post(f"{API_URL}/recordings/{int(recording_id)}/share", headers=_headers(token), timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()

def revoke_recording_share(token: str, recording_id: int) -> dict:
    response = requests.delete(f"{API_URL}/recordings/{int(recording_id)}/share", headers=_headers(token), timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json()

def get_recording_share_status(token: str, recording_id: int) -> dict:
    response = requests.get(f"{API_URL}/recordings/{int(recording_id)}/share", headers=_headers(token), timeout=TIMEOUT_SECONDS)
    if response.status_code == 404:
        return {"is_active": False}
    response.raise_for_status()
    return response.json()


def upload_recording_waveform(token: str, recording_id: int, signal, sampling_rate: int, lead_name: str = "Lead 0") -> dict:
    """Upload a compressed float32 waveform array for a saved patient recording."""
    import io
    import numpy as np
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        signal=np.asarray(signal, dtype=np.float32).reshape(-1),
        sampling_rate=np.array([int(sampling_rate)], dtype=np.int32),
    )
    buffer.seek(0)
    response = requests.put(
        f"{API_URL}/recordings/{int(recording_id)}/waveform",
        headers=_headers(token),
        files={"waveform_file": (f"recording_{recording_id}.npz", buffer.getvalue(), "application/octet-stream")},
        data={"lead_name": lead_name},
        timeout=TIMEOUT_SECONDS * 4,
    )
    response.raise_for_status()
    return response.json()


def get_recording_waveform(token: str, recording_id: int) -> bytes:
    response = requests.get(
        f"{API_URL}/recordings/{int(recording_id)}/waveform",
        headers=_headers(token),
        timeout=TIMEOUT_SECONDS * 4,
    )
    response.raise_for_status()
    return response.content


def get_recording_notes(token: str, recording_id: int) -> list[dict]:
    response = requests.get(
        f"{API_URL}/recordings/{int(recording_id)}/notes",
        headers=_headers(token),
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def create_recording_note(token: str, recording_id: int, note: str, patient_visible: bool = True) -> dict:
    response = requests.post(
        f"{API_URL}/recordings/{int(recording_id)}/notes",
        headers=_headers(token),
        json={"note": note, "patient_visible": bool(patient_visible)},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def save_ai_analysis(token: str, recording_id: int, report_text: str) -> dict:
    response = requests.post(
        f"{API_URL}/recordings/{int(recording_id)}/ai-analyses",
        headers=_headers(token),
        json={"report_text": report_text},
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_ai_analyses(token: str, recording_id: int) -> list[dict]:
    response = requests.get(
        f"{API_URL}/recordings/{int(recording_id)}/ai-analyses",
        headers=_headers(token),
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def get_chat_messages(token: str, patient_id: int, recording_id: Optional[int] = None) -> list[dict]:
    params = {"recording_id": int(recording_id)} if recording_id is not None else None
    response = requests.get(
        f"{API_URL}/chat/{int(patient_id)}/messages", headers=_headers(token),
        params=params, timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()


def send_chat_message(token: str, patient_id: int, message_text: str, recording_id: Optional[int] = None) -> dict:
    payload = {"patient_id": int(patient_id), "message_text": message_text}
    if recording_id is not None:
        payload["recording_id"] = int(recording_id)
    response = requests.post(
        f"{API_URL}/chat/messages", json=payload, headers=_headers(token),
        timeout=TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.json()
