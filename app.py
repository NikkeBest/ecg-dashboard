"""Streamlit dashboard for beat-level ECG classification with a Keras model."""
from __future__ import annotations

import math
from datetime import datetime
import shutil
import tempfile
import zipfile
from fractions import Fraction
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import wfdb
from wfdb.processing import xqrs_detect
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt
import tensorflow as tf

from groq_analysis import ask_groq, groq_available
from api_client import (
    register_user, login_user, save_recording, get_recordings,
    get_assigned_patients,
    get_recording_predictions, share_recording, revoke_recording_share, get_recording_share_status,
    upload_recording_waveform, get_recording_waveform, get_recording_notes, create_recording_note,
    save_ai_analysis, get_ai_analyses, get_chat_messages, send_chat_message,
)
import requests
import os


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "heartbeat_cnn.keras"
TRAINING_FS = 360
BEAT_PRE_SAMPLES = 90
BEAT_LENGTH = 187
CLASS_NAMES = {
    0: "N",
    1: "S",
    2: "V",
    3: "F",
    4: "Q",
}


def _saved_recording_ai_payload(recording: dict, predictions: list[dict]) -> dict:
    """Build a compact, role-neutral AI payload from one saved ECG recording."""
    if not isinstance(recording, dict):
        raise ValueError("The saved recording details are invalid.")
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("This recording has no saved beat predictions to explain.")

    counts = {name: 0 for name in CLASS_NAMES.values()}
    confidences = []
    beat_times = []

    for item in predictions:
        if not isinstance(item, dict):
            continue
        label = (
            item.get("heartbeat_class")
            or item.get("predicted_class")
            or item.get("class_name")
            or item.get("prediction")
        )
        if isinstance(label, str):
            label = label.strip().upper()
            if label in counts:
                counts[label] += 1

        confidence = item.get("confidence")
        try:
            confidence = float(confidence)
            if math.isfinite(confidence):
                confidences.append(confidence)
        except (TypeError, ValueError):
            pass

        beat_time = item.get("heartbeat_time", item.get("time_seconds"))
        try:
            beat_time = float(beat_time)
            if math.isfinite(beat_time):
                beat_times.append(beat_time)
        except (TypeError, ValueError):
            pass

    total = sum(counts.values())
    if total == 0:
        raise ValueError(
            "The saved predictions do not contain recognized heartbeat classes "
            "(expected N, S, V, F, or Q)."
        )

    rr_intervals = np.diff(np.asarray(sorted(beat_times), dtype=float))
    rr_intervals = rr_intervals[(rr_intervals > 0.25) & (rr_intervals < 2.5)]
    estimated_hr = (
        round(float(60.0 / np.median(rr_intervals)), 1)
        if len(rr_intervals) else None
    )

    return {
        "recording_id": recording.get("id"),
        "record_name": recording.get("file_name") or "Saved ECG recording",
        "sampling_rate_hz": recording.get("sampling_rate"),
        "lead_name": recording.get("lead_name"),
        "beats_classified": total,
        "predicted_class_counts": counts,
        "predicted_class_percentages": {
            name: round(100.0 * count / total, 2)
            for name, count in counts.items()
        },
        "mean_model_confidence": (
            round(float(np.mean(confidences)), 4) if confidences else None
        ),
        "low_confidence_beats_below_0_60": (
            sum(value < 0.60 for value in confidences) if confidences else None
        ),
        "estimated_heart_rate_bpm_from_saved_beat_times": estimated_hr,
        "limitations": [
            "These are CNN predictions, not clinical diagnoses.",
            "The AI summary is based on saved prediction metadata, not independent waveform interpretation.",
            "Model confidence is not a measure of clinical risk or diagnostic certainty.",
            "Prediction correctness cannot be established without independently verified reference labels.",
        ],
    }


def _generate_role_report(payload: dict, role: str) -> str:
    """Generate a patient-friendly or doctor-focused explanation with Groq."""
    role = str(role).strip().lower()
    if role == "patient":
        question = (
            "Write a clear, reassuring but not falsely reassuring explanation for a patient "
            "using plain everyday English. Use short sections: What the ECG model reported; "
            "What the numbers mean; What this result cannot tell us; What to discuss with "
            "your doctor. Explain the N, S, V, F and Q labels only as model classification "
            "categories, not as confirmed diagnoses. Do not infer a disease, prognosis, "
            "or treatment. Do not say the result is normal or safe merely because N is "
            "dominant or confidence is high. Encourage the patient to review the result "
            "with their doctor, especially if they have symptoms."
        )
    elif role == "doctor":
        question = (
            "Write a concise clinician-oriented decision-support report. Use sections: "
            "Summary of model outputs; Class distribution and confidence; Items to verify; "
            "Suggested review steps; Limitations. Identify the dominant predicted class "
            "and relevant non-N classes or low-confidence predictions. Recommend review "
            "of the corresponding ECG waveform segments, signal quality, peak alignment, "
            "symptoms, history and other clinical information. Do not diagnose, prescribe, "
            "or infer patient-specific risk from class labels or confidence alone. State "
            "that the AI received saved prediction metrics, not the raw ECG waveform."
        )
    else:
        raise ValueError("AI report role must be 'patient' or 'doctor'.")

    return ask_groq(payload, question)



@st.cache_resource(show_spinner="Loading the ECG classification model…")
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)


def normalize_beat(beat: np.ndarray) -> np.ndarray:
    beat = np.asarray(beat, dtype=np.float32)
    return (beat - beat.mean()) / (beat.std() + 1e-8)



def _save_uploads(uploaded_files, target_dir: Path) -> list[Path]:
    """Save uploads without relying on browser filenames being path-safe."""
    saved = []
    for uploaded in uploaded_files:
        safe_name = Path(uploaded.name.replace("\\", "/")).name
        if not safe_name or safe_name in {".", ".."}:
            continue
        destination = target_dir / safe_name
        destination.write_bytes(uploaded.getvalue())
        saved.append(destination)
    return saved


def _extract_zip(zip_path: Path, target_dir: Path) -> None:
    """Extract a WFDB bundle while rejecting paths that escape its temp folder."""
    base = target_dir.resolve()
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                relative = PurePosixPath(member.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("The ZIP contains an unsafe file path.")
                destination = (target_dir / Path(*relative.parts)).resolve()
                if base not in destination.parents and destination != base:
                    raise ValueError("The ZIP contains a file outside the upload folder.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
    except zipfile.BadZipFile as exc:
        raise ValueError("The uploaded ZIP file is damaged or invalid.") from exc


def _find_header_files(upload_paths: list[Path], root: Path) -> list[Path]:
    headers = [p for p in upload_paths if p.suffix.lower() == ".hea"]
    zip_paths = [p for p in upload_paths if p.suffix.lower() == ".zip"]
    for zip_path in zip_paths:
        _extract_zip(zip_path, root)
    # Find nested records too; de-duplicate case-insensitively.
    found = {str(p.resolve()).lower(): p for p in headers}
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".hea":
            found[str(path.resolve()).lower()] = path
    return sorted(found.values(), key=lambda p: p.name.lower())


def detect_r_peaks(signal: np.ndarray, fs: int) -> tuple[np.ndarray, str, str]:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    if len(signal) < max(20, int(fs * 2)):
        raise ValueError("The selected signal is too short for reliable beat detection.")
    try:
        peaks = np.asarray(xqrs_detect(sig=signal, fs=fs, verbose=False), dtype=int)
        if len(peaks) >= 2:
            return peaks, "WFDB XQRS", "XQRS completed successfully."
        xqrs_diagnostic = f"WFDB XQRS returned only {len(peaks)} peak(s); at least 2 are required."
    except Exception as exc:
        xqrs_diagnostic = f"WFDB XQRS failed: {type(exc).__name__}: {exc}"
    if fs <= 40:
        raise ValueError(f"{xqrs_diagnostic} Sampling frequency is too low for the fallback QRS detector.")
    high = min(20.0, fs * 0.45)
    sos = butter(2, [5.0, high], btype="bandpass", fs=fs, output="sos")
    try:
        filtered = sosfiltfilt(sos, signal)
    except ValueError as exc:
        raise ValueError(f"{xqrs_diagnostic} Signal is too short to filter for heartbeat detection.") from exc
    prominence = max(float(np.std(filtered)) * 0.35, 1e-6)
    # Absolute-valued ECG can make both lobes of one QRS complex look like peaks.
    # This is a best-effort fallback and its output count must be treated cautiously.
    peaks, _ = find_peaks(np.abs(filtered), distance=max(1, int(0.25 * fs)), prominence=prominence)
    if len(peaks) < 2:
        raise ValueError(
            f"{xqrs_diagnostic} SciPy fallback found only {len(peaks)} peak(s). "
            "Check the signal channel and sampling rate."
        )
    return peaks.astype(int), "SciPy fallback (approximate)", xqrs_diagnostic

def _validate_record_files(header_path: Path) -> None:
    """Give explicit errors for missing signal files referenced by the WFDB header."""
    try:
        lines = header_path.read_text(encoding="ascii", errors="replace").splitlines()
        fields = lines[0].split()
        signal_count = int(fields[1])
    except (OSError, IndexError, ValueError) as exc:
        raise ValueError(f"Cannot parse the WFDB header {header_path.name}.") from exc

    references = []
    for line in lines[1:1 + signal_count]:
        parts = line.split()
        if not parts or parts[0].startswith("#") or parts[0] == "~":
            continue
        relative = Path(parts[0].replace("\\", "/"))
        candidate = (header_path.parent / relative).resolve()
        if header_path.parent.resolve() not in candidate.parents and candidate != header_path.parent.resolve():
            raise ValueError("The WFDB header references a signal file outside its upload folder.")
        references.append((parts[0], candidate))

    if not references:
        raise ValueError(f"{header_path.name} does not name any readable WFDB signal file.")
    missing = [name for name, path in references if not path.is_file()]
    if missing:
        raise ValueError(
            "Missing signal file(s) referenced by the selected header: "
            + ", ".join(missing)
            + ". Upload the matching signal file(s) or a ZIP containing the complete record."
        )


def _read_record(header_path: Path):
    _validate_record_files(header_path)
    try:
        record = wfdb.rdrecord(str(header_path.with_suffix("")), physical=True)
    except Exception as exc:
        raise ValueError(
            f"WFDB could not read {header_path.name} and its signal file(s): "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    if record.p_signal is None or record.p_signal.ndim != 2 or record.p_signal.shape[1] == 0:
        raise ValueError("The record has no readable ECG signal data.")
    fs = float(record.fs)
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("The WFDB record reports an invalid sampling frequency.")
    lead_name = record.sig_name[0] if record.sig_name else "Lead 0"
    signal = np.asarray(record.p_signal[:, 0], dtype=np.float64)
    if not np.isfinite(signal).any():
        raise ValueError("The first ECG lead contains no finite samples.")
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return record, signal, fs, lead_name


def extract_beat_windows(signal: np.ndarray, centers: np.ndarray):
    """Shared training-matched windowing and per-beat normalization."""
    beats = []
    valid_centers = []
    valid_source_indices = []
    excluded = 0
    for source_index, center in enumerate(np.asarray(centers, dtype=int)):
        start = int(center) - BEAT_PRE_SAMPLES
        end = start + BEAT_LENGTH
        if start < 0 or end > len(signal):
            excluded += 1
            continue
        beat = np.asarray(signal[start:end], dtype=np.float32)
        beat = (beat - beat.mean()) / (beat.std() + 1e-8)
        beats.append(beat)
        valid_centers.append(int(center))
        valid_source_indices.append(source_index)
    if beats:
        batch = np.asarray(beats, dtype=np.float32)[..., np.newaxis]
    else:
        batch = np.empty((0, BEAT_LENGTH, 1), dtype=np.float32)
    return batch, np.asarray(valid_centers, dtype=int), np.asarray(valid_source_indices, dtype=int), excluded


def prepare_automatic_record(header_path: Path):
    """Prepare detector-centered beats for deployment mode only."""
    record, signal, original_fs, lead_name = _read_record(header_path)
    if not math.isclose(original_fs, TRAINING_FS, rel_tol=0, abs_tol=1e-6):
        ratio = Fraction(TRAINING_FS / original_fs).limit_denominator(10000)
        signal = resample_poly(signal, ratio.numerator, ratio.denominator)
    peaks, detector_name, diagnostic = detect_r_peaks(signal, TRAINING_FS)
    X, valid_peaks, _, excluded = extract_beat_windows(signal, peaks)
    if len(X) == 0:
        raise ValueError(f"{len(peaks)} peaks were detected, but none had a complete 187-sample window.")
    return {
        "record": record, "signal": signal, "original_fs": original_fs,
        "lead_name": lead_name, "detector": detector_name, "diagnostic": diagnostic,
        "all_peaks": peaks, "valid_peaks": valid_peaks, "X": X,
        "detected_count": len(peaks), "valid_count": len(valid_peaks), "excluded": excluded,
    }


def _get_user_from_login(result: dict) -> dict:
    """Accept the API's token response shape and extract the public user object."""
    user = result.get("user") if isinstance(result, dict) else None
    if isinstance(user, dict):
        return user
    if isinstance(result, dict) and ("email" in result or "role" in result):
        return result
    return {}


def _auth_screen() -> None:
    st.title("💓 ECG Heartbeat Monitoring")
    st.caption("Sign in to access the ECG research dashboard.")
    login_tab, register_tab = st.tabs(["Login", "Register"])

    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Email", key="auth_login_email")
            password = st.text_input("Password", type="password", key="auth_login_password")
            submitted = st.form_submit_button("Login", type="primary")
        if submitted:
            if not email.strip() or not password:
                st.error("Enter your email and password.")
            else:
                try:
                    result = login_user(email.strip(), password)
                    if not isinstance(result, dict):
                        raise ValueError("The login API returned an unexpected response.")
                    token = result.get("access_token") or result.get("token")
                    user = _get_user_from_login(result)
                    if not token:
                        st.error("Login response did not include an access token. Check the FastAPI response schema.")
                    elif not user:
                        st.error("Login succeeded, but the response did not include user details. Check the FastAPI TokenResponse schema.")
                    else:
                        st.session_state["auth_token"] = token
                        st.session_state["auth_user"] = user
                        st.rerun()
                except requests.RequestException as exc:
                    detail = getattr(getattr(exc, "response", None), "text", "") or str(exc)
                    st.error(f"Login failed: {detail}")
                except (ValueError, KeyError) as exc:
                    st.error(str(exc))

    with register_tab:
        with st.form("register_form"):
            full_name = st.text_input("Full name", key="auth_register_name")
            new_email = st.text_input("Email", key="auth_register_email")
            new_password = st.text_input("Password (at least 8 characters)", type="password", key="auth_register_password")
            role = st.selectbox("Register as", ["patient", "doctor"], key="auth_register_role")
            submitted = st.form_submit_button("Create account")
        if submitted:
            if not full_name.strip() or not new_email.strip() or not new_password:
                st.error("Complete all registration fields.")
            elif len(new_password) < 8:
                st.error("Password must contain at least 8 characters.")
            else:
                try:
                    register_user(full_name.strip(), new_email.strip(), new_password, role)
                    st.success("Registration submitted. Your account is pending administrator approval. You can log in after it is approved.")
                except requests.RequestException as exc:
                    detail = getattr(getattr(exc, "response", None), "text", "") or str(exc)
                    st.error(f"Registration failed: {detail}")



def _render_patient_doctor_chat(token: str, patient_id: int, recording_options: list[dict], key_prefix: str) -> None:
    """Render a persistent chat for the authenticated patient-doctor pair."""
    st.subheader("Messages with your doctor" if key_prefix == "patient_chat" else "Messages with patient")
    choices = {"General conversation": None}
    for item in recording_options:
        rid = int(item.get("id"))
        choices[f"ECG #{rid} — {item.get('file_name') or 'ECG recording'}"] = rid
    selected = st.selectbox("Conversation", list(choices), key=f"{key_prefix}_recording")
    recording_id = choices[selected]
    try:
        messages = get_chat_messages(token, patient_id, recording_id)
        if not messages:
            st.info("No messages in this conversation yet.")
        for msg in messages:
            mine = int(msg.get("sender_id", -1)) == int(st.session_state.get("auth_user", {}).get("id", -2))
            created = msg.get("created_at") or ""
            try:
                created = datetime.fromisoformat(str(created).replace("Z", "+00:00")).strftime("%d %b %Y, %I:%M %p")
            except (TypeError, ValueError):
                created = str(created)
            with st.chat_message("user" if mine else "assistant"):
                st.caption(created)
                st.write(msg.get("message_text", ""))
        with st.form(f"{key_prefix}_send_form", clear_on_submit=True):
            message = st.text_area("Write a message", max_chars=5000, key=f"{key_prefix}_message")
            submitted = st.form_submit_button("Send message")
        if submitted:
            if not message.strip():
                st.warning("Please enter a message before sending.")
            else:
                send_chat_message(token, patient_id, message.strip(), recording_id)
                st.rerun()
        if st.button("Refresh messages", key=f"{key_prefix}_refresh"):
            st.rerun()
    except requests.RequestException as exc:
        detail = getattr(getattr(exc, "response", None), "text", "") or str(exc)
        st.error(f"Could not load or send chat messages: {detail}")


def main() -> None:
    st.set_page_config(page_title="ECG Heartbeat Monitoring", page_icon="💓", layout="wide")

    # Apply the requested font only; keep the existing layout and styling unchanged.
    st.markdown(
        """
        <style>
        html, body, [class*="st-"], [class*="css"], button, input, textarea, select,
        h1, h2, h3, h4, h5, h6,
        [data-testid="stHeading"], [data-testid="stHeading"] *,
        [data-testid="stMarkdown"], [data-testid="stMarkdown"] *,
        [data-testid="stChatMessage"], [data-testid="stChatMessage"] *,
        [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] * {
            font-family: "Times New Roman", Times, serif !important;
        }
        /* Keep Streamlit Material icons using their icon font, not Times New Roman. */
        .material-symbols-rounded,
        .material-symbols-outlined,
        .material-icons,
        [data-testid="stIconMaterial"],
        [data-testid="stIconMaterial"] * {
            font-family: "Material Symbols Rounded" !important;
            font-weight: normal !important;
            font-style: normal !important;
            font-size: 24px;
            line-height: 1;
            letter-spacing: normal;
            text-transform: none;
            display: inline-block;
            white-space: nowrap;
            word-wrap: normal;
            direction: ltr;
            -webkit-font-feature-settings: "liga";
            -webkit-font-smoothing: antialiased;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    user = st.session_state.get("auth_user")
    if not user:
        _auth_screen()
        return

    user_name = user.get("full_name") or user.get("email") or "User"
    user_role = str(user.get("role", "unknown")).lower()
    with st.sidebar:
        st.title("Account")
        st.write(f"**{user_name}**")
        st.caption(f"Role: {user_role.title()}")
        if st.button("Log out", key="auth_logout"):
            st.session_state.pop("auth_user", None)
            st.session_state.pop("auth_token", None)
            st.rerun()
        st.divider()
        st.caption("ECG analysis")

    # Admin dashboard: account approvals and user directory.
    if user_role == "admin":
        st.title("Administrator Dashboard")
        st.caption("Review patient and doctor registrations and manage approved accounts.")
        api_url = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
        headers = {"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}

        st.subheader("Pending registrations")
        if st.button("Refresh pending registrations", key="admin_refresh_pending"):
            try:
                response = requests.get(
                    f"{api_url}/admin/pending-users", headers=headers, timeout=15
                )
                response.raise_for_status()
                st.session_state["admin_pending_users"] = response.json()
            except requests.RequestException as exc:
                detail = getattr(getattr(exc, "response", None), "text", "") or str(exc)
                st.error(f"Could not load pending registrations: {detail}")

        if "admin_pending_users" not in st.session_state:
            try:
                response = requests.get(
                    f"{api_url}/admin/pending-users", headers=headers, timeout=15
                )
                response.raise_for_status()
                st.session_state["admin_pending_users"] = response.json()
            except requests.RequestException as exc:
                detail = getattr(getattr(exc, "response", None), "text", "") or str(exc)
                st.error(f"Could not load pending registrations: {detail}")

        pending_users = st.session_state.get("admin_pending_users", [])
        if pending_users:
            for pending_user in pending_users:
                with st.container(border=True):
                    left, right = st.columns([3, 1])
                    with left:
                        st.markdown(f"**{pending_user.get('full_name')}**")
                        st.write(pending_user.get("email"))
                        st.caption(
                            f"Role: {str(pending_user.get('role', '')).title()} · "
                            f"Account ID: {pending_user.get('id')}"
                        )
                    with right:
                        if st.button(
                            "Approve",
                            key=f"admin_approve_{pending_user.get('id')}",
                            type="primary",
                        ):
                            try:
                                response = requests.post(
                                    f"{api_url}/admin/users/{pending_user.get('id')}/approve",
                                    headers=headers,
                                    timeout=15,
                                )
                                response.raise_for_status()
                                st.success("Account approved.")
                                st.session_state.pop("admin_pending_users", None)
                                st.rerun()
                            except requests.RequestException as exc:
                                detail = getattr(getattr(exc, "response", None), "text", "") or str(exc)
                                st.error(f"Approval failed: {detail}")
                        with st.popover("Reject"):
                            reason = st.text_area(
                                "Reason (optional)",
                                key=f"admin_reject_reason_{pending_user.get('id')}",
                            )
                            if st.button(
                                "Confirm rejection",
                                key=f"admin_reject_{pending_user.get('id')}",
                            ):
                                try:
                                    response = requests.post(
                                        f"{api_url}/admin/users/{pending_user.get('id')}/reject",
                                        headers=headers,
                                        json={"rejection_reason": reason or None},
                                        timeout=15,
                                    )
                                    response.raise_for_status()
                                    st.success("Registration rejected.")
                                    st.session_state.pop("admin_pending_users", None)
                                    st.rerun()
                                except requests.RequestException as exc:
                                    detail = getattr(getattr(exc, "response", None), "text", "") or str(exc)
                                    st.error(f"Rejection failed: {detail}")
        else:
            st.info("There are no pending registrations.")

        st.divider()
        st.subheader("Registered accounts")
        account_role = st.selectbox(
            "Account type",
            ["patient", "doctor", "admin"],
            format_func=lambda value: value.title(),
            key="admin_account_role",
        )
        if st.button("Load accounts", key="admin_load_accounts"):
            try:
                response = requests.get(
                    f"{api_url}/admin/users",
                    params={"role": account_role},
                    headers=headers,
                    timeout=15,
                )
                response.raise_for_status()
                st.session_state["admin_user_directory"] = response.json()
                st.session_state["admin_user_directory_role"] = account_role
            except requests.RequestException as exc:
                detail = getattr(getattr(exc, "response", None), "text", "") or str(exc)
                st.error(f"Could not load accounts: {detail}")

        directory = st.session_state.get("admin_user_directory")
        if directory is not None and st.session_state.get("admin_user_directory_role") == account_role:
            if directory:
                st.dataframe(
                    pd.DataFrame(directory)[
                        ["id", "full_name", "email", "role", "approval_status"]
                    ],
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info("No accounts found for this role.")

        st.divider()
        st.subheader("Patient–Doctor Assignment")
        st.caption("Each patient can have one assigned doctor. Reassigning replaces the previous doctor.")
        try:
            pr = requests.get(f"{api_url}/admin/users", params={"role": "patient"}, headers=headers, timeout=15)
            pr.raise_for_status()
            patients = [u for u in pr.json() if u.get("approval_status") == "approved"]
            dr = requests.get(f"{api_url}/admin/users", params={"role": "doctor"}, headers=headers, timeout=15)
            dr.raise_for_status()
            doctors = [u for u in dr.json() if u.get("approval_status") == "approved"]
            ar = requests.get(f"{api_url}/admin/assignments", headers=headers, timeout=15)
            ar.raise_for_status()
            assignments = ar.json()
            assigned = {a["patient_id"]: a for a in assignments}
            if not patients:
                st.info("No approved patients are available yet.")
            elif not doctors:
                st.warning("No approved doctors are available. Approve a doctor first.")
            else:
                patient_options = {f"{u['full_name']} — {u['email']} (ID {u['id']})": u for u in patients}
                doctor_options = {f"{u['full_name']} — {u['email']} (ID {u['id']})": u["id"] for u in doctors}
                patient_label = st.selectbox("Select patient", list(patient_options), key="admin_assignment_patient")
                patient = patient_options[patient_label]
                current = assigned.get(patient["id"])
                current_id = current.get("doctor_id") if current else None
                current_label = next((k for k,v in doctor_options.items() if v == current_id), "Not assigned")
                st.write(f"**Current doctor:** {current_label}")
                default_index = next((i for i,v in enumerate(doctor_options.values()) if v == current_id), 0)
                doctor_label = st.selectbox("Assign doctor", list(doctor_options), index=default_index, key="admin_assignment_doctor")
                if st.button("Save assignment", type="primary", key="admin_save_assignment"):
                    try:
                        rr = requests.put(f"{api_url}/admin/patients/{patient['id']}/assignment",
                            json={"doctor_id": doctor_options[doctor_label]}, headers=headers, timeout=15)
                        rr.raise_for_status()
                        st.success(rr.json().get("message", "Assignment saved."))
                        st.rerun()
                    except requests.RequestException as exc:
                        detail = getattr(getattr(exc, "response", None), "text", "") or str(exc)
                        st.error(f"Could not save assignment: {detail}")
            if assignments:
                st.markdown("**Current assignments**")
                st.dataframe(pd.DataFrame(assignments), width="stretch", hide_index=True)
            else:
                st.info("No patient–doctor assignments have been created yet.")
        except requests.RequestException as exc:
            detail = getattr(getattr(exc, "response", None), "text", "") or str(exc)
            st.error(f"Could not load assignment data: {detail}")
        return

    # Doctor dashboard: only patient-shared recordings are returned by the API.
    if user_role == "doctor":
        st.header("Doctor Dashboard")
        st.caption("Review ECG recordings that your assigned patients have explicitly shared.")
        st.divider()
        st.subheader("My Patients")
        try:
            patients = get_assigned_patients(st.session_state.get("auth_token", ""))
            if not patients:
                st.info("No patients are assigned to your account yet.")
            else:
                patient_options = {f"{p.get('full_name')} — {p.get('email')}": p for p in patients}
                selected_patient_label = st.selectbox(
                    "Select a patient", list(patient_options), key="doctor_selected_patient"
                )
                selected_patient = patient_options[selected_patient_label]
                token = st.session_state.get("auth_token", "")
                st.divider()
                doctor_recordings_for_chat = get_recordings(token)
                shared_for_chat = [r for r in doctor_recordings_for_chat if r.get("patient_id") == selected_patient.get("id")]
                _render_patient_doctor_chat(token, int(selected_patient["id"]), shared_for_chat, "doctor_chat")
                st.divider()
                doctor_recordings = get_recordings(token)
                patient_recordings = [
                    r for r in doctor_recordings if r.get("patient_id") == selected_patient.get("id")
                ]
                st.markdown(f"**Shared ECG recordings for {selected_patient.get('full_name')}**")
                if not patient_recordings:
                    st.info("This patient has not shared any ECG recordings with you.")
                else:
                    options = {
                        f"Recording #{r['id']} — {r.get('file_name') or 'ECG record'}": r
                        for r in patient_recordings
                    }
                    selected_label = st.selectbox(
                        "Select a shared recording", list(options), key="doctor_selected_recording"
                    )
                    selected_recording = options[selected_label]
                    recording_id = int(selected_recording["id"])
                    try:
                        detail = get_recording_predictions(token, recording_id)
                        st.subheader("CNN heartbeat predictions")
                        if detail:
                            pred_df = pd.DataFrame(detail)
                            st.dataframe(pred_df, width="stretch", hide_index=True)
                            if "heartbeat_class" in pred_df:
                                counts = pred_df["heartbeat_class"].value_counts().reindex(
                                    ["N", "S", "V", "F", "Q"], fill_value=0
                                )
                                st.plotly_chart(
                                    px.bar(
                                        pd.DataFrame({"Class": counts.index, "Beats": counts.values}),
                                        x="Class", y="Beats", color="Class",
                                    ),
                                    width="stretch",
                                )
                            if "confidence" in pred_df and pred_df["confidence"].notna().any():
                                st.metric("Mean model confidence", f"{pred_df['confidence'].mean():.1%}")
                        else:
                            st.info("No beat-level predictions were saved for this recording.")

                        st.subheader("ECG waveform")
                        if selected_recording.get("has_waveform"):
                            import io
                            waveform_bytes = get_recording_waveform(token, recording_id)
                            with np.load(io.BytesIO(waveform_bytes), allow_pickle=False) as archive:
                                waveform = np.asarray(archive["signal"], dtype=np.float32)
                                waveform_fs = int(np.asarray(archive["sampling_rate"]).reshape(-1)[0])
                            lead = selected_recording.get("lead_name") or "ECG lead"
                            max_points = min(len(waveform), waveform_fs * 30)
                            waveform_frame = pd.DataFrame({
                                "Time (s)": np.arange(max_points) / waveform_fs,
                                "Amplitude": waveform[:max_points],
                            })
                            st.caption(
                                f"{lead} · {waveform_fs} Hz · {len(waveform):,} samples · "
                                f"showing first {max_points / waveform_fs:.1f} seconds"
                            )
                            st.plotly_chart(
                                px.line(waveform_frame, x="Time (s)", y="Amplitude"),
                                width="stretch",
                            )
                        else:
                            st.warning(
                                "This older recording has predictions but no stored waveform. "
                                "The patient needs to upload and save it again using the updated app."
                            )

                        st.divider()
                        st.subheader("AI-assisted ECG review")
                        st.caption("Groq receives saved CNN summary metrics, not the raw waveform. This is decision support, not a diagnosis.")
                        if st.button("Generate doctor-focused AI report", key=f"doctor_ai_generate_{recording_id}"):
                            try:
                                if not groq_available():
                                    st.error("Groq is not configured. Add GROQ_API_KEY to Streamlit Secrets.")
                                else:
                                    payload = _saved_recording_ai_payload(selected_recording, detail)
                                    report_text = _generate_role_report(payload, "doctor")
                                    save_ai_analysis(token, recording_id, report_text)
                                    st.success("Doctor AI report saved to this ECG record.")
                                    st.rerun()
                            except Exception as exc:
                                st.error(f"Could not generate/save AI report: {exc}")
                        try:
                            saved_ai_reports = get_ai_analyses(token, recording_id)
                            if saved_ai_reports:
                                for ai_item in saved_ai_reports:
                                    with st.expander(f"AI report · {ai_item.get('created_at') or 'Date unavailable'}"):
                                        st.markdown(ai_item.get("report_text", ""))
                            else:
                                st.info("No saved doctor AI reports for this recording yet.")
                        except Exception as exc:
                            st.warning(f"Could not load AI report history: {exc}")

                        st.divider()
                        st.subheader("Clinical notes")
                        existing_notes = get_recording_notes(token, recording_id)
                        if existing_notes:
                            for note_item in existing_notes:
                                with st.container(border=True):
                                    st.caption(
                                        f"Created: {note_item.get('created_at') or 'Date unavailable'} · "
                                        + ("Visible to patient" if note_item.get("patient_visible") else "Doctor only")
                                    )
                                    st.write(note_item.get("note", ""))
                        else:
                            st.info("No clinical notes have been written for this recording.")
                        with st.form(f"doctor_note_form_{recording_id}"):
                            note_text = st.text_area("Write a clinical note", max_chars=10000)
                            patient_visible = st.checkbox(
                                "Make this note visible to the patient", value=True,
                                help="If unchecked, only the doctor can view this note."
                            )
                            note_submitted = st.form_submit_button("Save clinical note", type="primary")
                        if note_submitted:
                            if not note_text.strip():
                                st.error("Enter a note before saving.")
                            else:
                                create_recording_note(token, recording_id, note_text, patient_visible)
                                st.success("Clinical note saved.")
                                st.rerun()
                    except requests.RequestException as exc:
                        detail_text = getattr(getattr(exc, "response", None), "text", "") or str(exc)
                        st.error(f"Could not open this shared recording: {detail_text}")
                    except Exception as exc:
                        st.error(f"Could not display the recording: {exc}")
        except requests.RequestException as exc:
            detail_text = getattr(getattr(exc, "response", None), "text", "") or str(exc)
            st.error(f"Could not load assigned patients or shared recordings: {detail_text}")
        st.divider()

    if user_role == "patient":
        api_url = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
        try:
            ar = requests.get(f"{api_url}/patients/my-doctor",
                headers={"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}, timeout=15)
            ar.raise_for_status()
            info = ar.json()
            if info.get("assigned"):
                doctor = info["doctor"]
                st.info(f"Your assigned doctor: {doctor['full_name']} ({doctor['email']})")
            else:
                st.info("You do not have an assigned doctor yet. The administrator will assign one.")
            if info.get("assigned"):
                chat_recordings = get_recordings(st.session_state.get("auth_token", ""))
                active_shared = []
                for rec in chat_recordings:
                    try:
                        if get_recording_share_status(st.session_state.get("auth_token", ""), int(rec["id"])).get("is_active"):
                            active_shared.append(rec)
                    except requests.RequestException:
                        continue
                st.divider()
                _render_patient_doctor_chat(st.session_state.get("auth_token", ""), int(user["id"]), active_shared, "patient_chat")
        except requests.RequestException:
            st.warning("Your doctor assignment or chat could not be loaded right now.")

        st.divider()
        st.subheader("Your saved ECG recordings")
        patient_token = st.session_state.get("auth_token", "")
        try:
            patient_history = get_recordings(patient_token)
            if patient_history:
                history_frame = pd.DataFrame(patient_history)
                if "created_at" in history_frame.columns:
                    created = pd.to_datetime(history_frame["created_at"], errors="coerce", utc=True)
                    history_frame["created_at"] = created.dt.strftime("%d %b %Y, %I:%M %p UTC").fillna("")

                history_html = history_frame.to_html(index=False, escape=True, border=0, classes="ecg-history-table")
                st.markdown(
                    """
                    <style>
                    .ecg-history-table {
                        width: 100%;
                        border-collapse: collapse;
                        font-family: "Times New Roman", Times, serif !important;
                    }
                    .ecg-history-table th,
                    .ecg-history-table td {
                        font-family: "Times New Roman", Times, serif !important;
                        padding: 8px;
                        border: 1px solid rgba(128, 128, 128, 0.25);
                        text-align: left;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )
                st.markdown(history_html, unsafe_allow_html=True)
                st.subheader("Share an ECG recording with your assigned doctor")
                share_choices = {
                    f"Recording #{r['id']} — {r.get('file_name') or 'ECG record'}": r['id']
                    for r in patient_history
                }
                chosen_label = st.selectbox(
                    "Choose a saved recording",
                    list(share_choices),
                    key="patient_share_recording_choice",
                )
                chosen_id = share_choices[chosen_label]
                share_status = get_recording_share_status(patient_token, chosen_id)
                if share_status.get("is_active"):
                    st.success("This recording is currently shared with your assigned doctor.")
                    if st.button("Revoke sharing", key="revoke_ecg_share"):
                        try:
                            revoke_recording_share(patient_token, chosen_id)
                            st.success("Sharing revoked.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not revoke sharing: {exc}")
                else:
                    st.caption(
                        "Only your currently assigned doctor can access this recording after you share it."
                    )
                    if st.button("Share with my assigned doctor", key="share_ecg_recording"):
                        try:
                            share_recording(patient_token, chosen_id)
                            st.success("Recording shared with your assigned doctor.")
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not share recording: {exc}")
            else:
                st.info(
                    "You have no saved ECG recordings yet. The sharing section is available now; "
                    "save an ECG analysis to create a recording that you can share."
                )
                st.subheader("Share an ECG recording with your assigned doctor")
                st.caption("Your saved recordings will appear here when you have uploaded and saved an ECG analysis.")
        except requests.RequestException as exc:
            detail = getattr(getattr(exc, "response", None), "text", "") or str(exc)
            st.error(f"Could not load your saved ECG recordings: {detail}")
        except Exception as exc:
            st.error(f"Could not load sharing options: {exc}")

    if user_role == "patient":
        st.divider()
        st.subheader("AI explanation of a saved ECG")
        st.caption("A patient-friendly explanation of your saved CNN results. It is not a diagnosis and does not replace your doctor.")
        try:
            saved_for_ai = get_recordings(st.session_state.get("auth_token", ""))
            if saved_for_ai:
                ai_choices = {
                    f"Recording #{r['id']} — {r.get('file_name') or 'ECG record'}": r
                    for r in saved_for_ai
                }
                ai_label = st.selectbox("Choose an ECG record for AI explanation", list(ai_choices), key="patient_ai_record")
                ai_record = ai_choices[ai_label]
                ai_record_id = int(ai_record["id"])
                if st.button("Generate patient-friendly AI explanation", key=f"patient_ai_generate_{ai_record_id}"):
                    try:
                        if not groq_available():
                            st.error("Groq is not configured. Add GROQ_API_KEY to Streamlit Secrets.")
                        else:
                            ai_predictions = get_recording_predictions(st.session_state.get("auth_token", ""), ai_record_id)
                            ai_payload = _saved_recording_ai_payload(ai_record, ai_predictions)
                            ai_report = _generate_role_report(ai_payload, "patient")
                            save_ai_analysis(st.session_state.get("auth_token", ""), ai_record_id, ai_report)
                            st.success("Your AI explanation has been saved with this ECG record.")
                            st.rerun()
                    except Exception as exc:
                        st.error(f"Could not generate/save AI explanation: {exc}")
                try:
                    patient_ai_reports = get_ai_analyses(st.session_state.get("auth_token", ""), ai_record_id)
                    if patient_ai_reports:
                        for ai_item in patient_ai_reports:
                            with st.expander(f"AI explanation · {ai_item.get('created_at') or 'Date unavailable'}", expanded=True):
                                st.markdown(ai_item.get("report_text", ""))
                    else:
                        st.info("No AI explanations have been saved for this ECG record.")
                except Exception as exc:
                    st.warning(f"Could not load AI explanation history: {exc}")
            else:
                st.info("Save an ECG analysis first to generate a patient-friendly AI explanation.")
        except Exception as exc:
            st.warning(f"Could not load your saved ECG records: {exc}")

    if user_role == "patient":
        st.subheader("Clinical notes from your doctor")
        try:
            patient_saved = get_recordings(st.session_state.get("auth_token", ""))
            note_rows = []
            for item in patient_saved:
                try:
                    note_rows.extend(get_recording_notes(
                        st.session_state.get("auth_token", ""), int(item["id"])
                    ))
                except requests.RequestException:
                    continue
            if note_rows:
                for note_item in note_rows:
                    with st.container(border=True):
                        st.caption(f"Recording #{note_item.get('recording_id')} · {note_item.get('created_at') or ''}")
                        st.write(note_item.get("note", ""))
            else:
                st.info("No patient-visible clinical notes are available yet.")
        except requests.RequestException as exc:
            detail_text = getattr(getattr(exc, "response", None), "text", "") or str(exc)
            st.warning(f"Could not load clinical notes: {detail_text}")

    st.title("💓 ECG Heartbeat Classification")
    st.write(
        f"Welcome, {user_name}. "
        "Use the ECG analysis workspace below to review uploaded WFDB records."
    )
    st.warning(
        "Research/student prototype only — not clinically validated and not a medical diagnosis. "
        "Automatic peak positions can differ from reference annotations and affect predictions."
    )

    if not MODEL_PATH.is_file():
        st.error(f"Model file not found: {MODEL_PATH.name}. Put it beside app.py in the GitHub repository.")
        st.stop()
    try:
        model = load_model(str(MODEL_PATH))
    except Exception as exc:
        st.error(f"Could not load {MODEL_PATH.name}: {exc}")
        st.stop()

    output_shape = model.output_shape
    if isinstance(output_shape, list):
        if len(output_shape) != 1:
            st.error("The model must have one output with five classes.")
            st.stop()
        output_shape = output_shape[0]
    if not output_shape or output_shape[-1] != 5:
        st.error(f"Expected a five-class model output; model output shape is {output_shape}.")
        st.stop()

    st.info(
        "Upload a WFDB .hea file with its referenced ECG signal file (normally .dat). "
        "The app detects beats automatically using WFDB XQRS, with an approximate SciPy fallback "
        "if XQRS fails or returns fewer than two peaks."
    )

    uploaded_files = st.file_uploader(
        "WFDB files (.hea and matching signal file), or a ZIP bundle",
        type=["hea", "dat", "mat", "xyz", "zip"],
        accept_multiple_files=True,
    )
    if not uploaded_files:
        st.info("Upload the files required for the selected mode to begin.")
        return

    if "ecg_upload_dir" not in st.session_state:
        st.session_state.ecg_upload_dir = tempfile.mkdtemp(prefix="ecg_upload_")
    root = Path(st.session_state.ecg_upload_dir)
    root.mkdir(parents=True, exist_ok=True)
    try:
        for child in root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

        saved = _save_uploads(uploaded_files, root)
        headers = _find_header_files(saved, root)
        if not headers:
            raise ValueError("No .hea file was found. Upload it with its signal file or use a ZIP bundle.")

        header_options = [path.relative_to(root).as_posix() for path in headers]
        selected_name = st.selectbox("Choose ECG record", header_options)
        selected_header = root.joinpath(*Path(selected_name).parts)
        if st.button("Analyze ECG", type="primary"):
            try:
                data = prepare_automatic_record(selected_header)
            except ValueError as exc:
                st.error(str(exc))
                return
            with st.spinner("Predicting automatically detected beats…"):
                probabilities = np.asarray(model.predict(data["X"], verbose=0))
            if probabilities.ndim != 2 or probabilities.shape != (len(data["X"]), 5):
                raise ValueError(f"Expected prediction shape ({len(data['X'])}, 5); got {probabilities.shape}.")
            if not np.isfinite(probabilities).all():
                raise ValueError("The CNN returned NaN or infinite prediction values.")
            predictions = probabilities.argmax(axis=1)
            st.subheader("Automatic detection results")
            if data["detector"] != "WFDB XQRS":
                st.warning(
                    f"{data['diagnostic']} The app used the documented approximate SciPy fallback "
                    f"(5–{min(20.0, TRAINING_FS * 0.45):g} Hz band-pass, absolute-signal local maxima, "
                    "250 ms minimum spacing). Secondary waveform peaks can be counted; review detections."
                )
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Detector", data["detector"])
            c2.metric("Detected peaks", f"{data['detected_count']:,}")
            c3.metric("Valid 187-sample windows", f"{data['valid_count']:,}")
            c4.metric("Excluded: incomplete window", f"{data['excluded']:,}")
            class_counts = pd.Series(
                [CLASS_NAMES[int(i)] for i in predictions]
            ).value_counts().reindex(list(CLASS_NAMES.values()), fill_value=0)
            st.subheader("Predicted class distribution")
            st.plotly_chart(
                px.bar(
                    pd.DataFrame({"Class": class_counts.index, "Beats": class_counts.values}),
                    x="Class", y="Beats", color="Class",
                ),
                width="stretch",
            )
            results = pd.DataFrame({
                "Beat": np.arange(1, len(predictions) + 1),
                "Time (s)": data["valid_peaks"] / TRAINING_FS,
                "Class": [CLASS_NAMES[int(i)] for i in predictions],
                "Confidence": probabilities.max(axis=1),
            })
            signal = data["signal"]
            lead_name = data["lead_name"]
            sampling_rate = TRAINING_FS
            csv_stem = selected_header.stem

            st.subheader(f"ECG waveform — {lead_name}")
            plot_limit = min(len(signal), int(sampling_rate * 30))
            signal_frame = pd.DataFrame({
                "Time (s)": np.arange(plot_limit) / sampling_rate,
                "Amplitude": signal[:plot_limit],
            })
            st.plotly_chart(px.line(signal_frame, x="Time (s)", y="Amplitude"), width="stretch")
            st.subheader("Per-beat results")
            st.dataframe(results, width="stretch", hide_index=True)
            st.download_button(
                "Download predictions (CSV)",
                results.to_csv(index=False).encode("utf-8"),
                file_name=f"{csv_stem}_predictions.csv",
                mime="text/csv",
            )

            # Keep the completed analysis across Streamlit reruns so the user
            # can explicitly save it to the backend afterward.
            time_values = results["Time (s)"].to_numpy(dtype=float)
            st.session_state["last_ecg_analysis"] = {
                "file_name": csv_stem,
                "sampling_rate": int(round(float(sampling_rate))),
                "predictions": [
                    {
                        "heartbeat_class": CLASS_NAMES[int(class_id)],
                        "confidence": float(probabilities[index].max()),
                        "heartbeat_time": float(time_values[index]),
                    }
                    for index, class_id in enumerate(predictions)
                ],
                "mode": "automatic_detection",
                "waveform": np.asarray(signal, dtype=np.float32).copy(),
                "lead_name": str(lead_name),
            }
            counts = {name: int((predictions == idx).sum()) for idx, name in CLASS_NAMES.items()}
            confidence = np.asarray(probabilities).max(axis=1)
            rr_seconds = np.diff(np.asarray(results["Time (s)"], dtype=float))
            rr_seconds = rr_seconds[(rr_seconds > 0.25) & (rr_seconds < 2.5)]
            estimated_hr = float(60.0 / np.median(rr_seconds)) if len(rr_seconds) else None
            llm_payload = {
                "record_name": csv_stem,
                "mode": "automatic_detection",
                "sampling_rate_hz": float(sampling_rate),
                "lead_name": str(lead_name),
                "beats_classified": int(len(predictions)),
                "predicted_class_counts": counts,
                "predicted_class_percentages": {name: round(100 * count / max(1, len(predictions)), 2) for name, count in counts.items()},
                "mean_model_confidence": round(float(confidence.mean()), 4) if len(confidence) else None,
                "low_confidence_beats_below_0_60": int((confidence < 0.60).sum()),
                "estimated_heart_rate_bpm_from_detected_peak_intervals": round(estimated_hr, 1) if estimated_hr else None,
                "detector": data.get("detector"),
                "detector_diagnostic": data.get("diagnostic", None),
                "limitations": [
                    "CNN labels are model predictions, not clinical diagnoses.",
                    "Heart rate may be unreliable if peaks are missed or extra peaks are detected.",
                    "A single record and overall accuracy do not establish clinical validity.",
                ],
            }
            st.session_state["ecg_llm_payload"] = llm_payload
            st.session_state["ecg_llm_chat"] = []
            st.session_state.pop("ecg_llm_report", None)

        if st.session_state.get("ecg_llm_payload"):
            st.divider()
            st.header("Groq ECG Analysis (Research Prototype)")
            st.caption("Healthcare support summary: dominant model finding, possible concerns to verify, and practical next steps for doctors or nurses. Groq receives summary metrics and model outputs, not the raw ECG waveform or full beat table.")
            if not groq_available():
                st.info("To enable Groq, add GROQ_API_KEY to Streamlit Secrets. See GROQ_SETUP.md for setup.")
            payload = st.session_state["ecg_llm_payload"]
            if st.button("Generate English analysis report", type="primary", key="groq_report"):
                with st.spinner("Generating report with Groq…"):
                    try:
                        st.session_state["ecg_llm_report"] = ask_groq(
                            payload,
                            "Write an English ECG results interpretation and care-support summary for a doctor or nurse. Use sections: 1) Main Finding (state which predicted class is most common, with count and percentage), 2) Other Notable Model Outputs, 3) Potential Clinical Concerns to Verify (explain cautiously what the labels may warrant checking, without presenting them as confirmed abnormalities), 4) Practical Suggestions for the Healthcare User (review relevant waveform segments, correlate with symptoms/history and existing clinical observations, check signal/peak quality, and escalate according to established clinical protocols if the patient has concerning symptoms), and 5) Limitations. Base every statement on supplied metrics. Explain that model predictions alone cannot establish a patient-specific threat or diagnosis; the raw ECG waveform and patient context were not supplied to the LLM. If non-N classes or low-confidence beats are present, recommend checking those segments. Prediction correctness cannot be established without independently verified reference labels. Do not prescribe treatment or invent clinical findings. Do not provide a false sense of reassurance from a high mean confidence score.",
                        )
                    except Exception as exc:
                        st.error(f"Groq request failed: {exc}")
            if st.session_state.get("ecg_llm_report"):
                st.markdown(st.session_state["ecg_llm_report"])
                st.download_button("Download Groq report (.md)", st.session_state["ecg_llm_report"], file_name="ecg_groq_analysis_report.md", mime="text/markdown")
    except Exception as exc:
        st.error(str(exc))

    # Persisted save action: this remains available on the next Streamlit rerun.
    analysis_to_save = st.session_state.get("last_ecg_analysis")
    if analysis_to_save and user_role == "patient":
        st.divider()
        st.subheader("Save ECG analysis to patient history")
        st.caption(
            "This saves recording metadata and beat-level CNN predictions to PostgreSQL. "
            "The original raw ECG waveform file is not uploaded by this operation."
        )
        if st.button("Save this analysis to my history", type="primary", key="save_ecg_analysis"):
            try:
                with st.spinner("Saving ECG results to the database…"):
                    saved = save_recording(
                        token=st.session_state.get("auth_token", ""),
                        file_name=analysis_to_save["file_name"],
                        sampling_rate=analysis_to_save["sampling_rate"],
                        predictions=analysis_to_save["predictions"],
                    )
                    recording_id = int(saved.get("id"))
                    waveform_result = upload_recording_waveform(
                        token=st.session_state.get("auth_token", ""),
                        recording_id=recording_id,
                        signal=analysis_to_save["waveform"],
                        sampling_rate=analysis_to_save["sampling_rate"],
                        lead_name=analysis_to_save.get("lead_name", "Lead 0"),
                    )
                st.session_state["last_saved_recording_id"] = recording_id
                st.success(
                    f"ECG analysis and waveform saved successfully. Recording ID: {recording_id}"
                )
            except requests.RequestException as exc:
                detail = getattr(getattr(exc, "response", None), "text", "") or str(exc)
                st.error(f"Could not save ECG analysis: {detail}")
            except Exception as exc:
                st.error(f"Could not save ECG analysis: {exc}")

    # Recording history is available to patients and doctors, subject to backend access rules.
    st.divider()
    st.subheader("Saved ECG recording history")
    if st.button("Refresh recording history", key="refresh_recording_history"):
        try:
            history = get_recordings(st.session_state.get("auth_token", ""))
            st.session_state["recording_history"] = history
        except requests.RequestException as exc:
            detail = getattr(getattr(exc, "response", None), "text", "") or str(exc)
            st.error(f"Could not retrieve recording history: {detail}")

    if "recording_history" in st.session_state:
        history = st.session_state["recording_history"]
        if history:
            history_frame = pd.DataFrame(history)
            if "created_at" in history_frame.columns:
                created = pd.to_datetime(history_frame["created_at"], errors="coerce", utc=True)
                history_frame["created_at"] = created.dt.strftime("%d %b %Y, %I:%M %p UTC").fillna("")
            history_html = history_frame.to_html(index=False, escape=True, border=0, classes="ecg-history-table")
            st.markdown(
                """
                <style>
                .ecg-history-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-family: "Times New Roman", Times, serif !important;
                }
                .ecg-history-table th,
                .ecg-history-table td {
                    font-family: "Times New Roman", Times, serif !important;
                    padding: 8px;
                    border: 1px solid rgba(128, 128, 128, 0.25);
                    text-align: left;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )
            st.markdown(history_html, unsafe_allow_html=True)
        else:
            st.info("No saved ECG recordings were found for this account.")



if __name__ == "__main__":
    main()
