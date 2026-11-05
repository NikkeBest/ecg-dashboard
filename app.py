"""Streamlit dashboard for beat-level ECG classification with a Keras model."""
from __future__ import annotations

import math
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
from api_client import register_user, login_user
import requests


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


ANNOTATION_TO_CLASS = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
CLASS_TO_ID = {class_name: class_id for class_id, class_name in CLASS_NAMES.items()}


@st.cache_resource(show_spinner="Loading the ECG classification model…")
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)


def normalize_beat(beat: np.ndarray) -> np.ndarray:
    beat = np.asarray(beat, dtype=np.float32)
    return (beat - beat.mean()) / (beat.std() + 1e-8)



def compare_peak_positions(
    detected: np.ndarray,
    reference: np.ndarray,
    tolerance_samples: int,
) -> tuple[int, int, int]:
    """One-to-one evaluation match; reference annotations never feed inference."""
    detected = np.sort(np.asarray(detected, dtype=int))
    reference = np.sort(np.asarray(reference, dtype=int))
    i = j = matched = 0
    while i < len(detected) and j < len(reference):
        delta = int(detected[i]) - int(reference[j])
        if abs(delta) <= tolerance_samples:
            matched += 1
            i += 1
            j += 1
        elif delta < -tolerance_samples:
            i += 1
        else:
            j += 1
    return matched, len(detected) - matched, len(reference) - matched

def _read_header_frequency(header_path: Path) -> float:
    """Read WFDB's base sampling frequency, including optional counter suffixes."""
    try:
        first_line = header_path.read_text(encoding="ascii", errors="replace").splitlines()[0]
        fields = first_line.split()
        if len(fields) < 3:
            raise ValueError("The WFDB header's first line is incomplete.")
        # WFDB may write e.g. '360/1000' or '360 0 1000'; use only the rate.
        fs = float(fields[2].split("/")[0])
    except (OSError, IndexError, ValueError) as exc:
        raise ValueError(f"Could not read a valid sampling frequency from {header_path.name}.") from exc
    if not math.isfinite(fs) or fs <= 0:
        raise ValueError("The WFDB header contains an invalid sampling frequency.")
    return fs


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

def _matching_sidecar(header_path: Path, suffix: str) -> Path | None:
    """Find a sidecar by record stem without assuming extension case."""
    target = f"{header_path.stem}{suffix}".lower()
    for candidate in header_path.parent.iterdir():
        if candidate.is_file() and candidate.name.lower() == target:
            return candidate
    return None


def _validate_record_files(header_path: Path, require_dat: bool = False) -> None:
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
    if require_dat and not any(Path(name).suffix.lower() == ".dat" for name, _ in references):
        raise ValueError(
            "Reference Annotation Mode requires the matching .dat signal file named by the .hea header."
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


def prepare_reference_record(header_path: Path):
    """Prepare labeled annotation-centered beats; never invoke a peak detector."""
    _validate_record_files(header_path, require_dat=True)
    atr_path = _matching_sidecar(header_path, ".atr")
    if atr_path is None:
        raise ValueError(
            f"Reference Annotation Mode requires {header_path.stem}.atr alongside the selected record."
        )
    record, signal, fs, lead_name = _read_record(header_path)
    if not math.isclose(fs, TRAINING_FS, rel_tol=0, abs_tol=1e-6):
        raise ValueError(
            f"Training used 360 Hz records without resampling; this record is {fs:g} Hz. "
            "Reference-mode equivalence cannot be claimed for a different sampling rate."
        )
    try:
        annotation = wfdb.rdann(str(header_path.with_suffix("")), "atr")
    except Exception as exc:
        raise ValueError(f"Could not read matching annotation file {atr_path.name}: {exc}") from exc

    all_symbols = list(annotation.symbol)
    supported = [
        (int(sample), symbol, CLASS_TO_ID[ANNOTATION_TO_CLASS[symbol]])
        for sample, symbol in zip(annotation.sample, all_symbols)
        if symbol in ANNOTATION_TO_CLASS
    ]
    centers = np.asarray([item[0] for item in supported], dtype=int)
    y_all = np.asarray([item[2] for item in supported], dtype=np.int32)
    symbols_all = np.asarray([item[1] for item in supported], dtype=object)
    X, valid_centers, valid_indices, excluded = extract_beat_windows(signal, centers)
    return {
        "record": record, "signal": signal, "fs": fs, "lead_name": lead_name,
        "annotation": annotation, "atr_path": atr_path, "X": X,
        "centers": valid_centers, "y_true": y_all[valid_indices],
        "symbols": symbols_all[valid_indices], "total_annotations": len(all_symbols),
        "supported_annotations": len(supported), "ignored_annotations": len(all_symbols) - len(supported),
        "excluded": excluded,
    }


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


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    matrix = np.zeros((5, 5), dtype=np.int64)
    for truth, prediction in zip(y_true, y_pred):
        matrix[int(truth), int(prediction)] += 1
    rows = []
    for index, name in CLASS_NAMES.items():
        tp = int(matrix[index, index])
        predicted_count = int(matrix[:, index].sum())
        support = int(matrix[index, :].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append({"Class": name, "Precision": precision, "Recall": recall, "F1": f1, "Support": support})
    accuracy = float(np.mean(y_true == y_pred)) if len(y_true) else float("nan")
    report = pd.DataFrame(rows)
    report.loc[len(report)] = {
        "Class": "Macro avg", "Precision": report["Precision"].mean(),
        "Recall": report["Recall"].mean(), "F1": report["F1"].mean(), "Support": len(y_true),
    }
    if len(y_true):
        weights = report.iloc[:5]["Support"].to_numpy(dtype=float) / len(y_true)
        report.loc[len(report)] = {
            "Class": "Weighted avg",
            "Precision": float(np.sum(report.iloc[:5]["Precision"].to_numpy() * weights)),
            "Recall": float(np.sum(report.iloc[:5]["Recall"].to_numpy() * weights)),
            "F1": float(np.sum(report.iloc[:5]["F1"].to_numpy() * weights)),
            "Support": len(y_true),
        }
    return matrix, report, accuracy


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
                    st.success("Account created. Please sign in using the Login tab.")
                except requests.RequestException as exc:
                    detail = getattr(getattr(exc, "response", None), "text", "") or str(exc)
                    st.error(f"Registration failed: {detail}")


def _patient_dashboard(user_name: str) -> None:
    """Patient-facing dashboard. Live device and history data are not wired yet."""
    st.title("💓 My ECG Dashboard")
    st.write(f"Welcome, {user_name}.")
    st.caption("View your monitoring status, ECG history, and reviewed results.")
    st.warning(
        "This is a research prototype, not a medical device or diagnostic service. "
        "No ECG device is currently connected to this dashboard. Do not use it for emergencies."
    )

    overview, live, history, results, doctor = st.tabs(
        ["Overview", "Live Monitoring", "ECG History", "My Results", "My Doctor"]
    )
    with overview:
        a, b, c = st.columns(3)
        a.metric("Device status", "Not connected")
        b.metric("Latest heart rate", "-- BPM")
        c.metric("Latest recording", "No recordings")
        st.subheader("Monitoring")
        st.info(
            "Your ECG device has not been connected yet. Once device data ingestion is implemented, "
            "your monitoring status and latest ECG information will appear here."
        )
        st.subheader("Getting started")
        st.markdown(
            "1. Have your ECG monitoring device registered by your care team.\n"
            "2. Connect the device through the supported patient gateway.\n"
            "3. Keep this page open to view monitoring updates when live streaming is enabled."
        )
    with live:
        st.subheader("Live ECG")
        st.info("Waiting for a registered ECG device. Live waveform streaming is not connected yet.")
        st.caption("No live signal is being received. The display will become active after hardware ingestion is implemented.")
    with history:
        st.subheader("My ECG recordings")
        st.info("You do not have any saved ECG recordings yet.")
    with results:
        st.subheader("My ECG results")
        st.info("There are no reviewed results available yet. AI heartbeat classifications will appear here after recordings are processed and saved.")
        st.caption("Heartbeat classifications are research outputs, not diagnoses. A normal result does not rule out a health problem.")
    with doctor:
        st.subheader("My care team")
        st.info("No doctor has been linked to your account yet. Patient-doctor assignment is not implemented in this version.")


def _doctor_dashboard(user_name: str) -> None:
    """Doctor-facing dashboard with honest empty states until clinical workflows exist."""
    st.title("🩺 Doctor Dashboard")
    st.write(f"Welcome, {user_name}.")
    st.caption("Review assigned patients and their ECG monitoring information.")
    st.warning(
        "This is a research prototype, not a clinically validated diagnostic system. "
        "Do not use it to make treatment decisions or manage emergencies."
    )
    patients, monitoring, reviews, notes = st.tabs(
        ["My Patients", "Live Monitoring", "ECG Reviews", "Clinical Notes"]
    )
    with patients:
        st.subheader("Assigned patients")
        st.info("No patients are assigned to your account yet. Patient assignment and listing are not connected to the interface yet.")
    with monitoring:
        st.subheader("Patient monitoring")
        st.info("Live patient monitoring will be available after device ingestion and patient assignment are implemented.")
    with reviews:
        st.subheader("ECG recordings awaiting review")
        st.info("There are no ECG recordings awaiting review.")
    with notes:
        st.subheader("Clinical notes")
        st.info("Clinical notes will be available after patient-record access and note-saving workflows are connected.")


def main() -> None:
    st.set_page_config(page_title="ECG Heartbeat Monitoring", page_icon="💓", layout="wide")

    user = st.session_state.get("auth_user")
    if not user:
        _auth_screen()
        return

    user_name = user.get("full_name") or user.get("email") or "User"
    user_role = str(user.get("role", "patient")).lower()
    with st.sidebar:
        st.title("ECG Monitoring")
        st.write(f"**{user_name}**")
        st.caption(f"Account: {user_role.title()}")
        st.divider()
        if st.button("Log out", key="auth_logout", use_container_width=True):
            st.session_state.pop("auth_user", None)
            st.session_state.pop("auth_token", None)
            st.rerun()
        st.caption("ECG Heartbeat Monitoring · Research prototype")

    if user_role == "doctor":
        _doctor_dashboard(user_name)
    else:
        _patient_dashboard(user_name)


if __name__ == "__main__":
    main()
