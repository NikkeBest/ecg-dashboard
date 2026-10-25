TRAINING_FS = 360  # MIT-BIH training data sampling frequency
BEAT_PRE_SAMPLES = 90
BEAT_LENGTH = 187

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf

from pathlib import Path
from tempfile import TemporaryDirectory
from fractions import Fraction
import math

import wfdb
from scipy.signal import butter, sosfiltfilt, find_peaks, resample_poly

st.set_page_config(
    page_title="ECG Monitor",
    page_icon="💓",
    layout="wide"
)

@st.cache_resource
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)


def normalize_beat(beat: np.ndarray) -> np.ndarray:
    beat = beat.astype(np.float32)
    return (beat - beat.mean()) / (beat.std() + 1e-8)


def detect_r_peaks(signal: np.ndarray, fs: int) -> tuple[np.ndarray, str]:
    """Detect approximate R-peaks. WFDB XQRS is preferred; scipy is a fallback."""
    signal = np.asarray(signal, dtype=np.float64)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    try:
        peaks = wfdb.processing.xqrs_detect(sig=signal, fs=fs, verbose=False)
        peaks = np.asarray(peaks, dtype=int)
        if len(peaks) >= 2:
            return peaks, "WFDB XQRS"
    except Exception:
        pass

    # Fallback detector. This is an approximate detector and should be validated
    # for the recording/device used.
    if fs <= 40:
        raise ValueError("Sampling frequency is too low for the fallback QRS detector.")
    sos = butter(2, [5, min(20, fs * 0.45)], btype="bandpass", fs=fs, output="sos")
    filtered = sosfiltfilt(sos, signal)
    prominence = max(float(np.std(filtered)) * 0.35, 1e-6)
    peaks, _ = find_peaks(
        np.abs(filtered),
        distance=max(1, int(0.25 * fs)),
        prominence=prominence,
    )
    if len(peaks) < 2:
        raise ValueError(
            "Could not detect enough heartbeats. Check the signal channel and sampling rate."
        )
    return peaks.astype(int), "SciPy fallback (approximate)"


def prepare_record(record_path: Path):
    """Load a WFDB record, resample to the training rate, detect beats and prepare model input."""
    record = wfdb.rdrecord(str(record_path))
    if record.p_signal is None or record.p_signal.size == 0:
        raise ValueError("The record has no readable signal data.")

    signal = np.asarray(record.p_signal[:, 0], dtype=np.float64)
    original_fs = float(record.fs)
    if not np.isfinite(original_fs) or original_fs <= 0:
        raise ValueError("The header contains an invalid sampling frequency.")

    # The model was trained on MIT-BIH signals at 360 Hz. Resample if needed.
    if not math.isclose(original_fs, TRAINING_FS, rel_tol=0, abs_tol=1e-6):
        ratio = Fraction(TRAINING_FS / original_fs).limit_denominator(10000)
        signal_360 = resample_poly(signal, ratio.numerator, ratio.denominator)
    else:
        signal_360 = signal

    peaks, detector_name = detect_r_peaks(signal_360, TRAINING_FS)

    beats = []
    valid_peaks = []
    for peak in peaks:
        start = int(peak) - BEAT_PRE_SAMPLES
        end = start + BEAT_LENGTH
        if start < 0 or end > len(signal_360):
            continue
        beats.append(normalize_beat(signal_360[start:end]))
        valid_peaks.append(int(peak))

    if not beats:
        raise ValueError("No complete 187-sample heartbeat windows could be extracted.")

    X_input = np.asarray(beats, dtype=np.float32)[..., np.newaxis]
    return record, signal_360, np.asarray(valid_peaks), X_input, detector_name, original_fs


# ---------------------------------------------------------
# Sidebar and page header
# ---------------------------------------------------------
st.title("💓 ECG Heartbeat Classification")
st.write(
    "Upload a WFDB ECG record and run the trained CNN across the recording. "
    "The app reports predictions for detected heartbeats and a summary for the full record."
)
st.warning(
    "Research/student prototype only — not a medical diagnosis. "
    "Automatic R-peak detection can miss or misplace beats, especially on noisy or unfamiliar ECGs."
)

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: `{MODEL_PATH.name}`. "
        "Place `heartbeat_cnn.keras` in the same GitHub repository folder as `app.py`."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH))
except Exception as exc:
    st.error(f"Could not load the CNN model: {exc}")
    st.stop()

with st.sidebar:
    st.subheader("Model status")
    st.success("CNN model loaded")
    st.caption(f"Model file: {MODEL_PATH.name}")
    st.caption("Expected input: 187 samples per beat")
    st.caption("Output classes: N, S, V, F, Q")

st.subheader("1. Upload the ECG record")
st.markdown(
    """
    Upload the record's **`.hea` header and its associated signal file(s)**, usually a
    `.dat` file. A `.hea` file contains metadata and points to the signal file; it
    normally does **not** contain the ECG samples by itself. You do not need an `.atr`
    annotation file to predict a new record.
    """
)

uploaded_files = st.file_uploader(
    "Select the .hea file and all signal files it references",
    type=["hea", "dat", "mat", "atr"],
    accept_multiple_files=True,
    help="For a typical MIT-BIH record, select both 100.hea and 100.dat. The .atr file is optional for prediction.",
)

hea_files = [f for f in uploaded_files if f.name.lower().endswith(".hea")]
if len(hea_files) > 1:
    selected_hea_name = st.selectbox("Choose the record header", [f.name for f in hea_files])
    selected_hea = next(f for f in hea_files if f.name == selected_hea_name)
else:
    selected_hea = hea_files[0] if hea_files else None

run_prediction = st.button(
    "Run prediction for the full record",
    type="primary",
    disabled=(selected_hea is None or len(uploaded_files) == 0),
)

if run_prediction:
    with st.spinner("Reading ECG and running prediction..."):
        try:
            with TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)

                # Save all uploaded files using their original filenames
                for uploaded in uploaded_files:
                    file_path = temp_path / Path(uploaded.name).name
                    file_path.write_bytes(uploaded.getvalue())

                record_base = temp_path / Path(selected_hea.name).stem
                header_path = Path(str(record_base) + ".hea")

                if not header_path.exists():
                    raise FileNotFoundError(
                        "The selected .hea file was not saved correctly."
                    )

                # Verify the signal files referenced by the header
                header_lines = [
                    line.strip()
                    for line in header_path.read_text(
                        errors="replace"
                    ).splitlines()
                    if line.strip()
                ]

                if not header_lines:
                    raise ValueError("The .hea file is empty.")

                header_parts = header_lines[0].split()
                if len(header_parts) < 2:
                    raise ValueError("Invalid WFDB header format.")

                num_signals = int(header_parts[1])
                missing_files = []

                for line in header_lines[1:1 + num_signals]:
                    parts = line.split()
                    if not parts:
                        continue

                    referenced_file = parts[0]
                    expected_path = temp_path / referenced_file

                    if not expected_path.exists():
                        expected_path = (
                            temp_path / Path(referenced_file).name
                        )

                    if not expected_path.exists():
                        missing_files.append(referenced_file)

                if missing_files:
                    raise FileNotFoundError(
                        "Please upload the signal file(s) referenced by "
                        "the .hea file: " + ", ".join(missing_files)
                    )

                # Prepare heartbeat segments and run CNN prediction
                (
                    record,
                    signal_360,
                    peaks,
                    X_input,
                    detector_name,
                    original_fs,
                ) = prepare_record(record_base)

                model_output = model.predict(X_input, verbose=0)

                if model_output.ndim != 2 or model_output.shape[1] != 5:
                    raise ValueError(
                        f"Unexpected model output shape: {model_output.shape}"
                    )

                pred_indices = np.argmax(model_output, axis=1)
                pred_classes = [CLASS_NAMES[i] for i in pred_indices]
                confidences = np.max(model_output, axis=1) * 100

                results = pd.DataFrame({
                    "Beat #": np.arange(1, len(peaks) + 1),
                    "R-peak time (s)": np.round(
                        peaks / TRAINING_FS, 3
                    ),
                    "Predicted class": pred_classes,
                    "Top score (%)": np.round(confidences, 2),
                })

                for i, class_name in enumerate(CLASS_NAMES):
                    results[f"P({class_name}) (%)"] = np.round(
                        model_output[:, i] * 100, 2
                    )

                counts = (
                    pd.Series(pred_classes)
                    .value_counts()
                    .to_dict()
                )

                st.session_state["ecg_results"] = results
                st.session_state["ecg_summary"] = {
                    "record_name": record_base.name,
                    "original_fs": original_fs,
                    "duration": len(record.p_signal) / original_fs,
                    "detector": detector_name,
                    "num_beats": len(peaks),
                    "counts": counts,
                    "signal_360": signal_360,
                    "peaks": peaks,
                }

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
