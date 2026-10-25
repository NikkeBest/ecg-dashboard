"""Streamlit dashboard for beat-level ECG classification with a Keras model."""

from __future__ import annotations

import math
import tempfile
import zipfile
from fractions import Fraction
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import wfdb
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt
import tensorflow as tf


APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "heartbeat_cnn.keras"
TRAINING_FS = 360
BEAT_PRE_SAMPLES = 90
BEAT_LENGTH = 187
CLASS_NAMES = {
    0: "N — Normal",
    1: "S — Supraventricular ectopic",
    2: "V — Ventricular ectopic",
    3: "F — Fusion",
    4: "Q — Unknown / paced",
}


@st.cache_resource(show_spinner="Loading the ECG classification model…")
def load_model(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)


def normalize_beat(beat: np.ndarray) -> np.ndarray:
    beat = np.asarray(beat, dtype=np.float32)
    return (beat - float(beat.mean())) / (float(beat.std()) + 1e-8)


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


def detect_r_peaks(signal: np.ndarray, fs: int) -> tuple[np.ndarray, str]:
    signal = np.asarray(signal, dtype=np.float64).reshape(-1)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    if len(signal) < max(20, int(fs * 2)):
        raise ValueError("The selected signal is too short for reliable beat detection.")
    try:
        peaks = np.asarray(wfdb.processing.xqrs_detect(sig=signal, fs=fs, verbose=False), dtype=int)
        if len(peaks) >= 2:
            return peaks, "WFDB XQRS"
    except Exception:
        pass
    if fs <= 40:
        raise ValueError("Sampling frequency is too low for the fallback QRS detector.")
    high = min(20.0, fs * 0.45)
    sos = butter(2, [5.0, high], btype="bandpass", fs=fs, output="sos")
    try:
        filtered = sosfiltfilt(sos, signal)
    except ValueError as exc:
        raise ValueError("The signal is too short to filter for heartbeat detection.") from exc
    prominence = max(float(np.std(filtered)) * 0.35, 1e-6)
    peaks, _ = find_peaks(np.abs(filtered), distance=max(1, int(0.25 * fs)), prominence=prominence)
    if len(peaks) < 2:
        raise ValueError("Could not detect enough heartbeats. Check the signal channel and sampling rate.")
    return peaks.astype(int), "SciPy fallback (approximate)"


def prepare_record(header_path: Path, channel: int = 0):
    """Read a WFDB record, validate it, resample and create model-ready beats."""
    if header_path.suffix.lower() != ".hea" or not header_path.is_file():
        raise ValueError("Select a WFDB .hea header file.")
    header_fs = _read_header_frequency(header_path)
    try:
        record = wfdb.rdrecord(str(header_path.with_suffix("")), physical=True)
    except Exception as exc:
        raise ValueError(
            f"Could not load {header_path.name}. Include the matching signal file(s) "
            "(.dat, .mat, or the format named in the header) with the .hea file."
        ) from exc
    if record.p_signal is None or record.p_signal.size == 0:
        raise ValueError("The record has no readable signal data.")
    original_fs = float(record.fs)
    if not math.isfinite(original_fs) or original_fs <= 0:
        original_fs = header_fs
    if not math.isclose(original_fs, header_fs, rel_tol=0, abs_tol=1e-6):
        raise ValueError("The sampling frequency parsed from the header does not match WFDB.")
    if channel < 0 or channel >= record.p_signal.shape[1]:
        raise ValueError("The selected ECG channel does not exist in this record.")
    signal = np.asarray(record.p_signal[:, channel], dtype=np.float64)
    if not np.isfinite(signal).any():
        raise ValueError("The selected ECG channel contains no finite samples.")
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    if not math.isclose(original_fs, TRAINING_FS, rel_tol=0, abs_tol=1e-6):
        ratio = Fraction(TRAINING_FS / original_fs).limit_denominator(10000)
        signal = resample_poly(signal, ratio.numerator, ratio.denominator)
    peaks, detector_name = detect_r_peaks(signal, TRAINING_FS)
    beats, valid_peaks = [], []
    for peak in peaks:
        start = int(peak) - BEAT_PRE_SAMPLES
        end = start + BEAT_LENGTH
        if start >= 0 and end <= len(signal):
            beats.append(normalize_beat(signal[start:end]))
            valid_peaks.append(int(peak))
    if not beats:
        raise ValueError("No complete 187-sample heartbeat windows could be extracted.")
    model_input = np.asarray(beats, dtype=np.float32)[..., np.newaxis]
    return record, signal, np.asarray(valid_peaks), model_input, detector_name, original_fs


def main() -> None:
    st.set_page_config(page_title="ECG Heartbeat Classification", page_icon="💓", layout="wide")
    st.title("💓 ECG Heartbeat Classification")
    st.write(
        "Upload a WFDB ECG record and classify detected heartbeat windows with the trained CNN. "
        "Upload the `.hea` header and every signal file named by that header, or upload a ZIP bundle."
    )
    st.warning(
        "Research/student prototype only — not a medical diagnosis. R-peak detection may miss or "
        "misplace beats, especially on noisy or unfamiliar recordings."
    )

    if not MODEL_PATH.is_file():
        st.error(f"Model file not found: {MODEL_PATH.name}. Put it beside app.py in the GitHub repository.")
        st.stop()
    try:
        model = load_model(str(MODEL_PATH))
    except Exception as exc:
        st.error(f"Could not load {MODEL_PATH.name}: {exc}")
        st.stop()

    uploaded_files = st.file_uploader(
        "WFDB files (.hea and matching signal file), or a ZIP bundle",
        type=["hea", "dat", "mat", "atr", "xyz", "zip"],
        accept_multiple_files=True,
    )
    if not uploaded_files:
        st.info("Choose a WFDB header and its corresponding signal file(s) to begin.")
        return

    # Streamlit reruns this script after widget changes. Persist uploads in a
    # per-session folder so the selector never points to a deleted temp folder.
    if "ecg_upload_dir" not in st.session_state:
        st.session_state.ecg_upload_dir = tempfile.mkdtemp(prefix="ecg_upload_")
    root = Path(st.session_state.ecg_upload_dir)
    root.mkdir(parents=True, exist_ok=True)
    try:
            saved = _save_uploads(uploaded_files, root)
            headers = _find_header_files(saved, root)
            if not headers:
                raise ValueError("No .hea file was found. Upload it with its signal file, or use a ZIP bundle.")
            selected_header = st.selectbox("Choose ECG record", headers, format_func=lambda p: p.name)
            channels = None
            try:
                header_fs = _read_header_frequency(selected_header)
                record_for_channels = wfdb.rdrecord(str(selected_header.with_suffix("")), physical=True)
                channel_names = list(record_for_channels.sig_name or [])
                channel_count = len(channel_names) or record_for_channels.n_sig
                labels = [f"{i}: {channel_names[i] if i < len(channel_names) else f'Channel {i}'}" for i in range(channel_count)]
                channels = st.selectbox("ECG channel", range(channel_count), format_func=lambda i: labels[i])
            except Exception as exc:
                st.error(f"Unable to read the selected WFDB record: {exc}")
                return
            if st.button("Analyze recording", type="primary"):
                with st.spinner("Detecting beats and running the model…"):
                    record, signal, peaks, X_input, detector, original_fs = prepare_record(selected_header, channels)
                    predictions = np.asarray(model.predict(X_input, verbose=0))
                if predictions.ndim != 2 or predictions.shape[0] != len(X_input):
                    raise ValueError("The model returned an unexpected output shape.")
                if predictions.shape[1] != 5:
                    raise ValueError(f"Expected a five-class model output; received {predictions.shape[1]} classes.")
                class_ids = predictions.argmax(axis=1)
                results = pd.DataFrame({
                    "Beat": np.arange(1, len(class_ids) + 1),
                    "Time (s)": peaks / TRAINING_FS,
                    "Class": [CLASS_NAMES[int(i)] for i in class_ids],
                    "Confidence": predictions.max(axis=1),
                })
                st.success(f"Analyzed {len(results)} complete beats using {detector}.")
                c1, c2, c3 = st.columns(3)
                c1.metric("Record sampling rate", f"{original_fs:g} Hz")
                c2.metric("Model sampling rate", f"{TRAINING_FS} Hz")
                c3.metric("Detected beats", f"{len(results)}")
                counts = results["Class"].value_counts().rename_axis("Class").reset_index(name="Beats")
                st.subheader("Classification summary")
                st.plotly_chart(px.bar(counts, x="Class", y="Beats", color="Class"), width="stretch")
                st.subheader("Detected ECG signal")
                plot_limit = min(len(signal), TRAINING_FS * 30)
                signal_frame = pd.DataFrame({"Time (s)": np.arange(plot_limit) / TRAINING_FS, "Amplitude": signal[:plot_limit]})
                st.plotly_chart(px.line(signal_frame, x="Time (s)", y="Amplitude"), width="stretch")
                st.subheader("Beat predictions")
                st.dataframe(results, width="stretch", hide_index=True)
                st.download_button(
                    "Download predictions (CSV)",
                    results.to_csv(index=False).encode("utf-8"),
                    file_name=f"{selected_header.stem}_predictions.csv",
                    mime="text/csv",
                )
    except Exception as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()
