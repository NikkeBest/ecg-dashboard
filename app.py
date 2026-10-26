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


def main() -> None:
    st.set_page_config(page_title="ECG Heartbeat Classification", page_icon="💓", layout="wide")
    st.title("💓 ECG Heartbeat Classification")
    st.write(
        "Compare annotation-centered evaluation with deployment-style automatic R-peak detection "
        "using the same trained CNN and shared heartbeat preprocessing."
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

    mode = st.radio(
        "Inference mode",
        ["Reference Annotation Mode (evaluation)", "Automatic Detection Mode (deployment)"],
        horizontal=True,
    )
    if mode.startswith("Reference"):
        st.info(
            "Requires the matching .hea, .dat, and .atr. Uses the first signal lead and annotation "
            "sample positions directly; XQRS/SciPy detection is not run. Unsupported annotation symbols "
            "are reported and skipped, as in training."
        )
    else:
        st.info(
            "Requires the .hea and its referenced signal file (normally .dat). XQRS detects beats "
            "automatically; the approximate SciPy fallback is used only if XQRS fails or returns fewer "
            "than two peaks. An optional matching .atr is used only for evaluation comparison."
        )

    uploaded_files = st.file_uploader(
        "WFDB files (.hea and matching signal file; add .atr for reference evaluation), or a ZIP bundle",
        type=["hea", "dat", "mat", "atr", "xyz", "zip"],
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
        if st.button(
            "Run reference evaluation" if mode.startswith("Reference") else "Analyze automatically",
            type="primary",
        ):
            if mode.startswith("Reference"):
                try:
                    data = prepare_reference_record(selected_header)
                except ValueError as exc:
                    st.error(str(exc))
                    return
                if len(data["X"]) == 0:
                    st.error(
                        f"No eligible labeled beats: {data['total_annotations']} total annotations, "
                        f"{data['supported_annotations']} supported beat annotations, "
                        f"{data['excluded']} excluded at the signal boundaries."
                    )
                    return
                with st.spinner("Predicting annotation-centered beats…"):
                    probabilities = np.asarray(model.predict(data["X"], verbose=0))
                if probabilities.ndim != 2 or probabilities.shape != (len(data["X"]), 5):
                    raise ValueError(f"Expected prediction shape ({len(data['X'])}, 5); got {probabilities.shape}.")
                if not np.isfinite(probabilities).all():
                    raise ValueError("The CNN returned NaN or infinite prediction values.")
                predictions = probabilities.argmax(axis=1)
                matrix, report, accuracy = _classification_metrics(data["y_true"], predictions)

                st.subheader("Reference annotation evaluation")
                st.caption(
                    "Reference annotation symbols map as N/L/R/e/j→N, A/a/J/S→S, V/E→V, F→F, and /, f, Q→Q. "
                    "All other annotation symbols are ignored. The CNN output classes, in index order, are N, S, V, F, Q."
                )
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total annotations", f"{data['total_annotations']:,}")
                c2.metric("Supported beat annotations", f"{data['supported_annotations']:,}")
                c3.metric("Eligible beats", f"{len(data['X']):,}")
                c4.metric("Excluded: incomplete window", f"{data['excluded']:,}")
                st.caption(
                    f"Ignored unsupported/non-beat annotation symbols: {data['ignored_annotations']:,}. "
                    f"Lead used: {data['lead_name']} at {data['fs']:g} Hz."
                )
                st.metric("Accuracy on eligible annotated beats", f"{accuracy:.4f}")
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
                st.subheader("Confusion matrix")
                st.caption("Rows are reference labels; columns are CNN predictions.")
                st.dataframe(
                    pd.DataFrame(matrix, index=list(CLASS_NAMES.values()), columns=list(CLASS_NAMES.values()))
                    .rename_axis("Reference"),
                    width="stretch",
                )
                st.subheader("Classification report")
                st.dataframe(report, width="stretch", hide_index=True)
                results = pd.DataFrame({
                    "Beat": np.arange(1, len(predictions) + 1),
                    "Time (s)": data["centers"] / TRAINING_FS,
                    "Reference annotation symbol": data["symbols"],
                    "Reference class": [CLASS_NAMES[int(i)] for i in data["y_true"]],
                    "Predicted class": [CLASS_NAMES[int(i)] for i in predictions],
                    "Correct": data["y_true"] == predictions,
                    "Confidence": probabilities.max(axis=1),
                })
                signal = data["signal"]
                lead_name = data["lead_name"]
                sampling_rate = data["fs"]
                csv_stem = selected_header.stem
            else:
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

                atr_path = _matching_sidecar(selected_header, ".atr")
                if atr_path is not None:
                    try:
                        annotation = wfdb.rdann(str(selected_header.with_suffix("")), "atr")
                        reference_centers = np.asarray([
                            int(sample) for sample, symbol in zip(annotation.sample, annotation.symbol)
                            if symbol in ANNOTATION_TO_CLASS
                        ], dtype=int)
                        tolerance = int(round(0.15 * TRAINING_FS))
                        matched, extra, missed = compare_peak_positions(
                            data["all_peaks"], reference_centers, tolerance
                        )
                        st.caption(
                            f"Optional .atr evaluation only (one-to-one, ±150 ms): "
                            f"{matched:,} matched, {missed:,} reference beats missed, "
                            f"{extra:,} extra detections. Annotations did not generate these predictions."
                        )
                    except Exception as exc:
                        st.warning(f"Could not read the optional .atr file for comparison: {exc}")

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
                file_name=f"{csv_stem}_{'reference' if mode.startswith('Reference') else 'automatic'}_predictions.csv",
                mime="text/csv",
            )
            counts = {name: int((predictions == idx).sum()) for idx, name in CLASS_NAMES.items()}
            confidence = np.asarray(probabilities).max(axis=1)
            rr_seconds = np.diff(np.asarray(results["Time (s)"], dtype=float))
            rr_seconds = rr_seconds[(rr_seconds > 0.25) & (rr_seconds < 2.5)]
            estimated_hr = float(60.0 / np.median(rr_seconds)) if len(rr_seconds) else None
            llm_payload = {
                "record_name": csv_stem,
                "mode": "reference_annotation_evaluation" if mode.startswith("Reference") else "automatic_detection",
                "sampling_rate_hz": float(sampling_rate),
                "lead_name": str(lead_name),
                "beats_classified": int(len(predictions)),
                "predicted_class_counts": counts,
                "predicted_class_percentages": {name: round(100 * count / max(1, len(predictions)), 2) for name, count in counts.items()},
                "mean_model_confidence": round(float(confidence.mean()), 4) if len(confidence) else None,
                "low_confidence_beats_below_0_60": int((confidence < 0.60).sum()),
                "estimated_heart_rate_bpm_from_detected_peak_intervals": round(estimated_hr, 1) if estimated_hr else None,
                "detector": data.get("detector", "reference annotations"),
                "detector_diagnostic": data.get("diagnostic", None),
                "reference_accuracy": round(float(accuracy), 4) if mode.startswith("Reference") else None,
                "reference_support_by_class": {name: int((data["y_true"] == idx).sum()) for idx, name in CLASS_NAMES.items()} if mode.startswith("Reference") else None,
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
            st.caption("Groq receives summary metrics and model outputs, not the raw ECG waveform or full beat table.")
            if not groq_available():
                st.info("To enable Groq, add GROQ_API_KEY to Streamlit Secrets. See GROQ_SETUP.md for setup.")
            payload = st.session_state["ecg_llm_payload"]
            if st.button("Generate English analysis report", type="primary", key="gemini_report"):
                with st.spinner("Generating report with Gemini…"):
                    try:
                        st.session_state["ecg_llm_report"] = ask_groq(
                            payload,
                            "Write an English report with sections: Summary, Model Output Distribution, Data Quality and Uncertainty, Limitations, and Suggested Next Steps. Only describe supplied metrics. Do not diagnose.",
                        )
                    except Exception as exc:
                        st.error(f"Groq request failed: {exc}")
            if st.session_state.get("ecg_llm_report"):
                st.markdown(st.session_state["ecg_llm_report"])
                st.download_button("Download Groq report (.md)", st.session_state["ecg_llm_report"], file_name="ecg_groq_analysis_report.md", mime="text/markdown")
            st.subheader("Ask Groq about these results")
            if "ecg_llm_chat" not in st.session_state:
                st.session_state["ecg_llm_chat"] = []
            for item in st.session_state["ecg_llm_chat"]:
                with st.chat_message(item["role"]):
                    st.markdown(item["content"])
            question = st.chat_input("Ask about class counts, confidence, or detection limitations…")
            if question:
                prior_history = list(st.session_state["ecg_llm_chat"])
                st.session_state["ecg_llm_chat"].append({"role": "user", "content": question})
                with st.chat_message("user"):
                    st.markdown(question)
                with st.chat_message("assistant"):
                    with st.spinner("Groq is answering…"):
                        try:
                            answer = ask_groq(payload, question, prior_history)
                            st.markdown(answer)
                            st.session_state["ecg_llm_chat"].append({"role": "assistant", "content": answer})
                        except Exception as exc:
                            st.error(f"Groq request failed: {exc}")
    except Exception as exc:
        st.error(str(exc))


if __name__ == "__main__":
    main()








