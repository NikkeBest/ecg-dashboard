"""Run XQRS on MIT-BIH record 101 and compare with reference beat annotations.

Usage: python scripts/xqrs_record101_check.py /path/to/record101/101
WFDB 4.3.0 is the version pinned by requirements.txt.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import wfdb
from wfdb.processing import xqrs_detect


ANNOTATION_TO_CLASS = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}
FS = 360
WINDOW_PRE = 90
WINDOW_LENGTH = 187
MATCH_TOLERANCE = round(0.150 * FS)


def match_one_to_one(detected: np.ndarray, reference: np.ndarray, tolerance: int):
    detected = np.sort(np.asarray(detected, dtype=int))
    reference = np.sort(np.asarray(reference, dtype=int))
    i = j = matched = 0
    while i < len(detected) and j < len(reference):
        delta = int(detected[i]) - int(reference[j])
        if abs(delta) <= tolerance:
            matched += 1
            i += 1
            j += 1
        elif delta < -tolerance:
            i += 1
        else:
            j += 1
    return matched, len(detected) - matched, len(reference) - matched


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/xqrs_record101_check.py /path/to/101")
    record_base = Path(sys.argv[1])
    record = wfdb.rdrecord(str(record_base), physical=True)
    annotation = wfdb.rdann(str(record_base), "atr")
    if not np.isclose(float(record.fs), FS):
        raise RuntimeError(f"Expected record 101 at {FS} Hz, got {record.fs:g} Hz.")
    if record.p_signal is None or record.p_signal.ndim != 2 or record.p_signal.shape[1] == 0:
        raise RuntimeError("Record 101 has no readable first signal lead.")

    signal = np.asarray(record.p_signal[:, 0], dtype=np.float64)
    peaks = np.asarray(xqrs_detect(sig=signal, fs=record.fs, verbose=False), dtype=int)
    if peaks.ndim != 1:
        raise RuntimeError(f"XQRS returned an invalid peak array with shape {peaks.shape}.")

    supported_centers = np.asarray([
        int(sample)
        for sample, symbol in zip(annotation.sample, annotation.symbol)
        if symbol in ANNOTATION_TO_CLASS
    ], dtype=int)
    eligible_reference = supported_centers[
        (supported_centers - WINDOW_PRE >= 0)
        & (supported_centers - WINDOW_PRE + WINDOW_LENGTH <= len(signal))
    ]
    matched, extra, missed = match_one_to_one(peaks, eligible_reference, MATCH_TOLERANCE)

    print(json.dumps({
        "wfdb_version": getattr(wfdb, "__version__", "unknown"),
        "record": record_base.name,
        "sampling_rate_hz": float(record.fs),
        "lead": record.sig_name[0] if record.sig_name else "Lead 0",
        "xqrs_detected_peaks": int(len(peaks)),
        "recognized_reference_beats": int(len(supported_centers)),
        "eligible_reference_beats": int(len(eligible_reference)),
        "excluded_reference_boundary_beats": int(len(supported_centers) - len(eligible_reference)),
        "matching_tolerance_ms": 150,
        "matched": int(matched),
        "missed": int(missed),
        "extra": int(extra),
    }, indent=2))


if __name__ == "__main__":
    main()
