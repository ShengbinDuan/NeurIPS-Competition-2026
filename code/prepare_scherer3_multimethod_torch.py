from __future__ import annotations

r"""
Scherer2015 / BNCI2015_004 -> flexible 3-class EEG bank for cross-session experiments.

Classes (fixed):
    0 = SUB  / mental subtraction       (raw MATLAB y=2)
    1 = WORD / word association         (raw MATLAB y=1)
    2 = HAND / right-hand motor imagery (raw MATLAB y=4)

Why this preparation differs from the earlier 4-s script
---------------------------------------------------------
The published paradigm presents the task cue at t=3 s and the imagery/mental task
continues until t=10 s.  Instead of permanently cropping only [3, 7) s, this file
keeps a wider REAL-SIGNAL context [2, 11) s.  The training/sweep script can then
compare [3,7), [4,8), [5,9), [6,10), [3,10), multiple frequency bands, CAR/no-CAR,
etc. without rereading the MATLAB files.

Preparation intentionally does only signal conditioning that is common to all
experiments:
    continuous official MATLAB signal
      -> extract [2,11) s context per selected trial
      -> linear detrend on the full context
      -> 50-Hz notch
      -> broad zero-phase 1-48 Hz Butterworth band-pass
      -> save WITHOUT CAR (rereferencing is an experiment variable later)

No trial is silently removed.  Source artifact flags and a conservative automatic
flag are stored.  Only the TRAINING script may choose to exclude flagged training
rows. Validation/test rows are always preserved.

Output:
    prepared_data/Scherer2015_3Class_MultiMethod_Torch/
        calibration_session0.npz
        cross_session_test_session1.npz
"""

import gc
import os
import zlib
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("MOABB_DOWNLOAD_PROVIDER", "upstream")

import numpy as np
from scipy import signal
from scipy.io import loadmat
from tqdm.auto import tqdm


# =============================================================================
# Configuration
# =============================================================================
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = Path(
    os.environ.get(
        "SCHERER_TORCH_DATA_ROOT",
        str(PROJECT_DIR / "prepared_data" / "Scherer2015_3Class_MultiMethod_Torch"),
    )
)

SUBJECT_ID_TO_LETTER = {
    1: "A", 2: "C", 3: "D", 4: "E", 5: "F",
    6: "G", 7: "H", 8: "J", 9: "L",
}

_subject_env = os.environ.get("SUBJECT_IDS", "").strip()
if _subject_env:
    SUBJECTS = [int(x.strip()) for x in _subject_env.split(",") if x.strip()]
else:
    SUBJECTS = list(SUBJECT_ID_TO_LETTER)
for _sid in SUBJECTS:
    if _sid not in SUBJECT_ID_TO_LETTER:
        raise ValueError(f"Unknown subject id {_sid}; valid ids={list(SUBJECT_ID_TO_LETTER)}")

EXPECTED_SFREQ = 256.0
EXPECTED_N_CHANS = 30

CH_NAMES = [
    "AFz", "F7", "F3", "Fz", "F4", "F8",
    "FC3", "FCz", "FC4",
    "T3", "C3", "Cz", "C4", "T4",
    "CP3", "CPz", "CP4",
    "P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8",
    "PO3", "PO4", "O1", "O2",
]

RAW_TO_TARGET = {
    1: (1, "WORD"),
    2: (0, "SUB"),
    4: (2, "HAND"),
}
CLASS_NAMES = np.asarray(["SUB", "WORD", "HAND"])

# Keep 1 s real context before the cue and after the 7-s task interval.
CONTEXT_START_S = float(os.environ.get("CONTEXT_START_S", "2.0"))
CONTEXT_END_S = float(os.environ.get("CONTEXT_END_S", "11.0"))
TASK_START_S = float(os.environ.get("TASK_START_S", "3.0"))
TASK_END_S = float(os.environ.get("TASK_END_S", "10.0"))

NOTCH_HZ = float(os.environ.get("NOTCH_HZ", "50.0"))
NOTCH_Q = float(os.environ.get("NOTCH_Q", "30.0"))
BASE_BANDPASS_LO = float(os.environ.get("BASE_BANDPASS_LO", "1.0"))
BASE_BANDPASS_HI = float(os.environ.get("BASE_BANDPASS_HI", "48.0"))
BASE_BANDPASS_ORDER = int(os.environ.get("BASE_BANDPASS_ORDER", "4"))
AUTO_ARTIFACT_Z = float(os.environ.get("AUTO_ARTIFACT_Z", "6.0"))

# Official MATLAB X is represented in microvolts.
RAW_UV_TO_V = 1e-6


# =============================================================================
# MATLAB helpers
# =============================================================================
def _as_1d(value: Any, dtype=None) -> np.ndarray:
    arr = np.asarray(value).reshape(-1)
    return arr.astype(dtype, copy=False) if dtype is not None else arr


def _get_field(run: Any, name: str, default: Any = None) -> Any:
    if hasattr(run, name):
        return getattr(run, name)
    if isinstance(run, np.void) and run.dtype.names and name in run.dtype.names:
        return run[name]
    return default


def _split_sessions(mat: dict) -> list[Any]:
    if "data" not in mat:
        keys = sorted(k for k in mat if not k.startswith("__"))
        raise RuntimeError(f"MAT file is missing top-level 'data'. Keys={keys}")
    data = mat["data"]
    sessions = list(data.reshape(-1)) if isinstance(data, np.ndarray) else [data]
    if len(sessions) != 2:
        raise RuntimeError(f"Expected exactly 2 sessions, found {len(sessions)}")
    return sessions


def _flatten_paths(obj: Any) -> list[Path]:
    out: list[Path] = []
    if obj is None:
        return out
    if isinstance(obj, (str, os.PathLike)):
        out.append(Path(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_flatten_paths(v))
    elif isinstance(obj, (list, tuple, set, np.ndarray)):
        for v in obj:
            out.extend(_flatten_paths(v))
    return out


def locate_mat_file(subject_id: int, letter: str) -> Path:
    raw_dir = os.environ.get("SCHERER2015_RAW_DIR")
    candidates: list[Path] = []
    if raw_dir:
        candidates.append(Path(raw_dir).expanduser() / f"{letter}.mat")
    candidates.extend([
        PROJECT_DIR / "Scherer2015Individually" / "sourcedata" / f"{letter}.mat",
        PROJECT_DIR / "Scherer2015Individually" / f"{letter}.mat",
        PROJECT_DIR / "sourcedata" / f"{letter}.mat",
        PROJECT_DIR / f"{letter}.mat",
        PROJECT_DIR / "data" / f"{letter}.mat",
        PROJECT_DIR / "raw_data" / f"{letter}.mat",
        PROJECT_DIR / "raw" / f"{letter}.mat",
    ])
    for p in candidates:
        if p.is_file():
            return p.resolve()

    try:
        from moabb.datasets.bnci.bnci_2015 import _load_data_004_2015
        try:
            result = _load_data_004_2015(subject_id, only_filenames=True, verbose=False)
        except TypeError:
            result = _load_data_004_2015(subject_id, only_filenames=True)
        for p in _flatten_paths(result):
            if p.is_file() and p.suffix.lower() == ".mat" and p.stem.upper() == letter:
                return p.resolve()
    except Exception:
        pass

    try:
        from moabb.datasets import BNCI2015_004
        dataset = BNCI2015_004()
        attempts = [
            lambda: dataset.data_path(subject_id),
            lambda: dataset.data_path(subject=subject_id),
        ]
        for attempt in attempts:
            try:
                result = attempt()
            except Exception:
                continue
            for p in _flatten_paths(result):
                if p.is_file() and p.suffix.lower() == ".mat" and p.stem.upper() == letter:
                    return p.resolve()
    except Exception:
        pass

    raise FileNotFoundError(
        f"Cannot locate {letter}.mat. Put A.mat/C.mat/.../L.mat in a raw folder and set:\n"
        f'  PowerShell: $env:SCHERER2015_RAW_DIR="D:\\your\\raw\\folder"\n'
        "or install MOABB so the official BNCI2015_004 files can be resolved."
    )


def _force_redownload(subject_id: int, letter: str) -> Path | None:
    try:
        from moabb.datasets.bnci.bnci_2015 import _load_data_004_2015
        try:
            result = _load_data_004_2015(
                subject_id, force_update=True, only_filenames=True, verbose=False
            )
        except TypeError:
            result = _load_data_004_2015(
                subject_id, force_update=True, only_filenames=True
            )
        for p in _flatten_paths(result):
            if p.is_file() and p.suffix.lower() == ".mat" and p.stem.upper() == letter:
                return p.resolve()
    except Exception:
        return None
    return None


def load_mat_safe(subject_id: int, letter: str, path: Path) -> tuple[dict, Path]:
    kwargs = dict(struct_as_record=False, squeeze_me=True, verify_compressed_data_integrity=True)
    try:
        return loadmat(path, **kwargs), path
    except (OSError, EOFError, ValueError, zlib.error) as first:
        fresh = _force_redownload(subject_id, letter)
        if fresh is None:
            raise RuntimeError(
                f"Failed to read {path}: {type(first).__name__}: {first}. "
                "The file may be incomplete/corrupt."
            ) from first
        return loadmat(fresh, **kwargs), fresh


# =============================================================================
# Signal extraction + common conditioning
# =============================================================================
def _continuous_samples_channels(run: Any) -> np.ndarray:
    X = np.asarray(_get_field(run, "X"), dtype=np.float64)
    if X.ndim != 2:
        raise RuntimeError(f"run.X must be 2-D, got {X.shape}")
    if X.shape[1] == EXPECTED_N_CHANS:
        Xsc = X
    elif X.shape[0] == EXPECTED_N_CHANS:
        Xsc = X.T
    else:
        raise RuntimeError(f"Cannot infer channel axis from raw X shape={X.shape}")
    if not np.isfinite(Xsc).all():
        raise RuntimeError("Raw X contains NaN/Inf")
    return Xsc * RAW_UV_TO_V


def _source_artifact_flags(run: Any, n_trials: int) -> tuple[np.ndarray, bool]:
    value = _get_field(run, "artifacts", None)
    if value is None:
        return np.zeros(n_trials, dtype=np.int64), False
    flag = _as_1d(value, dtype=np.int64)
    if len(flag) != n_trials:
        raise RuntimeError(
            f"run.artifacts length mismatch: {len(flag)} vs number of trials {n_trials}"
        )
    return (flag != 0).astype(np.int64), True


def _build_filters(sfreq: float):
    if not (0.0 < BASE_BANDPASS_LO < BASE_BANDPASS_HI < sfreq / 2.0):
        raise ValueError(
            "Need 0 < BASE_BANDPASS_LO < BASE_BANDPASS_HI < Nyquist; got "
            f"{BASE_BANDPASS_LO}, {BASE_BANDPASS_HI}, sfreq={sfreq}"
        )
    b_notch, a_notch = signal.iirnotch(NOTCH_HZ, NOTCH_Q, fs=sfreq)
    notch_sos = signal.tf2sos(b_notch, a_notch)
    bp_sos = signal.butter(
        BASE_BANDPASS_ORDER,
        [BASE_BANDPASS_LO, BASE_BANDPASS_HI],
        btype="bandpass",
        fs=sfreq,
        output="sos",
    )
    return notch_sos, bp_sos


def _condition_context_epochs(x: np.ndarray, sfreq: float) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise RuntimeError("NaN/Inf before filtering")
    x = signal.detrend(x, axis=-1, type="linear", overwrite_data=False)
    notch_sos, bp_sos = _build_filters(sfreq)
    x = signal.sosfiltfilt(notch_sos, x, axis=-1)
    x = signal.sosfiltfilt(bp_sos, x, axis=-1)
    if not np.isfinite(x).all():
        raise RuntimeError("NaN/Inf after conditioning")
    return x.astype(np.float32, copy=False)


def _task_slice_from_context(sfreq: float) -> slice:
    a = int(round((TASK_START_S - CONTEXT_START_S) * sfreq))
    b = int(round((TASK_END_S - CONTEXT_START_S) * sfreq))
    return slice(a, b)


def _auto_artifact_flags(Xctx: np.ndarray, sfreq: float) -> np.ndarray:
    """Robust flag based on the full 7-s task interval; no rows are removed here."""
    X = np.asarray(Xctx[..., _task_slice_from_context(sfreq)], dtype=np.float64)
    # CAR only for artifact scoring. Saved signal remains non-CAR.
    X = X - X.mean(axis=1, keepdims=True)
    ptp_ch = np.ptp(X, axis=-1)
    score = np.percentile(ptp_ch, 95.0, axis=1)
    zbase = np.log(np.maximum(score, np.finfo(np.float64).tiny))
    med = float(np.median(zbase))
    mad = float(np.median(np.abs(zbase - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        scale = float(np.std(zbase))
    if not np.isfinite(scale) or scale < 1e-12:
        return np.zeros(len(X), dtype=np.int64)
    robust_z = (zbase - med) / scale
    return (robust_z > AUTO_ARTIFACT_Z).astype(np.int64)


def extract_and_condition_session(run: Any, subject_id: int, session_index: int) -> dict[str, np.ndarray]:
    sfreq = float(np.asarray(_get_field(run, "fs")).squeeze())
    if not np.isclose(sfreq, EXPECTED_SFREQ):
        raise RuntimeError(f"Expected {EXPECTED_SFREQ} Hz, got {sfreq}")

    Xcont = _continuous_samples_channels(run)
    y_all = _as_1d(_get_field(run, "y"), dtype=np.int64)
    trial_matlab = _as_1d(_get_field(run, "trial"), dtype=np.int64)
    if len(y_all) != len(trial_matlab):
        raise RuntimeError("len(y) != len(trial)")
    if not set(np.unique(y_all).tolist()).issuperset({1, 2, 4}):
        raise RuntimeError(f"Expected raw labels including 1/2/4, got {np.unique(y_all)}")

    src_flag_all, src_available = _source_artifact_flags(run, len(y_all))
    target_mask = np.isin(y_all, [1, 2, 4])
    paper_y = y_all[target_mask]
    starts = trial_matlab[target_mask] - 1
    src_flag = src_flag_all[target_mask]
    target_y = np.asarray([RAW_TO_TARGET[int(v)][0] for v in paper_y], dtype=np.int64)

    if CONTEXT_START_S < 0 or CONTEXT_END_S <= CONTEXT_START_S:
        raise ValueError("Invalid context interval")
    if not (CONTEXT_START_S <= TASK_START_S < TASK_END_S <= CONTEXT_END_S):
        raise ValueError("Task interval must lie inside saved context interval")

    offset = int(round(CONTEXT_START_S * sfreq))
    n_context = int(round((CONTEXT_END_S - CONTEXT_START_S) * sfreq))
    Xctx = np.empty((len(starts), EXPECTED_N_CHANS, n_context), dtype=np.float32)

    for i, trial_start in enumerate(starts):
        begin = int(trial_start) + offset
        stop = begin + n_context
        # Normally the continuous recording contains this whole interval.  For a
        # boundary trial in an export that stops immediately after the last trial,
        # reflect-pad only the missing FILTER CONTEXT.  The task interval itself
        # still has to be real recorded data.
        real_begin = max(0, begin)
        real_stop = min(len(Xcont), stop)
        seg = Xcont[real_begin:real_stop]
        pad_left = max(0, -begin)
        pad_right = max(0, stop - len(Xcont))
        task_begin = int(trial_start) + int(round(TASK_START_S * sfreq))
        task_stop = int(trial_start) + int(round(TASK_END_S * sfreq))
        if task_begin < 0 or task_stop > len(Xcont):
            raise RuntimeError(
                f"subject={subject_id} session={session_index} epoch {i}: real task interval "
                f"[{task_begin},{task_stop}) is outside recording length={len(Xcont)}"
            )
        if pad_left or pad_right:
            seg = np.pad(seg, ((pad_left, pad_right), (0, 0)), mode="reflect")
        if len(seg) != n_context:
            raise RuntimeError(f"Context extraction length mismatch: {len(seg)} != {n_context}")
        Xctx[i] = seg.T.astype(np.float32, copy=False)

    Xctx = _condition_context_epochs(Xctx, sfreq)
    auto_flag = _auto_artifact_flags(Xctx, sfreq)

    n = len(Xctx)
    return {
        "X": Xctx,
        "y": target_y,
        "paper_y": paper_y.astype(np.int64, copy=False),
        "subject": np.full(n, subject_id, dtype=np.int64),
        "session": np.full(n, session_index, dtype=np.int64),
        "event_sample": starts.astype(np.int64, copy=False),
        "source_artifact_flag": src_flag.astype(np.int64, copy=False),
        "source_artifact_available": np.full(n, int(src_available), dtype=np.int64),
        "auto_artifact_flag": auto_flag,
    }


def concatenate_parts(parts: list[dict[str, np.ndarray]], session_index: int) -> dict[str, np.ndarray]:
    if not parts:
        raise RuntimeError(f"No data collected for session {session_index}")
    keys = parts[0].keys()
    out = {k: np.concatenate([p[k] for p in parts], axis=0) for k in keys}
    if set(np.unique(out["y"]).tolist()) != {0, 1, 2}:
        raise RuntimeError(f"Target labels are not 0/1/2: {np.unique(out['y'])}")
    if not np.isfinite(out["X"]).all():
        raise RuntimeError("Prepared X contains NaN/Inf")
    return out


def save_split(path: Path, data: dict[str, np.ndarray]) -> None:
    np.savez_compressed(
        path,
        **data,
        sfreq=np.asarray(EXPECTED_SFREQ, dtype=np.float32),
        ch_names=np.asarray(CH_NAMES),
        class_names=CLASS_NAMES,
        context_start_s=np.asarray(CONTEXT_START_S, dtype=np.float32),
        context_end_s=np.asarray(CONTEXT_END_S, dtype=np.float32),
        task_start_s=np.asarray(TASK_START_S, dtype=np.float32),
        task_end_s=np.asarray(TASK_END_S, dtype=np.float32),
        base_bandpass_hz=np.asarray([BASE_BANDPASS_LO, BASE_BANDPASS_HI], dtype=np.float32),
        notch_hz=np.asarray(NOTCH_HZ, dtype=np.float32),
        rereference=np.asarray("none; original mastoid-reference relationship retained"),
        preprocessing=np.asarray(
            "real continuous context -> linear detrend -> 50Hz notch -> "
            f"{BASE_BANDPASS_LO:g}-{BASE_BANDPASS_HI:g}Hz zero-phase broad bandpass; no CAR"
        ),
    )


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    session_parts: dict[int, list[dict[str, np.ndarray]]] = {0: [], 1: []}

    bar = tqdm(SUBJECTS, desc="prepare multimethod subjects", unit="subject", dynamic_ncols=True)
    for subject_id in bar:
        letter = SUBJECT_ID_TO_LETTER[subject_id]
        mat_path = locate_mat_file(subject_id, letter)
        mat, effective_path = load_mat_safe(subject_id, letter, mat_path)
        sessions = _split_sessions(mat)

        for session_index, run in enumerate(sessions):
            part = extract_and_condition_session(run, subject_id, session_index)
            session_parts[session_index].append(part)

        bar.set_postfix(subject=f"{subject_id}:{letter}", file=effective_path.name, refresh=False)
        del sessions, mat
        gc.collect()

    cal = concatenate_parts(session_parts[0], 0)
    test = concatenate_parts(session_parts[1], 1)

    cal_path = OUTPUT_ROOT / "calibration_session0.npz"
    test_path = OUTPUT_ROOT / "cross_session_test_session1.npz"
    save_split(cal_path, cal)
    save_split(test_path, test)

    counts0 = np.bincount(cal["y"], minlength=3).tolist()
    counts1 = np.bincount(test["y"], minlength=3).tolist()
    src0 = int(cal["source_artifact_flag"].sum())
    src1 = int(test["source_artifact_flag"].sum())
    print(
        "Done | "
        f"session0={cal['X'].shape} classes={counts0} auto={int(cal['auto_artifact_flag'].sum())} source={src0} | "
        f"session1={test['X'].shape} classes={counts1} auto={int(test['auto_artifact_flag'].sum())} source={src1}"
    )
    print(
        f"Saved real-signal context [{CONTEXT_START_S:g},{CONTEXT_END_S:g}) s; "
        f"task interval [{TASK_START_S:g},{TASK_END_S:g}) s; no CAR."
    )
    print(f"Output: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
