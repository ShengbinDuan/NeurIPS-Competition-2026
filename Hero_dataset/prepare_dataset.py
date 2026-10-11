"""PROTEUS (nm000290) isolated complete 8-second trials, 500/250/120 Hz.

Design:
  - Cue (mi/sub/word) is t=0; crop only [0, 8) from the source EDF.
  - Detrend, 50-Hz notch, 1--45-Hz Butterworth, and common-average-reference
    are applied once at 500 Hz on that 8-s epoch *only*.
  - 250/120 Hz are anti-aliased polyphase resamplings of that same 500-Hz epoch.
  - Store volts, float32; do NOT fit data-dependent scalers or z-score.
  - Preserve the original SubN/sessionN/trialxxxxx_run... NPZ layout and labels.
  - Parse TSV feedback/end markers; no guessed 3/5-second phase boundary.
  - Reject malformed/overlapping/incomplete epochs and report every rejection.
  - Never overwrite an output produced by a different configuration.

NOTICE: Zero-phase filtering and full-epoch detrending mix information *within*
the 8-s epoch. For a strictly task-only, feedback-blind model, preprocess the
shorter task interval separately or implement causal filtering.

Usage:
  python prepare_proteus3_full8_stable.py --dry-run
  python prepare_proteus3_full8_stable.py
  python prepare_proteus3_full8_stable.py --self-test
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import mne
import numpy as np
from scipy import signal
from tqdm.auto import tqdm

# ----------- User configuration (keep original project roots) --------------
RAW_ROOT = Path(r"F:\ShengbinDuan\NIPS2026\dataset")
OUTPUT_ROOT = RAW_ROOT / "prepared_data" / "PROTEUS_Competition_TrialNPZ"
SOURCE_SFREQ = 500
NUM_EEG_CHANNELS = 41
SUBJECT_FOLDER_COUNT = 10
SESSION_FOLDER_COUNT = 3
OUTPUT_SFREQS = (500, 250, 120)
INCLUDE_ONLINE_RUNS = True
MAX_RUNS = 0                   # 0 means all; use CLI --max-runs for a small trial
RESUME_EXISTING = True
COMPRESS_NPZ = True
STRICT_COMPLETE_TRIALS = True  # requires feedback/end event, no overlap, ~8 s
END_TOLERANCE_S = 0.10         # tolerate event timing jitter, not shortened trials
BANDPASS_HZ = (1.0, 45.0)
FILTER_ORDER = 4
NOTCH_HZ = (50.0,)
NOTCH_Q = 30.0
DETREND_LINEAR = True
ARTIFACT_PEAK_UV = 200.0
ARTIFACT_PTP_UV = 300.0
ARTIFACT_FLAT_STD_UV = 0.10
ARTIFACT_FLAT_CHANNELS = 2
PIPELINE_VERSION = "proteus_full8_isolated_v1"

# Legacy mappings: do not change without updating your training labels.
CLASS_NAMES = ("SUB", "WORD", "MI")
LABELS = {"sub": 0, "word": 1, "mi": 2}
CODES = {"33025": 0, "33026": 1, "33024": 2}
EVENT_MARKERS = {"768": "start_trial", "786": "fixation_cross",
                 "781": "start_feedback", "800": "end_trial"}
TASKS = {"AcquisitionBH", "AcquisitionGraz", "OnlineRawBH", "OnlineRawGraz"}
INDEX_COLUMNS = (
    "path", "subject", "subject_bids", "session", "session_bids", "trial", "y", "class_name",
    "run", "run_bids", "task", "context", "event_onset_s", "event_sample", "sfreq",
    "n_channels", "n_samples", "window_start_s", "window_end_s", "artifact_flag",
    "artifact_peak_uv", "artifact_ptp_uv", "artifact_flat_channels", "source_file",
    "feedback_start_rel_s", "end_trial_rel_s", "task_duration_s", "feedback_duration_s",
    "cue_duration_s", "event_timing_valid", "timing_warnings",
)
REJECT_COLUMNS = ("subject", "session", "task", "run_bids", "trial", "class_name",
                  "cue_onset_s", "reason", "source_file")
mne.set_log_level("WARNING")


@dataclass(frozen=True)
class Profile:
    name: str
    sfreq: int
    band: tuple[float, float] = BANDPASS_HZ
    notch: tuple[float, ...] = NOTCH_HZ
    reference: str = "average"
    start: float = 0.0
    end: float = 8.0
    mode: str = "window"


PROFILES = tuple(Profile(f"{fs}Hz_BP1-45Hz_N50_CAR_Cue0to8_WindowOnly", fs)
                 for fs in OUTPUT_SFREQS)


def natural_key(s):
    return tuple((0, int(x)) if x.isdigit() else (1, x.lower())
                 for x in re.split(r"(\d+)", str(s)))


def entity(name, key):
    m = re.search(r"(?:^|_)" + re.escape(key) + r"-([^_]+)", name)
    if m is None:
        raise ValueError(f"Missing BIDS {key}: {name}")
    return m.group(1)


def atomic_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    try:
        tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    try:
        with tmp.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def atomic_npz(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    try:
        with tmp.open("wb") as fh:
            (np.savez_compressed if COMPRESS_NPZ else np.savez)(fh, **payload)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def discover_runs(raw_root, max_runs=0):
    runs = []
    for edf in sorted(raw_root.rglob("*_eeg.edf")):
        stem = edf.name.removesuffix("_eeg.edf")
        try:
            task = entity(stem, "task")
        except ValueError:
            continue
        if task not in TASKS or (not INCLUDE_ONLINE_RUNS and task.startswith("OnlineRaw")):
            continue
        events = edf.with_name(stem + "_events.tsv")
        if not events.is_file():
            raise FileNotFoundError(f"Missing TSV for {edf}: {events}")
        runs.append({"edf": edf, "events": events,
                     "subject_bids": entity(stem, "sub"),
                     "session_bids": entity(stem, "ses"),
                     "run_bids": entity(stem, "run"), "task": task})
    runs.sort(key=lambda r: (natural_key(r["subject_bids"]), natural_key(r["session_bids"]),
                             natural_key(r["run_bids"]), r["task"], str(r["edf"])))
    if max_runs:
        runs = runs[:max_runs]
    if not runs:
        raise FileNotFoundError(f"No supported *_eeg.edf under {raw_root}")
    return runs


def identify_subjects_and_sessions(runs):
    subjects = sorted({r["subject_bids"] for r in runs}, key=natural_key)
    if len(subjects) > SUBJECT_FOLDER_COUNT:
        raise ValueError(f"Found {len(subjects)} subjects, expected <= {SUBJECT_FOLDER_COUNT}")
    smap = {name: i + 1 for i, name in enumerate(subjects)}
    sesmap = {}
    for subj in subjects:
        sessions = sorted({r["session_bids"] for r in runs if r["subject_bids"] == subj},
                          key=natural_key)
        if len(sessions) > SESSION_FOLDER_COUNT:
            raise ValueError(f"Too many sessions for {subj}: {sessions}")
        sesmap[subj] = {name: i + 1 for i, name in enumerate(sessions)}
    return smap, sesmap


def normalize_code(value):
    s = str(value if value is not None else "").strip()
    if not s:
        return ""
    try:
        return str(int(float(s)))
    except (TypeError, ValueError, OverflowError):
        return s


def parse_label(row):
    for field in ("trial_type", "event_type", "description", "condition", "label"):
        value = str(row.get(field, "")).strip().lower().replace("-", "_").replace(" ", "_")
        if value in LABELS:
            return LABELS[value]
        if value.startswith(("cue_", "cue:")):
            for key, label in LABELS.items():
                if key in set(re.findall(r"[a-z]+", value)):
                    return label
    for field in ("value", "code", "event_code", "event_value"):
        v = normalize_code(row.get(field, ""))
        if v in CODES:
            return CODES[v]
    return None


def parse_marker(row):
    name = str(row.get("trial_type", "")).strip().lower().replace("-", "_").replace(" ", "_")
    if name in EVENT_MARKERS.values():
        return name
    return EVENT_MARKERS.get(normalize_code(row.get("value", "")))


def read_events(path):
    events = []
    with path.open("r", newline="", encoding="utf-8-sig") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        if not reader.fieldnames or "onset" not in reader.fieldnames:
            raise ValueError(f"Malformed events TSV: {path}")
        for line, row in enumerate(reader, start=2):
            try:
                onset = float(row["onset"])
            except (TypeError, ValueError):
                raise ValueError(f"Bad onset at {path}:{line}")
            if not np.isfinite(onset) or onset < 0:
                raise ValueError(f"Invalid onset at {path}:{line}: {onset}")
            duration_txt = str(row.get("duration", "")).strip()
            try:
                duration = float(duration_txt)
            except (ValueError, TypeError):
                duration = float("nan")
            sample_txt = str(row.get("sample", "")).strip()
            sample = None
            if sample_txt:
                try:
                    sample = int(float(sample_txt))
                except (ValueError, OverflowError):
                    raise ValueError(f"Invalid sample at {path}:{line}: {sample_txt}")
            events.append({"onset": onset, "duration": duration, "sample": sample,
                           "label": parse_label(row), "marker": parse_marker(row),
                           "line": line})
    events.sort(key=lambda e: (e["onset"], e["line"]))
    return events


def near(events, marker, left, right, last=False):
    found = [e for e in events if e["marker"] == marker and left < e["onset"] < right]
    return (found[-1] if last else found[0]) if found else None


def make_plans(runs, subject_map, session_map):
    counters = Counter()
    plans = []
    for run in tqdm(runs, desc="Read TSV", unit="run"):
        subject = subject_map[run["subject_bids"]]
        session = session_map[run["subject_bids"]][run["session_bids"]]
        events = read_events(run["events"])
        # Match the legacy cue de-duplication policy: one trial per cue onset.
        cues_by_onset = {}
        for ev in events:
            if ev["label"] is None:
                continue
            old = cues_by_onset.get(ev["onset"])
            if old is not None and old["label"] != ev["label"]:
                raise ValueError(f"Conflicting class labels at {ev['onset']}s in {run['events']}")
            cues_by_onset.setdefault(ev["onset"], ev)
        cues = list(cues_by_onset.values())
        cue_items = []
        for i, ev in enumerate(cues):
            cue = ev["onset"]
            prev_cue = cues[i - 1]["onset"] if i else -float("inf")
            next_cue = cues[i + 1]["onset"] if i + 1 < len(cues) else float("inf")
            # Do not pair markers from the next cue/trial.
            start = near(events, "start_trial", prev_cue, cue, last=True)
            fixation = near(events, "fixation_cross", max(prev_cue, start["onset"] if start else prev_cue), cue, last=True)
            feedback = near(events, "start_feedback", cue, min(next_cue, cue + 8.5))
            trial_end = near(events, "end_trial", cue, min(next_cue, cue + 8.5))
            next_start = near(events, "start_trial", cue, cue + 8.0 - END_TOLERANCE_S)
            issues, warnings = [], []
            # Source offset is exactly the BIDS cue onset rounded to samples.
            onset_sample = int(round(cue * SOURCE_SFREQ))
            if ev["sample"] is not None and abs(ev["sample"] - onset_sample) > 1:
                issues.append("cue_sample_onset_mismatch")
            if not np.isfinite(ev["duration"]) or abs(ev["duration"] - 8.0) > END_TOLERANCE_S:
                issues.append("cue_duration_not_8s")
            if feedback is None:
                issues.append("missing_feedback")
            elif not 0 < feedback["onset"] - cue < 8:
                issues.append("feedback_outside_trial")
            elif not 2.5 <= feedback["onset"] - cue <= 3.5:
                warnings.append("unusual_feedback_time")
            if trial_end is None:
                issues.append("missing_end_trial")
            elif abs(trial_end["onset"] - (cue + 8)) > END_TOLERANCE_S:
                issues.append("end_trial_not_cue_plus_8")
            if next_cue < cue + 8 - END_TOLERANCE_S or next_start:
                issues.append("overlap_next_trial")
            if start is None:
                warnings.append("missing_start_trial")
            if fixation is None:
                warnings.append("missing_fixation_cross")
            if feedback and trial_end and feedback["onset"] >= trial_end["onset"]:
                issues.append("feedback_not_before_end_trial")
            # Keep numbering exactly as the old script: all cue rows increment it.
            counters[(subject, session)] += 1
            trial_number = counters[(subject, session)]
            label = ev["label"]
            fname = f"trial{trial_number:05d}_run{run['run_bids']}_{run['task']}_{CLASS_NAMES[label]}.npz"
            cue_items.append({"number": trial_number, "label": label, "onset": cue,
                              "sample": onset_sample, "filename": fname,
                              "cue_duration": ev["duration"],
                              "start_trial_rel_s": (start["onset"] - cue if start else float("nan")),
                              "fixation_cross_rel_s": (fixation["onset"] - cue if fixation else float("nan")),
                              "feedback_start_rel_s": (feedback["onset"] - cue if feedback else float("nan")),
                              "end_trial_rel_s": (trial_end["onset"] - cue if trial_end else float("nan")),
                              "issues": tuple(dict.fromkeys(issues)), "warnings": tuple(dict.fromkeys(warnings))})
        plans.append({**run, "subject": subject, "session": session, "cues": cue_items})
    return plans


def project_config(max_runs):
    return {"pipeline_version": PIPELINE_VERSION, "raw_root": str(RAW_ROOT),
            "source_sfreq": SOURCE_SFREQ, "output_sfreqs": OUTPUT_SFREQS,
            "expected_eeg_channels": NUM_EEG_CHANNELS,
            "include_online_runs": INCLUDE_ONLINE_RUNS, "max_runs": max_runs,
            "complete_trials_only": STRICT_COMPLETE_TRIALS,
            "end_tolerance_s": END_TOLERANCE_S, "bandpass_hz": BANDPASS_HZ,
            "notch_hz": NOTCH_HZ, "filter_order": FILTER_ORDER, "notch_q": NOTCH_Q,
            "detrend_linear": DETREND_LINEAR, "reference": "average",
            "filter_padding_s": 0.0, "preprocessing_mode": "window",
            "anti_alias_resampling": "scipy.signal.resample_poly at 500Hz after filtering",
            "unit": "V", "dtype": "float32", "compress": COMPRESS_NPZ,
            "artifact_peak_uv": ARTIFACT_PEAK_UV, "artifact_ptp_uv": ARTIFACT_PTP_UV,
            "artifact_flat_std_uv": ARTIFACT_FLAT_STD_UV,
            "artifact_flat_channels": ARTIFACT_FLAT_CHANNELS}


def fingerprint(config, runs, channels):
    files = []
    for run in runs:
        for name in ("edf", "events"):
            path = run[name]
            s = path.stat()
            files.append((str(path.resolve()), s.st_size, s.st_mtime_ns))
    blob = json.dumps({"config": config, "files": files, "channels": channels},
                      sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def eeg_names(raw, expected=None):
    names = [name for name, kind in zip(raw.ch_names, raw.get_channel_types()) if kind == "eeg"]
    if len(names) != NUM_EEG_CHANNELS:
        raise ValueError(f"Expected {NUM_EEG_CHANNELS} EEG channels, got {len(names)}: {names}")
    if expected is not None and set(names) != set(expected):
        raise ValueError(f"Incompatible channel set: {set(names) ^ set(expected)}")
    return names if expected is None else list(expected)


def preprocess_full8(x):
    """x is [41,4000] V; no data from before/after the trial is accessed."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (NUM_EEG_CHANNELS, 8 * SOURCE_SFREQ) or not np.isfinite(x).all():
        raise ValueError(f"Invalid isolated trial: {x.shape}")
    y = signal.detrend(x, axis=-1, type="linear") if DETREND_LINEAR else x.copy()
    for freq in NOTCH_HZ:
        b, a = signal.iirnotch(freq, NOTCH_Q, fs=SOURCE_SFREQ)
        y = signal.sosfiltfilt(signal.tf2sos(b, a), y, axis=-1)
    band = signal.butter(FILTER_ORDER, BANDPASS_HZ, fs=SOURCE_SFREQ,
                         btype="bandpass", output="sos")
    y = signal.sosfiltfilt(band, y, axis=-1)
    y -= np.mean(y, axis=0, keepdims=True)  # CAR among the 41 EEG channels
    if not np.isfinite(y).all():
        raise ValueError("Non-finite output from filters")
    return y


def make_multirate(epoch500):
    result = {}
    for fs in OUTPUT_SFREQS:
        if fs == SOURCE_SFREQ:
            x = epoch500
        else:
            g = np.gcd(fs, SOURCE_SFREQ)
            x = signal.resample_poly(epoch500, fs // g, SOURCE_SFREQ // g,
                                     axis=-1, padtype="line")
        required = 8 * fs
        if x.shape != (NUM_EEG_CHANNELS, required) or not np.isfinite(x).all():
            raise RuntimeError(f"Resampling invalid at {fs} Hz: {x.shape}")
        result[fs] = x.astype(np.float32)
    return result


def artifact_stats(epoch500):
    micro = np.asarray(epoch500, dtype=np.float64) * 1e6
    peak = float(np.max(np.abs(micro)))
    ptp = float(np.max(np.ptp(micro, axis=-1)))
    flat = int(np.sum(np.std(micro, axis=-1) < ARTIFACT_FLAT_STD_UV))
    flag = int(peak > ARTIFACT_PEAK_UV or ptp > ARTIFACT_PTP_UV or flat >= ARTIFACT_FLAT_CHANNELS)
    return flag, peak, ptp, flat


def trial_path(folder, run, cue):
    return folder / f"Sub{run['subject']}" / f"session{run['session']}" / cue["filename"]


def quality_cached(path, profile, cue, channels):
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as a:
        expected = (NUM_EEG_CHANNELS, profile.sfreq * 8)
        if ("X" not in a or a["X"].shape != expected
                or int(a["y"]) != cue["label"] or float(a["sfreq"]) != profile.sfreq
                or float(a["context_start_s"]) != 0.0 or float(a["context_end_s"]) != 8.0
                or str(a["pipeline_version"]) != PIPELINE_VERSION
                or list(a["ch_names"].astype(str)) != list(channels)):
            raise RuntimeError(f"Existing NPZ does not match the current specification: {path}")
        return (int(a["artifact_flag"]), float(a["artifact_peak_uv"]),
                float(a["artifact_ptp_uv"]), int(a["artifact_flat_channels"]))


def make_payload(x, profile, run, cue, channels, metrics, raw_root):
    flag, peak, ptp, flat = metrics
    feedback = cue["feedback_start_rel_s"]
    end = cue["end_trial_rel_s"]
    return {
        "X": x, "y": np.asarray(cue["label"], dtype=np.int64),
        "class_name": np.asarray(CLASS_NAMES[cue["label"]]), "class_names": np.asarray(CLASS_NAMES),
        "subject": np.asarray(run["subject"], dtype=np.int64),
        "session": np.asarray(run["session"], dtype=np.int64),
        "trial": np.asarray(cue["number"], dtype=np.int64),
        "run": np.asarray(int(run["run_bids"]) if run["run_bids"].isdigit() else 0, dtype=np.int64),
        "subject_bids": np.asarray(run["subject_bids"]), "session_bids": np.asarray(run["session_bids"]),
        "run_bids": np.asarray(run["run_bids"]), "task": np.asarray(run["task"]),
        "context": np.asarray("BH" if run["task"].endswith("BH") else "Graz"),
        "event_onset_s": np.asarray(cue["onset"], dtype=np.float64),
        "event_sample": np.asarray(cue["sample"], dtype=np.int64),
        "sfreq": np.asarray(profile.sfreq, dtype=np.float32),
        "original_sfreq": np.asarray(SOURCE_SFREQ, dtype=np.float32),
        "ch_names": np.asarray(channels),
        "context_start_s": np.asarray(0.0, dtype=np.float32),
        "context_end_s": np.asarray(8.0, dtype=np.float32),
        "task_start_s": np.asarray(0.0, dtype=np.float32),
        "task_end_s": np.asarray(feedback, dtype=np.float64),
        "feedback_start_s": np.asarray(feedback, dtype=np.float64),
        "feedback_end_s": np.asarray(end, dtype=np.float64),
        "feedback_start_rel_s": np.asarray(feedback, dtype=np.float64),
        "end_trial_rel_s": np.asarray(end, dtype=np.float64),
        "start_trial_rel_s": np.asarray(cue["start_trial_rel_s"], dtype=np.float64),
        "fixation_cross_rel_s": np.asarray(cue["fixation_cross_rel_s"], dtype=np.float64),
        "cue_duration_s": np.asarray(cue["cue_duration"], dtype=np.float64),
        "task_duration_s": np.asarray(feedback, dtype=np.float64),
        "feedback_duration_s": np.asarray(end - feedback, dtype=np.float64),
        "event_timing_valid": np.asarray(int(not cue["issues"]), dtype=np.int8),
        "timing_warnings": np.asarray(";".join(cue["warnings"])),
        "bandpass_hz": np.asarray(BANDPASS_HZ, dtype=np.float32),
        "notch_hz": np.asarray(NOTCH_HZ, dtype=np.float32),
        "reference": np.asarray("average"), "preprocessing_mode": np.asarray("window"),
        "artifact_flag": np.asarray(flag, dtype=np.int64),
        "auto_artifact_flag": np.asarray(flag, dtype=np.int64),
        "artifact_peak_uv": np.asarray(peak, dtype=np.float32),
        "artifact_ptp_uv": np.asarray(ptp, dtype=np.float32),
        "artifact_flat_channels": np.asarray(flat, dtype=np.int16),
        "signal_unit": np.asarray("V"),
        "source_file": np.asarray(run["edf"].relative_to(raw_root).as_posix()),
        "pipeline_version": np.asarray(PIPELINE_VERSION),
    }


def make_index(folder, profile, run, cue, metrics, raw_root):
    flag, peak, ptp, flat = metrics
    feedback = cue["feedback_start_rel_s"]
    end = cue["end_trial_rel_s"]
    return {"path": trial_path(folder, run, cue).relative_to(folder).as_posix(),
            "subject": run["subject"], "subject_bids": run["subject_bids"],
            "session": run["session"], "session_bids": run["session_bids"],
            "trial": cue["number"], "y": cue["label"], "class_name": CLASS_NAMES[cue["label"]],
            "run": int(run["run_bids"]) if run["run_bids"].isdigit() else 0,
            "run_bids": run["run_bids"], "task": run["task"],
            "context": "BH" if run["task"].endswith("BH") else "Graz",
            "event_onset_s": cue["onset"], "event_sample": cue["sample"],
            "sfreq": profile.sfreq, "n_channels": NUM_EEG_CHANNELS,
            "n_samples": profile.sfreq * 8, "window_start_s": 0.0, "window_end_s": 8.0,
            "artifact_flag": flag, "artifact_peak_uv": peak, "artifact_ptp_uv": ptp,
            "artifact_flat_channels": flat,
            "source_file": run["edf"].relative_to(raw_root).as_posix(),
            "feedback_start_rel_s": feedback, "end_trial_rel_s": end,
            "task_duration_s": feedback, "feedback_duration_s": end - feedback,
            "cue_duration_s": cue["cue_duration"],
            "event_timing_valid": int(not cue["issues"]),
            "timing_warnings": ";".join(cue["warnings"])}


def rejected(run, cue, reason, raw_root):
    return {"subject": run["subject"], "session": run["session"], "task": run["task"],
            "run_bids": run["run_bids"], "trial": cue["number"],
            "class_name": CLASS_NAMES[cue["label"]], "cue_onset_s": cue["onset"],
            "reason": reason, "source_file": run["edf"].relative_to(raw_root).as_posix()}


def run_pipeline(raw_root=RAW_ROOT, output_root=OUTPUT_ROOT, max_runs=MAX_RUNS, dry_run=False):
    raw_root, output_root = Path(raw_root), Path(output_root)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"RAW_ROOT does not exist: {raw_root}")
    runs = discover_runs(raw_root, max_runs)
    smap, sesmap = identify_subjects_and_sessions(runs)
    plans = make_plans(runs, smap, sesmap)
    total_cues = sum(len(r["cues"]) for r in plans)
    if not total_cues:
        raise RuntimeError("No mi/sub/word cues in supported TSV runs")
    print(f"Runs={len(runs)}  Cues={total_cues}  Profiles={len(PROFILES)}")
    for profile in PROFILES:
        print(f"  {profile.name}: X=({NUM_EEG_CHANNELS}, {8*profile.sfreq})")
    if dry_run:
        reasons = Counter(s for r in plans for c in r["cues"] for s in c["issues"])
        valid = sum(not c["issues"] for r in plans for c in r["cues"])
        print(f"Dry run: valid event patterns={valid}/{total_cues}, issues={dict(reasons)}")
        return

    first = mne.io.read_raw_edf(plans[0]["edf"], preload=False, verbose="ERROR")
    try:
        channels = eeg_names(first)
    finally:
        first.close()
    config = project_config(max_runs)
    config["raw_root"] = str(raw_root)
    fp = fingerprint(config, runs, channels)
    folders = {}
    for profile in PROFILES:
        folder = output_root / profile.name
        state = folder / "preprocessing_state.json"
        if state.exists():
            old = json.loads(state.read_text(encoding="utf-8"))
            if old.get("fingerprint") != fp or old.get("profile") != json.loads(json.dumps(asdict(profile))):
                raise RuntimeError(
                    f"Refusing to overwrite another preprocessing configuration: {folder}\n"
                    "Do not delete your historical profile; move the conflicting NEW profile "
                    "directory out of the way or choose another OUTPUT_ROOT.")
            if not RESUME_EXISTING:
                raise RuntimeError(f"Existing profile when RESUME_EXISTING=False: {folder}")
        elif folder.exists() and any(folder.rglob("*.npz")):
            raise RuntimeError(f"Existing NPZ without matching state; refusing overwrite: {folder}")
        folders[profile.sfreq] = folder
    # All folder conflict checks have finished before any new output is written.
    for profile in PROFILES:
        folder = folders[profile.sfreq]
        folder.mkdir(parents=True, exist_ok=True)
        for subject in range(1, SUBJECT_FOLDER_COUNT + 1):
            for ses in range(1, SESSION_FOLDER_COUNT + 1):
                (folder / f"Sub{subject}" / f"session{ses}").mkdir(parents=True, exist_ok=True)
        state = folder / "preprocessing_state.json"
        if not state.exists():
            atomic_json(state, {"fingerprint": fp, "profile": asdict(profile),
                                "configuration": config, "subject_map": smap,
                                "session_map": sesmap, "channels": channels,
                                "class_map": dict(zip(CLASS_NAMES, range(3)))})

    indices = {fs: [] for fs in OUTPUT_SFREQS}
    new_counts, reuse_counts, exclude = Counter(), Counter(), Counter()
    rejected_rows = []
    try:
        for run in tqdm(plans, desc="Read isolated 8-s EEG", unit="run"):
            if not run["cues"]:
                continue
            raw = mne.io.read_raw_edf(run["edf"], preload=False, verbose="ERROR")
            try:
                if not np.isclose(raw.info["sfreq"], SOURCE_SFREQ):
                    raise ValueError(f"Unexpected sfreq {raw.info['sfreq']}: {run['edf']}")
                eeg_names(raw, channels)
                for cue in run["cues"]:
                    if STRICT_COMPLETE_TRIALS and cue["issues"]:
                        reason = ";".join(cue["issues"])
                        exclude[reason] += 1
                        rejected_rows.append(rejected(run, cue, reason, raw_root))
                        continue
                    a = cue["sample"]
                    b = a + 8 * SOURCE_SFREQ
                    if a < 0 or b > raw.n_times:
                        reason = "out_of_recording"
                        exclude[reason] += 1
                        rejected_rows.append(rejected(run, cue, reason, raw_root))
                        continue
                    cached = {}
                    for p in PROFILES:
                        target = trial_path(folders[p.sfreq], run, cue)
                        cached[p.sfreq] = quality_cached(target, p, cue, channels) if RESUME_EXISTING else None
                    if all(v is not None for v in cached.values()):
                        for p in PROFILES:
                            indices[p.sfreq].append(make_index(folders[p.sfreq], p, run, cue,
                                                                cached[p.sfreq], raw_root))
                            reuse_counts[p.sfreq] += 1
                        continue
                    x = raw.get_data(picks=channels, start=a, stop=b).astype(np.float64)
                    if x.shape != (NUM_EEG_CHANNELS, 8 * SOURCE_SFREQ) or not np.isfinite(x).all():
                        reason = "invalid_eeg_samples"
                        exclude[reason] += 1
                        rejected_rows.append(rejected(run, cue, reason, raw_root))
                        continue
                    epoch500 = preprocess_full8(x)
                    outputs = make_multirate(epoch500)
                    metrics = artifact_stats(epoch500)
                    for p in PROFILES:
                        target = trial_path(folders[p.sfreq], run, cue)
                        if cached[p.sfreq] is None:
                            payload = make_payload(outputs[p.sfreq], p, run, cue, channels, metrics, raw_root)
                            atomic_npz(target, payload)
                            new_counts[p.sfreq] += 1
                            current_stats = metrics
                        else:
                            current_stats = cached[p.sfreq]
                            reuse_counts[p.sfreq] += 1
                        indices[p.sfreq].append(make_index(folders[p.sfreq], p, run, cue,
                                                            current_stats, raw_root))
            finally:
                raw.close()
            # A partial run remains resumable; indices are rebuilt on the next run.
            for p in PROFILES:
                atomic_csv(folders[p.sfreq] / "dataset_index.csv", indices[p.sfreq], INDEX_COLUMNS)
    finally:
        for p in PROFILES:
            atomic_csv(folders[p.sfreq] / "dataset_index.csv", indices[p.sfreq], INDEX_COLUMNS)
        atomic_csv(output_root / "full8_excluded_trials.csv", rejected_rows, REJECT_COLUMNS)

    for p in PROFILES:
        rows = indices[p.sfreq]
        counts = Counter(row["class_name"] for row in rows)
        sessions = Counter(f"Sub{r['subject']}/session{r['session']}" for r in rows)
        manifest = {"fingerprint": fp, "profile": asdict(p), "configuration": config,
                    "subject_map": smap, "session_map": sesmap, "channels": channels,
                    "total_trials": len(rows), "class_counts": dict(counts),
                    "session_counts": dict(sorted(sessions.items(), key=lambda t: natural_key(t[0]))),
                    "artifact_flagged": sum(int(r["artifact_flag"]) for r in rows),
                    "newly_written": new_counts[p.sfreq], "reused_existing": reuse_counts[p.sfreq],
                    "excluded_global": dict(exclude), "input_unit": "V",
                    "splits_defined": False, "complete": True,
                    "trial_layout": "Sub{subject}/session{session}/trial{number}_run{run}_{task}_{class}.npz"}
        atomic_json(folders[p.sfreq] / "preprocessing_manifest.json", manifest)
        print(f"{p.name}: trials={len(rows)}, new={new_counts[p.sfreq]}, "
              f"reused={reuse_counts[p.sfreq]}, classes={dict(counts)}")
    atomic_json(output_root / "full8_dataset_summary.json",
                {"pipeline_version": PIPELINE_VERSION, "source_runs": len(runs),
                 "source_cues": total_cues, "strict_complete_trials": STRICT_COMPLETE_TRIALS,
                 "profiles": [p.name for p in PROFILES], "excluded": dict(exclude),
                 "rejected_rows": len(rejected_rows), "splits_defined": False})
    print(f"Done. Output root: {output_root}")


def self_test():
    """Synthetic signal and real-format TSV tests; NO download needed."""
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        tsv = tmp / "example_events.tsv"
        tsv.write_text("onset\tduration\ttrial_type\tvalue\tsample\n"
                       "5.952\t0.002\tstart_trial\t768\t2976\n"
                       "6.054\t0.002\tfixation_cross\t786\t3027\n"
                       "6.992\t8.0\tmi\t33024\t3496\n"
                       "10.006\t0.002\tstart_feedback\t781\t5003\n"
                       "14.992\t0.002\tend_trial\t800\t7496\n"
                       "16.710\t0.002\tstart_trial\t768\t8355\n"
                       "16.772\t0.002\tfixation_cross\t786\t8386\n"
                       "17.694\t8.0\tsub\t33025\t8847\n"
                       "20.678\t0.002\tstart_feedback\t781\t10339\n"
                       "25.734\t0.002\tend_trial\t800\t12867\n", encoding="utf-8")
        run = {"edf": tmp / "sub-P01_ses-01_task-AcquisitionBH_run-1_eeg.edf",
               "events": tsv, "subject_bids": "P01", "session_bids": "01",
               "run_bids": "1", "task": "AcquisitionBH"}
        plan = make_plans([run], {"P01": 1}, {"P01": {"01": 1}})[0]
        c1, c2 = plan["cues"]
        assert not c1["issues"] and not c2["issues"], (c1["issues"], c2["issues"])
        assert c1["label"] == 2 and c2["label"] == 0
        assert abs(c1["feedback_start_rel_s"] - 3.014) < 1e-9
        assert abs(c2["feedback_start_rel_s"] - 2.984) < 1e-9
        assert c1["sample"] == 3496 and c2["sample"] == 8847
        t = np.arange(SOURCE_SFREQ * 8) / SOURCE_SFREQ
        x = np.array([1e-6 * (np.sin(2*np.pi*10*t+0.13*i) +
                                    0.05*np.sin(2*np.pi*50*t)) for i in range(41)])
        processed = preprocess_full8(x)
        assert processed.shape == (41, 4000)
        assert np.max(np.abs(processed.mean(axis=0))) < 1e-17
        multi = make_multirate(processed)
        for hz, expected in ((500, 4000), (250, 2000), (120, 960)):
            assert multi[hz].shape == (41, expected)
            assert multi[hz].dtype == np.float32
            assert np.isfinite(multi[hz]).all()
        stats = artifact_stats(processed)
        for p in PROFILES:
            pack = make_payload(multi[p.sfreq], p, plan, c1,
                                [f"EEG{i}" for i in range(41)], stats, tmp)
            assert pack["X"].shape == (41, 8*p.sfreq)
            assert int(pack["y"]) == 2
            assert abs(float(pack["task_end_s"]) - 3.014) < 1e-9
            file = tmp / f"unit_{p.sfreq}.npz"
            atomic_npz(file, pack)
            assert quality_cached(file, p, c1, [f"EEG{i}" for i in range(41)]) is not None
        print("Self-test PASSED: TSV pairing, 8-s boundaries, 3 output rates, NPZ metadata")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw-root", type=Path, default=RAW_ROOT, help="Original PROTEUS dataset root")
    ap.add_argument("--output-root", type=Path, default=None, help="NPZ root (defaults to RAW_ROOT/prepared_data/PROTEUS_Competition_TrialNPZ)")
    ap.add_argument("--max-runs", type=int, default=MAX_RUNS, help="Only first N runs; 0 means all")
    ap.add_argument("--dry-run", action="store_true", help="Validate TSV without opening EEG / writing files")
    ap.add_argument("--self-test", action="store_true", help="Fast in-memory signal and TSV checks")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if args.max_runs < 0:
        ap.error("--max-runs must be nonnegative")
    output = args.output_root or args.raw_root / "prepared_data" / "PROTEUS_Competition_TrialNPZ"
    run_pipeline(args.raw_root, output, args.max_runs, args.dry_run)


if __name__ == "__main__":
    main()
