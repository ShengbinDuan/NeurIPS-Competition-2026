from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


CLASS_NAMES = ("SUB", "WORD", "MI")


@dataclass(frozen=True)
class Window:
    start: float
    end: float

    @classmethod
    def parse(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, (tuple, list)) and len(value) == 2:
            start, end = map(float, value)
        else:
            parts = str(value).replace(",", ":").split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid window {value!r}; use START:END")
            start, end = map(float, parts)
        if not np.isfinite([start, end]).all() or start >= end:
            raise ValueError(f"Invalid time interval: {start}, {end}")
        return cls(start, end)

    @property
    def name(self):
        def fmt(x):
            return format(x, ".9g").replace("-", "m").replace(".", "p")
        return f"Cue{fmt(self.start)}to{fmt(self.end)}"

    def indices(self, sfreq, source_start, source_end):
        if self.start < source_start - 1e-7 or self.end > source_end + 1e-7:
            raise ValueError(f"Window [{self.start},{self.end}) outside [{source_start},{source_end})")
        a = (self.start - source_start) * sfreq
        b = (self.end - source_start) * sfreq
        ia, ib = int(round(a)), int(round(b))
        if not np.isclose(a, ia, atol=1e-6, rtol=0) or not np.isclose(b, ib, atol=1e-6, rtol=0):
            raise ValueError(f"Window [{self.start},{self.end}) not sample-aligned at {sfreq} Hz")
        if ia < 0 or ib <= ia:
            raise ValueError("Empty or invalid sample interval")
        return ia, ib


@dataclass(frozen=True)
class ProfileIndex:
    root: Path
    rows: tuple[dict, ...]
    index_sha256: str
    preprocessing_sha256: str | None
    preprocessing_fingerprint: str | None


@dataclass
class PreparedSplit:
    path: Path
    y: np.ndarray
    meta: dict
    ch_names: tuple[str, ...]
    sfreq: float
    n_times: int
    signal_unit: str
    rms_microvolts: float

    @property
    def n_trials(self):
        return len(self.y)

    @property
    def n_channels(self):
        return len(self.ch_names)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_profiles(data_root, names=None):
    root = Path(data_root).expanduser().resolve()
    if names and not (len(names) == 1 and names[0].lower() == "all"):
        profiles = [(Path(name) if Path(name).is_absolute() else root / name).resolve() for name in names]
    else:
        profiles = sorted(p for p in root.iterdir() if p.is_dir() and (p / "dataset_index.csv").is_file()) if root.is_dir() else []
    if not profiles:
        raise FileNotFoundError(f"No NPZ profiles found under {root}")
    if len(set(profiles)) != len(profiles):
        raise ValueError("Duplicate profile paths")
    for profile in profiles:
        if not (profile / "dataset_index.csv").is_file():
            raise FileNotFoundError(profile / "dataset_index.csv")
    return profiles


def read_index(profile_root):
    root = Path(profile_root).expanduser().resolve()
    index_file = root / "dataset_index.csv"
    with index_file.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Empty dataset index: {index_file}")
    required = {"path", "y", "subject", "session"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Missing index columns: {sorted(required - set(rows[0]))}")
    seen = set()
    for index, row in enumerate(rows):
        trial_path = (root / row["path"]).resolve()
        if not trial_path.is_relative_to(root) or not trial_path.is_file():
            raise FileNotFoundError(f"Invalid NPZ at index row {index + 2}: {trial_path}")
        if row["path"] in seen:
            raise ValueError(f"Duplicate NPZ in index: {row['path']}")
        seen.add(row["path"])
        if int(row["y"]) not in range(len(CLASS_NAMES)):
            raise ValueError(f"Unknown class index in {row['path']}")
        row["absolute_path"] = str(trial_path)
        row["context"] = row.get("context") or _infer_context(row.get("task", ""))
    rows.sort(key=lambda r: (int(r["subject"]), int(r["session"]),
                             int(r.get("run") or 0), float(r.get("event_onset_s") or 0),
                             int(r.get("trial") or 0), r["path"]))
    manifest_file = root / "preprocessing_manifest.json"
    if manifest_file.is_file():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        if manifest.get("complete") is False:
            raise RuntimeError(f"Preprocessing not completed: {manifest_file}")
    state_file = root / "preprocessing_state.json"
    fingerprint = None
    if state_file.is_file():
        fingerprint = json.loads(state_file.read_text(encoding="utf-8")).get("fingerprint")
    return ProfileIndex(root, tuple(rows), sha256_file(index_file),
                        sha256_file(state_file) if state_file.is_file() else None, fingerprint)


def _infer_context(task):
    if str(task).endswith("BH"):
        return "BH"
    if str(task).endswith("Graz"):
        return "Graz"
    return "unknown"


def select_rows(index, sessions=None, subjects=None, contexts=None, exclude_artifacts=False):
    sessions = None if sessions is None else {int(x) for x in sessions}
    subjects = None if subjects is None else {int(x) for x in subjects}
    contexts = None if contexts is None else set(contexts)
    selected = []
    for row in index.rows:
        if sessions is not None and int(row["session"]) not in sessions:
            continue
        if subjects is not None and int(row["subject"]) not in subjects:
            continue
        if contexts is not None and row["context"] not in contexts:
            continue
        if exclude_artifacts and int(row.get("artifact_flag") or 0):
            continue
        selected.append(row)
    if not selected:
        raise ValueError(f"No trials selected in {index.root}")
    return selected


def row_metadata(rows):
    return {
        **{field: np.array([int(row.get(field) or 0) for row in rows], dtype=np.int64)
           for field in ("subject", "session", "run", "trial")},
        "context": np.array([row["context"] for row in rows]),
        "path": np.array([row["path"] for row in rows]),
    }


def inspect_trial(row, window, field="X", expected=None):
    window = Window.parse(window)
    with np.load(row["absolute_path"], allow_pickle=False) as archive:
        if field not in archive:
            raise KeyError(f"Missing {field}: {row['absolute_path']}")
        source = archive[field]
        if source.ndim != 2:
            raise ValueError(f"Expected [channels,time]: {row['absolute_path']}, {source.shape}")
        sfreq = float(archive["sfreq"])
        start = float(archive["context_start_s"]) if "context_start_s" in archive else float(row.get("window_start_s") or 0)
        end = float(archive["context_end_s"]) if "context_end_s" in archive else float(row.get("window_end_s") or start + source.shape[1] / sfreq)
        unit = str(archive["signal_unit"]) if "signal_unit" in archive else "V"
        channels = tuple(str(x) for x in archive["ch_names"]) if "ch_names" in archive else tuple(f"Ch{i}" for i in range(source.shape[0]))
        if not np.isfinite(sfreq) or sfreq <= 0 or unit not in {"V", "uV", "µV"}:
            raise ValueError(f"Unsupported sample frequency or unit in {row['absolute_path']}")
        if len(channels) != source.shape[0] or not np.isclose(source.shape[1], (end - start) * sfreq, atol=1e-3):
            raise ValueError(f"Invalid channel names or time metadata: {row['absolute_path']}")
        if "y" in archive and int(archive["y"]) != int(row["y"]):
            raise ValueError(f"CSV/NPZ label mismatch: {row['absolute_path']}")
        if row.get("sfreq") and not np.isclose(float(row["sfreq"]), sfreq):
            raise ValueError(f"CSV/NPZ sampling rate mismatch: {row['absolute_path']}")
        if row.get("window_start_s") and not np.isclose(float(row["window_start_s"]), start):
            raise ValueError(f"CSV/NPZ start mismatch: {row['absolute_path']}")
        if row.get("window_end_s") and not np.isclose(float(row["window_end_s"]), end):
            raise ValueError(f"CSV/NPZ end mismatch: {row['absolute_path']}")
        a, b = window.indices(sfreq, start, end)
        sample = np.asarray(source[:, a:b], dtype=np.float32)
    if not np.isfinite(sample).all():
        raise ValueError(f"Non-finite signal: {row['absolute_path']}")
    schema = (channels, sfreq, unit, sample.shape)
    if expected is not None and schema != expected:
        raise ValueError(f"Incompatible trial schema: {row['absolute_path']}: {schema} != {expected}")
    return sample, schema


def materialize_split(rows, window, destination, field="X"):
    if not rows:
        raise ValueError("Cannot materialize an empty split")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    first, schema = inspect_trial(rows[0], window, field)
    shape = (len(rows), *first.shape)
    mapped = np.lib.format.open_memmap(destination, mode="w+", dtype=np.float32, shape=shape)
    total_squares = 0.0
    for i, row in enumerate(rows):
        sample = first if i == 0 else inspect_trial(row, window, field, schema)[0]
        mapped[i] = sample
        total_squares += float(np.square(sample.astype(np.float64)).sum())
    mapped.flush()
    del mapped
    channels, sfreq, unit, signal_shape = schema
    scale = 1e6 if unit == "V" else 1.0
    rms = max(float(np.sqrt(total_squares / (len(rows) * np.prod(signal_shape))) * scale), 1e-4)
    return PreparedSplit(destination, np.array([int(row["y"]) for row in rows], dtype=np.int64),
                         row_metadata(rows), channels, sfreq, signal_shape[1], unit, rms)


class MappedTrialDataset(Dataset):
    def __init__(self, split, rms_microvolts, clip=8.0):
        self.data = np.load(split.path, mmap_mode="r")
        self.y = split.y
        self.scale = np.float32((1e6 if split.signal_unit == "V" else 1.0) / rms_microvolts)
        self.clip = float(clip)
        if not np.isfinite(self.scale) or self.clip <= 0:
            raise ValueError("Invalid normalization")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        signal = np.asarray(self.data[index], dtype=np.float32) * self.scale
        signal = np.clip(signal, -self.clip, self.clip)
        return torch.from_numpy(np.ascontiguousarray(signal)), int(self.y[index])


def read_trial(path, window, field="X"):
    row = {"absolute_path": str(Path(path).resolve()), "y": 0}
    with np.load(path, allow_pickle=False) as archive:
        row["y"] = int(archive["y"])
    signal, _ = inspect_trial(row, window, field)
    return signal, row["y"]
