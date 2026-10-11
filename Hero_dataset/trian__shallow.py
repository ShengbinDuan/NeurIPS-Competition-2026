from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import csv
import hashlib
import importlib.metadata
import inspect
import json
import platform
import random
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from torch import nn
from torch.utils.data import DataLoader

from dataloader import (
    CLASS_NAMES, MappedTrialDataset, Window, discover_profiles, inspect_trial,
    materialize_split, read_index, select_rows, sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = PROJECT_ROOT / "dataset" / "prepared_data" / "PROTEUS_Competition_TrialNPZ"
OUTPUT_ROOT = PROJECT_ROOT / "Weight" / "ShallowFBCSPNet_Full8_WindowSweep"
PROFILES = None
WINDOWS = (
    (0.0, 2.0),
    (0.0, 3.0),
    (0.0, 4.0),
    (0.0, 8.0),
    (0.5, 2.5),
    (0.5, 3.5),
    (0.5, 4.5),
    (1.0, 3.0),
    (1.0, 5.0),
    (1.0, 8.0),
    (2.0, 4.0),
    (2.0, 6.0),
    (3.0, 5.0),
    (3.0, 8.0),
    (4.0, 8.0),
)
TRAIN_SESSIONS = (1,)
VAL_SESSIONS = (2,)
TEST_SESSIONS = (3,)
SUBJECTS = None
CONTEXTS = None
EXCLUDE_ARTIFACTS = False
SIGNAL_FIELD = "X"
SELECTION_METRIC = "context_bacc"
REQUIRE_MATCHED_PROFILES = True
EVALUATE_WINNER_ON_TEST = True
CONTINUE_ON_ERROR = False
MAX_EPOCHS = 35
MIN_EPOCHS = 6
PATIENCE = 9
BATCH_SIZE = 64
PRED_BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.001
LABEL_SMOOTHING = 0.03
CLIP = 8.0
SEED = 2026
TORCH_THREADS = 4
NUM_WORKERS = 0
DEVICE = "auto"
USE_AMP = True
DETERMINISTIC = True


@dataclass(frozen=True)
class Settings:
    data_root: Path
    output_root: Path
    profiles: tuple[str, ...] | None
    windows: tuple[Window, ...]
    train_sessions: tuple[int, ...]
    val_sessions: tuple[int, ...]
    test_sessions: tuple[int, ...]
    subjects: tuple[int, ...] | None
    contexts: tuple[str, ...] | None
    exclude_artifacts: bool
    field: str
    selection_metric: str
    require_matched_profiles: bool
    evaluate_winner_on_test: bool
    continue_on_error: bool
    max_epochs: int
    min_epochs: int
    patience: int
    batch_size: int
    pred_batch_size: int
    learning_rate: float
    weight_decay: float
    label_smoothing: float
    clip: float
    seed: int
    threads: int
    workers: int
    device: str
    use_amp: bool
    deterministic: bool


def settings():
    config = Settings(
        data_root=Path(DATA_ROOT).expanduser().resolve(),
        output_root=Path(OUTPUT_ROOT).expanduser().resolve(),
        profiles=None if PROFILES is None else tuple(PROFILES),
        windows=tuple(dict.fromkeys(Window.parse(w) for w in WINDOWS)),
        train_sessions=tuple(TRAIN_SESSIONS),
        val_sessions=tuple(VAL_SESSIONS),
        test_sessions=tuple(TEST_SESSIONS),
        subjects=None if SUBJECTS is None else tuple(SUBJECTS),
        contexts=None if CONTEXTS is None else tuple(CONTEXTS),
        exclude_artifacts=EXCLUDE_ARTIFACTS,
        field=SIGNAL_FIELD,
        selection_metric=SELECTION_METRIC,
        require_matched_profiles=REQUIRE_MATCHED_PROFILES,
        evaluate_winner_on_test=EVALUATE_WINNER_ON_TEST,
        continue_on_error=CONTINUE_ON_ERROR,
        max_epochs=MAX_EPOCHS,
        min_epochs=MIN_EPOCHS,
        patience=PATIENCE,
        batch_size=BATCH_SIZE,
        pred_batch_size=PRED_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        label_smoothing=LABEL_SMOOTHING,
        clip=CLIP,
        seed=SEED,
        threads=TORCH_THREADS,
        workers=NUM_WORKERS,
        device=DEVICE,
        use_amp=USE_AMP,
        deterministic=DETERMINISTIC,
    )
    if not config.windows or config.selection_metric not in {"context_bacc", "session_bacc", "trial_bacc"}:
        raise ValueError("Invalid windows or selection metric")
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("Invalid device")
    if not (0 < config.min_epochs <= config.max_epochs) or config.patience < 1:
        raise ValueError("Invalid epoch configuration")
    if min(config.batch_size, config.pred_batch_size, config.threads) <= 0 or config.workers < 0:
        raise ValueError("Invalid batch/thread configuration")
    if config.learning_rate <= 0 or config.weight_decay < 0 or not 0 <= config.label_smoothing < 1:
        raise ValueError("Invalid optimization configuration")
    if config.clip <= 0 or not config.field:
        raise ValueError("Invalid preprocessing configuration")
    groups = [set(config.train_sessions), set(config.val_sessions), set(config.test_sessions)]
    if any(not group for group in groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("Train/validation/test sessions must be nonempty and disjoint")
    return config


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".writing")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_csv(path, records):
    path = Path(path)
    records = list(records)
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for record in records for key in record))
    temporary = path.with_name(path.name + ".writing")
    try:
        with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_npz(destination, **arrays):
    destination = Path(destination)
    temporary = destination.with_name(destination.name + ".writing")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def seed_all(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic)


def resolve_device(config):
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    use_cuda = config.device == "cuda" or (config.device == "auto" and torch.cuda.is_available())
    return torch.device("cuda" if use_cuda else "cpu")


def versions():
    try:
        braindecode_version = importlib.metadata.version("braindecode")
    except importlib.metadata.PackageNotFoundError:
        braindecode_version = None
    return {
        "python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__,
        "sklearn": sklearn.__version__, "braindecode": braindecode_version,
        "platform": platform.platform(), "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
    }


def row_identity(row):
    return (
        row.get("subject_bids") or row["subject"],
        row.get("session_bids") or row["session"],
        row.get("source_file") or row.get("task") or "",
        row.get("event_sample") or row.get("event_onset_s") or row.get("trial") or "",
    )


def verify_matched_profiles(indexes, config):
    if not config.require_matched_profiles or len(indexes) < 2:
        return
    reference = None
    for index in indexes:
        rows = select_rows(index, sessions=set(config.train_sessions + config.val_sessions + config.test_sessions),
                           subjects=config.subjects, contexts=config.contexts,
                           exclude_artifacts=config.exclude_artifacts)
        current = {}
        for row in rows:
            key = row_identity(row)
            if key in current:
                raise ValueError(f"Duplicate trial identity in {index.root}: {key}")
            current[key] = int(row["y"])
        if reference is None:
            reference = current
        elif current != reference:
            missing = len(reference.keys() - current.keys())
            additional = len(current.keys() - reference.keys())
            changed = sum(current[k] != reference[k] for k in current.keys() & reference.keys())
            raise ValueError(f"Profiles do not contain identical trials: {index.root.name}, missing={missing}, additional={additional}, changed_labels={changed}")


def select_splits(index, config, include_test=False):
    shared = {"subjects": config.subjects, "contexts": config.contexts,
              "exclude_artifacts": config.exclude_artifacts}
    result = {
        "train": select_rows(index, sessions=config.train_sessions, **shared),
        "val": select_rows(index, sessions=config.val_sessions, **shared),
    }
    if include_test:
        result["test"] = select_rows(index, sessions=config.test_sessions, **shared)
    for name, rows in result.items():
        for subject, session in sorted({(int(r["subject"]), int(r["session"])) for r in rows}):
            classes = {int(r["y"]) for r in rows if int(r["subject"]) == subject and int(r["session"]) == session}
            if classes != set(range(len(CLASS_NAMES))):
                raise ValueError(f"{name} subject={subject}, session={session} missing class(es): {classes}")
    return result


def model_for(n_channels, n_times, sfreq, device):
    try:
        from braindecode.models import ShallowFBCSPNet
    except ImportError as error:
        raise ImportError("Install braindecode==1.2.0 in the training environment") from error
    arguments = {"n_chans": n_channels, "n_times": n_times,
                 "n_outputs": len(CLASS_NAMES), "sfreq": sfreq, "final_conv_length": "auto"}
    try:
        model = ShallowFBCSPNet(**arguments)
    except TypeError:
        arguments.pop("sfreq")
        model = ShallowFBCSPNet(**arguments)
    return model.to(device)


def logits_of(output):
    if isinstance(output, dict):
        output = output.get("logits", next(iter(output.values())))
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output.ndim == 3 and output.shape[-1] == 1:
        output = output.squeeze(-1)
    if output.ndim != 2 or output.shape[1] != len(CLASS_NAMES):
        raise ValueError(f"Unexpected model output: {tuple(output.shape)}")
    return output


def grouped_metrics(y, probs, meta):
    y = np.asarray(y, dtype=np.int64)
    p = np.asarray(probs, dtype=np.float64)
    if p.shape != (len(y), len(CLASS_NAMES)) or not np.isfinite(p).all():
        raise ValueError("Invalid prediction probabilities")
    p = np.maximum(p, 1e-8)
    p /= p.sum(axis=1, keepdims=True)
    prediction = p.argmax(axis=1)
    ctx_cells = []
    for subject, session, context in sorted(set(zip(meta["subject"].tolist(), meta["session"].tolist(), meta["context"].tolist()))):
        mask = (meta["subject"] == subject) & (meta["session"] == session) & (meta["context"] == context)
        present = np.unique(y[mask])
        ctx_cells.append({"subject": int(subject), "session": int(session), "context": str(context),
                          "n": int(mask.sum()), "n_classes": int(len(present)),
                          "balanced_accuracy": float(balanced_accuracy_score(y[mask], prediction[mask]))})
    session_cells = []
    for subject, session in sorted(set(zip(meta["subject"].tolist(), meta["session"].tolist()))):
        mask = (meta["subject"] == subject) & (meta["session"] == session)
        session_cells.append({"subject": int(subject), "session": int(session), "context": "ALL",
                              "n": int(mask.sum()), "balanced_accuracy": float(balanced_accuracy_score(y[mask], prediction[mask]))})
    return {
        "context_bacc": float(np.mean([record["balanced_accuracy"] for record in ctx_cells])),
        "session_bacc": float(np.mean([record["balanced_accuracy"] for record in session_cells])),
        "trial_bacc": float(balanced_accuracy_score(y, prediction)),
        "accuracy": float(accuracy_score(y, prediction)),
        "logloss": float(log_loss(y, p, labels=list(range(len(CLASS_NAMES))))),
    }, ctx_cells, session_cells


def predict(model, split, rms, config, device):
    data = MappedTrialDataset(split, rms, config.clip)
    loader = DataLoader(data, batch_size=config.pred_batch_size, shuffle=False,
                        num_workers=config.workers, pin_memory=device.type == "cuda")
    model.eval()
    probabilities = []
    with torch.inference_mode():
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type,
                                enabled=config.use_amp and device.type == "cuda"):
                output = logits_of(model(x))
            probabilities.append(torch.softmax(output.float(), dim=1).cpu().numpy())
    return np.concatenate(probabilities).astype(np.float32)


def check_compatible(splits):
    first = next(iter(splits.values()))
    for name, split in splits.items():
        if (split.n_channels, split.n_times, split.sfreq, split.ch_names, split.signal_unit) != (
                first.n_channels, first.n_times, first.sfreq, first.ch_names, first.signal_unit):
            raise ValueError(f"Incompatible trial shapes, channel order, or units in {name}")


def trainer_core_hash():
    parts = (seed_all, select_splits, model_for, logits_of, grouped_metrics,
             predict, check_compatible, model_checkpoint, load_model, fit, evaluate)
    source = "\n".join(inspect.getsource(function) for function in parts)
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def training_record(index, window, rows, config, device):
    _, schema = inspect_trial(rows["train"][0], window, config.field)
    for name in rows:
        inspect_trial(rows[name][0], window, config.field, schema)
    channels, sfreq, unit, shape = schema
    return {
        "model": "ShallowFBCSPNet", "class_names": list(CLASS_NAMES),
        "profile": index.root.name, "profile_root": str(index.root),
        "preprocessing_fingerprint": index.preprocessing_fingerprint,
        "preprocessing_sha256": index.preprocessing_sha256,
        "dataset_index_sha256": index.index_sha256,
        "window": [window.start, window.end], "field": config.field,
        "n_channels": int(shape[0]), "n_times": int(shape[1]),
        "sfreq": float(sfreq), "signal_unit": unit, "ch_names": list(channels),
        "train_sessions": config.train_sessions, "val_sessions": config.val_sessions,
        "test_sessions": config.test_sessions, "subjects": config.subjects,
        "contexts": config.contexts, "exclude_artifacts": config.exclude_artifacts,
        "n_train": len(rows["train"]), "n_val": len(rows["val"]),
        "split_hashes": {name: hashlib.sha256(json.dumps(
            [(row["path"], int(row["y"])) for row in selected],
            ensure_ascii=False).encode("utf-8")).hexdigest() for name, selected in rows.items()},
        "selection_metric": config.selection_metric,
        "max_epochs": config.max_epochs, "min_epochs": config.min_epochs,
        "patience": config.patience, "batch_size": config.batch_size,
        "pred_batch_size": config.pred_batch_size, "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay, "label_smoothing": config.label_smoothing,
        "clip": config.clip, "seed": config.seed, "threads": config.threads,
        "workers": config.workers, "device": device.type,
        "use_amp": config.use_amp and device.type == "cuda", "deterministic": config.deterministic,
        "trainer_core_sha256": trainer_core_hash(),
        "dataloader_sha256": sha256_file(Path(__file__).with_name("dataloader.py")),
        "versions": versions(),
    }


def run_folder(index, window, record, config):
    signature = hashlib.sha256(json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return config.output_root / index.root.name / window.name / f"run_{signature}"


def model_checkpoint(model, split, epoch):
    return {
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_name": "ShallowFBCSPNet", "n_chans": split.n_channels,
        "n_times": split.n_times, "sfreq": split.sfreq, "ch_names": list(split.ch_names),
        "epoch": int(epoch), "class_names": list(CLASS_NAMES),
    }


def load_model(path, split, device):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if (int(checkpoint["n_chans"]), int(checkpoint["n_times"]), float(checkpoint["sfreq"]),
            tuple(checkpoint["ch_names"])) != (
            split.n_channels, split.n_times, split.sfreq, split.ch_names):
        raise ValueError(f"Checkpoint shape or channel mismatch: {path}")
    model = model_for(split.n_channels, split.n_times, split.sfreq, device)
    model.load_state_dict(checkpoint["state_dict"])
    return model, checkpoint


def fit(train, val, config, folder, device):
    seed_all(config.seed, config.deterministic)
    model = model_for(train.n_channels, train.n_times, train.sfreq, device)
    rms = train.rms_microvolts
    dataset = MappedTrialDataset(train, rms, config.clip)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True,
                        num_workers=config.workers, pin_memory=device.type == "cuda",
                        generator=torch.Generator().manual_seed(config.seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    counts = np.bincount(train.y, minlength=len(CLASS_NAMES)).astype(np.float64)
    weights = np.minimum(len(train.y) / (len(CLASS_NAMES) * np.maximum(counts, 1)), 3.0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device),
                                    label_smoothing=config.label_smoothing)
    amp = config.use_amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best = (-1.0, -1.0, -float("inf"))
    best_epoch = 0
    misses = 0
    history = []
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        loss_total = 0.0
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp):
                loss = criterion(logits_of(model(x)).float(), y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            scaler.step(optimizer)
            scaler.update()
            loss_total += float(loss.detach().item()) * len(y)
        scheduler.step()
        probabilities = predict(model, val, rms, config, device)
        metrics, _, _ = grouped_metrics(val.y, probabilities, val.meta)
        secondary = "session_bacc" if config.selection_metric != "session_bacc" else "trial_bacc"
        ranking = (metrics[config.selection_metric], metrics[secondary], -metrics["logloss"])
        if ranking > best:
            best = ranking
            best_epoch = epoch
            misses = 0
            target = folder / "network.pt"
            temporary = target.with_name(target.name + ".writing")
            torch.save(model_checkpoint(model, train, epoch), temporary)
            os.replace(temporary, target)
        else:
            misses += 1
        history.append({"epoch": epoch, "train_loss": loss_total / len(train.y),
                        **{f"val_{key}": value for key, value in metrics.items()}})
        if epoch == 1 or epoch % 5 == 0 or misses == config.patience:
            print(f"      epoch {epoch:02d} loss={history[-1]['train_loss']:.4f} "
                  f"val_{config.selection_metric}={metrics[config.selection_metric]:.4%} "
                  f"best={best[0]:.4%}", flush=True)
        if epoch >= config.min_epochs and misses >= config.patience:
            break
    save_csv(folder / "history.csv", history)
    del model
    return {"best_epoch": best_epoch, "epochs_used": len(history)}


def evaluate(model, split, rms, config, device, folder, prefix):
    probabilities = predict(model, split, rms, config, device)
    scores, context_cells, session_cells = grouped_metrics(split.y, probabilities, split.meta)
    save_npz(folder / f"{prefix}_probs.npz", probs=probabilities, y=split.y, **split.meta)
    save_csv(folder / f"{prefix}_context_scores.csv", context_cells)
    save_csv(folder / f"{prefix}_session_scores.csv", session_cells)
    return scores


def one_experiment(index, window, config, device):
    rows = select_splits(index, config)
    record = training_record(index, window, rows, config, device)
    folder = run_folder(index, window, record, config)
    config_file = folder / "config.json"
    result_file = folder / "result.json"
    if config_file.is_file() and json.loads(config_file.read_text(encoding="utf-8")) != json.loads(json.dumps(record)):
        raise RuntimeError(f"Existing run has a different configuration: {folder}")
    if config_file.is_file() and result_file.is_file() and (folder / "network.pt").is_file() and (folder / "val_probs.npz").is_file():
        result = json.loads(result_file.read_text(encoding="utf-8"))
        print(f"    Reused: {index.root.name} {window.name} {result['val'][config.selection_metric]:.4%}", flush=True)
        return result
    folder.mkdir(parents=True, exist_ok=True)
    save_json(config_file, record)
    print(f"    Training {index.root.name} {window.name}: {record['n_train']} train, "
          f"{record['n_val']} validation, {record['n_times']} samples", flush=True)
    with tempfile.TemporaryDirectory(prefix="shallow_window_", dir=config.output_root) as tmp:
        splits = {name: materialize_split(selected, window, Path(tmp) / f"{name}.npy", config.field)
                  for name, selected in rows.items()}
        check_compatible(splits)
        train, val = splits["train"], splits["val"]
        fit_result = fit(train, val, config, folder, device)
        model, checkpoint = load_model(folder / "network.pt", train, device)
        rms = train.rms_microvolts
        save_json(folder / "scaler.json", {"rms_microvolts": rms, "clip": config.clip,
                                           "signal_unit": train.signal_unit,
                                           "scale_to_microvolts": 1e6 if train.signal_unit == "V" else 1.0})
        scores = evaluate(model, val, rms, config, device, folder, "val")
        result = {"profile": index.root.name, "profile_root": str(index.root),
                  "window": [window.start, window.end], "sfreq": train.sfreq,
                  "n_channels": train.n_channels, "n_times": train.n_times,
                  "n_train": train.n_trials, "n_val": val.n_trials,
                  "best_epoch": checkpoint["epoch"], "epochs_used": fit_result["epochs_used"],
                  "selection_metric": config.selection_metric, "val": scores,
                  "run_folder": str(folder), "test": None}
        save_json(result_file, result)
        del model
    print(f"      val_context={scores['context_bacc']:.4%} "
          f"val_session={scores['session_bacc']:.4%}", flush=True)
    return result


def summary_row(result):
    row = {
        "profile": result["profile"], "sfreq": result["sfreq"],
        "start_s": result["window"][0], "end_s": result["window"][1],
        "duration_s": result["window"][1] - result["window"][0],
        "n_times": result["n_times"], "n_train": result["n_train"],
        "n_val": result["n_val"], "best_epoch": result["best_epoch"],
        "checkpoint": str(Path(result["run_folder"]) / "network.pt"),
    }
    row.update({f"val_{key}": value for key, value in result["val"].items()})
    if result.get("test") is not None:
        row.update({f"test_{key}": value for key, value in result["test"].items()})
    return row


def write_figures(results, config):
    if not results:
        return
    rows = [summary_row(result) for result in results]
    frequencies = sorted(set(float(row["sfreq"]) for row in rows))
    windows = list(config.windows)
    data = np.full((len(frequencies), len(windows)), np.nan, dtype=float)
    for row in rows:
        i = frequencies.index(float(row["sfreq"]))
        j = windows.index(Window(row["start_s"], row["end_s"]))
        data[i, j] = float(row[f"val_{config.selection_metric}"]) * 100
    fig, ax = plt.subplots(figsize=(max(12, len(windows) * 0.85), max(3.8, len(frequencies) * 0.9 + 1.8)))
    image = ax.imshow(np.ma.masked_invalid(data), cmap="viridis", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(windows)), [f"[{w.start:g},{w.end:g})" for w in windows], rotation=55, ha="right")
    ax.set_yticks(range(len(frequencies)), [f"{f:g} Hz" for f in frequencies])
    ax.set_xlabel("Cue-aligned window (seconds)")
    ax.set_ylabel("Sampling rate")
    ax.set_title(f"Validation {config.selection_metric} (%)")
    for i in range(len(frequencies)):
        for j in range(len(windows)):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if data[i, j] < 55 else "black")
    fig.colorbar(image, ax=ax, label="Balanced accuracy (%)")
    fig.tight_layout()
    fig.savefig(config.output_root / "Validation_Heatmap.png", dpi=180)
    plt.close(fig)
    for frequency in frequencies:
        rows_at_frequency = [row for row in rows if float(row["sfreq"]) == frequency]
        starts = sorted(set(row["start_s"] for row in rows_at_frequency))
        ends = sorted(set(row["end_s"] for row in rows_at_frequency))
        matrix = np.full((len(ends), len(starts)), np.nan)
        for row in rows_at_frequency:
            matrix[ends.index(row["end_s"]), starts.index(row["start_s"])] = row[f"val_{config.selection_metric}"] * 100
        fig, ax = plt.subplots(figsize=(max(6.5, len(starts) * 1.15 + 2), max(5, len(ends) * 0.75 + 2)))
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap="viridis", origin="lower",
                          aspect="auto", vmin=0, vmax=100)
        ax.set_xticks(range(len(starts)), [f"{s:g}" for s in starts])
        ax.set_yticks(range(len(ends)), [f"{e:g}" for e in ends])
        ax.set_xlabel("Window start after cue (s)")
        ax.set_ylabel("Window end after cue (s)")
        ax.set_title(f"{frequency:g} Hz | Validation {config.selection_metric} (%)")
        for i in range(len(ends)):
            for j in range(len(starts)):
                if np.isfinite(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=10,
                            color="white" if matrix[i, j] < 55 else "black")
        fig.colorbar(image, ax=ax, label="Balanced accuracy (%)")
        fig.tight_layout()
        fig.savefig(config.output_root / f"Window_Position_{frequency:g}Hz.png", dpi=180)
        plt.close(fig)


def write_summary(results, config):
    if not results:
        return None
    ranking = sorted(results, key=lambda result: (
        -result["val"][config.selection_metric],
        -result["val"]["session_bacc"],
        result["val"]["logloss"],
        result["sfreq"], result["window"][0], result["window"][1],
    ))
    save_csv(config.output_root / "Summary.csv", [summary_row(result) for result in ranking])
    save_json(config.output_root / "Summary.json", ranking)
    winner = ranking[0]
    save_json(config.output_root / "Best_Validation.json", {
        "selection_metric": config.selection_metric, "winning_profile": winner["profile"],
        "window": winner["window"], "validation": winner["val"],
        "checkpoint": str(Path(winner["run_folder"]) / "network.pt"),
        "ranking_rule": [config.selection_metric, "session_bacc", "lower logloss"],
    })
    write_figures(ranking, config)
    return winner


def evaluate_winner(winner, config, device):
    folder = Path(winner["run_folder"])
    result_file = folder / "result.json"
    if winner.get("test") is not None and (folder / "test_probs.npz").is_file():
        return winner
    index = read_index(winner["profile_root"])
    rows = select_rows(index, sessions=config.test_sessions, subjects=config.subjects,
                       contexts=config.contexts, exclude_artifacts=config.exclude_artifacts)
    window = Window(*winner["window"])
    with tempfile.TemporaryDirectory(prefix="shallow_test_", dir=config.output_root) as tmp:
        test = materialize_split(rows, window, Path(tmp) / "test.npy", config.field)
        model, _ = load_model(folder / "network.pt", test, device)
        scaler = json.loads((folder / "scaler.json").read_text(encoding="utf-8"))
        test_scores = evaluate(model, test, scaler["rms_microvolts"], config, device, folder, "test")
        del model
    winner["test"] = test_scores
    winner["n_test"] = len(rows)
    save_json(result_file, winner)
    save_json(config.output_root / "Final_Test.json", {
        "selected_using": f"validation {config.selection_metric}",
        "profile": winner["profile"], "window": winner["window"],
        "checkpoint": str(folder / "network.pt"), "test": test_scores,
    })
    print(f"Final winner Session 3: {config.selection_metric}={test_scores[config.selection_metric]:.4%}", flush=True)
    return winner


def main():
    config = settings()
    profile_paths = discover_profiles(config.data_root, config.profiles)
    indexes = [read_index(path) for path in profile_paths]
    verify_matched_profiles(indexes, config)
    torch.set_num_threads(config.threads)
    seed_all(config.seed, config.deterministic)
    device = resolve_device(config)
    config.output_root.mkdir(parents=True, exist_ok=True)
    print(f"Profiles: {len(indexes)}, windows: {len(config.windows)}, total: {len(indexes) * len(config.windows)}")
    print(f"Device: {device}; seed: {config.seed}; metric: {config.selection_metric}")
    started = time.perf_counter()
    results = []
    failures = []
    for index in indexes:
        for window in config.windows:
            try:
                results.append(one_experiment(index, window, config, device))
                write_summary(results, config)
            except Exception as error:
                failure = {"profile": index.root.name, "window": window.name,
                           "error": f"{type(error).__name__}: {error}"}
                failures.append(failure)
                save_json(config.output_root / "Failures.json", failures)
                if not config.continue_on_error:
                    raise
                print(f"FAILED: {failure}", flush=True)
    winner = write_summary(results, config)
    if winner is None:
        raise RuntimeError("No successful experiments")
    if config.evaluate_winner_on_test and not failures:
        evaluate_winner(winner, config, device)
        write_summary(results, config)
    save_json(config.output_root / "Run_Manifest.json", {
        "elapsed_s": time.perf_counter() - started,
        "completed": len(results), "failed": failures,
        "profiles": [index.root.name for index in indexes],
        "windows": [[w.start, w.end] for w in config.windows],
        "selection_metric": config.selection_metric,
        "best": {"profile": winner["profile"], "window": winner["window"]},
        "preprocessing_versions": {index.root.name: index.preprocessing_fingerprint for index in indexes},
        "code_hashes": {"trainer": sha256_file(Path(__file__)),
                        "dataloader": sha256_file(Path(__file__).with_name("dataloader.py"))},
    })
    print(f"Finished {len(results)} experiments in {time.perf_counter() - started:.1f}s")
    print(f"Best validation: {winner['profile']} {winner['window']} {winner['val'][config.selection_metric]:.4%}")
    print(f"Results: {config.output_root}")
    if failures:
        raise RuntimeError(f"{len(failures)} experiments failed; see Failures.json")


if __name__ == "__main__":
    main()
