from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import confusion_matrix

import trian__shallow as training
from dataloader import CLASS_NAMES, Window, materialize_split, read_index, select_rows, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = PROJECT_ROOT / "Weight" / "ShallowFBCSPNet_Full8_WindowSweep"
ANALYSIS_ROOT = EXPERIMENT_ROOT / "Analysis"
TEST_MODE = "all"
TOP_K_TO_TEST = 10
TOP_K_TO_REPORT = 10
SELECTION_METRIC = "auto"
DEVICE = "auto"
PRED_BATCH_SIZE = None
TORCH_THREADS = 4
CONTINUE_ON_TEST_ERROR = True
VERIFY_VALIDATION_PROBS = True
ANALYSIS_VERSION = "shallow_posthoc_analysis_v1"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".writing")
    try:
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_csv(path, rows, columns=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    columns = columns or list(dict.fromkeys(k for row in rows for k in row))
    tmp = path.with_name(path.name + ".writing")
    try:
        with tmp.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def write_npz(destination, **arrays):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".writing")
    try:
        with tmp.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path):
    if not Path(path).is_file():
        return []
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def score_key(row, metric):
    scores = row["val"]
    return (-float(scores[metric]), -float(scores.get("session_bacc", -1)),
            float(scores.get("logloss", float("inf"))), float(row["sfreq"]),
            float(row["window"][0]), float(row["window"][1]), row["profile"])


def load_experiments():
    manifest = EXPERIMENT_ROOT / "Run_Manifest.json"
    results_file = EXPERIMENT_ROOT / "Summary.json"
    if not results_file.is_file():
        raise FileNotFoundError(f"Training results not found: {results_file}")
    records = read_json(results_file)
    if not isinstance(records, list) or not records:
        raise ValueError(f"No successful experiments in {results_file}")
    unique = set()
    experiments = []
    for result in records:
        folder = Path(result["run_folder"]).expanduser().resolve()
        if folder in unique:
            raise ValueError(f"Duplicate experiment in Summary.json: {folder}")
        unique.add(folder)
        config_file = folder / "config.json"
        if not all((folder / name).is_file() for name in ("network.pt", "scaler.json", "val_probs.npz", "result.json", "config.json")):
            raise FileNotFoundError(f"Incomplete experiment: {folder}")
        record = read_json(config_file)
        if (record["profile"], [float(v) for v in record["window"]]) != (
                result["profile"], [float(v) for v in result["window"]]):
            raise ValueError(f"Summary/config disagreement: {folder}")
        for key, value in (("n_times", result["n_times"]), ("sfreq", result["sfreq"])):
            if not np.isclose(float(record[key]), float(value)):
                raise ValueError(f"Summary/config mismatch {key}: {folder}")
        if VERIFY_VALIDATION_PROBS:
            with np.load(folder / "val_probs.npz", allow_pickle=False) as saved:
                required = {"probs", "y", "subject", "session", "context"}
                if not required.issubset(saved.files):
                    raise ValueError(f"Validation NPZ missing fields {required - set(saved.files)}: {folder}")
                metadata = {k: np.asarray(saved[k]) for k in ("subject", "session", "context")}
                verified, _, _ = training.grouped_metrics(
                    np.asarray(saved["y"], dtype=np.int64),
                    np.asarray(saved["probs"], dtype=np.float32), metadata)
            for name, value in verified.items():
                if name in result["val"] and not np.isclose(value, float(result["val"][name]), atol=1e-7, rtol=0):
                    raise ValueError(f"Validation probability/score mismatch at {name}: {folder}")
        experiments.append({"result": result, "config": record, "folder": folder})
    metrics = {str(exp["config"]["selection_metric"]) for exp in experiments}
    if SELECTION_METRIC == "auto":
        if len(metrics) != 1:
            raise ValueError(f"Different selection metrics in sweep: {sorted(metrics)}")
        metric = next(iter(metrics))
    else:
        metric = SELECTION_METRIC
    if metric not in {"context_bacc", "session_bacc", "trial_bacc"}:
        raise ValueError(f"Unsupported selection metric: {metric}")
    for exp in experiments:
        if metric not in exp["result"]["val"]:
            raise ValueError(f"Missing validation metric in {exp['folder']}: {metric}")
    experiments.sort(key=lambda exp: score_key(exp["result"], metric))
    info = read_json(manifest) if manifest.is_file() else None
    return experiments, metric, info


def get_device():
    if DEVICE == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DEVICE='cuda' requires a CUDA-enabled PyTorch installation")
    if DEVICE not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {DEVICE}")
    return torch.device("cuda" if DEVICE == "cuda" or (DEVICE == "auto" and torch.cuda.is_available()) else "cpu")


def test_selection(experiments):
    if TEST_MODE == "none":
        return []
    if TEST_MODE == "winner":
        return experiments[:1]
    if TEST_MODE == "top_k":
        return experiments[:TOP_K_TO_TEST]
    if TEST_MODE == "all":
        return experiments
    raise ValueError("TEST_MODE must be 'all', 'top_k', 'winner' or 'none'")


def inspect_test_index(exp):
    record = exp["config"]
    root = Path(record["profile_root"]).expanduser().resolve()
    if root.name != record["profile"]:
        raise ValueError(f"Profile folder changed: {root}")
    index = read_index(root)
    if index.index_sha256 != record["dataset_index_sha256"]:
        raise ValueError(f"dataset_index.csv has changed since training: {root}")
    if record.get("preprocessing_sha256") != index.preprocessing_sha256:
        raise ValueError(f"Preprocessing state has changed since training: {root}")
    sessions = tuple(int(n) for n in record["test_sessions"])
    if set(sessions) & (set(record["train_sessions"]) | set(record["val_sessions"])):
        raise ValueError(f"Overlapping train/val/test sessions in {exp['folder']}")
    rows = select_rows(index, sessions=sessions, subjects=record["subjects"],
                       contexts=record["contexts"], exclude_artifacts=record["exclude_artifacts"])
    selected_hash = hashlib.sha256(json.dumps(
        [(row["path"], int(row["y"])) for row in rows], ensure_ascii=False).encode("utf-8")).hexdigest()
    return rows, selected_hash


def cache_signature(exp, test_hash, device):
    folder = exp["folder"]
    config = exp["config"]
    signature = {
        "version": ANALYSIS_VERSION,
        "config_sha256": sha256_file(folder / "config.json"),
        "checkpoint_sha256": sha256_file(folder / "network.pt"),
        "scaler_sha256": sha256_file(folder / "scaler.json"),
        "dataset_index_sha256": config["dataset_index_sha256"],
        "preprocessing_sha256": config.get("preprocessing_sha256"),
        "test_rows_sha256": test_hash,
        "trainer_sha256": sha256_file(Path(training.__file__)),
        "dataloader_sha256": sha256_file(Path(training.__file__).with_name("dataloader.py")),
        "device": device.type,
        "amp": bool(config["use_amp"] and device.type == "cuda"),
        "pred_batch_size": int(PRED_BATCH_SIZE or config["pred_batch_size"]),
        "torch_version": torch.__version__,
    }
    code = hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()[:20]
    return signature, ANALYSIS_ROOT / "TestCache" / code


def test_model(exp, device):
    record = exp["config"]
    folder = exp["folder"]
    rows, test_hash = inspect_test_index(exp)
    if "seed" in record:
        training.seed_all(int(record["seed"]), bool(record.get("deterministic", True)))
    signature, cached = cache_signature(exp, test_hash, device)
    required = [cached / "scores.json", cached / "test_probs.npz",
                cached / "context_scores.csv", cached / "session_scores.csv"]
    if all(path.is_file() for path in required):
        data = read_json(cached / "scores.json")
        if data.get("signature") == signature:
            print(f"  CACHE  {record['profile']} {record['window']}: {data['metrics']['context_bacc']:.4%}", flush=True)
            return data["metrics"], cached
    print(f"  TEST   {record['profile']} {record['window']} ({len(rows)} trials)", flush=True)
    window = Window.parse(record["window"])
    config = SimpleNamespace(pred_batch_size=int(PRED_BATCH_SIZE or record["pred_batch_size"]),
                             workers=0, use_amp=bool(record["use_amp"]), clip=float(record["clip"]))
    cached.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="shallow_analysis_", dir=ANALYSIS_ROOT) as work:
        split = materialize_split(rows, window, Path(work) / "test.npy", record["field"])
        if (split.n_channels, split.n_times, split.sfreq, split.signal_unit, list(split.ch_names)) != (
                record["n_channels"], record["n_times"], record["sfreq"],
                record["signal_unit"], record["ch_names"]):
            raise ValueError(f"Test data schema mismatch for {folder}")
        model, _ = training.load_model(folder / "network.pt", split, device)
        try:
            scaler = read_json(folder / "scaler.json")
            if scaler.get("signal_unit") != split.signal_unit or float(scaler.get("clip")) != float(record["clip"]):
                raise ValueError(f"Scaler mismatch: {folder}")
            probs = training.predict(model, split, float(scaler["rms_microvolts"]), config, device)
        finally:
            del model
        metrics, context_cells, session_cells = training.grouped_metrics(split.y, probs, split.meta)
        write_npz(cached / "test_probs.npz", probs=probs, y=split.y, **split.meta)
    write_csv(cached / "context_scores.csv", context_cells)
    write_csv(cached / "session_scores.csv", session_cells)
    write_json(cached / "scores.json", {"signature": signature, "metrics": metrics,
                                        "n_test": len(rows), "model_path": str(folder / "network.pt")})
    print(f"         context={metrics['context_bacc']:.4%} session={metrics['session_bacc']:.4%}", flush=True)
    return metrics, cached


def flat_row(exp, rank, test_result, metric):
    result = exp["result"]
    window = result["window"]
    row = {
        "val_rank": rank, "profile": result["profile"], "sfreq": float(result["sfreq"]),
        "window": f"[{float(window[0]):g}, {float(window[1]):g})",
        "start_s": float(window[0]), "end_s": float(window[1]),
        "duration_s": float(window[1]) - float(window[0]),
        "n_train": result["n_train"], "n_val": result["n_val"],
        "best_epoch": result["best_epoch"], "checkpoint": str(exp["folder"] / "network.pt"),
        "metric": metric,
    }
    for name, value in result["val"].items():
        row["val_" + name] = float(value)
    if test_result is not None:
        scores, cache = test_result
        for name, value in scores.items():
            row["test_" + name] = float(value)
        row["gap_val_minus_test_pp"] = 100 * (float(result["val"][metric]) - float(scores[metric]))
        row["test_cache"] = str(cache)
    return row


def clean_numeric(value):
    return float(value) if value is not None and value != "" else float("nan")


def plot_heatmap(rows, metric, which, destination):
    subset = [row for row in rows if np.isfinite(clean_numeric(row.get(f"{which}_{metric}")))]
    if not subset:
        return
    profiles = sorted(set((float(row["sfreq"]), row["profile"]) for row in rows))
    windows = sorted(set((float(row["start_s"]), float(row["end_s"])) for row in rows))
    matrix = np.full((len(profiles), len(windows)), np.nan)
    for row in subset:
        x = profiles.index((float(row["sfreq"]), row["profile"]))
        y = windows.index((float(row["start_s"]), float(row["end_s"])))
        if np.isfinite(matrix[x, y]):
            raise ValueError("Multiple runs with identical profile/window; filter experiments before plotting")
        matrix[x, y] = float(row[f"{which}_{metric}"]) * 100
    fig, ax = plt.subplots(figsize=(max(12, len(windows) * 0.85), 3.0 + len(profiles) * 0.75))
    plot = ax.imshow(np.ma.masked_invalid(matrix), cmap="viridis", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(windows)), [f"[{a:g},{b:g})" for a, b in windows], rotation=60, ha="right")
    labels = [name if len(profiles) > 3 else f"{rate:g} Hz" for rate, name in profiles]
    ax.set_yticks(range(len(profiles)), labels)
    ax.set_title(f"{which.upper()} {metric} (%)" + (" — exploratory, NOT for selection" if which == "test" else ""))
    for i in range(len(profiles)):
        for j in range(len(windows)):
            if np.isfinite(matrix[i, j]):
                ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=8,
                        color="white" if matrix[i, j] < 55 else "black")
    fig.colorbar(plot, ax=ax, label="Balanced accuracy (%)")
    fig.tight_layout()
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def plot_top(rows, metric, destination):
    chosen = rows[:TOP_K_TO_REPORT][::-1]
    if not chosen:
        return
    labels = [f"{r['sfreq']:g}Hz [{r['start_s']:g},{r['end_s']:g})" for r in chosen]
    val = np.array([100 * r[f"val_{metric}"] for r in chosen])
    test = np.array([100 * clean_numeric(r.get(f"test_{metric}")) for r in chosen])
    y = np.arange(len(chosen))
    fig, ax = plt.subplots(figsize=(11.5, max(4.5, len(chosen) * 0.57 + 1.4)))
    ax.barh(y - 0.19, val, height=0.36, label="Validation")
    if np.isfinite(test).any():
        ax.barh(y + 0.19, test, height=0.36, label="Test — diagnostic")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 100)
    ax.set_xlabel(f"{metric} (%)")
    ax.set_title(f"Top {len(chosen)} selected by validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=175)
    plt.close(fig)


def plot_gap(rows, metric, destination):
    tested = [r for r in rows if np.isfinite(clean_numeric(r.get(f"test_{metric}")))]
    if not tested:
        return
    fig, ax = plt.subplots(figsize=(8, 7))
    for hz in sorted(set(r["sfreq"] for r in tested)):
        group = [r for r in tested if r["sfreq"] == hz]
        ax.scatter([r[f"val_{metric}"] * 100 for r in group],
                   [r[f"test_{metric}"] * 100 for r in group],
                   label=f"{hz:g} Hz", s=48, alpha=0.8)
    ax.plot([0, 100], [0, 100], linestyle="--", linewidth=1)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Validation (%)")
    ax.set_ylabel("Test (%)")
    ax.set_title(f"Validation vs Test: {metric} (diagnostic)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(destination, dpi=175)
    plt.close(fig)


def plot_window_maps(rows, metric):
    profiles = sorted(set((row["sfreq"], row["profile"]) for row in rows))
    rates = [rate for rate, _ in profiles]
    for rate, profile in profiles:
        subset = [row for row in rows if row["profile"] == profile]
        starts = sorted(set(row["start_s"] for row in subset))
        ends = sorted(set(row["end_s"] for row in subset))
        views = ["val", "test"] if any("test_" + metric in row for row in subset) else ["val"]
        fig, axs = plt.subplots(1, len(views), figsize=(max(7, len(starts) * 1.05) * len(views),
                                                     max(5.3, len(ends) * 0.56 + 1.4)), squeeze=False)
        for ax, phase in zip(axs[0], views):
            grid = np.full((len(ends), len(starts)), np.nan)
            for row in subset:
                value = clean_numeric(row.get(f"{phase}_{metric}"))
                if np.isfinite(value):
                    grid[ends.index(row["end_s"]), starts.index(row["start_s"])] = value * 100
            im = ax.imshow(np.ma.masked_invalid(grid), origin="lower", cmap="viridis", aspect="auto", vmin=0, vmax=100)
            ax.set_xticks(range(len(starts)), [f"{start:g}" for start in starts])
            ax.set_yticks(range(len(ends)), [f"{end:g}" for end in ends])
            ax.set_xlabel("Window start (s)")
            ax.set_ylabel("Window end (s)")
            ax.set_title(f"{profile} | {phase.upper()}" + (" diagnostic" if phase == "test" else ""), fontsize=10)
            for i in range(len(ends)):
                for j in range(len(starts)):
                    if np.isfinite(grid[i, j]):
                        ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center", fontsize=9,
                                color="white" if grid[i, j] < 55 else "black")
            fig.colorbar(im, ax=ax, fraction=0.044, label=f"{metric} (%)")
        fig.tight_layout()
        filename = f"Window_Map_{rate:g}Hz.png" if rates.count(rate) == 1 else f"Window_Map_{profile}.png"
        fig.savefig(ANALYSIS_ROOT / filename, dpi=170)
        plt.close(fig)


def plot_confusion(exp, test_cache, destination):
    if test_cache is None:
        return
    with np.load(test_cache / "test_probs.npz", allow_pickle=False) as archive:
        truth = np.asarray(archive["y"], dtype=np.int64)
        pred = np.asarray(archive["probs"], dtype=np.float32).argmax(axis=1)
    matrix = confusion_matrix(truth, pred, labels=np.arange(len(CLASS_NAMES)), normalize="true")
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    plot = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_yticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Validation-selected winner: Test confusion matrix")
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{matrix[i, j] * 100:.1f}%", ha="center", va="center",
                    color="white" if matrix[i, j] > 0.55 else "black")
    fig.colorbar(plot, ax=ax, label="Row-normalized rate")
    fig.tight_layout()
    fig.savefig(destination, dpi=175)
    plt.close(fig)


def context_tables(experiments, tests):
    records = []
    class_warnings = []
    for position, exp in enumerate(experiments, start=1):
        folder = exp["folder"]
        for phase, path in (("val", folder / "val_context_scores.csv"),
                            ("test", tests[str(folder)][1] / "context_scores.csv") if str(folder) in tests else ("test", None)):
            if path is None:
                continue
            for row in read_csv(path):
                item = {"val_rank": position, "phase": phase, "profile": exp["result"]["profile"],
                        "sfreq": exp["result"]["sfreq"], "start_s": exp["result"]["window"][0],
                        "end_s": exp["result"]["window"][1],
                        "subject": int(row["subject"]), "session": int(row["session"]),
                        "context": row["context"], "n": int(row["n"]),
                        "n_classes": int(row.get("n_classes", len(CLASS_NAMES))),
                        "balanced_accuracy": float(row["balanced_accuracy"])}
                records.append(item)
                if item["n_classes"] != len(CLASS_NAMES):
                    class_warnings.append(item)
    write_csv(ANALYSIS_ROOT / "Per_Subject_Context.csv", records)
    aggregated = []
    keys = sorted(set((row["val_rank"], row["phase"], row["context"]) for row in records))
    for rank, phase, context in keys:
        match = [row for row in records if (row["val_rank"], row["phase"], row["context"]) == (rank, phase, context)]
        aggregated.append({"val_rank": rank, "phase": phase, "context": context,
                           "n_cells": len(match), "n_trials": sum(row["n"] for row in match),
                           "mean_context_bacc": float(np.mean([row["balanced_accuracy"] for row in match])),
                           "min_context_bacc": float(np.min([row["balanced_accuracy"] for row in match])),
                           "std_context_bacc": float(np.std([row["balanced_accuracy"] for row in match]))})
    write_csv(ANALYSIS_ROOT / "By_Context.csv", aggregated)
    write_csv(ANALYSIS_ROOT / "Missing_Classes_Cells.csv", class_warnings,
              columns=list(records[0]) if records else ["val_rank", "phase", "subject", "session", "context", "n_classes"])
    return aggregated, class_warnings


def as_percent(value):
    return f"{100 * float(value):.2f}%"


def markdown_rank(rows, metric, include_test=False, max_rows=10):
    lines = ["| Val rank | Hz | Window (s) | Val | Test | Gap (pp) |",
             "|---:|---:|---|---:|---:|---:|"]
    for row in rows[:max_rows]:
        v = as_percent(row[f"val_{metric}"])
        t = as_percent(row[f"test_{metric}"]) if include_test and f"test_{metric}" in row else "—"
        gap = f"{row['gap_val_minus_test_pp']:+.2f}" if include_test and "gap_val_minus_test_pp" in row else "—"
        lines.append(f"| {row['val_rank']} | {row['sfreq']:g} | {row['window']} | {v} | {t} | {gap} |")
    return lines


def write_report(rows, metric, manifest, grouped, incomplete):
    scored = [row for row in rows if f"test_{metric}" in row]
    scored.sort(key=lambda row: (-row[f"test_{metric}"], row["val_rank"]))
    best = rows[0]
    by_rate = {}
    for row in rows:
        by_rate.setdefault(row["sfreq"], row)
    lines = ["# ShallowFBCSPNet 时间窗实验结果分析", "",
             f"- 模型选择依据：**Validation {metric}**。所有 Test 数值仅作诊断对照，不用来选择最终模型。",
             f"- 已完成训练组合：**{len(rows)}**。本次成功计算或读取的 Test 组合：**{len(scored)}**。",
             f"- 测试模式：`{TEST_MODE}`。训练代码和模型权重不改动。",
             "- 原有分组评分规则由训练脚本 `grouped_metrics` 计算。", ""]
    if manifest is None:
        lines += ["**注意：** 未发现 `Run_Manifest.json`；这可能是尚未完成的实验快照。", ""]
    elif manifest.get("failed"):
        lines += [f"**注意：** 原实验有 {len(manifest['failed'])} 组失败；排名仅包含成功实验。", ""]
    if incomplete:
        lines += [f"**注意：** 有 {len(incomplete)} 个模型 Test 推理失败，详见 `Test_Errors.json`。", ""]
    lines += ["## 按 Validation 选择的最佳模型", "",
              f"**{best['sfreq']:g} Hz、{best['window']} 秒**，Validation {metric} = **{as_percent(best[f'val_{metric}'])}**。"]
    if f"test_{metric}" in best:
        lines[-1] += f" 同一模型的 Test {metric} = **{as_percent(best[f'test_{metric}'])}**。"
    lines += ["", "## Validation Top 10", ""] + markdown_rank(rows, metric, True, TOP_K_TO_REPORT) + [""]
    lines += ["## 每个采样率按 Validation 选出的最佳窗口", "",
              "| Hz | Window (s) | Val | Test (diagnostic) |",
              "|---:|---|---:|---:|"]
    for rate, row in sorted(by_rate.items()):
        t = as_percent(row[f"test_{metric}"]) if f"test_{metric}" in row else "—"
        lines.append(f"| {rate:g} | {row['window']} | {as_percent(row[f'val_{metric}'])} | {t} |")
    lines += ["", "## Test 前 10（只用于泛化诊断，禁止据此重新选模型）", ""]
    if scored:
        lines += markdown_rank(scored, metric, True, TOP_K_TO_REPORT)
        lines += ["", f"Test 数值最高的候选位于 Validation 排名第 {scored[0]['val_rank']} 位；仅用于诊断，不能用它替代按 Validation 预选的模型。"]
    else:
        lines += ["本次没有 Test 结果。"]
    lines += ["", "## Validated winner 的 Graz / BH 分组结果", "",
              "| Split | Context | BAcc (cell macro) | Cells |",
              "|---|---|---:|---:|"]
    for item in grouped:
        if item["val_rank"] == 1:
            lines.append(f"| {item['phase']} | {item['context']} | {as_percent(item['mean_context_bacc'])} | {item['n_cells']} |")
    lines += ["", "## 输出文件", "",
              "- `Scoreboard.csv`：全组合按 Validation 排序，包含 Val、Test、泛化差距和权重路径。",
              "- `Top_Candidates.csv`、`Best_Per_Frequency.csv`、`Best_Per_Duration.csv`：候选组合汇总。",
              "- `Ranking_Test_Exploratory.csv`：仅作为诊断的 Test 排名。",
              "- `Per_Subject_Context.csv`：所有被试、session 和 Graz/BH 单元分数。",
              "- `By_Context.csv`：各组合 Graz/BH 的宏平均表现。",
              "- `Heatmap_Validation.png`、`Heatmap_Test_Exploratory.png`、`Top_Validation_vs_Test.png`。",
              "- `Window_Map_*Hz.png`：时间窗起点/终点二维图。",
              "- `Validation_Test_Scatter.png`、`Winner_Test_Confusion.png`。", "",
              "## 结果解释", "",
              "对全部模型运行 Test 不需要重新训练，但一旦用 Test 分数决定使用哪个窗口，Session 3 就不再是独立测试集。",
              "如果要基于这些对照继续调参，应固定新的训练/验证协议，最后使用另一个未参与选择的测试集。",
              "当前结果通常是单次随机种子，接近的窗口差异还应通过多随机种子或被试层面的置信区间评估。", ""]
    (ANALYSIS_ROOT / "Analysis_Report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    started = time.perf_counter()
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    experiments, metric, manifest = load_experiments()
    torch.set_num_threads(TORCH_THREADS)
    device = get_device()
    print(f"Found {len(experiments)} completed models; selection=Validation {metric}; TEST_MODE={TEST_MODE}; device={device}", flush=True)
    chosen = test_selection(experiments)
    write_json(ANALYSIS_ROOT / "Test_Errors.json", [])
    tests = {}
    errors = []
    for i, exp in enumerate(chosen, start=1):
        print(f"[{i}/{len(chosen)}]", end="", flush=True)
        try:
            tests[str(exp["folder"])] = test_model(exp, device)
        except Exception as error:
            item = {"run_folder": str(exp["folder"]), "error": f"{type(error).__name__}: {error}"}
            errors.append(item)
            write_json(ANALYSIS_ROOT / "Test_Errors.json", errors)
            print(f"  ERROR {item['error']}", flush=True)
            if not CONTINUE_ON_TEST_ERROR:
                raise
    scoreboard = [flat_row(exp, rank, tests.get(str(exp["folder"])), metric)
                  for rank, exp in enumerate(experiments, start=1)]
    write_csv(ANALYSIS_ROOT / "Scoreboard.csv", scoreboard)
    write_csv(ANALYSIS_ROOT / "Top_Candidates.csv", scoreboard[:TOP_K_TO_REPORT])
    best_by_frequency = {}
    best_by_duration = {}
    for row in scoreboard:
        best_by_frequency.setdefault(row["sfreq"], row)
        best_by_duration.setdefault(row["duration_s"], row)
    write_csv(ANALYSIS_ROOT / "Best_Per_Frequency.csv", [best_by_frequency[k] for k in sorted(best_by_frequency)])
    write_csv(ANALYSIS_ROOT / "Best_Per_Duration.csv", [best_by_duration[k] for k in sorted(best_by_duration)])
    write_json(ANALYSIS_ROOT / "Scoreboard.json", scoreboard)
    write_csv(ANALYSIS_ROOT / "Ranking_Test_Exploratory.csv",
              sorted([row for row in scoreboard if f"test_{metric}" in row],
                     key=lambda row: (-row[f"test_{metric}"], row["val_rank"])))
    write_json(ANALYSIS_ROOT / "Best_By_Validation.json", scoreboard[0])
    tested = [row for row in scoreboard if f"test_{metric}" in row]
    if tested:
        write_json(ANALYSIS_ROOT / "Best_By_Test_Exploratory.json",
                   sorted(tested, key=lambda row: (-row[f"test_{metric}"], row["val_rank"]))[0])
    aggregated, missing = context_tables(experiments, tests)
    plot_heatmap(scoreboard, metric, "val", ANALYSIS_ROOT / "Heatmap_Validation.png")
    plot_heatmap(scoreboard, metric, "test", ANALYSIS_ROOT / "Heatmap_Test_Exploratory.png")
    plot_top(scoreboard, metric, ANALYSIS_ROOT / "Top_Validation_vs_Test.png")
    plot_gap(scoreboard, metric, ANALYSIS_ROOT / "Validation_Test_Scatter.png")
    plot_window_maps(scoreboard, metric)
    val_winner = experiments[0]
    if str(val_winner["folder"]) in tests:
        plot_confusion(val_winner, tests[str(val_winner["folder"])][1],
                       ANALYSIS_ROOT / "Winner_Test_Confusion.png")
    write_report(scoreboard, metric, manifest, aggregated, errors)
    write_json(ANALYSIS_ROOT / "Analysis_Manifest.json", {
        "version": ANALYSIS_VERSION, "selection_metric": metric, "test_mode": TEST_MODE,
        "n_models": len(experiments), "n_tested": len(tests), "n_failed_tests": len(errors),
        "missing_class_cells": len(missing), "device": str(device),
        "runtime_s": time.perf_counter() - started,
        "validation_winner": str(val_winner["folder"]),
    })
    print("\nValidation Top 10", flush=True)
    for row in scoreboard[:TOP_K_TO_REPORT]:
        ts = f"  test={row[f'test_{metric}']:.2%}" if f"test_{metric}" in row else ""
        print(f"  #{row['val_rank']:02d}  {row['sfreq']:g}Hz  {row['window']:>11}  val={row[f'val_{metric}']:.2%}{ts}", flush=True)
    print(f"\nReport: {ANALYSIS_ROOT / 'Analysis_Report.md'}", flush=True)
    print(f"Figures and tables: {ANALYSIS_ROOT}", flush=True)
    if errors:
        raise RuntimeError(f"{len(errors)} test evaluations failed; see {ANALYSIS_ROOT / 'Test_Errors.json'}")


if __name__ == "__main__":
    main()
