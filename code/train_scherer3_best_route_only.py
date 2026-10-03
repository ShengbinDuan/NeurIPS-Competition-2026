from __future__ import annotations

r"""
Frozen best-route Scherer2015 3-class cross-session training script.

This script is designed for the output of:
    prepare_scherer3_multimethod_torch.py

It compares, under one leakage-controlled protocol:
  - published Braindecode architectures: ShallowFBCSPNet, EEGNet, Deep4Net,
    ATCNet, EEGConformer, FBCNet;
  - CSP + shrinkage LDA;
  - covariance tangent-space + logistic regression;
  - multiple task windows and frequency bands;
  - weak EEG augmentation and class/subject-consistent Segmentation & Reconstruction;
  - subject-wise Euclidean Alignment (EA);
  - pooled global models and global->subject fine-tuning;
  - multi-seed probability averaging;
  - validation-selected temperature/prior post-processing;
  - multi-model uniform, weighted and greedy soft-voting ensembles.

Important evaluation rule
-------------------------
All model selection, epoch selection, post-processing selection and ensemble
selection use session0 validation labels ONLY. session1 labels, when present in the
local public dataset, are used only to report final benchmark metrics. Do not tune
again after reading session1 metrics if you want session1 to remain a meaningful
proxy for a hidden competition test set.

Recommended install:
    pip install -U braindecode mne scikit-learn scipy tqdm

Examples:
    python train_scherer3_multimethod_sweep.py --suite core
    python train_scherer3_multimethod_sweep.py --suite full --n-seeds 3
    python train_scherer3_multimethod_sweep.py --suite full --val-mode blocked --n-seeds 5

Windows-safe defaults: num_workers=0, CUDA required by default.
"""

import argparse
import copy
import csv
import gc
import inspect
import json
import math
import os
import random
import re
import warnings
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import optimize, signal
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


CODE_VERSION = "scherer3_best_route_only_2026-10-03_v1"
N_CLASSES = 3
CLASS_NAMES = ["SUB", "WORD", "HAND"]
CHANCE = 1.0 / N_CLASSES


# =============================================================================
# CLI / configuration
# =============================================================================
def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "--data-root",
        type=Path,
        default=here / "prepared_data" / "Scherer2015_3Class_MultiMethod_Torch",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=here / "scherer3_best_route_results",
    )
    p.add_argument("--suite", choices=["current", "core", "full"], default="full", help="Compatibility only; ignored in BEST_ONLY route.")
    p.add_argument("--only", type=str, default="", help="Compatibility only; ignored in BEST_ONLY route.")
    p.add_argument("--seed0", type=int, default=42)
    p.add_argument("--n-seeds", type=int, default=5)
    p.add_argument("--val-ratio", type=float, default=0.25)
    p.add_argument("--val-mode", choices=["random", "blocked"], default="random")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--max-epochs", type=int, default=180)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--label-smoothing", type=float, default=0.03)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--input-clip-z", type=float, default=8.0)
    p.add_argument("--ea-shrinkage", type=float, default=1e-3)
    p.add_argument("--ea-eig-floor", type=float, default=1e-10)
    p.add_argument("--artifact-mode", choices=["none", "auto", "source", "both"], default="both")
    p.add_argument("--ft-epochs", type=int, default=16, help="Global->subject fine-tuning epochs.")
    p.add_argument("--ft-lr", type=float, default=1e-4)
    p.add_argument("--cache-max", type=int, default=2, help="Max preprocessed signal variants kept in CPU RAM.")
    p.add_argument("--require-cuda", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--save-models", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


@dataclass(frozen=True)
class ExpConfig:
    name: str
    model: str
    crop_start: float
    crop_duration: float
    band_lo: float
    band_hi: float
    reref: str = "car"
    ea: str = "subject"
    augment: str = "sr"          # none | weak | sr
    mixup_alpha: float = 0.0
    scope: str = "pooled"        # pooled | subject_finetune | subjectwise(classical)
    lr: float = 6e-4
    dropout: float = 0.5


# The suite is intentionally diverse. Diversity matters for the final ensemble.
def experiment_suite(name: str = "best") -> list[ExpConfig]:
    """
    Frozen winning route only.

    Final ensemble members:
      1) conformer_3to10_2to45_sr
      2) shallow_current_3to7_4to40
      3) riemann_subject_4to8_4to40

    Their training / preprocessing / EA / calibration implementations remain
    exactly the same as in the original sweep script.
    """
    return [
        ExpConfig(
            name="conformer_3to10_2to45_sr",
            model="conformer",
            crop_start=3.0,
            crop_duration=7.0,
            band_lo=2.0,
            band_hi=45.0,
            reref="car",
            ea="subject",
            augment="sr",
            mixup_alpha=0.0,
            scope="pooled",
            lr=5e-4,
            dropout=0.4,
        ),
        ExpConfig(
            name="shallow_current_3to7_4to40",
            model="shallow",
            crop_start=3.0,
            crop_duration=4.0,
            band_lo=4.0,
            band_hi=40.0,
            reref="car",
            ea="subject",
            augment="weak",
            mixup_alpha=0.0,
            scope="pooled",
            lr=6e-4,
            dropout=0.5,
        ),
        ExpConfig(
            name="riemann_subject_4to8_4to40",
            model="riemann",
            crop_start=4.0,
            crop_duration=4.0,
            band_lo=4.0,
            band_hi=40.0,
            reref="car",
            ea="subject",
            augment="none",
            mixup_alpha=0.0,
            scope="subjectwise",
            lr=6e-4,
            dropout=0.5,
        ),
    ]


# =============================================================================
# Device / reproducibility
# =============================================================================
def configure_device(args: argparse.Namespace) -> torch.device:
    cuda_ok = torch.cuda.is_available()
    if args.require_cuda and not cuda_ok:
        raise RuntimeError(
            "CUDA is not available. Check with:\n"
            '  python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.version.cuda)"'
        )
    device = torch.device("cuda" if cuda_ok else "cpu")
    if device.type == "cuda":
        if args.tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(
            f"Device: CUDA | {torch.cuda.get_device_name(0)} | "
            f"torch={torch.__version__} | cuda={torch.version.cuda}"
        )
    else:
        print(f"Device: CPU | torch={torch.__version__}")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# I/O
# =============================================================================
def load_npz(path: Path, require_y: bool = True) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, float, dict[str, np.ndarray]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}\nRun prepare_scherer3_multimethod_torch.py first."
        )
    with np.load(path, allow_pickle=False) as f:
        payload = {k: f[k] for k in f.files}
    for k in ("X", "subject", "sfreq"):
        if k not in payload:
            raise KeyError(f"{path.name}: missing {k}")
    if require_y and "y" not in payload:
        raise KeyError(f"{path.name}: missing y")
    X = np.asarray(payload["X"], dtype=np.float32)
    s = np.asarray(payload["subject"], dtype=np.int64).reshape(-1)
    y = np.asarray(payload["y"], dtype=np.int64).reshape(-1) if "y" in payload else None
    sfreq = float(np.asarray(payload["sfreq"]).reshape(-1)[0])
    if X.ndim != 3 or len(X) != len(s) or (y is not None and len(y) != len(X)):
        raise ValueError(f"Bad shapes: X={X.shape} y={None if y is None else y.shape} subject={s.shape}")
    if require_y and set(np.unique(y).tolist()) != {0, 1, 2}:
        raise ValueError(f"Expected labels 0/1/2, got {np.unique(y)}")
    if not np.isfinite(X).all():
        raise ValueError(f"NaN/Inf in {path}")
    return X, y, s, sfreq, payload


def scalar(payload: dict[str, np.ndarray], key: str) -> float:
    if key not in payload:
        raise KeyError(
            f"Prepared file is missing {key!r}. This sweep needs the new wide-context preparation. "
            "Run prepare_scherer3_multimethod_torch.py."
        )
    return float(np.asarray(payload[key]).reshape(-1)[0])


def safe_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def safe_torch_save(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def sanitize_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


# =============================================================================
# Metrics
# =============================================================================
def subject_macro_bacc(y_true: np.ndarray, y_pred: np.ndarray, subjects: np.ndarray) -> float:
    vals = []
    for sid in sorted(np.unique(subjects).tolist()):
        idx = subjects == sid
        vals.append(balanced_accuracy_score(y_true[idx], y_pred[idx]))
    return float(np.mean(vals))


def metrics(y_true: np.ndarray, prob: np.ndarray, subjects: np.ndarray) -> dict[str, float]:
    pred = prob.argmax(axis=1)
    return {
        "bacc": float(balanced_accuracy_score(y_true, pred)),
        "subject_macro_bacc": subject_macro_bacc(y_true, pred, subjects),
        "acc": float(np.mean(y_true == pred)),
        "nll": float(log_loss(y_true, np.clip(prob, 1e-7, 1.0), labels=[0, 1, 2])),
    }


# =============================================================================
# Split / artifact handling
# =============================================================================
def split_session0_by_subject(
    y: np.ndarray,
    subjects: np.ndarray,
    event_sample: np.ndarray,
    val_ratio: float,
    seed: int,
    mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    tr: list[int] = []
    va: list[int] = []
    for sid in sorted(np.unique(subjects).tolist()):
        for cls in range(N_CLASSES):
            idx = np.flatnonzero((subjects == sid) & (y == cls))
            if len(idx) < 2:
                raise ValueError(f"Need >=2 trials for subject={sid}, class={cls}")
            n_val = max(1, min(len(idx) - 1, int(round(len(idx) * val_ratio))))
            if mode == "random":
                idx = idx.copy()
                rng.shuffle(idx)
                take_va = idx[:n_val]
                take_tr = idx[n_val:]
            elif mode == "blocked":
                idx = idx[np.argsort(event_sample[idx])]
                take_va = idx[-n_val:]
                take_tr = idx[:-n_val]
            else:
                raise ValueError(mode)
            va.extend(take_va.tolist())
            tr.extend(take_tr.tolist())
    return np.asarray(sorted(tr), np.int64), np.asarray(sorted(va), np.int64)


def training_flags(payload: dict[str, np.ndarray], n: int, mode: str) -> np.ndarray:
    flag = np.zeros(n, dtype=bool)
    if mode in ("auto", "both") and "auto_artifact_flag" in payload:
        flag |= np.asarray(payload["auto_artifact_flag"]).reshape(-1).astype(np.int64) != 0
    if mode in ("source", "both") and "source_artifact_flag" in payload:
        flag |= np.asarray(payload["source_artifact_flag"]).reshape(-1).astype(np.int64) != 0
    return flag


def keep_mask_preserve_groups(y: np.ndarray, subjects: np.ndarray, flagged: np.ndarray, min_per_group: int = 3) -> np.ndarray:
    keep = ~np.asarray(flagged, dtype=bool)
    for sid in np.unique(subjects):
        for cls in range(N_CLASSES):
            idx = np.flatnonzero((subjects == sid) & (y == cls))
            if len(idx) == 0:
                continue
            needed = min(min_per_group, len(idx))
            if int(keep[idx].sum()) < needed:
                keep[idx] = True
    return keep


# =============================================================================
# Band/crop/rereference cache
# =============================================================================
def bandpass_chunked(X: np.ndarray, sfreq: float, lo: float, hi: float, chunk: int = 96) -> np.ndarray:
    if not (0 < lo < hi < sfreq / 2):
        raise ValueError(f"Bad band {lo}-{hi} for sfreq={sfreq}")
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=sfreq, output="sos")
    out = np.empty_like(X, dtype=np.float32)
    for a in range(0, len(X), chunk):
        b = min(len(X), a + chunk)
        z = signal.sosfiltfilt(sos, np.asarray(X[a:b], dtype=np.float64), axis=-1)
        out[a:b] = z.astype(np.float32, copy=False)
    return out


def prepare_variant(
    Xctx: np.ndarray,
    sfreq: float,
    context_start: float,
    crop_start: float,
    crop_duration: float,
    lo: float,
    hi: float,
    reref: str,
) -> np.ndarray:
    crop_end = crop_start + crop_duration
    context_end = context_start + Xctx.shape[-1] / sfreq
    if crop_start < context_start - 1e-6 or crop_end > context_end + 1e-6:
        raise ValueError(
            f"Crop [{crop_start},{crop_end}) outside saved context [{context_start},{context_end})"
        )
    # Filter while real context is still present, then crop.
    z = bandpass_chunked(Xctx, sfreq, lo, hi)
    a = int(round((crop_start - context_start) * sfreq))
    n = int(round(crop_duration * sfreq))
    z = z[..., a:a + n].copy()
    if z.shape[-1] != n:
        raise RuntimeError("Crop length mismatch")
    if reref == "car":
        z -= z.mean(axis=1, keepdims=True)
    elif reref == "none":
        pass
    else:
        raise ValueError(reref)
    return z.astype(np.float32, copy=False)


class VariantCache:
    def __init__(self, max_items: int):
        self.max_items = max(1, int(max_items))
        self.data: OrderedDict[tuple, tuple[np.ndarray, np.ndarray]] = OrderedDict()

    def get(
        self,
        key: tuple,
        builder,
    ) -> tuple[np.ndarray, np.ndarray]:
        if key in self.data:
            val = self.data.pop(key)
            self.data[key] = val
            return val
        val = builder()
        self.data[key] = val
        while len(self.data) > self.max_items:
            self.data.popitem(last=False)
            gc.collect()
        return val


# =============================================================================
# Euclidean Alignment
# =============================================================================
def fit_ea_whitener(X: np.ndarray, shrinkage: float, eig_floor: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    Xc = X - X.mean(axis=-1, keepdims=True)
    n, c, t = Xc.shape
    R = np.einsum("nct,ndt->cd", Xc, Xc, optimize=True) / (n * max(1, t - 1))
    R = 0.5 * (R + R.T)
    scale = float(np.trace(R) / c)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("EA covariance has non-positive trace")
    R = (1.0 - shrinkage) * R + shrinkage * scale * np.eye(c)
    eigval, eigvec = np.linalg.eigh(R)
    floor = max(eig_floor * scale, np.finfo(np.float64).eps * scale)
    eigval = np.maximum(eigval, floor)
    return (eigvec * (1.0 / np.sqrt(eigval))[None, :]) @ eigvec.T


def ea_align_domainwise(X: np.ndarray, subjects: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    out = np.empty_like(X, dtype=np.float32)
    for sid in sorted(np.unique(subjects).tolist()):
        idx = np.flatnonzero(subjects == sid)
        W = fit_ea_whitener(X[idx], args.ea_shrinkage, args.ea_eig_floor)
        out[idx] = np.einsum("ij,njt->nit", W, np.asarray(X[idx], dtype=np.float64), optimize=True).astype(np.float32)
    return out


def maybe_ea(X: np.ndarray, subjects: np.ndarray, cfg: ExpConfig, args: argparse.Namespace) -> np.ndarray:
    if cfg.ea == "none":
        return X.astype(np.float32, copy=True)
    if cfg.ea == "subject":
        return ea_align_domainwise(X, subjects, args)
    raise ValueError(cfg.ea)


# =============================================================================
# Deep-model normalization / augmentation
# =============================================================================
class ChannelStandardizer:
    def __init__(self, clip_z: float):
        self.clip_z = float(clip_z)
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        self.mean_ = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        self.std_ = X.std(axis=(0, 2), keepdims=True).astype(np.float32)
        self.std_ = np.maximum(self.std_, np.float32(1e-6))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.mean_ is not None and self.std_ is not None
        z = ((X - self.mean_) / self.std_).astype(np.float32, copy=False)
        if self.clip_z > 0:
            z = np.clip(z, -self.clip_z, self.clip_z).astype(np.float32, copy=False)
        return z


def noncircular_shift(x: np.ndarray, shift: int) -> np.ndarray:
    if shift == 0:
        return x
    out = np.zeros_like(x)
    if shift > 0:
        out[:, shift:] = x[:, :-shift]
    else:
        k = -shift
        out[:, :-k] = x[:, k:]
    return out


class EEGDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        subjects: np.ndarray,
        augment: str = "none",
        sr_segments: int = 8,
    ):
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.subjects = np.asarray(subjects, dtype=np.int64)
        self.augment = augment
        self.sr_segments = int(sr_segments)
        self.same: dict[tuple[int, int], np.ndarray] = {}
        if augment == "sr":
            for sid in np.unique(self.subjects):
                for cls in range(N_CLASSES):
                    self.same[(int(sid), cls)] = np.flatnonzero((self.subjects == sid) & (self.y == cls))
            for cls in range(N_CLASSES):
                self.same[(-1, cls)] = np.flatnonzero(self.y == cls)

    def __len__(self) -> int:
        return len(self.y)

    def _segment_reconstruct(self, idx: int, x: np.ndarray) -> np.ndarray:
        if np.random.rand() >= 0.50:
            return x
        cls = int(self.y[idx])
        sid = int(self.subjects[idx])
        pool = self.same.get((sid, cls), np.empty(0, np.int64))
        if len(pool) < 2:
            pool = self.same[(-1, cls)]
        if len(pool) == 0:
            return x
        t = x.shape[-1]
        edges = np.linspace(0, t, self.sr_segments + 1, dtype=int)
        out = x.copy()
        for k in range(self.sr_segments):
            donor = int(pool[np.random.randint(0, len(pool))])
            a, b = int(edges[k]), int(edges[k + 1])
            out[:, a:b] = self.X[donor, :, a:b]
        return out

    @staticmethod
    def _weak(x: np.ndarray) -> np.ndarray:
        c, t = x.shape
        if np.random.rand() < 0.50:
            x *= np.float32(np.random.uniform(0.92, 1.08))
        if np.random.rand() < 0.35:
            sigma = max(float(np.std(x)) * np.random.uniform(0.005, 0.02), 1e-4)
            x += np.random.normal(0.0, sigma, size=x.shape).astype(np.float32)
        if np.random.rand() < 0.15:
            n_drop = int(np.random.randint(1, min(3, c + 1)))
            ch = np.random.choice(c, n_drop, replace=False)
            x[ch] = 0.0
        if np.random.rand() < 0.25:
            max_shift = max(1, int(round(0.025 * t)))
            x = noncircular_shift(x, int(np.random.randint(-max_shift, max_shift + 1)))
        if np.random.rand() < 0.15:
            width = max(1, int(round(np.random.uniform(0.02, 0.05) * t)))
            start = int(np.random.randint(0, max(1, t - width + 1)))
            x[:, start:start + width] = 0.0
        return x

    def __getitem__(self, idx: int):
        x = self.X[idx].copy()
        if self.augment == "sr":
            x = self._segment_reconstruct(idx, x)
            x = self._weak(x)
        elif self.augment == "weak":
            x = self._weak(x)
        elif self.augment != "none":
            raise ValueError(self.augment)
        return torch.from_numpy(x), torch.tensor(self.y[idx], dtype=torch.long)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    args: argparse.Namespace,
    seed: int,
    augment: str,
    shuffle: bool,
) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(
        EEGDataset(X, y, subjects, augment=augment),
        batch_size=args.batch_size,
        shuffle=shuffle,
        generator=g if shuffle else None,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(args.num_workers > 0),
        drop_last=False,
    )


# =============================================================================
# Braindecode model registry
# =============================================================================
def _braindecode_class(model_name: str):
    try:
        import braindecode.models as bm
    except Exception as e:
        raise RuntimeError(
            "Deep model sweep requires Braindecode. Install with: pip install -U braindecode mne"
        ) from e
    aliases = {
        "shallow": ["ShallowFBCSPNet"],
        "eegnet": ["EEGNet", "EEGNetv4"],
        "deep4": ["Deep4Net"],
        "atcnet": ["ATCNet"],
        "conformer": ["EEGConformer"],
        "fbcnet": ["FBCNet"],
    }
    for name in aliases[model_name]:
        if hasattr(bm, name):
            return getattr(bm, name)
    raise RuntimeError(f"Braindecode installation does not provide {aliases[model_name]}")


def build_deep_model(cfg: ExpConfig, n_chans: int, n_times: int, sfreq: float, device: torch.device) -> nn.Module:
    cls = _braindecode_class(cfg.model)
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    candidate = {
        "n_chans": n_chans,
        "n_outputs": N_CLASSES,
        "n_times": n_times,
        "sfreq": sfreq,
        "input_window_seconds": n_times / sfreq,
        "drop_prob": cfg.dropout,
        "dropout": cfg.dropout,
        "conv_block_dropout": cfg.dropout,
        "tcn_drop_prob": cfg.dropout,
        "att_drop_prob": min(0.5, cfg.dropout),
        "final_conv_length": "auto",
        "add_log_softmax": False,
    }
    kwargs = {k: v for k, v in candidate.items() if k in params}
    try:
        model = cls(**kwargs)
    except Exception as first:
        # Retry only with the universal signal args for compatibility across Braindecode releases.
        universal = {k: v for k, v in candidate.items() if k in {"n_chans", "n_outputs", "n_times", "sfreq", "input_window_seconds"} and k in params}
        try:
            model = cls(**universal)
        except Exception:
            raise RuntimeError(f"Failed to construct {cfg.model} with args={kwargs}: {first}") from first
    return model.to(device)


def forward_logits(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    out = model(x)
    if isinstance(out, dict):
        for key in ("logits", "output", "preds"):
            if key in out:
                out = out[key]
                break
        else:
            out = next(iter(out.values()))
    if isinstance(out, (tuple, list)):
        out = out[0]
    if not torch.is_tensor(out):
        raise TypeError(f"Model returned unsupported type {type(out)}")
    if out.ndim > 2:
        out = out.mean(dim=tuple(range(2, out.ndim)))
    if out.ndim != 2 or out.shape[1] != N_CLASSES:
        raise RuntimeError(f"Expected model output [B,{N_CLASSES}], got {tuple(out.shape)}")
    return out


def class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y.astype(np.int64), minlength=N_CLASSES).astype(np.float64)
    w = counts.sum() / (N_CLASSES * np.maximum(counts, 1))
    w /= w.mean()
    return torch.tensor(w, dtype=torch.float32, device=device)


def lr_at(epoch0: int, total_epochs: int, base_lr: float, warmup: int = 5) -> float:
    if epoch0 < warmup:
        return base_lr * (epoch0 + 1) / max(1, warmup)
    q = (epoch0 - warmup) / max(1, total_epochs - warmup)
    q = min(max(q, 0.0), 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * q))


def set_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = lr


def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except Exception:
        return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.inference_mode()
def predict_proba(
    model: nn.Module,
    X: np.ndarray,
    subjects: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    batch_size: int = 256,
) -> np.ndarray:
    model.eval()
    ds = EEGDataset(X, np.zeros(len(X), dtype=np.int64), subjects, augment="none")
    dl = DataLoader(
        ds, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"), persistent_workers=(args.num_workers > 0),
    )
    chunks = []
    for xb, _ in dl:
        xb = xb.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = forward_logits(model, xb)
        chunks.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler,
    loss_fn: nn.Module,
    mixup_alpha: float,
    device: torch.device,
    amp_enabled: bool,
) -> float:
    model.train()
    total = 0.0
    seen = 0
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        use_mix = mixup_alpha > 0 and len(yb) > 1
        if use_mix:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(len(yb), device=device)
            xmix = lam * xb + (1.0 - lam) * xb[perm]
        else:
            lam = 1.0
            perm = None
            xmix = xb
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            logits = forward_logits(model, xmix)
            if perm is None:
                loss = loss_fn(logits, yb)
            else:
                loss = lam * loss_fn(logits, yb) + (1.0 - lam) * loss_fn(logits, yb[perm])
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        b = len(yb)
        total += float(loss.detach().float().cpu()) * b
        seen += b
    return total / max(1, seen)


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_with_validation(
    cfg: ExpConfig,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    str_: np.ndarray,
    Xva: np.ndarray,
    yva: np.ndarray,
    sva: np.ndarray,
    sfreq: float,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
):
    set_seed(seed)
    model = build_deep_model(cfg, Xtr.shape[1], Xtr.shape[2], sfreq, device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=args.weight_decay)
    scaler = make_scaler(amp_enabled)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(ytr, device), label_smoothing=args.label_smoothing)
    dl = make_loader(Xtr, ytr, str_, args, seed, cfg.augment, shuffle=True)
    best = {"bacc": -np.inf, "subject_macro_bacc": -np.inf, "epoch": 0, "state": None, "prob": None}
    stale = 0
    total_epochs = 3 if args.quick else args.max_epochs
    patience = min(2, args.patience) if args.quick else args.patience
    bar = tqdm(range(1, total_epochs + 1), desc=f"{cfg.name} seed {seed} select", unit="ep", dynamic_ncols=True)
    for epoch in bar:
        set_lr(opt, lr_at(epoch - 1, total_epochs, cfg.lr))
        loss = train_epoch(model, dl, opt, scaler, loss_fn, cfg.mixup_alpha, device, amp_enabled)
        pva = predict_proba(model, Xva, sva, args, device, amp_enabled)
        met = metrics(yva, pva, sva)
        improved = (
            met["bacc"] > best["bacc"] + 1e-8
            or (abs(met["bacc"] - best["bacc"]) <= 1e-8 and met["subject_macro_bacc"] > best["subject_macro_bacc"] + 1e-8)
        )
        if improved:
            best.update(
                bacc=met["bacc"], subject_macro_bacc=met["subject_macro_bacc"], epoch=epoch,
                state=cpu_state_dict(model), prob=pva.copy(),
            )
            stale = 0
        else:
            stale += 1
        bar.set_postfix(loss=f"{loss:.4f}", val=f"{met['bacc']:.4f}", best=f"{best['bacc']:.4f}@{best['epoch']}", refresh=False)
        if stale >= patience:
            break
    bar.close()
    if best["state"] is None:
        raise RuntimeError("No best model captured")
    model.load_state_dict(best["state"])
    return model, best


def train_fixed_epochs(
    cfg: ExpConfig,
    X: np.ndarray,
    y: np.ndarray,
    subjects: np.ndarray,
    sfreq: float,
    seed: int,
    epochs: int,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    initial_state: dict[str, torch.Tensor] | None = None,
    lr_override: float | None = None,
    desc: str | None = None,
    show_bar: bool = True,
) -> nn.Module:
    set_seed(seed)
    model = build_deep_model(cfg, X.shape[1], X.shape[2], sfreq, device)
    if initial_state is not None:
        model.load_state_dict(initial_state)
    lr = cfg.lr if lr_override is None else float(lr_override)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scaler = make_scaler(amp_enabled)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(y, device), label_smoothing=args.label_smoothing)
    dl = make_loader(X, y, subjects, args, seed, cfg.augment, shuffle=True)
    epochs = max(1, int(epochs))
    iterable: Iterable[int] = range(1, epochs + 1)
    if show_bar:
        iterable = tqdm(iterable, desc=desc or f"{cfg.name} seed {seed} refit", unit="ep", dynamic_ncols=True)
    last_loss = float("nan")
    for epoch in iterable:
        set_lr(opt, lr_at(epoch - 1, epochs, lr, warmup=min(3, epochs)))
        last_loss = train_epoch(model, dl, opt, scaler, loss_fn, cfg.mixup_alpha, device, amp_enabled)
        if show_bar and hasattr(iterable, "set_postfix"):
            iterable.set_postfix(loss=f"{last_loss:.4f}", refresh=False)
    if show_bar and hasattr(iterable, "close"):
        iterable.close()
    return model


def personalize_probabilities(
    cfg: ExpConfig,
    base_state: dict[str, torch.Tensor],
    Xsource: np.ndarray,
    ysource: np.ndarray,
    ssource: np.ndarray,
    Xtarget: np.ndarray,
    starget: np.ndarray,
    sfreq: float,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
) -> np.ndarray:
    out = np.zeros((len(Xtarget), N_CLASSES), dtype=np.float32)
    bar = tqdm(sorted(np.unique(starget).tolist()), desc=f"{cfg.name} seed {seed} personalize", unit="subj", dynamic_ncols=True)
    for sid in bar:
        isrc = np.flatnonzero(ssource == sid)
        itgt = np.flatnonzero(starget == sid)
        if len(isrc) == 0 or len(itgt) == 0:
            continue
        ft_seed = seed + 10000 + int(sid)
        model = train_fixed_epochs(
            cfg, Xsource[isrc], ysource[isrc], ssource[isrc], sfreq, ft_seed,
            2 if args.quick else args.ft_epochs, args, device, amp_enabled,
            initial_state=base_state, lr_override=args.ft_lr, show_bar=False,
        )
        out[itgt] = predict_proba(model, Xtarget[itgt], starget[itgt], args, device, amp_enabled)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    bar.close()
    return out


# =============================================================================
# Classical CSP / Riemannian-tangent baselines
# =============================================================================
def fit_csp_lda(X: np.ndarray, y: np.ndarray):
    try:
        import mne
        from mne.decoding import CSP
        mne.set_log_level("ERROR")
    except Exception as e:
        raise RuntimeError("CSP baseline requires MNE (pip install -U mne).") from e
    n_comp = min(12, X.shape[1])
    csp = CSP(n_components=n_comp, reg="ledoit_wolf", log=True, norm_trace=False)
    lda = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    model = make_pipeline(csp, lda)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)
    return model



class FilterBankCSPClassifier:
    """Compact multiclass FBCSP: 8 sub-bands x 4 CSP components -> logistic regression."""
    def __init__(self, sfreq: float):
        self.sfreq = float(sfreq)
        self.bands = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 24), (24, 30), (30, 36), (36, 44)]
        self.csps = []
        self.pipe = None

    def _filter(self, X: np.ndarray, lo: float, hi: float) -> np.ndarray:
        sos = signal.butter(4, [lo, hi], btype="bandpass", fs=self.sfreq, output="sos")
        out = np.empty_like(X, dtype=np.float32)
        for a in range(0, len(X), 96):
            b = min(len(X), a + 96)
            out[a:b] = signal.sosfiltfilt(
                sos, np.asarray(X[a:b], dtype=np.float64), axis=-1
            ).astype(np.float32)
        return out

    def fit(self, X: np.ndarray, y: np.ndarray):
        try:
            import mne
            from mne.decoding import CSP
            mne.set_log_level("ERROR")
        except Exception as e:
            raise RuntimeError("FBCSP baseline requires MNE (pip install -U mne).") from e
        feats = []
        self.csps = []
        for lo, hi in self.bands:
            Xb = self._filter(X, lo, hi)
            csp = CSP(n_components=min(4, X.shape[1]), reg="ledoit_wolf", log=True, norm_trace=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                Fb = csp.fit_transform(Xb, y)
            self.csps.append(csp)
            feats.append(Fb)
        F = np.concatenate(feats, axis=1)
        self.pipe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced", solver="lbfgs"),
        )
        self.pipe.fit(F, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.pipe is None or not self.csps:
            raise RuntimeError("FBCSP not fitted")
        feats = []
        for (lo, hi), csp in zip(self.bands, self.csps):
            Xb = self._filter(X, lo, hi)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                feats.append(csp.transform(Xb))
        F = np.concatenate(feats, axis=1)
        return self.pipe.predict_proba(F).astype(np.float32)


def covariance_matrices(X: np.ndarray, shrink: float = 0.05) -> np.ndarray:
    Xd = np.asarray(X, dtype=np.float64)
    Xd = Xd - Xd.mean(axis=-1, keepdims=True)
    n, c, t = Xd.shape
    cov = np.einsum("nct,ndt->ncd", Xd, Xd, optimize=True) / max(1, t - 1)
    tr = np.trace(cov, axis1=1, axis2=2) / c
    tr = np.maximum(tr, 1e-18)
    eye = np.eye(c)[None, :, :]
    cov = (1.0 - shrink) * cov + shrink * tr[:, None, None] * eye
    # Trace normalization makes the feature less sensitive to session gain.
    cov /= np.maximum(np.trace(cov, axis1=1, axis2=2)[:, None, None], 1e-18)
    return cov


def spd_invsqrt(A: np.ndarray) -> np.ndarray:
    e, V = np.linalg.eigh(0.5 * (A + A.T))
    e = np.maximum(e, 1e-12)
    return (V * (1.0 / np.sqrt(e))[None, :]) @ V.T


def spd_log(A: np.ndarray) -> np.ndarray:
    e, V = np.linalg.eigh(0.5 * (A + A.T))
    e = np.maximum(e, 1e-12)
    return (V * np.log(e)[None, :]) @ V.T


def tangent_vectorize(M: np.ndarray) -> np.ndarray:
    c = M.shape[0]
    iu = np.triu_indices(c)
    v = M[iu].copy()
    off = iu[0] != iu[1]
    v[off] *= np.sqrt(2.0)
    return v


class TangentClassifier:
    def __init__(self):
        self.W: np.ndarray | None = None
        self.pipe = None

    def _features(self, X: np.ndarray) -> np.ndarray:
        assert self.W is not None
        cov = covariance_matrices(X)
        feat = np.empty((len(cov), X.shape[1] * (X.shape[1] + 1) // 2), dtype=np.float64)
        for i, C in enumerate(cov):
            S = self.W @ C @ self.W.T
            feat[i] = tangent_vectorize(spd_log(S))
        return feat

    def fit(self, X: np.ndarray, y: np.ndarray):
        cov = covariance_matrices(X)
        ref = cov.mean(axis=0)
        self.W = spd_invsqrt(ref)
        F = self._features(X)
        max_pc = min(128, F.shape[1], max(2, len(F) - N_CLASSES - 1))
        self.pipe = make_pipeline(
            StandardScaler(),
            PCA(n_components=max_pc, svd_solver="full", random_state=0),
            LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced", solver="lbfgs"),
        )
        self.pipe.fit(F, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert self.pipe is not None
        return self.pipe.predict_proba(self._features(X)).astype(np.float32)


def classical_predict(
    cfg: ExpConfig,
    sfreq: float,
    Xtr: np.ndarray,
    ytr: np.ndarray,
    str_: np.ndarray,
    Xva: np.ndarray,
    sva: np.ndarray,
    Xfull: np.ndarray,
    yfull: np.ndarray,
    sfull: np.ndarray,
    Xtest: np.ndarray,
    stest: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    def build(X, y):
        if cfg.model == "csp":
            return fit_csp_lda(X, y)
        if cfg.model == "riemann":
            return TangentClassifier().fit(X, y)
        if cfg.model == "fbcsp":
            return FilterBankCSPClassifier(sfreq).fit(X, y)
        raise ValueError(cfg.model)

    if cfg.scope == "pooled":
        mval = build(Xtr, ytr)
        pva = mval.predict_proba(Xva).astype(np.float32)
        mtest = build(Xfull, yfull)
        pte = mtest.predict_proba(Xtest).astype(np.float32)
        return pva, pte

    if cfg.scope == "subjectwise":
        pva = np.zeros((len(Xva), N_CLASSES), dtype=np.float32)
        pte = np.zeros((len(Xtest), N_CLASSES), dtype=np.float32)
        for sid in sorted(np.unique(sva).tolist()):
            a = np.flatnonzero(str_ == sid)
            b = np.flatnonzero(sva == sid)
            if len(a) and len(b):
                m = build(Xtr[a], ytr[a])
                pva[b] = m.predict_proba(Xva[b])
        for sid in sorted(np.unique(stest).tolist()):
            a = np.flatnonzero(sfull == sid)
            b = np.flatnonzero(stest == sid)
            if len(a) and len(b):
                m = build(Xfull[a], yfull[a])
                pte[b] = m.predict_proba(Xtest[b])
        return pva, pte
    raise ValueError(cfg.scope)


# =============================================================================
# Probability calibration / transductive prior balancing
# =============================================================================
def apply_temperature(prob: np.ndarray, T: float) -> np.ndarray:
    z = np.log(np.clip(prob, 1e-8, 1.0)) / max(float(T), 1e-3)
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    q /= q.sum(axis=1, keepdims=True)
    return q.astype(np.float32)


def fit_temperature(prob: np.ndarray, y: np.ndarray) -> float:
    def obj(logT: float) -> float:
        T = math.exp(float(logT))
        q = apply_temperature(prob, T)
        return float(log_loss(y, np.clip(q, 1e-7, 1.0), labels=[0, 1, 2]))
    res = optimize.minimize_scalar(obj, bounds=(math.log(0.5), math.log(3.0)), method="bounded")
    return float(math.exp(res.x)) if res.success else 1.0


def balance_group_probs(prob: np.ndarray, idx: np.ndarray, n_iter: int = 30) -> None:
    target = np.full(N_CLASSES, 1.0 / N_CLASSES, dtype=np.float64)
    q = prob[idx].astype(np.float64, copy=True)
    for _ in range(n_iter):
        mean = np.maximum(q.mean(axis=0), 1e-8)
        q *= (target / mean)[None, :]
        q /= np.maximum(q.sum(axis=1, keepdims=True), 1e-12)
    prob[idx] = q.astype(np.float32)


def balance_probs(prob: np.ndarray, subjects: np.ndarray | None, mode: str) -> np.ndarray:
    q = prob.copy()
    if mode == "global":
        balance_group_probs(q, np.arange(len(q)))
    elif mode == "subject":
        if subjects is None:
            raise ValueError("subjects required")
        for sid in np.unique(subjects):
            balance_group_probs(q, np.flatnonzero(subjects == sid))
    elif mode == "none":
        pass
    else:
        raise ValueError(mode)
    return q


def select_postprocess(
    pva: np.ndarray,
    pte: np.ndarray,
    yva: np.ndarray,
    sva: np.ndarray,
    ste: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, float]:
    T = fit_temperature(pva, yva)
    vaT = apply_temperature(pva, T)
    teT = apply_temperature(pte, T)
    candidates = []
    for mode in ("none", "global", "subject"):
        qv = balance_probs(vaT, sva, mode)
        qt = balance_probs(teT, ste, mode)
        met = metrics(yva, qv, sva)
        candidates.append((met["bacc"], -met["nll"], mode, qv, qt))
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    _, _, mode, qv, qt = candidates[0]
    return qv, qt, f"T={T:.3f}+prior={mode}", T


def choose_personal_blend(
    global_va: np.ndarray,
    personal_va: np.ndarray,
    global_te: np.ndarray,
    personal_te: np.ndarray,
    yva: np.ndarray,
    sva: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    best = None
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        q = (1.0 - alpha) * global_va + alpha * personal_va
        met = metrics(yva, q, sva)
        key = (met["bacc"], met["subject_macro_bacc"])
        if best is None or key > best[0]:
            best = (key, alpha)
    assert best is not None
    alpha = float(best[1])
    return (
        ((1.0 - alpha) * global_va + alpha * personal_va).astype(np.float32),
        ((1.0 - alpha) * global_te + alpha * personal_te).astype(np.float32),
        alpha,
    )


# =============================================================================
# Single experiment
# =============================================================================
def run_experiment(
    cfg: ExpConfig,
    X0p: np.ndarray,
    X1p: np.ndarray,
    y0: np.ndarray,
    y1: np.ndarray | None,
    s0: np.ndarray,
    s1: np.ndarray,
    sfreq: float,
    p0: dict[str, np.ndarray],
    idx_tr: np.ndarray,
    idx_va: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    amp_enabled: bool,
    seeds: list[int],
) -> dict[str, Any]:
    flags0 = training_flags(p0, len(X0p), args.artifact_mode)
    keep_tr_local = keep_mask_preserve_groups(y0[idx_tr], s0[idx_tr], flags0[idx_tr])
    idx_tr_clean = idx_tr[keep_tr_local]
    keep0 = keep_mask_preserve_groups(y0, s0, flags0)

    Xtr0, ytr, str_ = X0p[idx_tr_clean], y0[idx_tr_clean], s0[idx_tr_clean]
    Xva0, yva, sva = X0p[idx_va], y0[idx_va], s0[idx_va]
    Xfull0, yfull, sfull = X0p[keep0], y0[keep0], s0[keep0]

    Xtr_ea = maybe_ea(Xtr0, str_, cfg, args)
    Xva_ea = maybe_ea(Xva0, sva, cfg, args)
    Xfull_ea = maybe_ea(Xfull0, sfull, cfg, args)
    Xtest_ea = maybe_ea(X1p, s1, cfg, args)

    is_classical = cfg.model in {"csp", "riemann", "fbcsp"}
    if is_classical:
        pva, pte = classical_predict(
            cfg, sfreq, Xtr_ea, ytr, str_, Xva_ea, sva,
            Xfull_ea, yfull, sfull, Xtest_ea, s1,
        )
        pva, pte, post_name, T = select_postprocess(pva, pte, yva, sva, s1)
        val_met = metrics(yva, pva, sva)
        test_met = metrics(y1, pte, s1) if y1 is not None else None
        return {
            "name": cfg.name, "config": asdict(cfg), "val_prob": pva, "test_prob": pte,
            "val_y": yva, "val_subject": sva, "postprocess": post_name, "temperature": T,
            "best_epochs": [], "per_seed_val_bacc": [], "personal_alpha": [],
            "val_metrics": val_met, "test_metrics": test_met,
        }

    # Train-only standardization for neural models.
    sc_val = ChannelStandardizer(args.input_clip_z).fit(Xtr_ea)
    Xtr = sc_val.transform(Xtr_ea)
    Xva = sc_val.transform(Xva_ea)
    sc_final = ChannelStandardizer(args.input_clip_z).fit(Xfull_ea)
    Xfull = sc_final.transform(Xfull_ea)
    Xtest = sc_final.transform(Xtest_ea)

    val_probs: list[np.ndarray] = []
    test_probs: list[np.ndarray] = []
    best_epochs: list[int] = []
    seed_val: list[float] = []
    personal_alphas: list[float] = []

    for seed in seeds:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        select_model, best = train_with_validation(
            cfg, Xtr, ytr, str_, Xva, yva, sva, sfreq, seed, args, device, amp_enabled
        )
        pva_global = np.asarray(best["prob"], dtype=np.float32)
        best_epoch = int(best["epoch"])

        if cfg.scope == "subject_finetune":
            pva_personal = personalize_probabilities(
                cfg, best["state"], Xtr, ytr, str_, Xva, sva, sfreq,
                seed, args, device, amp_enabled,
            )
        else:
            pva_personal = None

        del select_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

        final_model = train_fixed_epochs(
            cfg, Xfull, yfull, sfull, sfreq, seed, best_epoch,
            args, device, amp_enabled, desc=f"{cfg.name} seed {seed} refit",
        )
        pte_global = predict_proba(final_model, Xtest, s1, args, device, amp_enabled)
        final_state = cpu_state_dict(final_model)

        if cfg.scope == "subject_finetune":
            pte_personal = personalize_probabilities(
                cfg, final_state, Xfull, yfull, sfull, Xtest, s1, sfreq,
                seed, args, device, amp_enabled,
            )
            pva_seed, pte_seed, alpha = choose_personal_blend(
                pva_global, pva_personal, pte_global, pte_personal, yva, sva
            )
            personal_alphas.append(alpha)
        else:
            pva_seed, pte_seed = pva_global, pte_global

        val_probs.append(pva_seed)
        test_probs.append(pte_seed)
        best_epochs.append(best_epoch)
        seed_val.append(metrics(yva, pva_seed, sva)["bacc"])

        if args.save_models:
            ckpt = {
                "code_version": CODE_VERSION,
                "config": asdict(cfg),
                "seed": seed,
                "best_epoch": best_epoch,
                "state_dict": final_state,
                "channel_mean": sc_final.mean_,
                "channel_std": sc_final.std_,
            }
            safe_torch_save(ckpt, args.outdir / "models" / f"{sanitize_name(cfg.name)}_seed{seed}.pt")

        del final_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    pva = np.mean(np.stack(val_probs, axis=0), axis=0).astype(np.float32)
    pte = np.mean(np.stack(test_probs, axis=0), axis=0).astype(np.float32)
    pva, pte, post_name, T = select_postprocess(pva, pte, yva, sva, s1)
    val_met = metrics(yva, pva, sva)
    test_met = metrics(y1, pte, s1) if y1 is not None else None

    return {
        "name": cfg.name, "config": asdict(cfg), "val_prob": pva, "test_prob": pte,
        "val_y": yva, "val_subject": sva, "postprocess": post_name, "temperature": T,
        "best_epochs": best_epochs, "per_seed_val_bacc": seed_val,
        "personal_alpha": personal_alphas,
        "val_metrics": val_met, "test_metrics": test_met,
    }


# =============================================================================
# Cache experiment output / leaderboard / ensembles
# =============================================================================
def save_experiment_cache(path: Path, result: dict[str, Any]) -> None:
    tm = result["test_metrics"] or {"bacc": np.nan, "subject_macro_bacc": np.nan, "acc": np.nan, "nll": np.nan}
    vm = result["val_metrics"]
    safe_npz(
        path,
        code_version=np.asarray(CODE_VERSION),
        name=np.asarray(result["name"]),
        config_json=np.asarray(json.dumps(result["config"], sort_keys=True)),
        val_prob=result["val_prob"], test_prob=result["test_prob"],
        val_y=result["val_y"], val_subject=result["val_subject"],
        postprocess=np.asarray(result["postprocess"]),
        temperature=np.asarray(result["temperature"], dtype=np.float64),
        best_epochs=np.asarray(result["best_epochs"], dtype=np.int64),
        per_seed_val_bacc=np.asarray(result["per_seed_val_bacc"], dtype=np.float64),
        personal_alpha=np.asarray(result["personal_alpha"], dtype=np.float64),
        val_metrics=np.asarray([vm["bacc"], vm["subject_macro_bacc"], vm["acc"], vm["nll"]], dtype=np.float64),
        test_metrics=np.asarray([tm["bacc"], tm["subject_macro_bacc"], tm["acc"], tm["nll"]], dtype=np.float64),
    )


def load_experiment_cache(path: Path, cfg: ExpConfig) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as f:
            if str(np.asarray(f["code_version"]).item()) != CODE_VERSION:
                return None
            saved_cfg = json.loads(str(np.asarray(f["config_json"]).item()))
            if saved_cfg != asdict(cfg):
                return None
            vm = np.asarray(f["val_metrics"], dtype=float)
            tm = np.asarray(f["test_metrics"], dtype=float)
            return {
                "name": str(np.asarray(f["name"]).item()),
                "config": saved_cfg,
                "val_prob": np.asarray(f["val_prob"], dtype=np.float32),
                "test_prob": np.asarray(f["test_prob"], dtype=np.float32),
                "val_y": np.asarray(f["val_y"], dtype=np.int64),
                "val_subject": np.asarray(f["val_subject"], dtype=np.int64),
                "postprocess": str(np.asarray(f["postprocess"]).item()),
                "temperature": float(np.asarray(f["temperature"]).item()),
                "best_epochs": np.asarray(f["best_epochs"], dtype=int).tolist(),
                "per_seed_val_bacc": np.asarray(f["per_seed_val_bacc"], dtype=float).tolist(),
                "personal_alpha": np.asarray(f["personal_alpha"], dtype=float).tolist(),
                "val_metrics": dict(zip(["bacc", "subject_macro_bacc", "acc", "nll"], vm.tolist())),
                "test_metrics": None if np.isnan(tm[0]) else dict(zip(["bacc", "subject_macro_bacc", "acc", "nll"], tm.tolist())),
            }
    except Exception:
        return None


def ensemble_result(
    name: str,
    members: list[dict[str, Any]],
    weights: np.ndarray,
    yva: np.ndarray,
    sva: np.ndarray,
    y1: np.ndarray | None,
    s1: np.ndarray,
) -> dict[str, Any]:
    weights = np.asarray(weights, dtype=np.float64)
    weights /= weights.sum()
    pva = sum(w * r["val_prob"] for w, r in zip(weights, members)).astype(np.float32)
    pte = sum(w * r["test_prob"] for w, r in zip(weights, members)).astype(np.float32)
    return {
        "name": name,
        "config": {"ensemble_members": [r["name"] for r in members], "weights": weights.tolist()},
        "val_prob": pva, "test_prob": pte, "val_y": yva, "val_subject": sva,
        "postprocess": "already_member_calibrated", "temperature": 1.0,
        "best_epochs": [], "per_seed_val_bacc": [], "personal_alpha": [],
        "val_metrics": metrics(yva, pva, sva),
        "test_metrics": metrics(y1, pte, s1) if y1 is not None else None,
    }


def make_ensembles(
    results: list[dict[str, Any]],
    yva: np.ndarray,
    sva: np.ndarray,
    y1: np.ndarray | None,
    s1: np.ndarray,
) -> list[dict[str, Any]]:
    """
    Frozen final route: equal-weight soft voting of the three known winning members.

    IMPORTANT:
    Each member has already completed its original member-level post-processing
    (multi-seed averaging where applicable + validation-selected temperature/prior).
    The final ensemble simply averages those calibrated probability matrices 1/3 each.
    """
    required = [
        "conformer_3to10_2to45_sr",
        "shallow_current_3to7_4to40",
        "riemann_subject_4to8_4to40",
    ]
    by_name = {r["name"]: r for r in results}
    missing = [name for name in required if name not in by_name]
    if missing:
        raise RuntimeError(f"Missing required winning-route members: {missing}")

    members = [by_name[name] for name in required]
    return [
        ensemble_result(
            "ensemble_greedy",
            members,
            np.ones(3, dtype=np.float64),
            yva,
            sva,
            y1,
            s1,
        )
    ]


def row_from_result(r: dict[str, Any]) -> dict[str, Any]:
    vm = r["val_metrics"]
    tm = r["test_metrics"] or {}
    return {
        "name": r["name"],
        "val_bacc": vm["bacc"],
        "val_subject_macro_bacc": vm["subject_macro_bacc"],
        "val_acc": vm["acc"],
        "val_nll": vm["nll"],
        "test_bacc": tm.get("bacc", np.nan),
        "test_subject_macro_bacc": tm.get("subject_macro_bacc", np.nan),
        "test_acc": tm.get("acc", np.nan),
        "test_nll": tm.get("nll", np.nan),
        "postprocess": r.get("postprocess", ""),
        "best_epochs": ";".join(map(str, r.get("best_epochs", []))),
        "personal_alpha": ";".join(f"{x:.2f}" for x in r.get("personal_alpha", [])),
        "config": json.dumps(r.get("config", {}), sort_keys=True),
    }


def write_leaderboard(path: Path, all_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [row_from_result(r) for r in all_results]
    rows.sort(key=lambda x: (x["val_bacc"], x["val_subject_macro_bacc"]), reverse=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return rows


def print_leaderboard(rows: list[dict[str, Any]], topn: int = 15) -> None:
    print("\nLeaderboard (RANKED BY SESSION0 VALIDATION ONLY; test is report-only)")
    print("-" * 112)
    print(f"{'#':>2}  {'name':<42} {'valBAcc':>8} {'valSubj':>8} {'testBAcc':>9} {'testSubj':>9}  post")
    for i, r in enumerate(rows[:topn], 1):
        tb = r["test_bacc"]
        ts = r["test_subject_macro_bacc"]
        print(
            f"{i:>2}  {r['name']:<42} {r['val_bacc']:>8.4f} {r['val_subject_macro_bacc']:>8.4f} "
            f"{tb:>9.4f} {ts:>9.4f}  {r['postprocess']}"
        )


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()
    if args.quick:
        args.n_seeds = 1
        args.max_epochs = min(args.max_epochs, 3)
        args.patience = min(args.patience, 2)
        args.ft_epochs = min(args.ft_epochs, 2)
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "cache").mkdir(parents=True, exist_ok=True)

    device = configure_device(args)
    amp_enabled = bool(args.amp and device.type == "cuda")
    seeds = [args.seed0 + 17 + 1000 * i for i in range(args.n_seeds)]

    X0ctx, y0, s0, sf0, p0 = load_npz(args.data_root / "calibration_session0.npz", require_y=True)
    X1ctx, y1, s1, sf1, p1 = load_npz(args.data_root / "cross_session_test_session1.npz", require_y=False)
    assert y0 is not None
    if abs(sf0 - sf1) > 1e-9:
        raise ValueError("sfreq mismatch")
    if X0ctx.shape[1:] != X1ctx.shape[1:]:
        raise ValueError("session0/session1 shape mismatch")

    context_start = scalar(p0, "context_start_s")
    context_start1 = scalar(p1, "context_start_s")
    if abs(context_start - context_start1) > 1e-9:
        raise ValueError("context_start mismatch")
    event0 = np.asarray(p0.get("event_sample", np.arange(len(X0ctx))), dtype=np.int64).reshape(-1)

    idx_tr, idx_va = split_session0_by_subject(
        y0, s0, event0, args.val_ratio, args.seed0, args.val_mode
    )
    yva = y0[idx_va]
    sva = s0[idx_va]

    configs = experiment_suite("best")
    if not configs:
        raise ValueError("No experiments selected")

    print(
        f"Data: session0={X0ctx.shape}, session1={X1ctx.shape}, sfreq={sf0:g}, "
        f"context_start={context_start:g}s | val_mode={args.val_mode} | artifact_mode={args.artifact_mode}"
    )
    print(f"Route=BEST_ONLY | configs={len(configs)} | seeds={seeds}")

    cache = VariantCache(args.cache_max)
    results: list[dict[str, Any]] = []

    for ci, cfg in enumerate(configs, 1):
        print(f"\n=== [{ci}/{len(configs)}] {cfg.name} ===")
        cache_file = args.outdir / "cache" / f"{sanitize_name(cfg.name)}.npz"
        if args.resume:
            cached = load_experiment_cache(cache_file, cfg)
            if cached is not None:
                print(
                    f"resume | val BAcc={cached['val_metrics']['bacc']:.4f}" +
                    (f" | test BAcc={cached['test_metrics']['bacc']:.4f}" if cached['test_metrics'] else "")
                )
                results.append(cached)
                continue

        key = (cfg.crop_start, cfg.crop_duration, cfg.band_lo, cfg.band_hi, cfg.reref)
        def builder():
            print(
                f"preprocess | crop=[{cfg.crop_start:g},{cfg.crop_start + cfg.crop_duration:g})s "
                f"band={cfg.band_lo:g}-{cfg.band_hi:g}Hz reref={cfg.reref}"
            )
            a = prepare_variant(X0ctx, sf0, context_start, cfg.crop_start, cfg.crop_duration, cfg.band_lo, cfg.band_hi, cfg.reref)
            b = prepare_variant(X1ctx, sf1, context_start, cfg.crop_start, cfg.crop_duration, cfg.band_lo, cfg.band_hi, cfg.reref)
            return a, b
        X0p, X1p = cache.get(key, builder)

        try:
            result = run_experiment(
                cfg, X0p, X1p, y0, y1, s0, s1, sf0, p0,
                idx_tr, idx_va, args, device, amp_enabled, seeds,
            )
        except Exception as e:
            # A full suite should continue if an optional model/classical dependency is unavailable.
            print(f"FAILED {cfg.name}: {type(e).__name__}: {e}")
            err = args.outdir / "failed_experiments.txt"
            with err.open("a", encoding="utf-8") as f:
                f.write(f"{cfg.name}\t{type(e).__name__}\t{e}\n")
            continue

        save_experiment_cache(cache_file, result)
        results.append(result)
        msg = f"done | val BAcc={result['val_metrics']['bacc']:.4f}"
        if result["test_metrics"] is not None:
            msg += f" | test BAcc={result['test_metrics']['bacc']:.4f}"
        msg += f" | {result['postprocess']}"
        print(msg)

    if not results:
        raise RuntimeError("All experiments failed. See failed_experiments.txt")

    # All configs share the same validation indices/order.
    ens = make_ensembles(results, yva, sva, y1, s1)
    all_results = results + ens
    rows = write_leaderboard(args.outdir / "leaderboard.csv", all_results)
    print_leaderboard(rows)

    # Save every probability array so you can build later ensembles without retraining.
    pred_arrays: dict[str, np.ndarray] = {
        "val_indices": idx_va,
        "val_y": yva,
        "val_subject": sva,
        "test_subject": s1,
        "class_names": np.asarray(CLASS_NAMES),
        "seeds": np.asarray(seeds, dtype=np.int64),
        "code_version": np.asarray(CODE_VERSION),
    }
    if y1 is not None:
        pred_arrays["test_y"] = y1
    for r in all_results:
        k = sanitize_name(r["name"])
        pred_arrays[f"val_prob__{k}"] = r["val_prob"]
        pred_arrays[f"test_prob__{k}"] = r["test_prob"]
    safe_npz(args.outdir / "all_predictions.npz", **pred_arrays)

    # For convenience: save the validation-ranked top method's test prediction.
    name_to_result = {r["name"]: r for r in all_results}
    best_name = rows[0]["name"]
    best = name_to_result[best_name]
    best_pred = best["test_prob"].argmax(axis=1).astype(np.int64)
    best_payload = {
        "method": np.asarray(best_name),
        "test_prob": best["test_prob"],
        "test_pred": best_pred,
        "test_subject": s1,
        "class_names": np.asarray(CLASS_NAMES),
    }
    if y1 is not None:
        best_payload["test_y"] = y1
        cm = confusion_matrix(y1, best_pred, labels=[0, 1, 2])
        best_payload["confusion_matrix"] = cm
        best_payload["per_class_recall"] = np.diag(cm) / np.maximum(cm.sum(axis=1), 1)
    safe_npz(args.outdir / "best_validation_ranked_prediction.npz", **best_payload)

    with (args.outdir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump({
            "code_version": CODE_VERSION,
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "seeds": seeds,
            "configs": [asdict(c) for c in configs],
            "selection_rule": "rank by session0 validation BAcc, tie by validation subject-macro BAcc; session1 labels report-only",
        }, f, indent=2, ensure_ascii=False)

    print(f"\nOutput: {args.outdir}")
    print(f"Best validation-ranked method: {best_name}")
    print("Do not choose a new method by session1 test BAcc if session1 is serving as your held-out proxy.")


if __name__ == "__main__":
    main()
