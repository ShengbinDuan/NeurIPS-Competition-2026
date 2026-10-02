# -*- coding: utf-8 -*-
"""去 benchopt 化的 BCI-decoding track 本地独立评分模块。

本模块忠实复现 NeurIPS EEG Challenge 2026（neural-interfaces26 /
2026-competition 仓库）BCI decoding track 的官方评分逻辑，使其可以在
不安装 benchopt 的情况下本地独立运行。

与官方代码的对应关系
--------------------
- ``to_numpy``            <- 官方 ``benchmark_utils/data.py::to_numpy``
                             （torch tensor / array-like -> numpy 的转换工具，
                             仅在 sklearn 边界处使用）。
- ``evaluate_model``      <- 官方 ``tracks/bci_decoding/objective.py::
                             Objective.evaluate_result``。逐行复现：遍历
                             test_loader 的 ``(X, y, info)`` 三元组，分别
                             收集 ``to_numpy(model.predict(X))`` 与
                             ``to_numpy(y)``，用 ``np.concatenate`` 按同样的
                             顺序拼接，再用 sklearn 计算
                             ``balanced_accuracy_score``（排名指标）与
                             ``accuracy_score``（附带报告），返回同样的 dict
                             （含 ``n_classes``）。
- ``score_predictions``   <- 官方 ``evaluate_result`` 中拼接完成之后的纯
                             指标计算部分。当用户已经有预测标签、没有模型或
                             loader 时，直接对 ``y_true / y_pred`` 算分，
                             返回与 ``evaluate_model`` 完全相同的 dict。
- ``ConstantClassifier``  <- 官方 ``benchmark_utils/baselines.py::
                             ConstantClassifier``（永远预测类别 0，官方
                             ``get_one_result`` 用的 sanity-check 模型）。
- ``make_synthetic_loader`` <- 官方 ``tracks/bci_decoding/datasets/
                             simulated.py``（Simulated 数据集）的数据生成
                             方式 + ``benchmark_utils/data.py::make_loader``
                             的打包方式：每类一个 channel-mean 模板
                             ``templates[k]``（shape ``(C,)``，由
                             ``rng.standard_normal((n_classes, C)) * 2.0``
                             生成），标签 ``y ~ rng.integers(0, n_classes)``，
                             每个窗口
                             ``X[i] = templates[y[i]][:, None] +
                             rng.standard_normal((C, T))``；
                             返回产出 ``(X, y, info)`` 三元组的
                             ``torch.utils.data.DataLoader``
                             （X 为 float32、y 为 int64、均在 CPU）。
"""

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset, default_collate

__all__ = [
    "to_numpy",
    "evaluate_model",
    "score_predictions",
    "ConstantClassifier",
    "make_synthetic_loader",
]


def to_numpy(x):
    """Convert a torch tensor (or array-like) to a numpy array.

    Used only at the scikit-learn boundaries — keep torch elsewhere.

    （与官方 ``benchmark_utils/data.py::to_numpy`` 完全一致。）
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def evaluate_model(model, test_loader, n_classes):
    """Evaluate ``model`` on ``test_loader`` — 复现官方 ``evaluate_result``。

    Parameters
    ----------
    model : object with ``predict(X)``
        接收 torch batch ``X``（shape ``(B, C, T)``），返回 ``(B,)`` 的
        整数类别预测（torch tensor 或 numpy 数组均可）。
    test_loader : iterable of ``(X, y, info)``
        X 为 float32 tensor ``(B, C, T)``，y 为 int64 tensor ``(B,)``。
    n_classes : int
        类别数（原样放入返回 dict）。

    Returns
    -------
    dict
        ``balanced_accuracy``（排名指标）、``accuracy``（附带报告）、
        ``n_classes``——与官方 ``evaluate_result`` 的返回完全一致。
    """
    y_true, y_pred = [], []
    for X, y, _info in test_loader:
        y_pred.append(to_numpy(model.predict(X)))
        y_true.append(to_numpy(y))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    return dict(
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        accuracy=accuracy_score(y_true, y_pred),
        n_classes=n_classes,
    )


def score_predictions(y_true, y_pred, n_classes):
    """Score ready-made predictions — 官方 ``evaluate_result`` 的纯标签版本。

    当用户已经有预测结果（没有模型 / loader）时使用。与
    :func:`evaluate_model` 中拼接后的计算完全相同。

    Parameters
    ----------
    y_true, y_pred : array-like, shape ``(N,)``
        真实标签与预测标签（torch tensor 或 numpy 数组均可）。
    n_classes : int
        类别数（原样放入返回 dict）。

    Returns
    -------
    dict
        与 :func:`evaluate_model` 相同结构的 dict。
    """
    y_true = to_numpy(y_true)
    y_pred = to_numpy(y_pred)

    return dict(
        balanced_accuracy=balanced_accuracy_score(y_true, y_pred),
        accuracy=accuracy_score(y_true, y_pred),
        n_classes=n_classes,
    )


class ConstantClassifier:
    """Always predict class 0 — one label per window.

    （与官方 ``benchmark_utils/baselines.py::ConstantClassifier`` 完全一致，
    用于 sanity check：其 balanced accuracy 应约为 ``1 / n_classes``。）
    """

    def fit(self, train_loader):
        return self

    def predict(self, X):
        return np.zeros(len(to_numpy(X)), dtype=np.int64)


class _ArrayWindows(Dataset):
    """In-memory windows dataset yielding ``(X, y, info)`` torch tensors.

    （对应官方 ``benchmark_utils/data.py::ArrayWindows``：X 存 float32，
    整数标签存 long，info 含 ``record_id`` 与 ``onset``。）
    """

    def __init__(self, X, y, record_id=None, onset=None):
        self.X = torch.as_tensor(to_numpy(X), dtype=torch.float32)
        self.y = torch.as_tensor(to_numpy(y)).to(torch.long)
        n = len(self.X)
        self.record_id = (
            np.zeros(n, dtype=np.int64) if record_id is None
            else np.asarray(record_id, dtype=np.int64)
        )
        self.onset = (
            np.arange(n, dtype=np.int64) if onset is None
            else np.asarray(onset, dtype=np.int64)
        )

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        info = {
            "record_id": int(self.record_id[i]),
            "onset": int(self.onset[i]),
        }
        return self.X[i], self.y[i], info


def _cpu_collate(batch):
    """Default collate, then keep signal/labels on CPU (info stays on CPU).

    （对应官方 ``benchmark_utils/data.py::_move_collate("cpu")``。）
    """
    X, y, info = default_collate(batch)
    return X.to("cpu"), y.to("cpu"), info


def make_synthetic_loader(n_samples, n_chans, n_times, n_classes,
                          batch_size=32, seed=0, templates=None):
    """Build a simulated test/train loader — 复现官方 Simulated 数据集。

    生成方式与官方 ``tracks/bci_decoding/datasets/simulated.py`` 一致：

    - 模板：``templates = rng.standard_normal((n_classes, n_chans)) * 2.0``；
    - 标签：``y = rng.integers(0, n_classes, size=n_samples)``（均匀随机）；
    - 窗口：``X[i] = templates[y[i]][:, None] +
      rng.standard_normal((n_chans, n_times))``。

    Parameters
    ----------
    n_samples : int
        窗口数量。
    n_chans, n_times : int
        每个窗口的通道数 C 与时间点数 T（X 的 shape 为 ``(N, C, T)``）。
    n_classes : int
        类别数。
    batch_size : int
        DataLoader 的 batch 大小。
    seed : int
        随机种子（``np.random.default_rng(seed)``）。
    templates : array-like, shape ``(n_classes, n_chans)``, optional
        类别模板。默认 ``None`` 时用 ``rng`` 按官方方式生成；传入已有模板
        可让训练集与测试集共享同一组模板（与官方 Simulated 数据集中
        train/test 共用 ``templates`` 的行为一致）。生成/使用的模板同时
        挂在返回 loader 的 ``dataset.templates`` 上，便于复用。

    Returns
    -------
    torch.utils.data.DataLoader
        产出 ``(X, y, info)`` 三元组：X 为 float32 tensor ``(B, C, T)``，
        y 为 int64 tensor ``(B,)``，info 为 ``{"record_id", "onset"}`` 的
        dict（batch 后为 tensor 的 dict），全部在 CPU。
    """
    rng = np.random.default_rng(seed)
    # One channel-mean template per class; scaled so a linear model can
    # separate the classes above the noise floor.（与官方一致）
    if templates is None:
        templates = rng.standard_normal((n_classes, n_chans)) * 2.0
    else:
        templates = np.asarray(templates, dtype=np.float64)

    X = np.empty((n_samples, n_chans, n_times), dtype=np.float32)
    y = rng.integers(0, n_classes, size=n_samples)
    for i, k in enumerate(y):
        mean = templates[k][:, None]
        X[i] = mean + rng.standard_normal((n_chans, n_times))

    dataset = _ArrayWindows(X, y.astype(np.int64))
    dataset.templates = templates  # 便于训练/测试集共享同一组模板
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=_cpu_collate,
    )
