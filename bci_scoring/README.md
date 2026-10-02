# BCI-decoding track 本地评分代码（去 benchopt 版）

本目录复现 NeurIPS EEG Challenge 2026（[neural-interfaces26 / 2026-competition](https://github.com/neural-interfaces26/2026-competition)）
**BCI decoding track** 的官方评分逻辑，使其**不依赖 benchopt** 即可在本地独立运行。

任务："Cued mental-command classification from short EEG windows"——模型的
`predict(X)` 接收 torch batch `X`（shape `(B, C, T)`，batch × channels × time），
返回 `(B,)` 的整数类别预测。排名指标 = **balanced accuracy**
（sklearn `balanced_accuracy_score`），plain accuracy 同时报告。

## 目录结构

```
bci_scoring/
├── objective_original.py   # 官方 tracks/bci_decoding/objective.py 原文拷贝（需 benchopt，仅供对照）
├── local_scoring.py        # 去 benchopt 化的独立评分模块（核心交付物）
├── demo_scoring.py         # 端到端演示 + 自测（断言 + 对比表格）
├── requirements.txt        # 仅 numpy / scikit-learn / torch（CPU 版即可）
└── README.md               # 本文件
```

## 如何运行 demo

```bash
pip install -r requirements.txt   # 或：pip install numpy scikit-learn torch --index-url https://download.pytorch.org/whl/cpu
python demo_scoring.py
```

demo 会：生成 4 类 / 19 通道 / T=400 的合成数据（模拟 Stieger2021 的 4 类设定）；
评估一个"记忆类别模板"的最近质心分类器（balanced accuracy 应远高于随机的 1/4）；
评估官方同款 `ConstantClassifier`（balanced accuracy 应 ≈ 1/4）；
并断言 `score_predictions` 路径与 `evaluate_model` 路径、以及 sklearn 独立计算的
结果完全一致，最后打印对比表格。

## 如何接入自己的模型 / 预测（两种路径）

### 路径一：loader + model（与官方 `evaluate_result` 完全相同的流程）

```python
from local_scoring import evaluate_model, make_synthetic_loader

test_loader = make_synthetic_loader(
    n_samples=200, n_chans=19, n_times=400, n_classes=4,
    batch_size=32, seed=1)

class MyModel:
    def predict(self, X):            # X: torch tensor (B, C, T)
        ...                          # 返回 (B,) 整数类别（torch 或 numpy 均可）

result = evaluate_model(MyModel(), test_loader, n_classes=4)
# -> dict(balanced_accuracy=..., accuracy=..., n_classes=4)
```

`test_loader` 需产出 `(X, y, info)` 三元组（X 为 float32 tensor `(B, C, T)`，
y 为 int64 tensor `(B,)`）。若有真实 EEG 窗口，可用
`local_scoring._ArrayWindows` + `torch.utils.data.DataLoader` 按同样契约打包
（参见 `make_synthetic_loader` 的实现）。

### 路径二：已有预测标签，直接算分

```python
from local_scoring import score_predictions

result = score_predictions(y_true, y_pred, n_classes=4)
# y_true / y_pred：shape (N,) 的标签数组（torch 或 numpy 均可）
# 返回与路径一完全相同的 dict，数值与官方 evaluate_result 一致
```

## 与官方代码的对应关系

| 本目录 | 官方代码（2026-competition 仓库） |
|---|---|
| `objective_original.py` | `tracks/bci_decoding/objective.py`（原文拷贝，含 benchopt 依赖，仅供对照） |
| `local_scoring.to_numpy` | `benchmark_utils/data.py::to_numpy`（逐行一致） |
| `local_scoring.evaluate_model` | `tracks/bci_decoding/objective.py::Objective.evaluate_result`（逐行一致：同样的遍历、同样的 `np.concatenate` 拼接顺序、同样的两个指标与返回 dict） |
| `local_scoring.score_predictions` | `evaluate_result` 拼接完成后的纯指标计算部分（纯标签版本） |
| `local_scoring.ConstantClassifier` | `benchmark_utils/baselines.py::ConstantClassifier`（永远预测类别 0） |
| `local_scoring.make_synthetic_loader` | `tracks/bci_decoding/datasets/simulated.py` 的 `_make_windows`（模板 `rng.standard_normal((n_classes, C)) * 2.0`，标签 `rng.integers(0, n_classes)`，窗口 `X[i] = templates[y[i]][:, None] + rng.standard_normal((C, T))`）+ `benchmark_utils/data.py` 的 `ArrayWindows` / `make_loader` 打包方式（`(X, y, info)` 三元组、float32/int64、CPU）。训练/测试集共享类别模板时（与官方一致），传 `templates=train_loader.dataset.templates`（见 demo） |

**偏差说明**：无逻辑偏差。仅去除了 benchopt 框架依赖（`BaseObjective`、`set_data` / `get_objective` 等框架钩子），
设备固定为 CPU（官方通过 `get_device()` 自动选择 cuda/cpu，对指标数值无影响）。
