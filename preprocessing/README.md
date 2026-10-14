# EEG 预处理：NeurIPS Competition 2026

本目录保存 warm-up 阶段的连续 EEG 预处理代码，使用 MNE 完成坏道插值、滤波、重参考、ICA 检查及伪迹重建。新版将 ICA 建模与绘图分开，并增加按上游缓存标识恢复处理的机制，变更记录见 [CHANGELOG.md](CHANGELOG.md)。

导出副本保留当前源码、方法参数配置和运行所需目录。原始 MAT、已有缓存、结果图片、虚拟环境及资料文件不随代码发布；空目录通过 `.gitkeep` 保留。两个人工标注 JSON 保留全部 session 键，但列表均清空，需结合自己的数据重新填写。

## 1. 安装与快速开始

使用 Python 3.12；整理时环境为 Python 3.12.13。核心依赖固定为 MNE 1.13.2、NumPy 2.5.3、SciPy 1.18.1、Matplotlib 3.11.2、python-picard 0.8.2，记录在 [`requirements-core.txt`](requirements-core.txt)。

在仓库根目录打开 PowerShell：

```powershell
cd preprocessing
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
```

Linux / macOS 可将上述虚拟环境解释器路径改为 `.venv/bin/python`。完整的 [`requirements.txt`](requirements.txt) 保留原实验环境的依赖，包含训练、下载及 Windows 工具等额外软件；只运行本目录的预处理时安装核心清单即可。

按第 3 节放好 MAT 后，先查看原始波形：

```powershell
.\.venv\Scripts\python.exe run_preprocessing.py --stage waveforms
```

根据波形填写坏道列表，再生成 ICA 地形图：

```powershell
.\.venv\Scripts\python.exe run_preprocessing.py --stage ica
```

首次运行会处理选定的全部被试和 session，ICA 拟合可能耗时。所有图片都写入磁盘；代码使用 `Agg` 后端，不弹出交互式标注窗口。

## 2. 目录与入口

```text
preprocessing/
├── run_preprocessing.py              # 发布副本的分阶段运行入口
├── src/
│   ├── Preprocessing.py              # 主版本：步骤字典、缓存标识及 start 恢复
│   ├── preprocessing_config_version.py # 独立的配置实验版本
│   └── preprocessing/__init__.py     # 现有包入口占位函数
├── configs/
│   ├── config.py                     # 仅供配置实验版本使用的方法参数
│   ├── bad_channels_exclude.json     # 各 session 的坏道名单
│   └── ica_exclude.json              # ICA 候选及最终排除名单
├── data/                             # 自行放入 A.mat、C.mat 等
├── cache/
│   ├── cache_v2/                     # 主版本的分版本缓存
│   │   ├── data/
│   │   ├── raw/                     # 预留，当前主流程未使用
│   │   ├── raw_interp/
│   │   ├── raw_pre_filter/
│   │   ├── raw_referenced/
│   │   ├── ica/
│   │   ├── raw_cleaned/
│   │   └── raw_after_filter/         # 初始声明目录；实际输出见下一项
│   ├── raw_after_filter_4-40/         # 主版本当前实际后滤波路径
│   ├── data/                         # 配置实验版本的 MAT 缓存
│   └── config_version/               # 配置实验版本按参数编号隔离的缓存
├── results/
│   ├── waveforms/                    # 主版本原始波形
│   ├── components/                   # 主版本 ICA 地形图
│   ├── properties/                   # 主版本候选 IC 诊断图，按 session 分目录
│   └── config_version/               # 配置实验版本的图和参数快照
├── assets/                           # 资料目录占位
├── CHANGELOG.md
├── requirements-core.txt
├── requirements.txt
└── pyproject.toml
```

缓存和结果的动态子目录在需要时自动创建。安装包后注册的 `preprocessing` 命令目前只打印 `Hello from preprocessing!`；实际运行请使用本文的脚本命令。

主版本的 `./data`、`./configs`、`./cache`、`./results` 相对于工作目录。辅助入口 `run_preprocessing.py` 会自动切换到本项目目录；直接运行或导入 `src/Preprocessing.py` 时，需要自行保证工作目录为 `preprocessing/`。配置实验版本以脚本位置定位项目目录。

## 3. 输入数据与人工标注

默认读取以下 9 个文件，须全部存在；想先处理少量数据，可修改对应版本源码中的 `subjects`：

```text
data/A.mat  data/C.mat  data/D.mat  data/E.mat  data/F.mat
data/G.mat  data/H.mat  data/J.mat  data/L.mat
```

代码不会下载数据。MAT 有效内容位于 `data` 字段，使用 `scipy.io.loadmat(..., simplify_cells=True)` 读取后应为可遍历的 session 序列，每个 session 为字典：

| 字段 | 当前代码的约定 |
| --- | --- |
| `X` | 连续电压，原始形状为 `(采样点, 30 个通道)`；转置为通道 × 采样点，再乘 `1e-6`，将 µV 转为 V |
| `trial` | trial 起始采样点；从 MATLAB 1-based 减 1 转为 Python 0-based |
| `y` | trial 标签，转为 `int16`；标签数值不减 1 |
| `fs` | 原采样率字段保留；构建 Raw 实际使用代码中的 `sfreq=256`，不会自动采用该字段 |
| `classes`、`session` | 类别和会话信息，保留在 MAT 读取结果中 |

30 个通道的顺序由源码 `ch_names` 指定，通道类型均为 EEG，电极布局为 `colin27_1005`。换数据时必须核对电压单位、矩阵方向、通道顺序及真实采样率。若单 session 文件被 `simplify_cells=True` 简化成单个字典，需先适配读取器。

Raw 和 JSON 使用被试名加 **0-based session 遍历序号**作为键，如 `A0`、`A1`；这些键不直接采用 MAT 中的 `session` 值。当前模板覆盖默认被试的两个 session；数据 session 数量变化时同步补全键。

[`configs/bad_channels_exclude.json`](configs/bad_channels_exclude.json) 保存坏道名称，例如：

```json
{
  "A0": ["F7", "T3"],
  "A1": []
}
```

上例仅展示部分键，使用时保留其他 session。名称须与 `ch_names` 一致；插值会恢复坏道并保留通道数量。空列表表示未指定坏道，缺少 session 键会报错。

[`configs/ica_exclude.json`](configs/ica_exclude.json) 分两层保存人工判断：

```json
{
  "topomap_exclude": {"A0": [0, 3], "A1": []},
  "final_exclude": {"A0": [3], "A1": []}
}
```

`topomap_exclude` 为地形图中初步选出的候选 IC，决定 properties 阶段画哪些成分；`final_exclude` 为进一步确认后实际排除的 IC。索引从 0 开始，必须小于对应模型的 `ica.n_components_`。候选列表不会自动成为最终列表。

发布模板的所有列表都为空。此时 properties 阶段会跳过各 session；clean 阶段仍会重建并保存，但不会排除任何 IC。空模板不表示数据已完成质量检查。调整坏道、滤波、参考、ICA 方法或输入后，应重新读图确认 IC 索引，不要沿用此前模型的标注。

## 4. 主版本：按阶段运行

[`run_preprocessing.py`](run_preprocessing.py) 只负责选择步骤、设置恢复起点和工作目录，实际处理仍调用 [`src/Preprocessing.py`](src/Preprocessing.py) 的 `main`。

| `--stage` | 开启的处理步骤 | 输出 |
| --- | --- | --- |
| `waveforms` | 读 MAT → 构建 Raw → 画原始波形 | `results/waveforms/` |
| `ica`，默认 | 读 MAT → 构建 Raw → 坏道插值 → 前滤波 → 重参考 → ICA → 地形图 | `results/components/` 和各阶段缓存 |
| `properties` | 与 `ica` 相同的上游和 ICA → 候选 IC properties 图 | `results/properties/<session>/` |
| `clean` | 与 `ica` 相同的上游和 ICA → 排除最终 IC → 后滤波 | cleaned 和后滤波 FIF |

推荐依次执行：

```powershell
# 1. 查看原始波形，然后编辑 bad_channels_exclude.json
.\.venv\Scripts\python.exe run_preprocessing.py --stage waveforms

# 2. 查看 ICA 地形图，然后填写 topomap_exclude
.\.venv\Scripts\python.exe run_preprocessing.py --stage ica

# 3. 查看候选成分诊断图，然后填写 final_exclude
.\.venv\Scripts\python.exe run_preprocessing.py --stage properties

# 4. 排除确认的伪迹并生成最终带通信号
.\.venv\Scripts\python.exe run_preprocessing.py --stage clean
```

默认从 `is_load_cached_mat` 开始。已有缓存且参数和标识匹配时会复用，缺失或失效时重新处理；不需要手动删除全部缓存来运行下一阶段。

源码直接入口 `python src/Preprocessing.py` 当前预设 `start="is_ica_cache"`，用于恢复已有 ICA 缓存，且开启 properties、关闭地形图和重建。**新克隆没有缓存时，请使用辅助入口；直接执行源码默认入口会因缓存缺失而失败。**

### 恢复到中间步骤

例如，已完成 ICA 后只画 properties：

```powershell
.\.venv\Scripts\python.exe run_preprocessing.py --stage properties --start is_ica_cache
```

`--start` 接受源码的步骤键。该键必须在选定阶段中开启；它之前的全部步骤会被关闭。有独立处理缓存的中间起点会读取本步已有缓存，在比较本步参数及 `version` 时忽略上游缓存标识；若所需键、标识或数据缺失、失效，则报错。`is_build_valid_raw` 和 `is_view_waveforms` 起点使用已有 MAT 缓存，不重新读原始文件。两个 ICA 绘图起点直接使用已有模型，不重新验证 ICA 方法参数及版本，使用前需确认模型仍适用。ICA 及绘图起点还要求能找到模型对应的上游 Raw 或 MAT 缓存。

常用步骤键为 `is_load_cached_mat`、`is_build_valid_raw`、`is_view_waveforms`、`is_exclude_bads_and_interpolate`、`is_pre_filter`、`is_set_reference`、`is_ica_cache`、`is_gen_components_topomap`、`is_gen_properties`、`is_mark_artifacts_and_build`、`is_after_filter`。

首次运行、调整处理参数或人工标注后，使用默认起点让依赖链重新检查。尤其不要把需要重新应用的清理步骤设为缓存恢复起点：恢复模式依赖已经存在且匹配的结果。

### 参数修改位置

| 目标 | 主版本中修改的位置与当前值 |
| --- | --- |
| 被试、通道、采样率、布局、缓存版本 | `src/Preprocessing.py` 顶部；`sfreq=256`、`version=2` |
| 输入转换 | `main()` 调用 `load_cached_mat` 的参数；转置、trial 减 1、缩放 `1e-6` |
| 插值、滤波、参考、并行设置 | `main()` 中相应函数调用；1 Hz 前高通、平均参考、4–40 Hz 后带通，前后陷波均关闭 |
| ICA 方法 | `ica_cache()` 中的 `mne.preprocessing.ICA(...)`；Picard、`rng=1`、`fit_params={"extended": True, "ortho": False}`；`decim=3` 在 `main()` 调用处 |
| properties 连续波形范围 | `main()` 调用 `gen_properties`；`t_start_s=0.0`、`duration=10.0` |

主版本不读取 `configs/config.py`。只修改函数定义的默认值不一定改变运行行为，因为 `main()` 多数会显式传参。`decim` 减少 ICA 拟合采样点，不改变 Raw 的采样率。启用陷波需同时核对频率与真实采样率；后滤波当前 `notch_freqs=None` 不能直接配合默认 FIR 陷波使用。

### 缓存与最终输出

主版本为各阶段保存缓存键、缓存标识和数据，上游重新计算会产生新标识并传给下游。坏道和最终 IC 标注文件的修改时间也参与对应阶段的键比较。ICA 建模与 Raw 的缓存标识分开，绘图使用已有模型。

缓存不会自动比较 MAT 文件的内容、大小或修改时间；替换同路径 MAT、修改通道信息或算法实现后，应递增顶部 `version`，从头检查所需阶段。主版本 ICA 的固定方法参数并未全部进入键，改 ICA 方法同样应递增版本并重新确认标注。中间恢复起点按本步缓存工作，不能用于验证上游的新设置。

当前清理结果为 `cache/cache_v2/raw_cleaned/<session>_cleaned_raw.fif`；4–40 Hz 后滤波结果位于 `cache/raw_after_filter_4-40/<session>_pre_filter_raw.fif`。后者复用统一滤波函数，因此文件名仍带 `pre_filter`，且当前路径由 `main()` 单独指定，未放在 `cache_v2/` 内；更改后滤波频段时请同步核对目录名。波形、地形图和 properties 开启后重新绘制，关闭不会删除旧图片。

`main()` 返回最终 Raw 字典，但代码不自动切 trial、提取特征或生成训练提交文件。trial 起点与标签仍来自 MAT 读取结果；下游训练须自行对齐分段。

## 5. 独立的配置实验版本

[`src/preprocessing_config_version.py`](src/preprocessing_config_version.py) 将插值、前滤波、参考、ICA 和后滤波的方法参数放在 [`configs/config.py`](configs/config.py)，并按实验参数隔离缓存、记录快照。它与主版本的缓存结构不同，不能混用缓存文件。

```powershell
.\.venv\Scripts\python.exe src/preprocessing_config_version.py
```

**该脚本当前默认只开启 ICA 缓存和成分地形图；坏道插值、前滤波、重参考、properties、伪迹重建和后滤波均关闭。** 即使 `config.py` 中写有滤波频率，也不会自动开启对应步骤。流程开关、滤波子步骤开关、路径、被试、并行度及版本在该脚本的 `main()` 内修改；它不使用辅助入口的 `--stage`。

此版本的 MAT 缓存位于 `cache/data/`；上游插值、滤波、参考及 ICA 位于 `cache/config_version/ica_<upstream_id>/`；清理和后滤波位于 `cache/config_version/run_<experiment_id>/`。图与快照位于 `results/config_version/<experiment_id>/`，快照文件为 `config_snapshot.json`。

参数、流程设置、坏道内容、数据约定、MNE 版本等决定上游编号；后滤波及最终 IC 列表决定运行编号，单独修改后滤波或最终 IC 列表可复用同一上游 ICA。绘图本身不作为信号处理缓存的依据。替换同路径 MAT 或修改算法实现仍需增加本版本 `main()` 中的 `version`。

恢复历史方法参数的示例：

```python
import json
from pathlib import Path
from src.preprocessing_config_version import main

snapshot = json.loads(Path("results/config_version/<实验编号>/config_snapshot.json").read_text(encoding="utf-8"))
raw_dic = main(config=snapshot["config"])
```

此调用只恢复 `config` 中的方法参数；流程开关、数据约定及人工标注仍以当前代码和 JSON 文件为准。快照也保存了这些信息，完整复现实验时需一并核对和恢复。该版本 `main()` 同样返回 Raw 字典；properties、地形图和重建只读取已建立的 ICA 模型，因此首次运行或更改上游时应保持 `is_ica_cache=True`。

## 6. 常见问题

- 找不到 MAT：检查 `data/`、`subjects` 以及工作目录；主版本推荐使用辅助入口。
- 提示起点缓存无法读取：新环境先省略 `--start`，完整运行相应阶段；旧版缓存没有当前所需标识时也应重新生成。
- properties 没有图：模板候选列表为空，先看地形图并填写 `topomap_exclude`。
- 没有清理后的 FIF：运行 `--stage clean`，填写 `final_exclude`；仅画图不会生成最终信号。
- 调整数据后结果似乎没变：缓存不追踪原始数据内容，增加对应版本的 `version` 后从头运行；也检查是否仍在看旧结果目录。
- 数据和生成文件无法提交到 Git：发布副本的 `.gitignore` 排除了数据、缓存、图和资料，保留目录占位符及需要共享的源码和配置。

## 7. 本次发布验证

已在 Python 3.12.13 环境使用隔离的合成 MAT / EEG 数据验证：分阶段入口、主版本从零处理及缓存恢复、缓存版本失效、插值与前后滤波、ICA 建模与人工排除、波形及诊断绘图，以及配置版 9 被试 / 18 session 的参数快照和缓存复用。原始源码与方法配置按 SHA-256 核对一致。本次验证没有运行真实数据的全量处理，也不评估解码性能；实际数据仍需人工检查坏道和 ICA。
