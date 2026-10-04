# EEG 预处理：NeurIPS Competition 2026

本目录保存 warm-up 阶段的连续 EEG 预处理工程。主程序是 [`src/Preprocessing.py`](src/Preprocessing.py)，使用 MNE 完成坏道插值、滤波、重参考、ICA 检查和伪迹重建，通过 `True` / `False` 控制各个阶段。

发布副本保留原工程的目录层级及 `src` 源码。`data`、`cache`、`results`、`assets` 和 `rubbish` 中不包含原始数据、缓存、图片、文档或历史脚本；`configs` 中保留两个 JSON 的键和嵌套结构，但所有列表都已清空。Git 用零字节 `.gitkeep` 保存空文件夹。虚拟环境、`__pycache__`、临时目录和 `tests` 不随本工程发布。

## 1. 安装与运行位置

使用 Python 3.12 或更新版本。工程的 `.python-version` 指定 3.12；整理时使用的环境为 Python 3.12.13、MNE 1.13.2、NumPy 2.5.3、SciPy 1.18.1、Matplotlib 3.11.2 和 python-picard 0.8.2。

在仓库根目录执行：

```powershell
cd preprocessing
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "mne>=1.13.2" "numpy>=2.5.3" "scipy>=1.18.1" "matplotlib>=3.11.2" "python-picard>=0.8.2"
```

上述依赖对应 `pyproject.toml` 中的预处理依赖。`requirements.txt` 另保留了原实验环境的完整依赖清单，包含训练、下载等额外工具；需要复现完整环境时可执行 `python -m pip install -r requirements.txt`。

准备好下一节的数据和配置后运行：

```powershell
.\.venv\Scripts\python.exe src/Preprocessing.py
```

**工作目录必须是本目录 `preprocessing/`**。代码中的 `./data`、`./configs`、`./cache`、`./results` 都相对于当前工作目录，而不是脚本所在位置。导入模块也会创建必要的缓存和结果目录。

当前 `pyproject.toml` 注册的 `preprocessing` 命令指向 `src/preprocessing/__init__.py`，该函数只打印 `Hello from preprocessing!`。实际预处理请使用上面的脚本入口，或导入 `src.Preprocessing.main`。Matplotlib 使用 `Agg` 后端，图片写到磁盘，不弹出交互式标注窗口。

## 2. 数据文件怎么放

主要目录如下；已有的多级资料与结果文件夹也保留在仓库中：

```text
preprocessing/
├── src/
│   ├── Preprocessing.py          # 真实的工作流与全部处理函数
│   └── preprocessing/__init__.py
├── configs/
│   ├── bad_channels_exclude.json
│   └── ica_exclude.json
├── data/                        # 自行放入 A.mat、C.mat 等
├── cache/
│   ├── data/
│   ├── raw/                     # 当前程序未使用的预留目录
│   ├── raw_interp/
│   ├── raw_pre_filter/
│   ├── raw_referenced/
│   ├── ica/
│   ├── raw_cleaned/
│   └── raw_after_filter/
├── results/
│   ├── waveforms/
│   ├── components/
│   ├── properties/              # 再按 A0、A1 等 session 分目录
│   ├── storage/
│   ├── waveforms_larger/
│   └── waveforms_smaller/
├── assets/                      # 保留资料归档目录层级
├── rubbish/                     # 保留空目录
├── .python-version
├── pyproject.toml
└── requirements.txt
```

默认 `subjects = ["A", "C", "D", "E", "F", "G", "H", "J", "L"]`，因此将以下文件直接放在 `data/`，不要再套一层被试文件夹：

```text
data/A.mat  data/C.mat  data/D.mat  data/E.mat  data/F.mat
data/G.mat  data/H.mat  data/J.mat  data/L.mat
```

只处理部分被试时，修改 `src/Preprocessing.py` 顶部的 `subjects`，例如 `subjects = ["A", "C"]`。选中的文件均需存在；仓库不含数据，也没有自动下载过程。

每个 MAT 文件的有效内容在 `data` 字段中，按 session 存放结构，包含：

| 字段 | 含义与代码处理 |
| --- | --- |
| `X` | 连续电压；当前调用假定原始形状为 `(采样点, 30 个通道)`，转置成 `(通道, 采样点)`，再乘 `1e-6`，把 µV 转成 V |
| `trial` | trial 起点的采样点索引；当前调用减 1，将 MATLAB 的 1-based 转成 Python 的 0-based |
| `y` | trial 标签，转成 `int16`，标签数值不做减 1 |
| `fs` | 原文件采样率；构建 Raw 时实际使用脚本顶部的 `sfreq`，不会自动采用此字段 |
| `classes`、`session` | 类别说明和原 session 信息；保留在读取结果中 |

`loadmat(..., simplify_cells=True)` 后，`data` 必须能被当前读取循环当作 session 序列处理。此读取器面向现有 MAT 结构；其他数据结构，尤其被简化成单个字典的单 session 文件，需要先适配。

默认通道顺序由顶部 `ch_names` 的 30 个名称决定，`ch_types` 全部为 `eeg`，`sfreq=256`，`montage_name="colin27_1005"`。通道顺序必须与 `X` 的列对应。换数据时同时核对单位、形状、采样率和电极布局。

Raw 字典和配置使用 **被试名 + session 的 0-based 序号**，例如 `A0`、`A1`。这些键依据遍历顺序生成，不直接读取 MAT 的 `session` 字段。

### 两个 JSON 怎么填

`configs/bad_channels_exclude.json` 每个 session 对应一个坏道名称列表。示例仅展示一个键；使用时保留全部需要处理的 session 键：

```json
{
  "A0": ["F7", "T3"],
  "A1": []
}
```

`[]` 表示该 session 暂未标记坏道；键缺失会触发 `KeyError`。通道名称必须与 `ch_names` 一致。插值保留通道数量，不是删除通道。

`configs/ica_exclude.json` 有两层标注：

```json
{
  "topomap_exclude": {
    "A0": [0, 3],
    "A1": []
  },
  "final_exclude": {
    "A0": [3],
    "A1": []
  }
}
```

`topomap_exclude` 是从成分地形图选出的候选 IC，决定绘制哪些 properties 图；`final_exclude` 是人工确认后实际排除的 IC。**IC 索引从 0 开始**，且应小于该 session 的 `ica.n_components_`。候选列表不会自动转成最终列表。

发布模板中所有列表均为 `[]`：properties 阶段会逐个跳过没有候选 IC 的 session；重建阶段仍会加载或拟合 ICA 并写出文件，但不排除任何 IC。这些空列表是待填写模板，不代表已经完成质量检查。

## 3. 用 True / False 控制工作流

实际执行顺序固定为：

```text
读取 MAT / 命中数据缓存 → 构建 Raw
  → 原始波形图（可选）
  → 坏道插值（可选）
  → 前滤波（可选）
  → 重参考（可选）
  → ICA 成分地形图（可选）
  → 候选 IC properties 图（可选）
  → 排除最终 IC 并重建（可选）
  → 后滤波（可选）
```

修改文件末尾 `if __name__ == '__main__':` 下的 `main(...)` 控制直接运行脚本时的行为。也可以在本目录调用：

```python
from src.Preprocessing import main

main(
    is_view_waveforms=False,
    is_exclude_bads_and_interpolate=True,
    is_pre_filter=True,
    is_set_reference=True,
    is_gen_components_topomap=True,
    is_gen_properties=False,
    is_mark_artifacts_and_build=False,
    is_after_filter=False,
)
```

`main` 的函数默认值是八个开关全部 `False`；文件末尾显式传入的值不同。当前源码的脚本入口开启 **坏道插值、前滤波、重参考、properties 图**，其余关闭。清空配置后，properties 不会生成图；想首次检查 ICA，请按上面的调用开启成分地形图。

| 开关 | `True` | `False` | 当前脚本入口 |
| --- | --- | --- | --- |
| `is_view_waveforms` | 将原始 Raw 波形图写到 `results/waveforms/` | 不画图 | `False` |
| `is_exclude_bads_and_interpolate` | 执行插值函数；命中对应缓存则加载插值后的 Raw | 不调用函数，也不读插值缓存 | `True` |
| `is_pre_filter` | 调用前滤波；当前内部设置为 1 Hz 高通、不开陷波 | 不读前滤波缓存，继续传递当前 Raw | `True` |
| `is_set_reference` | 重参考；当前为平均参考，可命中缓存 | 跳过重参考及其缓存 | `True` |
| `is_gen_components_topomap` | 加载或拟合 ICA，重新绘制全部 IC 地形图到 `results/components/` | 不画地形图；其他 ICA 阶段仍可独立执行 | `False` |
| `is_gen_properties` | 对候选列表非空的 session 加载或拟合 ICA，并画 properties 图 | 不画 properties 图 | `True` |
| `is_mark_artifacts_and_build` | 加载或拟合 ICA，用 `final_exclude` 重建并写入 `cache/raw_cleaned/` | 不重建，也不加载已有 cleaned 文件 | `False` |
| `is_after_filter` | 调用后滤波；当前为 8–30 Hz 带通，可命中缓存 | 不读后滤波缓存，也不生成该阶段输出 | `False` |

读取 MAT 和构建 Raw 始终执行。全部 `False` 仍会读取数据或数据缓存并构建 Raw，但不会拟合 ICA、绘图或自动写出最终 Raw；`main()` 当前没有返回值，也不输出 trial 分段、特征或训练数据。

### 分阶段操作示例

以下各阶段都从 MAT/数据缓存开始，再按开关执行；上游处理开关和参数应保持一致：

| 目标 | 波形 | 插值 | 前滤波 | 参考 | 地形图 | properties | 重建 | 后滤波 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 先看原始波形、填写坏道 JSON | True | False | False | False | False | False | False | False |
| 插值及预处理后，检查 ICA 地形图 | False | True | True | True | True | False | False | False |
| 填好候选列表后，检查候选 IC | False | True | True | True | False | True | False | False |
| 填好最终列表后，重建及后滤波 | False | True | True | True | False | False | True | True |

操作顺序：看波形 → 填坏道列表 → 使受影响的缓存失效 → 看 ICA 地形图 → 填候选列表 → 看 properties → 填最终列表 → 重建和后滤波。生成图不需要先运行另一个绘图阶段；三个 ICA 函数都能自己加载或拟合 ICA。

前后滤波还有函数内部的 `is_filter` 和 `is_notch_filter`：

| `is_filter` | `is_notch_filter` | 当前阶段效果 |
| --- | --- | --- |
| False | False | 直接返回输入 Raw，不查缓存、不写缓存 |
| True | False | 仅执行 `filter`，或读取对应参数的缓存 |
| False | True | 仅执行 `notch_filter`，或读取对应参数的缓存 |
| True | True | 未命中缓存时先陷波，再执行 `filter` |

外层 `is_pre_filter` / `is_after_filter=False` 时，内部开关不会被执行。布尔值须使用 Python 的 `True` / `False`，不要写成字符串 `"False"`。

## 4. 缓存如何运作，以及开关对缓存的影响

### 命中规则

`_load_cache` 将本阶段的处理参数与 `version` 组成缓存键，和 `.key.pkl` 中的键比较。**缓存数据文件和键文件都存在，且键相同，才视为命中**。未命中则重新计算，并覆盖同名缓存。

当前机制不比较输入 Raw 的数据、通道信息、annotations，也不比较 MAT 或 JSON 的文件内容、大小和修改时间。它不是自动追踪整条工作流依赖的缓存系统。

| 阶段 | 文件 / 目录 | 缓存键主要内容 |
| --- | --- | --- |
| MAT 读取 | `cache/data/scherer_read_mat.pkl` + `scherer_read_mat.key.pkl` | 数据目录绝对路径、被试列表及顺序、索引/形状转换、缩放系数、版本 |
| 坏道插值 | `cache/raw_interp/{session}_interp.fif` + `.key.pkl` | 坏道 JSON 的绝对路径、`reset_bads`、`method`、`mode`、版本；不含 JSON 内容 |
| 前滤波 | `cache/raw_pre_filter/{session}_pre_filter_raw.fif` + `{session}_pre_filter.key.pkl` | 两个内部开关、滤波及陷波参数、版本；不含输入 Raw |
| 重参考 | `cache/raw_referenced/{session}_referenced.fif` + `.key.pkl` | 参考方式、通道类型、`joint`、REST 参数、版本 |
| ICA | `cache/ica/{session}_ica.fif` + `.key.pkl` | **仅 `decim` 和版本**，三个 ICA 函数共用同一文件 |
| ICA 重建输出 | `cache/raw_cleaned/{session}_cleaned.fif` | **没有键文件、没有命中判断**；每次开启重建都重新 apply 并覆盖写出 |
| 后滤波 | `cache/raw_after_filter/{session}_after_filter_raw.fif` + `{session}_after_filter.key.pkl` | 两个内部开关、滤波及陷波参数、版本；不含输入 Raw |

FIF 缓存加载时使用 `preload=True`；命中避免重新计算，但仍有磁盘读取和内存占用。`build_valid_raw` 每次重建，`cache/raw/` 目前没有被实际使用。波形、地形图和 properties 图没有图片缓存命中机制，开启后会重新生成并覆盖同名图片；关闭不会删除旧图片。

### 需要记住的组合

| 控制与状态 | 实际效果 |
| --- | --- |
| 本阶段 `True`，缓存目录有效，键命中 | 调用函数，加载该阶段的处理结果；绘图函数加载 ICA 后仍重新画图 |
| 本阶段 `True`，缓存缺失或参数/版本不同 | 按当前输入计算，再写入缓存 |
| 本阶段 `False`，即使磁盘已有该阶段缓存 | 整步跳过，既不加载，也不清除缓存 |
| 函数支持的 `*_cache_dir=None` | 不读、不写该阶段缓存，执行时重新计算 |
| 上游阶段改成 `False`，下游仍 `True` 且命中旧缓存 | 下游可能载入以前包含该上游处理的结果；当前开关不一定反映在最终数据中 |

例如，先运行“插值 → 前滤波 → 平均参考”，随后仅把插值关掉，而前滤波参数和 `version` 不变，前滤波仍可能加载之前对**插值后数据**计算的缓存。改变上游滤波参数后，ICA 也可能命中旧模型，因为 ICA 的键不包含这些参数。

**改变数据、处理顺序、上游开关、配置内容或处理实现时，应同时让受影响的下游缓存失效。** 最方便的方式是递增脚本顶部的全局 `version`（当前为 `2`），`main` 会把新值传给所有缓存阶段；只有此次开启的阶段会在运行时重新计算。直接调用单个函数时，要自行传入一致的 `version`，函数默认是 `1`。

也可以删除受影响的缓存文件及对应 `.key.pkl`，并继续清理或更新其下游缓存。仅删除插值缓存不会自动使前滤波、参考或 ICA 失效；仅修改本阶段参数通常只使本阶段失效。

### 标注修改后怎样处理

| 改动 | 应采取的操作 |
| --- | --- |
| 替换同路径 MAT；改变通道、采样率、布局或上游工作流 | 增加全局版本，重新运行所需阶段，保证下游 ICA 对应新 Raw |
| 修改坏道 JSON 列表，但路径不变 | 增加版本，或清除插值及全部受影响的下游缓存 |
| 仅修改 `topomap_exclude` | 上游和 ICA 未改变时，可直接重跑 properties；该列表每次从 JSON 读取 |
| 仅修改 `final_exclude` | 上游和 ICA 未改变时，可直接重跑重建；**已有后滤波缓存还需失效**，否则可能得到旧的重建结果 |
| 修改 ICA 的 `n_components`、`method`、`rng`、`fit_params` | 这些参数当前没有进入 ICA 键，需要增加版本；重新检查 IC 编号和两个排除列表 |

只改变候选或最终排除列表，不需要重新拟合原 ICA；如果重新拟合了 ICA，则旧 IC 编号对应的成分可能变化，标注也需要重新确认。

支持 `None` 的缓存参数包括 `data_cache_dir`、`raw_interp_cache_dir`、`raw_pre_filter_cache_dir`、`raw_referenced_cache_dir`、`ica_cache_dir` 和 `raw_after_filter_cache_dir`。**`mark_artifacts_and_build` 的 `raw_cleaned_cache_dir` 必须是 `Path`，不能设为 `None`**；`ica_cache_dir=None` 只能关闭 ICA 模型缓存，重建结果仍会写出。

## 5. 重要参数怎么调

以下值指 `main()` 当前传入的值，不一定是函数定义的默认值。调整处理参数在 `main` 内相应函数调用处进行；只修改函数默认值，若 `main` 仍显式传值，就不会改变脚本运行行为。

| 函数 / 配置 | 最重要的参数与当前值 | 调整要点 |
| --- | --- | --- |
| 顶部配置 | `subjects`、`ch_names`、`ch_types`、`sfreq=256`、`montage_name`、`version=2` | 选取被试；按真实数据匹配通道及采样率。上游输入或处理逻辑变化时更新版本 |
| `load_cached_mat` / `_read_mat` | `convert_index=True`、`convert_shape=True`、`scale_factor=1e-6` | 输入已是 0-based 时关闭索引转换；已是通道×采样点时关闭转置；已是 V 时不再乘 `1e-6`，可传 `None` 或 `1.0` |
| `load_cached_mat` | `cache_file_name='scherer_read_mat.pkl'`、`data_cache_dir` | 不同实验可使用不同缓存文件名或目录；`None` 关闭读取层缓存 |
| `build_valid_raw` | `ch_names`、`ch_types`、`sfreq`、`montage_name` | 通道名称/类型数量与数据行数一致；采样率不会自动从 MAT 推断。这个函数不缓存，也不对 trial 分段 |
| `view_waveforms` | `scalings=35e-6`、`n_channels=len(ch_names)`、`start=0`、`duration=30` | `scalings` 按 V 调节显示尺度，也支持 `'auto'`；`start` 和 `duration` 单位为秒，换时间段检查。此处调参只影响图片 |
| `exclude_bads_and_interpolate` | `reset_bads=True`、`method={'eeg':'spline'}`、`mode='accurate'` | `reset_bads=True` 在插值后清空坏道标记；`False` 保留标记，可能影响后续通道选择。`mode` 支持 `'fast'` / `'accurate'`，主要针对 MNE 最小范数插值；当前 EEG 使用 spline |
| `pre_filter` | `is_filter=True`、`l_freq=1.0`、`h_freq=None`、`picks='eeg'` | 当前是 ICA 前的高通。`h_freq=None` 不做低通；改成数值后形成相应带通。修改上游频段后同步使 ICA 缓存失效 |
| `pre_filter` / `after_filter` | `is_notch_filter=False`、前滤波 `notch_freqs=[50.0,100.0]` | 列出频率不会自动开启陷波，必须同时设内部开关。按数据实际工频填写；所有频率须适合真实采样率 |
| `set_reference` | `ref_channels='average'`、`ch_type='eeg'`、`joint=False` | 可换参考通道名称列表，或使用 `'REST'`；当前直接重参考，没有使用 projection。改变参考方式后更新下游 ICA |
| 三个 ICA 函数 | `decim=3`、`ica_cache_dir` | `decim=1` 使用全部拟合采样点；更大值减少拟合点数，不改变原始 Raw 采样率。不是带抗混叠滤波的重采样，须结合输入带宽决定。保持三个函数的 `decim` 一致 |
| `gen_components_topomap` | `components_dir` | 画全部已拟合 IC；不自动识别伪迹，也不更改 Raw |
| `gen_properties` | `ica_exclude_file` 的 `topomap_exclude`、`properties_dir` | 仅绘制候选 IC；空列表直接跳过该 session。图片写到 `{properties_dir}/{session}/` |
| `mark_artifacts_and_build` | `final_exclude`、`raw_cleaned_cache_dir` | 仅排除最终确认 IC；`ica.apply` 修改字典中的 Raw，再保存。修改排除列表后需要处理后滤波旧缓存 |
| `after_filter` | `is_filter=True`、`l_freq=8.0`、`h_freq=30.0`、`picks='eeg'` | 当前是后处理的 8–30 Hz 频段，按下游任务改动。它在输入的当前 Raw 上工作，若关闭重建，也能对未经过 ICA 清理的数据执行 |

### 滤波与重参考的进阶参数

前后滤波都支持 `method='fir'` / `'iir'`。当前使用 `phase='zero'`、`filter_length='auto'`、自动过渡带、`fir_window='hamming'`、`fir_design='firwin'`。优先调截止频率和是否陷波，再根据需要调整这些设计参数。

- `l_freq=None`：不做高通；`h_freq=None`：不做低通。常规带通应满足 `0 < l_freq < h_freq < sfreq/2`，当前 Nyquist 频率为 128 Hz。
- `filter_length` 和 `l_trans_bandwidth` / `h_trans_bandwidth` 控制 FIR 长度及过渡带。遇到滤波器长于数据段的提示时，结合数据长度和目标频段调整；`iir_params` 仅用于 IIR。
- `n_jobs` / `notch_n_jobs` 控制对应计算的并行度，当前为 `None`。`verbose` / `notch_verbose` 控制日志，不是工作流开关，也不参与滤波缓存键。
- `notch_freqs=None` 仅用于 `notch_method='spectrum_fit'` 的自动检测；启用当前后滤波陷波时，需补充明确频率，或同时选择该方法。当前后滤波 `notch_method='fir'`，不能只把开关改成 `True` 而保留 `None`。
- REST 的 `pos=15.0` 控制源网格间距，`exclude=30.0` 控制球心附近的排除距离，单位为 mm；`trans=None` 是当前球模型的配置。REST 分支在检查每个 session 缓存之前就构建 forward，因此缓存命中仍可能有额外模型计算。

### ICA 拟合参数在哪里改

当前三个 ICA 函数内部都写有：

```python
mne.preprocessing.ICA(
    n_components=None,
    rng=1,
    method='picard',
    fit_params={'extended': True, 'ortho': False},
)
```

这些是函数体内的固定值，不能直接作为 `main()` 参数传入。需要调整时，应同时修改三个函数，或在自己的调用代码中统一实现拟合。`n_components=None` 交给 MNE 根据数据选择成分数，并不承诺恰好得到 30 个成分；`rng` 控制随机性；使用 `picard` 需要安装 `python-picard`。修改这些固定参数后更新 `version`，再重新确认 IC 标注。

参数的完整定义可参考 [MNE ICA 文档](https://mne.tools/stable/generated/mne.preprocessing.ICA.html)、[Raw 的滤波与插值 API](https://mne.tools/stable/generated/mne.io.Raw.html) 和 [EEG 重参考 API](https://mne.tools/stable/generated/mne.set_eeg_reference.html)。本 README 的流程及缓存行为以本仓库源码为准。

## 6. 输出与常见问题

- 看不到图形窗口：程序使用 `Agg`，到 `results/` 对应子目录查看 PNG。
- 首次运行没有 properties 图：发布模板的候选列表为空，先开启 ICA 地形图并填写 `topomap_exclude`。
- 修改 JSON 后结果没变：坏道内容不在插值缓存键中；最终 IC 改动也不会自动使后滤波缓存失效。按第 4 节更新版本或清理受影响的缓存。
- 没有最终 FIF：重建开启才写 `raw_cleaned`；后滤波开启且启用其缓存目录才写 `raw_after_filter`。`results/` 主要存图，不是最终信号导出目录。
- 想接入训练：自行读取对应的最终 FIF，或调用返回 Raw 字典的处理函数；trial 起点和标签在 MAT 读取字典里，需要另行对齐及分段。`main()` 不返回最终 Raw 字典。
- 数据或中间文件显示为 Git ignored：这是发布副本的正常行为。`.gitignore` 保留目录占位符，排除数据、缓存、结果、资料和废弃目录中新生成的文件；两个配置文件仍可正常提交。
