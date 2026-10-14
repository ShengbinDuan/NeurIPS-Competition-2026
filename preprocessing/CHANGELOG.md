# Changelog

## 2026-10-14 — GitHub 工程同步

### Added
- 同步 `src/preprocessing_config_version.py` 和 `configs/config.py`，集中管理插值、前后滤波、参考与 ICA 参数。
- 配置版按上游参数与实验参数隔离缓存，在 `results/config_version/<实验编号>/config_snapshot.json` 保存方法、流程、输入约定和人工标注快照。
- 增加发布辅助入口 `run_preprocessing.py`，通过 `waveforms`、`ica`、`properties`、`clean` 四个阶段调用原始主流程；默认从 MAT 读取开始，可显式指定已有缓存恢复点。
- 增加仅含五个核心包的 `requirements-core.txt`，保留原始完整环境清单 `requirements.txt`。

### Changed
- 同步当前主脚本的 `start` / `is_dic` 接口、上游缓存标识传播、ICA 独立建立与读取、重建结果缓存以及 properties 波形绘制逻辑。
- 更新中文 README，区分两套入口的默认流程、参数位置、缓存目录和恢复规则。
- 导出当前运行所需的空目录；删除上版仅用于历史图片归档、备份和废弃资料的空目录占位。

### Export notes
- `src/`、`configs/config.py`、`pyproject.toml`、`requirements.txt`、`.python-version` 与本次导出时的原工程保持字节一致。
- 两个人工标注 JSON 保留原始键和嵌套结构，列表全部清空；需要重新读图填写。
- 数据、虚拟环境、真实缓存、生成结果和历史导出包不随代码同步。
- 以下 0.1.1 记录保留原工程 CHANGELOG 内容；本次同步没有修改原始工程或其版本元数据。

## [0.1.1] - 2026-10-12

### Changed
- 将 ICA 缓存和建立从绘图、重建 Raw 的流程中独立出来，在去除伪迹之前统一缓存或建立 ICA。
- 将 前 / 后 滤波统一为一个滤波函数，再在主流程中通过不同顺序调用。
- 将缓存读取和写入逻辑从具体函数中抽取出来进行了独立实现，根据不同的读取写入接口需要分散实现辅助函数。
- 修补了整体缓存机制，更加精细地处理了上游变化、跳过上游直接利用本步缓存的规则，并添加了分开保存不同版本缓存的机制。
- 增补了 properties 绘图，增加了波形图绘制，并拉长了频谱图以便观察低频峰值。

### Added
