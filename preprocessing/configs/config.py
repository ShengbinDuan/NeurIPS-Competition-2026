"""预处理方法参数：默认值延续 src/Preprocessing.py。

修改下面的 config 字典，然后运行：
    python src/preprocessing_config_version.py

步骤开关、执行顺序、数据结构、路径、绘图和并行设置在新脚本的 main() 中。
新脚本会按参数隔离缓存，并在 results/config_version/<参数编号>/ 下
保存 config_snapshot.json，方便保留、比较和恢复不同实验的设置。
ICA 方法或上游预处理改变后，需要重新读图确认 ica_exclude.json 的 IC 索引。
配置值使用可保存为 JSON 的 Python 标量、列表和字典；路径使用字符串。
"""

config = {
    # 坏道插值；坏道名单仍由 bad_channels_exclude.json 提供。
    "interpolate": {
        "reset_bads": True,  # 插值后清除坏道标记，使这些通道参与后续处理
        "method": {"eeg": "spline"},  # EEG 插值方法
        "mode": "accurate",  # MNE 插值的精度模式，可改为 "fast"
    },
    # ICA 前滤波；带通 / 陷波是否执行由 main() 中的开关决定。
    "pre_filter": {
        "l_freq": 1.0,  # 高通截止频率（Hz）；None 表示不做高通
        "h_freq": None,  # 低通截止频率（Hz）；None 表示不做低通
        "picks": "eeg",  # 要滤波的通道类型、名称列表或索引列表
        "filter_length": "auto",  # FIR 长度："auto"、时长字符串或采样点数
        "l_trans_bandwidth": "auto",  # 高通端过渡带宽（Hz），仅用于 FIR
        "h_trans_bandwidth": "auto",  # 低通端过渡带宽（Hz），仅用于 FIR
        "method": "fir",  # "fir" 或 "iir"
        "iir_params": None,  # IIR 参数，如 {"order": 4, "ftype": "butter"}
        "phase": "zero",  # 相位处理方式
        "fir_window": "hamming",  # FIR 窗函数
        "fir_design": "firwin",  # FIR 设计方法
        "notch_freqs": [50.0, 100.0],  # 陷波中心频率（Hz）
        "notch_picks": "eeg",
        "notch_filter_length": "auto",
        "notch_notch_widths": None,  # 阻带宽（Hz）；None 使用 MNE 默认值
        "notch_trans_bandwidth": 1.0,  # 陷波过渡带宽（Hz）
        "notch_method": "fir",  # "fir"、"iir" 或 "spectrum_fit"
        "notch_iir_params": None,
        "notch_phase": "zero",
        "notch_fir_window": "hamming",
        "notch_fir_design": "firwin",
    },
    # EEG 重参考；REST 的球模型参数只在 ref_channels="REST" 时生效。
    "reference": {
        "ref_channels": "average",  # 平均参考、参考通道列表、"REST" 或参考映射
        "ch_type": "eeg",
        "joint": False,  # 多种通道类型是否合并计算平均参考
        "exclude": 30.0,  # REST：排除靠近球心的源点，单位 mm
        "pos": 15.0,  # REST：源点网格间距，单位 mm
        "trans": None,  # REST：头部到 MRI 的坐标变换文件；None 为单位变换
    },
    # ICA：原脚本内部固定的模型参数也在此显式保存。
    "ica": {
        "n_components": None,  # None 自动确定；整数指定数量，浮点数指定解释方差比例
        "rng": 1,  # 随机种子，用于复现实验
        "method": "picard",  # "picard"、"fastica"、"infomax" 等
        "fit_params": {"extended": True, "ortho": False},  # 与所选算法配套
        "max_iter": "auto",  # 最大迭代次数
        "decim": 3,  # 拟合时抽取采样点的间隔，不改变 Raw 的采样率
    },
    # ICA 后滤波，与 pre_filter 独立，方便比较不同运动想象频段。
    "after_filter": {
        "l_freq": 4.0,
        "h_freq": 40.0,
        "picks": "eeg",
        "filter_length": "auto",
        "l_trans_bandwidth": "auto",
        "h_trans_bandwidth": "auto",
        "method": "fir",
        "iir_params": None,
        "phase": "zero",
        "fir_window": "hamming",
        "fir_design": "firwin",
        "notch_freqs": None,  # 自动检测只适用于 notch_method="spectrum_fit"
        "notch_picks": "eeg",
        "notch_filter_length": "auto",
        "notch_notch_widths": None,
        "notch_trans_bandwidth": 1.0,
        "notch_method": "fir",
        "notch_iir_params": None,
        "notch_phase": "zero",
        "notch_fir_window": "hamming",
        "notch_fir_design": "firwin",
    },
}
