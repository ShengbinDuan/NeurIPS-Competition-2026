"""配置驱动的预处理版本，处理流程延续 Preprocessing.py。

方法参数来自 configs/config.py；流程设置集中在 main() 中。
运行：python src/preprocessing_config_version.py
也可导入 main(config=自定义字典)，传入从历史快照恢复的配置。
"""

import hashlib
import inspect
import json
import pickle
from copy import deepcopy
from pathlib import Path
from runpy import run_path
from typing import Any, Literal, Callable

import mne
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.io import loadmat


def _read_mat(
    data_folder: Path | str,  # 存放 mat 文件的文件夹
    subjects: list,  # 要读取的被试名列表
    convert_index: bool = True,  # True 将 mat 1-based 索引转化为 Python 的 0-based 索引
    convert_shape: bool = True,  # True 将 mat 的电压数据 shape 进行转置
    scale_factor: float | None = None,  # 电压单位转换系数
    ):
    """
    功能：
    对 data_folder 中的 mat 文件进行读取，返回各 subject 的数据字典 np_data_dic。

    输入：
    data 文件夹下的 mat 文件

    输出：
    np_data_dic 存放若干 np_data：
    表层是一个 list，保留各个 session 的数据；
    每个 session 由一个 dict 实现，包含下述字段：
    X：连续电压数据；
    y：各 trial 的标签；
    trial：trial 的起始采样点；原 mat 为 1-based 索引，convert_index=True 时转换为 0-based；
    fs：采样率
    classes：各标签对应的实际类别；
    session：会话标号，从 1 开始。    
    """

    data_folder = Path(data_folder)
    np_data_dic = {}
    for subject in subjects:
        data_file = data_folder / f'{subject}.mat'
        # simplify_cells=True 使得 load_mat 返回简化后的 dict，仅 'data' 字段下为有效数据
        np_data = loadmat(data_file, simplify_cells=True)
        # 仅取 'data' 字段下有效数据，data 为一个 list
        np_data = np_data['data']

        for i in range(len(np_data)):
            # np 数组转换，此后不再说明
            np_data[i]['trial'] = np.array(np_data[i]['trial'],dtype=np.int64)
            # 需要时转换为 0-based 索引
            if convert_index:
                np_data[i]['trial'] = np_data[i]['trial'] - 1
            
            np_data[i]['y'] = np.array(np_data[i]['y'],dtype=np.int16)

            np_data[i]['X'] = np.array(np_data[i]['X'],dtype=np.float32)
            # 需要时转置电压数据
            if convert_shape:
                np_data[i]['X'] = np_data[i]['X'].T
            # 需要时进行单位转化
            if scale_factor is not None:
                np_data[i]['X'] = np_data[i]['X'] * scale_factor
                
        np_data_dic[subject] = np_data

    return np_data_dic


def _normalize_legacy_cache_key(
    saved_key: Any,
    legacy_key_kind: Literal['raw', 'mat'] | None,
) -> Any:
    """将旧键转换为参数键，仅用于读取旧缓存，不检查数据或文件状态。"""
    if not isinstance(saved_key, tuple):
        return saved_key

    if legacy_key_kind == 'raw' and saved_key:
        raw_state = saved_key[0]
        if (
            isinstance(raw_state, tuple)
            and len(raw_state) == 3
            and isinstance(raw_state[0], int)
            and isinstance(raw_state[1], int)
            and isinstance(raw_state[2], bytes)
        ):
            return saved_key[1:]

    if legacy_key_kind == 'mat' and len(saved_key) == 5:
        data_state = saved_key[0]
        if isinstance(data_state, tuple) and data_state and all(
            isinstance(file_state, tuple)
            and len(file_state) == 3
            and isinstance(file_state[0], str)
            for file_state in data_state
        ):
            data_files = [Path(file_state[0]) for file_state in data_state]
            data_folder = data_files[0].parent
            if all(file.parent == data_folder and file.suffix == '.mat' for file in data_files):
                return (
                    str(data_folder.resolve()),
                    tuple(file.stem for file in data_files),
                    *saved_key[1:],
                )

    return saved_key

def _read_pickle(
    cache_file: Path,  # 缓存路径
    ):
    with cache_file.open('rb') as f:
        cache = pickle.load(f)
    return cache

def _read_raw(
    cache_file: Path,  # 缓存路径
    ):
    cache = mne.io.read_raw_fif(cache_file, preload=True, verbose=False)
    return cache

def _read_ica(
        cache_file: Path,  # 缓存路径
    ):
    cache = mne.preprocessing.read_ica(cache_file, verbose=False)
    return cache

def _load_cache(
    cache_key_file: Path,  # 缓存键目标路径
    cache_file: Path,  # 缓存路径
    key_components: tuple,  # 生成缓存键的特征元素
    version: int,  # 缓存版本号，内部组成 cache key，以防外部写 cache key 遗漏
    read_fn: Callable[[Path], Any] = _read_pickle,  # 根据读取方法，选择对应读取函数
    legacy_key_kind: Literal['raw', 'mat'] | None = None,  # 兼容旧键中已保存的参数
    ) -> tuple[tuple[Any, ...], Any | None]:
    """
    功能：
    辅助函数，统一完成中间数据缓存读取工作。

    机制：
    只对比函数的处理参数和版本号，相同则直接加载缓存。
    Raw 内容、Info、annotations 及文件大小、修改时间不参与比较。
    main() 按上游参数和人工标注隔离缓存；替换原始数据或修改算法实现后
    仍需增加 main() 中的 version。
    """
    cache_key = (*key_components, version)

    if cache_key_file.exists() and cache_file.exists():
        try:
            with cache_key_file.open('rb') as f:
                saved_key = pickle.load(f)
            saved_key = _normalize_legacy_cache_key(saved_key, legacy_key_kind)
            # 处理参数可能含有 np 数组，转成 bytes 后比较，避免数组比较报错
            if pickle.dumps(saved_key) == pickle.dumps(cache_key):
                cache = read_fn(cache_file)
                return cache_key, cache
        except (OSError, EOFError, pickle.UnpicklingError, ValueError, AttributeError):
            pass

    return cache_key, None


def _dump_pickle(
    data: Any,  # 需要缓存的数据
    cache_file: Path,  # 缓存路径
    ):

    with cache_file.open('wb') as f:
        cache = pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

def _dump_raw(
    raw: mne.io.BaseRaw,  # 需要缓存的数据
    cache_file: Path,  # 缓存路径
    fmt: Literal['single', 'double', 'int', 'short'] = 'double',  # 缓存写入格式
    ):

    raw.save(cache_file, overwrite=True, fmt=fmt, verbose=False)

def _dump_ica(
    ica: mne.preprocessing.ICA,  # 需要缓存的数据
    cache_file: Path,  # 缓存路径
    ):

    ica.save(cache_file, overwrite=True, verbose=False)

def _dump_cache(
        cache_key: tuple[Any, ...],  # 缓存键，通常和 _load_cache 统一
        data: Any,  # 需要缓存的数据
        cache_key_file: Path,  # 目标缓存键路径
        cache_data_file: Path,  # 目标缓存路径
        dump_fn: Callable[[Any, Path], Any],  # 用于缓存写入的方法函数
    ):
    """
    功能：
    辅助函数，统一完成中间数据缓存工作。
    """
    cache_key_file.parent.mkdir(parents=True, exist_ok=True)
    cache_data_file.parent.mkdir(parents=True, exist_ok=True)

    temporary_key_file = cache_key_file.with_suffix(cache_key_file.suffix + '.temp')
    # 保留 FIF 后缀和文件名约定，供 MNE 正常保存。
    temporary_data_file = cache_data_file.with_name('tmp-' + cache_data_file.name)

    try:
        with temporary_key_file.open('wb') as f:
            pickle.dump(cache_key, f, protocol=pickle.HIGHEST_PROTOCOL)
        dump_fn(data, temporary_data_file)

        temporary_data_file.replace(cache_data_file)
        temporary_key_file.replace(cache_key_file)
    finally:
        temporary_key_file.unlink(missing_ok=True)
        temporary_data_file.unlink(missing_ok=True)


def load_cached_mat(
    data_folder: Path,  # 存放 mat 文件的文件夹
    subjects: list,  # 要读取的被试名列表
    cache_file_name: str, # 缓存文件名
    data_cache_dir: Path | None = None, # 原始数据缓存目录，不给则不读取缓存数据，且不进行缓存
    convert_index: bool = True,  # True 将 mat 1-based 索引转化为 Python 的 0-based 索引
    convert_shape: bool = True,  # True 将 mat 的电压数据 shape 进行转置
    scale_factor: float | None = None,  # 电压单位转换系数
    version: int = 1, # 缓存版本号
    ):
    """
    功能：
    检测数据是否已缓存命中，若命中，则直接加载；若未命中，则重新读取。
    避免重复运行，节省数据读取时间。

    机制：
    数据目录、被试列表、转换参数和版本号未变更时，直接读取已有缓存。
    不检查 mat 文件内容、大小或修改时间；更新原始数据后需增加 version。

    输入：
    data 文件夹下的 mat 文件；
    或 data_cache_dir 下的 pkl 文件。

    输出：
    详见 _read_mat
    """

    np_data_dic = None  # 默认为 None，等待赋值，使得 data_cache_dir 为 None 时默认进行数据操作
    if data_cache_dir is not None:

        data_cache_key_components = (
            str(Path(data_folder).resolve()), tuple(subjects),
            convert_index, convert_shape, scale_factor,
        )
        data_cache_file = data_cache_dir / cache_file_name
        data_cache_key_file = data_cache_file.with_suffix('.key.pkl')

        # 缓存命中直接载入缓存
        data_cache_key, np_data_dic = _load_cache(
            cache_key_file=data_cache_key_file,
            cache_file=data_cache_file,
            key_components=data_cache_key_components,
            version=version,
            read_fn=_read_pickle,
            legacy_key_kind='mat',
        )

    if np_data_dic is None:
        # 缓存未命中则重新读取数据
        np_data_dic = _read_mat(
            data_folder=data_folder,
            subjects=subjects,
            convert_index=convert_index,
            convert_shape=convert_shape,
            scale_factor=scale_factor,
        )

        # 将读取数据导出到缓存文件
        if data_cache_dir is not None:
            _dump_cache(
                cache_key=data_cache_key,
                data=np_data_dic,
                cache_key_file=data_cache_key_file,
                cache_data_file=data_cache_file,
                dump_fn=_dump_pickle,
            )

    return np_data_dic


def build_valid_raw(
        data_dic: dict,  # 存放数据的 dict 对象
        ch_names: list,  # 通道名
        ch_types: list,  # 通道种类
        sfreq: int,  # 采样频率
        montage_name: str,  # 电极布局模板名
    ) -> dict[str, mne.io.BaseRaw]:
    """
    功能：
    将携带数据的字典读取并转化为配置完整的 Raw 对象字典，方便后续处理。
    建立 Raw 过程较快，因此不必缓存。

    输入：
    load_cached_mat 返回的数据字典

    输出：
    raw_dic 字典：
    字典的键为 {受试名}+{session 索引}，
    字典的值为该受试该 session 的原始电压 Raw 对象数据。
    """

    raw_dic = {}

    info = mne.create_info(ch_names=ch_names, ch_types=ch_types, sfreq=sfreq, verbose=False)

    for key, val in data_dic.items():
        for session in range(len(val)):
            raw_key = key + str(session)
            raw = mne.io.RawArray(val[session]['X'], info, verbose=False)
            raw.set_montage(montage_name)

            raw_dic[raw_key] = raw

    return raw_dic


def view_waveforms(
        raw_dic: dict,  # 输入 Raw 对象字典
        waveforms_dir: Path,  # 存储波形图路径
        scalings: float | Literal['auto'] = 'auto',  # 缩放尺度
        n_channels: int | None = 30,  # 不滚动滑轮时直接显示的通道数
        start: float = 0,  # 从什么时间开始显示，单位秒
        duration: float = 10.0,  # 显示时长，单位秒
    ):
    """
    功能：
    绘制各被试各 session 的波形图，以观察坏道。
    """

    waveforms_dir.mkdir(parents=True, exist_ok=True)
    for sbj_session, raw in raw_dic.items():

        # 后面通过 Matplotlib 保存图片，因此使用 Matplotlib 波形图后端
        with mne.viz.use_browser_backend('matplotlib'):
            fig = raw.plot(
                duration=duration,
                start=start,
                n_channels=n_channels,
                scalings=scalings,
                show=False,
            )

        fig.subplots_adjust(top=0.9)
        fig.suptitle(sbj_session, fontsize=16, fontweight='bold')

        waveforms_file = waveforms_dir / f'{sbj_session}_waveforms.png'
        fig.savefig(waveforms_file)

        plt.close(fig)

        print(f'已绘制波形图 {sbj_session}_waveforms.png')


def exclude_bads_and_interpolate(
        raw_dic: dict,  # 输入 Raw 对象字典
        bad_channels_exclude_file: Path,  # 读取人工标注的坏道字典
        raw_interp_cache_dir: Path | None = None,  # 插值后 Raw 对象缓存文件夹，为 None 时则重新插值，且不进行缓存
        reset_bads: bool = True,  # 是否将插值完的通道移出 raw.info['bads']
        method: dict | None = None,  # 插值方法，EEG 在 None 时默认使用 {'eeg':'spline'}
        mode: Literal['fast','accurate'] = 'accurate',  # 最小范数插值（MNE）下的计算模式，决定计算速度（与精度权衡）
        version: int = 1, # 缓存版本号
    ):
    """
    功能：
    根据人工筛选的坏道字典，标记坏道，并插值恢复通道。
    该操作仍会使用插值保留完整通道。

    输出：
    清理坏道并插值后的 Raw 对象字典。

    说明：
    所有 Raw 对象字典统一命名为 raw_dic；
    方便直接在 main 函数通过增加或去除操作统一工作流，以免反复修改传入下游函数的 Raw 字典变量名。
    """
    with open(bad_channels_exclude_file, 'r', encoding='utf-8') as f:
        bad_channels_dic = json.load(f)

    for sbj_session, raw in raw_dic.items():
        raw_interp = None

        if raw_interp_cache_dir is not None:
            raw_interp_cache_file = raw_interp_cache_dir / f'{sbj_session}_interp_raw.fif'
            raw_interp_cache_key_file = raw_interp_cache_dir / f'{sbj_session}_interp.key.pkl'
            # 把当前 session 的坏道名单纳入缓存键。
            raw_interp_cache_key_components = (
                str(Path(bad_channels_exclude_file).resolve()),
                tuple(bad_channels_dic[sbj_session]), reset_bads, method, mode,
            )

            raw_interp_cache_key, raw_interp = _load_cache(
                cache_key_file=raw_interp_cache_key_file,
                cache_file=raw_interp_cache_file,
                key_components=raw_interp_cache_key_components,
                read_fn=_read_raw,
                version=version,
            )


        if raw_interp is None:
            raw_interp = raw.copy().load_data()
            raw_interp.info['bads'] = bad_channels_dic[sbj_session]
            raw_interp.interpolate_bads(
                reset_bads=reset_bads,
                mode=mode,
                origin='auto',
                method=method,
            )

            # 如果 raw_interp 已缓存就不用再 dump，因此条件缩进
            if raw_interp_cache_dir is not None:
                raw_interp_cache_dir.mkdir(parents=True, exist_ok=True)
                # 先保存数据再保存缓存键，与前后滤波的保存顺序一致
                _dump_cache(
                    cache_key=raw_interp_cache_key,
                    data=raw_interp,
                    cache_key_file=raw_interp_cache_key_file,
                    cache_data_file=raw_interp_cache_file,
                    dump_fn=_dump_raw,
                )

        raw_dic[sbj_session] = raw_interp
        print(f'已完成{sbj_session}的坏道清除和插值')

    return raw_dic


def filter(
        # 对象参数
        raw_dic: dict,  # 输入 Raw 对象字典
        raw_filter_cache_dir: Path | None = None,  # 本次滤波的缓存目录；前后滤波分别传入不同目录
        
        # filter 参数
        is_filter: bool = False,  # 决定是否进行带通滤波
        l_freq: float | None = None,  # 低频边界，单位 Hz；None 表示不做高通
        h_freq: float | None = None,  # 高频边界，单位 Hz；None 表示不做低通
        picks: str | list[str] | list[int] | slice | NDArray | None = None,  # 滤波通道；可传类型、名称或索引
        filter_length: str | int = "auto",  # FIR 长度；可传 "auto"、"10s" 等时间字符串或采样点数
        l_trans_bandwidth: float | Literal["auto"] = "auto",  # 低频端过渡带宽度，单位 Hz，仅用于 FIR
        h_trans_bandwidth: float | Literal["auto"] = "auto",  # 高频端过渡带宽度，单位 Hz，仅用于 FIR
        n_jobs: int | Literal["cuda"] | None = None,  # 并行任务数；"cuda" 仅用于支持 CUDA 的 FIR
        method: Literal["fir", "iir"] = "fir",  # 滤波方法
        iir_params: dict | None = None,  # IIR 配置；None 时默认四阶 Butterworth，仅用于 IIR
        phase: Literal[
            "zero", "minimum", "zero-double", "minimum-half", "forward"
        ] = "zero",  # 相位处理方式；"forward" 仅用于 IIR，"minimum" 系列仅用于 FIR
        fir_window: Literal["hamming", "hann", "blackman"] = "hamming",  # FIR 窗函数，仅用于 FIR
        fir_design: Literal["firwin", "firwin2"] = "firwin",  # FIR 设计方法，仅用于 FIR
        verbose: bool | str | int | None = None,  # 日志级别；False 关闭日志，None 使用默认级别

        # notch_filter 参数
        is_notch_filter: bool = False,  # 决定是否进行陷波滤波
        notch_freqs: float | list[float] | NDArray | None = None,  # 陷波中心频率，单位 Hz；None 仅用于 spectrum_fit 自动检测
        notch_picks: str | list[str] | list[int] | slice | NDArray | None = None,  # 滤波通道；可传类型、名称或索引
        notch_filter_length: str | int = "auto",  # FIR 长度；spectrum_fit 中为拟合窗口长度，"auto" 对应 10 秒
        notch_notch_widths: float | list[float] | NDArray | None = None,  # 阻带宽度，单位 Hz；None 时采用 freqs / 200
        notch_trans_bandwidth: float = 1.0,  # 过渡带宽度，单位 Hz，用于 FIR 或 IIR
        notch_n_jobs: int | Literal["cuda"] | None = None,  # 并行任务数；"cuda" 仅用于支持 CUDA 的 FIR
        notch_method: Literal["fir", "iir", "spectrum_fit"] = "fir",  # 陷波方法；spectrum_fit 用于拟合并去除正弦噪声
        notch_iir_params: dict | None = None,  # IIR 配置；None 时默认四阶 Butterworth，仅用于 IIR
        notch_phase: Literal[
            "zero", "minimum", "zero-double", "minimum-half", "forward"
        ] = "zero",  # 相位处理方式；"forward" 仅用于 IIR，"minimum" 系列仅用于 FIR
        notch_fir_window: Literal["hamming", "hann", "blackman"] = "hamming",  # FIR 窗函数，仅用于 FIR
        notch_fir_design: Literal["firwin", "firwin2"] = "firwin",  # FIR 设计方法，仅用于 FIR
        notch_verbose: bool | str | int | None = None,  # 日志级别；False 关闭日志，None 使用默认级别

        version: int = 1,  # 缓存版本
    ):
    """
    功能：
    根据参数进行陷波滤波和带通滤波，供 ICA 前后分别调用。

    输出：
    滤波后的 Raw 对象字典。

    说明：
    先进行陷波滤波，再进行带通滤波，两步由各自的开关决定是否执行；
    滤波参数和版本号未变更时，直接加载缓存，不比较输入 Raw 状态；
    两个开关均为 False 时，直接返回 raw_dic。
    """

    if not is_filter and not is_notch_filter:
        return raw_dic

    for sbj_session, raw in raw_dic.items():
        raw_filter = None

        if raw_filter_cache_dir is not None:
            raw_filter_cache_file = raw_filter_cache_dir / f'{sbj_session}_filtered_raw.fif'
            raw_filter_cache_key_file = raw_filter_cache_dir / f'{sbj_session}_filter.key.pkl'
            raw_filter_cache_key_components = (
                is_filter, l_freq, h_freq, picks, filter_length,
                l_trans_bandwidth, h_trans_bandwidth, n_jobs, method,
                iir_params, phase, fir_window, fir_design,
                is_notch_filter, notch_freqs, notch_picks, notch_filter_length,
                notch_notch_widths, notch_trans_bandwidth, notch_n_jobs,
                notch_method, notch_iir_params, notch_phase,
                notch_fir_window, notch_fir_design,
            )

            raw_filter_cache_key, raw_filter = _load_cache(
                cache_key_file=raw_filter_cache_key_file,
                cache_file=raw_filter_cache_file,
                key_components=raw_filter_cache_key_components,
                version=version,
                read_fn=_read_raw,
                legacy_key_kind='raw',
            )

        if raw_filter is None:
            raw_filter = raw.copy().load_data()

            if is_notch_filter:
                raw_filter.notch_filter(
                    freqs=notch_freqs,
                    picks=notch_picks,
                    filter_length=notch_filter_length,
                    notch_widths=notch_notch_widths,
                    trans_bandwidth=notch_trans_bandwidth,
                    n_jobs=notch_n_jobs,
                    method=notch_method,
                    iir_params=notch_iir_params,
                    phase=notch_phase,
                    fir_window=notch_fir_window,
                    fir_design=notch_fir_design,
                    verbose=notch_verbose,
                )

            if is_filter:
                raw_filter.filter(
                    l_freq=l_freq,
                    h_freq=h_freq,
                    picks=picks,
                    filter_length=filter_length,
                    l_trans_bandwidth=l_trans_bandwidth,
                    h_trans_bandwidth=h_trans_bandwidth,
                    n_jobs=n_jobs,
                    method=method,
                    iir_params=iir_params,
                    phase=phase,
                    fir_window=fir_window,
                    fir_design=fir_design,
                    verbose=verbose,
                )

            # 如果 raw_filter 已缓存就不用再 dump，因此条件缩进
            if raw_filter_cache_dir is not None:
                raw_filter_cache_dir.mkdir(parents=True, exist_ok=True)
                # 先保存数据再保存缓存键，以免写入中断后误判缓存命中
                _dump_cache(
                    cache_key=raw_filter_cache_key,
                    data=raw_filter,
                    cache_key_file=raw_filter_cache_key_file,
                    cache_data_file=raw_filter_cache_file,
                    dump_fn=_dump_raw,
                )

        raw_dic[sbj_session] = raw_filter
        print(f'已完成{sbj_session}的滤波')

    return raw_dic


def set_reference(
        # 对象参数
        raw_dic: dict,  # 输入 Raw 对象字典
        raw_referenced_cache_dir: Path | None = None,  # 重参考后 Raw 对象缓存文件夹，为 None 时则重新重参考，且不进行缓存
        
        # set_eeg_reference 参数，projection 使用默认 False，直接执行重参考
        ref_channels: str | list[str] | dict[str, str | list[str]] = "average",  # 参考：平均参考、通道名列表、"REST"，或各通道的参考映射字典
        ch_type: (
            Literal["auto", "eeg", "ecog", "seeg", "dbs"] | list[str]
        ) = "auto",  # 要重参考的通道类型；auto 选择首个存在的类型，此处不逐个标注每个通道的类型
        joint: bool = False,  # 多类型平均参考时，False 各类型分别计算，True 合并所有指定类型计算
        verbose: bool | str | int | None = None,  # 日志级别；False 关闭日志，None 使用默认级别

        # 球模型参数，仅 ref_channels='REST' 时使用
        exclude: float = 30.0,  # 排除距球心小于该距离的源点，单位 mm；0.0 表示不额外排除中心区域
        pos: float = 15.0,  # 源点网格间距，单位 mm；越小，源点越密，计算量越大
        trans: str | Path | dict | None = None,  # 头部与 MRI 的坐标变换；可传变换文件路径、变换对象或 "fsaverage"；None 表示单位变换

        version: int = 1,  # 缓存版本号
    ):
    """
    功能：
    Raw 对象重参考；
    减少原参考干扰，统一分析标准。

    说明：
    根据后续分析方法要求，选取所需参考方式；
    REST 使用首个 Raw 的电极位置建立球模型，同一批 Raw 应使用相同的通道和电极布局。
    只比较参考参数和版本号；命中时返回已缓存的完整处理结果，包含此前的上游处理。

    输出：
    重参考后的 Raw 对象字典。
    """
    if not raw_dic:
        return raw_dic

    forward = None  # 平均参考或指定通道参考不需要正向模型
    if ref_channels == 'REST':  # 由于统一的几何配置，仅建立一次头模型
        raw = next(iter(raw_dic.values()))

        sphere = mne.make_sphere_model(r0='auto', head_radius='auto', info=raw.info, verbose=verbose)
        src = mne.setup_volume_source_space(pos=pos, sphere=sphere, exclude=exclude, verbose=verbose)
        forward = mne.make_forward_solution(
            info=raw.info,
            trans=trans,
            src=src,
            bem=sphere,
            meg=False,
            eeg=True,
            verbose=verbose,
        )

    for sbj_session, raw in raw_dic.items():
        raw_referenced = None

        if raw_referenced_cache_dir is not None:
            raw_referenced_cache_file = raw_referenced_cache_dir / f'{sbj_session}_referenced_raw.fif'
            raw_referenced_cache_key_file = raw_referenced_cache_dir / f'{sbj_session}_referenced.key.pkl'
            # 球模型参数也参与缓存键；verbose 只影响日志，不参与缓存键
            raw_referenced_cache_key_components = (ref_channels, ch_type, joint, exclude, pos, trans)

            raw_referenced_cache_key, raw_referenced = _load_cache(
                cache_key_file=raw_referenced_cache_key_file,
                cache_file=raw_referenced_cache_file,
                key_components=raw_referenced_cache_key_components,
                version=version,
                read_fn=_read_raw,
                legacy_key_kind='raw',
            )


        if raw_referenced is None:
            raw_referenced = raw.copy().load_data().set_eeg_reference(
                ref_channels=ref_channels,
                ch_type=ch_type,
                forward=forward,
                joint=joint,
                verbose=verbose,
            )

            if raw_referenced_cache_dir is not None:
                raw_referenced_cache_dir.mkdir(parents=True, exist_ok=True)
                _dump_cache(
                    cache_key=raw_referenced_cache_key,
                    data=raw_referenced,
                    cache_key_file=raw_referenced_cache_key_file,
                    cache_data_file=raw_referenced_cache_file,
                    dump_fn=_dump_raw,
                )

        raw_dic[sbj_session] = raw_referenced

        print(f'已完成{sbj_session}的{ref_channels}重参考')

    return raw_dic


def ica_cache(
        raw_dic: dict,  # 输入 Raw 对象字典
        decim: int,  # ICA 拟合时每隔多少个采样点取一个点，不改变 Raw 的采样率
        ica_cache_dir: Path | None = None,  # ICA 缓存文件夹，为 None 时则重新建立 ICA，且不进行缓存
        version: int = 1,  # 缓存版本号
        *,
        n_components: int | float | None = None,  # ICA 成分数量或解释方差比例
        rng: int | None = 1,  # 随机种子
        method: str = 'picard',  # ICA 算法
        fit_params: dict | None = None,  # 算法拟合参数，由配置显式提供
        max_iter: int | Literal['auto'] = 'auto',  # 最大迭代次数
        verbose: bool | str | int | None = None,  # 日志设置，留在 main()
    ):
    """统一读取 ICA 缓存；未命中时建立 ICA 并保存缓存。"""
    for sbj_session, raw in raw_dic.items():
        raw.load_data()
        ica = None

        if ica_cache_dir is not None:
            ica_cache_file = ica_cache_dir / f'{sbj_session}_ica.fif'
            ica_cache_key_file = ica_cache_dir / f'{sbj_session}_ica.key.pkl'
            # 模型参数全部参与缓存键；main() 另外隔离不同上游流程。
            ica_cache_key_components = (
                n_components, rng, method, fit_params, max_iter, decim,
            )

            ica_cache_key, ica = _load_cache(
                cache_key_file=ica_cache_key_file,
                cache_file=ica_cache_file,
                key_components=ica_cache_key_components,
                version=version,
                read_fn=_read_ica,
                legacy_key_kind='raw',
            )

        if ica is None:
            ica = mne.preprocessing.ICA(
                n_components=n_components,
                rng=rng,
                method=method,
                fit_params=fit_params,
                max_iter=max_iter,
                verbose=verbose,
            )

            ica.fit(raw, decim=decim, verbose=verbose)

            if ica_cache_dir is not None:
                ica_cache_dir.mkdir(parents=True, exist_ok=True)
                _dump_cache(
                    cache_key=ica_cache_key,
                    data=ica,
                    cache_key_file=ica_cache_key_file,
                    cache_data_file=ica_cache_file,
                    dump_fn=_dump_ica,
                )
            

def gen_components_topomap(
        raw_dic: dict,  # 输入 Raw 对象字典
        components_dir: Path,  # 保存 components topomap 图片的文件夹
        ica_cache_dir: Path,  # 读取已建立的 ICA 缓存的文件夹
    ):
    """
    功能：
    读取已建立的 ICA 缓存，生成各 session 的 ICA 成分地形图。
    """

    components_dir.mkdir(parents=True, exist_ok=True)
    for sbj_session, raw in raw_dic.items():
        ica_cache_file = ica_cache_dir / f'{sbj_session}_ica.fif'
        ica = mne.preprocessing.read_ica(ica_cache_file)

        fig = ica.plot_components(
            picks=range(ica.n_components_),
            show=False,
            # 将所有成分放在一张图上，避免成分较多时返回多个 Figure
            nrows=(ica.n_components_ + 4) // 5,
            ncols=5,
        )
        fig.subplots_adjust(top=0.9)
        fig.suptitle(sbj_session, fontsize=16, fontweight='bold')
        fig.savefig(components_dir / f'{sbj_session}_ica_components.png')
        plt.close(fig)

        print(f'已绘制{sbj_session}_ica_components.png')


def gen_properties(
        raw_dic: dict,  # 输入 Raw 对象字典
        properties_dir: Path,  # 保存 properties 图片的文件夹
        ica_exclude_file: Path,  # 读取人工标注的潜在 IC 字典
        ica_cache_dir: Path,  # 读取已建立的 ICA 缓存的文件夹
    ):
    """
    功能：
    绘制潜在伪迹主成分的 properties 图。

    使用说明：
    根据 topomap 图，分辨各 Raw 数据中的主成分哪些存在伪迹；
    将存在伪迹的主成分的索引，填写到 './configs/ica_exclude.json' 的 'topomap_exclude' 下对应 Raw 的列表里；
    该绘图会根据列表绘制潜在伪迹 IC （被 'topomap_exclude' 标记的 IC) 的 properties 图。
    本函数只读取 ICA 缓存，不建立 ICA 或写入模型缓存。
    运行前需通过 ica_cache 建立缓存；上游变更后需更新缓存，并重新确认 IC 索引。
    """
    
    with open(ica_exclude_file, 'r', encoding='utf-8') as f:
        pot_IC_dic = json.load(f)

    topomap_exclude_dic = pot_IC_dic['topomap_exclude']

    for sbj_session, raw in raw_dic.items():
        topomap_exclude = topomap_exclude_dic[sbj_session]
        if not topomap_exclude:
            print(f'{sbj_session}未标记潜在伪迹 IC，跳过 properties 图')
            continue

        raw.load_data()
        ica_cache_file = ica_cache_dir / f'{sbj_session}_ica.fif'
        ica = mne.preprocessing.read_ica(ica_cache_file)

        figs = ica.plot_properties(
            inst=raw,
            picks=topomap_exclude,
            show=False
        )

        properties_sbj_session_dir = properties_dir / sbj_session
        properties_sbj_session_dir.mkdir(parents=True, exist_ok=True)
        for ica_idx, fig in zip(topomap_exclude, figs):
            fig.subplots_adjust(top=0.9)
            fig.suptitle(sbj_session, fontsize=16, fontweight='bold')
            fig.savefig(properties_sbj_session_dir / f"{sbj_session}_{ica_idx}_properties.png")
            plt.close(fig)

        print(f'已绘制{sbj_session} 各主成分 properties 图')


def mark_artifacts_and_build(
        raw_dic: dict,  # 输入 Raw 对象字典
        raw_cleaned_cache_dir: Path,  # 缓存 ICA 处理后的 Raw 对象
        ica_exclude_file: Path,  # 读取人工确认的伪迹 IC 字典中的 final_exclude
        ica_cache_dir: Path,  # 读取已建立的 ICA 缓存的文件夹
    ):
    """
    功能：
    根据人工读图后指定的伪迹 IC 索引，清理伪迹，并重建 Raw 数据。

    使用说明：
    在通过 properties 图确认伪迹后，
    将伪迹的主成分的索引，填写到 './configs/ica_exclude.json' 的 'final_exclude' 下对应 Raw 的列表里；
    该操作会根据 'final_exclude' 中的标记从 ICA 对象中排除对应主成分，并重建 Raw 数据。
    本函数只读取 ICA 缓存；运行前需通过 ica_cache 建立缓存。

    输出：
    清理伪迹并重建后的 Raw 对象字典。
    """
    with open(ica_exclude_file, 'r', encoding='utf-8') as f:
        pot_IC_dic = json.load(f)
    final_exclude_dic = pot_IC_dic['final_exclude']

    raw_cleaned_cache_dir.mkdir(parents=True, exist_ok=True)
    for sbj_session, raw in raw_dic.items():
        raw.load_data()
        ica_cache_file = ica_cache_dir / f'{sbj_session}_ica.fif'
        ica = mne.preprocessing.read_ica(ica_cache_file)

        ica.exclude = final_exclude_dic[sbj_session]
        ica.apply(raw_dic[sbj_session])

        raw_cleaned_cache_file = raw_cleaned_cache_dir / f'{sbj_session}_cleaned_raw.fif'
        raw_dic[sbj_session].save(raw_cleaned_cache_file, overwrite=True)

        print(f'已完成{sbj_session}的伪迹清理和重建')

    return raw_dic


def _validate_config(config: dict) -> None:
    """在读取大体积数据前检查配置分组和参数名。"""
    if not isinstance(config, dict):
        raise TypeError('config 必须是字典')

    flow_parameters = {
        'raw_dic', 'version', 'verbose', 'notch_verbose',
        'bad_channels_exclude_file', 'raw_interp_cache_dir',
        'raw_filter_cache_dir', 'raw_referenced_cache_dir', 'ica_cache_dir',
        'is_filter', 'is_notch_filter', 'n_jobs', 'notch_n_jobs',
    }
    functions = {
        'interpolate': exclude_bads_and_interpolate,
        'pre_filter': filter,
        'reference': set_reference,
        'ica': ica_cache,
        'after_filter': filter,
    }
    missing = functions.keys() - config.keys()
    unknown = config.keys() - functions.keys()
    if missing or unknown:
        raise ValueError(f'config 分组不匹配：缺少 {sorted(missing)}，未知 {sorted(unknown)}')

    for section, function in functions.items():
        if not isinstance(config[section], dict):
            raise TypeError(f'config[{section!r}] 必须是参数字典')
        allowed = inspect.signature(function).parameters.keys() - flow_parameters
        invalid = config[section].keys() - allowed
        if invalid:
            raise ValueError(f'config[{section!r}] 包含未知或流程型参数：{sorted(invalid)}')
    if 'decim' not in config['ica']:
        raise ValueError("config['ica'] 缺少 decim")


def _settings_id(settings: dict) -> str:
    """相同设置得到同一个编号，便于复用缓存和保留实验参数。"""
    serialized = json.dumps(settings, ensure_ascii=False, sort_keys=True, allow_nan=False)
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()[:16]


def main(config: dict | None = None) -> dict[str, mne.io.BaseRaw]:
    """传入方法参数字典；省略时加载 configs/config.py 中的 config。

    可用 main(config=json.loads(快照路径.read_text())['config']) 恢复方法参数。
    流程开关仍以本函数中的设置为准；完整流程也会记录在参数快照中。
    """
    # 流程设置：延续原脚本 __main__ 中的开关，不放入方法参数配置。
    is_view_waveforms = False
    is_exclude_bads_and_interpolate = False
    is_pre_filter = False
    is_set_reference = False
    is_ica_cache = True
    is_gen_components_topomap = True
    is_gen_properties = False
    is_mark_artifacts_and_build = False
    is_after_filter = False

    # 滤波子步骤开关、并行和日志也属于流程设置。
    pre_is_filter = True
    pre_is_notch_filter = False
    after_is_filter = True
    after_is_notch_filter = False
    n_jobs = None
    verbose = False
    version = 2  # 替换原始 MAT 数据或修改算法实现后递增，使缓存失效。

    # 路径以脚本所在项目为基准，直接运行脚本或通过模块导入均可使用。
    project_root = Path(__file__).resolve().parents[1]
    configs_folder = project_root / 'configs'
    data_folder = project_root / 'data'
    cache_folder = project_root / 'cache'
    results_folder = project_root / 'results'
    bad_channels_exclude_file = configs_folder / 'bad_channels_exclude.json'
    ica_exclude_file = configs_folder / 'ica_exclude.json'

    # 数据格式约定，不作为预处理方法的实验变量。
    subjects = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'L']
    ch_names = [
        'AFz', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC3', 'FCz', 'FC4', 'T3', 'C3',
        'Cz', 'C4', 'T4', 'CP3', 'CPz', 'CP4', 'P7', 'P5', 'P3', 'P1', 'Pz',
        'P2', 'P4', 'P6', 'P8', 'PO3', 'PO4', 'O1', 'O2',
    ]
    ch_types = ['eeg'] * len(ch_names)
    sfreq = 256
    montage_name = 'colin27_1005'
    convert_index = True
    convert_shape = True
    scale_factor = 1e-6  # 原始电压从微伏转换为 MNE 使用的伏。

    if config is None:
        config = run_path(str(configs_folder / 'config.py'))['config']
    config = deepcopy(config)  # 保持传入的实验字典不被处理过程修改。
    _validate_config(config)

    workflow = {
        'is_view_waveforms': is_view_waveforms,
        'is_exclude_bads_and_interpolate': is_exclude_bads_and_interpolate,
        'is_pre_filter': is_pre_filter,
        'is_set_reference': is_set_reference,
        'is_ica_cache': is_ica_cache,
        'is_gen_components_topomap': is_gen_components_topomap,
        'is_gen_properties': is_gen_properties,
        'is_mark_artifacts_and_build': is_mark_artifacts_and_build,
        'is_after_filter': is_after_filter,
        'pre_is_filter': pre_is_filter,
        'pre_is_notch_filter': pre_is_notch_filter,
        'after_is_filter': after_is_filter,
        'after_is_notch_filter': after_is_notch_filter,
    }
    data_settings = {
        'data_folder': str(data_folder), 'subjects': subjects,
        'ch_names': ch_names, 'ch_types': ch_types, 'sfreq': sfreq,
        'montage_name': montage_name, 'convert_index': convert_index,
        'convert_shape': convert_shape, 'scale_factor': scale_factor,
    }
    bad_channels = None
    if is_exclude_bads_and_interpolate:
        bad_channels = json.loads(bad_channels_exclude_file.read_text(encoding='utf-8'))
    ica_exclude = None
    if is_gen_properties or is_mark_artifacts_and_build:
        ica_exclude = json.loads(ica_exclude_file.read_text(encoding='utf-8'))

    # ICA 上游编号不包含绘图、伪迹重建和后滤波设置。
    # 调整后滤波频段或人工 IC 标注时，复用相同的 ICA 拟合结果。
    upstream_settings = {
        'config': {key: value for key, value in config.items() if key != 'after_filter'},
        'pipeline': {
            'interpolate': is_exclude_bads_and_interpolate,
            'pre_filter': is_pre_filter, 'reference': is_set_reference,
            'is_filter': pre_is_filter, 'is_notch_filter': pre_is_notch_filter,
        },
        'data': data_settings, 'bad_channels': bad_channels,
        'version': version, 'mne_version': mne.__version__,
    }
    upstream_id = _settings_id(upstream_settings)
    experiment_settings = {
        'upstream_id': upstream_id,
        'after_filter': config['after_filter'],
        'pipeline': {
            'remove_artifacts': is_mark_artifacts_and_build,
            'after_filter': is_after_filter,
            'is_filter': after_is_filter, 'is_notch_filter': after_is_notch_filter,
        },
        'final_exclude': ica_exclude['final_exclude'] if is_mark_artifacts_and_build else None,
    }
    experiment_id = _settings_id(experiment_settings)
    upstream_cache_dir = cache_folder / 'config_version' / f'ica_{upstream_id}'
    experiment_cache_dir = cache_folder / 'config_version' / f'run_{experiment_id}'
    experiment_results_dir = results_folder / 'config_version' / experiment_id
    experiment_results_dir.mkdir(parents=True, exist_ok=True)

    # 快照同时记录方法参数、流程、数据约定及当次人工标注。
    snapshot = {
        'experiment_id': experiment_id, 'upstream_id': upstream_id,
        'config': config, 'workflow': workflow, 'data': data_settings,
        'version': version, 'mne_version': mne.__version__,
        'bad_channels': bad_channels, 'ica_exclude': ica_exclude,
        'cache_dir': str(experiment_cache_dir),
        'ica_cache_dir': str(upstream_cache_dir / 'ica'),
    }
    snapshot_file = experiment_results_dir / 'config_snapshot.json'
    snapshot_file.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
        encoding='utf-8',
    )
    print(f'实验参数编号：{experiment_id}；参数快照：{snapshot_file}')

    # 原始 MAT 缓存共用，避免为每组实验参数重复存储整份输入数据。
    np_data_dic = load_cached_mat(
        data_folder=data_folder,
        subjects=subjects,
        cache_file_name='scherer_read_mat.pkl',
        data_cache_dir=cache_folder / 'data',
        convert_index=convert_index,
        convert_shape=convert_shape,
        scale_factor=scale_factor,
        version=version,
    )
    raw_dic = build_valid_raw(
        data_dic=np_data_dic, ch_names=ch_names, ch_types=ch_types,
        sfreq=sfreq, montage_name=montage_name,
    )
    del np_data_dic

    if is_view_waveforms:
        view_waveforms(
            raw_dic=raw_dic,
            waveforms_dir=experiment_results_dir / 'waveforms',
            scalings=35e-6, n_channels=len(ch_names), start=0, duration=30,
        )

    if is_exclude_bads_and_interpolate:
        raw_dic = exclude_bads_and_interpolate(
            raw_dic=raw_dic,
            bad_channels_exclude_file=bad_channels_exclude_file,
            raw_interp_cache_dir=upstream_cache_dir / 'raw_interp',
            version=version,
            **config['interpolate'],
        )

    if is_pre_filter:
        raw_dic = filter(
            raw_dic=raw_dic,
            raw_filter_cache_dir=upstream_cache_dir / 'raw_pre_filter',
            is_filter=pre_is_filter, is_notch_filter=pre_is_notch_filter,
            n_jobs=n_jobs, notch_n_jobs=n_jobs,
            verbose=verbose, notch_verbose=verbose, version=version,
            **config['pre_filter'],
        )

    if is_set_reference:
        raw_dic = set_reference(
            raw_dic=raw_dic,
            raw_referenced_cache_dir=upstream_cache_dir / 'raw_referenced',
            verbose=verbose, version=version,
            **config['reference'],
        )

    ica_cache_dir = upstream_cache_dir / 'ica'
    if is_ica_cache:
        ica_cache(
            raw_dic=raw_dic, ica_cache_dir=ica_cache_dir,
            verbose=verbose, version=version,
            **config['ica'],
        )

    if is_gen_components_topomap:
        gen_components_topomap(
            raw_dic=raw_dic,
            components_dir=experiment_results_dir / 'components',
            ica_cache_dir=ica_cache_dir,
        )

    if is_gen_properties:
        gen_properties(
            raw_dic=raw_dic,
            properties_dir=experiment_results_dir / 'properties',
            ica_exclude_file=ica_exclude_file, ica_cache_dir=ica_cache_dir,
        )

    if is_mark_artifacts_and_build:
        raw_dic = mark_artifacts_and_build(
            raw_dic=raw_dic,
            raw_cleaned_cache_dir=experiment_cache_dir / 'raw_cleaned',
            ica_exclude_file=ica_exclude_file, ica_cache_dir=ica_cache_dir,
        )

    if is_after_filter:
        raw_dic = filter(
            raw_dic=raw_dic,
            raw_filter_cache_dir=experiment_cache_dir / 'raw_after_filter',
            is_filter=after_is_filter, is_notch_filter=after_is_notch_filter,
            n_jobs=n_jobs, notch_n_jobs=n_jobs,
            verbose=verbose, notch_verbose=verbose, version=version,
            **config['after_filter'],
        )

    return raw_dic


if __name__ == '__main__':
    main()
