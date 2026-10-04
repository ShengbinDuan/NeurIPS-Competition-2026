import json
import pickle
from pathlib import Path
from typing import Any, Literal

import mne
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.io import loadmat

# 路径配置
configs_folder = Path('./configs')
configs_folder.mkdir(parents=True, exist_ok=True)
bad_channels_exclude_file = configs_folder / 'bad_channels_exclude.json'
ica_exlude_file = configs_folder / 'ica_exclude.json'


data_folder = Path('./data')

cache_folder = Path('./cache')
data_cache_dir = cache_folder / 'data'
data_cache_dir.mkdir(parents=True, exist_ok=True)
raw_cache_dir = cache_folder / 'raw'
raw_cache_dir.mkdir(parents=True, exist_ok=True)
raw_interp_cache_dir = cache_folder / 'raw_interp'
raw_interp_cache_dir.mkdir(parents=True, exist_ok=True)
raw_pre_filter_cache_dir = cache_folder / 'raw_pre_filter'
raw_pre_filter_cache_dir.mkdir(parents=True, exist_ok=True)
raw_referenced_cache_dir = cache_folder / 'raw_referenced'
raw_referenced_cache_dir.mkdir(parents=True, exist_ok=True)
raw_cleaned_cache_dir = cache_folder / 'raw_cleaned'
raw_cleaned_cache_dir.mkdir(parents=True, exist_ok=True)
raw_after_filter_cache_dir = cache_folder / 'raw_after_filter'
raw_after_filter_cache_dir.mkdir(parents=True, exist_ok=True)
ica_cache_dir = cache_folder / 'ica'
ica_cache_dir.mkdir(parents=True, exist_ok=True)

results_folder = Path('./results')
waveforms_dir = results_folder / 'waveforms'
waveforms_dir.mkdir(parents=True, exist_ok=True)
components_dir = results_folder / 'components'
components_dir.mkdir(parents=True, exist_ok=True)
properties_dir = results_folder / 'properties'
properties_dir.mkdir(parents=True, exist_ok=True)

# 数据结构
subjects = ["A", "C", "D", "E", "F", "G", "H", "J", "L"]
ch_types = ['eeg'] * 30
ch_names = [
    "AFz", "F7", "F3", "Fz", "F4", "F8", "FC3", "FCz", "FC4", "T3", "C3",
    "Cz", "C4", "T4", "CP3", "CPz", "CP4", "P7", "P5", "P3", "P1", "Pz",
    "P2", "P4", "P6", "P8", "PO3", "PO4", "O1", "O2",
]
sfreq = 256
montage_name = "colin27_1005"
version = 2  # 缓存版本号，处理方法改变时可手动递增，使旧缓存失效

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


def _load_cache(
    cache_key_file: Path,  # 缓存键目标路径
    cache_file: Path,  # 缓存路径
    key_components: tuple,  # 生成缓存键的特征元素
    version: int,  # 缓存版本号，内部组成 cache key，以防外部写 cache key 遗漏
    is_pkl: bool = False,  # 数据是否以 pkl 格式统一读取，如果不是 pkl 格式则仅返回命中的缓存路径
    legacy_key_kind: Literal['raw', 'mat'] | None = None,  # 兼容旧键中已保存的参数
    ) -> tuple[tuple[Any, ...], Any | None]:
    """
    功能：
    辅助函数，统一完成中间数据缓存读取工作。

    机制：
    只对比函数的处理参数和版本号，相同则直接加载缓存。
    Raw 内容、Info、annotations 及文件大小、修改时间不参与比较。
    修改输入数据或配置文件内容后，需要手动增加 version 或删除对应缓存。
    """
    cache_key = (*key_components, version)

    if cache_key_file.exists() and cache_file.exists():
        try:
            with cache_key_file.open('rb') as f:
                saved_key = pickle.load(f)
            saved_key = _normalize_legacy_cache_key(saved_key, legacy_key_kind)
            # 处理参数可能含有 np 数组，转成 bytes 后比较，避免数组比较报错
            if pickle.dumps(saved_key) == pickle.dumps(cache_key):
                if is_pkl:
                    with cache_file.open('rb') as f:
                        return cache_key, pickle.load(f)
                return cache_key, cache_file
        except (OSError, EOFError, pickle.UnpicklingError):
            pass

    return cache_key, None
    

def _dump_cache(
        cache_key: tuple[Any, ...],  # 缓存键，通常和 _load_cache 统一
        data: Any,  # 需要缓存的数据
        cache_key_file: Path,  # 目标缓存键路径
        cache_data_file: Path,  # 目标缓存路径
        is_pkl: bool = False,  # 数据是否以 pkl 格式统一缓存，如果不是 pkl 格式无法通过该方法导出到本地
    ):
    """
    功能：
    辅助函数，统一完成中间数据缓存工作。
    """
    cache_key_file.parent.mkdir(parents=True, exist_ok=True)
    cache_data_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_key_file = cache_key_file.with_suffix(cache_key_file.suffix + '.temp')

    if is_pkl:
        temporary_data_file = cache_data_file.with_suffix(cache_data_file.suffix + '.temp')

        try:
            with temporary_key_file.open('wb') as f:
                pickle.dump(cache_key, f, protocol=pickle.HIGHEST_PROTOCOL)
            with temporary_data_file.open('wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            temporary_data_file.replace(cache_data_file)
            temporary_key_file.replace(cache_key_file)
        finally:
            temporary_key_file.unlink(missing_ok=True)
            temporary_data_file.unlink(missing_ok=True)

    else:
        try:
            with temporary_key_file.open('wb') as f:
                pickle.dump(cache_key, f, protocol=pickle.HIGHEST_PROTOCOL)
            temporary_key_file.replace(cache_key_file)
        finally:
            temporary_key_file.unlink(missing_ok=True)


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
            is_pkl=True,
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
                is_pkl=True,
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
            raw_interp_cache_file = raw_interp_cache_dir / f'{sbj_session}_interp.fif'
            raw_interp_cache_key_file = raw_interp_cache_dir / f'{sbj_session}_interp.key.pkl'
            # 只使用传入的配置文件路径和处理参数；文件内容变更时需增加 version
            raw_interp_cache_key_components = (
                str(Path(bad_channels_exclude_file).resolve()), reset_bads, method, mode,
            )

            raw_interp_cache_key, raw_interp = _load_cache(
                cache_key_file=raw_interp_cache_key_file,
                cache_file=raw_interp_cache_file,
                key_components=raw_interp_cache_key_components,
                version=version,
                is_pkl=False
            )

            # 只有缓存键命中才读取 fif；预加载后才能继续插值、重参考和 ICA
            if raw_interp is not None:
                raw_interp = mne.io.read_raw_fif(raw_interp_cache_file, preload=True, verbose=False)


        if raw_interp is None:
            raw.info['bads'] = bad_channels_dic[sbj_session]

            raw_interp = raw.copy().load_data().interpolate_bads(
                reset_bads=reset_bads,
                mode=mode,
                origin='auto',
                method=method,
            )

            # 如果 raw_interp 已缓存就不用再 dump，因此条件缩进
            if raw_interp_cache_dir is not None:
                raw_interp_cache_dir.mkdir(parents=True, exist_ok=True)
                # 先保存数据再保存缓存键，与前后滤波的保存顺序一致
                raw_interp_cache_key_file.unlink(missing_ok=True)
                raw_interp.save(raw_interp_cache_file, overwrite=True, fmt='double')
                _dump_cache(
                    cache_key=raw_interp_cache_key,
                    data=raw_interp,
                    cache_key_file=raw_interp_cache_key_file,
                    cache_data_file=raw_interp_cache_file,
                    is_pkl=False,
                )

        raw_dic[sbj_session] = raw_interp
        print(f'已完成{sbj_session}的坏道清除和插值')

    return raw_dic


def pre_filter(
        # 对象参数
        raw_dic: dict,  # 输入 Raw 对象字典
        raw_pre_filter_cache_dir: Path | None = None,  # 前滤波后 Raw 对象缓存文件夹，为 None 时则重新滤波，且不进行缓存
        
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
    在 ICA 处理前，根据参数进行陷波滤波和带通滤波。

    输出：
    前滤波后的 Raw 对象字典。

    说明：
    先进行陷波滤波，再进行带通滤波，两步由各自的开关决定是否执行；
    滤波参数和版本号未变更时，直接加载缓存，不比较输入 Raw 状态；
    两个开关均为 False 时，直接返回 raw_dic。
    """

    if not is_filter and not is_notch_filter:
        return raw_dic

    for sbj_session, raw in raw_dic.items():
        raw_pre_filter = None

        if raw_pre_filter_cache_dir is not None:
            raw_pre_filter_cache_file = raw_pre_filter_cache_dir / f'{sbj_session}_pre_filter_raw.fif'
            raw_pre_filter_cache_key_file = raw_pre_filter_cache_dir / f'{sbj_session}_pre_filter.key.pkl'
            raw_pre_filter_cache_key_components = (
                is_filter, l_freq, h_freq, picks, filter_length,
                l_trans_bandwidth, h_trans_bandwidth, n_jobs, method,
                iir_params, phase, fir_window, fir_design,
                is_notch_filter, notch_freqs, notch_picks, notch_filter_length,
                notch_notch_widths, notch_trans_bandwidth, notch_n_jobs,
                notch_method, notch_iir_params, notch_phase,
                notch_fir_window, notch_fir_design,
            )

            raw_pre_filter_cache_key, raw_pre_filter = _load_cache(
                cache_key_file=raw_pre_filter_cache_key_file,
                cache_file=raw_pre_filter_cache_file,
                key_components=raw_pre_filter_cache_key_components,
                version=version,
                is_pkl=False,
                legacy_key_kind='raw',
            )

            # 缓存键命中后再读取 fif，避免参数改变后仍使用旧缓存
            if raw_pre_filter is not None:
                try:
                    raw_pre_filter = mne.io.read_raw_fif(
                        raw_pre_filter_cache_file, preload=True, verbose=False
                    )
                except (OSError, EOFError, ValueError, AttributeError):
                    raw_pre_filter = None

        if raw_pre_filter is None:
            raw_pre_filter = raw.copy().load_data()

            if is_notch_filter:
                raw_pre_filter.notch_filter(
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
                raw_pre_filter.filter(
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

            # 如果 raw_pre_filter 已缓存就不用再 dump，因此条件缩进
            if raw_pre_filter_cache_dir is not None:
                raw_pre_filter_cache_dir.mkdir(parents=True, exist_ok=True)
                # 先保存数据再保存缓存键，以免写入中断后误判缓存命中
                raw_pre_filter_cache_key_file.unlink(missing_ok=True)
                raw_pre_filter.save(raw_pre_filter_cache_file, overwrite=True, fmt='double')
                _dump_cache(
                    cache_key=raw_pre_filter_cache_key,
                    data=raw_pre_filter,
                    cache_key_file=raw_pre_filter_cache_key_file,
                    cache_data_file=raw_pre_filter_cache_file,
                    is_pkl=False,
                )

        raw_dic[sbj_session] = raw_pre_filter
        print(f'已完成{sbj_session}的前滤波')

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
            raw_referenced_cache_file = raw_referenced_cache_dir / f'{sbj_session}_referenced.fif'
            raw_referenced_cache_key_file = raw_referenced_cache_dir / f'{sbj_session}_referenced.key.pkl'
            # 球模型参数也参与缓存键；verbose 只影响日志，不参与缓存键
            raw_referenced_cache_key_components = (ref_channels, ch_type, joint, exclude, pos, trans)

            raw_referenced_cache_key, raw_referenced = _load_cache(
                cache_key_file=raw_referenced_cache_key_file,
                cache_file=raw_referenced_cache_file,
                key_components=raw_referenced_cache_key_components,
                version=version,
                is_pkl=False,
                legacy_key_kind='raw',
            )

            if raw_referenced is not None:
                raw_referenced = mne.io.read_raw_fif(raw_referenced_cache_file, preload=True, verbose=False)

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
                raw_referenced_cache_key_file.unlink(missing_ok=True)
                raw_referenced.save(raw_referenced_cache_file, overwrite=True, fmt='double')
                _dump_cache(
                    cache_key=raw_referenced_cache_key,
                    data=raw_referenced,
                    cache_key_file=raw_referenced_cache_key_file,
                    cache_data_file=raw_referenced_cache_file,
                    is_pkl=False,
                )

        raw_dic[sbj_session] = raw_referenced

        print(f'已完成{sbj_session}的{ref_channels}重参考')

    return raw_dic
            

def gen_components_topomap(
        raw_dic: dict,  # 输入 Raw 对象字典
        components_dir: Path,  # 保存 components topomap 图片的文件夹
        decim: int,  # ICA 拟合时每隔多少个采样点取一个点，不改变 Raw 的采样率
        ica_cache_dir: Path | None = None,  # ICA 缓存文件夹，为 None 时则重新建立 ICA，且不进行缓存
        version: int = 1,  # 缓存版本号
    ):
    """
    功能：
    生成各 session 的 ICA 成分地形图。
    """

    components_dir.mkdir(parents=True, exist_ok=True)
    for sbj_session, raw in raw_dic.items():
        raw.load_data()
        ica = None

        if ica_cache_dir is not None:
            ica_cache_file = ica_cache_dir / f'{sbj_session}_ica.fif'
            ica_cache_key_file = ica_cache_dir / f'{sbj_session}_ica.key.pkl'
            # 三处 ICA 统一只比较 decim 和 version，输入 Raw 状态不参与缓存键
            ica_cache_key_components = (decim,)

            ica_cache_key, ica = _load_cache(
                cache_key_file=ica_cache_key_file,
                cache_file=ica_cache_file,
                key_components=ica_cache_key_components,
                version=version,
                is_pkl=False,
                legacy_key_kind='raw',
            )

            if ica is not None:
                ica = mne.preprocessing.read_ica(ica_cache_file)

        if ica is None:
            ica = mne.preprocessing.ICA(
                n_components=None,
                rng=1,
                method='picard',
                fit_params={
                    'extended':True,
                    'ortho':False,
                },
            )

            ica.fit(raw, decim=decim)

            if ica_cache_dir is not None:
                ica_cache_dir.mkdir(parents=True, exist_ok=True)
                ica_cache_key_file.unlink(missing_ok=True)
                ica.save(ica_cache_file, overwrite=True)
                _dump_cache(
                    cache_key=ica_cache_key,
                    data=ica,
                    cache_key_file=ica_cache_key_file,
                    cache_data_file=ica_cache_file,
                    is_pkl=False,
                )

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
        decim: int,  # ICA 拟合时每隔多少个采样点取一个点，不改变 Raw 的采样率
        ica_cache_dir: Path | None = None,  # ICA 缓存文件夹，为 None 时则重新建立 ICA，且不进行缓存
        version: int = 1,  # 缓存版本号
    ):
    """
    功能：
    绘制潜在伪迹主成分的 properties 图。

    使用说明：
    根据 topomap 图，分辨各 Raw 数据中的主成分哪些存在伪迹；
    将存在伪迹的主成分的索引，填写到 './configs/ica_exclude.json' 的 'topomap_exclude' 下对应 Raw 的列表里；
    该绘图会根据列表绘制潜在伪迹 IC （被 'topomap_exclude' 标记的 IC) 的 properties 图。
    三处 ICA 共享仅含 decim 和 version 的缓存键。
    要让上游变更作用于 ICA，需增加 version 或删除 ICA 缓存，并重新确认 IC 索引。
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
        ica = None

        if ica_cache_dir is not None:
            ica_cache_file = ica_cache_dir / f'{sbj_session}_ica.fif'
            ica_cache_key_file = ica_cache_dir / f'{sbj_session}_ica.key.pkl'
            ica_cache_key_components = (decim,)

            ica_cache_key, ica = _load_cache(
                cache_key_file=ica_cache_key_file,
                cache_file=ica_cache_file,
                key_components=ica_cache_key_components,
                version=version,
                is_pkl=False,
                legacy_key_kind='raw',
            ) 

            if ica is not None:
                ica = mne.preprocessing.read_ica(ica_cache_file)

        if ica is None:
            ica = mne.preprocessing.ICA(
                n_components=None,
                rng=1,
                method='picard',
                fit_params={
                    'extended':True,
                    'ortho':False,
                },
            )

            ica.fit(raw, decim=decim)

            if ica_cache_dir is not None:
                ica_cache_dir.mkdir(parents=True, exist_ok=True)
                ica_cache_key_file.unlink(missing_ok=True)
                ica.save(ica_cache_file, overwrite=True)
                _dump_cache(
                    cache_key=ica_cache_key,
                    data=ica,
                    cache_key_file=ica_cache_key_file,
                    cache_data_file=ica_cache_file,
                    is_pkl=False,
                )

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
        decim: int,  # ICA 拟合时每隔多少个采样点取一个点，不改变 Raw 的采样率
        ica_cache_dir: Path | None = None,  # ICA 缓存文件夹，为 None 时则重新建立 ICA，且不进行缓存
        version: int = 1,  # 缓存版本号
    ):
    """
    功能：
    根据人工读图后指定的伪迹 IC 索引，清理伪迹，并重建 Raw 数据。

    使用说明：
    在通过 properties 图确认伪迹后，
    将伪迹的主成分的索引，填写到 './configs/ica_exclude.json' 的 'final_exclude' 下对应 Raw 的列表里；
    该操作会根据 'final_exclude' 中的标记从 ICA 对象中排除对应主成分，并重建 Raw 数据。

    输出：
    清理伪迹并重建后的 Raw 对象字典。
    """
    with open(ica_exclude_file, 'r', encoding='utf-8') as f:
        pot_IC_dic = json.load(f)
    final_exclude_dic = pot_IC_dic['final_exclude']

    raw_cleaned_cache_dir.mkdir(parents=True, exist_ok=True)
    for sbj_session, raw in raw_dic.items():
        raw.load_data()
        ica = None

        if ica_cache_dir is not None:
            ica_cache_file = ica_cache_dir / f'{sbj_session}_ica.fif'
            ica_cache_key_file = ica_cache_dir / f'{sbj_session}_ica.key.pkl'
            ica_cache_key_components = (decim,)

            ica_cache_key, ica = _load_cache(
                cache_key_file=ica_cache_key_file,
                cache_file=ica_cache_file,
                key_components=ica_cache_key_components,
                version=version,
                is_pkl=False,
                legacy_key_kind='raw',
            ) 

            if ica is not None:
                ica = mne.preprocessing.read_ica(ica_cache_file)

        if ica is None:
            ica = mne.preprocessing.ICA(
                n_components=None,
                rng=1,
                method='picard',
                fit_params={
                    'extended':True,
                    'ortho':False,
                },
            )

            ica.fit(raw, decim=decim)

            if ica_cache_dir is not None:
                ica_cache_dir.mkdir(parents=True, exist_ok=True)
                ica_cache_key_file.unlink(missing_ok=True)
                ica.save(ica_cache_file, overwrite=True)
                _dump_cache(
                    cache_key=ica_cache_key,
                    data=ica,
                    cache_key_file=ica_cache_key_file,
                    cache_data_file=ica_cache_file,
                    is_pkl=False,
                )

        ica.exclude = final_exclude_dic[sbj_session]
        ica.apply(raw_dic[sbj_session])

        raw_cleaned_cache_file = raw_cleaned_cache_dir / f'{sbj_session}_cleaned.fif'
        raw_dic[sbj_session].save(raw_cleaned_cache_file, overwrite=True)

        print(f'已完成{sbj_session}的伪迹清理和重建')

    return raw_dic


def after_filter(
        # 对象参数
        raw_dic: dict,  # 输入 Raw 对象字典
        raw_after_filter_cache_dir: Path | None = None,  # 后滤波后 Raw 对象缓存文件夹，为 None 时则重新滤波，且不进行缓存
        
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
    在 ICA 处理后，根据参数进行陷波滤波和带通滤波。

    输出：
    后滤波后的 Raw 对象字典。

    说明：
    先进行陷波滤波，再进行带通滤波，两步由各自的开关决定是否执行；
    滤波参数和版本号未变更时，直接加载缓存，不比较输入 Raw 状态；
    两个开关均为 False 时，直接返回 raw_dic。
    """

    if not is_filter and not is_notch_filter:
        return raw_dic

    for sbj_session, raw in raw_dic.items():
        raw_after_filter = None

        if raw_after_filter_cache_dir is not None:
            raw_after_filter_cache_file = raw_after_filter_cache_dir / f'{sbj_session}_after_filter_raw.fif'
            raw_after_filter_cache_key_file = raw_after_filter_cache_dir / f'{sbj_session}_after_filter.key.pkl'
            raw_after_filter_cache_key_components = (
                is_filter, l_freq, h_freq, picks, filter_length,
                l_trans_bandwidth, h_trans_bandwidth, n_jobs, method,
                iir_params, phase, fir_window, fir_design,
                is_notch_filter, notch_freqs, notch_picks, notch_filter_length,
                notch_notch_widths, notch_trans_bandwidth, notch_n_jobs,
                notch_method, notch_iir_params, notch_phase,
                notch_fir_window, notch_fir_design,
            )

            raw_after_filter_cache_key, raw_after_filter = _load_cache(
                cache_key_file=raw_after_filter_cache_key_file,
                cache_file=raw_after_filter_cache_file,
                key_components=raw_after_filter_cache_key_components,
                version=version,
                is_pkl=False,
                legacy_key_kind='raw',
            )

            # 缓存键命中后再读取 fif，避免参数改变后仍使用旧缓存
            if raw_after_filter is not None:
                try:
                    raw_after_filter = mne.io.read_raw_fif(
                        raw_after_filter_cache_file, preload=True, verbose=False
                    )
                except (OSError, EOFError, ValueError, AttributeError):
                    raw_after_filter = None

        if raw_after_filter is None:
            raw_after_filter = raw.copy().load_data()

            if is_notch_filter:
                raw_after_filter.notch_filter(
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
                raw_after_filter.filter(
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

            # 如果 raw_after_filter 已缓存就不用再 dump，因此条件缩进
            if raw_after_filter_cache_dir is not None:
                raw_after_filter_cache_dir.mkdir(parents=True, exist_ok=True)
                # 先保存数据再保存缓存键，以免写入中断后误判缓存命中
                raw_after_filter_cache_key_file.unlink(missing_ok=True)
                raw_after_filter.save(raw_after_filter_cache_file, overwrite=True, fmt='double')
                _dump_cache(
                    cache_key=raw_after_filter_cache_key,
                    data=raw_after_filter,
                    cache_key_file=raw_after_filter_cache_key_file,
                    cache_data_file=raw_after_filter_cache_file,
                    is_pkl=False,
                )

        raw_dic[sbj_session] = raw_after_filter
        print(f'已完成{sbj_session}的后滤波')

    return raw_dic


def main(
        is_view_waveforms: bool = False,  # True 则画原始波形图， False 不画
        is_exclude_bads_and_interpolate: bool = False,  # True 则根据人工标记清理坏道并插值，False 则不标记和插值
        is_pre_filter: bool = False,  # True 则在 ICA 前进行滤波，False 则不滤波
        is_set_reference: bool = False,  # True 则进行重参考，False 则不重参考
        is_gen_components_topomap: bool = False,  # True 则画 ICA 成分地形图， False 不画
        is_gen_properties: bool = False,  # True 则画主成分 properties 图， False 不画
        is_mark_artifacts_and_build: bool = False,  # True 则根据标记伪迹 IC 重建 Raw，False 则不重建
        is_after_filter: bool = False,  # True 则在 ICA 后进行滤波，False 则不滤波
    ):

    # 读取原始数据（已缓存则直接加载）
    np_data_dic = load_cached_mat(
        data_folder=data_folder,  # 存放 mat 文件的文件夹
        subjects=subjects,  # 要读取的被试名列表
        cache_file_name='scherer_read_mat.pkl',  # 缓存文件名
        data_cache_dir=data_cache_dir,  # 原始数据缓存目录，为 None 时则不读取缓存数据，且不进行缓存
        convert_index=True,  # True 将 mat 1-based 索引转化为 Python 的 0-based 索引
        convert_shape=True,  # True 将 mat 的电压数据 shape 进行转置
        scale_factor=1e-6,  # 电压单位转换系数
        version=version,  # 缓存版本号
    )

    # 将原始数据转化为 MNE Raw 对象
    raw_dic = build_valid_raw(
        data_dic=np_data_dic,  # 存放数据的 dict 对象
        ch_names=ch_names,  # 通道名
        ch_types=ch_types,  # 通道种类
        sfreq=sfreq,  # 采样频率
        montage_name=montage_name,  # 电极布局模板名
    )

    # 绘制各被试各 session 的波形图，以观察坏道
    if is_view_waveforms:
        view_waveforms(
            raw_dic=raw_dic,  # 输入 Raw 对象字典
            waveforms_dir=waveforms_dir,  # 存储波形图路径
            scalings=35e-6,  # 缩放尺度，参数需自己尝试，大致 100e-6 左右上下调
            n_channels=len(ch_names),  # 不滚动滑轮时直接显示的通道数
            start=0,  # 从什么时间开始显示，单位秒
            duration=30,  # 显示时长，单位秒
        )

    # 根据人工标记的坏道进行插值，保留完整通道
    if is_exclude_bads_and_interpolate:
        raw_dic = exclude_bads_and_interpolate(
            raw_dic=raw_dic,  # 输入 Raw 对象字典
            bad_channels_exclude_file=bad_channels_exclude_file,  # 读取人工标注的坏道字典
            raw_interp_cache_dir=raw_interp_cache_dir,  # 插值后 Raw 对象缓存文件夹，为 None 时则重新插值，且不进行缓存
            reset_bads=True,  # 是否将插值完的通道移出 raw.info['bads']
            method={'eeg':'spline'},  # 插值方法，None 时默认方法为 {'eeg':'spline'}
            mode='accurate',  # 最小范数插值（MNE）下的计算模式，决定计算速度（与精度权衡）
            version=version, # 缓存版本号
        )

    # 在 ICA 前进行高通滤波和工频陷波
    if is_pre_filter:
        raw_dic = pre_filter(
            # 对象参数
            raw_dic=raw_dic,  # 输入 Raw 对象字典
            raw_pre_filter_cache_dir=raw_pre_filter_cache_dir,  # 前滤波后 Raw 对象缓存文件夹，为 None 时则重新滤波，且不进行缓存

            # filter 参数
            is_filter=True,  # 决定是否进行带通滤波
            l_freq=1.0,  # 低频边界，单位 Hz；None 表示不做高通
            h_freq=None,  # 高频边界，单位 Hz；None 表示不做低通
            picks='eeg',  # 滤波通道；可传类型、名称或索引
            filter_length='auto',  # FIR 长度；可传 "auto"、"10s" 等时间字符串或采样点数
            l_trans_bandwidth='auto',  # 低频端过渡带宽度，单位 Hz，仅用于 FIR
            h_trans_bandwidth='auto',  # 高频端过渡带宽度，单位 Hz，仅用于 FIR
            n_jobs=None,  # 并行任务数；"cuda" 仅用于支持 CUDA 的 FIR
            method='fir',  # 滤波方法
            iir_params=None,  # IIR 配置；None 时默认四阶 Butterworth，仅用于 IIR
            phase='zero',  # 相位处理方式；"forward" 仅用于 IIR，"minimum" 系列仅用于 FIR
            fir_window='hamming',  # FIR 窗函数，仅用于 FIR
            fir_design='firwin',  # FIR 设计方法，仅用于 FIR
            verbose=False,  # 日志级别；False 关闭日志，None 使用默认级别

            # notch_filter 参数
            is_notch_filter=False,  # 决定是否进行陷波滤波
            notch_freqs=[50.0, 100.0],  # 陷波中心频率，单位 Hz；None 仅用于 spectrum_fit 自动检测
            notch_picks='eeg',  # 滤波通道；可传类型、名称或索引
            notch_filter_length='auto',  # FIR 长度；spectrum_fit 中为拟合窗口长度，"auto" 对应 10 秒
            notch_notch_widths=None,  # 阻带宽度，单位 Hz；None 时采用 freqs / 200
            notch_trans_bandwidth=1.0,  # 过渡带宽度，单位 Hz，用于 FIR 或 IIR
            notch_n_jobs=None,  # 并行任务数；"cuda" 仅用于支持 CUDA 的 FIR
            notch_method='fir',  # 陷波方法；spectrum_fit 用于拟合并去除正弦噪声
            notch_iir_params=None,  # IIR 配置；None 时默认四阶 Butterworth，仅用于 IIR
            notch_phase='zero',  # 相位处理方式；"forward" 仅用于 IIR，"minimum" 系列仅用于 FIR
            notch_fir_window='hamming',  # FIR 窗函数，仅用于 FIR
            notch_fir_design='firwin',  # FIR 设计方法，仅用于 FIR
            notch_verbose=False,  # 日志级别；False 关闭日志，None 使用默认级别

            version=version,  # 缓存版本号
        )

    # 在 ICA 前进行重参考，后续各步骤继续使用同一个 Raw 对象字典
    if is_set_reference:
        raw_dic = set_reference(
            # 对象参数
            raw_dic=raw_dic,  # 输入 Raw 对象字典
            raw_referenced_cache_dir=raw_referenced_cache_dir,  # 重参考后 Raw 对象缓存文件夹，为 None 时则重新重参考，且不进行缓存

            # set_eeg_reference 参数，projection 使用默认 False，直接执行重参考
            ref_channels='average',  # 参考：平均参考、通道名列表、'REST'，或各通道的参考映射字典
            ch_type='eeg',  # 要重参考的通道类型，此处统一处理 EEG 通道
            joint=False,  # 多类型平均参考时，False 各类型分别计算，True 合并所有指定类型计算
            verbose=False,  # 日志级别；False 关闭日志，None 使用默认级别

            # 球模型参数，仅 ref_channels='REST' 时使用
            exclude=30.0,  # 排除距球心小于该距离的源点，单位 mm
            pos=15.0,  # 源点网格间距，单位 mm；越小，源点越密，计算量越大
            trans=None,  # 头部与 MRI 的坐标变换；当前球模型使用 None，即单位变换

            version=version,  # 缓存版本号
        )

    # 读取 Raw 数据，生成 ICA 成分 topomap
    if is_gen_components_topomap:
        gen_components_topomap(
            raw_dic=raw_dic,  # 输入 Raw 对象字典
            components_dir=components_dir,  # 保存 components topomap 图片的文件夹
            decim=3,  # ICA 拟合时每隔多少个采样点取一个点，不改变 Raw 的采样率
            ica_cache_dir=ica_cache_dir,  # ICA 缓存文件夹，为 None 时则重新建立 ICA，且不进行缓存
            version=version,  # 缓存版本号
        )

    # 根据 topomap 选出的潜在伪迹 IC，生成对应 properties 图供进一步伪迹确认
    if is_gen_properties:
        gen_properties(
            raw_dic=raw_dic,  # 输入 Raw 对象字典
            properties_dir=properties_dir,  # 保存 properties 图片的文件夹
            ica_exclude_file=ica_exlude_file,  # 读取人工标注的潜在 IC 字典
            decim=3,  # ICA 拟合时每隔多少个采样点取一个点，不改变 Raw 的采样率
            ica_cache_dir=ica_cache_dir,  # ICA 缓存文件夹，为 None 时则重新建立 ICA，且不进行缓存
            version=version,  # 缓存版本号
        )

    # 根据确认后的伪迹索引，排除对应 IC 并重建 Raw
    if is_mark_artifacts_and_build:
        raw_dic = mark_artifacts_and_build(
            raw_dic=raw_dic,  # 输入 Raw 对象字典
            raw_cleaned_cache_dir=raw_cleaned_cache_dir,  # 缓存 ICA 处理后的 Raw 对象
            ica_exclude_file=ica_exlude_file,  # 读取人工确认的伪迹 IC 字典中的 final_exclude
            decim=3,  # ICA 拟合时每隔多少个采样点取一个点，不改变 Raw 的采样率
            ica_cache_dir=ica_cache_dir,  # ICA 缓存文件夹，为 None 时则重新建立 ICA，且不进行缓存
            version=version,  # 缓存版本号
        )

    # 在 ICA 后进行运动想象频段滤波
    if is_after_filter:
        raw_dic = after_filter(
            # 对象参数
            raw_dic=raw_dic,  # 输入 Raw 对象字典
            raw_after_filter_cache_dir=raw_after_filter_cache_dir,  # 后滤波后 Raw 对象缓存文件夹，为 None 时则重新滤波，且不进行缓存

            # filter 参数
            is_filter=True,  # 决定是否进行带通滤波
            l_freq=8.0,  # 低频边界，单位 Hz；None 表示不做高通
            h_freq=30.0,  # 高频边界，单位 Hz；None 表示不做低通
            picks='eeg',  # 滤波通道；可传类型、名称或索引
            filter_length='auto',  # FIR 长度；可传 "auto"、"10s" 等时间字符串或采样点数
            l_trans_bandwidth='auto',  # 低频端过渡带宽度，单位 Hz，仅用于 FIR
            h_trans_bandwidth='auto',  # 高频端过渡带宽度，单位 Hz，仅用于 FIR
            n_jobs=None,  # 并行任务数；"cuda" 仅用于支持 CUDA 的 FIR
            method='fir',  # 滤波方法
            iir_params=None,  # IIR 配置；None 时默认四阶 Butterworth，仅用于 IIR
            phase='zero',  # 相位处理方式；"forward" 仅用于 IIR，"minimum" 系列仅用于 FIR
            fir_window='hamming',  # FIR 窗函数，仅用于 FIR
            fir_design='firwin',  # FIR 设计方法，仅用于 FIR
            verbose=False,  # 日志级别；False 关闭日志，None 使用默认级别

            # notch_filter 参数
            is_notch_filter=False,  # 决定是否进行陷波滤波
            notch_freqs=None,  # 陷波中心频率，单位 Hz；None 仅用于 spectrum_fit 自动检测
            notch_picks='eeg',  # 滤波通道；可传类型、名称或索引
            notch_filter_length='auto',  # FIR 长度；spectrum_fit 中为拟合窗口长度，"auto" 对应 10 秒
            notch_notch_widths=None,  # 阻带宽度，单位 Hz；None 时采用 freqs / 200
            notch_trans_bandwidth=1.0,  # 过渡带宽度，单位 Hz，用于 FIR 或 IIR
            notch_n_jobs=None,  # 并行任务数；"cuda" 仅用于支持 CUDA 的 FIR
            notch_method='fir',  # 陷波方法；spectrum_fit 用于拟合并去除正弦噪声
            notch_iir_params=None,  # IIR 配置；None 时默认四阶 Butterworth，仅用于 IIR
            notch_phase='zero',  # 相位处理方式；"forward" 仅用于 IIR，"minimum" 系列仅用于 FIR
            notch_fir_window='hamming',  # FIR 窗函数，仅用于 FIR
            notch_fir_design='firwin',  # FIR 设计方法，仅用于 FIR
            notch_verbose=False,  # 日志级别；False 关闭日志，None 使用默认级别

            version=version,  # 缓存版本号
        )

if __name__ == '__main__':
    main(
        is_view_waveforms=False,
        is_exclude_bads_and_interpolate=True,
        is_pre_filter=True,
        is_set_reference=True,
        is_gen_components_topomap=False,
        is_gen_properties=True,
        is_mark_artifacts_and_build=False,
        is_after_filter=False,
    )
