import json
import pickle
from pathlib import Path
from typing import Any, Literal, Callable
from uuid import uuid4

import mne
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from scipy.io import loadmat

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

# 路径配置
configs_folder = Path('./configs')
configs_folder.mkdir(parents=True, exist_ok=True)
bad_channels_exclude_file = configs_folder / 'bad_channels_exclude.json'
ica_exlude_file = configs_folder / 'ica_exclude.json'


data_folder = Path('./data')

cache_folder = Path('./cache')
cache_folder = cache_folder / f'cache_v{version}'  # 给不同版本的数据设置不同缓存路径

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
    cache_id_file: Path,  # 缓存标识路径
    cache_file: Path,  # 缓存路径
    key_components: tuple,  # 生成缓存键的特征元素
    version: int,  # 缓存版本号，内部组成 cache key，以防外部写 cache key 遗漏
    is_start: bool = False,  # 此时是否采用 start 处缓存机制，不依赖上游缓存标识
    read_fn: Callable[[Path], Any] = _read_pickle,  # 根据读取方法，选择对应读取函数
    legacy_key_kind: Literal['raw', 'mat'] | None = None,  # 兼容旧键中已保存的参数
    ) -> tuple[tuple[Any, ...], str | None, Any | None]:
    """
    功能：
    辅助函数，统一完成中间数据缓存读取工作。

    机制：
    比较调用方提供的 key_components 和版本号，相同则直接加载缓存。
    缓存起点忽略键的第一项，即上游缓存标识，仍比较本步参数和版本号。
    不额外检查 Raw 内容、Info、annotations 或源文件状态；配置修改时间由调用方纳入键。
    未纳入键的输入内容变化后，需要手动增加 version 或删除对应缓存。

    输出：
    传出缓存的 cache_key、cache_id 和缓存数据；未命中时后两项为 None。
    """
    cache_key = (*key_components, version)
    if cache_key_file.exists() and cache_file.exists():
        try:
            with cache_key_file.open('rb') as f:
                saved_key = pickle.load(f)
            # 将旧缓存键转换为当前参数键，再比较本步参数和版本号
            saved_key = _normalize_legacy_cache_key(saved_key, legacy_key_kind)

            with cache_id_file.open('rb') as f:
                cache_id = pickle.load(f)

            if is_start and (saved_key[1:] == cache_key[1:]):  # 若为缓存起点，则不比较上游缓存标识
                cache = read_fn(cache_file)
                return cache_key, cache_id, cache
            
            if saved_key == cache_key:
                cache = read_fn(cache_file)
                return cache_key, cache_id, cache
        except (OSError, EOFError, pickle.UnpicklingError):
            pass

    return cache_key, None, None


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
        cache_id: str,  # 缓存标识
        data: Any,  # 需要缓存的数据
        cache_key_file: Path,  # 目标缓存键路径
        cache_id_file: Path,  # 目标缓存标识路径
        cache_data_file: Path,  # 目标缓存路径
        dump_fn: Callable[[Any, Path], Any],  # 用于缓存写入的方法函数
    ):
    """
    功能：
    辅助函数，统一保存缓存键、缓存标识和数据，写入失败时向调用方抛出异常。
    """
    cache_key_file.parent.mkdir(parents=True, exist_ok=True)
    cache_data_file.parent.mkdir(parents=True, exist_ok=True)
    cache_id_file.parent.mkdir(parents=True, exist_ok=True)

    temporary_key_file = cache_key_file.with_suffix(cache_key_file.suffix + '.temp')
    temporary_id_file = cache_id_file.with_suffix(cache_id_file.suffix + '.temp')

    try:
        with temporary_key_file.open('wb') as f:
            pickle.dump(cache_key, f, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_key_file.replace(cache_key_file)
        
        with temporary_id_file.open('wb') as f:
            pickle.dump(cache_id, f, protocol=pickle.HIGHEST_PROTOCOL)
        temporary_id_file.replace(cache_id_file)
        
        dump_fn(data, cache_data_file)  # fif 对象写入方法限制了后缀前的字符，因此无法用 .temp
        
        
    finally:
        temporary_key_file.unlink(missing_ok=True)
        temporary_id_file.unlink(missing_ok=True)


def load_cached_mat(
    data_folder: Path,  # 存放 mat 文件的文件夹
    subjects: list,  # 要读取的被试名列表
    data_cache_dir: Path | None = None, # 原始数据缓存目录，不给则不读取缓存数据，且不进行缓存
    convert_index: bool = True,  # True 将 mat 1-based 索引转化为 Python 的 0-based 索引
    convert_shape: bool = True,  # True 将 mat 的电压数据 shape 进行转置
    scale_factor: float | None = None,  # 电压单位转换系数
    version: int = 1, # 缓存版本号
    ) -> tuple[Any, str]:
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

    cache_meta = {
        "cache_key": None,
        "cache_id": None,
    }
    np_data_dic = None  # 默认为 None，等待赋值，使得 data_cache_dir 为 None 时默认进行数据操作

    if data_cache_dir is not None:

        data_cache_key_components = (
            str(Path(data_folder).resolve()), tuple(subjects),
            convert_index, convert_shape, scale_factor,
        )

        data_cache_key_file = data_cache_dir / f'data.key.pkl'
        data_cache_id_file = data_cache_dir / f'data.id.pkl'
        data_cache_file = data_cache_dir / f'data.pkl'

        # 缓存命中直接载入缓存
        cache_meta['cache_key'], cache_meta['cache_id'], np_data_dic = _load_cache(
            cache_key_file=data_cache_key_file,
            cache_id_file=data_cache_id_file,
            cache_file=data_cache_file,
            key_components=data_cache_key_components,
            version=version,
            legacy_key_kind='mat',
        )

    need_new = (
        cache_meta['cache_id'] is None
        or np_data_dic is None
    )
    
    if need_new:
        cache_meta['cache_id'] = uuid4().hex

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
                cache_key=cache_meta['cache_key'],
                cache_id=cache_meta['cache_id'],
                data=np_data_dic,
                cache_key_file=data_cache_key_file,
                cache_id_file=data_cache_id_file,
                cache_data_file=data_cache_file,
                dump_fn=_dump_pickle,
            )

    return np_data_dic, cache_meta['cache_id']


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
        start: str,  # 缓存起点
        upstream_cache_id: str,  # 传入上游 cache_id
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
    清理坏道并插值后的 Raw 对象字典，以及各 session 的 cache_id。

    说明：
    所有 Raw 对象字典统一命名为 raw_dic；
    方便直接在 main 函数通过增加或去除操作统一工作流，以免反复修改传入下游函数的 Raw 字典变量名。
    """
    this_start = "is_exclude_bads_and_interpolate"
    is_start = (this_start == start)  # 判断是否为缓存起点，若为缓存起点，则不依赖上游标识决定是否读取缓存

    cache_meta = {
        'upstream_cache_id': upstream_cache_id,
        'cache_keys':{},
        'cache_ids':{},
    }

    with open(bad_channels_exclude_file, 'r', encoding='utf-8') as f:
        bad_channels_dic = json.load(f)

    for sbj_session, raw in raw_dic.items():
        raw_interp = None
        cache_meta['cache_ids'][sbj_session] = None  # 预先赋值保证 need_new 判断不报错

        if raw_interp_cache_dir is not None:
            raw_interp_cache_key_file = raw_interp_cache_dir / f'{sbj_session}_interp.key.pkl'
            raw_interp_cache_id_file = raw_interp_cache_dir / f'{sbj_session}_interp.cache_id.pkl'
            raw_interp_cache_file = raw_interp_cache_dir / f'{sbj_session}_interp_raw.fif'

            # 上游标识、配置文件修改时间和处理参数共同决定是否命中缓存
            raw_interp_cache_key_components = (
                upstream_cache_id,
                # 配置文件修改时间变化时，需要更新缓存
                Path(bad_channels_exclude_file).resolve().stat().st_mtime_ns,  
                reset_bads,
                method,
                mode,
            )

            cache_meta['cache_keys'][sbj_session], cache_meta['cache_ids'][sbj_session], raw_interp = _load_cache(
                is_start=is_start,
                cache_key_file=raw_interp_cache_key_file,
                cache_id_file=raw_interp_cache_id_file,
                cache_file=raw_interp_cache_file,
                key_components=raw_interp_cache_key_components,
                read_fn=_read_raw,
                version=version,
            )

        need_new = (
            cache_meta['cache_ids'][sbj_session] is None
            or raw_interp is None 
        )

        if need_new:
            if is_start:
                raise RuntimeError(f'{sbj_session}：当前步骤是缓存起点，但缓存数据和 cache_id 无法读取。')
            
            cache_meta['cache_ids'][sbj_session] = uuid4().hex

            raw_interp = raw.copy()
            raw_interp.info['bads'] = bad_channels_dic[sbj_session]
            raw_interp = raw_interp.load_data().interpolate_bads(
                reset_bads=reset_bads,
                mode=mode,
                origin='auto',
                method=method,
                verbose=False,
            )

            # 如果 raw_interp 已缓存就不用再 dump，因此条件缩进
            if raw_interp_cache_dir is not None:
                raw_interp_cache_dir.mkdir(parents=True, exist_ok=True)
                # 统一保存缓存键、缓存标识和 Raw 数据
                _dump_cache(
                    cache_key=cache_meta['cache_keys'][sbj_session],
                    cache_id=cache_meta['cache_ids'][sbj_session],
                    data=raw_interp,
                    cache_key_file=raw_interp_cache_key_file,
                    cache_id_file=raw_interp_cache_id_file,
                    cache_data_file=raw_interp_cache_file,
                    dump_fn=_dump_raw,
                )

        raw_dic[sbj_session] = raw_interp
        print(f'已完成{sbj_session}的坏道清除和插值')


    return raw_dic, cache_meta['cache_ids']


def filter(
        # 对象参数
        start: str,  # 缓存起点
        upstream_cache_id: dict,  # 传入上游各 session 的 cache_id
        raw_dic: dict,  # 输入 Raw 对象字典
        raw_pre_filter_cache_dir: Path | None = None,  # 前滤波后 Raw 对象缓存文件夹，为 None 时则重新滤波，且不进行缓存
        this_start: Literal['is_pre_filter', 'is_after_filter'] = 'is_pre_filter',  # 前后滤波共用本函数，指定本步标记
        
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
    在 ICA 处理前或处理后，根据参数进行陷波滤波和带通滤波。

    输出：
    滤波后的 Raw 对象字典，以及各 session 的 cache_id。

    说明：
    先进行陷波滤波，再进行带通滤波，两步由各自的开关决定是否执行；
    上游标识、滤波参数和版本号未变更时，直接加载缓存；
    两个开关均为 False 时，直接返回 raw_dic 和上游 cache_id。
    """
    is_start = (this_start == start)  # 判断是否为缓存起点，若为缓存起点，则不依赖上游标识决定是否读取缓存

    cache_meta = {
        'upstream_cache_id': upstream_cache_id,
        'cache_keys':{},
        'cache_ids':{},
    }

    if not is_filter and not is_notch_filter:
        return raw_dic, upstream_cache_id

    for sbj_session, raw in raw_dic.items():
        raw_pre_filter = None
        cache_meta['cache_ids'][sbj_session] = None  # 预先赋值保证 need_new 判断不报错

        if raw_pre_filter_cache_dir is not None:
            raw_pre_filter_cache_key_file = raw_pre_filter_cache_dir / f'{sbj_session}_pre_filter.key.pkl'
            raw_pre_filter_cache_id_file = raw_pre_filter_cache_dir / f'{sbj_session}_pre_filter.cache_id.pkl'
            raw_pre_filter_cache_file = raw_pre_filter_cache_dir / f'{sbj_session}_pre_filter_raw.fif'

            # 每个 session 只比较自己的上游标识，前后滤波通过不同目录区分
            raw_pre_filter_cache_key_components = (
                upstream_cache_id[sbj_session],
                is_filter, l_freq, h_freq, picks, filter_length,
                l_trans_bandwidth, h_trans_bandwidth, n_jobs, method,
                iir_params, phase, fir_window, fir_design,
                is_notch_filter, notch_freqs, notch_picks, notch_filter_length,
                notch_notch_widths, notch_trans_bandwidth, notch_n_jobs,
                notch_method, notch_iir_params, notch_phase,
                notch_fir_window, notch_fir_design,
            )

            cache_meta['cache_keys'][sbj_session], cache_meta['cache_ids'][sbj_session], raw_pre_filter = _load_cache(
                is_start=is_start,
                cache_key_file=raw_pre_filter_cache_key_file,
                cache_id_file=raw_pre_filter_cache_id_file,
                cache_file=raw_pre_filter_cache_file,
                key_components=raw_pre_filter_cache_key_components,
                read_fn=_read_raw,
                version=version,
            )

        need_new = (
            cache_meta['cache_ids'][sbj_session] is None
            or raw_pre_filter is None
        )

        if need_new:
            if is_start:
                raise RuntimeError(f'{sbj_session}：当前步骤是缓存起点，但缓存数据和 cache_id 无法读取。')

            cache_meta['cache_ids'][sbj_session] = uuid4().hex

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
                # 统一保存缓存键、缓存标识和 Raw 数据
                _dump_cache(
                    cache_key=cache_meta['cache_keys'][sbj_session],
                    cache_id=cache_meta['cache_ids'][sbj_session],
                    data=raw_pre_filter,
                    cache_key_file=raw_pre_filter_cache_key_file,
                    cache_id_file=raw_pre_filter_cache_id_file,
                    cache_data_file=raw_pre_filter_cache_file,
                    dump_fn=_dump_raw,
                )

        raw_dic[sbj_session] = raw_pre_filter
        print(f'已完成{sbj_session}的前滤波' if this_start == 'is_pre_filter' else f'已完成{sbj_session}的后滤波')

    return raw_dic, cache_meta['cache_ids']


def set_reference(
        # 对象参数
        start: str,  # 缓存起点
        upstream_cache_id: dict,  # 传入上游各 session 的 cache_id
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
    比较上游标识、参考参数和版本号；命中时直接加载已重参考的 Raw。

    输出：
    重参考后的 Raw 对象字典，以及各 session 的 cache_id。
    """
    this_start = "is_set_reference"
    is_start = (this_start == start)  # 判断是否为缓存起点，若为缓存起点，则不依赖上游标识决定是否读取缓存

    cache_meta = {
        'upstream_cache_id': upstream_cache_id,
        'cache_keys':{},
        'cache_ids':{},
    }

    if not raw_dic:
        return raw_dic, cache_meta['cache_ids']

    forward = None  # 平均参考或指定通道参考不需要正向模型

    for sbj_session, raw in raw_dic.items():
        raw_referenced = None
        cache_meta['cache_ids'][sbj_session] = None  # 预先赋值保证 need_new 判断不报错

        if raw_referenced_cache_dir is not None:
            raw_referenced_cache_key_file = raw_referenced_cache_dir / f'{sbj_session}_referenced.key.pkl'
            raw_referenced_cache_id_file = raw_referenced_cache_dir / f'{sbj_session}_referenced.cache_id.pkl'
            raw_referenced_cache_file = raw_referenced_cache_dir / f'{sbj_session}_referenced_raw.fif'

            # 球模型参数也参与缓存键；verbose 只影响日志，不参与缓存键
            raw_referenced_cache_key_components = (
                upstream_cache_id[sbj_session],
                ref_channels, ch_type, joint, exclude, pos, trans,
            )

            cache_meta['cache_keys'][sbj_session], cache_meta['cache_ids'][sbj_session], raw_referenced = _load_cache(
                is_start=is_start,
                cache_key_file=raw_referenced_cache_key_file,
                cache_id_file=raw_referenced_cache_id_file,
                cache_file=raw_referenced_cache_file,
                key_components=raw_referenced_cache_key_components,
                read_fn=_read_raw,
                version=version,
            )

        need_new = (
            cache_meta['cache_ids'][sbj_session] is None
            or raw_referenced is None
        )

        if need_new:
            if is_start:
                raise RuntimeError(f'{sbj_session}：当前步骤是缓存起点，但缓存数据和 cache_id 无法读取。')

            cache_meta['cache_ids'][sbj_session] = uuid4().hex

            if ref_channels == 'REST' and forward is None:  # 只在重新计算时建立一次头模型，命中缓存时不依赖输入 Raw
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

            raw_referenced = raw.copy().load_data().set_eeg_reference(
                ref_channels=ref_channels,
                ch_type=ch_type,
                forward=forward,
                joint=joint,
                verbose=verbose,
            )

            if raw_referenced_cache_dir is not None:
                raw_referenced_cache_dir.mkdir(parents=True, exist_ok=True)
                # 统一保存缓存键、缓存标识和 Raw 数据
                _dump_cache(
                    cache_key=cache_meta['cache_keys'][sbj_session],
                    cache_id=cache_meta['cache_ids'][sbj_session],
                    data=raw_referenced,
                    cache_key_file=raw_referenced_cache_key_file,
                    cache_id_file=raw_referenced_cache_id_file,
                    cache_data_file=raw_referenced_cache_file,
                    dump_fn=_dump_raw,
                )

        raw_dic[sbj_session] = raw_referenced

        print(f'已完成{sbj_session}的{ref_channels}重参考')

    return raw_dic, cache_meta['cache_ids']


def ica_cache(
        start: str,  # 缓存起点
        upstream_cache_id: dict,  # 传入上游各 session 的 cache_id
        raw_dic: dict,  # 输入 Raw 对象字典
        decim: int,  # ICA 拟合时每隔多少个采样点取一个点，不改变 Raw 的采样率
        ica_cache_dir: Path | None = None,  # ICA 缓存文件夹，为 None 时则重新建立 ICA，且不进行缓存
        version: int = 1,  # 缓存版本号
    ):
    """
    功能：
    统一读取 ICA 缓存；未命中时建立 ICA 并保存缓存。

    输出：
    各 session 的 ICA 对象字典，以及 ICA 模型对应的 cache_id。
    本步不修改 Raw，ICA 标识独立于 Raw 标识，供绘图和伪迹重建使用。
    """
    this_start = "is_ica_cache"
    is_start = (this_start == start)  # 判断是否为缓存起点，若为缓存起点，则不依赖上游标识决定是否读取缓存

    cache_meta = {
        'upstream_cache_id': upstream_cache_id,
        'cache_keys':{},
        'cache_ids':{},
    }
    ica_dic = {}

    for sbj_session, raw in raw_dic.items():
        ica = None
        cache_meta['cache_ids'][sbj_session] = None  # 预先赋值保证 need_new 判断不报错

        if ica_cache_dir is not None:
            ica_cache_key_file = ica_cache_dir / f'{sbj_session}_ica.key.pkl'
            ica_cache_id_file = ica_cache_dir / f'{sbj_session}_ica.cache_id.pkl'
            ica_cache_file = ica_cache_dir / f'{sbj_session}_ica.fif'

            # 比较上游标识和拟合参数；固定的 ICA 方法改变时需要增加 version
            ica_cache_key_components = (
                upstream_cache_id[sbj_session],
                decim,
            )

            cache_meta['cache_keys'][sbj_session], cache_meta['cache_ids'][sbj_session], ica = _load_cache(
                is_start=is_start,
                cache_key_file=ica_cache_key_file,
                cache_id_file=ica_cache_id_file,
                cache_file=ica_cache_file,
                key_components=ica_cache_key_components,
                read_fn=_read_ica,
                version=version,
            )

        need_new = (
            cache_meta['cache_ids'][sbj_session] is None
            or ica is None
        )

        if need_new:
            if is_start:
                raise RuntimeError(f'{sbj_session}：当前步骤是缓存起点，但缓存数据和 cache_id 无法读取。')

            cache_meta['cache_ids'][sbj_session] = uuid4().hex

            raw.load_data()
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
                # 统一保存缓存键、缓存标识和 ICA 模型
                _dump_cache(
                    cache_key=cache_meta['cache_keys'][sbj_session],
                    cache_id=cache_meta['cache_ids'][sbj_session],
                    data=ica,
                    cache_key_file=ica_cache_key_file,
                    cache_id_file=ica_cache_id_file,
                    cache_data_file=ica_cache_file,
                    dump_fn=_dump_ica,
                )

        ica_dic[sbj_session] = ica
        print(f'已完成{sbj_session}的 ICA 建立')

    return ica_dic, cache_meta['cache_ids']


def gen_components_topomap(
        raw_dic: dict,  # 输入 Raw 对象字典
        ica_dic: dict,  # 输入已读取或建立的 ICA 对象字典
        components_dir: Path,  # 保存 components topomap 图片的文件夹
    ):
    """
    功能：
    使用已读取或建立的 ICA 对象，生成各 session 的 ICA 成分地形图。
    """

    components_dir.mkdir(parents=True, exist_ok=True)
    for sbj_session, raw in raw_dic.items():
        ica = ica_dic[sbj_session]

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
        ica_dic: dict,  # 输入已读取或建立的 ICA 对象字典
        properties_dir: Path,  # 保存 properties 图片的文件夹
        ica_exclude_file: Path,  # 读取人工标注的潜在 IC 字典
        t_start_s: float,  # 绘制波形图起始时间，以秒为单位
        duration: float,  # 绘制波形图的时长，以秒为单位
    ):
    """
    功能：
    绘制潜在伪迹主成分的 properties 图。

    使用说明：
    根据 topomap 图，分辨各 Raw 数据中的主成分哪些存在伪迹；
    将存在伪迹的主成分的索引，填写到 './configs/ica_exclude.json' 的 'topomap_exclude' 下对应 Raw 的列表里；
    该绘图会根据列表绘制潜在伪迹 IC （被 'topomap_exclude' 标记的 IC) 的 properties 图。
    本函数只使用传入的 ICA 对象，不建立 ICA 或写入模型缓存。
    上游变更后需更新 ICA，并重新确认 IC 索引。
    """
    
    with open(ica_exclude_file, 'r', encoding='utf-8') as f:
        pot_IC_dic = json.load(f)

    topomap_exclude_dic = pot_IC_dic['topomap_exclude']

    for sbj_session, raw in raw_dic.items():
        topomap_exclude = topomap_exclude_dic[sbj_session]
        if not topomap_exclude:
            print(f'{sbj_session}未标记潜在伪迹 IC，跳过 properties 图')
            continue

        properties_sbj_session_dir = properties_dir / sbj_session
        properties_sbj_session_dir.mkdir(parents=True, exist_ok=True)

        raw.load_data()
        ica = ica_dic[sbj_session]

        sfreq = raw.info['sfreq']
        start = int(round(t_start_s * sfreq))
        stop = int(min(raw.n_times, round((t_start_s + duration) * sfreq)))

        sources = ica.get_sources(raw, start=start, stop=stop)
        times = raw.times[start:stop]

        for ica_idx in topomap_exclude:
            fig = plt.figure(figsize=(18, 10))
            ax_topomap = fig.add_axes([0.08, 0.62, 0.30, 0.30], label="topomap")  # 地形图。
            ax_image = fig.add_axes([0.60, 0.70, 0.35, 0.22], label="image")  # 分段热图。
            ax_erp = fig.add_axes([0.60, 0.62, 0.35, 0.08], label="erp")  # 分段平均波形。
            ax_spectrum = fig.add_axes([0.08, 0.35, 0.42, 0.19], label="spectrum")  # 功率谱。
            ax_variance = fig.add_axes([0.60, 0.35, 0.35, 0.19], label="variance")  # 分段方差。
            properties_axes = [ax_topomap, ax_image, ax_erp, ax_spectrum, ax_variance]  # MNE 要求的顺序。
            ica.plot_properties(
                inst=raw,
                picks=[ica_idx],
                axes=properties_axes,
                show=False,
            )

            waveform = sources.get_data(picks=[ica_idx])[0]
            ax_waveform = fig.add_axes([0.08, 0.08, 0.87, 0.16])
            ax_waveform.plot(times, waveform, color="tab:blue", linewidth=0.8)
            ax_waveform.set_ylabel(f"ICA{ica_idx:03d} continuous waveform")
            ax_waveform.set_xlabel('Time starting from {start} s')
            ax_waveform.set_xlim(times[0], times[-1])
            ax_waveform.grid(alpha=0.5)

            fig.subplots_adjust(top=0.9)
            fig.suptitle(f'{sbj_session} | ICA{ica_idx:03d}', fontsize=16, fontweight='bold')
            fig.savefig(properties_sbj_session_dir / f"{sbj_session}_ICA{ica_idx}_properties.png")
            plt.close(fig)

        print(f'已绘制{sbj_session} 各主成分 properties 图')


def mark_artifacts_and_build(
        start: str,  # 缓存起点
        upstream_cache_id: dict,  # 传入上游 Raw 对应的各 session cache_id
        raw_dic: dict,  # 输入 Raw 对象字典
        raw_cleaned_cache_dir: Path | None,  # 缓存 ICA 处理后的 Raw 对象，为 None 时则重新重建，且不进行缓存
        ica_exclude_file: Path,  # 读取人工确认的伪迹 IC 字典中的 final_exclude
        ica_dic: dict,  # 输入已读取或建立的 ICA 对象字典
        ica_cache_id: dict,  # 传入各 session 的 ICA 模型标识
        version: int = 1,  # 缓存版本号
    ):
    """
    功能：
    根据人工读图后指定的伪迹 IC 索引，清理伪迹，并重建 Raw 数据。

    使用说明：
    在通过 properties 图确认伪迹后，
    将伪迹的主成分的索引，填写到 './configs/ica_exclude.json' 的 'final_exclude' 下对应 Raw 的列表里；
    该操作会根据 'final_exclude' 中的标记从 ICA 对象中排除对应主成分，并重建 Raw 数据。
    Raw 标识和 ICA 标识共同作为上游依赖，人工标注修改时间参与缓存比较。

    输出：
    清理伪迹并重建后的 Raw 对象字典，以及各 session 的 cache_id。
    """
    this_start = "is_mark_artifacts_and_build"
    is_start = (this_start == start)  # 判断是否为缓存起点，若为缓存起点，则不依赖上游标识决定是否读取缓存

    cache_meta = {
        'upstream_cache_id': upstream_cache_id,
        'cache_keys':{},
        'cache_ids':{},
    }

    with open(ica_exclude_file, 'r', encoding='utf-8') as f:
        pot_IC_dic = json.load(f)
    final_exclude_dic = pot_IC_dic['final_exclude']

    for sbj_session, raw in raw_dic.items():
        raw_cleaned = None
        cache_meta['cache_ids'][sbj_session] = None  # 预先赋值保证 need_new 判断不报错

        if raw_cleaned_cache_dir is not None:
            raw_cleaned_cache_key_file = raw_cleaned_cache_dir / f'{sbj_session}_cleaned.key.pkl'
            raw_cleaned_cache_id_file = raw_cleaned_cache_dir / f'{sbj_session}_cleaned.cache_id.pkl'
            raw_cleaned_cache_file = raw_cleaned_cache_dir / f'{sbj_session}_cleaned_raw.fif'

            # 两路上游标识放在同一项，缓存起点统一跳过；标注修改后需要更新缓存
            raw_cleaned_cache_key_components = (
                (upstream_cache_id[sbj_session], ica_cache_id.get(sbj_session)),
                Path(ica_exclude_file).resolve().stat().st_mtime_ns,
            )

            cache_meta['cache_keys'][sbj_session], cache_meta['cache_ids'][sbj_session], raw_cleaned = _load_cache(
                is_start=is_start,
                cache_key_file=raw_cleaned_cache_key_file,
                cache_id_file=raw_cleaned_cache_id_file,
                cache_file=raw_cleaned_cache_file,
                key_components=raw_cleaned_cache_key_components,
                read_fn=_read_raw,
                version=version,
            )

        need_new = (
            cache_meta['cache_ids'][sbj_session] is None
            or raw_cleaned is None
        )

        if need_new:
            if is_start:
                raise RuntimeError(f'{sbj_session}：当前步骤是缓存起点，但缓存数据和 cache_id 无法读取。')

            cache_meta['cache_ids'][sbj_session] = uuid4().hex

            raw_cleaned = raw.copy().load_data()
            ica = ica_dic[sbj_session].copy()
            ica.exclude = final_exclude_dic[sbj_session]
            ica.apply(raw_cleaned)

            # 如果 raw_cleaned 已缓存就不用再 dump，因此条件缩进
            if raw_cleaned_cache_dir is not None:
                raw_cleaned_cache_dir.mkdir(parents=True, exist_ok=True)
                # 统一保存缓存键、缓存标识和 Raw 数据
                _dump_cache(
                    cache_key=cache_meta['cache_keys'][sbj_session],
                    cache_id=cache_meta['cache_ids'][sbj_session],
                    data=raw_cleaned,
                    cache_key_file=raw_cleaned_cache_key_file,
                    cache_id_file=raw_cleaned_cache_id_file,
                    cache_data_file=raw_cleaned_cache_file,
                    dump_fn=_dump_raw,
                )

        raw_dic[sbj_session] = raw_cleaned
        print(f'已完成{sbj_session}的伪迹清理和重建')

    return raw_dic, cache_meta['cache_ids']


def main(
        is_dic: dict,  # 决定操作开关的字典对象，用于指定是否执行某一操作
        start: str = "is_exclude_bads_and_interpolate",  # 缓存恢复起点，从其标记的步骤开始执行，且该步骤不依赖上游标记默认读取缓存。
    ):

    # 不修改调用方传入的字典；这样可复用同一份步骤设置重复运行。
    is_dic = is_dic.copy()

    # 处理步骤条件判别报错
    optional_steps = (
        "is_load_cached_mat",
        "is_build_valid_raw",
        "is_view_waveforms",
        "is_exclude_bads_and_interpolate",
        "is_pre_filter",
        "is_set_reference",
        "is_ica_cache",
        "is_gen_components_topomap",
        "is_gen_properties",
        "is_mark_artifacts_and_build",
        "is_after_filter",
    )
    missing_steps = [step for step in optional_steps if step not in is_dic]
    if missing_steps:
        raise KeyError(f'字典中缺少步骤开关：{missing_steps}')

    # 处理不合法 start
    if start not in optional_steps:
        raise KeyError(f'起点必须是可选步骤之一：{start}')
    if not is_dic[start]:
        raise ValueError(f'缓存起点对应的步骤必须开启：{start}')

    # start 前的步骤统一关闭，不执行处理，也不更新这些步骤的缓存
    for step in optional_steps:
        if step == start:
            break
        is_dic[step] = False

    np_data_dic = None
    cache_id = None  # MAT 使用整体标识，Raw 处理步骤使用各 session 的标识字典
    raw_dic = {}
    ica_dic = {}
    ica_cache_id = {}  # ICA 只建立模型，不覆盖当前 Raw 的 cache_id
    raw_after_filter_cache_dir = Path('./cache/raw_after_filter_4-40')

    # 从中间步骤恢复时，用本步已有缓存键取得 session 名；Raw 由起点函数读取
    start_cache_dirs = {
        'is_exclude_bads_and_interpolate': raw_interp_cache_dir,
        'is_pre_filter': raw_pre_filter_cache_dir,
        'is_set_reference': raw_referenced_cache_dir,
        'is_ica_cache': ica_cache_dir,
        'is_gen_components_topomap': ica_cache_dir,
        'is_gen_properties': ica_cache_dir,
        'is_mark_artifacts_and_build': raw_cleaned_cache_dir,
        'is_after_filter': raw_after_filter_cache_dir,
    }
    if start in start_cache_dirs:
        for subject in subjects:
            for cache_key_file in sorted(start_cache_dirs[start].glob(f'{subject}[0-9]*_*.key.pkl')):
                sbj_session = cache_key_file.name.split('_')[0]
                raw_dic[sbj_session] = None
        if not raw_dic:
            raise RuntimeError(f'{start}：当前步骤是缓存起点，但缓存键无法读取。')
        cache_id = {sbj_session: None for sbj_session in raw_dic}

    # 建立 Raw 和绘制原始波形没有独立数据缓存，只读取已有 MAT 缓存提供输入
    if start in ('is_build_valid_raw', 'is_view_waveforms'):
        _, cache_id, np_data_dic = _load_cache(
            cache_key_file=data_cache_dir / 'data.key.pkl',
            cache_id_file=data_cache_dir / 'data.id.pkl',
            cache_file=data_cache_dir / 'data.pkl',
            key_components=(str(Path(data_folder).resolve()), tuple(subjects), True, True, 1e-6),
            version=version,
            legacy_key_kind='mat',
        )
        if cache_id is None or np_data_dic is None:
            raise RuntimeError(f'{start}：起点所需的 MAT 缓存数据和 cache_id 无法读取。')
        if start == 'is_view_waveforms':
            raw_dic = build_valid_raw(
                data_dic=np_data_dic,
                ch_names=ch_names,
                ch_types=ch_types,
                sfreq=sfreq,
                montage_name=montage_name,
            )

    # ICA 缓存只保存模型；恢复到 ICA 或其绘图步骤时，还需读取模型对应的 Raw 输入
    if start in ('is_ica_cache', 'is_gen_components_topomap', 'is_gen_properties'):
        for sbj_session in raw_dic:
            ica_cache_key_file = ica_cache_dir / f'{sbj_session}_ica.key.pkl'
            upstream_cache_id = _read_pickle(ica_cache_key_file)[0]

            # 按标识寻找实际用于拟合 ICA 的 Raw，不重新执行上游处理
            raw_cache_files = (
                (raw_referenced_cache_dir / f'{sbj_session}_referenced_raw.fif', raw_referenced_cache_dir / f'{sbj_session}_referenced.cache_id.pkl'),
                (raw_pre_filter_cache_dir / f'{sbj_session}_pre_filter_raw.fif', raw_pre_filter_cache_dir / f'{sbj_session}_pre_filter.cache_id.pkl'),
                (raw_interp_cache_dir / f'{sbj_session}_interp_raw.fif', raw_interp_cache_dir / f'{sbj_session}_interp.cache_id.pkl'),
            )
            for raw_cache_file, raw_cache_id_file in raw_cache_files:
                if raw_cache_file.exists() and raw_cache_id_file.exists():
                    if _read_pickle(raw_cache_id_file) == upstream_cache_id:
                        raw_dic[sbj_session] = _read_raw(raw_cache_file)
                        cache_id[sbj_session] = upstream_cache_id
                        break

            # ICA 前未启用其他 Raw 处理时，输入来自已有 MAT 缓存
            if raw_dic[sbj_session] is None:
                data_cache_id = _read_pickle(data_cache_dir / 'data.id.pkl')
                if data_cache_id != upstream_cache_id:
                    raise RuntimeError(f'{sbj_session}：ICA 对应的上游 Raw 缓存无法读取。')
                if np_data_dic is None:
                    np_data_dic = _read_pickle(data_cache_dir / 'data.pkl')
                    raw_data_dic = build_valid_raw(
                        data_dic=np_data_dic,
                        ch_names=ch_names,
                        ch_types=ch_types,
                        sfreq=sfreq,
                        montage_name=montage_name,
                    )
                raw_dic[sbj_session] = raw_data_dic[sbj_session]
                cache_id[sbj_session] = data_cache_id

    # 读取原始数据（已缓存则直接加载）
    if is_dic['is_load_cached_mat']:
        np_data_dic, cache_id = load_cached_mat(
            data_folder=data_folder,  # 存放 mat 文件的文件夹
            subjects=subjects,  # 要读取的被试名列表
            data_cache_dir=data_cache_dir,  # 原始数据缓存目录，为 None 时则不读取缓存数据，且不进行缓存
            convert_index=True,  # True 将 mat 1-based 索引转化为 Python 的 0-based 索引
            convert_shape=True,  # True 将 mat 的电压数据 shape 进行转置
            scale_factor=1e-6,  # 电压单位转换系数
            version=version,  # 缓存版本号
        )

    # 将原始数据转化为 MNE Raw 对象
    if is_dic['is_build_valid_raw']:
        raw_dic = build_valid_raw(
            data_dic=np_data_dic,  # 存放数据的 dict 对象
            ch_names=ch_names,  # 通道名
            ch_types=ch_types,  # 通道种类
            sfreq=sfreq,  # 采样频率
            montage_name=montage_name,  # 电极布局模板名
        )

    # 绘制各被试各 session 的波形图，以观察坏道
    if is_dic["is_view_waveforms"]:
        view_waveforms(
            raw_dic=raw_dic,  # 输入 Raw 对象字典
            waveforms_dir=waveforms_dir,  # 存储波形图路径
            scalings=35e-6,  # 缩放尺度，参数需自己尝试，大致 100e-6 左右上下调
            n_channels=len(ch_names),  # 不滚动滑轮时直接显示的通道数
            start=0,  # 从什么时间开始显示，单位秒
            duration=30,  # 显示时长，单位秒
        )

    # 根据人工标记的坏道进行插值，保留完整通道
    if is_dic["is_exclude_bads_and_interpolate"]:
        raw_dic, cache_id = exclude_bads_and_interpolate(
            start=start,  # 缓存起点
            upstream_cache_id=cache_id,  # 传入上游 cache_id
            raw_dic=raw_dic,  # 输入 Raw 对象字典
            bad_channels_exclude_file=bad_channels_exclude_file,  # 读取人工标注的坏道字典
            raw_interp_cache_dir=raw_interp_cache_dir,  # 插值后 Raw 对象缓存文件夹，为 None 时则重新插值，且不进行缓存
            reset_bads=True,  # 是否将插值完的通道移出 raw.info['bads']
            method={'eeg':'spline'},  # 插值方法，None 时默认方法为 {'eeg':'spline'}
            mode='accurate',  # 最小范数插值（MNE）下的计算模式，决定计算速度（与精度权衡）
            version=version, # 缓存版本号
        )

    # 未启用插值时，将 MAT 的整体标识传给每个 session，便于后续逐个比较
    if not isinstance(cache_id, dict):
        cache_id = {sbj_session: cache_id for sbj_session in raw_dic}

    # 在 ICA 前进行高通滤波和工频陷波
    if is_dic["is_pre_filter"]:
        raw_dic, cache_id = filter(
            # 对象参数
            start=start,  # 缓存起点
            upstream_cache_id=cache_id,  # 传入上游各 session 的 cache_id
            raw_dic=raw_dic,  # 输入 Raw 对象字典
            raw_pre_filter_cache_dir=raw_pre_filter_cache_dir,  # 前滤波后 Raw 对象缓存文件夹，为 None 时则重新滤波，且不进行缓存
            this_start='is_pre_filter',  # 本次调用执行前滤波

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
    if is_dic["is_set_reference"]:
        raw_dic, cache_id = set_reference(
            # 对象参数
            start=start,  # 缓存起点
            upstream_cache_id=cache_id,  # 传入上游各 session 的 cache_id
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

    # 读取 Raw 数据生成 ICA，并缓存
    if is_dic["is_ica_cache"]:
        ica_dic, ica_cache_id = ica_cache(
            start=start,  # 缓存起点
            upstream_cache_id=cache_id,  # 传入当前 Raw 对应的各 session cache_id
            raw_dic=raw_dic,  # 输入 Raw 对象字典
            ica_cache_dir=ica_cache_dir,  # ICA 缓存文件夹，为 None 时则重新建立 ICA，且不进行缓存
            decim=3,  # ICA 拟合时每隔多少个采样点取一个点，不改变 Raw 的采样率
            version=version,  # 缓存版本号
        )

    # 未执行 ICA 建立步骤但需要绘图或重建时，只读取已有模型和标识，不更新 ICA 缓存
    # 若重建本身是缓存起点，则由本步读取已重建的 Raw，不要求读取此前的 ICA
    if not ica_dic and (
        is_dic['is_gen_components_topomap']
        or is_dic['is_gen_properties']
        or (is_dic['is_mark_artifacts_and_build'] and start != 'is_mark_artifacts_and_build')
    ):
        for sbj_session in raw_dic:
            ica_cache_file = ica_cache_dir / f'{sbj_session}_ica.fif'
            ica_cache_id_file = ica_cache_dir / f'{sbj_session}_ica.cache_id.pkl'
            ica_dic[sbj_session] = _read_ica(ica_cache_file)
            ica_cache_id[sbj_session] = _read_pickle(ica_cache_id_file)
            if ica_cache_id[sbj_session] is None:
                raise RuntimeError(f'{sbj_session}：ICA 的 cache_id 无法读取。')

    # 读取 Raw 数据，生成 ICA 成分 topomap
    if is_dic["is_gen_components_topomap"]:
        gen_components_topomap(
            raw_dic=raw_dic,  # 输入 Raw 对象字典
            ica_dic=ica_dic,  # 输入已读取或建立的 ICA 对象字典
            components_dir=components_dir,  # 保存 components topomap 图片的文件夹
        )

    # 根据 topomap 选出的潜在伪迹 IC，生成对应 properties 图供进一步伪迹确认
    if is_dic["is_gen_properties"]:
        gen_properties(
            raw_dic=raw_dic,  # 输入 Raw 对象字典
            ica_dic=ica_dic,  # 输入已读取或建立的 ICA 对象字典
            properties_dir=properties_dir,  # 保存 properties 图片的文件夹
            ica_exclude_file=ica_exlude_file,  # 读取人工标注的潜在 IC 字典
            t_start_s=0.0,  # 绘制波形图起始时间，以秒为单位
            duration=10.0,  # 绘制波形图的时长，以秒为单位
        )

    # 根据确认后的伪迹索引，排除对应 IC 并重建 Raw
    if is_dic["is_mark_artifacts_and_build"]:
        raw_dic, cache_id = mark_artifacts_and_build(
            start=start,  # 缓存起点
            upstream_cache_id=cache_id,  # 传入当前 Raw 对应的各 session cache_id
            raw_dic=raw_dic,  # 输入 Raw 对象字典
            raw_cleaned_cache_dir=raw_cleaned_cache_dir,  # 缓存 ICA 处理后的 Raw 对象
            ica_exclude_file=ica_exlude_file,  # 读取人工确认的伪迹 IC 字典中的 final_exclude
            ica_dic=ica_dic,  # 输入已读取或建立的 ICA 对象字典
            ica_cache_id=ica_cache_id,  # 传入各 session 的 ICA 模型标识
            version=version,  # 缓存版本号
        )

    # 在 ICA 后进行运动想象频段滤波
    if is_dic["is_after_filter"]:
        raw_dic, cache_id = filter(
            # 对象参数
            start=start,  # 缓存起点
            upstream_cache_id=cache_id,  # 传入上游各 session 的 cache_id
            raw_dic=raw_dic,  # 输入 Raw 对象字典
            raw_pre_filter_cache_dir=raw_after_filter_cache_dir,  # 前后滤波共用本函数，此处传入后滤波缓存目录
            this_start='is_after_filter',  # 本次调用执行后滤波

            # filter 参数
            is_filter=True,  # 决定是否进行带通滤波
            l_freq=4.0,  # 低频边界，单位 Hz；None 表示不做高通
            h_freq=40.0,  # 高频边界，单位 Hz；None 表示不做低通
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

    return raw_dic

if __name__ == '__main__':
    main(
        start = "is_ica_cache",  # 缓存恢复起点，从其标记的步骤开始执行，且该步骤不依赖上游标记默认读取缓存。
        is_dic = {  # 决定操作开关的字典对象，用于指定是否执行某一操作
        'is_load_cached_mat': True,  # True 则加载原始数据或原始数据缓存
        'is_build_valid_raw': True,  # True 则通过加载的 ndarray 数据建立 Raw 对象
        "is_view_waveforms": False,  # True 则画原始波形图
        "is_exclude_bads_and_interpolate": True,  # True 则根据人工标记清理坏道并插值
        "is_pre_filter": True,  # True 则在 ICA 前进行滤波
        "is_set_reference": True,  # True 则进行重参考
        "is_ica_cache": True,  # True 则读取 ICA 缓存，需要时或不存在缓存时则建立 ICA
        "is_gen_components_topomap": False,  # True 则画 ICA 成分地形图
        "is_gen_properties": True,  # True 则画主成分 properties 图
        "is_mark_artifacts_and_build": False,  # True 则根据标记伪迹 IC 重建 Raw
        "is_after_filter": False,  # True 则在 ICA 后进行滤波
        },
    )
