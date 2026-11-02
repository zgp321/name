"""Safe training-data and model-metadata inspection; does not import PyTorch."""
from pathlib import Path
import numpy as np
from .learn import load_pairs, validate_axes


def _axis_summary(times, depths):
    return {'n_times': len(times), 'n_depths': len(depths),
            'time_range_s': [float(times[0]), float(times[-1])],
            'depth_range_m': [float(depths[0]), float(depths[-1])]}


def inspect_dataset(path):
    x, y, times, depths = load_pairs(path)
    return {**_axis_summary(times, depths), 'samples': len(x),
            'rho_range': [float(y.min()), float(y.max())],
            'path': str(Path(path).resolve())}


def inspect_model(path_or_prefix):
    prefix = str(path_or_prefix)
    if prefix.endswith('.pt'): prefix = prefix[:-3]
    weights = Path(prefix + '.pt')
    meta = Path(prefix + '_metadata.npz')
    if not weights.is_file() or not meta.is_file():
        raise FileNotFoundError('模型需要同名前缀的 .pt 与 _metadata.npz 文件')
    with np.load(meta, allow_pickle=False) as ds:
        required = {'times_s', 'depths_m', 'xmean', 'xstd', 'ymean', 'ystd'}
        if not required.issubset(ds.files):
            raise ValueError('模型元数据缺少：' + ', '.join(sorted(required - set(ds.files))))
        times = np.asarray(ds['times_s'], dtype=float)
        depths = np.asarray(ds['depths_m'], dtype=float)
        validate_axes(times, depths)
        if len(times) < 8 or len(depths) < 4:
            raise ValueError('模型至少需要8个时间门、4个深度层')
        for key, length in [('xmean', len(times)), ('xstd', len(times)),
                            ('ymean', len(depths)), ('ystd', len(depths))]:
            value = np.asarray(ds[key], dtype=float)
            if value.shape != (length,) or not np.isfinite(value).all():
                raise ValueError(f'模型 {key} 长度不符或含有非有限值')
            if key.endswith('std') and np.any(value <= 0):
                raise ValueError(f'模型 {key} 必须为正数')
    return {**_axis_summary(times, depths), 'prefix': prefix}
