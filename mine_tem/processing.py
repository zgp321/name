"""Auditable central-loop late-time TEM quicklook and water-risk candidate extraction.

NOT a full time-domain inversion; no field-proven water classifier. Field geometry,
current, gating, waveform and calibration MUST be checked by an engineer.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.ndimage import label, median_filter, gaussian_filter
from scipy.interpolate import PchipInterpolator
from .model import MU0, Settings, Result, validate_measurements


def late_time_dbdt(rho_ohm_m, times_s, tx_moment_am2):
    """Ideal central-loop half-space asymptotic |dBz/dt| (T/s).

    Eq.: |dBz/dt| = [mu0 * M/20] * [mu0/(pi*rho)]**(3/2) / t**(5/2).
    Not valid for arbitrary early time, loop offsets, transmitter turn-off, or whole-space mine geometry.
    """
    rho, t = np.asarray(rho_ohm_m, dtype=float), np.asarray(times_s, dtype=float)
    return MU0 * tx_moment_am2 / 20 * (MU0 / (np.pi * rho)) ** 1.5 / t**2.5


def late_time_rho(dbdt_t_per_s, times_s, tx_moment_am2):
    """Algebraic inversion of the ideal late-time dB/dt half-space formula."""
    amp, t = np.asarray(dbdt_t_per_s, dtype=float), np.asarray(times_s, dtype=float)
    c = MU0 * tx_moment_am2 / 20 * (MU0 / np.pi) ** 1.5
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        return (c / (np.abs(amp) * t**2.5)) ** (2.0 / 3.0)


def diffusion_pseudo_depth(rho_ohm_m, times_s, factor=1.0):
    """Diffusion-depth proxy sqrt(rho * t / (2 mu0)), not true geological depth."""
    with np.errstate(invalid='ignore'):
        return factor * np.sqrt(np.asarray(rho_ohm_m) * np.asarray(times_s) / (2 * MU0))


def _repair_station(g: pd.DataFrame, cfg: Settings) -> pd.DataFrame:
    a = g.sort_values('time_s').copy()
    # By Faraday induction: voltage = -N*A*dB/dt. The data must be in SI units.
    if 'dbdt_t_s' in a:
        a['raw_dbdt_t_s'] = a['dbdt_t_s'].astype(float)
        if 'voltage_v' in a:
            m = ~np.isfinite(a['raw_dbdt_t_s'].to_numpy())
            a.loc[m, 'raw_dbdt_t_s'] = -a.loc[m, 'voltage_v'] / (cfg.rx_turns * cfg.rx_area_m2)
    else:
        a['raw_dbdt_t_s'] = -a['voltage_v'].astype(float) / (cfg.rx_turns * cfg.rx_area_m2)
    vals = a['raw_dbdt_t_s'].to_numpy(dtype=float)
    above_floor = np.isfinite(vals) & (np.abs(vals) >= cfg.noise_floor_t_per_s)
    good = above_floor & (np.abs(vals) / cfg.noise_floor_t_per_s >= cfg.min_snr)
    if good.any():
        sign = np.sign(np.median(np.sign(vals[good])))
        if sign == 0:
            sign = np.sign(vals[good][0])
    else:
        sign = 1.0
    flip = good & (np.sign(vals) != sign)
    quality = np.full(len(a), '有效', dtype=object)
    quality[~good] = '低于噪声门限'
    quality[above_floor & ~good] = '低信噪比(排除)'
    quality[flip] = '极性反转(排除)'
    valid = good & ~flip
    mag = np.abs(vals)

    # Optional baseline removal is explicitly opt-in; cannot assume final gates are noise.
    if cfg.late_offset_correction and np.sum(valid) >= 10:
        tail = vals[valid][-3:]
        offset = float(np.median(tail))
        # Only correct if it is small compared to peak, otherwise mark and keep raw.
        if abs(offset) < 0.03 * np.nanmax(mag[valid]):
            vals = vals - offset
            mag = np.abs(vals)
            valid &= ((np.sign(vals) == sign) & (mag >= cfg.noise_floor_t_per_s)
                      & (mag / cfg.noise_floor_t_per_s >= cfg.min_snr))
            quality[~valid & (quality == '有效')] = '基线校正后无效'

    # Detect isolated spikes in log magnitude. Only interpolate interior good points.
    logmag = np.log10(np.maximum(mag, cfg.noise_floor_t_per_s / 100))
    if len(logmag) >= 5:
        baseline = median_filter(logmag, size=min(7, (len(logmag)//2)*2+1), mode='nearest')
        delta = np.abs(logmag - baseline)
        outliers = valid & (delta > cfg.spike_log10_threshold)
        # Large contaminated groups are excluded rather than automatically "repaired".
        if outliers.sum() <= max(1, int(0.2 * len(a))):
            for i in np.flatnonzero(outliers):
                left = np.flatnonzero(valid[:i] & ~outliers[:i])
                right = np.flatnonzero(valid[i+1:] & ~outliers[i+1:]) + i + 1
                if len(left) and len(right):
                    lo, hi = left[-1], right[0]
                    logmag[i] = np.interp(np.log(a.iloc[i]['time_s']),
                                           np.log([a.iloc[lo]['time_s'], a.iloc[hi]['time_s']]),
                                           [logmag[lo], logmag[hi]])
                    quality[i] = '尖峰修复(插值)'
                else:
                    valid[i] = False
                    quality[i] = '尖峰无法修复'
        else:
            valid[outliers] = False
            quality[outliers] = '大范围干扰(排除)'
    cleaned = 10**logmag
    cleaned[~valid] = np.nan
    rho = late_time_rho(cleaned, a['time_s'].to_numpy(), cfg.tx_moment)
    depth = diffusion_pseudo_depth(rho, a['time_s'].to_numpy(), cfg.depth_factor)
    # Absurd values are filtered; they are not geological interpretive evidence.
    valid &= np.isfinite(rho) & np.isfinite(depth) & (rho > 0.01) & (rho < 1e7) & (depth > 0) & (depth < 1e5)
    quality[~valid & np.isin(quality, ['有效','尖峰修复(插值)'])] = '计算越界(排除)'
    a['dbdt_clean_t_s'] = np.where(valid, cleaned * sign, np.nan)
    a['rho_apparent_ohm_m'] = np.where(valid, rho, np.nan)
    a['pseudo_depth_m'] = np.where(valid, depth, np.nan)
    a['snr_min'] = mag / cfg.noise_floor_t_per_s
    a['qc'] = quality
    a['accepted'] = valid
    return a


def build_profile(readings: pd.DataFrame, n_depth: int = 140) -> dict:
    """Interpolate late-time apparent resistivity station-by-station (pseudo-section).

    Does not perform 2D electromagnetic inversion or interpolate between soundings.
    """
    d = readings[readings['accepted'] & np.isfinite(readings['rho_apparent_ohm_m'])]
    positions = readings.groupby('station', sort=False)['x_m'].first().sort_values()
    xs = positions.to_numpy(float)
    stations = positions.index.astype(str).tolist()
    if len(xs) < 2 or not np.all(np.diff(xs) > 0):
        raise ValueError('二维剖面至少需要两个不同里程的测点，且 x_m 不能重复')
    limits = []
    for _, g in d.groupby('station'):
        if len(g) >= 3:
            limits.append(float(g['pseudo_depth_m'].quantile(0.9)))
    if not limits:
        raise ValueError('有效时间门不足，无法建立剖面')
    max_depth = np.clip(np.nanquantile(limits, 0.85), 10.0, 2000.0)
    zs = np.linspace(0, max_depth, n_depth)
    grid = np.full((n_depth, len(xs)), np.nan)
    for j, station in enumerate(stations):
        original = readings[readings['station'].astype(str) == station].sort_values('time_s')
        usable = (original['accepted'] & np.isfinite(original['rho_apparent_ohm_m'])
                  & (original['rho_apparent_ohm_m'] > 0) & np.isfinite(original['pseudo_depth_m']))
        # A rejected time gate breaks interpolation support. Never join valid
        # points across a QC gap, even when their pseudo-depths bracket it.
        segments = (~usable).cumsum()
        for _, g in original[usable].groupby(segments[usable], sort=False):
            depths = g['pseudo_depth_m'].to_numpy(float)
            logs = np.log10(g['rho_apparent_ohm_m'].to_numpy(float))
            unique, idx = np.unique(depths, return_index=True)
            if len(unique) < 3:
                continue
            valid = (zs >= unique[0]) & (zs <= unique[-1])
            grid[valid, j] = 10**PchipInterpolator(unique, logs[idx])(zs[valid])
    if not np.isfinite(grid).any():
        raise ValueError('连续有效时间门不足，无法建立剖面；请检查质控记录')
    return {'x_m': xs, 'depth_m': zs, 'rho_ohm_m': grid, 'stations': stations,
            'view_type': 'late-time apparent resistivity pseudo-section; NOT 2D inversion'}


def pick_anomalies(profile: dict, cfg: Settings) -> tuple[pd.DataFrame, np.ndarray]:
    z = profile['depth_m']; x = profile['x_m']; rho = profile['rho_ohm_m']
    # Comparisons are only made for depths with enough valid sounding coverage.
    level_med = np.full(len(z), np.nan)
    valid_rho = np.isfinite(rho) & (rho > 0)
    usable_rows = np.any(valid_rho, axis=1)
    if usable_rows.any():
        level_med[usable_rows] = np.nanmedian(np.where(valid_rho, rho, np.nan)[usable_rows], axis=1)
    good = valid_rho & (rho < cfg.low_rho_ohm_m) & (
        rho < cfg.contrast_ratio * np.maximum(level_med[:, None], 1e-10))
    if len(x) < 2 or len(z) < 2:
        return pd.DataFrame(), good
    dx = np.median(np.diff(x)); dz = np.median(np.diff(z))
    # Adjacent stations should be contiguous only when their spacing is not wildly irregular.
    discontinuities = np.diff(x) > 2.5 * dx
    labels = np.zeros(good.shape, dtype=int)
    edges = np.r_[0, np.flatnonzero(discontinuities) + 1, len(x)]
    count = 0
    for left, right in zip(edges[:-1], edges[1:]):
        local, n = label(good[:, left:right], structure=np.ones((3,3), dtype=int))
        labels[:, left:right] = np.where(local, local + count, 0)
        count += n
    rows = []
    cell = float(dx*dz)
    for i in range(1, count+1):
        zi, xi = np.where(labels == i)
        area = len(zi)*cell
        if area < cfg.min_area_m2:
            good[labels == i] = False
            continue
        vals = rho[zi,xi]
        rows.append({'id':f'A{i:03d}', 'x_start_m':float(x[xi.min()]),'x_end_m':float(x[xi.max()]),
                     'depth_top_m':float(z[zi.min()]), 'depth_bottom_m':float(z[zi.max()]),
                     'area_pseudo_m2':round(area,1), 'min_rho_ohm_m':round(float(np.nanmin(vals)),2),
                     'median_rho_ohm_m':round(float(np.nanmedian(vals)),2),
                     'interpretation':'低阻候选异常（不等同于含水体）',
                     'evidence_level':'需要钻探/地质/其他物探验证'})
    return pd.DataFrame(rows,columns=['id','x_start_m','x_end_m','depth_top_m','depth_bottom_m',
            'area_pseudo_m2','min_rho_ohm_m','median_rho_ohm_m','interpretation','evidence_level']), good


def process(df: pd.DataFrame, cfg: Settings | None = None, line_id: str | None = None) -> Result:
    cfg = cfg or Settings()
    cfg.validate()
    raw = validate_measurements(df)
    lines = list(raw['line_id'].unique())
    if line_id is None:
        if len(lines) != 1:
            raise ValueError('数据包含多条测线，请明确指定 line_id。可使用 --line 命令行参数')
        line_id = lines[0]
    if line_id not in lines:
        raise ValueError(f'找不到测线 {line_id}; 可用测线：{lines}')
    raw = raw[raw['line_id'] == line_id].copy()
    chunks = [_repair_station(g,cfg) for _,g in raw.groupby('station',sort=False)]
    clean = pd.concat(chunks,ignore_index=True)
    profile = build_profile(clean)
    anomalies, mask = pick_anomalies(profile,cfg)
    profile['candidate_mask'] = mask
    counts = clean['qc'].value_counts().to_dict()
    quality = {'total_gates':int(len(clean)), 'accepted_gates':int(clean['accepted'].sum()),
               'rejected_gates':int((~clean['accepted']).sum()),
               'stations':int(clean['station'].nunique()), 'qc_breakdown':counts,
               'limitations':['仅中心回线晚期近似；未建模关断波形、井下全空间效应、偏移距、耦合与磁导率变化',
                              '2D 图为 1D 视电阻率柱状拼接的伪剖面，非 2D/3D 反演',
                              '异常为低阻候选区，泥岩、黏土和金属支护可能造成类似响应',
                              '伪深度及异常范围需要测井、钻探及现场地质验证']}
    return Result(clean,profile,anomalies,quality,cfg,line_id)
