"""Reproducible results, provenance and high-resolution vector/raster figures."""
from pathlib import Path
from datetime import datetime, timezone
import json
import shutil
import tempfile
import numpy as np
import pandas as pd
from .model import Result
from .visualize import plot_decay,plot_section,plot_3d_illustration


def export_result(result:Result,folder:str|Path,with_images=True) -> Path:
    """Stage a complete directory and roll back failed replacement attempts.

    Existing unrelated files are retained. This guards ordinary write/rename
    failures; it is not a power-loss transaction across two directory renames.
    """
    original = Path(folder)
    if original.is_symlink():
        raise ValueError('成果目录不能是符号链接')
    target = original.resolve()
    cwd = Path.cwd().resolve()
    if target.resolve() == cwd or target.resolve() in cwd.parents:
        raise ValueError('请使用独立成果子目录，不能替换当前工作目录或其父目录')
    if target.exists() and not target.is_dir():
        raise ValueError('成果路径已存在且不是目录')
    target.parent.mkdir(parents=True, exist_ok=True)
    transaction = Path(tempfile.mkdtemp(prefix='.tem-export-', dir=target.parent))
    stage = transaction / 'result'
    backup = transaction / 'previous'
    preserve_backup = False
    try:
        if target.exists():
            shutil.copytree(target, stage, symlinks=True)
        else:
            stage.mkdir()
        for name in ('gates_processed.csv', 'anomaly_candidates.csv', 'pseudo_section_xyz.csv',
                     'pseudo_section_arrays.npz', 'manifest.json', '成果说明.md',
                     'decay.png', 'pseudo_section.png', 'anomalies_2p5d.png'):
            if (stage / name).is_symlink():
                raise ValueError(f'成果文件不能是符号链接：{name}')
        if not with_images:
            for name in ('decay.png', 'pseudo_section.png', 'anomalies_2p5d.png'):
                (stage / name).unlink(missing_ok=True)
        _write_result(result, stage, with_images)
        if target.exists():
            target.rename(backup)
        try:
            stage.rename(target)
        except BaseException:
            if backup.exists():
                try:
                    backup.rename(target)
                except OSError as exc:
                    preserve_backup = True
                    raise RuntimeError(f'成果替换失败，原成果保留于 {backup}') from exc
            raise
        return original
    finally:
        if not preserve_backup:
            shutil.rmtree(transaction, ignore_errors=True)


def _write_result(result:Result,folder:str|Path,with_images=True) -> Path:
    out=Path(folder);out.mkdir(parents=True,exist_ok=True)
    result.readings.to_csv(out/'gates_processed.csv',index=False,encoding='utf-8-sig')
    result.anomalies.to_csv(out/'anomaly_candidates.csv',index=False,encoding='utf-8-sig')
    p=result.profile
    # Long format is easy to import into GIS/ParaView/Surfer.
    xx,zz=np.meshgrid(p['x_m'],p['depth_m'])
    pd.DataFrame({'x_m':xx.ravel(),'pseudo_depth_m':zz.ravel(),
                  'rho_apparent_ohm_m':p['rho_ohm_m'].ravel(),
                  'low_resistivity_candidate':p['candidate_mask'].ravel()}).to_csv(
                  out/'pseudo_section_xyz.csv',index=False,encoding='utf-8-sig')
    np.savez_compressed(out/'pseudo_section_arrays.npz',x_m=p['x_m'],depth_m=p['depth_m'],
                         rho_ohm_m=p['rho_ohm_m'],candidate_mask=p['candidate_mask'])
    manifest={'software':'矿井瞬变电磁TEM水害智能识别处理系统V1.0',
              'engine':result.quality.get('engine','late-time half-space apparent-resistivity quicklook (not inversion)'),
              'utc_created':datetime.now(timezone.utc).isoformat(),
              'line_id':result.line_id,'settings':result.settings.to_dict(),
              'quality':result.quality,'candidate_count':len(result.anomalies),
              'view_type':p['view_type'],'demo_only':result.line_id=='DEMO'}
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    readme=(f'# TEM 测线 {result.line_id} 辅助判读成果\n\n'
            f'- 成果性质：晚期视电阻率 **伪剖面，不是二维反演**。\n'
            f'- 时间门：{result.quality["total_gates"]}；有效：{result.quality["accepted_gates"]}。\n'
            f'- 低阻候选异常：{len(result.anomalies)} 处。**不能仅由低阻判定有水**。\n'
            '- 原始与质控结果见 gates_processed.csv，参数和质量统计见 manifest.json。\n'
            '- 伪深度与面积不能用于直接估算积水储量；三维图只是二维剖面沿假定宽度拉伸。\n'
            '- 请结合钻探、充水条件、水文地质、井下金属设施及其他物探成果解释。\n')
    (out/'成果说明.md').write_text(readme,encoding='utf-8')
    if with_images:
        for name,func in [('decay.png',plot_decay),('pseudo_section.png',plot_section),
                          ('anomalies_2p5d.png',plot_3d_illustration)]:
            fig=func(result)
            fig.savefig(out/name,dpi=240,facecolor='white')
    return out
