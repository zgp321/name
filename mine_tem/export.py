"""Reproducible results, provenance and high-resolution vector/raster figures."""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
import pandas as pd
from .model import Result
from .visualize import plot_decay,plot_section,plot_3d_illustration


def export_result(result:Result,folder:str|Path,with_images=True) -> Path:
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
