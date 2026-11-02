"""CLI entry for reproducible batch work; uses same core as GUI."""
import argparse
from pathlib import Path
from .model import Settings
from .io import read_files
from .processing import process
from .demo import save_demo
from .export import export_result


def main(argv=None):
    p=argparse.ArgumentParser(description='TEM 水害辅助识别处理系统 1.0')
    p.add_argument('inputs',nargs='*',help='CSV/TXT/XLSX 文件 (长表/宽表)')
    p.add_argument('--demo',action='store_true',help='生成演示数据并处理')
    p.add_argument('--output','-o',default='tem_output')
    p.add_argument('--line',default=None)
    p.add_argument('--tx-area',type=float,default=100.0)
    p.add_argument('--tx-turns',type=int,default=1)
    p.add_argument('--current',type=float,default=1.0)
    p.add_argument('--rx-area',type=float,default=1.0)
    p.add_argument('--rx-turns',type=int,default=1)
    p.add_argument('--noise-floor',type=float,default=1e-13)
    p.add_argument('--rho-threshold',type=float,default=35.0)
    p.add_argument('--depth-factor',type=float,default=1.0)
    p.add_argument('--no-images',action='store_true')
    opt=p.parse_args(argv)
    cfg=Settings(tx_area_m2=opt.tx_area,tx_turns=opt.tx_turns,current_a=opt.current,
                 rx_area_m2=opt.rx_area,rx_turns=opt.rx_turns,
                 noise_floor_t_per_s=opt.noise_floor,low_rho_ohm_m=opt.rho_threshold,
                 depth_factor=opt.depth_factor)
    if opt.demo:
        demo=Path(opt.output)/'demo_raw.csv';demo.parent.mkdir(parents=True,exist_ok=True)
        save_demo(demo,cfg);opt.inputs=[str(demo)];opt.line='DEMO'
    if not opt.inputs:
        p.error('需要至少一个数据文件或 --demo')
    result=process(read_files(opt.inputs),cfg,opt.line)
    path=export_result(result,opt.output,with_images=not opt.no_images)
    print(f'导出: {path.resolve()}\n有效时间门: {result.quality["accepted_gates"]}/{result.quality["total_gates"]}'
          f'\n低阻候选区: {len(result.anomalies)}（不是水体确证）')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
