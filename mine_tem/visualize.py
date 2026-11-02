"""Matplotlib figures for the measured/processed gates and pseudo-sections."""
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from .model import Result

matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK JP','Noto Serif CJK JP','Microsoft YaHei','Noto Sans CJK SC','SimHei','DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_decay(result: Result, fig: Figure | None=None) -> Figure:
    fig = fig or Figure(figsize=(10,5),dpi=100)
    fig.clear(); ax=fig.add_subplot(111)
    stations = result.readings['station'].unique()
    pick = stations[np.unique(np.linspace(0,len(stations)-1,min(6,len(stations))).astype(int))]
    for st in pick:
        d=result.readings[result.readings['station']==st]
        ax.loglog(d['time_s'],np.abs(d['raw_dbdt_t_s']),alpha=.45,linewidth=1)
        ax.loglog(d['time_s'],np.abs(d['dbdt_clean_t_s']),marker='.',markersize=3,linewidth=1,label=str(st))
    ax.set(title='TEM 衰减曲线 | 原始(浅色) / 质控后',xlabel='时间 t (s)',ylabel='|dBz/dt| (T/s)')
    ax.grid(True,which='both',alpha=.2)
    ax.legend(ncol=2,fontsize=8)
    fig.tight_layout()
    return fig


def plot_section(result: Result, fig: Figure | None = None, candidates=True) -> Figure:
    fig = fig or Figure(figsize=(10,5),dpi=100)
    fig.clear(); ax=fig.add_subplot(111)
    p=result.profile
    x=np.array(p['x_m']);z=np.array(p['depth_m']);grid=np.array(p['rho_ohm_m'])
    finite=grid[np.isfinite(grid)&(grid>0)]
    if not len(finite):
        ax.text(.5,.5,'无有效成像数据',transform=ax.transAxes,ha='center')
        return fig
    vmin=max(float(np.nanpercentile(finite,5)),.01)
    vmax=max(vmin*1.1,float(np.nanpercentile(finite,95)))
    im=ax.pcolormesh(x,z,np.ma.masked_invalid(grid),shading='nearest',cmap='turbo_r',norm=LogNorm(vmin=vmin,vmax=vmax))
    if candidates:
        mask=p.get('candidate_mask',np.zeros_like(grid,dtype=bool))
        if np.any(mask):
            ax.contour(x,z,mask.astype(float),levels=[.5],colors=['#aa163c'],linewidths=1.5)
    ax.invert_yaxis()
    ax.set(title='TEM 晚期视电阻率伪剖面 | 红框为低阻候选区',xlabel='测线里程 X (m)',ylabel='视深度 (m)')
    fig.colorbar(im,ax=ax,pad=.01,label='视电阻率 (Ω·m)')
    fig.tight_layout()
    return fig


def plot_3d_illustration(result: Result, fig: Figure | None=None) -> Figure:
    """2.5D extrusion of a 2D anomaly mask, illustrative only, no volume estimate."""
    fig=fig or Figure(figsize=(10,5),dpi=100)
    fig.clear();ax=fig.add_subplot(111,projection='3d')
    p=result.profile
    mask=p.get('candidate_mask',np.zeros_like(p['rho_ohm_m'],dtype=bool))
    iz,ix=np.where(mask)
    if len(ix):
        # Plot same profile at three synthetic y positions; this is no 3D inversion.
        for yi in np.linspace(-result.settings.display_extrusion_width_m/2,
                               result.settings.display_extrusion_width_m/2,3):
            ax.scatter(np.array(p['x_m'])[ix],np.full(len(ix),yi),
                       -np.array(p['depth_m'])[iz],s=3,alpha=.25,c='#db3f49')
    ax.set(xlabel='里程 X (m)',ylabel='假设横向延伸 Y (m)',zlabel='视深度 (m)')
    ax.set_title('异常区 2.5D 示意（非三维测量/反演）')
    fig.tight_layout()
    return fig


def plot_joint_section(result: Result, boreholes=None, fig: Figure | None = None) -> Figure:
    """Show a pseudo-section with borehole intervals as geometry-only overlays."""
    fig = fig or Figure(figsize=(10, 5), dpi=100)
    fig.clear(); ax = fig.add_subplot(111)
    p = result.profile
    x = np.asarray(p['x_m'], dtype=float); z = np.asarray(p['depth_m'], dtype=float)
    grid = np.asarray(p['rho_ohm_m'])
    finite = grid[np.isfinite(grid) & (grid > 0)]
    if len(finite):
        vmin = max(float(np.nanpercentile(finite, 5)), .01)
        vmax = max(vmin * 1.1, float(np.nanpercentile(finite, 95)))
        im = ax.pcolormesh(x, z, np.ma.masked_invalid(grid), shading='nearest',
                           cmap='turbo_r', norm=LogNorm(vmin=vmin, vmax=vmax))
        fig.colorbar(im, ax=ax, pad=.01, label='视电阻率 (Ω·m)')
    else:
        ax.text(.5, .5, '无有效成像数据', transform=ax.transAxes, ha='center')
    mask = p.get('candidate_mask', np.zeros_like(grid, dtype=bool))
    if np.any(mask):
        ax.contour(x, z, mask.astype(float), levels=[.5], colors=['#aa163c'], linewidths=1.5)
    if boreholes is not None and len(boreholes):
        colors = {}
        for row in boreholes.itertuples(index=False):
            lithology = str(getattr(row, 'lithology', ''))
            if lithology not in colors:
                # A stable, muted color derived from the text only supports
                # visual grouping; it has no geological semantics.
                colors[lithology] = plt_color = matplotlib.colormaps['tab20'](
                    len(colors) % 20 / 20)
            x0 = float(row.x_m)
            ax.axvline(x0, color='#263f48', linewidth=1, alpha=.7)
            ax.text(x0, 1.01, str(row.borehole_id), transform=ax.get_xaxis_transform(),
                    rotation=90, ha='center', va='bottom', fontsize=8, color='#263f48')
            top = float(row.depth_top_m); bottom = float(row.depth_bottom_m)
            ax.add_patch(Rectangle((x0 - max((x[-1] - x[0]) * .004, .4), top),
                                   max((x[-1] - x[0]) * .008, .8), bottom - top,
                                   facecolor=colors[lithology], edgecolor='none', alpha=.3,
                                   clip_on=False))
    ax.invert_yaxis()
    ax.set(title='TEM 伪剖面与钻孔区间叠加 | 伪深度几何对照',
           xlabel='测线里程 X (m)', ylabel='视深度 / 钻孔区间深度 (m)')
    fig.tight_layout()
    return fig
