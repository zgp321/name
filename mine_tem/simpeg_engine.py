"""Optional physical 1-D layered-earth TEM inversion powered by SimPEG.

Assumptions: surface central loop, loop approximated circular, step-off waveform,
flat non-magnetic layered half-space. Not valid for mines in whole-space without
appropriate physics or metal-support correction. Tested only when SimPEG installed.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from .model import Settings,Result
from .processing import process,pick_anomalies


@dataclass
class LayeredFit:
    station: str
    x_m: float
    depths_mid_m: np.ndarray
    resistivity_ohm_m: np.ndarray
    observed_dbdt: np.ndarray
    predicted_dbdt: np.ndarray
    rms_log10: float
    success: bool


def invert_station_simpeg(g:pd.DataFrame,settings:Settings,n_layers:int=8,max_depth_m:float=180.,
                          max_nfev:int=45, prior_rho:float=100., smooth_lambda:float=.16)->LayeredFit:
    try:
        from simpeg import maps
        import simpeg.electromagnetics.time_domain as tdem
    except ImportError as exc:
        raise RuntimeError('未安装 SimPEG。安装 python>=3.12，执行 pip install "simpeg>=0.25,<0.27"') from exc
    d=g[g['accepted'] & np.isfinite(g['dbdt_clean_t_s'])].sort_values('time_s')
    if len(d)<8:
        raise ValueError(f'测点 {g.iloc[0]["station"]} 有效门数不足(至少8门)')
    times=d['time_s'].to_numpy(float)
    obs=d['dbdt_clean_t_s'].to_numpy(float)
    radius=np.sqrt(settings.tx_area_m2/np.pi)
    rx=tdem.receivers.PointMagneticFluxTimeDerivative(np.array([[0.,0.,0.]]),times,'z')
    source=tdem.sources.CircularLoop([rx],radius=radius,location=np.array([0.,0.,0.]),
                waveform=tdem.sources.StepOffWaveform(),
                current=settings.current_a*settings.tx_turns)
    survey=tdem.Survey([source])
    thicknesses=np.geomspace(2.,max_depth_m/(n_layers*1.8),n_layers-1)
    # Scale thicknesses so sum reaches desired depth of investigation.
    thicknesses=thicknesses*(max_depth_m / thicknesses.sum())
    simulation=tdem.Simulation1DLayered(survey=survey,thicknesses=thicknesses,
                                        sigmaMap=maps.ExpMap(nP=n_layers))
    log_ref=np.log(1/prior_rho)
    # Define bounded log conductivity parameters and Tikhonov roughness prior.
    sigma0=np.full(n_layers,log_ref)
    scale=np.maximum(np.abs(obs),settings.noise_floor_t_per_s*5)
    def residual(m):
        prediction=np.asarray(simulation.dpred(m),float)
        if not np.all(np.isfinite(prediction)):
            return np.full(len(obs)+n_layers-1,1e5)
        return np.r_[(prediction-obs)/scale, smooth_lambda*np.diff(m)]
    fit=least_squares(residual,sigma0,bounds=(np.log(1e-5),np.log(10)),
                      max_nfev=max_nfev,xtol=1e-5,ftol=1e-5,gtol=1e-5)
    pred=np.asarray(simulation.dpred(fit.x),float)
    edges=np.r_[0.,np.cumsum(thicknesses),max_depth_m+thicknesses[-1]]
    mids=(edges[:-1]+edges[1:])/2
    rms=float(np.sqrt(np.mean(np.log10(np.maximum(np.abs(pred),1e-30)/
                         np.maximum(np.abs(obs),1e-30))**2)))
    return LayeredFit(str(g.iloc[0]['station']),float(g.iloc[0]['x_m']),mids,
                      np.exp(-fit.x),obs,pred,rms,bool(fit.success))


def invert_line_simpeg(df:pd.DataFrame,settings:Settings|None=None,line_id:str|None=None,
                       n_layers:int=8,max_depth_m:float=180.,limit_stations:int=250,
                       callback=None)->tuple[Result,list[LayeredFit]]:
    """Surface 1D inversions stitched into a pseudo-2D section; not 2D inverse."""
    settings=settings or Settings()
    basic=process(df,settings,line_id)
    sts=basic.readings['station'].unique()
    if len(sts)>limit_stations:
        raise ValueError(f'反演测点数 {len(sts)} 超出设定上限 {limit_stations}，请分段处理')
    fits=[]
    for i,st in enumerate(sts):
        try:
            f=invert_station_simpeg(basic.readings[basic.readings['station']==st],settings,
                                    n_layers,max_depth_m)
            fits.append(f)
        except ValueError:
            pass
        if callback:callback(i+1,len(sts),str(st))
    if len(fits)<2:
        raise ValueError('有效反演结果不足2个测点')
    xs=np.array([f.x_m for f in fits]);ix=np.argsort(xs);xs=xs[ix]
    zs=np.linspace(0,max_depth_m,140)
    grid=np.zeros((len(zs),len(xs)))
    for j,k in enumerate(ix):
        f=fits[k]
        grid[:,j]=np.interp(zs,f.depths_mid_m,f.resistivity_ohm_m)
    basic.profile={'x_m':xs,'depth_m':zs,'rho_ohm_m':grid,'stations':[fits[k].station for k in ix],
                    'view_type':'SimPEG 1D layered-earth soundings stitched into a section; NOT 2D inversion'}
    anomalies,mask=pick_anomalies(basic.profile,settings)
    basic.profile['candidate_mask']=mask
    basic.anomalies=anomalies
    basic.quality['engine']='SimPEG Simulation1DLayered; SciPy regularized least-squares'
    basic.quality['fit_rms_log10_median']=float(np.median([f.rms_log10 for f in fits]))
    basic.quality['limitations'] += ['地表半空间+理想阶跃关断+圆形中心回线模型，不适用于井下全空间原始数据；外业仪器波形/频带/增益未建模',
                                  '多个独立1D反演沿线拼接，不代表严格二维或三维反演']
    return basic,fits
