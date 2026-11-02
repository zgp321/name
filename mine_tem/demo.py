"""Reproducible synthetic central-loop halfspace response; NOT field observations."""
import numpy as np
import pandas as pd
from .model import Settings
from .processing import late_time_dbdt


def make_demo(settings: Settings | None = None, seed: int = 2026, n_stations=43) -> pd.DataFrame:
    cfg = settings or Settings()
    rng = np.random.default_rng(seed)
    times = np.geomspace(1.5e-5, 8e-4, 33)
    rows = []
    for idx, x in enumerate(np.linspace(0, 210, n_stations)):
        # Ideal independent halfspaces along the line. Change in resistivity is
        # lateral (one homogeneous sounding per station), not a layered target.
        rho = (110.0 - 96.0*np.exp(-0.5*((x-106)/23)**2)
               - 42.0*np.exp(-0.5*((x-177)/12)**2))
        amp = late_time_dbdt(rho, times, cfg.tx_moment)
        amplitude = amp*(1+rng.normal(0,0.025,len(times)))
        if idx in (5,17,28):
            amplitude[8] *= 20.0  # isolated synthetic metallic-style spike
        if idx in (12,38):
            amplitude[-2:] = cfg.noise_floor_t_per_s * 0.25
        for t, a in zip(times,amplitude):
            rows.append({'line_id':'DEMO','station':f'P{idx+1:03d}','x_m':round(float(x),3),
                         'y_m':0.0,'elevation_m':185.0,'time_s':float(t),
                         'dbdt_t_s':-float(a)})
    return pd.DataFrame(rows)


def save_demo(path, settings: Settings | None = None):
    df = make_demo(settings)
    df.to_csv(path,index=False,encoding='utf-8-sig')
    return str(path)
