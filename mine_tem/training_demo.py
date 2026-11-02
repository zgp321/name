"""Teaching-only halfspace examples for ML unit tests, not a useful ore/water classifier."""
import numpy as np
from .model import Settings
from .processing import late_time_dbdt


def make_training_demo(path,n_samples=80,n_times=28,n_depths=18,seed=2026):
    rng=np.random.default_rng(seed)
    time=np.geomspace(1.5e-5,8e-4,n_times)
    depth=np.linspace(5,150,n_depths)
    rho=10**rng.uniform(np.log10(15),np.log10(230),size=n_samples)
    sigma=np.array([late_time_dbdt(r,time,Settings().tx_moment) for r in rho])
    sigma*=1+rng.normal(0,.01,sigma.shape)
    target=np.repeat(rho[:,None],n_depths,axis=1)
    np.savez_compressed(path,signals=sigma.astype('float32'),
                        resistivity=target.astype('float32'),
                        times_s=time,depths_m=depth)
    return str(path)
