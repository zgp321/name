import numpy as np
import pytest
from mine_tem.learn import train,predict,load_pairs
from mine_tem.training_demo import make_training_demo


def test_supervised_training_smoke(tmp_path):
    pytest.importorskip('torch')
    data=tmp_path/'train.npz'
    make_training_demo(data,n_samples=20,n_times=12,n_depths=6)
    x,y,t,d=load_pairs(data)
    assert x.shape==(20,12)
    output=tmp_path/'model'
    summary=train(data,output,epochs=2)
    assert summary['epochs']==2
    got_depth,got_rho=predict(output,x[0],t)
    assert np.allclose(got_depth,d)
    assert np.isfinite(got_rho).all() and (got_rho>0).all()
