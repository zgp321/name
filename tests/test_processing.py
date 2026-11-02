import json
import numpy as np
import pandas as pd
import pytest
from mine_tem.model import Settings,validate_measurements
from mine_tem.io import read_files,canonicalize
from mine_tem.demo import make_demo
from mine_tem.processing import late_time_dbdt,late_time_rho,process,diffusion_pseudo_depth
from mine_tem.export import export_result


def test_ideal_halfspace_roundtrip():
    cfg=Settings();times=np.geomspace(1e-5,1e-3,30)
    for rho in (5,20,100,1000):
        amp=late_time_dbdt(rho,times,cfg.tx_moment)
        recovered=late_time_rho(amp,times,cfg.tx_moment)
        assert np.allclose(recovered,rho,rtol=1e-11)
    assert np.all(np.diff(diffusion_pseudo_depth(100,times))>0)


def test_demo_pipeline():
    result=process(make_demo(),Settings(),line_id='DEMO')
    assert result.quality['stations']==43
    assert result.quality['accepted_gates']>=.9*result.quality['total_gates']
    assert len(result.profile['x_m'])==43
    assert result.profile['rho_ohm_m'].shape[1]==43
    assert len(result.anomalies)>=1
    assert result.anomalies.iloc[0]['min_rho_ohm_m']<35


def test_measurement_units_cn_wide():
    d=pd.DataFrame({'测线':['L1','L1'],'测点':['S1','S2'],'横坐标':[0,10],
           't_0.1ms':[-1e-5,-2e-5],'t_0.2ms':[-1e-6,-2e-6]})
    out=canonicalize(d)
    assert len(out)==4
    assert sorted(out.time_s.unique())==[.0001,.0002]
    assert 'voltage_v' in out


def test_duplicate_gates_rejected():
    d=make_demo(n_stations=2)
    with pytest.raises(ValueError,match='重复'):
        validate_measurements(pd.concat([d,d.iloc[[0]]],ignore_index=True))


def test_multiline_needs_choice():
    d=make_demo(n_stations=2)
    d2=d.copy();d2['line_id']='B';d2['x_m']+=100
    with pytest.raises(ValueError,match='多条测线'):
        process(pd.concat([d,d2],ignore_index=True))


def test_csv_and_export(tmp_path):
    d=make_demo(n_stations=20)
    p=tmp_path/'raw.csv';d.to_csv(p,index=False)
    result=process(read_files([p]))
    folder=export_result(result,tmp_path/'deliverable',with_images=False)
    for name in ['manifest.json','gates_processed.csv','anomaly_candidates.csv',
                 'pseudo_section_arrays.npz','pseudo_section_xyz.csv','成果说明.md']:
        assert (folder/name).exists()
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['demo_only'] is True


def test_voltage_dbdt_conversion():
    cfg=Settings(rx_area_m2=5,rx_turns=10)
    d=make_demo(cfg,n_stations=4)
    raw=d.drop(columns=['dbdt_t_s']).copy()
    raw['voltage_v']=-d.dbdt_t_s*cfg.rx_area_m2*cfg.rx_turns
    r1=process(d,cfg);r2=process(raw,cfg)
    assert np.allclose(r1.readings['raw_dbdt_t_s'],r2.readings['raw_dbdt_t_s'])


def test_training_data_validation(tmp_path):
    from mine_tem.learn import load_pairs
    p=tmp_path/'bad.npz';np.savez(p,signals=np.ones((12,10)))
    with pytest.raises(ValueError,match='缺少'):
        load_pairs(p)
