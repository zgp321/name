from pathlib import Path
import numpy as np
import pytest
from mine_tem.model import Settings
from mine_tem.demo import make_demo
from mine_tem.processing import process, _repair_station, pick_anomalies
from mine_tem.export import export_result


def test_snr_threshold_rejects_signals_above_noise_but_below_minimum():
    g = make_demo(n_stations=2).query("station == 'P001'").copy()
    g['dbdt_t_s'] = -2e-13
    assert _repair_station(g, Settings(min_snr=1)).accepted.all()
    out = _repair_station(g, Settings(min_snr=3))
    assert not out.accepted.any()
    assert out.qc.str.contains('信噪比').all()


def test_rejected_time_interval_is_not_filled_by_profile_interpolation():
    raw = make_demo(n_stations=2)
    before = process(raw).readings.query("station == 'P001'").sort_values('time_s')
    lo, hi = before.iloc[9].pseudo_depth_m, before.iloc[20].pseudo_depth_m
    times = sorted(raw.time_s.unique())
    raw.loc[raw.time_s.isin(times[10:20]), 'dbdt_t_s'] = -2.5e-14
    result = process(raw)
    gap = (result.profile['depth_m'] > lo) & (result.profile['depth_m'] < hi)
    assert gap.any() and result.quality['rejected_gates'] == 20
    assert np.isnan(result.profile['rho_ohm_m'][gap, 0]).all()
    assert np.isfinite(result.profile['rho_ohm_m'][:, 0]).any()


def test_large_station_gap_splits_candidates_without_deleting_them():
    p = {'depth_m': np.array([0.,10.,20.]), 'x_m': np.array([0.,10.,20.,100.,110.,120.]),
         'rho_ohm_m': np.array([[100.,100.,10.,10.,100.,100.]] * 3)}
    found, mask = pick_anomalies(p, Settings(min_area_m2=1))
    assert len(found) == 2
    assert mask[:, 2:4].all()
    assert found.x_start_m.tolist() == [20.,100.]
    assert found.x_end_m.tolist() == [20.,100.]


@pytest.mark.parametrize('bad', [0., -1., np.inf, np.nan])
def test_invalid_rho_cannot_become_a_candidate(bad):
    p = {'depth_m': np.array([0.,10.,20.]), 'x_m': np.array([0.,10.,20.]),
         'rho_ohm_m': np.array([[100.,bad,100.]]*3)}
    found, mask = pick_anomalies(p, Settings(min_area_m2=1))
    assert found.empty and not mask.any()


def test_nonfinite_model_prediction_is_rejected(tmp_path):
    torch = pytest.importorskip('torch'); torch.set_num_threads(2)
    from mine_tem.learn import train, predict
    t = np.geomspace(1e-5,1e-3,8); signal = np.geomspace(1e-7,1e-10,8)
    data = tmp_path/'pairs.npz'; prefix = tmp_path/'model'
    np.savez(data, signals=np.tile(signal,(12,1)),
             resistivity=np.repeat(np.geomspace(1,1e4,12)[:,None],4,axis=1),
             times_s=t, depths_m=np.array([10.,20.,30.,40.]))
    train(data,prefix,epochs=2)
    with pytest.raises(ValueError, match='预测'):
        predict(prefix,signal*10,t)


def snapshot(folder):
    return {str(p.relative_to(folder)): p.read_bytes() for p in folder.rglob('*') if p.is_file()}


def test_failed_reexport_preserves_all_previous_artifacts(tmp_path, monkeypatch):
    old = process(make_demo(n_stations=4)); new = process(make_demo(n_stations=4),Settings(current_a=2))
    folder = tmp_path/'output'; export_result(old,folder)
    (folder/'user-note.txt').write_text('keep this')
    before = snapshot(folder)
    def fail(_): raise RuntimeError('render failed')
    monkeypatch.setattr('mine_tem.export.plot_decay', fail)
    with pytest.raises(RuntimeError): export_result(new,folder)
    assert snapshot(folder) == before


def test_failed_first_export_leaves_no_partial_output(tmp_path, monkeypatch):
    def fail(_): raise RuntimeError('render failed')
    monkeypatch.setattr('mine_tem.export.plot_decay', fail)
    folder = tmp_path/'output'
    with pytest.raises(RuntimeError): export_result(process(make_demo(n_stations=4)),folder)
    assert not folder.exists()


def test_successful_reexport_preserves_unrelated_files(tmp_path):
    folder = tmp_path/'output'; folder.mkdir(); (folder/'note.txt').write_text('keep')
    export_result(process(make_demo(n_stations=4)),folder,with_images=False)
    assert (folder/'note.txt').read_text() == 'keep'
    assert (folder/'manifest.json').is_file()


def test_directory_promotion_failure_restores_previous_output(tmp_path, monkeypatch):
    folder = tmp_path/'output'; result = process(make_demo(n_stations=4))
    export_result(result,folder,with_images=False); before = snapshot(folder)
    rename = Path.rename
    def fail_stage(source, target):
        if source.name == 'result': raise OSError('simulated directory rename failure')
        return rename(source,target)
    monkeypatch.setattr(Path,'rename',fail_stage)
    with pytest.raises(OSError): export_result(result,folder,with_images=False)
    assert snapshot(folder) == before
    assert not list(tmp_path.glob('.tem-export-*'))


def test_no_image_reexport_does_not_keep_figures_from_old_result(tmp_path):
    folder = tmp_path/'output'; result = process(make_demo(n_stations=4))
    export_result(result,folder)
    export_result(result,folder,with_images=False)
    assert not (folder/'pseudo_section.png').exists()
    assert (folder/'manifest.json').is_file()


def test_failed_rollback_retains_recoverable_previous_directory(tmp_path, monkeypatch):
    folder = tmp_path/'output'; result = process(make_demo(n_stations=4))
    export_result(result,folder,with_images=False); before = snapshot(folder)
    rename = Path.rename
    def fail_stage_and_restore(source, target):
        if source.name in ('result','previous'): raise OSError('simulated rename failure')
        return rename(source,target)
    monkeypatch.setattr(Path,'rename',fail_stage_and_restore)
    with pytest.raises(RuntimeError, match='原成果保留'):
        export_result(result,folder,with_images=False)
    backups = list(tmp_path.glob('.tem-export-*/previous'))
    assert len(backups) == 1 and snapshot(backups[0]) == before


def test_dotdot_output_never_stages_inside_itself(tmp_path, monkeypatch):
    import shutil
    folder = tmp_path/'output'; (folder/'nested').mkdir(parents=True)
    real_copy = shutil.copytree
    def check_location(source, destination, *a, **kw):
        assert not Path(destination).resolve().is_relative_to(Path(source).resolve()), 'recursive staging'
        return real_copy(source,destination,*a,**kw)
    monkeypatch.setattr(shutil,'copytree',check_location)
    export_result(process(make_demo(n_stations=4)), folder/'nested'/'..', with_images=False)
    assert (folder/'manifest.json').is_file()
