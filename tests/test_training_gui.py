from pathlib import Path
import numpy as np
import pytest
from PySide6.QtWidgets import QFileDialog
from test_gui import app, window, wait_until, run_demo
from mine_tem.training_demo import make_training_demo


@pytest.fixture
def dataset(tmp_path):
    p = tmp_path / 'pairs.npz'
    make_training_demo(p, n_samples=20, n_times=12, n_depths=6)
    return p


@pytest.fixture
def model(tmp_path):
    prefix = tmp_path / 'my.model'
    Path(str(prefix) + '.pt').write_bytes(b'not deserialized during inspection')
    np.savez(str(prefix) + '_metadata.npz', times_s=np.geomspace(1e-5,1e-3,12),
             depths_m=np.linspace(0,100,6), xmean=np.zeros(12), xstd=np.ones(12),
             ymean=np.zeros(6), ystd=np.ones(6))
    return prefix


def test_inspect_dataset_without_torch_and_clear_on_path_change(window, app, dataset):
    w = window; w._has_torch = False; w._refresh_controls()
    w.dataset.setText(str(dataset)); w.training.check_data_btn.click()
    wait_until(app, lambda: not w.busy)
    assert w.training.dataset_info['samples'] == 20
    assert '20' in w.training.data_summary.text() and '12' in w.training.data_summary.text()
    assert not w.trainbtn.isEnabled()
    w.dataset.setText(str(dataset.parent / 'other.npz'))
    assert w.training.dataset_info is None
    assert '检查' in w.training.data_summary.text()


def test_bad_dataset_check_recovers_without_losing_measurement(window, app, tmp_path):
    w = window; run_demo(w, app); previous = w.result
    bad = tmp_path / 'bad.npz'; np.savez(bad, signals=np.ones((12,8)))
    w.dataset.setText(str(bad)); w.check_training_data()
    wait_until(app, lambda: not w.busy)
    assert w.errors and w.result is previous and w.exportbtn.isEnabled()
    assert w.training.dataset_info is None
    assert '失败' in w.training.data_summary.text()


def test_model_picker_inspects_metadata_without_torch(window, app, model, monkeypatch):
    w = window; w._has_torch = False
    monkeypatch.setattr(QFileDialog, 'getOpenFileName', lambda *a, **kw: (str(model)+'.pt', ''))
    w.training.modelbtn.click(); wait_until(app, lambda: not w.busy)
    assert w.training.model_info['prefix'] == str(model)
    assert '12' in w.training.model_summary.text()
    assert '权重' in w.training.model_summary.text()
    w.training.model_input.setText(str(model.parent / 'new.pt'))
    assert w.training.model_info is None


def test_missing_model_pair_recovers(window, app, tmp_path):
    w = window; run_demo(w, app); previous = w.result
    w.training.model_input.setText(str(tmp_path / 'missing.pt'))
    w.check_model_metadata(); wait_until(app, lambda: not w.busy)
    assert w.errors and w.result is previous
    assert w.training.model_info is None and '失败' in w.training.model_summary.text()


def test_output_picker_keeps_model_extension_contract(window, tmp_path, monkeypatch):
    w = window
    monkeypatch.setattr(QFileDialog, 'getSaveFileName', lambda *a, **kw: (str(tmp_path / 'chosen.pt'), ''))
    w.training.outputbtn.click()
    assert w.modelpath.text() == str(tmp_path / 'chosen')


def test_structured_training_progress_and_early_completion(window, app, dataset, model, monkeypatch):
    import mine_tem.learn as learning
    def fake_train(path, prefix, epochs, callback):
        assert str(path) == str(dataset) and str(prefix) == str(model) and epochs == 60
        callback(1,60,.8,1.)
        callback(2,60,.4,.6)
        return {'epochs':2,'best_validation_mse':.6,'model_prefix':str(model)}
    monkeypatch.setattr(learning, 'train', fake_train)
    w = window; run_demo(w, app); previous = w.result
    w._has_torch = True; w._refresh_controls()
    w.dataset.setText(str(dataset)); w.modelpath.setText(str(model)); w.trainbtn.click()
    assert w.busy and not w.training.check_data_btn.isEnabled()
    wait_until(app, lambda: not w.busy)
    panel = w.training
    assert panel.epoch_bar.maximum() == 60 and panel.epoch_bar.value() == 2
    assert w.runprogress.maximum() == 60 and w.runprogress.value() == 2
    assert list(panel.train_line.get_xdata()) == [1,2]
    assert list(panel.val_line.get_ydata()) == [1.,.6]
    assert '早停' in panel.training_status.text()
    assert panel.model_info['prefix'] == str(model)
    assert w.result is previous and w.exportbtn.isEnabled()


def test_actual_short_training_reports_progress_and_auto_selects_model(window, app, dataset, tmp_path):
    torch = pytest.importorskip('torch'); torch.set_num_threads(2)
    w = window; run_demo(w, app); previous = w.result
    prefix = tmp_path / 'actual'
    actual_data = tmp_path / 'matching-gates.npz'
    make_training_demo(actual_data, n_samples=20, n_times=33, n_depths=6)
    w.dataset.setText(str(actual_data)); w.modelpath.setText(str(prefix)); w.epochs.setValue(2)
    w.start_train(); wait_until(app, lambda: not w.busy, seconds=45)
    assert not w.errors and w.training.model_info['prefix'] == str(prefix)
    assert w.training.epoch_bar.value() == 2
    assert len(w.training.val_line.get_ydata()) == 2
    assert np.isfinite(w.training.val_line.get_ydata()).all()
    assert Path(str(prefix)+'_history.csv').is_file()
    assert w.result is previous
    from mine_tem.demo import make_demo
    w._set_raw(make_demo(n_stations=2), 'matching demo')
    w.predict_model(); wait_until(app, lambda: not w.busy, seconds=45)
    assert not w.errors
    assert w.result.quality['stations'] == 2
    assert 'supervised' in w.result.quality['engine']


def test_changing_selected_model_invalidates_prediction_but_not_quicklook(window, app, model, monkeypatch):
    import mine_tem.learn as learning
    from mine_tem.processing import process
    w = window; run_demo(w, app); quicklook = w.result
    w.training.model_input.setText(str(model)+'.pt')
    assert w.result is quicklook
    monkeypatch.setattr(learning, 'predict_line', lambda raw, prefix, cfg, line: process(raw, cfg, line))
    w._has_torch = True; w.predict_model(); wait_until(app, lambda: not w.busy)
    assert w.result is not None and w.exportbtn.isEnabled()
    w.training.model_input.setText(str(model.parent/'another.pt'))
    assert w.result is None and not w.exportbtn.isEnabled()


def test_pt_ending_prefix_keeps_legacy_training_output_names(window, app, dataset, tmp_path):
    torch = pytest.importorskip('torch'); torch.set_num_threads(2)
    w = window; prefix = tmp_path / 'literal.pt'
    w.dataset.setText(str(dataset)); w.modelpath.setText(str(prefix)); w.epochs.setValue(2)
    w.start_train(); wait_until(app, lambda: not w.busy, seconds=45)
    assert not w.errors
    assert Path(str(prefix)+'.pt').is_file()
    assert w.training.model_info['prefix'] == str(prefix)
    assert w.training.model_input.text() == str(prefix)+'.pt'


@pytest.mark.parametrize('corrupt', ['negative_depths', 'negative_scale'])
def test_prediction_revalidates_metadata_after_successful_inspection(window, app, model, monkeypatch, corrupt):
    from mine_tem.processing import process
    import mine_tem.learn as learning
    w = window; run_demo(w, app)
    w.training.model_input.setText(str(model)+'.pt'); w.check_model_metadata()
    wait_until(app, lambda: not w.busy)
    assert w.training.model_info is not None
    path = str(model)+'_metadata.npz'
    with np.load(path) as z: metadata = dict(z)
    if corrupt == 'negative_depths': metadata['depths_m'] = np.linspace(-100,-1,6)
    else: metadata['ystd'][0] = -1
    np.savez(path, **metadata)
    calls = []
    def predict(raw, prefix, cfg, line):
        calls.append(prefix)
        return process(raw,cfg,line)
    monkeypatch.setattr(learning, 'predict_line', predict)
    w._has_torch = True; w.predict_model(); wait_until(app, lambda: not w.busy)
    assert w.errors and not calls
    assert w.result is None and not w.exportbtn.isEnabled()


def test_retraining_active_prefix_invalidates_previous_prediction(window, app, dataset, model, monkeypatch):
    import mine_tem.learn as learning
    from mine_tem.processing import process
    w = window; run_demo(w, app)
    monkeypatch.setattr(learning, 'predict_line', lambda raw, prefix, cfg, line: process(raw,cfg,line))
    w._has_torch = True; w.training.model_input.setText(str(model)+'.pt')
    w.predict_model(); wait_until(app, lambda: not w.busy)
    assert w.result is not None
    def failed_training(*a, **kw): raise RuntimeError('training failed')
    monkeypatch.setattr(learning, 'train', failed_training)
    w.dataset.setText(str(dataset)); w.modelpath.setText(str(model)); w.start_train()
    assert w.result is None and not w.exportbtn.isEnabled()
    wait_until(app, lambda: not w.busy)
    assert w.result is None and not w.exportbtn.isEnabled()
