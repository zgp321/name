import sys
import numpy as np
import pytest
from mine_tem.learn import load_pairs
from mine_tem.training_demo import make_training_demo


@pytest.fixture
def pairs(tmp_path):
    p = tmp_path / 'pairs.npz'
    make_training_demo(p, n_samples=20, n_times=12, n_depths=6)
    return p


@pytest.mark.parametrize('key,value', [('times_s', 0), ('times_s', -1), ('times_s', np.inf),
                                      ('depths_m', -1), ('depths_m', np.inf)])
def test_illegal_axis_values_rejected_by_training_loader(pairs, key, value):
    with np.load(pairs) as z: data = dict(z)
    data[key][-1 if np.isinf(value) else 0] = value
    np.savez(pairs, **data)
    with pytest.raises(ValueError): load_pairs(pairs)


def test_dataset_inspection_uses_real_training_validation_without_torch(pairs, monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    from mine_tem.training_info import inspect_dataset
    info = inspect_dataset(pairs)
    assert (info['samples'], info['n_times'], info['n_depths']) == (20, 12, 6)
    assert info['time_range_s'] == [1.5e-5, 8e-4]
    assert info['depth_range_m'] == [5., 150.]
    assert 15 <= info['rho_range'][0] <= info['rho_range'][1] <= 230
    with np.load(pairs) as z: data = dict(z)
    data['signals'][0, 0] = np.nan; np.savez(pairs, **data)
    with pytest.raises(ValueError): inspect_dataset(pairs)


@pytest.fixture
def model(tmp_path):
    prefix = tmp_path / 'trained.model'
    # Metadata inspection must never deserialize this deliberately invalid checkpoint.
    (tmp_path / 'trained.model.pt').write_bytes(b'not a pickle or a torch checkpoint')
    np.savez(str(prefix) + '_metadata.npz', times_s=np.geomspace(1e-5,1e-3,12),
             depths_m=np.linspace(0,100,6), xmean=np.zeros(12), xstd=np.ones(12),
             ymean=np.zeros(6), ystd=np.ones(6))
    return prefix


def test_model_inspection_matches_prefix_without_reading_weights(model, monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', None)
    from mine_tem.training_info import inspect_model
    info = inspect_model(str(model) + '.pt')
    assert info['prefix'] == str(model)
    assert (info['n_times'], info['n_depths']) == (12,6)
    assert info['depth_range_m'] == [0.,100.]
    assert inspect_model(model) == info


@pytest.mark.parametrize('change', ['missing_weights','missing_meta','missing_key','bad_scale','bad_shape','bad_axis'])
def test_invalid_model_metadata_is_rejected(model, change):
    from mine_tem.training_info import inspect_model
    meta = str(model) + '_metadata.npz'
    with np.load(meta) as z: data = dict(z)
    if change == 'missing_weights': model.with_suffix('.model.pt').unlink()
    elif change == 'missing_meta': __import__('pathlib').Path(meta).unlink()
    else:
        if change == 'missing_key': del data['xstd']
        elif change == 'bad_scale': data['ystd'][0] = 0
        elif change == 'bad_shape': data['xmean'] = np.zeros(2)
        elif change == 'bad_axis': data['times_s'][-1] = np.inf
        np.savez(meta, **data)
    with pytest.raises((ValueError, FileNotFoundError)): inspect_model(model)
