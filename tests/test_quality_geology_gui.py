import os
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pandas as pd
import pytest

pytest.importorskip('PySide6')
from PySide6.QtWidgets import QApplication, QMessageBox

from mine_tem.gui import create_window


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def wait_until(app, predicate, seconds=20):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        app.processEvents(); time.sleep(.01)
    app.processEvents()
    assert predicate(), 'GUI task did not complete'


@pytest.fixture
def window(app, monkeypatch):
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **kw: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'information', lambda *a, **kw: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'critical', lambda *a, **kw: QMessageBox.Ok)
    w = create_window(); w.errors = []
    monkeypatch.setattr(w, '_show_error', lambda title, detail: w.errors.append((title, detail)))
    yield w
    if w.worker is not None: w.worker.wait(20000)
    app.processEvents(); w.close(); w.deleteLater(); app.processEvents()


def run_demo(w, app):
    w.load_demo(); w.analyze(); wait_until(app, lambda: not w.busy)
    assert w.result is not None


def test_quality_tab_populates_station_summary_and_curve(window, app):
    run_demo(window, app)
    w = window
    assert w.analysis_tabs.count() == 3
    assert w.qc_station.count() == 43
    assert w.qc_reason_table.rowCount() >= 1
    w.qc_station.setCurrentText('P001'); app.processEvents()
    assert len(w.qc_canvas.figure.axes) == 1
    assert w.qc_canvas.figure.axes[0].lines
    assert w.qc_gate_table.rowCount() == 33


def test_borehole_import_overlay_and_export(window, app, tmp_path):
    run_demo(window, app)
    w = window; anomaly = w.result.anomalies.iloc[0]
    source = tmp_path / 'boreholes.csv'
    pd.DataFrame({
        'line_id': [w.result.line_id], 'borehole_id': ['ZK-01'],
        'x_m': [(anomaly.x_start_m + anomaly.x_end_m) / 2], 'y_m': [0.0],
        'elevation_m': [0.0], 'depth_top_m': [max(0, anomaly.depth_top_m - 2)],
        'depth_bottom_m': [anomaly.depth_bottom_m + 2], 'lithology': ['砂岩'],
        'water_evidence': ['原始记录待复核'], 'confidence': [0.5],
    }).to_csv(source, index=False)
    assert w.import_boreholes([source])
    wait_until(app, lambda: not w.busy)
    assert not w.errors
    assert len(w.boreholes) == 1 and len(w.borehole_links) == 1
    assert w.geology_canvas.figure.axes and w.geology_canvas.figure.axes[0].lines
    target = tmp_path / 'delivery'
    assert w.export_to(target)
    wait_until(app, lambda: not w.busy)
    out = target / f'TEM_{w.result.line_id}'
    assert (out / 'boreholes.csv').is_file()
    assert (out / 'borehole_candidate_links.csv').is_file()
    assert (out / 'geology_joint_section.png').is_file()
    assert (out / 'qc_station_summary.csv').is_file()


def test_invalid_borehole_import_keeps_current_result(window, app, tmp_path):
    run_demo(window, app)
    w = window; previous = w.result
    bad = tmp_path / 'bad.csv'; pd.DataFrame({'borehole_id': ['ZK']}).to_csv(bad, index=False)
    assert w.import_boreholes([bad]); wait_until(app, lambda: not w.busy)
    assert w.result is previous
    assert w.boreholes is None
    assert w.errors
