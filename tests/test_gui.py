"""Exercise the real offscreen workbench, including worker lifecycle and exports."""
import json
import os
import time

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import pandas as pd
import pytest

pytest.importorskip('PySide6')
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QKeyEvent
from PySide6.QtWidgets import QApplication, QAbstractItemView, QMessageBox, QTableWidgetSelectionRange
from mine_tem.demo import make_demo
from mine_tem.gui import create_window


@pytest.fixture(scope='module')
def app():
    return QApplication.instance() or QApplication([])


def wait_until(app, predicate, seconds=20):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(.01)
    app.processEvents()
    assert predicate(), 'GUI task did not complete'


@pytest.fixture
def window(app, monkeypatch):
    # Native modal dialogs are the only component replaced in ordinary tests.
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a, **kw: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'information', lambda *a, **kw: QMessageBox.Ok)
    monkeypatch.setattr(QMessageBox, 'critical', lambda *a, **kw: QMessageBox.Ok)
    w = create_window()
    w.errors = []
    # The failure state, real worker and log remain real; avoid blocking on a modal.
    if hasattr(w, '_show_error'):
        monkeypatch.setattr(w, '_show_error', lambda title, detail: w.errors.append((title, detail)))
    yield w
    if w.worker is not None:
        w.worker.wait(20000)
    app.processEvents()
    w.close()
    w.deleteLater()
    app.processEvents()


def run_demo(w, app):
    w.load_demo()
    w.analyze()
    wait_until(app, lambda: not w.busy)
    assert w.result is not None


def test_navigation_readonly_preview_and_copy(window, app):
    w = window
    w.show()
    assert w.nav.count() == 4
    for i in range(4):
        w.nav.setCurrentRow(i)
        assert w.pages.currentIndex() == i
    w.load_demo()
    assert w.preview.editTriggers() == QAbstractItemView.NoEditTriggers
    assert w.preview.rowCount() == 100
    w.preview.setRangeSelected(QTableWidgetSelectionRange(0, 0, 0, 1), True)
    QApplication.sendEvent(w.preview, QKeyEvent(QKeyEvent.KeyPress, Qt.Key_C, Qt.ControlModifier))
    assert QApplication.clipboard().text() == 'DEMO\tP001'


@pytest.mark.parametrize('change', ['line', 'parameter', 'engine', 'reload', 'reset'])
def test_context_changes_clear_old_results(window, app, change):
    w = window
    a = make_demo()
    b = a.copy(); b['line_id'] = 'SECOND'
    w._set_raw(pd.concat([a, b], ignore_index=True), 'two lines')
    w.analyze()
    wait_until(app, lambda: not w.busy)
    assert w.exportbtn.isEnabled()
    if change == 'line': w.lines.setCurrentText('SECOND')
    elif change == 'parameter': w.fields['current_a'].setValue(2)
    elif change == 'engine': w.method.setCurrentIndex(1)
    elif change == 'reload': w.load_demo()
    else:
        w.fields['current_a'].setValue(2)
        w.reset_settings()
    assert w.result is None
    assert not w.exportbtn.isEnabled()
    assert w.anom_table.rowCount() == 0
    assert all(not c.figure.axes for c in w.canvases)


def test_real_processing_busy_guard_and_close(window, app):
    w = window
    w.load_demo()
    w.analyze()
    first = w.worker
    assert w.busy and not w.importbtn.isEnabled() and not w.demobtn.isEnabled()
    assert not w.trainbtn.isEnabled() and not w.lines.isEnabled()
    w.analyze(); w.load_demo(); w.start_train(); w.predict_model()
    assert w.worker is first
    event = QCloseEvent(); w.closeEvent(event)
    assert not event.isAccepted()
    wait_until(app, lambda: not w.busy)
    assert w.result.quality['accepted_gates'] == 1415
    assert w.result.quality['stations'] == 43
    assert w.exportbtn.isEnabled() and w.importbtn.isEnabled()
    assert w.runprogress.value() == 100


def test_failed_import_preserves_previous_result(window, app, tmp_path):
    w = window
    run_demo(w, app)
    previous = w.result
    raw = w.raw
    bad = tmp_path / 'bad.csv'; bad.write_text('wrong,columns\n1,2\n')
    assert w.import_files([bad])
    wait_until(app, lambda: not w.busy)
    assert w.result is previous and w.raw is raw
    assert w.exportbtn.isEnabled() and w.errors


def test_processing_failure_recovers_and_allows_retry(window, app):
    w = window
    w._set_raw(make_demo(n_stations=1), 'single station')
    w.analyze()
    wait_until(app, lambda: not w.busy)
    assert w.result is None and not w.exportbtn.isEnabled()
    assert w.errors and w.runbtn.isEnabled()
    assert 'ValueError' in w.log.toPlainText()
    run_demo(w, app)
    assert w.exportbtn.isEnabled()


def test_async_import_export_and_contained_line_path(window, app, tmp_path):
    w = window
    data = make_demo(); data['line_id'] = '../../outside'
    src = tmp_path / 'source.csv'; data.to_csv(src, index=False)
    assert w.import_files([src])
    wait_until(app, lambda: not w.busy)
    assert len(w.raw) == 1419
    w.analyze()
    wait_until(app, lambda: not w.busy)
    target = tmp_path / 'delivery'
    assert w.export_to(target)
    assert w.busy and not w.plots.isEnabled()
    assert not w.export_to(tmp_path / 'duplicate')
    wait_until(app, lambda: not w.busy)
    manifests = list(target.rglob('manifest.json'))
    assert len(manifests) == 1
    out = manifests[0].parent
    assert out.resolve().is_relative_to(target.resolve())
    assert json.loads(manifests[0].read_text())['line_id'] == '../../outside'
    assert len(pd.read_csv(out / 'gates_processed.csv')) == 1419
    for name in ['decay.png', 'pseudo_section.png', 'anomalies_2p5d.png']:
        assert (out / name).read_bytes().startswith(b'\x89PNG\r\n\x1a\n')
    assert w.exportbtn.isEnabled() and w.plots.isEnabled()
    assert w.export_to(target)
    wait_until(app, lambda: not w.busy)
    assert not w.errors
    assert json.loads(manifests[0].read_text())['line_id'] == '../../outside'


def test_reset_and_minimum_size_have_scrollable_forms(window, app):
    from PySide6.QtWidgets import QScrollArea
    w = window
    w.fields['current_a'].setValue(2)
    w.reset_settings()
    assert w.fields['current_a'].value() == 1
    w.resize(1024, 720); w.show(); app.processEvents()
    assert w.width() == 1024 and w.height() == 720
    assert len(w.findChildren(QScrollArea)) >= 2


def test_absent_optional_dependencies_dont_block_base(window, app):
    import importlib.util
    w = window
    if importlib.util.find_spec('torch') is None:
        assert not w.trainbtn.isEnabled() and not w.predictbtn.isEnabled()
        assert 'PyTorch' in w.dependency_note.text()
    if importlib.util.find_spec('simpeg') is None:
        w.load_demo(); w.method.setCurrentIndex(1); w.analyze()
        assert not w.busy and w.result is None
        w.method.setCurrentIndex(0)
    run_demo(w, app)


def test_small_window_keeps_chart_axis_labels_inside_figure(window, app):
    w = window
    w.resize(1024, 720); w.show()
    run_demo(w, app)
    w.plots.setCurrentIndex(1)
    app.processEvents()
    canvas = w.canvases[1]; canvas.draw()
    axis = canvas.figure.axes[0]
    renderer = canvas.get_renderer()
    bounds = axis.xaxis.label.get_window_extent(renderer)
    assert bounds.y0 >= 0, 'X axis label is clipped at minimum window size'
    assert axis.get_window_extent(renderer).height >= 150, 'Plot squashed by surrounding controls'


def test_empty_import_is_rejected_without_losing_data(window, app, tmp_path):
    w = window
    run_demo(w, app)
    previous = w.result
    src = tmp_path / 'empty.csv'; make_demo().iloc[:0].to_csv(src, index=False)
    w.import_files([src]); wait_until(app, lambda: not w.busy)
    assert w.result is previous and len(w.raw) == 1419
    assert w.errors


def test_training_completion_does_not_replace_measurement_result(window, app, monkeypatch):
    # Replace only the expensive optional ML computation, not worker/signals/UI.
    from mine_tem.gui_support import Task
    w = window; run_demo(w, app); previous = w.result
    original = Task._execute
    monkeypatch.setattr(Task, '_execute', lambda t: {'epochs': 2} if t.kind == 'train' else original(t))
    w._start_task('train', dataset='unused', output='unused', epochs=2)
    wait_until(app, lambda: not w.busy)
    assert w.result is previous and w.exportbtn.isEnabled()
    assert '训练完成' in w.task_status.text()


@pytest.mark.parametrize('first,second', [('A/B', 'A\\B'), ('a' * 80 + 'X', 'a' * 80 + 'Y')])
def test_export_folder_does_not_merge_distinct_line_ids(tmp_path, first, second):
    from mine_tem.gui_support import export_folder
    a = export_folder(tmp_path, first); b = export_folder(tmp_path, second)
    assert a != b
    assert a.parent == b.parent == tmp_path


def test_export_rejects_existing_directory_for_another_line(window, app, tmp_path):
    from mine_tem.gui_support import export_folder
    w = window; run_demo(w, app)
    dest = export_folder(tmp_path, 'DEMO'); dest.mkdir()
    manifest = dest / 'manifest.json'
    manifest.write_text('{"line_id": "a different line"}')
    w.export_to(tmp_path); wait_until(app, lambda: not w.busy)
    assert w.errors
    assert json.loads(manifest.read_text())['line_id'] == 'a different line'
    assert not (dest / 'gates_processed.csv').exists()


def test_export_rejects_symlink_outside_selected_directory(tmp_path):
    from mine_tem.gui_support import export_folder
    parent = tmp_path / 'selected'; parent.mkdir()
    outside = tmp_path / 'outside'; outside.mkdir()
    (parent / 'TEM_DEMO').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError): export_folder(parent, 'DEMO')
