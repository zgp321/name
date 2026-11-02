"""Chinese desktop workflow; numerical operations remain in the existing core."""
from importlib.util import find_spec
from pathlib import Path
from numbers import Real, Integral
import traceback
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QPushButton, QComboBox, QLineEdit, QDoubleSpinBox, QSpinBox, QCheckBox,
    QTableWidgetItem, QFileDialog, QMessageBox, QPlainTextEdit, QProgressBar,
    QGroupBox, QHeaderView, QSplitter, QFrame, QListWidget, QStackedWidget,
    QScrollArea, QTabWidget, QSizePolicy,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from .model import Settings
from .demo import make_demo
from .visualize import plot_decay, plot_section, plot_3d_illustration, plot_joint_section
from .quality import build_quality_report
from .geology import read_boreholes, link_boreholes_to_candidates
from .gui_support import STYLE, ReadOnlyTable, Task, export_folder, Canvas


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('矿井 TEM 水害辅助判读工作台 · V1.0')
        self.resize(1360, 860); self.setMinimumSize(1024, 720)
        self.raw = self.result = self.worker = None
        self.boreholes = self.borehole_links = None
        self.qc_report = None
        self.paths = []; self.busy = False
        self._revision = 0; self._result_revision = None
        self._prediction_prefix = None
        self._has_torch = find_spec('torch') is not None
        self._has_simpeg = find_spec('simpeg') is not None
        self.setStyleSheet(STYLE)
        self._build_shell()
        self._data_page(); self._settings_page(); self._analysis_page(); self._learn_page()
        self.nav.currentRowChanged.connect(self._navigate)
        self.lines.currentTextChanged.connect(self._context_changed)
        for ctrl in self.fields.values(): ctrl.valueChanged.connect(self._context_changed)
        self.offset.toggled.connect(self._context_changed)
        self.method.currentIndexChanged.connect(self._context_changed)
        self.nav.setCurrentRow(0); self._refresh_controls()
        self.statusBar().showMessage('准备就绪 · 请导入测线文件，或使用合成演示')

    @staticmethod
    def _label(text, name=None):
        label = QLabel(text); label.setWordWrap(True)
        if name: label.setObjectName(name)
        return label

    @staticmethod
    def _button(text, callback, secondary=False):
        button = QPushButton(text); button.setProperty('secondary', secondary)
        button.clicked.connect(callback); button.setCursor(Qt.PointingHandCursor)
        return button

    def _build_shell(self):
        root = QWidget(); self.setCentralWidget(root)
        shell = QHBoxLayout(root); shell.setContentsMargins(0, 0, 0, 0); shell.setSpacing(0)
        side = QFrame(); side.setObjectName('sidebar'); side.setFixedWidth(195)
        rail = QVBoxLayout(side); rail.setContentsMargins(18, 28, 18, 22)
        rail.addWidget(self._label('TEM / 工作台', 'brand'))
        rail.addWidget(self._label('WATER-RISK WORKBENCH')); rail.addSpacing(28)
        self.nav = QListWidget()
        self.nav.addItems(['01　数据与测线', '02　处理参数', '03　结果判读', '04　模型训练'])
        rail.addWidget(self.nav, 1)
        rail.addWidget(self._label('工程研究原型 · V1.0\n结果须结合地质与钻探复核'))
        shell.addWidget(side)
        work = QWidget(); work.setObjectName('workspace')
        body = QVBoxLayout(work); body.setContentsMargins(22, 20, 22, 10); body.setSpacing(12)
        header = QHBoxLayout(); titles = QVBoxLayout()
        self.title = self._label('', 'pageTitle'); self.subtitle = self._label('', 'subtitle')
        titles.addWidget(self.title); titles.addWidget(self.subtitle); header.addLayout(titles, 1)
        header.addWidget(QLabel('当前测线'))
        self.lines = QComboBox(); self.lines.setMinimumWidth(125); self.lines.setMaximumWidth(220)
        header.addWidget(self.lines); body.addLayout(header)
        body.addWidget(self._label('中心回线晚期半空间近似 · 伪剖面非严格二维 / 三维反演 · 低阻候选区不等同于含水体', 'boundary'))
        self.pages = QStackedWidget(); self.tabs = self.pages; body.addWidget(self.pages, 1)
        self.task_status = self._label('准备就绪', 'subtitle')
        self.runprogress = QProgressBar(); self.runprogress.setTextVisible(False); self.runprogress.setValue(0)
        body.addWidget(self.task_status); body.addWidget(self.runprogress); shell.addWidget(work, 1)

    def _navigate(self, index):
        self.pages.setCurrentIndex(index)
        copy = [('准备一条待处理测线', '导入文件，检查来源、字段与时间单位。'),
                ('核对观测系统与参数', '确认设备信息及处理假设，再运行当前测线。'),
                ('查看测线处理成果', '先检查数据质量，再解释低阻候选异常。'),
                ('训练与使用监督模型', '仅使用来源可追溯的成对训练数据。')]
        self.title.setText(copy[index][0]); self.subtitle.setText(copy[index][1])

    def _scroll_page(self):
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); scroll.setWidget(content)
        layout = QVBoxLayout(content); layout.setContentsMargins(0, 0, 4, 0); layout.setSpacing(12)
        self.pages.addWidget(scroll)
        return layout

    def _data_page(self):
        page = QWidget(); v = QVBoxLayout(page); v.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.importbtn = self._button('导入测线文件', self.pick_files)
        self.demobtn = self._button('使用合成演示', self.load_demo, True)
        row.addWidget(self.importbtn); row.addWidget(self.demobtn); row.addStretch(); v.addLayout(row)
        v.addWidget(self._label('支持 CSV / TXT / TSV / XLSX / XLSM，多文件批量导入；支持中文列名、长表和时间门宽表。'))
        self.source = self._label('尚未导入文件。首次使用可点击“使用合成演示”体验完整流程。')
        v.addWidget(self.source)
        self.preview = ReadOnlyTable()
        self.preview.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        v.addWidget(self.preview, 1)
        v.addWidget(self._label('只读预览前 100 行，处理使用全部数据。选中单元格后按 Ctrl+C 复制。', 'subtitle'))
        self.nextbtn = self._button('下一步：核对处理参数 →', lambda: self.nav.setCurrentRow(1))
        v.addWidget(self.nextbtn, 0, Qt.AlignRight); self.pages.addWidget(page)

    def _settings_page(self):
        v = self._scroll_page(); self.fields = {}; defaults = Settings()
        sections = [
            ('发射与接收系统 · 请使用设备校准值', [
                ('tx_area_m2', '发射回线面积 (m²)', .1, 1e6), ('tx_turns', '发射匝数', 1, 100000),
                ('current_a', '发射电流 (A)', .001, 100000), ('rx_area_m2', '接收线圈面积 (m²)', .00001, 1e6),
                ('rx_turns', '接收匝数', 1, 100000)]),
            ('质量控制与候选阈值', [
                ('noise_floor_t_per_s', '|dB/dt| 噪声底 (T/s)', 1e-18, 1e-2),
                ('low_rho_ohm_m', '低阻阈值 (Ω·m)', .01, 100000), ('contrast_ratio', '相对背景比阈值', .05, 1),
                ('min_area_m2', '最小伪剖面面积 (m²)', 1, 1e7)]),
            ('显示与解释假设', [('depth_factor', '扩散视深度比例系数', .1, 5),
                ('display_extrusion_width_m', '2.5D 示意横向宽度 (m)', .5, 1000)])]
        for title, entries in sections:
            group = QGroupBox(title); form = QFormLayout(group)
            for key, caption, lo, hi in entries:
                ctrl = QSpinBox() if key in ('tx_turns', 'rx_turns') else QDoubleSpinBox()
                if isinstance(ctrl, QDoubleSpinBox):
                    ctrl.setDecimals(18 if key == 'noise_floor_t_per_s' else 5)
                    if key == 'noise_floor_t_per_s': ctrl.setSingleStep(1e-13)
                ctrl.setRange(lo, hi); ctrl.setValue(getattr(defaults, key))
                self.fields[key] = ctrl; form.addRow(caption, ctrl)
            v.addWidget(group)
        self.offset = QCheckBox('晚期尾门估计固定偏置（可选，默认关闭，需人工复核）'); v.addWidget(self.offset)
        group = QGroupBox('处理方法'); form = QFormLayout(group)
        self.method = QComboBox(); self.method.addItems(['晚期视电阻率快速成像', 'SimPEG 1D 地表分层反演（可选）'])
        form.addRow('处理引擎', self.method)
        note = 'SimPEG 仅适用于地表中心回线 1D 分层半空间，不是井下全空间或严格二维反演。'
        if not self._has_simpeg: note += '\n此环境未安装 SimPEG；请使用基础快速模式。'
        form.addRow(self._label(note, 'subtitle')); v.addWidget(group)
        row = QHBoxLayout()
        self.resetbtn = self._button('恢复默认参数', self.reset_settings, True)
        self.runbtn = self._button('运行当前测线 →', self.analyze)
        row.addWidget(self.resetbtn); row.addStretch(); row.addWidget(self.runbtn); v.addLayout(row); v.addStretch()

    def _analysis_page(self):
        v = self._scroll_page()
        self.analysis_tabs = QTabWidget()
        result_page = QWidget(); result_layout = QVBoxLayout(result_page)
        result_layout.setContentsMargins(0, 0, 0, 0); result_layout.setSpacing(12)
        self._build_result_tab(result_layout)
        self._build_quality_tab()
        self._build_geology_tab()
        self.analysis_tabs.addTab(result_page, '结果判读')
        self.analysis_tabs.addTab(self.quality_page, '质量诊断')
        self.analysis_tabs.addTab(self.geology_page, '地质 / 钻孔联合')
        v.addWidget(self.analysis_tabs, 1)

    def _build_result_tab(self, v):
        cards = QHBoxLayout(); self.metrics = []
        for caption in ('测点数量', '有效 / 总时间门', '拒绝时间门', '低阻候选区'):
            card = QFrame(); card.setObjectName('card'); sub = QVBoxLayout(card)
            sub.addWidget(self._label(caption, 'metricTitle'))
            value = self._label('—', 'metricValue'); self.metrics.append(value); sub.addWidget(value); cards.addWidget(card, 1)
        v.addLayout(cards)
        row = QHBoxLayout()
        self.rerunbtn = self._button('重新处理', self.analyze, True)
        self.adjustbtn = self._button('调整参数', lambda: self.nav.setCurrentRow(1), True)
        self.exportbtn = self._button('导出完整成果 ↗', self.save_results)
        row.addWidget(self.rerunbtn); row.addWidget(self.adjustbtn); row.addStretch(); row.addWidget(self.exportbtn); v.addLayout(row)
        splitter = QSplitter(Qt.Vertical); splitter.setChildrenCollapsible(False)
        splitter.setMinimumHeight(540)
        splitter.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
        self.plots = QTabWidget(); self.plots.setMinimumHeight(340); self.canvases = []
        for title in ('衰减曲线', '视电阻率伪剖面', '2.5D 示意（非三维反演）'):
            chart_page = QWidget(); sub = QVBoxLayout(chart_page); sub.setContentsMargins(0, 0, 0, 0)
            canvas = Canvas(Figure(figsize=(9, 4), dpi=100)); self.canvases.append(canvas)
            canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Ignored)
            toolbar = NavigationToolbar2QT(canvas, chart_page)
            translations = {'Home': '复位视图', 'Back': '上一视图', 'Forward': '下一视图', 'Pan': '平移',
                            'Zoom': '框选缩放', 'Subplots': '调整图形布局', 'Customize': '编辑图形', 'Save': '保存当前图形'}
            for action in toolbar.actions():
                if action.text() in translations:
                    action.setToolTip(translations[action.text()]); action.setText(translations[action.text()])
            sub.addWidget(toolbar); sub.addWidget(canvas, 1); self.plots.addTab(chart_page, title)
        splitter.addWidget(self.plots)
        group = QGroupBox('候选异常与质量记录'); sub = QVBoxLayout(group)
        self.summary = self._label('尚未处理。请先导入数据，再核对参数并运行当前测线。'); sub.addWidget(self.summary)
        self.anom_table = ReadOnlyTable(); self.anom_table.setMinimumHeight(105)
        self.anom_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        sub.addWidget(self.anom_table, 1)
        self.quality_note = self._label('', 'subtitle'); sub.addWidget(self.quality_note)
        splitter.addWidget(group); splitter.setSizes([380, 190]); v.addWidget(splitter, 1)

    @staticmethod
    def _fill_table(table, frame, names=None):
        frame = frame.copy()
        columns = list(frame.columns)
        table.setColumnCount(len(columns)); table.setRowCount(len(frame))
        table.setHorizontalHeaderLabels([(names or {}).get(c, c) for c in columns])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        for i, row in enumerate(frame.itertuples(index=False, name=None)):
            for j, value in enumerate(row):
                text = f'{value:.4g}' if isinstance(value, Real) and not isinstance(value, Integral) else str(value)
                item = QTableWidgetItem(text); item.setToolTip(str(value)); table.setItem(i, j, item)

    def _build_quality_tab(self):
        self.quality_page = QWidget(); v = QVBoxLayout(self.quality_page)
        v.setContentsMargins(0, 0, 0, 0); v.setSpacing(10)
        self.qc_summary = self._label('处理完成后显示测点级质量摘要；此页不改变任何 QC 决策。')
        v.addWidget(self.qc_summary)
        reason_group = QGroupBox('拒绝原因统计'); reason_layout = QVBoxLayout(reason_group)
        self.qc_reason_table = ReadOnlyTable(); self.qc_reason_table.setMinimumHeight(95)
        reason_layout.addWidget(self.qc_reason_table); v.addWidget(reason_group)
        station_group = QGroupBox('测点质量汇总'); station_layout = QVBoxLayout(station_group)
        row = QHBoxLayout(); row.addWidget(self._label('选中测点查看衰减曲线：'))
        self.qc_station = QComboBox(); self.qc_station.setMinimumWidth(160)
        self.qc_station.currentTextChanged.connect(self._show_qc_station)
        row.addWidget(self.qc_station); row.addStretch(); station_layout.addLayout(row)
        self.qc_station_table = ReadOnlyTable(); self.qc_station_table.setMinimumHeight(145)
        station_layout.addWidget(self.qc_station_table)
        v.addWidget(station_group)
        self.qc_canvas = Canvas(Figure(figsize=(8, 2.8), dpi=100)); self.qc_canvas.setMinimumHeight(245)
        v.addWidget(self.qc_canvas, 1)
        gate_group = QGroupBox('选中测点时间门明细'); gate_layout = QVBoxLayout(gate_group)
        self.qc_gate_table = ReadOnlyTable(); self.qc_gate_table.setMinimumHeight(125)
        gate_layout.addWidget(self.qc_gate_table); v.addWidget(gate_group)
        v.addWidget(self._label('原始与质控后曲线用于核查拒绝门和尖峰修复；伪深度插值不会在此页重新生成。', 'subtitle'))

    def _build_geology_tab(self):
        self.geology_page = QWidget(); v = QVBoxLayout(self.geology_page)
        v.setContentsMargins(0, 0, 0, 0); v.setSpacing(10)
        row = QHBoxLayout()
        self.boreholebtn = self._button('导入地质 / 钻孔 CSV', self.pick_boreholes, True)
        self.clear_boreholebtn = self._button('清除叠加', self.clear_boreholes, True)
        row.addWidget(self.boreholebtn); row.addWidget(self.clear_boreholebtn); row.addStretch(); v.addLayout(row)
        self.borehole_source = self._label('未导入钻孔区间。必需字段：line_id、borehole_id、x_m、depth_top_m、depth_bottom_m、lithology。')
        v.addWidget(self.borehole_source)
        self.borehole_table = ReadOnlyTable(); self.borehole_table.setMinimumHeight(120)
        v.addWidget(self.borehole_table)
        self.geology_canvas = Canvas(Figure(figsize=(8, 3), dpi=100)); self.geology_canvas.setMinimumHeight(260)
        v.addWidget(self.geology_canvas, 1)
        link_group = QGroupBox('候选异常空间关联（伪深度几何对照）'); link_layout = QVBoxLayout(link_group)
        self.borehole_link_table = ReadOnlyTable(); self.borehole_link_table.setMinimumHeight(120)
        link_layout.addWidget(self.borehole_link_table); v.addWidget(link_group)
        v.addWidget(self._label('钻孔深度与 TEM 视深度仅作几何对照；水文证据原样保留，需专业人员结合原始记录复核。', 'subtitle'))

    def _learn_page(self):
        from .gui_training import TrainingPanel
        v = self._scroll_page(); self.training = TrainingPanel(); v.addWidget(self.training)
        # Preserve the existing window-level control references for callers.
        for name in ('dataset', 'datasetbtn', 'modelpath', 'epochs', 'trainbtn', 'predictbtn', 'log', 'dependency_note'):
            setattr(self, name, getattr(self.training, name))
        self.training.browse_data.connect(self.pick_dataset)
        self.training.inspect_data.connect(self.check_training_data)
        self.training.browse_output.connect(self.choose_model_output)
        self.training.browse_model.connect(self.choose_model)
        self.training.inspect_model.connect(self.check_model_metadata)
        self.training.train_requested.connect(self.start_train)
        self.training.predict_requested.connect(self.predict_model)
        self.training.model_input.textChanged.connect(self._prediction_input_changed)
        self.modelpath.textChanged.connect(self._prediction_input_changed)

    def _prediction_input_changed(self):
        selected = self.training.model_input.text().strip() or self.modelpath.text().strip()
        if selected.endswith('.pt'): selected = selected[:-3]
        if self._prediction_prefix is not None and selected != self._prediction_prefix:
            self._context_changed()

    def _settings(self):
        return Settings(**{key: ctrl.value() for key, ctrl in self.fields.items()}, late_offset_correction=self.offset.isChecked())

    def _refresh_controls(self):
        free = not self.busy; has_data = self.raw is not None
        for ctrl in (self.importbtn, self.demobtn, self.resetbtn, self.offset, self.method,
                     self.dataset, self.datasetbtn, self.modelpath, self.epochs, *self.fields.values()): ctrl.setEnabled(free)
        self.lines.setEnabled(free and has_data); self.nextbtn.setEnabled(has_data)
        self.runbtn.setEnabled(free and has_data); self.rerunbtn.setEnabled(free and has_data)
        self.boreholebtn.setEnabled(free); self.clear_boreholebtn.setEnabled(free and self.boreholes is not None)
        self.trainbtn.setEnabled(free and self._has_torch); self.predictbtn.setEnabled(free and self._has_torch and has_data)
        self.exportbtn.setEnabled(free and self.result is not None and self._result_revision == self._revision)
        self.training.set_busy(self.busy, self._has_torch, has_data)

    def _context_changed(self, *_):
        self._revision += 1; self.result = None; self._result_revision = None
        self._prediction_prefix = None
        self.qc_report = None; self.borehole_links = None
        self.anom_table.setRowCount(0); self.quality_note.clear()
        for value in self.metrics: value.setText('—')
        for canvas in self.canvases: canvas.figure.clear(); canvas.draw_idle()
        self.qc_reason_table.setRowCount(0); self.qc_station_table.setRowCount(0); self.qc_station.clear()
        self.qc_summary.setText('处理完成后显示测点级质量摘要；此页不改变任何 QC 决策。')
        self.qc_canvas.figure.clear(); self.qc_canvas.draw_idle(); self.qc_gate_table.setRowCount(0)
        self.borehole_link_table.setRowCount(0); self.geology_canvas.figure.clear(); self.geology_canvas.draw_idle()
        text = '数据或参数已变化，请重新运行当前测线。' if self.raw is not None else '请先导入测线数据。'
        self.summary.setText(text)
        if not self.busy:
            self.task_status.setText('待处理 · ' + text); self.runprogress.setRange(0, 100); self.runprogress.setValue(0)
        self._refresh_controls()

    def reset_settings(self):
        if self.busy: return
        defaults = Settings()
        for key, ctrl in self.fields.items(): ctrl.setValue(getattr(defaults, key))
        self.offset.setChecked(False); self.method.setCurrentIndex(0); self._context_changed()

    def pick_files(self):
        if self.busy: return
        paths, _ = QFileDialog.getOpenFileNames(self, '选择 TEM 数据文件', '', 'TEM (*.csv *.txt *.tsv *.xlsx *.xlsm)')
        if paths: self.import_files(paths)

    def pick_boreholes(self):
        if self.busy: return
        paths, _ = QFileDialog.getOpenFileNames(
            self, '选择地质 / 钻孔文件', '', '地质表 (*.csv *.txt *.tsv *.xlsx *.xlsm)')
        if paths: self.import_boreholes(paths)

    def import_boreholes(self, paths):
        if self.busy or not paths: return False
        return self._start_task('boreholes', paths=[str(p) for p in paths])

    def clear_boreholes(self):
        if self.busy: return
        self.boreholes = self.borehole_links = None
        if self.result is not None:
            self.result.boreholes = None; self.result.borehole_links = None
        self.borehole_source.setText('未导入钻孔区间。必需字段：line_id、borehole_id、x_m、depth_top_m、depth_bottom_m、lithology。')
        self.borehole_table.setRowCount(0); self.borehole_link_table.setRowCount(0)
        self.geology_canvas.figure.clear(); self.geology_canvas.draw_idle(); self._refresh_controls()

    def import_files(self, paths):
        if self.busy or not paths: return False
        return self._start_task('import', paths=[str(p) for p in paths])

    def load_demo(self):
        if self.busy: return
        try:
            self._set_raw(make_demo(self._settings()), '合成演示数据（非实测 / 半空间晚期近似）'); self.paths = []
        except Exception: self._show_error('演示加载失败', traceback.format_exc())

    def _set_raw(self, raw, source):
        self.raw = raw; self.lines.blockSignals(True)
        self.lines.clear(); self.lines.addItems(sorted(raw['line_id'].unique().tolist())); self.lines.blockSignals(False)
        self.source.setText(f'来源：{source}\n{raw.line_id.nunique()} 条测线 · '
                            f'{len(raw[["line_id", "station"]].drop_duplicates())} 个测点 · {len(raw):,} 个时间门')
        view = raw.head(100); self.preview.setRowCount(len(view)); self.preview.setColumnCount(len(view.columns))
        self.preview.setHorizontalHeaderLabels([str(c) for c in view.columns])
        for i, row in enumerate(view.itertuples(index=False, name=None)):
            for j, value in enumerate(row): self.preview.setItem(i, j, QTableWidgetItem(str(value)))
        self._context_changed(); self.nav.setCurrentRow(0); self.statusBar().showMessage('已载入数据 · 下一步：核对处理参数')

    def analyze(self):
        if self.busy: return
        if self.raw is None:
            QMessageBox.warning(self, '无数据', '请先导入数据或使用合成演示。'); return
        if self.method.currentIndex() == 1 and not self._has_simpeg:
            QMessageBox.warning(self, '缺少可选依赖', '未安装 SimPEG。请切回晚期视电阻率快速成像。'); return
        try: cfg = self._settings(); cfg.validate()
        except Exception: self._show_error('参数错误', traceback.format_exc()); return
        self._context_changed()
        self._start_task('process', raw=self.raw.copy(), cfg=cfg, line=self.lines.currentText(), simpeg=self.method.currentIndex() == 1)
        self.nav.setCurrentRow(2)

    def _start_task(self, kind, **kwargs):
        if self.busy: return False
        if (kind == 'train' and self._prediction_prefix is not None
                and Path(kwargs['output']).resolve() == Path(self._prediction_prefix).resolve()):
            # The path is unchanged but training may replace its weights/metadata.
            # Invalidate before launch so even a partial failed save is safe.
            self._context_changed()
        if self.worker is not None: self.worker.deleteLater()
        self.worker = Task(kind, self, **kwargs); self.worker.context_revision = self._revision
        self.worker.data_revision = self.training.data_revision; self.worker.model_revision = self.training.model_revision
        self.worker.epoch_progress.connect(self._training_progress)
        self.worker.progress.connect(self._progress); self.worker.finished.connect(self._finish_task); self.busy = True
        labels = {'import': '正在导入数据', 'process': '正在处理当前测线', 'predict': '正在模型推理', 'train': '正在监督训练',
                  'export': '正在导出成果', 'dataset_info': '正在检查训练数据', 'model_info': '正在检查模型元数据',
                  'boreholes': '正在检查地质 / 钻孔文件'}
        self.task_status.setText(labels[kind] + ' · 请等待任务完成')
        self.statusBar().showMessage(self.task_status.text()); self.log.appendPlainText(labels[kind]); self.runprogress.setRange(0, 0)
        if kind == 'export':
            for canvas in [*self.canvases, self.training.canvas, self.qc_canvas, self.geology_canvas]:
                canvas.draw(); canvas.suspended = True
            self.plots.setEnabled(False)
        if kind == 'train':
            self.training.begin_training(kwargs['epochs'])
            self.runprogress.setRange(0, kwargs['epochs']); self.runprogress.setValue(0)
        self._refresh_controls(); self.worker.start(); return True

    def _progress(self, text):
        self.task_status.setText(text); self.statusBar().showMessage(text); self.log.appendPlainText(text)

    def _training_progress(self, epoch, total, train_mse, val_mse):
        self.training.update_progress(epoch, total, train_mse, val_mse)
        self.runprogress.setRange(0, total); self.runprogress.setValue(epoch)

    def _finish_task(self):
        task = self.worker; success = False
        try:
            if task.error: raise RuntimeError(task.error)
            if task.kind == 'import':
                self._set_raw(task.payload, '；'.join(Path(p).name for p in task.args['paths'])); self.paths = task.args['paths']
            elif task.kind in ('process', 'predict'):
                if task.context_revision != self._revision: raise RuntimeError('处理期间数据或参数发生变化，请重新处理。')
                self._present_result(task.payload); self._result_revision = task.context_revision
            elif task.kind == 'boreholes':
                self.boreholes = task.payload
                self.borehole_source.setText(
                    f'已导入 {self.boreholes["borehole_id"].nunique()} 个钻孔、{len(self.boreholes)} 个岩性区间。')
                self._fill_table(self.borehole_table, self.boreholes)
                self._refresh_geology()
            elif task.kind == 'dataset_info':
                if task.data_revision != self.training.data_revision: raise ValueError('数据路径已变化，请重新检查。')
                self.training.show_dataset(task.payload)
            elif task.kind == 'model_info':
                if task.model_revision != self.training.model_revision: raise ValueError('模型路径已变化，请重新检查。')
                self.training.show_model(task.payload)
            elif task.kind == 'train':
                self.training.complete_training(task.payload); self.log.appendPlainText(f'训练完成：{task.payload}')
            else: self.log.appendPlainText(f'成果已写入：{task.payload}')
            success = True
        except Exception:
            if task.kind in ('process', 'predict'): self._context_changed()
            if task.kind == 'dataset_info':
                self.training.dataset_info = None; self.training.data_summary.setText('检查失败，请核对训练数据。')
            elif task.kind == 'model_info':
                self.training.model_info = None; self.training.model_summary.setText('检查失败，请核对模型配套文件。')
            elif task.kind == 'train': self.training.training_status.setText('训练未完成 · 详情见运行日志')
            detail = traceback.format_exc(); self.log.appendPlainText(detail)
            self.task_status.setText('任务未完成 · 详情见模型训练页的运行日志'); self._show_error('操作未完成', detail)
        finally:
            self.busy = False
            for canvas in [*self.canvases, self.training.canvas, self.qc_canvas, self.geology_canvas]: canvas.suspended = False
            self.plots.setEnabled(True)
            for canvas in self.canvases: canvas.draw_idle()
            self.runprogress.setRange(0, 100); self.runprogress.setValue(100 if success else 0)
            if task.kind == 'train':
                self.runprogress.setRange(0, self.training.epoch_bar.maximum())
                self.runprogress.setValue(self.training.epoch_bar.value())
            if success:
                message = {'import': '导入完成 · 请核对处理参数', 'process': '处理完成 · 请复核质量与候选区',
                           'predict': '推理完成 · 请结合独立数据验证', 'train': '训练完成 · 请用独立数据检验',
                           'export': f'导出完成 · {task.payload}', 'dataset_info': '训练数据检查通过',
                           'model_info': '模型元数据检查通过（未加载权重）',
                           'boreholes': '地质 / 钻孔导入完成 · 请检查空间关联'}[task.kind]
                if task.kind == 'train': message = self.training.training_status.text()
                self.task_status.setText(message); self.log.appendPlainText(message)
            self.statusBar().showMessage(self.task_status.text()); self._refresh_controls()

    def _present_result(self, result):
        for canvas, plot in zip(self.canvases, (plot_decay, plot_section, plot_3d_illustration)):
            plot(result, fig=canvas.figure); canvas.draw_idle()
        self.result = result; q = result.quality
        for widget, value in zip(self.metrics, [str(q['stations']), f'{q["accepted_gates"]:,} / {q["total_gates"]:,}', str(q['rejected_gates']), str(len(result.anomalies))]): widget.setText(value)
        self.summary.setText(f'测线 {result.line_id} · 低阻候选区需结合地质 / 钻探验证。' if len(result.anomalies) else
                             f'测线 {result.line_id} · 未检出满足当前阈值的低阻候选区，不能据此排除水害。')
        columns = list(result.anomalies.columns)
        names = dict(zip(['id', 'x_start_m', 'x_end_m', 'depth_top_m', 'depth_bottom_m', 'area_pseudo_m2',
                          'min_rho_ohm_m', 'median_rho_ohm_m', 'interpretation', 'evidence_level'],
                         ['编号', '起始里程 (m)', '终止里程 (m)', '顶界视深度 (m)', '底界视深度 (m)', '伪剖面面积 (m²)',
                          '最小视电阻率 (Ω·m)', '中位视电阻率 (Ω·m)', '解释', '证据等级']))
        self.anom_table.setColumnCount(len(columns)); self.anom_table.setRowCount(len(result.anomalies))
        self.anom_table.setHorizontalHeaderLabels([names.get(c, c) for c in columns])
        for i, row in enumerate(result.anomalies.itertuples(index=False, name=None)):
            for j, value in enumerate(row):
                text = f'{value:.2f}' if isinstance(value, Real) and not isinstance(value, Integral) else str(value)
                item = QTableWidgetItem(text); item.setToolTip(str(value))
                self.anom_table.setItem(i, j, item)
        self.quality_note.setText('质量记录：' + '；'.join(f'{k} {v} 门' for k, v in q['qc_breakdown'].items()))
        self.qc_report = result.quality_report or build_quality_report(result.readings)
        totals = self.qc_report['totals']
        self.qc_summary.setText(
            f'时间门 {totals["total_gates"]:,} · 有效 {totals["accepted_gates"]:,} · '
            f'拒绝 {totals["rejected_gates"]:,} · 有效率 {totals["acceptance_rate"]:.1%}')
        self._fill_table(self.qc_reason_table, self.qc_report['reason_summary'],
                         {'qc': 'QC 状态', 'gate_count': '门数', 'accepted_gates': '有效门',
                          'rejected_gates': '拒绝门', 'share': '占比'})
        self._fill_table(self.qc_station_table, self.qc_report['station_summary'],
                         {'station': '测点', 'x_m': '里程 (m)', 'y_m': 'Y (m)', 'elevation_m': '高程 (m)',
                          'total_gates': '总门数', 'accepted_gates': '有效门', 'rejected_gates': '拒绝门',
                          'acceptance_rate': '有效率', 'snr_median': 'SNR 中位数', 'snr_min': 'SNR 最小值',
                          'first_valid_time_s': '首个有效时间 (s)', 'last_valid_time_s': '末个有效时间 (s)',
                          'qc_breakdown': 'QC 分布'})
        self.qc_station.blockSignals(True); self.qc_station.clear()
        self.qc_station.addItems(self.qc_report['station_summary']['station'].astype(str).tolist())
        self.qc_station.blockSignals(False)
        if self.qc_station.count():
            self.qc_station.setCurrentIndex(0)
            self._show_qc_station(self.qc_station.currentText())
        self._refresh_geology()

    def _show_qc_station(self, station):
        self.qc_canvas.figure.clear(); ax = self.qc_canvas.figure.add_subplot(111)
        self.qc_gate_table.setRowCount(0)
        if self.result is None or not station:
            ax.text(.5, .5, '处理完成后选择测点', transform=ax.transAxes, ha='center')
        else:
            d = self.result.readings[self.result.readings['station'].astype(str) == str(station)].sort_values('time_s')
            if len(d):
                self._fill_table(self.qc_gate_table, d[[
                    c for c in ('time_s', 'qc', 'accepted', 'snr_min', 'raw_dbdt_t_s',
                                'dbdt_clean_t_s', 'rho_apparent_ohm_m', 'pseudo_depth_m')
                    if c in d.columns
                ]], {'time_s': '时间 (s)', 'qc': 'QC 状态', 'accepted': '有效',
                     'snr_min': 'SNR', 'raw_dbdt_t_s': '原始 dB/dt (T/s)',
                     'dbdt_clean_t_s': '质控后 dB/dt (T/s)',
                     'rho_apparent_ohm_m': '视电阻率 (Ω·m)', 'pseudo_depth_m': '伪深度 (m)'})
                ax.loglog(d['time_s'], np.abs(d['raw_dbdt_t_s']), color='#9fb2bc', marker='.', label='原始')
                accepted = d[d['accepted']]
                if len(accepted):
                    ax.loglog(accepted['time_s'], np.abs(accepted['dbdt_clean_t_s']), color='#167b73', marker='.', label='质控后')
                rejected = d[~d['accepted']]
                if len(rejected):
                    ax.scatter(rejected['time_s'], np.maximum(np.abs(rejected['raw_dbdt_t_s']), 1e-300),
                               color='#ca8031', marker='x', label='拒绝门')
                ax.set(title=f'测点 {station} QC 衰减曲线', xlabel='时间 t (s)', ylabel='|dB/dt| (T/s)')
                ax.grid(True, which='both', alpha=.2); ax.legend(fontsize=8)
            else:
                ax.text(.5, .5, '没有该测点的时间门', transform=ax.transAxes, ha='center')
        self.qc_canvas.figure.tight_layout(); self.qc_canvas.draw_idle()

    def _refresh_geology(self):
        self.borehole_link_table.setRowCount(0)
        self.geology_canvas.figure.clear()
        if self.result is None:
            self.geology_canvas.draw_idle(); return
        try:
            if self.boreholes is not None and not self.boreholes.empty:
                self.borehole_links = link_boreholes_to_candidates(self.boreholes, self.result)
                self.result.boreholes = self.boreholes.copy()
                self.result.borehole_links = self.borehole_links.copy()
                self._fill_table(self.borehole_link_table, self.borehole_links,
                                 {'line_id': '测线', 'borehole_id': '钻孔', 'candidate_id': '候选区',
                                  'x_distance_m': 'X 距离 (m)', 'depth_overlap_m': '伪深度重叠 (m)',
                                  'candidate_x_start_m': '候选起始 X', 'candidate_x_end_m': '候选终止 X',
                                  'candidate_depth_top_m': '候选顶深', 'candidate_depth_bottom_m': '候选底深',
                                  'lithology': '岩性', 'water_evidence': '外部水文证据',
                                  'confidence': '置信度', 'notes': '备注', 'spatial_relation': '空间关系',
                                  'evidence_level': '证据级别'})
                self.borehole_source.setText(
                    f'已导入 {self.boreholes["borehole_id"].nunique()} 个钻孔、{len(self.boreholes)} 个岩性区间；'
                    f'当前测线几何关联 {len(self.borehole_links)} 条。')
                plot_joint_section(self.result, self.boreholes, fig=self.geology_canvas.figure)
            else:
                self.result.boreholes = None; self.result.borehole_links = None
                self.geology_canvas.figure.add_subplot(111).text(.5, .5, '导入钻孔区间后显示叠加图',
                                                                  transform=self.geology_canvas.figure.axes[0].transAxes,
                                                                  ha='center')
        except Exception as exc:
            self.borehole_links = None
            self.geology_canvas.figure.add_subplot(111).text(.5, .5, f'联合解释不可用：{exc}',
                                                              transform=self.geology_canvas.figure.axes[0].transAxes,
                                                              ha='center')
        self.geology_canvas.figure.tight_layout(); self.geology_canvas.draw_idle()

    def _show_error(self, title, detail):
        box = QMessageBox(self); box.setIcon(QMessageBox.Critical); box.setWindowTitle(title)
        lines = detail.strip().splitlines(); box.setText('本次操作未完成，请检查数据、参数或文件权限。')
        box.setInformativeText(lines[-1][:300] if lines else title); box.setDetailedText(detail); box.exec()

    def save_results(self):
        if self.busy or self.result is None: return
        folder = QFileDialog.getExistingDirectory(self, '选择成果输出目录')
        if folder: self.export_to(folder)

    def export_to(self, folder):
        if self.busy or self.result is None or self._result_revision != self._revision: return False
        try:
            target = export_folder(folder, self.result.line_id)
        except (OSError, ValueError):
            self._show_error('输出目录不可用', traceback.format_exc()); return False
        return self._start_task('export', result=self.result, folder=target)

    def predict_model(self):
        if self.busy: return
        if not self._has_torch:
            QMessageBox.warning(self, '缺少 PyTorch', '训练与模型推理需要安装 PyTorch。'); return
        if self.raw is None:
            QMessageBox.warning(self, '无数据', '请先导入测线。'); return
        model = self.training.model_input.text().strip() or self.modelpath.text().strip()
        if model.endswith('.pt'): model = model[:-3]
        if not Path(model + '.pt').is_file() or not Path(model + '_metadata.npz').is_file():
            QMessageBox.warning(self, '模型文件不存在', '请输入对应 .pt 与 _metadata.npz 的模型前缀。'); return
        try: cfg = self._settings(); cfg.validate()
        except Exception: self._show_error('参数错误', traceback.format_exc()); return
        self._context_changed()
        self._prediction_prefix = model
        self._start_task('predict', raw=self.raw.copy(), cfg=cfg, line=self.lines.currentText(), model=model); self.nav.setCurrentRow(2)

    def pick_dataset(self):
        if self.busy: return
        path, _ = QFileDialog.getOpenFileName(self, '选择成对训练数据', '', 'NumPy (*.npz)')
        if path: self.dataset.setText(path)

    def check_training_data(self):
        if self.busy: return
        if not self.dataset.text().strip():
            QMessageBox.warning(self, '未选择数据', '请先选择成对训练数据 NPZ。'); return
        self._start_task('dataset_info', dataset=self.dataset.text().strip())

    def choose_model_output(self):
        if self.busy: return
        path, _ = QFileDialog.getSaveFileName(self, '选择模型保存前缀', self.modelpath.text(), '模型权重 (*.pt)')
        if path: self.modelpath.setText(path[:-3] if path.endswith('.pt') else path)

    def choose_model(self):
        if self.busy: return
        path, _ = QFileDialog.getOpenFileName(self, '选择已有模型权重', '', '模型权重 (*.pt)')
        if path:
            self.training.model_input.setText(path); self.check_model_metadata()

    def check_model_metadata(self):
        if self.busy: return
        model = self.training.model_input.text().strip()
        if not model:
            QMessageBox.warning(self, '未选择模型', '请选择 .pt 权重文件。'); return
        self._start_task('model_info', model=model)

    def start_train(self):
        if self.busy: return
        if not self._has_torch:
            QMessageBox.warning(self, '缺少 PyTorch', '训练与模型推理需要安装 PyTorch。'); return
        if not Path(self.dataset.text()).is_file() or not self.modelpath.text().strip():
            QMessageBox.warning(self, '训练配置不完整', '请选择成对 NPZ 数据并填写模型输出前缀。'); return
        self._start_task('train', dataset=self.dataset.text(), output=self.modelpath.text().strip(), epochs=self.epochs.value())

    def closeEvent(self, event):
        if self.busy:
            QMessageBox.warning(self, '任务仍在运行', '请等待当前任务完成后再关闭窗口，避免数据或模型文件损坏。'); event.ignore()
        else: event.accept()
