"""Training-operation widgets; all methods are called on the GUI thread."""
from pathlib import Path
from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QGroupBox, QLabel, QLineEdit, QPushButton, QSpinBox, QProgressBar, QPlainTextEdit)
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from .gui_support import Canvas


class TrainingPanel(QWidget):
    browse_data = Signal()
    inspect_data = Signal()
    browse_output = Signal()
    browse_model = Signal()
    inspect_model = Signal()
    train_requested = Signal()
    predict_requested = Signal()

    def __init__(self):
        super().__init__()
        self.dataset_info = self.model_info = None
        self.data_revision = self.model_revision = 0
        self._busy = False; self._has_torch = False; self._has_data = False
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(14)
        self.dependency_note = self._label('', 'boundary'); v.addWidget(self.dependency_note)
        group = QGroupBox('01　选择与检查训练数据'); form = QFormLayout(group)
        self.dataset = QLineEdit(); self.dataset.setPlaceholderText('选择成对 TEM 响应 / 电阻率标签 .npz')
        self.datasetbtn = self._button('浏览', self.browse_data, True)
        self.check_data_btn = self._button('检查训练数据', self.inspect_data, True)
        row = QHBoxLayout(); row.addWidget(self.dataset, 1); row.addWidget(self.datasetbtn); row.addWidget(self.check_data_btn)
        form.addRow('训练数据', row)
        self.data_summary = self._label('尚未检查。需要 signals、resistivity、times_s、depths_m 四个数组。')
        form.addRow(self.data_summary)
        form.addRow(self._label('均匀半空间示例仅供教学和软件验证；训练集须有可追溯的物理 / 实测标签。', 'subtitle'))
        v.addWidget(group)
        group = QGroupBox('02　配置模型保存位置与训练轮次'); form = QFormLayout(group)
        self.modelpath = QLineEdit(str(Path.home() / 'tem_model'))
        self.outputbtn = self._button('选择保存位置', self.browse_output, True)
        row = QHBoxLayout(); row.addWidget(self.modelpath, 1); row.addWidget(self.outputbtn)
        form.addRow('保存前缀', row)
        self.epochs = QSpinBox(); self.epochs.setRange(2, 1000); self.epochs.setValue(60)
        form.addRow('最大训练轮次', self.epochs)
        form.addRow(self._label('保存 .pt 权重、_metadata.npz 归一化参数及 _history.csv 记录；保持原有 8:2 划分与早停策略。', 'subtitle'))
        self.trainbtn = self._button('开始监督训练', self.train_requested)
        form.addRow(self.trainbtn); v.addWidget(group)
        group = QGroupBox('03　训练监控'); sub = QVBoxLayout(group)
        self.training_status = self._label('尚未开始训练')
        self.epoch_bar = QProgressBar(); self.epoch_bar.setRange(0, 60); self.epoch_bar.setValue(0)
        self.epoch_bar.setFormat('等待训练'); self.epoch_bar.setStyleSheet('QProgressBar {min-height:22px;max-height:22px;}')
        sub.addWidget(self.training_status); sub.addWidget(self.epoch_bar)
        row = QHBoxLayout()
        self.train_loss = self._label('训练 MSE：—'); self.val_loss = self._label('验证 MSE：—'); self.best_loss = self._label('最低验证 MSE：—')
        for label in (self.train_loss, self.val_loss, self.best_loss): row.addWidget(label, 1)
        sub.addLayout(row)
        self.canvas = Canvas(Figure(figsize=(8, 2.8), dpi=100)); self.canvas.setMinimumHeight(245)
        sub.addWidget(self.canvas)
        sub.addWidget(self._label('损失为归一化对数电阻率目标空间的 MSE，不代表水害识别准确率。', 'subtitle'))
        v.addWidget(group)
        group = QGroupBox('04　选择已有模型用于推理'); form = QFormLayout(group)
        self.model_input = QLineEdit(); self.model_input.setPlaceholderText('选择 .pt 权重，自动匹配同名前缀的 _metadata.npz')
        self.modelbtn = self._button('选择已有模型', self.browse_model, True)
        self.check_model_btn = self._button('检查模型', self.inspect_model, True)
        row = QHBoxLayout(); row.addWidget(self.model_input, 1); row.addWidget(self.modelbtn); row.addWidget(self.check_model_btn)
        form.addRow('模型文件', row)
        self.model_summary = self._label('尚未选择模型。元数据检查不读取权重内容。')
        form.addRow(self.model_summary)
        self.predictbtn = self._button('对当前测线运行模型推理', self.predict_requested, True)
        form.addRow(self.predictbtn); v.addWidget(group)
        self.log = QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMinimumHeight(150)
        self.log.setPlaceholderText('检查、训练与其他任务的运行日志。'); v.addWidget(self.log)
        self.dataset.textChanged.connect(self._data_changed)
        self.model_input.textChanged.connect(self._model_changed)
        self._reset_curve()

    @staticmethod
    def _label(text, name=None):
        label = QLabel(text); label.setWordWrap(True); label.setTextFormat(Qt.PlainText)
        if name: label.setObjectName(name)
        return label

    @staticmethod
    def _button(text, signal, secondary=False):
        button = QPushButton(text); button.setProperty('secondary', secondary)
        button.clicked.connect(signal.emit)
        return button

    def _data_changed(self):
        self.data_revision += 1; self.dataset_info = None
        self.data_summary.setText('数据路径已变化，请重新检查训练数据。')

    def _model_changed(self):
        self.model_revision += 1; self.model_info = None
        self.model_summary.setText('模型路径已变化，请重新检查配套元数据。')

    @staticmethod
    def _ranges(info):
        t = info['time_range_s']; d = info['depth_range_m']
        return f'时间 {t[0]:.6g}–{t[1]:.6g} s　深度 {d[0]:.6g}–{d[1]:.6g} m'

    def show_dataset(self, info):
        self.dataset_info = info
        rho = info['rho_range']
        self.data_summary.setText(f'检查通过 · {info["samples"]} 组样本 · {info["n_times"]} 个时间门 · '
            f'{info["n_depths"]} 个深度层\n{self._ranges(info)}\n电阻率标签 {rho[0]:.6g}–{rho[1]:.6g} Ω·m')

    def show_model(self, info):
        self.model_info = info
        self.model_summary.setText(f'元数据检查通过 · {info["n_times"]} 个时间门 · {info["n_depths"]} 个深度层\n'
            f'{self._ranges(info)}\n权重内容未检查，将在实际推理时安全加载；请仅使用可信模型。')

    def set_busy(self, busy, has_torch, has_data):
        self._busy = busy; self._has_torch = has_torch; self._has_data = has_data
        for control in (self.dataset, self.datasetbtn, self.check_data_btn, self.modelpath, self.outputbtn,
                        self.epochs, self.model_input, self.modelbtn, self.check_model_btn): control.setEnabled(not busy)
        self.trainbtn.setEnabled(not busy and has_torch)
        self.predictbtn.setEnabled(not busy and has_torch and has_data)
        self.dependency_note.setText('PyTorch 已检测到 · 数据检查不等同于模型效果验证。' if has_torch else
            '未安装 PyTorch：可以检查数据和模型元数据；训练与推理需要安装 PyTorch。')

    def _reset_curve(self):
        self.canvas.figure.clear(); self.axis = self.canvas.figure.add_subplot(111)
        self.train_line, = self.axis.plot([], [], color='#167b73', label='训练 MSE')
        self.val_line, = self.axis.plot([], [], color='#ca8031', label='验证 MSE')
        self.axis.set(xlabel='训练轮次', ylabel='归一化对数目标 MSE')
        self.axis.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.axis.grid(alpha=.15); self.axis.legend(fontsize=8)
        self.canvas.figure.tight_layout(); self.canvas.draw_idle()
        self._best = float('inf')

    def begin_training(self, total):
        self._reset_curve(); self.epoch_bar.setRange(0, total); self.epoch_bar.setValue(0)
        self.epoch_bar.setFormat(f'0 / {total} 轮')
        self.training_status.setText('训练启动中 · 正在载入并重新校验数据')
        self.train_loss.setText('训练 MSE：—'); self.val_loss.setText('验证 MSE：—'); self.best_loss.setText('最低验证 MSE：—')

    def update_progress(self, epoch, total, train_mse, val_mse):
        self.epoch_bar.setRange(0, total); self.epoch_bar.setValue(epoch); self.epoch_bar.setFormat(f'{epoch} / {total} 轮')
        self.training_status.setText(f'训练中 · 第 {epoch} / {total} 轮')
        self.train_loss.setText(f'训练 MSE：{train_mse:.6g}'); self.val_loss.setText(f'验证 MSE：{val_mse:.6g}')
        self._best = min(self._best, val_mse); self.best_loss.setText(f'最低验证 MSE：{self._best:.6g}')
        for line, value in ((self.train_line, train_mse), (self.val_line, val_mse)):
            line.set_data([*line.get_xdata(), epoch], [*line.get_ydata(), value])
        self.axis.relim(); self.axis.autoscale_view(); self.canvas.draw_idle()

    def complete_training(self, summary):
        epochs = summary['epochs']; total = self.epoch_bar.maximum()
        self.epoch_bar.setValue(epochs); self.epoch_bar.setFormat(f'{epochs} / {total} 轮 · 已结束')
        end = '提前结束（早停）' if epochs < total else '训练完成'
        self.training_status.setText(f'{end} · 实际完成 {epochs} 轮')
        if 'best_validation_mse' in summary:
            self.best_loss.setText(f'已保存模型验证 MSE：{summary["best_validation_mse"]:.6g}')
        if 'model_info' in summary:
            self.model_input.setText(summary['model_prefix'] + '.pt'); self.show_model(summary['model_info'])
