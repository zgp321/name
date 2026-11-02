"""Desktop presentation helpers and serialized background operations."""
from pathlib import Path
import hashlib
import json
import re
import traceback

from PySide6.QtCore import QThread, Signal
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QApplication, QAbstractItemView, QTableWidget
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg


STYLE = '''
QWidget {font-family:"Microsoft YaHei","Noto Sans CJK SC",sans-serif;font-size:13px;color:#203940;}
QMainWindow, #workspace {background:#eef3f5;}
#sidebar {background:#102d39;} #sidebar QLabel {color:#b5cbd4;}
#brand {font-size:23px;font-weight:700;color:white;}
QListWidget {background:transparent;border:0;color:#b5cbd4;outline:0;}
QListWidget::item {padding:16px 12px;margin:5px 0;border-radius:7px;}
QListWidget::item:selected {background:#21505a;color:#e4fffa;border-left:3px solid #55d3bb;}
#pageTitle {font-size:25px;font-weight:700;} #subtitle {color:#6b7e88;}
#boundary {background:#fff6de;color:#7a5a19;border:1px solid #ebddb7;border-radius:7px;padding:10px;}
#card, QGroupBox {background:white;border:1px solid #dfe7eb;border-radius:9px;}
QGroupBox {margin-top:12px;padding:18px 12px 12px;}
QGroupBox::title {subcontrol-origin:margin;left:14px;font-weight:600;padding:0 6px;}
#metricValue {font-size:26px;font-weight:600;color:#173d49;}
#metricTitle {color:#71838e;}
QPushButton {background:#167b73;border:1px solid #167b73;color:white;border-radius:6px;padding:9px 14px;}
QPushButton:hover {background:#11665f;} QPushButton:disabled {background:#dce5e8;border-color:#dce5e8;color:#667d86;}
QPushButton[secondary="true"] {background:white;color:#45616d;border:1px solid #cddbe1;}
QPushButton[secondary="true"]:disabled {color:#9aabb2;background:#f0f4f5;}
QDoubleSpinBox,QSpinBox,QComboBox,QLineEdit {background:white;padding:6px;border:1px solid #cddbe1;border-radius:5px;min-height:20px;}
QTableWidget,QPlainTextEdit {background:white;border:1px solid #dfe7eb;gridline-color:#e9eef0;alternate-background-color:#f6f9fa;}
QHeaderView::section {background:#f2f6f7;color:#526c79;padding:8px;border:0;border-bottom:1px solid #dfe7eb;}
QProgressBar {background:#dce6e8;border:0;border-radius:4px;min-height:8px;max-height:8px;}
QProgressBar::chunk {background:#198d80;border-radius:4px;}
QTabWidget::pane {border:0;background:white;} QTabBar::tab {padding:10px 18px;background:white;color:#72868f;}
QTabBar::tab:selected {color:#167b73;border-bottom:2px solid #167b73;}
QScrollArea {border:0;background:transparent;} QScrollArea>QWidget>QWidget {background:#eef3f5;}
QStatusBar {background:#eef3f5;color:#627b85;}
'''


class Canvas(FigureCanvasQTAgg):
    suspended = False

    def draw(self, *args, **kwargs):
        if not self.suspended: return super().draw(*args, **kwargs)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.suspended and self.figure.axes: self.figure.tight_layout()


class ReadOnlyTable(QTableWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setAlternatingRowColors(True)

    def keyPressEvent(self, event):
        if event.matches(QKeySequence.Copy):
            indexes = self.selectedIndexes()
            if indexes:
                cells = {(i.row(), i.column()): i.data() for i in indexes}
                rows = range(min(i.row() for i in indexes), max(i.row() for i in indexes) + 1)
                cols = range(min(i.column() for i in indexes), max(i.column() for i in indexes) + 1)
                QApplication.clipboard().setText('\n'.join(
                    '\t'.join(str(cells.get((r, c), '')) for c in cols) for r in rows))
            event.accept()
        else:
            super().keyPressEvent(event)


def export_folder(folder, line_id):
    """Keep user-supplied line IDs inside the selected directory, including on Windows."""
    original = str(line_id)
    safe = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', original).strip(' .')[:64] or 'line'
    if safe != original:
        safe += '_' + hashlib.sha256(original.encode('utf-8')).hexdigest()[:12]
    root = Path(folder)
    candidate = root / ('TEM_' + safe)
    if not candidate.resolve().is_relative_to(root.resolve()):
        raise ValueError('成果子目录指向所选目录以外，请选择其他输出目录。')
    return candidate


class Task(QThread):
    progress = Signal(str)
    epoch_progress = Signal(int, int, float, float)

    def __init__(self, kind, parent=None, **kwargs):
        super().__init__(parent)
        self.kind = kind
        self.args = kwargs
        self.payload = None
        self.error = None

    def run(self):
        try:
            self.payload = self._execute()
        except Exception:
            self.error = traceback.format_exc(limit=8)

    def _execute(self):
        args = self.args
        if self.kind == 'dataset_info':
            from .training_info import inspect_dataset
            return inspect_dataset(args['dataset'])
        if self.kind == 'model_info':
            from .training_info import inspect_model
            return inspect_model(args['model'])
        if self.kind == 'boreholes':
            from .geology import read_boreholes
            return read_boreholes(args['paths'])
        if self.kind == 'import':
            from .io import read_files
            data = read_files(args['paths'])
            if data.empty:
                raise ValueError('文件没有有效数据行，请检查文件内容。')
            return data
        if self.kind == 'process':
            if args.get('simpeg'):
                from .simpeg_engine import invert_line_simpeg
                return invert_line_simpeg(args['raw'], args['cfg'], args['line'],
                    callback=lambda i, n, st: self.progress.emit(f'SimPEG 1D 拟合 {i}/{n}：{st}'))[0]
            from .processing import process
            return process(args['raw'], args['cfg'], args['line'])
        if self.kind == 'export':
            from .export import export_result
            folder = Path(args['folder'])
            if folder.is_symlink() or (folder.is_dir() and any(p.is_symlink() for p in folder.iterdir())):
                raise ValueError('成果目录包含符号链接，请选择其他输出目录。')
            manifest = folder / 'manifest.json'
            if folder.is_dir() and any(folder.iterdir()):
                if not manifest.is_file():
                    raise ValueError('目标子目录包含未知文件，请选择其他输出目录。')
                prior = json.loads(manifest.read_text(encoding='utf-8'))
                if prior.get('line_id') != args['result'].line_id:
                    raise ValueError('目标子目录属于另一条测线，请选择其他输出目录。')
            return export_result(args['result'], args['folder'])
        if self.kind == 'predict':
            from .learn import predict_line
            from .training_info import inspect_model
            # Use an explicit checkpoint path: a legacy prefix may itself end in .pt.
            # Revalidate on every inference; a previous inspection can become stale.
            inspect_model(str(args['model']) + '.pt')
            return predict_line(args['raw'], args['model'], args['cfg'], args['line'])
        if self.kind == 'train':
            from .learn import train
            from .training_info import inspect_model
            summary = train(args['dataset'], args['output'], args['epochs'], callback=self._training_callback)
            summary['model_info'] = inspect_model(str(summary['model_prefix']) + '.pt')
            return summary
        raise ValueError(f'未知任务: {self.kind}')

    def _training_callback(self, epoch, total, train_mse, val_mse):
        self.epoch_progress.emit(epoch, total, train_mse, val_mse)
        self.progress.emit(f'训练 {epoch}/{total}　训练损失={train_mse:.6g}　验证损失={val_mse:.6g}')
