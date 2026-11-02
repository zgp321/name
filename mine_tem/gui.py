"""Lazy desktop entry points; importing the CLI does not require Qt."""
import sys


def create_window():
    from .gui_window import Window
    return Window()


def main():
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError:
        print('未安装 PySide6。请运行: python -m pip install -e .', file=sys.stderr)
        return 2
    app = QApplication(sys.argv)
    app.setApplicationName('矿井瞬变电磁 TEM 水害辅助判读工作台')
    window = create_window()
    window.show()
    return app.exec()


if __name__ == '__main__':
    raise SystemExit(main())
