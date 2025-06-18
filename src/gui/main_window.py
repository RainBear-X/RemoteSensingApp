#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: main_window.py
模块: src.gui.main_window
功能: 使用 PyQt6 加载您在 Qt Designer 设计的 UI 文件，将前端界面与后端处理模块对接，并自动绑定信号槽
作者: 孟诣楠
版本: v1.0.3
创建时间: 2025-06-18
最近更新: 2025-06-18
较上一版本改进:
    a) 适配用户提供的 UI 文件 main_window.ui
    b) 使用 PyQt6.uic 动态加载 .ui，无需先转换为 .py，减少迭代成本
    c) 保持接口不变，提供可单独运行测试的入口
"""
import sys
from pathlib import Path
if __package__ is None or __package__ == "":
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src").is_dir():
            sys.path.insert(0, str(parent))
            break
import os
from PyQt6 import uic
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QProgressDialog, QDialog, QFileDialog
)
from functools import partial
from PyQt6.QtCore import QThread
from src.workers.display_worker import DisplayWorker
from src.workers.processing_worker import ProcessingWorker
from src.workers.file_worker import FileWorker
from src.workers.file_saver_worker import FileSaverWorker
from src.workers.vector_worker import VectorWorker
from src.processing.task_manager import TaskManager
from src.processing.task_result import TaskResult

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        # 动态加载 UI
        ui_path = os.path.join(os.path.dirname(__file__), 'ui', 'yaogan', 'yaogan.ui')
        uic.loadUi(ui_path, self)

        # 初始化任务管理器（加载默认配置）
        self.task_manager = TaskManager()

        # 进度对话框
        self.progressDialog = QProgressDialog(self)
        self.progressDialog.setAutoClose(False)
        self.progressDialog.setLabelText("准备中…")
        self.progressDialog.hide()
        # 取消按钮关闭当前线程
        self.progressDialog.canceled.connect(self.cancel_current_worker)

        # 当前运行的后台线程引用，避免被垃圾回收
        self.current_worker: QThread | None = None

        # 对应 UI 文件目录
        self.ui_dir = os.path.join(os.path.dirname(__file__), 'ui', 'yaogan')

        # ========== 菜单与对话框绑定 ==========
        # File 菜单的四个动作需要与后台任务交互，单独处理
        file_actions = {
            'actionOpenImageFile': self.show_open_image_dialog,
            'actionOpenVectorData': self.show_open_vector_dialog,
            'actionSaveImageFileAs': self.show_save_image_dialog,
            'actionSaveVectorFileAs': self.show_save_vector_dialog,
        }

        # 其余动作仅弹出对应的 ui
        dialog_map = {
            'actionBandextraction':     'ImageDisplay/Band_extraction.ui',
            'actionBandsynthesis':      'ImageDisplay/Band_synthesis.ui',
            'actionHistogram':          'ImageDisplay/Histogram.ui',
            'actionProjection':         'ImageDisplay/Projection.ui',
            'actionviewingmetadata':    'ImageDisplay/Viewing_metadata.ui',
            'actionBandMath':           'ImageProcessing/Band_math.ui',
            'actionEdgedetection':      'ImageProcessing/Edge_detection.ui',
            'actionSharpening':         'ImageProcessing/Sharpening.ui',
            'actionSmoothing':          'ImageProcessing/Smoothing.ui',
        }
        for action_name, slot in file_actions.items():
            if hasattr(self, action_name):
                getattr(self, action_name).triggered.connect(slot)

        for action_name, ui_rel in dialog_map.items():
            if hasattr(self, action_name):
                getattr(self, action_name).triggered.connect(
                    partial(self.show_ui_dialog, ui_rel)
                )

        if hasattr(self, 'actionExit'):
            self.actionExit.triggered.connect(self.close)

        # 旧版后台任务入口保留（未连接到菜单）

    def show_ui_dialog(self, ui_relative_path: str):
        """根据相对路径加载并显示一个对话框"""
        path = os.path.join(self.ui_dir, ui_relative_path)
        dialog = QDialog(self)
        uic.loadUi(path, dialog)
        dialog.exec()

    # ------ File 菜单专用对话框 ------
    def show_open_image_dialog(self):
        path = os.path.join(self.ui_dir, 'File', 'open_image_file.ui')
        dialog = QDialog(self)
        uic.loadUi(path, dialog)
        if hasattr(dialog, 'Open'):
            dialog.Open.clicked.connect(lambda: self._open_image(dialog))
        if hasattr(dialog, 'pushButton_2'):
            dialog.pushButton_2.clicked.connect(dialog.reject)
        dialog.exec()

    def _open_image(self, dialog: QDialog):
        directory = QFileDialog.getExistingDirectory(self, '选择影像文件夹')
        if not directory:
            return
        if hasattr(dialog, 'lineEdit'):
            dialog.lineEdit.setText(directory)
        params = {'input_dir': directory}
        self.run_file_operation(params)
        dialog.accept()

    def show_open_vector_dialog(self):
        path = os.path.join(self.ui_dir, 'File', 'open_vector_data.ui')
        dialog = QDialog(self)
        uic.loadUi(path, dialog)
        if hasattr(dialog, 'pushButton'):
            dialog.pushButton.clicked.connect(lambda: self._open_vector(dialog))
        if hasattr(dialog, 'pushButton_2'):
            dialog.pushButton_2.clicked.connect(dialog.reject)
        dialog.exec()

    def _open_vector(self, dialog: QDialog):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            '选择矢量文件',
            '',
            'Vector Files (*.shp *.geojson *.json *.gpkg)'
        )
        if not files:
            return
        if hasattr(dialog, 'lineEdit'):
            dialog.lineEdit.setText(files[0])
        params = {'input_paths': files}
        self.run_vector_processing(params)
        dialog.accept()

    def show_save_image_dialog(self):
        path = os.path.join(self.ui_dir, 'File', 'save_image_as.ui')
        dialog = QDialog(self)
        uic.loadUi(path, dialog)
        if hasattr(dialog, 'pushButton'):
            dialog.pushButton.clicked.connect(lambda: self._save_image(dialog))
        if hasattr(dialog, 'pushButton_2'):
            dialog.pushButton_2.clicked.connect(dialog.reject)
        dialog.exec()

    def _save_image(self, dialog: QDialog):
        directory = QFileDialog.getExistingDirectory(self, '选择保存目录')
        if not directory:
            return
        if hasattr(dialog, 'lineEdit'):
            dialog.lineEdit.setText(directory)
        params = {'save_dir': directory}
        self.run_file_save(params)
        dialog.accept()

    def show_save_vector_dialog(self):
        path = os.path.join(self.ui_dir, 'File', 'save_vector_as.ui')
        dialog = QDialog(self)
        uic.loadUi(path, dialog)
        if hasattr(dialog, 'pushButton'):
            dialog.pushButton.clicked.connect(lambda: self._save_vector(dialog))
        if hasattr(dialog, 'pushButton_2'):
            dialog.pushButton_2.clicked.connect(dialog.reject)
        dialog.exec()

    def _save_vector(self, dialog: QDialog):
        directory = QFileDialog.getExistingDirectory(self, '选择保存目录')
        if not directory:
            return
        if hasattr(dialog, 'lineEdit'):
            dialog.lineEdit.setText(directory)
        params = {'output_dir': directory}
        self.run_vector_processing(params)
        dialog.accept()



    # ===== 后台任务接口 =====
    def _start_worker(self, worker: QThread, title: str):
        """通用启动方法"""
        # 保存当前线程引用，避免被回收
        self.current_worker = worker

        worker.progress.connect(self.progressDialog.setLabelText)
        worker.finished.connect(lambda res: self._handle_result(title, res))
        worker.finished.connect(self._clear_current_worker)

        self.progressDialog.setLabelText(f"{title}…")
        self.progressDialog.show()
        worker.start()

    def _handle_result(self, title: str, result: TaskResult):
        self.progressDialog.hide()
        if result.status == "success":
            msg = f"{title}完成"
        else:
            msg = f"{title}失败: {result.message}"
        self.statusBar().showMessage(msg, 5000)


    def cancel_current_worker(self):
        """取消当前运行的后台线程"""
        if self.current_worker and self.current_worker.isRunning():
            self.current_worker.terminate()

    # 向后兼容旧接口
    def _cancel_current_worker(self):
        self.cancel_current_worker()

    def _clear_current_worker(self):
        self.current_worker = None

    def run_file_operation(self, override: dict | None = None):
        base = getattr(self.task_manager.config, "file_operation_params", {}).copy()
        if override:
            base.update(override)
        worker = FileWorker(params=base)
        self._start_worker(worker, "文件加载")

    def run_image_display(self):
        params = getattr(self.task_manager.config, "image_display_params", {})
        worker = DisplayWorker(params=params)
        self._start_worker(worker, "波段可视化")

    def run_image_processing(self):
        params = getattr(self.task_manager.config, "image_processing_params", {})
        worker = ProcessingWorker(params=params)
        self._start_worker(worker, "图像处理")

    def run_file_save(self, override: dict | None = None):
        base = getattr(self.task_manager.config, "file_saver_params", {}).copy()
        if override:
            base.update(override)
        worker = FileSaverWorker(params=base)
        self._start_worker(worker, "文件保存")

    def run_vector_processing(self, override: dict | None = None):
        base = getattr(self.task_manager.config, "vector_processing_params", {}).copy()
        if override:
            base.update(override)
        worker = VectorWorker(params=base)
        self._start_worker(worker, "矢量处理")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
