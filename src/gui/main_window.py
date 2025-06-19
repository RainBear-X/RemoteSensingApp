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
    QApplication, QMainWindow, QProgressDialog, QDialog, QFileDialog,
    QListWidget, QVBoxLayout, QLabel, QGraphicsScene, QCheckBox,
    QLineEdit, QPushButton, QSpinBox, QComboBox
)
from PyQt6.QtGui import QPixmap, QStandardItemModel, QStandardItem
from PyQt6.QtCore import Qt, QSize
from functools import partial
from PyQt6.QtCore import QThread
from src.workers.display_worker import DisplayWorker
from src.workers.processing_worker import ProcessingWorker
from src.workers.file_worker import FileWorker
from src.workers.file_saver_worker import FileSaverWorker
from src.workers.vector_worker import VectorWorker
from src.workers.classification_worker import ClassificationWorker
from shapely.geometry import Point, LineString, Polygon
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
        # 当前打开的原始影像文件列表(.tif 等)
        self.current_image_files: list[str] = []
        # 对应由 file_operation 生成的 numpy 文件列表
        self.current_numpy_files: list[str] = []
        # 当前创建的 ROI 或矢量对象
        self.current_roi = None
        self.current_vector = None

        # 对应 UI 文件目录
        self.ui_dir = os.path.join(os.path.dirname(__file__), 'ui', 'yaogan')

        # 在主界面右侧用于显示结果的 QLabel
        self.imageLabel = QLabel(self.frame_2)
        self.imageLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.imageLabel.setScaledContents(True)
        right_layout = QVBoxLayout(self.frame_2)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.imageLabel)

        # ========== 菜单与对话框绑定 ==========
        # File 菜单的四个动作需要与后台任务交互，单独处理
        file_actions = {
            'actionOpenImageFile': self.show_open_image_dialog,
            'actionOpenVectorData': self.show_open_vector_dialog,
            'actionSaveImageFileAs': self.show_save_image_dialog,
            'actionSaveVectorFileAs': self.show_save_vector_dialog,
        }

        # Image processing 动作将启动后台处理
        processing_actions = {
            'actionImagestretching': self.show_stretch_dialog,
            'actionEqualize':        self.show_equalize_dialog,
            'actionSmoothing':       self.show_smoothing_dialog,
            'actionSharpening':      self.show_sharpening_dialog,
            'actionEdgedetection':   self.show_edge_dialog,
            'actionBandMath':        self.show_band_math_dialog,
        }

        # 仅显示静态对话框的动作
        dialog_map = {
            # 其它模块可在此继续添加
        }

        image_actions = {
            'actionBandextraction':  self.show_band_extraction_dialog,
            'actionBandsynthesis':   self.show_band_synthesis_dialog,
            'actionHistogram':       self.show_histogram_dialog,
            'actionProjection':      self.show_projection_dialog,
            'actionviewingmetadata': self.show_metadata_dialog,
            'actionImageCutting':    self.show_cut_dialog,
            'actionSpectral_characteristics': self.show_spectral_dialog,
        }

        vector_actions = {
            'actionCreatingROI': self.show_create_roi_dialog,
            'actionSaveROIAs': self.show_save_roi_dialog,
            'actionEditingROI': self.show_edit_roi_dialog,
            'actionPoint': self.show_create_point_dialog,
            'actionPolyline': self.show_create_polyline_dialog,
            'actionPolygon': self.show_create_polygon_dialog,
        }

        classification_actions = {
            'actionDecision_Tree': partial(self.show_classification_dialog, 'decision_tree'),
            'actionRandom_Forest': partial(self.show_classification_dialog, 'random_forest'),
            'actionMaximum_Likelihood': partial(self.show_classification_dialog, 'maximum_likelihood'),
            'actionMinimum_Distance': partial(self.show_classification_dialog, 'minimum_distance'),
            'actionSVM': partial(self.show_classification_dialog, 'svm'),
            'actionK_means': partial(self.show_classification_dialog, 'kmeans'),
            'actionISODATA': partial(self.show_classification_dialog, 'isodata'),
        }
        for action_name, slot in file_actions.items():
            if hasattr(self, action_name):
                getattr(self, action_name).triggered.connect(slot)

        for action_name, slot in image_actions.items():
            if hasattr(self, action_name):
                getattr(self, action_name).triggered.connect(slot)

        for action_name, slot in vector_actions.items():
            if hasattr(self, action_name):
                getattr(self, action_name).triggered.connect(slot)

        for action_name, slot in classification_actions.items():
            if hasattr(self, action_name):
                getattr(self, action_name).triggered.connect(slot)

        for action_name, slot in processing_actions.items():
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
        # 在左侧 frame 中放入列表以展示文件
        list_widget = QListWidget(dialog.frame)
        layout = QVBoxLayout(dialog.frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(list_widget)

        if hasattr(dialog, 'pushButton_2'):
            dialog.pushButton_2.clicked.connect(dialog.reject)

        # 先选择目录并填充列表
        directory = QFileDialog.getExistingDirectory(self, '选择影像文件夹')
        if not directory:
            dialog.reject()
            return
        if hasattr(dialog, 'lineEdit'):
            dialog.lineEdit.setText(directory)

        self._populate_image_list(list_widget, directory)

        if hasattr(dialog, 'Open'):
            dialog.Open.clicked.connect(lambda: self._open_image(dialog, list_widget, directory))

        dialog.exec()

    def _populate_image_list(self, widget: QListWidget, directory: str):
        files = [f for f in os.listdir(directory) if f.lower().endswith(('.tif', '.tiff'))]
        widget.clear()
        widget.addItems(files)

    def _open_image(self, dialog: QDialog, widget: QListWidget, directory: str):
        selected = [os.path.join(directory, item.text()) for item in widget.selectedItems()]
        if not selected:
            return
        self.current_image_files = selected
        params = {'input_paths': selected}
        self.run_file_operation(params)
        dialog.accept()

    def show_open_vector_dialog(self):
        path = os.path.join(self.ui_dir, 'File', 'open_vector_data.ui')
        dialog = QDialog(self)
        uic.loadUi(path, dialog)

        list_widget = QListWidget(dialog.frame)
        layout = QVBoxLayout(dialog.frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(list_widget)

        if hasattr(dialog, 'pushButton_2'):
            dialog.pushButton_2.clicked.connect(dialog.reject)

        directory = QFileDialog.getExistingDirectory(self, '选择矢量文件夹')
        if not directory:
            dialog.reject()
            return
        if hasattr(dialog, 'lineEdit'):
            dialog.lineEdit.setText(directory)

        self._populate_vector_list(list_widget, directory)

        if hasattr(dialog, 'pushButton'):
            dialog.pushButton.clicked.connect(lambda: self._open_vector(dialog, list_widget, directory))

        dialog.exec()

    def _open_vector(self, dialog: QDialog, widget: QListWidget, directory: str):
        selected = [os.path.join(directory, item.text()) for item in widget.selectedItems()]
        if not selected:
            return
        params = {'input_paths': selected}
        self.run_vector_processing(params)
        dialog.accept()

    def _populate_vector_list(self, widget: QListWidget, directory: str):
        files = [f for f in os.listdir(directory) if f.lower().endswith(('.shp', '.geojson', '.json', '.gpkg'))]
        widget.clear()
        widget.addItems(files)

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

    # ------ Vector & ROI 操作 ------
    def show_create_roi_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle('Create ROI')
        layout = QVBoxLayout(dlg)
        edit = QLineEdit(dlg)
        edit.setPlaceholderText('x1,y1; x2,y2; x3,y3')
        btn = QPushButton('Create', dlg)
        layout.addWidget(edit)
        layout.addWidget(btn)

        def act():
            try:
                pts = [tuple(map(float, p.split(','))) for p in edit.text().split(';') if p.strip()]
                from src.processing.vector_processing.roi_creator import create_roi_polygon
                self.current_roi = create_roi_polygon(pts)
                self.statusBar().showMessage('ROI 已创建', 5000)
                dlg.accept()
            except Exception as e:
                self.statusBar().showMessage(f'创建失败: {e}', 5000)

        btn.clicked.connect(act)
        dlg.exec()

    def show_edit_roi_dialog(self):
        if self.current_roi is None:
            self.statusBar().showMessage('请先创建 ROI', 5000)
            return
        dlg = QDialog(self)
        dlg.setWindowTitle('Edit ROI')
        layout = QVBoxLayout(dlg)
        edit = QLineEdit(dlg)
        edit.setPlaceholderText('new_x1,new_y1; ...')
        btn = QPushButton('Update', dlg)
        layout.addWidget(edit)
        layout.addWidget(btn)

        def act():
            try:
                pts = [tuple(map(float, p.split(','))) for p in edit.text().split(';') if p.strip()]
                from src.processing.vector_processing.roi_editor import edit_roi_polygon
                self.current_roi = edit_roi_polygon(self.current_roi, pts)
                self.statusBar().showMessage('ROI 已更新', 5000)
                dlg.accept()
            except Exception as e:
                self.statusBar().showMessage(f'更新失败: {e}', 5000)

        btn.clicked.connect(act)
        dlg.exec()

    def show_save_roi_dialog(self):
        if self.current_roi is None:
            self.statusBar().showMessage('请先创建 ROI', 5000)
            return
        path, _ = QFileDialog.getSaveFileName(self, '保存 ROI', '', 'Shapefile (*.shp);;GeoJSON (*.geojson)')
        if not path:
            return
        try:
            from src.processing.vector_processing.roi_saver import save_roi_to_file
            save_roi_to_file(self.current_roi, path)
            self.statusBar().showMessage(f'ROI 已保存到 {path}', 5000)
        except Exception as e:
            self.statusBar().showMessage(f'保存失败: {e}', 5000)

    def show_create_point_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle('Create Point')
        layout = QVBoxLayout(dlg)
        x_edit = QLineEdit(dlg)
        x_edit.setPlaceholderText('x')
        y_edit = QLineEdit(dlg)
        y_edit.setPlaceholderText('y')
        btn = QPushButton('Create', dlg)
        for w in (x_edit, y_edit, btn):
            layout.addWidget(w)

        def act():
            try:
                x = float(x_edit.text())
                y = float(y_edit.text())
                self.current_vector = Point(x, y)
                self.statusBar().showMessage('点要素已创建', 5000)
                dlg.accept()
            except Exception as e:
                self.statusBar().showMessage(f'创建失败: {e}', 5000)

        btn.clicked.connect(act)
        dlg.exec()

    def show_create_polyline_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle('Create Polyline')
        layout = QVBoxLayout(dlg)
        edit = QLineEdit(dlg)
        edit.setPlaceholderText('x1,y1; x2,y2; ...')
        btn = QPushButton('Create', dlg)
        layout.addWidget(edit)
        layout.addWidget(btn)

        def act():
            try:
                pts = [tuple(map(float, p.split(','))) for p in edit.text().split(';') if p.strip()]
                self.current_vector = LineString(pts)
                self.statusBar().showMessage('折线已创建', 5000)
                dlg.accept()
            except Exception as e:
                self.statusBar().showMessage(f'创建失败: {e}', 5000)

        btn.clicked.connect(act)
        dlg.exec()

    def show_create_polygon_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle('Create Polygon')
        layout = QVBoxLayout(dlg)
        edit = QLineEdit(dlg)
        edit.setPlaceholderText('x1,y1; x2,y2; x3,y3')
        btn = QPushButton('Create', dlg)
        layout.addWidget(edit)
        layout.addWidget(btn)

        def act():
            try:
                pts = [tuple(map(float, p.split(','))) for p in edit.text().split(';') if p.strip()]
                self.current_vector = Polygon(pts)
                self.statusBar().showMessage('多边形已创建', 5000)
                dlg.accept()
            except Exception as e:
                self.statusBar().showMessage(f'创建失败: {e}', 5000)

        btn.clicked.connect(act)
        dlg.exec()

    # ------ Image Display 菜单 ------
    def show_band_extraction_dialog(self):
        path = os.path.join(self.ui_dir, 'ImageDisplay', 'Band_extraction.ui')
        dialog = QDialog(self)
        uic.loadUi(path, dialog)
        count = self._get_band_count()
        # 在滚动区域动态添加复选框供选择
        checks = []
        if hasattr(dialog, 'scrollAreaWidgetContents'):
            lay = QVBoxLayout(dialog.scrollAreaWidgetContents)
            for i in range(1, count + 1):
                cb = QCheckBox(f'Band {i}', dialog.scrollAreaWidgetContents)
                lay.addWidget(cb)
                checks.append(cb)
        if hasattr(dialog, 'pushButton'):
            dialog.pushButton.clicked.connect(lambda: self._band_extraction(dialog, checks))
        dialog.exec()

    def _band_extraction(self, dialog: QDialog, checks: list):
        bands = [i + 1 for i, cb in enumerate(checks) if cb.isChecked()]
        if not bands:
            text = dialog.lineEdit.text().strip() if hasattr(dialog, 'lineEdit') else ''
            bands = [int(b) for b in text.replace(' ', '').split(',') if b]
        if not bands:
            bands = [1]
        self.run_image_display({'bands': bands})
        dialog.accept()

    def show_band_synthesis_dialog(self):
        path = os.path.join(self.ui_dir, 'ImageDisplay', 'Band_synthesis.ui')
        dialog = QDialog(self)
        uic.loadUi(path, dialog)
        count = self._get_band_count()
        options = [str(i) for i in range(1, count + 1)]
        for cb_name in ('comboBox', 'comboBox_2', 'comboBox_3'):
            cb = getattr(dialog, cb_name, None)
            if cb is not None:
                cb.addItems(options)
        if hasattr(dialog, 'pushButton'):
            dialog.pushButton.clicked.connect(lambda: self._band_synthesis(dialog))
        dialog.exec()

    def _band_synthesis(self, dialog: QDialog):
        try:
            b1 = int(dialog.comboBox.currentText())
            b2 = int(dialog.comboBox_2.currentText())
            b3 = int(dialog.comboBox_3.currentText())
        except Exception:
            dialog.reject()
            return
        self.run_image_display({'bands': [b1, b2, b3]})
        dialog.accept()

    def show_histogram_dialog(self):
        path = os.path.join(self.ui_dir, 'ImageDisplay', 'Histogram.ui')
        dialog = QDialog(self)
        uic.loadUi(path, dialog)
        if hasattr(dialog, 'pushButton'):
            dialog.pushButton.clicked.connect(dialog.accept)
        paths = self.current_image_files
        if paths:
            try:
                from src.processing.image_display.histogram import band_histogram
                h = band_histogram(paths[0], 1)
                counts = list(h.values())[0]
                import matplotlib.pyplot as plt
                from io import BytesIO
                fig = plt.figure(figsize=(4, 3))
                plt.bar(range(len(counts)), counts)
                buf = BytesIO()
                fig.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                pix = QPixmap()
                pix.loadFromData(buf.getvalue())
                scene = QGraphicsScene(dialog.graphicsView)
                scene.addPixmap(pix)
                dialog.graphicsView.setScene(scene)
            except Exception as e:
                self.statusBar().showMessage(f'直方图绘制失败: {e}', 5000)
        else:
            self.statusBar().showMessage('请先加载影像文件', 5000)
        dialog.exec()

    def show_projection_dialog(self):
        path = os.path.join(self.ui_dir, 'ImageDisplay', 'Projection.ui')
        dialog = QDialog(self)
        uic.loadUi(path, dialog)
        file_path = self.current_image_files[0] if self.current_image_files else ''
        if hasattr(dialog, 'lineEdit'):
            dialog.lineEdit.setText(file_path)
        if not file_path:
            self.statusBar().showMessage('请先加载影像文件', 5000)
        if hasattr(dialog, 'pushButton'):
            dialog.pushButton.clicked.connect(lambda: self._run_projection(dialog))
        dialog.exec()

    def _run_projection(self, dialog: QDialog):
        input_path = dialog.lineEdit.text().strip() if hasattr(dialog, 'lineEdit') else ''
        if not input_path:
            return
        save_path, _ = QFileDialog.getSaveFileName(self, '保存为', os.path.splitext(input_path)[0] + '_proj.tif', 'TIFF Files (*.tif *.tiff)')
        if not save_path:
            return
        try:
            from osgeo import osr
            from src.processing.image_display.projection import reproject_image
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            wkt = srs.ExportToWkt()
            reproject_image(input_path, save_path, wkt)
            self.statusBar().showMessage(f'保存到 {save_path}', 5000)
        except Exception as e:
            self.statusBar().showMessage(f'投影转换失败: {e}', 5000)
        dialog.accept()

    def show_cut_dialog(self):
        if not self.current_image_files:
            self.statusBar().showMessage('请先加载影像文件', 5000)
            return
        dlg = QDialog(self)
        dlg.setWindowTitle('Image Cutting')
        layout = QVBoxLayout(dlg)
        edits = []
        for label in ('xoff', 'yoff', 'width', 'height'):
            sub = QLineEdit(dlg)
            sub.setPlaceholderText(label)
            layout.addWidget(sub)
            edits.append(sub)
        btn = QPushButton('Cut', dlg)
        layout.addWidget(btn)
        btn.clicked.connect(lambda: self._run_cut(dlg, edits))
        dlg.exec()

    def _run_cut(self, dlg: QDialog, edits: list):
        try:
            vals = [int(e.text()) for e in edits]
        except ValueError:
            self.statusBar().showMessage('请输入整数参数', 5000)
            return
        from src.processing.image_display.image_cutting import cut_image
        arr = cut_image(self.current_image_files[0], *vals)
        out_path = os.path.join(self.task_manager.config.image_display_params['output_dir'], 'cut_preview.png')
        from PIL import Image
        if arr.ndim == 3:
            img = Image.fromarray(arr)
        else:
            img = Image.fromarray(arr)
        img.save(out_path)
        self.display_image(out_path)
        dlg.accept()

    def show_spectral_dialog(self):
        if not self.current_image_files:
            self.statusBar().showMessage('请先加载影像文件', 5000)
            return
        dlg = QDialog(self)
        dlg.setWindowTitle('Spectral Analysis')
        layout = QVBoxLayout(dlg)
        row_edit = QLineEdit(dlg)
        row_edit.setPlaceholderText('row')
        col_edit = QLineEdit(dlg)
        col_edit.setPlaceholderText('col')
        layout.addWidget(row_edit)
        layout.addWidget(col_edit)
        btn = QPushButton('Analyze', dlg)
        layout.addWidget(btn)
        result_label = QLabel(dlg)
        layout.addWidget(result_label)
        btn.clicked.connect(lambda: self._run_spectral(row_edit, col_edit, result_label))
        dlg.exec()

    def _run_spectral(self, row_edit: QLineEdit, col_edit: QLineEdit, label: QLabel):
        try:
            row = int(row_edit.text())
            col = int(col_edit.text())
        except ValueError:
            self.statusBar().showMessage('请输入有效的行列号', 5000)
            return
        from src.processing.image_display.spectral_analysis import pixel_spectrum
        spec = pixel_spectrum(self.current_image_files[0], row, col)
        text = ', '.join(f'{k}:{v}' for k, v in spec.items())
        label.setText(text)

    def show_metadata_dialog(self):
        path = os.path.join(self.ui_dir, 'ImageDisplay', 'Viewing_metadata.ui')
        dialog = QDialog(self)
        uic.loadUi(path, dialog)
        if hasattr(dialog, 'pushButton'):
            dialog.pushButton.clicked.connect(dialog.accept)
        file_path = self.current_image_files[0] if self.current_image_files else ''
        if not file_path:
            self.statusBar().showMessage('请先加载影像文件', 5000)
            dialog.reject()
            return
        try:
            from src.processing.image_display.metadata_viewer import view_metadata
            meta = view_metadata(file_path)
        except Exception as e:
            self.statusBar().showMessage(f'读取元数据失败: {e}', 5000)
            meta = {}
        model = QStandardItemModel(dialog.listView)
        for k, v in meta.items():
            item = QStandardItem(f'{k}: {v}')
            model.appendRow(item)
        dialog.listView.setModel(model)
        dialog.exec()

    # ------ Image Processing 菜单 ------
    def _run_processing(self, methods: list[str], options: dict | None = None):
        """统一调用图像处理任务"""
        if not self.current_numpy_files:
            self.statusBar().showMessage('请先加载影像文件', 5000)
            return
        params = {
            'paths': self.current_numpy_files,
            'methods': methods,
            'options': options or {},
        }
        self.run_image_processing(params)

    def show_stretch_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle('Image Stretching')
        lay = QVBoxLayout(dlg)
        low = QSpinBox(dlg)
        low.setRange(0, 100)
        low.setValue(2)
        high = QSpinBox(dlg)
        high.setRange(0, 100)
        high.setValue(98)
        btn = QPushButton('Confirm', dlg)
        for w in (QLabel('Low %', dlg), low, QLabel('High %', dlg), high, btn):
            lay.addWidget(w)
        btn.clicked.connect(lambda: (self._run_processing(['stretch'], {'stretch': {'in_range': (low.value(), high.value())}}), dlg.accept()))
        dlg.exec()

    def show_equalize_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle('Histogram Equalization')
        lay = QVBoxLayout(dlg)
        btn = QPushButton('Run', dlg)
        lay.addWidget(btn)
        btn.clicked.connect(lambda: (self._run_processing(['equalization']), dlg.accept()))
        dlg.exec()

    def show_smoothing_dialog(self):
        path = os.path.join(self.ui_dir, 'ImageProcessing', 'Smoothing.ui')
        dlg = QDialog(self)
        uic.loadUi(path, dlg)
        if hasattr(dlg, 'pushButton'):
            dlg.pushButton.clicked.connect(lambda: self._run_smoothing(dlg))
        dlg.exec()

    def _run_smoothing(self, dlg: QDialog):
        if getattr(dlg, 'radioButton', None) and dlg.radioButton.isChecked():
            method = 'smooth_mean'
        elif getattr(dlg, 'radioButton_2', None) and dlg.radioButton_2.isChecked():
            method = 'smooth_gaussian'
        else:
            method = 'smooth_median'
        s1 = getattr(dlg, 'spinBox', None)
        s2 = getattr(dlg, 'spinBox_2', None)
        v1 = s1.value() if s1 else 3
        v2 = s2.value() if s2 else v1
        size = (v1, v2) if v1 != v2 else v1
        if method == 'smooth_gaussian':
            opts = {method: {'sigma': size}}
        else:
            opts = {method: {'size': size}}
        self._run_processing([method], opts)
        dlg.accept()

    def show_sharpening_dialog(self):
        path = os.path.join(self.ui_dir, 'ImageProcessing', 'Sharpening.ui')
        dlg = QDialog(self)
        uic.loadUi(path, dlg)
        if hasattr(dlg, 'pushButton'):
            dlg.pushButton.clicked.connect(lambda: self._run_sharpening(dlg))
        dlg.exec()

    def _run_sharpening(self, dlg: QDialog):
        if getattr(dlg, 'radioButton', None) and dlg.radioButton.isChecked():
            method = 'sharpen_unsharp'
        else:
            method = 'sharpen_laplacian'
        radius = getattr(dlg, 'spinBox', None)
        amount = getattr(dlg, 'spinBox_2', None)
        opts = {}
        if method == 'sharpen_unsharp':
            opts = {method: {'radius': radius.value() if radius else 1.0, 'amount': amount.value() if amount else 1.0}}
        else:
            opts = {method: {'alpha': radius.value() if radius else 1.0}}
        self._run_processing([method], opts)
        dlg.accept()

    def show_edge_dialog(self):
        path = os.path.join(self.ui_dir, 'ImageProcessing', 'Edge_detection.ui')
        dlg = QDialog(self)
        uic.loadUi(path, dlg)
        if hasattr(dlg, 'pushButton'):
            dlg.pushButton.clicked.connect(lambda: self._run_edge(dlg))
        dlg.exec()

    def _run_edge(self, dlg: QDialog):
        if getattr(dlg, 'radioButton', None) and dlg.radioButton.isChecked():
            method = 'edge_sobel'
        elif getattr(dlg, 'radioButton_3', None) and dlg.radioButton_3.isChecked():
            method = 'edge_roberts'
        else:
            method = 'edge_canny'
        s1 = getattr(dlg, 'spinBox', None)
        s2 = getattr(dlg, 'spinBox_2', None)
        val1 = s1.value() if s1 else 1
        val2 = s2.value() if s2 else val1
        sigma = (val1, val2) if val1 != val2 else float(val1)
        opts = {method: {'sigma': sigma}} if method == 'edge_canny' else {}
        self._run_processing([method], opts)
        dlg.accept()

    def show_band_math_dialog(self):
        path = os.path.join(self.ui_dir, 'ImageProcessing', 'Band_math.ui')
        dlg = QDialog(self)
        uic.loadUi(path, dlg)
        model = QStandardItemModel(dlg.listView)
        dlg.listView.setModel(model)

        history_path = self.task_manager.config.band_math_history

        def load_history():
            if not os.path.exists(history_path):
                return
            try:
                import json
                with open(history_path, 'r', encoding='utf-8') as f:
                    items = json.load(f)
                model.clear()
                for it in items:
                    model.appendRow(QStandardItem(it))
            except Exception as e:
                self.statusBar().showMessage(f'加载历史失败: {e}', 5000)

        def save_history():
            try:
                import json
                items = [model.item(i).text() for i in range(model.rowCount())]
                with open(history_path, 'w', encoding='utf-8') as f:
                    json.dump(items, f, ensure_ascii=False, indent=2)
            except Exception as e:
                self.statusBar().showMessage(f'保存历史失败: {e}', 5000)

        load_history()

        if hasattr(dlg, 'pushButton'):
            dlg.pushButton.clicked.connect(save_history)
        if hasattr(dlg, 'pushButton_2'):
            dlg.pushButton_2.clicked.connect(load_history)
        if hasattr(dlg, 'pushButton_3'):
            dlg.pushButton_3.clicked.connect(model.clear)
        if hasattr(dlg, 'pushButton_4'):
            dlg.pushButton_4.clicked.connect(lambda: model.removeRow(dlg.listView.currentIndex().row()))
        if hasattr(dlg, 'pushButton_5'):
            dlg.pushButton_5.clicked.connect(
                lambda: model.appendRow(QStandardItem(dlg.lineEdit.text().strip())) if dlg.lineEdit.text().strip() else None
            )
        if hasattr(dlg, 'pushButton_6'):
            dlg.pushButton_6.clicked.connect(lambda: self._run_band_math(dlg, model))
        if hasattr(dlg, 'pushButton_7'):
            dlg.pushButton_7.clicked.connect(dlg.reject)
        dlg.exec()

    def _run_band_math(self, dlg: QDialog, model: QStandardItemModel):
        expr = dlg.lineEdit.text().strip() if hasattr(dlg, 'lineEdit') else ''
        if not expr and model.rowCount() > 0:
            idx = dlg.listView.currentIndex()
            if idx.isValid():
                expr = model.item(idx.row()).text()
        if not expr:
            self.statusBar().showMessage('请输入表达式', 5000)
            return
        opts = {'band_math': {'expr': expr}}
        self._run_processing(['band_math'], opts)
        dlg.accept()

    def show_classification_dialog(self, algorithm: str):
        dlg = QDialog(self)
        dlg.setWindowTitle('Classification')
        layout = QVBoxLayout(dlg)
        feat_edit = QLineEdit(dlg)
        feat_edit.setPlaceholderText('features.npy')
        feat_btn = QPushButton('Browse Features', dlg)
        lbl_edit = QLineEdit(dlg)
        lbl_edit.setPlaceholderText('labels.npy (optional)')
        lbl_btn = QPushButton('Browse Labels', dlg)
        model_box = QComboBox(dlg)
        model_box.addItems(['decision_tree','random_forest','svm','maximum_likelihood','minimum_distance','kmeans','isodata'])
        if algorithm in [model_box.itemText(i) for i in range(model_box.count())]:
            idx = [model_box.itemText(i) for i in range(model_box.count())].index(algorithm)
            model_box.setCurrentIndex(idx)
        run_btn = QPushButton('Run', dlg)
        for w in (feat_edit, feat_btn, lbl_edit, lbl_btn, model_box, run_btn):
            layout.addWidget(w)

        feat_btn.clicked.connect(lambda: feat_edit.setText(QFileDialog.getOpenFileName(self, '选择 features', '', 'NumPy Files (*.npy)')[0]))
        lbl_btn.clicked.connect(lambda: lbl_edit.setText(QFileDialog.getOpenFileName(self, '选择 labels', '', 'NumPy Files (*.npy)')[0]))

        def act():
            feats = feat_edit.text().strip()
            if not feats:
                self.statusBar().showMessage('请选择特征文件', 5000)
                return
            data = {'features': feats}
            if lbl_edit.text().strip():
                data['labels'] = lbl_edit.text().strip()
            pipeline = {'classifiers': [{'name': model_box.currentText(), 'params': {}}], 'compare': False}
            params = {'data': data, 'pipeline_config': pipeline, 'model': model_box.currentText()}
            self.run_classification(params)
            dlg.accept()

        run_btn.clicked.connect(act)
        dlg.exec()

    def _get_band_count(self) -> int:
        if self.current_image_files:
            try:
                from osgeo import gdal
                ds = gdal.Open(self.current_image_files[0])
                if ds:
                    return ds.RasterCount
            except Exception:
                pass
        paths = self.task_manager.config.image_display_params.get('paths', [])
        if paths:
            try:
                import numpy as np
                arr = np.load(paths[0])
                return 1 if arr.ndim == 2 else arr.shape[0]
            except Exception:
                pass
        return 3

    def display_image(self, img_path: str) -> None:
        """在界面展示生成的 PNG 结果，并弹出预览对话框"""
        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            self.statusBar().showMessage(f"无法加载图像: {img_path}", 5000)
            return

        # 更新右侧预览标签
        scaled = pixmap.scaled(
            self.imageLabel.size() if self.imageLabel.size() != QSize(0, 0) else pixmap.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.imageLabel.setPixmap(scaled)

        # 同时弹出独立对话框便于查看完整图像
        dlg = QDialog(self)
        dlg.setWindowTitle("Image Preview")
        lbl = QLabel(dlg)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setPixmap(pixmap)
        layout = QVBoxLayout(dlg)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(lbl)
        dlg.resize(640, 480)
        dlg.exec()

    # 兼容旧代码中的 `show_image` 调用
    def show_image(self, img_path: str) -> None:
        self.display_image(img_path)



    # ===== 后台任务接口 =====
    def _start_worker(self, worker: QThread, title: str):
        """通用启动方法"""
        # 保存当前线程引用，避免被回收
        self.current_worker = worker

        worker.progress.connect(self.progressDialog.setLabelText)
        # 先清理旧线程，再回调处理结果，避免在回调中启动新线程时被覆盖
        worker.finished.connect(self._clear_current_worker)
        worker.finished.connect(lambda res: self._handle_result(title, res))

        self.progressDialog.setLabelText(f"{title}…")
        self.progressDialog.show()
        worker.start()

    def _handle_result(self, title: str, result: TaskResult):
        self.progressDialog.hide()
        if result.status == "success":
            msg = f"{title}完成"
            if title == "文件加载":
                # 更新显示任务输入并自动展示第一张影像
                self.current_numpy_files = result.outputs
                self.task_manager.config.image_display_params["paths"] = result.outputs
                self.task_manager.config.image_processing_params["paths"] = result.outputs
                if result.outputs:
                    self.run_image_display()
            elif title == "波段可视化":
                if result.outputs:
                    if hasattr(self, "display_image"):
                        self.display_image(result.outputs[0])
                    elif hasattr(self, "show_image"):
                        self.show_image(result.outputs[0])
            elif title == "图像处理":
                self.current_numpy_files = result.outputs
                self.task_manager.config.image_display_params["paths"] = result.outputs
                self.task_manager.config.image_processing_params["paths"] = result.outputs
                if result.outputs:
                    self.run_image_display()
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

    def run_image_display(self, override: dict | None = None):
        base = getattr(self.task_manager.config, "image_display_params", {}).copy()
        if override:
            base.update(override)
        worker = DisplayWorker(params=base)
        self._start_worker(worker, "波段可视化")

    def run_image_processing(self, override: dict | None = None):
        base = getattr(self.task_manager.config, "image_processing_params", {}).copy()
        if override:
            base.update(override)
        worker = ProcessingWorker(params=base)
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

    def run_classification(self, override: dict | None = None):
        base = getattr(self.task_manager.config, "classification_params", {}).copy()
        if override:
            base.update(override)
        worker = ClassificationWorker(params=base)
        self._start_worker(worker, "分类")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
