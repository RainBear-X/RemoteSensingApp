#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
文件: main_window.py
模块: src.gui.main_window
功能: 使用 PyQt6 加载您在 Qt Designer 设计的 UI 文件，将前端界面与后端处理模块对接，并自动绑定信号槽
作者: 张子涵、孟诣楠
版本: v1.1.5
创建时间: 2025-06-10
最近更新: 2025-06-18
较上一版本改进:
    a) 适配用户提供的 UI 文件 main_window.ui
    b) 使用 PyQt6.uic 动态加载 .ui，无需先转换为 .py，减少迭代成本
    c) 保持接口不变，提供可单独运行测试的入口
    d) 针对一位数组，提供更加友好的显示效果
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
import shutil
from PyQt6 import uic
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QProgressDialog,
    QDialog,
    QFileDialog,
    QListWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QGraphicsScene,
    QCheckBox,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QComboBox,
    QSizePolicy,
    QDockWidget,
    QMenu,
    QListWidgetItem,
    QGraphicsView,
    QGraphicsPixmapItem,
    QMessageBox,
)
from PyQt6.QtGui import (
    QPixmap,
    QStandardItemModel,
    QStandardItem,
    QImage,
    QCloseEvent,
    QPolygonF,
    QPen,
    QColor,
    QCloseEvent,
)
from PyQt6.QtCore import Qt, QThread, QTimer, QRectF, QPointF
from functools import partial
import numpy as np
import rasterio
import json
from src.workers.processing_worker import ProcessingWorker
from src.workers.file_worker import FileWorker
from src.workers.file_saver_worker import FileSaverWorker
from src.workers.vector_worker import VectorWorker
from src.workers.classification_worker import ClassificationWorker
from src.workers.feature_worker import FeatureWorker
from src.workers.evaluation_worker import EvaluationWorker
from shapely.geometry import Point, LineString, Polygon
from src.processing.task_manager import TaskManager
from src.processing.task_result import TaskResult
from src.utils.image_utils import load_tif_as_numpy
import tempfile



def _load_array_from_pkl(path: str):
    """从 .pkl 文件中提取数组"""
    import pickle
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        for k in ('features', 'data', 'array', 'image', 'arr'):
            if k in obj:
                return obj[k]
        if len(obj) == 1:
            val = next(iter(obj.values()))
            return val
        return obj
    if isinstance(obj, (list, tuple)):
        return obj[0]
    return obj

def _safe_reshape_for_display(data: np.ndarray) -> np.ndarray:
    """安全地重塑数组以用于显示"""
    if data.ndim == 1:
        # 尝试将一维数组重塑为正方形
        size = int(np.sqrt(len(data)))
        if size * size == len(data):
            return data.reshape(size, size)
        else:
            # 如果不是完全平方数，创建条状图像
            height = min(100, len(data))
            width = len(data) // height
            if width * height < len(data):
                width += 1
            # 用零填充到所需大小
            padded = np.zeros(height * width)
            padded[:len(data)] = data
            return padded.reshape(height, width)
    elif data.ndim == 2:
        return data
    elif data.ndim > 2:
        # 对于高维数组，取第一个"切片"
        return data.reshape(data.shape[-2], data.shape[-1])
    return data

class ImageViewer(QGraphicsView):
    """可缩放和拖动的图像查看器"""

    def __init__(self, parent=None):
        super().__init__(parent)
        scene = QGraphicsScene(self)
        self.setScene(scene)
        self._pix_item = QGraphicsPixmapItem()
        # 防止图像项截获鼠标事件
        self._pix_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        scene.addItem(self._pix_item)
        self._zoom = 1.0
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        # 启用手形拖动模式
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # ROI 绘制相关
        self._drawing_roi = False
        self._roi_points: list[QPointF] = []
        self._roi_item = None
        self.on_roi_complete = None

    def start_roi_drawing(self, callback=None):
        """进入 ROI 绘制模式"""
        self._drawing_roi = True
        self._roi_points.clear()
        if self._roi_item is not None:
            self.scene().removeItem(self._roi_item)
            self._roi_item = None
        self.on_roi_complete = callback
        self.setDragMode(QGraphicsView.DragMode.NoDrag)

    def _finish_roi_drawing(self):
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._drawing_roi = False
        if self._roi_item is not None:
            self.scene().removeItem(self._roi_item)
            self._roi_item = None
        pts = [(p.x(), p.y()) for p in self._roi_points]
        self._roi_points.clear()
        if len(pts) >= 3:
            poly = Polygon(pts)
        else:
            poly = None
        if self.on_roi_complete:
            cb = self.on_roi_complete
            self.on_roi_complete = None
            cb(poly)

    def mousePressEvent(self, event):
        if self._drawing_roi:
            if event.button() == Qt.MouseButton.LeftButton:
                pt = self.mapToScene(event.pos())
                self._roi_points.append(pt)
                if self._roi_item is None:
                    pen = QPen(QColor("red"))
                    pen.setWidth(2)
                    self._roi_item = self.scene().addPolygon(QPolygonF(self._roi_points), pen)
                else:
                    self._roi_item.setPolygon(QPolygonF(self._roi_points))
            elif event.button() == Qt.MouseButton.RightButton:
                self._finish_roi_drawing()
            return
        super().mousePressEvent(event)

    def setPixmap(self, pix: QPixmap) -> None:
        self._pix_item.setPixmap(pix)
        # 根据图像尺寸调整场景范围，确保可拖动
        self.scene().setSceneRect(self._pix_item.boundingRect())
        self.resetTransform()
        self._zoom = 1.0
        self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def clear(self) -> None:
        self._pix_item.setPixmap(QPixmap())
        self.scene().setSceneRect(QRectF())
        self.resetTransform()
        self._zoom = 1.0

    def resizeEvent(self, event) -> None:
        if self._pix_item.pixmap() and not self._pix_item.pixmap().isNull() and self._zoom == 1.0:
            self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)
        super().resizeEvent(event)

    def wheelEvent(self, event):
        if self._pix_item.pixmap().isNull():
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self._zoom *= factor
        self.scale(factor, factor)



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
        self.progressDialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progressDialog.setCancelButtonText("取消")
        self.progressDialog.setAutoClose(True)
        self.progressDialog.setAutoReset(True)
        self.progressDialog.setMinimumDuration(0)
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
         # 当前生成的矢量文件列表
        self.current_vector_files: list[str] = []
         # 记录运行中产生的临时文件
        self.temp_files: list[str] = []
        # 当前 ROI 对象及其临时文件路径
        self.current_roi = None
        self.current_roi_path: str | None = None

        # 对应 UI 文件目录
        self.ui_dir = os.path.join(os.path.dirname(__file__), 'ui', 'yaogan')

        # 调整 UI 布局，使窗口缩放时内容可自适应
        central = getattr(self, "centralwidget", None)
        layout_widget = getattr(self, "layoutWidget", None)
        if central and layout_widget:
            layout = layout_widget.layout()
            central.setLayout(layout)
            layout_widget.setParent(None)
            if hasattr(self, "frame"):
                layout.removeWidget(self.frame)
                self.frame.deleteLater()

        # 左侧文件列表停靠面板
        self.sideDock = QDockWidget("Files", self)
        self.sideList = QListWidget()
        self.sideDock.setWidget(self.sideList)
        self.sideDock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.sideDock)
        self.sideList.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.sideList.customContextMenuRequested.connect(self._show_side_list_menu)
        self.sideList.itemChanged.connect(self._on_side_item_changed)
        self.sideList.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)

        # 在主界面右侧用于显示结果的图像查看器
        self.imageLabel = ImageViewer(self.frame_2)
        self.imageLabel.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.current_pixmap: QPixmap | None = None
        right_layout = QVBoxLayout(self.frame_2)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.imageLabel)

        # 文件状态记录
        self.file_status: dict[str, str] = {}
        # 文件可见性
        self.file_visibility: dict[str, bool] = {}
        # 向下兼容的 PNG 映射表
        self.display_pngs: dict[str, str] = {}

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
        evaluation_actions = {
            'actionConfusion_Matrix': self.show_evaluation_dialog,
            'actionOverall_Accuracy': self.show_evaluation_dialog,
            'actionKappa': self.show_evaluation_dialog,
            'actionVerify_Sample_Accuracy_Test': self.show_evaluation_dialog,
            'actionGenerate_Accuracy_Evaluation_Table': self.show_evaluation_dialog,
        }
        vector_actions = {
            'actionCreatingROI': self.show_create_roi_dialog,
            'actionSaveROIAs':   self.show_save_roi_dialog,
            'actionEditingROI':  self.show_edit_roi_dialog,
            'actionPoint':       self.show_create_point_dialog,
            'actionPolyline':    self.show_create_polyline_dialog,
            'actionPolygon':     self.show_create_polygon_dialog,
        }
        # --- Feature 菜单中的光谱指数绑定 ---
        feature_actions = {
            'actionNDVI':  lambda: self._run_spectral_index('ndvi'),
            'actionEVI':   lambda: self._run_spectral_index('evi'),
            'actionMSAVI': lambda: self._run_spectral_index('msavi'),
            'actionNDWI':  lambda: self._run_spectral_index('ndwi'),
            'actionMNDWI': lambda: self._run_spectral_index('mndwi'),
            'actionNDBI':  lambda: self._run_spectral_index('ndbi'),
            'actionBSI':   lambda: self._run_spectral_index('bsi'),

            # 高级功能项
            'actionPCA_Transformation_4': self._run_pca_transformation,
            'actionMorphological_Filteers_4': self._run_morphological_filters,
            'actionFeature_SlectionMulti_scale_3': self._run_feature_selection_multiscale,
            'actionFeature_FusionContext_3': self._run_feature_fusion_context,
        }
        for action_name, slot in feature_actions.items():
            if hasattr(self, action_name):
                getattr(self, action_name).triggered.connect(slot)

        texture_actions = {
            'actionTexture_Features': self._run_texture_features,
        }
        for action_name, slot in texture_actions.items():
            if hasattr(self, action_name):
                getattr(self, action_name).triggered.connect(slot)



        classification_actions = {
            'actionMaximum_Likelihood':      lambda: self.show_classification_dialog('maximum_likelihood'),
            'actionMinimum_Distance':       lambda: self.show_classification_dialog('minimum_distance'),
            'actionSVM':                    lambda: self.show_classification_dialog('svm'),
            'actionDecision_Tree':          lambda: self.show_classification_dialog('decision_tree'),
            'actionRandom_Forest':          lambda: self.show_classification_dialog('random_forest'),
            'actionK_means':                lambda: self.show_classification_dialog('kmeans'),
            'actionISODATA':                lambda: self.show_classification_dialog('isodata'),
            'actionDeep_leraning_Classification': self.show_deep_learning_classification_dialog,
            'actionSave_Model_As':          self.show_save_model_as_dialog,
            'actionCustom_Color':           self.show_custom_color_dialog,
            'actionSmooth_Processing':      self.show_smooth_processing_dialog,
            'actionDenoising':              self.show_denoising_dialog,
            'actionGenerating':             self.show_generate_report_dialog,
        }
        for action_name, slot in file_actions.items():
            if hasattr(self, action_name):
                getattr(self, action_name).triggered.connect(slot)

        glcm_submenu_actions = {
            'actionPCA_Transformation': self._run_pca_transformation,
            'actionMorphological_Filteers': self._run_morphological_filters,
            'actionFeature_SelectionMulti_scale': self._run_feature_selection_multiscale,
            'actionFeature_FusionContext': self._run_feature_fusion_context,
        }

        for action_name, slot in glcm_submenu_actions.items():
            if hasattr(self, action_name):
                getattr(self, action_name).triggered.connect(slot)



        for action_name, slot in image_actions.items():
            if hasattr(self, action_name):
                getattr(self, action_name).triggered.connect(slot)

        for action_name, slot in processing_actions.items():
            if hasattr(self, action_name):
                getattr(self, action_name).triggered.connect(slot)

        for action_name, slot in evaluation_actions.items():
            if hasattr(self, action_name):
                getattr(self, action_name).triggered.connect(slot)
        
        for action_name, slot in vector_actions.items():
            if hasattr(self, action_name):
                getattr(self, action_name).triggered.connect(slot)
                
        
        
        for action_name, slot in classification_actions.items():
            if hasattr(self, action_name):
                getattr(self, action_name).triggered.connect(slot)
        
        for action_name, ui_rel in dialog_map.items():
            if hasattr(self, action_name):
                getattr(self, action_name).triggered.connect(
                    partial(self.show_ui_dialog, ui_rel)
                )

        if hasattr(self, 'actionExit'):
            self.actionExit.triggered.connect(self.close)

        if hasattr(self, 'actionFeature_Extraction'):
            self.actionFeature_Extraction.triggered.connect(self._run_feature_extraction_directly)

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
        files = [
            f
            for f in os.listdir(directory)
            if f.lower().endswith(
                ('.tif', '.tiff', '.npy', '.pkl', '.pickle')
            )
        ]
        widget.clear()
        widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        widget.addItems(files)

    def _open_image(self, dialog: QDialog, widget: QListWidget, directory: str):
        selected = [os.path.join(directory, item.text()) for item in widget.selectedItems()]
        if not selected:
            return
        
        tif_paths: list[str] = []
        for f in selected:
            ext = os.path.splitext(f)[1].lower()
            if f not in self.current_image_files:
                self.current_image_files.append(f)
            if ext in ('.npy', '.pkl', '.pickle'):
                if f not in self.current_numpy_files:
                    self.current_numpy_files.append(f)
                name = os.path.basename(f)
                self.file_status[name] = '已加载'
                self.file_visibility.setdefault(name, True)
            else:
                tif_paths.append(f)

        if tif_paths:
            params = {'input_paths': tif_paths}
            self.run_file_operation(params)
        else:
            self._update_file_list()
            self._refresh_display()
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

        for path in selected:
            if path not in self.current_vector_files:
                self.current_vector_files.append(path)
            name = os.path.basename(path)
            self.file_status[name] = '已加载'
            self.file_visibility.setdefault(name, False)

        self._update_file_list()
        self._refresh_display()

        dialog.accept()

    def _populate_vector_list(self, widget: QListWidget, directory: str):
        files = [
            f for f in os.listdir(directory)
            if f.lower().endswith((".shp", ".geojson", ".json", ".gpkg"))
        ]
        widget.clear()
        widget.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        widget.addItems(files)

    def show_save_image_dialog(self):
        items = self.sideList.selectedItems()
        if not items:
            self.statusBar().showMessage('请先在左侧选择要保存的文件', 5000)
            return
        name = items[0].text().split(' - ')[0]
        src_path = next(
            (p for p in self.current_image_files if os.path.basename(p) == name),
            '',
        )
        if not src_path or not os.path.exists(src_path):
            self.statusBar().showMessage('源文件不存在', 5000)
            return
        dest, _ = QFileDialog.getSaveFileName(
            self,
            '保存文件为',
            name,
            'TIFF Files (*.tif *.tiff);;NumPy Files (*.npy);;Pickle Files (*.pkl *.pickle);;All Files (*)',
        )
        if not dest:
            return
        try:
            shutil.copy2(src_path, dest)
            self.statusBar().showMessage(f'已保存到 {dest}', 5000)
            self.file_status[name] = '已保存'
            self._update_file_list()
        except Exception as e:
            self.statusBar().showMessage(f'保存失败: {e}', 5000)


    def show_save_vector_dialog(self):
        items = self.sideList.selectedItems()
        if not items:
            self.statusBar().showMessage('请先在左侧选择要保存的矢量文件', 5000)
            return
        name = items[0].text().split(' - ')[0]
        src_path = next((p for p in self.current_vector_files if os.path.basename(p) == name), '')
        if not src_path or not os.path.exists(src_path):
            self.statusBar().showMessage('源文件不存在', 5000)
            return
        dest, _ = QFileDialog.getSaveFileName(
            self,
            '保存矢量文件为',
            name,
            'Shapefile (*.shp);;GeoJSON (*.geojson);;All Files (*)',
        )
        if not dest:
            return
        try:
            base_src, ext_src = os.path.splitext(src_path)
            base_dest, ext_dest = os.path.splitext(dest)
            if ext_src == '.shp':
                for suf in ('.shp', '.shx', '.dbf', '.cpg', '.prj'):
                    sp = base_src + suf
                    if os.path.exists(sp):
                        dp = base_dest + suf
                        shutil.copy2(sp, dp)
            else:
                shutil.copy2(src_path, dest)
            self.statusBar().showMessage(f'已保存到 {dest}', 5000)
            self.file_status[name] = '已保存'
            self._update_file_list()
        except Exception as e:
            self.statusBar().showMessage(f'保存失败: {e}', 5000)

    # ------ Vector & ROI 操作 ------
    def show_create_roi_dialog(self):
        if not self.current_image_files:
            self.statusBar().showMessage('请先加载影像文件', 5000)
            return
        self.statusBar().showMessage('左键绘制 ROI，右键结束', 0)
        self.imageLabel.start_roi_drawing(self._on_roi_drawn)

    def _on_roi_drawn(self, poly: Polygon | None):
        """ROI 绘制完成后的回调"""
        self.statusBar().clearMessage()
        if poly is None:
            self.statusBar().showMessage('ROI 绘制取消或点数不足', 5000)
            return
        self.current_roi = poly
        try:
            import tempfile
            from src.processing.vector_processing.roi_saver import save_roi_to_file
            out_dir = self.task_manager.config.vector_processing_params['output_dir']
            os.makedirs(out_dir, exist_ok=True)
            tmp_path = tempfile.mktemp(prefix='roi_', suffix='.shp', dir=out_dir)
            save_roi_to_file(self.current_roi, tmp_path)
            self.temp_files.append(tmp_path)
            self.current_vector_files.append(tmp_path)
            self.current_roi_path = tmp_path
            name = os.path.basename(tmp_path)
            self.file_status[name] = '临时'
            self.file_visibility[name] = False
            self._update_file_list()
            self.statusBar().showMessage('ROI 已创建', 5000)
        except Exception as e:
            self.statusBar().showMessage(f'创建失败: {e}', 5000)


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
                from src.processing.vector_processing.roi_saver import save_roi_to_file
                if self.current_roi_path:
                    save_roi_to_file(self.current_roi, self.current_roi_path)
                    self._update_file_list()
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
            name = os.path.basename(path)
            if path not in self.current_vector_files:
                self.current_vector_files.append(path)
            self.file_status[name] = '已保存'
            self.file_visibility[name] = False
            self._update_file_list()
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
        
        if not self.current_image_files:
            self.statusBar().showMessage('请先加载影像文件', 5000)
            dialog.reject()
            return

        img_path = self.current_image_files[0]
        try:
            from src.processing.image_display.band_extraction import extract_band
            arr = extract_band(img_path, bands)
        except Exception as e:
            self.statusBar().showMessage(f'波段提取失败: {e}', 5000)
            dialog.reject()
            return

        import tempfile
        import rasterio
        import numpy as np
        import os

        out_dir = self.task_manager.config.file_operation_params['output_dir']
        os.makedirs(out_dir, exist_ok=True)
        prefix = f"band_{'_'.join(str(b) for b in bands)}_"
        tmp_tif = tempfile.mktemp(prefix=prefix, suffix='.tif', dir=out_dir)

        with rasterio.open(img_path) as src:
            meta = src.meta.copy()
            count = arr.shape[2] if arr.ndim == 3 else 1
            meta.update(count=count, dtype=arr.dtype)
            with rasterio.open(tmp_tif, 'w', **meta) as dst:
                if arr.ndim == 2:
                    dst.write(arr, 1)
                else:
                    dst.write(arr.transpose(2, 0, 1))

        npy_path = os.path.splitext(tmp_tif)[0] + '.npy'
        save_arr = arr if arr.ndim == 3 else arr[np.newaxis, ...]
        np.save(npy_path, save_arr)

        self.temp_files.extend([tmp_tif, npy_path])

        self.current_image_files.append(tmp_tif)
        self.current_numpy_files.append(npy_path)
        name = os.path.basename(tmp_tif)
        self.file_status[name] = '临时'
        self.file_visibility[name] = True
        self._update_file_list()

        pix = self._load_raster_pixmap(tmp_tif)
        if pix:
            self._update_image_label(pix)

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
        
        if not self.current_image_files:
            self.statusBar().showMessage('请先加载影像文件', 5000)
            dialog.reject()
            return

        img_path = self.current_image_files[0]
        try:
            from src.processing.image_display.band_synthesis import synthesize_band
            img = synthesize_band(img_path, (b1, b2, b3))
        except Exception as e:
            self.statusBar().showMessage(f'波段合成失败: {e}', 5000)
            dialog.reject()
            return

        import tempfile
        import rasterio
        import numpy as np
        import os

        out_dir = self.task_manager.config.file_operation_params['output_dir']
        os.makedirs(out_dir, exist_ok=True)
        prefix = f"syn_{b1}{b2}{b3}_"
        tmp_tif = tempfile.mktemp(prefix=prefix, suffix='.tif', dir=out_dir)

        with rasterio.open(img_path) as src:
            meta = src.meta.copy()
            meta.update(count=3, dtype=img.dtype)
            with rasterio.open(tmp_tif, 'w', **meta) as dst:
                dst.write(img.transpose(2, 0, 1))

        npy_path = os.path.splitext(tmp_tif)[0] + '.npy'
        np.save(npy_path, img.transpose(2, 0, 1))

        self.temp_files.extend([tmp_tif, npy_path])

        self.current_image_files.append(tmp_tif)
        self.current_numpy_files.append(npy_path)
        name = os.path.basename(tmp_tif)
        self.file_status[name] = '临时'
        self.file_visibility[name] = True
        self._update_file_list()

        pix = self._load_raster_pixmap(tmp_tif)
        if pix:
            self._update_image_label(pix)
        dialog.accept()

    def show_histogram_dialog(self):
        path = os.path.join(self.ui_dir, 'ImageDisplay', 'Histogram.ui')
        dialog = QDialog(self)
        uic.loadUi(path, dialog)
        if hasattr(dialog, 'pushButton'):
            dialog.pushButton.clicked.connect(dialog.accept)
        img_path = self._selected_image_path()
        if img_path:
            try:
                from src.processing.image_display.histogram import band_histogram
                h = band_histogram(img_path, 1)
                counts = list(h.values())[0]
                import matplotlib.pyplot as plt
                from io import BytesIO
                fig = plt.figure(figsize=(4, 3))
                plt.bar(range(len(counts)), counts)
                buf = BytesIO()
                import tempfile
                out_dir = self.task_manager.config.file_operation_params['output_dir']
                os.makedirs(out_dir, exist_ok=True)
                tmp_png = tempfile.mktemp(prefix='hist_', suffix='.png', dir=out_dir)
                fig.savefig(tmp_png)
                self.temp_files.append(tmp_png)
                self.current_image_files.append(tmp_png)
                self.current_numpy_files.append('')
                name = os.path.basename(tmp_png)
                self.file_status[name] = '临时'
                self.file_visibility[name] = True
                self._update_file_list()

                fig.savefig(buf, format='png')
                plt.close(fig)
                buf.seek(0)
                pix = QPixmap()
                pix.loadFromData(buf.getvalue())
                scene = QGraphicsScene(dialog.graphicsView)
                scene.addPixmap(pix)
                dialog.graphicsView.setScene(scene)
                self.display_image(tmp_png)
            except Exception as e:
                self.statusBar().showMessage(f'直方图绘制失败: {e}', 5000)
        else:
            QMessageBox.information(self, '提示', '请选择文件')
        dialog.exec()

    def show_projection_dialog(self):
        path = os.path.join(self.ui_dir, 'ImageDisplay', 'Projection.ui')
        dialog = QDialog(self)
        uic.loadUi(path, dialog)
        file_path = self._selected_image_path()
        if file_path and hasattr(dialog, 'lineEdit'):
            dialog.lineEdit.setText(file_path)
        if not file_path:
            QMessageBox.information(self, '提示', '请选择文件')
            dialog.reject()
            return
        if hasattr(dialog, 'pushButton'):
            dialog.pushButton.clicked.connect(lambda: self._run_projection(dialog))
        dialog.exec()

    def _run_projection(self, dialog: QDialog):
        input_path = dialog.lineEdit.text().strip() if hasattr(dialog, 'lineEdit') else ''
        if not input_path:
            QMessageBox.information(self, '提示', '请选择文件')
            return
        import tempfile, os
        out_dir = self.task_manager.config.file_operation_params['output_dir']
        os.makedirs(out_dir, exist_ok=True)
        save_path = tempfile.mktemp(prefix='proj_', suffix='.tif', dir=out_dir)
        try:
            from osgeo import osr
            from src.processing.image_display.projection import reproject_image
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            wkt = srs.ExportToWkt()
            reproject_image(input_path, save_path, wkt)
            import rasterio, numpy as np
            with rasterio.open(save_path) as src:
                arr = src.read()
            npy_path = os.path.splitext(save_path)[0] + '.npy'
            np.save(npy_path, arr)
            self.temp_files.extend([save_path, npy_path])
            self.current_image_files.append(save_path)
            self.current_numpy_files.append(npy_path)
            name = os.path.basename(save_path)
            self.file_status[name] = '临时'
            self.file_visibility[name] = True
            self._update_file_list()
            pix = self._load_raster_pixmap(save_path)
            if pix:
                self._update_image_label(pix)
            self.statusBar().showMessage(f'已保存到 {save_path}', 5000)
        except Exception as e:
            self.statusBar().showMessage(f'投影转换失败: {e}', 5000)
        dialog.accept()

    def show_cut_dialog(self):
        img_path = self._selected_image_path()
        if not img_path:
            QMessageBox.information(self, '提示', '请选择文件')
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
        btn.clicked.connect(lambda: self._run_cut(dlg, edits, img_path))
        dlg.exec()

    def _run_cut(self, dlg: QDialog, edits: list, path: str):
        try:
            vals = [int(e.text()) for e in edits]
        except ValueError:
            self.statusBar().showMessage('请输入整数参数', 5000)
            return
        from src.processing.image_display.image_cutting import cut_image
        arr = cut_image(path, *vals)
        import tempfile, rasterio, numpy as np, os
        out_dir = self.task_manager.config.file_operation_params['output_dir']
        os.makedirs(out_dir, exist_ok=True)
        tmp_tif = tempfile.mktemp(prefix='cut_', suffix='.tif', dir=out_dir)
        with rasterio.open(path) as src:
            meta = src.meta.copy()
            count = arr.shape[2] if arr.ndim == 3 else 1
            meta.update(count=count, dtype=arr.dtype)
            with rasterio.open(tmp_tif, 'w', **meta) as dst:
                if arr.ndim == 2:
                    dst.write(arr, 1)
                else:
                    dst.write(arr.transpose(2, 0, 1))
        npy_path = os.path.splitext(tmp_tif)[0] + '.npy'
        save_arr = arr if arr.ndim == 3 else arr[np.newaxis, ...]
        np.save(npy_path, save_arr)
        self.temp_files.extend([tmp_tif, npy_path])
        self.current_image_files.append(tmp_tif)
        self.current_numpy_files.append(npy_path)
        name = os.path.basename(tmp_tif)
        self.file_status[name] = '临时'
        self.file_visibility[name] = True
        self._update_file_list()
        pix = self._load_raster_pixmap(tmp_tif)
        if pix:
            self._update_image_label(pix)
        dlg.accept()

    def show_spectral_dialog(self):
        img_path = self._selected_image_path()
        if not img_path:
            QMessageBox.information(self, '提示', '请选择文件')
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
        btn.clicked.connect(lambda: self._run_spectral(row_edit, col_edit, result_label, img_path))
        dlg.exec()

    def _run_spectral(self, row_edit: QLineEdit, col_edit: QLineEdit, label: QLabel, path: str):
        try:
            row = int(row_edit.text())
            col = int(col_edit.text())
        except ValueError:
            self.statusBar().showMessage('请输入有效的行列号', 5000)
            return
        from src.processing.image_display.spectral_analysis import pixel_spectrum
        spec = pixel_spectrum(path, row, col)
        text = ', '.join(f'{k}:{v}' for k, v in spec.items())
        label.setText(text)
        import tempfile, os
        out_dir = self.task_manager.config.file_operation_params['output_dir']
        os.makedirs(out_dir, exist_ok=True)
        tmp_txt = tempfile.mktemp(prefix='spectral_', suffix='.txt', dir=out_dir)
        with open(tmp_txt, 'w', encoding='utf-8') as f:
            for k, v in spec.items():
                f.write(f'{k}: {v}\n')
        self.temp_files.append(tmp_txt)
        self.current_image_files.append(tmp_txt)
        self.current_numpy_files.append('')
        name = os.path.basename(tmp_txt)
        self.file_status[name] = '临时'
        self.file_visibility[name] = False
        self._update_file_list()

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

    def show_evaluation_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle('Accuracy Evaluation')
        layout = QVBoxLayout(dlg)
        class_edit = QLineEdit(dlg)
        class_edit.setPlaceholderText('class_map.npy')
        pre_selected = self._selected_numpy_path()
        if pre_selected:
            class_edit.setText(pre_selected)
        class_btn = QPushButton('Browse Classification', dlg)
        roi_edit = QLineEdit(dlg)
        roi_edit.setPlaceholderText('roi_mask.npy')
        roi_btn = QPushButton('Browse ROI', dlg)
        out_edit = QLineEdit(dlg)
        out_edit.setPlaceholderText('output directory (optional)')
        default_out = getattr(self.task_manager.config, 'evaluation_params', {}).get('output_dir')
        if default_out:
            out_edit.setText(default_out)
        out_btn = QPushButton('Browse Output', dlg)
        run_btn = QPushButton('Run', dlg)
        for w in (class_edit, class_btn, roi_edit, roi_btn, out_edit, out_btn, run_btn):
            layout.addWidget(w)
        class_btn.clicked.connect(lambda: class_edit.setText(QFileDialog.getOpenFileName(self, '选择分类结果')[0]))
        roi_btn.clicked.connect(lambda: roi_edit.setText(QFileDialog.getOpenFileName(self, '选择 ROI 掩膜')[0]))
        out_btn.clicked.connect(lambda: out_edit.setText(QFileDialog.getExistingDirectory(self, '选择输出目录')))

        def act():
            class_map = class_edit.text().strip() or self._selected_numpy_path()
            roi_mask = roi_edit.text().strip()
            if not class_map or not roi_mask:
                self.statusBar().showMessage('请选择输入文件', 5000)
                return
            params = {
                'class_map_path': class_map,
                'roi_mask_path': roi_mask,
            }
            if out_edit.text().strip():
                params['output_dir'] = out_edit.text().strip()
            elif default_out:
                params['output_dir'] = default_out
            self.run_evaluation(params)
            dlg.accept()

        run_btn.clicked.connect(act)
        dlg.exec()

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
        img_path = self._selected_image_path()
        if not img_path:
            QMessageBox.information(self, '提示', '请选择文件')
            return
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
        btn.clicked.connect(lambda: self._run_stretch(img_path, low.value(), high.value(), dlg))
        dlg.exec()

    def show_equalize_dialog(self):
        img_path = self._selected_image_path()
        if not img_path:
            QMessageBox.information(self, '提示', '请选择文件')
            return
        dlg = QDialog(self)
        dlg.setWindowTitle('Histogram Equalization')
        lay = QVBoxLayout(dlg)
        btn = QPushButton('Run', dlg)
        lay.addWidget(btn)
        btn.clicked.connect(lambda: self._run_equalize(img_path, dlg))
        dlg.exec()

    def _run_stretch(self, path: str, low: int, high: int, dlg: QDialog):
        try:
            import numpy as np, pickle, rasterio, tempfile, os
            ext = os.path.splitext(path)[1].lower()
            if ext == '.npy':
                arr = np.load(path)
            elif ext in ('.pkl', '.pickle'):
                with open(path, 'rb') as f:
                    obj = pickle.load(f)
                arr = obj[0] if isinstance(obj, tuple) else obj
            else:
                with rasterio.open(path) as src:
                    arr = src.read()
            from src.processing.image_processing.enhancement.image_stretching import stretch_percent
            result = stretch_percent(arr, low, high)
            out_dir = self.task_manager.config.file_operation_params['output_dir']
            os.makedirs(out_dir, exist_ok=True)
            tmp = tempfile.mktemp(prefix='stretch_', suffix='.npy', dir=out_dir)
            np.save(tmp, result)
            self.temp_files.append(tmp)
            self.current_image_files.append(tmp)
            self.current_numpy_files.append(tmp)
            name = os.path.basename(tmp)
            self.file_status[name] = '临时'
            self.file_visibility[name] = True
            self._update_file_list()
            pix = self._load_raster_pixmap(tmp)
            if pix:
                self._update_image_label(pix)
            self.statusBar().showMessage(f'已保存到 {tmp}', 5000)
        except Exception as e:
            self.statusBar().showMessage(f'拉伸失败: {e}', 5000)
        dlg.accept()

    def _run_equalize(self, path: str, dlg: QDialog):
        try:
            import numpy as np, pickle, rasterio, tempfile, os
            ext = os.path.splitext(path)[1].lower()
            if ext == '.npy':
                arr = np.load(path)
            elif ext in ('.pkl', '.pickle'):
                with open(path, 'rb') as f:
                    obj = pickle.load(f)
                arr = obj[0] if isinstance(obj, tuple) else obj
            else:
                with rasterio.open(path) as src:
                    arr = src.read()
            from src.processing.image_processing.enhancement.equalization import hist_equalize
            result = hist_equalize(arr)
            out_dir = self.task_manager.config.file_operation_params['output_dir']
            os.makedirs(out_dir, exist_ok=True)
            tmp = tempfile.mktemp(prefix='equalize_', suffix='.npy', dir=out_dir)
            np.save(tmp, result)
            self.temp_files.append(tmp)
            self.current_image_files.append(tmp)
            self.current_numpy_files.append(tmp)
            name = os.path.basename(tmp)
            self.file_status[name] = '临时'
            self.file_visibility[name] = True
            self._update_file_list()
            pix = self._load_raster_pixmap(tmp)
            if pix:
                self._update_image_label(pix)
            self.statusBar().showMessage(f'已保存到 {tmp}', 5000)
        except Exception as e:
            self.statusBar().showMessage(f'均衡化失败: {e}', 5000)
        dlg.accept()


    def show_smoothing_dialog(self):
        img_path = self._selected_image_path()
        if not img_path:
            QMessageBox.information(self, '提示', '请选择文件')
            return
        path = os.path.join(self.ui_dir, 'ImageProcessing', 'Smoothing.ui')
        dlg = QDialog(self)
        uic.loadUi(path, dlg)
        if hasattr(dlg, 'pushButton'):
            dlg.pushButton.clicked.connect(lambda: self._run_smoothing(dlg, img_path))
        dlg.exec()

    def _run_smoothing(self, dlg: QDialog, path: str):
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
        try:
            arr = self._load_image_array(path)
            from src.processing.image_processing.filtering import smoothing as sm
            if method == 'smooth_gaussian':
                result = sm.smooth_gaussian(arr, sigma=size)
            elif method == 'smooth_mean':
                result = sm.smooth_mean(arr, size=size)
            else:
                result = sm.smooth_median(arr, size=size)
            self._save_temp_array(result, f'{method}_')
        except Exception as e:
            self.statusBar().showMessage(f'平滑失败: {e}', 5000)
        dlg.accept()

    def show_sharpening_dialog(self):
        img_path = self._selected_image_path()
        if not img_path:
            QMessageBox.information(self, '提示', '请选择文件')
            return
        path = os.path.join(self.ui_dir, 'ImageProcessing', 'Sharpening.ui')
        dlg = QDialog(self)
        uic.loadUi(path, dlg)
        if hasattr(dlg, 'pushButton'):
            dlg.pushButton.clicked.connect(lambda: self._run_sharpening(dlg, img_path))
        dlg.exec()

    def _run_sharpening(self, dlg: QDialog, path: str):
        if getattr(dlg, 'radioButton', None) and dlg.radioButton.isChecked():
            method = 'sharpen_unsharp'
        else:
            method = 'sharpen_laplacian'
        radius = getattr(dlg, 'spinBox', None)
        amount = getattr(dlg, 'spinBox_2', None)
        try:
            arr = self._load_image_array(path)
            from src.processing.image_processing.filtering import sharpening as sp
            if method == 'sharpen_unsharp':
                result = sp.sharpen_unsharp(
                    arr,
                    radius=radius.value() if radius else 1.0,
                    amount=amount.value() if amount else 1.0,
                )
            else:
                result = sp.sharpen_laplacian(
                    arr,
                    alpha=radius.value() if radius else 1.0,
                )
            self._save_temp_array(result, f'{method}_')
        except Exception as e:
            self.statusBar().showMessage(f'锐化失败: {e}', 5000)
        dlg.accept()

    def show_edge_dialog(self):
        img_path = self._selected_image_path()
        if not img_path:
            QMessageBox.information(self, '提示', '请选择文件')
            return
        path = os.path.join(self.ui_dir, 'ImageProcessing', 'Edge_detection.ui')
        dlg = QDialog(self)
        uic.loadUi(path, dlg)
        if hasattr(dlg, 'pushButton'):
            dlg.pushButton.clicked.connect(lambda: self._run_edge(dlg, img_path))
        dlg.exec()

    def _run_edge(self, dlg: QDialog, path: str):
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
        try:
            arr = self._load_image_array(path)
            from src.processing.image_processing.filtering import edge_detection as ed
            if method == 'edge_sobel':
                result = ed.edge_sobel(arr)
            elif method == 'edge_roberts':
                result = ed.edge_roberts(arr)
            else:
                result = ed.edge_canny(arr, sigma=sigma)
            self._save_temp_array(result, f'{method}_')
        except Exception as e:
            self.statusBar().showMessage(f'边缘检测失败: {e}', 5000)
        dlg.accept()

    def show_band_math_dialog(self):
        items = self.sideList.selectedItems()
        if not items:
            QMessageBox.information(self, '提示', '请选择文件')
            return
        paths: list[str] = []
        for it in items:
            name = it.text().split(" - ")[0]
            for p in self.current_image_files:
                if os.path.basename(p) == name:
                    paths.append(p)
                    break
        if not paths:
            QMessageBox.information(self, '提示', '请选择文件')
            return
        img_path = paths[0]
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
            dlg.pushButton_6.clicked.connect(lambda: self._run_band_math(dlg, model, paths))
        if hasattr(dlg, 'pushButton_7'):
            dlg.pushButton_7.clicked.connect(dlg.reject)
        dlg.exec()

    def _run_band_math(self, dlg: QDialog, model: QStandardItemModel, paths: list[str]):
        expr = dlg.lineEdit.text().strip() if hasattr(dlg, 'lineEdit') else ''
        if not expr and model.rowCount() > 0:
            idx = dlg.listView.currentIndex()
            if idx.isValid():
                expr = model.item(idx.row()).text()
        if not expr:
            self.statusBar().showMessage('请输入表达式', 5000)
            return
        import re
        vars_found = sorted({int(x) for x in re.findall(r'B(\d+)', expr)})
        if not vars_found:
            self.statusBar().showMessage('表达式中未找到波段变量', 5000)
            return

        mapping = self._select_band_sources([f'B{i}' for i in vars_found], paths)
        if mapping is None:
            return

        bands = {}
        try:
            for var, (p, b_idx) in mapping.items():
                arr = self._load_image_array(p)
                if arr.ndim == 2:
                    if b_idx != 1:
                        raise ValueError(f'{os.path.basename(p)} 只有一个波段')
                    band = arr
                else:
                    if b_idx > arr.shape[0]:
                        raise ValueError(f'{os.path.basename(p)} 没有波段 {b_idx}')
                    band = arr[b_idx - 1]
                bands[var] = band

            from src.processing.image_processing.band_math import _safe_eval
            result = _safe_eval(expr, **bands)
            self._save_temp_array(result, 'band_math_')
        except Exception as e:
            self.statusBar().showMessage(f'波段运算失败: {e}', 5000)
            return
        dlg.accept()

    def _run_feature_extraction_directly(self):
        """点击菜单后直接运行特征提取功能"""
        if not self.current_numpy_files:
            self.statusBar().showMessage("请先加载影像文件", 5000)
            return
        npy = self._selected_numpy_path()
        if not npy:
            self.statusBar().showMessage("请先在左侧选中一个文件", 5000)
            return
        output_dir = self.task_manager.config.feature_extraction_params.get("output_dir", "")
        params = {
            "input_files": [npy],
            "output_dir": output_dir
        }
        self.run_feature_extraction(params)


    def show_feature_extraction_dialog(self):
        dlg = QDialog(self)
        dlg.setWindowTitle('Feature Extraction')
        layout = QVBoxLayout(dlg)
        input_edit = QLineEdit(dlg)
        input_edit.setPlaceholderText('input files (npy or tif)')
        input_btn = QPushButton('Browse', dlg)
        out_edit = QLineEdit(dlg)
        out_edit.setPlaceholderText('output directory')
        out_btn = QPushButton('Browse', dlg)
        run_btn = QPushButton('Run', dlg)
        for w in (input_edit, input_btn, out_edit, out_btn, run_btn):
            layout.addWidget(w)
        input_btn.clicked.connect(lambda: input_edit.setText(','.join(QFileDialog.getOpenFileNames(self, '选择输入文件')[0])))
        out_btn.clicked.connect(lambda: out_edit.setText(QFileDialog.getExistingDirectory(self, '选择输出目录')))

        def act():
            files = [f for f in input_edit.text().split(',') if f.strip()]
            if not files:
                self.statusBar().showMessage('请选择输入文件', 5000)
                return
            params = {
                'input_files': files,
                'output_dir': out_edit.text().strip() or self.task_manager.config.feature_extraction_params.get('output_dir')
            }
            self.run_feature_extraction(params)
            dlg.accept()

        run_btn.clicked.connect(act)
        dlg.exec()


    def _run_spectral_index(self, index_name: str):
        """运行指定光谱指数提取模块"""
        if not self.current_numpy_files:
            self.statusBar().showMessage('请先加载影像文件', 5000)
            return
    
        from src.processing.feature_extraction.indices import (
            calculate_ndvi, calculate_evi, calculate_msavi,
            calculate_ndwi, calculate_mndwi, calculate_ndbi, calculate_bsi
        )
    
        band_map = {
            'ndvi':  ('nir', 'red'),
            'evi':   ('nir', 'red', 'blue'),
            'msavi': ('nir', 'red'),
            'ndwi':  ('green', 'nir'),
            'mndwi': ('green', 'swir'),
            'ndbi':  ('swir', 'nir'),
            'bsi':   ('blue', 'red', 'nir', 'swir'),
        }
    
        func_map = {
            'ndvi':  calculate_ndvi,
            'evi':   calculate_evi,
            'msavi': calculate_msavi,
            'ndwi':  calculate_ndwi,
            'mndwi': calculate_mndwi,
            'ndbi':  calculate_ndbi,
            'bsi':   calculate_bsi,
        }
    
        if index_name not in func_map:
            self.statusBar().showMessage(f'未知指数: {index_name}', 5000)
            return

        npy = self._selected_numpy_path()
        if not npy:
            self.statusBar().showMessage('请先在左侧选中一个文件', 5000)
            return

        try:
            arr = np.load(npy)
            if arr.ndim == 2:
                bands = [arr]
            else:
                bands = [arr[i] for i in range(arr.shape[0])]
        except Exception as e:
            self.statusBar().showMessage(f'读取影像失败: {e}', 5000)
            return
    
        name_order = ['blue', 'green', 'red', 'nir', 'swir']
        band_dict = {}
        for i, name in enumerate(name_order):
            if i < len(bands):
                band_dict[name] = bands[i]
    
        needed = band_map[index_name]
        if not all(k in band_dict for k in needed):
            self.statusBar().showMessage(f'缺少所需波段: {needed}', 5000)
            return
    
        try:
            result = func_map[index_name](*(band_dict[k] for k in needed))
            self._save_temp_array(result, f"{index_name}_")
        except Exception as e:
            self.statusBar().showMessage(f'{index_name.upper()} 计算失败: {e}', 5000)

    def _run_texture_features(self):
        """执行纹理特征提取（GLCM+LBP+Gabor）并展示部分结果"""
        if not self.current_numpy_files:
            self.statusBar().showMessage('请先加载影像文件', 5000)
            return
        npy = self._selected_numpy_path()
        if not npy:
            self.statusBar().showMessage('请先在左侧选中一个文件', 5000)
            return
        try:
            arr = np.load(npy)
            if arr.ndim == 2:
                band = arr
            else:
                # 默认第4个波段是 NIR（第3或4）
                band = arr[3] if arr.shape[0] >= 4 else arr[0]
        except Exception as e:
            self.statusBar().showMessage(f"读取影像失败: {e}", 5000)
            return
        try:
            from src.processing.feature_extraction.texture import (
                calculate_glcm_features, calculate_lbp_features, calculate_gabor_features
            )
            glcm_feats = calculate_glcm_features(band)
            lbp_feat   = calculate_lbp_features(band)
            gabor_feats = calculate_gabor_features(band)
    
            # 保存所有结果
            for name, arr in glcm_feats.items():
                self._save_temp_array(arr, f"glcm_{name}_")
    
            self._save_temp_array(lbp_feat, "lbp_")
    
            for i, gabor in enumerate(gabor_feats):
                self._save_temp_array(gabor, f"gabor_{i:02d}_")
    
            self.statusBar().showMessage("纹理特征提取完成", 5000)
    
        except Exception as e:
            self.statusBar().showMessage(f"纹理特征提取失败: {e}", 5000)



    def show_classification_dialog(self, algorithm: str):
        dlg = QDialog(self)
        dlg.setWindowTitle('Classification')
        layout = QVBoxLayout(dlg)
        feat_edit = QLineEdit(dlg)
        feat_edit.setPlaceholderText('features.npy / features.pkl')
        feat_btn = QPushButton('Browse Features', dlg)
        lbl_edit = QLineEdit(dlg)
        lbl_edit.setPlaceholderText('labels.npy / labels.pkl (optional)')
        lbl_btn = QPushButton('Browse Labels', dlg)
        param_edit = QLineEdit(dlg)
        param_edit.setPlaceholderText('extra params as JSON {"n_estimators":100}')
        auto_box = QCheckBox('自动特征提取', dlg)
        model_box = QComboBox(dlg)
        model_box.addItems(['decision_tree','random_forest','svm','maximum_likelihood','minimum_distance','kmeans','isodata'])
        if algorithm in [model_box.itemText(i) for i in range(model_box.count())]:
            idx = [model_box.itemText(i) for i in range(model_box.count())].index(algorithm)
            model_box.setCurrentIndex(idx)
        run_btn = QPushButton('Run', dlg)
        for w in (feat_edit, feat_btn, lbl_edit, lbl_btn, param_edit, auto_box, model_box, run_btn):
            layout.addWidget(w)

        feat_btn.clicked.connect(lambda: feat_edit.setText(QFileDialog.getOpenFileName(self, '选择 features', '', 'Feature Files (*.npy *.pkl *.pickle)')[0]))
        lbl_btn.clicked.connect(lambda: lbl_edit.setText(QFileDialog.getOpenFileName(self, '选择 labels', '', 'Label Files (*.npy *.pkl *.pickle)')[0]))
        auto_box.toggled.connect(lambda s: (feat_edit.setDisabled(s), feat_btn.setDisabled(s)))
        def act():
            def start_classification(feat_path: str):
                data = {'features': feat_path}
                if lbl_edit.text().strip():
                    data['labels'] = lbl_edit.text().strip()
                extra = param_edit.text().strip()
                try:
                    params_dict = json.loads(extra) if extra else {}
                except Exception as e:
                    self.statusBar().showMessage(f'参数解析错误: {e}', 5000)
                    return
                pipeline = {
                    'classifiers': [{'name': model_box.currentText(), 'params': params_dict}],
                    'compare': False
                }
                import tempfile, os
                out_dir = self.task_manager.config.file_operation_params['output_dir']
                os.makedirs(out_dir, exist_ok=True)
                tmp = tempfile.mktemp(prefix='class_', suffix='.npy', dir=out_dir)
                params = {
                    'data': data,
                    'pipeline_config': pipeline,
                    'model': model_box.currentText(),
                    'class_map_path': tmp,
                }
                self.temp_files.append(tmp)
                self.run_classification(params)
                dlg.accept()

            # ✅ 自动特征提取分支逻辑
            if auto_box.isChecked():
                if not self.current_numpy_files:
                    self.statusBar().showMessage('请先加载影像文件', 5000)
                    return
                npy = self._selected_image_path()
                if not npy:
                    self.statusBar().showMessage('请先在左侧选中一个文件', 5000)
                    return
            
            
                # ✅ 若是 .tif 则自动转换并临时保存为 .npy，再调用 start_classification
                if npy.endswith('.tif'):
                    try:
                        image = load_tif_as_numpy(npy)
                        out_dir = self.task_manager.config.file_operation_params['output_dir']
                        os.makedirs(out_dir, exist_ok=True)
                        npy_path = tempfile.mktemp(prefix='converted_', suffix='.npy', dir=out_dir)
                        
                        print(f"✔️ 已将 {npy} 转换为临时特征文件: {npy_path}")

                        np.save(npy_path, image)
                        self.temp_files.append(npy_path)
                        start_classification(npy_path)
                        return
                    except Exception as e:
                        self.statusBar().showMessage(f'影像转换失败: {e}', 5000)
                        return
            
                # ✅ 若是已有 .npy 特征，直接调用
                elif npy.endswith('.npy'):
                    start_classification(npy)
                    return
            
                self.statusBar().showMessage('当前仅支持 .tif 或 .npy 影像用于自动分类', 5000)


            else:
                feats = feat_edit.text().strip()
                if not feats:
                    self.statusBar().showMessage('请选择特征文件', 5000)
                    return
                start_classification(feats)

        run_btn.clicked.connect(act)
        dlg.exec()

    def show_deep_learning_classification_dialog(self):
        self.show_ui_dialog(os.path.join('Classification', 'Deep_Learning_Classification_dialog.ui'))

    def show_save_model_as_dialog(self):
        self.show_ui_dialog(os.path.join('Classification', 'Save_Model_As_dialog.ui'))

    def show_custom_color_dialog(self):
        self.show_ui_dialog(os.path.join('Classification', 'ClassificationResultProcessing', 'Custom_color_dialog.ui'))

    def show_smooth_processing_dialog(self):
        self.show_ui_dialog(os.path.join('Classification', 'ClassificationResultProcessing', 'Smooth_Processing_dialog.ui'))

    def show_denoising_dialog(self):
        self.show_ui_dialog(os.path.join('Classification', 'ClassificationResultProcessing', 'Denoising_dialog.ui'))

    def show_generate_report_dialog(self):
        self.show_ui_dialog(os.path.join('Classification', 'Generating_Classification_Report_dialog.ui'))

    def _get_band_count(self) -> int:
        if self.current_image_files:
           path = self._selected_image_path() or self.current_image_files[0]
           ext = os.path.splitext(path)[1].lower()
           if ext in ('.npy', '.pkl', '.pickle'):
                try:
                    import numpy as np
                    if ext == '.npy':
                        arr = np.load(path)
                    else:
                        arr = _load_array_from_pkl(path)
                    return 1 if arr.ndim == 2 else arr.shape[0]
                except Exception:
                    pass
           elif ext in ('.png', '.jpg', '.jpeg'):
                try:
                    from PIL import Image
                    img = Image.open(path)
                    return 3 if img.mode in ('RGB', 'RGBA') else 1
                except Exception:
                    pass
           else:
                try:
                    from osgeo import gdal
                    ds = gdal.Open(path)
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

    def _update_image_label(self, pixmap: QPixmap) -> None:
        """在右侧标签展示给定的图像"""
        self.current_pixmap = pixmap
        self.imageLabel.setPixmap(pixmap)

    def _load_raster_pixmap(self, path: str, bands: list[int] | None = None) -> QPixmap | None:
        """读取遥感影像或数组文件并转换为 QPixmap"""
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in ('.npy', '.pkl', '.pickle'):
                import pickle
            if ext in ('.shp', '.geojson', '.gpkg', '.json'):
                return self._load_vector_pixmap(path)
            if ext in ('.png', '.jpg', '.jpeg'):
                pix = QPixmap(path)
                return pix if not pix.isNull() else None
            if ext in ('.npy', '.pkl', '.pickle'):
                import pickle
                if ext == '.npy':
                    data = np.load(path)
                else:
                    data = _load_array_from_pkl(path)
    
                # 使用辅助函数安全地处理数组维度
                original_shape = data.shape
    
                if data.ndim == 1:
                    # 将一维数组重塑为可显示的二维数组
                    data = _safe_reshape_for_display(data)
                    data = data[np.newaxis, ...]  # 添加波段维度
                elif data.ndim == 2:
                    data = data[np.newaxis, ...]
                # data 现在应该是 (bands, height, width) 的形状
    
                if bands is None:
                    bands = [1, 2, 3] if data.shape[0] >= 3 else [1]
                bands = [b for b in bands if 1 <= b <= data.shape[0]]
                if not bands:
                    return None
                data = data[[b - 1 for b in bands]]
            else:
                with rasterio.open(path) as src:
                    if bands is None:
                        bands = [1, 2, 3] if src.count >= 3 else [1]
                    bands = [b for b in bands if 1 <= b <= src.count]
                    if not bands:
                        return None
                    data = src.read(bands)
        except Exception as e:
            self.statusBar().showMessage(f"读取影像失败: {e}", 5000)
            return None
    
        data = data.astype(float)
    
        # 安全地计算最小值和最大值
        try:
            if data.ndim == 3 and data.shape[1] > 0 and data.shape[2] > 0:
                # 正常的三维数组 (bands, height, width)
                mn = data.min(axis=(1, 2), keepdims=True)
                mx = data.max(axis=(1, 2), keepdims=True)
            else:
                # 其他情况，使用全局最小最大值
                mn = np.array(data.min()).reshape(1, 1, 1)
                mx = np.array(data.max()).reshape(1, 1, 1)
        except Exception:
            # 如果计算失败，使用简单的全局值
            mn = np.array(data.min()).reshape(1, 1, 1)
            mx = np.array(data.max()).reshape(1, 1, 1)
    
        # 避免除零
        data = (data - mn) / (mx - mn + 1e-8)
        data = (data * 255).clip(0, 255).astype(np.uint8)
    
        if data.shape[0] == 1:
            img = data[0]
            # 确保图像至少是二维的
            if img.ndim == 1:
                # 创建一个简单的条状图像
                img = np.tile(img.reshape(-1, 1), (1, 10)).T
    
            qimg = QImage(
                img.tobytes(),
                img.shape[1],
                img.shape[0],
                img.strides[0],
                QImage.Format.Format_Grayscale8,
            )
        else:
            img = np.ascontiguousarray(np.transpose(data, (1, 2, 0)))
            # 确保图像是三维的 (height, width, channels)
            if img.ndim == 2:
                img = img[:, :, np.newaxis]
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2)
            elif img.shape[2] == 2:
                # 添加第三个通道
                img = np.concatenate([img, img[:, :, :1]], axis=2)
    
            qimg = QImage(
                img.tobytes(),
                img.shape[1],
                img.shape[0],
                img.strides[0],
                QImage.Format.Format_RGB888,
            )
    
        pixmap = QPixmap.fromImage(qimg.copy())
        if pixmap.isNull():
            return None
        return pixmap

    def _load_vector_pixmap(self, path: str) -> QPixmap | None:
        """读取矢量文件并转换为 QPixmap"""
        try:
            import geopandas as gpd
            import matplotlib.pyplot as plt
            from io import BytesIO
            import warnings
            from rasterio.errors import NotGeoreferencedWarning

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
                gdf = gpd.read_file(path)

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.axis("off")
            try:
                # GeoPandas 早期版本在纬度接近 90 度时使用 ``aspect='auto'`` 会
                # 计算出无穷大的纵横比，导致 ``matplotlib`` 报错。这里改为传入
                # ``aspect=None``，跳过 GeoPandas 的自动设置，再手动设为 ``auto``。
                gdf.plot(ax=ax, facecolor="none", edgecolor="red", aspect=None)
            except Exception as e:
                plt.close(fig)
                raise e

            ax.set_aspect("auto")

            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            pix = QPixmap()
            pix.loadFromData(buf.getvalue())
            return pix if not pix.isNull() else None
        except Exception as e:
            self.statusBar().showMessage(f"矢量显示失败: {e}", 5000)
            return None
            return None

    def _load_image_array(self, path: str):
        """读取影像或数组文件为 ndarray"""
        import numpy as np, rasterio, os
        ext = os.path.splitext(path)[1].lower()
        if ext == '.npy':
            return np.load(path)
        if ext in ('.pkl', '.pickle'):
            return _load_array_from_pkl(path)
        with rasterio.open(path) as src:
            return src.read()

    def _save_temp_array(self, arr, prefix: str) -> str:
        """保存数组为临时 .npy 文件并更新列表"""
        import numpy as np, tempfile, os
        out_dir = self.task_manager.config.file_operation_params['output_dir']
        os.makedirs(out_dir, exist_ok=True)
        tmp = tempfile.mktemp(prefix=prefix, suffix='.npy', dir=out_dir)
        np.save(tmp, arr)
        self.temp_files.append(tmp)
        self.current_image_files.append(tmp)
        self.current_numpy_files.append(tmp)
        name = os.path.basename(tmp)
        self.file_status[name] = '临时'
        self.file_visibility[name] = True
        self._update_file_list()
        pix = self._load_raster_pixmap(tmp)
        if pix:
            self._update_image_label(pix)
        self.statusBar().showMessage(f'已保存到 {tmp}', 5000)
        return tmp

    def _get_band_count_for_path(self, path: str) -> int:
        """返回指定文件的波段数"""
        ext = os.path.splitext(path)[1].lower()
        if ext in ('.npy', '.pkl', '.pickle'):
            try:
                import numpy as np
                if ext == '.npy':
                    arr = np.load(path)
                else:
                    arr = _load_array_from_pkl(path)
                return 1 if arr.ndim == 2 else arr.shape[0]
            except Exception:
                return 1
        if ext in ('.png', '.jpg', '.jpeg'):
            try:
                from PIL import Image
                img = Image.open(path)
                return 3 if img.mode in ('RGB', 'RGBA') else 1
            except Exception:
                return 1
        try:
            from osgeo import gdal
            ds = gdal.Open(path)
            if ds:
                return ds.RasterCount
        except Exception:
            pass
        return 1

    def _select_band_sources(self, variables: list[str], paths: list[str]):
        """弹出对话框让用户选择每个变量对应的文件及波段"""
        dlg = QDialog(self)
        dlg.setWindowTitle('选择波段来源')
        layout = QVBoxLayout(dlg)
        rows: list[tuple[QComboBox, QSpinBox]] = []
        names = [os.path.basename(p) for p in paths]
        for var in variables:
            row = QHBoxLayout()
            label = QLabel(var, dlg)
            combo = QComboBox(dlg)
            combo.addItems(names)
            spin = QSpinBox(dlg)
            spin.setMinimum(1)
            def update_range(index, sp=spin):
                cnt = self._get_band_count_for_path(paths[index])
                sp.setMaximum(max(cnt, 1))
            combo.currentIndexChanged.connect(update_range)
            update_range(0)
            row.addWidget(label)
            row.addWidget(combo)
            row.addWidget(spin)
            layout.addLayout(row)
            rows.append((combo, spin))
        btn_row = QHBoxLayout()
        ok_btn = QPushButton('OK', dlg)
        cancel_btn = QPushButton('Cancel', dlg)
        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        layout.addLayout(btn_row)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return None
        result: dict[str, tuple[str, int]] = {}
        for var, (combo, spin) in zip(variables, rows):
            path = paths[combo.currentIndex()]
            band = spin.value()
            result[var] = (path, band)
        return result

    def _preview_bands(self, bands: list[int]):
        """根据指定波段在界面预览当前文件"""
        for path in self.current_image_files + self.current_vector_files:
            name = os.path.basename(path)
            if self.file_visibility.get(name, True):
                pix = self._load_raster_pixmap(path, bands)
                if pix:
                    self._update_image_label(pix)
                return
        self.imageLabel.clear()
        self.current_pixmap = None

    def display_image(self, img_path: str) -> None:
        """在界面展示生成的 PNG 结果，并弹出预览对话框"""
        pixmap = QPixmap(img_path)
        if pixmap.isNull():
            self.statusBar().showMessage(f"无法加载图像: {img_path}", 5000)
            return

        # 更新右侧预览标签
        self._update_image_label(pixmap)

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
        self.current_worker.setParent(self)

        worker.progress.connect(self.progressDialog.setLabelText)
         # 先清理旧线程，再回调处理结果，避免在回调中启动新线程时被覆盖
        worker.finished.connect(self._clear_current_worker)
        worker.finished.connect(lambda res: self._handle_result(title, res))

        self.progressDialog.setLabelText(f"{title}…")
        self.progressDialog.show()
        QTimer.singleShot(2000, self.progressDialog.hide)
        worker.start()

    def _handle_result(self, title: str, result: TaskResult):
        self.progressDialog.hide()
        if result.status == "success":
            msg = f"{title}完成"
            if title == "文件加载":
                for o in result.outputs:
                    if o not in self.current_numpy_files:
                        self.current_numpy_files.append(o)
                for path in self.current_image_files:
                    name = os.path.basename(path)
                    self.file_status[name] = "已加载"
                    self.file_visibility.setdefault(name, True)
                self._update_file_list()
                self._refresh_display()
            elif title == "图像处理":
                for path in self.current_image_files:
                    self.file_status[os.path.basename(path)] = "已处理"
                self._update_file_list()

            elif title == "特征提取":
                for out in result.outputs:
                    if isinstance(out, str) and out.endswith('.npy'):
                        if out not in self.current_image_files:
                            self.current_image_files.append(out)
                            self.current_numpy_files.append(out)
                            self.temp_files.append(out)
                        name = os.path.basename(out)
                        self.file_status[name] = '临时'
                        self.file_visibility[name] = True
                self._update_file_list()
                self._refresh_display()

            elif title == "分类":
                for path in self.current_image_files:
                    self.file_status[os.path.basename(path)] = "已分类"
                for out in result.outputs:
                    if isinstance(out, str) and out.endswith('.npy'):
                        if out not in self.current_image_files:
                            self.current_image_files.append(out)
                            self.current_numpy_files.append(out)
                            self.temp_files.append(out)
                        name = os.path.basename(out)
                        self.file_status[name] = '临时'
                        self.file_visibility[name] = True
            
                self._update_file_list()
                self._refresh_display()
            elif title == "精度评估":
                show_img = None
                for out in result.outputs:
                    ext = os.path.splitext(out)[1].lower()
                    name = os.path.basename(out)
                    if ext in ('.png', '.jpg', '.jpeg'):
                        if out not in self.current_image_files:
                            self.current_image_files.append(out)
                            self.current_numpy_files.append('')
                            self.temp_files.append(out)
                        self.file_status[name] = '临时'
                        self.file_visibility[name] = True
                        show_img = out
                    else:
                        self.temp_files.append(out)
                        self.file_status[name] = '已保存'
                        self.file_visibility.setdefault(name, False)
                self._update_file_list()
                if show_img:
                    pix = self._load_raster_pixmap(show_img)
                    if pix:
                        self._update_image_label(pix)
            elif title == "矢量处理":
                for o in result.outputs:
                    if o not in self.current_vector_files:
                        self.current_vector_files.append(o)
                    name = os.path.basename(o)
                    self.file_status[name] = '已保存'
                    self.file_visibility.setdefault(name, False)
                self._update_file_list()
                self._refresh_display()
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

    def run_feature_extraction(self, override: dict | None = None):
        base = getattr(self.task_manager.config, "feature_extraction_params", {}).copy()
        if override:
            base.update(override)
        worker = FeatureWorker(params=base)
        self._start_worker(worker, "特征提取")


    def _run_pca_transformation(self):
        if not self.current_numpy_files:
            self.statusBar().showMessage("请先加载图像", 5000)
            return
        npy = self._selected_numpy_path()
        if not npy:
            self.statusBar().showMessage("请先在左侧选中一个文件", 5000)
            return
        from src.processing.feature_extraction.pca import perform_pca
        try:
            arr = np.load(npy)
            if arr.ndim == 2:
                bands = [arr]
            else:
                bands = [arr[i] for i in range(arr.shape[0])]
            pca_results, var_ratio, _ = perform_pca(bands, n_components=3)
            for i, comp in enumerate(pca_results):
                self._save_temp_array(comp, f"pca_{i}_")
            self.statusBar().showMessage("PCA 完成", 5000)
        except Exception as e:
            self.statusBar().showMessage(f"PCA 错误: {e}", 5000)

    def _run_morphological_filters(self):
        if not self.current_numpy_files:
            self.statusBar().showMessage("请先加载图像", 5000)
            return
        npy = self._selected_numpy_path()
        if not npy:
            self.statusBar().showMessage("请先在左侧选中一个文件", 5000)
            return
        from src.processing.feature_extraction.morphology import (
            calculate_morphological_features,
            calculate_filter_responses
        )
        try:
            arr = np.load(npy)
            band = arr[3] if arr.ndim == 3 and arr.shape[0] >= 4 else arr[0]
            morph = calculate_morphological_features(band)
            filt  = calculate_filter_responses(band)
            for name, img in {**morph, **filt}.items():
                self._save_temp_array(img, f"{name}_")
            self.statusBar().showMessage("形态学滤波完成", 5000)
        except Exception as e:
            self.statusBar().showMessage(f"形态滤波失败: {e}", 5000)

    def _run_feature_selection_multiscale(self):
        if not self.current_numpy_files:
            self.statusBar().showMessage("请先加载图像", 5000)
            return
        npy = self._selected_numpy_path()
        if not npy:
            self.statusBar().showMessage("请先在左侧选中一个文件", 5000)
            return
        from src.processing.feature_extraction.selection import (
            feature_selection_by_variance,
            calculate_multi_scale_features
        )
        try:
            arr = np.load(npy)
            band = arr[3] if arr.ndim == 3 and arr.shape[0] >= 4 else arr[0]
            multiscale = calculate_multi_scale_features(band)
            selected = feature_selection_by_variance(multiscale)
            for name, img in selected.items():
                self._save_temp_array(img, f"{name}_")
            self.statusBar().showMessage("多尺度特征选择完成", 5000)
        except Exception as e:
            self.statusBar().showMessage(f"特征选择失败: {e}", 5000)

    def _run_feature_fusion_context(self):
        if not self.current_numpy_files:
            self.statusBar().showMessage("请先加载图像", 5000)
            return
        npy = self._selected_numpy_path()
        if not npy:
            self.statusBar().showMessage("请先在左侧选中一个文件", 5000)
            return
        from src.processing.feature_extraction.fusion import (
            feature_fusion_for_segmentation,
            hierarchical_feature_fusion,
            add_spatial_context
        )
        try:
            arr = np.load(npy)
            if arr.ndim == 2:
                feats = {'band': arr}
            else:
                feats = {f'band_{i}': arr[i] for i in range(arr.shape[0])}
            fused = feature_fusion_for_segmentation(feats)
            self._save_temp_array(fused, "fused_")
    
            with_context = add_spatial_context(np.stack(list(feats.values()), axis=-1))
            for i in range(with_context.shape[-1]):
                self._save_temp_array(with_context[:, :, i], f"context_{i}_")
    
            hier = hierarchical_feature_fusion({'ndvi': fused})
            for lvl, data in hier.items():
                if isinstance(data, np.ndarray):
                    self._save_temp_array(data[:, :, 0], f"{lvl}_0_")
                elif isinstance(data, dict):
                    for k, v in data.items():
                        self._save_temp_array(v, f"{lvl}_{k}_")
    
            self.statusBar().showMessage("特征融合完成", 5000)
        except Exception as e:
            self.statusBar().showMessage(f"融合失败: {e}", 5000)


    def run_evaluation(self, override: dict | None = None):
        base = getattr(self.task_manager.config, "evaluation_params", {}).copy()
        if override:
            base.update(override)
        worker = EvaluationWorker(params=base)
        self._start_worker(worker, "精度评估")

    def _show_side_list_menu(self, pos):
        menu = QMenu(self.sideList)
        act_remove = menu.addAction("移除选中")
        act_clear = menu.addAction("清空列表")
        action = menu.exec(self.sideList.mapToGlobal(pos))
        if action == act_remove:
            for item in self.sideList.selectedItems():
                name = item.text().split(" - ")[0]
                self.file_status.pop(name, None)
                removed = False
                for i, p in enumerate(self.current_image_files):
                    if os.path.basename(p) == name:
                        self.current_image_files.pop(i)
                        if i < len(self.current_numpy_files):
                            self.current_numpy_files.pop(i)
                        removed = True
                        break
                if not removed:
                    for i, p in enumerate(self.current_vector_files):
                        if os.path.basename(p) == name:
                            self.current_vector_files.pop(i)
                            break
                self.file_visibility.pop(name, None)
            self._update_file_list()
            self._refresh_display()
        elif action == act_clear:
            self.sideList.clear()
            self.current_image_files.clear()
            self.current_numpy_files.clear()
            self.current_vector_files.clear()
            self.file_status.clear()
            self.file_visibility.clear()
            self._refresh_display()


    def _update_file_list(self):
        """在侧边栏刷新文件状态列表"""
        self.sideList.blockSignals(True)
        self.sideList.clear()
        for path in self.current_image_files + self.current_vector_files:
            name = os.path.basename(path)
            status = self.file_status.get(name, "")
            item = QListWidgetItem(f"{name} - {status}")
            visible = self.file_visibility.get(name, True)
            item.setCheckState(Qt.CheckState.Checked if visible else Qt.CheckState.Unchecked)
            self.sideList.addItem(item)
        self.sideList.blockSignals(False)

    def _on_side_item_changed(self, item: QListWidgetItem):
        name = item.text().split(" - ")[0]
        self.file_visibility[name] = item.checkState() == Qt.CheckState.Checked
        self._refresh_display()

    def _refresh_display(self):
        # 优先在当前影像文件中查找可见项并显示
        for img_path in self.current_image_files:
            name = os.path.basename(img_path)
            if self.file_visibility.get(name, True):
                pix = self._load_raster_pixmap(img_path)
                if pix:
                    self._update_image_label(pix)
                return

        # 如果没有可见影像，尝试显示矢量文件
        for vec_path in self.current_vector_files:
            name = os.path.basename(vec_path)
            if self.file_visibility.get(name, True):
                pix = self._load_vector_pixmap(vec_path)
                if pix:
                    self._update_image_label(pix)
                return

        # 兼容旧逻辑：尝试从 display_pngs 映射展示生成的 PNG
        for npy_path, png in self.display_pngs.items():
            name = os.path.basename(npy_path)
            if self.file_visibility.get(name, True) and os.path.exists(png):
                pixmap = QPixmap(png)
                if not pixmap.isNull():
                    self._update_image_label(pixmap)
                    return

        # 没有任何可显示内容时清空
        self.imageLabel.clear()
        self.current_pixmap = None
    
    def _selected_image_path(self) -> str | None:
        """返回侧边栏当前选中影像的路径"""
        items = self.sideList.selectedItems()
        if not items:
            return None
        name = items[0].text().split(" - ")[0]
        for p in self.current_image_files:
            if os.path.basename(p) == name:
                return p
        return None

    def _selected_numpy_path(self) -> str | None:
        """返回当前选中文件对应的 numpy 路径"""
        items = self.sideList.selectedItems()
        if not items:
            return None
        name = items[0].text().split(" - ")[0]
        for i, p in enumerate(self.current_image_files):
            if os.path.basename(p) == name:
                if i < len(self.current_numpy_files):
                    npy = self.current_numpy_files[i]
                    return npy or p
        return None

    def eventFilter(self, obj, event):
        if obj is self.imageLabel and event.type() == event.Type.Resize and self.current_pixmap:
            scaled = self.current_pixmap.scaled(
                obj.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.imageLabel.setPixmap(scaled)
        return super().eventFilter(obj, event)
    
    def closeEvent(self, event: QCloseEvent) -> None:
        for path in getattr(self, "temp_files", []):
            try:
                if os.path.exists(path):
                    os.remove(path)
                hdr = os.path.splitext(path)[0] + ".hdr"
                if os.path.exists(hdr):
                    os.remove(hdr)
                base, ext = os.path.splitext(path)
                for extra in (".shx", ".dbf", ".cpg", ".prj"):
                    side = base + extra
                    if os.path.exists(side):
                        os.remove(side)
            except Exception:
                pass
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
