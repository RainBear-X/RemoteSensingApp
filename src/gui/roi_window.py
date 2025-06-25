#!/usr/bin/env python3.12
# -*- coding: utf-8 -*-
"""
roi_window.py
--------------
一个简易的 ROI 编辑窗口。在独立的窗口中加载图像并支持绘制多边形 ROI，
可为每个 ROI 指定整数标签，最后导出为 mask.npy 文件。
"""

from __future__ import annotations

import os
from typing import List, Tuple

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPen, QPolygonF, QImage, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QInputDialog,
    QFileDialog, QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from shapely.geometry import Polygon
import numpy as np
from rasterio.features import rasterize
from src.utils.image_utils import load_tif_as_numpy
from src.processing.image_processing.enhancement.image_stretching import (
    stretch_percent,
)
import rasterio


class _ImageViewer(QGraphicsView):
    """可缩放、支持 ROI 绘制的图像查看器，使用鼠标滚轮缩放"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self._pix_item = QGraphicsPixmapItem()
        self._pix_item.setAcceptedMouseButtons(Qt.MouseButton.NoButton)
        self.scene().addItem(self._pix_item)
        self._zoom = 1.0
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._drawing = False
        self._points: List = []
        self._poly_item = None
        self.on_complete = None

    def start_drawing(self, callback=None):
        self._drawing = True
        self._points.clear()
        if self._poly_item is not None:
            self.scene().removeItem(self._poly_item)
            self._poly_item = None
        self.on_complete = callback
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def _finish_drawing(self):
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._drawing = False
        self.viewport().unsetCursor()
        if self._poly_item is not None:
            self.scene().removeItem(self._poly_item)
            self._poly_item = None
        pts = [(p.x(), p.y()) for p in self._points]
        self._points.clear()
        poly = Polygon(pts) if len(pts) >= 3 else None
        if self.on_complete:
            cb = self.on_complete
            self.on_complete = None
            cb(poly)

    def mousePressEvent(self, event):
        if self._drawing:
            if event.button() == Qt.MouseButton.LeftButton:
                pt = self.mapToScene(event.pos())
                self._points.append(pt)
                if self._poly_item is None:
                    pen = QPen(QColor("red"))
                    pen.setWidth(2)
                    self._poly_item = self.scene().addPolygon(QPolygonF(self._points), pen)
                else:
                    self._poly_item.setPolygon(QPolygonF(self._points))
            elif event.button() == Qt.MouseButton.RightButton:
                self._finish_drawing()
            return
        super().mousePressEvent(event)

    def setPixmap(self, pix: QPixmap) -> None:
        self._pix_item.setPixmap(pix)
        self.scene().setSceneRect(self._pix_item.boundingRect())
        self.resetTransform()
        self._zoom = 1.0
        self.fitInView(self._pix_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event):
        if self._pix_item.pixmap().isNull():
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self._zoom *= factor
        self.scale(factor, factor)


class ROIWindow(QDialog):
    """独立的 ROI 绘制与保存窗口"""

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ROI Editor - 滚轮缩放，右键结束绘制")
        # 调整窗口大小，方便在大影像上绘制
        self.viewer = _ImageViewer(self)
        self.viewer.setMinimumSize(800, 600)
        self.resize(1000, 800)

        self.polygons: List[Tuple[Polygon, int]] = []
        self.saved_mask_path: str | None = None
        # 转为绝对路径，避免在不同平台出现斜杠混用导致加载失败
        self.image_path = os.path.abspath(image_path)
        layout = QVBoxLayout(self)
        layout.addWidget(self.viewer)
        btn_layout = QHBoxLayout()
        self.draw_btn = QPushButton("Draw ROI", self)
        self.save_btn = QPushButton("Save mask", self)
        btn_layout.addWidget(self.draw_btn)
        btn_layout.addWidget(self.save_btn)
        layout.addLayout(btn_layout)

        self.draw_btn.clicked.connect(lambda: self.viewer.start_drawing(self._roi_done))
        self.save_btn.clicked.connect(self._export_mask)

        if not os.path.isfile(self.image_path):
            QMessageBox.warning(self, "ROI", f"找不到图像文件: {self.image_path}")
            return

        pix = self._prepare_preview_pixmap(self.image_path)
        if pix is None or pix.isNull():
            pix = self._load_pixmap(self.image_path)
        if pix is not None and not pix.isNull():
            self.viewer.setPixmap(pix)
        else:
            QMessageBox.warning(self, "ROI", f"无法加载图像: {self.image_path}")

    def _prepare_preview_pixmap(self, path: str) -> QPixmap | None:
        """直接生成预览 QPixmap，采用(3,2,1)波段并做 2% 拉伸"""
        try:
            with rasterio.open(path) as src:
                bands = src.count
                idx = [3, 2, 1]
                idx = [b for b in idx if b <= bands]
                if not idx:
                    idx = list(range(1, min(3, bands) + 1))
                data = src.read(idx).astype(np.float32)
            data = stretch_percent(data, 2.0, 98.0)
            data = (data * 255).clip(0, 255).astype(np.uint8)
            data = np.ascontiguousarray(data)
            if data.shape[0] == 1:
                img = data[0]
                img = np.ascontiguousarray(img)
                qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format.Format_Grayscale8).copy()
            else:
                img = np.transpose(data, (1, 2, 0))
                img = np.ascontiguousarray(img)
                qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format.Format_RGB888).copy()
            return QPixmap.fromImage(qimg)
        except Exception as e:
            print(f"[ROIWindow] 预览生成失败: {e}")
            return None

    def _load_pixmap(self, path: str) -> QPixmap | None:
        # 先尝试直接使用 QPixmap 加载
        pix = QPixmap(path)
        if not pix.isNull():
            return pix

        ext = os.path.splitext(path)[1].lower()
        try:
            if ext == '.npy':
                data = np.load(path)
            elif ext in ('.tif', '.tiff'):
                data = load_tif_as_numpy(path)
            else:
                return None

            if data.ndim == 3 and data.shape[0] <= 4 and data.shape[1] > 8:
                # 假设形状为 (bands, H, W)
                if data.shape[0] > 3:
                    data = data[:3]
                img = np.transpose(data, (1, 2, 0))
            elif data.ndim == 3:
                img = data[:, :, :3]
            else:
                img = data

            img = ((img - img.min()) / (img.ptp() + 1e-8) * 255).astype(np.uint8)
            img = np.ascontiguousarray(img)
            if img.ndim == 2:
                qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format.Format_Grayscale8).copy()
            else:
                qimg = QImage(img.data, img.shape[1], img.shape[0], img.strides[0], QImage.Format.Format_RGB888).copy()
            return QPixmap.fromImage(qimg)
        except Exception as e:
            print(f"[ROIWindow] 图像加载失败: {e}")
            return None

    def _roi_done(self, poly: Polygon | None):
        if poly is None:
            QMessageBox.warning(self, "ROI", "ROI 绘制取消或点数不足")
            return
        label, ok = QInputDialog.getInt(self, "Label", "输入 ROI 标签:", 1, 0, 255, 1)
        if ok:
            self.polygons.append((poly, label))

    def _export_mask(self):
        if not self.polygons:
            QMessageBox.warning(self, "ROI", "没有可保存的 ROI")
            return
        save_path, _ = QFileDialog.getSaveFileName(self, "保存 ROI mask", "roi_mask.npy", "NumPy File (*.npy)")
        if not save_path:
            return
        try:
            pix = self.viewer._pix_item.pixmap()
            height = pix.height()
            width = pix.width()
            shapes = [(poly, lbl) for poly, lbl in self.polygons]
            mask = rasterize(shapes, out_shape=(height, width), fill=0, dtype=np.int16)
            np.save(save_path, mask)
            self.saved_mask_path = save_path
            QMessageBox.information(self, "ROI", f"ROI mask 已保存到 {save_path}")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"保存失败: {e}")

    def closeEvent(self, event) -> None:
        super().closeEvent(event)


__all__ = ["ROIWindow"]
