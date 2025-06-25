from PyQt5 import QtWidgets
from .LoadModelDialog_ui import Ui_LoadModelDialog
from .ModelValidationDialog_ui import Ui_ModelValidationDialog
from .SaveModelDialog_ui import Ui_SaveModelDialog

class LoadModelDialog(QtWidgets.QDialog, Ui_LoadModelDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Load Model")
        self.btn_browse_file.clicked.connect(self.on_browse_file)

    def on_browse_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Model File", "", "Model Files (*.pth *.h5 *.pkl);;All Files (*)")
        if path:
            self.lineEdit_file.setText(path)

class ModelValidationDialog(QtWidgets.QDialog, Ui_ModelValidationDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Model Validation")
        self.btn_browse_dataset.clicked.connect(self.on_browse_dataset)
        self.btn_browse_output.clicked.connect(self.on_browse_output)

    def on_browse_dataset(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Validation Dataset", "", "Data Files (*.csv *.xlsx *.npy *.npz);;All Files (*)")
        if path:
            self.lineEdit_dataset.setText(path)

    def on_browse_output(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.lineEdit_output.setText(path)

class SaveModelDialog(QtWidgets.QDialog, Ui_SaveModelDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.setWindowTitle("Save Model")
        self.btn_browse_save.clicked.connect(self.on_browse_save)

    def on_browse_save(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select Save File Path", "", "Model Files (*.pth *.h5 *.pkl);;All Files (*)")
        if path:
            self.lineEdit_save.setText(path)