from PyQt5 import QtWidgets
from .AccuracyReportDialog_ui import Ui_AccuracyReportDialog
from .ConfusionMatrixDialog_ui import Ui_ConfusionMatrixDialog
from .KappaDialog_ui import Ui_KappaDialog
from .OverallAccuracyDialog_ui import Ui_OverallAccuracyDialog
from .SampleVerificationDialog_ui import Ui_SampleVerifiactionDialog

class AccuracyReportDialog(QtWidgets.QDialog, Ui_AccuracyReportDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.btn_browse_outputfile.clicked.connect(self.on_browse_outputfile)

    def on_browse_outputfile(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Select Output File", "", "Excel Files (*.xlsx);;CSV Files (*.csv);;Text Files (*.txt)"
        )
        if path:
            self.lineEdit_outputfile.setText(path)


class ConfusionMatrixDialog(QtWidgets.QDialog, Ui_ConfusionMatrixDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.btn_browse_result.clicked.connect(self.on_browse_result)
        self.btn_browse_reference.clicked.connect(self.on_browse_reference)
        self.btn_browse_mask.clicked.connect(self.on_browse_mask)
        self.btn_browse_outputdir.clicked.connect(self.on_browse_outputdir)

    def on_browse_result(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Classification Result File")
        if path:
            self.lineEdit_result.setText(path)

    def on_browse_reference(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Reference Data File")
        if path:
            self.lineEdit_reference.setText(path)

    def on_browse_mask(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Mask File")
        if path:
            self.lineEdit_mask.setText(path)

    def on_browse_outputdir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.lineEdit_outputdir.setText(path)


class KappaDialog(QtWidgets.QDialog, Ui_KappaDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.btn_browse_result.clicked.connect(self.on_browse_result)
        self.btn_browse_reference.clicked.connect(self.on_browse_reference)
        self.btn_browse_outputdir.clicked.connect(self.on_browse_outputdir)

    def on_browse_result(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Classification Result File")
        if path:
            self.lineEdit_result.setText(path)

    def on_browse_reference(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Reference Data File")
        if path:
            self.lineEdit_reference.setText(path)

    def on_browse_outputdir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.lineEdit_outputdir.setText(path)


class OverallAccuracyDialog(QtWidgets.QDialog, Ui_OverallAccuracyDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.btn_browse_result.clicked.connect(self.on_browse_result)
        self.btn_browse_reference.clicked.connect(self.on_browse_reference)
        self.btn_browse_outputdir.clicked.connect(self.on_browse_outputdir)

    def on_browse_result(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Classification Result File")
        if path:
            self.lineEdit_result.setText(path)

    def on_browse_reference(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Reference Data File")
        if path:
            self.lineEdit_reference.setText(path)

    def on_browse_outputdir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.lineEdit_outputdir.setText(path)


class SampleVerificationDialog(QtWidgets.QDialog, Ui_SampleVerifiactionDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
        self.btn_browse_sample.clicked.connect(self.on_browse_sample)
        self.btn_browse_outputdir.clicked.connect(self.on_browse_outputdir)

    def on_browse_sample(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Sample File")
        if path:
            self.lineEdit_sample.setText(path)

    def on_browse_outputdir(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if path:
            self.lineEdit_outputdir.setText(path)