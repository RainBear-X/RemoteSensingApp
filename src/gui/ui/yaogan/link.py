from PyQt5 import QtWidgets
from yaogan_ui import Ui_MainWindow

# ===== File（文件操作相关）UI类 import =====
from File.open_image_file import Ui_Form as Ui_OpenImageFile
from File.open_vector_data import Ui_Form2 as Ui_OpenVectorData
from File.save_image_as import Ui_Form3 as Ui_SaveImageAs
from File.save_vector_as import Ui_Form4 as Ui_SaveVectorAs

# ===== ImageDisplay（显示相关）UI类 import =====
from ImageDisplay.Band_extraction import Ui_Form5 as Ui_BandExtraction
from ImageDisplay.Band_synthesis import Ui_Form6 as Ui_BandSynthesis
from ImageDisplay.Histogram import Ui_Form9 as Ui_Histogram
from ImageDisplay.Projection import Ui_Form10 as Ui_Projection
from ImageDisplay.Viewing_metadata import Ui_Form13 as Ui_ViewingMetadata

# ===== ImageProcessing（处理相关）UI类 import =====
from ImageProcessing.Band_math import Ui_Form7 as Ui_BandMath
from ImageProcessing.Edge_detection import Ui_Form8 as Ui_EdgeDetection
from ImageProcessing.Sharpening import Ui_Form11 as Ui_Sharpening
from ImageProcessing.Smoothing import Ui_Form12 as Ui_Smoothing

# ===== Accuracy Evaluation（精度评估Dialogs）=====
from AccuracyEvaluation.accuracy_dialogs import (
    AccuracyReportDialog,
    ConfusionMatrixDialog,
    KappaDialog,
    OverallAccuracyDialog,
    SampleVerificationDialog
)

# ===== Model（模型Dialogs）=====
from Model.model_dialogs import (
    LoadModelDialog,
    SaveModelDialog,
    ModelValidationDialog
)

# ========== Classification（分类相关 Dialog/Widget）==========
# -- Classification Result Processing
from Classification.ClassificationResultProcessing.Custom_color_dialog import Ui_CustomColorDialog
from Classification.ClassificationResultProcessing.Denoising_dialog import Ui_DenoisingDialog
from Classification.ClassificationResultProcessing.Smooth_Processing_dialog import Ui_SmoothProcessingDialog

# -- Supervised Classification
from Classification.SupervisedClassification.Decision_Tree_dialog import Ui_DecisionTreeDialog
from Classification.SupervisedClassification.Maximum_Likelihood_dialog import Ui_MaximumLikelihoodDialog
from Classification.SupervisedClassification.Minimum_Distance_dialog import Ui_MinimumDistanceDialog
from Classification.SupervisedClassification.Random_Forest_dialog import Ui_RandomForestDialog
from Classification.SupervisedClassification.SVM_dialog import Ui_SVMDialog

# -- Unsupervised Classification
from Classification.UnsupervisedClassification.KMeans_dialog import Ui_KMeansDialog
from Classification.UnsupervisedClassification.ISODATA_dialog import Ui_ISODATADialog

# -- 其它分类功能
from Classification.Deep_Learning_Classification_dialog import Ui_DeepLearningClassificationDialog
from Classification.Generating_Classification_Report_dialog import Ui_GeneratingClassificationReportDialog
from Classification.Save_Model_As_dialog import Ui_SaveModelAsDialog

# ========== Vector（矢量与ROI相关 Dialog/Widget）==========
from Vector.CreatingVector.CreatePoint_dialog import Ui_CreatePointDialog
from Vector.CreatingVector.CreatePolyline_dialog import Ui_CreatePolylineDialog
from Vector.CreatingVector.CreatePolygon_dialog import Ui_CreatePolygonDialog
from Vector.creatingROI_dialog import Ui_CreatingROIDialog
from Vector.EditingROI_dialog import Ui_EditingROIDialog
from Vector.saveROIas_dialog import Ui_SaveROIASDialog

# ----------- 分类 Dialog 封装 -------------
class CustomColorDialog(QtWidgets.QDialog, Ui_CustomColorDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

class DenoisingDialog(QtWidgets.QDialog, Ui_DenoisingDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

class SmoothProcessingDialog(QtWidgets.QDialog, Ui_SmoothProcessingDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

class DecisionTreeDialog(QtWidgets.QDialog, Ui_DecisionTreeDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

class MaximumLikelihoodDialog(QtWidgets.QDialog, Ui_MaximumLikelihoodDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

class MinimumDistanceDialog(QtWidgets.QDialog, Ui_MinimumDistanceDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

class RandomForestDialog(QtWidgets.QDialog, Ui_RandomForestDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

class SVMDialog(QtWidgets.QDialog, Ui_SVMDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

class KMeansDialog(QtWidgets.QDialog, Ui_KMeansDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

class ISODATADialog(QtWidgets.QDialog, Ui_ISODATADialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

class DeepLearningClassificationDialog(QtWidgets.QDialog, Ui_DeepLearningClassificationDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

class SaveModelAsDialog(QtWidgets.QDialog, Ui_SaveModelAsDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

class GeneratingClassificationReportDialog(QtWidgets.QDialog, Ui_GeneratingClassificationReportDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

# ----------- 矢量/ROI Dialog 封装 -------------
class CreatingROIDialog(QtWidgets.QDialog, Ui_CreatingROIDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

class SaveROIASDialog(QtWidgets.QDialog, Ui_SaveROIASDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

class EditingROIDialog(QtWidgets.QDialog, Ui_EditingROIDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

class CreatePointDialog(QtWidgets.QDialog, Ui_CreatePointDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

class CreatePolylineDialog(QtWidgets.QDialog, Ui_CreatePolylineDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

class CreatePolygonDialog(QtWidgets.QDialog, Ui_CreatePolygonDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)

# ----------- 普通功能窗口类 -------------
class OpenFileDialog(QtWidgets.QWidget, Ui_OpenImageFile):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton_2.clicked.connect(self.close)

class OpenFileDialog2(QtWidgets.QWidget, Ui_OpenVectorData):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton_2.clicked.connect(self.close)

class OpenFileDialog3(QtWidgets.QWidget, Ui_SaveImageAs):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton_2.clicked.connect(self.close)

class OpenFileDialog4(QtWidgets.QWidget, Ui_SaveVectorAs):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.pushButton_2.clicked.connect(self.close)

class displayDialog(QtWidgets.QWidget, Ui_BandExtraction):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

class displayDialog2(QtWidgets.QWidget, Ui_BandSynthesis):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

class processsingDialog1(QtWidgets.QWidget, Ui_BandMath):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

class processsingDialog2(QtWidgets.QWidget, Ui_EdgeDetection):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

class displayDialog3(QtWidgets.QWidget, Ui_Histogram):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

class displayDialog4(QtWidgets.QWidget, Ui_Projection):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

class processsingDialog3(QtWidgets.QWidget, Ui_Sharpening):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

class processsingDialog4(QtWidgets.QWidget, Ui_Smoothing):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

class displayDialog5(QtWidgets.QWidget, Ui_ViewingMetadata):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

# ----------- 主窗口 -------------
class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # ========== 图像与矢量等功能 ==========
        self.actionOpenImageFile.triggered.connect(self.show_open_image_dialog)
        self.actionOpenVectorData.triggered.connect(self.show_open_image_dialog2)
        self.actionSaveImageFileAs.triggered.connect(self.show_open_image_dialog3)
        self.actionSaveVectorFileAs.triggered.connect(self.show_open_image_dialog4)
        self.actionBandextraction.triggered.connect(self.image_display)
        self.actionBandsynthesis.triggered.connect(self.image_display2)
        self.actionBandMath.triggered.connect(self.image_processing1)
        self.actionEdgedetection.triggered.connect(self.image_processing2)
        self.actionHistogram.triggered.connect(self.image_display3)
        self.actionProjection.triggered.connect(self.image_display4)
        self.actionSharpening.triggered.connect(self.image_processing3)
        self.actionSmoothing.triggered.connect(self.image_processing4)
        self.actionviewingmetadata.triggered.connect(self.image_display5)

        # ========== 精度评估菜单 ==========
        self.actionConfusion_Matrix.triggered.connect(self.show_confusion_matrix_dialog)
        self.actionOverall_Accuracy.triggered.connect(self.show_overall_accuracy_dialog)
        self.actionKappa.triggered.connect(self.show_kappa_dialog)
        self.actionVerify_Sample_Accuracy_Test.triggered.connect(self.show_sample_verification_dialog)
        self.actionGenerate_Accuracy_Evaluation_Table.triggered.connect(self.show_accuracy_report_dialog)

        # ========== 模型菜单 ==========
        self.actionLoad_Model.triggered.connect(self.show_load_model_dialog)
        self.actionSave_Model.triggered.connect(self.show_save_model_dialog)
        self.actionModel_Validation.triggered.connect(self.show_model_validation_dialog)

        # ========== 分类相关 ==========
        self.actionCustom_Color.triggered.connect(lambda: self.show_classification_dialog(CustomColorDialog))
        self.actionDenoising.triggered.connect(lambda: self.show_classification_dialog(DenoisingDialog))
        self.actionSmooth_Processing.triggered.connect(lambda: self.show_classification_dialog(SmoothProcessingDialog))

        self.actionDecision_Tree.triggered.connect(lambda: self.show_classification_dialog(DecisionTreeDialog))
        self.actionMaximum_Likelihood.triggered.connect(lambda: self.show_classification_dialog(MaximumLikelihoodDialog))
        self.actionMinimum_Distance.triggered.connect(lambda: self.show_classification_dialog(MinimumDistanceDialog))
        self.actionRandom_Forest.triggered.connect(lambda: self.show_classification_dialog(RandomForestDialog))
        self.actionSVM.triggered.connect(lambda: self.show_classification_dialog(SVMDialog))

        self.actionK_means.triggered.connect(lambda: self.show_classification_dialog(KMeansDialog))
        self.actionISODATA.triggered.connect(lambda: self.show_classification_dialog(ISODATADialog))

        self.actionDeep_leraning_Classification.triggered.connect(lambda: self.show_classification_dialog(DeepLearningClassificationDialog))
        self.actionSave_Model_As.triggered.connect(lambda: self.show_classification_dialog(SaveModelAsDialog))
        self.actionGenerating.triggered.connect(lambda: self.show_classification_dialog(GeneratingClassificationReportDialog))

        # ========== 矢量与ROI相关 ==========
        self.actionCreatingROI.triggered.connect(lambda: self.show_vector_dialog(CreatingROIDialog))
        self.actionSaveROIAs.triggered.connect(lambda: self.show_vector_dialog(SaveROIASDialog))
        self.actionEditingROI.triggered.connect(lambda: self.show_vector_dialog(EditingROIDialog))
        self.actionPoint.triggered.connect(lambda: self.show_vector_dialog(CreatePointDialog))
        self.actionPolyline.triggered.connect(lambda: self.show_vector_dialog(CreatePolylineDialog))
        self.actionPolygon.triggered.connect(lambda: self.show_vector_dialog(CreatePolygonDialog))

    # ========== 图像与矢量等功能 ==========
    def show_open_image_dialog(self): self.open_file_dialog = OpenFileDialog(); self.open_file_dialog.show()
    def show_open_image_dialog2(self): self.open_file_dialog = OpenFileDialog2(); self.open_file_dialog.show()
    def show_open_image_dialog3(self): self.open_file_dialog = OpenFileDialog3(); self.open_file_dialog.show()
    def show_open_image_dialog4(self): self.open_file_dialog = OpenFileDialog4(); self.open_file_dialog.show()
    def image_display(self): self.display_dialog = displayDialog(); self.display_dialog.show()
    def image_display2(self): self.display_dialog = displayDialog2(); self.display_dialog.show()
    def image_processing1(self): self.display_dialog = processsingDialog1(); self.display_dialog.show()
    def image_processing2(self): self.display_dialog = processsingDialog2(); self.display_dialog.show()
    def image_display3(self): self.display_dialog = displayDialog3(); self.display_dialog.show()
    def image_display4(self): self.display_dialog = displayDialog4(); self.display_dialog.show()
    def image_processing3(self): self.display_dialog = processsingDialog3(); self.display_dialog.show()
    def image_processing4(self): self.display_dialog = processsingDialog4(); self.display_dialog.show()
    def image_display5(self): self.display_dialog = displayDialog5(); self.display_dialog.show()

    # ========== 精度评估相关 ==========
    def show_confusion_matrix_dialog(self):
        dlg = ConfusionMatrixDialog(self)
        dlg.exec_()
    def show_overall_accuracy_dialog(self):
        dlg = OverallAccuracyDialog(self)
        dlg.exec_()
    def show_kappa_dialog(self):
        dlg = KappaDialog(self)
        dlg.exec_()
    def show_sample_verification_dialog(self):
        dlg = SampleVerificationDialog(self)
        dlg.exec_()
    def show_accuracy_report_dialog(self):
        dlg = AccuracyReportDialog(self)
        dlg.exec_()

    # ========== 模型相关 ==========
    def show_load_model_dialog(self):
        dlg = LoadModelDialog(self)
        dlg.exec_()
    def show_save_model_dialog(self):
        dlg = SaveModelDialog(self)
        dlg.exec_()
    def show_model_validation_dialog(self):
        dlg = ModelValidationDialog(self)
        dlg.exec_()

    # ========== 分类相关 ==========
    def show_classification_dialog(self, DialogClass):
        dlg = DialogClass(self)
        dlg.exec_()

    # ========== 矢量与ROI相关 ==========
    def show_vector_dialog(self, DialogClass):
        dlg = DialogClass(self)
        dlg.exec_()

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())