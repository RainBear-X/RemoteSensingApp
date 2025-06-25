import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5 import QtCore, QtGui, QtWidgets
from yaogan import Ui_MainWindow
from open_image_file import Ui_Form

class RemoteSensingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 菜单-文件（File) 信号绑定
        # 信号连接区域 - 新增OpenImageFile连接
        self.ui.actionOpenImageFile.triggered.connect(self.open_image_file)
        self.ui.actionOpenVectorData.triggered.connect(self.open_vector_data)
        self.ui.actionSaveImageFileAs.triggered.connect(self.save_image_file_as)
        self.ui.actionSaveVectorFileAs.triggered.connect(self.save_vector_file_as)
        self.ui.actionExit.triggered.connect(self.close_app)

        # 菜单-影像显示(Image display) 信号绑定
        self.ui.actionBandextraction.triggered.connect(self.band_extraction)
        self.ui.actionviewingmetadata.triggered.connect(self.viewing_metadata)
        self.ui.actionBandsynthesis.triggered.connect(self.band_synthesis)
        self.ui.actionImageCutting.triggered.connect(self.image_cutting)
        self.ui.actionProjection.triggered.connect(self.projection)
        self.ui.actionHistogram.triggered.connect(self.histogram)
        self.ui.actionSpectral_characteristics.triggered.connect(self.spectral_characteristics)

        # 菜单-影像处理(Image processing) 信号绑定
        self.ui.actionImagestretching.triggered.connect(self.image_stretching)
        self.ui.actionEqualize.triggered.connect(self.equalize)
        self.ui.actionSmoothing.triggered.connect(self.smoothing)
        self.ui.actionSharpening.triggered.connect(self.sharpening)
        self.ui.actionEdgedetection.triggered.connect(self.edge_detection)
        self.ui.actionBandMath.triggered.connect(self.band_math)

        # 菜单-ROI(Vector) 信号绑定
        self.ui.actionCreatingROI.triggered.connect(self.creating_roi)
        self.ui.actionSaveROIAs.triggered.connect(self.save_roi_as)
        self.ui.actionEditingROI.triggered.connect(self.editing_roi)
        self.ui.actionPoint.triggered.connect(self.create_point)
        self.ui.actionPolyline.triggered.connect(self.create_polyline)
        self.ui.actionPolygon.triggered.connect(self.create_polygon)

        # 分类（Classification）信号绑定
        self.ui.actionMaximum_Likelihood.triggered.connect(self.maximum_likelihood)
        self.ui.actionMinimum_Distance.triggered.connect(self.minimum_distance)
        self.ui.actionSVM.triggered.connect(self.svm_classification)
        self.ui.actionDecision_Tree.triggered.connect(self.decision_tree)
        self.ui.actionRandom_Forest.triggered.connect(self.random_forest)
        self.ui.actionK_means.triggered.connect(self.k_means)
        self.ui.actionISODATA.triggered.connect(self.isodata)
        self.ui.actionDeep_leraning_Classification.triggered.connect(self.deep_learning_classification)
        self.ui.actionSave_Model_As.triggered.connect(self.save_model_as)
        self.ui.actionGenerating.triggered.connect(self.generate_classification_report)
        self.ui.actionCustom_Color.triggered.connect(self.custom_color)
        self.ui.actionSmooth_Processing.triggered.connect(self.smooth_processing)
        self.ui.actionDenoising.triggered.connect(self.denoising)

        # 精度评价（Accuracy Evaluation）信号绑定
        self.ui.actionConfusion_Matrix.triggered.connect(self.confusion_matrix)
        self.ui.actionOverall_Accuracy.triggered.connect(self.overall_accuracy)
        self.ui.actionKappa.triggered.connect(self.kappa_evaluation)
        self.ui.actionVerify_Sample_Accuracy_Test.triggered.connect(self.verify_sample_accuracy_test)
        self.ui.actionGenerate_Accuracy_Evaluation_Table.triggered.connect(self.generate_accuracy_table)

        # 模型（Model）模块（暂未定义具体Action，预留接口）

    # 菜单-文件（File) 接口槽函数
    def open_image_file(self):
        """打开图像文件子窗体"""
        self.sub_window  = QtWidgets.QWidget()
        self.sub_ui  = Ui_Form()
        self.sub_ui.setupUi(self.sub_window)
        self.sub_window.setWindowTitle(" 图像文件浏览器")
    def open_vector_data(self): pass
    def save_image_file_as(self): pass
    def save_vector_file_as(self): pass
    def close_app(self): self.close()

    # 菜单-影像显示(Image display) 接口槽函数
    def band_extraction(self): pass
    def viewing_metadata(self): pass
    def band_synthesis(self): pass
    def image_cutting(self): pass
    def projection(self): pass
    def histogram(self): pass
    def spectral_characteristics(self): pass

    # 菜单-影像处理(Image processing) 接口槽函数
    def image_stretching(self): pass
    def equalize(self): pass
    def smoothing(self): pass
    def sharpening(self): pass
    def edge_detection(self): pass
    def band_math(self): pass

    # 菜单-ROI(Vector) 接口槽函数
    def creating_roi(self): pass
    def save_roi_as(self): pass
    def editing_roi(self): pass
    def create_point(self): pass
    def create_polyline(self): pass
    def create_polygon(self): pass

    # 分类相关槽函数
    def maximum_likelihood(self): pass

    def minimum_distance(self): pass

    def svm_classification(self): pass

    def decision_tree(self): pass

    def random_forest(self): pass

    def k_means(self): pass

    def isodata(self): pass

    def deep_learning_classification(self): pass

    def save_model_as(self): pass

    def generate_classification_report(self): pass

    def custom_color(self): pass

    def smooth_processing(self): pass

    def denoising(self): pass

    # 精度评价相关槽函数
    def confusion_matrix(self): pass

    def overall_accuracy(self): pass

    def kappa_evaluation(self): pass

    def verify_sample_accuracy_test(self): pass

    def generate_accuracy_table(self): pass

    # 模型模块预留函数（可按需添加）

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RemoteSensingApp()
    window.show()
    sys.exit(app.exec_())
