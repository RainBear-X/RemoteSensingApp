# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'c:\Users\11259\Desktop\ui\Maximum Likelihood_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.15.11
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MaximumLikelihoodDialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(277, 164)
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(20, 20, 111, 16))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.lineEdit_traingsample = QtWidgets.QLineEdit(Dialog)
        self.lineEdit_traingsample.setGeometry(QtCore.QRect(130, 20, 71, 21))
        self.lineEdit_traingsample.setObjectName("lineEdit_traingsample")
        self.button_browse = QtWidgets.QPushButton(Dialog)
        self.button_browse.setGeometry(QtCore.QRect(210, 30, 41, 16))
        self.button_browse.setObjectName("button_browse")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(20, 60, 91, 16))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.comboBox = QtWidgets.QComboBox(Dialog)
        self.comboBox.setGeometry(QtCore.QRect(130, 60, 60, 22))
        self.comboBox.setEditable(True)
        self.comboBox.setCurrentText("Full")
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.checkBox = QtWidgets.QCheckBox(Dialog)
        self.checkBox.setGeometry(QtCore.QRect(20, 100, 111, 16))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(11)
        self.checkBox.setFont(font)
        self.checkBox.setObjectName("checkBox")
        self.Run = QtWidgets.QPushButton(Dialog)
        self.Run.setGeometry(QtCore.QRect(120, 130, 61, 17))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        self.Run.setFont(font)
        self.Run.setObjectName("Run")
        self.Cancel = QtWidgets.QPushButton(Dialog)
        self.Cancel.setGeometry(QtCore.QRect(190, 130, 61, 17))
        font = QtGui.QFont()
        font.setFamily("Times New Roman")
        font.setPointSize(9)
        self.Cancel.setFont(font)
        self.Cancel.setObjectName("Cancel")

        self.retranslateUi(Dialog)
        self.comboBox.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Maximum Likelihood"))
        self.label.setText(_translate("Dialog", "Training Sample File："))
        self.button_browse.setText(_translate("Dialog", "Browse"))
        self.label_2.setText(_translate("Dialog", "Covariance Type"))
        self.comboBox.setItemText(0, _translate("Dialog", "Full"))
        self.comboBox.setItemText(1, _translate("Dialog", "Diagonal"))
        self.checkBox.setText(_translate("Dialog", "Use Prior Probability"))
        self.Run.setText(_translate("Dialog", "Run"))
        self.Cancel.setText(_translate("Dialog", "Cancel"))
