# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:\Users\User\Documents\study\turbine\gui\average_streamline\save_par_file_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(395, 146)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.rad_btn_compass = QtWidgets.QRadioButton(Dialog)
        self.rad_btn_compass.setChecked(True)
        self.rad_btn_compass.setObjectName("rad_btn_compass")
        self.verticalLayout.addWidget(self.rad_btn_compass)
        self.rad_btn_nx = QtWidgets.QRadioButton(Dialog)
        self.rad_btn_nx.setObjectName("rad_btn_nx")
        self.verticalLayout.addWidget(self.rad_btn_nx)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.prefix = QtWidgets.QLineEdit(Dialog)
        self.prefix.setObjectName("prefix")
        self.horizontalLayout.addWidget(self.prefix)
        self.verticalLayout.addLayout(self.horizontalLayout)
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.filename = QtWidgets.QLineEdit(Dialog)
        self.filename.setReadOnly(False)
        self.filename.setObjectName("filename")
        self.horizontalLayout_2.addWidget(self.filename)
        self.overview_btn = QtWidgets.QPushButton(Dialog)
        self.overview_btn.setObjectName("overview_btn")
        self.horizontalLayout_2.addWidget(self.overview_btn)
        self.verticalLayout.addLayout(self.horizontalLayout_2)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.verticalLayout.addWidget(self.buttonBox)
        self.buttonBox.raise_()
        self.rad_btn_compass.raise_()
        self.rad_btn_nx.raise_()
        self.label.raise_()

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Сохранить файл параметров"))
        self.rad_btn_compass.setText(_translate("Dialog", "КОМПАС"))
        self.rad_btn_nx.setText(_translate("Dialog", "NX"))
        self.label.setText(_translate("Dialog", "Префикс:"))
        self.prefix.setText(_translate("Dialog", "turb"))
        self.label_2.setText(_translate("Dialog", "Имя файла:"))
        self.filename.setText(_translate("Dialog", "params"))
        self.overview_btn.setText(_translate("Dialog", "Обзор"))

