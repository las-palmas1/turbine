# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window_sdi.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1650, 600)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        MainWindow.setMinimumSize(QtCore.QSize(1650, 0))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1650, 26))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.act_new = QtWidgets.QAction(MainWindow)
        self.act_new.setObjectName("act_new")
        self.act_open = QtWidgets.QAction(MainWindow)
        self.act_open.setObjectName("act_open")
        self.act_save = QtWidgets.QAction(MainWindow)
        self.act_save.setObjectName("act_save")
        self.act_save_as = QtWidgets.QAction(MainWindow)
        self.act_save_as.setObjectName("act_save_as")
        self.act_exit = QtWidgets.QAction(MainWindow)
        self.act_exit.setObjectName("act_exit")
        self.menu.addAction(self.act_new)
        self.menu.addAction(self.act_open)
        self.menu.addSeparator()
        self.menu.addAction(self.act_save)
        self.menu.addAction(self.act_save_as)
        self.menu.addSeparator()
        self.menu.addAction(self.act_exit)
        self.menubar.addAction(self.menu.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.menu.setTitle(_translate("MainWindow", "Файл"))
        self.act_new.setText(_translate("MainWindow", "Новый"))
        self.act_open.setText(_translate("MainWindow", "Открыть"))
        self.act_save.setText(_translate("MainWindow", "Сохранить"))
        self.act_save_as.setText(_translate("MainWindow", "Сохранить как..."))
        self.act_exit.setText(_translate("MainWindow", "Выход"))

