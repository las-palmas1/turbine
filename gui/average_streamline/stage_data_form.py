# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'stage_data_widget.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1528, 675)
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame_3 = QtWidgets.QFrame(Form)
        self.frame_3.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame_3)
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_stage_title = QtWidgets.QLabel(self.frame_3)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_stage_title.setFont(font)
        self.label_stage_title.setAlignment(QtCore.Qt.AlignCenter)
        self.label_stage_title.setObjectName("label_stage_title")
        self.verticalLayout.addWidget(self.label_stage_title)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.frame_5 = QtWidgets.QFrame(self.frame_3)
        self.frame_5.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_5.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_5.setObjectName("frame_5")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.frame_5)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_32 = QtWidgets.QLabel(self.frame_5)
        self.label_32.setEnabled(True)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_32.setFont(font)
        self.label_32.setAlignment(QtCore.Qt.AlignCenter)
        self.label_32.setObjectName("label_32")
        self.verticalLayout_2.addWidget(self.label_32)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.l1_b_sa_ratio = QtWidgets.QDoubleSpinBox(self.frame_5)
        self.l1_b_sa_ratio.setDecimals(3)
        self.l1_b_sa_ratio.setSingleStep(0.1)
        self.l1_b_sa_ratio.setProperty("value", 1.6)
        self.l1_b_sa_ratio.setObjectName("l1_b_sa_ratio")
        self.gridLayout.addWidget(self.l1_b_sa_ratio, 3, 1, 1, 1)
        self.epsilon = QtWidgets.QDoubleSpinBox(self.frame_5)
        self.epsilon.setMaximum(5.0)
        self.epsilon.setSingleStep(0.01)
        self.epsilon.setProperty("value", 1.0)
        self.epsilon.setObjectName("epsilon")
        self.gridLayout.addWidget(self.epsilon, 10, 1, 1, 1)
        self.label_46 = QtWidgets.QLabel(self.frame_5)
        self.label_46.setObjectName("label_46")
        self.gridLayout.addWidget(self.label_46, 10, 0, 1, 1)
        self.mu = QtWidgets.QDoubleSpinBox(self.frame_5)
        self.mu.setMaximum(1.0)
        self.mu.setSingleStep(0.01)
        self.mu.setProperty("value", 1.0)
        self.mu.setObjectName("mu")
        self.gridLayout.addWidget(self.mu, 11, 1, 1, 1)
        self.psi = QtWidgets.QDoubleSpinBox(self.frame_5)
        self.psi.setDecimals(3)
        self.psi.setMaximum(1.0)
        self.psi.setSingleStep(0.01)
        self.psi.setProperty("value", 0.97)
        self.psi.setObjectName("psi")
        self.gridLayout.addWidget(self.psi, 9, 1, 1, 1)
        self.label_50 = QtWidgets.QLabel(self.frame_5)
        self.label_50.setObjectName("label_50")
        self.gridLayout.addWidget(self.label_50, 14, 0, 1, 1)
        self.label_44 = QtWidgets.QLabel(self.frame_5)
        self.label_44.setObjectName("label_44")
        self.gridLayout.addWidget(self.label_44, 8, 0, 1, 1)
        self.label_43 = QtWidgets.QLabel(self.frame_5)
        self.label_43.setObjectName("label_43")
        self.gridLayout.addWidget(self.label_43, 7, 0, 1, 1)
        self.label_47 = QtWidgets.QLabel(self.frame_5)
        self.label_47.setObjectName("label_47")
        self.gridLayout.addWidget(self.label_47, 11, 0, 1, 1)
        self.delta_r_rel = QtWidgets.QDoubleSpinBox(self.frame_5)
        self.delta_r_rel.setMaximum(0.3)
        self.delta_r_rel.setSingleStep(0.01)
        self.delta_r_rel.setProperty("value", 0.01)
        self.delta_r_rel.setObjectName("delta_r_rel")
        self.gridLayout.addWidget(self.delta_r_rel, 7, 1, 1, 1)
        self.g_lb = QtWidgets.QDoubleSpinBox(self.frame_5)
        self.g_lb.setDecimals(3)
        self.g_lb.setMaximum(1.0)
        self.g_lb.setSingleStep(0.01)
        self.g_lb.setObjectName("g_lb")
        self.gridLayout.addWidget(self.g_lb, 14, 1, 1, 1)
        self.phi = QtWidgets.QDoubleSpinBox(self.frame_5)
        self.phi.setDecimals(3)
        self.phi.setMaximum(1.0)
        self.phi.setSingleStep(0.01)
        self.phi.setProperty("value", 0.97)
        self.phi.setObjectName("phi")
        self.gridLayout.addWidget(self.phi, 8, 1, 1, 1)
        self.l2_b_rk_ratio = QtWidgets.QDoubleSpinBox(self.frame_5)
        self.l2_b_rk_ratio.setDecimals(3)
        self.l2_b_rk_ratio.setSingleStep(0.1)
        self.l2_b_rk_ratio.setProperty("value", 1.8)
        self.l2_b_rk_ratio.setObjectName("l2_b_rk_ratio")
        self.gridLayout.addWidget(self.l2_b_rk_ratio, 2, 1, 1, 1)
        self.label_45 = QtWidgets.QLabel(self.frame_5)
        self.label_45.setObjectName("label_45")
        self.gridLayout.addWidget(self.label_45, 9, 0, 1, 1)
        self.label_41 = QtWidgets.QLabel(self.frame_5)
        self.label_41.setObjectName("label_41")
        self.gridLayout.addWidget(self.label_41, 5, 0, 1, 1)
        self.label_40 = QtWidgets.QLabel(self.frame_5)
        self.label_40.setObjectName("label_40")
        self.gridLayout.addWidget(self.label_40, 4, 0, 1, 1)
        self.g_lk = QtWidgets.QDoubleSpinBox(self.frame_5)
        self.g_lk.setDecimals(3)
        self.g_lk.setMaximum(1.0)
        self.g_lk.setSingleStep(0.01)
        self.g_lk.setObjectName("g_lk")
        self.gridLayout.addWidget(self.g_lk, 12, 1, 1, 1)
        self.label_35 = QtWidgets.QLabel(self.frame_5)
        self.label_35.setObjectName("label_35")
        self.gridLayout.addWidget(self.label_35, 2, 0, 1, 1)
        self.label_49 = QtWidgets.QLabel(self.frame_5)
        self.label_49.setObjectName("label_49")
        self.gridLayout.addWidget(self.label_49, 13, 0, 1, 1)
        self.label_36 = QtWidgets.QLabel(self.frame_5)
        self.label_36.setObjectName("label_36")
        self.gridLayout.addWidget(self.label_36, 3, 0, 1, 1)
        self.delta_a_b_rk_ratio = QtWidgets.QDoubleSpinBox(self.frame_5)
        self.delta_a_b_rk_ratio.setDecimals(3)
        self.delta_a_b_rk_ratio.setMaximum(10.0)
        self.delta_a_b_rk_ratio.setSingleStep(0.1)
        self.delta_a_b_rk_ratio.setProperty("value", 0.22)
        self.delta_a_b_rk_ratio.setObjectName("delta_a_b_rk_ratio")
        self.gridLayout.addWidget(self.delta_a_b_rk_ratio, 5, 1, 1, 1)
        self.label_48 = QtWidgets.QLabel(self.frame_5)
        self.label_48.setObjectName("label_48")
        self.gridLayout.addWidget(self.label_48, 12, 0, 1, 1)
        self.g_ld = QtWidgets.QDoubleSpinBox(self.frame_5)
        self.g_ld.setDecimals(3)
        self.g_ld.setMaximum(1.0)
        self.g_ld.setSingleStep(0.01)
        self.g_ld.setObjectName("g_ld")
        self.gridLayout.addWidget(self.g_ld, 13, 1, 1, 1)
        self.delta_a_b_sa_ratio = QtWidgets.QDoubleSpinBox(self.frame_5)
        self.delta_a_b_sa_ratio.setDecimals(3)
        self.delta_a_b_sa_ratio.setMaximum(10.0)
        self.delta_a_b_sa_ratio.setSingleStep(0.1)
        self.delta_a_b_sa_ratio.setProperty("value", 0.22)
        self.delta_a_b_sa_ratio.setObjectName("delta_a_b_sa_ratio")
        self.gridLayout.addWidget(self.delta_a_b_sa_ratio, 4, 1, 1, 1)
        self.label_rho = QtWidgets.QLabel(self.frame_5)
        self.label_rho.setObjectName("label_rho")
        self.gridLayout.addWidget(self.label_rho, 0, 0, 1, 1)
        self.rho = QtWidgets.QDoubleSpinBox(self.frame_5)
        self.rho.setEnabled(True)
        self.rho.setReadOnly(False)
        self.rho.setDecimals(3)
        self.rho.setMaximum(1.0)
        self.rho.setSingleStep(0.05)
        self.rho.setProperty("value", 0.4)
        self.rho.setObjectName("rho")
        self.gridLayout.addWidget(self.rho, 0, 1, 1, 1)
        self.label_H0 = QtWidgets.QLabel(self.frame_5)
        self.label_H0.setObjectName("label_H0")
        self.gridLayout.addWidget(self.label_H0, 1, 0, 1, 1)
        self.H0 = QtWidgets.QDoubleSpinBox(self.frame_5)
        self.H0.setMaximum(5000.0)
        self.H0.setSingleStep(10.0)
        self.H0.setProperty("value", 440.0)
        self.H0.setObjectName("H0")
        self.gridLayout.addWidget(self.H0, 1, 1, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_2.addItem(spacerItem)
        self.horizontalLayout_4.addWidget(self.frame_5)
        self.frame_6 = QtWidgets.QFrame(self.frame_3)
        self.frame_6.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_6.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_6.setObjectName("frame_6")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.frame_6)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.label_33 = QtWidgets.QLabel(self.frame_6)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_33.setFont(font)
        self.label_33.setAlignment(QtCore.Qt.AlignCenter)
        self.label_33.setObjectName("label_33")
        self.verticalLayout_7.addWidget(self.label_33)
        self.gridLayout_5 = QtWidgets.QGridLayout()
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.l0 = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.l0.setReadOnly(True)
        self.l0.setDecimals(4)
        self.l0.setMaximum(10000.0)
        self.l0.setObjectName("l0")
        self.gridLayout_5.addWidget(self.l0, 5, 1, 1, 1)
        self.label_55 = QtWidgets.QLabel(self.frame_6)
        self.label_55.setObjectName("label_55")
        self.gridLayout_5.addWidget(self.label_55, 0, 2, 1, 1)
        self.label_51 = QtWidgets.QLabel(self.frame_6)
        self.label_51.setObjectName("label_51")
        self.gridLayout_5.addWidget(self.label_51, 2, 0, 1, 1)
        self.label_37 = QtWidgets.QLabel(self.frame_6)
        self.label_37.setObjectName("label_37")
        self.gridLayout_5.addWidget(self.label_37, 0, 0, 1, 1)
        self.gamma_sum = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.gamma_sum.setReadOnly(True)
        self.gamma_sum.setMinimum(-100.0)
        self.gamma_sum.setMaximum(10000.0)
        self.gamma_sum.setObjectName("gamma_sum")
        self.gridLayout_5.addWidget(self.gamma_sum, 0, 1, 1, 1)
        self.label_38 = QtWidgets.QLabel(self.frame_6)
        self.label_38.setObjectName("label_38")
        self.gridLayout_5.addWidget(self.label_38, 1, 0, 1, 1)
        self.label_52 = QtWidgets.QLabel(self.frame_6)
        self.label_52.setObjectName("label_52")
        self.gridLayout_5.addWidget(self.label_52, 3, 0, 1, 1)
        self.D0 = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.D0.setReadOnly(True)
        self.D0.setDecimals(4)
        self.D0.setMaximum(10000.0)
        self.D0.setObjectName("D0")
        self.gridLayout_5.addWidget(self.D0, 4, 1, 1, 1)
        self.label_l0 = QtWidgets.QLabel(self.frame_6)
        self.label_l0.setObjectName("label_l0")
        self.gridLayout_5.addWidget(self.label_l0, 5, 0, 1, 1)
        self.label_53 = QtWidgets.QLabel(self.frame_6)
        self.label_53.setObjectName("label_53")
        self.gridLayout_5.addWidget(self.label_53, 4, 0, 1, 1)
        self.gamma_in = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.gamma_in.setReadOnly(True)
        self.gamma_in.setMinimum(-100.0)
        self.gamma_in.setMaximum(10000.0)
        self.gamma_in.setObjectName("gamma_in")
        self.gridLayout_5.addWidget(self.gamma_in, 2, 1, 1, 1)
        self.gamma_av = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.gamma_av.setReadOnly(True)
        self.gamma_av.setMinimum(-100.0)
        self.gamma_av.setMaximum(10000.0)
        self.gamma_av.setObjectName("gamma_av")
        self.gridLayout_5.addWidget(self.gamma_av, 1, 1, 1, 1)
        self.gamma_out = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.gamma_out.setReadOnly(True)
        self.gamma_out.setMinimum(-100.0)
        self.gamma_out.setMaximum(10000.0)
        self.gamma_out.setObjectName("gamma_out")
        self.gridLayout_5.addWidget(self.gamma_out, 3, 1, 1, 1)
        self.D1 = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.D1.setReadOnly(True)
        self.D1.setDecimals(4)
        self.D1.setMaximum(10000.0)
        self.D1.setObjectName("D1")
        self.gridLayout_5.addWidget(self.D1, 0, 3, 1, 1)
        self.label_56 = QtWidgets.QLabel(self.frame_6)
        self.label_56.setObjectName("label_56")
        self.gridLayout_5.addWidget(self.label_56, 1, 2, 1, 1)
        self.l1 = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.l1.setReadOnly(True)
        self.l1.setDecimals(4)
        self.l1.setMaximum(10000.0)
        self.l1.setObjectName("l1")
        self.gridLayout_5.addWidget(self.l1, 1, 3, 1, 1)
        self.label_57 = QtWidgets.QLabel(self.frame_6)
        self.label_57.setObjectName("label_57")
        self.gridLayout_5.addWidget(self.label_57, 2, 2, 1, 1)
        self.D2 = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.D2.setReadOnly(True)
        self.D2.setDecimals(4)
        self.D2.setMaximum(10000.0)
        self.D2.setObjectName("D2")
        self.gridLayout_5.addWidget(self.D2, 2, 3, 1, 1)
        self.label_58 = QtWidgets.QLabel(self.frame_6)
        self.label_58.setObjectName("label_58")
        self.gridLayout_5.addWidget(self.label_58, 3, 2, 1, 1)
        self.l2 = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.l2.setReadOnly(True)
        self.l2.setDecimals(4)
        self.l2.setMaximum(10000.0)
        self.l2.setObjectName("l2")
        self.gridLayout_5.addWidget(self.l2, 3, 3, 1, 1)
        self.label_59 = QtWidgets.QLabel(self.frame_6)
        self.label_59.setObjectName("label_59")
        self.gridLayout_5.addWidget(self.label_59, 4, 2, 1, 1)
        self.label_60 = QtWidgets.QLabel(self.frame_6)
        self.label_60.setObjectName("label_60")
        self.gridLayout_5.addWidget(self.label_60, 5, 2, 1, 1)
        self.b_sa = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.b_sa.setReadOnly(True)
        self.b_sa.setDecimals(4)
        self.b_sa.setMaximum(10000.0)
        self.b_sa.setObjectName("b_sa")
        self.gridLayout_5.addWidget(self.b_sa, 4, 3, 1, 1)
        self.b_rk = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.b_rk.setReadOnly(True)
        self.b_rk.setDecimals(4)
        self.b_rk.setMaximum(10000.0)
        self.b_rk.setObjectName("b_rk")
        self.gridLayout_5.addWidget(self.b_rk, 5, 3, 1, 1)
        self.verticalLayout_7.addLayout(self.gridLayout_5)
        self.label_34 = QtWidgets.QLabel(self.frame_6)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_34.setFont(font)
        self.label_34.setAlignment(QtCore.Qt.AlignCenter)
        self.label_34.setObjectName("label_34")
        self.verticalLayout_7.addWidget(self.label_34)
        self.gridLayout_6 = QtWidgets.QGridLayout()
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.w2 = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.w2.setReadOnly(True)
        self.w2.setMinimum(-10000.0)
        self.w2.setMaximum(10000.0)
        self.w2.setObjectName("w2")
        self.gridLayout_6.addWidget(self.w2, 5, 3, 1, 1)
        self.label_67 = QtWidgets.QLabel(self.frame_6)
        self.label_67.setObjectName("label_67")
        self.gridLayout_6.addWidget(self.label_67, 7, 0, 1, 1)
        self.label_69 = QtWidgets.QLabel(self.frame_6)
        self.label_69.setObjectName("label_69")
        self.gridLayout_6.addWidget(self.label_69, 0, 2, 1, 1)
        self.label_68 = QtWidgets.QLabel(self.frame_6)
        self.label_68.setObjectName("label_68")
        self.gridLayout_6.addWidget(self.label_68, 8, 0, 1, 1)
        self.label_64 = QtWidgets.QLabel(self.frame_6)
        self.label_64.setObjectName("label_64")
        self.gridLayout_6.addWidget(self.label_64, 4, 0, 1, 1)
        self.w1 = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.w1.setReadOnly(True)
        self.w1.setMinimum(-10000.0)
        self.w1.setMaximum(10000.0)
        self.w1.setObjectName("w1")
        self.gridLayout_6.addWidget(self.w1, 5, 1, 1, 1)
        self.label_66 = QtWidgets.QLabel(self.frame_6)
        self.label_66.setObjectName("label_66")
        self.gridLayout_6.addWidget(self.label_66, 6, 0, 1, 1)
        self.label_39 = QtWidgets.QLabel(self.frame_6)
        self.label_39.setObjectName("label_39")
        self.gridLayout_6.addWidget(self.label_39, 0, 0, 1, 1)
        self.c2 = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.c2.setReadOnly(True)
        self.c2.setMinimum(-10000.0)
        self.c2.setMaximum(10000.0)
        self.c2.setObjectName("c2")
        self.gridLayout_6.addWidget(self.c2, 0, 3, 1, 1)
        self.label_70 = QtWidgets.QLabel(self.frame_6)
        self.label_70.setObjectName("label_70")
        self.gridLayout_6.addWidget(self.label_70, 1, 2, 1, 1)
        self.c2_a = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.c2_a.setReadOnly(True)
        self.c2_a.setMinimum(-10000.0)
        self.c2_a.setMaximum(10000.0)
        self.c2_a.setObjectName("c2_a")
        self.gridLayout_6.addWidget(self.c2_a, 2, 3, 1, 1)
        self.w1_u = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.w1_u.setReadOnly(True)
        self.w1_u.setMinimum(-10000.0)
        self.w1_u.setMaximum(10000.0)
        self.w1_u.setObjectName("w1_u")
        self.gridLayout_6.addWidget(self.w1_u, 7, 1, 1, 1)
        self.beta1 = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.beta1.setReadOnly(True)
        self.beta1.setMinimum(-10000.0)
        self.beta1.setMaximum(10000.0)
        self.beta1.setObjectName("beta1")
        self.gridLayout_6.addWidget(self.beta1, 8, 1, 1, 1)
        self.w1_a = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.w1_a.setReadOnly(True)
        self.w1_a.setMinimum(-10000.0)
        self.w1_a.setMaximum(10000.0)
        self.w1_a.setObjectName("w1_a")
        self.gridLayout_6.addWidget(self.w1_a, 6, 1, 1, 1)
        self.c1 = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.c1.setReadOnly(True)
        self.c1.setMinimum(-10000.0)
        self.c1.setMaximum(10000.0)
        self.c1.setObjectName("c1")
        self.gridLayout_6.addWidget(self.c1, 0, 1, 1, 1)
        self.c1_u = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.c1_u.setReadOnly(True)
        self.c1_u.setMinimum(-10000.0)
        self.c1_u.setMaximum(10000.0)
        self.c1_u.setObjectName("c1_u")
        self.gridLayout_6.addWidget(self.c1_u, 3, 1, 1, 1)
        self.label_62 = QtWidgets.QLabel(self.frame_6)
        self.label_62.setObjectName("label_62")
        self.gridLayout_6.addWidget(self.label_62, 2, 0, 1, 1)
        self.c1_a = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.c1_a.setReadOnly(True)
        self.c1_a.setMinimum(-10000.0)
        self.c1_a.setMaximum(10000.0)
        self.c1_a.setObjectName("c1_a")
        self.gridLayout_6.addWidget(self.c1_a, 2, 1, 1, 1)
        self.label_73 = QtWidgets.QLabel(self.frame_6)
        self.label_73.setObjectName("label_73")
        self.gridLayout_6.addWidget(self.label_73, 4, 2, 1, 1)
        self.label_61 = QtWidgets.QLabel(self.frame_6)
        self.label_61.setObjectName("label_61")
        self.gridLayout_6.addWidget(self.label_61, 1, 0, 1, 1)
        self.u2 = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.u2.setReadOnly(True)
        self.u2.setMinimum(-10000.0)
        self.u2.setMaximum(10000.0)
        self.u2.setObjectName("u2")
        self.gridLayout_6.addWidget(self.u2, 1, 3, 1, 1)
        self.label_72 = QtWidgets.QLabel(self.frame_6)
        self.label_72.setObjectName("label_72")
        self.gridLayout_6.addWidget(self.label_72, 3, 2, 1, 1)
        self.c2_u = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.c2_u.setReadOnly(True)
        self.c2_u.setMinimum(-10000.0)
        self.c2_u.setMaximum(10000.0)
        self.c2_u.setObjectName("c2_u")
        self.gridLayout_6.addWidget(self.c2_u, 3, 3, 1, 1)
        self.alpha2 = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.alpha2.setReadOnly(True)
        self.alpha2.setMinimum(-10000.0)
        self.alpha2.setMaximum(10000.0)
        self.alpha2.setObjectName("alpha2")
        self.gridLayout_6.addWidget(self.alpha2, 4, 3, 1, 1)
        self.label_74 = QtWidgets.QLabel(self.frame_6)
        self.label_74.setObjectName("label_74")
        self.gridLayout_6.addWidget(self.label_74, 5, 2, 1, 1)
        self.label_63 = QtWidgets.QLabel(self.frame_6)
        self.label_63.setObjectName("label_63")
        self.gridLayout_6.addWidget(self.label_63, 3, 0, 1, 1)
        self.u1 = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.u1.setReadOnly(True)
        self.u1.setMinimum(-10000.0)
        self.u1.setMaximum(10000.0)
        self.u1.setObjectName("u1")
        self.gridLayout_6.addWidget(self.u1, 1, 1, 1, 1)
        self.label_75 = QtWidgets.QLabel(self.frame_6)
        self.label_75.setObjectName("label_75")
        self.gridLayout_6.addWidget(self.label_75, 6, 2, 1, 1)
        self.label_65 = QtWidgets.QLabel(self.frame_6)
        self.label_65.setObjectName("label_65")
        self.gridLayout_6.addWidget(self.label_65, 5, 0, 1, 1)
        self.alpha1 = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.alpha1.setReadOnly(True)
        self.alpha1.setMinimum(-10000.0)
        self.alpha1.setMaximum(10000.0)
        self.alpha1.setObjectName("alpha1")
        self.gridLayout_6.addWidget(self.alpha1, 4, 1, 1, 1)
        self.label_71 = QtWidgets.QLabel(self.frame_6)
        self.label_71.setObjectName("label_71")
        self.gridLayout_6.addWidget(self.label_71, 2, 2, 1, 1)
        self.label_77 = QtWidgets.QLabel(self.frame_6)
        self.label_77.setObjectName("label_77")
        self.gridLayout_6.addWidget(self.label_77, 7, 2, 1, 1)
        self.label_76 = QtWidgets.QLabel(self.frame_6)
        self.label_76.setObjectName("label_76")
        self.gridLayout_6.addWidget(self.label_76, 8, 2, 1, 1)
        self.beta2 = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.beta2.setReadOnly(True)
        self.beta2.setMinimum(-10000.0)
        self.beta2.setMaximum(10000.0)
        self.beta2.setObjectName("beta2")
        self.gridLayout_6.addWidget(self.beta2, 8, 3, 1, 1)
        self.w2_u = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.w2_u.setReadOnly(True)
        self.w2_u.setMinimum(-10000.0)
        self.w2_u.setMaximum(10000.0)
        self.w2_u.setObjectName("w2_u")
        self.gridLayout_6.addWidget(self.w2_u, 7, 3, 1, 1)
        self.w2_a = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.w2_a.setReadOnly(True)
        self.w2_a.setMinimum(-10000.0)
        self.w2_a.setMaximum(10000.0)
        self.w2_a.setObjectName("w2_a")
        self.gridLayout_6.addWidget(self.w2_a, 6, 3, 1, 1)
        self.label_82 = QtWidgets.QLabel(self.frame_6)
        self.label_82.setObjectName("label_82")
        self.gridLayout_6.addWidget(self.label_82, 4, 4, 1, 1)
        self.label_80 = QtWidgets.QLabel(self.frame_6)
        self.label_80.setObjectName("label_80")
        self.gridLayout_6.addWidget(self.label_80, 2, 4, 1, 1)
        self.label_81 = QtWidgets.QLabel(self.frame_6)
        self.label_81.setObjectName("label_81")
        self.gridLayout_6.addWidget(self.label_81, 3, 4, 1, 1)
        self.label_84 = QtWidgets.QLabel(self.frame_6)
        self.label_84.setObjectName("label_84")
        self.gridLayout_6.addWidget(self.label_84, 6, 4, 1, 1)
        self.H0_stag = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.H0_stag.setReadOnly(True)
        self.H0_stag.setMinimum(-10000.0)
        self.H0_stag.setMaximum(10000.0)
        self.H0_stag.setObjectName("H0_stag")
        self.gridLayout_6.addWidget(self.H0_stag, 1, 5, 1, 1)
        self.label_85 = QtWidgets.QLabel(self.frame_6)
        self.label_85.setObjectName("label_85")
        self.gridLayout_6.addWidget(self.label_85, 7, 4, 1, 1)
        self.label_78 = QtWidgets.QLabel(self.frame_6)
        self.label_78.setObjectName("label_78")
        self.gridLayout_6.addWidget(self.label_78, 0, 4, 1, 1)
        self.label_87 = QtWidgets.QLabel(self.frame_6)
        self.label_87.setObjectName("label_87")
        self.gridLayout_6.addWidget(self.label_87, 0, 6, 1, 1)
        self.label_86 = QtWidgets.QLabel(self.frame_6)
        self.label_86.setObjectName("label_86")
        self.gridLayout_6.addWidget(self.label_86, 8, 4, 1, 1)
        self.T2 = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.T2.setReadOnly(True)
        self.T2.setMinimum(-10000.0)
        self.T2.setMaximum(10000.0)
        self.T2.setObjectName("T2")
        self.gridLayout_6.addWidget(self.T2, 3, 5, 1, 1)
        self.T1 = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.T1.setReadOnly(True)
        self.T1.setMinimum(-10000.0)
        self.T1.setMaximum(10000.0)
        self.T1.setObjectName("T1")
        self.gridLayout_6.addWidget(self.T1, 2, 5, 1, 1)
        self.T_st_stag = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.T_st_stag.setReadOnly(True)
        self.T_st_stag.setMinimum(-10000.0)
        self.T_st_stag.setMaximum(10000.0)
        self.T_st_stag.setObjectName("T_st_stag")
        self.gridLayout_6.addWidget(self.T_st_stag, 5, 5, 1, 1)
        self.p2 = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.p2.setReadOnly(True)
        self.p2.setMinimum(-10000.0)
        self.p2.setMaximum(10000.0)
        self.p2.setObjectName("p2")
        self.gridLayout_6.addWidget(self.p2, 7, 5, 1, 1)
        self.T_st = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.T_st.setReadOnly(True)
        self.T_st.setMinimum(-10000.0)
        self.T_st.setMaximum(10000.0)
        self.T_st.setObjectName("T_st")
        self.gridLayout_6.addWidget(self.T_st, 4, 5, 1, 1)
        self.p1 = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.p1.setReadOnly(True)
        self.p1.setMinimum(-10000.0)
        self.p1.setMaximum(10000.0)
        self.p1.setObjectName("p1")
        self.gridLayout_6.addWidget(self.p1, 6, 5, 1, 1)
        self.p2_stag = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.p2_stag.setReadOnly(True)
        self.p2_stag.setMinimum(-10000.0)
        self.p2_stag.setMaximum(10000.0)
        self.p2_stag.setObjectName("p2_stag")
        self.gridLayout_6.addWidget(self.p2_stag, 8, 5, 1, 1)
        self.label_79 = QtWidgets.QLabel(self.frame_6)
        self.label_79.setObjectName("label_79")
        self.gridLayout_6.addWidget(self.label_79, 1, 4, 1, 1)
        self.label_83 = QtWidgets.QLabel(self.frame_6)
        self.label_83.setObjectName("label_83")
        self.gridLayout_6.addWidget(self.label_83, 5, 4, 1, 1)
        self.H0_out = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.H0_out.setReadOnly(True)
        self.H0_out.setMinimum(-10000.0)
        self.H0_out.setMaximum(10000.0)
        self.H0_out.setObjectName("H0_out")
        self.gridLayout_6.addWidget(self.H0_out, 0, 5, 1, 1)
        self.eta_u = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.eta_u.setReadOnly(True)
        self.eta_u.setDecimals(3)
        self.eta_u.setMinimum(-10000.0)
        self.eta_u.setMaximum(10000.0)
        self.eta_u.setObjectName("eta_u")
        self.gridLayout_6.addWidget(self.eta_u, 0, 7, 1, 1)
        self.label_88 = QtWidgets.QLabel(self.frame_6)
        self.label_88.setObjectName("label_88")
        self.gridLayout_6.addWidget(self.label_88, 1, 6, 1, 1)
        self.label_89 = QtWidgets.QLabel(self.frame_6)
        self.label_89.setObjectName("label_89")
        self.gridLayout_6.addWidget(self.label_89, 2, 6, 1, 1)
        self.label_90 = QtWidgets.QLabel(self.frame_6)
        self.label_90.setObjectName("label_90")
        self.gridLayout_6.addWidget(self.label_90, 3, 6, 1, 1)
        self.eta_l = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.eta_l.setReadOnly(True)
        self.eta_l.setDecimals(3)
        self.eta_l.setMinimum(-10000.0)
        self.eta_l.setMaximum(10000.0)
        self.eta_l.setObjectName("eta_l")
        self.gridLayout_6.addWidget(self.eta_l, 1, 7, 1, 1)
        self.eta_t = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.eta_t.setReadOnly(True)
        self.eta_t.setDecimals(3)
        self.eta_t.setMinimum(-10000.0)
        self.eta_t.setMaximum(10000.0)
        self.eta_t.setObjectName("eta_t")
        self.gridLayout_6.addWidget(self.eta_t, 2, 7, 1, 1)
        self.eta_t_stag = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.eta_t_stag.setReadOnly(True)
        self.eta_t_stag.setDecimals(3)
        self.eta_t_stag.setMinimum(-10000.0)
        self.eta_t_stag.setMaximum(10000.0)
        self.eta_t_stag.setObjectName("eta_t_stag")
        self.gridLayout_6.addWidget(self.eta_t_stag, 3, 7, 1, 1)
        self.label_91 = QtWidgets.QLabel(self.frame_6)
        self.label_91.setObjectName("label_91")
        self.gridLayout_6.addWidget(self.label_91, 4, 6, 1, 1)
        self.rho_out = QtWidgets.QDoubleSpinBox(self.frame_6)
        self.rho_out.setReadOnly(True)
        self.rho_out.setDecimals(3)
        self.rho_out.setMinimum(-10000.0)
        self.rho_out.setMaximum(10000.0)
        self.rho_out.setObjectName("rho_out")
        self.gridLayout_6.addWidget(self.rho_out, 4, 7, 1, 1)
        self.verticalLayout_7.addLayout(self.gridLayout_6)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_7.addItem(spacerItem1)
        self.horizontalLayout_4.addWidget(self.frame_6)
        self.verticalLayout.addLayout(self.horizontalLayout_4)
        spacerItem2 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem2)
        self.horizontalLayout.addWidget(self.frame_3)
        self.frame_triangle_plot = QtWidgets.QFrame(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_triangle_plot.sizePolicy().hasHeightForWidth())
        self.frame_triangle_plot.setSizePolicy(sizePolicy)
        self.frame_triangle_plot.setMinimumSize(QtCore.QSize(0, 0))
        self.frame_triangle_plot.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_triangle_plot.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_triangle_plot.setObjectName("frame_triangle_plot")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.frame_triangle_plot)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.horizontalLayout_triangle_plot = QtWidgets.QHBoxLayout()
        self.horizontalLayout_triangle_plot.setObjectName("horizontalLayout_triangle_plot")
        self.horizontalLayout_5.addLayout(self.horizontalLayout_triangle_plot)
        self.horizontalLayout.addWidget(self.frame_triangle_plot)
        spacerItem3 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem3)

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_stage_title.setText(_translate("Form", "Ступень 1"))
        self.label_32.setText(_translate("Form", "Исходные данные"))
        self.label_46.setText(_translate("Form", "Степень парциальности"))
        self.label_50.setText(_translate("Form", "Отн-ый расход перетечек поверх бондажа рабочих лопаток"))
        self.label_44.setText(_translate("Form", "К-т скорости в СА"))
        self.label_43.setText(_translate("Form", "Относительный радиальный зазор"))
        self.label_47.setText(_translate("Form", "К-т использования скорости на выходе"))
        self.label_45.setText(_translate("Form", "К-т скорости в РК"))
        self.label_41.setText(_translate("Form", "Относительный зазор после СА"))
        self.label_40.setText(_translate("Form", "Относительная зазор после РК"))
        self.label_35.setText(_translate("Form", "Относительное удлинение РК"))
        self.label_49.setText(_translate("Form", "Отно-ый расход ут-к в лаб-ых упл-их сопловых диафрагм"))
        self.label_36.setText(_translate("Form", "Относительное удинение СА"))
        self.label_48.setText(_translate("Form", "Отн-ый расход утечек в концевых лабиринтах["))
        self.label_rho.setText(_translate("Form", "Степень реактивности"))
        self.label_H0.setText(_translate("Form", "Статический теплоперепад, кДж/кг"))
        self.label_33.setText(_translate("Form", "Геометрические параметры"))
        self.label_55.setText(_translate("Form", "D1, м"))
        self.label_51.setText(_translate("Form", "gamma_вт, град"))
        self.label_37.setText(_translate("Form", "gamma_сум, град"))
        self.label_38.setText(_translate("Form", "gamma_ср, град"))
        self.label_52.setText(_translate("Form", "gamma_пер, град"))
        self.label_l0.setText(_translate("Form", "l0, м"))
        self.label_53.setText(_translate("Form", "D0, м"))
        self.label_56.setText(_translate("Form", "l1, м"))
        self.label_57.setText(_translate("Form", "D2, м"))
        self.label_58.setText(_translate("Form", "l2, м"))
        self.label_59.setText(_translate("Form", "b_са, м"))
        self.label_60.setText(_translate("Form", "b_рк, м"))
        self.label_34.setText(_translate("Form", "Газодинамические параметры"))
        self.label_67.setText(_translate("Form", "w1_u, м/с"))
        self.label_69.setText(_translate("Form", "c2, м/с"))
        self.label_68.setText(_translate("Form", "beta1, град"))
        self.label_64.setText(_translate("Form", "alpha1, град"))
        self.label_66.setText(_translate("Form", "w1_a, м/с"))
        self.label_39.setText(_translate("Form", "c1, м/с"))
        self.label_70.setText(_translate("Form", "u2, м/с"))
        self.label_62.setText(_translate("Form", "c1_a, м/с"))
        self.label_73.setText(_translate("Form", "alpha2, град"))
        self.label_61.setText(_translate("Form", "u1, м/с"))
        self.label_72.setText(_translate("Form", "c2_u, м/с"))
        self.label_74.setText(_translate("Form", "w2, м/с"))
        self.label_63.setText(_translate("Form", "c1_u, м/с"))
        self.label_75.setText(_translate("Form", "w2_a, м/с"))
        self.label_65.setText(_translate("Form", "w1, м/с"))
        self.label_71.setText(_translate("Form", "c2_a, м/с"))
        self.label_77.setText(_translate("Form", "w2_u, м/с"))
        self.label_76.setText(_translate("Form", "beta2, град"))
        self.label_82.setText(_translate("Form", "T_ст, К"))
        self.label_80.setText(_translate("Form", "T1, К"))
        self.label_81.setText(_translate("Form", "T2, К"))
        self.label_84.setText(_translate("Form", "p1, МПа"))
        self.label_85.setText(_translate("Form", "p2, МПа"))
        self.label_78.setText(_translate("Form", "H0, кДж/кг"))
        self.label_87.setText(_translate("Form", "eta_u"))
        self.label_86.setText(_translate("Form", "p2*, МПа"))
        self.label_79.setText(_translate("Form", "H0*, кДж/кг"))
        self.label_83.setText(_translate("Form", "T_ст*, К"))
        self.label_88.setText(_translate("Form", "eta_л"))
        self.label_89.setText(_translate("Form", "eta_т"))
        self.label_90.setText(_translate("Form", "eta_t*"))
        self.label_91.setText(_translate("Form", "rho"))

