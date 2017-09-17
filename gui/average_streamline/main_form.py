# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_widget.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1362, 741)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.tabWidget = QtWidgets.QTabWidget(Form)
        self.tabWidget.setTabPosition(QtWidgets.QTabWidget.North)
        self.tabWidget.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.tabWidget.setUsesScrollButtons(False)
        self.tabWidget.setDocumentMode(False)
        self.tabWidget.setTabsClosable(False)
        self.tabWidget.setMovable(False)
        self.tabWidget.setTabBarAutoHide(False)
        self.tabWidget.setObjectName("tabWidget")
        self.tab_init_data = QtWidgets.QWidget()
        self.tab_init_data.setObjectName("tab_init_data")
        self.horizontalLayout = QtWidgets.QHBoxLayout(self.tab_init_data)
        self.horizontalLayout.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(self.tab_init_data)
        self.frame.setFrameShape(QtWidgets.QFrame.Box)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.label_2 = QtWidgets.QLabel(self.frame)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.verticalLayout_3.addWidget(self.label_2)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_11 = QtWidgets.QLabel(self.frame)
        self.label_11.setObjectName("label_11")
        self.gridLayout_2.addWidget(self.label_11, 14, 0, 1, 1)
        self.gamma1 = QtWidgets.QDoubleSpinBox(self.frame)
        self.gamma1.setDecimals(1)
        self.gamma1.setMinimum(-30.0)
        self.gamma1.setMaximum(90.0)
        self.gamma1.setProperty("value", 10.0)
        self.gamma1.setObjectName("gamma1")
        self.gridLayout_2.addWidget(self.gamma1, 6, 1, 1, 1)
        self.label_5 = QtWidgets.QLabel(self.frame)
        self.label_5.setObjectName("label_5")
        self.gridLayout_2.addWidget(self.label_5, 8, 0, 1, 1)
        self.label_9 = QtWidgets.QLabel(self.frame)
        self.label_9.setObjectName("label_9")
        self.gridLayout_2.addWidget(self.label_9, 12, 0, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.frame)
        self.label_8.setObjectName("label_8")
        self.gridLayout_2.addWidget(self.label_8, 11, 0, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.frame)
        self.label_10.setObjectName("label_10")
        self.gridLayout_2.addWidget(self.label_10, 13, 0, 1, 1)
        self.label_gamma2 = QtWidgets.QLabel(self.frame)
        self.label_gamma2.setObjectName("label_gamma2")
        self.gridLayout_2.addWidget(self.label_gamma2, 7, 0, 1, 1)
        self.G_t = QtWidgets.QDoubleSpinBox(self.frame)
        self.G_t.setMinimum(0.1)
        self.G_t.setMaximum(1000.0)
        self.G_t.setProperty("value", 25.0)
        self.G_t.setObjectName("G_t")
        self.gridLayout_2.addWidget(self.G_t, 19, 1, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.frame)
        self.label_15.setObjectName("label_15")
        self.gridLayout_2.addWidget(self.label_15, 19, 0, 1, 1)
        self.n = QtWidgets.QDoubleSpinBox(self.frame)
        self.n.setDecimals(0)
        self.n.setMaximum(100000.0)
        self.n.setSingleStep(500.0)
        self.n.setProperty("value", 15000.0)
        self.n.setObjectName("n")
        self.gridLayout_2.addWidget(self.n, 20, 1, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.frame)
        self.label_16.setObjectName("label_16")
        self.gridLayout_2.addWidget(self.label_16, 20, 0, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.frame)
        self.label_12.setObjectName("label_12")
        self.gridLayout_2.addWidget(self.label_12, 15, 0, 1, 1)
        self.T_g_stag = QtWidgets.QDoubleSpinBox(self.frame)
        self.T_g_stag.setMinimum(500.0)
        self.T_g_stag.setMaximum(3000.0)
        self.T_g_stag.setSingleStep(50.0)
        self.T_g_stag.setProperty("value", 1400.0)
        self.T_g_stag.setObjectName("T_g_stag")
        self.gridLayout_2.addWidget(self.T_g_stag, 11, 1, 1, 1)
        self.p_g_stag = QtWidgets.QDoubleSpinBox(self.frame)
        self.p_g_stag.setDecimals(4)
        self.p_g_stag.setMinimum(0.1)
        self.p_g_stag.setMaximum(10.0)
        self.p_g_stag.setSingleStep(0.1)
        self.p_g_stag.setProperty("value", 0.3)
        self.p_g_stag.setObjectName("p_g_stag")
        self.gridLayout_2.addWidget(self.p_g_stag, 12, 1, 1, 1)
        self.alpha11 = QtWidgets.QDoubleSpinBox(self.frame)
        self.alpha11.setDecimals(1)
        self.alpha11.setMinimum(5.0)
        self.alpha11.setMaximum(45.0)
        self.alpha11.setProperty("value", 17.0)
        self.alpha11.setObjectName("alpha11")
        self.gridLayout_2.addWidget(self.alpha11, 8, 1, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.frame)
        self.label_20.setObjectName("label_20")
        self.gridLayout_2.addWidget(self.label_20, 22, 0, 1, 1)
        self.eta_m = QtWidgets.QDoubleSpinBox(self.frame)
        self.eta_m.setDecimals(3)
        self.eta_m.setMaximum(1.0)
        self.eta_m.setSingleStep(0.01)
        self.eta_m.setProperty("value", 0.98)
        self.eta_m.setObjectName("eta_m")
        self.gridLayout_2.addWidget(self.eta_m, 22, 1, 1, 1)
        self.label_6 = QtWidgets.QLabel(self.frame)
        self.label_6.setObjectName("label_6")
        self.gridLayout_2.addWidget(self.label_6, 9, 0, 1, 1)
        self.alpha_air = QtWidgets.QDoubleSpinBox(self.frame)
        self.alpha_air.setDecimals(3)
        self.alpha_air.setMaximum(1000.0)
        self.alpha_air.setSingleStep(0.5)
        self.alpha_air.setProperty("value", 2.5)
        self.alpha_air.setObjectName("alpha_air")
        self.gridLayout_2.addWidget(self.alpha_air, 9, 1, 1, 1)
        self.label_gamma1 = QtWidgets.QLabel(self.frame)
        self.label_gamma1.setWhatsThis("")
        self.label_gamma1.setObjectName("label_gamma1")
        self.gridLayout_2.addWidget(self.label_gamma1, 6, 0, 1, 1)
        self.c21 = QtWidgets.QDoubleSpinBox(self.frame)
        self.c21.setDecimals(1)
        self.c21.setMinimum(10.0)
        self.c21.setMaximum(600.0)
        self.c21.setSingleStep(10.0)
        self.c21.setProperty("value", 250.0)
        self.c21.setObjectName("c21")
        self.gridLayout_2.addWidget(self.c21, 10, 1, 1, 1)
        self.label_7 = QtWidgets.QLabel(self.frame)
        self.label_7.setObjectName("label_7")
        self.gridLayout_2.addWidget(self.label_7, 10, 0, 1, 1)
        self.eta_t_stag_cycle = QtWidgets.QDoubleSpinBox(self.frame)
        self.eta_t_stag_cycle.setDecimals(3)
        self.eta_t_stag_cycle.setMaximum(1.0)
        self.eta_t_stag_cycle.setSingleStep(0.01)
        self.eta_t_stag_cycle.setProperty("value", 0.91)
        self.eta_t_stag_cycle.setObjectName("eta_t_stag_cycle")
        self.gridLayout_2.addWidget(self.eta_t_stag_cycle, 18, 1, 1, 1)
        self.H01_init = QtWidgets.QDoubleSpinBox(self.frame)
        self.H01_init.setMaximum(5000.0)
        self.H01_init.setSingleStep(10.0)
        self.H01_init.setProperty("value", 120.0)
        self.H01_init.setObjectName("H01_init")
        self.gridLayout_2.addWidget(self.H01_init, 17, 1, 1, 1)
        self.T_t_stag_cycle = QtWidgets.QDoubleSpinBox(self.frame)
        self.T_t_stag_cycle.setMinimum(240.0)
        self.T_t_stag_cycle.setMaximum(2500.0)
        self.T_t_stag_cycle.setProperty("value", 800.0)
        self.T_t_stag_cycle.setObjectName("T_t_stag_cycle")
        self.gridLayout_2.addWidget(self.T_t_stag_cycle, 13, 1, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.frame)
        self.label_13.setObjectName("label_13")
        self.gridLayout_2.addWidget(self.label_13, 17, 0, 1, 1)
        self.label_14 = QtWidgets.QLabel(self.frame)
        self.label_14.setObjectName("label_14")
        self.gridLayout_2.addWidget(self.label_14, 18, 0, 1, 1)
        self.gamma2 = QtWidgets.QDoubleSpinBox(self.frame)
        self.gamma2.setMinimum(-30.0)
        self.gamma2.setMaximum(60.0)
        self.gamma2.setProperty("value", 4.0)
        self.gamma2.setObjectName("gamma2")
        self.gridLayout_2.addWidget(self.gamma2, 7, 1, 1, 1)
        self.p_t_stag_cycle = QtWidgets.QDoubleSpinBox(self.frame)
        self.p_t_stag_cycle.setDecimals(4)
        self.p_t_stag_cycle.setMinimum(0.06)
        self.p_t_stag_cycle.setMaximum(5.0)
        self.p_t_stag_cycle.setSingleStep(0.1)
        self.p_t_stag_cycle.setProperty("value", 0.1)
        self.p_t_stag_cycle.setObjectName("p_t_stag_cycle")
        self.gridLayout_2.addWidget(self.p_t_stag_cycle, 14, 1, 1, 1)
        self.H_t_stag_cycle = QtWidgets.QDoubleSpinBox(self.frame)
        self.H_t_stag_cycle.setMaximum(5000.0)
        self.H_t_stag_cycle.setSingleStep(10.0)
        self.H_t_stag_cycle.setProperty("value", 250.0)
        self.H_t_stag_cycle.setObjectName("H_t_stag_cycle")
        self.gridLayout_2.addWidget(self.H_t_stag_cycle, 15, 1, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.frame)
        self.label_17.setObjectName("label_17")
        self.gridLayout_2.addWidget(self.label_17, 21, 0, 1, 1)
        self.l1_D1_ratio = QtWidgets.QDoubleSpinBox(self.frame)
        self.l1_D1_ratio.setDecimals(3)
        self.l1_D1_ratio.setMinimum(0.03)
        self.l1_D1_ratio.setMaximum(0.9)
        self.l1_D1_ratio.setSingleStep(0.05)
        self.l1_D1_ratio.setProperty("value", 0.25)
        self.l1_D1_ratio.setObjectName("l1_D1_ratio")
        self.gridLayout_2.addWidget(self.l1_D1_ratio, 21, 1, 1, 1)
        self.label = QtWidgets.QLabel(self.frame)
        self.label.setObjectName("label")
        self.gridLayout_2.addWidget(self.label, 16, 0, 1, 1)
        self.L_t_cycle = QtWidgets.QDoubleSpinBox(self.frame)
        self.L_t_cycle.setMaximum(5000.0)
        self.L_t_cycle.setSingleStep(10.0)
        self.L_t_cycle.setProperty("value", 220.0)
        self.L_t_cycle.setObjectName("L_t_cycle")
        self.gridLayout_2.addWidget(self.L_t_cycle, 16, 1, 1, 1)
        self.verticalLayout_3.addLayout(self.gridLayout_2)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_3.addItem(spacerItem)
        self.horizontalLayout.addWidget(self.frame)
        self.frame_2 = QtWidgets.QFrame(self.tab_init_data)
        self.frame_2.setFrameShape(QtWidgets.QFrame.Box)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.verticalLayout_5 = QtWidgets.QVBoxLayout(self.frame_2)
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.groupBox = QtWidgets.QGroupBox(self.frame_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.groupBox.setFont(font)
        self.groupBox.setAlignment(QtCore.Qt.AlignCenter)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.stage_number = QtWidgets.QSpinBox(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.stage_number.setFont(font)
        self.stage_number.setMinimum(1)
        self.stage_number.setMaximum(9)
        self.stage_number.setProperty("value", 1)
        self.stage_number.setObjectName("stage_number")
        self.gridLayout_4.addWidget(self.stage_number, 1, 1, 1, 1)
        self.label_30 = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_30.setFont(font)
        self.label_30.setObjectName("label_30")
        self.gridLayout_4.addWidget(self.label_30, 2, 0, 1, 1)
        self.checkBox_rho_auto = QtWidgets.QCheckBox(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.checkBox_rho_auto.setFont(font)
        self.checkBox_rho_auto.setChecked(True)
        self.checkBox_rho_auto.setObjectName("checkBox_rho_auto")
        self.gridLayout_4.addWidget(self.checkBox_rho_auto, 3, 0, 1, 1)
        self.checkBox_h0_auto = QtWidgets.QCheckBox(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.checkBox_h0_auto.setFont(font)
        self.checkBox_h0_auto.setChecked(True)
        self.checkBox_h0_auto.setTristate(False)
        self.checkBox_h0_auto.setObjectName("checkBox_h0_auto")
        self.gridLayout_4.addWidget(self.checkBox_h0_auto, 4, 0, 1, 1)
        self.turbine_type = QtWidgets.QComboBox(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.turbine_type.setFont(font)
        self.turbine_type.setObjectName("turbine_type")
        self.turbine_type.addItem("")
        self.turbine_type.addItem("")
        self.gridLayout_4.addWidget(self.turbine_type, 0, 1, 1, 1)
        self.label_29 = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_29.setFont(font)
        self.label_29.setObjectName("label_29")
        self.gridLayout_4.addWidget(self.label_29, 1, 0, 1, 1)
        self.label_28 = QtWidgets.QLabel(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.label_28.setFont(font)
        self.label_28.setObjectName("label_28")
        self.gridLayout_4.addWidget(self.label_28, 0, 0, 1, 1)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.gamma_sum_av = QtWidgets.QRadioButton(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.gamma_sum_av.setFont(font)
        self.gamma_sum_av.setChecked(True)
        self.gamma_sum_av.setObjectName("gamma_sum_av")
        self.verticalLayout_6.addWidget(self.gamma_sum_av)
        self.gamma_in_out = QtWidgets.QRadioButton(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.gamma_in_out.setFont(font)
        self.gamma_in_out.setObjectName("gamma_in_out")
        self.verticalLayout_6.addWidget(self.gamma_in_out)
        self.gridLayout_4.addLayout(self.verticalLayout_6, 2, 1, 1, 1)
        self.checkBox_precise_h0 = QtWidgets.QCheckBox(self.groupBox)
        font = QtGui.QFont()
        font.setPointSize(8)
        self.checkBox_precise_h0.setFont(font)
        self.checkBox_precise_h0.setChecked(True)
        self.checkBox_precise_h0.setObjectName("checkBox_precise_h0")
        self.gridLayout_4.addWidget(self.checkBox_precise_h0, 5, 0, 1, 1)
        self.verticalLayout_5.addWidget(self.groupBox)
        self.label_18 = QtWidgets.QLabel(self.frame_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.label_18.setFont(font)
        self.label_18.setAlignment(QtCore.Qt.AlignCenter)
        self.label_18.setObjectName("label_18")
        self.verticalLayout_5.addWidget(self.label_18)
        self.gridLayout_3 = QtWidgets.QGridLayout()
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_23 = QtWidgets.QLabel(self.frame_2)
        self.label_23.setObjectName("label_23")
        self.gridLayout_3.addWidget(self.label_23, 3, 0, 1, 1)
        self.label_22 = QtWidgets.QLabel(self.frame_2)
        self.label_22.setObjectName("label_22")
        self.gridLayout_3.addWidget(self.label_22, 2, 0, 1, 1)
        self.label_25 = QtWidgets.QLabel(self.frame_2)
        self.label_25.setObjectName("label_25")
        self.gridLayout_3.addWidget(self.label_25, 5, 0, 1, 1)
        self.label_24 = QtWidgets.QLabel(self.frame_2)
        self.label_24.setObjectName("label_24")
        self.gridLayout_3.addWidget(self.label_24, 4, 0, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.frame_2)
        self.label_21.setObjectName("label_21")
        self.gridLayout_3.addWidget(self.label_21, 1, 0, 1, 1)
        self.label_19 = QtWidgets.QLabel(self.frame_2)
        self.label_19.setObjectName("label_19")
        self.gridLayout_3.addWidget(self.label_19, 0, 0, 1, 1)
        self.L_t_sum = QtWidgets.QDoubleSpinBox(self.frame_2)
        self.L_t_sum.setReadOnly(True)
        self.L_t_sum.setDecimals(2)
        self.L_t_sum.setMaximum(6000.0)
        self.L_t_sum.setObjectName("L_t_sum")
        self.gridLayout_3.addWidget(self.L_t_sum, 0, 1, 1, 1)
        self.label_26 = QtWidgets.QLabel(self.frame_2)
        self.label_26.setObjectName("label_26")
        self.gridLayout_3.addWidget(self.label_26, 6, 0, 1, 1)
        self.label_27 = QtWidgets.QLabel(self.frame_2)
        self.label_27.setObjectName("label_27")
        self.gridLayout_3.addWidget(self.label_27, 7, 0, 1, 1)
        self.H_t_stag = QtWidgets.QDoubleSpinBox(self.frame_2)
        self.H_t_stag.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.H_t_stag.setReadOnly(True)
        self.H_t_stag.setMaximum(5000.0)
        self.H_t_stag.setObjectName("H_t_stag")
        self.gridLayout_3.addWidget(self.H_t_stag, 1, 1, 1, 1)
        self.H_t = QtWidgets.QDoubleSpinBox(self.frame_2)
        self.H_t.setReadOnly(True)
        self.H_t.setMaximum(5000.0)
        self.H_t.setObjectName("H_t")
        self.gridLayout_3.addWidget(self.H_t, 2, 1, 1, 1)
        self.eta_t = QtWidgets.QDoubleSpinBox(self.frame_2)
        self.eta_t.setReadOnly(True)
        self.eta_t.setMaximum(1.0)
        self.eta_t.setObjectName("eta_t")
        self.gridLayout_3.addWidget(self.eta_t, 3, 1, 1, 1)
        self.eta_t_stag = QtWidgets.QDoubleSpinBox(self.frame_2)
        self.eta_t_stag.setReadOnly(True)
        self.eta_t_stag.setMaximum(1.0)
        self.eta_t_stag.setObjectName("eta_t_stag")
        self.gridLayout_3.addWidget(self.eta_t_stag, 4, 1, 1, 1)
        self.eta_l = QtWidgets.QDoubleSpinBox(self.frame_2)
        self.eta_l.setReadOnly(False)
        self.eta_l.setMaximum(1.0)
        self.eta_l.setObjectName("eta_l")
        self.gridLayout_3.addWidget(self.eta_l, 5, 1, 1, 1)
        self.c_p_gas_av = QtWidgets.QDoubleSpinBox(self.frame_2)
        self.c_p_gas_av.setReadOnly(True)
        self.c_p_gas_av.setMaximum(4000.0)
        self.c_p_gas_av.setObjectName("c_p_gas_av")
        self.gridLayout_3.addWidget(self.c_p_gas_av, 6, 1, 1, 1)
        self.k_gas_av = QtWidgets.QDoubleSpinBox(self.frame_2)
        self.k_gas_av.setReadOnly(True)
        self.k_gas_av.setMaximum(9.0)
        self.k_gas_av.setObjectName("k_gas_av")
        self.gridLayout_3.addWidget(self.k_gas_av, 7, 1, 1, 1)
        self.verticalLayout_5.addLayout(self.gridLayout_3)
        self.compute_btn = QtWidgets.QPushButton(self.frame_2)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.compute_btn.setFont(font)
        self.compute_btn.setObjectName("compute_btn")
        self.verticalLayout_5.addWidget(self.compute_btn)
        spacerItem1 = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout_5.addItem(spacerItem1)
        self.horizontalLayout.addWidget(self.frame_2)
        spacerItem2 = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum)
        self.horizontalLayout.addItem(spacerItem2)
        self.tabWidget.addTab(self.tab_init_data, "")
        self.tab_stage_data = QtWidgets.QWidget()
        self.tab_stage_data.setObjectName("tab_stage_data")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.tab_stage_data)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.stackedWidget = QtWidgets.QStackedWidget(self.tab_stage_data)
        self.stackedWidget.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.stackedWidget.setObjectName("stackedWidget")
        self.verticalLayout.addWidget(self.stackedWidget)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.prevPage_btn = QtWidgets.QPushButton(self.tab_stage_data)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.prevPage_btn.setFont(font)
        self.prevPage_btn.setObjectName("prevPage_btn")
        self.horizontalLayout_3.addWidget(self.prevPage_btn)
        self.nextPage_btn = QtWidgets.QPushButton(self.tab_stage_data)
        font = QtGui.QFont()
        font.setPointSize(12)
        self.nextPage_btn.setFont(font)
        self.nextPage_btn.setObjectName("nextPage_btn")
        self.horizontalLayout_3.addWidget(self.nextPage_btn)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.tabWidget.addTab(self.tab_stage_data, "")
        self.horizontalLayout_5.addWidget(self.tabWidget)

        self.retranslateUi(Form)
        self.tabWidget.setCurrentIndex(0)
        self.stackedWidget.setCurrentIndex(-1)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.label_2.setText(_translate("Form", "Исходные данные"))
        self.label_11.setText(_translate("Form", "Давление торможения после турбины, МПа"))
        self.label_5.setText(_translate("Form", "Угол потока после СА первой ступени, град"))
        self.label_9.setText(_translate("Form", "Давление перед турбиной, МПа"))
        self.label_8.setText(_translate("Form", "Температура перед турбиной, К"))
        self.label_10.setText(_translate("Form", "Температура торможения после турбины, К"))
        self.label_gamma2.setText(_translate("Form", "Угол наклона средней линии, град"))
        self.label_15.setText(_translate("Form", "Расход рабочего тела на входе в турбину, кг/с"))
        self.label_16.setText(_translate("Form", "Частота вращения, об/мин"))
        self.label_12.setText(_translate("Form", "Теплоперепад на турбине по параметрам торможения, кДж/кг"))
        self.label_20.setText(_translate("Form", "Механический КПД"))
        self.label_6.setText(_translate("Form", "Коэффициент избытка воздуха"))
        self.label_gamma1.setText(_translate("Form", "Угол раскрытия проточной части, град"))
        self.label_7.setText(_translate("Form", "Скорость на выходе их РК первой ступени, м/с"))
        self.label_13.setText(_translate("Form", "Теплоперепад на СА первой ступени, кДж/кг"))
        self.label_14.setText(_translate("Form", "КПД турбины по параметрам торможения"))
        self.label_17.setText(_translate("Form", "Относительная длина лопатки на входе в РК первой ступени"))
        self.label.setText(_translate("Form", "Удельная работа турбины, кДж/кг"))
        self.groupBox.setTitle(_translate("Form", "Настройки расчета"))
        self.label_30.setText(_translate("Form", "Способ задания формы проточной части"))
        self.checkBox_rho_auto.setText(_translate("Form", "Автонастройка степени реактивности"))
        self.checkBox_h0_auto.setText(_translate("Form", "Автонастройка теплоперепада по ступеням"))
        self.turbine_type.setItemText(0, _translate("Form", "Силовая"))
        self.turbine_type.setItemText(1, _translate("Form", "Компрессорная"))
        self.label_29.setText(_translate("Form", "Число ступеней"))
        self.label_28.setText(_translate("Form", "Тип турбины"))
        self.gamma_sum_av.setText(_translate("Form", "Угол раскрытия проточной части и угол наклона средней линии"))
        self.gamma_in_out.setText(_translate("Form", "Угол наклона периферии и втулки"))
        self.checkBox_precise_h0.setText(_translate("Form", "Уточнение теплоперепада по ступеням"))
        self.label_18.setText(_translate("Form", "Расчетные параметры"))
        self.label_23.setText(_translate("Form", "КПД по статическим параметрам"))
        self.label_22.setText(_translate("Form", "Статический теплоперепад, кДж/кг"))
        self.label_25.setText(_translate("Form", "Лопаточный КПД"))
        self.label_24.setText(_translate("Form", "КПД по параметрам торможения"))
        self.label_21.setText(_translate("Form", "Теплоперепад по параметрам торможения, кДж/кг"))
        self.label_19.setText(_translate("Form", "Удельная работа турбины, кДж/кг"))
        self.label_26.setText(_translate("Form", "Средняя теплоемкость, Дж/(кг*К)"))
        self.label_27.setText(_translate("Form", "Средний показатель адиабаты"))
        self.compute_btn.setText(_translate("Form", "Рассчитать"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_init_data), _translate("Form", "Интегральные параметры"))
        self.prevPage_btn.setText(_translate("Form", "Предыдущая ступень"))
        self.nextPage_btn.setText(_translate("Form", "Следующая ступень"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_stage_data), _translate("Form", "Данные по ступеням"))

