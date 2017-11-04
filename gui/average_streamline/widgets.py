from PyQt5 import QtWidgets
import sys
import pickle as pc
import numpy as np
from gas_turbine_cycle.gases import KeroseneCombustionProducts, NaturalGasCombustionProducts
from core.average_streamline.stage_geom import TurbineGeomAndHeatDropDistribution, StageGeomAndHeatDrop
from core.average_streamline.turbine import TurbineType, Turbine
from gui.average_streamline.main_form import Ui_Form
import gui.average_streamline.stage_data_form as stage_data_form
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from gas_turbine_cycle.tools.functions import create_logger
import os
from PyQt5.QtWidgets import QFileDialog
import gui.average_streamline.main_window_sdi_form as main_window_sdi_form
import logging

logger = create_logger(__name__, filename=os.path.join(os.path.dirname(__file__), 'error.log'),
                       loggerlevel=logging.ERROR,
                       add_file_handler=True, filemode='w',
                       add_console_handler=False, add_datetime=True, add_module_name=True)


class Canvas(FigureCanvas):
    def __init__(self, parent: QtWidgets.QWidget =None):
        fig = Figure(figsize=(5, 4))
        self.axes: Axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.setParent(parent)

    def plot_velocity_triangle(self, title: str, c1_u, c1_a, u1, c2_u, c2_a, u2):
        self.axes.cla()
        x_in = np.array([0, -c1_u, -c1_u + u1, 0])
        y_in = np.array([c1_a, 0, 0, c1_a])
        x_out = np.array([0, c2_u, c2_u + u2, 0])
        y_out = np.array([c1_a, c1_a - c2_a, c1_a - c2_a, c1_a])
        self.axes.plot(x_in, y_in, linewidth=2, color='red', label='inlet')
        self.axes.plot(x_out, y_out, linewidth=2, color='blue', label='outlet')
        self.axes.set_xlim(-c1_u, c2_u + u2)
        self.axes.set_ylim(-max(c1_a, c1_u), max(c1_a, c2_u + u2))
        self.axes.set_title(title, fontsize=9)
        self.axes.legend(fontsize=8)
        self.axes.grid()
        self.draw()

    def plot_heat_drop_distribution(self, stage_number, H0_arr):
        self.axes.cla()
        x_arr = list(range(1, stage_number + 1))
        self.axes.plot(x_arr, H0_arr, 'o', color='red', markersize=12)
        self.axes.plot(x_arr, H0_arr, color='red')
        self.axes.grid()
        self.axes.set_xticks(x_arr, x_arr)
        self.axes.set_xlim(0.7, stage_number + 0.3)
        self.axes.set_ylim(0, max(H0_arr) + 2e-2)
        self.axes.set_ylabel(r'$H_i,\ МДж/кг$', fontsize=8)
        self.axes.set_xlabel(r'$Stage\ number$', fontsize=8)
        self.axes.set_title('Распределение\n теплоперепадов', fontsize=9)
        self.draw()

    def plot_geometry(self, turbine_geom: TurbineGeomAndHeatDropDistribution):
        self.axes.cla()
        for num, geom in zip(range(len(turbine_geom)), turbine_geom):
            if num == 0:
                geom.x0 = 0
                geom.y0 = 0.5 * geom.D0 + 0.5 * (geom.l0 + geom.p_r_out + geom.p_r_in) - \
                             geom.p_r_out
            else:
                geom.x0 = turbine_geom[num - 1].x0 + turbine_geom[num - 1].delta_x0
                geom.y0 = turbine_geom[num - 1].y0 + turbine_geom[num - 1].delta_y0

            x_sa_arr = np.array([geom.x0, geom.x0, geom.x0 + geom.b_sa, geom.x0 + geom.b_sa, geom.x0])
            y_sa_arr = np.array([geom.y0 - geom.l0,
                                 geom.y0,
                                 geom.y0 + geom.b_sa * np.tan(geom.gamma_out),
                                 geom.y0 + geom.b_sa * np.tan(geom.gamma_out) - geom.l05,
                                 geom.y0 - geom.l0])
            x0_rk = geom.x0 + geom.b_sa + geom.delta_a_sa
            y0_rk = geom.y0 + np.tan(geom.gamma_out) * (geom.b_sa + geom.delta_a_sa) + geom.p_r_out
            x_rk_arr = np.array([x0_rk, x0_rk, x0_rk + geom.b_rk, x0_rk + geom.b_rk, x0_rk])
            y_rk_arr = np.array([y0_rk - geom.l1,
                                 y0_rk - geom.delta_r_rk,
                                 y0_rk - geom.delta_r_rk + np.tan(geom.gamma_out) * geom.b_rk,
                                 y0_rk + np.tan(geom.gamma_out) * geom.b_rk - geom.l2,
                                 y0_rk - geom.l1])
            x_out_arr = np.array([geom.x0, geom.x0 + geom.b_sa + geom.delta_a_sa - geom.p_a_out,
                                  geom.x0 + geom.b_sa + geom.delta_a_sa - geom.p_a_out,
                                  x0_rk + geom.b_rk + geom.delta_a_rk])
            x_av_arr = np.array([geom.x0, geom.x0 + geom.b_sa, x0_rk + geom.b_rk + geom.delta_a_rk])
            y_out_arr = np.array([geom.y0,
                                  geom.y0 + np.tan(geom.gamma_out) * (geom.b_sa + geom.delta_a_sa - geom.p_a_out),
                                  geom.y0 + np.tan(geom.gamma_out) * (geom.b_sa + geom.delta_a_sa - geom.p_a_out) +
                                  geom.p_r_out,
                                  geom.y0 + np.tan(geom.gamma_out) *
                                  (geom.b_sa + geom.delta_a_sa + geom.b_rk + geom.delta_a_rk) + geom.p_r_out])
            y_av_arr = np.array([0.5 * geom.D0, 0.5 * geom.D05,
                                 0.5 * geom.D2 + np.tan(geom.gamma_av) * geom.delta_a_rk])
            self.axes.plot(x_sa_arr, y_sa_arr, linewidth=1, color='red')
            self.axes.plot(x_rk_arr, y_rk_arr, linewidth=1, color='blue')
            self.axes.plot(x_out_arr, y_out_arr, linewidth=1, color='black')
            self.axes.plot(x_av_arr, y_av_arr, '--', linewidth=1, color='black')
        self.axes.grid()
        self.axes.set_title('Геометрия турбины', fontsize=9)
        self.axes.set_xlim(-0.01, turbine_geom[turbine_geom.stage_number - 1].x0 +
                           turbine_geom[turbine_geom.stage_number - 1].length + 0.01)
        self.axes.set_ylim(bottom=0)
        self.draw()


class StageDataWidget(QtWidgets.QWidget, stage_data_form.Ui_Form):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setupUi(self)
        self.triangle_canvas = Canvas(self)
        self.geometry_canvas = Canvas(self)
        self.verticalLayout_plot.addWidget(self.triangle_canvas)
        self.verticalLayout_plot.addWidget(self.geometry_canvas)


class AveLineWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setupUi(self)
        self.heat_drop_canvas = Canvas(self)
        self.geometry_canvas = Canvas(self)
        self.verticalLayout_plot.addWidget(self.heat_drop_canvas)
        self.verticalLayout_plot.addWidget(self.geometry_canvas)
        stage_widget = StageDataWidget()
        stage_widget.H0.setVisible(False)
        stage_widget.rho.setVisible(False)
        stage_widget.label_H0.setVisible(False)
        stage_widget.label_rho.setVisible(False)
        self.stackedWidget.addWidget(stage_widget)
        self.change_rho_auto()
        self.change_heat_drop_auto()
        self.prevPage_btn.clicked.connect(self.on_prev_page_btn_click)
        self.nextPage_btn.clicked.connect(self.on_next_page_btn_click)
        self.stage_number.valueChanged.connect(self.change_stage_number)
        self.gamma_sum_av.toggled.connect(self.change_flowing_channel_geom_setting_way)
        self.checkBox_rho_auto.stateChanged.connect(self.change_rho_auto)
        self.checkBox_h0_auto.stateChanged.connect(self.change_heat_drop_auto)
        self.compute_btn.clicked.connect(self.on_compute_btn_click)

    def on_prev_page_btn_click(self):
        cur_index = self.stackedWidget.currentIndex()
        if cur_index == 0:
            pass
        else:
            self.stackedWidget.setCurrentIndex(cur_index - 1)

    def on_next_page_btn_click(self):
        cur_index = self.stackedWidget.currentIndex()
        if cur_index == self.stackedWidget.count() - 1:
            pass
        else:
            self.stackedWidget.setCurrentIndex(cur_index + 1)

    def change_stage_number(self):
        new_stage_num = self.stage_number.value()
        old_stage_num = self.stackedWidget.count()
        if old_stage_num < new_stage_num:
            for i in range(old_stage_num, new_stage_num):
                widget = StageDataWidget()
                widget.label_stage_title.setText('Ступень %s' % (i + 1))
                self.stackedWidget.addWidget(widget)
                self.change_heat_drop_auto()
                self.change_rho_auto()
        elif new_stage_num < old_stage_num:
            while self.stackedWidget.count() != new_stage_num:
                self.stackedWidget.setCurrentIndex(self.stackedWidget.count() - 1)
                cur_page = self.stackedWidget.currentWidget()
                self.stackedWidget.removeWidget(cur_page)

    def change_flowing_channel_geom_setting_way(self):
        if self.gamma_sum_av.isChecked():
            self.label_gamma1.setText('Угол раскрытия проточной части, град')
            self.label_gamma2.setText('Угол наклона средней линии, град')
        else:
            self.label_gamma1.setText('Угол наклона периферийной поверхности, град')
            self.label_gamma2.setText('Угол наклона внутренней поверхности, град ')

    def change_rho_auto(self):
        if self.checkBox_rho_auto.isChecked():
            for i in range(self.stackedWidget.count()):
                widget: StageDataWidget = self.stackedWidget.widget(i)
                widget.rho.setVisible(False)
                widget.label_rho.setVisible(False)
        else:
            for i in range(self.stackedWidget.count()):
                widget: StageDataWidget = self.stackedWidget.widget(i)
                widget.rho.setVisible(True)
                widget.label_rho.setVisible(True)

    def change_heat_drop_auto(self):
        if self.checkBox_h0_auto.isChecked():
            self.H01_init.setVisible(True)
            self.c21.setVisible(True)
            self.label_c21.setVisible(True)
            self.label_H01_init.setVisible(True)
            for i in range(self.stackedWidget.count()):
                widget: StageDataWidget = self.stackedWidget.widget(i)
                widget.H0.setVisible(False)
                widget.label_H0.setVisible(False)
        else:
            self.H01_init.setVisible(False)
            self.c21.setVisible(False)
            self.label_c21.setVisible(False)
            self.label_H01_init.setVisible(False)
            for i in range(self.stackedWidget.count()):
                widget: StageDataWidget = self.stackedWidget.widget(i)
                widget.H0.setVisible(True)
                widget.label_H0.setVisible(True)

    def _set_output(self, turbine: Turbine):
        self.L_t_sum.setValue(turbine.L_t_sum / 1e3)
        self.H_t_stag.setValue(turbine.H_t_stag / 1e3)
        self.H_t.setValue(turbine.H_t / 1e3)
        self.eta_t.setValue(turbine.eta_t)
        self.eta_t_stag.setValue(turbine.eta_t_stag)
        self.eta_l.setValue(turbine.eta_l)
        self.c_p_gas_av.setValue(turbine.c_p_gas)
        self.k_gas_av.setValue(turbine.k_gas)
        H0_arr = [stage_geom.H0 / 1e6 for stage_geom in turbine.geom]
        self.heat_drop_canvas.plot_heat_drop_distribution(turbine.stage_number, H0_arr)
        self.geometry_canvas.plot_geometry(turbine.geom)
        for i in range(self.stackedWidget.count()):
            stage_data: StageDataWidget = self.stackedWidget.widget(i)
            stage_data.geometry_canvas.plot_geometry(turbine.geom)
        for i in range(self.stackedWidget.count()):
            stage_form: StageDataWidget = self.stackedWidget.widget(i)

            stage_form.gamma_sum.setValue(np.degrees(turbine.gamma_sum))
            stage_form.gamma_av.setValue(np.degrees(turbine.gamma_av))
            stage_form.gamma_in.setValue(np.degrees(turbine.gamma_in))
            stage_form.gamma_out.setValue(np.degrees(turbine.gamma_out))
            stage_form.D0.setValue(turbine.geom[i].D0)
            stage_form.l0.setValue(turbine.geom[i].l0)
            stage_form.D1.setValue(turbine.geom[i].D1)
            stage_form.l1.setValue(turbine.geom[i].l1)
            stage_form.D2.setValue(turbine.geom[i].D2)
            stage_form.l2.setValue(turbine.geom[i].l2)
            stage_form.b_sa.setValue(turbine.geom[i].b_sa)
            stage_form.b_rk.setValue(turbine.geom[i].b_rk)
            stage_form.delta_r.setValue(turbine.geom[i].delta_r_rk * 1e3)
            stage_form.p_in.setValue(turbine.geom[i].p_r_in * 1e3)
            stage_form.p_out.setValue(turbine.geom[i].p_r_out * 1e3)

            stage_form.c1.setValue(turbine[i].c1)
            stage_form.u1.setValue(turbine[i].u1)
            stage_form.c1_a.setValue(turbine[i].c1_a)
            stage_form.c1_u.setValue(turbine[i].c1_u)
            stage_form.alpha1.setValue(np.degrees(turbine[i].alpha1))
            stage_form.w1.setValue(turbine[i].w1)
            stage_form.w1_a.setValue(turbine[i].w1_a)
            stage_form.w1_u.setValue(turbine[i].w1_u)
            stage_form.beta1.setValue(np.degrees(turbine[i].beta1))
            stage_form.c2.setValue(turbine[i].c2)
            stage_form.u2.setValue(turbine[i].u2)
            stage_form.c2_a.setValue(turbine[i].c2_a)
            stage_form.c2_u.setValue(turbine[i].c2_u)
            stage_form.alpha2.setValue(np.degrees(turbine[i].alpha2))
            stage_form.w2.setValue(turbine[i].w2)
            stage_form.w2_a.setValue(turbine[i].w2_a)
            stage_form.w2_u.setValue(turbine[i].w2_u)
            stage_form.beta2.setValue(np.degrees(turbine[i].beta2))
            stage_form.H0_out.setValue(turbine[i].H0 / 1e3)
            stage_form.H0_stag.setValue(turbine[i].H0_stag / 1e3)
            stage_form.T1.setValue(turbine[i].T1)
            stage_form.T2.setValue(turbine[i].T2)
            stage_form.T_st.setValue(turbine[i].T_st)
            stage_form.T_st_stag.setValue(turbine[i].T_st_stag)
            stage_form.p1.setValue(turbine[i].p1 / 1e6)
            stage_form.p2.setValue(turbine[i].p2 / 1e6)
            stage_form.p2_stag.setValue(turbine[i].p2_stag / 1e6)
            stage_form.eta_u.setValue(turbine[i].eta_u)
            stage_form.eta_l.setValue(turbine[i].eta_l)
            stage_form.eta_t.setValue(turbine[i].eta_t)
            stage_form.eta_t_stag.setValue(turbine[i].eta_t_stag)
            stage_form.rho_out.setValue(turbine[i].rho)

            stage_form.H_s.setValue(turbine[i].H_s / 1e3)
            stage_form.H_l.setValue(turbine[i].H_l / 1e3)
            stage_form.T1_ad.setValue(turbine[i].T1_ad)
            stage_form.rho1.setValue(turbine[i].rho1)
            stage_form.rho2.setValue(turbine[i].rho2)
            stage_form.L_st.setValue(turbine[i].L_t / 1e3)
            stage_form.L_st_rel.setValue(turbine[i].L_t_rel / 1e3)
            stage_form.L_u.setValue(turbine[i].L_u / 1e3)
            stage_form.k_av.setValue(turbine[i].k_gas)
            stage_form.c_p_av.setValue(turbine[i].c_p_gas)
            stage_form.G_in.setValue(turbine[i].G_stage_in)
            stage_form.G_out.setValue(turbine[i].G_stage_out)

            stage_form.triangle_canvas.plot_velocity_triangle('Треугольник скоростей', turbine[i].c1_u,
                                                              turbine[i].c1_a, turbine[i].u1, turbine[i].c2_u,
                                                              turbine[i].c2_a, turbine[i].u2)

    def get_turbine(self):
        auto_set_rho = self.checkBox_rho_auto.isChecked()
        auto_compute_heat_drop = self.checkBox_h0_auto.isChecked()
        precise_heat_drop = self.checkBox_precise_h0.isChecked()
        if self.turbine_type.currentIndex() == 0:
            turbine_type = TurbineType.Power
        else:
            turbine_type = TurbineType.Compressor

        if self.fuel.currentIndex() == 0:
            work_fluid = KeroseneCombustionProducts()
        elif self.fuel.currentIndex() == 1:
            work_fluid = NaturalGasCombustionProducts()
        else:
            work_fluid = KeroseneCombustionProducts()

        H0_list = []
        rho_list = []
        for i in range(self.stage_number.value()):
            stage_data: StageDataWidget = self.stackedWidget.widget(i)
            if not auto_set_rho:
                rho_list.append(stage_data.rho.value())
            if not auto_compute_heat_drop:
                H0_list.append(stage_data.H0.value() * 1e3)

        if self.gamma_sum_av.isChecked():
            if auto_compute_heat_drop and auto_set_rho:
                turbine = Turbine(turbine_type,
                                  stage_number=self.stage_number.value(),
                                  T_g_stag=self.T_g_stag.value(),
                                  p_g_stag=self.p_g_stag.value() * 1e6,
                                  G_turbine=self.G_t.value(),
                                  work_fluid=work_fluid,
                                  alpha_air=self.alpha_air.value(),
                                  l1_D1_ratio=self.l1_D1_ratio.value(),
                                  n=self.n.value(),
                                  eta_m=self.eta_m.value(),
                                  T_t_stag_cycle=self.T_t_stag_cycle.value(),
                                  eta_t_stag_cycle=self.eta_t_stag_cycle.value(),
                                  auto_compute_heat_drop=auto_compute_heat_drop,
                                  auto_set_rho=auto_set_rho,
                                  H01_init=self.H01_init.value()*1e3,
                                  c21_init=self.c21.value(),
                                  alpha11=np.radians(self.alpha11.value()),
                                  gamma_sum=np.radians(self.gamma1.value()),
                                  gamma_av=np.radians(self.gamma2.value()))
            if auto_compute_heat_drop and not auto_set_rho:
                turbine = Turbine(turbine_type,
                                  stage_number=self.stage_number.value(),
                                  T_g_stag=self.T_g_stag.value(),
                                  p_g_stag=self.p_g_stag.value() * 1e6,
                                  G_turbine=self.G_t.value(),
                                  work_fluid=work_fluid,
                                  alpha_air=self.alpha_air.value(),
                                  l1_D1_ratio=self.l1_D1_ratio.value(),
                                  n=self.n.value(),
                                  eta_m=self.eta_m.value(),
                                  T_t_stag_cycle=self.T_t_stag_cycle.value(),
                                  eta_t_stag_cycle=self.eta_t_stag_cycle.value(),
                                  auto_compute_heat_drop=auto_compute_heat_drop,
                                  auto_set_rho=auto_set_rho,
                                  rho_list=rho_list,
                                  H01_init=self.H01_init.value() * 1e3,
                                  c21_init=self.c21.value(),
                                  alpha11=np.radians(self.alpha11.value()),
                                  gamma_sum=np.radians(self.gamma1.value()),
                                  gamma_av=np.radians(self.gamma2.value()))
            if not auto_compute_heat_drop and auto_set_rho:
                turbine = Turbine(turbine_type,
                                  stage_number=self.stage_number.value(),
                                  T_g_stag=self.T_g_stag.value(),
                                  p_g_stag=self.p_g_stag.value() * 1e6,
                                  G_turbine=self.G_t.value(),
                                  work_fluid=work_fluid,
                                  alpha_air=self.alpha_air.value(),
                                  l1_D1_ratio=self.l1_D1_ratio.value(),
                                  n=self.n.value(),
                                  eta_m=self.eta_m.value(),
                                  T_t_stag_cycle=self.T_t_stag_cycle.value(),
                                  eta_t_stag_cycle=self.eta_t_stag_cycle.value(),
                                  auto_compute_heat_drop=auto_compute_heat_drop,
                                  auto_set_rho=auto_set_rho,
                                  H0_list=H0_list,
                                  alpha11=np.radians(self.alpha11.value()),
                                  gamma_sum=np.radians(self.gamma1.value()),
                                  gamma_av=np.radians(self.gamma2.value()))
            if not auto_compute_heat_drop and not auto_set_rho:
                turbine = Turbine(turbine_type,
                                  stage_number=self.stage_number.value(),
                                  T_g_stag=self.T_g_stag.value(),
                                  p_g_stag=self.p_g_stag.value() * 1e6,
                                  G_turbine=self.G_t.value(),
                                  work_fluid=work_fluid,
                                  alpha_air=self.alpha_air.value(),
                                  l1_D1_ratio=self.l1_D1_ratio.value(),
                                  n=self.n.value(),
                                  eta_m=self.eta_m.value(),
                                  T_t_stag_cycle=self.T_t_stag_cycle.value(),
                                  eta_t_stag_cycle=self.eta_t_stag_cycle.value(),
                                  auto_compute_heat_drop=auto_compute_heat_drop,
                                  auto_set_rho=auto_set_rho,
                                  rho_list=rho_list,
                                  H0_list=H0_list,
                                  alpha11=np.radians(self.alpha11.value()),
                                  gamma_sum=np.radians(self.gamma1.value()),
                                  gamma_av=np.radians(self.gamma2.value()))
        else:
            if auto_compute_heat_drop and auto_set_rho:
                turbine = Turbine(turbine_type,
                                  stage_number=self.stage_number.value(),
                                  T_g_stag=self.T_g_stag.value(),
                                  p_g_stag=self.p_g_stag.value() * 1e6,
                                  G_turbine=self.G_t.value(),
                                  work_fluid=work_fluid,
                                  alpha_air=self.alpha_air.value(),
                                  l1_D1_ratio=self.l1_D1_ratio.value(),
                                  n=self.n.value(),
                                  eta_m=self.eta_m.value(),
                                  T_t_stag_cycle=self.T_t_stag_cycle.value(),
                                  eta_t_stag_cycle=self.eta_t_stag_cycle.value(),
                                  auto_compute_heat_drop=auto_compute_heat_drop,
                                  auto_set_rho=auto_set_rho,
                                  H01_init=self.H01_init.value()*1e3,
                                  c21_init=self.c21.value(),
                                  alpha11=np.radians(self.alpha11.value()),
                                  gamma_out=np.radians(self.gamma1.value()),
                                  gamma_in=np.radians(self.gamma2.value()))
            if auto_compute_heat_drop and not auto_set_rho:
                turbine = Turbine(turbine_type,
                                  stage_number=self.stage_number.value(),
                                  T_g_stag=self.T_g_stag.value(),
                                  p_g_stag=self.p_g_stag.value() * 1e6,
                                  G_turbine=self.G_t.value(),
                                  work_fluid=work_fluid,
                                  alpha_air=self.alpha_air.value(),
                                  l1_D1_ratio=self.l1_D1_ratio.value(),
                                  n=self.n.value(),
                                  eta_m=self.eta_m.value(),
                                  T_t_stag_cycle=self.T_t_stag_cycle.value(),
                                  eta_t_stag_cycle=self.eta_t_stag_cycle.value(),
                                  auto_compute_heat_drop=auto_compute_heat_drop,
                                  auto_set_rho=auto_set_rho,
                                  rho_list=rho_list,
                                  H01_init=self.H01_init.value() * 1e3,
                                  c21_init=self.c21.value(),
                                  alpha11=np.radians(self.alpha11.value()),
                                  gamma_out=np.radians(self.gamma1.value()),
                                  gamma_in=np.radians(self.gamma2.value()))
            if not auto_compute_heat_drop and auto_set_rho:
                turbine = Turbine(turbine_type,
                                  stage_number=self.stage_number.value(),
                                  T_g_stag=self.T_g_stag.value(),
                                  p_g_stag=self.p_g_stag.value() * 1e6,
                                  G_turbine=self.G_t.value(),
                                  work_fluid=work_fluid,
                                  alpha_air=self.alpha_air.value(),
                                  l1_D1_ratio=self.l1_D1_ratio.value(),
                                  n=self.n.value(),
                                  eta_m=self.eta_m.value(),
                                  T_t_stag_cycle=self.T_t_stag_cycle.value(),
                                  eta_t_stag_cycle=self.eta_t_stag_cycle.value(),
                                  auto_compute_heat_drop=auto_compute_heat_drop,
                                  auto_set_rho=auto_set_rho,
                                  H0_list=H0_list,
                                  alpha11=np.radians(self.alpha11.value()),
                                  gamma_out=np.radians(self.gamma1.value()),
                                  gamma_in=np.radians(self.gamma2.value()))
            if not auto_compute_heat_drop and not auto_set_rho:
                turbine = Turbine(turbine_type,
                                  stage_number=self.stage_number.value(),
                                  T_g_stag=self.T_g_stag.value(),
                                  p_g_stag=self.p_g_stag.value() * 1e6,
                                  G_turbine=self.G_t.value(),
                                  work_fluid=work_fluid,
                                  alpha_air=self.alpha_air.value(),
                                  l1_D1_ratio=self.l1_D1_ratio.value(),
                                  n=self.n.value(),
                                  eta_m=self.eta_m.value(),
                                  T_t_stag_cycle=self.T_t_stag_cycle.value(),
                                  eta_t_stag_cycle=self.eta_t_stag_cycle.value(),
                                  auto_compute_heat_drop=auto_compute_heat_drop,
                                  auto_set_rho=auto_set_rho,
                                  rho_list=rho_list,
                                  H0_list=H0_list,
                                  alpha11=np.radians(self.alpha11.value()),
                                  gamma_out=np.radians(self.gamma1.value()),
                                  gamma_in=np.radians(self.gamma2.value()))
        for i in range(self.stage_number.value()):
            stage_data: StageDataWidget = self.stackedWidget.widget(i)
            turbine.geom[i].l1_b_sa_ratio = stage_data.l1_b_sa_ratio.value()
            turbine.geom[i].l2_b_rk_ratio = stage_data.l2_b_rk_ratio.value()
            turbine.geom[i].delta_a_b_rk_ratio = stage_data.delta_a_b_rk_ratio.value()
            turbine.geom[i].delta_a_b_sa_ratio = stage_data.delta_a_b_sa_ratio.value()
            turbine.geom[i].delta_r_rk_l2_ratio = stage_data.delta_r_rel.value()
            turbine.geom[i].phi = stage_data.phi.value()
            turbine.geom[i].psi = stage_data.psi.value()
            turbine.geom[i].epsilon = stage_data.epsilon.value()
            turbine.geom[i].mu = stage_data.mu.value()
            turbine.geom[i].g_lk = stage_data.g_lk.value()
            turbine.geom[i].g_lb = stage_data.g_lb.value()
            turbine.geom[i].g_ld = stage_data.g_ld.value()
            turbine.geom[i].p_r_in_l1_ratio = stage_data.p_r_in_l1_ratio.value()
            turbine.geom[i].p_a_in_rel = stage_data.p_a_in_rel.value()
            turbine.geom[i].p_r_out_l1_ratio = stage_data.p_r_out_l1_ratio.value()
            turbine.geom[i].p_a_out_rel = stage_data.p_a_out_rel.value()
        return turbine, precise_heat_drop

    def show_error_message(self, message):
        err_message = QtWidgets.QMessageBox(self)
        err_message.setIcon(QtWidgets.QMessageBox.Warning)
        err_message.setWindowTitle('Error')
        err_message.setText('An error occurred during the calculation')
        err_message.setDetailedText('Exception message:\n%s' % message)
        err_message.show()

    def on_compute_btn_click(self):
        turbine, precise_heat_drop = self.get_turbine()
        try:
            turbine.compute_geometry()
            turbine.compute_stages_gas_dynamics(precise_heat_drop)
            turbine.compute_integrate_turbine_parameters()
            self._set_output(turbine)
        except Exception as ex:
            logger.error(ex)
            self.show_error_message(str(ex))

    def save_turbine_file(self, fname):
        turbine, precise_heat_drop = self.get_turbine()
        turbine.precise_heat_drop = precise_heat_drop
        file = open(fname, 'wb')
        pc.dump(turbine, file)
        file.close()

    def set_input_from_turbine_file(self, fname):
        file = open(fname, 'rb')
        turbine: Turbine = pc.load(file)
        file.close()
        precise_heat_drop = turbine.precise_heat_drop

        self.stage_number.setValue(turbine.stage_number)

        if turbine.turbine_type == TurbineType.Power:
            self.turbine_type.setCurrentIndex(0)
        elif turbine.turbine_type == TurbineType.Compressor:
            self.turbine_type.setCurrentIndex(1)

        if type(turbine.work_fluid) == KeroseneCombustionProducts:
            self.fuel.setCurrentIndex(0)
        elif type(turbine.work_fluid) == NaturalGasCombustionProducts:
            self.fuel.setCurrentIndex(1)
        if turbine.gamma_sum is not None and turbine.gamma_av is not None:
            self.gamma_sum_av.setChecked(True)
            self.gamma1.setValue(np.degrees(turbine.gamma_sum))
            self.gamma2.setValue(np.degrees(turbine.gamma_av))
        else:
            self.gamma_in_out.setChecked(True)
            self.gamma1.setValue(np.degrees(turbine.gamma_out))
            self.gamma2.setValue(np.degrees(turbine.gamma_in))

        if turbine.auto_set_rho:
            self.checkBox_rho_auto.setChecked(True)
        else:
            self.checkBox_rho_auto.setChecked(False)
            for i in range(self.stage_number.value()):
                stage_data: StageDataWidget = self.stackedWidget.widget(i)
                stage_data.rho.setValue(turbine.geom[i].rho)

        if turbine.auto_compute_heat_drop:
            self.checkBox_h0_auto.setChecked(True)
            self.H01_init.setValue(turbine.H01_init / 1e3)
            self.c21.setValue(turbine.c21_init)
        else:
            self.checkBox_h0_auto.setChecked(False)
            for i in range(self.stage_number.value()):
                stage_data: StageDataWidget = self.stackedWidget.widget(i)
                stage_data.H0.setValue(turbine.geom[i].H0 / 1e3)

        if precise_heat_drop:
            self.checkBox_precise_h0.setChecked(True)
        else:
            self.checkBox_precise_h0.setChecked(False)

        self.alpha11.setValue(np.degrees(turbine.alpha11))
        self.alpha_air.setValue(turbine.alpha_air)
        self.T_g_stag.setValue(turbine.T_g_stag)
        self.p_g_stag.setValue(turbine.p_g_stag / 1e6)
        self.T_t_stag_cycle.setValue(turbine.T_t_stag_cycle)
        self.eta_t_stag_cycle.setValue(turbine.eta_t_stag_cycle)
        self.G_t.setValue(turbine.G_turbine)
        self.n.setValue(turbine.n)
        self.l1_D1_ratio.setValue(turbine.l1_D1_ratio)
        self.eta_m.setValue(turbine.eta_m)

        for i in range(self.stage_number.value()):
            stage_data: StageDataWidget = self.stackedWidget.widget(i)

            stage_data.l1_b_sa_ratio.setValue(turbine.geom[i].l1_b_sa_ratio)
            stage_data.l2_b_rk_ratio.setValue(turbine.geom[i].l2_b_rk_ratio)
            stage_data.delta_a_b_rk_ratio.setValue(turbine.geom[i].delta_a_b_rk_ratio)
            stage_data.delta_a_b_sa_ratio.setValue(turbine.geom[i].delta_a_b_sa_ratio)
            stage_data.delta_r_rel.setValue(turbine.geom[i].delta_r_rk_l2_ratio)
            stage_data.phi.setValue(turbine.geom[i].phi)
            stage_data.psi.setValue(turbine.geom[i].psi)
            stage_data.epsilon.setValue(turbine.geom[i].epsilon)
            stage_data.mu.setValue(turbine.geom[i].mu)
            stage_data.g_lk.setValue(turbine.geom[i].g_lk)
            stage_data.g_ld.setValue(turbine.geom[i].g_lb)
            stage_data.g_ld.setValue(turbine.geom[i].g_ld)
            stage_data.p_r_in_l1_ratio.setValue(turbine.geom[i].p_r_in_l1_ratio)
            stage_data.p_r_out_l1_ratio.setValue(turbine.geom[i].p_r_out_l1_ratio)
            stage_data.p_a_in_rel.setValue(turbine.geom[i].p_a_in_rel)
            stage_data.p_a_out_rel.setValue(turbine.geom[i].p_a_out_rel)


class AveStreamLineMainWindow(QtWidgets.QMainWindow, main_window_sdi_form.Ui_MainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        self.save_count = 0
        self.setupUi(self)
        self.setCentralWidget(AveLineWidget())
        self.act_exit.triggered.connect(self.close)
        self.act_new.triggered.connect(self.on_new_action)
        self.act_open.triggered.connect(self.on_open_action)
        self.act_save_as.triggered.connect(self.on_save_as_action)
        self.act_save.triggered.connect(self.on_save_action)

    def on_new_action(self):
        self.setCentralWidget(AveLineWidget())
        self.setWindowTitle('Unnamed')
        self.save_count = 0

    def on_open_action(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'Открыть файл', os.getcwd(),
                                               'Turbine AveLine files  (*%s)' % 'avl')

        if fname:
            ave_line_widget: AveLineWidget = self.centralWidget()
            ave_line_widget.set_input_from_turbine_file(fname)
            self.setWindowTitle(fname)
            self.save_count = 1

    def on_save_as_action(self):
        fname, _ = QFileDialog.getSaveFileName(self, 'Сохранить как...', os.getcwd(),
                                               'Turbine AveLine files (*%s)' % 'avl')
        if fname:
            ave_line_widget: AveLineWidget = self.centralWidget()
            ave_line_widget.save_turbine_file(fname)
            self.setWindowTitle(fname)
            self.save_count += 1

    def on_save_action(self):
        if self.save_count > 0:
            fname = self.windowTitle()
            ave_line_widget: AveLineWidget = self.centralWidget()
            ave_line_widget.save_turbine_file(fname)
        else:
            fname, _ = QFileDialog.getSaveFileName(self, 'Сохранить как...', os.getcwd(),
                                                   'Turbine AveLine files (*%s)' % 'avl')
            if fname:
                ave_line_widget: AveLineWidget = self.centralWidget()
                ave_line_widget.save_turbine_file(fname)
                self.setWindowTitle(fname)
                self.save_count += 1

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = AveStreamLineMainWindow()
    window.show()
    sys.exit(app.exec_())