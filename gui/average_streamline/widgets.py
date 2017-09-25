from PyQt5 import QtWidgets
import sys
import numpy as np
from core.average_streamline.turbine import TurbineType, Turbine
from gui.average_streamline.main_form import Ui_Form
import gui.average_streamline.stage_data_form as stage_data_form
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.axes import Axes


class Canvas(FigureCanvas):
    def __init__(self, parent: QtWidgets.QWidget =None):
        fig = Figure()
        self.axes: Axes = fig.add_subplot(111)
        FigureCanvas.__init__(self, fig)
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
        self.axes.set_title(title, fontsize=10)
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
        self.axes.set_title('Предварительное распределение\n теплоперепадов', fontsize=10)
        self.draw()


class StageDataWidget(QtWidgets.QWidget, stage_data_form.Ui_Form):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setupUi(self)
        self.triangle_canvas = Canvas(self)
        self.horizontalLayout_triangle_plot.addWidget(self.triangle_canvas)


class AveLineWidget(QtWidgets.QWidget, Ui_Form):
    def __init__(self):
        QtWidgets.QWidget.__init__(self)
        self.setupUi(self)
        self.heat_drop_canvas = Canvas(self)
        self.tab_init_data.layout().addWidget(self.heat_drop_canvas)
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
            for i in range(self.stackedWidget.count()):
                widget: StageDataWidget = self.stackedWidget.widget(i)
                widget.H0.setVisible(False)
                widget.label_H0.setVisible(False)
        else:
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
            stage_form.triangle_canvas.plot_velocity_triangle('Треугольник скоростей', turbine[i].c1_u,
                                                              turbine[i].c1_a, turbine[i].u1, turbine[i].c2_u,
                                                              turbine[i].c2_a, turbine[i].u2)

    def get_turbine(self):
        if self.turbine_type.currentIndex() == 0:
            turbine_type = TurbineType.Power
        else:
            turbine_type = TurbineType.Compressor
        if self.gamma_sum_av.isChecked():
            turbine = Turbine(turbine_type, gamma_sum=np.radians(self.gamma1.value()),
                              gamma_av=np.radians(self.gamma2.value()))
        else:
            turbine = Turbine(turbine_type, gamma_out=np.radians(self.gamma1.value()),
                              gamma_in=np.radians(self.gamma2.value()))
        turbine.alpha11 = np.radians(self.alpha11.value())
        turbine.alpha_air = self.alpha_air.value()
        turbine.c21_init = self.c21.value()
        turbine.T_g_stag = self.T_g_stag.value()
        turbine.p_g_stag = self.p_g_stag.value() * 1e6
        turbine.T_t_stag_cycle = self.T_t_stag_cycle.value()
        turbine.p_t_stag_cycle = self.p_t_stag_cycle.value() * 1e6
        turbine.H_t_stag_cycle = self.H_t_stag_cycle.value() * 1e3
        turbine.L_t_cycle = self.L_t_cycle.value() * 1e3
        turbine.H01_init = self.H01_init.value() * 1e3
        turbine.eta_t_stag_cycle = self.eta_t_stag_cycle.value()
        turbine.G_turbine = self.G_t.value()
        turbine.n = self.n.value()
        turbine.l1_D1_ratio = self.l1_D1_ratio.value()
        turbine.eta_m = self.eta_m.value()
        turbine.stage_number = self.stage_number.value()
        auto_set_rho = self.checkBox_rho_auto.isChecked()
        compute_heat_drop_auto = self.checkBox_h0_auto.isChecked()
        precise_heat_drop = self.checkBox_precise_h0.isChecked()
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
            if not auto_set_rho:
                turbine.geom[i].rho = stage_data.rho.value()
            if not compute_heat_drop_auto:
                turbine.geom[i].H0 = stage_data.H0.value() * 1e3
        return turbine, compute_heat_drop_auto, precise_heat_drop, auto_set_rho

    def on_compute_btn_click(self):
        turbine, compute_heat_drop_auto, precise_heat_drop, auto_set_rho = self.get_turbine()
        turbine.compute_geometry(compute_heat_drop_auto, auto_set_rho)
        turbine.compute_stages_gas_dynamics(precise_heat_drop)
        turbine.compute_integrate_turbine_parameters()
        self._set_output(turbine)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = AveLineWidget()
    window.show()
    sys.exit(app.exec_())