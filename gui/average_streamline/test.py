import unittest
from PyQt5.QtWidgets import QApplication
from PyQt5.QtTest import QTest
from PyQt5.QtCore import Qt
from gui.average_streamline.widgets import AveLineWidget, StageDataWidget
from core.average_streamline.turbine import Turbine, TurbineType
import sys
from gas_turbine_cycle.gases import NaturalGasCombustionProducts
import numpy as np

app = QApplication(sys.argv)


class AveLineWidgetTest(unittest.TestCase):
    def setUp(self):
        self.form = AveLineWidget()

    def test_increase_stage_number(self):
        self.form.stage_number.setValue(3)
        count = self.form.stackedWidget.count()
        self.assertEqual(3, count)
        self.form.stage_number.setValue(1)

    def test_decrease_stage_number(self):
        self.form.stage_number.setValue(3)
        self.assertEqual(3, self.form.stackedWidget.count())
        self.form.stage_number.setValue(2)
        self.assertEqual(2, self.form.stackedWidget.count())
        self.form.stage_number.setValue(1)

    def test_change_rho_auto_to_unchecked(self):
        self.form.stage_number.setValue(4)
        self.form.checkBox_rho_auto.setChecked(False)
        for i in range(4):
            stage_data_widget: StageDataWidget = self.form.stackedWidget.widget(i)
            self.assertFalse(stage_data_widget.label_rho.isHidden())
            self.assertFalse(stage_data_widget.rho.isHidden())
        self.form.stage_number.setValue(1)

    def test_change_rho_auto_to_checked(self):
        self.form.stage_number.setValue(4)
        self.form.checkBox_rho_auto.setChecked(False)
        self.form.checkBox_rho_auto.setChecked(True)
        for i in range(4):
            stage_data_widget: StageDataWidget = self.form.stackedWidget.widget(i)
            self.assertTrue(stage_data_widget.label_rho.isHidden())
            self.assertTrue(stage_data_widget.rho.isHidden())
        self.form.stage_number.setValue(1)

    def test_change_heat_drop_auto_to_unchecked(self):
        self.form.stage_number.setValue(4)
        self.form.checkBox_h0_auto.setChecked(False)
        for i in range(4):
            stage_data_widget: StageDataWidget = self.form.stackedWidget.widget(i)
            self.assertFalse(stage_data_widget.label_H0.isHidden())
            self.assertFalse(stage_data_widget.H0.isHidden())
        self.form.stage_number.setValue(1)

    def test_change_heat_drop_auto_to_checked(self):
        self.form.stage_number.setValue(4)
        self.form.checkBox_h0_auto.setChecked(False)
        self.form.checkBox_h0_auto.setChecked(True)
        for i in range(4):
            stage_data_widget: StageDataWidget = self.form.stackedWidget.widget(i)
            self.assertTrue(stage_data_widget.label_H0.isHidden())
            self.assertTrue(stage_data_widget.H0.isHidden())
        self.form.stage_number.setValue(1)

    def test_compute_btn_click(self):
        """Тестируется правильность вывода значений в выходные поля"""
        turbine = self.form.get_turbine()
        turbine.compute_geometry()
        turbine.compute_stages_gas_dynamics()
        turbine.compute_integrate_turbine_parameters()
        QTest.mouseClick(self.form.compute_btn, Qt.LeftButton)
        self.assertNotEqual(self.form.turbine, None)
        self.assertAlmostEqual(self.form.eta_t.value(), turbine.eta_t, places=3)
        self.assertAlmostEqual(self.form.L_t_sum.value(), turbine.L_t_sum / 1e3, places=2)
        self.assertAlmostEqual(self.form.H_t_stag.value(), turbine.H_t_stag / 1e3, places=2)
        self.assertAlmostEqual(self.form.H_t.value(), turbine.H_t / 1e3, places=2)
        self.assertAlmostEqual(self.form.eta_t_stag.value(), turbine.eta_t_stag, places=3)
        self.assertAlmostEqual(self.form.eta_l.value(), turbine.eta_l, places=3)
        self.assertAlmostEqual(self.form.c_p_gas_av.value(), turbine.c_p_gas, places=2)
        self.assertAlmostEqual(self.form.k_gas_av.value(), turbine.k_gas, places=3)
        for i in range(self.form.stackedWidget.count()):
            stage_data: StageDataWidget = self.form.stackedWidget.widget(i)
            self.assertAlmostEqual(stage_data.gamma_sum.value(), np.degrees(turbine.geom.gamma_sum), places=2)
            self.assertAlmostEqual(stage_data.gamma_av.value(), np.degrees(turbine.geom.gamma_av), places=2)
            self.assertAlmostEqual(stage_data.gamma_in.value(), np.degrees(turbine.geom.gamma_in), places=2)
            self.assertAlmostEqual(stage_data.gamma_out.value(), np.degrees(turbine.geom.gamma_out), places=2)
            self.assertAlmostEqual(stage_data.D0.value(), turbine.geom[i].D0, places=3)
            self.assertAlmostEqual(stage_data.D1.value(), turbine.geom[i].D1, places=3)
            self.assertAlmostEqual(stage_data.D2.value(), turbine.geom[i].D2, places=3)
            self.assertAlmostEqual(stage_data.l0.value(), turbine.geom[i].l0, places=3)
            self.assertAlmostEqual(stage_data.l1.value(), turbine.geom[i].l1, places=3)
            self.assertAlmostEqual(stage_data.l2.value(), turbine.geom[i].l2, places=3)

            self.assertAlmostEqual(stage_data.delta_r.value(), turbine.geom[i].delta_r_rk * 1e3, places=2)
            self.assertAlmostEqual(stage_data.p_in.value(), turbine.geom[i].p_r_in * 1e3, places=2)
            self.assertAlmostEqual(stage_data.p_out.value(), turbine.geom[i].p_r_out * 1e3, places=2)

            self.assertAlmostEqual(stage_data.c1.value(), turbine[i].c1, places=2)
            self.assertAlmostEqual(stage_data.c1_a.value(), turbine[i].c1_a, places=2)
            self.assertAlmostEqual(stage_data.c1_u.value(), turbine[i].c1_u, places=2)
            self.assertAlmostEqual(stage_data.w1.value(), turbine[i].w1, places=2)
            self.assertAlmostEqual(stage_data.w1_u.value(), turbine[i].w1_u, places=2)
            self.assertAlmostEqual(stage_data.w1_a.value(), turbine[i].w1_a, places=2)
            self.assertAlmostEqual(stage_data.alpha1.value(), np.degrees(turbine[i].alpha1), places=2)
            self.assertAlmostEqual(stage_data.beta1.value(), np.degrees(turbine[i].beta1), places=2)

            self.assertAlmostEqual(stage_data.c2.value(), turbine[i].c2, places=2)
            self.assertAlmostEqual(stage_data.c2_a.value(), turbine[i].c2_a, places=2)
            self.assertAlmostEqual(stage_data.c2_u.value(), turbine[i].c2_u, places=2)
            self.assertAlmostEqual(stage_data.w2.value(), turbine[i].w2, places=2)
            self.assertAlmostEqual(stage_data.w2_u.value(), turbine[i].w2_u, places=2)
            self.assertAlmostEqual(stage_data.w2_a.value(), turbine[i].w2_a, places=2)
            self.assertAlmostEqual(stage_data.alpha2.value(), np.degrees(turbine[i].alpha2), places=2)
            self.assertAlmostEqual(stage_data.beta2.value(), np.degrees(turbine[i].beta2), places=2)

            self.assertAlmostEqual(stage_data.H0_out.value(), turbine[i].H0 / 1e3, places=2)
            self.assertAlmostEqual(stage_data.H0_stag.value(), turbine[i].H0_stag / 1e3, places=2)
            self.assertAlmostEqual(stage_data.T1.value(), turbine[i].T1, places=2)
            self.assertAlmostEqual(stage_data.T2.value(), turbine[i].T2, places=2)
            self.assertAlmostEqual(stage_data.T_st.value(), turbine[i].T_st, places=2)
            self.assertAlmostEqual(stage_data.T_st_stag.value(), turbine[i].T_st_stag, places=2)
            self.assertAlmostEqual(stage_data.p1.value(), turbine[i].p1 / 1e6, places=2)
            self.assertAlmostEqual(stage_data.p2.value(), turbine[i].p2 / 1e6, places=2)
            self.assertAlmostEqual(stage_data.p2_stag.value(), turbine[i].p2_stag / 1e6, places=2)
            self.assertAlmostEqual(stage_data.eta_t.value(), turbine[i].eta_t, places=3)
            self.assertAlmostEqual(stage_data.eta_t_stag.value(), turbine[i].eta_t_stag, places=3)
            self.assertAlmostEqual(stage_data.eta_l.value(), turbine[i].eta_l, places=3)
            self.assertAlmostEqual(stage_data.eta_u.value(), turbine[i].eta_u, places=3)
            self.assertAlmostEqual(stage_data.rho_out.value(), turbine[i].rho, places=3)

            self.assertAlmostEqual(stage_data.H_s.value(), turbine[i].H_s / 1e3, places=2)
            self.assertAlmostEqual(stage_data.H_l.value(), turbine[i].H_l / 1e3, places=2)
            self.assertAlmostEqual(stage_data.T1_ad.value(), turbine[i].T1_ad, places=2)
            self.assertAlmostEqual(stage_data.rho1.value(), turbine[i].rho1, places=2)
            self.assertAlmostEqual(stage_data.rho2.value(), turbine[i].rho2, places=2)
            self.assertAlmostEqual(stage_data.L_st.value(), turbine[i].L_t / 1e3, places=2)
            self.assertAlmostEqual(stage_data.L_st_rel.value(), turbine[i].L_t_rel / 1e3, places=2)
            self.assertAlmostEqual(stage_data.L_u.value(), turbine[i].L_u / 1e3, places=2)
            self.assertAlmostEqual(stage_data.k_av.value(), turbine[i].k_gas, places=2)
            self.assertAlmostEqual(stage_data.c_p_av.value(), turbine[i].c_p_gas, places=2)
            self.assertAlmostEqual(stage_data.G_in.value(), turbine[i].G_stage_in, places=2)
            self.assertAlmostEqual(stage_data.G_out.value(), turbine[i].G_stage_out, places=2)
            self.assertAlmostEqual(stage_data.T1_w_stag.value(), turbine[i].T1_w_stag, places=1)

    def test_setting_input_data_from_turbine(self):
        turbine = Turbine(turbine_type=TurbineType.Compressor,
                          stage_number=2,
                          T_g_stag=1450,
                          p_g_stag=400e3,
                          G_turbine=25,
                          work_fluid=NaturalGasCombustionProducts(),
                          alpha_air=2.6,
                          l1_D1_ratio=0.20,
                          n=15e3,
                          T_t_stag_cycle=1050,
                          eta_t_stag_cycle=0.91,
                          alpha11=np.radians([17])[0],
                          auto_set_rho=False,
                          rho_list=[0.35, 0.45],
                          auto_compute_heat_drop=True,
                          H01_init=220e3,
                          c21_init=250,
                          gamma_in=np.radians([7])[0],
                          gamma_out=np.radians([8])[0],
                          precise_heat_drop=False)

        turbine.geom[0].l1_b_sa_ratio = 2.1
        turbine.geom[0].l2_b_rk_ratio = 2.5
        turbine.geom[0].delta_a_b_sa_ratio = 0.4
        turbine.geom[0].delta_a_b_rk_ratio = 0.35
        turbine.geom[0].p_r_out_l1_ratio = 0.045
        turbine.geom[0].p_r_in_l1_ratio = 0.055
        turbine.geom[0].p_a_out_rel = 0.35
        turbine.geom[0].p_a_in_rel = 0.25
        turbine.geom[0].delta_r_rk_l2_ratio = 0.02
        turbine.geom[0].phi = 0.95
        turbine.geom[0].psi = 0.95
        turbine.geom[0].g_lb = 0.01
        turbine.geom[0].g_ld = 0.02
        turbine.geom[0].g_lk = 0.015
        turbine.geom[0].g_cool = 0.07

        self.form.set_input_from_turbine(turbine)

        self.assertEqual(turbine.stage_number, self.form.stage_number.value())
        self.assertEqual(self.form.turbine_type.currentIndex(), 1)
        self.assertEqual(self.form.fuel.currentIndex(), 1)
        self.assertTrue(self.form.gamma_in_out.isChecked())
        self.assertAlmostEqual(self.form.gamma1.value(), np.degrees(turbine.gamma_out), places=1)
        self.assertAlmostEqual(self.form.gamma2.value(), np.degrees(turbine.gamma_in), places=1)
        self.assertFalse(self.form.checkBox_rho_auto.isChecked())
        self.assertTrue(self.form.checkBox_h0_auto.isChecked())
        self.assertFalse(self.form.checkBox_precise_h0.isChecked())
        self.assertAlmostEqual(self.form.H01_init.value(), turbine.H01_init / 1e3, places=1)
        self.assertAlmostEqual(self.form.c21.value(), turbine.c21_init, places=1)

        self.assertAlmostEqual(self.form.T_g_stag.value(), turbine.T_g_stag, 2)
        self.assertAlmostEqual(self.form.p_g_stag.value(), turbine.p_g_stag / 1e6, places=3)
        self.assertAlmostEqual(self.form.G_t.value(), turbine.G_turbine, places=2)
        self.assertAlmostEqual(self.form.alpha_air.value(), turbine.alpha_air, places=2)
        self.assertAlmostEqual(self.form.alpha11.value(), np.degrees(turbine.alpha11), places=1)
        self.assertAlmostEqual(self.form.l1_D1_ratio.value(), turbine.l1_D1_ratio, places=2)
        self.assertAlmostEqual(self.form.n.value(), turbine.n, places=0)
        self.assertAlmostEqual(self.form.T_t_stag_cycle.value(), turbine.T_t_stag_cycle, places=0)
        self.assertAlmostEqual(self.form.eta_t_stag_cycle.value(), turbine.eta_t_stag_cycle, places=2)

        for i in range(turbine.stage_number):
            stage_data: StageDataWidget = self.form.stackedWidget.widget(i)
            geom = turbine.geom[i]
            self.assertAlmostEqual(stage_data.l1_b_sa_ratio.value(), geom.l1_b_sa_ratio, places=2)
            self.assertAlmostEqual(stage_data.l2_b_rk_ratio.value(), geom.l2_b_rk_ratio, places=2)
            self.assertAlmostEqual(stage_data.delta_a_b_sa_ratio.value(), geom.delta_a_b_sa_ratio, places=2)
            self.assertAlmostEqual(stage_data.delta_a_b_rk_ratio.value(), geom.delta_a_b_rk_ratio, places=2)
            self.assertAlmostEqual(stage_data.p_r_out_l1_ratio.value(), geom.p_r_out_l1_ratio, places=2)
            self.assertAlmostEqual(stage_data.p_r_in_l1_ratio.value(), geom.p_r_in_l1_ratio, places=2)
            self.assertAlmostEqual(stage_data.p_a_out_rel.value(), geom.p_a_out_rel, places=2)
            self.assertAlmostEqual(stage_data.p_a_in_rel.value(), geom.p_a_in_rel, places=2)
            self.assertAlmostEqual(stage_data.delta_r_rel.value(), geom.delta_r_rk_l2_ratio, places=2)
            self.assertAlmostEqual(stage_data.phi.value(), geom.phi, places=2)
            self.assertAlmostEqual(stage_data.psi.value(), geom.psi, places=2)
            self.assertAlmostEqual(stage_data.g_lb.value(), geom.g_lb, places=2)
            self.assertAlmostEqual(stage_data.g_ld.value(), geom.g_ld, places=2)
            self.assertAlmostEqual(stage_data.g_lk.value(), geom.g_lk, places=2)
            self.assertAlmostEqual(stage_data.g_cool.value(), geom.g_cool, places=2)


if __name__ == '__main__':
    unittest.main()