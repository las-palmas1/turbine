from core.average_streamline.turbine import TurbineType, Turbine
from gas_turbine_cycle.gases import KeroseneCombustionProducts
import unittest
import numpy as np


class TurbineTest(unittest.TestCase):
    def setUp(self):
        self.comp_turb_h0_auto = Turbine(TurbineType.Compressor,
                                         T_g_stag=1400,
                                         p_g_stag=5.5e5,
                                         G_turbine=25,
                                         work_fluid=KeroseneCombustionProducts(),
                                         alpha_air=2.87,
                                         l1_D1_ratio=0.25,
                                         n=15e3,
                                         T_t_stag_cycle=1200,
                                         stage_number=2,
                                         eta_t_stag_cycle=0.91,
                                         k_n=6.8,
                                         eta_m=0.99,
                                         auto_compute_heat_drop=True,
                                         auto_set_rho=True,
                                         H01_init=150e3,
                                         c21_init=250,
                                         alpha11=np.radians(17),
                                         gamma_av=np.radians(4),
                                         gamma_sum=np.radians(10))
        self.comp_turb_h0_hand = Turbine(TurbineType.Compressor,
                                         T_g_stag=1400,
                                         p_g_stag=5.5e5,
                                         G_turbine=25,
                                         work_fluid=KeroseneCombustionProducts(),
                                         alpha_air=2.87,
                                         l1_D1_ratio=0.25,
                                         n=15e3,
                                         T_t_stag_cycle=1200,
                                         stage_number=2,
                                         eta_t_stag_cycle=0.91,
                                         k_n=6.8,
                                         eta_m=0.99,
                                         auto_compute_heat_drop=False,
                                         auto_set_rho=True,
                                         H0_list=[160e3, 140e3],
                                         alpha11=np.radians(17),
                                         gamma_av=np.radians(4),
                                         gamma_sum=np.radians(10))
        self.comp_turb_rho_hand = Turbine(TurbineType.Compressor,
                                          T_g_stag=1400,
                                          p_g_stag=5.5e5,
                                          G_turbine=25,
                                          work_fluid=KeroseneCombustionProducts(),
                                          alpha_air=2.87,
                                          l1_D1_ratio=0.25,
                                          n=15e3,
                                          T_t_stag_cycle=1200,
                                          stage_number=2,
                                          eta_t_stag_cycle=0.91,
                                          k_n=6.8,
                                          eta_m=0.99,
                                          auto_compute_heat_drop=True,
                                          auto_set_rho=False,
                                          rho_list=[0.45, 0.4],
                                          H01_init=150e3,
                                          c21_init=250,
                                          alpha11=np.radians(17),
                                          gamma_av=np.radians(4),
                                          gamma_sum=np.radians(10))
        self.power_turb_h0_auto = Turbine(TurbineType.Power,
                                          T_g_stag=1400,
                                          p_g_stag=5.5e5,
                                          G_turbine=25,
                                          work_fluid=KeroseneCombustionProducts(),
                                          alpha_air=2.87,
                                          l1_D1_ratio=0.25,
                                          n=15e3,
                                          T_t_stag_cycle=1200,
                                          stage_number=2,
                                          eta_t_stag_cycle=0.91,
                                          k_n=6.8,
                                          eta_m=0.99,
                                          auto_compute_heat_drop=True,
                                          auto_set_rho=True,
                                          H01_init=150e3,
                                          c21_init=250,
                                          alpha11=np.radians(17),
                                          gamma_av=np.radians(4),
                                          gamma_sum=np.radians(10))

    def test_comp_turbine_h0_auto_rho_auto(self):
        """Тестирование расчета компрессорной турбины с автонастройкой теплоперепадов и степени реактивности, а также
        тестирование переинициализации геометрии турбины при изменении значений некоторых полей."""
        self.comp_turb_h0_auto.compute_geometry()
        self.comp_turb_h0_auto.compute_stages_gas_dynamics()
        self.comp_turb_h0_auto.compute_integrate_turbine_parameters()

        self.assertGreater(self.comp_turb_h0_auto.eta_t_stag, self.comp_turb_h0_auto.eta_t)
        self.assertGreater(self.comp_turb_h0_auto.eta_l, self.comp_turb_h0_auto.eta_t_stag)
        self.assertNotEqual(self.comp_turb_h0_auto.eta_t, None)
        self.assertNotEqual(self.comp_turb_h0_auto.eta_t_stag, None)

        L_st1 = self.comp_turb_h0_auto[0].c_p_gas * (self.comp_turb_h0_auto[0].T0_stag -
                                                     self.comp_turb_h0_auto[0].T_st_stag)
        L_st2 = self.comp_turb_h0_auto[1].c_p_gas * (self.comp_turb_h0_auto[1].T0_stag -
                                                     self.comp_turb_h0_auto[1].T_st_stag)
        self.assertAlmostEqual(abs((L_st1 + L_st2) - self.comp_turb_h0_auto.L_t_cycle) / (L_st1 + L_st2), 0, places=2)

        H_t_old = self.comp_turb_h0_auto.H_t_stag_cycle
        self.comp_turb_h0_auto.eta_t_stag_cycle = self.comp_turb_h0_auto.eta_t_stag_cycle - 0.01
        self.assertNotEqual(self.comp_turb_h0_auto.H_t_stag_cycle, H_t_old)
        self.assertNotEqual(self.comp_turb_h0_auto.geom.H_t_stag, H_t_old)
        self.assertEqual(self.comp_turb_h0_auto.H_t_stag_cycle, self.comp_turb_h0_auto.geom.H_t_stag)

        H_t_old = self.comp_turb_h0_auto.H_t_stag_cycle
        self.comp_turb_h0_auto.T_t_stag_cycle = self.comp_turb_h0_auto.T_t_stag_cycle - 10
        self.assertNotEqual(self.comp_turb_h0_auto.H_t_stag_cycle, H_t_old)
        self.assertNotEqual(self.comp_turb_h0_auto.geom.H_t_stag, H_t_old)
        self.assertEqual(self.comp_turb_h0_auto.H_t_stag_cycle, self.comp_turb_h0_auto.geom.H_t_stag)

        p_t_old = self.comp_turb_h0_auto.p_t_stag_cycle
        self.comp_turb_h0_auto.p_g_stag = self.comp_turb_h0_auto.p_g_stag - 10e3
        self.assertNotEqual(self.comp_turb_h0_auto.p_t_stag_cycle, p_t_old)
        self.assertNotEqual(self.comp_turb_h0_auto.geom.p_t_stag, p_t_old)
        self.assertEqual(self.comp_turb_h0_auto.p_t_stag_cycle, self.comp_turb_h0_auto.geom.p_t_stag)

        H_t_old = self.comp_turb_h0_auto.H_t_stag_cycle
        self.comp_turb_h0_auto.T_g_stag = self.comp_turb_h0_auto.T_g_stag - 10
        self.assertNotEqual(self.comp_turb_h0_auto.H_t_stag_cycle, H_t_old)
        self.assertNotEqual(self.comp_turb_h0_auto.geom.H_t_stag, H_t_old)
        self.assertEqual(self.comp_turb_h0_auto.H_t_stag_cycle, self.comp_turb_h0_auto.geom.H_t_stag)

    def test_comp_turbine_h0_hand_rho_auto(self):
        """Тестирование расчета компрессорной турбины с ручной настройкой теплоперепадо."""
        self.comp_turb_h0_hand.compute_geometry()
        self.comp_turb_h0_hand.compute_stages_gas_dynamics()
        self.comp_turb_h0_hand.compute_integrate_turbine_parameters()

        L_st1 = self.comp_turb_h0_hand[0].c_p_gas * (self.comp_turb_h0_hand[0].T0_stag -
                                                     self.comp_turb_h0_hand[0].T_st_stag)
        L_st2 = self.comp_turb_h0_hand[1].c_p_gas * (self.comp_turb_h0_hand[1].T0_stag -
                                                     self.comp_turb_h0_hand[1].T_st_stag)
        self.assertAlmostEqual(abs((L_st1 + L_st2) - self.comp_turb_h0_hand.L_t_cycle) / (L_st1 + L_st2), 0, places=2)
        self.assertGreater(self.comp_turb_h0_hand.eta_t_stag, self.comp_turb_h0_hand.eta_t)
        self.assertGreater(self.comp_turb_h0_hand.eta_l, self.comp_turb_h0_hand.eta_t_stag)
        self.assertNotEqual(self.comp_turb_h0_hand.eta_t, None)
        self.assertNotEqual(self.comp_turb_h0_hand.eta_t_stag, None)

    def test_comp_turbine_h0_auto_rho_hand(self):
        """Тестирование расчета компрессорной турбины с ручной настройкой степеней реактивности"""
        self.comp_turb_rho_hand.compute_geometry()
        self.comp_turb_rho_hand.compute_stages_gas_dynamics()
        self.comp_turb_rho_hand.compute_integrate_turbine_parameters()

        L_st1 = self.comp_turb_rho_hand[0].c_p_gas * (self.comp_turb_rho_hand[0].T0_stag -
                                                      self.comp_turb_rho_hand[0].T_st_stag)
        L_st2 = self.comp_turb_rho_hand[1].c_p_gas * (self.comp_turb_rho_hand[1].T0_stag -
                                                      self.comp_turb_rho_hand[1].T_st_stag)
        self.assertAlmostEqual(abs((L_st1 + L_st2) - self.comp_turb_rho_hand.L_t_cycle) / (L_st1 + L_st2), 0, places=2)
        self.assertGreater(self.comp_turb_rho_hand.eta_t_stag, self.comp_turb_rho_hand.eta_t)
        self.assertGreater(self.comp_turb_rho_hand.eta_l, self.comp_turb_rho_hand.eta_t_stag)
        self.assertNotEqual(self.comp_turb_rho_hand.eta_t, None)
        self.assertNotEqual(self.comp_turb_rho_hand.eta_t_stag, None)

    def test_power_turbine_h0_auto_rho_auto(self):
        """Тестирование расчета силовой турбины с автонастройкой теплоперепадов и степеней реактивности"""
        self.power_turb_h0_auto.compute_geometry()
        self.power_turb_h0_auto.compute_stages_gas_dynamics()
        self.power_turb_h0_auto.compute_integrate_turbine_parameters()

        self.assertGreater(self.power_turb_h0_auto.eta_t_stag, self.power_turb_h0_auto.eta_t)
        self.assertGreater(self.power_turb_h0_auto.eta_l, self.power_turb_h0_auto.eta_t_stag)
        self.assertNotEqual(self.power_turb_h0_auto.eta_t, None)
        self.assertNotEqual(self.power_turb_h0_auto.eta_t_stag, None)
        self.assertEqual(self.power_turb_h0_auto.last.p2, self.power_turb_h0_auto.geom.p_t)

    def test_parameter_transfer(self):
        """Тестирование передачи параметров между газодинамическими расчетами ступеней и между
        геометрией и газодинамикой"""
        self.comp_turb_h0_auto.compute_geometry()
        self.comp_turb_h0_auto.compute_stages_gas_dynamics()
        self.comp_turb_h0_auto.compute_integrate_turbine_parameters()

        self.assertEqual(self.comp_turb_h0_auto[0].T_st_stag, self.comp_turb_h0_auto[1].T0_stag)
        self.assertEqual(self.comp_turb_h0_auto[0].p2_stag, self.comp_turb_h0_auto[1].p0_stag)
        self.assertEqual(self.comp_turb_h0_auto[0].G_turbine, self.comp_turb_h0_auto[1].G_turbine)
        self.assertEqual(self.comp_turb_h0_auto[0].alpha_air, self.comp_turb_h0_auto[1].alpha_air)
        self.assertEqual(self.comp_turb_h0_auto.geom[0].l1, self.comp_turb_h0_auto[0].l1)
        self.assertEqual(self.comp_turb_h0_auto.geom[0].l2, self.comp_turb_h0_auto[0].l2)
        self.assertEqual(self.comp_turb_h0_auto.geom[0].D1, self.comp_turb_h0_auto[0].D1)
        self.assertEqual(self.comp_turb_h0_auto.geom[0].D2, self.comp_turb_h0_auto[0].D2)
        self.assertEqual(self.comp_turb_h0_auto.geom[0].rho, self.comp_turb_h0_auto[0].rho)
        self.assertEqual(self.comp_turb_h0_auto.geom[0].H0, self.comp_turb_h0_auto[0].H0)
        self.assertEqual(self.comp_turb_h0_auto.geom[1].l1, self.comp_turb_h0_auto[1].l1)
        self.assertEqual(self.comp_turb_h0_auto.geom[1].l2, self.comp_turb_h0_auto[1].l2)
        self.assertEqual(self.comp_turb_h0_auto.geom[1].D1, self.comp_turb_h0_auto[1].D1)
        self.assertEqual(self.comp_turb_h0_auto.geom[1].D2, self.comp_turb_h0_auto[1].D2)
        self.assertEqual(self.comp_turb_h0_auto.geom[1].rho, self.comp_turb_h0_auto[1].rho)
        self.assertEqual(self.comp_turb_h0_auto.geom[1].H0, self.comp_turb_h0_auto[1].H0)



