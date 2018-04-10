from .turbine import TurbineType, Turbine
from .stage_gas_dynamics import StageGasDynamics, StageType
from gas_turbine_cycle.gases import KeroseneCombustionProducts, NaturalGasCombustionProducts
import unittest
import numpy as np
import logging

logging.basicConfig(format='%(levelname)s - %(message)s', level=logging.INFO)


class StageGasDynamicsTest(unittest.TestCase):
    def setUp(self):
        self.precision = 0.001
        self.heat_drop_stage = StageGasDynamics(
            T0_stag=1400,
            p0_stag=10e5,
            T0_stag_ad_t=1400,
            G_stage_in=30,
            G_turbine=29.5,
            G_fuel=0.7,
            work_fluid=KeroseneCombustionProducts(),
            rho=0.3,
            phi=0.97,
            psi=0.97,
            l1=0.06,
            l2=0.068,
            D1=0.3,
            D2=0.3,
            delta_r_rk=0.01,
            n=11e3,
            epsilon=1,
            g_lb=0,
            g_lk=0,
            g_ld=0,
            g_cool=0,
            precision=self.precision,
            H0=120e3
        )
        self.heat_drop_stage.compute()

        self.work_stage = StageGasDynamics(
            T0_stag=1400,
            p0_stag=10e5,
            T0_stag_ad_t=1400,
            G_stage_in=30,
            G_turbine=29.5,
            G_fuel=0.7,
            work_fluid=KeroseneCombustionProducts(),
            rho=0.3,
            phi=0.97,
            psi=0.97,
            l1=0.06,
            l2=0.068,
            D1=0.3,
            D2=0.3,
            delta_r_rk=0.01,
            n=11e3,
            epsilon=1,
            g_lb=0,
            g_lk=0,
            g_ld=0,
            g_cool=0,
            precision=self.precision,
            L_t=50e3,
            eta_t0=0.9
        )
        self.work_stage.compute()

        self.press_stage = StageGasDynamics(
            T0_stag=1400,
            p0_stag=10e5,
            T0_stag_ad_t=1400,
            G_stage_in=30,
            G_turbine=29.5,
            G_fuel=0.7,
            work_fluid=KeroseneCombustionProducts(),
            rho=0.3,
            phi=0.97,
            psi=0.97,
            l1=0.06,
            l2=0.068,
            D1=0.3,
            D2=0.3,
            delta_r_rk=0.01,
            n=11e3,
            epsilon=1,
            g_lb=0,
            g_lk=0,
            g_ld=0,
            g_cool=0,
            precision=self.precision,
            p2_stag=7.9e5
        )
        self.press_stage.compute()

    def test_stages_types(self):
        self.assertEqual(self.heat_drop_stage.stage_type, StageType.HEAT_DROP)
        self.assertEqual(self.press_stage.stage_type, StageType.PRESSURE)
        self.assertEqual(self.work_stage.stage_type, StageType.WORK)

    def test_heat_drop_stage(self):
        self.assertTrue(self.heat_drop_stage.k_res <= self.precision)
        k_res = abs(self.heat_drop_stage.k_gas - self.heat_drop_stage.work_fluid.k_av_int) / self.heat_drop_stage.k_gas
        self.assertEqual(self.heat_drop_stage.k_res, k_res)

    def test_work_stage(self):
        self.assertTrue(self.work_stage.L_t_res <= self.precision)
        self.assertTrue(self.heat_drop_stage.k_res <= self.precision)
        k_res = abs(self.heat_drop_stage.k_gas - self.heat_drop_stage.work_fluid.k_av_int) / self.heat_drop_stage.k_gas
        self.assertEqual(self.heat_drop_stage.k_res, k_res)

    def test_press_stage(self):
        print('Pressures stage testing')
        print('p2_stag_res = %s' % self.press_stage.p2_stag_res)
        print('iter_number_p2_stag = %s' % self.press_stage.iter_number_p2_stag)
        self.assertTrue(self.heat_drop_stage.k_res <= self.precision)
        k_res = abs(self.heat_drop_stage.k_gas - self.heat_drop_stage.work_fluid.k_av_int) / self.heat_drop_stage.k_gas
        self.assertEqual(self.heat_drop_stage.k_res, k_res)
        p2_stag_old = self.press_stage.p2_stag
        self.press_stage._specified_heat_drop_calculation(self.press_stage.H0)
        self.assertEqual(self.press_stage.p2_stag, p2_stag_old)


class TurbineTest(unittest.TestCase):
    def setUp(self):
        self.precision = 0.0005
        self.comp_turb_h0_auto = Turbine(TurbineType.WORK,
                                         T_g_stag=1523,
                                         p_g_stag=16e5,
                                         G_turbine=40,
                                         G_fuel=0.8,
                                         work_fluid=KeroseneCombustionProducts(),
                                         l1_D1_ratio=0.25,
                                         n=13e3,
                                         T_t_stag_cycle=1100,
                                         stage_number=2,
                                         eta_t_stag_cycle=0.91,
                                         k_n=6.8,
                                         eta_m=0.99,
                                         auto_compute_heat_drop=True,
                                         auto_set_rho=True,
                                         precise_heat_drop=False,
                                         precision=self.precision,
                                         H01_init=150e3,
                                         c21_init=250,
                                         alpha11=np.radians([17])[0],
                                         gamma_av=np.radians([4])[0],
                                         gamma_sum=np.radians([10])[0])
        self.comp_turb_h0_auto.geom[0].g_cool = 0.03
        self.comp_turb_h0_auto.geom[0].g_lb = 0.00
        self.comp_turb_h0_auto.geom[0].g_lk = 0.00
        self.comp_turb_h0_auto.geom[0].g_ld = 0.00
        self.comp_turb_h0_auto.geom[0].T_cool = 750
        self.comp_turb_h0_auto.geom[1].g_cool = 0.03
        self.comp_turb_h0_auto.compute_geometry()
        self.comp_turb_h0_auto.compute_stages_gas_dynamics()
        self.comp_turb_h0_auto.compute_integrate_turbine_parameters()

        self.comp_turb_h0_hand = Turbine(TurbineType.WORK,
                                         T_g_stag=1523,
                                         p_g_stag=16e5,
                                         G_turbine=40,
                                         G_fuel=0.8,
                                         work_fluid=NaturalGasCombustionProducts(),
                                         l1_D1_ratio=0.12,
                                         n=13e3,
                                         T_t_stag_cycle=1100,
                                         stage_number=2,
                                         eta_t_stag_cycle=0.91,
                                         k_n=6.8,
                                         eta_m=0.99,
                                         auto_compute_heat_drop=False,
                                         auto_set_rho=True,
                                         precise_heat_drop=False,
                                         H0_list=[160e3, 140e3],
                                         alpha11=np.radians([17])[0],
                                         gamma_av=np.radians([4])[0],
                                         gamma_sum=np.radians([10])[0])
        self.comp_turb_h0_hand.compute_geometry()
        self.comp_turb_h0_hand.compute_stages_gas_dynamics()
        self.comp_turb_h0_hand.compute_integrate_turbine_parameters()

        self.comp_turb_rho_hand = Turbine(TurbineType.WORK,
                                          T_g_stag=1400,
                                          p_g_stag=5.5e5,
                                          G_turbine=25,
                                          G_fuel=1,
                                          work_fluid=KeroseneCombustionProducts(),
                                          l1_D1_ratio=0.25,
                                          n=15e3,
                                          T_t_stag_cycle=1200,
                                          stage_number=2,
                                          eta_t_stag_cycle=0.91,
                                          k_n=6.8,
                                          eta_m=0.99,
                                          auto_compute_heat_drop=True,
                                          auto_set_rho=False,
                                          precise_heat_drop=False,
                                          rho_list=[0.45, 0.4],
                                          H01_init=150e3,
                                          c21_init=250,
                                          alpha11=np.radians([17])[0],
                                          gamma_av=np.radians([4])[0],
                                          gamma_sum=np.radians([10])[0])
        self.comp_turb_rho_hand.compute_geometry()
        self.comp_turb_rho_hand.compute_stages_gas_dynamics()
        self.comp_turb_rho_hand.compute_integrate_turbine_parameters()

        self.power_turb_h0_auto = Turbine(TurbineType.PRESSURE,
                                          T_g_stag=1400,
                                          p_g_stag=5.5e5,
                                          G_turbine=25,
                                          G_fuel=1,
                                          work_fluid=NaturalGasCombustionProducts(),
                                          l1_D1_ratio=0.25,
                                          n=15e3,
                                          T_t_stag_cycle=1200,
                                          stage_number=3,
                                          eta_t_stag_cycle=0.91,
                                          k_n=6.8,
                                          eta_m=0.99,
                                          auto_compute_heat_drop=True,
                                          auto_set_rho=True,
                                          precise_heat_drop=False,
                                          H01_init=150e3,
                                          c21_init=250,
                                          alpha11=np.radians([17])[0],
                                          gamma_in=np.radians([0])[0],
                                          gamma_out=np.radians([10])[0])
        self.power_turb_h0_auto.compute_geometry()
        self.power_turb_h0_auto.compute_stages_gas_dynamics()
        self.power_turb_h0_auto.compute_integrate_turbine_parameters()

    def test_gas_dynamics_work_fluid_type_comp_turbine(self):
        main_type = type(self.comp_turb_h0_hand.work_fluid)
        for i in range(len(self.comp_turb_h0_hand)):
            stage = self.comp_turb_h0_hand[i]
            self.assertEqual(type(stage.work_fluid), main_type)

    def test_gas_dynamics_work_fluid_type_power_turbine(self):
        main_type = type(self.power_turb_h0_auto.work_fluid)
        self.assertEqual(main_type, NaturalGasCombustionProducts)
        for i in range(len(self.power_turb_h0_auto)):
            stage = self.power_turb_h0_auto[i]
            self.assertEqual(type(stage.work_fluid), main_type)

    def test_geom_work_fluid_type(self):
        main_type = type(self.comp_turb_h0_hand.work_fluid)
        self.assertEqual(main_type, NaturalGasCombustionProducts)
        self.assertEqual(main_type, type(self.comp_turb_h0_hand.geom.work_fluid))

    def test_comp_turbine_h0_auto_rho_auto(self):
        """Тестирование расчета компрессорной турбины с автонастройкой теплоперепадов и степени реактивности, а также
        тестирование переинициализации геометрии турбины при изменении значений некоторых полей."""
        self.assertGreater(self.comp_turb_h0_auto.eta_t_stag, self.comp_turb_h0_auto.eta_t)
        self.assertGreater(self.comp_turb_h0_auto.eta_l, self.comp_turb_h0_auto.eta_t_stag)
        self.assertNotEqual(self.comp_turb_h0_auto.eta_t, None)
        self.assertNotEqual(self.comp_turb_h0_auto.eta_t_stag, None)

    def test_precision_setting(self):
        self.assertEqual(self.comp_turb_h0_auto.geom.precision, self.precision)
        self.assertTrue(self.comp_turb_h0_auto.geom.T11_res <= self.precision)
        self.assertTrue(self.comp_turb_h0_auto.geom.T_t_res <= self.precision)
        for i in range(len(self.comp_turb_h0_auto)):
            self.assertEqual(self.comp_turb_h0_auto[i].precision, self.precision)
            self.assertTrue(self.comp_turb_h0_auto[i].k_res <= self.precision)

    def test_changing_eta_t_stag_cycle(self):
        H_t_old = self.comp_turb_h0_auto.H_t_stag_cycle
        self.comp_turb_h0_auto.eta_t_stag_cycle = self.comp_turb_h0_auto.eta_t_stag_cycle - 0.01
        self.assertNotEqual(self.comp_turb_h0_auto.H_t_stag_cycle, H_t_old)
        self.assertNotEqual(self.comp_turb_h0_auto.geom.H_t_stag, H_t_old)
        self.assertEqual(self.comp_turb_h0_auto.H_t_stag_cycle, self.comp_turb_h0_auto.geom.H_t_stag)

    def test_changing_T_stag_cycle(self):
        H_t_old = self.comp_turb_h0_auto.H_t_stag_cycle
        self.comp_turb_h0_auto.T_t_stag_cycle = self.comp_turb_h0_auto.T_t_stag_cycle - 10
        self.assertNotEqual(self.comp_turb_h0_auto.H_t_stag_cycle, H_t_old)
        self.assertNotEqual(self.comp_turb_h0_auto.geom.H_t_stag, H_t_old)
        self.assertEqual(self.comp_turb_h0_auto.H_t_stag_cycle, self.comp_turb_h0_auto.geom.H_t_stag)

    def test_changing_p_g_stag(self):
        p_t_old = self.comp_turb_h0_auto.p_t_stag_cycle
        self.comp_turb_h0_auto.p_g_stag = self.comp_turb_h0_auto.p_g_stag - 10e3
        self.assertNotEqual(self.comp_turb_h0_auto.p_t_stag_cycle, p_t_old)
        self.assertNotEqual(self.comp_turb_h0_auto.geom.p_t_stag, p_t_old)
        self.assertEqual(self.comp_turb_h0_auto.p_t_stag_cycle, self.comp_turb_h0_auto.geom.p_t_stag)

    def test_changing_T_g_stag(self):
        H_t_old = self.comp_turb_h0_auto.H_t_stag_cycle
        self.comp_turb_h0_auto.T_g_stag = self.comp_turb_h0_auto.T_g_stag - 10
        self.assertNotEqual(self.comp_turb_h0_auto.H_t_stag_cycle, H_t_old)
        self.assertNotEqual(self.comp_turb_h0_auto.geom.H_t_stag, H_t_old)
        self.assertEqual(self.comp_turb_h0_auto.H_t_stag_cycle, self.comp_turb_h0_auto.geom.H_t_stag)

    def test_comp_turbine_h0_hand_rho_auto(self):
        """Тестирование расчета компрессорной турбины с ручной настройкой теплоперепадо."""
        self.assertGreater(self.comp_turb_h0_hand.eta_t_stag, self.comp_turb_h0_hand.eta_t)
        self.assertGreater(self.comp_turb_h0_hand.eta_l, self.comp_turb_h0_hand.eta_t_stag)
        self.assertNotEqual(self.comp_turb_h0_hand.eta_t, None)
        self.assertNotEqual(self.comp_turb_h0_hand.eta_t_stag, None)
        self.assertAlmostEqual(self.comp_turb_h0_hand[0].alpha_air_in, self.comp_turb_h0_hand[0].alpha_air_out,
                               places=5)
        self.assertAlmostEqual(self.comp_turb_h0_hand[1].alpha_air_in, self.comp_turb_h0_hand[1].alpha_air_out,
                               places=5)
        self.assertAlmostEqual(self.comp_turb_h0_hand[0].T_st_stag, self.comp_turb_h0_hand[0].T_mix_stag, places=5)
        self.assertAlmostEqual(self.comp_turb_h0_hand[1].T_st_stag, self.comp_turb_h0_hand[1].T_mix_stag, places=5)

    def test_comp_turbine_h0_auto_rho_hand(self):
        """Тестирование расчета компрессорной турбины с ручной настройкой степеней реактивности"""
        self.assertGreater(self.comp_turb_rho_hand.eta_t_stag, self.comp_turb_rho_hand.eta_t)
        self.assertGreater(self.comp_turb_rho_hand.eta_l, self.comp_turb_rho_hand.eta_t_stag)
        self.assertNotEqual(self.comp_turb_rho_hand.eta_t, None)
        self.assertNotEqual(self.comp_turb_rho_hand.eta_t_stag, None)

    def test_power_turbine_h0_auto_rho_auto(self):
        """Тестирование расчета силовой турбины с автонастройкой теплоперепадов и степеней реактивности"""
        self.assertGreater(self.power_turb_h0_auto.eta_t_stag, self.power_turb_h0_auto.eta_t)
        self.assertGreater(self.power_turb_h0_auto.eta_l, self.power_turb_h0_auto.eta_t_stag)
        self.assertNotEqual(self.power_turb_h0_auto.eta_t, None)
        self.assertNotEqual(self.power_turb_h0_auto.eta_t_stag, None)

    def test_parameter_transfer(self):
        """Тестирование передачи параметров между газодинамическими расчетами ступеней и между
        геометрией и газодинамикой"""
        self.assertEqual(self.comp_turb_h0_auto[0].T_mix_stag, self.comp_turb_h0_auto[1].T0_stag)
        self.assertAlmostEqual(self.comp_turb_h0_auto[0].p2_stag, self.comp_turb_h0_auto[1].p0_stag, places=3)
        self.assertEqual(self.comp_turb_h0_auto[0].G_turbine, self.comp_turb_h0_auto[1].G_turbine)
        self.assertEqual(self.comp_turb_h0_auto[0].G_stage_out, self.comp_turb_h0_auto[1].G_stage_in)
        self.assertEqual(self.comp_turb_h0_auto[0].alpha_air_out, self.comp_turb_h0_auto[1].alpha_air_in)
        self.assertEqual(self.comp_turb_h0_auto[0].G_fuel, self.comp_turb_h0_auto[1].G_fuel)
        self.assertEqual(self.comp_turb_h0_auto[0].T_mix_stag_ad_t, self.comp_turb_h0_auto[1].T0_stag_ad_t)
        self.assertEqual(self.comp_turb_h0_auto.geom[0].l1, self.comp_turb_h0_auto[0].l1)
        self.assertEqual(self.comp_turb_h0_auto.geom[0].l2, self.comp_turb_h0_auto[0].l2)
        self.assertEqual(self.comp_turb_h0_auto.geom[0].D1, self.comp_turb_h0_auto[0].D1)
        self.assertEqual(self.comp_turb_h0_auto.geom[0].D2, self.comp_turb_h0_auto[0].D2)
        self.assertEqual(self.comp_turb_h0_auto.geom[0].rho, self.comp_turb_h0_auto[0].rho)
        self.assertEqual(self.comp_turb_h0_auto.geom[0].H0, self.comp_turb_h0_auto[0].H0)
        self.assertEqual(self.comp_turb_h0_auto.geom[0].T_cool, self.comp_turb_h0_auto[0].T_cool)
        self.assertEqual(self.comp_turb_h0_auto.geom[0].g_cool, self.comp_turb_h0_auto[0].g_cool)
        self.assertEqual(self.comp_turb_h0_auto.geom[0].g_lb, self.comp_turb_h0_auto[0].g_lb)
        self.assertEqual(self.comp_turb_h0_auto.geom[0].g_lk, self.comp_turb_h0_auto[0].g_lk)
        self.assertEqual(self.comp_turb_h0_auto.geom[0].g_ld, self.comp_turb_h0_auto[0].g_ld)
        self.assertEqual(self.comp_turb_h0_auto.geom[0].epsilon, self.comp_turb_h0_auto[0].epsilon)
        self.assertEqual(self.comp_turb_h0_auto.geom[0].delta_r_rk, self.comp_turb_h0_auto[0].delta_r_rk)

        self.assertEqual(self.comp_turb_h0_auto.geom[1].l1, self.comp_turb_h0_auto[1].l1)
        self.assertEqual(self.comp_turb_h0_auto.geom[1].l2, self.comp_turb_h0_auto[1].l2)
        self.assertEqual(self.comp_turb_h0_auto.geom[1].D1, self.comp_turb_h0_auto[1].D1)
        self.assertEqual(self.comp_turb_h0_auto.geom[1].D2, self.comp_turb_h0_auto[1].D2)
        self.assertEqual(self.comp_turb_h0_auto.geom[1].rho, self.comp_turb_h0_auto[1].rho)
        self.assertEqual(self.comp_turb_h0_auto.geom[1].H0, self.comp_turb_h0_auto[1].H0)
        self.assertEqual(self.comp_turb_h0_auto.geom[1].T_cool, self.comp_turb_h0_auto[1].T_cool)
        self.assertEqual(self.comp_turb_h0_auto.geom[1].g_cool, self.comp_turb_h0_auto[1].g_cool)
        self.assertEqual(self.comp_turb_h0_auto.geom[1].g_lb, self.comp_turb_h0_auto[1].g_lb)
        self.assertEqual(self.comp_turb_h0_auto.geom[1].g_lk, self.comp_turb_h0_auto[1].g_lk)
        self.assertEqual(self.comp_turb_h0_auto.geom[1].g_ld, self.comp_turb_h0_auto[1].g_ld)
        self.assertEqual(self.comp_turb_h0_auto.geom[1].epsilon, self.comp_turb_h0_auto[1].epsilon)
        self.assertEqual(self.comp_turb_h0_auto.geom[1].delta_r_rk, self.comp_turb_h0_auto[1].delta_r_rk)

        self.assertLessEqual(self.comp_turb_h0_auto[0].T_mix_stag, self.comp_turb_h0_auto[0].T_st_stag)
        self.assertLessEqual(self.comp_turb_h0_auto[0].alpha_air_in, self.comp_turb_h0_auto[0].alpha_air_out)


class TestSpecificHeatAveraging(unittest.TestCase):
    def setUp(self):
        self.precision = 0.00001
        self.comp_turbine_ker_wcool = Turbine(
            TurbineType.WORK,
            T_g_stag=1523,
            p_g_stag=16e5,
            G_turbine=40,
            G_fuel=0.8,
            work_fluid=KeroseneCombustionProducts(),
            l1_D1_ratio=0.25,
            n=13e3,
            T_t_stag_cycle=1000,
            stage_number=3,
            eta_t_stag_cycle=0.90,
            k_n=6.8,
            eta_m=0.99,
            auto_compute_heat_drop=False,
            auto_set_rho=True,
            precise_heat_drop=True,
            precision=self.precision,
            H0_list=[290e3, 290e3, 290e3],
            alpha11=np.radians([17])[0],
            gamma_av=np.radians([4])[0],
            gamma_sum=np.radians([10])[0]
        )
        self.comp_turbine_ker_wcool.compute_geometry()
        self.comp_turbine_ker_wcool.compute_stages_gas_dynamics()
        self.comp_turbine_ker_wcool.compute_integrate_turbine_parameters()

        self.comp_turbine_ngas_wcool = Turbine(
            TurbineType.WORK,
            T_g_stag=1523,
            p_g_stag=16e5,
            G_turbine=40,
            G_fuel=0.8,
            work_fluid=NaturalGasCombustionProducts(),
            l1_D1_ratio=0.25,
            n=13e3,
            T_t_stag_cycle=1000,
            stage_number=3,
            eta_t_stag_cycle=0.90,
            k_n=6.8,
            eta_m=0.99,
            auto_compute_heat_drop=False,
            auto_set_rho=True,
            precise_heat_drop=True,
            precision=self.precision,
            H0_list=[290e3, 290e3, 290e3],
            alpha11=np.radians([17])[0],
            gamma_av=np.radians([4])[0],
            gamma_sum=np.radians([10])[0]
        )
        self.comp_turbine_ngas_wcool.compute_geometry()
        self.comp_turbine_ngas_wcool.compute_stages_gas_dynamics()
        self.comp_turbine_ngas_wcool.compute_integrate_turbine_parameters()

        self.power_turbine_ngas_wcool = Turbine(
            TurbineType.PRESSURE,
            T_g_stag=1523,
            p_g_stag=16e5,
            G_turbine=40,
            G_fuel=0.8,
            work_fluid=NaturalGasCombustionProducts(),
            l1_D1_ratio=0.25,
            n=13e3,
            T_t_stag_cycle=1000,
            stage_number=3,
            eta_t_stag_cycle=0.90,
            k_n=6.8,
            eta_m=0.99,
            auto_compute_heat_drop=False,
            auto_set_rho=True,
            precise_heat_drop=True,
            precision=self.precision,
            H0_list=[290e3, 290e3, 290e3],
            alpha11=np.radians([17])[0],
            gamma_av=np.radians([4])[0],
            gamma_sum=np.radians([10])[0]
        )
        eta_t_res = 1
        eta_t_stag_new = self.power_turbine_ngas_wcool.eta_t_stag_cycle
        self.precision_eta_t = 0.0001
        while eta_t_res >= self.precision_eta_t:
            self.power_turbine_ngas_wcool.eta_t_stag_cycle = eta_t_stag_new
            self.power_turbine_ngas_wcool.compute_geometry()
            self.power_turbine_ngas_wcool.compute_stages_gas_dynamics()
            self.power_turbine_ngas_wcool.compute_integrate_turbine_parameters()
            eta_t_stag_new = self.power_turbine_ngas_wcool.eta_t_stag
            eta_t_stag_old = self.power_turbine_ngas_wcool.eta_t_stag_cycle
            eta_t_res = abs(eta_t_stag_old - eta_t_stag_new) / eta_t_stag_old

        self.comp_turbine_ngas_cool = Turbine(
            TurbineType.WORK,
            T_g_stag=1523,
            p_g_stag=16e5,
            G_turbine=40,
            G_fuel=0.8,
            work_fluid=NaturalGasCombustionProducts(),
            l1_D1_ratio=0.25,
            n=13e3,
            T_t_stag_cycle=1000,
            stage_number=3,
            eta_t_stag_cycle=0.90,
            k_n=6.8,
            eta_m=0.99,
            auto_compute_heat_drop=False,
            auto_set_rho=True,
            precise_heat_drop=True,
            precision=self.precision,
            H0_list=[290e3, 290e3, 290e3],
            alpha11=np.radians([17])[0],
            gamma_av=np.radians([4])[0],
            gamma_sum=np.radians([10])[0]
        )
        self.comp_turbine_ngas_cool.geom[0].g_cool = 0.05
        self.comp_turbine_ngas_cool.geom[1].g_cool = 0.05
        self.comp_turbine_ngas_cool.geom[2].g_cool = 0.01
        self.comp_turbine_ngas_cool.compute_geometry()
        self.comp_turbine_ngas_cool.compute_stages_gas_dynamics()
        self.comp_turbine_ngas_cool.compute_integrate_turbine_parameters()

    def test_comp_turbine_stages_types(self):
        self.assertEqual(self.comp_turbine_ker_wcool[0].stage_type, StageType.HEAT_DROP)
        self.assertEqual(self.comp_turbine_ker_wcool[1].stage_type, StageType.HEAT_DROP)
        self.assertEqual(self.comp_turbine_ker_wcool[2].stage_type, StageType.WORK)

    def test_power_turbine_stages_types(self):
        self.assertEqual(self.power_turbine_ngas_wcool[0].stage_type, StageType.HEAT_DROP)
        self.assertEqual(self.power_turbine_ngas_wcool[1].stage_type, StageType.HEAT_DROP)
        self.assertEqual(self.power_turbine_ngas_wcool[2].stage_type, StageType.PRESSURE)

    def test_separate_stages_work_balance_comp_turbine_ker_wcool(self):
        """Равенство работы, определенной по сумме работ ступеней, заданной величине."""
        L_t = sum([i.L_t_prime for i in self.comp_turbine_ker_wcool])
        L_t_res = abs(L_t - self.comp_turbine_ker_wcool.L_t_cycle) / self.comp_turbine_ker_wcool.L_t_cycle
        self.assertAlmostEqual(L_t_res, 0, places=3)

    def test_separate_stages_work_balance_comp_turbine_ngas_wcool(self):
        """Равенство работы, определенной по сумме работ ступеней, заданной величине."""
        L_t = sum([i.L_t_prime for i in self.comp_turbine_ngas_wcool])
        L_t_res = abs(L_t - self.comp_turbine_ngas_wcool.L_t_cycle) / self.comp_turbine_ngas_wcool.L_t_cycle
        self.assertAlmostEqual(L_t_res, 0, places=3)

    def test_separate_stages_work_balance_power_turbine_ngas_wcool(self):
        """Равенство работы, определенной по сумме работ ступеней, заданной величине."""
        L_t = sum([i.L_t_prime for i in self.power_turbine_ngas_wcool])
        L_t_res = abs(L_t - self.power_turbine_ngas_wcool.L_t_cycle) / self.power_turbine_ngas_wcool.L_t_cycle
        self.assertAlmostEqual(L_t_res, 0, places=2)

    def test_separate_stages_work_balance_comp_turbine_ngas_cool(self):
        """Равенство работы, определенной по сумме работ ступеней, заданной величине."""
        L_t = 0
        for i in range(len(self.comp_turbine_ngas_cool)):
            stage = self.comp_turbine_ngas_cool[i]
            L_t += stage.L_t_prime
        L_t_res = abs(L_t - self.comp_turbine_ngas_cool.L_t_cycle) / self.comp_turbine_ngas_cool.L_t_cycle
        self.assertAlmostEqual(L_t_res, 0, places=3)

    def test_out_temp_coincidence_comp_turbine_ker_wcool(self):
        T_out_stag = self.comp_turbine_ker_wcool.last.T_st_stag
        T_out_stag_cycle = self.comp_turbine_ker_wcool.T_t_stag_cycle
        T_out_stag_res = abs(T_out_stag - T_out_stag_cycle) / T_out_stag_cycle
        self.assertAlmostEqual(T_out_stag_res, 0, places=3)

    def test_out_temp_coincidence_comp_turbine_ngas_wcool(self):
        T_out_stag = self.comp_turbine_ngas_wcool.last.T_st_stag
        T_out_stag_cycle = self.comp_turbine_ngas_wcool.T_t_stag_cycle
        T_out_stag_res = abs(T_out_stag - T_out_stag_cycle) / T_out_stag_cycle
        self.assertAlmostEqual(T_out_stag_res, 0, places=3)

    def test_out_temp_coincidence_power_turbine_ngas_wcool(self):
        T_out_stag = self.power_turbine_ngas_wcool.last.T_st_stag
        T_out_stag_cycle = self.power_turbine_ngas_wcool.T_t_stag_cycle
        T_out_stag_res = abs(T_out_stag - T_out_stag_cycle) / T_out_stag_cycle
        self.assertAlmostEqual(T_out_stag_res, 0, places=2)

    def test_out_press_coincidence_comp_turbine_ker_wcool(self):
        self.comp_turbine_ker_wcool.eta_t_stag_cycle = self.comp_turbine_ker_wcool.eta_t_stag
        p_t_stag = self.comp_turbine_ker_wcool.last.p2_stag
        p_t_stag_cycle = self.comp_turbine_ker_wcool.p_t_stag_cycle
        p_t_stag_res = abs(p_t_stag - p_t_stag_cycle) / p_t_stag_cycle
        self.assertAlmostEqual(p_t_stag_res, 0, places=2)

    def test_out_press_coincidence_comp_turbine_ngas_wcool(self):
        self.comp_turbine_ngas_wcool.eta_t_stag_cycle = self.comp_turbine_ngas_wcool.eta_t_stag
        p_t_stag = self.comp_turbine_ngas_wcool.last.p2_stag
        p_t_stag_cycle = self.comp_turbine_ngas_wcool.p_t_stag_cycle
        p_t_stag_res = abs(p_t_stag - p_t_stag_cycle) / p_t_stag_cycle
        self.assertAlmostEqual(p_t_stag_res, 0, places=2)

    def test_out_press_coincidence_power_turbine_ngas_wcool(self):
        p_t_stag = self.power_turbine_ngas_wcool.last.p2_stag
        p_t_stag_cycle = self.power_turbine_ngas_wcool.p_t_stag_cycle
        p_t_stag_res = abs(p_t_stag - p_t_stag_cycle) / p_t_stag_cycle
        self.assertAlmostEqual(p_t_stag_res, 0, places=3)

    def test_turbine_heat_drop_comp_turbine_ker_wcool(self):
        """
        Равенство суммы теплоперепада, определенного по входным и выходным параметрам турбины,
        туплоперепаду, определенному как сумма теплоперепадов по ступеням при адиабатическом процессе в турбине.
        """
        work_fluid = type(self.comp_turbine_ker_wcool.work_fluid)()

        T_g_stag = self.comp_turbine_ker_wcool.T_g_stag
        T_t_stag_ad = self.comp_turbine_ker_wcool.last.T2_stag_ad_t
        T_t_ad = self.comp_turbine_ker_wcool.last.T2_ad_t
        alpha = self.comp_turbine_ker_wcool.first.alpha_air_in

        c_p_ad = work_fluid.c_p_av_int_func(T_g_stag, T_t_ad, alpha=alpha)
        c_p_stag_ad = work_fluid.c_p_av_int_func(T_g_stag, T_t_stag_ad, alpha=alpha)
        H_t_stag = c_p_stag_ad * (T_g_stag - T_t_stag_ad)
        H_t = c_p_ad * (T_g_stag - T_t_ad)

        H_t_stag_res = abs(H_t_stag - self.comp_turbine_ker_wcool.H_t_stag) / self.comp_turbine_ker_wcool.H_t_stag
        H_t_res = abs(H_t - self.comp_turbine_ker_wcool.H_t) / self.comp_turbine_ker_wcool.H_t

        self.assertAlmostEqual(H_t_stag_res, 0, places=3)
        self.assertAlmostEqual(H_t_res, 0, places=3)

    def test_turbine_heat_drop_comp_turbine_ngas_wcool(self):
        """
        Равенство суммы теплоперепада, определенного по входным и выходным параметрам турбины,
        туплоперепаду, определенному как сумма теплоперепадов по ступеням при адиабатическом процессе в турбине.
        """
        work_fluid = type(self.comp_turbine_ngas_wcool.work_fluid)()

        T_g_stag = self.comp_turbine_ngas_wcool.T_g_stag
        T_t_stag_ad = self.comp_turbine_ngas_wcool.last.T2_stag_ad_t
        T_t_ad = self.comp_turbine_ngas_wcool.last.T2_ad_t
        alpha = self.comp_turbine_ngas_wcool.first.alpha_air_in

        c_p_ad = work_fluid.c_p_av_int_func(T_g_stag, T_t_ad, alpha=alpha)
        c_p_stag_ad = work_fluid.c_p_av_int_func(T_g_stag, T_t_stag_ad, alpha=alpha)
        H_t_stag = c_p_stag_ad * (T_g_stag - T_t_stag_ad)
        H_t = c_p_ad * (T_g_stag - T_t_ad)

        H_t_stag_res = abs(H_t_stag - self.comp_turbine_ngas_wcool.H_t_stag) / self.comp_turbine_ngas_wcool.H_t_stag
        H_t_res = abs(H_t - self.comp_turbine_ngas_wcool.H_t) / self.comp_turbine_ngas_wcool.H_t

        self.assertAlmostEqual(H_t_stag_res, 0, places=3)
        self.assertAlmostEqual(H_t_res, 0, places=3)

    def test_turbine_heat_drop_power_turbine_ngas_wcool(self):
        """
        Равенство суммы теплоперепада, определенного по входным и выходным параметрам турбины,
        туплоперепаду, определенному как сумма теплоперепадов по ступеням при адиабатическом процессе в турбине.
        """
        work_fluid = type(self.power_turbine_ngas_wcool.work_fluid)()

        T_g_stag = self.power_turbine_ngas_wcool.T_g_stag
        T_t_stag_ad = self.power_turbine_ngas_wcool.last.T2_stag_ad_t
        T_t_ad = self.power_turbine_ngas_wcool.last.T2_ad_t
        alpha = self.power_turbine_ngas_wcool.first.alpha_air_in

        c_p_ad = work_fluid.c_p_av_int_func(T_g_stag, T_t_ad, alpha=alpha)
        c_p_stag_ad = work_fluid.c_p_av_int_func(T_g_stag, T_t_stag_ad, alpha=alpha)
        H_t_stag = c_p_stag_ad * (T_g_stag - T_t_stag_ad)
        H_t = c_p_ad * (T_g_stag - T_t_ad)

        H_t_stag_res = abs(H_t_stag - self.power_turbine_ngas_wcool.H_t_stag) / self.power_turbine_ngas_wcool.H_t_stag
        H_t_res = abs(H_t - self.power_turbine_ngas_wcool.H_t) / self.power_turbine_ngas_wcool.H_t

        self.assertAlmostEqual(H_t_stag_res, 0, places=3)
        self.assertAlmostEqual(H_t_res, 0, places=3)

    def test_turbine_heat_drop_comp_turbine_ngas_cool(self):
        """
        Равенство суммы теплоперепада, определенного по входным и выходным параметрам турбины,
        теплоперепаду, определенному как сумма теплоперепадов по ступеням при адиабатическом процессе в турбине.
        """
        i_stag_in = self.comp_turbine_ngas_cool.first.i_stag_ad_t_in
        i_ad_out = self.comp_turbine_ngas_cool.last.i_ad_t_out
        i_stag_ad_out = self.comp_turbine_ngas_cool.last.i_stag_ad_t_out

        i_cool = 0
        for n, stage in enumerate(self.comp_turbine_ngas_cool):
            if n != self.comp_turbine_ngas_cool.stage_number - 1:
                i_cool += stage.i_cool

        H_t_stag = i_stag_in - i_stag_ad_out + i_cool
        H_t = i_stag_in - i_ad_out + i_cool
        H_t_stag_res = abs(H_t_stag - self.comp_turbine_ngas_cool.H_t_stag) / self.comp_turbine_ngas_cool.H_t_stag
        H_t_res = abs(H_t - self.comp_turbine_ngas_cool.H_t) / self.comp_turbine_ngas_cool.H_t

        self.assertAlmostEqual(H_t_stag_res, 0, places=3)
        self.assertAlmostEqual(H_t_res, 0, places=3)

    def test_separate_stages_heat_drop_comp_turbine_ngas_cool(self):
        """Равенство теплоперепада на ступени при адиабатическом процессе в турбине изменение энтальпии."""
        for i in range(len(self.comp_turbine_ngas_cool)):
            stage = self.comp_turbine_ngas_cool[i]
            H0_stag_ad_t = stage.i_stag_ad_t_in - stage.i_stag_ad_t_out
            H0_ad_t = stage.i_stag_ad_t_in - stage.i_ad_t_out
            H0_stag_ad_t_res = abs(H0_stag_ad_t - stage.H0_stag_ad_t_prime) / stage.H0_stag_ad_t_prime
            H0_ad_t_res = abs(H0_ad_t - stage.H0_ad_t_prime) / stage.H0_ad_t_prime
            self.assertAlmostEqual(H0_stag_ad_t_res, 0, places=3)
            self.assertAlmostEqual(H0_ad_t_res, 0, places=3)

    def test_integrate_work_balance_comp_turbine_ker_wcool(self):
        """Соврадение работы, определенной на разности входной и выходной энтальпий, заданной величине."""
        work_fluid = KeroseneCombustionProducts()

        T_g_stag = self.comp_turbine_ker_wcool.T_g_stag
        T_t_stag = self.comp_turbine_ker_wcool.last.T_st_stag
        alpha = self.comp_turbine_ker_wcool.first.alpha_air_in

        c_p = work_fluid.c_p_av_int_func(T_g_stag, T_t_stag, alpha=alpha)
        L_t = c_p * (T_g_stag - T_t_stag)
        L_t_res = abs(L_t - self.comp_turbine_ker_wcool.L_t_cycle) / self.comp_turbine_ker_wcool.L_t_cycle
        self.assertAlmostEqual(L_t_res, 0, places=3)

    def test_integrate_work_balance_comp_turbine_ngas_wcool(self):
        """Соврадение работы, определенной на разности входной и выходной энтальпий, заданной величине."""
        work_fluid = NaturalGasCombustionProducts()
        T_g_stag = self.comp_turbine_ngas_wcool.T_g_stag
        T_t_stag = self.comp_turbine_ngas_wcool.last.T_st_stag
        work_fluid.T1 = T_g_stag
        work_fluid.T2 = T_t_stag
        work_fluid.alpha = self.comp_turbine_ngas_wcool.first.alpha_air_in
        L_t = work_fluid.c_p_av_int * (T_g_stag - T_t_stag)
        L_t_res = abs(L_t - self.comp_turbine_ngas_wcool.L_t_cycle) / self.comp_turbine_ngas_wcool.L_t_cycle
        self.assertAlmostEqual(L_t_res, 0, places=3)

    def test_integrate_work_balance_power_turbine_ngas_wcool(self):
        """Соврадение работы, определенной на разности входной и выходной энтальпий, заданной величине."""
        work_fluid = NaturalGasCombustionProducts()
        T_g_stag = self.power_turbine_ngas_wcool.T_g_stag
        T_t_stag = self.power_turbine_ngas_wcool.last.T_st_stag
        work_fluid.T1 = T_g_stag
        work_fluid.T2 = T_t_stag
        work_fluid.alpha = self.power_turbine_ngas_wcool.first.alpha_air_in
        L_t = work_fluid.c_p_av_int * (T_g_stag - T_t_stag)
        L_t_res = abs(L_t - self.power_turbine_ngas_wcool.L_t_cycle) / self.power_turbine_ngas_wcool.L_t_cycle
        self.assertAlmostEqual(L_t_res, 0, places=2)

    def test_integrate_work_balance_comp_turbine_ngas_cool(self):
        """Соврадение работы, определенной на разности входной и выходной энтальпий, заданной величине."""
        work_fluid = NaturalGasCombustionProducts()
        T_g_stag = self.comp_turbine_ngas_cool.T_g_stag
        T_t_stag = self.comp_turbine_ngas_cool.last.T_st_stag
        g_outlet = self.comp_turbine_ngas_cool.last.G_stage_in / self.comp_turbine_ngas_cool.G_turbine
        i_stag_in = work_fluid.get_specific_enthalpy(T_g_stag, alpha=self.comp_turbine_ngas_cool.first.alpha_air_in)
        i_stag_out = work_fluid.get_specific_enthalpy(
            T_t_stag, alpha=self.comp_turbine_ngas_cool.last.alpha_air_in
        ) * g_outlet
        i_cool = 0
        for n, stage in enumerate(self.comp_turbine_ngas_cool):
            if n != self.comp_turbine_ngas_cool.stage_number - 1:
                i_cool += stage.i_cool
        L_t = i_stag_in - i_stag_out + i_cool
        L_t_res = abs(L_t - self.comp_turbine_ngas_cool.L_t_cycle) / self.comp_turbine_ngas_cool.L_t_cycle
        self.assertAlmostEqual(L_t_res, 0, places=3)

    def test_separate_stages_averaging_comp_turbine_ker_wcool(self):
        """Равенство работы ступеней разности энатльпий на входе и на выходе ступеней."""
        for i in range(len(self.comp_turbine_ker_wcool)):
            logging.info('Stage %s' % (i + 1))
            stage = self.comp_turbine_ker_wcool[i]
            L_st = stage.c_p_gas * (stage.T0_stag - stage.T_st_stag)
            L_res = abs(L_st - stage.L_t_prime) / stage.L_t_prime
            di = self.comp_turbine_ker_wcool[i].i_stag_in - self.comp_turbine_ker_wcool[i].i_stag_out
            i_res = abs(stage.L_t_prime - di) / stage.L_t_prime
            self.assertAlmostEqual(i_res, 0, places=3)
            self.assertAlmostEqual(L_res, 0, places=3)
            self.assertEqual(stage.work_fluid.alpha, stage.alpha_air_in)
            self.assertEqual(stage.work_fluid.T2, stage.T_st_stag)

    def test_separate_stages_averaging_comp_turbine_ngas_wcool(self):
        """Равенство работы ступеней разности энатльпий на входе и на выходе ступеней."""
        for i in range(len(self.comp_turbine_ngas_wcool)):
            logging.info('Stage %s' % (i + 1))
            stage = self.comp_turbine_ngas_wcool[i]
            L_st = stage.c_p_gas * (stage.T0_stag - stage.T_st_stag)
            L_res = abs(L_st - stage.L_t_prime) / stage.L_t_prime
            di = self.comp_turbine_ngas_wcool[i].i_stag_in - self.comp_turbine_ngas_wcool[i].i_stag_out
            i_res = abs(stage.L_t_prime - di) / stage.L_t_prime
            self.assertAlmostEqual(i_res, 0, places=3)
            self.assertAlmostEqual(L_res, 0, places=3)
            self.assertEqual(stage.work_fluid.alpha, stage.alpha_air_in)
            self.assertEqual(stage.work_fluid.T2, stage.T_st_stag)

    def test_separate_stages_averaging_power_turbine_ngas_wcool(self):
        """Равенство работы ступеней разности энатльпий на входе и на выходе ступеней."""
        for i in range(len(self.power_turbine_ngas_wcool)):
            logging.info('Stage %s' % (i + 1))
            stage = self.power_turbine_ngas_wcool[i]
            L_st = stage.c_p_gas * (stage.T0_stag - stage.T_st_stag)
            L_res = abs(L_st - stage.L_t_prime) / stage.L_t_prime
            di = self.power_turbine_ngas_wcool[i].i_stag_in - self.power_turbine_ngas_wcool[i].i_stag_out
            i_res = abs(stage.L_t_prime - di) / stage.L_t_prime
            self.assertAlmostEqual(i_res, 0, places=3)
            self.assertAlmostEqual(L_res, 0, places=3)
            self.assertEqual(stage.work_fluid.alpha, stage.alpha_air_in)
            self.assertEqual(stage.work_fluid.T2, stage.T_st_stag)

    def test_mixture_enthalpy_conservation_cool_turbine(self):
        """Сохранение величины суммарной энтальпии после подмешвания."""
        for i in range(len(self.comp_turbine_ngas_cool)):
            logging.info('Stage %s' % (i + 1))
            stage = self.comp_turbine_ngas_cool[i]
            i1 = stage.i_stag_out + stage.i_cool
            i2 = stage.i_stag_mixture
            i1_ad_t = stage.i_stag_ad_t_out + stage.i_cool
            i2_ad_t = stage.i_stag_ad_t_mixture
            i_res = abs(i1 - i2) / i1
            i_ad_t_res = abs(i1_ad_t - i2_ad_t) / i1_ad_t
            self.assertAlmostEqual(i_res, 0, places=3)
            self.assertAlmostEqual(i_ad_t_res, 0, places=3)

    def test_separate_stages_averaging_comp_turbine_ngas_cool(self):
        """Равенство работы ступеней разности энатльпий на входе и на выходе ступеней."""
        for i in range(len(self.comp_turbine_ngas_cool)):
            logging.info('Stage %s' % (i + 1))
            stage = self.comp_turbine_ngas_cool[i]
            work_fluid_in = NaturalGasCombustionProducts()
            work_fluid_out = NaturalGasCombustionProducts()

            work_fluid_in.T = stage.T0_stag
            work_fluid_in.alpha = stage.alpha_air_in
            work_fluid_out.T = stage.T_st_stag
            work_fluid_out.alpha = stage.alpha_air_in

            enthalpy_in = work_fluid_in.c_p_av * (stage.T0_stag - work_fluid_in.T0)
            enthalpy_out = work_fluid_out.c_p_av * (stage.T_st_stag - work_fluid_out.T0)
            L_st = enthalpy_in - enthalpy_out
            L_res = abs(L_st - stage.L_t) / stage.L_t
            self.assertAlmostEqual(L_res, 0, places=3)
            self.assertEqual(stage.work_fluid.T2, stage.T_st_stag)




