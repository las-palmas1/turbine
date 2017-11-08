import unittest
from core.profiling.stage import StageProfiler, ProfilingType
from core.average_streamline.turbine import Turbine, TurbineType
from gas_turbine_cycle.gases import KeroseneCombustionProducts
from core.average_streamline.stage_gas_dynamics import StageGasDynamics
from core.average_streamline.stage_geom import StageGeomAndHeatDrop
import numpy as np
from core.profiling.turbine import TurbineProfiler


class StageProfilerTest(unittest.TestCase):
    def setUp(self):
        self.turbine = Turbine(TurbineType.Compressor,
                               T_g_stag=1400,
                               p_g_stag=300e3,
                               G_turbine=25,
                               work_fluid=KeroseneCombustionProducts(),
                               alpha_air=2.5,
                               l1_D1_ratio=0.25,
                               n=15e3,
                               T_t_stag_cycle=1150,
                               stage_number=2,
                               eta_t_stag_cycle=0.91,
                               k_n=6.8,
                               eta_m=0.99,
                               auto_compute_heat_drop=True,
                               precise_heat_drop=False,
                               auto_set_rho=True,
                               H01_init=120e3,
                               c21_init=250,
                               alpha11=np.radians([17])[0],
                               gamma_av=np.radians([4])[0],
                               gamma_sum=np.radians([10])[0])
        self.turbine.compute_geometry()
        self.turbine.compute_stages_gas_dynamics()
        self.turbine.compute_integrate_turbine_parameters()
        self.stage_prof = StageProfiler(profiling_type=ProfilingType.ConstantCirculation,
                                        p0_stag=lambda r: self.turbine[0].p0_stag,
                                        T0_stag=lambda r: self.turbine[0].T0_stag,
                                        c0=lambda r: 100,
                                        alpha0=lambda r: np.radians([90])[0],
                                        c_p=self.turbine.c_p_gas,
                                        k=self.turbine.k_gas,
                                        D1_in=self.turbine.geom[0].D1 - self.turbine.geom[0].l1,
                                        D1_av=self.turbine.geom[0].D1,
                                        D1_out=self.turbine.geom[0].D1 + self.turbine.geom[0].l1,
                                        n=self.turbine.n,
                                        c1_av=self.turbine[0].c1,
                                        alpha1_av=self.turbine[0].alpha1,
                                        L_u_av=self.turbine[0].L_u,
                                        c2_a_av=self.turbine[0].c2_a,
                                        c2_u_av=self.turbine[0].c2_u,
                                        b_a_sa=self.turbine.geom[0].b_sa,
                                        b_a_rk=self.turbine.geom[0].b_rk,
                                        delta_a_sa=self.turbine.geom[0].delta_a_sa,
                                        delta_a_rk=self.turbine.geom[0].delta_a_rk,
                                        t_rel_av_sa=0.8,
                                        t_rel_av_rk=0.7,
                                        center=True,
                                        section_num=3,
                                        x0=0.,
                                        y0=0.,
                                        pnt_cnt=35)

    def test_blade_number_computing(self):
        self.stage_prof.init_sections()
        self.stage_prof.compute_sections()

        self.assertNotEqual(self.stage_prof.z_sa, None)
        self.assertNotEqual(self.stage_prof.z_rk, None)
        self.assertNotEqual(self.stage_prof.t_av_rk, None)
        self.assertNotEqual(self.stage_prof.t_av_sa, None)

        self.assertEqual(self.stage_prof.z_sa * self.stage_prof.t_av_sa, np.pi * self.stage_prof.D1_av)
        self.assertEqual(self.stage_prof.z_rk * self.stage_prof.t_av_rk, np.pi * self.stage_prof.D1_av)

    def test_sections_computing(self):
        self.stage_prof.init_sections()
        self.stage_prof.compute_sections()

        self.assertNotEqual(self.stage_prof.sa_sections[0].x_k, None)
        self.assertNotEqual(self.stage_prof.sa_sections[0].y_k, None)
        self.assertNotEqual(self.stage_prof.sa_sections[0].x_s, None)
        self.assertNotEqual(self.stage_prof.sa_sections[0].y_s, None)
        self.assertNotEqual(self.stage_prof.sa_sections[0].x_in_edge, None)
        self.assertNotEqual(self.stage_prof.sa_sections[0].y_in_edge, None)
        self.assertNotEqual(self.stage_prof.sa_sections[0].x_out_edge, None)
        self.assertNotEqual(self.stage_prof.sa_sections[0].y_out_edge, None)

        self.assertNotEqual(self.stage_prof.sa_sections[1].x_k, None)
        self.assertNotEqual(self.stage_prof.sa_sections[1].y_k, None)
        self.assertNotEqual(self.stage_prof.sa_sections[1].x_s, None)
        self.assertNotEqual(self.stage_prof.sa_sections[1].y_s, None)
        self.assertNotEqual(self.stage_prof.sa_sections[1].x_in_edge, None)
        self.assertNotEqual(self.stage_prof.sa_sections[1].y_in_edge, None)
        self.assertNotEqual(self.stage_prof.sa_sections[1].x_out_edge, None)
        self.assertNotEqual(self.stage_prof.sa_sections[1].y_out_edge, None)

        self.assertNotEqual(self.stage_prof.sa_sections[2].x_k, None)
        self.assertNotEqual(self.stage_prof.sa_sections[2].y_k, None)
        self.assertNotEqual(self.stage_prof.sa_sections[2].x_s, None)
        self.assertNotEqual(self.stage_prof.sa_sections[2].y_s, None)
        self.assertNotEqual(self.stage_prof.sa_sections[2].x_in_edge, None)
        self.assertNotEqual(self.stage_prof.sa_sections[2].y_in_edge, None)
        self.assertNotEqual(self.stage_prof.sa_sections[2].x_out_edge, None)
        self.assertNotEqual(self.stage_prof.sa_sections[2].y_out_edge, None)

        self.assertNotEqual(self.stage_prof.rk_sections[0].x_k, None)
        self.assertNotEqual(self.stage_prof.rk_sections[0].y_k, None)
        self.assertNotEqual(self.stage_prof.rk_sections[0].x_s, None)
        self.assertNotEqual(self.stage_prof.rk_sections[0].y_s, None)
        self.assertNotEqual(self.stage_prof.rk_sections[0].x_in_edge, None)
        self.assertNotEqual(self.stage_prof.rk_sections[0].y_in_edge, None)
        self.assertNotEqual(self.stage_prof.rk_sections[0].x_out_edge, None)
        self.assertNotEqual(self.stage_prof.rk_sections[0].y_out_edge, None)

        self.assertNotEqual(self.stage_prof.rk_sections[1].x_k, None)
        self.assertNotEqual(self.stage_prof.rk_sections[1].y_k, None)
        self.assertNotEqual(self.stage_prof.rk_sections[1].x_s, None)
        self.assertNotEqual(self.stage_prof.rk_sections[1].y_s, None)
        self.assertNotEqual(self.stage_prof.rk_sections[1].x_in_edge, None)
        self.assertNotEqual(self.stage_prof.rk_sections[1].y_in_edge, None)
        self.assertNotEqual(self.stage_prof.rk_sections[1].x_out_edge, None)
        self.assertNotEqual(self.stage_prof.rk_sections[1].y_out_edge, None)

        self.assertNotEqual(self.stage_prof.rk_sections[2].x_k, None)
        self.assertNotEqual(self.stage_prof.rk_sections[2].y_k, None)
        self.assertNotEqual(self.stage_prof.rk_sections[2].x_s, None)
        self.assertNotEqual(self.stage_prof.rk_sections[2].y_s, None)
        self.assertNotEqual(self.stage_prof.rk_sections[2].x_in_edge, None)
        self.assertNotEqual(self.stage_prof.rk_sections[2].y_in_edge, None)
        self.assertNotEqual(self.stage_prof.rk_sections[2].x_out_edge, None)
        self.assertNotEqual(self.stage_prof.rk_sections[2].y_out_edge, None)

        r_in = self.stage_prof.D1_in / 2
        r_av = self.stage_prof.D1_av / 2
        r_out = self.stage_prof.D1_out / 2

        section = self.stage_prof.sa_sections[0]
        self.assertEqual(section.b_a, self.stage_prof.b_a_sa)
        self.assertEqual(section.angle1, self.stage_prof.alpha0(r_in))
        self.assertEqual(section.delta1, self.stage_prof.get_delta(self.stage_prof.M_c0(r_in)))
        self.assertEqual(section.angle2, self.stage_prof.alpha1(r_in))
        self.assertEqual(section.delta2, self.stage_prof.get_delta(self.stage_prof.M_c1(r_in)))
        self.assertEqual(section.angle1_l, section.angle1 - section.delta1)
        self.assertEqual(section.angle2_l, section.angle2 - section.delta2)
        self.assertEqual(section.r1, self.stage_prof.r1_rel_sa * self.stage_prof.b_a_sa)
        self.assertEqual(section.gamma1_k,
                         self.stage_prof.get_gamma1(section.angle1_l) * self.stage_prof.gamma1_k_rel_sa)
        self.assertEqual(section.gamma1_s,
                         self.stage_prof.get_gamma1(section.angle1_l) * (1 - self.stage_prof.gamma1_k_rel_sa))
        self.assertEqual(section.gamma2_k, self.stage_prof.gamma2_sa * self.stage_prof.gamma2_k_rel_sa)
        self.assertEqual(section.gamma2_s, self.stage_prof.gamma2_sa * (1 - self.stage_prof.gamma2_k_rel_sa))

        section = self.stage_prof.sa_sections[1]
        self.assertEqual(section.b_a, self.stage_prof.b_a_sa)
        self.assertEqual(section.angle1, self.stage_prof.alpha0(r_av))
        self.assertEqual(section.delta1, self.stage_prof.get_delta(self.stage_prof.M_c0(r_av)))
        self.assertEqual(section.angle2, self.stage_prof.alpha1(r_av))
        self.assertEqual(section.delta2, self.stage_prof.get_delta(self.stage_prof.M_c1(r_av)))
        self.assertEqual(section.angle1_l, section.angle1 - section.delta1)
        self.assertEqual(section.angle2_l, section.angle2 - section.delta2)
        self.assertEqual(section.r1, self.stage_prof.r1_rel_sa * self.stage_prof.b_a_sa)
        self.assertEqual(section.gamma1_k,
                         self.stage_prof.get_gamma1(section.angle1_l) * self.stage_prof.gamma1_k_rel_sa)
        self.assertEqual(section.gamma1_s,
                         self.stage_prof.get_gamma1(section.angle1_l) * (1 - self.stage_prof.gamma1_k_rel_sa))
        self.assertEqual(section.gamma2_k, self.stage_prof.gamma2_sa * self.stage_prof.gamma2_k_rel_sa)
        self.assertEqual(section.gamma2_s, self.stage_prof.gamma2_sa * (1 - self.stage_prof.gamma2_k_rel_sa))

        section = self.stage_prof.sa_sections[2]
        self.assertEqual(section.b_a, self.stage_prof.b_a_sa)
        self.assertEqual(section.angle1, self.stage_prof.alpha0(r_out))
        self.assertEqual(section.delta1, self.stage_prof.get_delta(self.stage_prof.M_c0(r_out)))
        self.assertEqual(section.angle2, self.stage_prof.alpha1(r_out))
        self.assertEqual(section.delta2, self.stage_prof.get_delta(self.stage_prof.M_c1(r_out)))
        self.assertEqual(section.angle1_l, section.angle1 - section.delta1)
        self.assertEqual(section.angle2_l, section.angle2 - section.delta2)
        self.assertEqual(section.r1, self.stage_prof.r1_rel_sa * self.stage_prof.b_a_sa)
        self.assertEqual(section.gamma1_k,
                         self.stage_prof.get_gamma1(section.angle1_l) * self.stage_prof.gamma1_k_rel_sa)
        self.assertEqual(section.gamma1_s,
                         self.stage_prof.get_gamma1(section.angle1_l) * (1 - self.stage_prof.gamma1_k_rel_sa))
        self.assertEqual(section.gamma2_k, self.stage_prof.gamma2_sa * self.stage_prof.gamma2_k_rel_sa)
        self.assertEqual(section.gamma2_s, self.stage_prof.gamma2_sa * (1 - self.stage_prof.gamma2_k_rel_sa))

        section = self.stage_prof.rk_sections[0]
        self.assertEqual(section.b_a, self.stage_prof.b_a_rk)
        self.assertEqual(section.angle1, self.stage_prof.beta1(r_in))
        self.assertEqual(section.delta1, self.stage_prof.get_delta(self.stage_prof.M_w1(r_in)))
        self.assertEqual(section.angle2, self.stage_prof.beta2(r_in))
        self.assertEqual(section.delta2, self.stage_prof.get_delta(self.stage_prof.M_w2(r_in)))
        self.assertEqual(section.angle1_l, section.angle1 - section.delta1)
        self.assertEqual(section.angle2_l, section.angle2 - section.delta2)
        self.assertEqual(section.r1, self.stage_prof.r1_rel_rk * self.stage_prof.b_a_rk)
        self.assertEqual(section.gamma1_k,
                         self.stage_prof.get_gamma1(section.angle1_l) * self.stage_prof.gamma1_k_rel_rk)
        self.assertEqual(section.gamma1_s,
                         self.stage_prof.get_gamma1(section.angle1_l) * (1 - self.stage_prof.gamma1_k_rel_rk))
        self.assertEqual(section.gamma2_k, self.stage_prof.gamma2_rk * self.stage_prof.gamma2_k_rel_rk)
        self.assertEqual(section.gamma2_s, self.stage_prof.gamma2_rk * (1 - self.stage_prof.gamma2_k_rel_rk))

        section = self.stage_prof.rk_sections[1]
        self.assertEqual(section.b_a, self.stage_prof.b_a_rk)
        self.assertEqual(section.angle1, self.stage_prof.beta1(r_av))
        self.assertEqual(section.delta1, self.stage_prof.get_delta(self.stage_prof.M_w1(r_av)))
        self.assertEqual(section.angle2, self.stage_prof.beta2(r_av))
        self.assertEqual(section.delta2, self.stage_prof.get_delta(self.stage_prof.M_w2(r_av)))
        self.assertEqual(section.angle1_l, section.angle1 - section.delta1)
        self.assertEqual(section.angle2_l, section.angle2 - section.delta2)
        self.assertEqual(section.r1, self.stage_prof.r1_rel_rk * self.stage_prof.b_a_rk)
        self.assertEqual(section.gamma1_k,
                         self.stage_prof.get_gamma1(section.angle1_l) * self.stage_prof.gamma1_k_rel_rk)
        self.assertEqual(section.gamma1_s,
                         self.stage_prof.get_gamma1(section.angle1_l) * (1 - self.stage_prof.gamma1_k_rel_rk))
        self.assertEqual(section.gamma2_k, self.stage_prof.gamma2_rk * self.stage_prof.gamma2_k_rel_rk)
        self.assertEqual(section.gamma2_s, self.stage_prof.gamma2_rk * (1 - self.stage_prof.gamma2_k_rel_rk))

        section = self.stage_prof.rk_sections[2]
        self.assertEqual(section.b_a, self.stage_prof.b_a_rk)
        self.assertEqual(section.angle1, self.stage_prof.beta1(r_out))
        self.assertEqual(section.delta1, self.stage_prof.get_delta(self.stage_prof.M_w1(r_out)))
        self.assertEqual(section.angle2, self.stage_prof.beta2(r_out))
        self.assertEqual(section.delta2, self.stage_prof.get_delta(self.stage_prof.M_w2(r_out)))
        self.assertEqual(section.angle1_l, section.angle1 - section.delta1)
        self.assertEqual(section.angle2_l, section.angle2 - section.delta2)
        self.assertEqual(section.r1, self.stage_prof.r1_rel_rk * self.stage_prof.b_a_rk)
        self.assertEqual(section.gamma1_k,
                         self.stage_prof.get_gamma1(section.angle1_l) * self.stage_prof.gamma1_k_rel_rk)
        self.assertEqual(section.gamma1_s,
                         self.stage_prof.get_gamma1(section.angle1_l) * (1 - self.stage_prof.gamma1_k_rel_rk))
        self.assertEqual(section.gamma2_k, self.stage_prof.gamma2_rk * self.stage_prof.gamma2_k_rel_rk)
        self.assertEqual(section.gamma2_s, self.stage_prof.gamma2_rk * (1 - self.stage_prof.gamma2_k_rel_rk))

    def test_plots(self):
        self.stage_prof.init_sections()
        self.stage_prof.compute_sections()
        self.stage_prof.plot_parameter_distribution(['alpha1', 'alpha2', 'beta1', 'beta2'])
        self.stage_prof.plot_profile_2d(1, width_rel=3)
        self.stage_prof.plot_profile_3d()


class TurbineProfilerTest(unittest.TestCase):
    def setUp(self):
        self.turbine = Turbine(TurbineType.Compressor,
                               T_g_stag=1450,
                               p_g_stag=400e3,
                               G_turbine=25,
                               work_fluid=KeroseneCombustionProducts(),
                               alpha_air=2.5,
                               l1_D1_ratio=0.25,
                               n=15e3,
                               T_t_stag_cycle=1050,
                               stage_number=2,
                               eta_t_stag_cycle=0.91,
                               k_n=6.8,
                               eta_m=0.99,
                               auto_compute_heat_drop=False,
                               precise_heat_drop=False,
                               auto_set_rho=False,
                               rho_list=[0.4, 0.20],
                               H0_list=[235e3, 200e3],
                               alpha11=np.radians([17])[0],
                               gamma_av=np.radians([4])[0],
                               gamma_sum=np.radians([10])[0])
        self.turbine.compute_geometry()
        self.turbine.compute_stages_gas_dynamics()
        self.turbine.compute_integrate_turbine_parameters()

        self.turbine_profiler = TurbineProfiler(turbine=self.turbine,
                                                p_in_stag=lambda r: self.turbine[0].p0_stag,
                                                T_in_stag=lambda r: self.turbine[0].T0_stag,
                                                c_in=lambda r: 90,
                                                alpha_in=lambda r: np.radians([90])[0],
                                                center=True)

    # def test_parameter_transfer(self):
    #     """Тестирование правильности передачи параметров из расчета по средней линии тока в профайлер"""
    #     self.turbine_profiler.compute_stage_profiles()
