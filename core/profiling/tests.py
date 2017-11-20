import unittest
from .stage import StageProfiler, ProfilingType
from ..average_streamline.turbine import Turbine, TurbineType
from .section import BladeSection
from gas_turbine_cycle.gases import KeroseneCombustionProducts
from ..average_streamline.stage_gas_dynamics import StageGasDynamics
from ..average_streamline.stage_geom import StageGeomAndHeatDrop
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from .turbine import TurbineProfiler


class SectionTest(unittest.TestCase):
    def setUp(self):
        self.bs1 = BladeSection(angle1=np.radians([30])[0],
                                angle2=np.radians([30])[0],
                                delta1=np.radians([5])[0],
                                delta2=np.radians([2])[0],
                                b_a=0.03,
                                r1=0.002,
                                convex='left',
                                pnt_count=30,
                                s2=0.0003)
        self.bs1.compute_profile()

        self.bs2 = BladeSection(angle1=np.radians([130])[0],
                                angle2=np.radians([30])[0],
                                delta1=np.radians([5])[0],
                                delta2=np.radians([2])[0],
                                b_a=0.03,
                                r1=0.002,
                                convex='right',
                                pnt_count=30,
                                s2=0.0003)
        self.bs2.compute_profile()

    def test_output(self):
        self.assertNotEqual(self.bs1.x_s, None)
        self.assertNotEqual(self.bs1.x_k, None)
        self.assertNotEqual(self.bs1.x_in_edge, None)
        self.assertNotEqual(self.bs1.x_out_edge, None)
        self.assertNotEqual(self.bs1.y_s, None)
        self.assertNotEqual(self.bs1.y_k, None)
        self.assertNotEqual(self.bs1.y_in_edge, None)
        self.assertNotEqual(self.bs1.y_out_edge, None)

        self.assertNotEqual(self.bs2.x_s, None)
        self.assertNotEqual(self.bs2.x_k, None)
        self.assertNotEqual(self.bs2.x_in_edge, None)
        self.assertNotEqual(self.bs2.x_out_edge, None)
        self.assertNotEqual(self.bs2.y_s, None)
        self.assertNotEqual(self.bs2.y_k, None)
        self.assertNotEqual(self.bs2.y_in_edge, None)
        self.assertNotEqual(self.bs2.y_out_edge, None)

    def test_plots(self):
        self.bs1.plot()
        self.bs1.plot()

    def test_partition(self):
        plt.figure(figsize=(6, 4))
        plt.plot(self.bs1.y_av, self.bs1.x_av, lw=0.5, ls='--', color='black')
        plt.plot(self.bs1.y_s, self.bs1.x_s, lw=1, color='red')
        plt.plot(self.bs1.y_k, self.bs1.x_k, lw=1, color='red')
        plt.plot(self.bs1.y_in_edge, self.bs1.x_in_edge, lw=1, color='red')
        plt.plot(self.bs1.y_out_edge, self.bs1.x_out_edge, lw=1, color='red')
        plt.plot([self.bs1.y0_av, self.bs1.y1_av, self.bs1.y2_av],
                 [self.bs1.x0_av, self.bs1.x1_av, self.bs1.x2_av], lw=0.5, ls=':',
                 color='blue', marker='o', ms=2)
        plt.plot([self.bs1.y0_k, self.bs1.y1_k, self.bs1.y2_k], [self.bs1.x0_k, self.bs1.x1_k, self.bs1.x2_k],
                 lw=0.5, ls=':',
                 color='blue', marker='o', ms=2)
        plt.plot([self.bs1.y0_s, self.bs1.y1_s, self.bs1.y2_s], [self.bs1.x0_s, self.bs1.x1_s, self.bs1.x2_s],
                 lw=0.5, ls=':',
                 color='blue', marker='o', ms=2)
        plt.plot([self.bs1.y_c], [self.bs1.x_c], linestyle='', marker='o', ms=8, mfc='black', color='red')
        plt.plot([self.bs1.y0], [self.bs1.x0], linestyle='', marker='o', ms=4, mfc='black', color='green')
        x12, y12, x23, y23 = self.bs1.get_heat_transfer_regions_bound_points(self.bs1.x_s, self.bs1.y_s,
                                                                             [self.bs1.length_s * 0.45,
                                                                              self.bs1.length_s -
                                                                              self.bs1.chord_length / 3])
        y_s_int = interp1d(self.bs1.x_s, self.bs1.y_s)
        self.assertEqual(self.bs1.get_length(x12, self.bs1.x_s, self.bs1.y_s, lambda x: y_s_int(x).__float__()),
                         self.bs1.length_s * 0.45)
        plt.plot([y12], [x12], linestyle='', marker='o', ms=4, mfc='black', color='green')
        plt.plot([y23], [x23], linestyle='', marker='o', ms=4, mfc='black', color='green')
        plt.grid()
        plt.show()


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
                                        pnt_cnt=35,
                                        gamma1_k_rel_rk=lambda r_rel: 0.5,
                                        gamma1_rk=None,
                                        gamma1_sa=None,
                                        center_point_rk=lambda r_rel: 0.46 + 0.02 * r_rel,
                                        center_point_sa=lambda r_rel: 0.51 + 0.02 * r_rel)

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
        self.assertEqual(section.r1, self.stage_prof.r1_rel_sa(0) * self.stage_prof.b_a_sa)
        self.assertEqual(section.gamma1_k,
                         self.stage_prof.get_gamma1(section.angle1_l) * self.stage_prof.gamma1_k_rel_sa(0))
        self.assertEqual(section.gamma1_s,
                         self.stage_prof.get_gamma1(section.angle1_l) * (1 - self.stage_prof.gamma1_k_rel_sa(0)))
        self.assertEqual(section.gamma2_k, self.stage_prof.gamma2_sa * self.stage_prof.gamma2_k_rel_sa(0))
        self.assertEqual(section.gamma2_s, self.stage_prof.gamma2_sa * (1 - self.stage_prof.gamma2_k_rel_sa(0)))
        self.assertEqual(section.center_point_pos, self.stage_prof.center_point_sa(0))

        section = self.stage_prof.sa_sections[1]
        self.assertEqual(section.b_a, self.stage_prof.b_a_sa)
        self.assertEqual(section.angle1, self.stage_prof.alpha0(r_av))
        self.assertEqual(section.delta1, self.stage_prof.get_delta(self.stage_prof.M_c0(r_av)))
        self.assertEqual(section.angle2, self.stage_prof.alpha1(r_av))
        self.assertEqual(section.delta2, self.stage_prof.get_delta(self.stage_prof.M_c1(r_av)))
        self.assertEqual(section.angle1_l, section.angle1 - section.delta1)
        self.assertEqual(section.angle2_l, section.angle2 - section.delta2)
        self.assertEqual(section.r1, self.stage_prof.r1_rel_sa(0.5) * self.stage_prof.b_a_sa)
        self.assertEqual(section.gamma1_k,
                         self.stage_prof.get_gamma1(section.angle1_l) * self.stage_prof.gamma1_k_rel_sa(0.5))
        self.assertEqual(section.gamma1_s,
                         self.stage_prof.get_gamma1(section.angle1_l) * (1 - self.stage_prof.gamma1_k_rel_sa(0.5)))
        self.assertEqual(section.gamma2_k, self.stage_prof.gamma2_sa * self.stage_prof.gamma2_k_rel_sa(0.5))
        self.assertEqual(section.gamma2_s, self.stage_prof.gamma2_sa * (1 - self.stage_prof.gamma2_k_rel_sa(0.5)))
        self.assertEqual(section.center_point_pos, self.stage_prof.center_point_sa(0.5))

        section = self.stage_prof.sa_sections[2]
        self.assertEqual(section.b_a, self.stage_prof.b_a_sa)
        self.assertEqual(section.angle1, self.stage_prof.alpha0(r_out))
        self.assertEqual(section.delta1, self.stage_prof.get_delta(self.stage_prof.M_c0(r_out)))
        self.assertEqual(section.angle2, self.stage_prof.alpha1(r_out))
        self.assertEqual(section.delta2, self.stage_prof.get_delta(self.stage_prof.M_c1(r_out)))
        self.assertEqual(section.angle1_l, section.angle1 - section.delta1)
        self.assertEqual(section.angle2_l, section.angle2 - section.delta2)
        self.assertEqual(section.r1, self.stage_prof.r1_rel_sa(1) * self.stage_prof.b_a_sa)
        self.assertEqual(section.gamma1_k,
                         self.stage_prof.get_gamma1(section.angle1_l) * self.stage_prof.gamma1_k_rel_sa(1))
        self.assertEqual(section.gamma1_s,
                         self.stage_prof.get_gamma1(section.angle1_l) * (1 - self.stage_prof.gamma1_k_rel_sa(1)))
        self.assertEqual(section.gamma2_k, self.stage_prof.gamma2_sa * self.stage_prof.gamma2_k_rel_sa(1))
        self.assertEqual(section.gamma2_s, self.stage_prof.gamma2_sa * (1 - self.stage_prof.gamma2_k_rel_sa(1)))
        self.assertEqual(section.center_point_pos, self.stage_prof.center_point_sa(1))

        section = self.stage_prof.rk_sections[0]
        self.assertEqual(section.b_a, self.stage_prof.b_a_rk)
        self.assertEqual(section.angle1, self.stage_prof.beta1(r_in))
        self.assertEqual(section.delta1, self.stage_prof.get_delta(self.stage_prof.M_w1(r_in)))
        self.assertEqual(section.angle2, self.stage_prof.beta2(r_in))
        self.assertEqual(section.delta2, self.stage_prof.get_delta(self.stage_prof.M_w2(r_in)))
        self.assertEqual(section.angle1_l, section.angle1 - section.delta1)
        self.assertEqual(section.angle2_l, section.angle2 - section.delta2)
        self.assertEqual(section.r1, self.stage_prof.r1_rel_rk(0) * self.stage_prof.b_a_rk)
        self.assertEqual(section.gamma1_k,
                         self.stage_prof.get_gamma1(section.angle1_l) * self.stage_prof.gamma1_k_rel_rk(0))
        self.assertEqual(section.gamma1_s,
                         self.stage_prof.get_gamma1(section.angle1_l) * (1 - self.stage_prof.gamma1_k_rel_rk(0)))
        self.assertEqual(section.gamma2_k, self.stage_prof.gamma2_rk * self.stage_prof.gamma2_k_rel_rk(0))
        self.assertEqual(section.gamma2_s, self.stage_prof.gamma2_rk * (1 - self.stage_prof.gamma2_k_rel_rk(0)))
        self.assertEqual(section.center_point_pos, self.stage_prof.center_point_rk(0))

        section = self.stage_prof.rk_sections[1]
        self.assertEqual(section.b_a, self.stage_prof.b_a_rk)
        self.assertEqual(section.angle1, self.stage_prof.beta1(r_av))
        self.assertEqual(section.delta1, self.stage_prof.get_delta(self.stage_prof.M_w1(r_av)))
        self.assertEqual(section.angle2, self.stage_prof.beta2(r_av))
        self.assertEqual(section.delta2, self.stage_prof.get_delta(self.stage_prof.M_w2(r_av)))
        self.assertEqual(section.angle1_l, section.angle1 - section.delta1)
        self.assertEqual(section.angle2_l, section.angle2 - section.delta2)
        self.assertEqual(section.r1, self.stage_prof.r1_rel_rk(0.5) * self.stage_prof.b_a_rk)
        self.assertEqual(section.gamma1_k,
                         self.stage_prof.get_gamma1(section.angle1_l) * self.stage_prof.gamma1_k_rel_rk(0.5))
        self.assertEqual(section.gamma1_s,
                         self.stage_prof.get_gamma1(section.angle1_l) * (1 - self.stage_prof.gamma1_k_rel_rk(0.5)))
        self.assertEqual(section.gamma2_k, self.stage_prof.gamma2_rk * self.stage_prof.gamma2_k_rel_rk(0.5))
        self.assertEqual(section.gamma2_s, self.stage_prof.gamma2_rk * (1 - self.stage_prof.gamma2_k_rel_rk(0.5)))
        self.assertEqual(section.center_point_pos, self.stage_prof.center_point_rk(0.5))

        section = self.stage_prof.rk_sections[2]
        self.assertEqual(section.b_a, self.stage_prof.b_a_rk)
        self.assertEqual(section.angle1, self.stage_prof.beta1(r_out))
        self.assertEqual(section.delta1, self.stage_prof.get_delta(self.stage_prof.M_w1(r_out)))
        self.assertEqual(section.angle2, self.stage_prof.beta2(r_out))
        self.assertEqual(section.delta2, self.stage_prof.get_delta(self.stage_prof.M_w2(r_out)))
        self.assertEqual(section.angle1_l, section.angle1 - section.delta1)
        self.assertEqual(section.angle2_l, section.angle2 - section.delta2)
        self.assertEqual(section.r1, self.stage_prof.r1_rel_rk(1) * self.stage_prof.b_a_rk)
        self.assertEqual(section.gamma1_k,
                         self.stage_prof.get_gamma1(section.angle1_l) * self.stage_prof.gamma1_k_rel_rk(1))
        self.assertEqual(section.gamma1_s,
                         self.stage_prof.get_gamma1(section.angle1_l) * (1 - self.stage_prof.gamma1_k_rel_rk(1)))
        self.assertEqual(section.gamma2_k, self.stage_prof.gamma2_rk * self.stage_prof.gamma2_k_rel_rk(1))
        self.assertEqual(section.gamma2_s, self.stage_prof.gamma2_rk * (1 - self.stage_prof.gamma2_k_rel_rk(1)))
        self.assertEqual(section.center_point_pos, self.stage_prof.center_point_rk(1))

    def test_plots(self):
        self.stage_prof.init_sections()
        self.stage_prof.compute_sections()
        self.stage_prof.plot_parameter_distribution(['alpha1', 'alpha2', 'beta1', 'beta2'])
        self.stage_prof.plot_profile_2d(1, width_rel=3)
        self.stage_prof.plot_profile_3d()
        self.stage_prof.plot_sa_sections()
        self.stage_prof.plot_rk_sections()


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
        self.turbine_profiler[0].profiling_type = ProfilingType.ConstantAngle
        self.turbine_profiler[0].gamma1_k_rel_rk = lambda r_rel: 0.5
        self.turbine_profiler[1].profiling_type = ProfilingType.ConstantAngle
        self.turbine_profiler[1].auto_sections_par = False
        self.turbine_profiler[1].rk_sections[0].gamma1_s = np.radians([15])[0]
        self.turbine_profiler[1].rk_sections[0].gamma1_k = np.radians([7])[0]
        self.turbine_profiler.compute_stage_profiles()

    def test_parameter_transfer(self):
        """Тестирование правильности передачи параметров из расчета по средней линии тока в профайлер"""

        stage_prof = self.turbine_profiler[0]
        geom = self.turbine.geom[0]
        gas_dynamic = self.turbine[0]

        self.assertEqual(stage_prof.c_p, gas_dynamic.c_p_gas)
        self.assertEqual(stage_prof.k, gas_dynamic.k_gas)
        self.assertEqual(stage_prof.D1_in, geom.D1 - geom.l1)
        self.assertEqual(stage_prof.D1_av, geom.D1)
        self.assertEqual(stage_prof.D1_out, geom.D1 + geom.l1)
        self.assertEqual(stage_prof.c1_av, gas_dynamic.c1)
        self.assertEqual(stage_prof.alpha1_av, gas_dynamic.alpha1)
        self.assertEqual(stage_prof.n, gas_dynamic.n)
        self.assertEqual(stage_prof.L_u_av, gas_dynamic.L_u)
        self.assertEqual(stage_prof.c2_a_av, gas_dynamic.c2_a)
        self.assertEqual(stage_prof.c2_u_av, gas_dynamic.c2_u)
        self.assertEqual(stage_prof.b_a_sa, geom.b_sa)
        self.assertEqual(stage_prof.b_a_rk, geom.b_rk)
        self.assertEqual(stage_prof.delta_a_sa, geom.delta_a_sa)
        self.assertEqual(stage_prof.delta_a_rk, geom.delta_a_rk)

        stage_prof = self.turbine_profiler[1]
        geom = self.turbine.geom[1]
        gas_dynamic = self.turbine[1]

        self.assertEqual(self.turbine_profiler[0].T2_stag, stage_prof.T0_stag)
        self.assertEqual(self.turbine_profiler[0].p2_stag, stage_prof.p0_stag)
        self.assertEqual(self.turbine_profiler[0].alpha2, stage_prof.alpha0)
        self.assertEqual(self.turbine_profiler[0].c2, stage_prof.c0)
        self.assertEqual(stage_prof.c_p, gas_dynamic.c_p_gas)
        self.assertEqual(stage_prof.k, gas_dynamic.k_gas)
        self.assertEqual(stage_prof.D1_in, geom.D1 - geom.l1)
        self.assertEqual(stage_prof.D1_av, geom.D1)
        self.assertEqual(stage_prof.D1_out, geom.D1 + geom.l1)
        self.assertEqual(stage_prof.c1_av, gas_dynamic.c1)
        self.assertEqual(stage_prof.alpha1_av, gas_dynamic.alpha1)
        self.assertEqual(stage_prof.n, gas_dynamic.n)
        self.assertEqual(stage_prof.L_u_av, gas_dynamic.L_u)
        self.assertEqual(stage_prof.c2_a_av, gas_dynamic.c2_a)
        self.assertEqual(stage_prof.c2_u_av, gas_dynamic.c2_u)
        self.assertEqual(stage_prof.b_a_sa, geom.b_sa)
        self.assertEqual(stage_prof.b_a_rk, geom.b_rk)
        self.assertEqual(stage_prof.delta_a_sa, geom.delta_a_sa)
        self.assertEqual(stage_prof.delta_a_rk, geom.delta_a_rk)

    def test_plot(self):
        self.turbine_profiler.plot_3d()