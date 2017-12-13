from .convective_defl import SectorCooler, BladeSection, GasBladeHeatExchange
from .tools import DeflectorAverageParamCalculator, FilmCalculator
from .film_defl import FilmSectorCooler, FilmBladeCooler
from gas_turbine_cycle.gases import Air, KeroseneCombustionProducts, NaturalGasCombustionProducts
import unittest
import numpy as np
from scipy.interpolate import interp1d
from ..average_streamline.turbine import TurbineType, Turbine
from ..profiling.turbine import TurbineProfiler, ProfilingType


class BladeSectorTest(unittest.TestCase):
    def setUp(self):
        self.bs = BladeSection(angle1=np.radians([90])[0],
                               angle2=np.radians([21])[0],
                               delta1=np.radians([2])[0],
                               delta2=np.radians([2])[0],
                               b_a=0.07,
                               r1=0.01 / 2,
                               convex='left',
                               pnt_count=30,
                               s2=0.0003)
        self.bs.compute_profile()
        lam_blade_arr = np.array([19, 22, 24, 27])
        T_wall_arr = np.array([600, 700, 800, 900]) + 273
        lam_blade_int = interp1d(T_wall_arr, lam_blade_arr, bounds_error=False, fill_value='extrapolate')
        self.conv_cooler = SectorCooler(section=self.bs,
                                        height=67.6e-3,
                                        T_gas_stag=1223,
                                        G_gas=439,
                                        D_av=0.6476,
                                        wall_thickness=1.8e-3,
                                        T_wall_av=980,
                                        T_cool_fluid0=600,
                                        G_cool=0.144,
                                        lam_blade=lambda T: lam_blade_int(T).__float__(),
                                        cool_fluid=Air(),
                                        work_fluid=KeroseneCombustionProducts(),
                                        node_num=250,
                                        channel_width=0.001
                                        )
        mu_gas, lam_gas, Re_gas, Nu_gas, alpha_gas_av = GasBladeHeatExchange.get_alpha_gas_av(self.conv_cooler.section,
                                                                                              self.conv_cooler.height,
                                                                                              self.conv_cooler.T_gas_stag,
                                                                                              self.conv_cooler.G_gas,
                                                                                              self.conv_cooler.D_av,
                                                                                              self.conv_cooler.work_fluid)
        self.ave_param = DeflectorAverageParamCalculator(self.conv_cooler.section,
                                                         self.conv_cooler.height,
                                                         self.conv_cooler.D_av,
                                                         self.conv_cooler.wall_thickness,
                                                         1023,
                                                         self.conv_cooler.T_cool_fluid0,
                                                         self.conv_cooler.T_gas_stag,
                                                         alpha_gas_av,
                                                         self.conv_cooler.G_cool,
                                                         self.conv_cooler.lam_blade,
                                                         self.conv_cooler.cool_fluid
                                                         )

        self.x_hole1 = [-0.06, -0.04, -0.015, 0.008, 0.015, 0.04]
        self.film1 = FilmCalculator(x_hole=self.x_hole1,
                                    hole_num=[15 for _ in self.x_hole1],
                                    d_hole=[0.001 for _ in self.x_hole1],
                                    phi_hole=[0.9 for _ in self.x_hole1],
                                    mu_hole=[0.85 for _ in self.x_hole1],
                                    T_gas_stag=self.conv_cooler.T_gas_stag,
                                    p_gas_stag=9.5e5,
                                    v_gas=lambda x: 100,
                                    c_p_gas_av=1150,
                                    work_fluid=KeroseneCombustionProducts(),
                                    alpha_gas=lambda x: 4500,
                                    T_cool=lambda x: 650,
                                    G_cool0=0.25,
                                    p_cool_stag0=10e5,
                                    c_p_cool_av=1100,
                                    cool_fluid=Air(),
                                    height=0.03)

        self.x_hole2 = [-0.015, 0.025]
        self.film2 = FilmCalculator(x_hole=self.x_hole2,
                                    hole_num=[15 for _ in self.x_hole2],
                                    d_hole=[0.001 for _ in self.x_hole2],
                                    phi_hole=[0.9 for _ in self.x_hole2],
                                    mu_hole=[0.85 for _ in self.x_hole2],
                                    T_gas_stag=self.conv_cooler.T_gas_stag,
                                    p_gas_stag=9.5e5,
                                    v_gas=lambda x: 100,
                                    c_p_gas_av=1150,
                                    work_fluid=KeroseneCombustionProducts(),
                                    alpha_gas=lambda x: 4500,
                                    T_cool=lambda x: 650,
                                    G_cool0=0.25,
                                    p_cool_stag0=10e5,
                                    c_p_cool_av=1100,
                                    cool_fluid=Air(),
                                    height=0.03)

        self.x_hole_rel = [-0.8, -0.6, -0.2, 0.2, 0.4, 0.7]
        self.film_cooler = FilmSectorCooler(section=self.bs,
                                            height=self.conv_cooler.height,
                                            channel_width=1e-3,
                                            wall_thickness=self.conv_cooler.wall_thickness,
                                            D_av=self.conv_cooler.D_av,
                                            lam_blade=lambda T: lam_blade_int(T).__float__(),
                                            T_wall_av=1000,
                                            x_hole_rel=self.x_hole_rel,
                                            hole_num=[15 for _ in self.x_hole_rel],
                                            d_hole=[1.5e-3 for _ in self.x_hole_rel],
                                            phi_hole=[0.9 for _ in self.x_hole_rel],
                                            mu_hole=[0.85 for _ in self.x_hole_rel],
                                            T_gas_stag=self.conv_cooler.T_gas_stag,
                                            p_gas_stag=9.5e5,
                                            G_gas=self.conv_cooler.G_gas,
                                            c_p_gas_av=1150,
                                            lam_gas_in=0.1,
                                            lam_gas_out=0.2,
                                            work_fluid=KeroseneCombustionProducts(),
                                            T_cool0=self.conv_cooler.T_cool_fluid0,
                                            p_cool_stag0=10e5,
                                            G_cool0=self.conv_cooler.G_cool,
                                            c_p_cool_av=1100,
                                            cool_fluid=Air(),
                                            node_num=250)

    def test_av_params(self):
        """Проверка теплового баланса."""
        self.ave_param.compute()
        k = 1 / (1 / self.ave_param.alpha_out + 1 / self.ave_param.alpha_cool_fluid_av +
                 self.conv_cooler.wall_thickness / self.conv_cooler.lam_blade(self.ave_param.T_wall_av))
        Q1 = k * (self.conv_cooler.T_gas_stag - self.ave_param.T_cool_fluid_av)
        Q2 = self.ave_param.alpha_out * (self.ave_param.T_out_stag - self.ave_param.T_wall_out)
        Q3 = self.ave_param.alpha_cool_fluid_av * (self.ave_param.T_wall_in -
                                                   self.ave_param.T_cool_fluid_av)
        self.assertAlmostEqual(Q1, Q2, places=7)
        self.assertAlmostEqual(Q2, Q3, places=7)

    # def test_channel_width_plot(self):
    #     self.cooler.plot_channel_width_plot(np.linspace(0.05, 0.25, 10))

    def test_local_params(self):
        """Проверка решения дифференциального уравнения."""
        cool_fluid = self.conv_cooler.cool_fluid
        ave_param = self.ave_param
        local_param = self.conv_cooler.local_param
        ave_param.compute()
        self.conv_cooler.channel_width = ave_param.channel_width
        self.conv_cooler.T_wall_av = ave_param.T_wall_av
        self.conv_cooler.compute()

        def alpha_cool(x, T_cool):
            return (0.02 * cool_fluid.lam(T_cool) / (2 * ave_param.channel_width) *
                    (local_param.G_cool(x) / (self.conv_cooler.height * cool_fluid.mu(T_cool))) ** 0.8)

        def heat_trans_coef(x, T_cool):
            return 1 / (1 / (alpha_cool(x, T_cool)) + 1 / (local_param.alpha_out(x)) +
                        self.conv_cooler.wall_thickness / self.conv_cooler.lam_blade(ave_param.T_wall_av).__float__())

        def T_cool_der(x, T_cool):
            if x >= 0:
                return (heat_trans_coef(x, T_cool) * self.conv_cooler.height * (self.conv_cooler.T_gas_stag - T_cool) /
                        (0.5 * local_param.G_cool(x) * cool_fluid.c_p_real_func(T_cool)))
            else:
                return -(heat_trans_coef(x, T_cool) * self.conv_cooler.height * (self.conv_cooler.T_gas_stag - T_cool) /
                         (0.5 * local_param.G_cool(x) * cool_fluid.c_p_real_func(T_cool)))

        for i in range(1, len(local_param.x_arr)):
            T_der1 = ((local_param.T_cool_fluid_arr[i] - local_param.T_cool_fluid_arr[i - 1]) /
                     (local_param.x_arr[i] - local_param.x_arr[i - 1]))

            if local_param.x_arr[i] <= 0:
                T_der2 = T_cool_der(local_param.x_arr[i], local_param.T_cool_fluid_arr[i])
            else:
                T_der2 = T_cool_der(local_param.x_arr[i - 1], local_param.T_cool_fluid_arr[i - 1])

            self.assertAlmostEqual(abs(T_der1 - T_der2) / T_der1, 0, places=4)

        self.conv_cooler.local_param.plot_T_wall()
        self.conv_cooler.local_param.plot_all()

    def test_film_calculator_with_six_holes_rows(self):
        self.film1.compute()
        x_arr = np.linspace(self.x_hole1[0] - 0.06, self.x_hole1[len(self.x_hole1) - 1] + 0.06, 400)
        self.film1.plot_film_eff(0, x_arr, create_fig=True, show=False)
        self.film1.plot_film_eff(1, x_arr, show=False)
        self.film1.plot_film_eff(2, x_arr, show=False)
        self.film1.plot_film_eff(3, x_arr, show=False)
        self.film1.plot_film_eff(4, x_arr, show=False)
        self.film1.plot_film_eff(5, x_arr, show=True)

        self.film1.plot_G_cool(x_arr)
        self.film1.plot_T_film(x_arr)
        self.film1.plot_alpha_film(x_arr, (4000, 8000))

    def test_film_calculator_with_two_holes_rows(self):
        self.film2.compute()
        x_arr = np.linspace(self.x_hole2[0] - 0.20, self.x_hole2[len(self.x_hole2) - 1] + 0.20, 400)
        self.film2.plot_film_eff(0, x_arr, create_fig=True, show=False)
        self.film2.plot_film_eff(1, x_arr, show=True)

        self.film2.plot_G_cool(x_arr)
        self.film2.plot_T_film(x_arr)
        self.film2.plot_alpha_film(x_arr, (4000, 8000))

    def test_deflector_film_sector_cooling(self):
        ave_param = self.ave_param
        ave_param.compute()
        self.film_cooler.T_wall_av = ave_param.T_wall_av
        self.film_cooler.channel_width = ave_param.channel_width

        self.film_cooler.set_calculators()
        self.film_cooler.compute()
        self.assertNotEqual(self.film_cooler.alpha_film_av, None)
        self.assertNotEqual(self.film_cooler.T_film_av, None)
        self.assertNotEqual(self.film_cooler.G_cool_av, None)

        print('alpha_film_av = %.3f' % self.film_cooler.alpha_film_av)
        print('T_film_av = %.3f' % self.film_cooler.T_film_av)
        print('G_cool_av = %.4f' % self.film_cooler.G_cool_av)
        print('T_wall_out_av = %.3f' % self.film_cooler.T_wall_out_av)

        self.film_cooler.local_param.plot_all()
        self.film_cooler.film.plot_T_film(self.film_cooler.local_param.x_arr)


class DeflectorBladeFilmCoolingTest(unittest.TestCase):
    def setUp(self):
        self.turbine = Turbine(TurbineType.Compressor,
                               T_g_stag=1450,
                               p_g_stag=400e3,
                               G_turbine=25,
                               work_fluid=KeroseneCombustionProducts(),
                               G_fuel=2.5,
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

        stage_prof = self.turbine_profiler[0]
        lam_blade_arr = np.array([17, 19, 22, 24, 27, 29])
        T_wall_arr = np.array([200, 600, 700, 800, 900, 1200]) + 273
        lam_blade_int = interp1d(T_wall_arr, lam_blade_arr, bounds_error=False, fill_value='extrapolate')
        x_hole_rel = [-0.8, -0.5, -0.15, 0.1, 0.35, 0.6]

        self.film_blade = FilmBladeCooler(sections=stage_prof.sa_sections,
                                          channel_width=0.001,
                                          wall_thickness=0.001,
                                          D_in=stage_prof.D1_in,
                                          D_out=stage_prof.D1_out,
                                          T_wall_out_av_init=1123,
                                          lam_blade=lambda T: lam_blade_int(T).__float__(),
                                          x_hole_rel=x_hole_rel,
                                          hole_num=[20 for _ in x_hole_rel],
                                          d_hole=[0.5e-3 for _ in x_hole_rel],
                                          phi_hole=[0.9 for _ in x_hole_rel],
                                          mu_hole=[0.86 for _ in x_hole_rel],
                                          T_gas_stag=stage_prof.T0_stag,
                                          p_gas_stag=stage_prof.p0_stag,
                                          G_gas=self.turbine.G_turbine,
                                          c_p_gas_av=stage_prof.c_p,
                                          lam_gas_in=stage_prof.lam_c0,
                                          lam_gas_out=stage_prof.lam_c1,
                                          work_fluid=type(self.turbine.work_fluid)(),
                                          T_cool0=650,
                                          p_cool_stag0=self.turbine.p_g_stag + 20e3,
                                          G_cool0=0.2,
                                          cool_fluid=Air())

    def test_partition(self):
        self.assertEqual(sum(self.film_blade.G_gas_arr), self.film_blade.G_gas)
        self.assertEqual(self.film_blade.G_gas_arr[0] * 2, self.film_blade.G_gas_arr[1])
        self.assertEqual(self.film_blade.G_gas_arr[2] * 2, self.film_blade.G_gas_arr[1])
        self.assertEqual(sum(self.film_blade.G_cool0_arr), self.film_blade.G_cool0)
        self.assertEqual(self.film_blade.G_cool0_arr[0] * 2, self.film_blade.G_cool0_arr[1])
        self.assertEqual(self.film_blade.G_cool0_arr[2] * 2, self.film_blade.G_cool0_arr[1])

        self.assertEqual(sum(self.film_blade.sector_height_arr), 0.5 * (self.film_blade.D_out - self.film_blade.D_in))
        self.assertEqual(self.film_blade.sector_height_arr[0] * 2, self.film_blade.sector_height_arr[1])
        self.assertEqual(self.film_blade.sector_height_arr[2] * 2, self.film_blade.sector_height_arr[1])

        self.film_blade.plot_lam_gas_out()

    def test_calculation(self):
        self.film_blade.compute()

        print('alpha_film_av = %.3f' % self.film_blade.alpha_film_av)
        print('T_film_av = %.3f' % self.film_blade.T_film_av)
        print('G_cool_av = %.4f' % self.film_blade.G_cool_av)
        print('T_wall_out_av = %.3f' % self.film_blade.T_wall_out_av)

        self.assertLess(abs(self.film_blade.av_param.T_cool_fluid_av - self.film_blade.T_cool_av) /
                        self.film_blade.T_cool_av * 100, 1)

        self.film_blade.sectors[0].local_param.plot_all()
        self.film_blade.sectors[1].local_param.plot_all()
        self.film_blade.sectors[2].local_param.plot_all()

        self.film_blade.sectors[0].plot_v_gas()
        self.film_blade.plot_T_wall()



