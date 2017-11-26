from .convective_defl import SectorCooler, BladeSection, GasBladeHeatExchange
from .tools import DeflectorAverageParamCalculator, FilmCalculator
from gas_turbine_cycle.gases import Air, KeroseneCombustionProducts
import unittest
import numpy as np
from scipy.interpolate import interp1d
from scipy.misc import derivative


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
        self.cooler = SectorCooler(section=self.bs,
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
        mu_gas, lam_gas, Re_gas, Nu_gas, alpha_gas_av = GasBladeHeatExchange.get_alpha_gas_av(self.cooler.section,
                                                                                              self.cooler.height,
                                                                                              self.cooler.T_gas_stag,
                                                                                              self.cooler.G_gas,
                                                                                              self.cooler.D_av,
                                                                                              self.cooler.work_fluid)
        self.ave_param = DeflectorAverageParamCalculator(self.cooler.section,
                                                         self.cooler.height,
                                                         self.cooler.D_av,
                                                         self.cooler.wall_thickness,
                                                         1023,
                                                         self.cooler.T_cool_fluid0,
                                                         self.cooler.T_gas_stag,
                                                         alpha_gas_av,
                                                         self.cooler.G_cool,
                                                         self.cooler.lam_blade,
                                                         self.cooler.cool_fluid
                                                         )

    def test_av_params(self):
        """Проверка теплового баланса."""
        self.ave_param.compute()
        k = 1 / (1 / self.ave_param.alpha_out + 1 / self.ave_param.alpha_cool_fluid_av +
                 self.cooler.wall_thickness / self.cooler.lam_blade(self.ave_param.T_wall_av))
        Q1 = k * (self.cooler.T_gas_stag - self.ave_param.T_cool_fluid_av)
        Q2 = self.ave_param.alpha_out * (self.ave_param.T_out_stag - self.ave_param.T_wall_out)
        Q3 = self.ave_param.alpha_cool_fluid_av * (self.ave_param.T_wall_in -
                                                   self.ave_param.T_cool_fluid_av)
        self.assertAlmostEqual(Q1, Q2, places=7)
        self.assertAlmostEqual(Q2, Q3, places=7)

    # def test_channel_width_plot(self):
    #     self.cooler.plot_channel_width_plot(np.linspace(0.05, 0.25, 10))

    def test_local_params(self):
        "Проверка решения дифференциального уравнения."
        cool_fluid = self.cooler.cool_fluid
        ave_param = self.ave_param
        local_param = self.cooler.local_param
        ave_param.compute()
        self.cooler.channel_width = ave_param.channel_width
        self.cooler.T_wall_av = ave_param.T_wall_av
        self.cooler.compute()

        def alpha_cool(x, T_cool):
            return (0.02 * cool_fluid.lam(T_cool) / (2 * ave_param.channel_width) *
                    (local_param.G_cool(x) / (self.cooler.height * cool_fluid.mu(T_cool))) ** 0.8)

        def heat_trans_coef(x, T_cool):
            return 1 / (1 / (alpha_cool(x, T_cool)) + 1 / (local_param.alpha_out(x)) +
                        self.cooler.wall_thickness / self.cooler.lam_blade(ave_param.T_wall_av).__float__())

        def T_cool_der(x, T_cool):
            if x >= 0:
                return (heat_trans_coef(x, T_cool) * self.cooler.height * (self.cooler.T_gas_stag - T_cool) /
                        (0.5 * local_param.G_cool(x) * cool_fluid.c_p_real_func(T_cool)))
            else:
                return -(heat_trans_coef(x, T_cool) * self.cooler.height * (self.cooler.T_gas_stag - T_cool) /
                         (0.5 * local_param.G_cool(x) * cool_fluid.c_p_real_func(T_cool)))

        for i in range(1, len(local_param.x_arr)):
            T_der1 = ((local_param.T_cool_fluid_arr[i] - local_param.T_cool_fluid_arr[i - 1]) /
                     (local_param.x_arr[i] - local_param.x_arr[i - 1]))

            if local_param.x_arr[i] <= 0:
                T_der2 = T_cool_der(local_param.x_arr[i], local_param.T_cool_fluid_arr[i])
            else:
                T_der2 = T_cool_der(local_param.x_arr[i - 1], local_param.T_cool_fluid_arr[i - 1])

            self.assertAlmostEqual(abs(T_der1 - T_der2) / T_der1, 0, places=4)
        self.cooler.local_param.plot_T_wall()

    def test_film_calculator(self):
        x_hole = [-0.06, -0.04, -0.015, 0.008, 0.015, 0.04]
        film = FilmCalculator(x_hole=x_hole,
                              hole_num=[15 for _ in x_hole],
                              d_hole=[0.001 for _ in x_hole],
                              phi_hole=[0.9 for _ in x_hole],
                              mu_hole=[0.85 for _ in x_hole],
                              T_gas_stag=self.cooler.T_gas_stag,
                              p_gas_stag=9.5e5,
                              v_gas=lambda x: 100,
                              c_p_gas_av=1150,
                              work_fluid=KeroseneCombustionProducts(),
                              alpha_gas=lambda x: 4500,
                              T_cool=lambda x: 650,
                              G_cool=lambda x: 0.25,
                              p_cool_stag0=10e5,
                              c_p_cool_av=1100,
                              cool_fluid=Air(),
                              channel_width=0.001,
                              height=0.03)
        film.compute()
        x_arr = np.linspace(x_hole[0] - 0.06, x_hole[len(x_hole) - 1] + 0.06, 400)
        film.plot_film_eff(0, x_arr, create_fig=True, show=False)
        film.plot_film_eff(1, x_arr, show=False)
        film.plot_film_eff(2, x_arr, show=False)
        film.plot_film_eff(3, x_arr, show=False)
        film.plot_film_eff(4, x_arr, show=False)
        film.plot_film_eff(5, x_arr, show=True)

        film.plot_G_cool(x_arr)
        film.plot_T_film(x_arr)
        film.plot_alpha_film(x_arr, (4000, 8000))


