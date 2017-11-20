from .convective import BladeSectorCooler, BladeSection
from gas_turbine_cycle.gases import Air, KeroseneCombustionProducts
import unittest
import numpy as np
from scipy.interpolate import interp1d


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
        self.cooler = BladeSectorCooler(section=self.bs,
                                        height=67.6e-3,
                                        T_gas_stag=1223,
                                        G_gas=439,
                                        D_av=0.6476,
                                        wall_thickness=1.8e-3,
                                        T_wall_out_av=1023,
                                        T_cool_fluid0=600,
                                        G_air=0.144,
                                        lam_blade=lambda T: lam_blade_int(T).__float__(),
                                        cool_fluid=Air(),
                                        work_fluid=KeroseneCombustionProducts(),
                                        node_num=250)

    def test_av_params(self):
        """Проверка теплового баланса."""
        self.cooler.compute_av_params()
        k = 1 / (1 / self.cooler.alpha_gas_av + 1 / self.cooler.alpha_cool_fluid_av +
                 self.cooler.wall_thickness / self.cooler.lam_blade(self.cooler.T_wall_av))
        Q1 = k * (self.cooler.T_gas_stag - self.cooler.T_cool_fluid_av)
        Q2 = self.cooler.alpha_gas_av * (self.cooler.T_gas_stag - self.cooler.T_wall_out)
        self.assertAlmostEqual(Q1, Q2, places=5)

    def test_channel_width_plot(self):
        self.cooler.plot_channel_width_plot(np.linspace(0.05, 0.25, 10))

    def test_local_params(self):
        self.cooler.compute_av_params()
        self.cooler.compute_local_params()
        self.cooler.plot_T_wall()


