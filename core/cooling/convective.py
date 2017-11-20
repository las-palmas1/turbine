from ..profiling.section import BladeSection
import numpy as np
from gas_turbine_cycle.gases import Air, IdealGas
import typing
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


class BladeSectorCooler:
    """Класс для расчета участка лопатки с постоянным профилем и параметрами газа на входе."""
    def __init__(self, section: BladeSection, height, T_gas_stag, G_gas, D_av,
                 wall_thickness, T_wall_out_av, T_cool_fluid0, G_air,
                 lam_blade: typing.Callable[[float], float],
                 cool_fluid: IdealGas, work_fluid: IdealGas, node_num: int=500):
        """
        :param section: BladeSection. \n
            Сечение рассчитываемого участка лопатки.
        :param height: float. \n
            Высота рассчитываемого участка лопакти.
        :param T_gas_stag: float. \n
            Температура торможения газа на входе в лопаточный венец.
        :param G_gas: float. \n
            Расход газа через лопаточный венец.
        :param D_av: float. \n
            Средний диаметр участка.
        :param wall_thickness: float. \n
            Толщина стенки.
        :param T_wall_out_av: float. \n
            Средняя температура наружной поверхности стенки участка лопатки.ы
        :param T_cool_fluid0: float. \n
            Температура охраждающего тела на входе в лопатку.
        :param G_air: float. \n
            Расход охлаждающего тела.
        :param lam_blade:
            Тепопроводность материала лопатки в зависимости от температуры
        :param cool_fluid: IdealGas. \n
            Охлаждающее тело.
        :param work_fluid: IdealGas. \n
            Рабочее тело турбины.
        :param node_num: int, optional. \n
            Число узлов на интервале решения уравнение теплового баланса.
        """
        self.section = section
        self.height = height
        self.T_gas_stag = T_gas_stag
        self.G_gas = G_gas
        self.D_av = D_av
        self.angle1 = section.angle1
        self.angle2 = section.angle2
        self.wall_thickness = wall_thickness
        self.T_wall_out = T_wall_out_av
        self.T_cool_fluid0 = T_cool_fluid0
        self.G_air = G_air
        self.lam_blade = lam_blade
        self.cool_fluid = cool_fluid
        self.work_fluid = work_fluid
        self.node_num = node_num

        self.x_2k = None
        self.x_1k = None
        self.x_0 = None
        self.x_1s = None
        self.x_2s = None
        self.x_3s = None
        self.alpha_gas_inlet = None
        self.T_cool_fluid_s_arr = None
        self.T_cool_fluid_k_arr = None
        self.alpha_cool_fluid_s_arr = None
        self.alpha_cool_fluid_k_arr = None
        self.alpha_gas_s_arr = None
        self.alpha_gas_k_arr = None
        self.T_wall_s_arr = None
        self.T_wall_k_arr = None
        self.x_s_arr = None
        self.x_k_arr = None

    @classmethod
    def get_nusselt_coef(cls, angle1, angle2):
        return 0.07 + 100 * (np.degrees(angle1) + np.degrees(angle2)) ** (-2)

    def _get_av_cool_fluid_temp(self, Q_blade, G_air, T_cool_fluid0):
        cool_fluid_temp_res = 1.
        self.cool_fluid.T1 = T_cool_fluid0
        T_cool_fluid_av = None
        T_cool_fluid_av_old = None

        while cool_fluid_temp_res > 0.001:
            T_cool_fluid_av_old = self.cool_fluid.T2
            T_cool_fluid_av = T_cool_fluid0 + Q_blade / (2 * G_air * self.cool_fluid.c_p_av_int)
            cool_fluid_temp_res = abs(T_cool_fluid_av_old - T_cool_fluid_av) / T_cool_fluid_av
            self.cool_fluid.T2 = T_cool_fluid_av
        return T_cool_fluid_av, T_cool_fluid_av_old, cool_fluid_temp_res

    def _compute_average_parameters(self, G_air):
        self.mu_gas = self.work_fluid.mu(self.T_gas_stag)
        self.lam_gas = self.work_fluid.lam(self.T_gas_stag)
        self.Re_gas = self.G_gas * self.section.chord_length / (np.pi * self.D_av * self.height *
                                                                self.mu_gas * np.sin(self.angle2))
        self.nusselt_coef = self.get_nusselt_coef(self.angle1, self.angle2)
        self.Nu_gas = self.nusselt_coef * self.Re_gas ** 0.68
        self.alpha_gas_av = self.Nu_gas * self.lam_gas / self.section.chord_length
        self.perimeter = (self.section.length_in_edge + self.section.length_s + self.section.length_k +
                          self.section.length_out_edge)
        self.Q_blade = self.alpha_gas_av * self.perimeter * self.height * (self.T_gas_stag - self.T_wall_out)
        self.square = self.perimeter * self.height
        self.T_wall_av: float = fsolve(lambda x: [self.T_wall_out -
                                                  self.Q_blade * self.wall_thickness /
                                                  (2 * self.square * self.lam_blade(x[0])) - x[0]],
                                       np.array([self.T_wall_out]))[0]
        self.delta_T = self.Q_blade * self.wall_thickness / (self.square * self.lam_blade(self.T_wall_av))
        self.T_wall_in = self.T_wall_out - self.delta_T
        self.T_cool_fluid_av, self.T_cool_fluid_old, \
        self.cool_fluid_temp_res = self._get_av_cool_fluid_temp(self.Q_blade, G_air, self.T_cool_fluid0)
        self.c_p_air_av = self.cool_fluid.c_p_av_int
        self.mu_cool_fluid = self.cool_fluid.mu(self.T_cool_fluid_av)
        self.Re_cool_fluid = G_air / (self.height * self.cool_fluid.mu(self.T_cool_fluid_av))
        self.alpha_cool_fluid_av = self.Q_blade / (self.square * (self.T_wall_in - self.T_cool_fluid_av))
        self.lam_cool_fluid = self.cool_fluid.lam(self.T_cool_fluid_av)
        self.epsilon = 0.01 * self.lam_cool_fluid * (1 / (self.height * self.mu_cool_fluid)) ** 0.8
        self.D = (1 / self.alpha_gas_av * (self.T_gas_stag - self.T_cool_fluid0) / (self.T_gas_stag - self.T_wall_out) -
                  1 / self.alpha_gas_av - self.wall_thickness / self.lam_blade(self.T_wall_av))
        self.channel_width = self.epsilon * G_air**0.8 * (self.D - self.square / (2 * G_air * self.c_p_air_av))

    def _get_regions_bound(self):
        x_2k = self.section.length_k + self.section.length_in_edge_k
        x_1k = self.section.length_in_edge_k
        x_0 = 0
        x_1s = -self.section.length_in_edge_s
        x_2s = -self.section.length_in_edge_s - self.section.length_s + self.section.chord_length / 3
        x_3s = -self.section.length_in_edge_s - self.section.length_s
        return x_3s, x_2s, x_1s, x_0, x_1k, x_2k

    @classmethod
    def _get_alpha_gas(cls, x, alpha_gas_inlet, alpha_gas_av, x_3s, x_2s, x_1s, x_0, x_1k, x_2k):
        if x_2s > x >= x_3s:
            return 1.5 * alpha_gas_av
        elif x_1s > x >= x_2s:
            return 0.6 * alpha_gas_av
        elif x_0 > x >= x_1s:
            return alpha_gas_inlet
        elif x_1k > x >= x_0:
            return alpha_gas_inlet
        elif x_2k >= x >= x_1k:
            return alpha_gas_av

    def _get_alpha_cool_fluid(self, T_cool_fluid):
        return (0.02 * self.cool_fluid.lam(T_cool_fluid) / (2 * self.channel_width) *
                (self.G_air / (self.height * self.cool_fluid.mu(T_cool_fluid))) ** 0.8)

    def _get_heat_transfer_coef(self, x, T_cool_fluid, alpha_gas_inlet, alpha_gas_av,
                                x_3s, x_2s, x_1s, x_0, x_1k, x_2k):
        return 1 / (1 / self._get_alpha_cool_fluid(T_cool_fluid) +
                    1 / self._get_alpha_gas(x, alpha_gas_inlet, alpha_gas_av, x_3s, x_2s, x_1s, x_0, x_1k, x_2k) +
                    self.wall_thickness / self.lam_blade(self.T_wall_av)).__float__()

    @classmethod
    def _solve_equation(cls, x_arr: np.ndarray, der: typing.Callable[[float, float], float], val0, eq_type='s'):
        val_arr = [val0]
        print()
        for i in range(1, len(x_arr)):
            if eq_type == 'k':
                val = val_arr[i - 1] + (x_arr[i] - x_arr[i - 1]) * der(val_arr[i - 1], x_arr[i - 1])
            elif eq_type == 's':
                val = val_arr[i - 1] + (x_arr[i - 1] - x_arr[i]) * der(val_arr[i - 1], x_arr[i - 1])
            print(i, val, x_arr[i])
            val_arr.append(val)
        return val_arr

    def _get_T_cool_fluid_der(self, x, T_cool_fluid, alpha_gas_inlet, alpha_gas_av,
                              x_3s, x_2s, x_1s, x_0, x_1k, x_2k):
        heat_trans_coef = self._get_heat_transfer_coef(x, T_cool_fluid, alpha_gas_inlet, alpha_gas_av,
                                                       x_3s, x_2s, x_1s, x_0, x_1k, x_2k)
        return (self.height * (self.T_gas_stag - T_cool_fluid) * heat_trans_coef /
                (0.5 * self.G_air * self.c_p_air_av))

    def _get_T_wall(self, x, T_cool_fluid, alpha_gas_inlet, alpha_gas_av,
                    x_3s, x_2s, x_1s, x_0, x_1k, x_2k):
        heat_trans_coef = self._get_heat_transfer_coef(x, T_cool_fluid, alpha_gas_inlet, alpha_gas_av,
                                                       x_3s, x_2s, x_1s, x_0, x_1k, x_2k)
        alpha_gas = self._get_alpha_gas(x, alpha_gas_inlet, alpha_gas_av, x_3s, x_2s, x_1s, x_0, x_1k, x_2k)
        return self.T_gas_stag - heat_trans_coef / alpha_gas * (self.T_gas_stag - T_cool_fluid)

    def compute_local_params(self):
        self.x_3s, self.x_2s, self.x_1s, self.x_0, self.x_1k, self.x_2k = self._get_regions_bound()
        self.alpha_gas_inlet = (0.74 * self.lam_gas / (2 * self.section.r1) *
                                (self.G_gas * 2 * self.section.r1 /
                                 (np.pi * self.D_av * self.height * np.sin(self.angle1) * self.mu_gas)) ** 0.5)
        self.x_s_arr = np.array(np.linspace(self.x_0, self.x_3s, self.node_num))
        self.x_k_arr = np.array(np.linspace(self.x_0, self.x_2k, self.node_num))
        self.T_cool_fluid_s_arr = self._solve_equation(self.x_s_arr,
                                                       lambda T, x:
                                                       self._get_heat_transfer_coef(x, T, self.alpha_gas_inlet,
                                                                                    self.alpha_gas_av,
                                                                                    *self._get_regions_bound()),
                                                       self.T_cool_fluid0,
                                                       eq_type='s')
        self.T_cool_fluid_k_arr = self._solve_equation(self.x_k_arr,
                                                       lambda T, x:
                                                       self._get_heat_transfer_coef(x, T, self.alpha_gas_inlet,
                                                                                    self.alpha_gas_av,
                                                                                    *self._get_regions_bound()),
                                                       self.T_cool_fluid0,
                                                       eq_type='k')

        self.T_wall_s_arr = []

        for T_cool_fluid_s, x_s in zip(self.T_cool_fluid_s_arr, self.x_s_arr):
            T_wall = self._get_T_wall(x_s, T_cool_fluid_s, self.alpha_gas_inlet, self.alpha_gas_av,
                                      *self._get_regions_bound())
            self.T_wall_s_arr.append(T_wall)

        self.T_wall_k_arr = []

        for T_cool_fluid_k, x_k in zip(self.T_cool_fluid_k_arr, self.x_k_arr):
            T_wall = self._get_T_wall(x_k, T_cool_fluid_k, self.alpha_gas_inlet, self.alpha_gas_av,
                                      *self._get_regions_bound())
            self.T_wall_k_arr.append(T_wall)

        self.alpha_cool_fluid_s_arr = []
        for T_cool_fluid_s in self.T_cool_fluid_s_arr:
            alpha_cool = self._get_alpha_cool_fluid(T_cool_fluid_s)
            self.alpha_cool_fluid_s_arr.append(alpha_cool)

        self.alpha_cool_fluid_k_arr = []
        for T_cool_fluid_k in self.T_cool_fluid_k_arr:
            alpha_cool = self._get_alpha_cool_fluid(T_cool_fluid_k)
            self.alpha_cool_fluid_k_arr.append(alpha_cool)

        self.alpha_gas_s_arr = []
        for x_s in self.x_s_arr:
            alpha_gas = self._get_alpha_gas(x_s, self.alpha_gas_inlet, self.alpha_gas_av,
                                            *self._get_regions_bound())
            self.alpha_gas_s_arr.append(alpha_gas)

        self.alpha_gas_k_arr = []
        for x_k in self.x_k_arr:
            alpha_gas = self._get_alpha_gas(x_k, self.alpha_gas_inlet, self.alpha_gas_av,
                                            *self._get_regions_bound())
            self.alpha_gas_k_arr.append(alpha_gas)

    def plot_channel_width_plot(self, G_air_arr, figsize=(6, 4)):
        channel_width_arr = []
        for G_air in G_air_arr:
            self._compute_average_parameters(G_air)
            channel_width_arr.append(self.channel_width)

        plt.figure(figsize=figsize)
        plt.plot(G_air_arr, np.array(channel_width_arr) * 1e3, lw=1, color='red')
        plt.xlabel(r'$G_в,\ кг/с$', fontsize=10)
        plt.ylabel(r'$\delta,\ мм$', fontsize=10)
        plt.grid()
        plt.show()

    def compute_av_params(self):
        self._compute_average_parameters(self.G_air)

    def plot_T_wall(self, figsize=(8, 6)):
        plt.figure(figsize=figsize)
        plt.plot(self.x_s_arr * 1e3, self.T_wall_s_arr, lw=1, linestyle='--', color='red')
        plt.plot(self.x_k_arr * 1e3, self.T_wall_k_arr, lw=1, linestyle='--', color='red')
        plt.xlim(self.x_3s * 1e3, self.x_2k * 1e3)
        T_max = max(max(self.T_wall_k_arr), max(self.T_wall_s_arr))
        plt.text(0.4 * self.x_2k * 1e3, T_max - 30, r'$корыто$', fontsize=16)
        plt.text(0.6 * self.x_3s * 1e3, T_max - 30, r'$спинка$', fontsize=16)
        plt.xlabel(r'$x,\ мм$', fontsize=12)
        plt.ylabel(r'$T_{ст},\ К$', fontsize=12)
        plt.grid()
        plt.show()


class BladeCooler:
    def __init__(self):
        pass




