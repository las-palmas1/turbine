from ..profiling.section import BladeSection
import numpy as np
from gas_turbine_cycle.gases import Air, IdealGas
from gas_turbine_cycle.tools.gas_dynamics import GasDynamicFunctions as gd
import typing
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


class GasBladeHeatExchange:

    @classmethod
    def get_nusselt_coef(cls, angle1, angle2):
        return 0.07 + 100 * (np.degrees(angle1) + np.degrees(angle2)) ** (-2)

    @classmethod
    def get_regions_bound(cls, section: BladeSection):
        x_2k = section.length_k + section.length_in_edge_k
        x_1k = section.length_in_edge_k
        x_0 = 0
        x_1s = -section.length_in_edge_s
        x_2s = -section.length_in_edge_s - section.length_s + section.chord_length / 3
        x_3s = -section.length_in_edge_s - section.length_s
        return x_3s, x_2s, x_1s, x_0, x_1k, x_2k

    @classmethod
    def get_alpha_gas(cls, x, alpha_gas_inlet, alpha_gas_av, x_3s, x_2s, x_1s, x_0, x_1k, x_2k):
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

    @classmethod
    def get_alpha_gas_av(cls, section: BladeSection, height, T_gas_stag, G_gas, D_av, work_fluid: IdealGas):
        mu_gas = work_fluid.mu(T_gas_stag)
        lam_gas = work_fluid.lam(T_gas_stag)
        Re_gas = G_gas * section.chord_length / (np.pi * D_av * height * mu_gas * np.sin(section.angle2))
        nusselt_coef = cls.get_nusselt_coef(section.angle1, section.angle2)
        Nu_gas = nusselt_coef * Re_gas ** 0.68
        alpha_gas_av = Nu_gas * lam_gas / section.chord_length
        return mu_gas, lam_gas, Re_gas, Nu_gas, alpha_gas_av

    @classmethod
    def get_alpha_gas_inlet(cls, section: BladeSection, height, T_gas_stag, G_gas, D_av, work_fluid: IdealGas):
        mu_gas = work_fluid.mu(T_gas_stag)
        lam_gas = work_fluid.lam(T_gas_stag)
        alpha_gas_inlet = (0.74 * lam_gas / (2 * section.r1) *
                                (G_gas * 2 * section.r1 /
                                 (np.pi * D_av * height * np.sin(section.angle1) * mu_gas)) ** 0.5)
        return alpha_gas_inlet


class DeflectorAverageParamCalculator:
    """
    Класс для расчета средних параметров дефлекторной лопатки.
    """
    def __init__(self, section: BladeSection,
                 height=None,
                 D_av=None,
                 wall_thickness=None,
                 T_wall_out=None,
                 T_cool_fluid0=None,
                 T_out_stag=None,
                 alpha_out=None,
                 G_cool=None,
                 lam_blade: typing.Callable[[float], float]=None,
                 cool_fluid: IdealGas=Air()):
        """
        :param section: BladeSection. \n
            Сечение рассчитываемого участка лопатки.
        :param height: float. \n
            Высота рассчитываемого участка лопакти.
        :param D_av: float. \n
            Средний диаметр участка.
        :param wall_thickness: float. \n
            Толщина стенки.
        :param T_wall_out: float. \n
            Средняя температура наружной поверхности стенки участка лопатки.
        :param T_cool_fluid0: float. \n
            Температура охраждающего тела на входе в лопатку.
        :param T_out_stag: float. \n
            Средняя температура торможения обтекающей лопатку среды.
        :param alpha_out: float. \n
            Средний коэффициент теплоотдачи от обтекающей лопакту среды к лопатке.
        :param G_cool: float. \n
            Расход охлаждающего тела.
        :param lam_blade: callable. \n
            Тепопроводность материала лопатки в зависимости от температуры
        :param cool_fluid: IdealGas. \n
            Охлаждающее тело.
        """
        self.section = section
        self.height = height
        self.D_av = D_av
        self.wall_thickness = wall_thickness
        self.T_wall_out = T_wall_out
        self.T_cool_fluid0 = T_cool_fluid0
        self.T_out_stag = T_out_stag
        self.alpha_out = alpha_out
        self.G_cool = G_cool
        self.lam_blade = lam_blade
        self.cool_fluid = cool_fluid

        self.perimeter = None
        self.square = None
        self.Q_blade = None
        self.T_wall_av = None
        self.T_wall_in = None
        self.delta_T = None
        self.T_cool_fluid_av = None
        self.T_cool_fluid_old = None
        self.cool_fluid_temp_res = None
        self.c_p_cool_av = None
        self.mu_cool_fluid = None
        self.Re_cool_fluid = None
        self.alpha_cool_fluid_av = None
        self.lam_cool_fluid = None
        self.epsilon = None
        self.D = None
        self.channel_width = None

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

    def compute(self):
        self.perimeter = (self.section.length_in_edge + self.section.length_s + self.section.length_k +
                          self.section.length_out_edge)
        self.Q_blade = self.alpha_out * self.perimeter * self.height * (self.T_out_stag - self.T_wall_out)
        self.square = self.perimeter * self.height
        self.T_wall_av: float = fsolve(lambda x: [self.T_wall_out -
                                                  self.Q_blade * self.wall_thickness /
                                                  (2 * self.square * self.lam_blade(x[0])) - x[0]],
                                       np.array([self.T_wall_out]))[0]
        self.delta_T = self.Q_blade * self.wall_thickness / (self.square * self.lam_blade(self.T_wall_av))
        self.T_wall_in = self.T_wall_out - self.delta_T
        self.T_cool_fluid_av, self.T_cool_fluid_old, \
        self.cool_fluid_temp_res = self._get_av_cool_fluid_temp(self.Q_blade, self.G_cool, self.T_cool_fluid0)
        self.c_p_cool_av = self.cool_fluid.c_p_av_int
        self.mu_cool_fluid = self.cool_fluid.mu(self.T_cool_fluid_av)
        self.Re_cool_fluid = self.G_cool / (self.height * self.cool_fluid.mu(self.T_cool_fluid_av))
        self.alpha_cool_fluid_av = self.Q_blade / (self.square * (self.T_wall_in - self.T_cool_fluid_av))
        self.lam_cool_fluid = self.cool_fluid.lam(self.T_cool_fluid_av)
        self.epsilon = 0.01 * self.lam_cool_fluid * (1 / (self.height * self.mu_cool_fluid)) ** 0.8
        self.D = (1 / self.alpha_out * (self.T_out_stag - self.T_cool_fluid0) / (self.T_out_stag - self.T_wall_out) -
                  1 / self.alpha_out - self.wall_thickness / self.lam_blade(self.T_wall_av))
        self.channel_width = self.epsilon * self.G_cool ** 0.8 * (self.D - self.square /
                                                                  (2 * self.G_cool * self.c_p_cool_av))


class LocalParamCalculator:
    """Класс для расчет локальных параметров участка лопатки, в пределах которого постоянны по высоте
    параметры внешней среды и форма профиля."""
    def __init__(self, section: BladeSection,
                 height=None,
                 wall_thickness=None,
                 T_cool_fluid0=None,
                 T_wall_av=None,
                 channel_width=None,
                 alpha_cool: typing.Callable[[float, float], float]=None,
                 T_out_stag: typing.Callable[[float], float]=None,
                 alpha_out: typing.Callable[[float], float]=None,
                 G_cool: typing.Callable[[float], float]=None,
                 lam_blade: typing.Callable[[float], float]=None,
                 cool_fluid: IdealGas=Air(),
                 node_num: int = 500):
        """
        :param section: BladeSection. \n
            Сечение рассчитываемого участка лопатки.
        :param height: float. \n
            Высота рассчитываемого участка лопакти.
        :param wall_thickness: float. \n
            Толщина стенки.
        :param T_cool_fluid0: float. \n
            Температура охраждающего тела на входе в лопатку.
        :param T_wall_av: float. \n
            Средняя температура стенки лопатки.
        :param channel_width: float. \n
            Ширина канала.
        :param alpha_cool: callable. \n
            Зависимость коэффициента теплоотдачи от охлаждающего тела к стенке лопатки от координаты вдоль
            обвода профиля и температуры охлаждающего тела.
        :param T_out_stag: callable. \n
            Распределение температуры обтекающей лопатку среды вдоль обвода профиля.
        :param alpha_out: callable. \n
            Распределение коэффициента теплоотдачи от обтекающей лопатку среды к лопатке вдоль обвода профиля.
        :param G_cool: callable. \n
            Распределение расхода охлаждающего тела к лопатке вдоль обвода профиля.
        :param lam_blade: callable. \n
            Тепопроводность материала лопатки в зависимости от температуры.
        :param cool_fluid: IdealGas. \n
            Охлаждающее тело.
        :param node_num: int, optional. \n
            Число узлов на интервале решения уравнения теплового баланса.
        """
        self.section = section
        self.height = height
        self.wall_thickness = wall_thickness
        self.T_cool_fluid0 = T_cool_fluid0
        self.channel_width = channel_width
        self.alpha_cool = alpha_cool
        self.T_wall_av = T_wall_av
        self.T_out_stag = T_out_stag
        self.alpha_out = alpha_out
        self.G_cool = G_cool
        self.lam_blade = lam_blade
        self.cool_fluid = cool_fluid
        self.node_num = node_num

        self.T_cool_fluid_s_arr = None
        self.T_cool_fluid_k_arr = None
        self.T_cool_fluid_arr = None
        self.alpha_cool_fluid_arr = None
        self.alpha_cool_fluid_k_arr = None
        self.alpha_gas_arr = None
        self.alpha_gas_k_arr = None
        self.T_wall_arr = None
        self.T_wall_k_arr = None
        self.x_s_arr = None
        self.x_k_arr = None
        self.x_arr = None

    def _get_heat_transfer_coef(self, x, T_cool_fluid):
        return 1 / (1 / self.alpha_cool(x, T_cool_fluid) +
                    1 / self.alpha_out(x) +
                    self.wall_thickness / self.lam_blade(self.T_wall_av)).__float__()

    @classmethod
    def _solve_equation(cls, x_arr: np.ndarray, der: typing.Callable[[float, float], float], val0, eq_type='s'):
        val_arr = [val0]
        print()
        for i in range(1, len(x_arr)):
            if eq_type == 'k':
                val = val_arr[i - 1] + (x_arr[i] - x_arr[i - 1]) * der(val_arr[i - 1], x_arr[i - 1])
            elif eq_type == 's':
                val = val_arr[i - 1] - (x_arr[i - 1] - x_arr[i]) * der(val_arr[i - 1], x_arr[i - 1])
            print('i = %s' % i, 'T = %.3f' % val, 'x = %.5f' % x_arr[i], 'der_T = %.3f' % der(val_arr[i - 1],
                                                                                              x_arr[i - 1]))
            val_arr.append(val)
        return np.array(val_arr)

    def _get_T_cool_fluid_der(self, x, T_cool_fluid):
        heat_trans_coef = self._get_heat_transfer_coef(x, T_cool_fluid)
        if x >= 0:
            return (self.height * (self.T_out_stag(x) - T_cool_fluid) * heat_trans_coef /
                    (0.5 * self.G_cool(x) * self.cool_fluid.c_p_real_func(T_cool_fluid)))
        else:
            return -(self.height * (self.T_out_stag(x) - T_cool_fluid) * heat_trans_coef /
                     (0.5 * self.G_cool(x) * self.cool_fluid.c_p_real_func(T_cool_fluid)))

    def _get_T_wall(self, x, T_cool_fluid):
        heat_trans_coef = self._get_heat_transfer_coef(x, T_cool_fluid)
        return self.T_out_stag(x) - heat_trans_coef / self.alpha_out(x) * (self.T_out_stag(x) - T_cool_fluid)

    def compute(self):
        self.x_s_arr = np.array(np.linspace(0, -self.section.length_in_edge_s - self.section.length_s, self.node_num))
        self.x_k_arr = np.array(np.linspace(0, self.section.length_in_edge_k + self.section.length_k, self.node_num))
        self.T_cool_fluid_s_arr = self._solve_equation(self.x_s_arr,
                                                       lambda T, x:
                                                       self._get_T_cool_fluid_der(x, T),
                                                       self.T_cool_fluid0,
                                                       eq_type='s')
        self.T_cool_fluid_k_arr = self._solve_equation(self.x_k_arr,
                                                       lambda T, x:
                                                       self._get_T_cool_fluid_der(x, T),
                                                       self.T_cool_fluid0,
                                                       eq_type='k')
        x_s_list = list(self.x_s_arr[1: self.x_s_arr.shape[0]])
        x_s_list.reverse()
        self.x_arr = np.array(x_s_list + list(self.x_k_arr))

        T_cool_fluid_s_list = list(self.T_cool_fluid_s_arr[1: self.T_cool_fluid_s_arr.shape[0]])
        T_cool_fluid_s_list.reverse()
        self.T_cool_fluid_arr = np.array(T_cool_fluid_s_list + list(self.T_cool_fluid_k_arr))

        self.T_wall_arr = []

        for T_cool_fluid, x in zip(self.T_cool_fluid_arr, self.x_arr):
            T_wall = self._get_T_wall(x, T_cool_fluid)
            self.T_wall_arr.append(T_wall)

        self.alpha_cool_fluid_arr = []
        for T_cool_fluid, x in zip(self.T_cool_fluid_arr, self.x_arr):
            alpha_cool = self.alpha_cool(x, T_cool_fluid)
            self.alpha_cool_fluid_arr.append(alpha_cool)

        self.alpha_gas_arr = []
        for x in self.x_arr:
            alpha_gas = self.alpha_out(x)
            self.alpha_gas_arr.append(alpha_gas)

    def plot_T_wall(self, figsize=(8, 6)):
        plt.figure(figsize=figsize)
        plt.plot(self.x_arr * 1e3, self.T_wall_arr, lw=1, linestyle='--', color='red')
        plt.xlim(min(self.x_arr) * 1e3, max(self.x_arr) * 1e3)
        T_max = max(self.T_wall_arr)
        plt.text(0.6 * min(self.x_arr) * 1e3, T_max - 30, r'$корыто$', fontsize=16)
        plt.text(0.4 * max(self.x_arr) * 1e3, T_max - 30, r'$спинка$', fontsize=16)
        plt.xlabel(r'$x,\ мм$', fontsize=12)
        plt.ylabel(r'$T_{ст},\ К$', fontsize=12)
        plt.grid()
        plt.show()


class FilmCalculator:
    def __init__(self, x_hole: typing.List[float],
                 hole_num: typing.List[int],
                 d_hole: typing.List[float],
                 phi_hole: typing.List[float],
                 mu_hole: typing.List[float],
                 T_gas_stag,
                 p_gas_stag,
                 v_gas: typing.Callable[[float], float],
                 c_p_gas_av,
                 work_fluid: IdealGas,
                 alpha_gas: typing.Callable[[float], float],
                 T_cool: typing.Callable[[float], float],
                 G_cool: typing.Callable[[float], float],
                 p_cool_stag0,
                 c_p_cool_av,
                 cool_fluid: IdealGas,
                 channel_width,
                 height
                 ):
        """
        :param x_hole: Координаты рядов отверстий.
        :param hole_num:  Число отверстий в рядах.
        :param d_hole: Диаметры отверстий в рядах.
        :param phi_hole: Коэффициенты скорости в рядах отверстий.
        :param mu_hole: Коэффициенты расхода в рядах отверстий.
        :param T_gas_stag: Температура торможения газа.
        :param p_gas_stag: Давление тороможения газа.
        :param v_gas: Распределение скорости газа по профилю.
        :param c_p_gas_av: Средняя теплоемкость газа
        :param work_fluid: Рабочее тело турбины.
        :param alpha_gas: Распределение коэффициента теплоотдачи со стророны газа.
        :param T_cool: Распределение температуры торможения охлаждающего воздуха вдоль профиля.
        :param G_cool: Распределение расхода охлаждающего воздуха вдоль профиля.
        :param p_cool_stag0: Полное давление охлаждающего воздуха на входе в канал.
        :param c_p_cool_av: Средняя теплоемкость охлаждающего воздуха.
        :param cool_fluid: Охлаждающее тело.
        :param channel_width: Ширина канала охлаждения.
        :param height: Высота участка лопатки
        """
        self.x_hole = np.array(x_hole)
        self.hole_num = np.array(hole_num)
        self.d_hole = np.array(d_hole)
        self.phi_hole = phi_hole
        self.mu_hole = mu_hole
        self.T_gas_stag = T_gas_stag
        self.p_gas_stag = p_gas_stag
        self.v_gas = v_gas
        self.c_p_gas_av = c_p_gas_av
        self.work_fluid = work_fluid
        self.alpha_gas = alpha_gas
        self.T_cool = T_cool
        self.G_cool = G_cool
        self.p_cool_stag0 = p_cool_stag0
        self.G_cool0 = G_cool(0)
        self.c_p_cool_av = c_p_cool_av
        self.cool_fluid = cool_fluid
        self.channel_width = channel_width
        self.height = height
        self.k_gas_av = self.work_fluid.k_func(self.c_p_gas_av)
        self.k_cool_av = self.cool_fluid.k_func(self.c_p_cool_av)

        self.hole_step = np.zeros(self.x_hole.shape[0])
        self.s = np.zeros(self.x_hole.shape[0])
        self.T_cool_hole = np.zeros(self.x_hole.shape[0])
        "Температура торможения охлаждающей среды у входа в отверстия"
        self.v_gas_hole = np.zeros(self.x_hole.shape[0])
        "Скорость газа у отверстий"
        self.T_gas_hole = np.zeros(self.x_hole.shape[0])
        "Статическая температура газа у отверстий"
        self.p_gas_hole = np.zeros(self.x_hole.shape[0])
        "Статическое давление газа у отверстий"
        self.rho_gas_hole = np.zeros(self.x_hole.shape[0])
        "Статичекая плотность газа у отверстий"
        self.v_cool_hole_out = np.zeros(self.x_hole.shape[0])
        "Скорость истечения охлаждающей среды из отвестий"
        self.rho_cool_stag_hole = np.zeros(self.x_hole.shape[0])
        "Плотность охлаждающей среды по параметрам торможения на входе в отверстия"
        self.rho_cool_hole_out = np.zeros(self.x_hole.shape[0])
        "Статическая плотность охлаждающей среды на выходе из отверстия"
        self.m = np.zeros(self.x_hole.shape[0])
        "Параметры вдува на отверстиях"
        self.Re_s = np.zeros(self.x_hole.shape[0])
        "Число Рейнольдса по ширине щели на отверстиях"
        self.phi_temp = np.zeros(self.x_hole.shape[0])
        "Температурный фактор на отверстиях"
        self.G_cool_hole = np.zeros(self.x_hole.shape[0])
        "Расход охлаждающей среды в сечении канала перед отверстиями"
        self.dG_cool_hole = np.zeros(self.x_hole.shape[0])
        "Расход охлаждающей среды перед отверстиями"
        self.film_eff_list = []
        "Список функций эффективности пленок от каждого ряда отверстий"

    @classmethod
    def get_film_eff_func(cls, args):
        Re_s = args[0]
        m = args[1]
        phi = args[2]
        s = args[3]
        x_hole = args[4]

        def A(x):
            if x_hole >= 0:
                res = Re_s ** (-0.25) * m ** (-1.3) * phi ** (-1.25) * (x - x_hole) / s
            else:
                res = Re_s ** (-0.25) * m ** (-1.3) * phi ** (-1.25) * (x_hole - x) / s
            return res

        def res(x):
            if A(x) < 0.:
                return 0
            elif 0. <= A(x) < 3.:
                return 1.
            elif 3. <= A(x) < 11:
                return (A(x) / 3) ** (-0.285)
            else:
                return (A(x) / 7.43) ** (-0.95)

        return res

    def compute(self):
        self.hole_step = self.height / (self.hole_num + 1)
        self.s = self.hole_num * np.pi * self.d_hole**2 / 4 * 1 / self.height

        for i, x in enumerate(self.x_hole):
            self.v_gas_hole[i] = self.v_gas(x)
            self.T_gas_hole[i] = self.T_gas_stag - self.v_gas_hole[i] ** 2 / (2 * self.c_p_gas_av)

            self.p_gas_hole[i] = (self.p_gas_stag /
                                  (self.T_gas_stag / self.T_gas_hole[i]) ** (self.k_gas_av / (self.k_gas_av - 1)))

            self.rho_gas_hole[i] = self.p_gas_hole[i] / (self.work_fluid.R * self.T_gas_hole[i])
            self.T_cool_hole[i] = self.T_cool(x)

            self.v_cool_hole_out[i] = (self.phi_hole[i] *
                                       np.sqrt(
                                           2 * self.k_cool_av / (self.k_cool_av - 1) *
                                           self.cool_fluid.R * self.T_cool_hole[i] *
                                           (1 - (self.p_gas_hole[i] / self.p_cool_stag0) **
                                            ((self.k_cool_av - 1) / self.k_cool_av))
                                       ))
            self.rho_cool_stag_hole[i] = self.p_cool_stag0 / (self.cool_fluid.R * self.T_cool_hole[i])

            self.rho_cool_hole_out[i] = self.p_gas_hole[i] / (self.cool_fluid.R *
                                                              (self.T_cool_hole[i] -
                                                               self.v_cool_hole_out[i]**2 / (2 * self.c_p_cool_av)))
            self.G_cool_hole[i] = self.G_cool(x)

            self.m[i] = (self.rho_cool_hole_out[i] * self.v_cool_hole_out[i] /
                         (self.rho_gas_hole[i] * self.v_gas_hole[i]))

            self.Re_s[i] = (self.rho_gas_hole[i] * self.v_gas_hole[i] * self.s[i] /
                            self.work_fluid.mu(self.T_gas_hole[i]))

            self.phi_temp[i] = self.T_cool_hole[i] / self.T_gas_stag

            self.dG_cool_hole[i] = (self.hole_num[i] * np.pi * self.d_hole[i] ** 2 / 4 * self.mu_hole[i] *
                                    np.sqrt(
                                        2 * self.k_cool_av / (self.k_cool_av - 1) * self.p_cool_stag0 *
                                        self.rho_cool_stag_hole[i] *
                                        (self.p_gas_hole[i] / self.p_cool_stag0) ** (2 / self.k_cool_av) *
                                        (1 - (self.p_gas_hole[i] / self.p_cool_stag0) **
                                         ((self.k_cool_av - 1) / self.k_cool_av))
                                    ))

        self.film_eff_list = list(map(self.get_film_eff_func,
                                      zip(self.Re_s, self.m, self.phi_temp, self.s, self.x_hole)))

    def get_T_film(self, x):
        term1 = 1
        for film_eff in self.film_eff_list:
            term1 *= (1 - film_eff(x))

        term2 = 0
        for i in range(len(self.film_eff_list)):
            term3 = 1
            for j in range(i + 1, len(self.film_eff_list)):
                term3 *= (1 - self.film_eff_list[j](x))
            term2 += self.film_eff_list[i](x) * self.T_cool(x) * term3

        res = self.T_gas_stag * term1 + term2
        return res

    def get_alpha_item(self, i, x):
        if x != self.x_hole[i]:
            if x >= 0:
                return self.alpha_gas(x) * (1 + 2 * self.m[i] * self.s[i] / (x - self.x_hole[i]))
            else:
                return self.alpha_gas(x) * (1 + 2 * self.m[i] * self.s[i] / (self.x_hole[i] - x))
        else:
            return 0

    def get_alpha_film(self, x):
        x_hole_s = self.x_hole[self.x_hole < 0]
        x_hole_k = self.x_hole[self.x_hole >= 0]

        if x <= x_hole_s[0]:
            return self.get_alpha_item(0, x)
        elif x >= self.x_hole[len(self.x_hole) - 1]:
            return self.get_alpha_item(len(self.x_hole) - 1, x)
        elif x_hole_s[len(x_hole_s) - 1] < x < x_hole_k[0]:
            return self.alpha_gas(x)

        else:
            for i in range(0, len(x_hole_s) - 1):
                if x_hole_s[i] < x <= x_hole_s[i + 1]:
                    return self.get_alpha_item(i + 1, x)

            for i in range(0, len(x_hole_k) - 1):
                if x_hole_k[i] <= x_hole_k[i + 1]:
                    return self.get_alpha_item(i + len(x_hole_s), x)

    def get_G_cool(self, x):
        x_hole_s = self.x_hole[self.x_hole < 0]
        x_hole_k = self.x_hole[self.x_hole >= 0]

        dG_hole_s = self.dG_cool_hole[self.x_hole < 0]
        dG_hole_k = self.dG_cool_hole[self.x_hole >= 0]

        if x <= x_hole_s[0]:
            return self.G_cool0 - dG_hole_s.sum()
        elif x >= self.x_hole[len(self.x_hole) - 1]:
            return self.G_cool0 - dG_hole_k.sum()
        elif x_hole_s[len(x_hole_s) - 1] < x < x_hole_k[0]:
            return self.G_cool0

        else:
            x_hole_s = list(x_hole_s)
            x_hole_s.reverse()
            x_hole_s = -1 * np.array(x_hole_s)

            if x >= 0:
                dG = 0
                for i in range(len(x_hole_k) - 1):
                    dG += dG_hole_k[i]
                    if x_hole_k[i] < x <= x_hole_k[i + 1]:
                        return self.G_cool0 - dG
            else:
                dG = 0
                for i in range(len(x_hole_s) - 1):
                    dG += dG_hole_s[i]
                    if x_hole_s[i] < -x <= x_hole_s[i + 1]:
                        return self.G_cool0 - dG

    @classmethod
    def _get_lim(cls, value_arr, places, delta):
        if round(max(value_arr), places) > max(value_arr):
            max_value = round(max(value_arr), places)
        else:
            max_value = round(max(value_arr), places) + delta

        if round(min(value_arr)) < min(value_arr):
            min_value = round(min(value_arr), places)
        else:
            min_value = round(min(value_arr), places) - delta
        return min_value, max_value

    def plot_T_film(self, x_arr, places=-2, delta=50, figsize=(7, 5)):
        plt.figure(figsize=figsize)
        T_arr = [self.get_T_film(x) for x in x_arr]

        T_min, T_max = self._get_lim(T_arr, places, delta)
        plt.plot(x_arr, T_arr, lw=1.5)
        for x in self.x_hole:
            plt.plot([x, x], [T_min, T_max], color='black', lw=0.8, linestyle='--')
        plt.grid()
        plt.ylim(T_min, T_max)
        plt.xlabel(r'$x,\ м$', fontsize=12)
        plt.ylabel(r'$T_{пл}^*,\ К$', fontsize=12)
        plt.show()

    def plot_film_eff(self, hole_num, x_arr, create_fig=False, show=False, figsize=(9, 7)):
        if create_fig:
            plt.figure(figsize=figsize)
        plt.plot(x_arr, [self.film_eff_list[hole_num](x) for x in x_arr], lw=1, label='Отверстие %s' % (hole_num + 1))
        plt.legend(fontsize=8)
        plt.xlabel(r'$x,\ м$', fontsize=12)
        plt.ylabel(r'$\theta_{пл}$', fontsize=12)
        plt.grid(True, axis='both')
        if show:
            plt.show()

    def plot_alpha_film(self, x_arr, ylim, figsize=(7, 5)):
        plt.figure(figsize=figsize)
        alpha_arr = [self.get_alpha_film(x) for x in x_arr]
        plt.plot(x_arr, alpha_arr, lw=1.5)
        alpha_min, alpha_max = ylim
        for x in self.x_hole:
            plt.plot([x, x], [alpha_min, alpha_max], color='black', lw=0.8, linestyle='--')
        plt.grid()
        plt.ylim(alpha_min, alpha_max)
        plt.xlabel(r'$x,\ м$', fontsize=12)
        plt.ylabel(r'$\alpha_{пл},\ \frac{Вт}{м^2 \cdot К}$', fontsize=12)
        plt.show()

    def plot_G_cool(self, x_arr, places=4, delta=1e-4, figsize=(7, 5)):
        plt.figure(figsize=figsize)
        G_arr = [self.get_G_cool(x) for x in x_arr]
        plt.plot(x_arr, G_arr, lw=1.5)

        G_min, G_max = self._get_lim(G_arr, places, delta)
        for x in self.x_hole:
            plt.plot([x, x], [G_min, G_max], color='black', lw=0.8, linestyle='--')
        plt.grid()
        plt.xlabel(r'$x,\ м$', fontsize=12)
        plt.ylabel(r'$G_в,\ кг/с$', fontsize=12)
        plt.show()
