from .tools import GasBladeHeatExchange, LocalParamCalculator, FilmCalculator, DeflectorAverageParamCalculator
from ..profiling.section import BladeSection
from ..profiling.stage import StageProfiler, ProfilingResultsForCooling
from gas_turbine_cycle.gases import IdealGas, Air
import numpy as np
import pandas as pd
import pickle
import typing
from gas_turbine_cycle.tools.gas_dynamics import GasDynamicFunctions as gd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.integrate import quad
import xlwt

log_level = 'INFO'


class Logger:
    def __init__(self, level_name: str= 'INFO'):
        self.level_name = level_name

    @classmethod
    def get_level(cls, level_name):
        if level_name == 'INFO':
            return 2
        elif level_name == 'DEBUG':
            return 1
        elif level_name == 'ERROR':
            return 0

    def send_msg(self, msg: str, level_name: str):
        print('%s - %s' % (level_name, msg))

    def info(self, msg: str):
        self.send_msg(msg, 'INFO')

    def debug(self, msg: str):
        if self.get_level(self.level_name) <= self.get_level('DEBUG'):
            self.send_msg(msg, 'DEBUG')

    def error(self, msg: str):
        if self.get_level(self.level_name) <= self.get_level('ERROR'):
            self.send_msg(msg, 'ERROR')


class FilmSectorCooler(GasBladeHeatExchange):
    def __init__(self,
                 section: BladeSection,
                 height=None,
                 channel_width=None,
                 wall_thickness=None,
                 D_av=None,
                 lam_blade: typing.Callable[[float], float]=None,
                 T_wall_av=None,
                 x_hole_rel: typing.List[float]=None,
                 hole_num: typing.List[float]=None,
                 d_hole: typing.List[float]=None,
                 phi_hole: typing.List[float]=None,
                 mu_hole: typing.List[float]=None,
                 T_gas_stag=None,
                 p_gas_stag=None,
                 G_gas=None,
                 c_p_gas_av=None,
                 lam_gas_in=None,
                 lam_gas_out=None,
                 work_fluid: IdealGas=None,
                 T_cool0=None,
                 p_cool_stag0=None,
                 G_cool0=None,
                 g_cool0_s=0.5,
                 c_p_cool_av=None,
                 cover_thickness=0.15e-3,
                 lam_cover=2,
                 cool_fluid: IdealGas=Air(),
                 node_num: int=500,
                 accuracy: float=0.01
                 ):
        """
        :param section: Сечение сектора лопатки.
        :param height: Высота сектора лопатки.
        :param channel_width: Ширина канала для охлаждения.
        :param wall_thickness: Толщина стенки.
        :param D_av: Средний диаметр сектора.
        :param lam_blade: Зависимость теплопроводности материала лопатки от температуры.
        :param T_wall_av: Средняя температура стенки лопатки.
        :param x_hole_rel: Относительный координаты рядов отверстий.
        :param hole_num: Количества отверстий в рядах.
        :param d_hole: Диаметры отверстий в рядах.
        :param phi_hole: Коэффициенты скорости для истечения из отверстий в рядах.
        :param mu_hole: Коэффициенты расхода для истечения из отверстий в рядах.
        :param T_gas_stag: Температура торможения газа.
        :param p_gas_stag: Давление торможения газа.
        :param G_gas: Расход газа.
        :param c_p_gas_av: Средняя теплоемкость газа.
        :param lam_gas_in: Приведенная скорость газа на входе.
        :param lam_gas_out: Приведенная скосроть газа на выходе.
        :param work_fluid: Рабочее тело турбины.
        :param T_cool0: Температура охлаждающей среды на входе в канал.
        :param p_cool_stag0: Полное давление охлаждающей среды на входе в канал.
        :param G_cool0: Расход охлаждающей среды на входе в канал.
        :param g_cool0_s: Относительный расход охлаждающего воздуха на входе в канал спинки.
        :param c_p_cool_av: Средняя теплоемкость охлаждающей среды.
        :param cool_fluid: Охлаждающая среда.
        :param cover_thickness: Толщина защитного покрытия.
        :param lam_cover: Теплопроводность защитного покрытия.
        :param node_num: Число узлов для решения уравнения теплового баланса.
        :param accuracy: Точность сходимости.
        """
        self.section = section
        self.height = height
        self.channel_width = channel_width
        self.wall_thickness = wall_thickness
        self.D_av = D_av
        self.lam_blade = lam_blade
        self.T_wall_av = T_wall_av
        self.x_hole_rel = x_hole_rel
        self.hole_num = hole_num
        self.d_hole = d_hole
        self.phi_hole = phi_hole
        self.mu_hole = mu_hole
        self.T_gas_stag = T_gas_stag
        self.p_gas_stag = p_gas_stag
        self.G_gas = G_gas
        self.c_p_gas_av = c_p_gas_av
        self.lam_gas_in = lam_gas_in
        self.lam_gas_out = lam_gas_out
        self.work_fluid = work_fluid
        self.T_cool0 = T_cool0
        self.p_cool_stag0 = p_cool_stag0
        self.G_cool0 = G_cool0
        self.g_cool0_s = g_cool0_s
        self.c_p_cool_av = c_p_cool_av
        self.cool_fluid = cool_fluid
        self.node_num = node_num
        self.accuracy = accuracy
        self.lam_cover = lam_cover
        self.cover_thickness = cover_thickness

        self.k_gas_av = None
        self.k_cool_av = None
        self.x_hole = None
        self.local_param: LocalParamCalculator = None
        self.film: FilmCalculator = None

        self.mu_gas = None
        self.lam_gas = None
        self.Re_gas = None
        self.Nu_gas = None
        self.alpha_gas_av = None
        self.alpha_gas_inlet = None
        self.res = None
        self._T_wall_list = []
        "Список для хранения температур стенки на разных итерациях."
        self.iter_num = None
        self.alpha_film_av = None
        self.T_film_av = None
        self.G_cool_av = None
        self.T_wall_out_av = None
        self.T_cool_av = None
        self.logger = Logger(level_name=log_level)

    def set_calculators(self):
        self.k_gas_av = self.work_fluid.k_func(self.c_p_gas_av)
        self.k_cool_av = self.work_fluid.k_func(self.c_p_cool_av)
        self.x_hole = self.get_x_hole(self.x_hole_rel, self.section)
        self.local_param = LocalParamCalculator(section=self.section,
                                                height=self.height,
                                                wall_thickness=self.wall_thickness,
                                                T_cool_fluid0=self.T_cool0,
                                                T_wall_av=self.T_wall_av,
                                                lam_blade=self.lam_blade,
                                                cool_fluid=self.cool_fluid,
                                                node_num=self.node_num,
                                                lam_cover=self.lam_cover,
                                                cover_thickness=self.cover_thickness)
        self.film = FilmCalculator(x_hole=self.x_hole,
                                   hole_num=self.hole_num,
                                   d_hole=self.d_hole,
                                   phi_hole=self.phi_hole,
                                   mu_hole=self.mu_hole,
                                   T_gas_stag=self.T_gas_stag,
                                   p_gas_stag=self.p_gas_stag,
                                   c_p_gas_av=self.c_p_gas_av,
                                   work_fluid=self.work_fluid,
                                   G_cool0=self.G_cool0,
                                   p_cool_stag0=self.p_cool_stag0,
                                   c_p_cool_av=self.c_p_cool_av,
                                   cool_fluid=self.cool_fluid,
                                   height=self.height,
                                   g_cool0_s=self.g_cool0_s)

        self.mu_gas, self.lam_gas, self.Re_gas, self.Nu_gas, self.alpha_gas_av = self.get_alpha_gas_av(self.section,
                                                                                                       self.height,
                                                                                                       self.T_gas_stag,
                                                                                                       self.G_gas,
                                                                                                       self.D_av,
                                                                                                       self.work_fluid)
        self.alpha_gas_inlet = self.get_alpha_gas_inlet(self.section, self.height, self.T_gas_stag, self.G_gas,
                                                        self.D_av, self.work_fluid)

    @classmethod
    def get_x_hole(cls, x_hole_rel, section: BladeSection):
        l_s = section.length_s + section.length_in_edge_s
        l_k = section.length_k + section.length_in_edge_k

        x_hole = []
        for x_rel in x_hole_rel:
            if x_rel < 0:
                x_hole.append(x_rel * l_s)
            else:
                x_hole.append(x_rel * l_k)
        return x_hole

    def get_alpha_cool_fluid(self, G_cool, T_cool_fluid):
        assert G_cool > 0, 'G_cool must not be negative.'
        return (0.02 * self.cool_fluid.lam(T_cool_fluid) / (2 * self.channel_width) *
                (G_cool / (self.height * self.cool_fluid.mu(T_cool_fluid))) ** 0.8)

    def _get_residual(self):

        if len(self._T_wall_list) > 1:
            T_wall_arr_curr = self._T_wall_list[len(self._T_wall_list) - 1]
            T_wall_arr_prev = self._T_wall_list[len(self._T_wall_list) - 2]
            res = np.linalg.norm(T_wall_arr_curr - T_wall_arr_prev)
            return res
        else:
            return 1

    def make_iteration(self,
                       alpha_cool: typing.Callable[[float, float], float],
                       T_out_stag: typing.Callable[[float], float],
                       alpha_out: typing.Callable[[float], float],
                       G_cool: typing.Callable[[float], float]
                       ):

        self.local_param.alpha_cool = alpha_cool
        self.local_param.T_out_stag = T_out_stag
        self.local_param.alpha_out = alpha_out
        self.local_param.G_cool = G_cool
        self.local_param.compute()
        self._T_wall_list.append(np.array(self.local_param.T_wall_arr))

        self.film.T_cool = self.local_param.get_T_cool
        self.film.compute()

        return (lambda x, T: self.get_alpha_cool_fluid(self.film.get_G_cool(x), T),
                self.film.get_T_film,
                self.film.get_alpha_film,
                self.film.get_G_cool)

    def _get_lam_gas_k(self, x_rel):
        return (1 + ((self.lam_gas_out / self.lam_gas_in) ** 0.5 - 1) * x_rel) ** 2 * self.lam_gas_in

    def _get_lam_gas_s(self, x_rel):
        return (1 + ((self.lam_gas_out / self.lam_gas_in) ** 4 - 1) * x_rel) ** 0.25 * self.lam_gas_in

    def get_v_gas(self, x):
        l_k = self.section.length_k + self.section.length_in_edge_k
        l_s = self.section.length_s + self.section.length_in_edge_s
        a_cr = np.sqrt(2 * self.k_gas_av / (self.k_gas_av + 1) * self.work_fluid.R * self.T_gas_stag)

        if x < 0:
            v = a_cr * self._get_lam_gas_s(-x / l_s)
        else:
            v = a_cr * self._get_lam_gas_k(x / l_k)

        return v

    def _get_alpha_gas(self, x):
        return self.get_alpha_gas(x, self.alpha_gas_inlet, self.alpha_gas_av,
                                  *self.get_regions_bound(self.section))

    @classmethod
    def _get_average_value(cls, func: typing.Callable[[float], float], x_arr):
        square = 0

        for i in range(len(x_arr) - 1):
            value = 0.5 * (func(x_arr[i]) + func(x_arr[i + 1]))
            square += value * (x_arr[i + 1] - x_arr[i])

        res = square / (x_arr[len(x_arr) - 1] - x_arr[0])

        return res

    def compute(self):
        self.film.alpha_gas = self._get_alpha_gas
        self.film.v_gas = self.get_v_gas

        self.res = 1.
        self.iter_num = 0

        alpha_cool = None
        T_out_stag = None
        alpha_out = None
        G_cool = None

        def G_cool_init(x):
            if x < 0:
                return self.G_cool0 * self.g_cool0_s
            elif x > 0:
                return self.G_cool0 * (1 - self.g_cool0_s)
            elif x == 0:
                return self.G_cool0

        while self.res >= self.accuracy:
            if not alpha_cool and not T_out_stag and not alpha_out and not G_cool:
                self.logger.debug(self.get_regions_bound(self.section))
                self.logger.debug('l_ie = ' + str(self.section.length_in_edge))
                self.logger.debug('l_ie_s = ' + str(self.section.length_in_edge_s))
                self.logger.debug('l_k = ' + str(self.section.length_k))
                self.logger.debug('y0 = ' + str(self.section.y0))
                self.logger.debug('y01 = ' + str(self.section.y01))
                self.logger.debug('x0 = ' + str(self.section.x0))
                self.logger.debug('x01 = ' + str(self.section.x01))
                (alpha_cool, T_out_stag,
                 alpha_out, G_cool) = self.make_iteration(

                    alpha_cool=lambda x, T: self.get_alpha_cool_fluid(self.G_cool0, T),
                    T_out_stag=lambda x: self.T_gas_stag,
                    alpha_out=self._get_alpha_gas,
                    G_cool=G_cool_init
                )
            else:
                (alpha_cool, T_out_stag,
                 alpha_out, G_cool) = self.make_iteration(

                    alpha_cool=alpha_cool,
                    T_out_stag=T_out_stag,
                    alpha_out=alpha_out,
                    G_cool=G_cool
                )
            self.iter_num += 1
            self.res = self._get_residual()
            self.logger.info('Sector computing: Iter №%s, residual = %.4f' % (self.iter_num, self.res))

        self.alpha_film_av = self._get_average_value(self.film.get_alpha_film, self.local_param.x_arr)
        self.T_film_av = self._get_average_value(self.film.get_T_film, self.local_param.x_arr)
        self.G_cool_av = self._get_average_value(self.film.get_G_cool, self.local_param.x_arr)
        self.T_wall_out_av = self._get_average_value(self.local_param.get_T_wall, self.local_param.x_arr)
        self.T_cool_av = self._get_average_value(self.local_param.get_T_cool, self.local_param.x_arr)
        self.logger.debug('T_cool_av = %.2f' % self.T_cool_av)

    def get_cool_eff(self, x):
        return (self.T_gas_stag - self.local_param.get_T_wall(x)) / (self.T_gas_stag - self.local_param.get_T_cool(x))

    def plot_cool_eff(self, figsize=(7, 5), filename=None, label=True):
        plt.figure(figsize=figsize)
        plt.plot(self.local_param.x_arr * 1e3,
                 [self.get_cool_eff(x) for x in self.local_param.x_arr], lw=2, color='red')

        if label:
            plt.xlabel(r'$x,\ мм$', fontsize=14)
            plt.ylabel(r'$\theta_{охл}$', fontsize=14)
        plt.xlim(min(self.local_param.x_arr) * 1e3, max(self.local_param.x_arr) * 1e3)
        plt.grid(linewidth=1)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if filename:
            plt.savefig(filename)
        plt.show()

    def plot_alpha_gas(self, figsize=(7, 5), filename=None, label=True):
        plt.figure(figsize=figsize)
        plt.plot(self.local_param.x_arr * 1e3,
                 [self._get_alpha_gas(x) for x in self.local_param.x_arr], lw=2, color='red')

        if label:
            plt.xlabel(r'$x,\ мм$', fontsize=14)
            plt.ylabel(r'$\alpha_г,\ \frac{Вт}{м^2 \cdot К}$', fontsize=14)
        plt.xlim(min(self.local_param.x_arr) * 1e3, max(self.local_param.x_arr) * 1e3)
        plt.grid(linewidth=1)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.ylim(ymin=0)
        if filename:
            plt.savefig(filename)
        plt.show()

    def plot_v_gas(self, figsize=(6, 4), filename=None, label=True):
        plt.figure(figsize=figsize)

        x_arr = self.local_param.x_arr
        v_arr = [self.get_v_gas(x) for x in x_arr]

        plt.plot(x_arr * 1e3, v_arr, lw=2, color='red')

        if label:
            plt.xlabel(r'$x,\ мм$', fontsize=14)
            plt.ylabel(r'$v_г,\ м/с$', fontsize=14)
        plt.xlim(min(x_arr) * 1e3, max(x_arr) * 1e3)
        plt.grid(linewidth=1)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if filename:
            plt.savefig(filename)
        plt.show()

    def get_film_params(self):

        params = pd.DataFrame.from_dict(
            {
                'Name': [
                    r'$x,\ мм$',
                    r'$s,\ 10^{-3}\ мм$',
                    r'$\phi_{отв}$',
                    r'$\mu_{отв}$',
                    r'$m$',
                    r'$\phi$',
                    r'$G_{отв},\ г/с$',
                    r'$G_{отв}/G_{в0}$',
                ],
                'Value': [
                    [round(x * 1e3, 1) for x in self.film.x_hole],
                    [round(s * 1e6, 1) for s in self.film.s],
                    [round(phi, 3) for phi in self.film.phi_hole],
                    [round(mu, 3) for mu in self.film.mu_hole],
                    [round(m, 3) for m in self.film.m],
                    [round(phi, 3) for phi in self.film.phi_temp],
                    [round(G * 1e3, 3) for G in self.film.dG_cool_hole],
                    [round(G / self.film.G_cool0, 3) for G in self.film.dG_cool_hole],
                ]
            }
        )

        return params

    def get_local_params(self, pnt_num=40):
        x_arr = np.array(np.linspace(self.local_param.x_arr.min(), self.local_param.x_arr.max(), pnt_num))

        params = pd.DataFrame.from_dict(
            {
                'Name': [
                    r'$x,\ мм$',
                    r'$\alpha_{пл},\ \frac{Вт}{м^2 \cdot К}$',
                    r'$\alpha_{в},\ \frac{Вт}{м^2 \cdot К}$',
                    r'$\alpha_{г},\ \frac{Вт}{м^2 \cdot К}$',
                    r'$T_{в}^*,\ К$',
                    r'$T_{пл}^*,\ К$',
                    r'$T_{ст},\ К$',
                    r'$\theta_{пл}$',
                    r'$\theta_{охл}$'
                ],
                'Value': [
                    [round(x * 1e3, 1) for x in x_arr],
                    [round(self.film.get_alpha_film(x), 1) for x in x_arr],
                    [round(self.get_alpha_cool_fluid(self.film.get_G_cool(x),
                                                     self.local_param.get_T_cool(x)), 1) for x in x_arr],
                    [round(self._get_alpha_gas(x), 1) for x in x_arr],
                    [round(self.local_param.get_T_cool(x)) for x in x_arr],
                    [round(self.film.get_T_film(x)) for x in x_arr],
                    [round(self.local_param.get_T_wall(x)) for x in x_arr],
                    [round(self.film.get_integrate_film_eff(x), 3) for x in x_arr],
                    [round(self.get_cool_eff(x), 3) for x in x_arr],
                ]
            }
        )
        return params


class FilmBladeCoolingResults:
    def __init__(self, p_cool_stag0, T_cool0, G_cool0, channel_width, wall_thickness, cool_eff_av,
                 lam_cover, cover_thickness,
                 film_params: typing.List[pd.DataFrame],
                 local_params: typing.List[pd.DataFrame]):
        self.p_cool_stag0 = p_cool_stag0
        self.T_cool0 = T_cool0
        self.G_cool0 = G_cool0
        self.channel_width = channel_width
        self.wall_thickness = wall_thickness
        self.cool_eff_av = cool_eff_av
        self.lam_cover = lam_cover
        self.cover_thickness = cover_thickness
        self.film_params = film_params
        self.local_params = local_params


class FilmBladeCooler(GasBladeHeatExchange):
    """
    NOTE: Расчет лопатки происходит итерационно. Сначала производится расчет по среднем параметрам с
    начальным приближением для средней температуры обтекающей среды, среднеого коэффициента теплоотдачи
    обтекающей среды, среднего расхода воздуха через канал и средней температуры наружней поверхности лопатки. Далее
    производится расчет пленки и локальных параметров. В результате уточняются средняя температура
    наружней поверхности лопатки, расход воздуха и средние параметры обтекающей среды. После этого расчет повторяется
    до схождения по средней температуры по толщине стенки лопатки.
    """
    def __init__(self,
                 sections: typing.List[BladeSection],
                 channel_width,
                 wall_thickness,
                 D_in,
                 D_out,
                 T_wall_out_av_init,
                 lam_blade: typing.Callable[[float], float],
                 x_hole_rel: typing.List[float],
                 hole_num: typing.List[int],
                 d_hole: typing.List[float],
                 phi_hole: typing.List[float],
                 mu_hole: typing.List[float],
                 T_gas_stag: typing.Callable[[float], float],
                 p_gas_stag: typing.Callable[[float], float],
                 c_p_gas_av,
                 lam_gas_in: typing.Callable[[float], float],
                 lam_gas_out: typing.Callable[[float], float],
                 work_fluid: IdealGas,
                 T_cool0,
                 p_cool_stag0,
                 g_cool0: typing.Callable[[float], float],
                 cool_fluid: IdealGas,
                 g_cool0_s=0.5,
                 cover_thickness=0.015e-3,
                 lam_cover=2,
                 node_num: int = 500,
                 accuracy: float = 0.01
                 ):
        """
        :param sections: Массив сечений.
        :param channel_width: Ширина канала.
        :param wall_thickness: Толщина стенки лопатки.
        :param D_in: Внутренний диаметр.
        :param D_out: Наружний диаметр.
        :param T_wall_out_av_init: Начальное приближение для средней наружней температуры стенки.
        :param lam_blade: Зависимость теплопроводности лопатки от температуры.
        :param x_hole_rel: Относительный координаты рядов отверстий.
        :param hole_num: Количества отверстий в рядах.
        :param d_hole: Диаметры отверстий в рядах.
        :param phi_hole: Коэффициенты скорости для истечения из отверстий в рядах.
        :param mu_hole: Коэффициенты расхода для истечения из отверстий в рядах.
        :param T_gas_stag: Распределение по радиусу температуры торможения газа.
        :param p_gas_stag: Распределение по радиусу давление торможения газа.
        :param c_p_gas_av: Теплоемкость газа.
        :param lam_gas_in: Распределение по радиусу приведенной скорости газа на входе.
        :param lam_gas_out: Распределение по радиусу приведенной скорости газа на выходе.
        :param work_fluid: Рабочее тело турбины.
        :param T_cool0: Температура охлаждающего воздуха на входе в канала.
        :param p_cool_stag0: Давление охлажадющего воздуха на входе в канал.
        :param g_cool0: Распределение плотности расхода охлаждающего воздуха на входе в канал.
        :param cool_fluid: Охлаждающее тело.
        :param g_cool0_s: Относительный расход охлаждающего воздуха на входе в канал спинки.
        :param cover_thickness: Толщина защитного покрытия.
        :param lam_cover: Теплопроводность защитного покрытия.
        :param node_num: Число узлов для решения дифура.
        :param accuracy:
        """
        self.sections = sections
        self.sector_num = len(sections)
        assert self.sector_num % 2 == 1, 'Number of sections must be odd.'
        self.av_section = sections[int(self.sector_num / 2)]
        self.channel_width = channel_width
        self.wall_thickness = wall_thickness
        self.D_sec_arr = np.linspace(D_in, D_out, self.sector_num)
        self.D_in = D_in
        self.D_out = D_out
        self.T_wall_out_av_init = T_wall_out_av_init
        self.lam_blade = lam_blade
        self.x_hole_rel = x_hole_rel
        self.hole_num = hole_num
        self.d_hole = d_hole
        self.phi_hole = phi_hole
        self.mu_hole = mu_hole
        self.T_gas_stag = T_gas_stag
        self.p_gas_stag = p_gas_stag
        self.c_p_gas_av = c_p_gas_av
        self.k_gas = work_fluid.k_func(c_p_gas_av)
        self.lam_gas_in = lam_gas_in
        self.lam_gas_out = lam_gas_out
        self.work_fluid = work_fluid
        self.T_cool0 = T_cool0
        self.p_cool_stag0 = p_cool_stag0
        self.g_cool0 = g_cool0
        self.cool_fluid = cool_fluid
        self.g_cool0_s = g_cool0_s
        self.cover_thickness = cover_thickness
        self.lam_cover = lam_cover
        self.node_num = node_num
        self.accuracy = accuracy
        self.logger = Logger(level_name=log_level)

        self.hole_step = 0.5 * (self.D_out - self.D_in) / np.array(self.hole_num)

        self.av_param = DeflectorAverageParamCalculator(section=self.av_section,
                                                        height=0.5 * (D_out - D_in),
                                                        D_av=0.5 * (D_in + D_out),
                                                        wall_thickness=wall_thickness,
                                                        T_cool_fluid0=T_cool0,
                                                        T_out_stag=T_gas_stag(0.25 * (D_in + D_out)),
                                                        cool_fluid=type(cool_fluid)(),
                                                        lam_blade=lam_blade,
                                                        cover_thickness=cover_thickness,
                                                        lam_cover=lam_cover)

        self.G_cool0 = quad(g_cool0, 0.5 * D_in, 0.5 * D_out)[0]
        self.G_gas = quad(self._get_g_gas, 0.5 * D_in, 0.5 * D_out)[0]

        self.D_av_arr = None
        self.D_bound_arr = None
        self.sector_height_arr = None
        self.g_gas_arr = None
        self.g_cool0_arr = None
        self.T_gas_stag_arr = None
        self.p_gas_stag_arr = None
        self.lam_gas_in_arr = None
        self.lam_gas_out_arr = None
        self.G_gas_arr = None
        self.G_cool0_arr = None
        self.hole_num_arr = None
        self.sectors: typing.List[FilmSectorCooler] = None

        self.make_partition()

        self._T_wall_av_arr = []
        "Список, в котором записывается история вычислений средней температуры стенки"
        self.res = None
        self.iter_num = None
        self.alpha_film_av = None
        self.T_film_av = None
        self.G_cool_av = None
        self.T_cool_av = None
        self.T_wall_out_av = None
        "Фактическая средняя температура внешеней поверхности стенки"

        (self.mu_gas, self.lam_gas, self.Re_gas,
         self.Nu_gas, self.alpha_gas_av) = self.get_alpha_gas_av(self.av_section,
                                                                 0.5 * (D_out - D_in),
                                                                 T_gas_stag(0.25 * (D_in + D_out)),
                                                                 self.G_gas,
                                                                 0.5 * (D_in + D_out),
                                                                 type(work_fluid)())

    def make_partition(self):
        self.D_av_arr, self.D_bound_arr, self.sector_height_arr = self._get_D_av_arr_and_height_arr()

        self.g_gas_arr = self._get_partition(self._get_g_gas)
        self.g_cool0_arr = self._get_partition(self.g_cool0)
        self.T_gas_stag_arr = self._get_partition(self.T_gas_stag)
        self.p_gas_stag_arr = self._get_partition(self.p_gas_stag)
        self.lam_gas_in_arr = self._get_partition(self.lam_gas_in)
        self.lam_gas_out_arr = self._get_partition(self.lam_gas_out)

        self.G_gas_arr = self._get_G_arr(self.g_gas_arr)
        self.G_cool0_arr = self._get_G_arr(self.g_cool0_arr)

        self.hole_num_arr = self._get_hole_num_arr()

        self.sectors = [FilmSectorCooler(section=self.sections[i],
                                         height=self.sector_height_arr[i],
                                         channel_width=self.channel_width,
                                         wall_thickness=self.wall_thickness,
                                         D_av=self.D_av_arr[i],
                                         lam_blade=self.lam_blade,
                                         x_hole_rel=self.x_hole_rel,
                                         hole_num=self.hole_num_arr[i],
                                         d_hole=self.d_hole,
                                         phi_hole=self.phi_hole,
                                         mu_hole=self.mu_hole,
                                         T_gas_stag=self.T_gas_stag_arr[i],
                                         p_gas_stag=self.p_gas_stag_arr[i],
                                         G_gas=self.G_gas_arr[i],
                                         c_p_gas_av=self.c_p_gas_av,
                                         lam_gas_in=self.lam_gas_in_arr[i],
                                         lam_gas_out=self.lam_gas_out_arr[i],
                                         work_fluid=type(self.work_fluid)(),
                                         T_cool0=self.T_cool0,
                                         p_cool_stag0=self.p_cool_stag0,
                                         G_cool0=self.G_cool0_arr[i],
                                         cool_fluid=type(self.cool_fluid)(),
                                         node_num=self.node_num,
                                         accuracy=self.accuracy,
                                         g_cool0_s=self.g_cool0_s,
                                         lam_cover=self.lam_cover,
                                         cover_thickness=self.cover_thickness
                                         ) for i in range(self.sector_num)]

    def _get_D_av_arr_and_height_arr(self):
        """Массив средних диаметров, массив диаметров расположения границ между секторами и высот секторов."""
        D_arr = []
        h_arr = []
        D_bound_arr = [self.D_in]

        for i in range(self.sector_num):
            if i == 0:
                h = 0.5 * (self.D_sec_arr[i + 1] - self.D_sec_arr[i]) / 2
                D_av = self.D_sec_arr[i] + h
                D_bound_arr.append(self.D_sec_arr[i] + 2 * h)
            elif i == self.sector_num - 1:
                h = 0.5 * (self.D_sec_arr[i] - self.D_sec_arr[i - 1]) / 2
                D_av = self.D_sec_arr[i] - h
                D_bound_arr.append(self.D_sec_arr[i])
            else:
                h = 0.5 * ((self.D_sec_arr[i] - self.D_sec_arr[i - 1]) / 2 +
                           (self.D_sec_arr[i + 1] - self.D_sec_arr[i]) / 2)
                D_av = self.D_sec_arr[i]
                D_bound_arr.append(self.D_sec_arr[i] + h)
            D_arr.append(D_av)
            h_arr.append(h)

        return np.array(D_arr), np.array(D_bound_arr), np.array(h_arr)

    def _get_partition(self, param):
        """Возвращает массив среднеинтегральных значений величины на участках, с заданным
        распределением величины по радиусу."""
        r_arr = 0.5 * np.array(self.D_bound_arr)
        res = []

        for i in range(len(r_arr) - 1):
            res.append(quad(param, r_arr[i], r_arr[i + 1])[0] / (r_arr[i + 1] - r_arr[i]))

        return res

    def _get_g_gas(self, r):
        """Возвращает плотность расхода газа на заданном радиусе."""
        a_cr = gd.a_cr(self.T_gas_stag(r), self.k_gas, self.work_fluid.R)
        v = a_cr * self.lam_gas_in(r)
        p_gas = self.p_gas_stag(r) * gd.pi_lam(self.lam_gas_in(r), self.k_gas)
        T_gas = self.T_gas_stag(r) * gd.tau_lam(self.lam_gas_in(r), self.k_gas)
        rho = p_gas / (T_gas * self.work_fluid.R)

        return rho * v * 2 * np.pi * r

    def _get_G_arr(self, g_arr: typing.List[float]):
        """Возвращает массив значение расхода по заданному массиву среднеинтегральных
        значений плотностей расхода на участках"""
        r_arr = 0.5 * np.array(self.D_bound_arr)
        res = []

        for i in range(len(r_arr) - 1):
            res.append(g_arr[i] * (r_arr[i + 1] - r_arr[i]))
        return res

    def _get_hole_num_arr(self):
        delta = np.array(self.hole_num) / (self.sector_num - 1)
        hole_num_arr = []

        for i in range(self.sector_num):
            if i == 0 or i == self.sector_num - 1:
                hole_num_arr.append(list(delta / 2))
            else:
                hole_num_arr.append(list(delta))
        return hole_num_arr

    def _compute_sectors(self):
        for i, sector in enumerate(self.sectors):
            self.logger.info('Computing sector №%s' % i)
            sector.T_wall_av = self.av_param.T_wall_av
            sector.c_p_cool_av = self.av_param.c_p_cool_av
            sector.set_calculators()
            sector.compute()

    def _get_average_value(self, value_arr):
        square = 0
        for i in range(self.sector_num):
            square += value_arr[i] * self.sector_height_arr[i]

        res = square / (0.5 * (self.D_out - self.D_in))
        return res

    def _make_iteration(self, alpha_film_av, T_film_av, G_cool_av, T_wall_out_av):
        self.logger.debug('alpha_film_av = %.3f' % alpha_film_av)
        self.logger.debug('T_film_av = %.3f' % T_film_av)
        self.logger.debug('G_cool_av = %.5f' % G_cool_av)
        self.logger.debug('T_wall_out_av = %.3f' % T_wall_out_av)

        self.av_param.alpha_out = alpha_film_av
        self.av_param.T_out_stag = T_film_av
        self.av_param.G_cool = G_cool_av
        self.av_param.T_wall_out = T_wall_out_av
        self.av_param.compute()

        self.logger.debug('channel_width = %.4f' % self.av_param.channel_width)
        self.logger.debug('T_wall_av = %.3f' % self.av_param.T_wall_av)
        self._T_wall_av_arr.append(self.av_param.T_wall_av)
        self._compute_sectors()

        self.T_cool_av = self._get_average_value([sector.T_cool_av for sector in self.sectors])
        self.logger.debug('From loc param T_cool_av = %.3f' % self.T_cool_av)
        self.logger.debug('From av param T_cool_av = %.3f' % self.av_param.T_cool_fluid_av)
        return (self._get_average_value([sector.alpha_film_av for sector in self.sectors]),
                self._get_average_value([sector.T_film_av for sector in self.sectors]),
                sum([sector.G_cool_av for sector in self.sectors]),
                self._get_average_value([sector.T_wall_out_av for sector in self.sectors]))

    def _get_residual(self):
        l = len(self._T_wall_av_arr)
        if l > 1:
            res = abs(self._T_wall_av_arr[l - 1] - self._T_wall_av_arr[l - 2]) / self._T_wall_av_arr[l - 1]
        else:
            res = 1.
        return res

    def compute(self):
        self.iter_num = 0
        self.res = 1.

        while self.res >= self.accuracy:
            if not self.alpha_film_av and not self.T_film_av and not self.G_cool_av:
                (self.alpha_film_av,
                 self.T_film_av,
                 self.G_cool_av,
                 self.T_wall_out_av) = self._make_iteration(alpha_film_av=self.alpha_gas_av,
                                                            T_film_av=self.T_gas_stag(0.25 * (self.D_in + self.D_out)),
                                                            G_cool_av=self.G_cool0,
                                                            T_wall_out_av=self.T_wall_out_av_init)
            else:
                (self.alpha_film_av,
                 self.T_film_av,
                 self.G_cool_av,
                 self.T_wall_out_av) = self._make_iteration(alpha_film_av=self.alpha_film_av,
                                                            T_film_av=self.T_film_av,
                                                            G_cool_av=self.G_cool_av,
                                                            T_wall_out_av=self.T_wall_out_av)
            self.iter_num += 1
            self.res = self._get_residual()
            self.logger.info('Blade computing: Iter №%s, residual = %.4f\n' % (self.iter_num, self.res))

    def get_cool_eff_av(self):
        T_gas_av = 2 * quad(lambda r: self.T_gas_stag(r), 0.5 * self.D_in, 0.5 * self.D_out)[0] / (self.D_out - self.D_in)
        return (T_gas_av - self.T_wall_out_av) / (T_gas_av - self.T_cool_av)

    def get_cooling_results(self) -> FilmBladeCoolingResults:
        film_params = []
        local_params = []

        for sector in self.sectors:
            film_params.append(sector.get_film_params())
            local_params.append(sector.get_local_params())

        results = FilmBladeCoolingResults(self.p_cool_stag0,
                                          self.T_cool0,
                                          self.G_cool0,
                                          self.channel_width,
                                          self.wall_thickness,
                                          self.get_cool_eff_av(),
                                          self.lam_cover,
                                          self.cover_thickness,
                                          film_params,
                                          local_params)
        return results

    def save_results(self, filename: str):
        with open(filename, 'wb') as file:
            results = self.get_cooling_results()
            pickle.dump(results, file)

    def _plot_partition(self, param_arr, param_func, figsize):
        plt.figure(figsize=figsize)
        r_arr = 0.5 * np.array(np.linspace(self.D_in, self.D_out, 100))
        param_func_arr = [param_func(r) for r in r_arr]
        plt.plot(param_func_arr, r_arr, lw=2, color='red', label='Исходный профиль')

        r_bound_arr = []
        param_arr_to_plot = []
        for i in range(self.sector_num):
            if i == 0:
                r_bound_arr.append(0.5 * self.D_sec_arr[i])
                r_bound_arr.append(0.5 * self.D_sec_arr[i] + self.sector_height_arr[i])
            elif i == self.sector_num - 1:
                r_bound_arr.append(0.5 * self.D_sec_arr[i] - self.sector_height_arr[i])
                r_bound_arr.append(0.5 * self.D_sec_arr[i])
            else:
                r_bound_arr.append(0.5 * self.D_sec_arr[i] -
                                   0.5 * (self.D_sec_arr[i] - self.D_sec_arr[i - 1]) / 2)
                r_bound_arr.append(0.5 * self.D_sec_arr[i] +
                                   0.5 * (self.D_sec_arr[i + 1] - self.D_sec_arr[i]) / 2)
            param_arr_to_plot.append(param_arr[i])
            param_arr_to_plot.append(param_arr[i])
        plt.plot(param_arr_to_plot, r_bound_arr, lw=2, color='blue', label='Разбиение')

        plt.ylim(min(r_arr), max(r_arr))
        plt.grid(linewidth=1)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=14)

    def plot_lam_gas_in(self, figsize=(6, 4), filename=None, label=True):
        self._plot_partition(self.lam_gas_in_arr, self.lam_gas_in, figsize)

        if label:
            plt.ylabel(r'$r,\ м$', fontsize=14)
            plt.xlabel(r'$\lambda_{вх}$', fontsize=14)
        if filename:
            plt.savefig(filename)
        plt.show()

    def plot_lam_gas_out(self, figsize=(6, 4), filename=None, label=True):
        self._plot_partition(self.lam_gas_out_arr, self.lam_gas_out, figsize)

        if label:
            plt.ylabel(r'$r,\ м$', fontsize=14)
            plt.xlabel(r'$\lambda_{вых}$', fontsize=14)
        if filename:
            plt.savefig(filename)
        plt.show()

    def plot_p_gas_stag(self, figsize=(6, 4), filename=None, label=True):
        self._plot_partition(self.p_gas_stag_arr, self.p_gas_stag, figsize)

        if label:
            plt.ylabel(r'$r,\ м$', fontsize=14)
            plt.xlabel(r'$p_{г}^*,\ Па$', fontsize=14)
        if filename:
            plt.savefig(filename)
        plt.show()

    def plot_T_gas_stag(self, figsize=(6, 4), filename=None, label=True):
        self._plot_partition(self.T_gas_stag_arr, self.T_gas_stag, figsize)

        if label:
            plt.ylabel(r'$r,\ м$', fontsize=14)
            plt.xlabel(r'$T_{г}^*,\ К$', fontsize=14)
        if filename:
            plt.savefig(filename)
        plt.show()

    def plot_g_gas(self, figsize=(6, 4), filename=None, label=True):
        self._plot_partition(self.g_gas_arr, self._get_g_gas, figsize)

        if label:
            plt.ylabel(r'$r,\ м$', fontsize=14)
            plt.xlabel(r'$g_г$', fontsize=14)
        if filename:
            plt.savefig(filename)
        plt.show()

    def plot_g_cool(self, figsize=(6, 4), filename=None, label=True):
        self._plot_partition(self.g_cool0_arr, self.g_cool0, figsize)

        if label:
            plt.ylabel(r'$r$', fontsize=14)
            plt.xlabel(r'$g_в$', fontsize=14)
        if filename:
            plt.savefig(filename)
        plt.show()

    def plot_T_wall(self, T_material, figsize=(7, 5), filename=None, label=True):
        plt.figure(figsize=figsize)

        for i, sector in enumerate(self.sectors):
            plt.plot(sector.local_param.x_arr * 1e3, sector.local_param.T_wall_arr, lw=2, label='Sector %s' % i)

        x_min = min(self.sectors[0].local_param.x_arr) * 1e3
        x_max = max(self.sectors[0].local_param.x_arr) * 1e3
        plt.plot([x_min, x_max], [T_material, T_material], lw=2, linestyle='--', color='black')
        plt.xlim(x_min, x_max)

        if label:
            plt.xlabel(r'$x,\ мм$', fontsize=14)
            plt.ylabel(r'$T_{ст},\ К$', fontsize=14)
        plt.grid(linewidth=1)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        if filename:
            plt.savefig(filename)
        plt.show()

    def plot_T_film(self, figsize=(7, 5), filename=None, label=True):
        plt.figure(figsize=figsize)

        for i, sector in enumerate(self.sectors):
            plt.plot(sector.local_param.x_arr * 1e3, [sector.film.get_T_film(x) for x in sector.local_param.x_arr],
                     lw=2, label='Sector %s' % i)

        x_min = min(self.sectors[0].local_param.x_arr) * 1e3
        x_max = max(self.sectors[0].local_param.x_arr) * 1e3
        plt.xlim(x_min, x_max)

        if label:
            plt.xlabel(r'$x,\ мм$', fontsize=14)
            plt.ylabel(r'$T_{пл}^*,\ К$', fontsize=14)
        plt.grid(linewidth=1)
        plt.legend(fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if filename:
            plt.savefig(filename)
        plt.show()

    def plot_T_cool(self, figsize=(7, 5), filename=None, label=True):
        plt.figure(figsize=figsize)

        for i, sector in enumerate(self.sectors):
            plt.plot(sector.local_param.x_arr * 1e3, sector.local_param.T_cool_fluid_arr,
                     lw=2, label='Sector %s' % i)

        x_min = min(self.sectors[0].local_param.x_arr) * 1e3
        x_max = max(self.sectors[0].local_param.x_arr) * 1e3
        plt.xlim(x_min, x_max)

        if label:
            plt.xlabel(r'$x,\ мм$', fontsize=14)
            plt.ylabel(r'$T_{в}^*,\ К$', fontsize=14)
        plt.grid(linewidth=1)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        if filename:
            plt.savefig(filename)
        plt.show()

    def plot_T_wall_in_point(self, x, T_material_max, figsize=(7, 5), filename=None, label=True):
        plt.figure(figsize=figsize)
        r_arr = self.D_av_arr * 0.5
        T_arr = [self.sectors[i].local_param.get_T_wall(x) for i in range(self.sector_num)]
        plt.plot(T_arr, r_arr, lw=2)
        plt.plot([T_material_max, T_material_max], [min(r_arr), max(r_arr)], lw=2, linestyle='--', color='black')
        plt.ylim(min(r_arr), max(r_arr))
        if label:
            plt.ylabel(r'$r,\ м$', fontsize=14)
            plt.xlabel(r'$T_{ст},\ К$', fontsize=14)
        plt.grid(linewidth=1)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        if filename:
            plt.savefig(filename)
        plt.show()

    def write_nx_exp_file(self, filename: str, sector_num: int):
        sector = self.sectors[sector_num]
        section = sector.section

        x_holes, y_holes = section.get_holes_coordinates(sector.x_hole_rel)

        with open(filename, 'w') as file:
            lines = []

            for i in range(len(self.x_hole_rel)):
                lines.append('[mm]x_hole_%s%s=%s\n' % (sector_num, i, x_holes[i] * 1e3))
                lines.append('[mm]y_hole_%s%s=%s\n' % (sector_num, i, y_holes[i] * 1e3))
                lines.append('z_hole_%s%s=%s\n' % (sector_num, i, self.hole_num[i]))
                lines.append('[mm]d_hole_%s%s=%s\n' % (sector_num, i, self.d_hole[i] * 1e3))
            lines.append('[mm]wall_thickness=%s\n' % (self.wall_thickness * 1.e3))
            lines.append('[mm]delta=%s\n' % (self.channel_width * 1.e3))

            file.writelines(lines)

    def write_excel_table(self, filename: str, sector_num: int):
        sector = self.sectors[sector_num]
        section = sector.section

        hole_step = 0.5 * (self.D_out - self.D_in) / np.array(self.hole_num)
        x_holes, y_holes = section.get_holes_coordinates(sector.x_hole_rel)

        book = xlwt.Workbook()
        sheet = book.add_sheet('Sheet 1')
        sheet.write(0, 1, 'Спинка')
        sheet.write(0, 5, 'Корыто')
        sheet.write(1, 0, '№')
        sheet.write(2, 0, 'X, мм')
        sheet.write(3, 0, 'Y, мм')
        sheet.write(4, 0, 'Кол-во')
        sheet.write(5, 0, 'Шаг, мм')
        sheet.write(6, 0, '<o>, мм')

        for i in range(len(self.x_hole_rel)):
            sheet.write(1, i + 1, i + 1)
            sheet.write(2, i + 1, round(x_holes[i] * 1e3, 1))
            sheet.write(3, i + 1, round(y_holes[i] * 1e3, 1))
            sheet.write(4, i + 1, self.hole_num[i])
            sheet.write(5, i + 1, round(hole_step[i] * 1e3, 2))
            sheet.write(6, i + 1, self.d_hole[i] * 1e3)

        book.save(filename)


@typing.overload
def get_sa_cooler(
        profiling: ProfilingResultsForCooling,
        channel_width,
        wall_thickness,
        T_wall_out_av_init,
        lam_blade: typing.Callable[[float], float],
        x_hole_rel: typing.List[float],
        hole_num: typing.List[int],
        d_hole: typing.List[float],
        phi_hole: typing.List[float],
        mu_hole: typing.List[float],
        work_fluid: IdealGas,
        T_cool0,
        p_cool_stag0,
        g_cool0: typing.Callable[[float], float],
        cool_fluid: IdealGas,
        cover_thickness=0.15e-3,
        lam_cover=2,
        node_num=500,
        accuracy=0.01,
        g_cool0_s=0.5
) -> FilmBladeCooler:
    ...


@typing.overload
def get_sa_cooler(
        profiling: StageProfiler,
        channel_width,
        wall_thickness,
        T_wall_out_av_init,
        lam_blade: typing.Callable[[float], float],
        x_hole_rel: typing.List[float],
        hole_num: typing.List[int],
        d_hole: typing.List[float],
        phi_hole: typing.List[float],
        mu_hole: typing.List[float],
        work_fluid: IdealGas,
        T_cool0,
        p_cool_stag0,
        g_cool0: typing.Callable[[float], float],
        cool_fluid: IdealGas,
        cover_thickness=0.15e-3,
        lam_cover=2,
        node_num=500,
        accuracy=0.01,
        g_cool0_s=0.5
) -> FilmBladeCooler:
    ...


def get_sa_cooler(
        profiling,
        channel_width,
        wall_thickness,
        T_wall_out_av_init,
        lam_blade: typing.Callable[[float], float],
        x_hole_rel: typing.List[float],
        hole_num: typing.List[int],
        d_hole: typing.List[float],
        phi_hole: typing.List[float],
        mu_hole: typing.List[float],
        work_fluid: IdealGas,
        T_cool0,
        p_cool_stag0,
        g_cool0: typing.Callable[[float], float],
        cool_fluid: IdealGas,
        cover_thickness=0.15e-3,
        lam_cover=2,
        node_num=500,
        accuracy=0.01,
        g_cool0_s=0.5
) -> FilmBladeCooler:

    if type(profiling) == ProfilingResultsForCooling:
        r_arr = np.linspace(0.5 * profiling.D_in, 0.5 * profiling.D_out, len(profiling.T_gas_stag))
        T_gas_stag_int = interp1d(r_arr, profiling.T_gas_stag, bounds_error=False, fill_value='extrapolate')
        p_gas_stag_int = interp1d(r_arr, profiling.p_gas_stag, bounds_error=False, fill_value='extrapolate')
        lam_gas_in_int = interp1d(r_arr, profiling.lam_gas_in, bounds_error=False, fill_value='extrapolate')
        lam_gas_out_int = interp1d(r_arr, profiling.lam_gas_out, bounds_error=False, fill_value='extrapolate')

        cooler = FilmBladeCooler(sections=profiling.sections,
                                 channel_width=channel_width,
                                 wall_thickness=wall_thickness,
                                 D_in=profiling.D_in,
                                 D_out=profiling.D_out,
                                 T_wall_out_av_init=T_wall_out_av_init,
                                 lam_blade=lam_blade,
                                 x_hole_rel=x_hole_rel,
                                 hole_num=hole_num,
                                 d_hole=d_hole,
                                 phi_hole=phi_hole,
                                 mu_hole=mu_hole,
                                 T_gas_stag=lambda r: T_gas_stag_int(r).__float__(),
                                 p_gas_stag=lambda r: p_gas_stag_int(r).__float__(),
                                 c_p_gas_av=profiling.c_p,
                                 lam_gas_in=lambda r: lam_gas_in_int(r).__float__(),
                                 lam_gas_out=lambda r: lam_gas_out_int(r).__float__(),
                                 work_fluid=work_fluid,
                                 T_cool0=T_cool0,
                                 p_cool_stag0=p_cool_stag0,
                                 g_cool0=g_cool0,
                                 cool_fluid=cool_fluid,
                                 cover_thickness=cover_thickness,
                                 lam_cover=lam_cover,
                                 node_num=node_num,
                                 accuracy=accuracy,
                                 g_cool0_s=g_cool0_s)
    elif type(profiling) == StageProfiler:
        cooler = FilmBladeCooler(sections=profiling.sa_sections,
                                 channel_width=channel_width,
                                 wall_thickness=wall_thickness,
                                 D_in=profiling.D1_in,
                                 D_out=profiling.D1_out,
                                 T_wall_out_av_init=T_wall_out_av_init,
                                 lam_blade=lam_blade,
                                 x_hole_rel=x_hole_rel,
                                 hole_num=hole_num,
                                 d_hole=d_hole,
                                 phi_hole=phi_hole,
                                 mu_hole=mu_hole,
                                 T_gas_stag=profiling.T0_stag,
                                 p_gas_stag=profiling.p0_stag,
                                 c_p_gas_av=profiling.c_p,
                                 lam_gas_in=profiling.lam_c0,
                                 lam_gas_out=profiling.lam_c1,
                                 work_fluid=work_fluid,
                                 T_cool0=T_cool0,
                                 p_cool_stag0=p_cool_stag0,
                                 g_cool0=g_cool0,
                                 cool_fluid=cool_fluid,
                                 cover_thickness=cover_thickness,
                                 lam_cover=lam_cover,
                                 node_num=node_num,
                                 accuracy=accuracy,
                                 g_cool0_s=g_cool0_s)
    else:
        raise TypeError("profiling can not have this type: %s" % type(profiling))
    return cooler


@typing.overload
def get_rk_cooler(
        profiling: ProfilingResultsForCooling,
        channel_width,
        wall_thickness,
        T_wall_out_av_init,
        lam_blade: typing.Callable[[float], float],
        x_hole_rel: typing.List[float],
        hole_num: typing.List[int],
        d_hole: typing.List[float],
        phi_hole: typing.List[float],
        mu_hole: typing.List[float],
        work_fluid: IdealGas,
        T_cool0,
        p_cool_stag0,
        g_cool0: typing.Callable[[float], float],
        cool_fluid: IdealGas,
        cover_thickness=0.15e-3,
        lam_cover=2,
        node_num=500,
        accuracy=0.01,
        g_cool0_s=0.5
):
    ...


@typing.overload
def get_rk_cooler(
        profiling: StageProfiler,
        channel_width,
        wall_thickness,
        T_wall_out_av_init,
        lam_blade: typing.Callable[[float], float],
        x_hole_rel: typing.List[float],
        hole_num: typing.List[int],
        d_hole: typing.List[float],
        phi_hole: typing.List[float],
        mu_hole: typing.List[float],
        work_fluid: IdealGas,
        T_cool0,
        p_cool_stag0,
        g_cool0: typing.Callable[[float], float],
        cool_fluid: IdealGas,
        cover_thickness=0.15e-3,
        lam_cover=2,
        node_num=500,
        accuracy=0.01,
        g_cool0_s=0.5
):
    ...


def get_rk_cooler(
        profiling,
        channel_width,
        wall_thickness,
        T_wall_out_av_init,
        lam_blade: typing.Callable[[float], float],
        x_hole_rel: typing.List[float],
        hole_num: typing.List[int],
        d_hole: typing.List[float],
        phi_hole: typing.List[float],
        mu_hole: typing.List[float],
        work_fluid: IdealGas,
        T_cool0,
        p_cool_stag0,
        g_cool0: typing.Callable[[float], float],
        cool_fluid: IdealGas,
        cover_thickness=0.15e-3,
        lam_cover=2,
        node_num=500,
        accuracy=0.01,
        g_cool0_s=0.5
) -> FilmBladeCooler:

    if type(profiling) == ProfilingResultsForCooling:
        r_arr = np.linspace(0.5 * profiling.D_in, 0.5 * profiling.D_out, len(profiling.T_gas_stag))
        T_gas_stag_int = interp1d(r_arr, profiling.T_gas_stag, bounds_error=False, fill_value='extrapolate')
        p_gas_stag_int = interp1d(r_arr, profiling.p_gas_stag, bounds_error=False, fill_value='extrapolate')
        lam_gas_in_int = interp1d(r_arr, profiling.lam_gas_in, bounds_error=False, fill_value='extrapolate')
        lam_gas_out_int = interp1d(r_arr, profiling.lam_gas_out, bounds_error=False, fill_value='extrapolate')

        cooler = FilmBladeCooler(sections=profiling.sections,
                                 channel_width=channel_width,
                                 wall_thickness=wall_thickness,
                                 D_in=profiling.D_in,
                                 D_out=profiling.D_out,
                                 T_wall_out_av_init=T_wall_out_av_init,
                                 lam_blade=lam_blade,
                                 x_hole_rel=x_hole_rel,
                                 hole_num=hole_num,
                                 d_hole=d_hole,
                                 phi_hole=phi_hole,
                                 mu_hole=mu_hole,
                                 T_gas_stag=lambda r: T_gas_stag_int(r).__float__(),
                                 p_gas_stag=lambda r: p_gas_stag_int(r).__float__(),
                                 c_p_gas_av=profiling.c_p,
                                 lam_gas_in=lambda r: lam_gas_in_int(r).__float__(),
                                 lam_gas_out=lambda r: lam_gas_out_int(r).__float__(),
                                 work_fluid=work_fluid,
                                 T_cool0=T_cool0,
                                 p_cool_stag0=p_cool_stag0,
                                 g_cool0=g_cool0,
                                 cool_fluid=cool_fluid,
                                 cover_thickness=cover_thickness,
                                 lam_cover=lam_cover,
                                 node_num=node_num,
                                 accuracy=accuracy,
                                 g_cool0_s=g_cool0_s)
    elif type(profiling) == StageProfiler:
        cooler = FilmBladeCooler(sections=profiling.rk_sections,
                                 channel_width=channel_width,
                                 wall_thickness=wall_thickness,
                                 D_in=profiling.D1_in,
                                 D_out=profiling.D1_out,
                                 T_wall_out_av_init=T_wall_out_av_init,
                                 lam_blade=lam_blade,
                                 x_hole_rel=x_hole_rel,
                                 hole_num=hole_num,
                                 d_hole=d_hole,
                                 phi_hole=phi_hole,
                                 mu_hole=mu_hole,
                                 T_gas_stag=profiling.T1_w_stag,
                                 p_gas_stag=profiling.p0_stag,
                                 c_p_gas_av=profiling.c_p,
                                 lam_gas_in=profiling.lam_w1,
                                 lam_gas_out=profiling.lam_w2,
                                 work_fluid=work_fluid,
                                 T_cool0=T_cool0,
                                 p_cool_stag0=p_cool_stag0,
                                 g_cool0=g_cool0,
                                 cool_fluid=cool_fluid,
                                 cover_thickness=cover_thickness,
                                 lam_cover=lam_cover,
                                 node_num=node_num,
                                 accuracy=accuracy,
                                 g_cool0_s=g_cool0_s)
    else:
        raise TypeError("profiling can not have this type: %s" % type(profiling))
    return cooler