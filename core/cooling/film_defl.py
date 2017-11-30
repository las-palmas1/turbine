from .tools import GasBladeHeatExchange, LocalParamCalculator, FilmCalculator, DeflectorAverageParamCalculator
from ..profiling.section import BladeSection
from ..profiling.stage import StageProfiler
from gas_turbine_cycle.gases import IdealGas, Air
import numpy as np
import typing
import matplotlib.pyplot as plt

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
                 c_p_cool_av=None,
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
        :param c_p_cool_av: Средняя теплоемкость охлаждающей среды.
        :param cool_fluid: Охлаждающая среда.
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
        self.c_p_cool_av = c_p_cool_av
        self.cool_fluid = cool_fluid
        self.node_num = node_num
        self.accuracy = accuracy

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
                                                node_num=self.node_num)
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
                                   height=self.height)

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
                    G_cool=lambda x: self.G_cool0
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

    def plot_v_gas(self, figsize=(6, 4)):
        plt.figure(figsize=figsize)

        x_arr = self.local_param.x_arr
        v_arr = [self.get_v_gas(x) for x in x_arr]

        plt.plot(x_arr * 1e3, v_arr, lw=2, color='red')
        plt.xlabel(r'$x,\ мм$', fontsize=14)
        plt.ylabel(r'$v_г,\ м/с$', fontsize=14)
        plt.grid()
        plt.show()


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
                 G_gas,
                 c_p_gas_av,
                 lam_gas_in: typing.Callable[[float], float],
                 lam_gas_out: typing.Callable[[float], float],
                 work_fluid: IdealGas,
                 T_cool0,
                 p_cool_stag0,
                 G_cool0,
                 cool_fluid: IdealGas,
                 node_num: int = 500,
                 accuracy: float = 0.01
                 ):
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
        self.G_gas = G_gas
        self.c_p_gas_av = c_p_gas_av
        self.lam_gas_in = lam_gas_in
        self.lam_gas_out = lam_gas_out
        self.work_fluid = work_fluid
        self.T_cool0 = T_cool0
        self.p_cool_stag0 = p_cool_stag0
        self.G_cool0 = G_cool0
        self.cool_fluid = cool_fluid
        self.node_num = node_num
        self.accuracy = accuracy
        self.logger = Logger(level_name=log_level)

        self.av_param = DeflectorAverageParamCalculator(section=self.av_section,
                                                        height=0.5 * (D_out - D_in),
                                                        D_av=0.5 * (D_in + D_out),
                                                        wall_thickness=wall_thickness,
                                                        T_cool_fluid0=T_cool0,
                                                        T_out_stag=T_gas_stag(0.25 * (D_in + D_out)),
                                                        cool_fluid=type(cool_fluid)(),
                                                        lam_blade=lam_blade)

        self.D_av_arr, self.sector_height_arr = self._get_D_av_arr_and_height_arr()
        (self.T_gas_stag_arr,
         self.p_gas_stag_arr,
         self.lam_gas_in_arr,
         self.lam_gas_out_arr) = self._get_gas_param_arr(self.D_av_arr)
        self.G_gas_arr = self._get_G_arr(self.G_gas)
        self.G_cool0_arr = self._get_G_arr(self.G_cool0)
        self.hole_num_arr = self._get_hole_num_arr()

        self.sectors = [FilmSectorCooler(section=sections[i],
                                         height=self.sector_height_arr[i],
                                         channel_width=channel_width,
                                         wall_thickness=wall_thickness,
                                         D_av=self.D_av_arr[i],
                                         lam_blade=lam_blade,
                                         x_hole_rel=x_hole_rel,
                                         hole_num=self.hole_num_arr[i],
                                         d_hole=d_hole,
                                         phi_hole=phi_hole,
                                         mu_hole=mu_hole,
                                         T_gas_stag=self.T_gas_stag_arr[i],
                                         p_gas_stag=self.p_gas_stag_arr[i],
                                         G_gas=self.G_gas_arr[i],
                                         c_p_gas_av=c_p_gas_av,
                                         lam_gas_in=self.lam_gas_in_arr[i],
                                         lam_gas_out=self.lam_gas_out_arr[i],
                                         work_fluid=type(work_fluid)(),
                                         T_cool0=T_cool0,
                                         p_cool_stag0=p_cool_stag0,
                                         G_cool0=self.G_cool0_arr[i],
                                         cool_fluid=type(cool_fluid)(),
                                         node_num=node_num,
                                         accuracy=accuracy
                                         ) for i in range(self.sector_num)]

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

    def _get_D_av_arr_and_height_arr(self):
        """Массив средних диаметров для секторов."""
        D_arr = []
        h_arr = []

        for i in range(self.sector_num):
            if i == 0:
                h = 0.5 * (self.D_sec_arr[i + 1] - self.D_sec_arr[i]) / 2
                D_av = self.D_sec_arr[i] + h
            elif i == self.sector_num - 1:
                h = 0.5 * (self.D_sec_arr[i] - self.D_sec_arr[i - 1]) / 2
                D_av = self.D_sec_arr[i] - h
            else:
                h = 0.5 * ((self.D_sec_arr[i] - self.D_sec_arr[i - 1]) / 2 +
                           (self.D_sec_arr[i + 1] - self.D_sec_arr[i]) / 2)
                D_av = self.D_sec_arr[i]
            D_arr.append(D_av)
            h_arr.append(h)

        return np.array(D_arr), np.array(h_arr)

    def _get_G_arr(self, G):
        dG = G / (self.sector_num - 1)
        G_arr = []

        for i in range(self.sector_num):
            if i == 0 or i == self.sector_num - 1:
                G_arr.append(dG / 2)
            else:
                G_arr.append(dG)
        return G_arr

    def _get_hole_num_arr(self):
        delta = np.array(self.hole_num) / (self.sector_num - 1)
        hole_num_arr = []

        for i in range(self.sector_num):
            if i == 0 or i == self.sector_num - 1:
                hole_num_arr.append(list(delta / 2))
            else:
                hole_num_arr.append(list(delta))
        return hole_num_arr

    def _get_gas_param_arr(self, D_av_arr):
        """Массивы параметров газа для секторов."""
        T_gas_stag = []
        p_gas_stag = []
        lam_gas_in = []
        lam_gas_out = []

        for D in D_av_arr:
            T_gas_stag.append(self.T_gas_stag(0.5 * D))
            p_gas_stag.append(self.p_gas_stag(0.5 * D))
            lam_gas_in.append(self.lam_gas_in(0.5 * D))
            lam_gas_out.append(self.lam_gas_out(0.5 * D))

        return T_gas_stag, p_gas_stag, lam_gas_in, lam_gas_out

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

    def _plot_partition(self, param_arr):
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

    def plot_lam_gas_in(self, figsize=(6, 4)):
        plt.figure(figsize=figsize)
        r_arr = 0.5 * np.array(np.linspace(self.D_in, self.D_out, 100))
        lam_in_arr = [self.lam_gas_in(r) for r in r_arr]

        plt.plot(lam_in_arr, r_arr, lw=2, color='red', label='Исходный профиль')
        self._plot_partition(self.lam_gas_in_arr)
        plt.ylim(min(r_arr), max(r_arr))
        plt.grid()
        plt.ylabel(r'$r$', fontsize=14)
        plt.xlabel(r'$\lambda_{вх}$', fontsize=14)
        plt.legend(fontsize=10)
        plt.show()

    def plot_lam_gas_out(self, figsize=(6, 4)):
        plt.figure(figsize=figsize)
        r_arr = 0.5 * np.array(np.linspace(self.D_in, self.D_out, 100))
        lam_in_arr = [self.lam_gas_out(r) for r in r_arr]

        plt.plot(lam_in_arr, r_arr, lw=2, color='red', label='Исходный профиль')
        self._plot_partition(self.lam_gas_out_arr)
        plt.ylim(min(r_arr), max(r_arr))
        plt.grid()
        plt.ylabel(r'$r$', fontsize=14)
        plt.xlabel(r'$\lambda_{вых}$', fontsize=14)
        plt.legend(fontsize=10)
        plt.show()

    def plot_p_gas_stag(self, figsize=(6, 4)):
        plt.figure(figsize=figsize)
        r_arr = 0.5 * np.array(np.linspace(self.D_in, self.D_out, 100))
        lam_in_arr = [self.p_gas_stag(r) for r in r_arr]

        plt.plot(lam_in_arr, r_arr, lw=2, color='red', label='Исходный профиль')
        self._plot_partition(self.p_gas_stag_arr)
        plt.ylim(min(r_arr), max(r_arr))
        plt.grid()
        plt.ylabel(r'$r$', fontsize=14)
        plt.xlabel(r'$p_{г}^*$', fontsize=14)
        plt.legend(fontsize=10)
        plt.show()

    def plot_T_gas_stag(self, figsize=(6, 4)):
        plt.figure(figsize=figsize)
        r_arr = 0.5 * np.array(np.linspace(self.D_in, self.D_out, 100))
        lam_in_arr = [self.T_gas_stag(r) for r in r_arr]

        plt.plot(lam_in_arr, r_arr, lw=2, color='red', label='Исходный профиль')
        self._plot_partition(self.T_gas_stag_arr)
        plt.ylim(min(r_arr), max(r_arr))
        plt.grid()
        plt.ylabel(r'$r$', fontsize=14)
        plt.xlabel(r'$T_{г}^*$', fontsize=14)
        plt.legend(fontsize=10)
        plt.show()

    def plot_T_wall(self, figsize=(7, 5)):
        plt.figure(figsize=figsize)

        for i, sector in enumerate(self.sectors):
            plt.plot(sector.local_param.x_arr * 1e3, sector.local_param.T_wall_arr, lw=2, label='Sector %s' % i)

        plt.xlabel(r'$x,\ мм$', fontsize=14)
        plt.ylabel(r'$T_{ст},\ К$', fontsize=14)
        plt.grid()
        plt.legend(fontsize=10)
        plt.show()

    def plot_T_film(self, figsize=(7, 5)):
        plt.figure(figsize=figsize)

        for i, sector in enumerate(self.sectors):
            plt.plot(sector.local_param.x_arr * 1e3, [sector.film.get_T_film(x) for x in sector.local_param.x_arr],
                     lw=2, label='Sector %s' % i)

        plt.xlabel(r'$x,\ мм$', fontsize=14)
        plt.ylabel(r'$T_{пл}^*,\ К$', fontsize=14)
        plt.grid()
        plt.legend(fontsize=10)
        plt.show()

    def plot_T_cool(self, figsize=(7, 5)):
        plt.figure(figsize=figsize)

        for i, sector in enumerate(self.sectors):
            plt.plot(sector.local_param.x_arr * 1e3, sector.local_param.T_cool_fluid_arr,
                     lw=2, label='Sector %s' % i)

        plt.xlabel(r'$x,\ мм$', fontsize=14)
        plt.ylabel(r'$T_{в}^*,\ К$', fontsize=14)
        plt.grid()
        plt.legend(fontsize=10)
        plt.show()


def get_sa_cooler(
        stage_profiler: StageProfiler,
        channel_width,
        wall_thickness,
        T_wall_out_av_init,
        lam_blade: typing.Callable[[float], float],
        x_hole_rel: typing.List[float],
        hole_num: typing.List[int],
        d_hole: typing.List[float],
        phi_hole: typing.List[float],
        mu_hole: typing.List[float],
        G_gas,
        work_fluid: IdealGas,
        T_cool0,
        p_cool_stag0,
        G_cool0,
        cool_fluid: IdealGas,
        node_num=500,
        accuracy=0.01
) -> FilmBladeCooler:
    cooler = FilmBladeCooler(sections=stage_profiler.sa_sections,
                             channel_width=channel_width,
                             wall_thickness=wall_thickness,
                             D_in=stage_profiler.D1_in,
                             D_out=stage_profiler.D1_out,
                             T_wall_out_av_init=T_wall_out_av_init,
                             lam_blade=lam_blade,
                             x_hole_rel=x_hole_rel,
                             hole_num=hole_num,
                             d_hole=d_hole,
                             phi_hole=phi_hole,
                             mu_hole=mu_hole,
                             T_gas_stag=stage_profiler.T0_stag,
                             p_gas_stag=stage_profiler.p0_stag,
                             G_gas=G_gas,
                             c_p_gas_av=stage_profiler.c_p,
                             lam_gas_in=stage_profiler.lam_c0,
                             lam_gas_out=stage_profiler.lam_c1,
                             work_fluid=work_fluid,
                             T_cool0=T_cool0,
                             p_cool_stag0=p_cool_stag0,
                             G_cool0=G_cool0,
                             cool_fluid=cool_fluid,
                             node_num=node_num,
                             accuracy=accuracy)
    return cooler


def get_rk_cooler(
        stage_profiler: StageProfiler,
        channel_width,
        wall_thickness,
        T_wall_out_av_init,
        lam_blade: typing.Callable[[float], float],
        x_hole_rel: typing.List[float],
        hole_num: typing.List[int],
        d_hole: typing.List[float],
        phi_hole: typing.List[float],
        mu_hole: typing.List[float],
        G_gas,
        work_fluid: IdealGas,
        T_cool0,
        p_cool_stag0,
        G_cool0,
        cool_fluid: IdealGas,
        node_num=500,
        accuracy=0.01
) -> FilmBladeCooler:
    cooler = FilmBladeCooler(sections=stage_profiler.rk_sections,
                             channel_width=channel_width,
                             wall_thickness=wall_thickness,
                             D_in=stage_profiler.D1_in,
                             D_out=stage_profiler.D1_out,
                             T_wall_out_av_init=T_wall_out_av_init,
                             lam_blade=lam_blade,
                             x_hole_rel=x_hole_rel,
                             hole_num=hole_num,
                             d_hole=d_hole,
                             phi_hole=phi_hole,
                             mu_hole=mu_hole,
                             T_gas_stag=stage_profiler.T1_w_stag,
                             p_gas_stag=stage_profiler.p1_w_stag,
                             G_gas=G_gas,
                             c_p_gas_av=stage_profiler.c_p,
                             lam_gas_in=stage_profiler.lam_w1,
                             lam_gas_out=stage_profiler.lam_w2,
                             work_fluid=work_fluid,
                             T_cool0=T_cool0,
                             p_cool_stag0=p_cool_stag0,
                             G_cool0=G_cool0,
                             cool_fluid=cool_fluid,
                             node_num=node_num,
                             accuracy=accuracy)
    return cooler