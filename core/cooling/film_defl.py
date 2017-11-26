from .tools import GasBladeHeatExchange, LocalParamCalculator, FilmCalculator
from ..profiling.section import BladeSection
from gas_turbine_cycle.gases import IdealGas
import numpy as np
import typing


class FilmSectorCooler(GasBladeHeatExchange):
    def __init__(self,
                 section: BladeSection,
                 height,
                 channel_width,
                 wall_thickness,
                 D_av,
                 lam_blade: typing.Callable[[float], float],
                 T_wall_av,
                 x_hole_rel: typing.List[float],
                 hole_num: typing.List[int],
                 d_hole: typing.List[float],
                 phi_hole: typing.List[float],
                 mu_hole: typing.List[float],
                 T_gas_stag,
                 p_gas_stag,
                 G_gas,
                 c_p_gas_av,
                 lam_gas_in,
                 lam_gas_out,
                 work_fluid: IdealGas,
                 T_cool0,
                 p_cool_stag0,
                 G_cool0,
                 c_p_cool_av,
                 cool_fluid: IdealGas,
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
        self.k_gas_av = self.work_fluid.k_func(self.c_p_gas_av)
        self.k_cool_av = self.work_fluid.k_func(self.c_p_cool_av)
        self.x_hole = self.get_x_hole(x_hole_rel, section)

        self.local_param = LocalParamCalculator(section=section,
                                                height=height,
                                                wall_thickness=wall_thickness,
                                                T_cool_fluid0=T_cool0,
                                                T_wall_av=T_wall_av,
                                                lam_blade=lam_blade,
                                                cool_fluid=cool_fluid,
                                                node_num=node_num)
        self.film = FilmCalculator(x_hole=self.x_hole,
                                   hole_num=hole_num,
                                   d_hole=d_hole,
                                   phi_hole=phi_hole,
                                   mu_hole=mu_hole,
                                   T_gas_stag=T_gas_stag,
                                   p_gas_stag=p_gas_stag,
                                   c_p_gas_av=c_p_gas_av,
                                   work_fluid=work_fluid,
                                   G_cool0=G_cool0,
                                   p_cool_stag0=p_cool_stag0,
                                   c_p_cool_av=c_p_cool_av,
                                   cool_fluid=cool_fluid,
                                   height=height)

        self.mu_gas, self.lam_gas, self.Re_gas, self.Nu_gas, self.alpha_gas_av = self.get_alpha_gas_av(self.section,
                                                                                                       self.height,
                                                                                                       self.T_gas_stag,
                                                                                                       self.G_gas,
                                                                                                       self.D_av,
                                                                                                       self.work_fluid)
        self.alpha_gas_inlet = self.get_alpha_gas_inlet(self.section, self.height, self.T_gas_stag, self.G_gas,
                                                        self.D_av, self.work_fluid)

        self.res = None
        self._T_wall_list = []
        "Список для хранения температур стенки на разных итерациях."
        self.iter_num = None

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


class FilmBladeCooler:
    def __init__(self,
                 sections: typing.List[BladeSection],
                 wall_thickness,
                 D_in,
                 D_out,
                 T_wall_out,
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
        self.wall_thickness = wall_thickness
        self.D_in = D_in
        self.D_out = D_out
        self.T_wall_out = T_wall_out
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


    def _get_D_av_arr(self):
        """Массив средних диаметров для секторов."""
        pass

    def _get_gas_param_arr(self):
        """Массивы параметров газа для секторов."""
        pass
