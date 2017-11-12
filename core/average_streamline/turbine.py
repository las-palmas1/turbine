from gas_turbine_cycle.gases import KeroseneCombustionProducts, IdealGas
from .stage_geom import InvalidStageSizeValue, StageGeomAndHeatDrop, \
    TurbineGeomAndHeatDropDistribution, set_logging
import logging
import numpy as np
from .stage_gas_dynamics import StageGasDynamics, get_first_stage, \
    get_intermediate_stage, get_last_pressure_stage, get_last_work_stage, get_only_pressure_stage, get_only_work_stage
from enum import Enum
from scipy.interpolate import interp1d
import os
import pickle as pk


class TurbineType(Enum):
    Power = 0
    Compressor = 1


class Turbine:
    def __init__(self, turbine_type: TurbineType, stage_number, T_g_stag, p_g_stag, G_turbine, work_fluid: IdealGas,
                 alpha_air, l1_D1_ratio, n, T_t_stag_cycle, eta_t_stag_cycle,
                 alpha11, k_n=6.8, eta_m=0.99,
                 auto_set_rho: bool=True,
                 auto_compute_heat_drop: bool=True,
                 precise_heat_drop: bool=False,
                 **kwargs):
        """
        :param turbine_type: TurbineType \n
            Тип турбины, компрессорная или силовая.
        :param stage_number: float \n
            Число ступеней.
        :param T_g_stag: float \n
            Температура торможения после КС.
        :param p_g_stag: float \n
            Давление торможения после КС.
        :param G_turbine: float \n
            Расход газа через СА первой ступени.
        :param work_fluid: IdealGas \n
            Рабочее тело.
        :param alpha_air: float \n
            Коэффициент избытка воздуха.
        :param l1_D1_ratio: float \n
            Отношение длины лопатки РК первой ступени к среднему диаметру.
        :param n: float \n
            Частота вращения ротора турбины.
        :param T_t_stag_cycle: float \n
            Температура торможения на выходе из турбины. Необходимо на этапе расчета геометрии при
            вычислении статических параметров на выходе из турбины. Их определение необходимо для расчета
            коэффициента избытка теплоты и расчета последней ступени силовой турбины.
        :param eta_t_stag_cycle: float \n
            Величина КПД турбины по параметрам торможения из расчета цикла. Необходим при вычислении коэффициента
            возврата теплоты при расчете предварительного рапределения теплоперепадов по ступеням.
        :param alpha11: float \n
            Угол потока после СА первой ступени. Необходим для вычисления размеров входного сечения на
            этапе расчета геометрии.
        :param k_n: float, optional \n
            Коэффициент в формуле для приблизительно расчета напряжения в лопатке.
        :param eta_m: float, optional \n
            Механический КПД.
        :param auto_set_rho: bool, optional \n
            Если True, то после расчета геометрии по относительному удлинению лопатки РК на выходе
            на каждой ступени вычисляется степень реактивности.
        :param auto_compute_heat_drop: bool, optional \n
            Если True, то при расчете геометрии происходит расчет преварительного распределения
            теплоперепада по ступеням.
        :param precise_heat_drop: bool, optional. \n
            Если True, то теплоперепад на ступене будет уточняться с учетом величигы
            коэффициента использования скорости.
        :param kwargs: Возможны следующие наборы ключевых слов: \n
            1.  Должны быть обязаетльно указаны gamma_av и gamma_sum или gamma_in и gamma_out. Это соответственно
                угол наклона средней линии и суммарный угол раскрытия проточной част и углы наклона образующей
                периферии и втулки. \n
            2.  Если параметр auto_set_rho равен False, то должен быть указан список степеней реактивности для
                всех ступеней турбины rho_list.
            3.  Если параметр auto_compute_heat_drop равен True, то должны быть указаны начальные приближения для
                теплоперепада на первой ступени H01_init и скорости на выходе из РК первой ступени c21_init. Если же
                он равен False, то должен быть указан список теплоперепадо для всех ступеней H0_list
        """
        set_logging()
        self._T_g_stag = T_g_stag
        self._p_g_stag = p_g_stag
        self._G_turbine = G_turbine
        self._kwargs = kwargs
        self._alpha_air = alpha_air
        self._stage_number = stage_number
        self._k_n = k_n
        self._work_fluid: IdealGas = work_fluid
        self._l1_D1_ratio = l1_D1_ratio
        self._n = n
        self._T_t_stag_cycle = T_t_stag_cycle
        self._p_t_stag_cycle, self._L_t_cycle, self._H_t_stag_cycle = self._get_p_t_stag_L_t_and_H_t(work_fluid,
                                                                                                     p_g_stag,
                                                                                                     T_g_stag,
                                                                                                     T_t_stag_cycle,
                                                                                                     eta_t_stag_cycle)
        self._alpha11 = alpha11
        self._eta_t_stag_cycle = eta_t_stag_cycle
        self._geom = None
        self.c_p_gas = None
        self.k_gas = None
        self.c_p_gas_stag = None
        self.k_gas_stag = None
        self._gas_dynamics = list()
        self._type = turbine_type
        self.precise_heat_drop = precise_heat_drop
        self.auto_set_rho = auto_set_rho
        self.auto_compute_heat_drop = auto_compute_heat_drop
        if not auto_set_rho:
            assert 'rho_list' in kwargs, 'rho_list is not specified.'
            assert len(kwargs['rho_list']) == stage_number, "Length of rho_list isn't equal to the stage number."
            self._rho_list = kwargs['rho_list']
        self._auto_set_rho = auto_set_rho
        if not auto_compute_heat_drop:
            assert 'H0_list' in kwargs, 'H0_list is not specified.'
            assert len(kwargs['H0_list']) == stage_number, "Length of H0_list isn't equal to the stage number."
            self._H0_list = kwargs['H0_list']
        if auto_compute_heat_drop:
            assert 'H01_init' in kwargs, 'H01_init is not specified.'
            assert 'c21_init' in kwargs, 'c21_init is not specified.'
            self._c21_init = kwargs['c21_init']
            self._H01_init = kwargs['H01_init']
        else:
            self._c21_init = None
            self._H01_init = None
        if ('gamma_av' in kwargs) and ('gamma_sum' in kwargs):
            self._gamma_av = kwargs['gamma_av']
            self._gamma_in = None
            self._gamma_out = None
            self._gamma_sum = kwargs['gamma_sum']
        elif ('gamma_in' in kwargs) and ('gamma_out' in kwargs):
            self._gamma_av = None
            self._gamma_in = kwargs['gamma_in']
            self._gamma_out = kwargs['gamma_out']
            self._gamma_sum = None
        else:
            assert False, 'gamma_av and gamma_sum or gamma_in and gamma_out must be set'
        self.L_t_sum: float = None
        self.H_t = None
        self.H_t_stag = None
        self.eta_t = None
        self.eta_t_stag = None
        self.eta_l = None
        self.N = None
        self._eta_m = eta_m
        self._init_turbine_geom()
        self.log_filename = os.path.join(os.getcwd(), 'average_streamline.log')

    @classmethod
    def _get_p_t_stag_L_t_and_H_t(cls, work_fluid: IdealGas, p_g_stag, T_g_stag, T_t_stag, eta_t_stag):
        work_fluid.__init__()
        work_fluid.T1 = T_t_stag
        work_fluid.T2 = T_g_stag
        L_t = work_fluid.c_p_av_int * (T_g_stag - T_t_stag)
        H_t = L_t / eta_t_stag
        pi_t = (1 - L_t / (work_fluid.c_p_av_int * T_g_stag * eta_t_stag)) ** \
               (work_fluid.k_av_int / (1 - work_fluid.k_av_int))
        p_t_stag = p_g_stag / pi_t
        return p_t_stag, L_t, H_t

    def __getitem__(self, item) -> StageGasDynamics:
        try:
            if 0 <= item < self.stage_number:
                return self._gas_dynamics[item]
            else:
                raise IndexError('invalid index')
        except IndexError:
            assert False, 'turbine stages have not computed yet'

    def __len__(self):
        return self.stage_number

    def __iter__(self):
        self._num = 0
        return self

    def __next__(self):
        try:
            if self._num < self.stage_number:
                current = self._gas_dynamics[self._num]
                self._num += 1
                return current
            else:
                raise StopIteration()
        except IndexError:
            assert False, 'turbine stages have not computed yet'

    def str(self):
        str_arr = str(self).split()
        return str_arr[0][1:] + ' ' + str_arr[1]

    def _init_turbine_geom(self):
        try:
            if self._gamma_out and self._gamma_in:
                self._geom = TurbineGeomAndHeatDropDistribution(self.stage_number, self.eta_t_stag_cycle,
                                                                self.n, KeroseneCombustionProducts(),
                                                                self.T_g_stag, self.p_g_stag, self.alpha_air,
                                                                self.G_turbine, self.l1_D1_ratio,
                                                                self.alpha11, self.k_n,
                                                                self.T_t_stag_cycle,
                                                                auto_compute_heat_drop=self.auto_compute_heat_drop,
                                                                c21=self.c21_init,
                                                                gamma_in=self.gamma_in,
                                                                gamma_out=self.gamma_out)
            elif self._gamma_av and self._gamma_sum:
                self._geom = TurbineGeomAndHeatDropDistribution(self.stage_number, self.eta_t_stag_cycle,
                                                                self.n, KeroseneCombustionProducts(),
                                                                self.T_g_stag, self.p_g_stag, self.alpha_air,
                                                                self.G_turbine, self.l1_D1_ratio,
                                                                self.alpha11, self.k_n,
                                                                self.T_t_stag_cycle,
                                                                auto_compute_heat_drop=self.auto_compute_heat_drop,
                                                                c21=self.c21_init,
                                                                gamma_av=self.gamma_av,
                                                                gamma_sum=self.gamma_sum)
            if not self.auto_set_rho:
                for geom, rho in zip(self.geom, self._rho_list):
                    geom.rho = rho
            if not self.auto_compute_heat_drop:
                for geom, H0 in zip(self.geom, self._H0_list):
                    geom.H0 = H0
            if self.auto_compute_heat_drop:
                self.geom[0].H0 = self.H01_init
        except AssertionError:
            pass

    def compute_geometry(self):
        logging.info('%s РАСЧЕТ ГЕОМЕТРИИ ТУРБИНЫ %s\n' % (30 * '*', 30 * '*'))
        if self.auto_set_rho:
            self.geom[0].rho = self._rho_func(self.l1_D1_ratio)
        self.geom.compute()
        if self.auto_set_rho:
            self._set_rho()
        self._gamma_in = self.geom.gamma_in
        self._gamma_out = self.geom.gamma_out
        self._gamma_av = self.geom.gamma_av
        self._gamma_sum = self.geom.gamma_sum

    def compute_stages_gas_dynamics(self):
        """
        Вычисление газодинамических параметров на ступенях турбины
        """
        logging.info('\n%s РАСЧЕТ ГАЗОДИНАМИЧЕСКИХ ПАРАМЕТРОВ ТУРБИНЫ %s\n' % (30 * '*', 30 * '*'))
        if self.turbine_type == TurbineType.Power:
            for num, item in enumerate(self.geom):
                logging.info('\n%s СТУПЕНЬ %s %s\n' % (15 * '*', num + 1, 15 * '*'))
                logging.debug('%s compute_gas_dynamics num = %s' % (self.str(), num))
                if num == 0 and self.stage_number > 1:
                    # расчет первой ступени при числе ступеней, больше одной
                    stage_gas_dyn = get_first_stage(item, self.T_g_stag, self.p_g_stag, self.G_turbine,
                                                    self.alpha_air)
                    self._gas_dynamics.append(stage_gas_dyn)
                elif num == 0 and self.stage_number == 1:
                    # расчет первой ступени при числе ступеней, равному единице
                    stage_gas_dyn = get_only_pressure_stage(self.geom, self.T_g_stag, self.p_g_stag, self.G_turbine,
                                                            self.alpha_air)
                    self._gas_dynamics.append(stage_gas_dyn)
                elif num < self.stage_number - 1:
                    # расчет промежуточных ступеней
                    stage_gas_dyn = get_intermediate_stage(item, self._gas_dynamics[num - 1], self.geom[num - 1],
                                                           precise_heat_drop=self.precise_heat_drop)
                    self._gas_dynamics.append(stage_gas_dyn)
                elif num == self.stage_number - 1:
                    # расчет последней ступени
                    stage_gas_dyn = get_last_pressure_stage(self.geom, item,
                                                            self._gas_dynamics[num - 1], self.geom[num - 1])
                    self._gas_dynamics.append(stage_gas_dyn)
        elif self.turbine_type == TurbineType.Compressor:
            L_last_stage_rel = self.L_t_cycle
            for num, item in enumerate(self.geom):
                logging.info('\n%s СТУПЕНЬ %s %s\n' % (15 * '*', num + 1, 15 * '*'))
                logging.debug('%s compute_gas_dynamics num = %s' % (self.str(), num))
                if num == 0 and self.stage_number > 1:
                    # расчет первой ступени при числе ступеней, больше одной
                    stage_gas_dyn = get_first_stage(item, self.T_g_stag, self.p_g_stag, self.G_turbine,
                                                    self.alpha_air)
                    self._gas_dynamics.append(stage_gas_dyn)
                    L_last_stage_rel -= stage_gas_dyn.L_t_rel
                elif num == 0 and self.stage_number == 1:
                    # расчет первой ступени при числе ступеней, равному единице
                    stage_gas_dyn = get_only_work_stage(item, L_last_stage_rel, 0.9, self.T_g_stag, self.p_g_stag,
                                                        self.G_turbine, self.alpha_air)
                    self._gas_dynamics.append(stage_gas_dyn)
                elif num < self.stage_number - 1:
                    # расчет промежуточных ступеней
                    stage_gas_dyn = get_intermediate_stage(item, self._gas_dynamics[num - 1], self.geom[num - 1],
                                                           precise_heat_drop=self.precise_heat_drop)
                    self._gas_dynamics.append(stage_gas_dyn)
                    L_last_stage_rel -= stage_gas_dyn.L_t_rel
                elif num == self.stage_number - 1:
                    # расчет последней ступени
                    if L_last_stage_rel < 0:
                        raise InvalidStageSizeValue('L_last_stage_rel must not be negative')
                    stage_gas_dyn = get_last_work_stage(item, self._gas_dynamics[num - 1], self.geom[num - 1],
                                                        L_last_stage_rel, 0.9)
                    self._gas_dynamics.append(stage_gas_dyn)
        for n, i in enumerate(self):
            self.geom[n].H0 = i.H0

    def compute_integrate_turbine_parameters(self):
        logging.info('\n%s РАСЧЕТ ИНТЕГРАЛЬНЫХ ПАРАМЕТРОВ ТУРБИНЫ %s\n' % (30 * '*', 30 * '*'))
        self.work_fluid.__init__()
        self.L_t_sum = 0
        for item in self:
            self.L_t_sum += item.L_t_rel
        self.work_fluid.T1 = self.T_g_stag
        self.work_fluid.T2 = self.last.T_st
        self.c_p_gas = self.work_fluid.c_p_av_int
        self.k_gas = self.work_fluid.k_av_int
        self.H_t = self.work_fluid.c_p_av_int * self.T_g_stag * \
                   (1 - (self.p_g_stag / self.last.p2) ** ((1 - self.work_fluid.k_av_int) / self.work_fluid.k_av_int))
        self.eta_t = self.L_t_sum / self.H_t
        self.eta_l = (self.L_t_sum + self.last.c2 ** 2 / 2) / self.H_t
        self.work_fluid.T2 = self.last.T_st_stag
        self.c_p_gas_stag = self.work_fluid.c_p_av_int
        self.k_gas_stag = self.work_fluid.k_av_int
        self.H_t_stag = self.work_fluid.c_p_av_int * self.T_g_stag * (1 - (self.p_g_stag / self.last.p2_stag) **
                                                                      ((1 - self.work_fluid.k_av_int) /
                                                                       self.work_fluid.k_av_int))
        self.eta_t_stag = self.L_t_sum / self.H_t_stag
        self.N = self.L_t_sum * self.G_turbine * self.eta_m

    def save(self, filename='average_streamline_calculation_results'):
        file = open(os.path.join(os.path.dirname(__file__), filename), 'wb')
        pk.dump(self, file)
        file.close()

    @property
    def first(self) -> StageGasDynamics:
        try:
            return self._gas_dynamics[0]
        except IndexError:
            assert False, 'turbine stages have not computed yet'

    @property
    def last(self) -> StageGasDynamics:
        try:
            return self._gas_dynamics[self.stage_number - 1]
        except IndexError:
            assert False, 'turbine stages have not computed yet'

    @classmethod
    def _rho_func(cls, l2_d2_ratio):
        x = np.array([0.7, 1 / 2, 1 / 3, 1 / 4, 1 / 6, 1 / 9])
        y = np.array([0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        rho_interp = interp1d(x, y)
        return float(rho_interp(l2_d2_ratio))

    def _set_rho(self):
        """
        Задает степени реактивности на всех ступенях в зависимости от отношения
        длины лопатки РК к диаметру на выходе из РК
        """
        for i in self.geom:
            i.rho = self._rho_func(i.l1 / i.D1)

    def set_l_b_ratio(self, x0, delta, sa_rk_ratio):
        """
        :param x0: значение относительного удлинения РК первой ступени
        :param delta: изменение относительного удлинения РК от ступени к ступени
        :param sa_rk_ratio: отношение относительных удлинений СА и РК
        :return: None
        Задает относительные удлинения РК и СА на всех ступенях
        """
        def l_b_ratio(n):
            return x0 + delta * n
        for num, item in enumerate(self.geom):
            item.l2_b_rk_ratio = l_b_ratio(num)
            item.l1_b_sa_ratio = l_b_ratio(num) * sa_rk_ratio

    def set_delta_a_b_ratio(self, x0, delta):
        """
        :param x0: значение относительных зазоров на первой ступени
        :param delta: изменение значения относительных зазоров от ступени к ступени
        :return: None
        Задает значения относительных зазоров на всех ступенях
        """
        def delta_a_b_ratio(n):
            return x0 + delta * n
        for num, item in enumerate(self.geom):
            item.delta_a_b_rk_ratio = delta_a_b_ratio(num)
            item.delta_a_b_sa_ratio = delta_a_b_ratio(num)

    def set_g_loss(self, g_lk, g_ld, g_lb):
        """
        Задает значения относительных расходов утечек на всех ступенях
        """
        for num, item in enumerate(self.geom):
            if num == 0:
                item.g_lk = g_lk
                item.g_ld = 0
                item.g_lb = g_lb
            else:
                item.g_lk = g_lk
                item.g_ld = g_ld
                item.g_lb = g_lb

    @property
    def turbine_type(self) -> TurbineType:
        assert self._type is not None, 'turbine_type must not be None'
        return self._type

    @turbine_type.setter
    def turbine_type(self, value: TurbineType):
        self._type = value

    @property
    def eta_m(self):
        """
        Механический КПД.
        """
        assert self._eta_m is not None, 'eta_m must not be None'
        return self._eta_m

    @eta_m.setter
    def eta_m(self, value):
        self._eta_m = value

    @property
    def L_t_cycle(self):
        """
        Величниа работы турбины из расчета цикла. Необходима для расчета параметров последней ступени
        при расчете турбины компрессора.
        """
        assert self._L_t_cycle is not None, 'L_t_cycle must not be None'
        return self._L_t_cycle

    @property
    def geom(self) -> TurbineGeomAndHeatDropDistribution:
        assert self._geom is not None, 'geom must not be None'
        return self._geom

    @property
    def H_t_stag_cycle(self):
        """
        Величина теплоперепада в турбине по параметрам торможения. Необходим при вычислении коэффициента
        возврата теплоты при расчете предварительного рапределения теплоперепадов по ступеням.
        """
        assert self._H_t_stag_cycle is not None, 'H_t_stag_cycle must not be None'
        return self._H_t_stag_cycle

    @property
    def eta_t_stag_cycle(self):
        """
        Величина КПД турбины по параметрам торможения из расчета цикла. Необходим при вычислении коэффициента
        возврата теплоты при расчете предварительного рапределения теплоперепадов по ступеням.
        """
        assert self._eta_t_stag_cycle is not None, 'eta_t_stag_cycle must not be None'
        return self._eta_t_stag_cycle

    @eta_t_stag_cycle.setter
    def eta_t_stag_cycle(self, value):
        self._eta_t_stag_cycle = value
        self._p_t_stag_cycle, self._L_t_cycle, \
        self._H_t_stag_cycle = self._get_p_t_stag_L_t_and_H_t(self.work_fluid,
                                                              self.p_g_stag,
                                                              self.T_g_stag,
                                                              self.T_t_stag_cycle,
                                                              self.eta_t_stag_cycle)
        self._init_turbine_geom()

    @property
    def alpha11(self):
        """
        Угол потока после СА первой ступени. Необходим для вычисления размеров входного сечения на
        этапе расчета геометрии
        """
        assert self._alpha11 is not None, 'alpha11 must not be None'
        return self._alpha11

    @alpha11.setter
    def alpha11(self, value):
        self._alpha11 = value
        self._init_turbine_geom()

    @property
    def p_t_stag_cycle(self):
        """
        Давление торможения на выходе из турбины. Необходимо на этапе расчета геометрии при
        вычислении статических параметров на выходе из турбины. Их определение необходимо для расчета
        коэффициента избытка теплоты и расчета последней ступени силовой турбины.
        """
        assert self._p_t_stag_cycle is not None, 'p_t_stag_cycle must not be None'
        return self._p_t_stag_cycle

    @property
    def T_t_stag_cycle(self):
        """
        Температура торможения на выходе из турбины. Необходимо на этапе расчета геометрии при
        вычислении статических параметров на выходе из турбины. Их определение необходимо для расчета
        коэффициента избытка теплоты и расчета последней ступени силовой турбины.
        """
        assert self._T_t_stag_cycle is not None, 'T_t_stag_cycle must not be None'
        return self._T_t_stag_cycle

    @T_t_stag_cycle.setter
    def T_t_stag_cycle(self, value):
        self._T_t_stag_cycle = value
        self._p_t_stag_cycle, self._L_t_cycle, \
        self._H_t_stag_cycle = self._get_p_t_stag_L_t_and_H_t(self.work_fluid,
                                                              self.p_g_stag,
                                                              self.T_g_stag,
                                                              self.T_t_stag_cycle,
                                                              self.eta_t_stag_cycle)
        self._init_turbine_geom()

    @property
    def H01_init(self):
        """
        Начальное приближение для адиабатического теплоперепада на первой ступени. Необходимо
        на этапе расчета геометрии при вычисления кольцевой площади на входе в рк первой ступени.
        """
        if self.auto_compute_heat_drop:
            assert self._H01_init is not None, 'H01_init must not be None'
        return self._H01_init

    @H01_init.setter
    def H01_init(self, value):
        self._H01_init = value
        self._init_turbine_geom()

    @property
    def c21_init(self):
        """
        Начальное приближение для скорости на выходе их РК первой ступени. Необходимо на этапе расчета геометрии
        при вычислении рапределения теплоперепадов по ступеням.
        """
        if self.auto_compute_heat_drop:
            assert self._c21_init is not None, 'c21_init must not be None'
        return self._c21_init

    @c21_init.setter
    def c21_init(self, value):
        self._c21_init = value
        self._init_turbine_geom()

    @property
    def n(self):
        """
        Частота вращения.
        """
        assert self._n is not None, 'n must not be None'
        return self._n

    @n.setter
    def n(self, value):
        self._n = value
        self._init_turbine_geom()

    @property
    def l1_D1_ratio(self):
        """
        Отношение длины лопатки РК первой ступени к среднему диаметру
        """
        assert self._l1_D1_ratio is not None, 'l1_D1_ratio must not be None'
        return self._l1_D1_ratio

    @l1_D1_ratio.setter
    def l1_D1_ratio(self, value):
        self._l1_D1_ratio = value
        self._init_turbine_geom()

    @property
    def work_fluid(self) -> IdealGas:
        assert self._work_fluid is not None, 'work_fluid must not be None'
        return self._work_fluid

    @property
    def k_n(self):
        """
        Коэффициент в формуле для приблизительно расчета напряжения в лопатке.
        """
        assert self._k_n is not None, 'k_n must not be None'
        return self._k_n

    @k_n.setter
    def k_n(self, value):
        self._k_n = value
        self._init_turbine_geom()

    @property
    def stage_number(self):
        """
        Число ступеней турбины.
        """
        assert self._stage_number is not None, 'stage_number must not be None'
        return self._stage_number

    @stage_number.setter
    def stage_number(self, value):
        self._stage_number = value
        self._init_turbine_geom()

    @property
    def alpha_air(self):
        """
        Коэффициент избытка воздуха
        """
        assert self._alpha_air is not None, 'alpha_air must not be None'
        return self._alpha_air

    @alpha_air.setter
    def alpha_air(self, value):
        self._alpha_air = value
        self._init_turbine_geom()

    @property
    def G_turbine(self):
        """
        Расход газа через СА первой ступени.
        """
        assert self._G_turbine is not None, 'G_turbine must not be None'
        return self._G_turbine

    @G_turbine.setter
    def G_turbine(self, value):
        self._G_turbine = value
        self._init_turbine_geom()

    @property
    def p_g_stag(self):
        """Давление торможения после КС."""
        assert self._p_g_stag is not None, 'p_g_stag must not be None'
        return self._p_g_stag

    @p_g_stag.setter
    def p_g_stag(self, value):
        self._p_g_stag = value
        self._p_t_stag_cycle, self._L_t_cycle, \
        self._H_t_stag_cycle = self._get_p_t_stag_L_t_and_H_t(self.work_fluid,
                                                              self.p_g_stag,
                                                              self.T_g_stag,
                                                              self.T_t_stag_cycle,
                                                              self.eta_t_stag_cycle)
        self._init_turbine_geom()

    @property
    def T_g_stag(self):
        """Температура торможения после КС."""
        assert self._T_g_stag is not None, 'T_g_stag must not be None'
        return self._T_g_stag

    @T_g_stag.setter
    def T_g_stag(self, value):
        self._T_g_stag = value
        self._p_t_stag_cycle, self._L_t_cycle, \
        self._H_t_stag_cycle = self._get_p_t_stag_L_t_and_H_t(self.work_fluid,
                                                              self.p_g_stag,
                                                              self.T_g_stag,
                                                              self.T_t_stag_cycle,
                                                              self.eta_t_stag_cycle)
        self._init_turbine_geom()

    @property
    def gamma_sum(self):
        """Суммарный угол раскрытия проточной части"""
        if ('gamma_av' in self._kwargs) and ('gamma_sum' in self._kwargs):
            assert self._gamma_sum is not None, 'gamma_sum must not be None'
        return self._gamma_sum

    @gamma_sum.setter
    def gamma_sum(self, value):
        self._gamma_sum = value
        if ('gamma_av' in self._kwargs) and ('gamma_sum' in self._kwargs):
            self._init_turbine_geom()

    @property
    def gamma_out(self):
        """Угол наклона периферии."""
        if 'gamma_out' in self._kwargs:
            assert self._gamma_out is not None, 'gamma_out must not be None'
        return self._gamma_out

    @gamma_out.setter
    def gamma_out(self, value):
        self._gamma_out = value
        if 'gamma_in' in self._kwargs and ('gamma_out' in self._kwargs):
            self._init_turbine_geom()

    @property
    def gamma_in(self):
        """Угол наклона внутренней поверхности проточной части."""
        if 'gamma_in' in self._kwargs and ('gamma_out' in self._kwargs):
            assert self._gamma_in is not None, 'gamma_in must not be None'
        return self._gamma_in

    @gamma_in.setter
    def gamma_in(self, value):
        self._gamma_in = value
        if 'gamma_in' in self._kwargs and ('gamma_out' in self._kwargs):
            self._init_turbine_geom()

    @property
    def gamma_av(self):
        """Угол наклона средней линии тока."""
        if 'gamma_av' in self._kwargs and ('gamma_sum' in self._kwargs):
            assert self._gamma_av is not None, 'gamma_av must not be None'
        return self._gamma_av

    @gamma_av.setter
    def gamma_av(self, value):
        self._gamma_av = value
        if ('gamma_av' in self._kwargs) and ('gamma_sum' in self._kwargs):
            self._init_turbine_geom()


if __name__ == '__main__':
    # TODO: выяснить, как считается коэффициент использования скорости
    # TODO: сделать уточнение коэффициентов скорости
    deg = np.pi / 180
    turbine = Turbine(TurbineType.Compressor,
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
                      auto_set_rho=True,
                      precise_heat_drop=False,
                      H01_init=120e3,
                      c21_init=250,
                      alpha11=17*deg,
                      gamma_av=4 * deg,
                      gamma_sum=10 * deg)

    turbine.compute_geometry()
    turbine.compute_stages_gas_dynamics()
    turbine.geom.plot_geometry(figsize=(5, 5))
    turbine.compute_integrate_turbine_parameters()
    # turbine.geom.plot_heat_drop_distribution()
    # for num, i in enumerate(turbine):
    #     i.plot_velocity_triangle('Stage %s' % (num + 1))
    print('l1/b_sa = %.4f' % turbine.geom[0].l1_b_sa_ratio)
    print('l2/b_rk = %.4f' % turbine.geom[0].l2_b_rk_ratio)
    print('delta_a_sa / b_sa = %.4f' % turbine.geom[0].delta_a_b_sa_ratio)
    print('delta_a_rk / b_rk = %.4f' % turbine.geom[0].delta_a_b_rk_ratio)
    print('pho = %.3f' % turbine.geom[0].rho)
    print('H0 = %.3f' % turbine.geom[0].H0)
    print('L_t = %.3f' % turbine.L_t_sum)
    print('eta_t = %.3f' % turbine.eta_t)






