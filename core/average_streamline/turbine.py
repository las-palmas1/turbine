from gas_turbine_cycle.gases import KeroseneCombustionProducts, IdealGas, Air
from .stage_geom import InvalidStageSizeValue, StageGeomAndHeatDrop, \
    TurbineGeomAndHeatDropDistribution, set_logging
import logging
import numpy as np
from .stage_gas_dynamics import StageGasDynamics, get_first_stage, \
    get_intermediate_stage, get_last_pressure_stage, get_last_work_stage, get_only_pressure_stage, get_only_work_stage
from enum import Enum
from scipy.interpolate import interp1d
import os
from gas_turbine_cycle.tools.functions import eta_turb_stag_p
import pickle as pk
import xlwt as xl


# TODO: поменять названия типов турбин


class TurbineType(Enum):
    PRESSURE = 0
    WORK = 1


class TurbineInput:
    """ Содержит входные данные для расчета турбина по средней линии тока, известные из расчета цикла. """

    ext = '.avlinit'

    def __init__(self, turbine_type: TurbineType, T_g_stag, p_g_stag, G_turbine, G_fuel,
                 work_fluid: IdealGas, T_t_stag_cycle, eta_t_stag_cycle):
        self.turbine_type = turbine_type
        self.T_g_stag = T_g_stag
        self.p_g_stag = p_g_stag
        self.G_turbine = G_turbine
        self.G_fuel = G_fuel
        self.work_fluid = work_fluid
        self.T_t_stag_cycle = T_t_stag_cycle
        self.eta_t_stag_cycle = eta_t_stag_cycle

    def write_input_file(self, fname: str):
        with open(os.path.splitext(fname)[0] + self.ext, 'wb') as f:
            pk.dump(self, f)


class Turbine:
    def __init__(self, turbine_type: TurbineType, stage_number, T_g_stag, p_g_stag, G_turbine, G_fuel,
                 work_fluid: IdealGas, l1_D1_ratio, n,
                 T_t_stag_cycle, eta_t_stag_cycle,
                 alpha11, k_n=6.8, eta_m=0.99,
                 auto_set_rho: bool=True,
                 auto_compute_heat_drop: bool=True,
                 precise_heat_drop: bool=False,
                 precision=0.001,
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
        :param G_fuel: float \n
            Суммарный расход топлива по тракту перед турбиной.
        :param work_fluid: IdealGas \n
            Рабочее тело.
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
        :param precision: float, optional. \n
            Точность схождения вычислений в цикле.
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
        self._G_fuel = G_fuel
        self._kwargs = kwargs
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
                                                                                                     eta_t_stag_cycle,
                                                                                                     G_turbine,
                                                                                                     G_fuel)
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
        self.precision = precision
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
        self.pi_t = None
        self.pi_t_stag = None
        self.N = None
        self.eta_t_stag_p = None
        self.i_cool_sum = None
        self._eta_m = eta_m
        self._init_turbine_geom()
        self.log_filename = os.path.join(os.getcwd(), 'average_streamline.log')

    @classmethod
    def _get_p_t_stag_L_t_and_H_t(cls, work_fluid: IdealGas, p_g_stag, T_g_stag, T_t_stag, eta_t_stag,
                                  G_turbine, G_fuel):
        work_fluid.__init__()
        g_fuel = G_fuel / (G_turbine - G_fuel)
        alpha_air = 1 / (work_fluid.l0 * g_fuel)
        work_fluid.T1 = T_t_stag
        work_fluid.T2 = T_g_stag
        work_fluid.alpha = alpha_air
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
            if self._gamma_out is not None and self._gamma_in is not None:
                self._geom = TurbineGeomAndHeatDropDistribution(self.stage_number, self.eta_t_stag_cycle,
                                                                self.n, type(self.work_fluid)(),
                                                                self.T_g_stag, self.p_g_stag, self.G_fuel,
                                                                self.G_turbine, self.l1_D1_ratio,
                                                                self.alpha11, self.k_n,
                                                                self.T_t_stag_cycle,
                                                                precision=self.precision,
                                                                auto_compute_heat_drop=self.auto_compute_heat_drop,
                                                                c21=self.c21_init,
                                                                gamma_in=self.gamma_in,
                                                                gamma_out=self.gamma_out)
            elif self._gamma_av is not None and self._gamma_sum is not None:
                self._geom = TurbineGeomAndHeatDropDistribution(self.stage_number, self.eta_t_stag_cycle,
                                                                self.n, type(self.work_fluid)(),
                                                                self.T_g_stag, self.p_g_stag, self.G_fuel,
                                                                self.G_turbine, self.l1_D1_ratio,
                                                                self.alpha11, self.k_n,
                                                                self.T_t_stag_cycle,
                                                                precision=self.precision,
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
        self._gas_dynamics.clear()
        logging.info('\n%s РАСЧЕТ ГАЗОДИНАМИЧЕСКИХ ПАРАМЕТРОВ ТУРБИНЫ %s\n' % (30 * '*', 30 * '*'))
        if self.turbine_type == TurbineType.PRESSURE:
            for num, item in enumerate(self.geom):
                logging.info('\n%s СТУПЕНЬ %s %s\n' % (15 * '*', num + 1, 15 * '*'))
                logging.debug('%s compute_gas_dynamics num = %s' % (self.str(), num))
                if num == 0 and self.stage_number > 1:
                    # расчет первой ступени при числе ступеней, больше одной
                    stage_gas_dyn = get_first_stage(type(self.work_fluid)(), item, self.T_g_stag, self.p_g_stag,
                                                    self.G_turbine, self.G_fuel, precision=self.precision)
                    self._gas_dynamics.append(stage_gas_dyn)
                elif num == 0 and self.stage_number == 1:
                    # расчет первой ступени при числе ступеней, равному единице
                    stage_gas_dyn = get_only_pressure_stage(type(self.work_fluid)(), self.geom,
                                                            p2_stag=self.p_t_stag_cycle,
                                                            T0_stag=self.T_g_stag,
                                                            p0_stag=self.p_g_stag,
                                                            G_turbine=self.G_turbine,
                                                            G_fuel=self.G_fuel,
                                                            precision=self.precision)
                    self._gas_dynamics.append(stage_gas_dyn)
                elif num < self.stage_number - 1:
                    # расчет промежуточных ступеней
                    stage_gas_dyn = get_intermediate_stage(type(self.work_fluid)(), item, self._gas_dynamics[num - 1],
                                                           self.geom[num - 1],
                                                           self.G_fuel, precise_heat_drop=self.precise_heat_drop,
                                                           precision=self.precision)
                    self._gas_dynamics.append(stage_gas_dyn)
                elif num == self.stage_number - 1:
                    # расчет последней ступени
                    stage_gas_dyn = get_last_pressure_stage(type(self.work_fluid)(), item,
                                                            self._gas_dynamics[num - 1], self.geom[num - 1],
                                                            p2_stag=self.p_t_stag_cycle,
                                                            G_fuel=self.G_fuel,
                                                            precision=self.precision)
                    self._gas_dynamics.append(stage_gas_dyn)
        elif self.turbine_type == TurbineType.WORK:
            L_last_stage_rel = self.L_t_cycle
            for num, item in enumerate(self.geom):
                logging.info('\n%s СТУПЕНЬ %s %s\n' % (15 * '*', num + 1, 15 * '*'))
                logging.debug('%s compute_gas_dynamics num = %s' % (self.str(), num))
                if num == 0 and self.stage_number > 1:
                    # расчет первой ступени при числе ступеней, больше одной
                    stage_gas_dyn = get_first_stage(type(self.work_fluid)(), item, self.T_g_stag, self.p_g_stag,
                                                    self.G_turbine, self.G_fuel,
                                                    precision=self.precision)
                    self._gas_dynamics.append(stage_gas_dyn)
                    L_last_stage_rel -= stage_gas_dyn.L_t_prime
                elif num == 0 and self.stage_number == 1:
                    # расчет первой ступени при числе ступеней, равному единице
                    stage_gas_dyn = get_only_work_stage(type(self.work_fluid)(), item, L_last_stage_rel, 0.9,
                                                        self.T_g_stag, self.p_g_stag,
                                                        self.G_turbine, self.G_fuel,
                                                        precision=self.precision)
                    self._gas_dynamics.append(stage_gas_dyn)
                elif num < self.stage_number - 1:
                    # расчет промежуточных ступеней
                    stage_gas_dyn = get_intermediate_stage(type(self.work_fluid)(), item, self._gas_dynamics[num - 1],
                                                           self.geom[num - 1],
                                                           self.G_fuel,
                                                           precise_heat_drop=self.precise_heat_drop,
                                                           precision=self.precision)
                    self._gas_dynamics.append(stage_gas_dyn)
                    L_last_stage_rel -= stage_gas_dyn.L_t_prime
                elif num == self.stage_number - 1:
                    # расчет последней ступени
                    if L_last_stage_rel < 0:
                        raise InvalidStageSizeValue('L_last_stage_rel must not be negative')
                    stage_gas_dyn = get_last_work_stage(type(self.work_fluid)(), item, self._gas_dynamics[num - 1],
                                                        self.geom[num - 1], L_last_stage_rel, 0.9, self.G_fuel,
                                                        precision=self.precision)
                    self._gas_dynamics.append(stage_gas_dyn)
        for n, i in enumerate(self):
            self.geom[n].H0 = i.H0

    def compute_integrate_turbine_parameters(self):
        logging.info('\n%s РАСЧЕТ ИНТЕГРАЛЬНЫХ ПАРАМЕТРОВ ТУРБИНЫ %s\n' % (30 * '*', 30 * '*'))
        self.work_fluid.__init__()
        self.work_fluid.alpha = self.first.alpha_air_in
        self.L_t_sum = 0
        self.H_t_stag = 0
        self.H_t = 0
        self.i_cool_sum = 0
        for n, item in enumerate(self):
            self.L_t_sum += item.L_t_prime
            self.H_t_stag += item.H0_stag_ad_t_prime
            if n != self.stage_number - 1:
                self.H_t += item.H0_stag_ad_t_prime
                self.i_cool_sum += item.i_cool
            else:
                self.H_t += item.H0_ad_t_prime
        self.work_fluid.T1 = self.T_g_stag
        self.work_fluid.T2 = self.last.T_st
        self.work_fluid.alpha = self.last.alpha_air_in
        self.c_p_gas = self.work_fluid.c_p_av_int
        self.k_gas = self.work_fluid.k_av_int
        self.eta_t = self.L_t_sum / self.H_t
        self.eta_l = (self.L_t_sum + self.last.c2 ** 2 / 2) / self.H_t
        self.pi_t = self.first.p0_stag / self.last.p2
        self.pi_t_stag = self.first.p0_stag / self.last.p2_stag
        self.work_fluid.T2 = self.last.T_st_stag
        self.c_p_gas_stag = self.work_fluid.c_p_av_int
        self.k_gas_stag = self.work_fluid.k_av_int
        self.eta_t_stag = self.L_t_sum / self.H_t_stag
        self.eta_t_stag_p = eta_turb_stag_p(self.pi_t_stag, self.k_gas_stag, self.eta_t_stag)
        self.N = self.L_t_sum * self.G_turbine * self.eta_m

    def write_compass_parameter_file(self, fname, prefix='turb'):
        wb = xl.Workbook()
        ws = wb.add_sheet('VarTable')
        ws.write(0, 0, 'Комментарий')
        ws.write(1, 0, 'row')
        length = sum([i.length for i in self.geom]) - self.geom.last.delta_a_rk
        ws.write(0, 1, '%s_length' % prefix)
        ws.write(1, 1, length * 1e3)
        ws.write(0, 2, '%s_D_inlet_av' % prefix)
        ws.write(1, 2, self.geom.first.D0 * 1e3)
        ws.write(0, 3, '%s_l_inlet' % prefix)
        ws.write(1, 3, self.geom.first.l0 * 1e3)
        ws.write(0, 4, '%s_D_outlet_av' % prefix)
        ws.write(1, 4, self.geom.last.D2 * 1e3)
        ws.write(0, 5, '%s_l_outlet' % prefix)
        ws.write(1, 5, self.geom.last.l2 * 1e3)
        start = 5
        num = 12
        for i in range(self.stage_number):
            ws.write(0, start + i * num + 1, '%s_b_sa_%s' % (prefix, i + 1))
            ws.write(1, start + i * num + 1, self.geom[i].b_sa * 1e3)
            ws.write(0, start + i * num + 2, '%s_delta_a_sa_%s' % (prefix, i + 1))
            ws.write(1, start + i * num + 2, self.geom[i].delta_a_sa * 1e3)
            ws.write(0, start + i * num + 3, '%s_b_rk_%s' % (prefix, i + 1))
            ws.write(1, start + i * num + 3, self.geom[i].b_rk * 1e3)
            ws.write(0, start + i * num + 4, '%s_delta_a_rk_%s' % (prefix, i + 1))
            ws.write(1, start + i * num + 4, self.geom[i].delta_a_rk * 1e3)
            ws.write(0, start + i * num + 5, '%s_D0_%s' % (prefix, i + 1))
            ws.write(1, start + i * num + 5, self.geom[i].D0 * 1e3)
            ws.write(0, start + i * num + 6, '%s_D05_%s' % (prefix, i + 1))
            ws.write(1, start + i * num + 6, self.geom[i].D05 * 1e3)
            ws.write(0, start + i * num + 7, '%s_D1_%s' % (prefix, i + 1))
            ws.write(1, start + i * num + 7, self.geom[i].D1 * 1e3)
            ws.write(0, start + i * num + 8, '%s_D2_%s' % (prefix, i + 1))
            ws.write(1, start + i * num + 8, self.geom[i].D2 * 1e3)
            ws.write(0, start + i * num + 9, '%s_l0_%s' % (prefix, i + 1))
            ws.write(1, start + i * num + 9, self.geom[i].l0 * 1e3)
            ws.write(0, start + i * num + 10, '%s_l05_%s' % (prefix, i + 1))
            ws.write(1, start + i * num + 10, self.geom[i].l05 * 1e3)
            ws.write(0, start + i * num + 11, '%s_l1_%s' % (prefix, i + 1))
            ws.write(1, start + i * num + 11, self.geom[i].l1 * 1e3)
            ws.write(0, start + i * num + 12, '%s_l2_%s' % (prefix, i + 1))
            ws.write(1, start + i * num + 12, self.geom[i].l2 * 1e3)
        wb.save(os.path.splitext(fname)[0] + '.xls')

    def write_nx_parameter_file(self, fname, prefix='turb'):
        if prefix:
            prefix = prefix + '_'
        lines = []
        length = sum([i.length for i in self.geom])
        lines.append('[mm]%slength=%s\n' % (prefix, length * 1e3))
        lines.append('[mm]%sD_inlet_av=%s\n' % (prefix, self.geom.first.D0 * 1e3))
        lines.append('[mm]%sl_inlet=%s\n' % (prefix, self.geom.first.l0 * 1e3))
        lines.append('[mm]%sD_outlet_av=%s\n' % (prefix, self.geom.last.D2 * 1e3))
        lines.append('[mm]%sl_outlet=%s\n' % (prefix, self.geom.last.l2 * 1e3))
        for i in range(self.stage_number):
            lines.append('[mm]%sb_sa_%s=%s\n' % (prefix, i + 1, self.geom[i].b_sa * 1e3))
            lines.append('[mm]%sdelta_a_sa_%s=%s\n' % (prefix, i + 1, self.geom[i].delta_a_sa * 1e3))
            lines.append('[mm]%sb_rk_%s=%s\n' % (prefix, i + 1, self.geom[i].b_rk * 1e3))
            lines.append('[mm]%sdelta_a_rk_%s=%s\n' % (prefix, i + 1, self.geom[i].delta_a_rk * 1e3))
            lines.append('[mm]%sD0_%s=%s\n' % (prefix, i + 1, self.geom[i].D0 * 1e3))
            lines.append('[mm]%sD05_%s=%s\n' % (prefix, i + 1, self.geom[i].D05 * 1e3))
            lines.append('[mm]%sD1_%s=%s\n' % (prefix, i + 1, self.geom[i].D1 * 1e3))
            lines.append('[mm]%sD2_%s=%s\n' % (prefix, i + 1, self.geom[i].D2 * 1e3))
            lines.append('[mm]%sl0_%s=%s\n' % (prefix, i + 1, self.geom[i].l0 * 1e3))
            lines.append('[mm]%sl05_%s=%s\n' % (prefix, i + 1, self.geom[i].l05 * 1e3))
            lines.append('[mm]%sl1_%s=%s\n' % (prefix, i + 1, self.geom[i].l1 * 1e3))
            lines.append('[mm]%sl2_%s=%s\n' % (prefix, i + 1, self.geom[i].l2 * 1e3))
            lines.append('[mm]%sdelta_r_%s=%s\n' % (prefix, i + 1, self.geom[i].delta_r_rk * 1e3))
            lines.append('[mm]%sp_in_%s=%s\n' % (prefix, i + 1, self.geom[i].p_r_in * 1e3))
            lines.append('[mm]%sp_out_%s=%s\n' % (prefix, i + 1, self.geom[i].p_r_out * 1e3))

        with open(os.path.splitext(fname)[0] + '.exp', 'w') as f:
            f.writelines(lines)

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
                                                              self.eta_t_stag_cycle,
                                                              self.G_turbine,
                                                              self.G_fuel
                                                              )
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
                                                              self.eta_t_stag_cycle,
                                                              self.G_turbine,
                                                              self.G_fuel
                                                              )

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
    def G_fuel(self):
        """
        Суммарный расход топлива по тракту до турбины.
        """
        assert self._G_fuel is not None, 'G_fuel must not be None'
        return self._G_fuel

    @G_fuel.setter
    def G_fuel(self, value):
        self._G_fuel = value
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
                                                              self.eta_t_stag_cycle,
                                                              self.G_turbine,
                                                              self.G_fuel
                                                              )
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
                                                              self.eta_t_stag_cycle,
                                                              self.G_turbine,
                                                              self.G_fuel
                                                              )
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
    pass






