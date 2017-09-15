from core.average_streamline.stage_geom import InvalidStageSizeValue, StageGeomAndHeatDrop, \
    TurbineGeomAndHeatDropDistribution
import core.functions as func
import logging
from core.gas import KeroseneCombustionProducts
import numpy as np
import matplotlib.pyplot as plt
import os

log_filename = os.path.join(os.path.dirname(__file__), 'average_streamline_calculation.log')
logger = func.create_logger(__name__, logging.INFO, add_console_handler=True, add_file_handler=False)


class StageGasDynamics:
    def __init__(self, T0_stag, p0_stag, G_stage_in, G_turbine, alpha_air, work_fluid: KeroseneCombustionProducts, rho,
                 phi, psi, l1, l2, D1, D2, delta_r_rk, n, epsilon, g_lk, g_ld, g_lb, **kwargs):
        """

        :param T0_stag:
        :param p0_stag:
        :param G_stage_in: расход газа на входе в ступень
        :param G_turbine: расход газа через СА первой ступени
        :param alpha_air:
        :param work_fluid:
        :param rho:
        :param phi:
        :param psi:
        :param l1:
        :param l2:
        :param D1:
        :param D2:
        :param delta_r_rk:
        :param n:
        :param epsilon: Степень парциальности. Необходима для вычисления затрат на трение и вентиляцию
        :param g_lk: относительный расход утечек в концевых лабиринтах
        :param g_ld: относительный расход перетечек в лабиринтных уплотнениях сопловых диафрагм
        :param g_lb: относительный расход перетечек поверх бондажа рабочих лопаток
        :param kwargs: H0, p2, L_t, eta_t0
        """
        self.T0_stag = T0_stag
        self.p0_stag = p0_stag
        self.G_stage_in = G_stage_in
        self.G_turbine = G_turbine
        self.work_fluid = work_fluid
        self.alpha_air = alpha_air
        self.phi = phi
        self.psi = psi
        self.l1 = l1
        self.l2 = l2
        self.D1 = D1
        self.D2 = D2
        self.rho = rho
        self.n = n
        self.delta_r_rk = delta_r_rk
        self.epsilon = epsilon
        self.g_lk = g_lk
        self.g_ld = g_ld
        self.g_lb = g_lb
        self._kwargs = kwargs
        if 'H0' in kwargs:
            self.H0 = kwargs['H0']
            self.p2 = None
            self.L_t = None
            self._eta_t0 = None
        elif 'p2' in kwargs:
            self.H0 = None
            self.p2 = kwargs['p2']
            self.L_t = None
            self._eta_t0 = None
        elif ('L_t' in kwargs) and ('eta_t0' in kwargs):
            self.L_t = kwargs['L_t']
            self._eta_t0 = kwargs['eta_t0']
            self.H0 = None
            self.p2 = None
        else:
            assert False, 'H0 or p2 or (L_t and eta_t0) must be set'

    def str(self):
        str_arr = str(self).split()
        return str_arr[0][1:] + ' ' + str_arr[1]

    def compute(self):
        if 'H0' in self._kwargs:
            logger.info('%s РАСЧЕТ ГАЗОДИНАМИЧЕСКИХ ПАРАМЕТРОВ СТУПЕНИ ПО ЗАДАННОМУ ТЕПЛОПЕРЕПАДУ %s\n' %
                        (25 * '#', 25 * '#'))
            self._specified_heat_drop_calculation()
        elif 'p2' in self._kwargs:
            logger.info('%s РАСЧЕТ ГАЗОДИНАМИЧЕСКИХ ПАРАМЕТРОВ СТУПЕНИ ПО ДАВЛЕНИЮ НА ВЫХОДЕ %s\n' %
                        (25 * '#', 25 * '#'))
            self._specified_outlet_pressure_calculation()
        elif 'L_t' in self._kwargs and 'eta_t0' in self._kwargs:
            logger.info('%s РАСЧЕТ ГАЗОДИНАМИЧЕСКИХ ПАРАМЕТРОВ СТУПЕНИ ПО РАБОТЕ %s\n' %
                        (25 * '#', 25 * '#'))
            self._specified_work_calculation()

    def _specified_heat_drop_calculation(self):
        """Вычисляет параметры ступени по заданному теплоперепаду с уточнением k"""
        self.work_fluid.__init__()
        self.work_fluid.T1 = self.T0_stag
        self.work_fluid.alpha = self.alpha_air
        self.dk_rel = 1.
        self._iter_number_k_gas = 0
        logger.info('%s Расчет параметров ступени с уточнение k %s\n' % (15 * '-', 15 * '-'))
        while self.dk_rel >= 0.001:
            self._iter_number_k_gas += 1
            logger.info('ИТЕРАЦИЯ %s' % self._iter_number_k_gas)
            logger.debug('%s _specified_heat_drop_calculation _iter_number_k_gas = %s' %
                         (self.str(), self._iter_number_k_gas))
            self._compute_stage_parameters()
            logger.info('Residual = %.4f' % self.dk_rel)

    def _specified_outlet_pressure_calculation(self):
        """Вычисляет параметры ступени по заданному выходному давлению"""
        self.work_fluid.__init__()
        self.work_fluid.T1 = self.T0_stag
        self.work_fluid.alpha = self.alpha_air
        self.dk_rel = 1.
        self._iter_number_k_gas = 0
        logger.info('%s Расчет параметров ступени с уточнение k %s\n' % (15 * '-', 15 * '-'))
        while self.dk_rel >= 0.001:
            self._iter_number_k_gas += 1
            logger.info('ИТЕРАЦИЯ %s' % self._iter_number_k_gas)
            logger.debug('%s _specified_outlet_pressure_calculation _iter_number_k_gas = %s' %
                         (self.str(), self._iter_number_k_gas))
            self.H0 = self.work_fluid.c_p_av_int * \
                        self.T0_stag * (1 - (self.p0_stag / self.p2) **
                                       ((1 - self.work_fluid.k_av_int) / self.work_fluid.k_av_int))
            logger.debug('%s _specified_outlet_pressure_calculation H0 = %s' % (self.str(), self.H0))
            self._compute_stage_parameters()
            logger.info('Residual(k) = %.4f' % self.dk_rel)

    def _specified_work_calculation(self):
        """Вычисляет параметры ступени по заданной работе"""
        self.d_eta_t_rel = 1
        eta_t = self._eta_t0
        self._iter_number_eta_t = 0
        logger.info('%s Расчет параметров ступени с уточнение КПД по статическим параметрам %s\n' %
                    (15 * '-', 15 * '-'))
        while self.d_eta_t_rel >= 0.001:
            self._iter_number_eta_t += 1
            logger.info('%s ИТЕРАЦИЯ %s %s\n' % ('-' * 10, self._iter_number_eta_t, '-' * 10))
            logger.debug('%s _specified_work_calculation _iter_number_eta_t = %s' %
                         (self.str(), self._iter_number_eta_t))
            self.H0 = self.L_t / eta_t
            self._specified_heat_drop_calculation()
            self.d_eta_t_rel = abs(self.eta_t - eta_t) / eta_t
            eta_t = self.eta_t
            logger.info('')
            logger.info('Residual(eta_t) = %.4f\n' % self.d_eta_t_rel)

    def _compute_stage_parameters(self):
        """Вычисляет параметры ступени по известному теплоперепаду без уточнения k"""
        self.u1 = np.pi * self.D1 * self.n / 60
        self.H_s = self.H0 * (1 - self.rho)
        self.c1 = self.phi * np.sqrt(2 * self.H_s)
        self.c_p_gas = self.work_fluid.c_p_av_int
        self.T1 = self.T0_stag - self.H_s * self.phi ** 2 / self.c_p_gas
        self.T1_ad = self.T0_stag - self.H_s / self.c_p_gas
        self.k_gas = self.work_fluid.k_av_int
        logger.debug('%s _compute_stage_parameters k = %s' % (self.str(), self.k_gas))
        self.p1 = self.p0_stag * (self.T1_ad / self.T0_stag) ** (self.k_gas / (self.k_gas - 1))
        self.A1_a = np.pi * self.D1 * self.l1
        self.rho1 = self.p1 / (self.work_fluid.R * self.T1)
        self.G_sa = self.G_stage_in - self.g_ld * self.G_turbine
        self.c1_a = self.G_sa / (self.rho1 * self.A1_a)
        if self.c1_a > self.c1:
            raise InvalidStageSizeValue('c1_a must be less than c1')
        self.alpha1 = np.arcsin(self.c1_a / self.c1)
        self.c1_u = self.c1 * np.cos(self.alpha1)
        self.w1 = np.sqrt(self.c1**2 + self.u1**2 - 2 * self.c1 * self.u1 * np.cos(self.alpha1))
        if self.c1 * np.cos(self.alpha1) - self.u1 >= 0:
            self.beta1 = np.arctan(self.c1_a / (self.c1 * np.cos(self.alpha1) - self.u1))
        else:
            self.beta1 = np.pi + np.arctan(self.c1_a / (self.c1 * np.cos(self.alpha1) - self.u1))
        self.H_l = self.rho * self.H0 * self.T1 / self.T1_ad
        self.u2 = np.pi * self.D2 * self.n / 60
        self.w2 = self.psi * np.sqrt(self.w1**2 + 2 * self.H_l + self.u2**2 - self.u1**2)
        self.T2 = self.T1 - (self.w2**2 - self.w1**2 - self.u2**2 + self.u1**2) / (2 * self.c_p_gas)
        self.T2_ad = self.T1 - self.H_l / self.c_p_gas
        if ('H0' in self._kwargs) or ('L_t' in self._kwargs and 'eta_t0' in self._kwargs):
            self.p2 = self.p1 * (self.T2_ad / self.T1) ** (self.k_gas / (self.k_gas - 1))
        elif 'p2' in self._kwargs:
            self.p2_check = self.p1 * (self.T2_ad / self.T1) ** (self.k_gas / (self.k_gas - 1))
        self.rho2 = self.p2 / (self.work_fluid.R * self.T2)
        self.A2_a = np.pi * self.D2 * self.l2
        self.G_rk = self.G_stage_in - self.G_turbine * (self.g_lb + self.g_lk)
        self.c2_a = self.G_rk / (self.A2_a * self.rho2)
        if self.c2_a / self.w2 >= 1:
            raise InvalidStageSizeValue('c2_a must be less than w2')
        self.beta2 = np.arcsin(self.c2_a / self.w2)
        self.c2_u = self.w2 * np.cos(self.beta2) - self.u2
        if self.c2_u >= 0:
            self.alpha2 = np.arctan(self.c2_a / self.c2_u)
        else:
            self.alpha2 = np.pi + np.arctan(self.c2_a / self.c2_u)
        self.c2 = np.sqrt(self.c2_a**2 + self.c2_u**2)
        self.L_u = self.c1_u * self.u1 + self.c2_u * self.u2
        self.eta_u = self.L_u / self.H0
        self.h_s = (1 / self.phi ** 2 - 1) * self.c1 ** 2 / 2
        self.h_s_touch = self.h_s * self.T2_ad / self.T1
        self.zeta_s = self.h_s / self.H0
        self.zeta_s_touch = self.h_s_touch / self.H0
        self.h_l = (1 / self.psi ** 2 - 1) * self.w2 ** 2 / 2
        self.zeta_l = self.h_l / self.H0
        self.h_v = self.c2 ** 2 / 2
        self.zeta_v = self.h_v / self.H0
        self.eta_u_check = 1 - self.zeta_s_touch - self.zeta_l - self.zeta_v
        self.D_av = 0.5 * (self.D1 + self.D2)
        self.h_z = 1.37 * (1 + 1.6 * self.rho) * (1 + self.l2 / self.D_av) * self.delta_r_rk / self.l2 * self.L_u
        self.zeta_z = self.h_z / self.H0
        self.L_uz = self.L_u - self.h_z
        self.eta_t_touch = self.eta_u - self.zeta_z
        self.eta_l_touch = self.eta_t_touch + self.zeta_v
        self.l_av = 0.5 * (self.l1 + self.l2)
        self.u_av = 0.5 * (self.u1 + self.u2)
        self.N_tv = (1.07 * self.D_av**2 + 61 * (1 - self.epsilon) * self.D_av * self.l_av) * \
                    (self.u_av / 100)**3 * self.rho
        self.h_tv = self.N_tv / self.G_stage_in
        self.zeta_tv = self.h_tv / self.H0
        self.eta_t = self.eta_t_touch - self.zeta_tv
        self.eta_l = self.eta_l_touch - self.zeta_tv
        if ('H0' in self._kwargs) or ('p2' in self._kwargs):
            self.L_t = self.H0 * self.eta_t
            logger.debug('%s _compute_stage_parameters L_t = %s' % (self.str(), self.L_t))
        self.L_t_rel = self.L_t * self.G_stage_in / self.G_turbine - self.L_t * (self.g_ld + self.g_lk + self.g_lb)
        "Удельная работа ступени, отнесенная к расходу через СА первой ступени с учетом потерь из-за утечек"
        self.T_st = self.T2 + self. h_z / self.c_p_gas + self.h_tv / self.c_p_gas
        self.T_st_stag = self.T_st + self.h_v / self.c_p_gas
        self.work_fluid.T2 = self.T_st
        try:
            self.dk_rel = abs(self.k_gas - self.work_fluid.k_av_int) / self.k_gas
        except TypeError:
            self.dk_rel = max(abs(self.k_gas - self.work_fluid.k_av_int) / self.k_gas)
        logger.debug('%s _compute_stage_parameters dk_rel = %s' % (self.str(), self.dk_rel))
        self.p2_stag = self.p2 * (self.T_st_stag / self.T_st) ** (self.k_gas / (self.k_gas - 1))
        self.H0_stag = self.c_p_gas * self.T0_stag * (1 - (self.p2_stag / self.p0_stag) ** ((self.k_gas - 1) / self.k_gas))
        self.eta_t_stag = self.L_t / self.H0_stag
        self.G_stage_out = self.G_stage_in - self.G_turbine * self.g_lk

    def plot_velocity_triangle(self, title='', figsize=(8, 8)):
        x_in = np.array([0, -self.c1_u, -self.c1_u + self.u1, 0])
        y_in = np.array([self.c1_a, 0, 0, self.c1_a])
        x_out = np.array([0, self.c2_u, self.c2_u + self.u2, 0])
        y_out = np.array([self.c1_a, self.c1_a - self.c2_a, self.c1_a - self.c2_a, self.c1_a])
        plt.figure(figsize=figsize)
        plt.plot(x_in, y_in, linewidth=2, color='red', label='inlet')
        plt.plot(x_out, y_out, linewidth=2, color='blue', label='outlet')
        plt.xlim(-self.c1_u, self.c2_u + self.u2)
        plt.ylim(-max(self.c1_a, self.c1_u), max(self.c1_a, self.c2_u + self.u2))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.legend()
        plt.show()


def get_first_stage_gas_dynamics(stage_geom: StageGeomAndHeatDrop, T0_stag, p0_stag, G_turbine,
                                 alpha_air) -> StageGasDynamics:
    result = StageGasDynamics(T0_stag, p0_stag, G_turbine, G_turbine, alpha_air, KeroseneCombustionProducts(),
                              stage_geom.rho, stage_geom.phi, stage_geom.psi, stage_geom.l1,
                              stage_geom.l2, stage_geom.D1, stage_geom.D2, stage_geom.delta_r_rk,
                              stage_geom.n, stage_geom.epsilon, stage_geom.g_lk, stage_geom.g_ld,
                              stage_geom.g_lb, H0=stage_geom.H0)
    result.compute()
    return result


def get_intermediate_stage(stage_geom: StageGeomAndHeatDrop, prev_stage: StageGasDynamics, precise_heat_drop=True) -> \
        StageGasDynamics:
    if precise_heat_drop:
        H0 = stage_geom.H0 * (1 + (1 - stage_geom.mu) ** 2 * prev_stage.c2 ** 2 /
                              (2 * prev_stage.c_p_gas * prev_stage.T_st)) + 0.5 * (stage_geom.mu * prev_stage.c2) ** 2
    else:
        H0 = stage_geom.H0
    p0_stag = prev_stage.p2 * (1 + (stage_geom.mu * prev_stage.c2)**2 / (2 * prev_stage.c_p_gas * prev_stage.T_st)) ** \
                              (prev_stage.k_gas / (prev_stage.k_gas - 1))
    result = StageGasDynamics(prev_stage.T_st_stag, p0_stag, prev_stage.G_stage_out, prev_stage.G_turbine,
                              prev_stage.alpha_air, KeroseneCombustionProducts(), stage_geom.rho, stage_geom.phi,
                              stage_geom.psi, stage_geom.l1, stage_geom.l2, stage_geom.D1, stage_geom.D2,
                              stage_geom.delta_r_rk, stage_geom.n, stage_geom.epsilon, stage_geom.g_lk,
                              stage_geom.g_ld, stage_geom.g_lb, H0=H0)
    result.compute()
    return result


def get_last_pressure_stage(turbine_geom: TurbineGeomAndHeatDropDistribution,
                            stage_geom: StageGeomAndHeatDrop, prev_stage: StageGasDynamics) -> StageGasDynamics:
    p0_stag = prev_stage.p2 * (1 + (stage_geom.mu * prev_stage.c2)**2 / (2 * prev_stage.c_p_gas * prev_stage.T_st)) ** \
                              (prev_stage.k_gas / (prev_stage.k_gas - 1))
    result = StageGasDynamics(prev_stage.T_st_stag, p0_stag, prev_stage.G_stage_out, prev_stage.G_turbine,
                              prev_stage.alpha_air, KeroseneCombustionProducts(), stage_geom.rho, stage_geom.phi,
                              stage_geom.psi, stage_geom.l1, stage_geom.l2, stage_geom.D1, stage_geom.D2,
                              stage_geom.delta_r_rk, stage_geom.n, stage_geom.epsilon, stage_geom.g_lk,
                              stage_geom.g_ld, stage_geom.g_lb, p2=turbine_geom.p_t)
    result.compute()
    return result


def get_last_work_stage(stage_geom: StageGeomAndHeatDrop, prev_stage: StageGasDynamics,
                        L_stage_rel, eta_t0) -> StageGasDynamics:
    L_stage = L_stage_rel / (prev_stage.G_stage_out / prev_stage.G_turbine -
                             (stage_geom.g_lb + stage_geom.g_ld + stage_geom.g_lk))
    p0_stag = prev_stage.p2 * (1 + (stage_geom.mu * prev_stage.c2) ** 2 / (2 * prev_stage.c_p_gas *
                                                                           prev_stage.T_st)) ** \
                              (prev_stage.k_gas / (prev_stage.k_gas - 1))
    result = StageGasDynamics(prev_stage.T_st_stag, p0_stag, prev_stage.G_stage_out, prev_stage.G_turbine,
                              prev_stage.alpha_air, KeroseneCombustionProducts(), stage_geom.rho, stage_geom.phi,
                              stage_geom.psi, stage_geom.l1, stage_geom.l2, stage_geom.D1, stage_geom.D2,
                              stage_geom.delta_r_rk, stage_geom.n, stage_geom.epsilon, stage_geom.g_lk,
                              stage_geom.g_ld, stage_geom.g_lb, L_t=L_stage, eta_t0=eta_t0)
    result.compute()
    return result


if __name__ == '__main__':
    deg = np.pi / 180
    stage_gd = StageGasDynamics(1400, 1100e3, 30, 30, 2.5, KeroseneCombustionProducts(), 0.8, 0.96, 0.96, 0.08, 0.10,
                                0.34, 0.37, 0.001, 15e3, 1, 0, 0, 0, L_t=250e3, eta_t0=0.85)  # H0=300e3)
    stage_gd.compute()
    print(stage_gd.alpha1 / deg)
    print(stage_gd.alpha2 / deg)
    print(stage_gd.beta1 / deg)
    print(stage_gd.beta2 / deg)
    print(stage_gd.H_l)
    print(stage_gd.H_s)
    print(stage_gd.c1_a)
    print(stage_gd.c2_a)
    print(stage_gd.T2)
    print(stage_gd.T_st)
    print(stage_gd.T_st_stag)
    print(stage_gd.eta_t)
    stage_gd.plot_velocity_triangle()
    # stage_gd.L_t = 190e3
    # print(stage_gd.eta_t)