from .stage_geom import InvalidStageSizeValue, StageGeomAndHeatDrop, \
    TurbineGeomAndHeatDropDistribution, set_logging
import logging
from gas_turbine_cycle.gases import KeroseneCombustionProducts, IdealGas, Air
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root
import enum
from gas_turbine_cycle.tools.functions import get_mixture_temp


class StageType(enum.Enum):
    HEAT_DROP = 0
    PRESSURE = 1
    WORK = 2


class StageGasDynamics:
    def __init__(self, T0_stag, p0_stag, T0_stag_ad_t, G_stage_in, G_turbine, G_fuel, work_fluid: IdealGas,
                 rho, phi, psi, l1, l2, D1, D2, delta_r_rk, n, epsilon,
                 g_lk, g_ld, g_lb, g_cool, T_cool=700, precision=0.001, cool_fluid: IdealGas=Air(), **kwargs):
        """
        :param T0_stag:
        :param p0_stag:
        :param T0_stag_ad_t: Температура на входе в ступень при адиабатическом процессе в турбине.
        :param G_stage_in: расход газа на входе в ступень
        :param G_turbine: расход газа через СА первой ступени
        :param G_fuel: расход топлива по тракту до турбины
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
        :param g_cool: Отношение расхода охлаждающего воздуха к расходу газа через СА первой ступени.
        :param T_cool: температура охлаждающего воздуха.
        :param precision: точность.
        :param cool_fluid: рабочее тело для охлажденияю
        :param kwargs: H0 - теплоперепад на ступени, p2_stag - полное давление на выходе из ступени,
                L_t - работа ступени, eta_t0 - КПД ступени в первом приближении.
        """
        set_logging()
        self.T0_stag = T0_stag
        self.p0_stag = p0_stag
        self.T0_stag_ad_t = T0_stag_ad_t
        self.G_stage_in = G_stage_in
        self.G_turbine = G_turbine
        self.work_fluid = work_fluid
        self.work_fluid_ad = type(work_fluid)()
        self.work_fluid_stag_ad = type(work_fluid)()
        self.G_fuel = G_fuel
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
        self.g_cool = g_cool
        self.T_cool = T_cool
        self.precision = precision
        self.cool_fluid = cool_fluid
        (self.stage_type, self.H0, self.p2_stag_pre,
         self.L_t_pre, self._eta_t0) = self._get_turbine_type_specific_params(**kwargs)
        self.k_res = None
        self.L_t_res = None
        self.p2_stag_res = None
        self.T_st_stag = 500
        self.T2_ad_t = 500
        self.T2_stag_ad_t = 500

    @classmethod
    def _get_turbine_type_specific_params(cls, **kwargs):
        if 'H0' in kwargs:
            stage_type = StageType.HEAT_DROP
            H0 = kwargs['H0']
            p2_stag = None
            L_t = None
            eta_t0 = None
        elif 'p2_stag' in kwargs:
            stage_type = StageType.PRESSURE
            H0 = None
            p2_stag = kwargs['p2_stag']
            L_t = None
            eta_t0 = None
        elif ('L_t' in kwargs) and ('eta_t0' in kwargs):
            stage_type = StageType.WORK
            L_t = kwargs['L_t']
            eta_t0 = kwargs['eta_t0']
            H0 = None
            p2_stag = None
        else:
            raise Exception('Incorrect set of kwags. Stage type can not be define.')
        return stage_type, H0, p2_stag, L_t, eta_t0

    def compute(self):
        if self.stage_type == StageType.HEAT_DROP:
            logging.info('%s РАСЧЕТ ГАЗОДИНАМИЧЕСКИХ ПАРАМЕТРОВ СТУПЕНИ ПО ЗАДАННОМУ ТЕПЛОПЕРЕПАДУ %s\n' %
                         (25 * '#', 25 * '#'))
            self._specified_heat_drop_calculation(self.H0)
        elif self.stage_type == StageType.PRESSURE:
            logging.info('%s РАСЧЕТ ГАЗОДИНАМИЧЕСКИХ ПАРАМЕТРОВ СТУПЕНИ ПО ДАВЛЕНИЮ НА ВЫХОДЕ %s\n' %
                         (25 * '#', 25 * '#'))
            self._specified_outlet_pressure_calculation(self.p2_stag_pre)
        elif self.stage_type == StageType.WORK:
            logging.info('%s РАСЧЕТ ГАЗОДИНАМИЧЕСКИХ ПАРАМЕТРОВ СТУПЕНИ ПО РАБОТЕ %s\n' %
                         (25 * '#', 25 * '#'))
            self._specified_work_calculation(self.L_t_pre, self._eta_t0)

    def _specified_heat_drop_calculation(self, H0):
        """Вычисляет параметры ступени по заданному теплоперепаду с уточнением k"""
        self.work_fluid.__init__()
        self.work_fluid_ad.__init__()
        self.work_fluid_stag_ad.__init__()

        self.g_fuel_in = self.G_fuel / (self.G_stage_in - self.G_fuel)
        self.alpha_air_in = 1 / (self.work_fluid.l0 * self.g_fuel_in)

        self.work_fluid.T1 = self.T0_stag
        self.work_fluid_ad.T1 = self.T0_stag_ad_t
        self.work_fluid_stag_ad.T1 = self.T0_stag_ad_t

        self.work_fluid.alpha = self.alpha_air_in
        self.work_fluid_ad.alpha = self.alpha_air_in
        self.work_fluid_stag_ad.alpha = self.alpha_air_in

        self.k_res = 1.
        self.iter_number_k_gas = 0
        logging.info('%s Расчет параметров ступени с уточнение k %s\n' % (15 * '-', 15 * '-'))
        while self.k_res >= self.precision:
            self.iter_number_k_gas += 1
            logging.info('ИТЕРАЦИЯ %s' % self.iter_number_k_gas)
            self._compute_stage_parameters(
                H0, self.work_fluid.c_p_av_int, self.work_fluid_ad.c_p_av_int, self.work_fluid_stag_ad.c_p_av_int)
            self.work_fluid.T2 = self.T_st_stag
            self.work_fluid_ad.T2 = self.T2_ad_t
            self.work_fluid_stag_ad.T2 = self.T2_stag_ad_t
            self.k_res = abs(self.k_gas - self.work_fluid.k_av_int) / self.k_gas
            logging.info('Residual = %.4f' % self.k_res)

    def _get_p2_stag(self, H0):
        self._specified_heat_drop_calculation(H0)
        return self.p2_stag

    def _specified_outlet_pressure_calculation(self, p2_stag_pre):
        """Вычисляет параметры ступени по заданному выходному давлению"""
        H0_init = self.work_fluid.c_p_av_int * self.T0_stag * (
                1 - (p2_stag_pre / self.p0_stag) ** ((self.work_fluid.k_av_int - 1) / self.work_fluid.k_av_int)
        )
        sol = root(lambda x: [self._get_p2_stag(x[0]) - p2_stag_pre], np.array([H0_init]))
        logging.info(sol)
        self.iter_number_p2_stag = sol.nfev
        self.H0 = sol.x[0]
        self.p2_stag_res = abs(self.p2_stag - p2_stag_pre) / p2_stag_pre

    def _specified_work_calculation(self, L_t, eta_t0):
        """Вычисляет параметры ступени по заданной работе"""
        self.L_t_res = 1
        self.iter_number_L_t = 0
        logging.info('%s Расчет параметров ступени с уточнение КПД по статическим параметрам %s\n' %
                     (15 * '-', 15 * '-'))
        self.eta_t = eta_t0
        while self.L_t_res >= self.precision:
            self.iter_number_L_t += 1
            logging.info('%s ИТЕРАЦИЯ %s %s\n' % ('-' * 10, self.iter_number_L_t, '-' * 10))
            self.H0 = L_t / self.eta_t
            self._specified_heat_drop_calculation(self.H0)
            self.L_t_res = abs(self.L_t - L_t) / self.L_t
            logging.info('')
            logging.info('Residual(eta_t) = %.4f\n' % self.L_t_res)

    def _compute_stage_parameters(self, H0, c_p_gas, c_p_gas_ad_t, c_p_gas_stag_ad_t):
        """Вычисляет параметры ступени по известному теплоперепаду без уточнения k"""
        self.g_stage_in = self.G_stage_in / self.G_turbine
        self.i_stag_in = self.work_fluid.get_specific_enthalpy(self.T0_stag, alpha=self.alpha_air_in) * self.g_stage_in
        self.i_stag_ad_t_in = self.work_fluid.get_specific_enthalpy(
            self.T0_stag_ad_t, alpha=self.alpha_air_in
        ) * self.g_stage_in
        self.u1 = np.pi * self.D1 * self.n / 60
        self.H_s = H0 * (1 - self.rho)
        self.c1 = self.phi * np.sqrt(2 * self.H_s)
        self.c_p_gas = c_p_gas
        self.c_p_gas_ad_t = c_p_gas_ad_t
        self.c_p_gas_stag_ad_t = c_p_gas_stag_ad_t
        self.k_gas = self.work_fluid.k_func(c_p_gas)
        self.k_gas_ad_t = self.work_fluid_ad.k_func(c_p_gas_ad_t)
        self.k_gas_stag_ad_t = self.work_fluid_ad.k_func(c_p_gas_stag_ad_t)
        self.T1 = self.T0_stag - self.H_s * self.phi ** 2 / self.c_p_gas
        self.T1_ad = self.T0_stag - self.H_s / self.c_p_gas
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
        self.w1_a = self.w1 * np.sin(self.beta1)
        self.w1_u = self.w1 * np.cos(self.beta1)
        self.H_l = self.rho * H0 * self.T1 / self.T1_ad
        self.u2 = np.pi * self.D2 * self.n / 60
        self.T1_w_stag = self.T1 + self.w1 ** 2 / (2 * self.c_p_gas)
        self.w2 = self.psi * np.sqrt(self.w1**2 + 2 * self.H_l + self.u2**2 - self.u1**2)
        self.T2 = self.T1 - (self.w2**2 - self.w1**2 - self.u2**2 + self.u1**2) / (2 * self.c_p_gas)
        self.T2_ad = self.T1 - self.H_l / self.c_p_gas
        self.p2 = self.p1 * (self.T2_ad / self.T1) ** (self.k_gas / (self.k_gas - 1))
        self.rho2 = self.p2 / (self.work_fluid.R * self.T2)
        self.A2_a = np.pi * self.D2 * self.l2
        self.G_rk = self.G_stage_in - self.G_turbine * (self.g_lb + self.g_lk)
        self.c2_a = self.G_rk / (self.A2_a * self.rho2)
        if self.c2_a / self.w2 >= 1:
            raise InvalidStageSizeValue('c2_a must be less than w2')
        self.beta2 = np.arcsin(self.c2_a / self.w2)
        self.w2_a = self.w2 * np.sin(self.beta2)
        self.w2_u = self.w2 * np.cos(self.beta2)
        self.c2_u = self.w2 * np.cos(self.beta2) - self.u2
        if self.c2_u >= 0:
            self.alpha2 = np.arctan(self.c2_a / self.c2_u)
        else:
            self.alpha2 = np.pi + np.arctan(self.c2_a / self.c2_u)
        self.c2 = np.sqrt(self.c2_a**2 + self.c2_u**2)
        self.L_u = self.c1_u * self.u1 + self.c2_u * self.u2
        self.eta_u = self.L_u / H0
        self.h_s = (1 / self.phi ** 2 - 1) * self.c1 ** 2 / 2
        self.h_s_touch = self.h_s * self.T2_ad / self.T1
        self.zeta_s = self.h_s / H0
        self.zeta_s_touch = self.h_s_touch / H0
        self.h_l = (1 / self.psi ** 2 - 1) * self.w2 ** 2 / 2
        self.zeta_l = self.h_l / H0
        self.h_v = self.c2 ** 2 / 2
        self.zeta_v = self.h_v / H0
        self.eta_u_check = 1 - self.zeta_s_touch - self.zeta_l - self.zeta_v
        self.D_av = 0.5 * (self.D1 + self.D2)
        self.h_z = 1.37 * (1 + 1.6 * self.rho) * (1 + self.l2 / self.D_av) * self.delta_r_rk / self.l2 * self.L_u
        self.zeta_z = self.h_z / H0
        self.L_uz = self.L_u - self.h_z
        self.eta_t_touch = self.eta_u - self.zeta_z
        self.eta_l_touch = self.eta_t_touch + self.zeta_v
        self.l_av = 0.5 * (self.l1 + self.l2)
        self.u_av = 0.5 * (self.u1 + self.u2)
        self.N_tv = (1.07 * self.D_av**2 + 61 * (1 - self.epsilon) * self.D_av * self.l_av) * \
                    (self.u_av / 100)**3 * self.rho
        self.h_tv = self.N_tv / self.G_stage_in
        self.zeta_tv = self.h_tv / H0
        self.eta_t = self.eta_t_touch - self.zeta_tv
        self.eta_l = self.eta_l_touch - self.zeta_tv
        self.L_t = H0 * self.eta_t
        self.L_t_prime = self.L_t * (self.G_stage_in / self.G_turbine - (self.g_ld + self.g_lk + self.g_lb))
        "Удельная работа ступени, отнесенная к расходу через СА первой ступени с учетом потерь из-за утечек и " \
        "добавки охлаждающего воздуха"
        self.T_st = self.T2 + self. h_z / self.c_p_gas + self.h_tv / self.c_p_gas
        self.T_st_stag = self.T_st + self.h_v / self.c_p_gas
        self.p2_stag = self.p2 * (self.T_st_stag / self.T_st) ** (self.k_gas / (self.k_gas - 1))
        self.T2_stag_ad_t = self.T0_stag_ad_t * (self.p2_stag / self.p0_stag) ** (
                (self.k_gas_stag_ad_t - 1) / self.k_gas_stag_ad_t
        )
        self.T2_ad_t = self.T0_stag_ad_t * (self.p2 / self.p0_stag) ** (
                (self.k_gas_ad_t - 1) / self.k_gas_ad_t
        )
        self.H0_stag_ad_t = self.c_p_gas_stag_ad_t * (self.T0_stag_ad_t - self.T2_stag_ad_t)
        self.H0_ad_t = c_p_gas_ad_t * (self.T0_stag_ad_t - self.T2_ad_t)
        self.H0_stag = self.c_p_gas * self.T0_stag * (
                1 - (self.p2_stag / self.p0_stag) ** ((self.k_gas - 1) / self.k_gas)
        )
        self.H0_ad_t_prime = self.H0_ad_t * (self.G_stage_in / self.G_turbine - (self.g_ld + self.g_lk + self.g_lb))
        self.H0_stag_ad_t_prime = self.H0_stag_ad_t * (self.G_stage_in / self.G_turbine -
                                                       (self.g_ld + self.g_lk + self.g_lb))
        self.eta_t_stag = self.L_t / self.H0_stag
        self.G_stage_out_prime = self.G_stage_in - self.G_turbine * self.g_lk
        "Расход на выходе из ступени без учета охлаждения"
        self.G_stage_out = self.G_stage_in - self.G_turbine * self.g_lk + self.G_turbine * self.g_cool
        self.g_stage_out = self.G_stage_out / self.G_turbine
        self.g_fuel_out = self.G_fuel / (self.G_stage_out_prime - self.G_fuel + self.g_cool * self.G_turbine)
        self.alpha_air_out = 1 / (self.work_fluid.l0 * self.g_fuel_out)
        self.pi = self.p0_stag / self.p2
        self.pi_stag = self.p0_stag / self.p2_stag
        self.G_cool = self.g_cool * self.G_turbine
        self.i_stag_out = self.work_fluid.get_specific_enthalpy(
            self.T_st_stag, alpha=self.alpha_air_in
        ) * self.g_stage_in
        self.i_stag_ad_t_out = self.work_fluid.get_specific_enthalpy(
            self.T2_stag_ad_t, alpha=self.alpha_air_in
        ) * self.g_stage_in
        self.i_ad_t_out = self.work_fluid.get_specific_enthalpy(
            self.T2_ad_t, alpha=self.alpha_air_in
        ) * self.g_stage_in
        self.i_cool = self.cool_fluid.get_specific_enthalpy(self.T_cool) * self.g_cool

        self.T_mix_stag_new, self.mixture, self.c_p_gas_av_stag_out, \
        self.c_p_cool_av, self.T_mix_stag, self.T_mix_stag_res = get_mixture_temp(
            self.work_fluid, self.cool_fluid, self.T_st_stag, self.T_cool,
            self.G_stage_out_prime, self.G_cool, self.alpha_air_out, self.precision
        )
        self.T_mix_stag_ad_t_new, self.mixture_ad_t, self.c_p_gas_av_stag_ad_t_out, \
        self.c_p_cool_av_ad_t, self.T_mix_stag_ad_t, self.T_mix_stag_ad_t_res = get_mixture_temp(
            self.work_fluid_stag_ad, self.cool_fluid, self.T2_stag_ad_t, self.T_cool,
            self.G_stage_out_prime, self.G_cool, self.alpha_air_out, self.precision
        )
        self.i_stag_mixture = self.mixture.get_specific_enthalpy(
            self.T_mix_stag, alpha=self.alpha_air_out
        ) * self.g_stage_out

        self.i_stag_ad_t_mixture = self.mixture_ad_t.get_specific_enthalpy(
            self.T_mix_stag_ad_t, alpha=self.alpha_air_out
        ) * self.g_stage_out

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


def get_first_stage(work_fluid: IdealGas, stage_geom: StageGeomAndHeatDrop, T0_stag, p0_stag, G_turbine, G_fuel,
                    precision=0.001) -> StageGasDynamics:
    """Расчет первой ступени( если она не единственная)"""
    result = StageGasDynamics(T0_stag, p0_stag, T0_stag, G_turbine, G_turbine, G_fuel, work_fluid,
                              stage_geom.rho, stage_geom.phi, stage_geom.psi, stage_geom.l1,
                              stage_geom.l2, stage_geom.D1, stage_geom.D2, stage_geom.delta_r_rk,
                              stage_geom.n, stage_geom.epsilon, stage_geom.g_lk, stage_geom.g_ld,
                              stage_geom.g_lb, stage_geom.g_cool, stage_geom.T_cool,
                              precision=precision,
                              H0=stage_geom.H0)
    result.compute()
    return result


def get_intermediate_stage(work_fluid: IdealGas, stage_geom: StageGeomAndHeatDrop, prev_stage: StageGasDynamics,
                           prev_stage_geom: StageGeomAndHeatDrop, G_fuel, precision=0.001,
                           precise_heat_drop=True) -> StageGasDynamics:
    """Расчет промежуточной ступени"""
    if precise_heat_drop:
        H0 = stage_geom.H0 * (1 + (1 - prev_stage_geom.mu) ** 2 * prev_stage.c2 ** 2 /
                              (2 * prev_stage.c_p_gas * prev_stage.T_st)) + 0.5 * (stage_geom.mu * prev_stage.c2) ** 2
    else:
        H0 = stage_geom.H0
    p0_stag = prev_stage.p2 * (1 + (stage_geom.mu * prev_stage.c2)**2 / (2 * prev_stage.c_p_gas * prev_stage.T_st)) ** \
                              (prev_stage.k_gas / (prev_stage.k_gas - 1))
    result = StageGasDynamics(prev_stage.T_mix_stag, p0_stag, prev_stage.T_mix_stag_ad_t, prev_stage.G_stage_out,
                              prev_stage.G_turbine, G_fuel,
                              work_fluid, stage_geom.rho, stage_geom.phi,
                              stage_geom.psi, stage_geom.l1, stage_geom.l2, stage_geom.D1, stage_geom.D2,
                              stage_geom.delta_r_rk, stage_geom.n, stage_geom.epsilon, stage_geom.g_lk,
                              stage_geom.g_ld, stage_geom.g_lb, stage_geom.g_cool, stage_geom.T_cool,
                              precision=precision,
                              H0=H0)
    result.compute()
    return result


def get_last_pressure_stage(work_fluid: IdealGas,
                            stage_geom: StageGeomAndHeatDrop,
                            prev_stage: StageGasDynamics, prev_stage_geom: StageGeomAndHeatDrop, p2_stag,
                            G_fuel, precision=0.001, ) -> StageGasDynamics:
    """Расчет последней ступени по выходному давлению (для силовой турбины)"""
    p0_stag = prev_stage.p2 * (1 + (prev_stage_geom.mu * prev_stage.c2)**2 / (2 * prev_stage.c_p_gas * prev_stage.T_st)) ** \
                              (prev_stage.k_gas / (prev_stage.k_gas - 1))
    result = StageGasDynamics(prev_stage.T_mix_stag, p0_stag, prev_stage.T_mix_stag_ad_t, prev_stage.G_stage_out,
                              prev_stage.G_turbine, G_fuel,
                              work_fluid, stage_geom.rho, stage_geom.phi,
                              stage_geom.psi, stage_geom.l1, stage_geom.l2, stage_geom.D1, stage_geom.D2,
                              stage_geom.delta_r_rk, stage_geom.n, stage_geom.epsilon, stage_geom.g_lk,
                              stage_geom.g_ld, stage_geom.g_lb, stage_geom.g_cool, stage_geom.T_cool,
                              precision=precision,
                              p2_stag=p2_stag)
    result.compute()
    return result


def get_only_pressure_stage(work_fluid: IdealGas, turbine_geom: TurbineGeomAndHeatDropDistribution, p2_stag,
                            T0_stag, p0_stag, G_turbine, G_fuel, precision=0.001) -> StageGasDynamics:
    """Расчет первой ступени по давлению на выходе, в случае когда она единственная в турбине"""
    result = StageGasDynamics(T0_stag, p0_stag, T0_stag, G_turbine, G_turbine, G_fuel, work_fluid,
                              turbine_geom[0].rho, turbine_geom[0].phi, turbine_geom[0].psi, turbine_geom[0].l1,
                              turbine_geom[0].l2, turbine_geom[0].D1, turbine_geom[0].D2, turbine_geom[0].delta_r_rk,
                              turbine_geom[0].n, turbine_geom[0].epsilon, turbine_geom[0].g_lk, turbine_geom[0].g_ld,
                              turbine_geom[0].g_lb, turbine_geom[0].g_cool, turbine_geom[0].T_cool,
                              precision=precision,
                              p2_stag=p2_stag)
    result.compute()
    return result


def get_last_work_stage(work_fluid: IdealGas, stage_geom: StageGeomAndHeatDrop, prev_stage: StageGasDynamics,
                        prev_stage_geom: StageGeomAndHeatDrop, L_stage_rel, eta_t0,
                        G_fuel, precision=0.001) -> StageGasDynamics:
    """Расчет последней ступени по работе турбины (для компрессорной турбины)"""
    L_stage = L_stage_rel / (prev_stage.G_stage_out / prev_stage.G_turbine -
                             (stage_geom.g_lb + stage_geom.g_ld + stage_geom.g_lk))
    p0_stag = prev_stage.p2 * (1 + (prev_stage_geom.mu * prev_stage.c2) ** 2 / (2 * prev_stage.c_p_gas *
                                                                           prev_stage.T_st)) ** \
                              (prev_stage.k_gas / (prev_stage.k_gas - 1))
    result = StageGasDynamics(prev_stage.T_mix_stag, p0_stag, prev_stage.T_mix_stag_ad_t, prev_stage.G_stage_out,
                              prev_stage.G_turbine, G_fuel,
                              work_fluid, stage_geom.rho, stage_geom.phi,
                              stage_geom.psi, stage_geom.l1, stage_geom.l2, stage_geom.D1, stage_geom.D2,
                              stage_geom.delta_r_rk, stage_geom.n, stage_geom.epsilon, stage_geom.g_lk,
                              stage_geom.g_ld, stage_geom.g_lb, stage_geom.g_cool, stage_geom.T_cool,
                              precision=precision,
                              L_t=L_stage, eta_t0=eta_t0)
    result.compute()
    return result


def get_only_work_stage(work_fluid: IdealGas, stage_geom: StageGeomAndHeatDrop, L_stage_rel, eta_t0,
                        T0_stag, p0_stag, G_turbine, G_fuel, precision=0.001) -> StageGasDynamics:
    L_stage = L_stage_rel / (1 - (stage_geom.g_lb + stage_geom.g_ld + stage_geom.g_lk))
    result = StageGasDynamics(T0_stag, p0_stag, T0_stag, G_turbine, G_turbine, G_fuel, work_fluid,
                              stage_geom.rho, stage_geom.phi, stage_geom.psi, stage_geom.l1, stage_geom.l2,
                              stage_geom.D1, stage_geom.D2, stage_geom.delta_r_rk, stage_geom.n, stage_geom.epsilon,
                              stage_geom.g_lk, stage_geom.g_ld, stage_geom.g_lb, stage_geom.g_cool, stage_geom.T_cool,
                              precision=precision,
                              L_t=L_stage, eta_t0=eta_t0)
    result.compute()
    return result


if __name__ == '__main__':
    pass
