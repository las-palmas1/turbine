from gas_turbine_cycle.gases import KeroseneCombustionProducts, IdealGas
import gas_turbine_cycle.tools.functions as func
import logging
import matplotlib.pyplot as plt
from gas_turbine_cycle.tools.gas_dynamics import *
from scipy.optimize import fsolve
import os


class InvalidStageSizeValue(Exception):
    def __init__(self, message):
        Exception.__init__(self, message)
        self.message = message

    def __str__(self):
        return self.message


def set_logging():
    logging.basicConfig(filemode='a', filename=os.path.join(os.getcwd(), 'average_streamline.log'),
                        format='%(levelname)s - %(message)s', level=logging.INFO)


class StageGeomAndHeatDrop:
    """
    Индекс 0 используется для обозначения параметров на входе в СА, 05 - на выходе из СА,
    1 - на входе в РК, 2 - на выходе из РК
    """
    def __init__(self):
        self.gamma_out = None
        self.gamma_in = None
        self.gamma_av = None
        self.p_r_out_l1_ratio = 0.03
        "Относительный размер перекрыши на периферии"
        self.p_r_in_l1_ratio = 0.03
        "Относительный размер перекрыши на втулке"
        self.p_a_out_rel = 0.2
        "Отношение расстояния от РК до периферийной перекрыши к величине осевого зазора"
        self.p_a_in_rel = 0.2
        "Отношение расстояния от РК до втулочной перекрыши к величине осевого зазора"
        self.l1_b_sa_ratio = 1.6
        "Отношение длины лопатки на входе в РК к ширине лопатки СА"
        self.l2_b_rk_ratio = 1.8
        "Отношение длины лопатки на выходе из РК к ширине лопатки РК"
        self.delta_a_b_sa_ratio = 0.22
        "Отношение осевого зазора после СА к ширине лопатки СА"
        self.delta_a_b_rk_ratio = 0.22
        "Отношение осевого зазора после РК к ширине лопатки РК"
        self.mu = 1
        "Коэффициент использования скорости на выходе их ступени"
        self._k_b_sa = None
        self._k_b_rk = None
        self._k_delta_a_sa = None
        self._k_delta_a_rk = None
        self.l1 = None
        self.D1 = None
        self.delta_r_rk_l2_ratio = 0.01
        "Относительный радиальный зазор"
        self.n = None
        self.x0 = None
        self.y0 = None
        "x0, y0 - координаты верхнего левого угла на графике"
        self.delta_x0 = None
        self.delta_y0 = None
        self.rho = None
        "Степень реактивности"
        self.phi = 0.97
        "Коэффициент скорости в сопловых лопатках"
        self.psi = 0.97
        "Коэффициент скорости в рабочих лопатках"
        self.epsilon = 1
        "Степень парциальности. Необходима для вычисления затрат на трение и вентиляцию"
        self.g_lk = 0
        "Отношение расхода утечек в концевых лабиринтах к расходу газа через СА первой ступени"
        self.g_ld = 0
        "Отношение расход перетечек в лабиринтных уплотнениях сопловых диафрагм к расходу газа через СА первой ступени"
        self.g_lb = 0
        "Отношение расход перетечек поверх бондажа рабочих лопаток к расходу газа через СА первой ступени"
        self.g_cool = 0.
        "Отношение расхода охлаждающего воздуха к расходу газа через СА первой ступени."
        self.T_cool = 700
        "Температура охлаждающего воздуха"

        self.D2 = None
        self.D0 = None
        self.D05 = None
        self.l0 = None
        self.l05 = None
        self.l2 = None
        self.u1 = None
        self.u2 = None
        self.u_av = None
        """Средняя окружная скорость на ступени. Необходима для вычисления рапределения теплоперепадов по
        коэффициентам возврата теплоты"""
        self.H0 = None
        "Теплоперепад на ступени"
        self.delta_a_sa = None
        self.delta_a_rk = None
        self.b_sa = None
        self.b_rk = None
        self.p_r_in = None
        self.p_r_out = None
        self.p_a_in = None
        self.p_a_out = None
        self.length = None
        "Суммарная длина ступени"
        self.delta_r_rk = None
        "Радиальный зазор"
        self.l0_next = None
        self.A1 = None
        "Кольцевая площадь на входе"
        self.A2 = None
        "Кольцевая площадь на выходе"

    def str(self):
        str_arr = str(self).split()
        return str_arr[0][1:] + ' ' + str_arr[1]

    def compute_coefficients(self):
        self._k_b_sa = 1 / self.l1_b_sa_ratio
        self._k_b_rk = 1 / self.l2_b_rk_ratio
        self._k_delta_a_sa = self.delta_a_b_sa_ratio / self.l1_b_sa_ratio
        self._k_delta_a_rk = self.delta_a_b_rk_ratio / self.l2_b_rk_ratio

    def compute_geometry(self):
        logging.info('Вычисление геометрических параметров ступени')
        self.b_sa = self.l1 * self.k_b_sa
        self.delta_a_sa = self.l1 * self.k_delta_a_sa
        self.p_r_in = self.p_r_in_l1_ratio * self.l1
        self.p_r_out = self.p_r_out_l1_ratio * self.l1
        self.p_a_in = self.p_a_in_rel * self.delta_a_sa
        self.p_a_out = self.p_a_out_rel * self.delta_a_sa
        self.l0 = self.l1 - (np.tan(self.gamma_in) + np.tan(self.gamma_out)) * (self.b_sa + self.delta_a_sa) - \
                  self.p_r_in - self.p_r_out
        self.l2 = self.l1 / (1 - self.k_b_rk * (np.tan(self.gamma_in) + np.tan(self.gamma_out)))
        self.delta_r_rk = self.delta_r_rk_l2_ratio * self.l2
        self.b_rk = self.l2 * self.k_b_rk
        self.delta_a_rk = self.l2 * self.k_delta_a_rk
        self.D2 = self.D1 + 2 * np.tan(self.gamma_av) * self.b_rk
        self.D05 = self.D1 - 2 * np.tan(self.gamma_av) * self.delta_a_sa
        self.D0 = self.D05 - 2 * np.tan(self.gamma_av) * self.b_sa
        self.l05 = self.l0 + self.b_sa * (np.tan(self.gamma_in) + np.tan(self.gamma_out))
        self.length = self.b_sa + self.delta_a_sa + self.b_rk + self.delta_a_rk
        self.l0_next = self.l2 + self.delta_a_rk * (np.tan(self.gamma_in) + np.tan(self.gamma_out))
        self.A1 = np.pi * self.D1 * self.l1
        self.A2 = np.pi * self.D2 * self.l2
        self.delta_x0 = self.length
        self.delta_y0 = np.tan(self.gamma_out) * (self.b_sa + self.delta_a_sa +
                                                  self.b_rk + self.delta_a_rk) + self.p_r_out

    def compute_velocities(self):
        logging.info('Вычисление окружных скоростей')
        self.u1 = np.pi * self.D1 * self.n / 60
        self.u2 = np.pi * self.D2 * self.n / 60
        self.u_av = np.pi * 0.5 * (self.D2 + self.D1) * self.n / 60

    def plot(self):
        logging.info('Вырисовывание ступени')
        x_sa_arr = np.array([self.x0, self.x0, self.x0 + self.b_sa, self.x0 + self.b_sa, self.x0])
        y_sa_arr = np.array([self.y0 - self.l0,
                             self.y0,
                             self.y0 + self.b_sa * np.tan(self.gamma_out),
                             self.y0 + self.b_sa * np.tan(self.gamma_out) - self.l05,
                             self.y0 - self.l0])
        x0_rk = self.x0 + self.b_sa + self.delta_a_sa
        y0_rk = self.y0 + np.tan(self.gamma_out) * (self.b_sa + self.delta_a_sa) + self.p_r_out
        x_rk_arr = np.array([x0_rk, x0_rk, x0_rk + self.b_rk, x0_rk + self.b_rk, x0_rk])
        y_rk_arr = np.array([y0_rk - self.l1,
                             y0_rk - self.delta_r_rk,
                             y0_rk - self.delta_r_rk + np.tan(self.gamma_out) * self.b_rk,
                             y0_rk + np.tan(self.gamma_out) * self.b_rk - self.l2,
                             y0_rk - self.l1])
        x_out_arr = np.array([self.x0, self.x0 + self.b_sa + self.delta_a_sa - self.p_a_out,
                              self.x0 + self.b_sa + self.delta_a_sa - self.p_a_out,
                              x0_rk + self.b_rk + self.delta_a_rk])
        x_av_arr = np.array([self.x0, self.x0 + self.b_sa, x0_rk + self.b_rk + self.delta_a_rk])
        y_out_arr = np.array([self.y0,
                              self.y0 + np.tan(self.gamma_out) * (self.b_sa + self.delta_a_sa - self.p_a_out),
                              self.y0 + np.tan(self.gamma_out) * (self.b_sa + self.delta_a_sa - self.p_a_out) +
                              self.p_r_out,
                              self.y0 + np.tan(self.gamma_out) *
                              (self.b_sa + self.delta_a_sa + self.b_rk + self.delta_a_rk) + self.p_r_out])
        y_av_arr = np.array([0.5 * self.D0, 0.5 * self.D05,
                             0.5 * self.D2 + np.tan(self.gamma_av) * self.delta_a_rk])
        plt.plot(x_sa_arr, y_sa_arr, linewidth=2, color='red')
        plt.plot(x_rk_arr, y_rk_arr, linewidth=2, color='blue')
        plt.plot(x_out_arr, y_out_arr, linewidth=2, color='black')
        plt.plot(x_av_arr, y_av_arr, '--', linewidth=2, color='black')

    @property
    def k_delta_a_rk(self):
        return self._k_delta_a_rk

    @property
    def k_delta_a_sa(self):
        return self._k_delta_a_sa

    @property
    def k_b_rk(self):
        return self._k_b_rk

    @property
    def k_b_sa(self):
        return self._k_b_sa


class TurbineGeomAndHeatDropDistribution:
    """
    NOTE: расчет геометрии производится по относительным геометрическим параметрам и площади на входе в
    РК первой ступени. Входная площадь определяется по расходу и скорости перед РК первой ступени.
    Скорость в свою очередь определяется по заданному на первой ступене теплоперепаду и степени реактивности.
    В результате вычисления предварительного распределения теплоперепадов по ступеням значение теплоперепада на
    перовй ступени может оказаться отличным от того значения, которое использовалось при вычислении геометрии.
    Производить вычисление распределения теплоперепадов по ступеням до вычисления геометрии не представляется
    возможным, так как геометрическии параметры необходимы при расчете этого распределения. Поэтому расчет
    распределения теплоперепадов производится итерационно, с уточнением на каждой итерации значения теплоперепада
    на первой ступени.
    """
    def __init__(self, stage_number, eta_t_stag, n, work_fluid: IdealGas, T_g_stag,
                 p_g_stag, G_fuel, G_turbine, l1_D1_ratio, alpha11, k_n, T_t_stag,
                 auto_compute_heat_drop: bool=True, precision=0.001, **kwargs):
        """
        :param stage_number:
        :param eta_t_stag:
        :param n: частота вращения
        :param work_fluid:
        :param T_g_stag:
        :param p_g_stag:
        :param G_fuel: Суммарный расход топлива перед турбиной.
        :param G_turbine:
        :param l1_D1_ratio:
        :param alpha11: угол потока после СА первой ступени
        :param k_n:
        :param T_t_stag:
        :param auto_compute_heat_drop:
        :param precision: точность.
        :param kwargs: gamma_av, gamma_sum, gamma_in, gamma_out, c21
        """
        self.stage_number = stage_number
        self.eta_t_stag = eta_t_stag
        self.n = n
        self.work_fluid = work_fluid
        self.T_g_stag = T_g_stag
        self.p_g_stag = p_g_stag
        self.G_fuel = G_fuel
        self.G_turbine = G_turbine
        self.l1_D1_ratio = l1_D1_ratio
        self.alpha11 = alpha11
        self.k_n = k_n
        self.T_t_stag = T_t_stag
        self.p_t_stag, self.H_t_stag = self._get_p_t_stag_and_H_t(self.work_fluid, p_g_stag, T_g_stag,
                                                                  T_t_stag, eta_t_stag, G_turbine, G_fuel)
        self.auto_compute_heat_drop = auto_compute_heat_drop
        self.precision = precision
        self._kwargs = kwargs
        if ('gamma_av' in kwargs) and ('gamma_sum' in kwargs):
            self.gamma_av = kwargs['gamma_av']
            self.gamma_in = None
            self.gamma_out = None
            self.gamma_sum = kwargs['gamma_sum']
        elif ('gamma_in' in kwargs) and ('gamma_out' in kwargs):
            self.gamma_av = None
            self.gamma_in = kwargs['gamma_in']
            self.gamma_out = kwargs['gamma_out']
            self.gamma_sum = None
        else:
            assert False, 'gamma_av and gamma_sum or gamma_in and gamma_out must be set'
        self._stages = [StageGeomAndHeatDrop() for _ in range(self.stage_number)]
        if auto_compute_heat_drop:
            assert 'c21' in kwargs, 'c21 is not specified'
            self._c21 = kwargs['c21']
        for item in self._stages:
            item.n = self.n

    @classmethod
    def _get_p_t_stag_and_H_t(cls, work_fluid: IdealGas, p_g_stag, T_g_stag, T_t_stag, eta_t_stag, G_turbine, G_fuel):
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
        return p_t_stag, H_t

    @property
    def c21(self):
        """Скорость на выходе из РК первой ступени"""
        assert self._c21 is not None, 'c21 is not specified.'
        return self._c21

    @property
    def H01(self):
        """Теплоперепад на первой ступени"""
        assert self[0].H0 is not None, 'H01 is not specified'
        return self[0].H0

    @H01.setter
    def H01(self, value):
        self[0].H0 = value

    @property
    def rho1(self):
        """Степень реактивности на первой ступени"""
        assert self[0].rho is not None, 'rho1 is not specified'
        return self[0].rho

    @property
    def last(self) -> StageGeomAndHeatDrop:
        return self._stages[self.stage_number - 1]

    @property
    def first(self) -> StageGeomAndHeatDrop:
        return self._stages[0]

    def str(self):
        str_arr = str(self).split()
        return str_arr[0][1:] + ' ' + str_arr[1]

    def _compute_d1_and_l1(self):
        """Вычисление среднего диаметра и длины лопатки на входе в РК первой ступени"""
        logging.info('Вычисление среднего диаметра и длины лопатки на входе в РК первой ступени')
        self.work_fluid.__init__()
        self._compute_t_11()
        self.p_11 = self.p_g_stag * (1 - self.H_s1 /
                                     (self.c_p_gas11 * self.T_g_stag)) ** (self.k_gas11 / (self.k_gas11 - 1))
        self.rho11 = self.p_11 / (self.work_fluid.R * self.T_11)
        self.D1_l1_product = self.G_turbine / (np.pi * self.rho11 * np.sin(self.alpha11) * self.c11)
        logging.debug('%s _compute_d1_and_l1 D1_l1_product = %s' % (self.str(), self.D1_l1_product))
        self.D1 = np.sqrt(self.D1_l1_product / self.l1_D1_ratio)
        self.l1 = self.D1_l1_product / self.D1
        logging.debug('%s _compute_d1_and_l1 D1 = %s, l1 = %s' % (self.str(), self.D1, self.l1))

    def _compute_t_11(self):
        """Вычисляет величину температуры перед РК первой ступени T_11"""
        logging.info('Вычисление температуры перед РК первой ступени')
        self.T11_res = 1.
        self._iter_number_d1_l1 = 0
        while self.T11_res >= self.precision:
            self._iter_number_d1_l1 += 1
            logging.debug('%s _compute_t_11 iter_number = %s' % (self.str(), self._iter_number_d1_l1))
            self.work_fluid.T1 = self.T_g_stag
            self.g_fuel = self.G_fuel / (self.G_turbine - self.G_fuel)
            self.alpha_air = 1 / (self.work_fluid.l0 * self.g_fuel)
            self.work_fluid.alpha = self.alpha_air
            self.H_s1 = self.H01 * (1 - self.rho1)
            self.c11 = self._stages[0].phi * np.sqrt(2 * self.H_s1)
            self.k_gas11 = self.work_fluid.k_av_int
            self.c_p_gas11 = self.work_fluid.c_p_av_int
            logging.debug('%s _compute_t_11 c_p_gas11 = %s' % (self.str(), self.k_gas11))
            self.T_11 = self.T_g_stag - self.H_s1 * self._stages[0].phi ** 2 / self.c_p_gas11
            self.T11_res = abs(self.T_11 - self.work_fluid.T2) / self.work_fluid.T2
            self.work_fluid.T2 = self.T_11
            logging.debug('%s _compute_t_11 T11_res = %s' % (self.str(), self.T11_res))

    def _compute_angles(self):
        logging.info('Вычисление углов')
        if ('gamma_in' in self._kwargs) and ('gamma_out' in self._kwargs):
            self.gamma_av = np.arctan(0.5 * (np.tan(self.gamma_out) - np.tan(self.gamma_in)))
            self.gamma_sum = self.gamma_in + self.gamma_out
        elif ('gamma_av' in self._kwargs) and ('gamma_sum' in self._kwargs):
            a = np.tan(self.gamma_sum)
            b = 2 - 2 * np.tan(self.gamma_sum) * np.tan(self.gamma_av)
            c = -2 * np.tan(self.gamma_av) - np.tan(self.gamma_sum)
            d = b**2 - 4 * a * c
            if d < 0:
                raise InvalidStageSizeValue('d < 0')
            self.gamma_out = np.arctan((-b + np.sqrt(d)) / (2 * a))
            self.gamma_in = np.arctan(np.tan(self.gamma_out) - 2 * np.tan(self.gamma_av))

    def __getitem__(self, item) -> StageGeomAndHeatDrop:
        if 0 <= item < self.stage_number:
            return self._stages[item]
        else:
            raise IndexError('invalid index')

    def __len__(self):
        return len(self._stages)

    def __iter__(self):
        self._num = 0
        return self

    def __next__(self) -> StageGeomAndHeatDrop:
        if self._num < self.stage_number:
            current = self._stages[self._num]
            self._num += 1
            return current
        else:
            raise StopIteration()

    def _compute_linear_dimensions(self):
        logging.info('Расчет линейных размеров ступеней\n')
        for num, item in enumerate(self._stages):
            logging.info('Ступень %s' % (num + 1))
            item.gamma_av = self.gamma_av
            item.gamma_out = self.gamma_out
            item.gamma_in = self.gamma_in
            item.compute_coefficients()
            if num == 0:
                item.l1 = self.l1
                item.D1 = self.D1
                item.D0 = item.D05
                item.l0 = item.l05
            else:
                l0 = self._stages[num - 1].l0_next
                item.l1 = l0 / (1 - (item.k_b_sa + item.k_delta_a_sa) *
                                (np.tan(self.gamma_in) + np.tan(self.gamma_out)) -
                                (item.p_r_out_l1_ratio + item.p_r_in_l1_ratio))
                b_sa = item.l1 * item.k_b_sa
                delta_a_sa = item.l1 * item.k_delta_a_sa
                item.D1 = self._stages[num - 1].D2 + 2 * np.tan(self.gamma_av) * (self._stages[num - 1].delta_a_rk +
                                                                                  b_sa + delta_a_sa)
            item.compute_geometry()
            item.compute_velocities()

    def _compute_geometry(self):
        logging.info('%s РАСЧЕТ ГЕОМЕТРИИ ТУРБИНЫ %s' % ('-'*10, '-'*10))
        self._compute_d1_and_l1()
        self._compute_angles()
        self._compute_linear_dimensions()

    def plot_geometry(self, figsize=(5, 6), title='Turbine geometry'):
        logging.info('%s РИСОВАНИЕ ГЕОМЕТРИИ ТУРБИНЫ %s ' % ('-'*10, '-'*10))
        plt.figure(figsize=figsize)
        for num, item in enumerate(self._stages):
            if num == 0:
                item.x0 = 0
                item.y0 = 0.5 * item.D0 + 0.5 * (item.l0 + item.p_r_out + item.p_r_in) - item.p_r_out
                item.plot()
            else:
                item.x0 = self._stages[num - 1].x0 + self._stages[num - 1].delta_x0
                item.y0 = self._stages[num - 1].y0 + self._stages[num - 1].delta_y0
                item.plot()
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlim(-0.01, self._stages[self.stage_number - 1].x0 + self._stages[self.stage_number - 1].length + 0.01)
        plt.ylim(bottom=0)
        plt.show()

    def _compute_heat_drop_distribution(self):
        logging.info('%s ВЫЧИСЛЕНИЕ ПРЕДВАРИТЕЛЬНОГО РАСПРЕДЕЛЕНИЯ ТЕПЛОПЕРЕПАДОВ ПО СТУПЕНЯМ %s' % ('-'*10, '-'*10))
        u_av_squared_sum = 0
        for num, item in enumerate(self._stages):
            u_av_squared_sum += item.u_av ** 2
        for num, item in enumerate(self._stages):
            if num == 0:
                item.H0 = self.H_t * (1 + self.alpha) * item.u_av**2 / u_av_squared_sum + 0.5 * (item.mu * self.c21)**2
            else:
                item.H0 = self.H_t * (1 + self.alpha) * item.u_av ** 2 / u_av_squared_sum

    def plot_heat_drop_distribution(self, figsize=(6, 4), title='Распределение\n теплоперепадов'):
        logging.info('%s СОЗДАНИЕ ГРАФИКА РАСПРЕДЕЛЕНИЯ ТЕПЛОПЕРЕПАДОВ %s' % ('-'*10, '-'*10))
        x_arr = list(range(1, self.stage_number + 1))
        y_arr = [item.H0 for item in self._stages]
        plt.figure(figsize=figsize)
        plt.plot(x_arr, y_arr, 'o', color='red', markersize=12)
        plt.plot(x_arr, y_arr, color='red')
        plt.grid()
        plt.xticks(x_arr, x_arr)
        plt.xlim(0.7, self.stage_number + 0.3)
        plt.ylim(0, max(y_arr) + 2e4)
        plt.ylabel(r'$H_i$', fontsize=20)
        plt.xlabel(r'$Stage\ number$', fontsize=20)
        plt.title(title, fontsize=20)
        plt.show()

    def _compute_sigma_l(self):
        self.sigma_l = self.n ** 2 * self.k_n * self.last.A2

    def _compute_outlet_static_parameters(self):
        """
        Расчет статических параметров на выходе необходим для расчета силовой турбины (посленяя ступень
        рассчитываается по выходному статическому давлению), а также для определения коэффициента возврата теплоты
        """
        logging.info('%s ВЫЧИСЛЕНИЕ СТАТИЧЕСКИХ ПАРАМЕТРОВ НА ВЫХОДЕ ИХ ТУРБИНЫ %s' % ('-'*10, '-'*10))
        self.work_fluid.__init__()
        self.work_fluid.alpha = self.alpha_air
        self.rho_t_stag = self.p_t_stag / (self.work_fluid.R * self.T_t_stag)
        self.work_fluid.T1 = self.T_t_stag
        self.T_t_res = 1
        self._iter_number_static_par = 0
        while self.T_t_res >= self.precision:
            self._iter_number_static_par += 1
            logging.debug('%s _compute_outlet_static_parameters _iter_number = %s' %
                         (self.str(), self._iter_number_static_par))
            self.c_p_gas_t = self.work_fluid.c_p_av_int
            self.k_gas_t = self.work_fluid.k_av_int
            logging.debug('%s _compute_outlet_static_parameters k_gas_t = %s' % (self.str(), self.k_gas_t))
            self.a_cr_t = GasDynamicFunctions.a_cr(self.T_t_stag, self.work_fluid.k_av_int, self.work_fluid.R)
            G_outlet = self.G_turbine
            for i in self:
                G_outlet += i.g_cool * self.G_turbine

            def eps(c):
                return GasDynamicFunctions.eps_lam(c / self.a_cr_t, self.work_fluid.k)

            def func_to_solve(x):
                return [x[0] * eps(x[0]) - G_outlet / (self.rho_t_stag * self.last.A2)]

            x = fsolve(func_to_solve, np.array([200]))
            self.c_t = x[0]
            logging.debug('%s _compute_outlet_static_parameters c_t = %s' % (self.str(), self.c_t))
            self.lam_t = self.c_t / self.a_cr_t
            self.p_t = self.p_t_stag * GasDynamicFunctions.pi_lam(self.lam_t, self.work_fluid.k_av_int)
            self.T_t = self.T_t_stag * GasDynamicFunctions.tau_lam(self.lam_t, self.work_fluid.k_av_int)
            logging.debug('%s _compute_outlet_static_parameters T_t = %s' % (self.str(), self.T_t))
            self.T_t_res = abs(self.T_t - self.work_fluid.T2) / self.work_fluid.T2
            logging.debug('%s _compute_outlet_static_parameters T_t_res = %s' % (self.str(), self.T_t_res))
            self.work_fluid.T2 = self.T_t
        self.work_fluid.T1 = self.T_g_stag
        self.work_fluid.T2 = self.T_t
        self.k_gas = self.work_fluid.k_av_int
        self.c_p_gas = self.work_fluid.c_p_av_int
        self.H_t = self.c_p_gas * self.T_g_stag * (1 - (self.p_g_stag / self.p_t) ** ((1 - self.k_gas) / self.k_gas))
        self.eta_l = func.eta_turb_l(self.eta_t_stag, self.H_t_stag, self.H_t, self.c_t)
        self.alpha = (self.stage_number - 1) / (2 * self.stage_number) * (1 - self.eta_l) * \
                     ((self.p_g_stag / self.p_t) ** ((self.k_gas - 1) / self.k_gas) - 1)
        "Коэффициент возврата теплоты"

    def _compute_inlet_velocity(self):
        square = np.pi * self[0].D0 * self[0].l0
        rho_stag = self.p_g_stag / (self.T_g_stag * self.work_fluid.R)
        a_cr = GasDynamicFunctions.a_cr(self.T_g_stag, self.k_gas, self.work_fluid.R)
        c_in = fsolve(
            lambda X: [
                self.G_turbine -
                square * X[0] * rho_stag * GasDynamicFunctions.eps_lam(X[0] / a_cr, self.k_gas)
            ], np.array([100])
        )[0]
        self.c_inlet = c_in

    def compute(self):
        if self.auto_compute_heat_drop:
            self._specify_h01()
        else:
            self._compute_output()

    def _compute_output(self):
        self._compute_geometry()
        self._compute_sigma_l()
        self._compute_outlet_static_parameters()
        self._compute_inlet_velocity()

    def _specify_h01(self):
        logging.info('')
        logging.info(
            '%s РАСЧЕТ ГЕОМЕТРИИ ТУРИБНЫ С УТОЧНЕНИЕМ ТЕПЛОПЕРЕПАДА НА СА ПЕРВОЙ СТУПЕНИ %s\n' % ('#' * 15, '#' * 15))
        dh01_rel = 1.
        H01 = self.H01
        iter_number = 0
        while dh01_rel >= self.precision:
            iter_number += 1
            logging.info('%s ИТЕРАЦИЯ %s %s\n' % ('-' * 20, iter_number, '-' * 20))
            logging.debug('specify_h01 iter_number = %s' % iter_number)
            self.H01 = H01
            logging.debug('specify_h01 H01 = %s' % H01)
            self._compute_output()
            self._compute_heat_drop_distribution()
            dh01_rel = abs(self.first.H0 - self.H01) / self.H01
            logging.debug('specify_h01 dh01_rel = %s' % dh01_rel)
            H01 = self.first.H0
            logging.info('')


if __name__ == '__main__':
    pass





