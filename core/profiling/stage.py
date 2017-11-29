import enum
import typing
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from ..average_streamline.stage_geom import InvalidStageSizeValue
from scipy.integrate import quad
from gas_turbine_cycle.tools.gas_dynamics import GasDynamicFunctions
from .section import BladeSection
from mpl_toolkits.mplot3d import Axes3D


class ProfilingType(enum.Enum):
    ConstantCirculation = 0
    ConstantAngle = 1


class StageParametersRadialDistribution:
    def __init__(self, profiling_type: ProfilingType, p0_stag: typing.Callable[[float], float],
                 T0_stag: typing.Callable[[float], float], c0: typing.Callable[[float], float],
                 alpha0: typing.Callable[[float], float],
                 c_p, k, D1_in, D1_av, D1_out, n,
                 c1_av, alpha1_av, L_u_av, c2_a_av, c2_u_av, phi=1.0, psi=1.0):
        """
        :param profiling_type: ProfilingType. \n
                Тип профилирования: по постоянной циркуляции или постоянному углу
        :param p0_stag: callable. \n
                Вызываемый объект, возвращающий значение давления торможения на входе в ступень для заданного радиуса.
        :param T0_stag: callable. \n
                Вызываемый объект, возвращающий значение температуры торможения на входе в ступень
                для заданного радиуса.
        :param c0: callable. \n
                Вызываемый объект, возвращающий значение скорости на входе в ступень для заданного радиуса.
        :param alpha0: callable. \n
                Профиль угла потока на входе в СА.
        :param c_p: float. \n
                Средняя теплоемкость в ступени.
        :param k: float. \n
                Средний показатель адиабаты в ступени.
        :param D1_in: float. \n
                Внутренний диаметр на входе в РК.
        :param D1_av: float. \n
                Средний диаметр на входе в РК.
        :param D1_out: float. \n
                Наружний диаметр на входе в РК.
        :param n: float. \n
                Частота вращения ротора турбины.
        :param c1_av: float. \n
                Средняя скорость на среднем радиусе на входе в РК.
        :param alpha1_av: float. \n
                Угол потока на среднем радиусе на входе в РК.
        :param L_u_av: float. \n
                Работа на окружности колеса.
        :param c2_a_av: float. \n
                Осевая скорость на выходе из РК на средней линии тока.
        :param c2_u_av: float. \n
                Окружная скорость на выходе из РК на средней линии тока.
        :param phi:
        :param psi:

        NOTE: При расчете распределения параметров принимаются следующие допущения:
                1. Течение рассматривается идеальным.
                2. Теплоемкость принимается постоянной и равной средней из расчета по средней линии тока.
                3. Работа на окружности колеса принимается постоянной.
                4. Изменение энтальпии по параметрам торможения принимается равным работе на окружности колеса.
              Осевая скорость на выходе из РК находится интегрированием системы дииференциальных уравнений
              радиального равновесия и Бернулли.
        """
        self.profiling_type = profiling_type
        self.p0_stag = p0_stag
        self.T0_stag = T0_stag
        self.alpha0 = alpha0
        self.c0 = c0
        self.phi = phi
        self.psi = psi
        self.c_p = c_p
        self.k = k
        self.D1_in = D1_in
        self.D1_av = D1_av
        self.D1_out = D1_out
        self.c1_av = c1_av
        self.alpha1_av = alpha1_av
        self.n = n
        self.L_u_av = L_u_av
        self.c2_a_av = c2_a_av
        self.c2_u_av = c2_u_av

    @property
    def R(self):
        return self.c_p * (self.k - 1) / self.k

    def T0(self, r):
        return self.T0_stag(r) - self.c0(r) ** 2 / (2 * self.c_p)

    def c1_u(self, r):
        c1_u_av = self.c1_av * np.cos(self.alpha1_av)
        if self.profiling_type == ProfilingType.ConstantCirculation:
            return 0.5 * self.D1_av * c1_u_av / r
        elif self.profiling_type == ProfilingType.ConstantAngle:
            return c1_u_av * (0.5 * self.D1_av / r) ** (np.cos(self.alpha1_av) ** 2)

    def c1_a(self, r):
        c1_a_av = self.c1_av * np.sin(self.alpha1_av)
        if self.profiling_type == ProfilingType.ConstantCirculation:
            return c1_a_av
        elif self.profiling_type == ProfilingType.ConstantAngle:
            return c1_a_av * (0.5 * self.D1_av / r) ** (np.cos(self.alpha1_av) ** 2)

    def c1(self, r):
        return np.sqrt(self.c1_a(r) ** 2 + self.c1_u(r) ** 2)

    def alpha1(self, r):
        if self.c1_a(r) / self.c1(r) > 1:
            raise InvalidStageSizeValue('c1_a must be less than c1')
        return np.arcsin(self.c1_a(r) / self.c1(r))

    def H_s(self, r):
        return self.c1(r) ** 2 / (2 * self.phi ** 2)

    def p1(self, r):
        return self.p0_stag(r) * (1 - self.H_s(r) / (self.T0_stag(r) * self.c_p)) ** (self.k / (self.k - 1))

    def T1_ad(self, r):
        return self.T0_stag(r) - self.H_s(r) / self.c_p

    def T1(self, r):
        return self.T0_stag(r) - self.H_s(r) * self.phi ** 2 / self.c_p

    def p1_balance_equation(self, r):
        """p2, определенное из уравнения радиального равновесия"""
        def int_func(r):
            return self.c1_u(r) ** 2 / (self.R * self.T1(r) * r)
        return self.p1(self.D1_av / 2) * np.exp(quad(int_func, 0.5 * self.D1_av, r)[0])

    def u(self, r):
        return 2 * np.pi * r * self.n / 60

    def L_u(self, r):
        return self.L_u_av

    def T2_stag(self, r):
        return self.T0_stag(r) - self.L_u(r) / self.c_p

    def c2_u(self, r):
        return (self.L_u(r) - self.c1_u(r) * self.u(r)) / self.u(r)

    def c2_a(self, r):
        def int_func(r):
            return self.c2_u(r) ** 2 / r
        return np.sqrt(self.c2_a_av ** 2 + self.c2_u_av ** 2 - self.c2_u(r) ** 2 -
                       2 * quad(int_func, 0.5 * self.D1_av, r)[0])

    def c2(self, r):
        return np.sqrt(self.c2_a(r) ** 2 + self.c2_u(r) ** 2)

    def alpha2(self, r):
        if self.c2_u(r) >= 0:
            return np.arctan(self.c2_a(r) / self.c2_u(r))
        else:
            return np.pi + np.arctan(self.c2_a(r) / self.c2_u(r))

    def w2_u(self, r):
        return self.c2_u(r) + self.u(r)

    def w2(self, r):
        return np.sqrt(self.w2_u(r) ** 2 + self.c2_a(r) ** 2)

    def w1(self, r):
        a = self.c1(r) ** 2 + self.u(r) ** 2 - 2 * self.u(r) * self.c1(r) * self.alpha1(r)
        if a < 0:
            raise InvalidStageSizeValue('w1 can not be calculated')
        return np.sqrt(self.c1(r) ** 2 + self.u(r) ** 2 - 2 * self.u(r) * self.c1(r) * np.cos(self.alpha1(r)))

    def T1_w_stag(self, r):
        return self.T1(r) + self.w1(r) ** 2 / (2 * self.c_p)

    def beta1(self, r):
        if self.c1(r) * np.cos(self.alpha1(r)) - self.u(r) >= 0:
            return np.arctan(self.c1_a(r) / (self.c1(r) * np.cos(self.alpha1(r)) - self.u(r)))
        else:
            return np.pi + np.arctan(self.c1_a(r) / (self.c1(r) * np.cos(self.alpha1(r)) - self.u(r)))

    def w1_a(self, r):
        return self.w1(r) * np.sin(self.beta1(r))

    def w1_u(self, r):
        return self.w1(r) * np.cos(self.beta1(r))

    def beta2(self, r):
        if self.c2_a(r) / self.w2(r) > 1:
            raise InvalidStageSizeValue('c2_a must be less than w2')
        return np.arcsin(self.c2_a(r) / self.w2(r))

    def w2_a(self, r):
        return self.w2(r) * np.sin(self.beta2(r))

    def H_l(self, r):
        return 0.5 * ((self.w2(r) / self.psi) ** 2 - self.w1(r) ** 2)

    def p1_w_stag(self, r):
        return self.p1(r) / GasDynamicFunctions.pi_lam(self.lam_w1(r), self.k)

    def p2(self, r):
        return self.p1(r) * (1 - self.H_l(r) / (self.c_p * self.T1(r))) ** (self.k / (self.k - 1))

    def p2_w_stag(self, r):
        return self.p2(r) / GasDynamicFunctions.pi_lam(self.lam_w2(r), self.k)

    def p2_balance_equation(self, r):
        """p2, определенное из уравнения радиального равновесия"""
        def int_func(r):
            return self.c2_u(r) ** 2 / (self.R * self.T2(r) * r)
        return self.p2(self.D1_av / 2) * np.exp(quad(int_func, 0.5 * self.D1_av, r)[0])

    def p2_stag(self, r):
        return self.p2(r) / GasDynamicFunctions.pi_lam(self.c2(r) /
                                                       GasDynamicFunctions.a_cr(self.T2_stag(r), self.k, self.R), self.k)

    def T2(self, r):
        return self.T1(r) - (self.w2(r) ** 2 - self.w1(r) ** 2) / (2 * self.c_p)

    def H0(self, r):
        return self.c_p * self.T0_stag(r) * (1 - (self.p0_stag(r) / self.p2(r)) ** ((1 - self.k) / self.k))

    def rho(self, r):
        return self.H_l(r) * self.T1_ad(r) / (self.H0(r) * self.T1(r))

    def M_c0(self, r):
        return self.c0(r) / np.sqrt(self.k * self.R * self.T0(r))

    def M_c1(self, r):
        return self.c1(r) / np.sqrt(self.k * self.R * self.T1(r))

    def M_w1(self, r):
        return self.w1(r) / np.sqrt(self.k * self.R * self.T1(r))

    def M_w2(self, r):
        return self.w2(r) / np.sqrt(self.k * self.R * self.T2(r))

    def lam_c0(self, r):
        a_cr = np.sqrt(2 * self.k / (self.k + 1) * self.R * self.T0_stag(r))
        return self.c0(r) / a_cr

    def lam_c1(self, r):
        a_cr = np.sqrt(2 * self.k / (self.k + 1) * self.R * self.T0_stag(r))
        return self.c1(r) / a_cr

    def lam_w1(self, r):
        a_cr = np.sqrt(2 * self.k / (self.k + 1) * self.R * self.T1_w_stag(r))
        return self.w1(r) / a_cr

    def lam_w2(self, r):
        a_cr = np.sqrt(2 * self.k / (self.k + 1) * self.R * self.T1_w_stag(r))
        return self.w2(r) / a_cr

    def plot_parameter_distribution(self, par_names: typing.List[str], colors=None, figsize=(9, 7), filename=None):
        r_in = 0.5 * self.D1_in
        r_out = 0.5 * self.D1_out
        r_av = 0.5 * self.D1_av
        get_atr = object.__getattribute__
        y1 = np.array(np.linspace(r_in, r_out, 100)) / r_av
        y = np.array(np.linspace(r_in, r_out, 100))
        deg = np.pi / 180
        plt.figure(figsize=figsize)
        for n, item in enumerate(par_names):
            par = get_atr(self, item)
            x = [par(i) for i in y]
            if item.startswith('alpha') or item.startswith('beta') or item.startswith('delta') or \
                           item.startswith('gamma'):
                x = [i / deg for i in x]
            if colors:
                plt.plot(x, y1, linewidth=2, color=colors[n], label=item)
            else:
                plt.plot(x, y1, linewidth=2, label=item)
        plt.legend(fontsize=16)
        plt.ylabel(r'$\frac{r}{r_{av}}$', fontsize=22)
        plt.grid()
        if filename:
            plt.savefig(filename)
        plt.show()

    def plot_velocity_triangles(self, r_rel=(0, 0.5, 1), figsize=(8, 8), filename=None,):
        r_arr = [0.5 * (self.D1_in + i * (self.D1_out - self.D1_in)) for i in r_rel]
        title = [r'$r_{rel} = %s$' % i for i in r_rel]
        for (n, i) in enumerate(r_arr):
            plt.figure(figsize=figsize)
            x_in = np.array([0, -self.c1_u(i), -self.c1_u(i) + self.u(i), 0])
            y_in = np.array([self.c1_a(i), 0, 0, self.c1_a(i)])
            x_out = np.array([0, self.c2_u(i), self.c2_u(i) + self.u(i), 0])
            y_out = np.array([self.c1_a(i), self.c1_a(i) - self.c2_a(i), self.c1_a(i) - self.c2_a(i), self.c1_a(i)])
            plt.plot(x_in, y_in, linewidth=2, color='red', label='inlet')
            plt.plot(x_out, y_out, linewidth=2, color='blue', label='outlet')
            plt.xlim(-self.c1_u(i), self.c2_u(i) + self.u(i))
            plt.ylim(-max(self.c1_a(i), self.c1_u(i)), max(self.c1_a(i), self.c2_u(i) + self.u(i)))
            plt.grid()
            plt.title(title[n], fontsize=20)
            plt.legend()
            if filename:
                plt.savefig('%s_%s' % (filename, n))
        plt.show()


class StageProfiler(StageParametersRadialDistribution):
    def __init__(self, profiling_type: ProfilingType, p0_stag: typing.Callable[[float], float],
                 T0_stag: typing.Callable[[float], float], c0: typing.Callable[[float], float],
                 alpha0: typing.Callable[[float], float],
                 c_p, k, D1_in, D1_av, D1_out, n,
                 c1_av, alpha1_av, L_u_av, c2_a_av, c2_u_av,
                 b_a_sa, b_a_rk,
                 delta_a_sa, delta_a_rk,
                 t_rel_av_sa=0.8, t_rel_av_rk=0.7,
                 x0=0., y0=0.,
                 auto_sections_par: bool = True,
                 r1_rel_sa: typing.Callable[[float], float] = lambda r_rel: 0.04,
                 r1_rel_rk: typing.Callable[[float], float] = lambda r_rel: 0.04,
                 s2_sa=0.001,
                 s2_rk=0.001,
                 gamma2_sa=np.radians([3])[0],
                 gamma2_rk=np.radians([3])[0],
                 gamma1_k_rel_sa: typing.Callable[[float], float] = lambda r_rel: 0.3,
                 gamma1_k_rel_rk: typing.Callable[[float], float] = lambda r_rel: 0.3,
                 gamma2_k_rel_sa: typing.Callable[[float], float] = lambda r_rel: 0.3,
                 gamma2_k_rel_rk: typing.Callable[[float], float] = lambda r_rel: 0.3,
                 gamma1_sa: typing.Callable[[float], float] = None,
                 gamma1_rk: typing.Callable[[float], float] = None,
                 center_point_sa: typing.Callable[[float], float] = lambda r_rel: 0.5,
                 center_point_rk: typing.Callable[[float], float] = lambda r_rel: 0.5,
                 section_num: int=3, pnt_cnt=20,
                 center: bool =True,
                 phi=1.0, psi=1.0):
        """
        :param profiling_type: ProfilingType. \n
                Тип профилирования: по постоянной циркуляции или постоянному углу
        :param p0_stag: callable. \n
                Вызываемый объект, возвращающий значение давления торможения на входе в ступень для заданного радиуса.
        :param T0_stag: callable. \n
                Вызываемый объект, возвращающий значение температуры торможения на входе в ступень
                для заданного радиуса.
        :param c0: callable. \n
                Вызываемый объект, возвращающий значение скорости на входе в ступень для заданного радиуса.
        :param alpha0: callable. \n
                Профиль угла потока на входе в СА.
        :param c_p: float. \n
                Средняя теплоемкость в ступени.
        :param k: float. \n
                Средний показатель адиабаты в ступени.
        :param D1_in: float. \n
                Внутренний диаметр на входе в РК.
        :param D1_av: float. \n
                Средний диаметр на входе в РК.
        :param D1_out: float. \n
                Наружний диаметр на входе в РК.
        :param n: float. \n
                Частота вращения ротора турбины.
        :param c1_av: float. \n
                Средняя скорость на среднем радиусе на входе в РК.
        :param alpha1_av: float. \n
                Угол потока на среднем радиусе на входе в РК.
        :param L_u_av: float. \n
                Работа на окружности колеса.
        :param c2_a_av: float. \n
                Осевая скорость на выходе из РК на средней линии тока.
        :param c2_u_av: float. \n
                Окружная скорость на выходе из РК на средней линии тока.
        :param b_a_sa: float. \n
                Осевая ширина СА.
        :param b_a_rk: float. \n
                Осевая ширина РК.
        :param delta_a_sa: float. \n
                Величина осевого зазора после СА.
        :param delta_a_rk: float. \n
                Величина осевого зазора после РК.
        :param t_rel_av_sa: float. \n
                Относительный шаг на среднем радиусе СА.
        :param t_rel_av_rk: float. \n
                Относительный шаг на среднем радиусе РК.
        :param x0: float. \n
                Координата входного полюса средней линии втулочного профиля СА.
        :param y0: float. \n
                Координата входного полюса средней линии втулочного профиля СА.
        :param auto_sections_par: bool, optional. \n
                Если равен True, то параметры сечений (углы отставания, скругления,
                углы между касательными на входе и выходе) будут определяться автоматически по интегральным
                показателям для ступени. Если равен False, то параметры каждого сечения можно задавать вручную или
                оставить равными по умолчанию.
        :param r1_rel_sa: callable, optional. \n
                Зависимость относительного радиуса входной кромки СА (по отношению к осевой ширине) от
                относительного радиуса сечения.
        :param r1_rel_rk: callable, optional. \n
                Зависимость относительного радиуса входной кромки РК от относительного радиуса сечения.
        :param s2_sa: float, optional. \n
                Радиус скругления выходной кромки СА.
        :param s2_rk: float, optional. \n
                Радиус скругления выходной кромки РК.
        :param gamma2_sa: float, optional. \n
                Угол между касательными к профилю на выходе из СА.
        :param gamma2_rk: float, optional. \n
                Угол между касательными к профилю на выходе из РК.
        :param gamma1_k_rel_sa: callable, optional. \n
                Зависимость отношения угла gamma1_k к углу gamma1 в СА от относительного радиуса сечения.
        :param gamma1_k_rel_rk: callable, optional. \n
                Зависимость отношения угла gamma1_k к углу gamma1 в РК от относительного радиуса сечения.
        :param gamma2_k_rel_sa: callable, optional. \n
                Зависимость отношения угла gamma2_k к углу gamma2 в СА от относительного радиуса сечения.
        :param gamma2_k_rel_rk: callable, optional. \n
                Зависимость отношения угла gamma2_k к углу gamma2 в РК.
        :param gamma1_sa: callable, optional. \n
                Зависимость угла gamma1 на СА от относительного радиуса сечения. По умолчанию равень None,
                в таком случае он будет определятся в зависимости от угла лопатки на входе.
        :param gamma1_rk: callable, optional. \n
                Зависимость угла gamma1 на РК от относительного радиуса сечения. По умолчанию равень None,
                в таком случае он будет определятся в зависимости от угла лопатки на входе.
        :param center_point_sa: list, optional. \n
                Позиция, заданная в относительном виде, центрального полюса средней линии профиля СА.
        :param center_point_rk: list, optional. \n
                Позиция, заданная в относительном виде, центрального полюса средней линии профиля РК.
        :param section_num: int, optional. \n
                Число рассчитываемых сечений.
        :param pnt_cnt: int, optional. \n
                Количество рассчитываемых на линиях профиля каждого сечения точек.
        :param center: bool, optional. \n
                Если равен True, то будет произведено обнуление выносов, если равен False, то не будет.
        :param phi:
        :param psi:

        NOTE: Сечения распределяются равномерно по радиусу от втулки до периферии на входе в РК.
        """
        StageParametersRadialDistribution.__init__(self, profiling_type, p0_stag, T0_stag, c0, alpha0,
                                                   c_p, k, D1_in, D1_av, D1_out, n, c1_av,
                                                   alpha1_av, L_u_av, c2_a_av, c2_u_av, phi, psi)
        assert section_num % 2 == 1, 'The section number must be odd.'
        self._section_num = section_num
        self.b_a_sa = b_a_sa
        self.t_rel_av_sa = t_rel_av_sa
        self.t_rel_av_rk = t_rel_av_rk
        self.b_a_rk = b_a_rk
        self.pnt_cnt = pnt_cnt
        self.x0 = x0
        self.y0 = y0
        self.delta_a_sa = delta_a_sa
        self.delta_a_rk = delta_a_rk
        self.r1_rel_sa = r1_rel_sa
        self.r1_rel_rk = r1_rel_rk
        self.gamma2_sa = gamma2_sa
        self.gamma2_rk = gamma2_rk
        self.s2_sa = s2_sa
        self.s2_rk = s2_rk
        self.gamma1_k_rel_sa = gamma1_k_rel_sa
        self.gamma1_k_rel_rk = gamma1_k_rel_rk
        self.gamma2_k_rel_sa = gamma2_k_rel_sa
        self.gamma2_k_rel_rk = gamma2_k_rel_rk
        self.gamma1_sa = gamma1_sa
        self.gamma1_rk = gamma1_rk

        self.center_point_sa = center_point_sa
        self.center_point_rk = center_point_rk

        self.center: bool = center
        self.auto_sections_par = auto_sections_par
        self._sa_sections: typing.List[BladeSection] = [BladeSection() for _ in range(section_num)]
        self._rk_sections: typing.List[BladeSection] = [BladeSection() for _ in range(section_num)]
        self.x0_next = None
        "Координата x0 для следующей ступени"
        self.z_sa = None
        self.z_rk = None
        self.t_av_sa = None
        self.t_av_rk = None

    @property
    def section_num(self):
        return self._section_num

    @section_num.setter
    def section_num(self, value: int):
        self._section_num = value
        self._sa_sections: typing.List[BladeSection] = [BladeSection() for _ in range(self.section_num)]
        self._rk_sections: typing.List[BladeSection] = [BladeSection() for _ in range(self.section_num)]

    @property
    def sa_sections(self) -> typing.List[BladeSection]:
        return self._sa_sections

    @property
    def rk_sections(self) -> typing.List[BladeSection]:
        return self._rk_sections

    def _get_radius(self):
        res = np.linspace(0.5 * self.D1_in, 0.5 * self.D1_out, self.section_num)
        return np.array(res)

    def get_delta(self, M):
        """Возвращает значение угла отставания в зависимости от величина числа Маха на входе в профиль."""
        mach = np.array([0, 0.1, 0.2, 0.6, 0.8, 0.95, 1.4, 1.9, 3])
        delta = np.radians([0, 0, 0.1, 0.33, 0.7, 1.5, 3, 5, 5])
        res = interp1d(mach, delta)(M)
        return res

    def get_gamma1(self, angle1_l):
        """Возвращает значение угла между касательными к контуру у входной кромки."""
        angle1_l_arr = np.radians([10, 30, 150, 170])
        gamma1_arr = np.radians([48, 40, 8, 4])
        res = interp1d(angle1_l_arr, gamma1_arr)(angle1_l)
        return res

    def init_sections(self):
        """Инициализация входных параметров всех сечений"""
        radiuses = self._get_radius()
        for section, radius, n in zip(self.sa_sections, radiuses, range(self.section_num)):
            radius_rel = (radius - 0.5 * self.D1_in) / (0.5 * (self.D1_out - self.D1_in))
            if self.auto_sections_par:
                section.b_a = self.b_a_sa
                section.angle1 = self.alpha0(radius)
                section.delta1 = self.get_delta(self.M_c0(radius))
                section.angle2 = self.alpha1(radius)
                section.delta2 = self.get_delta(self.M_c1(radius))
                section.r1 = self.r1_rel_sa(radius_rel) * self.b_a_sa
                section.x0_av = self.x0
                section.y0_av = self.y0
                section.s2 = self.s2_sa
                if self.gamma1_sa is not None:
                    section.gamma1_k = self.gamma1_sa(radius_rel) * self.gamma1_k_rel_sa(radius_rel)
                    section.gamma1_s = self.gamma1_sa(radius_rel) * (1 - self.gamma1_k_rel_sa(radius_rel))
                else:
                    section.gamma1_k = self.get_gamma1(section.angle1_l) * self.gamma1_k_rel_sa(radius_rel)
                    section.gamma1_s = self.get_gamma1(section.angle1_l) * (1 - self.gamma1_k_rel_sa(radius_rel))
                section.gamma2_k = self.gamma2_sa * self.gamma2_k_rel_sa(radius_rel)
                section.gamma2_s = self.gamma2_sa * (1 - self.gamma2_k_rel_sa(radius_rel))
                section.convex = 'left'
                section.pnt_count = self.pnt_cnt
                section.center_point_pos = self.center_point_sa(radius_rel)
            else:
                section.b_a = self.b_a_sa
                section.angle1 = self.alpha0(radius)
                section.angle2 = self.alpha1(radius)
                section.convex = 'left'
                section.pnt_count = self.pnt_cnt
                section.x0_av = self.x0
                section.y0_av = self.y0

        for section, radius, n in zip(self.rk_sections, radiuses, range(self.section_num)):
            radius_rel = (radius - 0.5 * self.D1_in) / (0.5 * (self.D1_out - self.D1_in))
            if self.auto_sections_par:
                section.b_a = self.b_a_rk
                section.angle1 = self.beta1(radius)
                section.delta1 = self.get_delta(self.M_w1(radius))
                section.angle2 = self.beta2(radius)
                section.delta2 = self.get_delta(self.M_w2(radius))
                section.r1 = self.r1_rel_rk(radius_rel) * self.b_a_rk
                section.x0_av = self.x0 + self.b_a_sa + self.delta_a_sa
                section.y0_av = self.y0
                section.s2 = self.s2_rk
                if self.gamma1_rk is not None:
                    section.gamma1_k = self.gamma1_rk(radius_rel) * self.gamma1_k_rel_rk(radius_rel)
                    section.gamma1_s = self.gamma1_rk(radius_rel) * (1 - self.gamma1_k_rel_rk(radius_rel))
                else:
                    section.gamma1_k = self.get_gamma1(section.angle1_l) * self.gamma1_k_rel_rk(radius_rel)
                    section.gamma1_s = self.get_gamma1(section.angle1_l) * (1 - self.gamma1_k_rel_rk(radius_rel))
                section.gamma2_k = self.gamma2_rk * self.gamma2_k_rel_rk(radius_rel)
                section.gamma2_s = self.gamma2_rk * (1 - self.gamma2_k_rel_rk(radius_rel))
                section.convex = 'right'
                section.pnt_count = self.pnt_cnt
                section.center_point_pos = self.center_point_rk(radius_rel)
            else:
                section.b_a = self.b_a_rk
                section.angle1 = self.beta1(radius)
                section.angle2 = self.beta2(radius)
                section.x0_av = self.x0 + self.b_a_sa + self.delta_a_sa
                section.y0_av = self.y0
                section.convex = 'right'
                section.pnt_count = self.pnt_cnt

    def _get_average_sections(self):
        sa_section = self.sa_sections[int(self.section_num / 2)]
        rk_section = self.rk_sections[int(self.section_num / 2)]
        return sa_section, rk_section

    def _compute_step(self):
        """Рассчитывает величины абсолютных и относительных шагов."""
        sa_av_section, rk_av_section = self._get_average_sections()

        self.z_sa = int(np.pi * self.D1_av / (self.t_rel_av_sa * sa_av_section.chord_length))
        self.t_av_sa = np.pi * self.D1_av / self.z_sa
        self.t_rel_av_sa = self.t_av_sa / sa_av_section.chord_length

        self.z_rk = int(np.pi * self.D1_av / (self.t_rel_av_rk * rk_av_section.chord_length))
        self.t_av_rk = np.pi * self.D1_av / self.z_rk
        self.t_rel_av_rk = self.t_av_rk / rk_av_section.chord_length

        radiuses = self._get_radius()
        for section, radius in zip(self.sa_sections, radiuses):
            section.t = 2 * np.pi * radius / self.z_sa
        for section, radius in zip(self.rk_sections, radiuses):
            section.t = 2 * np.pi * radius / self.z_rk

    def compute_sections(self):
        for section in self.sa_sections:
            section.compute_profile()
        for section in self.rk_sections:
            section.compute_profile()

        if self.center:
            # координаты центра среднего сечения СА, или ближайшего к центру сверху, если число сечений четное
            x_c_av_sa = self.sa_sections[int(self.section_num / 2)].x_c
            y_c_av_sa = self.sa_sections[int(self.section_num / 2)].y_c
            # координаты центра среднего сечения РК, или ближайшего к центру сверху, если число сечений четное
            x_c_av_rk = self.rk_sections[int(self.section_num / 2)].x_c
            y_c_av_rk = self.rk_sections[int(self.section_num / 2)].y_c

            for section in self.sa_sections:
                section.move_to(x_c_av_sa, y_c_av_sa)
            for section in self.rk_sections:
                section.move_to(x_c_av_rk, y_c_av_rk)

        self._compute_step()
        self.x0_next = self.x0 + self.delta_a_sa + self.delta_a_rk + self.b_a_sa + self.b_a_rk

    def plot_profile_2d(self, section_number=0, width_rel=4, figsize=(8, 5)):
        sa_section = self.sa_sections[section_number]
        t_sa = sa_section.t
        rk_section = self.rk_sections[section_number]
        t_rk = rk_section.t
        radius = self._get_radius()[section_number]
        radius_rel = (radius - 0.5 * self.D1_in) / (0.5 * self.D1_out - 0.5 * self.D1_in)
        i = 0
        y_max = width_rel * t_sa
        plt.figure(figsize=figsize)
        while i * t_sa < y_max:
            plt.plot(sa_section.y_in_edge + i * t_sa, sa_section.x_in_edge, lw=1, color='red')
            plt.plot(sa_section.y_out_edge + i * t_sa, sa_section.x_out_edge, lw=1, color='red')
            plt.plot(sa_section.y_s + i * t_sa, sa_section.x_s, lw=1, color='red')
            plt.plot(sa_section.y_k + i * t_sa, sa_section.x_k, lw=1, color='red')
            plt.plot(sa_section.y_av + i * t_sa, sa_section.x_av, lw=0.5, color='red', linestyle=':')
            i += 1

        i = 0
        while i * t_rk < y_max:
            plt.plot(rk_section.y_in_edge + i * t_rk, rk_section.x_in_edge, lw=1, color='blue')
            plt.plot(rk_section.y_out_edge + i * t_rk, rk_section.x_out_edge, lw=1, color='blue')
            plt.plot(rk_section.y_s + i * t_rk, rk_section.x_s, lw=1, color='blue')
            plt.plot(rk_section.y_k + i * t_rk, rk_section.x_k, lw=1, color='blue')
            plt.plot(rk_section.y_av + i * t_rk, rk_section.x_av, lw=0.5, color='blue', linestyle=':')
            i += 1

        plt.grid()
        plt.title(r'$\frac{r - r_{вт}}{r_{п} - r_{вт}} = %.3f$' % radius_rel, fontsize=14)
        plt.show()

    def plot_profile_3d(self, axes=None):
        radiuses = self._get_radius()
        show = not bool(axes)
        if not axes:
            fig = plt.figure(figsize=(8, 6))
            axes = Axes3D(fig)
        for n, radius in enumerate(radiuses):
            sa_section = self.sa_sections[n]
            rk_section = self.rk_sections[n]
            axes.plot(xs=sa_section.x_in_edge, ys=sa_section.y_in_edge, zs=radius, color='red', lw=0.7)
            axes.plot(xs=sa_section.x_out_edge, ys=sa_section.y_out_edge, zs=radius, color='red', lw=0.7)
            axes.plot(xs=sa_section.x_s, ys=sa_section.y_s, zs=radius, color='red', lw=0.7)
            axes.plot(xs=sa_section.x_k, ys=sa_section.y_k, zs=radius, color='red', lw=0.7)

            axes.plot(xs=rk_section.x_in_edge, ys=rk_section.y_in_edge, zs=radius, color='blue', lw=0.7)
            axes.plot(xs=rk_section.x_out_edge, ys=rk_section.y_out_edge, zs=radius, color='blue', lw=0.7)
            axes.plot(xs=rk_section.x_s, ys=rk_section.y_s, zs=radius, color='blue', lw=0.7)
            axes.plot(xs=rk_section.x_k, ys=rk_section.y_k, zs=radius, color='blue', lw=0.7)
        if show:
            plt.show()

    @classmethod
    def _plot_section(cls, section: BladeSection, label: str):
        y_s = list(section.y_s)
        y_s.reverse()
        x_s = list(section.x_s)
        x_s.reverse()
        y = list(section.y_in_edge) + list(section.y_k) + list(section.y_out_edge) + y_s
        x = list(section.x_in_edge) + list(section.x_k) + list(section.x_out_edge) + x_s
        plt.plot(y, x, lw=1, label=label)
        plt.plot(section.y_c, section.x_c, linestyle='', marker='o', ms=8, mfc='black', color='black')

    def plot_sa_sections(self, r_arr=(0, 0.5, 1.0), figsize=(6, 4)):
        plt.figure(figsize=figsize)
        radiuses = (self._get_radius() - 0.5 * self.D1_in) / (0.5 * (self.D1_out - self.D1_in))
        for section, rad in zip(self.sa_sections, radiuses):
            if rad in r_arr:
                self._plot_section(section, label=r'$\bar{r} = %.3f$' % rad)
        plt.grid()
        plt.legend(fontsize=10)
        plt.show()

    def plot_rk_sections(self, r_arr=(0, 0.5, 1.0), figsize=(6, 4)):
        plt.figure(figsize=figsize)
        radiuses = (self._get_radius() - 0.5 * self.D1_in) / (0.5 * (self.D1_out - self.D1_in))
        for section, rad in zip(self.rk_sections, radiuses):
            if rad in r_arr:
                self._plot_section(section, label=r'$\bar{r} = %.3f$' % rad)
        plt.grid()
        plt.legend(fontsize=10)
        plt.show()


if __name__ == '__main__':
    pass
