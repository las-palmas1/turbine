import enum
import typing
import numpy as np
from gas_turbine_cycle.gases import KeroseneCombustionProducts
import matplotlib.pyplot as plt
from core.average_streamline.stage_geom import InvalidStageSizeValue
from core.average_streamline.turbine import TurbineType, Turbine
from scipy.integrate import quad
from gas_turbine_cycle.tools.gas_dynamics import GasDynamicFunctions
from core.profiling.section import BladeSection


class ProfilingType(enum.Enum):
    ConstantCirculation = 0
    ConstantAngle = 1


class StageParametersRadialDistribution:
    def __init__(self, profiling_type: ProfilingType, p0_stag: typing.Callable[[float], float],
                 T0_stag: typing.Callable[[float], float], c0: typing.Callable[[float], float],
                 c_p, k, D1_in, D1_av, D1_out, n,
                 c1_av, alpha1_av, L_u_av, c2_a_av, c2_u_av, phi=1.0, psi=1.0):
        """
        :param profiling_type: ProfilingType. \n
                Тип профилирования: по постоянной циркуляции или постоянному углу
        :param p0_stag: callable. \n
                Вызываемый объект, возвращающий значение давления торможения на входе для заданного радиуса.
        :param T0_stag: callable. \n
                Вызываемый объект, возвращающий значение температуры торможения на входе для заданного радиуса.
        :param c0: callable. \n
                Вызываемый объект, возвращающий значение скорости на входе в ступень для заданного радиуса.
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
        self.R = self.c_p * (self.k - 1) / self.k

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
                       2 * quad(int_func, 0.5 * self.D1_av, r)[0] +
                       2 * self.c_p * (self.T2_stag(r) - self.T2_stag(0.5 * self.D1_av)))

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

    def T1_w(self, r):
        return self.T0(r) + self.w1(r) / (2 * self.c_p)

    def beta1(self, r):
        if self.c1(r) * np.cos(self.alpha1(r)) - self.u(r) >= 0:
            return np.arctan(self.c1_a(r) / (self.c1(r) * np.cos(self.alpha1(r)) - self.u(r)))
        else:
            return np.pi + np.arctan(self.c1_a(r) / (self.c1(r) * np.cos(self.alpha1(r)) - self.u(r)))

    def beta2(self, r):
        if self.c2_a(r) / self.w2(r) > 1:
            raise InvalidStageSizeValue('c2_a must be less than w2')
        return np.arcsin(self.c2_a(r) / self.w2(r))

    def H_l(self, r):
        return 0.5 * ((self.w2(r) / self.psi) ** 2 - self.w1(r) ** 2)

    def p2(self, r):
        return self.p1(r) * (1 - self.H_l(r) / (self.c_p * self.T1(r))) ** (self.k / (self.k - 1))

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
        return self.c_p * self.T0_stag * (1 - (self.p0_stag(r) / self.p2(r)) ** ((1 - self.k) / self.k))

    def rho(self, r):
        return self.H_l(r) * self.T1_ad(r) / (self.H0(r) * self.T1(r))

    def M_c1(self, r):
        return self.c1(r) / np.sqrt(self.k * self.R * self.T1(r))

    def M_w2(self, r):
        return self.w2(r) / np.sqrt(self.k * self.R * self.T2(r))

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


if __name__ == '__main__':
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
                      H01_init=120e3,
                      c21_init=250,
                      alpha11=np.radians([17])[0],
                      gamma_av=np.radians([4])[0],
                      gamma_sum=np.radians([10])[0])
    turbine.compute_geometry()
    turbine.compute_stages_gas_dynamics(precise_heat_drop=True)
    turbine.compute_integrate_turbine_parameters()
    # turbine[0].plot_velocity_triangle()
    # turbine.geom.plot_geometry(figsize=(5, 5))

    rad_dist = StageParametersRadialDistribution(ProfilingType.ConstantCirculation,
                                                 lambda r: turbine[0].p0_stag,
                                                 lambda r: turbine[0].T0_stag,
                                                 lambda r: 100,
                                                 turbine.c_p_gas,
                                                 turbine.k_gas,
                                                 turbine.geom[0].D1 - turbine.geom[0].l1,
                                                 turbine.geom[0].D1,
                                                 turbine.geom[0].D1 + turbine.geom[0].l1,
                                                 turbine.n,
                                                 turbine[0].c1,
                                                 turbine[0].alpha1,
                                                 turbine[0].L_u,
                                                 turbine[0].c2_a,
                                                 turbine[0].c2_u)
    rad_dist.plot_parameter_distribution(['alpha1', 'alpha2', 'beta1', 'beta2'])

