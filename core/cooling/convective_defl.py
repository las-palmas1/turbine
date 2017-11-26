import matplotlib.pyplot as plt
from ..profiling.section import BladeSection
import numpy as np
from gas_turbine_cycle.gases import Air, IdealGas
import typing
from .tools import LocalParamCalculator, GasBladeHeatExchange


class SectorCooler(GasBladeHeatExchange):
    """Класс для расчета локальных параметров на участке дефлекторной лопатки с постоянным профилем и
    параметрами газа на входе."""
    def __init__(self, section: BladeSection,
                 height,
                 T_gas_stag,
                 G_gas,
                 D_av,
                 wall_thickness,
                 T_wall_av,
                 T_cool_fluid0,
                 G_cool,
                 channel_width,
                 lam_blade: typing.Callable[[float], float],
                 cool_fluid: IdealGas,
                 work_fluid: IdealGas,
                 node_num: int=500):
        """
        :param section: BladeSection. \n
            Сечение рассчитываемого участка лопатки.
        :param height: float. \n
            Высота рассчитываемого участка лопакти.
        :param T_gas_stag: float. \n
            Температура торможения газа на входе в лопаточный венец.
        :param G_gas: float. \n
            Расход газа через лопаточный венец.
        :param D_av: float. \n
            Средний диаметр участка.
        :param wall_thickness: float. \n
            Толщина стенки.
        :param T_wall_av: float. \n
            Средняя температура срединной поверхности участка лопатки.
        :param T_cool_fluid0: float. \n
            Температура охраждающего тела на входе в лопатку.
        :param G_cool: float. \n
            Расход охлаждающего тела.
        :param channel_width: float. \n
            Ширина канала между дефлектором и профилем.
        :param lam_blade:
            Тепопроводность материала лопатки в зависимости от температуры
        :param cool_fluid: IdealGas. \n
            Охлаждающее тело.
        :param work_fluid: IdealGas. \n
            Рабочее тело турбины.
        :param node_num: int, optional. \n
            Число узлов на интервале решения уравнения теплового баланса.
        """
        self.section = section
        self.height = height
        self.T_gas_stag = T_gas_stag
        self.G_gas = G_gas
        self.D_av = D_av
        self.angle1 = section.angle1
        self.angle2 = section.angle2
        self.wall_thickness = wall_thickness
        self.T_wall_av = T_wall_av
        self.T_cool_fluid0 = T_cool_fluid0
        self.G_cool = G_cool
        self.channel_width = channel_width
        self.lam_blade = lam_blade
        self.cool_fluid = cool_fluid
        self.work_fluid = work_fluid
        self.node_num = node_num
        # self.ave_param = DeflectorAverageParamCalculator(section=self.section,
        #                                                  height=height,
        #                                                  D_av=D_av,
        #                                                  wall_thickness=wall_thickness,
        #                                                  T_wall_out=T_wall_out,
        #                                                  T_cool_fluid0=T_cool_fluid0,
        #                                                  T_out_stag=T_gas_stag,
        #                                                  lam_blade=lam_blade,
        #                                                  cool_fluid=cool_fluid)

        self.local_param = LocalParamCalculator(section=section,
                                                height=height,
                                                wall_thickness=wall_thickness,
                                                T_cool_fluid0=T_cool_fluid0,
                                                G_cool=lambda x: G_cool,
                                                T_out_stag=lambda x: T_gas_stag,
                                                cool_fluid=cool_fluid,
                                                lam_blade=lam_blade,
                                                node_num=node_num)

        self.x_2k = None
        self.x_1k = None
        self.x_0 = None
        self.x_1s = None
        self.x_2s = None
        self.x_3s = None
        self.alpha_gas_inlet = None
        self.mu_gas = None
        self.lam_gas = None
        self.Re_gas = None
        self.Nu_gas = None
        self.alpha_gas_av = None

    def _get_alpha_cool_fluid(self, G_cool, T_cool_fluid):
        return (0.02 * self.cool_fluid.lam(T_cool_fluid) / (2 * self.channel_width) *
                (G_cool / (self.height * self.cool_fluid.mu(T_cool_fluid))) ** 0.8)

    # def _compute_average_parameters(self, G_cool):
    #     self.mu_gas = self.work_fluid.mu(self.T_gas_stag)
    #     self.lam_gas = self.work_fluid.lam(self.T_gas_stag)
    #     self.Re_gas = self.G_gas * self.section.chord_length / (np.pi * self.D_av * self.height *
    #                                                             self.mu_gas * np.sin(self.angle2))
    #     self.nusselt_coef = self.get_nusselt_coef(self.angle1, self.angle2)
    #     self.Nu_gas = self.nusselt_coef * self.Re_gas ** 0.68
    #     self.alpha_gas_av = self.Nu_gas * self.lam_gas / self.section.chord_length
    #     self.ave_param.G_cool = G_cool
    #     self.ave_param.alpha_out = self.alpha_gas_av
    #     self.ave_param.compute()

    # def plot_channel_width_plot(self, G_air_arr, figsize=(6, 4)):
    #     channel_width_arr = []
    #     for G_air in G_air_arr:
    #         self._compute_average_parameters(G_air)
    #         channel_width_arr.append(self.channel_width)
    #
    #     plt.figure(figsize=figsize)
    #     plt.plot(G_air_arr, np.array(channel_width_arr) * 1e3, lw=1, color='red')
    #     plt.xlabel(r'$G_в,\ кг/с$', fontsize=10)
    #     plt.ylabel(r'$\delta,\ мм$', fontsize=10)
    #     plt.grid()
    #     plt.show()

    def compute(self):
        self.mu_gas, self.lam_gas, self.Re_gas, self.Nu_gas, self.alpha_gas_av = self.get_alpha_gas_av(self.section,
                                                                                                       self.height,
                                                                                                       self.T_gas_stag,
                                                                                                       self.G_gas,
                                                                                                       self.D_av,
                                                                                                       self.work_fluid)
        self.alpha_gas_inlet = self.get_alpha_gas_inlet(self.section, self.height, self.T_gas_stag, self.G_gas,
                                                        self.D_av, self.work_fluid)
        self.x_3s, self.x_2s, self.x_1s, self.x_0, self.x_1k, self.x_2k = self.get_regions_bound(self.section)
        self.local_param.T_wall_av = self.T_wall_av
        self.local_param.alpha_cool = lambda x, T: self._get_alpha_cool_fluid(self.local_param.G_cool(x), T)
        self.local_param.alpha_out = lambda x: self.get_alpha_gas(x, self.alpha_gas_inlet, self.alpha_gas_av,
                                                                  *self.get_regions_bound(self.section))
        self.local_param.compute()


class BladeCooler:
    def __init__(self):
        pass