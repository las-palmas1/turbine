from .stage import StageProfiler, ProfilingType
from ..average_streamline.turbine import Turbine, TurbineType
from gas_turbine_cycle.gases import KeroseneCombustionProducts
from ..average_streamline.stage_gas_dynamics import StageGasDynamics
from ..average_streamline.stage_geom import StageGeomAndHeatDrop
import typing
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class TurbineProfiler:
    def __init__(self, turbine: Turbine, p_in_stag: typing.Callable[[float], float],
                 T_in_stag: typing.Callable[[float], float], c_in: typing.Callable[[float], float],
                 alpha_in: typing.Callable[[float], float], section_num: int=3, pnt_cnt: int=30,
                 center: bool=False):
        self._turbine = turbine
        self.section_num = section_num
        self.pnt_cnt = pnt_cnt
        self.center = center
        self.stage_number = turbine.stage_number
        self.p_in_stag = p_in_stag
        self.T_in_stag = T_in_stag
        self.c_in = c_in
        self.alpha_in = alpha_in
        self._stages = self._get_empty_stage_profilers()
        self._set_stage_profilers(self._turbine, section_num, pnt_cnt, center)

    def __getitem__(self, item) -> StageProfiler:
        if 0 <= item < self.stage_number:
            return self._stages[item]
        else:
            IndexError('invalid index')

    def __len__(self):
        return self.stage_number

    def __iter__(self):
        self._item = 0
        return self

    def __next__(self) -> StageProfiler:
        if 0 <= self._item < self._turbine.stage_number:
            current = self._stages[self._item]
            self._item += 1
            return current
        else:
            raise StopIteration()

    def compute_stage_profiles(self):
        for n in range(self.stage_number):
            if n == 0:
                self[n].init_sections()
                self[n].compute_sections()
            else:
                self[n].p0_stag = self[n - 1].p2_stag
                self[n].T0_stag = self[n - 1].T2_stag
                self[n].c0 = self[n - 1].c2
                self[n].alpha0 = self[n - 1].alpha2
                self[n].x0 = self[n - 1].x0_next
                self[n].init_sections()
                self[n].compute_sections()

    def _set_stage_profilers(self, turbine: Turbine, section_num: int, pnt_cnt: int, center: bool):
        for n in range(turbine.stage_number):
            self._set_av_par_in_stage_profiler(self._stages[n], self._turbine[n], self._turbine.geom[n])
            self._stages[n].section_num = section_num
            self._stages[n].pnt_cnt = pnt_cnt
            self._stages[n].center = center

    def plot_3d(self):
        fig = plt.figure(figsize=(11, 7))
        axes = Axes3D(fig)
        for i in range(self.stage_number):
            self[i].plot_profile_3d(axes)
        plt.show()

    @classmethod
    def _set_av_par_in_stage_profiler(cls, stage_profiler: StageProfiler, gas_dynamic: StageGasDynamics,
                                      geom: StageGeomAndHeatDrop):
        """Передача параметров из расчета по средней линии в профилировщик ступени."""
        stage_profiler.c_p = gas_dynamic.c_p_gas
        stage_profiler.k = gas_dynamic.k_gas
        stage_profiler.b_a_rk = geom.b_rk
        stage_profiler.b_a_sa = geom.b_sa
        stage_profiler.D1_in = geom.D1 - geom.l1
        stage_profiler.D1_av = geom.D1
        stage_profiler.D1_out = geom.D1 + geom.l1
        stage_profiler.n = gas_dynamic.n
        stage_profiler.c1_av = gas_dynamic.c1
        stage_profiler.alpha1_av = gas_dynamic.alpha1
        stage_profiler.L_u_av = gas_dynamic.L_u
        stage_profiler.c2_a_av = gas_dynamic.c2_a
        stage_profiler.c2_u_av = gas_dynamic.c2_u
        stage_profiler.delta_a_sa = geom.delta_a_sa
        stage_profiler.delta_a_rk = geom.delta_a_rk

    def _get_empty_stage_profilers(self) -> typing.List[StageProfiler]:
        res = []
        for i in range(self._turbine.stage_number):
            if i == 0:
                res.append(StageProfiler(profiling_type=ProfilingType.ConstantCirculation,
                                         p0_stag=self.p_in_stag,
                                         T0_stag=self.T_in_stag,
                                         c0=self.c_in,
                                         alpha0=self.alpha_in,
                                         c_p=None,
                                         k=None,
                                         D1_in=None,
                                         D1_av=None,
                                         D1_out=None,
                                         n=None,
                                         c1_av=None,
                                         alpha1_av=None,
                                         L_u_av=None,
                                         c2_a_av=None,
                                         c2_u_av=None,
                                         b_a_sa=None,
                                         b_a_rk=None,
                                         delta_a_sa=None,
                                         delta_a_rk=None))
            else:
                res.append(StageProfiler(profiling_type=ProfilingType.ConstantCirculation,
                                         p0_stag=lambda r: 0.,
                                         T0_stag=lambda r: 0.,
                                         c0=lambda r: 0.,
                                         alpha0=lambda r: 0.,
                                         c_p=None,
                                         k=None,
                                         D1_in=None,
                                         D1_av=None,
                                         D1_out=None,
                                         n=None,
                                         c1_av=None,
                                         alpha1_av=None,
                                         L_u_av=None,
                                         c2_a_av=None,
                                         c2_u_av=None,
                                         b_a_sa=None,
                                         b_a_rk=None,
                                         delta_a_sa=None,
                                         delta_a_rk=None))
        return res


if __name__ == '__main__':
    turbine = Turbine(TurbineType.Compressor,
                      T_g_stag=1450,
                      p_g_stag=400e3,
                      G_turbine=25,
                      work_fluid=KeroseneCombustionProducts(),
                      alpha_air=2.5,
                      l1_D1_ratio=0.25,
                      n=15e3,
                      T_t_stag_cycle=1050,
                      stage_number=2,
                      eta_t_stag_cycle=0.91,
                      k_n=6.8,
                      eta_m=0.99,
                      auto_compute_heat_drop=False,
                      precise_heat_drop=False,
                      auto_set_rho=False,
                      rho_list=[0.4, 0.20],
                      H0_list=[235e3, 200e3],
                      alpha11=np.radians([17])[0],
                      gamma_av=np.radians([4])[0],
                      gamma_sum=np.radians([10])[0])
    turbine.compute_geometry()
    turbine.compute_stages_gas_dynamics()
    turbine.compute_integrate_turbine_parameters()
    # turbine[0].plot_velocity_triangle()
    # turbine[1].plot_velocity_triangle()
    # turbine.geom.plot_geometry(figsize=(5, 5))

    turbine_profiler = TurbineProfiler(turbine=turbine,
                                       p_in_stag=lambda r: turbine[0].p0_stag,
                                       T_in_stag=lambda r: turbine[0].T0_stag,
                                       c_in=lambda r: 90,
                                       alpha_in=lambda r: np.radians([90])[0],
                                       center=True,
                                       )
    turbine_profiler[0].profiling_type = ProfilingType.ConstantAngle
    turbine_profiler[0].gamma1_k_rel_rk = 0.5
    turbine_profiler[1].profiling_type = ProfilingType.ConstantAngle
    turbine_profiler[1].auto_sections_par = False
    turbine_profiler[1].rk_sections[0].gamma1_s = np.radians([15])[0]
    turbine_profiler[1].rk_sections[0].gamma1_k = np.radians([7])[0]
    turbine_profiler.compute_stage_profiles()
    turbine_profiler.plot_3d()
    turbine_profiler[0].plot_profile_2d(0)
    turbine_profiler[1].plot_parameter_distribution(['alpha1', 'alpha2', 'beta1', 'beta2'])
    turbine_profiler[1].plot_parameter_distribution(['M_c0', 'M_c1', 'M_w1', 'M_w2'])
    turbine_profiler[1].plot_velocity_triangles()