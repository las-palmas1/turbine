import unittest
import numpy as np
from gas_turbine_cycle.gases import KeroseneCombustionProducts, Air
from core.average_streamline.turbine import TurbineType, Turbine
from core.profiling.turbine import TurbineProfiler, ProfilingType
from core.cooling.film_defl import FilmBladeCooler, get_sa_cooler
import os
from jinja2 import Template, Environment, select_autoescape, FileSystemLoader
import core.templates.tests
import core.templates


class TemplateTester(unittest.TestCase):
    def setUp(self):
        self.comp_turb_2st = Turbine(TurbineType.Compressor,
                                     T_g_stag=1400,
                                     p_g_stag=5.5e5,
                                     G_turbine=25,
                                     G_fuel=1,
                                     work_fluid=KeroseneCombustionProducts(),
                                     alpha_air=2.87,
                                     l1_D1_ratio=0.25,
                                     n=15e3,
                                     T_t_stag_cycle=1200,
                                     stage_number=2,
                                     eta_t_stag_cycle=0.91,
                                     k_n=6.8,
                                     eta_m=0.99,
                                     auto_compute_heat_drop=True,
                                     auto_set_rho=True,
                                     precise_heat_drop=False,
                                     H01_init=150e3,
                                     c21_init=250,
                                     alpha11=np.radians([17])[0],
                                     gamma_av=np.radians([4])[0],
                                     gamma_sum=np.radians([10])[0])
        self.comp_turb_2st.geom[0].g_cool = 0.004
        self.comp_turb_2st.geom[1].g_cool = 0.003
        self.comp_turb_2st.compute_geometry()
        self.comp_turb_2st.compute_stages_gas_dynamics()
        self.comp_turb_2st.compute_integrate_turbine_parameters()

        self.turb_profiler = TurbineProfiler(
            turbine=self.comp_turb_2st,
            p_in_stag=lambda r: self.comp_turb_2st[0].p0_stag,
            T_in_stag=lambda r: self.comp_turb_2st[0].T0_stag,
            c_in=lambda r: 100,
            alpha_in=lambda r: np.radians([90])[0],
            section_num=3,
            center=True
        )
        self.turb_profiler[0].profiling_type = ProfilingType.ConstantAngle
        self.turb_profiler[1].profiling_type = ProfilingType.ConstantAngle
        self.turb_profiler.compute_stage_profiles()

        x_hole_rel = [-0.5, -0.25, 0., 0.2, 0.6]
        G_cool0 = 0.2
        g_cool0 = G_cool0 * 2 / (self.turb_profiler[0].D1_out - self.turb_profiler[0].D1_in)

        self.cooler = get_sa_cooler(
            self.turb_profiler[0],
            channel_width=0.001,
            wall_thickness=0.001,
            T_wall_out_av_init=1000,
            lam_blade=lambda T: 24,
            x_hole_rel=x_hole_rel,
            hole_num=[35 for _ in x_hole_rel],
            d_hole=[0.5e-3 for _ in x_hole_rel],
            phi_hole=[0.98 for _ in x_hole_rel],
            mu_hole=[0.95 for _ in x_hole_rel],
            work_fluid=KeroseneCombustionProducts(),
            T_cool0=650,
            p_cool_stag0=0.99 * self.comp_turb_2st[0].p0_stag,
            g_cool0=lambda r: g_cool0,
            cool_fluid=Air(),
            cover_thickness=0,
        )
        self.cooler.compute()
        self.cooler.plot_T_wall(T_material=1100)

    def test_comp_turb_2st(self):
        loader = FileSystemLoader(
            [
                core.templates.__path__[0],
                core.templates.tests.__path__[0],
            ]
        )
        env = Environment(
            loader=loader,
            autoescape=select_autoescape(['tex']),
            block_start_string='</',
            block_end_string='/>',
            variable_start_string='<<',
            variable_end_string='>>',
            comment_start_string='<#',
            comment_end_string='#>'
        )
        st1_params = self.turb_profiler[0].get_flow_params(r_rel=np.array([0, 0.5, 1.0]))

        film_params = self.cooler.sectors[1].get_film_params()
        local_params = self.cooler.sectors[1].get_local_params()

        cooling_results = self.cooler.get_cooling_results()

        template = env.get_template('2stage_comp_turb_templ.tex')
        content = template.render(
            turb=self.comp_turb_2st,
            st1_params=st1_params,
            st1_prof_type='const angle',
            film_params=film_params,
            local_params=local_params,
            cooling_results=cooling_results,
            G_comp=self.comp_turb_2st.G_turbine,
            blade_num=self.turb_profiler[0].z_sa
        )

        with open('2stage_comp_turb.tex', 'w', encoding='utf-8') as file:
            file.write(content)
