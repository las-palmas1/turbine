import unittest
import numpy as np
from gas_turbine_cycle.gases import KeroseneCombustionProducts
from core.average_streamline.turbine import TurbineType, Turbine
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

    def test_comp_turb_2st(self):
        loader = FileSystemLoader([core.templates.__path__[0], core.templates.tests.__path__[0]])
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
        template = env.get_template('2stage_comp_turb_templ.tex')
        content = template.render(turb=self.comp_turb_2st)

        with open('2stage_comp_turb.tex', 'w', encoding='utf-8') as file:
            file.write(content)
