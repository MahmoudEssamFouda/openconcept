"""
The MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Copyright: (c) 2022, Deutsches Zentrum fuer Luft- und Raumfahrt e.V.
Contact: jasper.bussemaker@dlr.de
         mahmoud.fouda@dlr.de
"""

import openmdao.api as om
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openconcept.architecting.propulsion_systems.conventional_twin_turboprop import ConventionalTwinTurboprop
from openconcept.architecting.propulsion_systems.conventional_twin_turboprop import ConventionalTwinTurbopropExpanded


class ConventionalTwinTurbopropTestGroup(om.Group):
    """
    Test the conventional Twin turboprop propulsion system
    """

    def initialize(self):
        self.options.declare('vec_size', default=11, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options['vec_size']

        controls = self.add_subsystem('controls', om.IndepVarComp(), promotes_outputs=['*'])
        controls.add_output('prop|rpm', val=np.ones(nn) * 1900, units='rpm')

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes_outputs=['*'])
        dvs.add_output('ac|propulsion|engine|rating', val=850, units='hp')
        dvs.add_output('ac|propulsion|propeller|diameter', val=2.3, units='m')
        dvs.add_output('ac|propulsion|propeller|power_rating', val=850, units='hp')

        flt_cond = self.add_subsystem('flt_cond', om.IndepVarComp(), promotes_outputs=['*'])
        flt_cond.add_output('fltcond|rho', units='kg / m ** 3', desc='Air density', val=np.ones(nn) * 0.475448)
        flt_cond.add_output('fltcond|Utrue', units='m/s', desc='Flight speed', val=np.ones(nn) * 92.5)

        mission_prms = self.add_subsystem('mission_prms', om.IndepVarComp(), promotes_outputs=['*'])
        mission_prms.add_output('throttle', val=np.ones(nn) * 0.90)
        mission_prms.add_output('propulsor_active', val=np.ones(nn) * 1.0)

        # define propulsion system
        propulsion_promotes_outputs = ['fuel_flow', 'thrust']
        propulsion_promotes_inputs = ["ac|propulsion|*", "fltcond|*", "throttle", "propulsor_active"]

        self.add_subsystem('propmodel', ConventionalTwinTurboprop(num_nodes=nn),
                           promotes_inputs=propulsion_promotes_inputs,
                           promotes_outputs=propulsion_promotes_outputs)
        self.connect('prop|rpm', ['propmodel.prop1.rpm', 'propmodel.prop2.rpm'])


class ConventionalTwinTurbopropExpandedTestGroup(om.Group):
    """
    Test the conventional Twin turboprop expanded propulsion system
    """

    def initialize(self):
        self.options.declare('vec_size', default=11, desc="Number of mission analysis points to run")
        self.options.declare('engine_out', default=True)

    def setup(self):
        nn = self.options['vec_size']
        engine_out = self.options['engine_out']

        controls = self.add_subsystem('controls', om.IndepVarComp(), promotes_outputs=['*'])
        controls.add_output('prop|rpm', val=np.ones(nn) * 1900, units='rpm')

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes_outputs=['*'])
        dvs.add_output('ac|propulsion|engine|rating', val=850, units='hp')
        dvs.add_output('ac|propulsion|engine|output_rpm', val=6000.0, units='rpm')
        dvs.add_output('ac|propulsion|propeller|diameter', val=2.3, units='m')
        dvs.add_output('ac|propulsion|propeller|power_rating', val=850, units='hp')
        dvs.add_output('ac|propulsion|propeller|max_speed', val=2900.0, units='rpm')
        dvs.add_output('ac|propulsion|propeller|min_speed', val=1100.0, units='rpm')

        flt_cond = self.add_subsystem('flt_cond', om.IndepVarComp(), promotes_outputs=['*'])
        flt_cond.add_output('fltcond|rho', units='kg / m ** 3', desc='Air density', val=np.ones(nn) * 0.475448)
        flt_cond.add_output('fltcond|Utrue', units='m/s', desc='Flight speed', val=np.ones(nn) * 92.5)

        mission_prms = self.add_subsystem('mission_prms', om.IndepVarComp(), promotes_outputs=['*'])
        mission_prms.add_output('throttle', val=np.ones(nn) * 0.90)
        if engine_out:
            mission_prms.add_output('propulsor_active', val=np.ones(nn) * 0.0)
        else:
            mission_prms.add_output('propulsor_active', val=np.ones(nn) * 1.0)

        # define propulsion system
        propulsion_promotes_outputs = ['fuel_flow', 'thrust']
        propulsion_promotes_inputs = ["ac|propulsion|*", "fltcond|*", "throttle", "propulsor_active"]

        self.add_subsystem('propmodel', ConventionalTwinTurbopropExpanded(num_nodes=nn),
                           promotes_inputs=propulsion_promotes_inputs,
                           promotes_outputs=propulsion_promotes_outputs)
        self.connect('prop|rpm', ['propmodel.prop1.rpm', 'propmodel.prop2.rpm'])


class ConventionalTwinTurbopropTestCase(unittest.TestCase):

    def test_default_settings(self):
        prob = om.Problem(ConventionalTwinTurbopropTestGroup(vec_size=11))
        prob.setup(check=True, force_alloc_complex=True)
        # om.n2(prob, show_browser=True)
        prob.run_model()

        assert_near_equal(prob.get_val('propmodel.eng1.shaft_power_out', units='W'),
                          np.ones(11) * 0.9 * 850 * 745.7, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.eng1.component_cost', units='USD'),
                          850 * 745.7 * 1.04 + 0, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.eng1.component_weight', units='kg'),
                          (850 * 745.7 * 0.14 / 1000) + 104, tolerance=1e-6)
        #
        assert_near_equal(prob.get_val('propmodel.prop1.cp'),
                          np.ones(11) * 0.9 * 850 * 745.7 / (0.475448 * ((1900 / 60) ** 3) * 2.3 ** 5), tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.prop1.propmap.J'),
                          np.ones(11) * 92.5 / ((1900 / 60) * 2.3), tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.prop1.prop_Vtip'),
                          np.ones(11) * ((1900 / 60) * 2.3 * np.pi), tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.prop1.eta_prop'), np.ones(11) * 0.7842547, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.prop1.ct_over_cp'), np.ones(11) * 0.1314537, tolerance=1e-6)

        assert_near_equal(prob.get_val('propmodel.prop1.component_weight', units='lb'),
                          0.108 * (2.3 * 3.28084 * 850 * (4 ** 0.5)) ** 0.782, tolerance=1e-6)  # 1 m = 3.28 ft

        assert_near_equal(prob.get_val('thrust', units='N'),
                          (2 * np.ones(11) * (
                                  0.9 * 850 * 745.7 / (0.475448 * ((1900 / 60) ** 3) * 2.3 ** 5)) * 0.7842547 /
                           (92.5 / ((1900 / 60) * 2.3))) * 0.475448 * ((1900 / 60) ** 2) * 2.3 ** 4, tolerance=1e-6)
        assert_near_equal(prob.get_val('fuel_flow', units='kg/s'),
                          2 * np.ones(11) * 0.9 * 850 * 745.7 * 0.6 * 1.68965774e-7, tolerance=1e-6)

        prob.model.list_inputs(units=True,
                               prom_name=True,
                               shape=True,
                               hierarchical=False,
                               print_arrays=True)

        # show outputs
        prob.model.list_outputs(implicit=True,
                                explicit=True,
                                prom_name=True,
                                units=True,
                                shape=True,
                                bounds=False,
                                residuals=False,
                                scaling=False,
                                hierarchical=False,
                                print_arrays=True)

        # partials = prob.check_partials(method='cs', compact_print=True)
        # assert_check_partials(partials) -> fails for propeller component, partial derivatives for eff map parameters


class ConventionalTwinTurbopropExpandedTestCase(unittest.TestCase):
    def test_default_settings(self):
        prob = om.Problem(ConventionalTwinTurbopropExpandedTestGroup(vec_size=11, engine_out=False))
        prob.setup(check=True, force_alloc_complex=True)
        # om.n2(prob, show_browser=True)
        prob.run_model()
        assert_near_equal(prob.get_val('propmodel.eng1.shaft_power_out', units='W'),
                          np.ones(11) * 0.9 * 850 * 745.7, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.eng1.component_cost', units='USD'),
                          850 * 745.7 * 1.04 + 0, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.eng1.component_weight', units='kg'),
                          (850 * 745.7 * 0.14 / 1000) + 104, tolerance=1e-6)
        # gearbox efficiency for these inputs  = 0.9875942790377175
        # Note: gearbox efficiency changes based on input power and rated power
        assert_near_equal(prob.get_val('propmodel.prop1.cp'),
                          np.ones(11) * 0.9 * 850 * 745.7 * 0.9875942790377175
                          / (0.475448 * ((1900 / 60) ** 3) * 2.3 ** 5), tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.prop1.propmap.J'),
                          np.ones(11) * 92.5 / ((1900 / 60) * 2.3), tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.prop1.prop_Vtip'),
                          np.ones(11) * ((1900 / 60) * 2.3 * np.pi), tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.prop1.eta_prop'), np.ones(11) * 0.78457275, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.prop1.ct_over_cp'), np.ones(11) * 0.13903385, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.prop1.component_weight', units='lb'),
                          0.108 * (2.3 * 3.28084 * 850 * (4 ** 0.5)) ** 0.782, tolerance=1e-6)  # 1 m = 3.28 ft

        assert_near_equal(prob.get_val('thrust', units='N'),
                          (2 * np.ones(11) * (
                                  0.9875942790377175 * 0.9 * 850 * 745.7 / (
                                  0.475448 * ((1900 / 60) ** 3) * 2.3 ** 5)) * 0.78457275 /
                           (92.5 / ((1900 / 60) * 2.3))) * 0.475448 * ((1900 / 60) ** 2) * 2.3 ** 4, tolerance=1e-6)
        assert_near_equal(prob.get_val('fuel_flow', units='kg/s'),
                          2 * np.ones(11) * 0.9 * 850 * 745.7 * 0.6 * 1.68965774e-7, tolerance=1e-6)

        prob.model.list_inputs(units=True,
                               prom_name=True,
                               shape=True,
                               hierarchical=False,
                               print_arrays=True)

        # show outputs
        prob.model.list_outputs(implicit=True,
                                explicit=True,
                                prom_name=True,
                                units=True,
                                shape=True,
                                bounds=False,
                                residuals=False,
                                scaling=False,
                                hierarchical=False,
                                print_arrays=True)

    def test_engine_out(self):  # check one engine out case when propulsor_active = 0
        prob = om.Problem(ConventionalTwinTurbopropExpandedTestGroup(vec_size=11, engine_out=True))
        prob.setup(check=True, force_alloc_complex=True)
        # om.n2(prob, show_browser=True)
        prob.run_model()
        assert_near_equal(prob.get_val('propmodel.eng2.shaft_power_out', units='W'),
                          np.ones(11) * 0.0, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.gearbox2.shaft_power_out', units='W'),
                          np.ones(11) * 0.0, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.prop2.thrust', units='N'),
                          np.ones(11) * 0.0, tolerance=1e-6)

        prob.model.list_inputs(units=True,
                               prom_name=True,
                               shape=True,
                               hierarchical=False,
                               print_arrays=True)

        # show outputs
        prob.model.list_outputs(implicit=True,
                                explicit=True,
                                prom_name=True,
                                units=True,
                                shape=True,
                                bounds=False,
                                residuals=False,
                                scaling=False,
                                hierarchical=False,
                                print_arrays=True)


# if __name__ == '__main__':  # run the prop system model, check n2, inputs and outputs
#     prob = om.Problem(ConventionalTwinTurbopropExpandedTestGroup(vec_size=11, engine_out=False))
#     prob.setup(check=True, force_alloc_complex=True)
#     om.n2(prob, show_browser=True)
#     prob.run_model()
