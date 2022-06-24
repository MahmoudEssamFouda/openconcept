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
from examples.propulsion_layouts.simple_series_hybrid import TwinSeriesHybridElectricPropulsionSystem
from openconcept.architecting.propulsion_systems.electric_twin_prop import ElectricTwinProp, ElectricTwinPropExpanded
from openconcept.utilities.linearinterp import LinearInterpolator
from openmdao.api import DirectSolver, IndepVarComp, NewtonSolver, BoundsEnforceLS


class ElectricTwinPropTestGroup(om.Group):
    """
    Test the Twin series hybrid propulsion system
    """

    def initialize(self):
        self.options.declare("vec_size", default=11, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options["vec_size"]

        controls = self.add_subsystem("controls", om.IndepVarComp(), promotes_outputs=["*"])
        controls.add_output("prop|rpm", val=np.ones(nn) * 1900, units="rpm")

        dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes_outputs=["*"])
        dvs.add_output("ac|propulsion|propeller|diameter", val=2.3, units="m")
        dvs.add_output("ac|propulsion|propeller|power_rating", val=240, units="kW")
        dvs.add_output("ac|propulsion|motor|rating", val=240, units="kW")
        dvs.add_output("ac|weights|W_battery", val=1000, units="kg")
        dvs.add_output("ac|propulsion|battery|specific_energy", val=300, units="W*h/kg")

        flt_cond = self.add_subsystem("flt_cond", om.IndepVarComp(), promotes_outputs=["*"])
        flt_cond.add_output("fltcond|rho", units="kg / m ** 3", desc="Air density", val=np.ones(nn) * 0.475448)
        flt_cond.add_output("fltcond|Utrue", units="m/s", desc="Flight speed", val=np.ones(nn) * 92.5)

        mission_prms = self.add_subsystem("mission_prms", om.IndepVarComp(), promotes_outputs=["*"])
        mission_prms.add_output("throttle", val=np.ones(nn) * 0.90)
        mission_prms.add_output("propulsor_active", val=np.ones(nn) * 1.0)
        mission_prms.add_output("duration", val=300, units="s")

        # define propulsion system
        propulsion_promotes_outputs = ["thrust"]
        propulsion_promotes_inputs = [
            "ac|propulsion|*",
            "fltcond|*",
            "throttle",
            "propulsor_active",
            "ac|weights*",
            "duration",
        ]  # ac|weights* is only used for battery weight

        self.add_subsystem(
            "propmodel",
            ElectricTwinProp(num_nodes=nn),
            promotes_inputs=propulsion_promotes_inputs,
            promotes_outputs=propulsion_promotes_outputs,
        )

        self.connect("prop|rpm", ["propmodel.prop1.rpm", "propmodel.prop2.rpm"])


class ElectricTwinPropExpandedTestGroup(om.Group):
    """
    Test the Twin all electric expanded propulsion system
    """

    def initialize(self):
        self.options.declare("vec_size", default=11, desc="Number of mission analysis points to run")
        self.options.declare("engine_out", default=False)

    def setup(self):
        nn = self.options["vec_size"]
        engine_out = self.options["engine_out"]

        controls = self.add_subsystem("controls", om.IndepVarComp(), promotes_outputs=["*"])
        controls.add_output("prop|rpm", val=np.ones(nn) * 1900, units="rpm")

        dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes_outputs=["*"])
        dvs.add_output("ac|propulsion|propeller|diameter", val=2.3, units="m")
        dvs.add_output("ac|propulsion|propeller|max_speed", val=2900.0, units="rpm")
        dvs.add_output("ac|propulsion|propeller|min_speed", val=1900.0, units="rpm")
        dvs.add_output("ac|propulsion|motor|rating", val=240, units="kW")
        dvs.add_output("ac|propulsion|motor|max_speed", val=5500.0, units="rpm")
        dvs.add_output("ac|propulsion|propeller|power_rating", val=240, units="kW")
        dvs.add_output("ac|weights|W_battery", val=1000, units="kg")
        dvs.add_output("ac|propulsion|battery|specific_energy", val=300, units="W*h/kg")
        dvs.add_output("ac|propulsion|battery|SOC_initial", val=1.0)

        flt_cond = self.add_subsystem("flt_cond", om.IndepVarComp(), promotes_outputs=["*"])
        flt_cond.add_output("fltcond|rho", units="kg / m ** 3", desc="Air density", val=np.ones(nn) * 0.475448)
        flt_cond.add_output("fltcond|Utrue", units="m/s", desc="Flight speed", val=np.ones(nn) * 92.5)

        mission_prms = self.add_subsystem("mission_prms", om.IndepVarComp(), promotes_outputs=["*"])
        mission_prms.add_output("throttle", val=np.ones(nn) * 0.90)
        if engine_out:
            mission_prms.add_output("propulsor_active", val=np.ones(nn) * 0.0)
        else:
            mission_prms.add_output("propulsor_active", val=np.ones(nn) * 1.0)

        mission_prms.add_output("duration", val=300, units="s")

        # define propulsion system
        propulsion_promotes_outputs = ["thrust"]
        propulsion_promotes_inputs = [
            "ac|propulsion|*",
            "fltcond|*",
            "throttle",
            "propulsor_active",
            "ac|weights*",
            "duration",
        ]  # ac|weights* is only used for battery weight

        self.add_subsystem(
            "propmodel",
            ElectricTwinPropExpanded(num_nodes=nn),
            promotes_inputs=propulsion_promotes_inputs,
            promotes_outputs=propulsion_promotes_outputs,
        )

        self.connect("prop|rpm", ["propmodel.prop1.rpm", "propmodel.prop2.rpm"])


class ElectricTwinPropTestCase(unittest.TestCase):
    def test_default_settings(self):
        prob = om.Problem(ElectricTwinPropTestGroup(vec_size=11))
        prob.model.nonlinear_solver = NewtonSolver(iprint=1)
        prob.model.options["assembled_jac_type"] = "csc"
        prob.model.linear_solver = DirectSolver(assemble_jac=True)
        prob.model.nonlinear_solver.options["solve_subsystems"] = True  # important to solve subsystem implicit eqns
        prob.model.nonlinear_solver.options["maxiter"] = 10
        prob.model.nonlinear_solver.options["atol"] = 1e-7
        prob.model.nonlinear_solver.options["rtol"] = 1e-7
        prob.setup(check=True, force_alloc_complex=True)
        # om.n2(prob, show_browser=True)
        prob.run_model()
        prob.model.list_inputs(units=True, prom_name=True, shape=True, hierarchical=False, print_arrays=True)

        # show outputs
        prob.model.list_outputs(
            implicit=True,
            explicit=True,
            prom_name=True,
            units=True,
            shape=True,
            bounds=False,
            residuals=False,
            scaling=False,
            hierarchical=False,
            print_arrays=True,
        )


class ElectricTwinPropExpandedTestCase(unittest.TestCase):
    def test_default_settings(self):
        prob = om.Problem(ElectricTwinPropExpandedTestGroup(vec_size=11, engine_out=False))
        prob.model.nonlinear_solver = NewtonSolver(iprint=1)
        prob.model.options["assembled_jac_type"] = "csc"
        prob.model.linear_solver = DirectSolver(assemble_jac=True)
        prob.model.nonlinear_solver.options["solve_subsystems"] = True  # important to solve subsystem implicit eqns
        prob.model.nonlinear_solver.options["maxiter"] = 10
        prob.model.nonlinear_solver.options["atol"] = 1e-7
        prob.model.nonlinear_solver.options["rtol"] = 1e-7
        prob.setup(check=True, force_alloc_complex=True)
        # om.n2(prob, show_browser=True)
        prob.run_model()

        assert_near_equal(
            prob.get_val("propmodel.motor1.shaft_power_out", units="kW"), np.ones(11) * 0.9 * 240 * 0.97, tolerance=1e-6
        )
        assert_near_equal(
            prob.get_val("propmodel.motor1.elec_load", units="kW"), np.ones(11) * 0.9 * 240, tolerance=1e-6
        )
        assert_near_equal(
            prob.get_val("propmodel.motor2.shaft_power_out", units="kW"), np.ones(11) * 0.9 * 240 * 0.97, tolerance=1e-6
        )
        assert_near_equal(
            prob.get_val("propmodel.motor2.elec_load", units="kW"), np.ones(11) * 0.9 * 240, tolerance=1e-6
        )
        assert_near_equal(
            prob.get_val("propmodel.inverter1.elec_power_in", units="kW"),
            np.ones(11) * 0.9 * 240 / 0.97,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob.get_val("propmodel.inverter2.elec_power_in", units="kW"),
            np.ones(11) * 0.9 * 240 / 0.97,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob.get_val("propmodel.dc_bus.elec_power_out", units="kW"),
            np.ones(11) * 2 * 0.9 * 240 / 0.97,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob.get_val("propmodel.dc_bus.elec_power_in", units="kW"),
            (np.ones(11) * 2 * 0.9 * 240 / 0.97) / 0.99,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob.get_val("propmodel.batt1.elec_load", units="kW"),
            (np.ones(11) * 2 * 0.9 * 240 / 0.97) / 0.99,
            tolerance=1e-6,
        )

        prob.model.list_inputs(units=True, prom_name=True, shape=True, hierarchical=False, print_arrays=True)

        # show outputs
        prob.model.list_outputs(
            implicit=True,
            explicit=True,
            prom_name=True,
            units=True,
            shape=True,
            bounds=False,
            residuals=False,
            scaling=False,
            hierarchical=False,
            print_arrays=True,
        )

    def test_engine_out_settings(self):  # test nondefault values, with propulsive_active = 0
        prob = om.Problem(ElectricTwinPropExpandedTestGroup(vec_size=11, engine_out=True))
        prob.model.nonlinear_solver = NewtonSolver(iprint=1)
        prob.model.options["assembled_jac_type"] = "csc"
        prob.model.linear_solver = DirectSolver(assemble_jac=True)
        prob.model.nonlinear_solver.options["solve_subsystems"] = True  # important to solve subsystem implicit eqns
        prob.model.nonlinear_solver.options["maxiter"] = 10
        prob.model.nonlinear_solver.options["atol"] = 1e-7
        prob.model.nonlinear_solver.options["rtol"] = 1e-7
        prob.setup(check=True, force_alloc_complex=True)
        # om.n2(prob, show_browser=True)
        prob.run_model()

        # test power and thrust on the inactive branch, should be all zero
        assert_near_equal(prob.get_val("propmodel.inverter2.elec_power_out", units="W"), np.zeros(11), tolerance=1e-6)
        assert_near_equal(prob.get_val("propmodel.inverter2.elec_power_in", units="W"), np.zeros(11), tolerance=1e-6)
        assert_near_equal(prob.get_val("propmodel.motor2.shaft_power_out", units="W"), np.zeros(11), tolerance=1e-6)
        assert_near_equal(prob.get_val("propmodel.gearbox2.shaft_power_out", units="W"), np.zeros(11), tolerance=1e-6)
        assert_near_equal(prob.get_val("propmodel.prop2.thrust", units="N"), np.zeros(11), tolerance=1e-6)
        # test total elec power is only from one motor
        assert_near_equal(
            prob.get_val("propmodel.dc_bus.elec_power_out", units="kW"), np.ones(11) * 0.9 * 240 / 0.97, tolerance=1e-6
        )
        assert_near_equal(
            prob.get_val("propmodel.dc_bus.elec_power_in", units="kW"),
            (np.ones(11) * 0.9 * 240 / 0.97) / 0.99,
            tolerance=1e-6,
        )

        prob.model.list_inputs(units=True, prom_name=True, shape=True, hierarchical=False, print_arrays=True)

        # show outputs
        prob.model.list_outputs(
            implicit=True,
            explicit=True,
            prom_name=True,
            units=True,
            shape=True,
            bounds=False,
            residuals=False,
            scaling=False,
            hierarchical=False,
            print_arrays=True,
        )


# if __name__ == '__main__':  # run the prop system model, check n2, inputs and outputs
#     prob = om.Problem(ElectricTwinPropExpandedTestGroup(vec_size=11, engine_out=False))
#     prob.model.nonlinear_solver = NewtonSolver(iprint=1)
#     prob.model.nonlinear_solver = NewtonSolver(iprint=1)
#     prob.model.options['assembled_jac_type'] = 'csc'
#     prob.model.linear_solver = DirectSolver(assemble_jac=True)
#     prob.model.nonlinear_solver.options['solve_subsystems'] = True
#     prob.model.nonlinear_solver.options['maxiter'] = 10
#     prob.model.nonlinear_solver.options['atol'] = 1e-7
#     prob.model.nonlinear_solver.options['rtol'] = 1e-7
#     prob.setup(check=True, force_alloc_complex=True)
#     om.n2(prob, show_browser=True)
#     prob.run_model()
#
#     prob.model.list_inputs(units=True,
#                            prom_name=True,
#                            shape=True,
#                            hierarchical=False,
#                            print_arrays=True)
#
#     # show outputs
#     prob.model.list_outputs(implicit=True,
#                             explicit=True,
#                             prom_name=True,
#                             units=True,
#                             shape=True,
#                             bounds=False,
#                             residuals=False,
#                             scaling=False,
#                             hierarchical=False,
#                             print_arrays=True)
