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

import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from typing import *
import openmdao.api as om
from openconcept.architecting.builder.utils import *
from openconcept.architecting.builder.architecture import *
from openconcept.architecting.builder.arch_group import DynamicPropulsionArchitecture


class DynamicConventionalTwinTurbopropTestGroup(om.Group):
    """
    Test the dynamic conventional Twin turboprop propulsion system
    """

    def initialize(self):
        self.options.declare("vec_size", default=11, desc="Number of mission analysis points to run")
        self.options.declare("architecture", types=PropSysArch, desc="The propulsion system architecture definition")
        self.options.declare("engine_out", default=False)

    def setup(self):
        nn = self.options["vec_size"]
        arch = self.options["architecture"]
        engine_out = self.options["engine_out"]

        controls = self.add_subsystem("controls", om.IndepVarComp(), promotes_outputs=["*"])
        controls.add_output("prop|rpm", val=np.ones(nn) * 1900, units="rpm")

        flt_cond = self.add_subsystem("flt_cond", om.IndepVarComp(), promotes_outputs=["*"])
        flt_cond.add_output("fltcond|rho", units="kg / m ** 3", desc="Air density", val=np.ones(nn) * 0.475448)
        flt_cond.add_output("fltcond|Utrue", units="m/s", desc="Flight speed", val=np.ones(nn) * 92.5)

        mission_prms = self.add_subsystem("mission_prms", om.IndepVarComp(), promotes_outputs=["*"])
        mission_prms.add_output("throttle", val=np.ones(nn) * 0.90)
        mission_prms.add_output("duration", val=300, units="s")
        if engine_out:
            mission_prms.add_output("propulsor_active", val=np.ones(nn) * 0.0)
        else:
            mission_prms.add_output("propulsor_active", val=np.ones(nn) * 1.0)

        # define propulsion system
        propulsion_promotes_outputs = ["propulsion_system_weight", "fuel_flow", "thrust", "SOC"]
        propulsion_promotes_inputs = ["fltcond|*", "throttle", "propulsor_active", "duration", "prop|rpm"]

        self.add_subsystem(
            "propmodel",
            DynamicPropulsionArchitecture(num_nodes=nn, architecture=arch),
            promotes_inputs=propulsion_promotes_inputs,
            promotes_outputs=propulsion_promotes_outputs,
        )


class DynamicConventionalTwinTurbopropTestCase(unittest.TestCase):
    def test_default_settings(self):  # test all engines active
        arch = PropSysArch(  # Conventional with gearbox
            thrust=ThrustGenElements(
                propellers=[
                    Propeller(name="prop1", blades=4, diameter=2.3),
                    Propeller(name="prop2", blades=4, diameter=2.3),
                ],
                gearboxes=[
                    Gearbox(name="gearbox1"),
                    Gearbox(name="gearbox2"),
                ],
            ),
            mech=MechPowerElements(
                engines=[
                    Engine(name="turboshaft", power_rating=850 * 0.7457, output_rpm=6000),
                    Engine(name="turboshaft", power_rating=850 * 0.7457, output_rpm=6000),
                ]
            ),
        )
        prob = om.Problem(DynamicConventionalTwinTurbopropTestGroup(vec_size=11, architecture=arch, engine_out=False))
        prob.setup(check=True, force_alloc_complex=True)
        # om.n2(prob, show_browser=True)
        prob.run_model()

        assert_near_equal(
            prob.get_val("fuel_flow", units="kg/s"),
            2 * np.ones(11) * 0.9 * 850 * 745.7 * 0.6 * 1.68965774e-7,
            tolerance=1e-6,
        )
        # check weight components and sum
        assert_near_equal(
            prob.get_val("propmodel.mech.mech1.turboshaft.component_weight", units="kg"),
            850 * 745.7 * 0.14 / 1000 + 104,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob.get_val("propmodel.thrust1.prop1.component_weight", units="lbm"),
            0.108 * (2.3 * 3.28084 * 850 * (4**0.5)) ** 0.782,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob.get_val("propmodel.thrust1.gearbox1.component_weight", units="kg"),
            26 * ((850 * 0.7457) ** 0.76) * (6000**0.13) / (1900**0.89),
            tolerance=1e-6,
        )
        # 1 m = 3.28084 ft
        assert_near_equal(
            prob.get_val("propulsion_system_weight", units="kg"),
            2 * (850 * 745.7 * 0.14 / 1000 + 104)
            + 2 * 0.453592 * (0.108 * (2.3 * 3.28084 * 850 * (4**0.5)) ** 0.782)
            + 2 * (26 * ((850 * 0.7457) ** 0.76) * (6000**0.13) / (1900**0.89)),
            tolerance=1e-6,
        )
        # check thrust output
        # gearbox efficiency with these parameters = 0.9875942790377175
        # air density = 0.475448
        # prop efficiency based on forward flight map at these flight conditions = 0.78457275
        assert_near_equal(
            prob.get_val("thrust", units="N"),
            (
                2
                * np.ones(11)
                * (0.9875942790377175 * 0.9 * 850 * 745.7 / (0.475448 * ((1900 / 60) ** 3) * 2.3**5))
                * 0.78457275
                / (92.5 / ((1900 / 60) * 2.3))
            )
            * 0.475448
            * ((1900 / 60) ** 2)
            * 2.3**4,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob.get_val("propmodel.SOC", units=None),
            np.zeros(
                11,
            ),
            tolerance=1e-6,
        )  # no battery

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

    def test_nondefault_settings(self):  # test one engine inoperative case
        arch = PropSysArch(  # Conventional with gearbox
            thrust=ThrustGenElements(
                propellers=[
                    Propeller(name="prop1", blades=4, diameter=2.3),
                    Propeller(name="prop2", blades=4, diameter=2.3),
                ],
                gearboxes=[
                    Gearbox(name="gearbox1"),
                    Gearbox(name="gearbox2"),
                ],
            ),
            mech=MechPowerElements(
                engines=[
                    Engine(name="turboshaft", power_rating=850 * 0.7457, output_rpm=6000),
                    Engine(name="turboshaft", power_rating=850 * 0.7457, output_rpm=6000),
                ]
            ),
        )
        prob = om.Problem(DynamicConventionalTwinTurbopropTestGroup(vec_size=11, architecture=arch, engine_out=True))
        prob.setup(check=True, force_alloc_complex=True)
        # om.n2(prob, show_browser=True)
        prob.run_model()

        # check fuel flow is half, only one engine is active
        assert_near_equal(
            prob.get_val("fuel_flow", units="kg/s"),
            1 * np.ones(11) * 0.9 * 850 * 745.7 * 0.6 * 1.68965774e-7,
            tolerance=1e-6,
        )

        # check weight components and sum
        assert_near_equal(
            prob.get_val("propmodel.mech.mech1.turboshaft.component_weight", units="kg"),
            850 * 745.7 * 0.14 / 1000 + 104,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob.get_val("propmodel.thrust1.prop1.component_weight", units="lbm"),
            0.108 * (2.3 * 3.28084 * 850 * (4**0.5)) ** 0.782,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob.get_val("propmodel.thrust1.gearbox1.component_weight", units="kg"),
            26 * ((850 * 0.7457) ** 0.76) * (6000**0.13) / (1900**0.89),
            tolerance=1e-6,
        )
        # 1 m = 3.28084 ft
        assert_near_equal(
            prob.get_val("propulsion_system_weight", units="kg"),
            2 * (850 * 745.7 * 0.14 / 1000 + 104)
            + 2 * 0.453592 * (0.108 * (2.3 * 3.28084 * 850 * (4**0.5)) ** 0.782)
            + 2 * (26 * ((850 * 0.7457) ** 0.76) * (6000**0.13) / (1900**0.89)),
            tolerance=1e-6,
        )

        # # check thrust is half, only one engine is active
        # gearbox efficiency with these parameters = 0.9875942790377175
        # air density = 0.475448
        # prop efficiency based on forward flight map at these flight conditions = 0.78457275
        assert_near_equal(
            prob.get_val("thrust", units="N"),
            (
                1
                * np.ones(11)
                * (0.9875942790377175 * 0.9 * 850 * 745.7 / (0.475448 * ((1900 / 60) ** 3) * 2.3**5))
                * 0.78457275
                / (92.5 / ((1900 / 60) * 2.3))
            )
            * 0.475448
            * ((1900 / 60) ** 2)
            * 2.3**4,
            tolerance=1e-6,
        )

        # check shaft power output and thrust of inactive components
        assert_near_equal(
            prob.get_val("propmodel.mech.mech2.turboshaft.shaft_power_out", units="W"),
            np.ones(11) * 0.0,
            tolerance=1e-6,
        )
        assert_near_equal(
            prob.get_val("propmodel.thrust2.gearbox2.shaft_power_out", units="W"), np.ones(11) * 0.0, tolerance=1e-6
        )
        assert_near_equal(prob.get_val("propmodel.thrust2.thrust", units="N"), np.ones(11) * 0.0, tolerance=1e-6)

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
