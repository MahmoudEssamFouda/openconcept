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

from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem, n2
from openconcept.components.bus import SimpleDCBus, SimpleDCBusInverted, SimpleMechBus


class SimpleDCBusTestGroup(Group):
    """
    Test the SimpleDCBus component
    """

    def initialize(self):
        self.options.declare('vec_size', default=1, desc="Number of mission analysis points to run")
        self.options.declare('use_defaults', default=True)

    def setup(self):
        use_defaults = self.options['use_defaults']
        nn = self.options['vec_size']
        iv = self.add_subsystem('iv', IndepVarComp())
        if not use_defaults:
            iv.add_output('elec_power_in', val=np.ones(nn) * 0.0, units='kW')
        else:
            iv.add_output('elec_power_in', val=np.ones(nn) * 200, units='kW')
        self.add_subsystem('dc_bus', SimpleDCBus(num_nodes=nn))

        self.connect('iv.elec_power_in', 'dc_bus.elec_power_in')


class SimpleDCBusInvertedTestGroup(Group):
    """
    Test the SimpleDCBusInverted component
    """

    def initialize(self):
        self.options.declare('vec_size', default=1, desc="Number of mission analysis points to run")
        self.options.declare('use_defaults', default=True)

    def setup(self):
        use_defaults = self.options['use_defaults']
        nn = self.options['vec_size']
        iv = self.add_subsystem('iv', IndepVarComp())
        if not use_defaults:
            iv.add_output('elec_power_out', val=np.ones(nn) * 0.0, units='kW')
        else:
            iv.add_output('elec_power_out', val=np.ones(nn) * 200, units='kW')
        self.add_subsystem('dc_bus', SimpleDCBusInverted(num_nodes=nn))

        self.connect('iv.elec_power_out', 'dc_bus.elec_power_out')


class SimpleMechBusTestGroup(Group):
    """
    Test the SimpleMechBus component
    """

    def initialize(self):
        self.options.declare('vec_size', default=1, desc="Number of mission analysis points to run")
        self.options.declare('use_defaults', default=True)

    def setup(self):
        use_defaults = self.options['use_defaults']
        nn = self.options['vec_size']
        iv = self.add_subsystem('iv', IndepVarComp())
        if not use_defaults:
            iv.add_output('shaft_power_in', val=np.ones(nn) * 0.0, units='kW')
        else:
            iv.add_output('shaft_power_in', val=np.ones(nn) * 200, units='kW')
        self.add_subsystem('mech_bus', SimpleMechBus(num_nodes=nn, efficiency=0.95, rpm_out=6000))

        self.connect('iv.shaft_power_in', 'mech_bus.shaft_power_in')


class SimpleDCBusTestCase(unittest.TestCase):

    def test_default_settings(self):
        prob = Problem(SimpleDCBusTestGroup(vec_size=11, use_defaults=True))
        prob.setup(check=True, force_alloc_complex=True)
        # n2(prob, show_browser=False)
        prob.run_model()
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
        assert_near_equal(prob.get_val('dc_bus.elec_power_out', units='kW'), np.ones(11) * 200 * 0.99, tolerance=1e-6)
        assert_near_equal(prob.get_val('dc_bus.heat_out', units='kW'), np.ones(11) * 200 * 0.01, tolerance=1e-6)
        partials = prob.check_partials(method='cs', compact_print=True)
        assert_check_partials(partials)

    def test_nondefault_settings(self):  # test the case when power_in = 0, check power_out
        prob = Problem(SimpleDCBusTestGroup(vec_size=11, use_defaults=False))
        prob.setup(check=True, force_alloc_complex=True)
        n2(prob, show_browser=False)
        prob.run_model()
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
        assert_near_equal(prob.get_val('dc_bus.elec_power_out', units='kW'), np.zeros(11), tolerance=1e-6)
        assert_near_equal(prob.get_val('dc_bus.heat_out', units='kW'), np.zeros(11), tolerance=1e-6)
        partials = prob.check_partials(method='cs', compact_print=True)
        assert_check_partials(partials)


class SimpleDCBusInvertedTestCase(unittest.TestCase):

    def test_default_settings(self):
        prob = Problem(SimpleDCBusInvertedTestGroup(vec_size=11, use_defaults=True))
        prob.setup(check=True, force_alloc_complex=True)
        # n2(prob, show_browser=False)
        prob.run_model()
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
        assert_near_equal(prob.get_val('dc_bus.elec_power_in', units='kW'), np.ones(11) * 200 / 0.99, tolerance=1e-6)
        assert_near_equal(prob.get_val('dc_bus.heat_out', units='kW'), np.ones(11) * 200/0.99 * 0.01, tolerance=1e-6)
        partials = prob.check_partials(method='cs', compact_print=True)
        assert_check_partials(partials)

    def test_nondefault_settings(self):  # test the case when power_in = 0, check power_out
        prob = Problem(SimpleDCBusInvertedTestGroup(vec_size=11, use_defaults=False))
        prob.setup(check=True, force_alloc_complex=True)
        # n2(prob, show_browser=False)
        prob.run_model()
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
        assert_near_equal(prob.get_val('dc_bus.elec_power_in', units='kW'), np.zeros(11), tolerance=1e-6)
        assert_near_equal(prob.get_val('dc_bus.heat_out', units='kW'), np.zeros(11), tolerance=1e-6)
        partials = prob.check_partials(method='cs', compact_print=True)
        assert_check_partials(partials)


class SimpleMechBusTestCase(unittest.TestCase):

    def test_default_settings(self):
        prob = Problem(SimpleMechBusTestGroup(vec_size=11, use_defaults=True))
        prob.setup(check=True, force_alloc_complex=True)
        # n2(prob, show_browser=False)
        prob.run_model()
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
        assert_near_equal(prob.get_val('mech_bus.shaft_power_out', units='kW'),
                          np.ones(11) * 200 * 0.95, tolerance=1e-6)
        assert_near_equal(prob.get_val('mech_bus.heat_out', units='kW'), np.ones(11) * 200 * 0.05, tolerance=1e-6)
        assert_near_equal(prob.get_val('mech_bus.output_rpm', units='rpm'), 6000, tolerance=1e-6)
        partials = prob.check_partials(method='cs', compact_print=True)
        assert_check_partials(partials)

    def test_nondefault_settings(self):  # test the case when power_in = 0, check power_out
        prob = Problem(SimpleMechBusTestGroup(vec_size=11, use_defaults=False))
        prob.setup(check=True, force_alloc_complex=True)
        # n2(prob, show_browser=False)
        prob.run_model()
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
        assert_near_equal(prob.get_val('mech_bus.shaft_power_out', units='kW'), np.zeros(11), tolerance=1e-6)
        assert_near_equal(prob.get_val('mech_bus.heat_out', units='kW'), np.zeros(11), tolerance=1e-6)
        assert_near_equal(prob.get_val('mech_bus.output_rpm', units='rpm'), 6000, tolerance=1e-6)
        partials = prob.check_partials(method='cs', compact_print=True)
        assert_check_partials(partials)
