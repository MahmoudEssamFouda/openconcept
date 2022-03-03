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
import openmdao.api as om
from openmdao.api import IndepVarComp, Group, Problem, n2
from openconcept.components.gearbox import SimpleGearbox


class GearboxTestGroup(Group):
    """
    Test the converter component
    """

    def initialize(self):
        self.options.declare('vec_size', default=1, desc="Number of mission analysis points to run")
        self.options.declare('use_defaults', default=True)

    def setup(self):
        use_defaults = self.options['use_defaults']
        nn = self.options['vec_size']
        iv = self.add_subsystem('iv', IndepVarComp())

        iv.add_output('shaft_power_rating', val=250, units='kW')
        iv.add_output('shaft_speed_in', val=6000, units='rpm')
        iv.add_output('shaft_speed_out', val=2000, units='rpm')

        if not use_defaults:
            iv.add_output('shaft_power_in', val=np.ones(nn) * 0.0, units='kW')
        else:
            iv.add_output('shaft_power_in', val=np.ones(nn) * 200.0, units='kW')

        self.add_subsystem('gearbox', SimpleGearbox(num_nodes=nn))
        self.connect('iv.shaft_power_rating', 'gearbox.shaft_power_rating')
        self.connect('iv.shaft_power_in', 'gearbox.shaft_power_in')
        self.connect('iv.shaft_speed_in', 'gearbox.shaft_speed_in')
        self.connect('iv.shaft_speed_out', 'gearbox.shaft_speed_out')


class SimpleGearboxTestCase(unittest.TestCase):

    def test_default_settings(self):
        prob = Problem(GearboxTestGroup(vec_size=11,
                                        use_defaults=True))
        prob.setup(check=True, force_alloc_complex=True)
        n2(prob, show_browser=False)
        prob.run_model()

        assert_near_equal(prob.get_val('gearbox.shaft_power_out', units='kW'), np.ones(11) * 200 * 0.98602518,
                          tolerance=1e-6)
        assert_near_equal(prob.get_val('gearbox.heat_out', units='kW'), np.ones(11) * 200 * (1 - 0.98602518),
                          tolerance=1e-6)
        assert_near_equal(prob.get_val('gearbox.component_weight', units='kg'), 26 * (250 ** 0.76) * (6000 ** 0.13) / (
                2000 ** 0.89), tolerance=1e-6)
        partials = prob.check_partials(method='cs', compact_print=True)
        assert_check_partials(partials)

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

    def test_nondefault_settings(self):  # check if shaft_power_in = 0, when propulsor_active = 0
        prob = Problem(GearboxTestGroup(vec_size=11,
                                        use_defaults=False))
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
        assert_near_equal(prob.get_val('gearbox.shaft_power_out', units='kW'), np.ones(11) * 0.0, tolerance=1e-6)
        assert_near_equal(prob.get_val('gearbox.heat_out', units='kW'), np.ones(11) * 0, tolerance=1e-6)
        assert_near_equal(prob.get_val('gearbox.component_weight', units='kg'), 26 * (250 ** 0.76) * (6000 ** 0.13) / (
                2000 ** 0.89), tolerance=1e-6)
        partials = prob.check_partials(method='cs', compact_print=True)
        # check partials fails for 'efficiency' wrt 'shaft_power_in' for gearbox component when shaft_power_in = 0
        # this is because partial derivatives for efficiency equation is undefined at zero "shaft_power_in"
        # assert_check_partials(partials)
        # om.partial_deriv_plot('efficiency', 'shaft_power_in', partials, binary=False)


