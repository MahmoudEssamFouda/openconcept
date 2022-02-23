from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem, n2
from openconcept.components.converter import SimpleConverter, SimpleConverterInverted


class SimpleConverterTestGroup(Group):
    """
    Test the SimpleConverter component
    """

    def initialize(self):
        self.options.declare('vec_size', default=1, desc="Number of mission analysis points to run")
        self.options.declare('efficiency', default=0.95, desc='Efficiency (dimensionless)')
        self.options.declare('weight_inc', default=1 / (10 * 1000), desc='kg/W')
        self.options.declare('weight_base', default=0., desc='kg base weight')
        self.options.declare('cost_inc', default=100.0 / 745.0, desc='$ cost per watt')
        self.options.declare('cost_base', default=1., desc='$ cost base')
        self.options.declare('use_defaults', default=True)

    def setup(self):
        use_defaults = self.options['use_defaults']
        nn = self.options['vec_size']
        iv = self.add_subsystem('iv', IndepVarComp())
        iv.add_output('elec_power_rating', val=250, units='kW')
        if not use_defaults:
            iv.add_output('elec_power_out', val=np.ones(nn) * 0.0, units='kW')
        else:
            iv.add_output('elec_power_out', val=np.ones(nn) * 200.0, units='kW')
        self.add_subsystem('rectifier', SimpleConverter(num_nodes=nn))

        self.connect('iv.elec_power_rating', 'rectifier.elec_power_rating')
        self.connect('iv.elec_power_out', 'rectifier.elec_power_in')


class SimpleConverterInvertedTestGroup(Group):
    """
    Test the SimpleConverterInverted component
    """

    def initialize(self):
        self.options.declare('vec_size', default=1, desc="Number of mission analysis points to run")
        self.options.declare('efficiency', default=0.95, desc='Efficiency (dimensionless)')
        self.options.declare('weight_inc', default=1 / (10 * 1000), desc='kg/W')
        self.options.declare('weight_base', default=0., desc='kg base weight')
        self.options.declare('cost_inc', default=100.0 / 745.0, desc='$ cost per watt')
        self.options.declare('cost_base', default=1., desc='$ cost base')
        self.options.declare('use_defaults', default=True)

    def setup(self):
        use_defaults = self.options['use_defaults']
        nn = self.options['vec_size']
        iv = self.add_subsystem('iv', IndepVarComp())
        iv.add_output('elec_power_rating', val=250, units='kW')
        if not use_defaults:
            iv.add_output('elec_power_out', val=np.ones(nn) * 0.0, units='kW')
        else:
            iv.add_output('elec_power_out', val=np.ones(nn) * 200.0, units='kW')
        self.add_subsystem('inverter', SimpleConverterInverted(num_nodes=nn))

        self.connect('iv.elec_power_rating', 'inverter.elec_power_rating')
        self.connect('iv.elec_power_out', 'inverter.elec_power_out')


class SimpleConverterTestCase(unittest.TestCase):

    def test_default_settings(self):
        prob = Problem(SimpleConverterTestGroup(vec_size=11,use_defaults=True))
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
        assert_near_equal(prob.get_val('rectifier.elec_power_out', units='kW'), np.ones(11) * 200 * 0.95,
                          tolerance=1e-6)
        assert_near_equal(prob.get_val('rectifier.heat_out', units='kW'), np.ones(11) * 200 * 0.05, tolerance=1e-6)
        assert_near_equal(prob.get_val('rectifier.component_sizing_margin'),
                          np.ones(11) * 200 * 0.95 / 250, tolerance=1e-6)
        assert_near_equal(prob.get_val('rectifier.component_cost', units='USD'),
                          250 * 1000 * 100.0 / 745.0 + 1.0, tolerance=1e-6)
        assert_near_equal(prob.get_val('rectifier.component_weight', units='kg'), 250 * 1000 / (10 * 1000),
                          tolerance=1e-6)
        partials = prob.check_partials(method='cs', compact_print=True)
        assert_check_partials(partials)

    def test_nondefault_settings(self):  # test the case when power_in is zero, , check power_out and heat_out
        prob = Problem(SimpleConverterTestGroup(vec_size=11, use_defaults=False))
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
        assert_near_equal(prob.get_val('rectifier.elec_power_out', units='kW'), np.zeros(11), tolerance=1e-6)
        assert_near_equal(prob.get_val('rectifier.heat_out', units='kW'), np.zeros(11), tolerance=1e-6)
        assert_near_equal(prob.get_val('rectifier.component_sizing_margin'), np.zeros(11), tolerance=1e-6)
        assert_near_equal(prob.get_val('rectifier.component_cost', units='USD'),
                          250 * 1000 * 100.0/745.0 + 1.0, tolerance=1e-6)
        assert_near_equal(prob.get_val('rectifier.component_weight', units='kg'), 250 * 1000 / (10 * 1000),
                          tolerance=1e-6)
        partials = prob.check_partials(method='cs', compact_print=True)
        assert_check_partials(partials)


class SimpleConverterInvertedTestCase(unittest.TestCase):

    def test_default_settings(self):
        prob = Problem(SimpleConverterInvertedTestGroup(vec_size=11, use_defaults=True))
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
        assert_near_equal(prob.get_val('inverter.elec_power_in', units='kW'), np.ones(11) * (200 / 0.95),
                          tolerance=1e-6)
        assert_near_equal(prob.get_val('inverter.heat_out', units='kW'), np.ones(11) * (200 / 0.95) * 0.05,
                          tolerance=1e-6)
        assert_near_equal(prob.get_val('inverter.component_sizing_margin'),
                          np.ones(11) * 200 / 250, tolerance=1e-6)
        assert_near_equal(prob.get_val('inverter.component_cost', units='USD'),
                          250 * 1000 * 100.0 / 745.0 + 1.0, tolerance=1e-6)
        assert_near_equal(prob.get_val('inverter.component_weight', units='kg'), 250 * 1000 / (10 * 1000),
                          tolerance=1e-6)
        partials = prob.check_partials(method='cs', compact_print=True)
        assert_check_partials(partials)

    def test_nondefault_settings(self):  # test when power_output = 0, check power_in and heat_out
        prob = Problem(SimpleConverterInvertedTestGroup(vec_size=11, use_defaults=False))
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
        assert_near_equal(prob.get_val('inverter.elec_power_in', units='kW'), np.zeros(11), tolerance=1e-6)
        assert_near_equal(prob.get_val('inverter.heat_out', units='kW'), np.zeros(11), tolerance=1e-6)
        assert_near_equal(prob.get_val('inverter.component_sizing_margin'), np.zeros(11), tolerance=1e-6)
        assert_near_equal(prob.get_val('inverter.component_cost', units='USD'),
                          250 * 1000 * 100.0 / 745.0 + 1.0, tolerance=1e-6)
        assert_near_equal(prob.get_val('inverter.component_weight', units='kg'), 250 * 1000 / (10 * 1000),
                          tolerance=1e-6)
        partials = prob.check_partials(method='cs', compact_print=True)
        assert_check_partials(partials)

