from __future__ import division
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from openmdao.api import IndepVarComp, Group, Problem, n2
from openconcept.components.battery import SOCBattery


class BatterySOCTestGroup(Group):
    """
    Test the battery group that tracks the SOC of the battery during flight
    """

    def initialize(self):
        self.options.declare('vec_size', default=11, desc="Number of mission analysis points to run")
        self.options.declare('efficiency', default=1., desc='Efficiency (dimensionless)')
        self.options.declare('p', default=5000., desc='Battery specific power (W/kg)')
        self.options.declare('e', default=300., desc='Battery spec energy CAREFUL: (Wh/kg)')
        self.options.declare('cost_inc', default=50., desc='$ cost per kg')
        self.options.declare('cost_base', default=1., desc='$ cost base')
        self.options.declare('use_defaults', default=True)

    def setup(self):
        use_defaults = self.options['use_defaults']
        nn = self.options['vec_size']

        iv = self.add_subsystem('iv', IndepVarComp())
        iv.add_output('battery_weight', val=100, units='kg')
        iv.add_output('elec_load', val=np.ones(nn) * 100000, units='W')
        iv.add_output('duration', val=300, units='s')

        if use_defaults:
            battery_soc = self.add_subsystem('BatteryWithSOC', SOCBattery(num_nodes=nn))
        else:
            eta_b = self.options['efficiency']
            p = self.options['p']
            e = self.options['e']
            ci = self.options['cost_inc']
            cb = self.options['cost_base']
            battery_soc = self.add_subsystem('BatteryWithSOC', SOCBattery(num_nodes=nn,
                                                                            efficiency=eta_b,
                                                                            specific_power=p,
                                                                            specific_energy=e,
                                                                            cost_inc=ci,
                                                                            cost_base=cb))

        self.connect('iv.battery_weight', 'BatteryWithSOC.battery_weight')
        self.connect('iv.elec_load', 'BatteryWithSOC.elec_load')
        self.connect('iv.duration', 'BatteryWithSOC.duration')


class BatterySOCTestCase(unittest.TestCase):

    def test_default_settings(self):
        prob = Problem(BatterySOCTestGroup(vec_size=11, use_defaults=True))
        prob.setup(check=True, force_alloc_complex=True)
        # n2(prob, show_browser=True)
        prob.run_model()
        assert_near_equal(prob['BatteryWithSOC.heat_out'], np.ones(11) * 100 * 0.0, tolerance=1e-15)
        assert_near_equal(prob['BatteryWithSOC.component_sizing_margin'], np.ones(11) * 0.20, tolerance=1e-15)
        assert_near_equal(prob['BatteryWithSOC.component_cost'], 5001, tolerance=1e-15)
        assert_near_equal(prob.get_val('BatteryWithSOC.max_energy', units='W*h'), 300 * 100, tolerance=1e-15)
        assert_near_equal(prob.get_val('BatteryWithSOC.dSOCdt', units='W/(kJ)')[0],
                          -100000/(300*100*3.6), tolerance=1e-15)
        assert_near_equal(prob.get_val('BatteryWithSOC.dSOCdt', units='W/(kJ)')[-1],
                          -100000 / (300 * 100 * 3.6), tolerance=1e-15)
        assert_near_equal(prob.get_val('BatteryWithSOC.SOC_initial'), 1.0, tolerance=1e-15)
        assert_near_equal(prob.get_val('BatteryWithSOC.SOC_final'), 1-300*(100/(300 * 100 * 3.6)), tolerance=1e-3)
        assert_near_equal(prob.get_val('BatteryWithSOC.SOC')[1], 1 - 30 * (100 / (300 * 100 * 3.6)), tolerance=1e-3)
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
        partials = prob.check_partials(method='cs', compact_print=True)
        assert_check_partials(partials)

    def test_nondefault_settings(self):
        prob = Problem(BatterySOCTestGroup(vec_size=11,
                                        use_defaults=False,
                                        efficiency=0.95,
                                        p=3000,
                                        e=500,
                                        cost_inc=100,
                                        cost_base=0))
        prob.setup(check=True, force_alloc_complex=True)
        prob.run_model()

        assert_near_equal(prob.get_val('BatteryWithSOC.heat_out', units='kW'), np.ones(11) * 100 * 0.05, tolerance=1e-15)
        assert_near_equal(prob['BatteryWithSOC.component_sizing_margin'], np.ones(11) / 3, tolerance=1e-15)
        assert_near_equal(prob['BatteryWithSOC.component_cost'], 10000, tolerance=1e-15)
        assert_near_equal(prob.get_val('BatteryWithSOC.max_energy', units='W*h'), 500 * 100, tolerance=1e-15)

        assert_near_equal(prob.get_val('BatteryWithSOC.dSOCdt', units='W/(kJ)')[0],
                          -100000/(500*100*3.6), tolerance=1e-15)
        assert_near_equal(prob.get_val('BatteryWithSOC.dSOCdt', units='W/(kJ)')[-1],
                          -100000 / (500 * 100 * 3.6), tolerance=1e-15)
        assert_near_equal(prob.get_val('BatteryWithSOC.SOC_initial'), 1.0, tolerance=1e-15)
        assert_near_equal(prob.get_val('BatteryWithSOC.SOC_final'), 1-300*(100/(500 * 100 * 3.6)), tolerance=1e-3)
        assert_near_equal(prob.get_val('BatteryWithSOC.SOC')[1], 1 - 30 * (100 / (500 * 100 * 3.6)), tolerance=1e-3)
        #     # show outputs
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
        partials = prob.check_partials(method='cs', compact_print=True)
        assert_check_partials(partials)




# if __name__ == '__main__':
#     prob = Problem(BatterySOCTestGroup(vec_size=11, use_defaults=True))
#     prob.setup(check=True, force_alloc_complex=True)
#     n2(prob, show_browser=True)
#     prob.run_model()
#
#     prob.model.list_inputs(units=True,
#                            prom_name=True,
#                            shape=True,
#                            hierarchical=False,
#                            print_arrays=False)
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
#                             print_arrays=False)
#
#
#     # print(prob.get_val('BatteryWithSOC.max_energy'))
#     print(prob.get_val('BatteryWithSOC.elec_load')[0])
#     print(prob.get_val('BatteryWithSOC.battery_weight'))
#     print(prob.get_val('BatteryWithSOC.dSOCdt')[0])
#     print(prob.get_val('BatteryWithSOC.duration'))
#     print(prob.get_val('BatteryWithSOC.SOC_initial'))
#     print(prob.get_val('BatteryWithSOC.SOC_final'))
#     print(prob.get_val('BatteryWithSOC.SOC'))
