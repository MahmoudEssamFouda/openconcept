import openmdao.api as om
import unittest
import numpy as np
from openmdao.utils.assert_utils import assert_near_equal, assert_check_partials
from examples.propulsion_layouts.simple_series_hybrid import TwinSeriesHybridElectricPropulsionSystem
from openconcept.architecting.propulsion_systems.series_hybrid_twin_prop \
    import SeriesHybridTwinProp, SeriesHybridTwinPropExpanded
from openconcept.utilities.linearinterp import LinearInterpolator
from openmdao.api import DirectSolver, IndepVarComp, NewtonSolver, BoundsEnforceLS


class TwinSeriesHybridTestGroup(om.Group):
    """
    Test the Twin series hybrid propulsion system
    """

    def initialize(self):
        self.options.declare('vec_size', default=11, desc="Number of mission analysis points to run")
        self.options.declare('engine_out', default=True)

    def setup(self):
        nn = self.options['vec_size']
        engine_out = self.options['engine_out']

        controls = self.add_subsystem('controls', om.IndepVarComp(), promotes_outputs=['*'])
        controls.add_output('prop|rpm', val=np.ones(nn) * 1900, units='rpm')
        controls.add_output('hybridization', val=0.6)

        hybrid_factor = self.add_subsystem('hybrid_factor', LinearInterpolator(num_nodes=nn),
                                           promotes_inputs=[('start_val', 'hybridization'),
                                                            ('end_val', 'hybridization')])

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes_outputs=['*'])
        dvs.add_output('ac|propulsion|propeller|diameter', val=2.3, units='m')
        dvs.add_output('ac|propulsion|engine|rating', val=260, units='kW')
        dvs.add_output('ac|propulsion|generator|rating', val=250, units='kW')
        dvs.add_output('ac|propulsion|motor|rating', val=240, units='kW')
        dvs.add_output('ac|propulsion|propeller|power_rating', val=240, units='kW')
        dvs.add_output('ac|weights|W_battery', val=1000, units='kg')
        dvs.add_output('ac|propulsion|battery|specific_energy', val=300, units='W*h/kg')

        flt_cond = self.add_subsystem('flt_cond', om.IndepVarComp(), promotes_outputs=['*'])
        flt_cond.add_output('fltcond|rho', units='kg / m ** 3', desc='Air density', val=np.ones(nn) * 0.475448)
        flt_cond.add_output('fltcond|Utrue', units='m/s', desc='Flight speed', val=np.ones(nn) * 92.5)

        mission_prms = self.add_subsystem('mission_prms', om.IndepVarComp(), promotes_outputs=['*'])
        mission_prms.add_output('throttle', val=np.ones(nn) * 0.90)
        mission_prms.add_output('duration', val=300, units='s')
        if engine_out:
            mission_prms.add_output('propulsor_active', val=np.ones(nn) * 0.0)
        else:
            mission_prms.add_output('propulsor_active', val=np.ones(nn) * 1.0)

        # define propulsion system
        propulsion_promotes_outputs = ['fuel_flow', 'thrust']
        propulsion_promotes_inputs = ["ac|propulsion|*", "fltcond|*", "throttle", "propulsor_active",
                                      "ac|weights*", 'duration']  # ac|weights* is only used for battery weight

        self.add_subsystem('propmodel', SeriesHybridTwinProp(num_nodes=nn),
                           promotes_inputs=propulsion_promotes_inputs,
                           promotes_outputs=propulsion_promotes_outputs)

        self.connect('prop|rpm', ['propmodel.prop1.rpm', 'propmodel.prop2.rpm'])
        self.connect('hybrid_factor.vec', 'propmodel.hybrid_split.power_split_fraction')


class TwinSeriesHybridExpandedTestGroup(om.Group):
    """
    Test the Twin series hybrid expanded propulsion system
    """

    def initialize(self):
        self.options.declare('vec_size', default=11, desc="Number of mission analysis points to run")
        self.options.declare('engine_out', default=True)

    def setup(self):
        nn = self.options['vec_size']
        engine_out = self.options['engine_out']

        controls = self.add_subsystem('controls', om.IndepVarComp(), promotes_outputs=['*'])
        controls.add_output('prop|rpm', val=np.ones(nn) * 1900, units='rpm')
        controls.add_output('hybridization', val=0.6)

        hybrid_factor = self.add_subsystem('hybrid_factor', LinearInterpolator(num_nodes=nn),
                                           promotes_inputs=[('start_val', 'hybridization'),
                                                            ('end_val', 'hybridization')])

        dvs = self.add_subsystem('dvs', om.IndepVarComp(), promotes_outputs=['*'])
        dvs.add_output('ac|propulsion|propeller|diameter', val=2.3, units='m')
        dvs.add_output('ac|propulsion|propeller|max_speed', val=2900.0, units='rpm')
        dvs.add_output('ac|propulsion|propeller|min_speed', val=400.0, units='rpm')
        dvs.add_output('ac|propulsion|motor|rating', val=240, units='kW')
        dvs.add_output('ac|propulsion|motor|max_speed', val=5500.0, units='rpm')
        dvs.add_output('ac|propulsion|propeller|power_rating', val=240, units='kW')
        dvs.add_output('ac|propulsion|splitter|power_rating', val=99999999, units='W')
        dvs.add_output('ac|propulsion|engine|rating', val=260, units='kW')
        dvs.add_output('ac|propulsion|generator|rating', val=250, units='kW')
        dvs.add_output('ac|weights|W_battery', val=1000, units='kg')
        dvs.add_output('ac|propulsion|battery|specific_energy', val=300, units='W*h/kg')
        dvs.add_output('ac|propulsion|battery|SOC_initial', val=1.0)

        flt_cond = self.add_subsystem('flt_cond', om.IndepVarComp(), promotes_outputs=['*'])
        flt_cond.add_output('fltcond|rho', units='kg / m ** 3', desc='Air density', val=np.ones(nn) * 0.475448)
        flt_cond.add_output('fltcond|Utrue', units='m/s', desc='Flight speed', val=np.ones(nn) * 92.5)

        mission_prms = self.add_subsystem('mission_prms', om.IndepVarComp(), promotes_outputs=['*'])
        mission_prms.add_output('throttle', val=np.ones(nn) * 0.90)
        mission_prms.add_output('duration', val=300, units='s')
        if engine_out:
            mission_prms.add_output('propulsor_active', val=np.ones(nn) * 0.0)
        else:
            mission_prms.add_output('propulsor_active', val=np.ones(nn) * 1.0)

        # define propulsion system
        propulsion_promotes_outputs = ['fuel_flow', 'thrust']
        propulsion_promotes_inputs = ["ac|propulsion|*", "fltcond|*", "throttle", "propulsor_active",
                                      "ac|weights*", 'duration']  # ac|weights* is only used for battery weight

        self.add_subsystem('propmodel', SeriesHybridTwinPropExpanded(num_nodes=nn),
                           promotes_inputs=propulsion_promotes_inputs,
                           promotes_outputs=propulsion_promotes_outputs)

        self.connect('prop|rpm', ['propmodel.prop1.rpm', 'propmodel.prop2.rpm'])
        self.connect('hybrid_factor.vec', 'propmodel.hybrid_split.power_split_fraction')


class TwinSeriesHybridTestCase(unittest.TestCase):

    def test_default_settings(self):
        prob = om.Problem(TwinSeriesHybridTestGroup(vec_size=11, engine_out=False))
        prob.model.nonlinear_solver = NewtonSolver(iprint=1)
        prob.model.options['assembled_jac_type'] = 'csc'
        prob.model.linear_solver = DirectSolver(assemble_jac=True)
        prob.model.nonlinear_solver.options['solve_subsystems'] = True  # important to solve subsystem implicit eqns
        prob.model.nonlinear_solver.options['maxiter'] = 10
        prob.model.nonlinear_solver.options['atol'] = 1e-7
        prob.model.nonlinear_solver.options['rtol'] = 1e-7
        prob.setup(check=True, force_alloc_complex=True)
        # om.n2(prob, show_browser=False)
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
        # check motors calculations
        assert_near_equal(prob.get_val('propmodel.motor1.shaft_power_out', units='W'),
                          np.ones(11) * 0.9 * 240 * 1000 * 0.97, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.motor1.elec_load', units='W'),
                          np.ones(11) * 0.9 * 240 * 1000, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.failedmotor.motor2throttle'),
                          np.ones(11) * 0.9, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.motor2.shaft_power_out', units='W'),
                          np.ones(11) * 0.9 * 240 * 1000 * 0.97, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.motor2.elec_load', units='W'),
                          np.ones(11) * 0.9 * 240 * 1000, tolerance=1e-6)

        # check propeller 1 and thrust calculations
        assert_near_equal(prob.get_val('propmodel.prop1.cp'),
                          np.ones(11) * 0.9 * 240 * 1000 * 0.97 / (0.475448 * ((1900 / 60) ** 3) * 2.3 ** 5),
                          tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.prop1.propmap.J'),
                          np.ones(11) * 92.5 / ((1900 / 60) * 2.3), tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.prop1.prop_Vtip'),
                          np.ones(11) * ((1900 / 60) * 2.3 * np.pi), tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.prop1.eta_prop'), np.ones(11) * 0.7818752716122873, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.prop1.ct_over_cp'), np.ones(11) * 1.072771961229587, tolerance=1e-6)
        # due to unit conversions, prop weight is calculated with 3 significant figures, so tolerance is 1e-3
        assert_near_equal(prob.get_val('propmodel.prop1.component_weight', units='lb'),
                          0.108 * (2.3 * 3.28084 * 240 * 1.34102 * (4 ** 0.5)) ** 0.782,
                          tolerance=1e-3)  # 1 m = 3.28 ft
        # check the calculations for prop 2
        assert_near_equal(prob.get_val('propmodel.prop2.cp'),
                          np.ones(11) * 0.9 * 240 * 1000 * 0.97 / (0.475448 * ((1900 / 60) ** 3) * 2.3 ** 5),
                          tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.prop2.propmap.J'),
                          np.ones(11) * 92.5 / ((1900 / 60) * 2.3), tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.prop2.prop_Vtip'),
                          np.ones(11) * ((1900 / 60) * 2.3 * np.pi), tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.prop2.eta_prop'), np.ones(11) * 0.7818752716122873, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.prop2.ct_over_cp'), np.ones(11) * 1.072771961229587, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.prop2.component_weight', units='lb'),
                          0.108 * (2.3 * 3.28084 * 240 * 1.34102 * (4 ** 0.5)) ** 0.782,
                          tolerance=1e-3)  # 1 m = 3.28 ft
        # check the thrust calculations and elec load
        assert_near_equal(prob.get_val('thrust', units='N'),
                          (2 * np.ones(11) * (0.9 * 240 * 1000 * 0.97 / (
                                  0.475448 * ((1900 / 60) ** 3) * 2.3 ** 5)) * 0.7818752716122873 /
                           (92.5 / ((1900 / 60) * 2.3))) * 0.475448 * ((1900 / 60) ** 2) * 2.3 ** 4, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.motors_elec_load', units='W'),
                          np.ones(11) * 0.9 * 240 * 1000 * 2, tolerance=1e-6)
        # check power splitter calculations
        assert_near_equal(prob.get_val('propmodel.hybrid_split.power_out_A', units='W'),
                          np.ones(11) * 0.9 * 240 * 1000 * 2 * 0.6, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.hybrid_split.power_out_B', units='W'),
                          np.ones(11) * 0.9 * 240 * 1000 * 2 * 0.4, tolerance=1e-6)

        # check battery calculations
        assert_near_equal(prob.get_val('propmodel.batt1.batt_base.heat_out', units='W'),
                          np.ones(11) * 0.9 * 240 * 1000 * 2 * 0.6 * 0.03, tolerance=1e-15)
        assert_near_equal(prob.get_val('propmodel.batt1.batt_base.max_energy', units='W*h'),
                          300 * 1000, tolerance=1e-15)
        assert_near_equal(prob.get_val('propmodel.batt1.dSOCdt', units='W/(kJ)')[0],
                          - 0.9 * 240 * 1000 * 2 * 0.6 / (300 * 1000 * 3.6), tolerance=1e-15)  # 1 Wh = 3.6 kJ
        assert_near_equal(prob.get_val('propmodel.batt1.dSOCdt', units='W/(kJ)')[-1],
                          - 0.9 * 240 * 1000 * 2 * 0.6 / (300 * 1000 * 3.6), tolerance=1e-15)  # # 1 Wh = 3.6 kJ
        assert_near_equal(prob.get_val('propmodel.batt1.SOC_initial'), 1.0, tolerance=1e-15)
        assert_near_equal(prob.get_val('propmodel.batt1.SOC_final'),
                          1 - 300 * ((0.9 * 240 * 2 * 0.6) / (300 * 1000 * 3.6)), tolerance=1e-3)
        assert_near_equal(prob.get_val('propmodel.batt1.SOC')[1],
                          1 - 30 * ((0.9 * 240 * 2 * 0.6) / (300 * 1000 * 3.6)), tolerance=1e-3)

        # check the generator calculations
        assert_near_equal(prob.get_val('propmodel.gen1.elec_power_out', units='W'),
                          np.ones(11) * 0.9 * 240 * 1000 * 2 * 0.4, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.gen1.shaft_power_in', units='W'),
                          (np.ones(11) * 0.9 * 240 * 1000 * 2 * 0.4) / 0.97, tolerance=1e-6)

        # check the engine calculations
        assert_near_equal(prob.get_val('propmodel.eng1.shaft_power_out', units='W'),
                          (np.ones(11) * 0.9 * 240 * 1000 * 2 * 0.4) / 0.97, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.eng1.throttle'),
                          ((np.ones(11) * 0.9 * 240 * 1000 * 2 * 0.4) / 0.97) / (260 * 1000), tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.eng1.fuel_flow', units='kg/s'),
                          np.ones(11) * 0.6851705 * 260 * 1000 * 0.6 * 1.69e-7, tolerance=1e-3)

    def test_engine_out_settings(self):  # check one engine out case when propulsor_active = 0
        prob = om.Problem(TwinSeriesHybridTestGroup(vec_size=11, engine_out=True))
        prob.model.nonlinear_solver = NewtonSolver(iprint=1)
        prob.model.options['assembled_jac_type'] = 'csc'
        prob.model.linear_solver = DirectSolver(assemble_jac=True)
        prob.model.nonlinear_solver.options['solve_subsystems'] = True  # important to solve subsystem implicit eqns
        prob.model.nonlinear_solver.options['maxiter'] = 10
        prob.model.nonlinear_solver.options['atol'] = 1e-7
        prob.model.nonlinear_solver.options['rtol'] = 1e-7
        prob.setup(check=True, force_alloc_complex=True)
        # om.n2(prob, show_browser=False)
        prob.run_model()
        assert_near_equal(prob.get_val('propmodel.motor2.shaft_power_out', units='W'),
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


class TwinSeriesHybridExpandedTestCase(unittest.TestCase):

    def test_default_settings(self):
        prob = om.Problem(TwinSeriesHybridExpandedTestGroup(vec_size=11, engine_out=False))
        prob.model.nonlinear_solver = NewtonSolver(iprint=1)
        prob.model.nonlinear_solver = NewtonSolver(iprint=1)
        prob.model.options['assembled_jac_type'] = 'csc'
        prob.model.linear_solver = DirectSolver(assemble_jac=True)
        prob.model.nonlinear_solver.options['solve_subsystems'] = True
        prob.model.nonlinear_solver.options['maxiter'] = 20
        prob.model.nonlinear_solver.options['atol'] = 1e-7
        prob.model.nonlinear_solver.options['rtol'] = 1e-7
        prob.setup(check=True, force_alloc_complex=True)
        # om.n2(prob, show_browser=True)
        prob.run_model()

        assert_near_equal(prob.get_val('propmodel.motor1.shaft_power_out', units='kW'),
                          np.ones(11) * 0.9 * 240 * 0.97, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.motor1.elec_load', units='kW'),
                          np.ones(11) * 0.9 * 240, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.motor2.shaft_power_out', units='kW'),
                          np.ones(11) * 0.9 * 240 * 0.97, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.motor2.elec_load', units='kW'),
                          np.ones(11) * 0.9 * 240, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.inverter1.elec_power_in', units='kW'),
                          np.ones(11) * 0.9 * 240 / 0.97, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.inverter2.elec_power_in', units='kW'),
                          np.ones(11) * 0.9 * 240 / 0.97, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.dc_bus.elec_power_out', units='kW'),
                          np.ones(11) * 2 * 0.9 * 240 / 0.97, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.dc_bus.elec_power_in', units='kW'),
                          (np.ones(11) * 2 * 0.9 * 240 / 0.97) / 0.99, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.batt1.elec_load', units='kW'),
                          ((np.ones(11) * 2 * 0.9 * 240 / 0.97) / 0.99)*0.6, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.rectifier1.elec_power_out', units='kW'),
                          ((np.ones(11) * 2 * 0.9 * 240 / 0.97) / 0.99) * 0.4, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.rectifier1.elec_power_out', units='kW'),
                          prob.get_val('propmodel.hybrid_split.power_out_B', units='kW'), tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.batt1.elec_load', units='kW'),
                          prob.get_val('propmodel.hybrid_split.power_out_A', units='kW'), tolerance=1e-6)

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

    def test_engine_out_settings(self):  # test one engine out case with propulsive_active = 0
        prob = om.Problem(TwinSeriesHybridExpandedTestGroup(vec_size=11, engine_out=True))
        prob.model.nonlinear_solver = NewtonSolver(iprint=1)
        prob.model.options['assembled_jac_type'] = 'csc'
        prob.model.linear_solver = DirectSolver(assemble_jac=True)
        prob.model.nonlinear_solver.options['solve_subsystems'] = True  # important to solve subsystem implicit eqns
        prob.model.nonlinear_solver.options['maxiter'] = 10
        prob.model.nonlinear_solver.options['atol'] = 1e-7
        prob.model.nonlinear_solver.options['rtol'] = 1e-7
        prob.setup(check=True, force_alloc_complex=True)
        # om.n2(prob, show_browser=True)
        prob.run_model()
        # test power and thrust on the inactive branch, should be all zero
        assert_near_equal(prob.get_val('propmodel.inverter2.elec_power_out', units='W'),
                          np.zeros(11), tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.motor2.shaft_power_out', units='W'),
                          np.zeros(11), tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.gearbox2.shaft_power_out', units='W'),
                          np.zeros(11), tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.prop2.thrust', units='N'),
                          np.zeros(11), tolerance=1e-6)
        # test total elec power is only from one motor
        assert_near_equal(prob.get_val('propmodel.dc_bus.elec_power_out', units='kW'),
                          np.ones(11) * 0.9 * 240 / 0.97, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.dc_bus.elec_power_in', units='kW'),
                          (np.ones(11) * 0.9 * 240 / 0.97) / 0.99, tolerance=1e-6)
        assert_near_equal(prob.get_val('propmodel.rectifier1.elec_power_out', units='kW'),
                          prob.get_val('propmodel.hybrid_split.power_out_B', units='kW'), tolerance=1e-6)

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
#     prob = om.Problem(TwinSeriesHybridExpandedTestGroup(vec_size=11, engine_out=False))
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
