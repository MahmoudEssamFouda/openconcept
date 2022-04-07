import numpy as np
import openmdao.api as om
import openconcept.api as oc
from openconcept.architecting.builder.architecture import PropSysArch, ThrustGenElements, \
                                                          MechPowerElements, Propeller, Gearbox, Engine
from openconcept.architecting.builder.ac_model import DynamicACModel
from openconcept.analysis.performance.mission_profiles import FullMissionAnalysis
from examples.aircraft_data.KingAirC90GT import data as acdata

class DynamicKingAirAnalysisGroup(om.Group):
    """
    This is an example similar to the default OpenConcept King Air, but
    with a propulsion system defined with the propulsion system builder.
    """
    def setup(self):
        # Take parameters that are not already defined in the DynamicACModel
        # from the default King Air C90GT data dictionary
        dv_comp = self.add_subsystem('dv_comp',  oc.DictIndepVarComp(acdata),
                                     promotes_outputs=["*"])
        dv_comp.add_output_from_dict('ac|aero|polar|e')
        dv_comp.add_output_from_dict('ac|aero|polar|CD0_TO')
        dv_comp.add_output_from_dict('ac|aero|polar|CD0_cruise')
        dv_comp.add_output_from_dict('ac|geom|wing|S_ref')
        dv_comp.add_output_from_dict('ac|geom|wing|AR')
        dv_comp.add_output_from_dict('ac|weights|MTOW')

        arch = PropSysArch(  # Conventional with gearbox
            thrust=ThrustGenElements(propellers=[Propeller('prop1'), Propeller('prop2')],
                                    gearboxes=[Gearbox('gearbox1'), Gearbox('gearbox2')]),
            mech=MechPowerElements(engines=Engine('turboshaft', power_rating=560.)),
        )

        self.add_subsystem('mission', FullMissionAnalysis(
                                num_nodes=11, aircraft_model=DynamicACModel.factory(arch),
                            ), promotes_inputs=['*'], promotes_outputs=['*'])
        
        # Compute OEW using the original aircraft OEW, original propulsion
        # system weight, and current propulsion system weight
        self.add_subsystem(
            'OEW_calc', oc.AddSubtractComp(output_name='OEW', input_names=['OEW_orig', 'W_prop_orig', 'W_prop'], units='kg',
                                         scaling_factors=[1, -1, 1]),
            promotes_outputs=[('OEW', 'ac|weights|OEW')])
        self.set_input_defaults('OEW_calc.OEW_orig', 7000., units='lb')
        self.set_input_defaults('OEW_calc.W_prop_orig', 444.873, units='kg')
        self.connect('cruise.propulsion_system_weight', 'OEW_calc.W_prop')
        
        # Compute MTOW - fuel burn - ZFW (OEW and payload) residual to be used as a constraint in the
        # optimization problem (> 0) when MTOW is a design variable
        self.add_subsystem(
            'TOW_res', oc.AddSubtractComp(output_name='residual', input_names=['MTOW', 'fuel_used', 'OEW', 'payload'], units='kg',
                                         scaling_factors=[1, -1, -1, -1]),
            promotes_inputs=[('MTOW', 'ac|weights|MTOW'), ('OEW', 'ac|weights|OEW'), 'payload'])
        self.connect('descent.fuel_used_final', 'TOW_res.fuel_used')

        # Promote propulsion system variables that are used in the optimization problem as DVs
        segments = ['v0v1', 'v1vr', 'rotate', 'v1v0', 'engineoutclimb', 'climb', 'cruise', 'descent']
        var_promote = [(['propmodel.mech.mech1.eng_rating', 'propmodel.mech.mech2.eng_rating'], 'ac|propulsion|engine|rating'),
                       (['propmodel.thrust1.diameter', 'propmodel.thrust2.diameter'], 'ac|propulsion|propeller|diameter')]
        promotes_list = []
        for segment in segments:
            for var in var_promote:
                for abs_var in var[0]:
                    promotes_list.append((f"{segment}.{abs_var}", var[1]))
        self.promotes('mission', inputs=promotes_list)


def opt_prob():
    """
    Optimization problem definition. Can be used for analyses too.
    """
    prob = om.Problem()
    prob.model = DynamicKingAirAnalysisGroup()
    prob.model.nonlinear_solver = om.NewtonSolver(iprint=2, err_on_non_converge=False)
    prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(print_bound_enforce=False)
    prob.model.options['assembled_jac_type'] = 'csc'
    prob.model.linear_solver = om.DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.options['maxiter'] = 15
    prob.model.nonlinear_solver.options['atol'] = 1e-8
    prob.model.nonlinear_solver.options['rtol'] = 1e-8

    # Setup the optimization
    prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", maxiter=200, tol=1e-8, disp=True)
    prob.driver.options['debug_print'] = ['objs', 'desvars', 'nl_cons']

    prob.model.add_design_var('ac|propulsion|engine|rating', lower=200., upper=1000., ref=5e2)
    prob.model.add_design_var('ac|propulsion|propeller|diameter', lower=2.2)
    prob.model.add_objective('descent.fuel_used_final')

    # Throttle can't exceed 1 in any flight segments (limited to max power of engine)
    segments = ['climb', 'cruise', 'descent']
    for segment in segments:
        prob.model.add_constraint(f"{segment}.throttle", lower=0., upper=1.)
    
    # Use the optimizer to solve for initial weight (called MTOW here) by having initial
    # weight as a design variable and preventing the final weight from being less than
    # the zero fuel weight (OEW + payload)
    prob.model.add_design_var('ac|weights|MTOW', lower=2e3, ref=5e3)
    prob.model.add_constraint('TOW_res.residual', lower=0.)

    # Balanced field length must be at least as good as baseline
    prob.model.add_constraint('rotate.range_final', upper=4452., units='ft')
    prob.model.add_constraint('v1v0.range_final', upper=4452., units='ft')

    return prob

def setup_problem():
    prob = opt_prob()
    prob.setup()

    # Set required mission parameters
    num_nodes = 11
    prob.set_val('ac|weights|MTOW', 10e3, units='lb')
    prob.set_val('ac|propulsion|engine|rating', 500, units='kW')
    prob.set_val('climb.fltcond|vs', np.ones((num_nodes,))*1500, units='ft/min')
    prob.set_val('climb.fltcond|Ueas', np.ones((num_nodes,))*124, units='kn')
    prob.set_val('cruise.fltcond|vs', np.ones((num_nodes,))*0.01, units='ft/min')
    prob.set_val('cruise.fltcond|Ueas', np.ones((num_nodes,))*170, units='kn')
    prob.set_val('descent.fltcond|vs', np.ones((num_nodes,))*(-600), units='ft/min')
    prob.set_val('descent.fltcond|Ueas', np.ones((num_nodes,))*140, units='kn')

    prob.set_val('cruise|h0',29000,units='ft')
    prob.set_val('mission_range',1000,units='NM')
    prob.set_val('payload',1000,units='lb')

    # (optional) guesses for takeoff speeds may help with convergence
    prob.set_val('v0v1.fltcond|Utrue',np.ones((num_nodes))*50,units='kn')
    prob.set_val('v1vr.fltcond|Utrue',np.ones((num_nodes))*85,units='kn')
    prob.set_val('v1v0.fltcond|Utrue',np.ones((num_nodes))*85,units='kn')

    # set some airplane-specific values. The throttle edits are to derate the takeoff power of the PT6A
    # prob['climb.OEW.structural_fudge'] = 1.67
    prob.set_val('v0v1.throttle', np.ones((num_nodes)) * 0.75)
    prob.set_val('v1vr.throttle', np.ones((num_nodes)) * 0.75)
    prob.set_val('rotate.throttle', np.ones((num_nodes)) * 0.75)

    return prob

if __name__ == '__main__':
    prob = setup_problem()
    # prob.run_model()
    prob.run_driver()
    om.n2(prob, show_browser=False)

    # Print some results
    vars_list = ['ac|weights|MTOW','descent.fuel_used_final','rotate.range_final']
    units = ['lb','lb','ft']
    nice_print_names = ['MTOW', 'Fuel used', 'TOFL (over 35ft obstacle)']
    print("=======================================================================")
    for i, thing in enumerate(vars_list):
        print(nice_print_names[i]+': '+str(prob.get_val(thing,units=units[i])[0])+' '+units[i])
