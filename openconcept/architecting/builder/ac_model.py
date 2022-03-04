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

import numpy as np
import openmdao.api as om
import openconcept.api as oc
from openconcept.analysis.aerodynamics import PolarDrag
from openconcept.utilities.math.integrals import Integrator
from openconcept.architecting.builder.architecture import *
from openconcept.architecting.builder.arch_group import DynamicPropulsionArchitecture
from openconcept.analysis.performance.mission_profiles import FullMissionAnalysis
from examples.aircraft_data.KingAirC90GT import data as acdata

__all__ = ['DynamicACModel', 'DynamicKingAirAnalysisGroup']


class DynamicACModel(oc.IntegratorGroup):
    """
    OpenConcept-compliant aircraft model. Should be created using the DynamicACModel.factory function (see below).

    Note: the aircraft weight is calculated by simply subtracting the fuel flow from the MTOW over the course of the
    mission. The propulsion architecture system weight is added as an output, and should be integrated into the OEW
    external to OpenConcept!

    Usage in the setup function of your main analysis group:
    ```
    arch = PropSysArch(...)

    mission_model = MissionWithReserve(  # Can be replaced by any other OpenConcept mission class
        num_nodes=nn,
        aircraft_model=DynamicACModel.factory(arch),
    )
    ```

    Options
    -------
        num_nodes : float
            Number of analysis points to run (default 1)
        flight_phase : str|None
            Name of the flight phase (default: None)
        architecture: PropSysArch
            Propulsion system architecture description to use.

    Inputs
    --------
        fltcond|*: float
            Flight conditions during the mission segment (vec)
                fltcond|rho     Air density     (kg/m**3)
                fltcond|Utrue   True airspeed   (m/s)
                fltcond|CL      Trimmed CL      (-)
                fltcond|q       Dynamic pressure (Pa)
        throttle: float
            Throttle input to the engine, fraction from 0-1 (vec, -)
        propulsor_active: float (either 0 or 1)
            A flag to indicate on or off for the connected propulsor either 1 or 0 (vec, -)
        duration: float
            The amount of time to finish the segment in seconds (scalar, s)
        ac|*: float
            Aircraft design parameters (scalar)
                ac|aero|polar|CD0_cruise    CD0 in cruise       (-)
                ac|aero|polar|CD0_TO        CD0 in take-off     (-)
                ac|aero|polar|e             Oswald factor       (-)
                ac|aero|wing|S_ref          Wing reference area (m**2)
                ac|aero|wing|AR             Wing aspect ratio   (-)
                ac|weights|MTOW             Max take-off weight (kg) <-- used as initial wt during mission simulation

    Outputs
    ------
        drag: float
            Total drag of the aircraft (Vec, N)
        thrust: float
            Total thrust of the propulsion system (Vec, N)
        weight: float
            Total weight of the aircraft, calculated from MTOW and fuel flow (Vec, kg)
        seg_fuel_used: float
            Total fuel used in the mission segment (Scalar, kg)
        propulsion_system_weight : float
            The weight of the propulsion system (Scalar, kg)
    """

    @classmethod
    def factory(cls, architecture: PropSysArch):
        def _factory(num_nodes=1, flight_phase=None):
            return cls(num_nodes=num_nodes, flight_phase=flight_phase, architecture=architecture)
        return _factory

    def initialize(self):
        self.options.declare('num_nodes', default=1)
        self.options.declare('flight_phase', default=None)
        self.options.declare('architecture', types=PropSysArch, desc='The propulsion system architecture definition')

    def setup(self):
        nn = self.options['num_nodes']
        self._add_propulsion_model(nn)
        self._add_drag_model(nn)
        self._add_weight_model(nn)

    def _add_propulsion_model(self, nn):
        self.add_subsystem(
            'propmodel', DynamicPropulsionArchitecture(num_nodes=nn, architecture=self.options['architecture']),
            promotes_inputs=['fltcond|*', 'throttle', 'propulsor_active', 'duration'],
            promotes_outputs=['fuel_flow', 'thrust', 'propulsion_system_weight'],
        )

    def _add_drag_model(self, nn):
        # Determine CD0 source based on flight phase
        flight_phase = self.options['flight_phase']
        if flight_phase not in ['v0v1', 'v1v0', 'v1vr', 'rotate']:
            cd0_source = 'ac|aero|polar|CD0_cruise'
        else:
            cd0_source = 'ac|aero|polar|CD0_TO'

        # Add drag model based on simple drag polar model
        self.add_subsystem(
            'drag', PolarDrag(num_nodes=nn),
            promotes_inputs=['fltcond|CL', 'ac|geom|*', ('CD0', cd0_source), 'fltcond|q', ('e', 'ac|aero|polar|e')],
            promotes_outputs=['drag'],
        )

    def _add_weight_model(self, nn):
        # Integrate fuel flow
        fuel_int = self.add_subsystem(
            'fuel_int', Integrator(num_nodes=nn, method='simpson', diff_units='s', time_setup='duration'),
            promotes_inputs=['*'], promotes_outputs=['*'])
        fuel_int.add_integrand('fuel_used', rate_name='fuel_flow', val=1.0, units='kg')

        # Calculate weight by subtracting fuel used from MTOW
        # Note that fuel used is accumulated over all mission phases, therefore fuel_used here represents the total fuel
        # used since the first mission phase
        self.add_subsystem(
            'weight', oc.AddSubtractComp(output_name='weight', input_names=['ac|weights|MTOW', 'fuel_used'], units='kg',
                                         vec_size=(1, nn), scaling_factors=[1, -1]),
            promotes_inputs=['*'], promotes_outputs=['weight'])

        # Calculate total fuel used in this mission segment
        self.add_subsystem(
            'seg_fuel_used', om.ExecComp(['seg_fuel_used=sum(fuel_used)'],
                                         seg_fuel_used={'val': 1.0, 'units': 'kg'},
                                         fuel_used={'val': np.ones((nn,)), 'units': 'kg'},
        ), promotes_inputs=['*'], promotes_outputs=['*'])

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

def setup_problem():
    prob = om.Problem()
    prob.model = DynamicKingAirAnalysisGroup()
    prob.model.nonlinear_solver = om.NewtonSolver(iprint=2)
    prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(print_bound_enforce=False)
    prob.model.options['assembled_jac_type'] = 'csc'
    prob.model.linear_solver = om.DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.options['solve_subsystems'] = True
    prob.model.nonlinear_solver.options['maxiter'] = 15
    prob.model.nonlinear_solver.options['atol'] = 1e-6
    prob.model.nonlinear_solver.options['rtol'] = 1e-6

    prob.setup()

    # Set required mission parameters
    num_nodes = 11
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
    prob.run_model()
    om.n2(prob, show_browser=False)

    # Print some results
    vars_list = ['ac|weights|MTOW','descent.fuel_used_final','rotate.range_final']
    units = ['lb','lb','ft']
    nice_print_names = ['MTOW', 'Fuel used', 'TOFL (over 35ft obstacle)']
    print("=======================================================================")
    for i, thing in enumerate(vars_list):
        print(nice_print_names[i]+': '+str(prob.get_val(thing,units=units[i])[0])+' '+units[i])
