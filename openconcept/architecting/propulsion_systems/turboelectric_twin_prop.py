from __future__ import division
from openconcept.components.motor import SimpleMotor
from openconcept.components.splitter import PowerSplit
from openconcept.components.generator import SimpleGenerator
from openconcept.components.turboshaft import SimpleTurboshaft
from openconcept.components.battery import SimpleBattery, SOCBattery
from openconcept.components.propeller import SimplePropeller
from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties
from openconcept.utilities.dvlabel import DVLabel
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp
from openconcept.components.converter import SimpleConverter, SimpleConverterInverted
from openconcept.components.bus import SimpleDCBus, SimpleDCBusInverted
from openconcept.components.gearbox import SimpleGearbox

from openmdao.api import Problem, Group, IndepVarComp, BalanceComp, DirectSolver, NewtonSolver, ScipyKrylov, ExecComp
import numpy as np
from openmdao.api import ExplicitComponent


class TurboelectricTwinPropOneEngine(Group):
    """
    a model for a twin turboelectric propulsion system with one engine

    components included:
        1 turboshaft
        1 generator
        2 motors
        2 constant speed propellers (propeller has either 3 or 4 blades by default)
        1 power adder: to add electric loads from the motors
        1 "propulsor_active" flag component (failedengine)
        1 adder: to add weights and thrust of components
        1 probmodel_weight: to output the total prop system weight (propellers + engines)


    Inputs
    ------
        ac|propulsion|engine|rating : float
            The maximum rated shaft power of the engine (Scalar, W)
        ac|propulsion|propeller|diameter : float
            Diameter of the propeller (Scalar, m)
        ac|propulsion|propeller|power_rating : float
            Power rating of the propeller (Scalar, 'kW')
        ac|propulsion|motor|rating: float
            the rated power of the motor (Scalar, 'kW')
        ac|propulsion|generator|rating: float
            the rated power of the generator (Scalar, 'kW')

    Options
    -------
        num_nodes : float
            Number of analysis points to run (default 1)

    Controls
    --------
        prop|rpm: float
            the propeller rpm (vec, RPM)

    Outputs
    ------
        propulsion_system_weight : float
            The weight of the propulsion system (Scalar, 'kg') Note: battery weight is included
        fuel_flow: float
            The fuel flow consumed in the segment (Vec, 'kg/s')
        thrust: float
            The total thrust of the propulsion system (Vec, 'N')

    needs (external information)
        --------
        fltcond|rho: float
            air density at the specific flight condition (vec, kg/m**3)
        fltcond|Utrue: float
            true airspeed at the flight condition (vec, m/s)
        throttle: float
            throttle input to the engine, fraction from 0-1 (vec, '')
        propulsor_active: float (either 0 or 1)
            a flag to indicate on or off for the connected propulsor either 1 or 0 (vec, '')

    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options['num_nodes']

        # define design variables that are independent of flight condition or control states
        dvlist = [['ac|propulsion|engine|rating', 'eng_rating', 260.0, 'kW'],
                  ['ac|propulsion|propeller|diameter', 'prop_diameter', 2.5, 'm'],
                  ['ac|propulsion|motor|rating', 'motor_rating', 240.0, 'kW'],
                  ['ac|propulsion|propeller|power_rating', 'prop_rating', 240.0, 'kW'],
                  ['ac|propulsion|generator|rating', 'gen_rating', 250.0, 'kW'],
                  ]

        self.add_subsystem('dvs', DVLabel(dvlist), promotes_inputs=["*"], promotes_outputs=["*"])
        # introduce model components
        self.add_subsystem('motor1', SimpleMotor(efficiency=0.97, num_nodes=nn), promotes_inputs=['throttle'])
        self.add_subsystem('prop1', SimplePropeller(num_nodes=nn, num_blades=4),
                           promotes_inputs=["fltcond|*"])
        self.connect('motor1.shaft_power_out', 'prop1.shaft_power_in')

        # propulsion models expect a high-level 'throttle' parameter and a 'propulsor_active' flag to set individual throttles
        failedengine = ElementMultiplyDivideComp()
        failedengine.add_equation('motor2throttle', input_names=['throttle', 'propulsor_active'], vec_size=nn)
        self.add_subsystem('failedmotor', failedengine,
                           promotes_inputs=['throttle', 'propulsor_active'])

        self.add_subsystem('motor2', SimpleMotor(efficiency=0.97, num_nodes=nn))
        self.add_subsystem('prop2', SimplePropeller(num_nodes=nn, num_blades=4),
                           promotes_inputs=["fltcond|*"])
        self.connect('motor2.shaft_power_out', 'prop2.shaft_power_in')
        self.connect('failedmotor.motor2throttle', 'motor2.throttle')

        addpower = AddSubtractComp(output_name='motors_elec_load', input_names=['motor1_elec_load', 'motor2_elec_load'],
                                   units='kW', vec_size=nn)
        addpower.add_equation(output_name='thrust', input_names=['prop1_thrust', 'prop2_thrust'], units='N',
                              vec_size=nn)
        self.add_subsystem('add_power', subsys=addpower, promotes_outputs=['*'])
        self.connect('motor1.elec_load', 'add_power.motor1_elec_load')
        self.connect('motor2.elec_load', 'add_power.motor2_elec_load')
        self.connect('prop1.thrust', 'add_power.prop1_thrust')
        self.connect('prop2.thrust', 'add_power.prop2_thrust')

        self.add_subsystem('eng1', SimpleTurboshaft(num_nodes=nn, weight_inc=0.14 / 1000, weight_base=104),
                           promotes_outputs=["fuel_flow"])
        self.add_subsystem('gen1', SimpleGenerator(efficiency=0.97, num_nodes=nn))

        self.connect('eng1.shaft_power_out', 'gen1.shaft_power_in')

        self.add_subsystem('eng_throttle_set',
                           BalanceComp(name='eng_throttle', val=np.ones((nn,)) * 0.5, units=None, eq_units='kW',
                                       rhs_name='gen_power_required', lhs_name='gen_power_available'))
        # need to use the optimizer to drive hybrid_split.power_out_B to the same value as gen1.elec_power_out
        self.connect('motors_elec_load', 'eng_throttle_set.gen_power_required')
        self.connect('gen1.elec_power_out', 'eng_throttle_set.gen_power_available')
        self.connect('eng_throttle_set.eng_throttle', 'eng1.throttle')

        addweights = AddSubtractComp(output_name='motors_weight', input_names=['motor1_weight', 'motor2_weight'],
                                     units='kg')
        addweights.add_equation(output_name='propellers_weight', input_names=['prop1_weight', 'prop2_weight'],
                                units='kg')
        self.add_subsystem('add_weights', subsys=addweights, promotes_inputs=['*'], promotes_outputs=['*'])

        self.connect('motor1.component_weight', 'motor1_weight')
        self.connect('motor2.component_weight', 'motor2_weight')
        self.connect('prop1.component_weight', 'prop1_weight')
        self.connect('prop2.component_weight', 'prop2_weight')

        # connect design variables to model component inputs
        self.connect('eng_rating', 'eng1.shaft_power_rating')
        self.connect('prop_diameter', ['prop1.diameter', 'prop2.diameter'])
        self.connect('motor_rating', ['motor1.elec_power_rating', 'motor2.elec_power_rating'])
        self.connect('prop_rating', ['prop1.power_rating', 'prop2.power_rating'])
        self.connect('gen_rating', 'gen1.elec_power_rating')

        propmodel_weight = self.add_subsystem('propmodel_weight', ExecComp([
            'propulsion_system_weight = total_engines_weight + total_propellers_weight + total_motors_weight + '
            'total_generators_weight'
        ],
            propulsion_system_weight={'value': 1.0, 'units': 'kg'},
            total_engines_weight={'value': 1.0, 'units': 'kg'},
            total_propellers_weight={'value': 1.0, 'units': 'kg'},
            total_motors_weight={'value': 1.0, 'units': 'kg'},
            total_generators_weight={'value': 1.0, 'units': 'kg'},
        ),
                                              promotes_inputs=['*'],
                                              promotes_outputs=['propulsion_system_weight'])
        self.connect('eng1.component_weight', 'total_engines_weight')
        self.connect('gen1.component_weight', 'total_generators_weight')
        self.connect('motors_weight', 'total_motors_weight')
        self.connect('propellers_weight', 'total_propellers_weight')


class TurboelectricTwinPropOneEngineExpanded(Group):
    """
    a model for a turboelectric twin prop expanded propulsion system
    Expanded model additionally includes rectifiers, inverters, DC bus and gearbox

    components included:
        1 engine
        1 generator
        1 rectifier
        2 splitter
        1 DC bus
        2 inverters: convertes DC power to AC power
        2 motors
        1 power adder: to add electric loads from the inverters
        2 gearbox: to decouple the shaft speed of motors and propellers
        2 constant speed propellers (propeller has either 3 or 4 blades by default)
        1 "propulsor_active" flag component (failedengine)
        1 adder: to add weights and thrust of components
        1 probmodel_weight: to output the total propulsion system weight


    Inputs
    ------
        ac|propulsion|engine|rating : float
            The maximum rated shaft power of the engine (Scalar, W)
        ac|propulsion|propeller|diameter : float
            Diameter of the propeller (Scalar, m)
        ac|propulsion|propeller|max_speed : float
            maximum rotational speed of the propeller (Scalar, rpm)
        ac|propulsion|propeller|min_speed : float
            minimum rotational speed of the propeller (Scalar, rpm)
        ac|propulsion|propeller|power_rating : float
            Power rating of the propeller (Scalar, 'kW')
        ac|propulsion|motor|rating: float
            the rated power of the motor (Scalar, 'kW')
        ac|propulsion|motor|max_speed: float
            the maximum shaft speed provided by the motor (Scalar, 'rpm')
        ac|propulsion|generator|rating: float
            the rated power of the generator (Scalar, 'kW')

    Options
    -------
        num_nodes : float
            Number of analysis points to run (default 1)

    Controls
    --------
        prop|rpm: float
            the propeller rpm (vec, RPM)

    Outputs
    ------
        propulsion_system_weight : float
            The weight of the propulsion system (Scalar, 'kg') Note: battery weight is included
        fuel_flow: float
            The fuel flow consumed in the segment (Vec, 'kg/s')
        thrust: float
            The total thrust of the propulsion system (Vec, 'N')

    needs (external information)
        --------
        fltcond|rho: float
            air density at the specific flight condition (vec, kg/m**3)
        fltcond|Utrue: float
            true airspeed at the flight condition (vec, m/s)
        throttle: float
            throttle input to the engine, fraction from 0-1 (vec, '')
        propulsor_active: float (either 0 or 1)
            a flag to indicate on or off for the connected propulsor either 1 or 0 (vec, '')

    """


    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options['num_nodes']

        # define design variables that are independent of flight condition or control states
        dvlist = [['ac|propulsion|engine|rating', 'eng_rating', 260.0, 'kW'],
                  ['ac|propulsion|propeller|diameter', 'prop_diameter', 2.5, 'm'],
                  ['ac|propulsion|propeller|max_speed', 'prop_max_speed', 2900.0, 'rpm'],
                  ['ac|propulsion|propeller|min_speed', 'prop_min_speed', 400.0, 'rpm'],
                  ['ac|propulsion|motor|rating', 'motor_rating', 240.0, 'kW'],
                  ['ac|propulsion|motor|max_speed', 'motor_max_speed', 5500.0, 'rpm'],
                  ['ac|propulsion|propeller|power_rating', 'prop_rating', 240.0, 'kW'],
                  ['ac|propulsion|generator|rating', 'gen_rating', 250.0, 'kW'],
                  ]

        self.add_subsystem('dvs', DVLabel(dvlist), promotes_inputs=["*"], promotes_outputs=["*"])
        # introduce model components
        self.add_subsystem('motor1', SimpleMotor(efficiency=0.97, num_nodes=nn), promotes_inputs=['throttle'])
        self.add_subsystem('gearbox1', SimpleGearbox(num_nodes=nn))
        self.add_subsystem('prop1', SimplePropeller(num_nodes=nn, num_blades=4),
                           promotes_inputs=["fltcond|*"])
        self.connect('motor1.shaft_power_out', 'gearbox1.shaft_power_in')
        self.connect('gearbox1.shaft_power_out', 'prop1.shaft_power_in')

        # propulsion models expect a high-level 'throttle' parameter and a 'propulsor_active' flag to set individual throttles
        failedengine = ElementMultiplyDivideComp()
        failedengine.add_equation('motor2throttle', input_names=['throttle', 'propulsor_active'], vec_size=nn)
        self.add_subsystem('failedmotor', failedengine,
                           promotes_inputs=['throttle', 'propulsor_active'])

        self.add_subsystem('motor2', SimpleMotor(efficiency=0.97, num_nodes=nn))
        self.add_subsystem('gearbox2', SimpleGearbox(num_nodes=nn))
        self.add_subsystem('prop2', SimplePropeller(num_nodes=nn, num_blades=4),
                           promotes_inputs=["fltcond|*"])
        self.connect('motor2.shaft_power_out', 'gearbox2.shaft_power_in')
        self.connect('gearbox2.shaft_power_out', 'prop2.shaft_power_in')
        self.connect('failedmotor.motor2throttle', 'motor2.throttle')

        self.add_subsystem('inverter1', SimpleConverterInverted(efficiency=0.97, num_nodes=nn))
        self.add_subsystem('inverter2', SimpleConverterInverted(efficiency=0.97, num_nodes=nn))
        self.connect('motor1.elec_load', 'inverter1.elec_power_out')
        self.connect('motor2.elec_load', 'inverter2.elec_power_out')

        addpower = AddSubtractComp(output_name='total_elec_load', input_names=['inverter1_elec_load',
                                                                               'inverter2_elec_load'],
                                   units='kW', vec_size=nn)
        addpower.add_equation(output_name='thrust', input_names=['prop1_thrust', 'prop2_thrust'], units='N',
                              vec_size=nn)
        self.add_subsystem('add_power', subsys=addpower, promotes_outputs=['*'])
        self.connect('inverter1.elec_power_in', 'add_power.inverter1_elec_load')
        self.connect('inverter2.elec_power_in', 'add_power.inverter2_elec_load')
        self.connect('prop1.thrust', 'add_power.prop1_thrust')
        self.connect('prop2.thrust', 'add_power.prop2_thrust')

        self.add_subsystem('dc_bus', SimpleDCBusInverted(efficiency=0.99, num_nodes=nn))
        self.connect('total_elec_load', 'dc_bus.elec_power_out')

        # add the fuel energy components and engine throttle balancer
        self.add_subsystem('eng1', SimpleTurboshaft(num_nodes=nn, weight_inc=0.14 / 1000, weight_base=104),
                           promotes_outputs=["fuel_flow"])
        self.add_subsystem('gen1', SimpleGenerator(efficiency=0.97, num_nodes=nn))
        self.add_subsystem('rectifier1', SimpleConverter(efficiency=0.97, num_nodes=nn))

        self.connect('eng1.shaft_power_out', 'gen1.shaft_power_in')
        self.connect('gen1.elec_power_out', 'rectifier1.elec_power_in')

        # add a balance comp to solve for engine throttle
        self.add_subsystem('eng_throttle_set',
                           BalanceComp(name='eng_throttle', val=np.ones((nn,)) * 0.5, units=None, eq_units='kW',
                                       rhs_name='rec_power_required', lhs_name='rec_power_available'))
        self.connect('dc_bus.elec_power_in', 'eng_throttle_set.rec_power_required')
        self.connect('rectifier1.elec_power_out', 'eng_throttle_set.rec_power_available')
        self.connect('eng_throttle_set.eng_throttle', 'eng1.throttle')

        addweights = AddSubtractComp(output_name='motors_weight', input_names=['motor1_weight', 'motor2_weight'],
                                     units='kg')
        addweights.add_equation(output_name='propellers_weight', input_names=['prop1_weight', 'prop2_weight'],
                                units='kg')
        addweights.add_equation(output_name='converters_weight',
                                input_names=['inverter1_weight', 'inverter2_weight', 'rectifier1_weight'],
                                units='kg')
        addweights.add_equation(output_name='gearboxes_weight',
                                input_names=['gearbox1_weight', 'gearbox2_weight'],
                                units='kg')
        self.add_subsystem('add_weights', subsys=addweights, promotes_inputs=['*'], promotes_outputs=['*'])

        self.connect('motor1.component_weight', 'motor1_weight')
        self.connect('motor2.component_weight', 'motor2_weight')
        self.connect('prop1.component_weight', 'prop1_weight')
        self.connect('prop2.component_weight', 'prop2_weight')
        self.connect('inverter1.component_weight', 'inverter1_weight')
        self.connect('inverter2.component_weight', 'inverter2_weight')
        self.connect('rectifier1.component_weight', 'rectifier1_weight')
        self.connect('gearbox1.component_weight', 'gearbox1_weight')
        self.connect('gearbox2.component_weight', 'gearbox2_weight')

        # connect design variables to model component inputs
        self.connect('eng_rating', 'eng1.shaft_power_rating')
        self.connect('prop_diameter', ['prop1.diameter', 'prop2.diameter'])
        self.connect('motor_rating', ['motor1.elec_power_rating', 'motor2.elec_power_rating'])
        self.connect('motor_rating', ['inverter1.elec_power_rating', 'inverter2.elec_power_rating'])
        self.connect('prop_rating', ['prop1.power_rating', 'prop2.power_rating'])
        self.connect('prop_rating', ['gearbox1.shaft_power_rating', 'gearbox2.shaft_power_rating'])
        self.connect('motor_max_speed', ['gearbox1.shaft_speed_in', 'gearbox2.shaft_speed_in'])  # used for sizing
        self.connect('prop_min_speed', ['gearbox1.shaft_speed_out', 'gearbox2.shaft_speed_out'])  # used for sizing
        self.connect('gen_rating', 'gen1.elec_power_rating')
        self.connect('gen_rating', 'rectifier1.elec_power_rating')

        propmodel_weight = self.add_subsystem('propmodel_weight', ExecComp([
            'propulsion_system_weight = total_engines_weight + total_propellers_weight + total_motors_weight + '
            'total_generators_weight + total_converters_weight + total_gearboxes_weight'
        ],
            propulsion_system_weight={'value': 1.0, 'units': 'kg'},
            total_engines_weight={'value': 1.0, 'units': 'kg'},
            total_propellers_weight={'value': 1.0, 'units': 'kg'},
            total_motors_weight={'value': 1.0, 'units': 'kg'},
            total_generators_weight={'value': 1.0, 'units': 'kg'},
            total_converters_weight={'value': 1.0, 'units': 'kg'},
            total_gearboxes_weight={'value': 1.0, 'units': 'kg'},
        ),
                                              promotes_inputs=['*'],
                                              promotes_outputs=['propulsion_system_weight'])
        self.connect('eng1.component_weight', 'total_engines_weight')
        self.connect('gen1.component_weight', 'total_generators_weight')
        self.connect('motors_weight', 'total_motors_weight')
        self.connect('propellers_weight', 'total_propellers_weight')
        self.connect('converters_weight', 'total_converters_weight')
        self.connect('gearboxes_weight', 'total_gearboxes_weight')


class TurboelectricTwinPropTwoEnginesExpanded(Group):
    """
    a model for a turboelectric twin prop expanded propulsion system with two engines
    Expanded model additionally includes rectifiers, inverters, DC bus and gearbox

    components included:
        2 engines
        2 generators
        2 rectifiers
        2 splitter
        1 DC bus
        2 inverters: convertes DC power to AC power
        2 motors
        2 power adder: to add electric loads from the inverters, and rectifiers
        2 gearbox: to decouple the shaft speed of motors and propellers
        2 constant speed propellers (propeller has either 3 or 4 blades by default)
        1 "propulsor_active" flag component (failedengine)
        1 adder: to add weights and thrust of components
        1 probmodel_weight: to output the total propulsion system weight


    Inputs
    ------
        ac|propulsion|engine|rating : float
            The maximum rated shaft power of the engine (Scalar, W)
        ac|propulsion|propeller|diameter : float
            Diameter of the propeller (Scalar, m)
        ac|propulsion|propeller|max_speed : float
            maximum rotational speed of the propeller (Scalar, rpm)
        ac|propulsion|propeller|min_speed : float
            minimum rotational speed of the propeller (Scalar, rpm)
        ac|propulsion|propeller|power_rating : float
            Power rating of the propeller (Scalar, 'kW')
        ac|propulsion|motor|rating: float
            the rated power of the motor (Scalar, 'kW')
        ac|propulsion|motor|max_speed: float
            the maximum shaft speed provided by the motor (Scalar, 'rpm')
        ac|propulsion|generator|rating: float
            the rated power of the generator (Scalar, 'kW')

    Options
    -------
        num_nodes : float
            Number of analysis points to run (default 1)

    Controls
    --------
        prop|rpm: float
            the propeller rpm (vec, RPM)

    Outputs
    ------
        propulsion_system_weight : float
            The weight of the propulsion system (Scalar, 'kg') Note: battery weight is included
        fuel_flow: float
            The fuel flow consumed in the segment (Vec, 'kg/s')
        thrust: float
            The total thrust of the propulsion system (Vec, 'N')

    needs (external information)
        --------
        fltcond|rho: float
            air density at the specific flight condition (vec, kg/m**3)
        fltcond|Utrue: float
            true airspeed at the flight condition (vec, m/s)
        throttle: float
            throttle input to the engine, fraction from 0-1 (vec, '')
        propulsor_active: float (either 0 or 1)
            a flag to indicate on or off for the connected propulsor either 1 or 0 (vec, '')

    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options['num_nodes']

        # define design variables that are independent of flight condition or control states
        dvlist = [['ac|propulsion|engine|rating', 'eng_rating', 260.0, 'kW'],
                  ['ac|propulsion|propeller|diameter', 'prop_diameter', 2.5, 'm'],
                  ['ac|propulsion|propeller|max_speed', 'prop_max_speed', 2900.0, 'rpm'],
                  ['ac|propulsion|propeller|min_speed', 'prop_min_speed', 400.0, 'rpm'],
                  ['ac|propulsion|motor|rating', 'motor_rating', 240.0, 'kW'],
                  ['ac|propulsion|motor|max_speed', 'motor_max_speed', 5500.0, 'rpm'],
                  ['ac|propulsion|propeller|power_rating', 'prop_rating', 240.0, 'kW'],
                  ['ac|propulsion|generator|rating', 'gen_rating', 250.0, 'kW'],
                  ]

        self.add_subsystem('dvs', DVLabel(dvlist), promotes_inputs=["*"], promotes_outputs=["*"])
        # introduce model components
        self.add_subsystem('motor1', SimpleMotor(efficiency=0.97, num_nodes=nn), promotes_inputs=['throttle'])
        self.add_subsystem('gearbox1', SimpleGearbox(num_nodes=nn))
        self.add_subsystem('prop1', SimplePropeller(num_nodes=nn, num_blades=4),
                           promotes_inputs=["fltcond|*"])
        self.connect('motor1.shaft_power_out', 'gearbox1.shaft_power_in')
        self.connect('gearbox1.shaft_power_out', 'prop1.shaft_power_in')

        # propulsion models expect a high-level 'throttle' parameter and a 'propulsor_active' flag to set individual throttles
        failedengine = ElementMultiplyDivideComp()
        failedengine.add_equation('motor2throttle', input_names=['throttle', 'propulsor_active'], vec_size=nn)
        self.add_subsystem('failedmotor', failedengine,
                           promotes_inputs=['throttle', 'propulsor_active'])

        self.add_subsystem('motor2', SimpleMotor(efficiency=0.97, num_nodes=nn))
        self.add_subsystem('gearbox2', SimpleGearbox(num_nodes=nn))
        self.add_subsystem('prop2', SimplePropeller(num_nodes=nn, num_blades=4),
                           promotes_inputs=["fltcond|*"])
        self.connect('motor2.shaft_power_out', 'gearbox2.shaft_power_in')
        self.connect('gearbox2.shaft_power_out', 'prop2.shaft_power_in')
        self.connect('failedmotor.motor2throttle', 'motor2.throttle')

        self.add_subsystem('inverter1', SimpleConverterInverted(efficiency=0.97, num_nodes=nn))
        self.add_subsystem('inverter2', SimpleConverterInverted(efficiency=0.97, num_nodes=nn))
        self.connect('motor1.elec_load', 'inverter1.elec_power_out')
        self.connect('motor2.elec_load', 'inverter2.elec_power_out')

        addpower = AddSubtractComp(output_name='total_elec_load', input_names=['inverter1_elec_load',
                                                                               'inverter2_elec_load'],
                                   units='kW', vec_size=nn)
        addpower.add_equation(output_name='thrust', input_names=['prop1_thrust', 'prop2_thrust'], units='N',
                              vec_size=nn)
        self.add_subsystem('add_power', subsys=addpower, promotes_outputs=['*'])
        self.connect('inverter1.elec_power_in', 'add_power.inverter1_elec_load')
        self.connect('inverter2.elec_power_in', 'add_power.inverter2_elec_load')
        self.connect('prop1.thrust', 'add_power.prop1_thrust')
        self.connect('prop2.thrust', 'add_power.prop2_thrust')

        self.add_subsystem('dc_bus', SimpleDCBusInverted(efficiency=0.99, num_nodes=nn))
        self.connect('total_elec_load', 'dc_bus.elec_power_out')

        # add the fuel energy components and engine throttle balancer
        self.add_subsystem('eng1', SimpleTurboshaft(num_nodes=nn, weight_inc=0.14 / 1000, weight_base=104))
        self.add_subsystem('gen1', SimpleGenerator(efficiency=0.97, num_nodes=nn))
        self.add_subsystem('rectifier1', SimpleConverter(efficiency=0.97, num_nodes=nn))

        self.add_subsystem('eng2', SimpleTurboshaft(num_nodes=nn, weight_inc=0.14 / 1000, weight_base=104))
        self.add_subsystem('gen2', SimpleGenerator(efficiency=0.97, num_nodes=nn))
        self.add_subsystem('rectifier2', SimpleConverter(efficiency=0.97, num_nodes=nn))

        self.connect('eng1.shaft_power_out', 'gen1.shaft_power_in')
        self.connect('gen1.elec_power_out', 'rectifier1.elec_power_in')
        self.connect('eng2.shaft_power_out', 'gen2.shaft_power_in')
        self.connect('gen2.elec_power_out', 'rectifier2.elec_power_in')

        # add rectifier powers and engines fuel flow
        addfuel = AddSubtractComp(output_name='fuel_flow', input_names=['eng1_fuel_flow', 'eng2_fuel_flow'],
                                  vec_size=nn, units='kg/s')
        addfuel.add_equation(output_name='rectifiers_elec_power_out',
                             input_names=['rectifier1_power_out', 'rectifier2_power_out'],
                             vec_size=nn, units='W')
        self.add_subsystem('addfuel', subsys=addfuel, promotes_inputs=["*"], promotes_outputs=["*"])
        self.connect('eng1.fuel_flow', 'eng1_fuel_flow')
        self.connect('eng2.fuel_flow', 'eng2_fuel_flow')
        self.connect('rectifier1.elec_power_out', 'rectifier1_power_out')
        self.connect('rectifier2.elec_power_out', 'rectifier2_power_out')

        # add a balance comp to solve for engine throttle
        self.add_subsystem('eng_throttle_set',
                           BalanceComp(name='eng_throttle', val=np.ones((nn,)) * 0.5, units=None, eq_units='kW',
                                       rhs_name='rec_power_required', lhs_name='rec_power_available'))
        self.connect('dc_bus.elec_power_in', 'eng_throttle_set.rec_power_required')
        self.connect('rectifiers_elec_power_out', 'eng_throttle_set.rec_power_available')
        self.connect('eng_throttle_set.eng_throttle', 'eng1.throttle')
        self.connect('eng_throttle_set.eng_throttle', 'eng2.throttle')

        addweights = AddSubtractComp(output_name='motors_weight', input_names=['motor1_weight', 'motor2_weight'],
                                     units='kg')
        addweights.add_equation(output_name='propellers_weight', input_names=['prop1_weight', 'prop2_weight'],
                                units='kg')
        addweights.add_equation(output_name='converters_weight',
                                input_names=['inverter1_weight', 'inverter2_weight',
                                             'rectifier1_weight', 'rectifier2_weight'],
                                units='kg')
        addweights.add_equation(output_name='gearboxes_weight',
                                input_names=['gearbox1_weight', 'gearbox2_weight'],
                                units='kg')
        addweights.add_equation(output_name='engines_weight', input_names=['eng1_weight', 'eng2_weight'], units='kg')
        addweights.add_equation(output_name='generators_weight', input_names=['gen1_weight', 'gen2_weight'], units='kg')
        self.add_subsystem('add_weights', subsys=addweights, promotes_inputs=['*'], promotes_outputs=['*'])

        self.connect('motor1.component_weight', 'motor1_weight')
        self.connect('motor2.component_weight', 'motor2_weight')
        self.connect('prop1.component_weight', 'prop1_weight')
        self.connect('prop2.component_weight', 'prop2_weight')
        self.connect('eng1.component_weight', 'eng1_weight')
        self.connect('eng2.component_weight', 'eng2_weight')
        self.connect('gen1.component_weight', 'gen1_weight')
        self.connect('gen2.component_weight', 'gen2_weight')
        self.connect('inverter1.component_weight', 'inverter1_weight')
        self.connect('inverter2.component_weight', 'inverter2_weight')
        self.connect('rectifier1.component_weight', 'rectifier1_weight')
        self.connect('rectifier2.component_weight', 'rectifier2_weight')
        self.connect('gearbox1.component_weight', 'gearbox1_weight')
        self.connect('gearbox2.component_weight', 'gearbox2_weight')

        # connect design variables to model component inputs
        self.connect('eng_rating', ['eng1.shaft_power_rating', 'eng2.shaft_power_rating'])
        self.connect('prop_diameter', ['prop1.diameter', 'prop2.diameter'])
        self.connect('motor_rating', ['motor1.elec_power_rating', 'motor2.elec_power_rating'])
        self.connect('motor_rating', ['inverter1.elec_power_rating', 'inverter2.elec_power_rating'])
        self.connect('prop_rating', ['prop1.power_rating', 'prop2.power_rating'])
        self.connect('prop_rating', ['gearbox1.shaft_power_rating', 'gearbox2.shaft_power_rating'])
        self.connect('motor_max_speed', ['gearbox1.shaft_speed_in', 'gearbox2.shaft_speed_in'])  # used for sizing
        self.connect('prop_min_speed', ['gearbox1.shaft_speed_out', 'gearbox2.shaft_speed_out'])  # used for sizing
        self.connect('gen_rating', ['gen1.elec_power_rating', 'gen2.elec_power_rating'])
        self.connect('gen_rating', ['rectifier1.elec_power_rating', 'rectifier2.elec_power_rating'])

        propmodel_weight = self.add_subsystem('propmodel_weight', ExecComp([
            'propulsion_system_weight = total_engines_weight + total_propellers_weight + total_motors_weight + '
            'total_generators_weight + total_converters_weight + total_gearboxes_weight'
        ],
            propulsion_system_weight={'value': 1.0, 'units': 'kg'},
            total_engines_weight={'value': 1.0, 'units': 'kg'},
            total_propellers_weight={'value': 1.0, 'units': 'kg'},
            total_motors_weight={'value': 1.0, 'units': 'kg'},
            total_generators_weight={'value': 1.0, 'units': 'kg'},
            total_converters_weight={'value': 1.0, 'units': 'kg'},
            total_gearboxes_weight={'value': 1.0, 'units': 'kg'},
        ),
                                              promotes_inputs=['*'],
                                              promotes_outputs=['propulsion_system_weight'])
        self.connect('engines_weight', 'total_engines_weight')
        self.connect('generators_weight', 'total_generators_weight')
        self.connect('motors_weight', 'total_motors_weight')
        self.connect('propellers_weight', 'total_propellers_weight')
        self.connect('converters_weight', 'total_converters_weight')
        self.connect('gearboxes_weight', 'total_gearboxes_weight')
