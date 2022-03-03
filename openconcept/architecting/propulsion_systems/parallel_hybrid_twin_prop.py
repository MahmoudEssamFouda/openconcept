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
from openconcept.components.motor import SimpleMotor
from openconcept.components.splitter import PowerSplit
from openconcept.components.generator import SimpleGenerator
from openconcept.components.turboshaft import SimpleTurboshaft
from openconcept.components.battery import SimpleBattery, SOCBattery
from openconcept.components.propeller import SimplePropeller
from openconcept.components.converter import SimpleConverter, SimpleConverterInverted
from openconcept.components.gearbox import SimpleGearbox
from openconcept.components.bus import SimpleDCBus, SimpleDCBusInverted, SimpleMechBus

from openconcept.analysis.atmospherics.compute_atmos_props import ComputeAtmosphericProperties
from openconcept.utilities.dvlabel import DVLabel
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp

from openmdao.api import Problem, Group, IndepVarComp, BalanceComp, DirectSolver, NewtonSolver, ScipyKrylov, ExecComp
from openmdao.api import ExplicitComponent

import numpy as np


class ParallelHybridTwinPropExpanded(Group):
    """
    a model for a parallel hybrid twin prop propulsion system
    Expanded model additionally includes rectifiers, inverters, DC bus and gearbox

    components included:
        2 turboshaft engines
        1 battery
        1 DC bus
        2 inverters: convertes DC power to AC power
        1 splitter components
            to split dc power output
        2 motors
        1 power adder: to add elec power of the inverters
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
        ac|weights|W_battery: float
            the weight of the battery (Scalar, 'kg')
        ac|propulsion|battery|specific_energy: float
            the specific energy density of the battery (Scalar, 'Wh/kg')

    Options
    -------
        num_nodes : float
            Number of analysis points to run (default 1)

    Controls
    --------
        prop|rpm: float
            the propeller rpm (vec, RPM)
        hybridization: float -> might need to be transferred to a vector
            the degree of hybridization of the prop system, fraction form 0 to 1 (Scalar, '')

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
        duration: float
            the amount of time to finish the segment in seconds (Scalar, 's')

    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options['num_nodes']

        # define design variables that are independent of flight condition or control states
        dvlist = [['ac|propulsion|engine|rating', 'eng_rating', 260.0, 'kW'],
                  ['ac|propulsion|propeller|diameter', 'prop_diameter', 2.5, 'm'],
                  ['ac|propulsion|propeller|max_speed', 'prop_max_speed', 2900.0, 'rpm'],  # not used
                  ['ac|propulsion|propeller|min_speed', 'prop_min_speed', 400.0, 'rpm'],  # used for sizing the gearbox
                  ['ac|propulsion|motor|rating', 'motor_rating', 260.0, 'kW'],
                  ['ac|propulsion|motor|max_speed', 'motor_max_speed', 5500.0, 'rpm'],
                  ['ac|propulsion|propeller|power_rating', 'prop_rating', 240.0, 'kW'],
                  ['ac|propulsion|splitter|power_rating', 'splitter_rating', 99999999.0, 'W'],
                  ['ac|weights|W_battery', 'batt_weight', 1000, 'kg'],
                  ['ac|propulsion|battery|specific_energy', 'specific_energy', 300, 'W*h/kg'],
                  ]

        self.add_subsystem('dvs', DVLabel(dvlist), promotes_inputs=["*"], promotes_outputs=["*"])
        # introduce model components
        self.add_subsystem('motor1', SimpleMotor(efficiency=0.97, num_nodes=nn))
        self.add_subsystem('eng1', SimpleTurboshaft(num_nodes=nn, weight_inc=0.14 / 1000, weight_base=104))

        self.add_subsystem('motor2', SimpleMotor(efficiency=0.97, num_nodes=nn))
        self.add_subsystem('eng2', SimpleTurboshaft(num_nodes=nn, weight_inc=0.14 / 1000, weight_base=104))

        # add rated powers with Exec component, eng+motor
        # OEI case: total_rated_power_2 = power of eng2 only
        get_rated_power = self.add_subsystem('get_rated_power', ExecComp([
            'total_rated_power_1 = eng1_rated_power + motor1_rated_power',
            'total_rated_power_2 = eng2_rated_power + propulsor_active[0]*motor2_rated_power',
        ],
            total_rated_power_1={'value': 1.0, 'units': 'kW'},
            total_rated_power_2={'value': 1.0, 'units': 'kW'},
            propulsor_active={'value': np.ones(nn)},
            eng1_rated_power={'value': 1.0, 'units': 'kW'},
            eng2_rated_power={'value': 1.0, 'units': 'kW'},
            motor1_rated_power={'value': 1.0, 'units': 'kW'},
            motor2_rated_power={'value': 1.0, 'units': 'kW'},


        ), promotes_inputs=['eng1_rated_power', 'motor1_rated_power', 'eng2_rated_power',
                            'propulsor_active', 'motor2_rated_power'],
                                             promotes_outputs=['total_rated_power_1', 'total_rated_power_2'])
        self.connect('eng_rating', ['eng1_rated_power', 'eng2_rated_power'])
        self.connect('motor_rating', ['motor1_rated_power', 'motor2_rated_power'])

        # find total shaft power
        get_shaft_power = self.add_subsystem('get_shaft_power', ExecComp([
            'tot_shaft_power_1 = throttle * total_rated_power_1',
            'tot_shaft_power_2 = throttle * total_rated_power_2',
        ],
            tot_shaft_power_1={'value': np.ones(nn), 'units': 'kW'},
            tot_shaft_power_2={'value': np.ones(nn), 'units': 'kW'},
            throttle={'value': np.ones(nn)},
            total_rated_power_1={'value': 1.0, 'units': 'kW'},
            total_rated_power_2={'value': 1.0, 'units': 'kW'},

        ), promotes_inputs=['throttle', 'total_rated_power_1', 'total_rated_power_2'],
                                             promotes_outputs=['tot_shaft_power_1', 'tot_shaft_power_2'])

        # add hybrid split for motor 1 and eng 1
        self.add_subsystem('hybrid_split_1', PowerSplit(rule='fraction', num_nodes=nn, efficiency=1))
        # connect total shaft power for motor 1 and eng 1 to the power into the splitter
        self.connect('tot_shaft_power_1', 'hybrid_split_1.power_in')

        # add a balance comp to solve for eng1 throttle
        self.add_subsystem('eng1_throttle_set',
                           BalanceComp(name='eng_throttle', val=np.ones((nn,)) * 0.5, units=None, eq_units='kW',
                                       rhs_name='shaft_power_required', lhs_name='shaft_power_available'))
        self.connect('hybrid_split_1.power_out_B', 'eng1_throttle_set.shaft_power_required')
        self.connect('eng1.shaft_power_out', 'eng1_throttle_set.shaft_power_available')
        self.connect('eng1_throttle_set.eng_throttle', 'eng1.throttle')

        # add a balance comp to solve for motor1 throttle
        self.add_subsystem('motor1_throttle_set',
                           BalanceComp(name='motor1_throttle', val=np.ones((nn,)) * 0.5, units=None, eq_units='kW',
                                       rhs_name='shaft_power_required', lhs_name='shaft_power_available'))
        self.connect('hybrid_split_1.power_out_A', 'motor1_throttle_set.shaft_power_required')
        self.connect('motor1.shaft_power_out', 'motor1_throttle_set.shaft_power_available')
        self.connect('motor1_throttle_set.motor1_throttle', 'motor1.throttle')

        # add hybrid split for motor 2 and eng 2
        # add failed motor flag
        failedmotor = ElementMultiplyDivideComp()
        failedmotor.add_equation('motor2flag', input_names=['hybrid_split_vec_2', 'propulsor_active'], vec_size=nn)
        self.add_subsystem('failedmotor', failedmotor,
                           promotes_inputs=['propulsor_active'])
        self.add_subsystem('hybrid_split_2', PowerSplit(rule='fraction', num_nodes=nn, efficiency=1))
        self.connect('tot_shaft_power_2', 'hybrid_split_2.power_in')
        # motor flag will make motor2 not operative in case propulsor_active is zero, this means all power
        # requirement will be provided by eng2 in that condition
        self.connect('failedmotor.motor2flag', 'hybrid_split_2.power_split_fraction')

        # add a balance comp to solve for eng2 throttle
        self.add_subsystem('eng2_throttle_set',
                           BalanceComp(name='eng_throttle', val=np.ones((nn,)) * 0.5, units=None, eq_units='kW',
                                       rhs_name='shaft_power_required', lhs_name='shaft_power_available'))
        self.connect('hybrid_split_2.power_out_B', 'eng2_throttle_set.shaft_power_required')
        self.connect('eng2.shaft_power_out', 'eng2_throttle_set.shaft_power_available')
        self.connect('eng2_throttle_set.eng_throttle', 'eng2.throttle')

        # add a balance comp to solve for motor2 throttle
        self.add_subsystem('motor2_throttle_set',
                           BalanceComp(name='motor2_throttle', val=np.ones((nn,)) * 0.5, units=None, eq_units='kW',
                                       rhs_name='shaft_power_required', lhs_name='shaft_power_available'))
        self.connect('hybrid_split_2.power_out_A', 'motor2_throttle_set.shaft_power_required')
        self.connect('motor2.shaft_power_out', 'motor2_throttle_set.shaft_power_available')
        self.connect('motor2_throttle_set.motor2_throttle', 'motor2.throttle')

        # add gearboxes and propellers
        self.add_subsystem('MechBus1', SimpleMechBus(efficiency=0.95, rpm_out=5500, num_nodes=nn))
        self.add_subsystem('gearbox1', SimpleGearbox(num_nodes=nn))
        self.add_subsystem('prop1', SimplePropeller(num_nodes=nn, num_blades=4),
                           promotes_inputs=["fltcond|*"])

        self.add_subsystem('MechBus2', SimpleMechBus(efficiency=0.95, rpm_out=5500, num_nodes=nn))
        self.add_subsystem('gearbox2', SimpleGearbox(num_nodes=nn))
        self.add_subsystem('prop2', SimplePropeller(num_nodes=nn, num_blades=4),
                           promotes_inputs=["fltcond|*"])

        self.connect('tot_shaft_power_1', 'MechBus1.shaft_power_in')
        self.connect('MechBus1.shaft_power_out', 'gearbox1.shaft_power_in')
        self.connect('gearbox1.shaft_power_out', 'prop1.shaft_power_in')

        self.connect('tot_shaft_power_2', 'MechBus2.shaft_power_in')
        self.connect('MechBus2.shaft_power_out', 'gearbox2.shaft_power_in')
        self.connect('gearbox2.shaft_power_out', 'prop2.shaft_power_in')

        # add components after motor elec load
        self.add_subsystem('inverter1', SimpleConverterInverted(efficiency=0.97, num_nodes=nn))
        self.add_subsystem('inverter2', SimpleConverterInverted(efficiency=0.97, num_nodes=nn))
        self.connect('motor1.elec_load', 'inverter1.elec_power_out')
        self.connect('motor2.elec_load', 'inverter2.elec_power_out')

        addpower = AddSubtractComp(output_name='total_elec_load', input_names=['inverter1_elec_load',
                                                                               'inverter2_elec_load'],
                                   units='kW', vec_size=nn)
        self.add_subsystem('add_power', subsys=addpower, promotes_outputs=['*'])
        self.connect('inverter1.elec_power_in', 'add_power.inverter1_elec_load')
        self.connect('inverter2.elec_power_in', 'add_power.inverter2_elec_load')

        self.add_subsystem('dc_bus', SimpleDCBusInverted(efficiency=0.99, num_nodes=nn))
        self.connect('total_elec_load', 'dc_bus.elec_power_out')

        self.add_subsystem('batt1', SOCBattery(num_nodes=nn, efficiency=0.97),
                           promotes_inputs=["duration", "specific_energy"])
        self.connect('dc_bus.elec_power_in', 'batt1.elec_load')

        relabel = [['dc_power_in', 'battery_load', np.ones(nn) * 260.0, 'kW']]
        self.add_subsystem('relabel', DVLabel(relabel), promotes_outputs=["battery_load"])
        self.connect('dc_bus.elec_power_in', 'relabel.dc_power_in')

        # add engines fuel flow
        addfuel = AddSubtractComp(output_name='fuel_flow', input_names=['eng1_fuel_flow', 'eng2_fuel_flow'],
                                  vec_size=nn, units='kg/s')
        self.add_subsystem('addfuel', subsys=addfuel, promotes_inputs=["*"], promotes_outputs=["*"])
        self.connect('eng1.fuel_flow', 'eng1_fuel_flow')
        self.connect('eng2.fuel_flow', 'eng2_fuel_flow')

        # add thrust of both propellers and promote it
        addthrust = AddSubtractComp(output_name='thrust', input_names=['prop1_thrust', 'prop2_thrust'], units='N',
                                    vec_size=nn)
        self.add_subsystem('add_thrust', subsys=addthrust, promotes_outputs=['*'])
        self.connect('prop1.thrust', 'add_thrust.prop1_thrust')
        self.connect('prop2.thrust', 'add_thrust.prop2_thrust')

        addweights = AddSubtractComp(output_name='motors_weight', input_names=['motor1_weight', 'motor2_weight'],
                                     units='kg')
        addweights.add_equation(output_name='propellers_weight', input_names=['prop1_weight', 'prop2_weight'],
                                units='kg')
        addweights.add_equation(output_name='converters_weight',
                                input_names=['inverter1_weight', 'inverter2_weight'],
                                units='kg')
        addweights.add_equation(output_name='gearboxes_weight',
                                input_names=['gearbox1_weight', 'gearbox2_weight'],
                                units='kg')
        addweights.add_equation(output_name='engines_weight', input_names=['eng1_weight', 'eng2_weight'], units='kg')
        self.add_subsystem('add_weights', subsys=addweights, promotes_inputs=['*'], promotes_outputs=['*'])

        self.connect('motor1.component_weight', 'motor1_weight')
        self.connect('motor2.component_weight', 'motor2_weight')
        self.connect('prop1.component_weight', 'prop1_weight')
        self.connect('prop2.component_weight', 'prop2_weight')
        self.connect('inverter1.component_weight', 'inverter1_weight')
        self.connect('inverter2.component_weight', 'inverter2_weight')
        self.connect('gearbox1.component_weight', 'gearbox1_weight')
        self.connect('gearbox2.component_weight', 'gearbox2_weight')
        self.connect('eng1.component_weight', 'eng1_weight')
        self.connect('eng2.component_weight', 'eng2_weight')

        # connect design variables to model component inputs
        self.connect('eng_rating', ['eng1.shaft_power_rating', 'eng2.shaft_power_rating'])
        self.connect('prop_diameter', ['prop1.diameter', 'prop2.diameter'])
        self.connect('motor_rating', ['motor1.elec_power_rating', 'motor2.elec_power_rating'])
        self.connect('motor_rating', ['inverter1.elec_power_rating', 'inverter2.elec_power_rating'])
        self.connect('splitter_rating', ['hybrid_split_1.power_rating',
                                         'hybrid_split_2.power_rating'])
        self.connect('prop_rating', ['prop1.power_rating', 'prop2.power_rating'])
        self.connect('prop_rating', ['gearbox1.shaft_power_rating', 'gearbox2.shaft_power_rating'])
        self.connect('motor_max_speed', ['gearbox1.shaft_speed_in', 'gearbox2.shaft_speed_in'])  # used for sizing
        self.connect('prop_min_speed', ['gearbox1.shaft_speed_out', 'gearbox2.shaft_speed_out'])  # used for sizing
        self.connect('batt_weight', 'batt1.battery_weight')
        # self.connect('battery_SOC_inital', 'batt1.SOC_inital')

        propmodel_weight = self.add_subsystem('propmodel_weight', ExecComp([
            'propulsion_system_weight = total_engines_weight + total_propellers_weight + total_motors_weight + '
            'total_batteries_weight + total_converters_weight + total_gearboxes_weight'
        ],
            propulsion_system_weight={'value': 1.0, 'units': 'kg'},
            total_engines_weight={'value': 1.0, 'units': 'kg'},
            total_propellers_weight={'value': 1.0, 'units': 'kg'},
            total_motors_weight={'value': 1.0, 'units': 'kg'},
            total_batteries_weight={'value': 1.0, 'units': 'kg'},
            total_converters_weight={'value': 1.0, 'units': 'kg'},
            total_gearboxes_weight={'value': 1.0, 'units': 'kg'},
        ),
                                              promotes_inputs=['*'],
                                              promotes_outputs=['propulsion_system_weight'])
        self.connect('engines_weight', 'total_engines_weight')
        self.connect('motors_weight', 'total_motors_weight')
        self.connect('propellers_weight', 'total_propellers_weight')
        self.connect('batt_weight', 'total_batteries_weight')
        self.connect('converters_weight', 'total_converters_weight')
        self.connect('gearboxes_weight', 'total_gearboxes_weight')
