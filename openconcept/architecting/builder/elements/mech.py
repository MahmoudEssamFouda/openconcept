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
from typing import *
import openmdao.api as om
from dataclasses import dataclass
from openconcept.architecting.builder.defs import *
from openconcept.architecting.builder.utils import *
from openconcept.architecting.builder.elements.thrust import *

from openconcept.components import SimpleTurboshaft, SimpleMotor, SimpleConverterInverted, SimpleMechBus, PowerSplit
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp

__all__ = ['MechPowerElements', 'Engine', 'Motor', 'Inverter', 'FUEL_FLOW_OUTPUT', 'ELECTRIC_POWER_OUTPUT',
           'MechSplitter', 'MechBus', 'THROTTLE_INPUT', 'ACTIVE_INPUT']

THROTTLE_INPUT = 'throttle'
ACTIVE_INPUT = 'propulsor_active'

FUEL_FLOW_OUTPUT = 'fuel_flow'
ELECTRIC_POWER_OUTPUT = 'motors_elec_power'


@dataclass(frozen=False)
class Engine(ArchElement):
    """Conventional turboshaft engine."""

    power_rating: float = 260.  # kW

    specific_weight: float = .14 / 1000  # kg/kW
    base_weight: float = 104  # kg
    psfc: float = .6  # kg/W/s
    output_rpm: float = 6000  # rpm


@dataclass(frozen=False)
class Motor(ArchElement):
    """Electric motor."""

    power_rating: float = 260.  # kW
    efficiency: float = .97
    specific_weight: float = 1. / 5000  # kg/kW
    base_weight: float = 0.  # kg
    cost_inc: float = 100.0 / 745.0  # $ per watt
    cost_base: float = 1.  # $ per base
    output_rpm: float = 5500  # rpm


@dataclass(frozen=False)
class Inverter(ArchElement):
    """A DC to AC inverter."""
    efficiency: float = 0.97
    # power_rating: float = 260.  # kW, passed from electric motor
    specific_weight: float = 1. / (10 * 1000)  # kg/kW
    base_weight: float = 0.  # kg
    cost_inc: float = 100.0 / 745.0  # $ per watt
    cost_base: float = 1.  # $ per base


@dataclass(frozen=False)
class MechSplitter(ArchElement):
    """ mech power splitter to divide a power input to two outputs A and B based on a split fraction and
    efficiency loss"""

    power_rating: float = 99999999  # 'W', maximum power rating of split component
    efficiency: float = 1.0  # efficiency defines the loss of combining eng+motor shaft power
    split_rule: str = "fraction"  # this sets the rule to always use a fraction between 0 and 1
    mech_DoH: float = 0.5  # degree of hybridization between eng & motor for delivering shaft power, 0 =< mech_DoH =< 1


@dataclass(frozen=False)
class MechBus(ArchElement):
    """electric dc bus"""
    efficiency: float = 0.95  # efficiency loss to combine eng and motor shaft powers
    rpm_out: float = 5500  # output rpm of the mechanical bus to be connected to gearbox


@dataclass(frozen=False)
class MechPowerElements(ArchSubSystem):
    """Mechanical power generation elements in the propulsion system architecture. It is assumed that mechanical power
    is generated near the propellers, and therefore the local sub-architecture of the mechanical elements is replicated
    for each propeller."""

    # Either specify one element to be replicated for each propeller,
    # or specify a list of elements to distribute over the propellers
    engines: Optional[Union[Engine, List[Optional[Engine]]]] = None
    motors: Optional[Union[Motor, List[Optional[Motor]]]] = None
    inverters: Optional[Union[Inverter, List[Optional[Inverter]]]] = None
    mech_buses: Optional[Union[MechBus, List[Optional[MechBus]]]] = None
    mech_splitters: Optional[Union[MechSplitter, List[Optional[MechSplitter]]]] = None

    def create_mech_group(self, arch: om.Group, thrust_groups: List[om.Group], nn: int) -> Tuple[om.Group, bool]:
        """
        Creates the mechanical power group and returns whether electric power generation is needed.

        Inputs: THROTTLE_INPUT[nn], DURATION_INPUT, FLTCOND_RHO_INPUT[nn], FLTCOND_TAS_INPUT[nn], ACTIVE_INPUT
        Outputs: FUEL_FLOW_OUTPUT[nn], ELECTRIC_LOAD_OUTPUT[nn], WEIGHT_OUTPUT
        """

        # Prepare components
        n_thrust = len(thrust_groups)
        engines = self.engines
        motors = self.motors
        inverters = self.inverters
        mech_buses = self.mech_buses
        mech_splitters = self.mech_splitters

        if engines is None or isinstance(engines, Engine):
            engines = [engines for _ in range(n_thrust)]
        if motors is None or isinstance(motors, Motor):
            motors = [motors for _ in range(n_thrust)]
        if inverters is None or isinstance(inverters, Inverter):
            inverters = [inverters for _ in range(n_thrust)]
        if mech_buses is None or isinstance(mech_buses, MechBus):
            mech_buses = [mech_buses for _ in range(n_thrust)]
        if mech_splitters is None or isinstance(mech_splitters, MechSplitter):
            mech_splitters = [mech_splitters for _ in range(n_thrust)]

        # Create mechanical power group
        mech_group: om.Group = arch.add_subsystem('mech', om.Group())

        # Define inputs
        _, input_map = collect_inputs(mech_group, [
            (THROTTLE_INPUT, None, np.tile(1., nn)),
            (DURATION_INPUT, 's', 1.),
            (ACTIVE_INPUT, None, np.tile(1., nn)),
            (FLTCOND_RHO_INPUT, 'kg/m**3', np.tile(1.225, nn)),
            (FLTCOND_TAS_INPUT, 'm/s', np.tile(100., nn)),
        ], name="mech_in_collect")

        # Create and add components
        fuel_flow_outputs = []
        weight_outputs = []
        electric_load_outputs = []
        for i, thrust_group in enumerate(thrust_groups):
            engine, motor, inverter, mech_splitter, mech_bus, = \
                engines[i], motors[i], inverters[i], mech_splitters[i], mech_buses[i]

            if engine is None and motor is None:
                raise RuntimeError('Either engine or motor should be present for thrust group %d!' % (i + 1,))

            # Create group for mechanical power generation components for this specific thrust group
            mech_thrust_group: om.Group = mech_group.add_subsystem('mech%d' % (i + 1,), om.Group())
            shaft_power_out_param = None
            shaft_speed_out_param = None
            rated_power_out_param = None
            throttle_param = None

            if engine is not None and motor is not None:  # used usually for parallel hybrid
                if mech_bus is None or mech_splitter is None:
                    raise RuntimeError("engines and motors are added as inputs, therefore, "
                                       "mech_buses and mech_splitters must be provided as well")

                # define design params for eng
                _, eng_input_map = collect_inputs(mech_thrust_group, [
                    ('eng_rating', 'kW', engine.power_rating),
                    ('eng_output_rpm', 'rpm', engine.output_rpm),
                ], name="eng_in_collect")

                eng = mech_thrust_group.add_subsystem(
                    engine.name, SimpleTurboshaft(num_nodes=nn, psfc=engine.psfc * 1.68965774e-7,
                                                  weight_inc=engine.specific_weight, weight_base=engine.base_weight))

                mech_thrust_group.connect(eng_input_map['eng_rating'], eng.name + '.shaft_power_rating')

                fuel_flow_outputs += ['.'.join([mech_thrust_group.name, eng.name, 'fuel_flow'])]
                weight_outputs += ['.'.join([mech_thrust_group.name, eng.name, 'component_weight'])]

                # define design params for motor
                _, mot_input_map = collect_inputs(mech_thrust_group, [
                    ('motor_rating', 'kW', motor.power_rating),
                    ('motor_output_rpm', 'rpm', motor.output_rpm),
                ], name="motor_in_collect")

                # Add electric motor component
                mot = mech_thrust_group.add_subsystem(
                    motor.name, SimpleMotor(efficiency=motor.efficiency, num_nodes=nn, weight_inc=motor.specific_weight,
                                            weight_base=motor.base_weight, cost_inc=motor.cost_inc,
                                            cost_base=motor.cost_base))

                mech_thrust_group.connect(mot_input_map['motor_rating'], mot.name + '.elec_power_rating')

                weight_outputs += ['.'.join([mech_thrust_group.name, mot.name, 'component_weight'])]

                if inverter is None:  # override if inverter is added
                    electric_load_outputs += ['.'.join([mech_thrust_group.name, mot.name, 'elec_load'])]

                if i == 1:  # add OEI condition
                    # get propulsor active flag as scalar to use it for OEI
                    scalify_active_flag = ScalifyComponent(
                        vars=[
                            ('propulsor_active_vector', ACTIVE_INPUT + '_scalar', nn, None),
                        ])
                    mech_thrust_group.add_subsystem('scalify_active_input', subsys=scalify_active_flag)
                    mech_group.connect(input_map[ACTIVE_INPUT],
                                       mech_thrust_group.name + '.scalify_active_input' + '.propulsor_active_vector')

                    # add rated powers
                    sum_rated_power = om.ExecComp([
                        'tot_rated_power = engine_rated_power + active_flag * motor_rated_power',
                    ],
                        tot_rated_power={'val': 1, 'units': 'kW'},
                        engine_rated_power={'val': 1, 'units': 'kW'},
                        active_flag={'val': 1},
                        motor_rated_power={'val': 1.0, 'units': 'kW'},
                    )
                    mech_thrust_group.add_subsystem('sum_rated_power', subsys=sum_rated_power)
                    mech_thrust_group.connect(eng_input_map['eng_rating'],
                                              'sum_rated_power' + '.engine_rated_power')
                    mech_thrust_group.connect(mot_input_map['motor_rating'],
                                              'sum_rated_power' + '.motor_rated_power')
                    mech_thrust_group.connect('scalify_active_input' + '.' + ACTIVE_INPUT + '_scalar',
                                              'sum_rated_power' + '.active_flag')
                else:  # all engines active condition
                    sum_rated_power = om.ExecComp([
                        'tot_rated_power = engine_rated_power + motor_rated_power',
                    ],
                        tot_rated_power={'val': 1, 'units': 'kW'},
                        engine_rated_power={'val': 1, 'units': 'kW'},
                        motor_rated_power={'val': 1.0, 'units': 'kW'},
                    )
                    mech_thrust_group.add_subsystem('sum_rated_power', subsys=sum_rated_power)
                    mech_thrust_group.connect(eng_input_map['eng_rating'],
                                              'sum_rated_power' + '.engine_rated_power')
                    mech_thrust_group.connect(mot_input_map['motor_rating'],
                                              'sum_rated_power' + '.motor_rated_power')

                # add rated powers of eng and motor for sizing
                sizing_rated_power = AddSubtractComp()
                sizing_rated_power.add_equation(
                    output_name='tot_rated_power',
                    input_names=[eng.name + '_rated_power', mot.name + '_rated_power'],
                    units='kW')
                mech_thrust_group.add_subsystem('sizing_rated_power', subsys=sizing_rated_power)
                mech_thrust_group.connect(eng_input_map['eng_rating'],
                                          'sizing_rated_power' + '.' + eng.name + '_rated_power')
                mech_thrust_group.connect(mot_input_map['motor_rating'],
                                          'sizing_rated_power' + '.' + mot.name + '_rated_power')

                # get total shaft power output of (engine + motor) system
                get_shaft_power = om.ExecComp([
                    'tot_shaft_power = throttle_vec * total_rated_power',
                ],
                    tot_shaft_power={'val': np.ones(nn), 'units': 'kW'},
                    throttle_vec={'val': np.ones(nn)},
                    total_rated_power={'val': 1.0, 'units': 'kW'},
                )
                mech_thrust_group.add_subsystem('eng_motor_shaft_power', subsys=get_shaft_power)
                mech_thrust_group.connect('sum_rated_power' + '.tot_rated_power',
                                          'eng_motor_shaft_power' + '.total_rated_power')

                # define throttle parameter
                throttle_param = '.'.join([mech_thrust_group.name, 'eng_motor_shaft_power', 'throttle_vec'])

                # add MechBus
                bus = mech_thrust_group.add_subsystem(
                    mech_bus.name, SimpleMechBus(num_nodes=nn, efficiency=mech_bus.efficiency,
                                                 rpm_out=mech_bus.rpm_out))
                mech_thrust_group.connect('eng_motor_shaft_power' + '.tot_shaft_power',
                                          bus.name + '.shaft_power_in')

                # add splitter
                # Define design params for splitter
                _, splitter_input_map = collect_inputs(mech_thrust_group, [
                    ('mech_DoH', None, np.ones(nn) * mech_splitter.mech_DoH),
                ], name='splitter_in_collect')

                if i == 1:
                    # add get split fraction component for OEI
                    get_split_fraction = om.ExecComp([
                        'split_fraction_vec = active_flag_vec * mech_DoH_vec',
                    ],
                        split_fraction_vec={'val': np.zeros(nn)},
                        active_flag_vec={'val': np.zeros(nn)},
                        mech_DoH_vec={'val': np.ones(nn)},
                    )
                    mech_thrust_group.add_subsystem('get_split_fraction', subsys=get_split_fraction)
                    mech_thrust_group.connect(splitter_input_map['mech_DoH'],
                                              'get_split_fraction' + '.mech_DoH_vec')
                    mech_group.connect(input_map[ACTIVE_INPUT],
                                       mech_thrust_group.name + '.get_split_fraction' + '.active_flag_vec')

                split = mech_thrust_group.add_subsystem(
                    mech_splitter.name, PowerSplit(num_nodes=nn, efficiency=mech_splitter.efficiency,
                                                   rule=mech_splitter.split_rule))
                # add OEI condition
                if i == 1:  # use get split fraction to connect to split fraction
                    mech_thrust_group.connect('get_split_fraction' + '.split_fraction_vec',
                                              split.name + '.power_split_fraction')
                else:  # use input map to connect to split fraction
                    mech_thrust_group.connect(splitter_input_map['mech_DoH'], split.name + '.power_split_fraction')

                # connect shaft power input to splitter
                mech_thrust_group.connect('eng_motor_shaft_power' + '.tot_shaft_power',
                                          split.name + '.power_in')
                # track weight
                weight_outputs += ['.'.join([mech_thrust_group.name, split.name, 'component_weight'])]

                # define required power outputs from motor and engine
                motor_req_power_out = '.'.join([split.name, 'power_out_A'])
                eng_req_power_out = '.'.join([split.name, 'power_out_B'])

                # define available power for engine and motor
                motor_avail_power_out = '.'.join([mot.name, 'shaft_power_out'])
                eng_avail_power_out = '.'.join([eng.name, 'shaft_power_out'])

                # find engine throttle
                throttle_from_power_balance(group=mech_thrust_group, power_req=eng_req_power_out,
                                            power_avail=eng_avail_power_out, units='kW',
                                            comp_name=eng.name, n=nn)

                # find motor throttle
                throttle_from_power_balance(group=mech_thrust_group, power_req=motor_req_power_out,
                                            power_avail=motor_avail_power_out, units='kW',
                                            comp_name=mot.name, n=nn)

                # define output parameters
                shaft_power_out_param = '.'.join([mech_group.name, mech_thrust_group.name,
                                                  bus.name, 'shaft_power_out'])
                shaft_speed_out_param = '.'.join([mech_group.name, mech_thrust_group.name,
                                                  bus.name, 'output_rpm'])
                rated_power_out_param = '.'.join([mech_group.name, mech_thrust_group.name,
                                                  'sizing_rated_power', 'tot_rated_power'])

            # Add turboshaft engine
            if engine is not None and motor is None:  # used usually for conventional architectures
                # Define design params
                _, eng_input_map = collect_inputs(mech_thrust_group, [
                    ('rating', 'kW', engine.power_rating),
                    ('output_rpm', 'rpm', engine.output_rpm),
                ], name="eng_in_collect")

                # add one engine inoperative case for prop systems with two or more engines
                if i == 1 and motor is None:  # if no of engines >=2, if yes, add a failed engine component to mech2
                    failedengine = ElementMultiplyDivideComp()
                    failedengine.add_equation('eng2throttle',
                                              input_names=['throttle_vec', 'propulsor_active_flag'], vec_size=nn)
                    failedengine = mech_thrust_group.add_subsystem('failedengine', failedengine)
                    mech_group.connect(input_map[ACTIVE_INPUT],
                                       mech_thrust_group.name + '.failedengine' + '.propulsor_active_flag')

                # Add engine component
                eng = mech_thrust_group.add_subsystem(
                    engine.name, SimpleTurboshaft(num_nodes=nn, psfc=engine.psfc * 1.68965774e-7,
                                                  weight_inc=engine.specific_weight, weight_base=engine.base_weight))

                mech_thrust_group.connect(eng_input_map['rating'], eng.name + '.shaft_power_rating')

                fuel_flow_outputs += ['.'.join([mech_thrust_group.name, eng.name, 'fuel_flow'])]
                weight_outputs += ['.'.join([mech_thrust_group.name, eng.name, 'component_weight'])]

                # define out_params
                shaft_power_out_param = '.'.join([mech_group.name, mech_thrust_group.name, eng.name, 'shaft_power_out'])
                shaft_speed_out_param = '.'.join([mech_group.name, mech_thrust_group.name, eng_input_map['output_rpm']])
                rated_power_out_param = '.'.join([mech_group.name, mech_thrust_group.name, eng_input_map['rating']])

                # define throttle parameter in case of one engine inoperative OEI or Normal
                if i == 1:  # in the case of OEI, for mech2, connect throttle to failedengine
                    throttle_param = '.'.join([mech_thrust_group.name, 'failedengine', 'throttle_vec'])
                    mech_thrust_group.connect('failedengine' + '.eng2throttle', eng.name + '.throttle')
                else:  # Normal conditions
                    throttle_param = '.'.join([mech_thrust_group.name, eng.name, 'throttle'])

            # Add electric motor
            if motor is not None and engine is None:
                # Defined design params
                _, mot_input_map = collect_inputs(mech_thrust_group, [
                    ('rating', 'kW', motor.power_rating),
                    ('output_rpm', 'rpm', motor.output_rpm),
                ], name="motor_in_collect")

                # add one motor inoperative case for prop systems with two or more motors
                if i == 1:  # check if no of motors >=2, if yes, add a failed motor component to mech2 group
                    failedmotor = ElementMultiplyDivideComp()
                    failedmotor.add_equation('motor2throttle',
                                             input_names=['throttle_vec', 'propulsor_active_flag'], vec_size=nn)
                    failedmotor = mech_thrust_group.add_subsystem('failedmotor', failedmotor)
                    mech_group.connect(input_map[ACTIVE_INPUT],
                                       mech_thrust_group.name + '.failedmotor' + '.propulsor_active_flag')

                # Add electric motor component
                mot = mech_thrust_group.add_subsystem(
                    motor.name, SimpleMotor(efficiency=motor.efficiency, num_nodes=nn, weight_inc=motor.specific_weight,
                                            weight_base=motor.base_weight, cost_inc=motor.cost_inc,
                                            cost_base=motor.cost_base))

                mech_thrust_group.connect(mot_input_map['rating'], mot.name + '.elec_power_rating')

                weight_outputs += ['.'.join([mech_thrust_group.name, mot.name, 'component_weight'])]

                if inverter is None:  # override if inverter is added
                    electric_load_outputs += ['.'.join([mech_thrust_group.name, mot.name, 'elec_load'])]

                # define out_params
                shaft_power_out_param = '.'.join([mech_group.name, mech_thrust_group.name, mot.name, 'shaft_power_out'])
                shaft_speed_out_param = '.'.join([mech_group.name, mech_thrust_group.name, mot_input_map['output_rpm']])
                rated_power_out_param = '.'.join([mech_group.name, mech_thrust_group.name, mot_input_map['rating']])

                # define throttle parameter in case of one motor inoperative OEI or Normal
                if i == 1:  # in the case of OEI, for mech2, connect throttle to failedmotor
                    throttle_param = '.'.join([mech_thrust_group.name, 'failedmotor', 'throttle_vec'])
                    mech_thrust_group.connect('failedmotor' + '.motor2throttle', mot.name + '.throttle')
                else:  # Normal conditions
                    throttle_param = '.'.join([mech_thrust_group.name, mot.name, 'throttle'])

            if inverter is not None:
                if motor is None:
                    raise RuntimeError('Inverter is added but no Motor!')
                else:
                    # inverter does not need in_collect, it has no inputs, only options
                    invert = mech_thrust_group.add_subsystem(
                        inverter.name, SimpleConverterInverted(
                            num_nodes=nn, efficiency=inverter.efficiency, weight_inc=inverter.specific_weight,
                            weight_base=inverter.base_weight, cost_inc=inverter.cost_inc, cost_base=inverter.cost_base))

                    weight_outputs += ['.'.join([mech_thrust_group.name, invert.name, 'component_weight'])]
                    electric_load_outputs += ['.'.join([mech_thrust_group.name, invert.name, 'elec_power_in'])]

                    if engine is None:
                        mech_thrust_group.connect(mot_input_map['rating'], invert.name + '.elec_power_rating')
                        mech_thrust_group.connect(mot.name + '.elec_load', invert.name + '.elec_power_out')
                    else:
                        mech_thrust_group.connect(mot_input_map['motor_rating'], invert.name + '.elec_power_rating')
                        mech_thrust_group.connect(mot.name + '.elec_load', invert.name + '.elec_power_out')

            # Connect throttle input
            mech_group.connect(input_map[THROTTLE_INPUT], throttle_param)

            # Connect output shaft power to thrust generation group
            if shaft_power_out_param is None:
                raise RuntimeError('No shaft power generated for thrust group %d!' % (i + 1,))
            arch.connect(shaft_power_out_param, thrust_group.name + '.' + SHAFT_POWER_INPUT)
            arch.connect(shaft_speed_out_param, thrust_group.name + '.' + SHAFT_SPEED_INPUT)
            arch.connect(rated_power_out_param, thrust_group.name + '.' + RATED_POWER_INPUT)

        # Calculate output sums
        create_output_sum(mech_group, FUEL_FLOW_OUTPUT, fuel_flow_outputs, 'kg/s', n=nn)
        create_output_sum(mech_group, WEIGHT_OUTPUT, weight_outputs, 'kg')
        create_output_sum(mech_group, ELECTRIC_POWER_OUTPUT, electric_load_outputs, 'kW', n=nn)

        # Determine whether electric power generation is needed
        electric_power_needed = len(electric_load_outputs) > 0

        return mech_group, electric_power_needed
