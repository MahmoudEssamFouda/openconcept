import numpy as np
from typing import *
import openmdao.api as om
from dataclasses import dataclass
from openconcept.architecting.builder.defs import *
from openconcept.architecting.builder.utils import *
from openconcept.architecting.builder.elements.thrust import *

from openconcept.components import SimpleTurboshaft, SimpleMotor

__all__ = ['MechPowerElements', 'Engine', 'Motor', 'Inverter', 'FUEL_FLOW_OUTPUT', 'ELECTRIC_POWER_OUTPUT',
           'THROTTLE_INPUT', 'ACTIVE_INPUT']

THROTTLE_INPUT = 'throttle'
ACTIVE_INPUT = 'propulsor_active'

FUEL_FLOW_OUTPUT = 'fuel_flow'
ELECTRIC_POWER_OUTPUT = 'motors_elec_power'


@dataclass(frozen=False)
class Engine(ArchElement):
    """Conventional turboshaft engine."""

    power_rating: float = 260.  # kW

    specific_weight: float = .14  # kg/kW
    base_weight: float = 104  # kg
    psfc: float = .6  # kg/W/s


@dataclass(frozen=False)
class Motor(ArchElement):
    """Electric motor."""

    power_rating: float = 260.  # kW
    efficiency: float = .97


@dataclass(frozen=False)
class Inverter(ArchElement):
    """A DC to AC inverter."""


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

        if engines is None or isinstance(engines, Engine):
            engines = [engines for _ in range(n_thrust)]
        if motors is None or isinstance(motors, Motor):
            motors = [motors for _ in range(n_thrust)]
        if inverters is None or isinstance(inverters, Inverter):
            inverters = [inverters for _ in range(n_thrust)]

        # Create mechanical power group
        mech_group: om.Group = arch.add_subsystem('mech', om.Group())

        # Define inputs
        _, input_map = collect_inputs(mech_group, [
            (THROTTLE_INPUT, None, np.tile(1., nn)),
            (DURATION_INPUT, 's', 1.),
            (ACTIVE_INPUT, None, np.tile(1., nn)),
            (FLTCOND_RHO_INPUT, 'kg/m**3', np.tile(1.225, nn)),
            (FLTCOND_TAS_INPUT, 'm/s', np.tile(100., nn)),
        ])

        # Create and add components
        fuel_flow_outputs = []
        weight_outputs = []
        electric_load_outputs = []
        for i, thrust_group in enumerate(thrust_groups):
            engine, motor, inverter = engines[i], motors[i], inverters[i]
            if engine is None and motor is None:
                raise RuntimeError('Either engine or motor should be present for thrust group %d!' % (i+1,))

            if engine is not None and motor is not None:  # Temporary!!
                raise NotImplementedError('Hybrid is not implemented yet!')
            if inverter is not None:
                raise NotImplementedError('Inverters not supported yet!')

            # Create group for mechanical power generation components for this specific thrust group
            mech_thrust_group: om.Group = mech_group.add_subsystem('mech%d' % (i+1,), om.Group())
            shaft_power_out_param = None
            throttle_param = None

            # Add turboshaft engine
            if engine is not None:
                # Define design params
                _, eng_input_map = collect_inputs(mech_thrust_group, [
                    ('rating', 'kW', engine.power_rating),
                ])

                # Add engine component
                eng = mech_thrust_group.add_subsystem(
                    engine.name, SimpleTurboshaft(num_nodes=nn, psfc=engine.psfc*1.68965774e-7,
                                                  weight_inc=engine.specific_weight, weight_base=engine.base_weight))

                fuel_flow_outputs += ['.'.join([mech_thrust_group.name, eng.name, 'fuel_flow'])]
                weight_outputs += ['.'.join([mech_thrust_group.name, eng.name, 'component_weight'])]

                shaft_power_out_param = '.'.join([mech_group.name, mech_thrust_group.name, eng.name, 'shaft_power_out'])
                throttle_param = '.'.join([mech_thrust_group.name, eng.name, 'throttle'])

                mech_thrust_group.connect(eng_input_map['rating'], eng.name+'.shaft_power_rating')

            # Add electric motor
            if motor is not None:
                # Defined design params
                _, mot_input_map = collect_inputs(mech_thrust_group, [
                    ('rating', 'kW', motor.power_rating),
                ])

                # Add electric motor component
                mot = mech_thrust_group.add_subsystem(
                    motor.name, SimpleMotor(efficiency=motor.efficiency, num_nodes=nn))

                weight_outputs += ['.'.join([mech_thrust_group.name, mot.name, 'component_weight'])]
                electric_load_outputs += ['.'.join([mech_thrust_group.name, mot.name, 'elec_load'])]

                shaft_power_out_param = '.'.join([mech_group.name, mech_thrust_group.name, mot.name, 'shaft_power_out'])
                throttle_param = '.'.join([mech_thrust_group.name, mot.name, 'throttle'])

                mech_thrust_group.connect(mot_input_map['rating'], mot.name+'.elec_power_rating')

            # Connect throttle input
            mech_group.connect(input_map[THROTTLE_INPUT], throttle_param)

            # Connect output shaft power to thrust generation group
            if shaft_power_out_param is None:
                raise RuntimeError('No shaft power generated for thrust group %d!' % (i+1,))
            arch.connect(shaft_power_out_param, thrust_group.name+'.'+SHAFT_POWER_INPUT)

        # Calculate output sums
        create_output_sum(mech_group, FUEL_FLOW_OUTPUT, fuel_flow_outputs, 'kg/s', n=nn)
        create_output_sum(mech_group, WEIGHT_OUTPUT, weight_outputs, 'kg')
        create_output_sum(mech_group, ELECTRIC_POWER_OUTPUT, electric_load_outputs, 'kW', n=nn)

        # Determine whether electric power generation is needed
        electric_power_needed = len(electric_load_outputs) > 0

        return mech_group, electric_power_needed
