import numpy as np
from typing import *
import openmdao.api as om
from dataclasses import dataclass
from openconcept.architecting.builder.defs import *
from openconcept.architecting.builder.utils import *
from openconcept.architecting.builder.elements.mech import *

from openconcept.components.battery import SOCBattery

__all__ = ['ElectricPowerElements', 'Batteries', 'Engine', 'Generator', 'Rectifier', 'EngineChain', 'FUEL_FLOW_OUTPUT',
           'SOC_OUTPUT']

SOC_OUTPUT = 'SOC'


@dataclass(frozen=False)
class Batteries(ArchElement):
    """Battery pack."""

    weight: float = 2000.  # kg
    specific_power: float = 5000  # W/kg
    specific_energy: float = 300  # Wh/kg
    efficiency: float = .97


@dataclass(frozen=False)
class Generator(ArchElement):
    """An AC electric power generator."""

    power_rating: float = 250.  # kW


@dataclass(frozen=False)
class Rectifier(ArchElement):
    """An AC to DC rectifier."""


EngineChain = Tuple[Engine, Generator, Rectifier]


@dataclass(frozen=False)
class ElectricPowerElements(ArchSubSystem):
    """Electrical power generation elements in the propulsion system architecture."""

    batteries: Batteries = None
    engines: List[EngineChain] = None

    def create_electric_group(self, arch: om.Group, mech_power_group: om.Group, nn: int) -> om.Group:
        """
        Creates the electrical power generation group.

        Inputs: DURATION_INPUT, FLTCOND_RHO_INPUT[nn], FLTCOND_TAS_INPUT[nn]
        Outputs: FUEL_FLOW_OUTPUT[nn], WEIGHT_OUTPUT, SOC_OUTPUT[nn]
        """

        # Prepare components
        batteries = self.batteries
        engine_chains = self.engines

        # Create electrical power group
        elec_group: om.Group = arch.add_subsystem('elec', om.Group())

        # Define inputs
        _, input_map = collect_inputs(elec_group, [
            (DURATION_INPUT, 's', 1.),
            (FLTCOND_RHO_INPUT, 'kg/m**3', np.tile(1.225, nn)),
            (FLTCOND_TAS_INPUT, 'm/s', np.tile(100., nn)),
        ])

        elec_load_input = None
        fuel_flow_outputs = []
        weight_outputs = []
        soc_outputs = []

        # Create and add batteries
        if batteries is not None:
            # Define design params
            _, bat_input_map = collect_inputs(elec_group, [
                ('weight', 'kg', batteries.weight),
            ], name='bat_in_collect')

            # Add battery component
            bat = elec_group.add_subsystem(
                batteries.name, SOCBattery(
                    num_nodes=nn, efficiency=batteries.efficiency, specific_power=batteries.specific_power,
                    specific_energy=batteries.specific_power))

            weight_outputs += [bat_input_map['weight']]
            soc_outputs += [bat.name+'.SOC']

            elec_load_input = bat.name+'.elec_load'

            elec_group.connect(bat_input_map['weight'], bat.name+'.battery_weight')
            elec_group.connect(input_map[DURATION_INPUT], bat.name+'.duration')

        # Add conventional engine chains
        if engine_chains is not None:
            raise NotImplementedError('Series hybrid not implemented yet!')

        # Connect electric load
        if elec_load_input is None:
            raise RuntimeError('Cannot connect electric load!')
        elec_load_output_param = mech_power_group.name+'.'+ELECTRIC_POWER_OUTPUT
        arch.connect(elec_load_output_param, elec_group.name+'.'+elec_load_input)

        # Calculate output sums
        create_output_sum(elec_group, FUEL_FLOW_OUTPUT, fuel_flow_outputs, 'kg/s', n=nn)
        create_output_sum(elec_group, WEIGHT_OUTPUT, weight_outputs, 'kg')
        create_output_sum(elec_group, SOC_OUTPUT, soc_outputs, n=nn)

        return elec_group
