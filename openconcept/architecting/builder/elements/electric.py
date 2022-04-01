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
from openconcept.architecting.builder.elements.mech import *

from openconcept.components.battery import SOCBattery
from openconcept.components import SimpleTurboshaft, SimpleGenerator, SimpleDCBusInverted, SimpleConverter, PowerSplit

__all__ = ['ElectricPowerElements', 'DCBus', 'Batteries', 'Engine', 'Generator', 'Rectifier', 'DCEngineChain',
           'ACEngineChain', 'Splitter', 'FUEL_FLOW_OUTPUT', 'SOC_OUTPUT']

SOC_OUTPUT = 'SOC'


@dataclass(frozen=False)
class DCBus(ArchElement):
    """electric dc bus"""
    efficiency: float = 0.99  #


@dataclass(frozen=False)
class Splitter(ArchElement):
    """power splitter to divide a power input to two outputs A and B based on a split fraction and efficiency loss"""
    efficiency: float = 1.0  # always keep as 1, apply required efficiency in DC Bus component
    split_rule: str = "fraction"  # this sets the rule to always use a fraction between 0 and 1
    split_fraction: float = 0.5  # degree of hybridization between Battery pack and EngineChains, 0 =< DoH =< 1


@dataclass(frozen=False)
class Batteries(ArchElement):
    """Battery pack."""

    weight: float = 1000.  # kg
    specific_power: float = 5000  # W/kg
    specific_energy: float = 300  # W/kg
    efficiency: float = .97
    cost_inc: float = 50.  # $ per kg
    cost_base: float = 1.  # $ per base


@dataclass(frozen=False)
class Generator(ArchElement):
    """An AC electric power generator."""

    power_rating: float = 250.  # kW


@dataclass(frozen=False)
class Rectifier(ArchElement):
    """An AC to DC rectifier."""


DCEngineChain = Tuple[Engine, Generator, Rectifier]
ACEngineChain = Tuple[Engine, Generator]


@dataclass(frozen=False)
class ElectricPowerElements(ArchSubSystem):
    """Electrical power generation elements in the propulsion system architecture."""

    dc_bus: Optional[DCBus] = None
    splitter: Optional[Splitter] = None
    batteries: Union[Batteries, List[Batteries]] = None
    engines_dc: Union[DCEngineChain, List[DCEngineChain]] = None
    engines_ac: Union[ACEngineChain, List[ACEngineChain]] = None

    def create_electric_group(self, arch: om.Group, mech_power_group: om.Group, nn: int) -> om.Group:
        """
        Creates the electrical power generation group.

        Inputs: DURATION_INPUT, FLTCOND_RHO_INPUT[nn], FLTCOND_TAS_INPUT[nn]
        Outputs: FUEL_FLOW_OUTPUT[nn], WEIGHT_OUTPUT, SOC_OUTPUT[nn]
        """

        # Prepare components
        dc_bus = self.dc_bus
        splitter = self.splitter
        batteries = self.batteries
        engine_chains_dc = self.engines_dc
        engine_chains_ac = self.engines_ac

        # Create electrical power group
        elec_group: om.Group = arch.add_subsystem('elec', om.Group())

        # Define inputs
        _, input_map = collect_inputs(elec_group, [
            (DURATION_INPUT, 's', 1.),
            (FLTCOND_RHO_INPUT, 'kg/m**3', np.tile(1.225, nn)),
            (FLTCOND_TAS_INPUT, 'm/s', np.tile(100., nn)),
        ], name="elec_in_collect")

        elec_load_input = None
        fuel_flow_outputs = []
        weight_outputs = []
        soc_outputs = []

        if dc_bus is not None:
            bus = elec_group.add_subsystem(dc_bus.name, SimpleDCBusInverted(num_nodes=nn, efficiency=dc_bus.efficiency))
            elec_load_input = dc_bus.name+'.elec_power_out'  # connect elec load to dc bus

        if splitter is not None:
            if dc_bus is None:
                raise RuntimeError("Currently, dc_bus is required with a splitter")
            else:
                if engine_chains_dc is None or batteries is None:
                    raise RuntimeError("engines and batteries are needed with splitter")
                else:
                    raise NotImplementedError('Splitter not implemented yet!')

        # check batteries input
        if type(batteries) == list:
            if len(batteries) == 1:
                raise RuntimeError("for one Battery pack, "
                                   "use batteries=Batteries('bat_pack') instead of [Batteries('bat_pack')]")
            elif len(batteries) > 1:
                raise NotImplementedError('multiple battery pack design is not implemented yet')
            else:
                raise RuntimeError("Battery pack list cannot be empty")

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
                    specific_energy=batteries.specific_energy, cost_inc=batteries.cost_inc,
                    cost_base=batteries.cost_base))

            weight_outputs += [bat_input_map['weight']]
            soc_outputs += [bat.name+'.SOC']

            if dc_bus is None:
                elec_load_input = bat.name+'.elec_load'  # directly connect elec load to battery
            else:
                if splitter is None and engine_chains_dc is None:
                    elec_group.connect(dc_bus.name+'.elec_power_in', bat.name+'.elec_load')

            elec_group.connect(bat_input_map['weight'], bat.name+'.battery_weight')
            elec_group.connect(input_map[DURATION_INPUT], bat.name+'.duration')

        # Add conventional engine chains
        if engine_chains_dc is not None:
            raise NotImplementedError('Series hybrid with DC Architecture not implemented yet!')

        if engine_chains_ac is not None:
            raise NotImplementedError('AC architectures not implemented yet!')

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
