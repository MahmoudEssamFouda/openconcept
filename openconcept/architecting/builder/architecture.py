from typing import *
from dataclasses import dataclass

from openconcept.architecting.builder.defs import *
from openconcept.architecting.builder.elements.thrust import *
from openconcept.architecting.builder.elements.mech import *
from openconcept.architecting.builder.elements.electric import *

__all__ = [
    'ArchElement', 'ArchSubSystem', 'WEIGHT_OUTPUT', 'PropSysArch', 'DURATION_INPUT', 'FLTCOND_RHO_INPUT',
    'FLTCOND_TAS_INPUT',

    'ThrustGenElements', 'Propeller', 'Gearbox', 'SHAFT_POWER_INPUT', 'THRUST_OUTPUT', 'RPM_INPUT',
    'MechPowerElements', 'Engine', 'Motor', 'Inverter', 'FUEL_FLOW_OUTPUT', 'ELECTRIC_POWER_OUTPUT', 'THROTTLE_INPUT',
    'ACTIVE_INPUT',
    'ElectricPowerElements', 'Batteries', 'Engine', 'Generator', 'Rectifier', 'EngineChain', 'SOC_OUTPUT',
]


@dataclass(frozen=False)
class PropSysArch:
    """
    Describes an instance of a propulsion system architecture.

    In general the propulsion system architectures analyzed with this software package are built-up as follows:
    - A thrust generation sub-system:
      - Propeller
      - Gearbox
    - A mechanical power generation sub-system (connected to thrust generation sub-system)
      - Motor (electric) + inverter
      - OR: engine (conventional turboshaft)
      - OR:
        - Power splitter
        - Motor + inverter
        - Engine
    - An electric power generation sub-system (connected to the electric motors)
      - Batteries
      - OR: Engine + generator + rectifier
      - OR:
        - Hybrid splitter
        - Batteries
        - Engine + generator + rectifier

    Note that the nr of props, engines (for electricity generation) and batteries are all dynamic and independent.
    """

    thrust: ThrustGenElements
    mech: MechPowerElements
    electric: Optional[ElectricPowerElements] = None

    def __eq__(self, other):
        return hash(self) == hash(other)

    def __hash__(self):
        return id(self)
