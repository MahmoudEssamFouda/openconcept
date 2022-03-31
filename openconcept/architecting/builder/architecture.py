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

from typing import *
from dataclasses import dataclass

from openconcept.architecting.builder.defs import *
from openconcept.architecting.builder.elements.thrust import *
from openconcept.architecting.builder.elements.mech import *
from openconcept.architecting.builder.elements.electric import *

__all__ = [
    'ArchElement', 'ArchSubSystem', 'WEIGHT_OUTPUT', 'PropSysArch', 'DURATION_INPUT', 'FLTCOND_RHO_INPUT',
    'FLTCOND_TAS_INPUT',

    'ThrustGenElements', 'Propeller', 'Gearbox',
    'SHAFT_POWER_INPUT', 'THRUST_OUTPUT', 'RPM_INPUT', 'SHAFT_SPEED_INPUT', 'RATED_POWER_INPUT',
    'MechPowerElements', 'Engine', 'Motor', 'Inverter', 'FUEL_FLOW_OUTPUT', 'ELECTRIC_POWER_OUTPUT', 'THROTTLE_INPUT',
    'ACTIVE_INPUT',
    'ElectricPowerElements', 'DCBus', 'Batteries', 'Engine', 'Generator', 'Rectifier', 'EngineChain', 'SOC_OUTPUT',
]
# any interface variables must be added here as well


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
