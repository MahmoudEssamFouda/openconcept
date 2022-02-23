import numpy as np
from typing import *
import openmdao.api as om
from dataclasses import dataclass
from openconcept.architecting.builder.defs import *
from openconcept.architecting.builder.utils import *

from openconcept.components import SimplePropeller
from openconcept.components.gearbox import SimpleGearbox

__all__ = ['ThrustGenElements', 'Propeller', 'Gearbox', 'SHAFT_POWER_INPUT', 'THRUST_OUTPUT', 'RPM_INPUT']

RPM_INPUT = 'prop|rpm'
SHAFT_POWER_INPUT = 'shaft_power'

THRUST_OUTPUT = 'thrust'


@dataclass(frozen=False)
class Propeller(ArchElement):
    """Represents a propeller for thrust generation."""

    blades: int = 4
    diameter: float = 2.5  # m
    power_rating: float = 240  # kW

    design_cp: float = .2  # Cruise power coefficient
    design_adv_ratio: float = 2.2  # V/n/D (advance ratio)

    default_rpm: float = 2000.  # Default RPM if not overriden by design variables


@dataclass(frozen=False)
class Gearbox(ArchElement):
    """Mechanical reduction gearbox for the propeller. Output RPM is set by the propeller RPM."""

    input_rpm: float = 5500  # rpm


@dataclass(frozen=False)
class ThrustGenElements(ArchSubSystem):
    """Thrust generation elements in the propulsion system architecture."""

    propellers: List[Propeller]
    gearboxes: Optional[List[Optional[Gearbox]]] = None

    def create_thrust_groups(self, arch: om.Group, nn: int) -> List[om.Group]:
        """
        Creates thrust generation groups for each propeller.

        Inputs: RPM_INPUT[nn], DURATION_INPUT, FLTCOND_RHO_INPUT[nn], FLTCOND_TAS_INPUT[nn], SHAFT_POWER_INPUT[nn]
        Outputs: THRUST_OUTPUT[nn], WEIGHT_OUTPUT
        """

        # Get propeller and gearbox definitions
        props = self.propellers
        gearboxes = self.gearboxes
        if gearboxes is not None:
            if len(gearboxes) != len(props):
                raise RuntimeError('Nr of props (%d) not the same as gearboxes (%d)' % (len(props), len(gearboxes)))
        else:
            gearboxes = [None for _ in range(len(props))]

        # Create thrust groups
        groups = []
        for i, prop in enumerate(props):
            thrust_group: om.Group = arch.add_subsystem('thrust%d' % (i+1,), om.Group())
            groups.append(thrust_group)

            # Define inputs
            _, input_map = collect_inputs(thrust_group, [
                (RPM_INPUT, 'rpm', np.tile(prop.default_rpm, nn)),
                (DURATION_INPUT, 's', 1.),
                (SHAFT_POWER_INPUT, 'kW', np.tile(1., nn)),

                ('diameter', 'm', prop.diameter),
                ('power', 'kW', prop.power_rating),
            ])

            # Create propeller
            prop_sys = thrust_group.add_subsystem(
                prop.name, SimplePropeller(num_nodes=nn, num_blades=prop.blades, design_J=prop.design_adv_ratio,
                                           design_cp=prop.design_cp),
                promotes_inputs=[FLTCOND_RHO_INPUT, FLTCOND_TAS_INPUT], promotes_outputs=[THRUST_OUTPUT])
            shaft_power_input_param = prop_shaft_power_in = prop_sys.name+'.shaft_power_in'
            weights = [prop_sys.name+'.component_weight']

            thrust_group.connect(input_map[RPM_INPUT], prop_sys.name+'.rpm')
            thrust_group.connect(input_map['diameter'], prop_sys.name+'.diameter')
            thrust_group.connect(input_map['power'], prop_sys.name+'.power_rating')

            # Create optional gearbox
            gearbox = gearboxes[i]
            if gearbox is not None:
                gear_sys = thrust_group.add_subsystem(gearbox.name, SimpleGearbox(num_nodes=nn))
                shaft_power_input_param = gear_sys.name+'.shaft_power_in'
                weights += [gear_sys.name+'.component_weight']

                # Connect gearbox to propeller
                thrust_group.connect(gear_sys.name+'.shaft_power_out', prop_shaft_power_in)

            # Define shaft power input parameter
            thrust_group.connect(input_map[SHAFT_POWER_INPUT], shaft_power_input_param)

            # Sum weight outputs
            create_output_sum(thrust_group, WEIGHT_OUTPUT, weights, 'kg')

        return groups
