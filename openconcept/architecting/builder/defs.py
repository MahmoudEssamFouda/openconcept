from dataclasses import dataclass

__all__ = ['ArchElement', 'ArchSubSystem', 'WEIGHT_OUTPUT', 'DURATION_INPUT', 'FLTCOND_RHO_INPUT', 'FLTCOND_TAS_INPUT']

DURATION_INPUT = 'duration'
FLTCOND_RHO_INPUT = 'fltcond|rho'
FLTCOND_TAS_INPUT = 'fltcond|Utrue'

WEIGHT_OUTPUT = 'subsystem_weight'


@dataclass(frozen=False)
class ArchElement:
    """Base class for an architecture element."""

    name: str

    def __hash__(self):
        return id(self)


class ArchSubSystem:
    """Base class for a subdivision of the propsulsion system architecture."""
