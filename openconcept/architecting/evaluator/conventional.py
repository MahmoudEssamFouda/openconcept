import openmdao.api as om
from openconcept.architecting.evaluator.analysis_group import opt_prob, set_problem_vars
from openconcept.architecting.builder.architecture import *

prop_arch = PropSysArch(  # Conventional with gearbox
    thrust=ThrustGenElements(
        propellers=[Propeller("prop1"), Propeller("prop2")],
        gearboxes=[Gearbox("gearbox1"), Gearbox("gearbox2")],
    ),
    mech=MechPowerElements(engines=Engine("turboshaft", power_rating=560.0)),
)

# obj = {"var": "descent.fuel_used_final"}
obj = {"var": "mixed_objective"}
DVs = [
    {"var": "ac|propulsion|propeller|diameter", "kwargs": {"lower": 2.2, "units": "m"}},
    {"var": "ac|propulsion|engine|rating", "kwargs": {"lower": 200, "upper": 1e3, "ref": 5e2, "units": "kW"}}
]
cons = [
    {"var": "climb.throttle", "kwargs": {"lower": 0.0, "upper": 1.0}},
    {"var": "cruise.throttle", "kwargs": {"lower": 0.0, "upper": 1.0}},
    {"var": "descent.throttle", "kwargs": {"lower": 0.0, "upper": 1.0}},
]

p = opt_prob(prop_arch=prop_arch, obj=obj, prop_sys_DVs=DVs, prop_sys_cons=cons)
p.setup()
set_problem_vars(p)
p.run_driver()