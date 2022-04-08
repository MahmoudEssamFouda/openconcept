import openmdao.api as om
from openconcept.architecting.evaluator.analysis_group import opt_prob, set_problem_vars, DynamicKingAirAnalysisGroup
from openconcept.architecting.builder.architecture import *

prop_arch = PropSysArch(  # Conventional with gearbox
    thrust=ThrustGenElements(
        propellers=[Propeller("prop1"), Propeller("prop2")],
        gearboxes=[Gearbox("gearbox1"), Gearbox("gearbox2")],
    ),
    mech=MechPowerElements(engines=Engine("turboshaft", power_rating=560.0)),
)

# Temporary class to handle DV promotion while the dynamic prop system promotion gets sorted out
class Analysis(DynamicKingAirAnalysisGroup):
    def setup(self):
        super().setup()

        # Promote propulsion system variables that are used in the optimization problem as DVs
        segments = ["v0v1", "v1vr", "rotate", "v1v0", "engineoutclimb", "climb", "cruise", "descent"]
        var_promote = [
            (["propmodel.mech.mech1.eng_rating", "propmodel.mech.mech2.eng_rating"], "ac|propulsion|engine|rating"),
            (["propmodel.thrust1.diameter", "propmodel.thrust2.diameter"], "ac|propulsion|propeller|diameter"),
        ]
        promotes_list = []
        for segment in segments:
            for var in var_promote:
                for abs_var in var[0]:
                    promotes_list.append((f"{segment}.{abs_var}", var[1]))
        self.promotes("mission", inputs=promotes_list)

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

p = opt_prob(prop_arch=prop_arch, obj=obj, prop_sys_DVs=DVs, prop_sys_cons=cons, model=Analysis)
p.setup()
set_problem_vars(p)
p.run_model()
om.n2(p, show_browser=False)
# p.run_driver()