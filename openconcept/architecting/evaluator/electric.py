import openmdao.api as om
import os
from pathlib import Path
from copy import deepcopy
import pickle as pkl
import numpy as np
from openconcept.architecting.evaluator.analysis_group import (
    add_recorder,
    opt_prob,
    set_problem_vars,
    DynamicKingAirAnalysisGroup,
)
from openconcept.architecting.builder.architecture import *

# TODO: add DVs/constraints to properly size other electrical propulsion system components
#       (for example, the inverter) so they can handle the electrical power
# obj = {"var": "descent.fuel_used_final"}
obj = {"var": "mixed_objective"}
DVs = [
    {"var": "ac|propulsion|propeller|diameter", "kwargs": {"lower": 2.2, "units": "m"}},
    {"var": "ac|propulsion|motor|rating", "kwargs": {"lower": 200, "ref": 5e2, "units": "kW"}},
    {"var": "ac|weights|W_battery", "kwargs": {"lower": 50, "ref": 1e3, "units": "kg"}}
]
# Constraints to be enforced at every flight segment; full variable name will be
# <mission segment>.<var name>
seg_cons = [
    {"var": "throttle", "kwargs": {"lower": 0.0, "upper": 1.0}},
    {"var": "propmodel.elec.bat_pack.SOC", "kwargs": {"lower": 0.0}},
]

# Generate constraints necessary for every mission segment
segments = ["v0v1", "v1vr", "rotate", "v1v0", "engineoutclimb", "climb", "cruise", "descent"]
cons = []
for seg_con in seg_cons:
    for seg in segments:
        cons.append(deepcopy(seg_con))
        cons[-1]["var"] = ".".join((seg, cons[-1]["var"]))

curDir = os.path.abspath(os.path.dirname(__file__))
filepath = os.path.join(curDir, "data", "electric")
Path(filepath).mkdir(parents=True, exist_ok=True)

mission_ranges = np.linspace(300, 800, 2)
spec_energies = np.linspace(300, 800, 2)
# Will have a dictionary containing:
#       "range" (nmi)
#       "fuel burn" (kg)
#       "fuel energy" (kWh)
#       "battery energy" (kWh)
#       "MTOW" (kg)
#       "mixed objective" (kg, fuel burn + MTOW / 100)
results = []

for mission_range in mission_ranges:
    for e_batt in spec_energies:
        prop_arch = PropSysArch(  # All-electric with inverters
            thrust=ThrustGenElements(
                propellers=[Propeller("prop1"), Propeller("prop2")],
                gearboxes=[Gearbox("gearbox1"), Gearbox("gearbox2")],
            ),
            mech=MechPowerElements(motors=[Motor("elec_motor", power_rating=600), Motor("elec_motor", power_rating=600)], inverters=Inverter("inverter")),
            electric=ElectricPowerElements(
                dc_bus=DCBus("elec_bus"), batteries=Batteries("bat_pack", weight=1e3, specific_energy=e_batt)
            ),
        )

        p = opt_prob(
            prop_arch=prop_arch,
            obj=obj,
            prop_sys_DVs=DVs,
            prop_sys_cons=cons,
            model=DynamicKingAirAnalysisGroup,
            hst_file=os.path.join(filepath, f"range{int(mission_range)}nmi_eBatt{int(e_batt)}.hst"),
        )
        add_recorder(p, filename=os.path.join(filepath, f"range{int(mission_range)}nmi_eBatt{int(e_batt)}.sql"))
        p.setup()
        set_problem_vars(p)
        p.set_val("mission_range", mission_range, units="nmi")
        p.run_driver()
        p.record("optimized")
        om.n2(p, show_browser=False, outfile=os.path.join(filepath, f"range{int(mission_range)}nmi_eBatt{int(e_batt)}_n2.html"))

        # Set values in the results vector
        results.append({})
        results[-1]["range"] = mission_range
        results[-1]["battery specific energy"] = e_batt
        results[-1]["fuel burn"] = 0.0
        results[-1]["fuel energy"] = 0.0
        results[-1]["battery energy"] = 1. - p.get_val("descent.propmodel.elec.bat_pack.SOC_final").item()
        results[-1]["battery energy"] *= e_batt * p.get_val("ac|weights|W_battery", units="kg")
        results[-1]["MTOW"] = p.get_val("ac|weights|MTOW", units="kg").item()
        results[-1]["mixed objective"] = p.get_val("mixed_objective", units="kg").item()

with open(os.path.join(filepath, "results_electric.pkl"), "wb") as f:
    pkl.dump(results, f, protocol=pkl.HIGHEST_PROTOCOL)
print(results)
