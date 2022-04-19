import openmdao.api as om
import os
from pathlib import Path
import numpy as np
import pickle as pkl
from openconcept.architecting.evaluator.analysis_group import (
    add_recorder,
    opt_prob,
    set_problem_vars,
    DynamicKingAirAnalysisGroup,
)
from openconcept.architecting.builder.architecture import *

prop_arch = PropSysArch(  # Conventional with gearbox
    thrust=ThrustGenElements(
        propellers=[Propeller("prop1"), Propeller("prop2")],
        gearboxes=[Gearbox("gearbox1"), Gearbox("gearbox2")],
    ),
    mech=MechPowerElements(
        engines=[Engine("turboshaft", power_rating=560.0), Engine("turboshaft", power_rating=560.0)]
    ),
)

# obj = {"var": "descent.fuel_used_final"}
obj = {"var": "mixed_objective"}
DVs = [
    {"var": "ac|propulsion|propeller|diameter", "kwargs": {"lower": 2.2, "units": "m"}},
    {"var": "ac|propulsion|mech_engine|rating", "kwargs": {"lower": 200, "upper": 1e3, "ref": 5e2, "units": "kW"}},
]
cons = [
    {"var": "climb.throttle", "kwargs": {"lower": 0.0, "upper": 1.0}},
    {"var": "cruise.throttle", "kwargs": {"lower": 0.0, "upper": 1.0}},
    {"var": "descent.throttle", "kwargs": {"lower": 0.0, "upper": 1.0}},
]

curDir = os.path.abspath(os.path.dirname(__file__))
filepath = os.path.join(curDir, "data", "conventional")
Path(filepath).mkdir(parents=True, exist_ok=True)

mission_ranges = np.linspace(300, 800, 10)
spec_energies = np.linspace(300, 800, 10)
# Will have a dictionary containing:
#       "range" (nmi)
#       "battery specific energy" (Wh/kg, None)
#       "fuel burn" (kg)
#       "fuel energy" (kWh)
#       "battery energy" (kWh)
#       "MTOW" (kg)
#       "mixed objective" (kg, fuel burn + MTOW / 100)
results = []

for mission_range in mission_ranges:
    p = opt_prob(
        prop_arch=prop_arch,
        obj=obj,
        prop_sys_DVs=DVs,
        prop_sys_cons=cons,
        model=DynamicKingAirAnalysisGroup,
        hst_file=os.path.join(filepath, f"range{int(mission_range)}nmi.hst"),
    )
    add_recorder(p, filename=os.path.join(filepath, f"range{int(mission_range)}nmi.sql"))
    p.setup()
    set_problem_vars(p)
    p.set_val("mission_range", mission_range, units="nmi")
    p.run_driver()
    p.record("optimized")
    om.n2(p, show_browser=False, outfile=os.path.join(filepath, f"range{int(mission_range)}nmi_n2.html"))

    # Set values in the results vector
    results.append({})
    results[-1]["range"] = mission_range
    results[-1]["battery specific energy"] = None
    results[-1]["fuel burn"] = p.get_val("descent.fuel_used_final", units="kg").item()
    # Jet A specific energy is 11.95 kWh/kg
    results[-1]["fuel energy"] = 11.95 * p.get_val("descent.fuel_used_final", units="kg").item()
    results["battery energy"] = 0.0
    results["MTOW"] = p.get_val("ac|weights|MTOW", units="kg").item()
    results["mixed objective"] = p.get_val("mixed_objective", units="kg").item()

    with open(os.path.join(filepath, f"range{int(mission_range)}nmi.pkl"), "wb") as f:
        pkl.dump(results, f, protocol=pkl.HIGHEST_PROTOCOL)
    print(results)
