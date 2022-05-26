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

# obj = {"var": "descent.fuel_used_final"}
obj = {"var": "mixed_objective"}
DVs = [
    {"var": "ac|propulsion|propeller|diameter", "kwargs": {"lower": 2.2, "units": "m"}},
    {"var": "ac|propulsion|motor|rating", "kwargs": {"lower": 500, "ref": 5e2, "units": "kW"}},
    {"var": "ac|weights|W_battery", "kwargs": {"lower": 50, "ref": 1e3, "units": "kg"}}
]
# Constraints to be enforced at every flight segment; full variable name will be
# <mission segment>.<var name>
seg_cons = [
    {"var": "throttle", "kwargs": {"lower": 0.0, "upper": 1.0}},
    {"var": "propmodel.elec.bat_pack.component_sizing_margin", "kwargs": {"upper": 1.0}},
]
cons = [  # constraints that are enforce at a single variable
    {"var": "descent.propmodel.elec.bat_pack.SOC_final", "kwargs": {"lower": 0.0}},
]

# Generate constraints necessary for every mission segment
segments = ["v0v1", "v1vr", "rotate", "v1v0", "engineoutclimb", "climb", "cruise", "descent"]
for seg_con in seg_cons:
    for seg in segments:
        cons.append(deepcopy(seg_con))
        cons[-1]["var"] = ".".join((seg, cons[-1]["var"]))

curDir = os.path.abspath(os.path.dirname(__file__))
filepath = os.path.join(curDir, "data", "mtow_bound", "grid", "electric")
Path(filepath).mkdir(parents=True, exist_ok=True)

mission_ranges = np.linspace(300, 800, 11)
spec_energies = np.linspace(300, 800, 11)

# mission_ranges = [300.]  # nmi (skip 500 case because it can't be feasible with MTOW bound)
# spec_energies = [500]  # Wh/kg

for mission_range in mission_ranges:
    for e_batt in spec_energies:
        prop_arch = PropSysArch(  # All-electric with inverters
            thrust=ThrustGenElements(
                propellers=[Propeller("prop1"), Propeller("prop2")],
                gearboxes=[Gearbox("gearbox1"), Gearbox("gearbox2")],
            ),
            mech=MechPowerElements(motors=[Motor("elec_motor", power_rating=600), Motor("elec_motor", power_rating=600)], inverters=Inverter("inverter")),
            electric=ElectricPowerElements(
                dc_bus=DCBus("elec_bus"), batteries=Batteries("bat_pack", weight=3e3, specific_energy=e_batt)
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
        p.model.set_input_defaults("ac|weights|W_battery", val=3e3, units="kg")
        add_recorder(p, filename=os.path.join(filepath, f"range{int(mission_range)}nmi_eBatt{int(e_batt)}.sql"))
        p.setup()
        set_problem_vars(p, e_batt=e_batt)
        # p.set_val("ac|weights|MTOW", val=5750., units="kg")
        p.set_val("mission_range", mission_range, units="nmi")
        # p.run_model()
        p.run_driver()
        p.record("optimized")
        om.n2(p, show_browser=False, outfile=os.path.join(filepath, f"range{int(mission_range)}nmi_eBatt{int(e_batt)}_n2.html"))

        # Set values in the results vector
        # Will have a dictionary containing:
        #       "range" (nmi)
        #       "fuel burn" (kg)
        #       "fuel energy" (kWh)
        #       "battery energy" (kWh)
        #       "MTOW" (kg)
        #       "cruise DoH"
        #       "S_ref" (m^2)
        #       "mixed objective" (kg, fuel burn + MTOW / 100)
        results = {}
        results["range"] = mission_range
        results["battery specific energy"] = e_batt
        results["fuel burn"] = 0.0
        results["fuel energy"] = 0.0
        results["battery energy"] = 1. - p.get_val("descent.propmodel.elec.bat_pack.SOC_final").item()
        results["battery energy"] *= e_batt / 1e3 * p.get_val("ac|weights|W_battery", units="kg").item()
        results["MTOW"] = p.get_val("ac|weights|MTOW", units="kg").item()
        results["mixed objective"] = p.get_val("mixed_objective", units="kg").item()
        results["cruise DoH"] = 1.
        results["S_ref"] = p.get_val("ac|geom|wing|S_ref", units="m**2").item()

        with open(os.path.join(filepath, f"range{int(mission_range)}nmi_eBatt{int(e_batt)}.pkl"), "wb") as f:
            pkl.dump(results, f, protocol=pkl.HIGHEST_PROTOCOL)
        print(results)
