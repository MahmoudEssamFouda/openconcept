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
# obj = {"var": "energy_used"}
obj = {"var": "mixed_objective"}
DVs = [
    {"var": "ac|propulsion|propeller|diameter", "kwargs": {"lower": 2.2, "units": "m"}},
    {"var": "ac|propulsion|elec_engine|rating", "kwargs": {"lower": 200, "ref": 5e2, "units": "kW"}},
    {"var": "ac|propulsion|motor|rating", "kwargs": {"lower": 300, "ref": 5e2, "units": "kW"}},
    {"var": "ac|weights|W_battery", "kwargs": {"lower": 0.1, "ref": 1e3, "units": "kg"}},
    {"var": "cruise.propmodel.elec.elec_DoH", "kwargs": {"lower": 0.01, "upper": 0.99, "ref": 0.5}},
]
# Constraints to be enforced at every flight segment; full variable name will be
# <mission segment>.<var name>
seg_cons = [
    {"var": "throttle", "kwargs": {"lower": 0.0, "upper": 1.0}},
    {"var": "propmodel.elec.turboshaft.component_sizing_margin", "kwargs": {"upper": 1.0}},
    {"var": "propmodel.mech.mech1.elec_motor.component_sizing_margin", "kwargs": {"upper": 1.0}},
    {"var": "propmodel.mech.mech2.elec_motor.component_sizing_margin", "kwargs": {"upper": 1.0}},
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
filepath = os.path.join(curDir, "data", "mtow_bound", "grid", "series_hybrid")
Path(filepath).mkdir(parents=True, exist_ok=True)

mission_ranges = np.linspace(300, 800, 11)
spec_energies = np.linspace(300, 800, 11)

# mission_ranges = [550.]  # nmi
# spec_energies = [300., 350., 400., 450., 500., 550., 600., 650., 700., 750., 800.]  # Wh/kg
# spec_energies = [550., 800.]  # Wh/kg

for mission_range in mission_ranges:
    for e_batt in spec_energies:
        prop_arch = PropSysArch(  # series hybrid with one engine, battery and two motors
            thrust=ThrustGenElements(propellers=[Propeller('prop1'), Propeller('prop2')],
                                    gearboxes=[Gearbox('gearbox1'), Gearbox('gearbox2')]),
            mech=MechPowerElements(motors=[Motor('elec_motor', power_rating=2000), Motor('elec_motor', power_rating=2000)],
                                inverters=Inverter('inverter')),
            electric=ElectricPowerElements(dc_bus=DCBus('elec_bus'),
                                        splitter=ElecSplitter('splitter', elec_DoH=0.25),  # 0.4 hybridization to match parallel
                                        batteries=Batteries('bat_pack', weight=4e3, specific_energy=e_batt),
                                        engines_dc=(Engine(name='turboshaft', power_rating=2e3), Generator(name='generator'),
                                                    Rectifier(name='rectifier'))),
        )

        p = opt_prob(
            prop_arch=prop_arch,
            obj=obj,
            prop_sys_DVs=DVs,
            prop_sys_cons=cons,
            model=DynamicKingAirAnalysisGroup,
            hst_file=os.path.join(filepath, f"range{int(mission_range)}nmi_eBatt{int(e_batt)}.hst"),
        )
        p.model.set_input_defaults("ac|weights|W_battery", val=4e3, units="kg")
        add_recorder(p, filename=os.path.join(filepath, f"range{int(mission_range)}nmi_eBatt{int(e_batt)}.sql"))
        p.setup()
        set_problem_vars(p, e_batt=e_batt)
        p.set_val("mission_range", mission_range, units="nmi")
        p.run_driver()
        p.record("optimized")
        om.n2(p, show_browser=False, outfile=os.path.join(filepath, f"range{int(mission_range)}nmi_eBatt{int(e_batt)}_n2.html"))

        # Set values in the results vector
        # Will have a dictionary containing:
        #       "range" (nmi)
        #       "battery specific energy" (Wh/kg)
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
        results["fuel burn"] = p.get_val("descent.fuel_used_final", units="kg").item()
        # Jet A specific energy is 11.95 kWh/kg
        results["fuel energy"] = 11.95 * p.get_val("descent.fuel_used_final", units="kg").item()
        results["battery energy"] = 1. - p.get_val("descent.propmodel.elec.bat_pack.SOC_final").item()
        results["battery energy"] *= e_batt / 1e3 * p.get_val("ac|weights|W_battery", units="kg").item()
        results["MTOW"] = p.get_val("ac|weights|MTOW", units="kg").item()
        results["mixed objective"] = p.get_val("mixed_objective", units="kg").item()
        results["cruise DoH"] = p.get_val("cruise.propmodel.elec.elec_DoH").item()
        results["S_ref"] = p.get_val("ac|geom|wing|S_ref", units="m**2").item()

        with open(os.path.join(filepath, f"range{int(mission_range)}nmi_eBatt{int(e_batt)}.pkl"), "wb") as f:
            pkl.dump(results, f, protocol=pkl.HIGHEST_PROTOCOL)
        print(results)
