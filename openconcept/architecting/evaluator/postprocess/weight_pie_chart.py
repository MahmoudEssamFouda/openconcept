import matplotlib.pyplot as plt
import niceplots as nice
import os
import pickle as pkl
import openmdao.api as om
import numpy as np

# ================= PLOT SETTINGS =================
categories = ["Propulsion", "Structure", "Fuel system", "Equipment", "Fluids", "Payload"]
subcategories = [
    ["Fuel", "Battery", "Propulsion system"],
    ["Wing", "Fuselage", "Nacelle", "Empennage", "Gear"],
    ["Fuel system"],
    ["Equipment"],
    ["Fluids"],
    ["Payload"],
]

vars = [
    [{"name": "fuel burn", "pkl": True}, {"name": "ac|weights|W_battery_pass", "pkl": False}, {"name": "Other propulsion"}],
    [{"name": "climb.OEW_calc.W_wing", "pkl": False}, {"name": "climb.OEW_calc.W_fuselage", "pkl": False}, {"name": "climb.OEW_calc.W_nacelle", "pkl": False}, {"name": "climb.OEW_calc.W_empennage", "pkl": False}, {"name": "climb.OEW_calc.W_gear", "pkl": False}],
    [{"name": "climb.OEW_calc.W_fuelsystem", "pkl": False}],
    [{"name": "climb.OEW_calc.W_equipment", "pkl": False}],
    [{"name": "climb.OEW_calc.W_fluids", "pkl": False}],
    [{"name": "payload", "pkl": False}],
]

cases = [{"range": 466, "e_batt": 522}, {"range": 300, "e_batt": 522}]

archs = ["electric", "series_hybrid", "parallel_hybrid", "conventional", "turboelectric"]
is_oneD = [
    False,
    False,
    False,
    True,
    True,
]  # True if only sweep over mission range and not battery spec energy (because no battery)

payload = 453.592  # kg, 1000 lbs

# ================= CREATE PLOTS =================
nice.setRCParams(dark_mode=False, set_dark_background=False)
plt.rcParams["font.size"] = 12

curDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for i_arch, arch in enumerate(archs):
    oneD = is_oneD[i_arch]
    filepath = os.path.join(curDir, "data", arch)
    filepath_save = os.path.join(curDir, "postprocess", "figures", "result_grids")

    for c in cases:
        mission_range = c["range"]
        e_batt = c["e_batt"]

        filename = f"range{mission_range}nmi_eBatt{e_batt}"
        if oneD:
            filename = f"range{mission_range}nmi"

        try:
            with open(os.path.join(filepath, filename + ".pkl"), "rb") as f:
                results = pkl.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No data is available for {arch} with a range of {mission_range} nmi and specific energy of {e_batt} Wh/kg"
            )

        try:
            cr = om.CaseReader(os.path.join(filepath, filename + ".sql"))
            case = cr.get_case("optimized")
        except OSError:
            raise OSError(
                f"No data is available for {arch} with a range of {mission_range} nmi and specific energy of {e_batt} Wh/kg"
            )

        # Collect values
        vals = []
        MTOW = 0.
        for i_cat, cat in enumerate(categories):
            vals.append([])

            # Use the structural fudge multiplier when necessary
            multiplier = case.get_val("cruise.OEW_calc.structural_fudge").item() if cat == "Structure" else 1.0

            for i_subcat, subcat in enumerate(subcategories[i_cat]):
                var = vars[i_cat][i_subcat]

                try:
                    W_batt = case.get_val("ac|weights|W_battery_pass", units="kg").item()
                except KeyError:
                    W_batt = 0.0

                if subcat == "Battery":
                    vals[-1].append(W_batt)
                elif subcat == "Propulsion system":
                    W_prop_sys = case.get_val("cruise.propulsion_system_weight", units="kg").item()
                    vals[-1].append(W_prop_sys - W_batt)
                elif var["pkl"]:
                    vals[-1].append(results[var["name"]])
                else:
                    vals[-1].append(case.get_val(var["name"], units="kg").item())

                vals[-1][-1] *= multiplier

                MTOW += vals[-1][-1]
