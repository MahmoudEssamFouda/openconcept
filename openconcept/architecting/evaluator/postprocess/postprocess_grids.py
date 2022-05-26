import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import niceplots as nice
import os
import openmdao.api as om
from openconcept.architecting.evaluator.postprocess.utils import get_snopt_exit

# ================= PLOT SETTINGS =================
archs = [
    "electric",
    "series_hybrid",
    "parallel_hybrid",
    "conventional",
    "turboelectric"
]
is_oneD = [
    False,
    False,
    False,
    True,
    True,
]  # True if only sweep over mission range and not battery spec energy (because no battery)

variable = ["mixed objective", "MTOW", "energy", "cruise DoH", "payload_frac", "batt_frac", "prop_sys_frac", "bfl", "motor_rating", "eng_rating"]
nice_var_name = [
    "Fuel burn + MTOW / 100 (kg)",
    "MTOW (kg)",
    "Energy used (kWh)",
    "Cruise hybridization (0 all fuel, 1 all electric)",
    "Paylod weight / MTOW",
    "Battery weight / MTOW",
    "Propulsion system weight (incl. fuel and battery) / MTOW",
    "Balanced field length (ft)",
    "Parallel hybrid motor rating (kW)",
    "Parallel hybrid engine rating (kW)",
]
file_var_name = ["obj", "mtow", "energy", "cruise_doh", "payload_frac", "batt_frac", "prop_sys_frac", "bfl", "motor_rating", "eng_rating"]
cbar_lim = [
    (39.07724584033572, 818.8),
    (3078.8021790291486, 5700.),
    (598.4980632723381, 9349.168),
    (0.0, 1.0),
    (0.0, 0.1474),
    (0.0, 0.3253),
    (0.2, 0.4527),
    (0., 4452.),
    (0., 400.),
    (0., 570.),
]  # colorbar limits for each variable (currently set as variables' min and max values across all architectures)

payload = 453.592  # kg, 1000 lbs

# ================= CREATE PLOTS =================
nice.setRCParams(dark_mode=False, set_dark_background=False)
plt.rcParams["font.size"] = 12

curDir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))

# Min and max value for each variable
var_minmax = [
    (np.inf, -np.inf),
] * len(variable)

mission_ranges = np.linspace(300, 800, 11)
spec_energies = np.linspace(300, 800, 11)
range_grid, e_batt_grid = np.meshgrid(mission_ranges, spec_energies, indexing="xy")

x_spacing = mission_ranges[1] - mission_ranges[0]
y_spacing = spec_energies[1] - spec_energies[0]
x_list = np.hstack((mission_ranges - x_spacing / 2, np.array([mission_ranges[-1] + x_spacing / 2])))
y_list = np.hstack((spec_energies - y_spacing / 2, np.array([spec_energies[-1] + y_spacing / 2])))
x, y = np.meshgrid(x_list, y_list, indexing="xy")

for i_arch, arch in enumerate(archs):
    filepath = os.path.join(curDir, "data", "mtow_bound", "grid", arch)
    filepath_save = os.path.join(curDir, "postprocess", "figures", "result_grids")
    oneD = is_oneD[i_arch]

    for i_var in range(len(variable)):
        var = variable[i_var]
        var_nice = nice_var_name[i_var]
        var_file = file_var_name[i_var]
        data = np.zeros_like(range_grid)
        clim = cbar_lim[i_var]

        if var in ["motor_rating", "eng_rating"] and arch != "parallel_hybrid":
            continue

        # Series hybrid
        for i in range(range_grid.shape[0]):
            for j in range(range_grid.shape[1]):
                e_batt = e_batt_grid[i, j]
                mission_range = range_grid[i, j]

                filename = f"range{int(mission_range)}nmi_eBatt{int(e_batt)}"
                if oneD:
                    filename = f"range{int(mission_range)}nmi"

                try:
                    with open(os.path.join(filepath, filename + ".pkl"), "rb") as f:
                        results = pkl.load(f)
                except FileNotFoundError:
                    data[i, j] = np.NaN
                    continue
                
                # Only plot if the SNOPT exit code is 0/1
                exit_code = get_snopt_exit(os.path.join(filepath, filename + "_SNOPT_print.out"))
                if exit_code != (0, 1):
                    data[i, j] = np.NaN
                    continue

                if var == "energy":
                    results[var] = results["fuel energy"] + results["battery energy"]
                elif var == "payload_frac":
                    results[var] = payload / results["MTOW"]
                elif var in ["batt_frac", "prop_sys_frac", "bfl", "motor_rating", "eng_rating"]:
                    try:
                        cr = om.CaseReader(os.path.join(filepath, filename + ".sql"))
                        case = cr.get_case("optimized")
                    except OSError:
                        data[i, j] = np.NaN
                        continue

                    # Get the battery weight
                    try:
                        W_batt = case.get_val("ac|weights|W_battery_pass", units="kg").item()
                    except KeyError:
                        W_batt = 0.0

                    # Get the propulsion system weight (including battery)
                    W_prop_sys = case.get_val("cruise.propulsion_system_weight", units="kg").item()
                    W_prop_sys += results["fuel burn"]

                    results["batt_frac"] = W_batt / results["MTOW"]
                    results["prop_sys_frac"] = W_prop_sys / results["MTOW"]
                    results["bfl"] = max(case.get_val("rotate.range_final", units="ft").item(), case.get_val("v1v0.range_final", units="ft").item())

                    if var in ["motor_rating", "eng_rating"]:
                        results["motor_rating"] = case.get_val("ac|propulsion|motor|rating", units="kW").item()
                        results["eng_rating"] = case.get_val("ac|propulsion|mech_engine|rating", units="kW").item()

                    # if var == "batt_frac":
                    #     # print(case.get_val("ac|propulsion|mech_engine|rating", units="kW").item())
                    #     # print(case.get_val("ac|propulsion|motor|rating", units="kW").item())
                    #     print(case.get_val("cruise_DoH").item())

                print(results)

                # Set the min and max of the current variable
                var_minmax[i_var] = (min(var_minmax[i_var][0], results[var]), max(var_minmax[i_var][1], results[var]))

                data[i, j] = results[var]

        plt.figure(figsize=[5.75, 4.8])
        plt.pcolormesh(x, y, data)
        plt.ylabel("Battery specific energy (Wh/kg)")
        plt.xlabel("Mission range (nmi)")
        plt.xlim((x_list[0], x_list[-1]))
        plt.ylim((y_list[0], y_list[-1]))
        cbar = plt.colorbar()
        # plt.title(var_nice, fontsize="medium")
        if clim is not None:
            plt.clim(clim)
        plt.savefig(os.path.join(filepath_save, f"{arch}_{var_file}.pdf"))

        plt.close("all")

# Print out the mins and maxes for each var
print(var_minmax)
for i, var in enumerate(nice_var_name):
    print(var)
    print(f"    Minumum: {var_minmax[i][0]}")
    print(f"    Maximum: {var_minmax[i][1]}")
