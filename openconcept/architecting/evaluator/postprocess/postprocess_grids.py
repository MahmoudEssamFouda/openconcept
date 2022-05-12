import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import niceplots as nice
import os
import openmdao.api as om

# ================= PLOT SETTINGS =================
archs = ["electric", "series_hybrid", "parallel_hybrid", "conventional", "turboelectric"]
is_oneD = [
    False,
    False,
    False,
    True,
    True,
]  # True if only sweep over mission range and not battery spec energy (because no battery)

variable = ["mixed objective", "MTOW", "energy", "cruise DoH", "payload_frac", "batt_frac", "prop_sys_frac"]
nice_var_name = [
    "Fuel burn + MTOW / 100 (kg)",
    "MTOW (kg)",
    "Energy used (kWh)",
    "Cruise hybridization (0 all fuel, 1 all electric)",
    "Paylod weight / MTOW",
    "Battery weight / MTOW",
    "Propulsion system weight (incl. fuel and battery) / MTOW",
]
file_var_name = ["obj", "mtow", "energy", "cruise_doh", "payload_frac", "batt_frac", "prop_sys_frac"]
cbar_lim = [
    (39.07724584033572, 908.759127400349),
    (3078.8021790291486, 12439.5875678016),
    (598.4980632723381, 10375.395528122435),
    (0.0, 1.0),
    (0.0, 0.15),
    (0.0, 0.5),
    (0.2, 0.6),
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

mission_ranges = np.linspace(300, 800, 10)
spec_energies = np.linspace(300, 800, 10)
e_batt_grid, range_grid = np.meshgrid(spec_energies, mission_ranges, indexing="xy")

x_spacing = spec_energies[1] - spec_energies[0]
y_spacing = mission_ranges[1] - mission_ranges[0]
x_list = np.hstack((spec_energies - x_spacing / 2, np.array([spec_energies[-1] + x_spacing / 2])))
y_list = np.hstack((mission_ranges - y_spacing / 2, np.array([mission_ranges[-1] + y_spacing / 2])))
x, y = np.meshgrid(x_list, y_list, indexing="xy")

for i_arch, arch in enumerate(archs):
    filepath = os.path.join(curDir, "data", arch)
    filepath_save = os.path.join(curDir, "postprocess", "figures", "result_grids")
    oneD = is_oneD[i_arch]

    for i_var in range(len(variable)):
        var = variable[i_var]
        var_nice = nice_var_name[i_var]
        var_file = file_var_name[i_var]
        data = np.zeros_like(range_grid)
        clim = cbar_lim[i_var]

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

                if var == "energy":
                    results[var] = results["fuel energy"] + results["battery energy"]
                elif var == "payload_frac":
                    results[var] = payload / results["MTOW"]
                elif var == "batt_frac" or var == "prop_sys_frac":
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

                print(results)

                # Set the min and max of the current variable
                var_minmax[i_var] = (min(var_minmax[i_var][0], results[var]), max(var_minmax[i_var][1], results[var]))

                data[i, j] = results[var]

        plt.figure(figsize=[5.75, 5.])
        plt.pcolormesh(x, y, data)
        plt.xlabel("Battery specific energy (Wh/kg)")
        plt.ylabel("Mission range (nmi)")
        plt.xlim((x_list[0], x_list[-1]))
        plt.ylim((y_list[0], y_list[-1]))
        cbar = plt.colorbar()
        plt.title(var_nice, fontsize="medium")
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
