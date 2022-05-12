import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle as pkl
import numpy as np
import niceplots as nice
import os
import openmdao.api as om

# ================= PLOT SETTINGS =================
archs = ["series_hybrid", "parallel_hybrid"]  # results will be (arch[1] - arch[0]) / arch[0]

variable = ["mixed objective", "MTOW", "energy", "cruise DoH", "payload_frac", "batt_frac", "prop_sys_frac"]
nice_var_name = [
    "Parallel hybrid fuel burn + MTOW / 100 relative to series (%)",
    "Parallel hybrid MTOW relative to series (%)",
    "Parallel hybrid energy used relative to series (%)",
    "Parallel hybrid cruise hybridization relative to series (%)",
    "Parallel hybrid paylod weight / MTOW relative to series (%)",
    "Parallel hybrid battery weight / MTOW relative to series (%)",
    "Parallel hybrid propulsion system weight (incl. fuel and battery) / MTOW relative to series (%)",
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
plt.rcParams["font.size"] = 14

curDir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
filepath_save = os.path.join(curDir, "postprocess", "figures", "series_vs_parallel_grids")

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

for i_var in range(len(variable)):
    var = variable[i_var]
    var_nice = nice_var_name[i_var]
    var_file = file_var_name[i_var]
    data = np.zeros_like(range_grid)
    lim = cbar_lim[i_var]

    # Series hybrid
    for i in range(range_grid.shape[0]):
        for j in range(range_grid.shape[1]):
            for i_arch, arch in enumerate(archs):
                filepath = os.path.join(curDir, "data", arch)

                e_batt = e_batt_grid[i, j]
                mission_range = range_grid[i, j]

                filename = f"range{int(mission_range)}nmi_eBatt{int(e_batt)}"

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
                        W_batt = 0.
                    
                    # Get the propulsion system weight (including battery)
                    W_prop_sys = case.get_val("cruise.propulsion_system_weight", units="kg").item()
                    W_prop_sys += results["fuel burn"]

                    results["batt_frac"] = W_batt / results["MTOW"]
                    results["prop_sys_frac"] = W_prop_sys / results["MTOW"]

                print(results)

                if i_arch == 0:
                    data[i, j] = results[var]
                else:
                    data[i, j] = (results[var] - data[i, j]) / data[i, j] * 100.

    # Set the min and max of the current variable
    var_minmax[i_var] = (np.min(data), np.max(data))

    print("\n\n======================================================================================")
    print(f"      Done with {var_nice}")
    print("======================================================================================\n\n")

    lim = np.array([np.min(data), np.max(data)])
    lim[0] = min(-1e-10, lim[0])
    lim[1] = max(1e-10, lim[1])

    plt.figure(figsize=[6.4, 4.8])
    plt.pcolormesh(x, y, data, cmap="coolwarm", norm=mcolors.TwoSlopeNorm(vmin=lim[0], vcenter=0., vmax=lim[1]))
    plt.xlabel("Battery specific energy (Wh/kg)")
    plt.ylabel("Mission range (nmi)")
    plt.xlim((x_list[0], x_list[-1]))
    plt.ylim((y_list[0], y_list[-1]))
    cbar = plt.colorbar()
    plt.title(var_nice, fontsize="medium")

    plt.clim(lim)
    plt.savefig(os.path.join(filepath_save, f"{archs[0]}_vs_{archs[1]}_{var_file}.pdf"))

    plt.close("all")

# Print out the mins and maxes for each var
print(var_minmax)
for i, var in enumerate(nice_var_name):
    print(var)
    print(f"    Minumum: {var_minmax[i][0]}")
    print(f"    Maximum: {var_minmax[i][1]}")
