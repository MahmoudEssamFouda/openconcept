import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle as pkl
import numpy as np
import niceplots as nice
import os
import openmdao.api as om
from openconcept.architecting.evaluator.postprocess.utils import get_snopt_exit

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
    "Parallel hybrid propulsion system weight\nincluding fuel and battery / MTOW relative to series (%)",
]
file_var_name = ["obj", "mtow", "energy", "cruise_doh", "payload_frac", "batt_frac", "prop_sys_frac"]
clim = [
    None,
    None,
    None,
    None,
    None,
    None,
    None,
]  # colorbar limits for each variable

payload = 453.592  # kg, 1000 lbs

# ================= CREATE PLOTS =================
nice.setRCParams(dark_mode=False, set_dark_background=False)
plt.rcParams["font.size"] = 12

curDir = os.path.dirname(os.path.abspath(os.path.dirname(__file__)))
filepath_save = os.path.join(curDir, "postprocess", "figures", "series_vs_parallel_grids")

# Min and max value for each variable
var_minmax = [
    (np.inf, -np.inf),
] * len(variable)

mission_ranges = np.linspace(300, 800, 11)
spec_energies = np.linspace(300, 800, 11)
range_grid, e_batt_grid = np.meshgrid(mission_ranges, spec_energies, indexing="xy")

x_spacing = mission_ranges[1] - mission_ranges[0]
y_spacing = spec_energies[1] - spec_energies[0]
y_list = np.hstack((spec_energies - y_spacing / 2, np.array([spec_energies[-1] + y_spacing / 2])))
x_list = np.hstack((mission_ranges - x_spacing / 2, np.array([mission_ranges[-1] + x_spacing / 2])))
x, y = np.meshgrid(x_list, y_list, indexing="xy")

for i_var in range(len(variable)):
    var = variable[i_var]
    var_nice = nice_var_name[i_var]
    var_file = file_var_name[i_var]
    data = np.zeros_like(range_grid)

    # Series hybrid
    for i in range(range_grid.shape[0]):
        for j in range(range_grid.shape[1]):
            for i_arch, arch in enumerate(archs):
                filepath = os.path.join(curDir, "data", "mtow_bound", "grid", arch)

                e_batt = e_batt_grid[i, j]
                mission_range = range_grid[i, j]

                filename = f"range{int(mission_range)}nmi_eBatt{int(e_batt)}"

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

                if i_arch == 0:
                    data[i, j] = results[var]
                else:
                    data[i, j] = (results[var] - data[i, j]) / data[i, j] * 100.0

    # Set the min and max of the current variable
    var_minmax[i_var] = (np.nanmin(data), np.nanmax(data))

    print("\n\n======================================================================================")
    print(f"      Done with {var_nice}")
    print("======================================================================================\n\n")

    lim = np.array([np.nanmin(data), np.nanmax(data)])

    # Take the biggest distance from zero to compute vmin and vmax
    colormap_lim = max(abs(np.nanmin(data)), abs(np.nanmax(data)))
    if clim[i_var]:
        vmin = clim[i_var][0]
        vmax = clim[i_var][1]
        lim[0] = max(lim[0], vmin)
        lim[1] = min(lim[1], vmax)
    else:
        vmin = -colormap_lim
        vmax = colormap_lim

    plt.figure(figsize=[5.75, 4.8])
    plt.pcolormesh(x, y, data, cmap="coolwarm", vmin=vmin, vmax=vmax)
    plt.xlabel("Mission range (nmi)")
    plt.ylabel("Battery specific energy (Wh/kg)")
    plt.xlim((x_list[0], x_list[-1]))
    plt.ylim((y_list[0], y_list[-1]))
    cbar = plt.colorbar()
    # plt.title(var_nice, fontsize="medium")

    cbar.ax.set_ylim(lim)
    plt.savefig(os.path.join(filepath_save, f"{archs[0]}_vs_{archs[1]}_{var_file}.pdf"))

    plt.close("all")

# Print out the mins and maxes for each var
print(var_minmax)
for i, var in enumerate(nice_var_name):
    print(var)
    print(f"    Minumum: {var_minmax[i][0]}")
    print(f"    Maximum: {var_minmax[i][1]}")
