import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import os

curDir = os.path.abspath(os.path.dirname(__file__))
arch = "parallel_hybrid"
filepath = os.path.join(curDir, "data", arch)

variable = ["mixed objective", "MTOW", "energy", "cruise DoH"]
nice_var_name = ["fuel burn + MTOW / 100 (kg)", "MTOW (kg)", "Energy used (kWh)", "Cruise hybridization (0 all battery, 1 all fuel)"]
file_var_name = ["obj", "mtow", "energy", "cruise_doh"]

mission_ranges = np.linspace(300, 800, 10)
spec_energies = np.linspace(300, 800, 10)
e_batt_grid, range_grid = np.meshgrid(spec_energies, mission_ranges, indexing='xy')

x_spacing = spec_energies[1] - spec_energies[0]
y_spacing = mission_ranges[1] - mission_ranges[0]
x_list = np.hstack((spec_energies - x_spacing / 2, np.array([spec_energies[-1] + x_spacing / 2])))
y_list = np.hstack((mission_ranges - y_spacing / 2, np.array([mission_ranges[-1] + y_spacing / 2])))
x, y = np.meshgrid(x_list, y_list, indexing='xy')

for i_var in range(len(variable)):
    var = variable[i_var]
    var_nice = nice_var_name[i_var]
    var_file = file_var_name[i_var]
    data = np.zeros_like(range_grid)

    # Series hybrid
    for i in range(range_grid.shape[0]):
        for j in range(range_grid.shape[1]):
            e_batt = e_batt_grid[i, j]
            mission_range = range_grid[i, j]

            try:
                with open(os.path.join(filepath, f"range{int(mission_range)}nmi_eBatt{int(e_batt)}.pkl"), "rb") as f:
                    results = pkl.load(f)
            except FileNotFoundError:
                results = {"range": mission_range, "battery specific energy": e_batt}
                results[var] = np.NaN
                if var == "energy":
                    results["fuel energy"] = np.NaN
                    results["battery energy"] = np.NaN
            
            if var == "energy":
                results[var] = results["fuel energy"] + results["battery energy"]

            print(results)

            data[i, j] = results[var]

    plt.pcolormesh(x, y, data)
    plt.xlabel("Battery specific energy (Wh/kg)")
    plt.ylabel("Mission range (nmi)")
    plt.colorbar()
    plt.title(var_nice)
    plt.savefig(os.path.join(filepath, f"{arch}_{var_file}.pdf"))

    plt.close('all')
