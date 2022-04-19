import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import os

curDir = os.path.abspath(os.path.dirname(__file__))
filepath = os.path.join(curDir, "data", "series_hybrid")

mission_ranges = np.linspace(300, 800, 10)
spec_energies = np.linspace(300, 800, 10)
e_batt_grid, range_grid = np.meshgrid(spec_energies, mission_ranges, indexing='xy')

x_spacing = spec_energies[1] - spec_energies[0]
y_spacing = mission_ranges[1] - mission_ranges[0]
x_list = np.hstack((spec_energies - x_spacing / 2, np.array([spec_energies[-1] + x_spacing / 2])))
y_list = np.hstack((mission_ranges - y_spacing / 2, np.array([mission_ranges[-1] + y_spacing / 2])))
x, y = np.meshgrid(x_list, y_list, indexing='xy')
obj = np.zeros_like(range_grid)

# Series hybrid
for i in range(range_grid.shape[0]):
    for j in range(range_grid.shape[1]):
        e_batt = e_batt_grid[i, j]
        mission_range = range_grid[i, j]

        try:
            with open(os.path.join(filepath, f"range{int(mission_range)}nmi_eBatt{int(e_batt)}.pkl"), "rb") as f:
                results = pkl.load(f)
        except FileNotFoundError:
            results = {"range": mission_range, "battery specific energy": e_batt, "fuel burn": np.NaN, "fuel energy": np.NaN, "battery energy": np.NaN, "MTOW": np.NaN, "mixed objective": np.NaN}
        
        print(results)
        
        obj[i, j] = results["mixed objective"]

plt.pcolormesh(x, y, obj)
plt.xlabel("Battery specific energy (Wh/kg)")
plt.ylabel("Mission range (nmi)")
plt.colorbar()
plt.title("fuel burn + MTOW / 100 (kg)")
plt.savefig(os.path.join(filepath, "series_hybrid.pdf"))
