import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import os

curDir = os.path.abspath(os.path.dirname(__file__))
arch = "turboelectric"
filepath = os.path.join(curDir, "data", arch)

mission_ranges = np.linspace(300, 800, 10)
obj = np.zeros_like(mission_ranges)

# Series hybrid
for i, mission_range in enumerate(mission_ranges):
    try:
        with open(os.path.join(filepath, f"range{int(mission_range)}nmi.pkl"), "rb") as f:
            results = pkl.load(f)
    except FileNotFoundError:
        results = {"range": mission_range, "battery specific energy": np.NaN, "fuel burn": np.NaN, "fuel energy": np.NaN, "battery energy": np.NaN, "MTOW": np.NaN, "mixed objective": np.NaN}
    
    print(results)
    
    obj[i] = results["mixed objective"]

plt.scatter(mission_ranges, obj)
plt.xlabel("Mission range (nmi)")
plt.ylabel("fuel burn + MTOW / 100 (kg)")
plt.savefig(os.path.join(filepath, f"{arch}.pdf"))
