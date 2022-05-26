from copy import deepcopy
import enum
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import matplotlib.patches as mpatches
import niceplots as nice
import os
import pickle as pkl
import openmdao.api as om
import numpy as np
from openconcept.architecting.evaluator.postprocess.utils import get_snopt_exit

# ================= PLOT SETTINGS =================
categories = ["Propulsion", "Structure", "Hydraulics, avionics, etc.", "Payload", "Other"]
subcategories = [
    ["Battery", "Fuel", "Propulsion system"],
    ["Wing", "Fuselage", None, None, None],  # Nones are gear, empennage, nacelle
    [None],  # Equipment
    [None],  # Payload
    [None, None],  # Fuel system, fluids
]

vars = [
    [{"name": "ac|weights|W_battery_pass", "pkl": False}, {"name": "fuel burn", "pkl": True}, {"name": "Other propulsion"}],
    [{"name": "climb.OEW_calc.W_wing", "pkl": False}, {"name": "climb.OEW_calc.W_fuselage", "pkl": False}, {"name": "climb.OEW_calc.W_gear", "pkl": False}, {"name": "climb.OEW_calc.W_empennage", "pkl": False}, {"name": "climb.OEW_calc.W_nacelle", "pkl": False}],
    [{"name": "climb.OEW_calc.W_equipment", "pkl": False}],
    [{"name": "payload", "pkl": False}],
    [{"name": "climb.OEW_calc.W_fuelsystem", "pkl": False}, {"name": "climb.OEW_calc.W_fluids", "pkl": False}],
]

cases = [{"range": 500, "e_batt": 500}, {"range": 300, "e_batt": 500}]
figsize = [(5.7, 5.3), (7, 5.3)]

archs = ["electric", "series_hybrid", "parallel_hybrid", "turboelectric", "conventional"]
is_oneD = [
    False,
    False,
    False,
    True,
    True,
]  # True if only sweep over mission range and not battery spec energy (because no battery)

payload = 453.592  # kg, 1000 lbs

# ================= COLOR DEFINITION =================
colors = ['#56b2f0', '#F0BC4A', '#E98F51', '#D47F82', '#A97B95']  # custom colors #4FA2CE old blue
# colors = [mcolor.rgb2hex(plt.colormaps["Set2"](i)) for i in range(len(categories))]
alphas = np.linspace(0.2, 0.95, 10)[-1::-1]
sub_colors = []
for i, sublist in enumerate(subcategories):
    sub_colors.append([])
    for i_alpha in range(len(sublist)):
        sub_colors[-1].append(colors[i] + float.hex(alphas[i_alpha])[4:6])


# ================= FUNCTIONS FOR PIE CHART LAYERS =================
def sum_categories(vals_list):
    """
    Sums the sublists of a 2D Python list.

    Example:
        Input: [[1], [2, 5], [1, -1]]
        Output: [1, 7, 0]
    """
    result = []
    for sublist in vals_list:
        total = sublist[0] * 0  # keeps the type of the sublist
        for item in sublist:
            total += item
        result.append(total)
    return result


def expand_subcategories(vals_list):
    """
    Flattens a 2D Python list.

    Example:
        Input: [[1], [2, 5], [1, -1]]
        Output: [1,2, 5, 1, -1]
    """
    result = []
    for sublist in vals_list:
        for item in sublist:
            result.append(item)
    return result

# ================= COLLECT DATA AND PLOT =================
nice.setRCParams(dark_mode=False, set_dark_background=False)
plt.rcParams["font.size"] = 12

curDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

for i_c, c in enumerate(cases):
    mission_range = c["range"]
    e_batt = c["e_batt"]

    fig = plt.figure(figsize=figsize[i_c])
    ax = plt.gca()
    ax.set_ylabel("Weight (kg)")

    for i_arch, arch in enumerate(archs):
        oneD = is_oneD[i_arch]
        filepath = os.path.join(curDir, "data", "mtow_bound", "grid", arch)
        filepath_save = os.path.join(curDir, "postprocess", "figures", "weight_pie_charts")
        subcat_labels = deepcopy(subcategories)

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

        nice_arch = arch.capitalize().replace("_", " ")
        if get_snopt_exit(os.path.join(filepath, filename + "_SNOPT_print.out")) != (0, 1):
            continue

        try:
            cr = om.CaseReader(os.path.join(filepath, filename + ".sql"))
            case = cr.get_case("optimized")
        except OSError:
            raise OSError(
                f"No data is available for {arch} with a range of {mission_range} nmi and specific energy of {e_batt} Wh/kg"
            )

        # Collect values
        vals = []
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
                    if W_batt == 0.0:
                        subcat_labels[i_cat][i_subcat] = None
                elif subcat == "Propulsion system":
                    W_prop_sys = case.get_val("cruise.propulsion_system_weight", units="kg").item()
                    vals[-1].append(W_prop_sys - W_batt)
                elif var["pkl"]:
                    vals[-1].append(results[var["name"]])
                else:
                    vals[-1].append(case.get_val(var["name"], units="kg").item())
                
                if subcat == "Fuel" and vals[-1][-1] == 0.0:
                    subcat_labels[i_cat][i_subcat] = None

                vals[-1][-1] *= multiplier

        size = 0.3
        cmap = plt.colormaps["tab20c"]
        outer_colors = cmap(np.arange(3)*4)
        inner_colors = cmap([1, 2, 5, 6, 9, 10])

        outer_vals = expand_subcategories(vals)
        outer_colors = expand_subcategories(sub_colors)
        outer_labels = expand_subcategories(subcat_labels)
        for i_lab, label in enumerate(outer_labels):
            if label is None:
                continue
            if label == "Fuel":
                outer_labels[i_lab] += f", {outer_vals[i_lab] / results['MTOW'] * 100:1.1f}%"
            else:
                outer_labels[i_lab] += f"\n{outer_vals[i_lab] / results['MTOW'] * 100:1.1f}%"

        bottom = 0
        labels = []
        for i, val in enumerate(outer_vals):
            bar_container = ax.bar(nice_arch, val, bottom=bottom, color=outer_colors[i])
            if outer_labels[i]:
                labels.append(ax.bar_label(bar_container, labels=[outer_labels[i]], label_type="center", color="w")[0])
            else:
                labels.append(ax.bar_label(bar_container, labels=[""], label_type="center", color="w")[0])
            bottom += val
        labels[0].set_fontsize(8)  # "Battery"
        labels[1].set_fontsize(8)  # "Fuel"
        labels[2].set_fontsize(8)  # "Propulsion system"
        labels[3].set_fontsize(8)  # "Wing"
        labels[4].set_fontsize(8)  # "Fuselage"

    nice.adjust_spines(ax, spines=["left", "bottom"])
    ax.spines["bottom"].set_position(("outward", 0))
    ax.tick_params(axis='x', tick1On=False)

    # Make label
    handles = []
    for i in range(len(categories)):
        i_cat = len(categories) - 1 - i
        handles.append(mpatches.Patch(color=colors[i_cat], label=categories[i_cat]))
    fig.legend(handles=handles, markerfirst=False)

    fig.savefig(os.path.join(filepath_save, f"weight_{mission_range}nmi_eBatt{e_batt}.pdf"))
