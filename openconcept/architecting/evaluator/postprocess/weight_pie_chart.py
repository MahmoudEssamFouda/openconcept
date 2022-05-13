from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import niceplots as nice
import os
import pickle as pkl
import openmdao.api as om
import numpy as np

# ================= PLOT SETTINGS =================
categories = ["Propulsion", "Structure", "Hydraulics,\navionics, etc.", "Payload", "Other"]
subcategories = [
    ["Battery", "Fuel", "Propulsion\nsystem"],
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

for i_arch, arch in enumerate(archs):
    oneD = is_oneD[i_arch]
    filepath = os.path.join(curDir, "data", arch)
    filepath_save = os.path.join(curDir, "postprocess", "figures", "weight_pie_charts")
    subcat_labels = deepcopy(subcategories)

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
                elif subcat == "Propulsion\nsystem":
                    W_prop_sys = case.get_val("cruise.propulsion_system_weight", units="kg").item()
                    vals[-1].append(W_prop_sys - W_batt)
                elif var["pkl"]:
                    vals[-1].append(results[var["name"]])
                else:
                    vals[-1].append(case.get_val(var["name"], units="kg").item())
                
                if subcat == "Fuel" and vals[-1][-1] == 0.0:
                    subcat_labels[i_cat][i_subcat] = None

                vals[-1][-1] *= multiplier

        fig = plt.figure(figsize=(5, 5))
        ax = plt.gca()
        nice_arch = arch.capitalize().replace("_", " ")
        size = 0.3
        cmap = plt.colormaps["tab20c"]
        outer_colors = cmap(np.arange(3)*4)
        inner_colors = cmap([1, 2, 5, 6, 9, 10])

        _, outer_text = ax.pie(expand_subcategories(vals), radius=1, colors=expand_subcategories(sub_colors),
            wedgeprops=dict(width=size, edgecolor=None), textprops=dict(rotation_mode='anchor', va='center', ha='center', color="w"),
            labels=expand_subcategories(subcat_labels), rotatelabels=False, labeldistance=.85)

        _, inner_text = ax.pie(sum_categories(vals), radius=1-size-0.01, colors=colors,
            wedgeprops=dict(width=size, edgecolor=None), textprops=dict(rotation_mode='anchor', va='center', ha='center', color="w"),
            labels=categories, rotatelabels=False, labeldistance=.75)

        ax.set(aspect="equal")
        ax.set_title(f"{nice_arch}, {mission_range} nmi and {e_batt} Wh/kg", y=0.935)

        # Manually set some font sizes so they fit better
        inner_text[-4].set_fontsize(8)  # "Structure"
        inner_text[-3].set_fontsize(6)  # "Equipment"
        inner_text[-2].set_fontsize(8)  # "Payload"
        inner_text[-1].set_fontsize(4)  # "Other"
        outer_text[3].set_fontsize(8)  # "Wing"
        outer_text[4].set_fontsize(8)  # "Fuselage"
        outer_text[1].set_fontsize(10)  # "Fuel"
        outer_text[2].set_fontsize(8)  # "Propulsion system"
        if arch == "conventional":
            inner_text[0].set_fontsize(10)

        # Hack the matplotlib pie chart command to put text in the center
        ax.pie([1.], radius=0.01, colors=["w"], labels=[f"MTOW\n{results['MTOW']:0.0f} kg"], textprops=dict(rotation_mode='anchor', va='center', ha='center'))

        fig.savefig(os.path.join(filepath_save, f"{arch}_weight_{mission_range}nmi_eBatt{e_batt}.pdf"))
    