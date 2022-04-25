import numpy as np
import openmdao.api as om
import openconcept.api as oc
from openconcept.architecting.builder.architecture import (
    PropSysArch,
    ThrustGenElements,
    MechPowerElements,
    Propeller,
    Gearbox,
    Engine,
)
from openconcept.utilities.math import ElementMultiplyDivideComp
from openconcept.architecting.builder.ac_model import DynamicACModel
from openconcept.analysis.performance.mission_profiles import FullMissionAnalysis
from examples.aircraft_data.KingAirC90GT import data as acdata


class DynamicKingAirAnalysisGroup(om.Group):
    """
    Mission analysis model for the King Air with a customizable
    propulsion system (given as an option). It uses the aerodynamics
    and weights from the OpenConcept example HybridTwin.
    """

    def initialize(self):
        # Default propulsion system (conventional with gearbox)
        arch = PropSysArch(  # Conventional with gearbox
            thrust=ThrustGenElements(
                propellers=[Propeller("prop1"), Propeller("prop2")],
                gearboxes=[Gearbox("gearbox1"), Gearbox("gearbox2")],
            ),
            mech=MechPowerElements(
                engines=[Engine("turboshaft", power_rating=560.0), Engine("turboshaft", power_rating=560.0)]
            ),
        )
        self.options.declare(name="prop_arch", default=arch, types=PropSysArch, desc="Propulsion system architecture")
        self.options.declare(name="num_nodes", default=11, desc="Number of integration points per flight segment")

    def setup(self):
        # Take parameters that are not already defined in the DynamicACModel
        # from the default King Air C90GT data dictionary. This information
        # is used for aerodynamics and weights.
        dv_comp = self.add_subsystem("dv_comp", oc.DictIndepVarComp(acdata), promotes_outputs=["*"])
        dv_comp.add_output_from_dict("ac|aero|polar|e")
        dv_comp.add_output_from_dict("ac|aero|polar|CD0_TO")
        dv_comp.add_output_from_dict("ac|aero|polar|CD0_cruise")
        # dv_comp.add_output_from_dict("ac|geom|wing|S_ref")
        dv_comp.add_output_from_dict("ac|geom|wing|AR")
        dv_comp.add_output_from_dict("ac|geom|wing|c4sweep")
        dv_comp.add_output_from_dict("ac|geom|wing|taper")
        dv_comp.add_output_from_dict("ac|geom|wing|toverc")
        dv_comp.add_output_from_dict("ac|geom|hstab|S_ref")
        dv_comp.add_output_from_dict("ac|geom|hstab|c4_to_wing_c4")
        dv_comp.add_output_from_dict("ac|geom|vstab|S_ref")
        dv_comp.add_output_from_dict("ac|geom|fuselage|height")
        dv_comp.add_output_from_dict("ac|geom|fuselage|length")
        dv_comp.add_output_from_dict("ac|geom|fuselage|width")
        dv_comp.add_output_from_dict("ac|geom|fuselage|S_wet")
        dv_comp.add_output_from_dict("ac|geom|maingear|length")
        dv_comp.add_output_from_dict("ac|geom|nosegear|length")
        dv_comp.add_output_from_dict("ac|propulsion|engine|rating")  # used for baseline nacelle weight
        dv_comp.add_output_from_dict("ac|q_cruise")
        dv_comp.add_output_from_dict("ac|num_passengers_max")
        dv_comp.add_output_from_dict("ac|weights|MTOW")
        dv_comp.add_output_from_dict("ac|weights|MLW")
        dv_comp.add_output_from_dict("ac|weights|W_fuel_max")

        # Wing loading of the King Air (kg/m^2)
        wing_loading = acdata["ac"]["weights"]["MTOW"]["value"] / acdata["ac"]["geom"]["wing"]["S_ref"]["value"]
        self.add_subsystem(
            "wing_area",
            ElementMultiplyDivideComp(
                output_name="ac|geom|wing|S_ref",
                input_names=["ac|weights|MTOW", "wing_loading"],
                input_units=["kg", "kg/m**2"],
                divide=[False, True],
            ),
            promotes_inputs=["ac|weights|MTOW"],
            promotes_outputs=["ac|geom|wing|S_ref"],
        )
        self.set_input_defaults("wing_area.wing_loading", val=wing_loading, units="kg/m**2")

        mission = self.add_subsystem(
            "mission",
            FullMissionAnalysis(
                num_nodes=11,
                aircraft_model=DynamicACModel.factory(self.options["prop_arch"]),
            ),
            promotes_inputs=["*"],
            promotes_outputs=["*"],
        )

        self.options["prop_arch"].create_top_level(
            mission, ["v0v1", "v1vr", "rotate", "v1v0", "engineoutclimb", "climb", "cruise", "descent"], "propmodel"
        )

        # Compute MTOW - fuel burn - ZFW (OEW and payload) residual to be used as a constraint in the
        # optimization problem (> 0) when MTOW is a design variable
        self.add_subsystem(
            "TOW_margin",
            oc.AddSubtractComp(
                output_name="residual",
                input_names=["MTOW", "fuel_used", "OEW", "payload"],
                units="kg",
                scaling_factors=[1, -1, -1, -1],
            ),
            promotes_inputs=[("MTOW", "ac|weights|MTOW"), "payload"],
        )
        self.connect("cruise.ac|weights|OEW", "TOW_margin.OEW")  # connect one of the OEWs from within the mission
        self.connect("descent.fuel_used_final", "TOW_margin.fuel_used")

        self.add_subsystem(
            "aug_obj", AugmentedFBObjective(), promotes_inputs=["ac|weights|MTOW"], promotes_outputs=["mixed_objective"]
        )
        self.connect("descent.fuel_used_final", "aug_obj.fuel_burn")

        self.add_subsystem("energy", EnergyObjective(), promotes_inputs=[("W_battery", "ac|weights|W_battery")], promotes_outputs=["energy_used"])
        self.connect("descent.fuel_used_final", "energy.fuel_burn")


class AugmentedFBObjective(om.ExplicitComponent):
    def setup(self):
        self.add_input("fuel_burn", units="kg")
        self.add_input("ac|weights|MTOW", units="kg")
        self.add_output("mixed_objective", units="kg")
        self.declare_partials(["mixed_objective"], ["fuel_burn"], val=1)
        self.declare_partials(["mixed_objective"], ["ac|weights|MTOW"], val=1 / 100)

    def compute(self, inputs, outputs):
        outputs["mixed_objective"] = inputs["fuel_burn"] + inputs["ac|weights|MTOW"] / 100


class EnergyObjective(om.ExplicitComponent):
    def initialize(self):
        self.options.declare("e_fuel", default=11.95e3, desc="Specific energy of Jet A-1 (Wh/kg)")
    def setup(self):
        self.add_input("fuel_burn", units="kg")
        self.add_input("W_battery", val=0., units="kg")
        self.add_input("e_battery", val=300., units="W*h/kg")
        self.add_output("energy_used", units="W*h")
        self.declare_partials(["energy_used"], ["fuel_burn", "W_battery", "e_battery"])

    def compute(self, inputs, outputs):
        outputs["energy_used"] = inputs["fuel_burn"] * self.options["e_fuel"] + inputs["W_battery"] * inputs["e_battery"]
    
    def compute_partials(self, inputs, J):
        J["energy_used", "fuel_burn"] = self.options["e_fuel"]
        J["energy_used", "W_battery"] = inputs["e_battery"]
        J["energy_used", "e_battery"] = inputs["W_battery"]


def opt_prob(
    prop_arch=None,
    num_nodes=11,
    obj={"var": "mixed_objective", "kwargs": {}},
    prop_sys_DVs=[{"var": "ac|propulsion|propeller|diameter", "kwargs": {"lower": 2.2, "units": "m"}}],
    prop_sys_cons=[
        {"var": "climb.throttle", "kwargs": {"lower": 0.0, "upper": 1.0}},
        {"var": "cruise.throttle", "kwargs": {"lower": 0.0, "upper": 1.0}},
        {"var": "descent.throttle", "kwargs": {"lower": 0.0, "upper": 1.0}},
    ],
    model=DynamicKingAirAnalysisGroup,
    hst_file="opt.hst",
):
    """
    Optimization problem definition. Can be used for analyses too.

    Inputs
    ------
    prop_arch : PropSysArch
        Propulsion system architecture. If none is specified, it will default
        to a conventional architecture with a gearbox.
    num_nodes : int
        Number of numerical integration points per flight segment, by default 11.
    obj : dict
        Variable to use for objective function. The format of the dictionary should be
        {"var": <OpenMDAO variable name>, "kwargs": {<any optional keyword arguments>}}.
        Keyword arguments include ref, ref0, etc.
    prop_sys_DVs : list of dict
        List of propulsion system design variables. This is a list of dictionaries, each
        of which are in the same format as the obj dictionary.
    prop_sys_cons : dict
        List of propulsion system constraints. This is a list of dictionaries, each of which
        are in the same format as the obj dictionary.
    model : OpenMDAO Group
        Top level model for OpenMDAO problem, by default DynamicKingAriAnalysisGroup
    hst_file : str
        If the pyOptSparse driver is used, this is the optimization history file name.
    """
    # Check the inputs and set defaults
    if prop_arch is None:
        prop_arch = PropSysArch(  # Conventional with gearbox
            thrust=ThrustGenElements(
                propellers=[Propeller("prop1"), Propeller("prop2")],
                gearboxes=[Gearbox("gearbox1"), Gearbox("gearbox2")],
            ),
            mech=MechPowerElements(
                engines=[Engine("turboshaft", power_rating=560.0), Engine("turboshaft", power_rating=560.0)]
            ),
        )
    if "kwargs" not in obj:
        obj["kwargs"] = {}
    for DV in prop_sys_DVs:
        if "kwargs" not in DV:
            DV["kwargs"] = {}
    for con in prop_sys_cons:
        if "kwargs" not in con:
            con["kwargs"] = {}

    prob = om.Problem()
    prob.model = model(num_nodes=num_nodes, prop_arch=prop_arch)
    prob.model.nonlinear_solver = om.NewtonSolver(iprint=2, err_on_non_converge=True)
    prob.model.nonlinear_solver.linesearch = om.BoundsEnforceLS(print_bound_enforce=False)
    prob.model.options["assembled_jac_type"] = "csc"
    prob.model.linear_solver = om.DirectSolver(assemble_jac=True)
    prob.model.nonlinear_solver.options["solve_subsystems"] = True
    prob.model.nonlinear_solver.options["maxiter"] = 15
    prob.model.nonlinear_solver.options["atol"] = 1e-8
    prob.model.nonlinear_solver.options["rtol"] = 1e-8

    # Setup the optimization
    # prob.driver = om.ScipyOptimizeDriver(optimizer="SLSQP", maxiter=200, tol=1e-8, disp=True)
    # prob.driver.options["debug_print"] = ["objs", "desvars", "nl_cons"]

    prob.driver = om.pyOptSparseDriver(optimizer="SNOPT")
    prob.driver.hist_file = hst_file
    prob.driver.options["debug_print"] = ["objs", "desvars", "nl_cons"]
    prob.driver.opt_settings["Major iterations limit"] = 400
    prob.driver.opt_settings["Major feasibility tolerance"] = 1e-6
    prob.driver.opt_settings["Major optimality tolerance"] = 1e-5
    prefix = "".join(hst_file.split(".")[:-1])
    prob.driver.opt_settings["Print file"] = f"{prefix}_SNOPT_print.out"
    prob.driver.opt_settings["Summary file"] = f"{prefix}_SNOPT_summary.out"

    # prob.driver = om.pyOptSparseDriver(optimizer="IPOPT")
    # prob.driver.hist_file = hst_file
    # prob.driver.options["debug_print"] = ["objs", "desvars", "nl_cons"]
    # prob.driver.opt_settings["max_iter"] = 400
    # prob.driver.opt_settings["constr_viol_tol"] = 1e-6
    # prob.driver.opt_settings["nlp_scaling_method"] = "gradient-based"
    # prob.driver.opt_settings["acceptable_tol"] = 1e-5
    # prob.driver.opt_settings["acceptable_iter"] = 0
    # prob.driver.opt_settings["tol"] = 1e-5
    # prob.driver.opt_settings["mu_strategy"] = "adaptive"
    # prob.driver.opt_settings["corrector_type"] = "affine"
    # prob.driver.opt_settings["limited_memory_max_history"] = 1000
    # prob.driver.opt_settings["corrector_type"] = "primal-dual"

    # Add the objective
    prob.model.add_objective(obj["var"], **obj["kwargs"])

    # Use the optimizer to solve for initial weight (called MTOW here) by having initial
    # weight as a design variable and preventing the final weight from being less than
    # the zero fuel weight (OEW + payload)
    prob.model.add_design_var("ac|weights|MTOW", lower=1e3, ref=5e3)
    prob.model.add_constraint("TOW_margin.residual", lower=0.0)

    # Balanced field length must be at least as good as baseline
    prob.model.add_constraint("rotate.range_final", upper=4452.0, units="ft")
    prob.model.add_constraint("v1v0.range_final", upper=4452.0, units="ft")

    # Add propulsion system design variables and constraints
    for DV in prop_sys_DVs:
        prob.model.add_design_var(DV["var"], **DV["kwargs"])
    for con in prop_sys_cons:
        prob.model.add_constraint(con["var"], **con["kwargs"])

    return prob


def add_recorder(prob, filename="data.sql"):
    """
    Adds a recorder to the OpenMDAO problem and driver.

    Parameters
    ----------
    prob : OpenMDAO Problem
        Problem that has already been setup (prob.setup() has been called).
    filename : str
        Filename for recorder to save to.
    """
    recorder = om.SqliteRecorder(filename)
    prob.add_recorder(recorder)
    prob.driver.add_recorder(recorder)


def set_problem_vars(prob, num_nodes=11, e_batt=300.):
    """
    Sets mission profile and payload values. Also sets takeoff speed
    guesses (to improve convergence) and takeoff throttles.

    Parameters
    ----------
    prob : OpenMDAO Problem
        Problem that has already been setup (prob.setup() has been called).
    num_nodes : int
        Number of numerical integration points per flight segment, by default 11.
    e_batt : float
        Specific energy of batteries (Wh/kg), by default 300.
    """
    # Set required mission parameters
    prob.set_val("ac|weights|MTOW", 10099.0, units="lb")
    prob.set_val("ac|propulsion|engine|rating", 560.0, units="kW")
    prob.set_val("climb.fltcond|vs", np.ones((num_nodes,)) * 1500, units="ft/min")
    prob.set_val("climb.fltcond|Ueas", np.ones((num_nodes,)) * 124, units="kn")
    prob.set_val("cruise.fltcond|vs", np.ones((num_nodes,)) * 0.01, units="ft/min")
    prob.set_val("cruise.fltcond|Ueas", np.ones((num_nodes,)) * 170, units="kn")
    prob.set_val("descent.fltcond|vs", np.ones((num_nodes,)) * (-600), units="ft/min")
    prob.set_val("descent.fltcond|Ueas", np.ones((num_nodes,)) * 140, units="kn")

    prob.set_val("cruise|h0", 29000, units="ft")
    prob.set_val("mission_range", 1000, units="NM")
    prob.set_val("payload", 1000, units="lb")

    # (optional) guesses for takeoff speeds may help with convergence
    prob.set_val("v0v1.fltcond|Utrue", np.ones((num_nodes)) * 50, units="kn")
    prob.set_val("v1vr.fltcond|Utrue", np.ones((num_nodes)) * 85, units="kn")
    prob.set_val("v1v0.fltcond|Utrue", np.ones((num_nodes)) * 85, units="kn")

    # set some airplane-specific values. The throttle edits are to derate the takeoff power of the PT6A
    prob.set_val("v0v1.throttle", np.ones((num_nodes)) * 0.75)
    prob.set_val("v1vr.throttle", np.ones((num_nodes)) * 0.75)
    prob.set_val("rotate.throttle", np.ones((num_nodes)) * 0.75)

    # Set battery specific energy for energy objective
    prob.set_val("energy.e_battery", e_batt, units="W*h/kg")


if __name__ == "__main__":
    prob = opt_prob(prop_sys_DVs=[])
    prob.setup()
    set_problem_vars(prob)
    prob.run_model()
    # prob.run_driver()
    om.n2(prob, show_browser=False)

    # Print some results
    vars_list = ["ac|weights|MTOW", "descent.fuel_used_final", "rotate.range_final"]
    units = ["lb", "lb", "ft"]
    nice_print_names = ["MTOW", "Fuel used", "TOFL (over 35ft obstacle)"]
    print("=======================================================================")
    for i, thing in enumerate(vars_list):
        print(nice_print_names[i] + ": " + str(prob.get_val(thing, units=units[i])[0]) + " " + units[i])
