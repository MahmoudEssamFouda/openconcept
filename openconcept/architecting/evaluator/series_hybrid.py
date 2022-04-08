import openmdao.api as om
from openconcept.architecting.evaluator.analysis_group import opt_prob, set_problem_vars
from openconcept.architecting.builder.architecture import *

prop_arch = PropSysArch(  # Series Hybrid Electric with gearbox,  inverter, dc_bus, splitter, and engine chain
    thrust=ThrustGenElements(
        propellers=[
            Propeller(name='prop1', blades=4, diameter=2.3, design_adv_ratio=2.2, design_cp=0.55),
            Propeller(name='prop2', blades=4, diameter=2.3, design_adv_ratio=2.2, design_cp=0.55)
        ],
        gearboxes=[
            Gearbox(name='gearbox1'), Gearbox(name='gearbox2')
        ]
    ),
    mech=MechPowerElements(motors=Motor(name='elec_motor', power_rating=240, efficiency=0.97, output_rpm=5500,
                                        specific_weight=1. / 5000, base_weight=0.,
                                        cost_inc=100.0 / 745.0, cost_base=1.),
                            inverters=Inverter(name='inverter', efficiency=0.97,
                                                specific_weight=1. / (10 * 1000), base_weight=0.,
                                                cost_inc=100.0 / 745.0, cost_base=1.)),

    electric=ElectricPowerElements(dc_bus=DCBus(name='elec_bus', efficiency=0.99),
                                    splitter=ElecSplitter(name='splitter', efficiency=1, split_rule='fraction',
                                                            elec_DoH=0.6),
                                    batteries=Batteries(name='bat_pack', weight=1000, efficiency=0.97,
                                                        specific_power=5000, specific_energy=300,
                                                        cost_inc=50., cost_base=1.),
                                    engines_dc=(Engine(name='turboshaft', power_rating=260,
                                                        specific_weight=.14 / 1000, base_weight=104, psfc=0.6,
                                                        output_rpm=6000),
                                                Generator(name='generator', efficiency=0.97,
                                                            specific_weight=1. / 5000, base_weight=0.,
                                                            cost_inc=100.0 / 745.0, cost_base=1.),
                                                Rectifier(name='rectifier', efficiency=0.97,
                                                            specific_weight=1. / (10 * 1000), base_weight=0.,
                                                            cost_inc=100.0 / 745.0, cost_base=1.)
                                                ),
                                    ),
)

obj={"var": "mixed_objective", "kwargs": {}}
DVs=[
    {"var": "ac|propulsion|propeller|diameter", "kwargs": {"lower": 2.2, "units": "m"}},
    {"var": "ac|propulsion|engine|rating", "kwargs": {"lower": 200, "upper": 1e3, "ref": 5e2, "units": "kW"}}
]
cons=[
    {"var": "climb.throttle", "kwargs": {"lower": 0.0, "upper": 1.0}},
    {"var": "cruise.throttle", "kwargs": {"lower": 0.0, "upper": 1.0}},
    {"var": "descent.throttle", "kwargs": {"lower": 0.0, "upper": 1.0}},
]

p = opt_prob(prop_arch=prop_arch, obj=obj, prop_sys_DVs=DVs, prop_sys_cons=cons)
p.setup()
set_problem_vars(p)
p.run_model()
om.n2(p, show_browser=False)
# p.run_driver()
