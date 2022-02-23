
import openmdao.api as om
from openconcept.components import SimplePropeller, SimpleTurboshaft
from openconcept.components.gearbox import SimpleGearbox
from openconcept.utilities.dvlabel import DVLabel
from openconcept.utilities.math import AddSubtractComp, ElementMultiplyDivideComp
from openmdao.api import ExecComp


class ConventionalTwinTurboprop(om.Group):
    """
    an example model for a conventional Twin turboprop propulsion system

    components included:
        2 turboshaft components
        2 constant speed propellers (propeller has either 3 or 4 blades by default)
        1 "propulsor_active" flag component (failedengine)
        1 adder: to add weights and thrust of components
        1 probmodel_weight: to output the total prop system weight (propellers + engines)


    Inputs
    ------
        ac|propulsion|engine|rating : float
            The maximum rated shaft power of the engine (Scalar, W)
        ac|propulsion|propeller|diameter : float
            Diameter of the propeller (Scalar, m)
        ac|propulsion|propeller|power_rating : float
            Power rating of the propeller (Scalar, 'kW')

    Options
    -------
        num_nodes : float
            Number of analysis points to run (default 1)

    Controls
    --------
        prop|rpm: float
            the propeller rpm (vec, RPM)

    Outputs
    ------
        propulsion_system_weight : float
            The weight of the propulsion system (Scalar, 'kg')
        fuel_flow: float
            The fuel flow consumed in the segment (Vec, 'kg/s')
        thrust: float
            The total thrust of the propulsion system (Vec, 'N')

    needs (external information)
        --------
        fltcond|rho: float
            air density at the specific flight condition (vec, kg/m**3)
        fltcond|Utrue: float
            true airspeed at the flight condition (vec, m/s)
        throttle: float
            throttle input to the engine, fraction from 0-1 (vec, '')
        propulsor_active: float (either 0 or 1)
            a flag to indicate on or off for the connected propulsor either 1 or 0 (vec, '')

    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options['num_nodes']

        # rename incoming design variables
        dvlist = [['ac|propulsion|engine|rating', 'eng_rating', 850, 'hp'],
                  ['ac|propulsion|propeller|diameter', 'prop_diameter', 2.3, 'm'],
                  ['ac|propulsion|propeller|power_rating', 'prop_rating', 850, 'hp']]

        self.add_subsystem('dvs', DVLabel(dvlist),
                           promotes_inputs=["*"], promotes_outputs=["*"])

        # introduce model components
        self.add_subsystem('eng1',
                           SimpleTurboshaft(num_nodes=nn, weight_inc=0.14 / 1000, weight_base=104),
                           promotes_inputs=["throttle"])
        self.add_subsystem('prop1',
                           SimplePropeller(num_nodes=nn, num_blades=4,
                                           design_J=2.2, design_cp=0.55),
                           promotes_inputs=["fltcond|*"])

        # propmodel expect a high-level 'throttle' parameter and a 'propulsor_active' flag to set individual throttles
        failedengine = ElementMultiplyDivideComp()
        failedengine.add_equation('eng2throttle', input_names=['throttle', 'propulsor_active'], vec_size=nn)
        self.add_subsystem('failedengine', failedengine,
                           promotes_inputs=['throttle', 'propulsor_active'])

        self.add_subsystem('eng2',
                           SimpleTurboshaft(num_nodes=nn, weight_inc=0.14 / 1000, weight_base=104))

        self.add_subsystem('prop2',
                           SimplePropeller(num_nodes=nn, num_blades=4,
                                           design_J=2.2, design_cp=0.55),
                           promotes_inputs=["fltcond|*"])

        # connect design variables to model component inputs
        self.connect('eng_rating', 'eng1.shaft_power_rating')
        self.connect('eng_rating', 'eng2.shaft_power_rating')
        self.connect('prop_rating', 'prop1.power_rating')
        self.connect('prop_rating', 'prop2.power_rating')
        self.connect('prop_diameter', 'prop1.diameter')
        self.connect('prop_diameter', 'prop2.diameter')

        # conncet propulsor_active flag to second engine
        self.connect('failedengine.eng2throttle', 'eng2.throttle')

        # connect components to each other
        self.connect('eng1.shaft_power_out', 'prop1.shaft_power_in')
        self.connect('eng2.shaft_power_out', 'prop2.shaft_power_in')

        # add up the weights, thrusts and fuel flows
        add1 = AddSubtractComp(output_name='fuel_flow', input_names=['eng1_fuel_flow', 'eng2_fuel_flow'], vec_size=nn,
                               units='kg/s')
        add1.add_equation(output_name='thrust', input_names=['prop1_thrust', 'prop2_thrust'], vec_size=nn, units='N')
        add1.add_equation(output_name='engines_weight', input_names=['eng1_weight', 'eng2_weight'], units='kg')
        add1.add_equation(output_name='propellers_weight', input_names=['prop1_weight', 'prop2_weight'], units='kg')
        self.add_subsystem('adder', subsys=add1, promotes_inputs=["*"], promotes_outputs=["*"])
        self.connect('prop1.thrust', 'prop1_thrust')
        self.connect('prop2.thrust', 'prop2_thrust')
        self.connect('eng1.fuel_flow', 'eng1_fuel_flow')
        self.connect('eng2.fuel_flow', 'eng2_fuel_flow')
        self.connect('prop1.component_weight', 'prop1_weight')
        self.connect('prop2.component_weight', 'prop2_weight')
        self.connect('eng1.component_weight', 'eng1_weight')
        self.connect('eng2.component_weight', 'eng2_weight')

        # calculate prop system weight
        propmodel_weight = self.add_subsystem('propmodel_weight', ExecComp([
            'propulsion_system_weight = total_engines_weight + total_propellers_weight'
        ],
            propulsion_system_weight={'value': 1.0, 'units': 'kg'},
            total_engines_weight={'value': 1.0, 'units': 'kg'},
            total_propellers_weight={'value': 1.0, 'units': 'kg'},
        ),
                                              promotes_inputs=['*'],
                                              promotes_outputs=['propulsion_system_weight'])
        self.connect('engines_weight', 'total_engines_weight')
        self.connect('propellers_weight', 'total_propellers_weight')


class ConventionalTwinTurbopropExpanded(om.Group):
    """
    an example model for a conventional Twin turboprop propulsion system
    expanded model additionally includes a gearbox component between propeller and turboshaft engine

    components included:
        2 turboshaft components
        2 gearbox components
        2 constant speed propellers (propeller has either 3 or 4 blades by default)
        1 "propulsor_active" flag component (failedengine)
        1 adder: to add weights and thrust of components
        1 probmodel_weight: to output the total prop system weight (propellers + engines)


    Inputs
    ------
        ac|propulsion|engine|rating : float
            The maximum rated shaft power of the engine (Scalar, W)
        ac|propulsion|engine|output_rpm : float
            The operating shaft rpm output of the engine (Scalar, rpm)
        ac|propulsion|propeller|diameter : float
            Diameter of the propeller (Scalar, m)
        ac|propulsion|propeller|power_rating : float
            Power rating of the propeller (Scalar, 'kW')
        ac|propulsion|propeller|max_speed
            the maximum operating rpm of the propeller (Scalar, rpm)
        ac|propulsion|propeller|min_speed
            the minimum operating rpm of the propeller (Scalar, rpm)

    Options
    -------
        num_nodes : float
            Number of analysis points to run (default 1)

    Controls
    --------
        prop|rpm: float
            the propeller rpm (vec, RPM)

    Outputs
    ------
        propulsion_system_weight : float
            The weight of the propulsion system (Scalar, 'kg')
        fuel_flow: float
            The fuel flow consumed in the segment (Vec, 'kg/s')
        thrust: float
            The total thrust of the propulsion system (Vec, 'N')

    needs (external information)
        --------
        fltcond|rho: float
            air density at the specific flight condition (vec, kg/m**3)
        fltcond|Utrue: float
            true airspeed at the flight condition (vec, m/s)
        throttle: float
            throttle input to the engine, fraction from 0-1 (vec, '')
        propulsor_active: float (either 0 or 1)
            a flag to indicate on or off for the connected propulsor either 1 or 0 (vec, '')

    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc="Number of mission analysis points to run")

    def setup(self):
        nn = self.options['num_nodes']

        # rename incoming design variables
        dvlist = [['ac|propulsion|engine|rating', 'eng_rating', 850, 'hp'],
                  ['ac|propulsion|engine|output_rpm', 'engine_output_rpm', 6000.0, 'rpm'],
                  ['ac|propulsion|propeller|diameter', 'prop_diameter', 2.3, 'm'],
                  ['ac|propulsion|propeller|max_speed', 'prop_max_speed', 2420.0, 'rpm'],  # not used
                  ['ac|propulsion|propeller|min_speed', 'prop_min_speed', 400.0, 'rpm'],  # gearbox sizing parameter
                  ['ac|propulsion|propeller|power_rating', 'prop_rating', 850, 'hp']]

        self.add_subsystem('dvs', DVLabel(dvlist),
                           promotes_inputs=["*"], promotes_outputs=["*"])

        # introduce model components
        self.add_subsystem('eng1',
                           SimpleTurboshaft(num_nodes=nn, weight_inc=0.14 / 1000, weight_base=104),
                           promotes_inputs=["throttle"])
        self.add_subsystem('gearbox1', SimpleGearbox(num_nodes=nn))
        self.add_subsystem('prop1',
                           SimplePropeller(num_nodes=nn, num_blades=4,
                                           design_J=2.2, design_cp=0.55),
                           promotes_inputs=["fltcond|*"])

        # propmodel expect a high-level 'throttle' parameter and a 'propulsor_active' flag to set individual throttles
        failedengine = ElementMultiplyDivideComp()
        failedengine.add_equation('eng2throttle', input_names=['throttle', 'propulsor_active'], vec_size=nn)
        self.add_subsystem('failedengine', failedengine,
                           promotes_inputs=['throttle', 'propulsor_active'])

        self.add_subsystem('eng2',
                           SimpleTurboshaft(num_nodes=nn, weight_inc=0.14 / 1000, weight_base=104))
        self.add_subsystem('gearbox2', SimpleGearbox(num_nodes=nn))
        self.add_subsystem('prop2',
                           SimplePropeller(num_nodes=nn, num_blades=4,
                                           design_J=2.2, design_cp=0.55),
                           promotes_inputs=["fltcond|*"])

        # connect design variables to model component inputs
        self.connect('eng_rating', 'eng1.shaft_power_rating')
        self.connect('eng_rating', 'eng2.shaft_power_rating')
        self.connect('eng_rating', ['gearbox1.shaft_power_rating', 'gearbox2.shaft_power_rating'])
        self.connect('engine_output_rpm', ['gearbox1.shaft_speed_in', 'gearbox2.shaft_speed_in'])
        self.connect('prop_min_speed', ['gearbox1.shaft_speed_out', 'gearbox2.shaft_speed_out'])
        self.connect('prop_rating', 'prop1.power_rating')
        self.connect('prop_rating', 'prop2.power_rating')
        self.connect('prop_diameter', 'prop1.diameter')
        self.connect('prop_diameter', 'prop2.diameter')

        # conncet propulsor_active flag to second engine
        self.connect('failedengine.eng2throttle', 'eng2.throttle')

        # connect components to each others to pass the power
        self.connect('eng1.shaft_power_out', 'gearbox1.shaft_power_in')
        self.connect('gearbox1.shaft_power_out', 'prop1.shaft_power_in')
        self.connect('eng2.shaft_power_out', 'gearbox2.shaft_power_in')
        self.connect('gearbox2.shaft_power_out', 'prop2.shaft_power_in')

        # add up the weights, thrusts and fuel flows
        add1 = AddSubtractComp(output_name='fuel_flow', input_names=['eng1_fuel_flow', 'eng2_fuel_flow'], vec_size=nn,
                               units='kg/s')
        add1.add_equation(output_name='thrust', input_names=['prop1_thrust', 'prop2_thrust'], vec_size=nn, units='N')
        add1.add_equation(output_name='engines_weight', input_names=['eng1_weight', 'eng2_weight'], units='kg')
        add1.add_equation(output_name='propellers_weight', input_names=['prop1_weight', 'prop2_weight'], units='kg')
        add1.add_equation(output_name='gearboxes_weight', input_names=['gearbox1_weight',
                                                                       'gearbox2_weight'], units='kg')
        self.add_subsystem('adder', subsys=add1, promotes_inputs=["*"], promotes_outputs=["*"])
        self.connect('prop1.thrust', 'prop1_thrust')
        self.connect('prop2.thrust', 'prop2_thrust')
        self.connect('eng1.fuel_flow', 'eng1_fuel_flow')
        self.connect('eng2.fuel_flow', 'eng2_fuel_flow')
        self.connect('prop1.component_weight', 'prop1_weight')
        self.connect('prop2.component_weight', 'prop2_weight')
        self.connect('eng1.component_weight', 'eng1_weight')
        self.connect('eng2.component_weight', 'eng2_weight')
        self.connect('gearbox1.component_weight', 'gearbox1_weight')
        self.connect('gearbox2.component_weight', 'gearbox2_weight')

        # calculate prop system weight
        propmodel_weight = self.add_subsystem('propmodel_weight', ExecComp([
            'propulsion_system_weight = total_engines_weight + total_propellers_weight + total_gearboxes_weight'
        ],
            propulsion_system_weight={'value': 1.0, 'units': 'kg'},
            total_engines_weight={'value': 1.0, 'units': 'kg'},
            total_propellers_weight={'value': 1.0, 'units': 'kg'},
            total_gearboxes_weight={'value': 1.0, 'units': 'kg'},
        ),
                                              promotes_inputs=['*'],
                                              promotes_outputs=['propulsion_system_weight'])
        self.connect('engines_weight', 'total_engines_weight')
        self.connect('propellers_weight', 'total_propellers_weight')
        self.connect('gearboxes_weight', 'total_gearboxes_weight')


