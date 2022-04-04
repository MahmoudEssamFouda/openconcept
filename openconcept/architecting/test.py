# test runner for propulsion architecture builder module

import unittest
from builder.tests.dynamic_conventional_twin_turboprop_test import DynamicConventionalTwinTurbopropTestCase
from builder.tests.dynamic_electric_twin_prop_test import DynamicElectricTwinTurbopropTestCase
from builder.tests.dynamic_series_hybrid_twin_prop_test import DynamicSeriesHybridTwinTurbopropTestCase
from builder.tests.dynamic_turboelectric_twin_prop_test import DynamicTurboelectricTwinTurbopropTestCase

# tests to be added for more prop systems that can be constructed with current functionality
# fully electric distributed propulsion system with any number of propellers
# turboelectric distributed propulsion system with any number of propellers
# series distributed propulsion system with any number of propellers

if __name__ == '__main__':
    unittest.main()
