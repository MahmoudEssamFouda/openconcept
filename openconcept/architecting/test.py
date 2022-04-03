# test runner for propulsion architecture builder module

import unittest
from builder.tests.dynamic_conventional_twin_turboprop_test import DynamicConventionalTwinTurbopropTestCase
from builder.tests.dynamic_electric_twin_prop_test import DynamicElectricTwinTurbopropTestCase
from builder.tests.dynamic_series_hybrid_twin_prop_test import DynamicSeriesHybridTwinTurbopropTestCase
from builder.tests.dynamic_turboelectric_twin_prop_test import DynamicTurboelectricTwinTurbopropTestCase


if __name__ == '__main__':
    unittest.main()
