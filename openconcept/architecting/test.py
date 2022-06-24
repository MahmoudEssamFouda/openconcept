# test runner for propulsion architecture builder module

import unittest
from builder.tests.dynamic_conventional_twin_turboprop_test import DynamicConventionalTwinTurbopropTestCase
from builder.tests.dynamic_electric_twin_prop_test import DynamicElectricTwinTurbopropTestCase
from builder.tests.dynamic_series_hybrid_twin_prop_test import DynamicSeriesHybridTwinTurbopropTestCase
from builder.tests.dynamic_turboelectric_twin_prop_test import DynamicTurboelectricTwinTurbopropTestCase
from builder.tests.dynamic_parallel_hybrid_twin_prop_test import DynamicParallelHybridTwinTurbopropTestCase

# tests to be added for more prop systems that can be constructed with current functionality
# fully electric distributed propulsion system with any number of propellers
# turboelectric distributed propulsion system with any number of propellers and one engine chain
# series distributed propulsion system with any number of propellers and one engine chain and one battery pack

# possible expansions with current interface
# AC architectures
# mutliple engines chains
# multiple battery backs

if __name__ == "__main__":
    unittest.main()
