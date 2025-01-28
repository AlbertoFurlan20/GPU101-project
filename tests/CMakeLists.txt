# Integration test
add_executable(integration_test integration/integration_test.cpp)
target_link_libraries(integration_test PRIVATE gtest gtest_main OpenMP::OpenMP_CXX)

# Performance test
add_executable(performance_test performance/performance_test.cpp)
target_link_libraries(performance_test PRIVATE OpenMP::OpenMP_CXX)

# Register the tests
add_test(NAME IntegrationTest COMMAND integration_test)
add_test(NAME PerformanceTest COMMAND performance_test)