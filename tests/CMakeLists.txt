# Pulls Catch2 once at configure time -- no system install needed, so this
# works identically on your machine and both teammates' machines.
include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.5.4
)
FetchContent_MakeAvailable(Catch2)

# One test binary for the whole engine. Add a source file here every time
# you add an op to src/CMakeLists.txt -- this list should always mirror
# that one 1:1.
add_executable(microtorch_tests
    test_tensor.cpp
    test_matmul.cpp
    test_broadcast_add.cpp
    test_relu.cpp
    test_neighbor_aggregation.cpp
    test_reduction.cpp
    test_mse_loss.cpp
    test_softmax_cross_entropy.cpp
    test_sgd.cpp
    test_adam.cpp
    test_gradcheck_harness.cpp   # the harness itself gets its own test
)

target_link_libraries(microtorch_tests
    PRIVATE microtorch Catch2::Catch2WithMain
)

# registers each Catch2 TEST_CASE as its own ctest entry, so a failure
# points you straight at which op broke instead of "tests failed" as a blob
include(Catch)
catch_discover_tests(microtorch_tests
    PROPERTIES SKIP_REGULAR_EXPRESSION "SKIPPED:"     #prevents failures from skipped tests
)