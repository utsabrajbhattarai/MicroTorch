#include <catch2/catch_test_macros.hpp>
#include "microtorch/Tensor.hpp"

// TODO: forward-pass correctness check for neighbor_aggregation.
TEST_CASE("neighbor_aggregation forward is correct", "[neighbor_aggregation]") {
    // SKIP() until implemented -- keeps ctest's count honest (skipped,
    // not silently passing on an empty body).
    SKIP("not yet implemented");
}

// TODO: gradient check via finite differences for neighbor_aggregation.
// This is the non-negotiable harness call -- every op gets one of these.
TEST_CASE("neighbor_aggregation gradient check", "[neighbor_aggregation][gradcheck]") {
    SKIP("not yet implemented");
}
