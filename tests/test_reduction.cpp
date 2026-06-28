#include <catch2/catch_test_macros.hpp>
#include "microtorch/Tensor.hpp"

// TODO: forward-pass correctness check for sum/mean reduction.
TEST_CASE("sum/mean reduction forward is correct", "[reduction]") {
    // SKIP() until implemented -- keeps ctest's count honest (skipped,
    // not silently passing on an empty body).
    SKIP("not yet implemented");
}

// TODO: gradient check via finite differences for sum/mean reduction.
// This is the non-negotiable harness call -- every op gets one of these.
TEST_CASE("sum/mean reduction gradient check", "[reduction][gradcheck]") {
    SKIP("not yet implemented");
}
