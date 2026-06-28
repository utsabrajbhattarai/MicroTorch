#include <catch2/catch_test_macros.hpp>
#include "microtorch/Tensor.hpp"

// TODO: forward-pass correctness check for relu.
TEST_CASE("relu forward is correct", "[relu]") {
    // SKIP() until implemented -- keeps ctest's count honest (skipped,
    // not silently passing on an empty body).
    SKIP("not yet implemented");
}

// TODO: gradient check via finite differences for relu.
// This is the non-negotiable harness call -- every op gets one of these.
TEST_CASE("relu gradient check", "[relu][gradcheck]") {
    SKIP("not yet implemented");
}
