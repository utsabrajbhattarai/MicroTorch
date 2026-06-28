#include <catch2/catch_test_macros.hpp>
#include "microtorch/Tensor.hpp"

// TODO: forward-pass correctness check for Adam optimizer.
TEST_CASE("Adam optimizer forward is correct", "[adam]") {
    // SKIP() until implemented -- keeps ctest's count honest (skipped,
    // not silently passing on an empty body).
    SKIP("not yet implemented");
}

// TODO: gradient check via finite differences for Adam optimizer.
// This is the non-negotiable harness call -- every op gets one of these.
TEST_CASE("Adam optimizer gradient check", "[adam][gradcheck]") {
    SKIP("not yet implemented");
}
