#include <catch2/catch_test_macros.hpp>
#include "microtorch/Tensor.hpp"

// TODO: forward-pass correctness check for SGD optimizer.
TEST_CASE("SGD optimizer forward is correct", "[sgd]") {
    // SKIP() until implemented -- keeps ctest's count honest (skipped,
    // not silently passing on an empty body).
    SKIP("not yet implemented");
}

// TODO: gradient check via finite differences for SGD optimizer.
// This is the non-negotiable harness call -- every op gets one of these.
TEST_CASE("SGD optimizer gradient check", "[sgd][gradcheck]") {
    SKIP("not yet implemented");
}
