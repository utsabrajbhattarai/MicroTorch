#include <catch2/catch_test_macros.hpp>
#include "microtorch/Tensor.hpp"

// TODO: forward-pass correctness check for MSE loss.
TEST_CASE("MSE loss forward is correct", "[mse_loss]") {
    // SKIP() until implemented -- keeps ctest's count honest (skipped,
    // not silently passing on an empty body).
    SKIP("not yet implemented");
}

// TODO: gradient check via finite differences for MSE loss.
// This is the non-negotiable harness call -- every op gets one of these.
TEST_CASE("MSE loss gradient check", "[mse_loss][gradcheck]") {
    SKIP("not yet implemented");
}
