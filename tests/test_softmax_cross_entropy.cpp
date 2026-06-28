#include <catch2/catch_test_macros.hpp>
#include "microtorch/Tensor.hpp"

// TODO: forward-pass correctness check for fused softmax + cross-entropy.
TEST_CASE("fused softmax + cross-entropy forward is correct", "[softmax_cross_entropy]") {
    // SKIP() until implemented -- keeps ctest's count honest (skipped,
    // not silently passing on an empty body).
    SKIP("not yet implemented");
}

// TODO: gradient check via finite differences for fused softmax + cross-entropy.
// This is the non-negotiable harness call -- every op gets one of these.
TEST_CASE("fused softmax + cross-entropy gradient check", "[softmax_cross_entropy][gradcheck]") {
    SKIP("not yet implemented");
}
