#include <catch2/catch_test_macros.hpp>
#include "microtorch/Tensor.hpp"

// TODO: forward-pass correctness check for matmul.
TEST_CASE("matmul forward is correct", "[matmul]") {
    // SKIP() until implemented -- keeps ctest's count honest (skipped,
    // not silently passing on an empty body).
    SKIP("not yet implemented");
}

// TODO: gradient check via finite differences for matmul.
// This is the non-negotiable harness call -- every op gets one of these.
TEST_CASE("matmul gradient check", "[matmul][gradcheck]") {
    SKIP("not yet implemented");
}
