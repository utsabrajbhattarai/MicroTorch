#include <catch2/catch_test_macros.hpp>
#include "microtorch/Tensor.hpp"

// TODO: forward-pass correctness check for broadcast_add.
TEST_CASE("broadcast_add forward is correct", "[broadcast_add]") {
    // SKIP() until implemented -- keeps ctest's count honest (skipped,
    // not silently passing on an empty body).
    SKIP("not yet implemented");
}

// TODO: gradient check via finite differences for broadcast_add.
// This is the non-negotiable harness call -- every op gets one of these.
TEST_CASE("broadcast_add gradient check", "[broadcast_add][gradcheck]") {
    SKIP("not yet implemented");
}
