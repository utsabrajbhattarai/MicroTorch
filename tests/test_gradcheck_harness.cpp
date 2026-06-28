#include <catch2/catch_test_macros.hpp>

// TODO: the gradient-check harness itself needs its own test -- verify
// that a deliberately-wrong analytic gradient gets CAUGHT by the harness
// (not just that correct gradients pass). A harness that never fails on
// bad input isn't actually checking anything.
TEST_CASE("gradient-check harness catches a deliberately wrong gradient", "[gradcheck][meta]") {
    SKIP("not yet implemented");
}
