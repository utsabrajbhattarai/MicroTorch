#include <catch2/catch_test_macros.hpp>
#include "microtorch/Tensor.hpp"

// Placeholder: confirms the build + Catch2 + ctest pipeline works end to
// end before any real Tensor logic exists. Replace/expand once Tensor
// actually has data and grad members.
TEST_CASE("build pipeline is wired correctly", "[tensor][smoke]") {
    REQUIRE(1 + 1 == 2);
}
