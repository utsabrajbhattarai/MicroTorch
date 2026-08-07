#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "microtorch/diffusion/noise_schedule.hpp"

using namespace microtorch;

TEST_CASE("noise schedule: alpha_bar slides from ~1 to ~0, monotonically", "[noise_schedule]") {
    int T = 200;
    NoiseSchedule ns = make_noise_schedule(T);

    SECTION("sizes are all T") {
        REQUIRE(ns.beta.size()      == T);
        REQUIRE(ns.alpha.size()     == T);
        REQUIRE(ns.alpha_bar.size() == T);
    }

    SECTION("beta is linear from start to end") {
        REQUIRE(ns.beta[0]   == Catch::Approx(1e-4));
        REQUIRE(ns.beta[T-1] == Catch::Approx(0.02));
    }

    SECTION("alpha = 1 - beta") {
        for (int t = 0; t < T; ++t)
            REQUIRE(ns.alpha[t] == Catch::Approx(1.0 - ns.beta[t]));
    }

    SECTION("alpha_bar starts near 1 and decreases monotonically") {
        REQUIRE(ns.alpha_bar[0] == Catch::Approx(1.0 - 1e-4));
        for (int t = 1; t < T; ++t)
            REQUIRE(ns.alpha_bar[t] < ns.alpha_bar[t-1]);   // strictly decreasing
    }

    SECTION("alpha_bar is the running product of alpha") {
        double prod = 1.0;
        for (int t = 0; t < T; ++t) {
            prod *= ns.alpha[t];
            REQUIRE(ns.alpha_bar[t] == Catch::Approx(prod));
        }
    }
}