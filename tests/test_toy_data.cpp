#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "microtorch/data/toy_data.hpp"

TEST_CASE("make_spiral returns correct shape", "[toy_data]") {
    Eigen::MatrixXd pts = make_spiral(500);
    REQUIRE(pts.rows() == 500);
    REQUIRE(pts.cols() == 2);
}

TEST_CASE("make_spiral first point is at origin", "[toy_data]") {
    Eigen::MatrixXd pts = make_spiral(500, 0.0);  //noise=0 for exact values
    REQUIRE(pts(0, 0) == Catch::Approx(0.0).margin(1e-9));
    REQUIRE(pts(0, 1) == Catch::Approx(0.0).margin(1e-9));
}

TEST_CASE("make_spiral distance increases with t", "[toy_data]") {
    Eigen::MatrixXd pts = make_spiral(500, 0.0);  // noise=0
    // point at t=π is closer to origin than point at t=2π
    double d_half = std::sqrt(pts(249, 0)*pts(249, 0) + pts(249, 1)*pts(249, 1));
    double d_full = std::sqrt(pts(499, 0)*pts(499, 0) + pts(499, 1)*pts(499, 1));
    REQUIRE(d_full > d_half);
}

TEST_CASE("make_two_moons returns correct shape", "[toy_data]") {
    Eigen::MatrixXd pts = make_two_moons(500, 0.0);
    REQUIRE(pts.rows() == 500);
    REQUIRE(pts.cols() == 2);
}

TEST_CASE("make_two_moons upper moon first point correct", "[toy_data]") {
    Eigen::MatrixXd pts = make_two_moons(500, 0.0);
    REQUIRE(pts(0, 0) == Catch::Approx(1.0).margin(1e-9));   // x = cos(0) = 1
    REQUIRE(pts(0, 1) == Catch::Approx(0.0).margin(1e-9));   // y = sin(0) = 0
}

TEST_CASE("make_two_moons lower moon first point correct", "[toy_data]") {
    Eigen::MatrixXd pts = make_two_moons(500, 0.0);
    REQUIRE(pts(250, 0) == Catch::Approx(0.0).margin(1e-9));    // x = 1 - cos(0) = 0
    REQUIRE(pts(250, 1) == Catch::Approx(-0.5).margin(1e-9));   // y = -sin(0) - 0.5 = -0.5
}