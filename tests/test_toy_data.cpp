#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include "microtorch/toy_data.hpp"

using namespace microtorch;

TEST_CASE("make_spiral returns correct shape", "[toy_data]") {
    Eigen::MatrixXd pts = make_spiral(500);
    REQUIRE(pts.rows() == 500);
    REQUIRE(pts.cols() == 2);
}

TEST_CASE("make_spiral points stay within expected radius", "[toy_data]") {
    Eigen::MatrixXd pts = make_spiral(500, 0.0);  //noise=0 for exact bound
    for (int i = 0; i < pts.rows(); ++i) {
        double dist = std::sqrt(pts(i, 0)*pts(i, 0) + pts(i, 1)*pts(i, 1));
        REQUIRE(dist <= 2.0 + 1e-9);  //r = t*2.0, t in [0,1], so max radius is 2.0
    }
}

TEST_CASE("make_two_moons returns correct shape", "[toy_data]") {
    Eigen::MatrixXd pts = make_two_moons(500, 0.0);
    REQUIRE(pts.rows() == 500);
    REQUIRE(pts.cols() == 2);
}

TEST_CASE("make_two_moons upper moon has non-negative y", "[toy_data]") {
    Eigen::MatrixXd pts = make_two_moons(500, 0.0);
    int m_points = 500 / 2;
    for (int j = 0; j < m_points; ++j) {
        REQUIRE(pts(j, 1) >= -1e-9);  //angle in [0,π] means sin(angle) >= 0
    }
}

TEST_CASE("make_two_moons lower moon sits below upper moon", "[toy_data]") {
    Eigen::MatrixXd pts = make_two_moons(500, 0.0);
    int m_points = 500 / 2;
    for (int j = m_points; j < 500; ++j) {
        REQUIRE(pts(j, 1) <= 0.0 + 1e-9);  //shifted down, should stay non-positive
    }
}

TEST_CASE("make_infinity returns correct shape", "[toy_data]") {
    Eigen::MatrixXd pts = make_infinity(500);
    REQUIRE(pts.rows() == 500);
    REQUIRE(pts.cols() == 2);
}

TEST_CASE("make_infinity points stay within expected bounds", "[toy_data]") {
    Eigen::MatrixXd pts = make_infinity(500, 0.0);  //noise=0 for exact bound
    for (int i = 0; i < pts.rows(); ++i) {
        REQUIRE(pts(i, 0) >= -2.0 - 1e-9);
        REQUIRE(pts(i, 0) <= 2.0 + 1e-9);
        REQUIRE(pts(i, 1) >= -1.0 - 1e-9);  //y = sin(t)*cos(t)*2 has max amplitude 1
        REQUIRE(pts(i, 1) <= 1.0 + 1e-9);
    }
}

TEST_CASE("make_concentric_circles returns correct shape", "[toy_data]") {
    Eigen::MatrixXd pts = make_concentric_circles(500);
    REQUIRE(pts.rows() == 500);
    REQUIRE(pts.cols() == 2);
}

TEST_CASE("make_concentric_circles points sit on expected radii", "[toy_data]") {
    Eigen::MatrixXd pts = make_concentric_circles(500, 0.0);  //noise=0 for exact radius
    for (int i = 0; i < pts.rows(); ++i) {
        double dist = std::sqrt(pts(i, 0)*pts(i, 0) + pts(i, 1)*pts(i, 1));
        bool matches_known_radius = (std::abs(dist - 0.7) < 1e-9) || (std::abs(dist - 1.5) < 1e-9);
        REQUIRE(matches_known_radius);
    }
}

TEST_CASE("make_heart returns correct shape", "[toy_data]") {
    Eigen::MatrixXd pts = make_heart(500);
    REQUIRE(pts.rows() == 500);
    REQUIRE(pts.cols() == 2);
}

TEST_CASE("make_heart points stay within expected bounds", "[toy_data]") {
    Eigen::MatrixXd pts = make_heart(500, 0.0);  //noise=0 for exact bound
    for (int i = 0; i < pts.rows(); ++i) {
        REQUIRE(pts(i, 0) >= -2.5);
        REQUIRE(pts(i, 0) <= 2.5);
        REQUIRE(pts(i, 1) >= -2.5);
        REQUIRE(pts(i, 1) <= 2.5);
    }
}

TEST_CASE("make_checkerboard returns exactly n points", "[toy_data]") {
    Eigen::MatrixXd pts = make_checkerboard(500); //500 points, noise=0.0 by default
    REQUIRE(pts.rows() == 500); //500 points requested, should return 500 points
    REQUIRE(pts.cols() == 2); //2D points, so 2 columns
}

TEST_CASE("make_checkerboard points satisfy the checkerboard rule", "[toy_data]") {
    Eigen::MatrixXd pts = make_checkerboard(500, 0.0);  //noise=0 for exact checkerboard pattern
    for (int i = 0; i < pts.rows(); ++i) {
        int gx = (int)std::floor(pts(i, 0)); //floor the x coordinate to get the grid cell in x direction
        int gy = (int)std::floor(pts(i, 1)); //floor the y coordinate to get the grid cell in y direction
        REQUIRE((gx + gy) % 2 == 0); //checkerboard rule: sum of grid coordinates should be even
    }
}