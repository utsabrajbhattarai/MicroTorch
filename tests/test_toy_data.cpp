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