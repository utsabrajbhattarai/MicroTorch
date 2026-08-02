#pragma once
#include <Eigen/Dense>

namespace microtorch {

Eigen::MatrixXd make_spiral(int n, double noise = 0.05);
Eigen::MatrixXd make_two_moons(int n, double noise = 0.05);
Eigen::MatrixXd make_checkerboard(int n, double noise = 0.0);
Eigen::MatrixXd make_heart(int n, double noise = 0.05);
Eigen::MatrixXd make_concentric_circles(int n, double noise = 0.05);
Eigen::MatrixXd make_infinity(int n, double noise = 0.05);

} 