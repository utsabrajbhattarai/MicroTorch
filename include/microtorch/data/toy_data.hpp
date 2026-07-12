#pragma once
#include <Eigen/Dense>

Eigen::MatrixXd make_spiral(int n_points, double noise = 0.05);
Eigen::MatrixXd make_two_moons(int n_points, double noise = 0.05);