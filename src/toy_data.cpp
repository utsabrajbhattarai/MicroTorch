#include "microtorch/toy_data.hpp" //header file from the include directory
#include <cmath>
#include <random>

Eigen::MatrixXd make_spiral(int n_points, double noise) {
    Eigen::MatrixXd points(n_points, 2);
    double t_max = 4 * M_PI;
    std::mt19937 rng(42); //Fixed seed for reproducibility
    std::normal_distribution<double> dist(0.0, noise>0.0?noise:1.0); //mean=0, std=noise

    for (int i = 0; i < n_points; ++i) {
        double t = (double)i / (n_points - 1) * t_max; //Scale t to range from 0 to 4π for 2 full revolutions
        double x = t * std::cos(t);
        double y = t * std::sin(t);
        double nx=(noise>0.0)?dist(rng):0.0;
        double ny=(noise>0.0)?dist(rng):0.0;
        points(i, 0) = x + nx; //Add noise to x
        points(i, 1) = y + ny; //Add noise to y

    }

    return points;
}