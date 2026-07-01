#include "microtorch/toy_data.hpp" //header file from the include directory
#include <cmath>

Eigen::MatrixXd make_spiral(int n_points, double noise) {
    Eigen::MatrixXd points(n_points, 2);
    double t_max = 4 * M_PI;

    for (int i = 0; i < n_points; ++i) {
        double t = (double)i / (n_points - 1) * t_max; //Scale t to range from 0 to 4π for 2 full revolutions
        double x = t * std::cos(t);
        double y = t * std::sin(t);
        points(i, 0) = x;
        points(i, 1) = y;
    }

    return points;
}