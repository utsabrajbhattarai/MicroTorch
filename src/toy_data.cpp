#include "microtorch/toy_data.hpp" //header file from the include directory
#include <cmath>
#include <random>

namespace microtorch {

Eigen::MatrixXd make_spiral(int n, double noise) {
    Eigen::MatrixXd points(n, 2); //create an Eigen matrix to hold the points, with n rows and 2 columns (x and y).
    std::mt19937 rng(42); //a random number generator seeded with 42 for reproducibility; can be anything.
    std::uniform_real_distribution<double> t_dist(0.0, 1.0);
    std::normal_distribution<double> noise_dist(0.0, noise > 0.0 ? noise : 1.0);

    for (int i = 0; i < n; ++i) {
        double t = t_dist(rng);
        double r = t * 2.0; //radius increases linearly with t; basic formula.
        double theta = t * 3 * M_PI; //angle increases with 3; for 1.5 rotations, theta = 0 to 3π; basic formula.

        double x, y;
        if (i % 2 == 0) {
            x = r * std::cos(theta); //formula for x coordinate
            y = r * std::sin(theta); //formula for y coordinate
        } else {
            x = -r * std::cos(theta); //mirror the spiral for odd indices to create a double spiral effect
            y = -r * std::sin(theta); //mirror the spiral for odd indices to create a double spiral effect
        }

        double nx = (noise > 0.0) ? noise_dist(rng) : 0.0; //add noise to x coordinate
        double ny = (noise > 0.0) ? noise_dist(rng) : 0.0; //add noise to y coordinate
        points(i, 0) = x + nx;
        points(i, 1) = y + ny;
    }

    return points;
}

Eigen::MatrixXd make_two_moons(int n, double noise) {
    Eigen::MatrixXd points(n, 2); //create an Eigen matrix to hold the points, with n rows and 2 columns (x and y).
    int m_points = n / 2;
    std::mt19937 rng(42); //a random number generator seeded with 42 for reproducibility; can be anything.
    std::uniform_real_distribution<double> angle_dist(0.0, M_PI); //angle distribution for the half-circle
    std::normal_distribution<double> noise_dist(0.0, noise > 0.0 ? noise : 1.0); //noise distribution
    double scale = 1.5; //scale factor to adjust the size of the moons; can be changed to make the moons larger or smaller.

    //upper moon: rows 0 to m_points-1
    for (int j = 0; j < m_points; ++j) {
        double angle = angle_dist(rng);
        double x = std::cos(angle) * scale; //formula for x coordinate of the upper moon
        double y = std::sin(angle) * scale; //formula for y coordinate of the upper moon
        double nx = (noise > 0.0) ? noise_dist(rng) : 0.0;
        double ny = (noise > 0.0) ? noise_dist(rng) : 0.0;
        points(j, 0) = x + nx;
        points(j, 1) = y + ny;
    }

    //lower moon: rows m_points to n-1
    for (int j = 0; j < m_points; ++j) {
        double angle = angle_dist(rng);
        double x = (1 - std::cos(angle)) * scale; //formula for x coordinate of the lower moon, shifted to the right by 1 unit
        double y = (-std::sin(angle) - 0.5) * scale; //formula for y coordinate of the lower moon, shifted down by 0.5 units
        double nx = (noise > 0.0) ? noise_dist(rng) : 0.0;
        double ny = (noise > 0.0) ? noise_dist(rng) : 0.0;
        points(m_points + j, 0) = x + nx;
        points(m_points + j, 1) = y + ny;
    }

    return points;
}

}