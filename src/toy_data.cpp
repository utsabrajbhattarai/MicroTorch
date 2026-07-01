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

Eigen::MatrixXd make_two_moons(int n_points, double noise) {
    Eigen::MatrixXd points(n_points, 2);
    int m_points = n_points / 2;  //half for each moon
    double t_max = M_PI;          //0 to π for each moon
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, noise > 0.0 ? noise : 1.0);

    for (int j = 0; j < m_points; ++j) {
        double t = (double)j / (m_points - 1) * t_max;
        double x = std::cos(t);
        double y = std::sin(t);
        double nx = (noise > 0.0) ? dist(rng) : 0.0;
        double ny = (noise > 0.0) ? dist(rng) : 0.0;
        points(j, 0) = x + nx;
        points(j, 1) = y + ny;
    }

    //lower moon: rows m_points to n_points-1
    for (int j = 0; j < m_points; ++j) {
        double t = (double)j / (m_points - 1) * t_max;
        double x = 1 - std::cos(t);
        double y = -std::sin(t) - 0.5;
        double nx = (noise > 0.0) ? dist(rng) : 0.0;
        double ny = (noise > 0.0) ? dist(rng) : 0.0;
        points(m_points + j, 0) = x + nx;
        points(m_points + j, 1) = y + ny;
    }

    return points;
}