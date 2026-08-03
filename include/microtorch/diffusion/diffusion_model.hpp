#pragma once
#include "microtorch/Tensor.hpp"
#include "microtorch/diffusion/noise_schedule.hpp"
#include <vector>
#include "microtorch/ops.hpp"
#include <random>


namespace microtorch {

    //forward of Diffsion=> Returns noisy x_t ie, data for training and also fills noise_out with noise
    Eigen::MatrixXd forward_noise(const Eigen::MatrixXd& x0, int t, const NoiseSchedule& ns, Eigen::MatrixXd& noise_out, std::mt19937& rng);




}