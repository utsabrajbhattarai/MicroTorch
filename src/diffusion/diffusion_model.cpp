#include "microtorch/Tensor.hpp"
#include "microtorch/diffusion/diffusion_model.hpp"
#include "microtorch/ops.hpp"
#include <cmath>
#include <random>

namespace microtorch {

    Eigen::MatrixXd forward_noise(const Eigen::MatrixXd& x0, int t, const NoiseSchedule& ns, Eigen::MatrixXd& noise_out, std::mt19937& rng){

        //Step 1: fill noise_out with a normal noise (same shape as x0)
        noise_out = Eigen::MatrixXd(x0.rows(), x0.cols());
        std::normal_distribution<double> dist(0.0, 1.0);

        for(int i = 0; i < noise_out.rows(); i++){
            for(int j = 0; j < noise_out.cols(); j++){
                noise_out(i,j) = dist(rng);
            }
        }

        //Step 2: calculating coefficients of equation:
        double c1 = sqrt(ns.alpha_bar[t]);
        double c2 = sqrt(1.0 - ns.alpha_bar[t]);

        //Step 3: blending (creating x_t and return)
        Eigen::MatrixXd x_t = c1 * x0 + c2 * noise_out;

        return x_t;

    }

}  // namespace microtorch
