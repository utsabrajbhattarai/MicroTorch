#include "microtorch/Tensor.hpp"
#include "microtorch/diffusion/noise_schedule.hpp"

namespace microtorch {

    NoiseSchedule make_noise_schedule(int T, double beta_start, double beta_end){

        NoiseSchedule ns;   //ns to return
        ns.T = T;
        ns.beta = Eigen::VectorXd(T);
        ns.alpha = Eigen::VectorXd(T);
        ns.alpha_bar = Eigen::VectorXd(T);

        for(int t = 0; t < T; t++){
            ns.beta[t] = beta_start + (beta_end - beta_start) * t / (T - 1);    //linearly increasing beta
            ns.alpha[t] = 1.0 - ns.beta[t];             //alpha  = 1-B
            ns.alpha_bar[t] = (t > 0) ? (ns.alpha_bar[t-1] * ns.alpha[t]) : ns.alpha[0]; //alpha_bar for 0 = alpha for 0 other wise its a cumulative product
        }
        return ns;
    }

}