#pragma once
#include "microtorch/Tensor.hpp"
#include <vector>
#include <cmath>

namespace microtorch {

class Adam {
public:
    Adam(std::vector<TensorPtr> params, double lr, //stores actual tensor parameters to be optimized, learning rate, beta1, beta2, epsilon
         double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8);

    void step();
    void zero_grad();

private:
    std::vector<TensorPtr> params_;
    double lr_, beta1_, beta2_, eps_;
    int t_;  //step counter that starts at 0

    std::vector<Eigen::MatrixXd> m_;  //per-parameter; first moment vector
    std::vector<Eigen::MatrixXd> v_;  //per-parameter; squared-gradient average OR second moment vector
};

}