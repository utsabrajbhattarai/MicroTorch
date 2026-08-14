#pragma once
#include "microtorch/optim/optimizer.hpp"
#include <vector>

namespace microtorch {

    //Adam: adaptive moments (m, v) with bias correction aso inherits from optimizer
    class Adam : public Optimizer {
    public:
        Adam(const std::vector<TensorPtr>& params, double lr,
             double beta1 = 0.9, double beta2 = 0.999, double eps = 1e-8);

        void step() override;

    private:
        double lr_;
        double beta1_, beta2_, eps_;
        int t_;                              //timestep counter for bias correction
        std::vector<Eigen::MatrixXd> m_;     //first moment per parameter
        std::vector<Eigen::MatrixXd> v_;     //second moment per parameter
    };

}  // namespace microtorch