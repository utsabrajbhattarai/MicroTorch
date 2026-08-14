#pragma once
#include "microtorch/optim/optimizer.hpp"

namespace microtorch {

    //SGD: subtract lr * gradient from each parameter
    class SGD : public Optimizer {
    public:
        SGD(const std::vector<TensorPtr>& params, double lr);

        void step() override;   //override the base's pure-virtual step

    private:
        double lr_;   //learning rate
    };

}  // namespace microtorch