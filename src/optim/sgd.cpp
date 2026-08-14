#include "microtorch/optim/sgd.hpp"

namespace microtorch {

    //constructor: hand params up to the base class, store the learning rate
    SGD::SGD(const std::vector<TensorPtr>& params, double lr)
        : Optimizer(params), lr_(lr) {}

    //SGD update: each parameter moves against its gradient by lr
    void SGD::step() {
        for (auto& p : params_) {
            p->data -= lr_ * p->grad;
        }
    }

}  // namespace microtorch