#pragma once
#include "microtorch/Tensor.hpp"
#include <vector>

namespace microtorch {

    //base class for all optimizers: owns the parameters and the shared zero_grad,
    //declares step() as pure virtual so each optimizer implements its own update rule
    //optimizer itself is ABSTRACT CLASS though
    class Optimizer {
    public:
        //constructor takes the parameters to optimize (shared with the model)
        Optimizer(const std::vector<TensorPtr>& params) : params_(params) {}

        //virtual destructor so deleting through an Optimizer* cleans up the subclass properly
        virtual ~Optimizer() = default;

        //shared across all optimizers: reset every parameter's gradient to zero
        void zero_grad() {
            for (auto& p : params_) {
                p->grad.setZero();
            }
        }

        //each optimizer defines its own parameter update (pure virtual = subclasses must implement)
        virtual void step() = 0;

    protected:
        std::vector<TensorPtr> params_;   //the parameters being optimized (accessible to subclasses)
    };

}  // namespace microtorch