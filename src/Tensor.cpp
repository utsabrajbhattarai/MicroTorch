#include "microtorch/Tensor.hpp"

namespace microtorch {

    void Tensor::zero_grad() {
        grad.setZero(); //sets grad zero
    }

}  // namespace microtorch
