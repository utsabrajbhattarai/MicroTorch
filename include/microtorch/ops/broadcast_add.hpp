#pragma once
#include "microtorch/Tensor.hpp"

namespace microtorch {

// broadcast-add -- biases, residuals
// TODO: declare forward + backward signatures once Tensor's design is set.
Tensor broadcast_add(const Tensor& a, const Tensor& b);

}  // namespace microtorch
