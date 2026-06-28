#pragma once
#include "microtorch/Tensor.hpp"

namespace microtorch {

// ReLU nonlinearity -- trivial backward
// TODO: declare forward + backward signatures once Tensor's design is set.
Tensor relu(const Tensor& a, const Tensor& b);

}  // namespace microtorch
