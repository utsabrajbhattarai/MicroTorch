#pragma once
#include "microtorch/Tensor.hpp"

namespace microtorch {

// matmul -- backbone of every layer; backward is the transpose-multiply pattern
// TODO: declare forward + backward signatures once Tensor's design is set.
Tensor matmul(const Tensor& a, const Tensor& b);

}  // namespace microtorch
