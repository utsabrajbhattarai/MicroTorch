#pragma once
#include "microtorch/Tensor.hpp"

namespace microtorch {

// matmul:backbone of every layer; backward is the transpose-multiply pattern
TensorPtr matmul(const TensorPtr& a, const TensorPtr& b);

}  // namespace microtorch
