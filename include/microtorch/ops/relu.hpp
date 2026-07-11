#pragma once
#include "microtorch/Tensor.hpp"

namespace microtorch {

//ReLU nonlinearity
TensorPtr relu(const TensorPtr& a);

}  // namespace microtorch
