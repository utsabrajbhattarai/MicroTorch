#pragma once
#include "microtorch/Tensor.hpp"

namespace microtorch {

// sum/mean reduction turning a whole matrix into either sums or means based on the given loss function derivation
TensorPtr sum(const TensorPtr& a);
TensorPtr mean(const TensorPtr& a);

}  // namespace microtorch
