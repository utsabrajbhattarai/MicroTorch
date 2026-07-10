#pragma once
#include "microtorch/Tensor.hpp"

namespace microtorch {

// sum/mean reduction -- shared primitive every loss needs

TensorPtr mean(const TensorPtr& a);

}  // namespace microtorch
